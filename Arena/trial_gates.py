from __future__ import annotations
import json, time, threading
from dataclasses import dataclass
from typing import Dict, Any

try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None

def _now_ms() -> int:
    return int(time.monotonic() * 1000)

class BaseGate:
    def wait_for_edge(self, timeout_s: float) -> bool: raise NotImplementedError
    def close(self): pass

# ---------------- pose_distance (MQTT) ----------------

@dataclass
class PoseDistanceSpec:
    camera: str
    threshold: float
    cam_magnification: float
    delta: float
    direction: str = 'enter_far'   # or 'enter_near'
    is_screen_on_left: bool = False,
    arena_size_y: float = 96.0,   # cm
    debounce_ms: int = 200
    source_topic: str = 'arena/pose/head'
    source_field: str = 'nose_y'
    camera_field: str = 'camera'
    host: str = 'localhost'
    port: int = 1883

class PoseDistanceGate(BaseGate):
    """
    Subscribes to pose MQTT (JSON), watches `source_field` from the specified `camera`,
    applies hysteresis on threshold±delta, emits on entering 'far' or 'near' per `direction`.
    """
    def __init__(self, spec: PoseDistanceSpec, logger):
        if mqtt is None:
            raise RuntimeError("paho-mqtt not available for PoseDistanceGate")
        self.spec = spec
        self.logger = logger

        self.low  = float(self.spec.threshold) - float(self.spec.delta)
        self.high = float(self.spec.threshold) + float(self.spec.delta)
        self.logger.info(f"[PoseDistanceGate] band=[{self.low:.3f},{self.high:.3f}] (threshold={self.spec.threshold}±{self.spec.delta})")
        self.is_real_world_values = True
        assert self.low <= self.high
        assert self.spec.direction in ('enter_far', 'enter_near')

        self._evt = threading.Event()
        self._last_state = None      # 'near' | 'far'
        self._last_change_ms = 0

        # mqtt client
        self._client = mqtt.Client(client_id=f"pose-gate-{int(time.time()*1000)}", clean_session=True)
        self._client.on_message = self._on_message
        self._client.connect(self.spec.host, int(self.spec.port), keepalive=30)
        self._client.subscribe(self.spec.source_topic, qos=0)
        self._client.loop_start()

        self.logger.info(
            f"[PoseDistanceGate] topic={self.spec.source_topic} cam={self.spec.camera} "
            f"field={self.spec.source_field} band=[{self.low:.3f},{self.high:.3f}] "
            f"dir={self.spec.direction} debounce={self.spec.debounce_ms}ms"
        )

    def _classify(self, v: float):
        if self.is_real_world_values:
            high, low, arena_size_y = self.high, self.low, self.spec.arena_size_y
        else:
            high = self.high * self.spec.cam_magnification
            low = self.low * self.spec.cam_magnification
            arena_size_y = self.spec.arena_size_y * self.spec.cam_magnification
        # self.logger.info(f"[PoseDistanceGate] raw={v:.3f} band=[{low},{high}] rw_values={self.is_real_world_values}, cam_inverted={self.spec.is_screen_on_left}")
        if self.spec.is_screen_on_left:
            low = arena_size_y - high
            high = arena_size_y - low
            # self.logger.info(f"[PoseDistanceGate] raw={v:.3f} band=[{low},{high}]" 
            #                  f"rw_values={self.is_real_world_values}, cam_inverted={self.spec.is_screen_on_left}")

            if v >= low:  return 'near'
            if v <= high: return 'far'
        else:
            if v <= low:  return 'near'
            if v >= high: return 'far'
        return self._last_state  # within deadband -> keep state

    def _on_message(self, _client, _userdata, msg):
        try:
            obj = json.loads(msg.payload.decode('utf-8', errors='ignore'))
            cam = obj.get(self.spec.camera_field)
            if cam != self.spec.camera:
                return
            raw = obj.get(self.spec.source_field)
            if raw is None:
                return
            v = float(raw)
            is_real_world_values = bool(obj.get('is_real_world_values', False))
        except Exception:
            return
        self.is_real_world_values = is_real_world_values
        st = {
            'value': v,
            'state': self._classify(v)
        }
        now = _now_ms()

        if self._last_state is None:
            self._last_state, self._last_change_ms = st, now
            return

        if st is not None and st['state'] != self._last_state['state'] and (now - self._last_change_ms) >= int(self.spec.debounce_ms):
            fire = (
                (self.spec.direction == 'enter_far' and st['state'] == 'far')
                or
                (self.spec.direction == 'enter_near' and st['state'] == 'near')
            )
            if fire:
                self.logger.info(
                    f"[PoseDistanceGate] {self.spec.source_field}="
                    f"{self._last_state['value']:.3f} → {st['value']:.3f} : {st['state']}"
                )
                self._evt.set()
            self._last_state, self._last_change_ms = st, now

    def wait_for_edge(self, timeout_s: float) -> bool:
        # arm once per wait
        self._evt.clear()
        return self._evt.wait(timeout=max(0.0, float(timeout_s)))

    def close(self):
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass

# --------------- builder ----------------

def build_gate(spec: Dict[str, Any], logger) -> BaseGate:
    typ = (spec.get('type') or 'mqtt_edge').lower()

    if typ == 'mqtt_edge':
        # existing lights/feeder etc. path (unchanged)
        from subscribers import MqttEdgeGate
        return MqttEdgeGate(
            topic=spec.get('topic', 'arena/value'),
            field=spec.get('field', 'day_lights'),
            edge=spec.get('edge', 'rising'),
            debounce_ms=int(spec.get('debounce_ms', 300)),
            logger=logger
        )

    if typ == 'pose_distance':
        pd = PoseDistanceSpec(
            camera              = spec.get('camera') or 'front',
            threshold           = float(spec.get('threshold', 0.30)),
            cam_magnification   = float(spec.get('cam_magnification', 100)),
            is_screen_on_left     = bool(int(spec.get('is_screen_on_left', 0))),
            arena_size_y        = float(spec.get('arena_size_y', 96.0)),
            delta               = float(spec.get('delta',  0.0)),
            direction           = spec.get('direction', 'enter_far'),
            debounce_ms         = int(spec.get('debounce_ms', 200)),
            source_topic        = spec.get('source_topic', 'arena/pose/head'),
            source_field        = spec.get('source_field', 'nose_y'),
            camera_field        = spec.get('camera_field', 'camera'),
            host                = spec.get('host', 'localhost'),
            port                = int(spec.get('port', 1883)),
        )
        return PoseDistanceGate(pd, logger)

    raise ValueError(f"Unknown gate type: {typ}")
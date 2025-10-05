import json
import os
import threading
import time
from datetime import datetime
import paho.mqtt.client as mqtt
import paho.mqtt.subscribe as subscribe
from pathlib import Path
import utils
from cache import RedisCache, CacheColumns as cc, Column
from loggers import get_logger
import serial
from serial.tools import list_ports
from db_models import ORM
import config

HEALTHCHECK_PREFIX = 'periphery_healthcheck_'
TOGGLES_STATE_PREFIX = 'periphery_toggles_state_'
HEALTHCHECK_TIMEOUT = 10
TOGGLES_STATE_TIMEOUT = 120


class PeripheryIntegrator:
    """class for communicating with reptilearn's arena.py"""
    def __init__(self):
        self.logger = get_logger('Periphery')
        self.logger.debug('periphery integration created')
        self.cache = RedisCache()
        self.mqtt_client = mqtt.Client()
        self.orm = ORM()
        self.periphery_config = config.load_configuration('periphery')
        if self.periphery_config and config.ARENA_ARDUINO_NAME in self.periphery_config:
            self.devices = {d['name']: d for d in self.periphery_config[config.ARENA_ARDUINO_NAME]['interfaces']}
        else:
            self.devices = {}

    def save_config_to_file(self):
        with open(config.configurations["periphery"][0], 'w') as f:
            json.dump(self.periphery_config, f, indent=4)

    def switch(self, name, state):
        assert state in [0, 1]
        if name not in self.toggles:
            self.logger.warning(f'No toggle named {name}; abort switch command')
            return
        if self.devices[name].get('nc_toggle'):
            state = int(not state)
        self.mqtt_publish(config.mqtt['publish_topic'], f'["set","{name}",{state}]')
        time.sleep(1)
        self.mqtt_publish(config.mqtt['publish_topic'], f'["get","{name}"]')

    def cam_trigger(self, state):
        assert state in [0, 1]
        self.mqtt_publish(config.mqtt['publish_topic'], f'["set","Camera Trigger",{state}]')
        self.cache.set(cc.CAM_TRIGGER_STATE, state)

    def publish_cam_trigger_state(self):
        self.mqtt_publish(config.mqtt['publish_topic'], f'["get","Camera Trigger"]')

    def change_trigger_fps(self, new_fps):
        new_duration = round(1000 / new_fps)
        trig_inters = self.periphery_config[config.CAM_TRIGGER_ARDUINO_NAME]['interfaces'][0]
        trig_inters['pulse_len'] = new_duration
        self.save_config_to_file()
        next(utils.run_command('cd ../docker && docker-compose restart periphery'))

        # self.mqtt_publish('change_cam_trigger_duration', new_duration)

        time.sleep(5)
        self.cam_trigger(1)
        self.logger.info(f'Published cam trigger FPS change to {new_fps}')

    def feed(self, is_manual=False):
        if self.cache.get(cc.IS_REWARD_TIMEOUT):
            return
        self.cache.set(cc.IS_REWARD_TIMEOUT, True)

        feed_counts = self.get_feeders_counts()
        if all([c == 0 for c in feed_counts.values()]):
            self.logger.warning('No reward left in feeders')
            return

        for feeder_name, count in self.get_feeders_counts().items():
            if count == 0:
                continue

            log_str = f'Reward given by {feeder_name}'
            if config.FEEDER_AUDIO_PATH:
                self.activate_feeder_audio()
                log_str += f' and playing audio {config.FEEDER_AUDIO_PATH}'
            if self.cache.get(cc.FEEDER_DELAY):
                self.logger.info(f'Feeder delay active for {feeder_name}')
                time.sleep(int(self.cache.get(cc.FEEDER_DELAY)))

            self.mqtt_publish(config.mqtt['publish_topic'], f'["dispense","{feeder_name}"]')
            self.update_reward_count(feeder_name, count - 1)
            self.logger.info(log_str)
            self.orm.commit_reward(datetime.now(), is_manual=is_manual)
            break

    def activate_feeder_audio(self):
        """Plays sound for the feeder activation"""
        wav_path = config.FEEDER_AUDIO_PATH
        if not Path(wav_path).exists():
            self.logger.error(f"Cannot play sound: {wav_path} not found.")
            return
        self.play_wav_file(wav_path)

    @staticmethod
    def play_wav_file(wav_path):
        cmd = f'XDG_RUNTIME_DIR="/run/user/$(id -u)" paplay {wav_path}'
        next(utils.run_command(cmd))

    def mqtt_publish(self, topic, payload):
        self.mqtt_client.connect(config.mqtt['host'], config.mqtt['port'], keepalive=60)
        self.mqtt_client.publish(topic, payload)

    def get_feeders_counts(self) -> dict:
        counts = self.cache.get(cc.REWARD_LEFT)
        if counts is None:
            counts = [0 for _ in self.feeders]
        return {n: int(c) for n, c in zip(self.feeders, counts)}

    def update_reward_count(self, feeder_name, reward_count):
        c = self.get_feeders_counts()
        c[feeder_name] = reward_count
        new_counts = [str(c.get(feeder, 0)) for feeder in self.feeders]
        self.cache.set(cc.REWARD_LEFT, new_counts)

    def get_feeder_delay(self):
        delay = self.cache.get(cc.FEEDER_DELAY)
        return delay or 0

    def update_feeder_delay(self, delay):
        self.logger.info(f'Updating feeder delay to {delay}')
        self.cache.set(cc.FEEDER_DELAY, delay)

    def check_periphery_healthcheck(self):
        res = []
        for port_name in self.periphery_config.keys():
            if self.cache.get(Column(f'{HEALTHCHECK_PREFIX}{port_name}', bool, HEALTHCHECK_TIMEOUT)):
                res.append(port_name)
        return res

    def check_toggles_states(self):
        res = {}
        for toggle in self.toggles:
            value = self.check_toggle(toggle)
            if value is not None:
                res[toggle] = value
        return res

    def check_toggle(self, toggle):
        return self.cache.get(Column(f'{TOGGLES_STATE_PREFIX}{toggle}', bool, TOGGLES_STATE_TIMEOUT))

    def send_toggles_healthcheck(self):
        for toggle in self.toggles:
            self.mqtt_publish(config.mqtt['publish_topic'], f'["get","{toggle}"]')
            time.sleep(1)

    @property
    def toggles(self) -> list:
        return [k for k, dev in self.devices.items() if dev['type'] == 'line']

    @property
    def feeders(self) -> list:
        feeds = [(k, dev['order']) for k, dev in self.devices.items() if dev['type'] == 'feeder']
        return [x[0] for x in sorted(feeds, key=lambda x: x[1])]


class MQTTListener:
    topics = []

    def __init__(self, topics=None, callback=None, is_debug=True, stop_event=None, is_loop_forever=False):
        self.client = mqtt.Client()
        self.callback = callback
        self.topics = topics or self.topics
        self.topics = self.topics if isinstance(self.topics, (tuple, list)) else [self.topics]
        self.stop_event = stop_event
        self.is_debug = is_debug
        self.is_loop_forever = is_loop_forever
        self.is_initiated = False

    def init(self):
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(config.mqtt['host'], config.mqtt['port'])
        self.is_initiated = True

    def loop(self):
        if not self.is_initiated:
            self.init()
        if self.stop_event is not None:
            while not self.stop_event.is_set():
                self.client.loop()
        elif self.is_loop_forever:
            self.client.loop_forever()
        else:
            self.client.loop()

    def on_connect(self, client, userdata, flags, rc):
        if self.is_debug:
            print(f'MQTT connecting to host: {config.mqtt["host"]}; rc: {rc}')
        client.subscribe([(topic, 0) for topic in self.topics])

    def parse_payload(self, payload):
        return payload

    def on_message(self, client, userdata, msg):
        payload = self.parse_payload(msg.payload.decode('utf-8'))
        if self.callback is not None:
            self.callback(payload)
        if self.is_debug:
            print(f'received message with topic {msg.topic}: {payload}')


class ArenaListener(MQTTListener):
    topics = ['arena/value', 'arena/listening']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = RedisCache()
        self.toggles = self.load_toggles()

    @staticmethod
    def load_toggles():
        toggles = {}
        periphery_config = config.load_configuration('periphery')
        if periphery_config and config.ARENA_ARDUINO_NAME in periphery_config:
            toggles = {d['name']: d for d in periphery_config[config.ARENA_ARDUINO_NAME]['interfaces'] if d['type'] == 'line'}
        return toggles

    def on_message(self, client, userdata, msg):
        if msg.topic == 'arena/listening':
            port_name = msg.payload.decode('utf-8')
            col = Column(f'{HEALTHCHECK_PREFIX}{port_name}', bool, HEALTHCHECK_TIMEOUT)
            self.cache.set(col, True)
        else:
            super().on_message(client, userdata, msg)

    def parse_payload(self, payload):
        payload = json.loads(payload)
        if not isinstance(payload, dict):
            return payload

        return_payload = {}
        for name, value in payload.items():
            if name in config.mqtt['temperature_sensors']:
                return_payload = self.parse_temperature(name, value)
            elif name == 'Camera Trigger':
                self.parse_trigger_state(value)
            elif name in self.toggles:
                col = Column(f'{TOGGLES_STATE_PREFIX}{name}', bool, TOGGLES_STATE_TIMEOUT)
                value = bool(value)
                if self.toggles[name].get('nc_toggle'):
                    value = not value
                self.cache.set(col, value)

        return return_payload

    @staticmethod
    def parse_temperature(name, data):
        res = {}
        if not data:
            return
        elif len(data) == 1:
            res[name] = data[0]
        else:
            res.update({f'{name}{i}': v for i, v in enumerate(data)})
        return res

    def parse_trigger_state(self, value):
        if value in [0, 1]:
            self.cache.set(cc.CAM_TRIGGER_STATE, value)
        return


class LightSTIM:
    """Lights stimulations"""
    def __init__(self):
        self.port = None
        self.logger = get_logger('lightSTIM')
        if config.LIGHT_STIM_SERIAL:
            for p in list_ports.comports():
                if p.serial_number == config.LIGHT_STIM_SERIAL:
                    self.port = p.device
                    break
            if not self.port:
                self.logger.warning(f'could not find lightSTIM arduino with serial: {config.LIGHT_STIM_SERIAL}')
        else:
            self.logger.warning(f'LIGHT_STIM_PORT is not configured, cannot run lightSTIM command')

    def run_stim_command(self, stim_cmd):
        if self.port is None:
            self.logger.warning(f'cannot run stim light command, no port found')
            return

        ser = serial.Serial(self.port, config.LIGHT_STIM_BAUD, timeout=1)
        time.sleep(1)
        self.logger.info(f'Start LightSTIM command: "{stim_cmd}"')
        ser.write((stim_cmd + '\r\n').encode('ascii'))
        ser.close()

    def stop_stim(self):
        if self.port is None:
            self.logger.warning(f'cannot run stop stim light, no port found')
            return

        ser = serial.Serial(self.port, config.LIGHT_STIM_BAUD, timeout=1)
        time.sleep(1)
        self.logger.info(f'Stop LightSTIM')
        ser.write('STOP\r\n'.encode('ascii'))
        ser.close()

if __name__ == "__main__":
    def hc_callback(payload):
        print(payload)


    e = threading.Event()
    listener = ArenaListener(is_debug=False, callback=hc_callback, is_loop_forever=True, stop_event=e)
    listener.topics = ['arena/value', 'arena/listening']
    t = threading.Thread(target=listener.loop)
    t.start()

    pi = PeripheryIntegrator()
    pi.send_toggles_healthcheck()
    # pi.mqtt_publish(config.mqtt['publish_topic'], f'["set","Camera Trigger",1]')
    # time.sleep(1)
    # pi.mqtt_publish(config.mqtt['publish_topic'], f'["get","Camera Trigger"]')
    # time.sleep(1)
    # pi.mqtt_publish(config.mqtt['publish_topic'], f'["set","Camera Trigger",0]')
    # time.sleep(1)
    # pi.mqtt_publish(config.mqtt['publish_topic'], f'["get","Camera Trigger"]')
    # time.sleep(10)
    # e.set()
    # listener.loop()
    # while True:
    #     listener.loop()
    #     time.sleep(0.1)


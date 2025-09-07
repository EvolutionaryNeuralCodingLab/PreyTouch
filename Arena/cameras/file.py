# cameras/file.py
import cv2
import time
import numpy as np
import pandas as pd
from arrayqueues.shared_arrays import Full
from loggers import get_logger
from arena import Camera
import config

class FileCamera(Camera):
    """
    Drop-in virtual camera that mimics FLIRCamera behavior but reads from a video file.
    - Same base class (Camera / ArenaProcess)
    - Same frame emission path: frames_queue.put(img, timestamp) + calc_fps(timestamp)
    - Real-time pacing to match configured fps (and optional speed multiplier)
    """

    def __init__(self, *args, **kwargs):
        # IMPORTANT: accept the same signature as other cameras
        super().__init__(*args, **kwargs)
        self.logger = get_logger(f'Cam-{self.cam_name}')

        # Pull settings from cam_config (keep defaults safe)
        cfg = self.cam_config or {}
        self.src            = cfg.get('source')                       # REQUIRED
        assert self.src, f"FileCamera {self.cam_name} requires cam_config['source'] (path to video)"

        self.fps            = float(cfg.get('fps', 30))
        self.writing_fps    = int(cfg.get('writing_fps', self.fps))
        self.loop           = bool(cfg.get('loop', False))
        self.preview        = bool(cfg.get('preview', False))
        self.speed          = float(cfg.get('speed', 1.0))
        self.start_offset_s = float(cfg.get('start_offset_s', 0.0))

        # Target resize to match other cams / predictors (optional)
        img_sz = cfg.get('image_size')
        self._resize = None
        if isinstance(img_sz, (list, tuple)) and len(img_sz) >= 2:
            # OpenCV expects (W, H)
            self._resize = (int(img_sz[1]), int(img_sz[0]))

        # pacing: period in seconds (respect speed multiplier)
        base_fps = max(0.001, self.fps)
        spd      = max(0.001, self.speed)
        self._period_s = 1.0 / (base_fps * spd)

        self.cap = None

    # Do not override start/stop — base class handles process creation and joins.
    # Implement only the _run loop, like FLIRCamera.

    def _open_cap(self):
        cap = cv2.VideoCapture(self.src)
        if not cap or not cap.isOpened():
            raise RuntimeError(f"FileCamera {self.cam_name}: cannot open video: {self.src}")
        if self.start_offset_s > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, self.start_offset_s * 1000.0)
        return cap

    def _read_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        is_color = bool(self.cam_config.get('is_color'))
        if not is_color:
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # keep SHM shape (H,W,1)
            if frame.ndim == 2:
                frame = np.expand_dims(frame, axis=2)
        if self._resize is not None:
            frame = cv2.resize(frame, self._resize)
        return frame

    def _run(self):
        """Mirror FLIRCamera loop but drive frames from a file at real-time pace."""
        self.camera_time_delta = 0.0  # file stream -> just use server time
        cap_open_time = None
        try:
            self.cap = self._open_cap()
            cap_open_time = time.time()
            next_due = time.time()
            last_queue_warn = 0.0

            while not self.stop_signal.is_set():
                frame = self._read_frame()
                if frame is None:
                    if self.loop:
                        # restart file
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        self.cap = self._open_cap()
                        continue
                    else:
                        break  # EOF

                # Timestamp: use server time to behave like FLIR after delta applied
                timestamp = time.time()

                # Emit exactly like FLIR.image_handler
                t0 = time.time()
                while True:
                    try:
                        self.frames_queue.put(frame, timestamp)
                        self.calc_fps(timestamp)
                        break
                    except Full:
                        # Match FLIR behavior: warn at most once per ~60s
                        if (time.time() - last_queue_warn) > 60.0:
                            self.logger.warning('Queue is still full after waiting 0.1')
                            last_queue_warn = time.time()
                        if (time.time() - t0) > 0.1:
                            break
                        time.sleep(0.01)

                # Real-time pacing to target FPS*speed
                next_due += self._period_s
                sleep_s = next_due - time.time()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    # if we’re behind badly, reset schedule to now
                    next_due = time.time()

        except Exception:
            self.logger.exception('Error in FileCamera:')
        finally:
            try:
                if self.cap:
                    self.cap.release()
            except Exception:
                pass


def scan_cameras(is_print=True) -> pd.DataFrame:
    """
    Arena expects each camera module to provide scan_cameras() that returns a df
    whose index are the camera names detected for this module.
    For file cameras, “detection” = entries in config.cameras with module == 'file'.
    """
    rows, names = [], []
    for cam_name, d in (config.cameras or {}).items():
        if d.get('module') == 'file':
            rows.append({
                'DeviceID': d.get('id', cam_name),
                'Source':   d.get('source', ''),
                'FPS':      d.get('fps', None),
            })
            names.append(cam_name)

    df = pd.DataFrame(rows, index=names)
    if is_print:
        if not df.empty:
            print(f'\nFile Cameras:\n\n{df.to_string()}\n')
        else:
            print('No File Cameras configured')
    return df
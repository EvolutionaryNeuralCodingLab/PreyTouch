import cv2
import time
import numpy as np
import pandas as pd

import config
from pathlib import Path
from arena import ImageHandler, QueueException
from cache import RedisCache, CacheColumns as cc
from analysis.pose_utils import put_text
from analysis.predictors.tongue_out import TongueOutAnalyzer, TONGUE_CLASS
from analysis.pose import ArenaPose, PogonaHeadPose


class PredictHandler(ImageHandler):
    def __init__(self, *args, logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.cache = RedisCache()
        self.last_timestamp = None
        self.is_initiated = False
        self.prediction_summary = ''

    def __str__(self):
        return f'Predictor-{self.cam_name}'

    def _init(self, img):
        try:
            self.init(img)
        except Exception as exc:
            raise Exception(f'Could not initiate predictor; {exc}')

    def init(self, img):
        pass

    def loop(self):
        self.logger.info('start predictor loop')
        try:
            while not self.stop_signal.is_set() and not self.mp_metadata['predictors_stop'].is_set():
                timestamp = self.wait_for_next_frame()
                img = np.frombuffer(self.shm.buf, dtype=config.shm_buffer_dtype).reshape(self.cam_config['image_size'])
                img = self.before_predict(img)

                pred, img = self.predict_frame(img, timestamp)
                if not self.is_initiated:
                    self._init(img)
                self.analyze_prediction(timestamp, pred)

                # copy the image+predictions to pred_shm
                if self.pred_image_size is not None and self.is_streaming:
                    img = self.draw_pred_on_image(pred, img)
                    buf_np = np.frombuffer(self.pred_shm.buf, dtype=config.shm_buffer_dtype).reshape(
                        self.pred_image_size)
                    np.copyto(buf_np, img)

                t_end = time.time()
                self.calc_fps(t_end)
                self.calc_pred_delay(timestamp, t_end)
                self.last_timestamp = timestamp
        finally:
            self.on_stop()
            if self.mp_metadata['is_pred_on'].is_set():
                self.mp_metadata['is_pred_on'].clear()
            self.mp_metadata[self.calc_fps_name].value = 0.0
            self.mp_metadata['pred_delay'].value = 0.0
            self.logger.info('predict loop is closed')

    def on_stop(self):
        pass

    def before_predict(self, img):
        return img

    def predict_frame(self, img, timestamp):
        """Return the prediction vector and the image itself in case it was changed"""
        return None, img

    def analyze_prediction(self, timestamp, pred):
        pass

    def get_db_video_id(self):
        return self.mp_metadata['db_video_id'].value or None

    def draw_pred_on_image(self, det, img):
        return img

    def wait_for_next_frame(self, timeout=config.PREDICTORS_WAIT_FRAME_TIMEOUT):
        current_timestamp = self.mp_metadata['shm_frame_timestamp'].value
        t0 = time.time()
        while self.last_timestamp and current_timestamp == self.last_timestamp:
            if time.time() - t0 > timeout:
                raise QueueException(f'{str(self)} waited for {timeout} seconds and no new frames came; abort')
            current_timestamp = self.mp_metadata['shm_frame_timestamp'].value
            time.sleep(0.001)
        return current_timestamp


class TongueOutHandler(PredictHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyzer = TongueOutAnalyzer(action_callback=self.publish_tongue_out)
        self.last_detected_ts = None
        self.frames_probas = []
        self.mp_metadata['is_pred_on'].set()

    def __str__(self):
        return f'tongue-out-{self.cam_name}'

    def publish_tongue_out(self):
        self.cache.publish_command('strike_predicted')
        # self.logger.info('Tongue detected!')

    def predict_frame(self, img, timestamp):
        is_tongue, resized_img, prob, _ = self.analyzer.predict(img, timestamp)
        self.frames_probas.append((timestamp, prob))
        if is_tongue:
            self.publish_tongue_out()
            self.last_detected_ts = timestamp
        return is_tongue, resized_img

    def log_prediction(self, is_tongue, timestamp):
        pass

    def on_stop(self):
        try:
            block_path = self.cache.get(cc.EXPERIMENT_BLOCK_PATH)
            if not block_path:
                self.logger.warning('unable to save tounge predictions; block path is None')
                return

            df = pd.DataFrame(self.frames_probas, columns=['timestamp', 'prob'])
            df.to_csv(Path(block_path) / 'tongue_probas.csv')
        except Exception as exc:
            self.logger.error(f'unable to save tongue_probas; {exc}')

    def draw_pred_on_image(self, prob, img):
        if not prob:
            return img

        h, w = img.shape[:2]
        font, color = cv2.FONT_HERSHEY_SIMPLEX, (255, 0, 255)
        img = cv2.putText(img, f'P={prob:.2f}', (20, h - 30), font, 1, color, 2, cv2.LINE_AA)
        # img = put_text(f'P={prob:.2f}', img, img.shape[1] - 120, 30, color=(255, 0, 0))
        return img


class PogonaHeadHandler(PredictHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arena_pose = PogonaHeadPose(self.cam_name)

    def __str__(self):
        return f'pogona-head-{self.cam_name}'

    def loop(self):
        detector = self.arena_pose.predictor.detector
        self.logger.info(f"YOLO detector loaded successfully")
        super().loop()

    def _init(self, img):
        self.arena_pose.init(img)
        self.is_initiated = True

    def predict_frame(self, img, timestamp):
        """Get detection of pogona head on frame"""
        is_plot_preds = self.pred_image_size is not None and self.is_streaming
        if len(img.shape) == 2 or img.shape[-1] == 1:
            # convert gray image to 3-channels
            img = cv2.merge((img, img, img))
        pred_row_df, img = self.arena_pose.predictor.predict(img, is_plot_preds=is_plot_preds)
        return pred_row_df, img

    def analyze_prediction(self, timestamp, pred_row_df):
        db_video_id = self.get_db_video_id()
        self.prediction_summary = self.arena_pose.analyze_frame(timestamp, pred_row_df, db_video_id)

    def draw_pred_on_image(self, pred_row, img, font=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0)):
        angle = pred_row.iloc[0][('angle', '')]
        x, y = pred_row.iloc[0][('nose', 'x')], pred_row.iloc[0][('nose', 'y')]
        text = f'({round(x), round(y)}), head_angle={round(np.rad2deg(angle))}' if not np.isnan(x) else 'No Lizard Detection'
        img = put_text(text, img, 20, self.pred_image_size[0]-20)
        return img
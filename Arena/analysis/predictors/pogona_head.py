import pandas as pd
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from analysis.predictors.base import Predictor


class PogonaHead(Predictor):
    def __init__(self, cam_name, model_path=None):
        super(PogonaHead, self).__init__(model_path)
        self.model_name = Path(self.model_path).parent.name
        self.cam_name = cam_name
        self.bodyparts = ['nose', 'left_ear', 'right_ear']
        self.detector = YOLO(self.model_path)
        self.is_initialized = False

    def init(self, img):
        self.is_initialized = True

    def predict(self, frame, frame_id=0, is_plot_preds=False):
        res = self.detector(frame, save=False, imgsz=640, conf=self.threshold, iou=0.45, device='cuda:0', verbose=False)[0]
        preds = res.summary()
        pred = preds[0] if preds else None
        pred_row_df = self.convert_to_dlc_row(pred, frame_id)
        if is_plot_preds:
            frame = res.plot()
        return pred_row_df, frame

    def convert_to_dlc_row(self, res, frame_id):
        d = {}
        for i, bp in enumerate(self.bodyparts):
            for c, new_c in [('x', 'cam_x'), ('y', 'cam_y'), ('visible', 'prob')]:
                d[(bp, new_c)] = res['keypoints'][c][i] if res is not None else np.nan
        return pd.DataFrame(d, index=[frame_id])

    def create_pred_row(self, res):
        return res

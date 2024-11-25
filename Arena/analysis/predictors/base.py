import yaml
from pathlib import Path
import cv2
import pandas as pd
if __name__ == '__main__':
    import os
    os.chdir('../..')
import config



class Predictor:
    def __init__(self, model_path=None):
        self.pred_config = dict()
        self.load_pred_config()
        self.threshold = self.pred_config['threshold']
        self.model_path = model_path or self.pred_config['model_path']
        self.model_name = Path(self.model_path).name

    def init(self, *args, **kwargs):
        pass

    def predict(self, frame, timestamp):
        raise NotImplemented('No predict method')

    def create_pred_row(self, res):
        raise NotImplemented('No create_pred_row method')

    def predict_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        df = []
        for frame_id in range(n_frames):
            ret, frame = cap.read()
            res = self.predict(frame, frame_id)
            row = self.create_pred_row(res)
            df.append(row)
        cap.release()
        df = pd.DataFrame(df)
        df.to_parquet(self.get_predicted_cache_path(video_path))

    def get_predicted_cache_path(self, video_path) -> Path:
        preds_dir = Path(video_path).parent / 'predictions'
        preds_dir.mkdir(exist_ok=True)
        vid_name = Path(video_path).with_suffix('.parquet').name
        return preds_dir / f'{self.model_name}__{vid_name}'

    def plot_predictions(self, frame, *args, **kwargs):
        return frame

    def load_pred_config(self):
        pconfig = config.load_configuration('predict')
        predictor_name = type(self).__name__
        for k, d in pconfig.items():
            if d.get('predictor_name') == predictor_name:
                self.pred_config = d
                break
        assert self.pred_config, f'Could not find config for {predictor_name}'

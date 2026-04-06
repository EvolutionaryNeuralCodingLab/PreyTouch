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
        resume_df = self.load_resume_video(video_path)
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        df = resume_df.to_dict('records') if resume_df is not None else []
        df = df[:n_frames]
        start_frame = min(n_frames, len(df))
        checkpoint_stride = max(1, n_frames // 5)
        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_id in range(start_frame, n_frames):
            ret, frame = cap.read()
            res = self.predict(frame, frame_id)
            row = self.create_pred_row(res)
            df.append(row)
            if (frame_id == n_frames - 1) or ((frame_id + 1) % checkpoint_stride == 0):
                self.save_resume_video(pd.DataFrame(df), video_path)
        cap.release()
        df = pd.DataFrame(df)
        self.save_predicted_video(df, video_path)

    def get_predicted_cache_path(self, video_path) -> Path:
        preds_dir = Path(video_path).parent / 'predictions'
        preds_dir.mkdir(exist_ok=True)
        vid_name = Path(video_path).with_suffix('.parquet').name
        return preds_dir / f'{self.model_name}__{vid_name}'

    def get_resume_cache_path(self, video_path) -> Path:
        return self.get_predicted_cache_path(video_path).with_suffix('.resume')

    def get_resume_cache_tmp_path(self, video_path) -> Path:
        resume_path = self.get_resume_cache_path(video_path)
        return resume_path.parent / f'{resume_path.name}.tmp'

    def load_resume_video(self, video_path):
        for resume_path in [self.get_resume_cache_path(video_path), self.get_resume_cache_tmp_path(video_path)]:
            if not resume_path.exists():
                continue
            try:
                return pd.read_parquet(resume_path)
            except Exception:
                continue
        return None

    def save_resume_video(self, df: pd.DataFrame, video_path):
        tmp_path = self.get_resume_cache_tmp_path(video_path)
        resume_path = self.get_resume_cache_path(video_path)
        df.to_parquet(tmp_path)
        tmp_path.replace(resume_path)

    def cleanup_prediction_progress(self, video_path):
        cache_path = self.get_predicted_cache_path(video_path)
        for path in [
            cache_path.with_suffix('.processing'),
            self.get_resume_cache_path(video_path),
            self.get_resume_cache_tmp_path(video_path),
        ]:
            if path.exists():
                path.unlink()

    def save_predicted_video(self, df: pd.DataFrame, video_path):
        cache_path = self.get_predicted_cache_path(video_path)
        df.to_parquet(cache_path)
        self.cleanup_prediction_progress(video_path)

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

import yaml
from pathlib import Path
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

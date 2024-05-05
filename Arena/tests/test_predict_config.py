import config
import yaml
import json
from pathlib import Path
import config
import warnings


class TestPredictConfig:
    pred_config = config.load_configuration('predict')
    
    def run_all(self, pred_config):
        self.pred_config = pred_config
        self.test_main_structure()
        self.test_preds()
        self.test_deeplabcut()

    def test_main_structure(self):
        for pred_name, v in self.pred_config.items():
            assert isinstance(pred_name, str) and isinstance(v, dict), f'bad types for {pred_name}: ({type(pred_name)},{type(v)}) expected (str,dict)'
    
    def test_preds(self):
        mandatory_cols = {'model_path': str, 'threshold': float, 'predictor_name': str}
        for pred_name, d in self.pred_config.items():
            for col, types in mandatory_cols.items():
                assert col in d, f'{pred_name} dict is missing the key "{col}"'
                assert isinstance(d[col], types), f'{pred_name}, {col} must be of type {types}. found: {type(d[col])}'
                if col == 'model_path':
                    assert Path(d[col]).exists(), f'{pred_name}, {col} must be a valid path. found: {d[col]}'
    
    def test_deeplabcut(self):
        dlc_predictors = [k for k, v in self.pred_config.items() if v['predictor_name'] == 'DLCPose']
        if not dlc_predictors:
            return
        
        for pred_name in dlc_predictors:
            dlc_config = self.pred_config[pred_name]
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    from dlclive import DLCLive, Processor
                DLCLive(dlc_config['model_path'], processor=Processor())
            except ImportError:
                raise Exception('dlclive is not installed and deeplabcut model is configured in predict_config')
            except Exception as exc:
                raise Exception(f'Unable to load {pred_name} model from {dlc_config["model_path"]}; {exc}')
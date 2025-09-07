import config
import yaml
from pathlib import Path


class TestCamConfig:
    cam_config = config.cameras
    
    def run_all(self, cam_config):
        self.cam_config = cam_config
        self.test_main_structure()
        self.test_cams()
        self.test_cam_predictors()

    def test_main_structure(self):
        for cam_name, v in self.cam_config.items():
            assert isinstance(cam_name, str) and isinstance(v, dict), f'bad types for {cam_name}: ({type(cam_name)},{type(v)}) expected (str,dict)'

    def test_cams(self):
        mandatory_cols = {'id': (str, int), 'module': str, 'exposure': int, 'image_size': (list, tuple), 'output_dir': (str, type(None))}
        optional_cols = {'mode': str, 'always_on': (int, bool), 'is_color': (int, bool), 'writing_fps': int, 'fps': int, 'predictors': dict,
                         'trigger_source': str}
        for cam_name, d in self.cam_config.items():
            for col, types in mandatory_cols.items():
                assert col in d, f'{cam_name} dict is missing the key "{col}"'
                assert isinstance(d[col], types), f'{cam_name}, {col} must be of type {types}. found: {type(d[col])}'
            for col, types in optional_cols.items():
                if col in d:
                    assert isinstance(d[col], types), f'{cam_name}, {col} must be of type {types}. found: {type(d[col])}'

            assert not d['output_dir'], f'{cam_name} - output_dir must be empty'
            assert d['module'] in ['allied_vision', 'flir', 'file'], f'{cam_name} module: {d["module"]}, but possible options are "allied_vision" and "flir"'
            assert len(d['image_size']) in [2, 3], f'{cam_name} image_size must be list with size 2 or 3'
            if d.get('is_color'):
                assert len(d['image_size']) == 3, f'{cam_name} image_size must be of size 3 in color cameras'
            if d.get('mode'):
                assert d['mode'] in ['tracking', 'manual'], f'{cam_name} mode: {d["mode"]}, but possible options are "tracking" and "manual"'
            assert ('fps' in d) ^ ('trigger_source' in d), f'{cam_name} is configured to both trigger and fps. You must either specify "fps" or "trigger_source"'
            if d.get('fps') and d.get('writing_fps'):
                assert d['fps'] >= d['writing_fps'], f'{cam_name} "fps" must be greater or equal to "writing_fps"'

    def test_cam_predictors(self):
        pconfig = config.load_configuration('predict')
        possible_predictors = list(pconfig.keys())
        for cam_name, d in self.cam_config.items():
            if 'predictors' not in d:
                continue

            for pred_name, pred_dict in d['predictors'].items():
                assert pred_name in possible_predictors, f'Unknwon predictor {pred_name} for {cam_name}. Possible options from predict_config are: {possible_predictors}'
                assert 'mode' in pred_dict, f'predictors dict of {cam_name} lacks the key "mode"'
                assert pred_dict['mode'] in ['experiment', 'no_experiment', 'always'], f'predictors "mode" of {cam_name} must be "experiment", "no_experiment" or "always"'
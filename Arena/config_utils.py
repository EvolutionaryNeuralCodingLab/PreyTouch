import re
import json
from environs import Env
from pathlib import Path
from functools import wraps
from dotenv import set_key
from typing import Dict, Union

_env = Env()
env_path = 'configurations/.env'
_env.read_env(env_path)
if _env.bool('IS_PROD', 0):
    _env.read_env('configurations/.env.prod', override=True)


configurations = {
    'cameras': ('configurations/cam_config.json', ('tests.test_cam_config', 'TestCamConfig')),
    'periphery': ('configurations/periphery_config.json', ('tests.test_periphery', 'TestPeripheryConfig')),
    'predict': ('configurations/predict_config.json', ('tests.test_predict_config', 'TestPredictConfig')),
    'agent': ('configurations/agent_config.json', ('tests.test_agent_config', 'TestAgentConfig'))
}


def load_configuration(conf_name, is_assert_exist=False):
    assert conf_name in configurations, f'Supported configurations are: {list(configurations.keys())}'
    p = Path(configurations[conf_name][0])
    if p.exists():
        with p.open('r') as f:
            d = json.load(f)
    else:
        if is_assert_exist:
            raise FileNotFoundError(f'Configuration file {p} does not exist')
        else:
            with p.open('w') as f:
                json.dump({}, f)
            d = {}
    return d


class Conf:
    def __init__(self):
        self.env_map = {}

    def env_func(func):
        @wraps(func)
        def wrapper(self, env_name, default_value, group='Other', desc='', is_map=True, validator=None, **kwargs):
            env_type = func.__name__
            env_val = getattr(_env, env_type)(env_name, default_value, **kwargs)
            if is_map:
                self.env_map[env_name] = {'value': env_val, 'type': env_type, 'default': default_value, 'group': group, 'desc': desc, 'validator': validator,
                                          'is_changed': env_val != default_value}
            return env_val

        return wrapper

    def set_in_env_file(self, key, map):
        print(f'create new key for {key} in .env file')
        if map.get('validator') is not None:
            self.validate(key, map)
        self.env_map[key] = map
        set_key(env_path, key, self.get_value_as_string(map))

    def get_value_as_string(self, map):
        if map['type'] == 'list':
            return ','.join([str(x) for x in map['value']])
        elif map['type'] == 'bool':
            return str(int(map['value']))
        elif map['type'] == 'dict':
            return ','.join([f'{k}={v}' for k, v  in map['value'].items()])
        else:
            return str(map['value'])

    def validate(self, key, map):
        valid = getattr(Validator, map['validator'])
        try:
            valid(map['value'])
        except Exception as e:
            raise Exception(f'Validation of {key} failed: {e}')

    def get_all_from_cache(self):
        res = {}
        for k, m in self.env_map.items():
            d = res.setdefault(m['group'], {})
            d[k] = m

        res = dict(sorted(res.items()))
        res = {k: dict(sorted(d.items())) for k, d in res.items()}
        return res

    def update_from_api(self, key, value):
        map = self.env_map[key]
        if map['type'] == "__call__":  # string env. variable
            assert isinstance(value, str), f'value received: {value}, expected type: string'
        else:
            try:
                value = eval(value)
            except Exception as e:
                raise Exception(f'value received: {value}, expected type: {map["type"]}')

            if map['type'] == 'list':
                assert isinstance(value, list), f'value received: {value}, expected type: list'
            elif map['type'] in ['int', 'float']:
                assert isinstance(value, (int, float)), f'value received: {value}, expected type: number'
            elif map['type'] == 'bool':
                assert isinstance(value, bool), f'value received: {value}, expected type: boolean'

        map['value'] = value
        self.set_in_env_file(key, map)

    @env_func
    def __call__(self, *args, **kwargs):
        pass

    @env_func
    def bool(self, *args, **kwargs):
        pass

    @env_func
    def int(self, *args, **kwargs):
        pass

    @env_func
    def float(self, *args, **kwargs):
        pass

    @env_func
    def dict(self, *args, **kwargs):
        pass

    @env_func
    def list(self, *args, **kwargs):
        pass


class Validator:
    def hour_validator(x):
        assert re.match(r'\d{2}:\d{2}', x), 'Hour format is 00:00'

    def cam_exist(cam_name):
        if not cam_name:
            return
        cam_config = load_configuration('cameras')
        assert cam_name in cam_config, f'Camera {cam_name} does not exist'

    def predict_model_exist(pred_name):
        if not pred_name:
            return
        pred_config = load_configuration('predict')
        assert pred_name in pred_config, f'Predictor {pred_name} does not exist'
        
        

def _fill_gate_defaults(g: Dict) -> Dict:
    """Fill missing keys and host/port defaults."""
    g = (g or {}).copy()
    g.setdefault('topic', 'arena/value')
    g.setdefault('field', 'day_lights')
    g.setdefault('edge',  'rising')
    g.setdefault('debounce_ms', 300)

    # Prefer values from .env via _env (config_utils already loads .env)
    try:
        host = _env('MQTT_HOST', 'localhost')
    except Exception:
        host = 'localhost'
    try:
        port = _env.int('MQTT_PORT', 1883)
    except Exception:
        port = 1883
    g.setdefault('host', host)
    g.setdefault('port', int(port))
    return g

def parse_gated_trigger(val: Union[str, Dict, None]) -> Dict:
    """
    Accept a dict or a JSON string (e.g. from GATED_BLOCK_TRIGGER env).
    Returns a dict with sane defaults merged.
    """
    if not val:
        return _fill_gate_defaults({})
    if isinstance(val, dict):
        return _fill_gate_defaults(val)
    if isinstance(val, str):
        try:
            return _fill_gate_defaults(json.loads(val))
        except Exception:
            # Optional compact "topic:field:edge:debounce_ms" form
            parts = [p.strip() for p in val.split(':')]
            g: Dict[str, Union[str, int]] = {}
            if len(parts) >= 1: g['topic'] = parts[0] or 'arena/value'
            if len(parts) >= 2: g['field'] = parts[1] or 'day_lights'
            if len(parts) >= 3: g['edge']  = parts[2] or 'rising'
            if len(parts) >= 4:
                try: g['debounce_ms'] = int(parts[3])
                except: g['debounce_ms'] = 300
            return _fill_gate_defaults(g)
    return _fill_gate_defaults({})        
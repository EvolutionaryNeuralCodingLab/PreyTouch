import pytest
import json
from pathlib import Path
import config
import yaml
import re
import pandas as pd
from pathlib import Path
import config


class TestPeripheryConfig:
    periphery_config = config.load_configuration('periphery', is_assert_exist=False)
    
    # def test_exist(self):
    #     if not self.periphery_config:
    #         with open(config.configurations['periphery'][0], 'w') as f:
    #             json.dump({}, f)

    def run_all(self, periphery_config):
        self.periphery_config = periphery_config
        self.test_structure()

    def test_structure(self):
        if not self.periphery_config:
            return

        assert 'arena' in self.periphery_config
        # assert 'camera_trigger' in cfg
        for i, device in enumerate(self.periphery_config['arena']['interfaces']):
            assert 'name' in device, f'Device #{i + 1} has no name'
            device_name = device['name']
            assert 'type' in device, f'Device {device_name} has no "type"'
            assert 'pins' in device or 'pin' in device, f'Device {device_name} has no "pin" or "pins"'
            if device['type'] == 'feeder':
                assert 'order' in device, f'Feeder {device["name"]} must have "order" field'

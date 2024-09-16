import config
import yaml
import re
from pathlib import Path
import config


class TestAgentConfig:
    agent_config = config.load_configuration('agent')
    
    def run_all(self, agent_config):
        self.agent_config = agent_config
        self.test_main_structure()
        self.test_times()
        self.test_trials()

    def test_main_structure(self):
        if not self.agent_config:
            config.IS_AGENT_ENABLED = False
            print('Agent config is empty; Setting IS_AGENT_ENABLED=False in config')
            return
        main_keys = ['trials', 'default_struct', 'times']
        for k in main_keys:
            assert k in self.agent_config, f'{k} must be in agent config'
            assert isinstance(self.agent_config[k], dict), f'{k} must be a dictionary'
    
    def test_times(self):
        if not self.agent_config:
            return
        times_keys = ['start_time', 'end_time', 'time_between_experiments']
        tms = self.agent_config['times']
        for k in times_keys:
            assert k in tms, f'{k} must be in times dict'
            if k in ['start_time', 'end_time']:
                assert re.match(r'\d{2}:\d{2}', tms[k]), 'format of times.{k} must be "00:00"'
    
    def test_trials(self):
        if not self.agent_config:
            return
        for trial_name, trial_dict in self.agent_config['trials'].items():
            assert 'count' in trial_dict, f'"count" must be in trial {trial_name}'
            for k in ['key', 'amount']:
                assert k in trial_dict['count'], f'{k} must be in "count" of trial {trial_name}'

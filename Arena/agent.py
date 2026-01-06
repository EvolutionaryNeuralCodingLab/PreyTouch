import time
import yaml
import json
import copy
from datetime import datetime, timedelta
from pathlib import Path
import re
import utils
from db_models import ORM, Block, Experiment
from cache import RedisCache, CacheColumns as cc
from experiment import ExperimentCache, ExperimentValidation
from loggers import get_logger
import config

EXIT_HOLES = ['left', 'right']


class Agent:
    def __init__(self, orm=None):
        self.orm = orm if orm is not None else ORM()
        self.logger = get_logger('Agent')
        self.cache = RedisCache()
        self.exp_validation = ExperimentValidation(logger=self.logger, orm=self.orm, cache=self.cache, is_silent=True)
        self.animal_id = None
        self.history = {}
        self.next_trial_name = None

        if Path(config.configurations['agent'][0]).exists():
            agent_config = config.load_configuration('agent')
            self.check_agent_config(agent_config)
            self.trials = agent_config['trials']
            self.default_struct = agent_config['default_struct']
            self.times = agent_config['times']
            self.history_scope = agent_config.get('history_scope', 'all')
            self.schedule_mode = agent_config.get('schedule_mode', 'next_available')
        else:
            self.trials = {}

    def update(self, animal_id=None):
        if not self.trials:
            return
        if self.cache.get_current_experiment():
            return
        self.animal_id = animal_id or self.cache.get(cc.CURRENT_ANIMAL_ID)
        if self.animal_id == 'test':
            return
        self.init_history()
        self.load_history()
        if self.cache.get(cc.HOLD_AGENT):  # stop here if agent is on hold
            return
        self.next_trial_name = self.get_next_trial_name()
        if not self.next_trial_name:
            # all experiments are over
            party_emoji = u'\U0001F389'
            self.publish(f'Animal {self.animal_id} has finished all its experiments {party_emoji}')
            return
        self.create_cached_experiment()
        if self.exp_validation.is_ready():
            self.schedule_next_block()
        else:
            error_msg = f'Unable to schedule an experiment using agent since the following checks failed: ' \
                        f'{",".join(self.exp_validation.failed_checks)}'
            self.publish(error_msg)

    def schedule_next_block(self):
        next_schedules = self.get_upcoming_agent_schedules()
        if next_schedules:
            # if there are scheduled agent trials, do nothing
            return

        schedule_time = self.get_next_schedule_time()
        if not schedule_time:
            return
        self.orm.commit_schedule(schedule_time, self.cached_experiment_name)

    def get_next_trial_name(self):
        for trial_name in self.trials:
            if self.is_trial_type_finished(trial_name):
                continue
            else:
                return trial_name

    def get_upcoming_agent_schedules(self):
        res = {}
        for s in self.orm.get_upcoming_schedules().all():
            # ignore all the non-experiment schedules (e.g. SWITCH:, FEEDER:,...)
            if not re.match(r'[A-Z]+\:.*', s.experiment_name):
                res[s.experiment_name] = s.date
        return res

    def get_possible_times(self):
        now = datetime.now()
        start_hour, start_minute = self.times['start_time'].split(':')
        dt = now.replace(hour=int(start_hour), minute=int(start_minute), second=0, microsecond=0)
        end_hour, end_minute = self.times['end_time'].split(':')
        end_dt = now.replace(hour=int(end_hour), minute=int(end_minute), second=0, microsecond=0)
        possible_times = []
        while dt <= end_dt:
            if dt >= now:
                possible_times.append(dt)
            dt += timedelta(minutes=self.times['time_between_experiments'])
        return possible_times

    def get_next_schedule_time(self):
        if self.schedule_mode == 'trial_order':
            return self.get_trial_order_time(self.next_trial_name)
        possible_times = self.get_possible_times()
        return possible_times[0] if possible_times else None

    def get_trial_order_time(self, trial_name):
        if not trial_name or trial_name not in self.trials:
            return None
        now = datetime.now()
        start_hour, start_minute = self.times['start_time'].split(':')
        start_dt = now.replace(hour=int(start_hour), minute=int(start_minute), second=0, microsecond=0)
        end_hour, end_minute = self.times['end_time'].split(':')
        end_dt = now.replace(hour=int(end_hour), minute=int(end_minute), second=0, microsecond=0)
        try:
            idx = list(self.trials.keys()).index(trial_name)
        except ValueError:
            return None
        slot_dt = start_dt + timedelta(minutes=self.times['time_between_experiments']) * idx
        if slot_dt > end_dt:
            self.logger.warning(f'No available slot for {trial_name}; slot {slot_dt} beyond end_time {end_dt}')
            return None
        if slot_dt <= now:
            if now > end_dt:
                return None
            schedule_dt = now + timedelta(seconds=5)
            if schedule_dt > end_dt:
                return None
            return schedule_dt
        return slot_dt

    def init_history(self):
        self.history = {}
        for trial_name, trial_dict in self.trials.items():
            self.history[trial_name] = {'key': trial_dict['count']['key'], 'count_target': trial_dict['count']['amount']}
            if 'per' in trial_dict['count']:
                self.history[trial_name]['counts'] = {}
                for group_name, group_vals in trial_dict['count']['per'].items():
                    self.history[trial_name]['counts'][group_name] = {k: 0 for k in group_vals}
            else:
                self.history[trial_name]['counts'] = 0

    def load_history(self):
        # create a dict of movement types and their relevant trial names
        agent_trial_names = list(self.trials.keys())
        # movements = {}
        # for trial_name, trial_dict in self.trials.items():
        #     movements.setdefault(trial_dict['movement_type'], []).append(trial_name)

        with self.orm.session() as s:
            exp_query = s.query(Experiment).filter_by(animal_id=self.animal_id)
            if self.history_scope == 'today':
                day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                day_end = day_start + timedelta(days=1)
                exp_query = exp_query.filter(Experiment.start_time >= day_start, Experiment.start_time < day_end)
            exps = exp_query.all()
            for exp in exps:
                for blk in exp.blocks:
                    if blk.agent_label in agent_trial_names:
                        trial_name = blk.agent_label
                        count_key = self.history[trial_name]['key']
                        counts = self.history[trial_name]['counts']

                        if count_key == 'engaged_trials':
                            blk_count = len([tr for tr in blk.trials if len(tr.strikes) > 0])
                        else:
                            blk_count = len(getattr(blk, count_key))

                        if isinstance(counts, dict):  # case of per
                            for metric_name, metric_counts in counts.items():
                                if metric_name == 'bug_speed':
                                    self.parse_bug_speed_from_trials(blk, metric_counts, count_key)
                                else:
                                    if getattr(blk, metric_name) in metric_counts:
                                        metric_counts[getattr(blk, metric_name)] += blk_count
                        elif isinstance(counts, int):
                            self.history[trial_name]['counts'] += blk_count

    @staticmethod
    def parse_bug_speed_from_trials(blk, metric_counts, count_key):
        """bug speed is not saved in the block level, thus the count is done over the trials"""
        for tr in blk.trials:
            trial_speed = getattr(tr, 'bug_speed')
            if trial_speed in metric_counts:
                if count_key == 'engaged_trials':
                    if len(tr.strikes) > 0:
                        metric_counts[trial_speed] += 1
                else:
                    metric_counts[trial_speed] += len(getattr(tr, count_key))

    def publish(self, msg):
        last_publish = self.cache.get(cc.LAST_TIME_AGENT_MESSAGE)
        if not last_publish or time.time() - float(last_publish) > config.AGENT_MIN_DURATION_BETWEEN_PUBLISH:
            self.logger.error(msg)
            utils.send_telegram_message(f'Agent Message:\n{msg}')
            self.cache.set(cc.LAST_TIME_AGENT_MESSAGE, time.time())

    def get_animal_history(self):
        txt = f'Animal ID: {self.animal_id}\n'
        for trial_name in self.history:
            self.history[trial_name]['is_finished'] = self.is_trial_type_finished(trial_name)
        txt += json.dumps(self.history, indent=4)
        return txt

    def create_cached_experiment(self):
        # load the agent config
        block_dict_ = self.trials[self.next_trial_name].copy()
        block_dict_['agent_label'] = self.next_trial_name
        count_dict = block_dict_.pop('count')
        for k, v in block_dict_.copy().items():
            if isinstance(v, str) and v.startswith('per_'):
                per_left = [x for x in count_dict['per'][k]
                            if self.history[self.next_trial_name]['counts'][k][x] < count_dict['amount']]
                if v == 'per_random':
                    block_dict_[k] = per_left
                elif v == 'per_ordered':
                    block_dict_[k] = per_left[0]

        json_struct = copy.deepcopy(self.default_struct)
        json_struct['blocks'][0].update(block_dict_)
        exp_name = self.save_cached_experiment(json_struct)
        return exp_name

    def save_cached_experiment(self, trial_dict):
        trial_dict['name'] = self.cached_experiment_name
        ExperimentCache().save(trial_dict)
        return trial_dict['name']

    @property
    def cached_experiment_name(self):
        return f'agent_{self.next_trial_name}'

    def is_trial_type_finished(self, trial_name):
        count_dict = self.trials[trial_name]['count']
        is_finished = False
        history = self.history[trial_name]['counts']
        if isinstance(history, dict):
            is_finished = all(v >= count_dict['amount'] for group_name, group_vals in history.items()
                              for v in group_vals.values())
        elif isinstance(history, (int, float)):
            is_finished = history >= count_dict['amount']
        return is_finished

    def check_agent_config(self, agent_config):
        main_keys = ['trials', 'default_struct', 'times']
        for k in main_keys:
            assert k in agent_config, f'{k} must be in agent config'

        times_keys = ['start_time', 'end_time', 'time_between_experiments']
        for k in times_keys:
            assert k in agent_config['times'], f'{k} must be in times'

        for trial_name, trial_dict in agent_config['trials'].items():
            assert 'count' in trial_dict, f'"count" must be in trial {trial_name}'
            for k in ['key', 'amount']:
                assert k in trial_dict['count'], f'{k} must be in "count" of trial {trial_name}'


if __name__ == '__main__':
    ag = Agent()
    ag.update()
    print(ag.get_animal_history())

import time
import yaml
import json
import copy
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
import re
import utils
from db_models import ORM, Block, Experiment, Trial
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
        self.success_announce = {}

        if Path(config.configurations['agent'][0]).exists():
            agent_config = config.load_configuration('agent')
            self.check_agent_config(agent_config)
            self.default_struct = agent_config['default_struct']
            self.times = agent_config['times']
            self.history_scope = agent_config.get('history_scope', 'all')
            self.schedule_mode = agent_config.get('schedule_mode', 'next_available')
            self.success_announce = agent_config.get('success_announce', {})
            self.trials = self.expand_trials(agent_config['trials'])
        else:
            self.trials = {}

    def update(self, animal_id=None):
        if not self.trials:
            self.set_last_error('Agent trials config is empty')
            return
        self.animal_id = animal_id or self.cache.get(cc.CURRENT_ANIMAL_ID)
        if not self.animal_id:
            self.set_last_error('No current animal ID set')
            return
        if self.animal_id == 'test':
            self.set_last_error('Agent disabled for test animal')
            return
        self.exp_validation.animal_id = self.animal_id
        if self.exp_validation.is_max_reward_reached():
            self.clear_upcoming_agent_schedules()
            self.set_last_error(
                f'Max daily rewards of {config.MAX_DAILY_REWARD} reached; agent will not schedule more experiments today'
            )
            return
        self.check_daily_success_announce()
        if self.cache.get_current_experiment():
            if not self.clear_stale_experiment_cache():
                self.set_last_error('Experiment already running')
                return
        self.init_history()
        self.load_history()
        if self.cache.get(cc.HOLD_AGENT):  # stop here if agent is on hold
            self.set_last_error('Agent is on hold')
            return
        if not self.is_within_time_window():
            self.clear_last_error()
            return
        self.next_trial_name = self.get_next_trial_name()
        if not self.next_trial_name:
            # all experiments are over
            party_emoji = u'\U0001F389'
            self.set_last_error('All agent trials are complete')
            self.publish(f'Animal {self.animal_id} has finished all its experiments {party_emoji}')
            return
        self.create_cached_experiment()
        if self.exp_validation.is_ready():
            self.schedule_next_block()
        else:
            error_msg = f'Unable to schedule an experiment using agent since the following checks failed: ' \
                        f'{",".join(self.exp_validation.failed_checks)}'
            self.set_last_error(error_msg)
            self.publish(error_msg)

    def schedule_next_block(self):
        next_schedules = self.get_upcoming_agent_schedules()
        if next_schedules:
            # if there are scheduled agent trials, do nothing
            self.clear_last_error()
            return

        schedule_time = self.get_next_schedule_time()
        if not schedule_time:
            self.set_last_error('No available schedule time')
            return
        self.orm.commit_schedule(schedule_time, self.cached_experiment_name)
        self.clear_last_error()

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

    def clear_upcoming_agent_schedules(self):
        for s in self.orm.get_upcoming_schedules().all():
            if s.experiment_name.startswith('agent_'):
                self.orm.delete_schedule(s.id)

    def get_daily_time_slots(self):
        now = datetime.now()
        start_hour, start_minute = self.times['start_time'].split(':')
        dt = now.replace(hour=int(start_hour), minute=int(start_minute), second=0, microsecond=0)
        end_hour, end_minute = self.times['end_time'].split(':')
        end_dt = now.replace(hour=int(end_hour), minute=int(end_minute), second=0, microsecond=0)
        slots = []
        while dt <= end_dt:
            slots.append(dt)
            dt += timedelta(minutes=self.times['time_between_experiments'])
        return slots

    def is_within_time_window(self):
        now = datetime.now()
        start_hour, start_minute = self.times['start_time'].split(':')
        start_dt = now.replace(hour=int(start_hour), minute=int(start_minute), second=0, microsecond=0)
        end_hour, end_minute = self.times['end_time'].split(':')
        end_dt = now.replace(hour=int(end_hour), minute=int(end_minute), second=0, microsecond=0)
        return start_dt <= now <= end_dt

    def get_possible_times(self):
        now = datetime.now()
        return [dt for dt in self.get_daily_time_slots() if dt >= now]

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

    def expand_trials(self, trials):
        if not trials:
            return trials
        needs_expansion = any(
            ('repeat' in trial_dict) or ('name_template' in trial_dict) or ('{time}' in trial_name)
            for trial_name, trial_dict in trials.items()
        )
        if not needs_expansion:
            return trials

        slots = self.get_daily_time_slots()
        total_slots = len(slots)
        repeats = {}
        fill_trials = []
        for trial_name, trial_dict in trials.items():
            repeat = trial_dict.get('repeat', 1)
            if isinstance(repeat, str):
                assert repeat == 'fill', f'repeat must be int or "fill" for trial {trial_name}'
                repeats[trial_name] = None
                fill_trials.append(trial_name)
            else:
                assert isinstance(repeat, int), f'repeat must be int or "fill" for trial {trial_name}'
                assert repeat >= 0, f'repeat must be >= 0 for trial {trial_name}'
                repeats[trial_name] = repeat

        if fill_trials:
            assert len(fill_trials) == 1, 'Only one trial can use repeat="fill"'
            fixed_count = sum(v for v in repeats.values() if v is not None)
            fill_count = total_slots - fixed_count
            assert fill_count >= 0, 'repeat="fill" exceeds available time slots'
            repeats[fill_trials[0]] = fill_count

        expanded = OrderedDict()
        slot_index = 0
        for trial_name, trial_dict in trials.items():
            repeat_count = repeats[trial_name]
            if repeat_count == 0:
                continue
            for i in range(repeat_count):
                slot_time = slots[slot_index] if slot_index < len(slots) else None
                expanded_name = self.format_trial_name(trial_name, trial_dict, slot_time, i, repeat_count)
                expanded_name = self.ensure_unique_trial_name(expanded_name, expanded)
                expanded[expanded_name] = {k: v for k, v in trial_dict.items()
                                           if k not in ['repeat', 'name_template']}
                slot_index += 1
        return expanded

    @staticmethod
    def format_trial_name(base_name, trial_dict, slot_time, repeat_index, repeat_count):
        template = trial_dict.get('name_template', base_name)
        name = template
        time_token = slot_time.strftime('%H%M') if slot_time else None
        if '{time}' in name:
            name = name.replace('{time}', time_token or f'{repeat_index + 1:02d}')
        if '{index}' in name:
            name = name.replace('{index}', str(repeat_index + 1))
        if repeat_count > 1 and name == template:
            if time_token:
                name = f'{name}_{time_token}'
            else:
                name = f'{name}_{repeat_index + 1}'
        return name

    @staticmethod
    def ensure_unique_trial_name(trial_name, trial_map):
        if trial_name not in trial_map:
            return trial_name
        suffix = 2
        while f'{trial_name}_{suffix}' in trial_map:
            suffix += 1
        return f'{trial_name}_{suffix}'

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

    def should_evaluate_success(self, agent_label):
        if not agent_label:
            return False
        trial_cfg = self.trials.get(agent_label)
        if not trial_cfg:
            return False
        return trial_cfg.get('evaluate_success', True)

    def check_daily_success_announce(self):
        if not self.success_announce or self.success_announce.get('enabled', True) is False:
            return
        if config.DISABLE_DB:
            return
        if not self.animal_id:
            return
        day_string = datetime.now().strftime('%Y-%m-%d')
        if self.cache.get(cc.DAILY_SUCCESS_ANNOUNCED_DATE) == day_string:
            return

        successes, trials_with_strikes = self.get_daily_success_stats()
        min_trials = self.success_announce.get('min_trials_with_strikes', 5)
        threshold = self.success_announce.get('success_threshold', 0.8)
        if trials_with_strikes < min_trials:
            return
        if trials_with_strikes == 0:
            return
        rate = successes / trials_with_strikes
        if rate < threshold:
            return
        msg = (f'Animal {self.animal_id} reached {rate:.0%} daily success '
               f'({successes}/{trials_with_strikes} trials with strikes; '
               f'last strike is reward bug)')
        self.publish(msg, force=True)
        self.cache.set(cc.DAILY_SUCCESS_ANNOUNCED_DATE, day_string)

    def get_daily_success_stats(self):
        day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        successes = 0
        trials_with_strikes = 0
        with self.orm.session() as s:
            trials = s.query(Trial, Block).join(
                Block, Block.id == Trial.block_id).join(
                Experiment, Experiment.id == Block.experiment_id).filter(
                Trial.start_time >= day_start,
                Trial.start_time < day_end,
                Experiment.animal_id == self.animal_id,
                Experiment.arena == config.ARENA_NAME,
                Block.block_type == 'bugs'
            ).all()
            for tr, blk in trials:
                if not self.should_evaluate_success(blk.agent_label):
                    continue
                if not tr.strikes:
                    continue
                trials_with_strikes += 1
                last_strike = max(tr.strikes, key=lambda s: s.time or datetime.min)
                if last_strike.is_reward_bug:
                    successes += 1
        return successes, trials_with_strikes

    def publish(self, msg, force=False):
        last_publish = self.cache.get(cc.LAST_TIME_AGENT_MESSAGE)
        if force or not last_publish or time.time() - float(last_publish) > config.AGENT_MIN_DURATION_BETWEEN_PUBLISH:
            self.logger.error(msg)
            utils.send_telegram_message(f'Agent Message:\n{msg}')
            self.cache.set(cc.LAST_TIME_AGENT_MESSAGE, time.time())

    def set_last_error(self, msg):
        if msg:
            stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.cache.set(cc.LAST_AGENT_ERROR, f'{stamp} - {msg}')
        else:
            self.clear_last_error()

    def clear_last_error(self):
        self.cache.delete(cc.LAST_AGENT_ERROR)

    def clear_stale_experiment_cache(self):
        if config.DISABLE_DB:
            return False
        animal_id = self.animal_id or self.cache.get(cc.CURRENT_ANIMAL_ID)
        if not animal_id:
            return False
        try:
            with self.orm.session() as s:
                running_exp = s.query(Experiment).filter_by(animal_id=animal_id,
                                                           arena=config.ARENA_NAME,
                                                           end_time=None).first()
            if running_exp:
                return False
            for col in [cc.EXPERIMENT_NAME, cc.EXPERIMENT_PATH, cc.EXPERIMENT_BLOCK_ID, cc.EXPERIMENT_BLOCK_PATH,
                        cc.IS_ALWAYS_REWARD, cc.IS_EXPERIMENT_CONTROL_CAMERAS, cc.IS_REWARD_TIMEOUT,
                        cc.IS_VISUAL_APP_ON, cc.CURRENT_BLOCK_DB_INDEX]:
                self.cache.delete(col)
            self.logger.warning('Cleared stale experiment cache to unblock agent scheduling')
            return True
        except Exception as exc:
            self.logger.warning(f'Unable to verify stale experiment cache; {exc}')
            return False

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
            if 'name_template' in trial_dict:
                assert isinstance(trial_dict['name_template'], str), f'name_template must be a string in {trial_name}'
            repeat = trial_dict.get('repeat', 1)
            if isinstance(repeat, str):
                assert repeat == 'fill', f'repeat must be int or "fill" for trial {trial_name}'
            else:
                assert isinstance(repeat, int), f'repeat must be int or "fill" for trial {trial_name}'
                assert repeat >= 0, f'repeat must be >= 0 for trial {trial_name}'
            if 'evaluate_success' in trial_dict:
                assert isinstance(trial_dict['evaluate_success'], bool), \
                    f'evaluate_success must be a boolean in {trial_name}'

        fill_trials = [name for name, trial in agent_config['trials'].items()
                       if isinstance(trial.get('repeat', 1), str) and trial.get('repeat') == 'fill']
        if fill_trials:
            assert len(fill_trials) == 1, 'Only one trial can use repeat="fill"'
            slots_count = 0
            start_hour, start_minute = agent_config['times']['start_time'].split(':')
            start_dt = datetime.now().replace(hour=int(start_hour), minute=int(start_minute), second=0, microsecond=0)
            end_hour, end_minute = agent_config['times']['end_time'].split(':')
            end_dt = datetime.now().replace(hour=int(end_hour), minute=int(end_minute), second=0, microsecond=0)
            dt = start_dt
            while dt <= end_dt:
                slots_count += 1
                dt += timedelta(minutes=agent_config['times']['time_between_experiments'])
            fixed_count = 0
            for trial_name, trial_dict in agent_config['trials'].items():
                repeat = trial_dict.get('repeat', 1)
                if isinstance(repeat, int):
                    fixed_count += repeat
            assert slots_count >= fixed_count, 'repeat="fill" exceeds available time slots'


if __name__ == '__main__':
    ag = Agent()
    ag.update()
    print(ag.get_animal_history())

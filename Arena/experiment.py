import os
import subprocess
import signal
import argparse
import inspect
import random
import websocket
import json
import threading
import time
import cv2
from typing import Union
import humanize
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from multiprocessing.pool import ThreadPool
from threading import Event
from pathlib import Path
import yaml
import requests
import pandas as pd

import config
import utils
from loggers import get_logger
from cache import RedisCache, CacheColumns as cc
from utils import mkdir, to_integer, turn_display_on, turn_display_off, run_command, get_hdmi_xinput_id, get_psycho_files
from subscribers import Subscriber, start_experiment_subscribers
from periphery_integration import PeripheryIntegrator
from db_models import ORM, Experiment as Experiment_Model, Strike


@dataclass
class Experiment:
    cam_units: dict
    animal_id: str
    cameras: dict
    num_blocks: int = 1
    name: str = ''
    blocks: list = field(default_factory=list, repr=False)
    time_between_blocks: int = config.TIME_BETWEEN_BLOCKS
    extra_time_recording: int = config.EXTRA_TIME_RECORDING
    is_identical_blocks: bool = False
    is_test: bool = False
    cache = RedisCache()

    def __post_init__(self):
        self.start_time = datetime.now()
        if self.is_test:
            self.animal_id = 'test'
        self.day = self.start_time.strftime('%Y%m%d')
        self.name = str(self)
        self.orm = ORM()
        blocks_ids = range(self.first_block, self.first_block + len(self.blocks))
        self.blocks = [Block(i, self.cameras, str(self), self.experiment_path, self.animal_id, self.cam_units, self.orm,
                             self.cache, extra_time_recording=self.extra_time_recording, **kwargs)
                       for i, kwargs in zip(blocks_ids, self.blocks)]
        self.logger = get_logger('Experiment')
        self.exp_validation = ExperimentValidation(logger=self.logger, cache=self.cache, orm=self.orm)
        self.threads = {}
        self.experiment_stop_flag = threading.Event()
        self.init_experiment_cache()

    def __str__(self):
        return f'EXP{self.day}'

    @property
    def first_block(self):
        mkdir(self.experiment_path)
        blocks = Path(self.experiment_path).glob('block*')
        blocks = [to_integer(x.name.split('block')[-1]) for x in blocks]
        blocks = sorted([b for b in blocks if isinstance(b, int)])
        return blocks[-1] + 1 if blocks else 1

    @property
    def info(self):
        non_relevant_fields = ['self', 'blocks', 'cache']
        info = {k: getattr(self, k) for k in get_arguments(self) if k not in non_relevant_fields}
        for block in self.blocks:
            info[f'block{block.block_id}'] = block.info

        return info

    def start(self):
        """Main Function for starting an experiment"""
        def _start():
            self.logger.info(f'Experiment started for {humanize.precisedelta(self.experiment_duration)}'
                             f' with cameras: {",".join(self.cameras.keys())}')
            self.orm.commit_experiment(self)
            self.turn_screen('on', board=self.get_board())
            if not config.IS_CHECK_SCREEN_MAPPING:
                # if you don't check the screen's mapping, map it again in each experiment
                self.exp_validation.map_touchscreen_to_hdmi(is_display_on=True)

            try:
                for i, block in enumerate(self.blocks):
                    if i > 0:
                        time.sleep(self.time_between_blocks)
                    block.start()
            except EndExperimentException as exc:
                self.logger.warning(f'Experiment stopped externally; {exc}')

            self.turn_screen('off')
            time.sleep(3)
            self.orm.update_experiment_end_time()
            self.cache.publish_command('end_experiment')

        if not self.exp_validation.is_ready():
            self.stop_experiment()
            return
        self.threads['experiment_stop'] = Subscriber(self.experiment_stop_flag,
                                                     channel=config.subscription_topics['end_experiment'],
                                                     callback=self.stop_experiment)

        self.threads['main'] = threading.Thread(target=_start)
        [t.start() for t in self.threads.values()]
        return str(self)

    def stop_experiment(self, *args):
        self.logger.debug('closing experiment...')
        self.experiment_stop_flag.set()
        self.cache.delete(cc.IS_VISUAL_APP_ON)
        self.cache.delete(cc.EXPERIMENT_BLOCK_ID)
        self.cache.delete(cc.EXPERIMENT_BLOCK_PATH)
        self.cache.delete(cc.IS_ALWAYS_REWARD)
        self.cache.delete(cc.EXPERIMENT_NAME)
        self.cache.delete(cc.EXPERIMENT_PATH)
        self.cache.delete(cc.IS_EXPERIMENT_CONTROL_CAMERAS)
        self.cache.delete(cc.IS_REWARD_TIMEOUT)
        if 'main' in self.threads:
            self.threads['main'].join()
        time.sleep(0.2)
        self.logger.info('Experiment ended')

    def save(self, data):
        """Save experiment arguments"""

    def turn_screen(self, val, board='holes'):
        """val must be on or off"""
        if config.DISABLE_ARENA_SCREEN or board == 'blank':
            return
        assert val in ['on', 'off'], 'val must be either "on" or "off"'
        try:
            if val.lower() == 'on':
                turn_display_on(board, is_test=self.is_test)
            else:
                turn_display_off(app_only=self.is_test)
            self.logger.debug(f'screen turned {val}')
        except Exception as exc:
            self.logger.exception(f'Error turning off screen: {exc}')

    def init_experiment_cache(self):
        self.cache.set(cc.EXPERIMENT_NAME, str(self), timeout=self.experiment_duration)
        self.cache.set(cc.EXPERIMENT_PATH, self.experiment_path, timeout=self.experiment_duration)
        self.cache.set(cc.IS_EXPERIMENT_CONTROL_CAMERAS, True, timeout=self.experiment_duration)
        if self.is_test:
            # cancel reward in test experiments
            self.cache.set(cc.IS_REWARD_TIMEOUT, True, timeout=self.experiment_duration)
        else:
            self.cache.set(cc.IS_REWARD_TIMEOUT, False)

    @property
    def experiment_path(self):
        return utils.get_todays_experiment_dir(self.animal_id)

    @property
    def experiment_duration(self):
        return int(sum(b.overall_block_duration for b in self.blocks) + self.time_between_blocks * (len(self.blocks) - 1))

    def get_board(self):
        with open('../pogona_hunter/src/config.json', 'r') as f:
            app_config = json.load(f)

        block_type = self.blocks[0].block_type
        if block_type in ['psycho', 'blank', 'media']:
            return block_type
        curr_mt = self.blocks[0].movement_type
        curr_board = None
        for board, boards_mts in app_config['boards'].items():
            if curr_mt in boards_mts:
                curr_board = board
        if not curr_board:
            raise EndExperimentException(f'unable to find board for movement type: {curr_mt}')
        return curr_board


@dataclass
class Block:
    block_id: int
    cameras: dict
    experiment_name: str
    experiment_path: str
    animal_id: str
    cam_units: dict
    orm: ORM
    cache: RedisCache
    periphery: PeripheryIntegrator = None
    extra_time_recording: int = config.EXTRA_TIME_RECORDING
    start_time = None
    bug_types: list = field(default_factory=list)
    num_trials: int = 1
    trial_duration: int = 10
    iti: int = 10
    notes: str = ''
    background_color: str = ''
    block_type: str = 'bugs'  # options: 'bugs', 'blank', 'media', 'psycho'

    num_of_bugs: int = 1
    is_split_bugs_view: bool = False
    split_repeated_pos_ratio: float = 1
    split_bugs_order: list = None
    split_randomize_timing: bool = True
    movement_type: str = None
    bug_speed: [int, list] = None
    bug_size: int = None
    holes_height_scale: float = 0.1
    circle_height_scale: float = 0.5
    circle_radius_scale: float = 0.2
    time_between_bugs: int = None
    is_default_bug_size: bool = True
    exit_hole: str = None
    reward_type: str = 'always'
    reward_bugs: list = None
    reward_any_touch_prob: float = 0.0
    agent_label: str = None
    accelerate_multiplier: float = 3.0

    media_url: str = ''

    psycho_file: str = ''
    psycho_proc_pid: int = None

    blank_rec_type: str = 'trials'  # options: 'trials', 'continuous'
    trial_images = []

    def __post_init__(self):
        self.logger = get_logger(f'{self.experiment_name}-Block {self.block_id}')
        self.exp_validation = ExperimentValidation(logger=self.logger, cache=self.cache, orm=self.orm)
        if isinstance(self.bug_speed, list):
            bug_speed_choices = ','.join([str(s) for s in self.bug_speed])
            # self.bug_speed = random.choice(self.bug_speed)
            self.logger.info(f'Multiple bug speeds. options: {bug_speed_choices}')
        elif self.block_type == 'bugs':
            self.logger.info(f'Start block with bug speed: {self.bug_speed}')
        if self.periphery is None:
            self.periphery = PeripheryIntegrator()
        if isinstance(self.bug_types, str):
            self.bug_types = self.bug_types.split(',')
        if isinstance(self.reward_bugs, str):
            self.reward_bugs = self.reward_bugs.split(',')
        elif not self.reward_bugs:
            self.logger.debug(f'No reward bugs were given, using all bug types as reward; {self.reward_bugs}')
            self.reward_bugs = self.bug_types

        if self.is_continuous_blank:
            self.num_trials, self.iti = 1, 0
            self.trial_duration = config.MAX_DURATION_CONT_BLANK
        if self.is_long_recording:
            if not self.is_continuous_blank:
                self.logger.warning('Block is very long. switched to compressed video writer')
            self.cache.set(cc.IS_COMPRESSED_LONG_RECORDING, True, timeout=self.overall_block_duration)

    @property
    def info(self):
        non_relevant_fields = ['self', 'cache', 'is_use_predictions', 'cameras', 'experiment_path', 'orm', 'cam_units',
                               'periphery']
        for block_type, block_type_fields in config.experiment_types.items():
            if block_type != self.block_type:
                non_relevant_fields += block_type_fields
        info = {k: getattr(self, k) for k in get_arguments(self) if k not in non_relevant_fields}
        info['start_time'] = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        info['version'] = config.version
        return info

    def start(self):
        self.start_time = datetime.now()
        self.logger.debug('block start')
        self.orm.commit_block(self)

        mkdir(self.block_path)
        if self.is_always_reward:
            self.cache.set(cc.IS_ALWAYS_REWARD, True, timeout=self.block_duration)
        self.hide_visual_app_content()
        try:
            self.run_block()
        except EndExperimentException as exc:
            self.hide_visual_app_content()
            self.logger.warning('block stopped externally')
            raise exc
        finally:
            self.orm.update_block_end_time()
            self.end_block()

        self.logger.info(self.block_summary)

    def run_block(self):
        """Run block flow"""
        self.init_block()
        self.wait(self.extra_time_recording, label='Extra Time Rec')
        # if self.is_split_bugs_view, take self.ratio and change the order of self.bug_types in each trial by creating
        # an array the size of the trial that combinatorially put 1 and -1, where 1 indicates the current bugs order and -1 indicates the opposite order
        if self.is_split_bugs_view:
            self.create_bugs_order()

        for trial_id in range(1, self.num_trials + 1):
            if self.block_type == 'bugs':
                if not self.exp_validation.is_reward_left():
                    utils.send_telegram_message('No reward left in feeder; stopping experiment')
                    raise EndExperimentException('No reward left; stopping experiment')
                elif self.exp_validation.is_max_reward_reached():
                    utils.send_telegram_message(f'Max daily rewards of {config.MAX_DAILY_REWARD} reached; stopping experiment')
                    raise EndExperimentException(f'Max daily rewards of {config.MAX_DAILY_REWARD} reached; stopping experiment')
            self.start_trial(trial_id)
            self.wait(self.trial_duration, check_visual_app_on=True, label=f'Trial {trial_id}',
                      take_img_after=self.trial_duration/2)
            self.end_trial()
            self.save_trial_images(trial_id)
            self.wait(self.iti, label='ITI')

        self.wait(self.extra_time_recording, label='Extra Time Rec')
        self.end_block()

    def init_block(self):
        mkdir(self.block_path)
        self.save_block_log_files()
        self.cache.set(cc.EXPERIMENT_BLOCK_ID, self.block_id, timeout=self.overall_block_duration)
        self.cache.set(cc.EXPERIMENT_BLOCK_PATH, self.block_path, timeout=self.overall_block_duration + self.iti)
        # check engagement of the animal
        self.check_engagement_level()
        # start cameras for experiment with their predictors and set the output dir for videos
        if config.CAM_TRIGGER_DELAY_AROUND_BLOCK:
            self.periphery.cam_trigger(0)  # turn off trigger
            self.logger.info('trigger is off')
        t0 = time.time()
        self.turn_cameras('on')
        # screencast
        if config.IS_RECORD_SCREEN_IN_EXPERIMENT:
            threading.Thread(target=self.record_screen).start()

        for cam_name in self.cameras.keys():
            output_dir = mkdir(f'{self.block_path}/videos')
            self.cache.set_cam_output_dir(cam_name, output_dir)

        if config.CAM_TRIGGER_DELAY_AROUND_BLOCK:
            while time.time() - t0 < config.CAM_TRIGGER_DELAY_AROUND_BLOCK:
                time.sleep(0.05)
            self.periphery.cam_trigger(1)  # turn trigger on
            self.logger.info(f'Trigger was off for {time.time() - t0:.2f} sec')

        if config.IR_TOGGLE_DELAY_AROUND_BLOCK:
            time.sleep(1)
            self.periphery.switch(config.IR_LIGHT_NAME, 1)
            time.sleep(config.IR_TOGGLE_DELAY_AROUND_BLOCK)
            self.periphery.switch(config.IR_LIGHT_NAME, 0)

    def end_block(self):
        if self.block_type == 'psycho' and self.psycho_proc_pid:
            os.killpg(os.getpgid(self.psycho_proc_pid), signal.SIGTERM)
            self.psycho_proc_pid = 0

        if config.IR_TOGGLE_DELAY_AROUND_BLOCK:
            self.periphery.switch(config.IR_LIGHT_NAME, 1)
            time.sleep(config.IR_TOGGLE_DELAY_AROUND_BLOCK)
            self.periphery.switch(config.IR_LIGHT_NAME, 0)
            time.sleep(1)

        t0 = time.time()
        if config.CAM_TRIGGER_DELAY_AROUND_BLOCK:
            self.periphery.cam_trigger(0)
            self.logger.info('trigger is off')
            time.sleep(config.CAM_TRIGGER_DELAY_AROUND_BLOCK)

        self.cache.delete(cc.EXPERIMENT_BLOCK_ID)
        for cam_name in self.cameras.keys():
            self.cache.set_cam_output_dir(cam_name, '')

        self.turn_cameras('off')
        self.cache.delete(cc.EXPERIMENT_BLOCK_PATH)
        if self.is_long_recording:
            self.cache.delete(cc.IS_COMPRESSED_LONG_RECORDING)

        if config.CAM_TRIGGER_DELAY_AROUND_BLOCK:
            self.periphery.cam_trigger(1)
            self.logger.info(f'Trigger was off for {time.time() - t0:.2f} sec')

    def create_bugs_order(self) -> list:
        """
        Create an order array for a trial in which 1 indicates the current bugs order
        and -1 indicates the opposite bugs order.
        :return: list of 1 and -1 of length equal to trial_length.
        """
        num_current_order = int(round(self.split_repeated_pos_ratio * self.num_trials))
        num_opposite_order = self.num_trials - num_current_order

        bugs_order = [1] * num_current_order + [-1] * num_opposite_order
        random.shuffle(bugs_order)
        self.split_bugs_order = bugs_order

        return bugs_order

    def turn_cameras(self, required_state):
        """Turn on cameras if needed, and load the experiment predictors"""
        assert required_state in ['on', 'off']
        for cam_name, cu in self.cam_units.items():
            # If there are no predictors configured and camera is on
            # or writing_fps is 0 - do nothing.
            configured_predictors = cu.get_conf_predictors()
            if (not configured_predictors and cu.is_on()) or \
                    (cu.cam_config.get('writing_fps') == 0):
                continue

            t0 = time.time()
            # wait maximum 10 seconds if CU is starting or stopping
            while (cu.is_starting or cu.is_stopping) and (time.time() - t0 < 10):
                time.sleep(0.1)

            if required_state == 'on':
                if not cu.is_on():
                    cu.start(is_experiment=True, movement_type=self.movement_type)
                else:
                    cu.reload_predictors(is_experiment=True, movement_type=self.movement_type)
            else:  # required_state == 'off'
                cu.reload_predictors(is_experiment=False, movement_type=self.movement_type)

        t0 = time.time()
        # wait maximum 30 seconds for cameras to finish start / stop and predictors to initiate
        while any(cu.is_starting or cu.is_stopping for cu in self.cam_units.values()) and \
                (time.time() - t0 < 30):
            time.sleep(0.1)

        if required_state == 'on':  # check predictors are up
            for cam_name, cu in self.cam_units.items():
                exp_predictors = [pr_name for pr_name, pr_dict in cu.get_conf_predictors().items()
                                  if pr_dict.get('mode') == 'experiment' and self.movement_type in pr_dict.get('movement_type', [])]
                for ep in exp_predictors:
                    is_on = False
                    t0 = time.time()
                    while time.time() - t0 < 60:
                        if ep in cu.processes and cu.processes[ep].is_on():
                            is_on = True
                            break
                    if not is_on:
                        msg = f'Aborting experiment since predictor {ep} is not alive and is configured for camera {cam_name}'
                        utils.send_telegram_message(msg)
                        self.logger.error(msg)
                        raise EndExperimentException()
            self.logger.info('finished cameras initialization for experiment')

    def start_trial(self, trial_id):
        trial_db_id = self.orm.commit_trial({
            'start_time': datetime.now(),
            'in_block_trial_id': trial_id})

        if self.block_type == 'psycho':
            self.run_psycho()
            self.cache.set(cc.IS_VISUAL_APP_ON, True)

        if self.block_type in ['bugs', 'media']:
            if self.is_media_block:
                command, options = 'init_media', self.media_options
            else:
                command, options = 'init_bugs', self.bug_options
                if self.is_random_low_horizontal:
                    options = self.set_random_low_horizontal_trial(options)
                if self.is_split_bugs_view and self.split_repeated_pos_ratio < 1:
                    ordered_bugs = self.bug_types[::self.split_bugs_order[trial_id - 1]]
                    options = self.set_bugs_order_trial(options, ordered_bugs)
            options['trialID'] = trial_id
            options['trialDBId'] = trial_db_id
            self.cache.publish_command(command, json.dumps(options))
            self.cache.set(cc.IS_VISUAL_APP_ON, True)
            time.sleep(1)  # wait for data to be sent
        self.take_trial_image()

        self.logger.info(f'Trial #{trial_id} started')

    def end_trial(self):
        self.hide_visual_app_content()
        self.take_trial_image()
        time.sleep(1)

    def get_bug_speed_for_trial(self):
        if isinstance(self.bug_speed, list):
            return random.choice(self.bug_speed)
        else:
            return self.bug_speed

    def hide_visual_app_content(self):
        if self.is_blank_block:
            return
        if self.is_media_block:
            self.cache.publish_command('hide_media')
        else:
            self.cache.publish_command('hide_bugs')

    def save_block_log_files(self):
        with open(f'{self.block_path}/info.yaml', 'w') as f:
            yaml.dump(self.info, f)
        with open(f'{self.block_path}/config.yaml', 'w') as f:
            yaml.dump(config_log(), f)
        if self.notes:
            with open(f'{self.block_path}/notes.txt', 'w') as f:
                f.write(self.notes)

    def wait(self, duration, check_visual_app_on=False, label='', take_img_after=None):
        """Sleep while checking for experiment end"""
        if label:
            label = f'({label}): '
        self.logger.info(f'{label}waiting for {duration} seconds...')
        t0 = time.time()
        while time.time() - t0 < duration:
            # check for external stop of the experiment
            if not self.cache.get(cc.EXPERIMENT_NAME):
                raise EndExperimentException()
            # check for visual app finish (due to bug catch, etc...)
            if check_visual_app_on and not self.is_blank_block and not self.cache.get(cc.IS_VISUAL_APP_ON):
                self.logger.debug('Trial ended')
                return
            # If take_img_after is set, and we have not taken 2 images yet, take one now.
            # This is used for taking a middle trial image during the wait loop
            if take_img_after and len(self.trial_images) < 2 and time.time() - t0 > take_img_after:
                self.take_trial_image()
            time.sleep(0.1)

    def take_trial_image(self):
        if not config.TRIAL_IMAGE_CAMERA:
            return
        try:
            img = self.cam_units[config.TRIAL_IMAGE_CAMERA].get_frame()
            img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
            self.trial_images.append(img)
        except Exception as e:
            self.trial_images.append(None)
            self.logger.error(f'Error taking trial image: {str(e)}')

    def save_trial_images(self, trial_num):
        trials_img_dir = Path(f'{self.block_path}/trials_images')
        trials_img_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(self.trial_images):
            if img is not None:
                cv2.imwrite(f'{trials_img_dir}/trial_{trial_num}_{i}.png', img)
        self.trial_images = []

    def check_engagement_level(self):
        """check if there are any strikes in the previous 2 hours. If not, give a manual reward"""
        if not config.CHECK_ENGAGEMENT_HOURS:
            return
        with self.orm.session() as s:
            res = s.query(Strike).filter(Strike.time > datetime.now() - timedelta(hours=config.CHECK_ENGAGEMENT_HOURS)).all()
        if not res:
            self.periphery.feed(is_manual=True)

    def run_psycho(self):
        psycho_files = get_psycho_files()
        cmd = f'cd {psycho_files[self.psycho_file]} && DISPLAY="{config.APP_SCREEN}" {config.PSYCHO_PYTHON_INTERPRETER} {self.psycho_file}.py'
        self.logger.info(f'Running the following psycho cmd: {cmd}')
        proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid, stderr=subprocess.STDOUT)
        self.psycho_proc_pid = proc.pid

    def record_screen(self):
        filename = f'{self.block_path}/screen_record.mp4'
        next(run_command(
            f'ffmpeg -video_size 1920x1080 -framerate 30 -f x11grab '
            f'-i :0.0+1920+0 -f pulse -i default -ac 2 -t {int(self.block_duration)} '
            f'''-vf "drawtext=fontfile=/Windows/Fonts/Arial.ttf: 
            text='%{{localtime}}':x=30:y=30:fontcolor=red:fontsize=30" {filename}''', is_debug=False)
        )

    @staticmethod
    def set_random_low_horizontal_trial(options):
        options['movementType'] = 'low_horizontal'
        return options

    @staticmethod
    def set_bugs_order_trial(options, ordered_bugs):
        options['bugTypes'] = ordered_bugs
        return options

    @property
    def media_options(self) -> dict:
        return {
            'trialID': 1,  # default value, changed in init_media,
            'url': f'{config.MANAGEMENT_URL}/media/{self.media_url}'
        }

    @property
    def bug_options(self) -> dict:
        return {
            'numOfBugs': self.num_of_bugs,
            'isSplitBugsView': self.is_split_bugs_view,
            'splitRandomizeTiming': self.split_randomize_timing,
            'trialID': 1,  # default value, changed in init_bugs
            'trialDBId': 1, # default value, changed in init_bugs
            'numTrials': self.num_trials,
            'iti': self.iti,
            'trialDuration': self.trial_duration,
            'speed': self.get_bug_speed_for_trial(),
            'bugTypes': self.bug_types,
            'rewardBugs': self.reward_bugs,
            'movementType': self.movement_type,
            'isLogTrajectory': True,
            'bugSize': self.bug_size,
            'backgroundColor': self.background_color,
            'exitHole': random.choice(['left', 'right']) if self.exit_hole == 'random' else self.exit_hole,
            'rewardAnyTouchProb': self.reward_any_touch_prob,
            'holesHeightScale': self.holes_height_scale,
            'circleHeightScale': self.circle_height_scale,
            'circleRadiusScale': self.circle_radius_scale,
            'accelerateMultiplier': self.accelerate_multiplier
        }

    @property
    def block_summary(self):
        log_string = f'Summary of Block {self.block_id}:\n'
        touches_file = Path(self.block_path) / config.experiment_metrics.get("touch", {}).get('csv_file', 'touch.csv')
        num_hits = 0
        if touches_file.exists() and touches_file.is_file():
            touches_df = pd.read_csv(touches_file, parse_dates=['time'], index_col=0).reset_index(drop=True)
            log_string += f'  Number of touches on the screen: {len(touches_df)}\n'
            num_hits = len(touches_df.query("is_hit == True"))
            log_string += f'  Number of successful hits: {num_hits}\n'
            num_hits_rewarded = len(touches_df.query("is_hit == True & is_reward_bug == True"))
            log_string += f'  Number of Rewarded hits: {num_hits_rewarded}'
        else:
            log_string += 'No screen strikes were recorded.'

        log_string += 2 * '\n'

        if num_hits and self.reward_type == 'end_trial':
            self.cache.publish(config.subscription_topics['reward'], '')

        return log_string

    @property
    def is_media_block(self):
        return self.block_type == 'media'

    @property
    def is_blank_block(self):
        return self.block_type == 'blank'

    @property
    def is_continuous_blank(self):
        return self.blank_rec_type == 'continuous'

    @property
    def is_long_recording(self):
        return self.overall_block_duration > config.MAX_TIME_SHORT_RECORDING_DURATION

    @property
    def is_random_low_horizontal(self):
        return self.movement_type == 'random_low_horizontal'

    @property
    def is_always_reward(self):
        return self.reward_type == 'always'

    @property
    def overall_block_duration(self):
        if not self.is_blank_block:
            return self.block_duration + 2 * self.extra_time_recording
        else:
            return config.MAX_DURATION_CONT_BLANK

    @property
    def block_duration(self):
        return round((self.num_trials * self.trial_duration + (self.num_trials - 1) * self.iti) * 1.5)

    @property
    def block_path(self):
        return f'{self.experiment_path}/block{self.block_id}'

    @property
    def videos_path(self):
        return f'{self.block_path}/videos'


class ExperimentValidation:
    def __init__(self, logger=None, cache=None, orm=None, is_silent=False):
        self.logger = logger if logger is not None else get_logger('Experiment-Validation')
        self.cache = cache if cache is not None else RedisCache()
        self.orm = orm if orm is not None else ORM()
        self.animal_id = self.cache.get(cc.CURRENT_ANIMAL_ID)
        self.is_silent = is_silent
        self.failed_checks = []

    def is_ready(self):
        self.failed_checks = []
        checks = {
            'websocket_server_on': self.is_websocket_server_on(),
            'pogona_hunter_app_up': self.is_pogona_hunter_up(),
            'reward_left': self.is_reward_left(),
            'touchscreen_mapped': self.is_touchscreen_mapped_to_hdmi()
        }
        if all(checks.values()):
            return True
        else:
            errs = [k for k, v in checks.items() if not v]
            msg = f'Aborting experiment due to violation of {", ".join(errs)}.'
            if not self.is_silent:
                utils.send_telegram_message(msg)
                self.logger.error(msg)
            self.failed_checks.extend(errs)
            return False

    def is_websocket_server_on(self):
        try:
            ws = websocket.WebSocket()
            ws.connect(config.WEBSOCKET_URL)
            return True
        except Exception:
            if not self.is_silent:
                self.logger.error(f'Websocket server on {config.WEBSOCKET_URL} is dead')

    def is_pogona_hunter_up(self):
        try:
            res = requests.get(f'http://0.0.0.0:{config.POGONA_HUNTER_PORT}')
            return res.ok
        except Exception:
            if not self.is_silent:
                self.logger.error('pogona hunter app is down')

    def is_touchscreen_mapped_to_hdmi(self):
        if not config.IS_CHECK_SCREEN_MAPPING:
            return True

        touchscreen_device_id = self.get_touchscreen_device_id()

        def _check_mapped():
            # if the matrix under "Coordinate Transformation Matrix" has values different from 0,1 - that means
            # that the mapping is working
            cmd = f'DISPLAY="{config.APP_SCREEN}"  xinput list-props {touchscreen_device_id} | grep "Coordinate Transformation Matrix"'
            res = next(run_command(cmd)).decode()
            return any(z not in [0.0, 1.0] for z in [float(x) for x in re.findall(r'\d\.\d+', res)])

        try:
            is_mapped = _check_mapped()
            if not is_mapped:
                self.logger.info('Fixing mapping of touchscreen output')
                cmd = f'DISPLAY="{config.APP_SCREEN}" xinput map-to-output {touchscreen_device_id} {config.APP_DISPLAY}'
                self.map_touchscreen_to_hdmi()
                time.sleep(1)
                is_mapped = _check_mapped()
                if not is_mapped and not self.is_silent:
                    self.logger.error(
                        f'Touch detection is not mapped to {config.APP_DISPLAY} screen\nFix by running: {cmd}')
            return is_mapped
        except Exception:
            if not self.is_silent:
                self.logger.exception('Error in is_touchscreen_mapped_to_hdmi')

    def map_touchscreen_to_hdmi(self, is_display_on=False):
        touchscreen_device_id = self.get_touchscreen_device_id()
        cmd = f'DISPLAY="{config.APP_SCREEN}" xinput map-to-output {touchscreen_device_id} {config.APP_DISPLAY}'
        if not is_display_on:
            turn_display_on()
            time.sleep(5)
        next(run_command(cmd))

    def is_reward_left(self):
        return self.get_reward_left() > 0

    def is_max_reward_reached(self):
        rewards_dict = self.orm.get_rewards_for_day(animal_id=self.animal_id)
        return sum(rewards_dict.values()) >= config.MAX_DAILY_REWARD

    def get_reward_left(self):
        try:
            return sum([int(x) for x in self.cache.get(cc.REWARD_LEFT)])
        except Exception:
            return 0

    @staticmethod
    def get_touchscreen_device_id():
        touchscreen_device_id = get_hdmi_xinput_id()
        if not touchscreen_device_id:
            raise Exception('unable to find touch USB')
        return touchscreen_device_id


class ExperimentCache:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or config.CACHED_EXPERIMENTS_DIR
        mkdir(self.cache_dir)
        self.saved_caches = self.get_saved_caches()

    def load(self, name):
        path = Path(self.get_cache_path(name))
        assert path.exists(), f'experiment {name} does not exist'
        with path.open('r') as f:
            data = json.load(f)
        return data

    def save(self, data):
        name = data.get('name')
        with Path(self.get_cache_path(name)).open('w') as f:
            json.dump(data, f)

    def get_saved_caches(self):
        return list(Path(self.cache_dir).glob('*.json'))

    def get_cache_path(self, name):
        return f"{self.cache_dir}/{name}.json"


class EndExperimentException(Exception):
    """End Experiment"""


def config_log():
    """Get the config for logging"""
    drop_config_fields = ['Env', 'env']
    config_dict = config.__dict__
    for k in config_dict.copy():
        if k.startswith('__') or k in drop_config_fields:
            config_dict.pop(k)
    return config_dict


def get_arguments(cls):
    """Get the arguments of the Experiment class"""
    sig = inspect.signature(cls.__init__)
    return sig.parameters.keys()


def main():
    signature = inspect.signature(Experiment.__init__)
    arg_parser = argparse.ArgumentParser(description='Experiments')

    for k, v in signature.parameters.items():
        if k == 'self':
            continue
        default = v.default if v.default is not signature.empty else None
        annotation = v.annotation if v.annotation is not signature.empty else None
        arg_parser.add_argument('--' + k, type=annotation, default=default)

    args = arg_parser.parse_args()
    kwargs = dict(args._get_kwargs())

    e = Experiment(**kwargs)
    e.start()
    print(e)


def start_trial():
    arg_parser = argparse.ArgumentParser(description='Experiments')
    arg_parser.add_argument('-m', '--movement_type', default='circle')
    arg_parser.add_argument('--exit_hole', default='right', choices=['left', 'right'])
    arg_parser.add_argument('--speed', type=int, default=5)
    args = arg_parser.parse_args()
    cache_ = RedisCache()
    options = {
        'numOfBugs': 1,
        'isSplitBugsView': True,
        'trialID': 1,  # default value, changed in init_bugs
        'trialDBId': 1, # default value, changed in init_bugs
        'numTrials': 1,
        'iti': 30,
        'trialDuration': 5,
        'speed': args.speed,
        'bugTypes': ['cockroach'],
        'rewardBugs': [],
        'movementType': args.movement_type,
        'isLogTrajectory': True,
        # 'bugSize': self.bug_size,
        # 'backgroundColor': self.background_color,
        'exitHole': args.exit_hole,
        'rewardAnyTouchProb': 0,
        'circleRadiusScale': 0.2,
        'circleHeightScale': 0.5,
        'holesHeightScale': 0.1
    }
    cache_.publish_command('init_bugs', json.dumps(options))


if __name__ == "__main__":
    start_trial()
import re
import time
import json
from functools import wraps
from datetime import datetime, timedelta
import threading
import multiprocessing
from loggers import get_logger
import config
from cache import RedisCache, CacheColumns as cc
from compress_videos import get_videos_ids_for_compression, compress
from periphery_integration import PeripheryIntegrator, LightSTIM
from analysis.pose import VideoPoseScanner
from analysis.strikes.strikes import StrikeScanner
from db_models import DWH
from agent import Agent
import utils
try:
    from analysis.predictors.pogona_head import predict_tracking
except Exception:
    predict_tracking = None

TIME_TABLE = {
    'cameras_on': (config.CAMERAS_ON_TIME, config.CAMERAS_OFF_TIME),
    'run_pose': (config.POSE_ON_TIME, config.POSE_OFF_TIME),
    'tracking_pose': (config.TRACKING_POSE_ON_TIME, config.TRACKING_POSE_OFF_TIME),
    'lights_sunrise': config.LIGHTS_SUNRISE,
    'lights_sunset': config.LIGHTS_SUNSET,
    'dwh_commit_time': config.DWH_COMMIT_TIME,
    'strike_analysis_time': config.STRIKE_ANALYSIS_TIME,
    'daily_summary': config.DAILY_SUMMARY_TIME
}
ALWAYS_ON_CAMERAS_RESTART_DURATION = 30 * 60  # seconds
cache = RedisCache()


def schedule_method(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            args[0].logger.error(f'Error in {func.__name__}: {exc}')
    return wrapper


class Scheduler(threading.Thread):
    def __init__(self, arena_mgr):
        super().__init__()
        self.logger = get_logger('Scheduler')
        self.logger.debug('Scheduler started...')
        self.arena_mgr = arena_mgr
        self.periphery = PeripheryIntegrator()
        if config.IS_AGENT_ENABLED:
            self.agent = Agent()
        self.next_experiment_time = None
        self.dlc_on = multiprocessing.Event()
        self.dlc_errors_cache = []
        self.tracking_pose_on = multiprocessing.Event()
        self.compress_threads = {}
        self.current_animal_id = None
        self.dwh_commit_tries = 0
        self.start_lights()

    def run(self):
        time.sleep(10)  # let all other arena processes and threads to start
        t0 = None  # every minute
        t1 = None  # every 5 minutes
        t2 = None  # every 15 minutes
        while not self.is_stop_set():
            if not t0 or time.time() - t0 >= 60:  # every minute
                t0 = time.time()
                self.current_animal_id = cache.get(cc.CURRENT_ANIMAL_ID)
                self.check_lights()
                self.check_camera_status()
                self.set_tracking_cameras()
                self.check_scheduled_experiments()
                self.periphery.send_toggles_healthcheck()
                self.arena_mgr.update_upcoming_schedules()
                self.analyze_strikes()
                self.dwh_commit()
                self.daily_summary()

            if not t1 or time.time() - t1 >= 60 * 5:  # every 5 minutes
                t1 = time.time()
                self.compress_videos()
                self.run_pose()
                self.tracking_pose()

            if not t2 or time.time() - t2 >= 60 * 15:  # every 15 minutes
                t2 = time.time()
                self.agent_update()

    @schedule_method
    def check_lights(self):
        """Check that during the day LEDs are on and IR is off, and vice versa during the night"""
        if self.is_in_range('lights_sunrise'):
            self.turn_light(config.IR_LIGHT_NAME, 0)
            self.turn_light(config.DAY_LIGHT_NAME, 1)
        elif self.is_in_range('lights_sunset'):
            self.turn_light(config.IR_LIGHT_NAME, 1)
            self.turn_light(config.DAY_LIGHT_NAME, 0)

    def turn_light(self, name, state):
        if not name:
            return
        self.periphery.switch(name, state)
        self.logger.info(f'turn {name} {"on" if state else "off"}')

    def start_lights(self):
        """function to run only when scheduler is initiated to match lightning to expected lighting conditions"""
        if self.is_in_range((TIME_TABLE['lights_sunrise'], TIME_TABLE['lights_sunset'])):
            self.turn_light(config.IR_LIGHT_NAME, 0)
            self.turn_light(config.DAY_LIGHT_NAME, 1)
        else:
            self.turn_light(config.IR_LIGHT_NAME, 1)
            self.turn_light(config.DAY_LIGHT_NAME, 0)

    @schedule_method
    def check_scheduled_experiments(self):
        """Check if a scheduled experiment needs to be executed and run it"""
        for schedule_id, schedule_string in self.arena_mgr.schedules.items():
            m = re.search(r'(?P<date>.{16}) - (?P<name>.*)', schedule_string)
            if m:
                schedule_date = datetime.strptime(m.group('date'), config.SCHEDULER_DATE_FORMAT)
                if (schedule_date - datetime.now()).total_seconds() <= 0:
                    exp_name = m.group('name')
                    # lights stimulation (in case the name starts with "LightSTIM:")
                    if exp_name.startswith('LightSTIM:'):
                        stim_cmd = exp_name.replace('LightSTIM:', '')
                        LightSTIM().run_stim_command(stim_cmd)
                    # schedule of periphery switches (in case the name starts with "SWITCH:")
                    elif exp_name.startswith('SWITCH:'):
                        switch_name, switch_state = exp_name.replace('SWITCH:', '').split(',')
                        self.periphery.switch(switch_name, int(switch_state == 'on'))
                    # schedule of agent activation
                    elif exp_name.startswith('AGENT:'):
                        agent_state = exp_name.replace('AGENT:', '')
                        cache.set(cc.HOLD_AGENT, agent_state == 'off')
                    elif exp_name.startswith('FEEDER:'):
                        self.periphery.feed()
                    else:  # otherwise, start the cached experiment
                        self.arena_mgr.start_cached_experiment(m.group('name'))

    @schedule_method
    def dwh_commit(self):
        if self.is_in_range('dwh_commit_time') and config.DWH_HOST and not cache.get_current_experiment():
            if self.dwh_commit_tries >= config.DWH_N_TRIES:
                self.dwh_commit_tries = 0
                utils.send_telegram_message(f'Commit to DWH failed after {config.DWH_N_TRIES} times')
                return
            try:
                DWH().commit()
            except Exception as exc:
                self.dwh_commit_tries += 1
                self.logger.warning(f'Failed committing to DWH ({self.dwh_commit_tries}/{config.DWH_N_TRIES}): {exc}')
            else:
                self.dwh_commit_tries = 0

    @schedule_method
    def agent_update(self):
        if config.IS_AGENT_ENABLED and self.is_in_range('cameras_on') and not self.is_test_animal() and \
                not cache.get(cc.HOLD_AGENT):
            self.agent.update()

    @schedule_method
    def analyze_strikes(self):
        if self.is_in_range('strike_analysis_time') and config.IS_RUN_NIGHTLY_POSE_ESTIMATION:
            StrikeScanner().scan()

    @staticmethod
    def is_in_range(label):
        now = datetime.now()
        if isinstance(label, str):
            val = TIME_TABLE[label]
        else:
            val = label
        if isinstance(val, tuple):
            start, end = [datetime.combine(now, datetime.strptime(t, '%H:%M').time()) for t in val]
        elif isinstance(val, str):
            dt = datetime.combine(now, datetime.strptime(val, '%H:%M').time())
            start, end = dt, dt + timedelta(minutes=1)
        else:
            raise Exception(f'bad value for {label}: {val}')

        if start > end:
            return ((now - start).total_seconds() >= 0) or ((now - end).total_seconds() <= 0)
        else:
            return ((now - start).total_seconds() >= 0) and ((now - end).total_seconds() <= 0)

    @schedule_method
    def check_camera_status(self):
        """turn off cameras outside working hours, and restart predictors. Does nothing during
        an active experiment"""
        if cache.get(cc.IS_EXPERIMENT_CONTROL_CAMERAS) or config.DISABLE_CAMERAS_CHECK:
            return

        for cam_name, cu in self.arena_mgr.units.copy().items():
            if cu.is_starting or cu.is_stopping:
                continue

            if not self.is_in_range('cameras_on'):
                # outside the active hours
                self.stop_camera(cu)
            else:
                # in active hours
                if cu.cam_config.get('mode') == 'manual':
                    # camera will be stopped only if min_duration was reached
                    self.stop_camera(cu)
                else:
                    self.start_camera(cu)
                    # if there are any alive_predictors on, restart every x minutes.
                    if cu.is_on() and cu.get_alive_predictors() and cu.preds_start_time and \
                            time.time() - cu.preds_start_time > ALWAYS_ON_CAMERAS_RESTART_DURATION:
                        self.logger.info(f'restarting camera unit of {cu.cam_name}')
                        cu.stop()
                        if cam_name == self.arena_mgr.get_streaming_camera():
                            self.arena_mgr.stop_stream()
                        time.sleep(1)
                        cu.start()

    def stop_camera(self, cu):
        if cu.is_on() and cu.time_on > config.CAMERA_ON_MIN_DURATION:
            self.logger.info(f'stopping CU {cu.cam_name}')
            cu.stop()

    def start_camera(self, cu):
        if not cu.is_on():
            self.logger.debug(f'starting CU {cu.cam_name}')
            cu.start()
            time.sleep(5)

    @schedule_method
    def set_tracking_cameras(self):
        if not self.is_in_range('cameras_on') or not config.IS_TRACKING_CAMERAS_ALLOWED or \
                cache.get(cc.IS_EXPERIMENT_CONTROL_CAMERAS) or self.is_test_animal():
            return

        for cam_name, cu in self.arena_mgr.units.copy().items():
            if cu.is_starting or cu.is_stopping:
                continue

            if cu.cam_config.get('mode') == 'tracking':
                tracking_output_path = utils.get_todays_experiment_dir(cache.get(cc.CURRENT_ANIMAL_ID)) + '/tracking'
                utils.mkdir(tracking_output_path)
                cache.set_cam_output_dir(cam_name, tracking_output_path)

    @schedule_method
    def daily_summary(self):
        if self.is_in_range('daily_summary') and not self.is_test_animal():
            struct = self.arena_mgr.orm.today_summary()
            msg = json.dumps(struct, indent=4)
            utils.send_telegram_message(f'Daily Summary:\n{msg}')

    @schedule_method
    def compress_videos(self):
        if self.is_in_range('cameras_on') or not self.is_compression_thread_available() or config.DISABLE_DB or config.IS_COMPRESSION_DISABLED:
            return

        videos = get_videos_ids_for_compression(self.arena_mgr.orm, sort_by_size=True)
        if not videos:
            return

        while self.is_compression_thread_available():
            currently_compressed_vids = [v for _, v in self.compress_threads.values()]
            vids_ = [v for v in videos if v not in currently_compressed_vids]
            if not vids_:
                return
            t = threading.Thread(target=compress, args=(vids_[0], self.logger, self.arena_mgr.orm))
            t.start()
            self.compress_threads[t.name] = (t, vids_[0])

    def is_compression_thread_available(self):
        for thread_name in list(self.compress_threads.keys()):
            t, _ = self.compress_threads[thread_name]
            if not t.is_alive():
                self.compress_threads.pop(thread_name)

        return len(self.compress_threads) < config.MAX_COMPRESSION_THREADS

    @schedule_method
    def run_pose(self):
        if not self.is_in_range('run_pose') or cache.get(cc.IS_COMPRESSED_LONG_RECORDING) or self.dlc_on.is_set() or \
                not config.IS_RUN_NIGHTLY_POSE_ESTIMATION:
            return

        multiprocessing.Process(target=_run_pose_callback, args=(self.dlc_on, self.dlc_errors_cache)).start()
        self.dlc_on.set()

    @schedule_method
    def tracking_pose(self):
        if predict_tracking is None or not self.is_in_range('tracking_pose') or \
            cache.get(cc.IS_COMPRESSED_LONG_RECORDING) or self.dlc_on.is_set() or \
                not config.IS_RUN_NIGHTLY_POSE_ESTIMATION or self.tracking_pose_on.is_set():
            return

        multiprocessing.Process(target=_run_tracking_pose, args=(self.tracking_pose_on,)).start()
        self.tracking_pose_on.set()

    def is_test_animal(self):
        return self.current_animal_id in ['test']

    def is_stop_set(self):
        try:
            return self.arena_mgr.arena_shutdown_event.is_set()
        except Exception:
            return True


def _run_pose_callback(dlc_on, errors_cache):
    try:
        VideoPoseScanner().predict_all(max_videos=20, errors_cache=errors_cache, is_tqdm=False)
    finally:
        dlc_on.clear()

def _run_tracking_pose(tracking_pose_on):
    try:
        predict_tracking(max_videos=30, is_tqdm=False)
    finally:
        tracking_pose_on.clear()
import re
import time
import json
import subprocess
import tempfile
import shutil
from functools import wraps
from datetime import datetime, timedelta
from pathlib import Path
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
    'daily_summary': config.DAILY_SUMMARY_TIME,
    'daily_timelapse_push': config.TIMELAPSE_DAILY_PUSH_TIME
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
        self.last_daily_summary_sent = cache.get(cc.DAILY_SUMMARY_SENT_DATE)
        self.last_timelapse_summary_sent = cache.get(cc.DAILY_TIMELAPSE_SUMMARY_SENT_DATE)
        self.last_daily_timelapse_sent = cache.get(cc.DAILY_TIMELAPSE_SENT_DATE)
        self.timelapse_watchdog_started_at = time.time()
        self.last_timelapse_capture_restart = 0
        self._timelapse_restart_inflight = False
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
                self.timelapse_capture_watchdog()
                self.periphery.send_toggles_healthcheck()
                self.arena_mgr.update_upcoming_schedules()
                self.analyze_strikes()
                self.dwh_commit()
                self.daily_summary()
                self.daily_timelapse_push()

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
                    try:
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
                        elif exp_name.startswith('SOUND:'):
                            wav_name = exp_name.replace('SOUND:', '')
                            wav_file = f'{config.STATIC_FILES_DIR}/{wav_name}'
                            self.periphery.play_wav_file(wav_file)
                        else:  # otherwise, start the cached experiment
                            self.arena_mgr.start_cached_experiment(m.group('name'))
                    finally:
                        self.arena_mgr.orm.delete_schedule(schedule_id)

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
            if self._is_in_agent_window() and not self.agent.get_upcoming_agent_schedules() \
                and cache.get(cc.LAST_AGENT_ERROR) is None:
                msg = 'Agent is active but no upcoming schedules were found.'
                self.agent.publish(msg)

    def _is_in_agent_window(self):
        try:
            start_time = self.agent.times['start_time']
            end_time = self.agent.times['end_time']
            start_hour, start_minute = start_time.split(':')
            end_hour, end_minute = end_time.split(':')
        except Exception:
            return True
        now = datetime.now()
        start_dt = now.replace(hour=int(start_hour), minute=int(start_minute), second=0, microsecond=0)
        end_dt = now.replace(hour=int(end_hour), minute=int(end_minute), second=0, microsecond=0)
        if start_dt <= end_dt:
            return start_dt <= now <= end_dt
        return now >= start_dt or now <= end_dt

    @schedule_method
    def analyze_strikes(self):
        if self.is_in_range('strike_analysis_time') and config.IS_RUN_NIGHTLY_POSE_ESTIMATION:
            StrikeScanner().scan()

    def _load_attempts(self, cache_column):
        entries = cache.get(cache_column) or []
        attempts = {}
        for entry in entries:
            if not entry:
                continue
            token, sep, count_str = entry.partition('=')
            if not sep:
                continue
            try:
                attempts[token] = int(count_str)
            except ValueError:
                continue
        return attempts

    def _save_attempts(self, cache_column, attempts):
        payload = [f'{token}={count}' for token, count in attempts.items()]
        cache.set(cache_column, payload)

    @staticmethod
    def _is_after_final_retry_time(now):
        summary_time = datetime.strptime(config.DAILY_SUMMARY_TIME, '%H:%M').time()
        cameras_off_time = datetime.strptime(config.CAMERAS_OFF_TIME, '%H:%M').time()
        summary_dt = datetime.combine(now.date(), summary_time)
        cameras_off_dt = datetime.combine(now.date(), cameras_off_time)
        final_dt = max(summary_dt, cameras_off_dt)
        return now >= final_dt

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
        if self.is_test_animal():
            return
        now = datetime.now()
        summary_time = datetime.strptime(config.DAILY_SUMMARY_TIME, '%H:%M').time()
        summary_dt = datetime.combine(now.date(), summary_time)
        if now < summary_dt:
            return
        today_key = now.strftime('%Y%m%d')
        cached_summary = cache.get(cc.DAILY_SUMMARY_SENT_DATE)
        cached_video = cache.get(cc.DAILY_TIMELAPSE_SUMMARY_SENT_DATE)
        summary_sent = (self.last_daily_summary_sent == today_key) or (cached_summary == today_key)
        video_sent = (self.last_timelapse_summary_sent == today_key) or (cached_video == today_key)
        if summary_sent and video_sent:
            return
        if not summary_sent:
            attempts = self._load_attempts(cc.DAILY_SUMMARY_ATTEMPTS)
            attempt_count = attempts.get(today_key, 0)
            max_attempts = 4 if self._is_after_final_retry_time(now) else 3
            if attempt_count >= max_attempts:
                return
            attempts[today_key] = attempt_count + 1
            self._save_attempts(cc.DAILY_SUMMARY_ATTEMPTS, attempts)
            struct = self.arena_mgr.orm.today_summary()
            msg_lines = [f'Daily Summary:\n{json.dumps(struct, indent=4)}']
            skipped_rewards = self.arena_mgr.orm.get_skipped_rewards_for_day()
            if skipped_rewards:
                msg_lines.append(f'Skipped rewards today: {skipped_rewards}')
            resp = utils.send_telegram_message('\n'.join(msg_lines))
            if resp is not None and resp.ok:
                self.last_daily_summary_sent = today_key
                cache.set(cc.DAILY_SUMMARY_SENT_DATE, today_key)
                attempts.pop(today_key, None)
                self._save_attempts(cc.DAILY_SUMMARY_ATTEMPTS, attempts)
                self.logger.info('Daily summary sent for %s', today_key)
            else:
                self.logger.warning('Daily summary failed for %s', today_key)
        if not video_sent:
            if self.send_timelapse_summary_clip():
                self.last_timelapse_summary_sent = today_key
                cache.set(cc.DAILY_TIMELAPSE_SUMMARY_SENT_DATE, today_key)
                self.logger.info('Timelapse summary sent for %s', today_key)
            else:
                self.logger.warning('Timelapse summary failed or missing clips for %s', today_key)

    @schedule_method
    def daily_timelapse_push(self):
        if not config.TIMELAPSE_DAILY_PUSH_ENABLE or self.is_test_animal():
            return
        settings = getattr(config, 'TIMELAPSE_SETTINGS', None)
        if not settings or not settings.get('enable'):
            return
        now = datetime.now()
        push_time = datetime.strptime(config.TIMELAPSE_DAILY_PUSH_TIME, '%H:%M').time()
        push_dt = datetime.combine(now.date(), push_time)
        if now < push_dt:
            return
        today = now.date()
        target_date = today - timedelta(days=1)
        target_key = target_date.strftime('%Y%m%d')
        if self.last_daily_timelapse_sent == target_key:
            return
        if target_date >= today:
            return
        cached_sent = cache.get(cc.DAILY_TIMELAPSE_SENT_DATE)
        if cached_sent == target_key:
            self.last_daily_timelapse_sent = target_key
            return
        if self._send_full_day_timelapse(target_date, settings):
            self.last_daily_timelapse_sent = target_key
            cache.set(cc.DAILY_TIMELAPSE_SENT_DATE, target_key)
            self.logger.info('Timelapse daily push sent for %s', target_key)
        else:
            self.logger.warning('Timelapse daily push failed or missing clips for %s', target_key)

    def send_timelapse_summary_clip(self):
        settings = getattr(config, 'TIMELAPSE_SETTINGS', None)
        if not settings or not settings.get('enable'):
            return False
        camera_names = self._get_timelapse_camera_names(settings)
        if not camera_names:
            return False

        today = datetime.now().date()
        date_key = today.strftime('%Y%m%d')
        sent_cameras = cache.get(cc.DAILY_TIMELAPSE_SUMMARY_SENT_CAMERAS) or []
        sent_cameras_set = set(sent_cameras)
        attempts = self._load_attempts(cc.DAILY_TIMELAPSE_SUMMARY_ATTEMPTS)
        max_attempts = config.TIMELAPSE_MAX_SEND_ATTEMPTS
        start_time = datetime.strptime(config.CAMERAS_ON_TIME, '%H:%M').time()
        summary_time = datetime.strptime(config.DAILY_SUMMARY_TIME, '%H:%M').time()
        cameras_off_time = datetime.strptime(config.CAMERAS_OFF_TIME, '%H:%M').time()
        start_dt = datetime.combine(today, start_time)
        summary_dt = datetime.combine(today, summary_time)
        cameras_off_dt = datetime.combine(today, cameras_off_time)
        end_dt = max(summary_dt, cameras_off_dt)
        if end_dt <= start_dt:
            self.logger.warning('Skipping timelapse summary - summary time %s is not after start time %s',
                                config.DAILY_SUMMARY_TIME, config.CAMERAS_ON_TIME)
            return
        start_hour = start_time.hour
        end_hour = min(end_dt.hour + 1, 24)
        if start_hour >= end_hour:
            self.logger.warning('Skipping timelapse summary - invalid hours range %s-%s', start_hour, end_hour)
            return

        base_dir = Path(settings['base_dir'])
        hourly_dir = Path(settings.get('hourly_dir') or (base_dir / 'hourly'))
        captures_dir = Path(settings.get('captures_dir') or (base_dir / 'captures'))
        daily_dir = Path(settings.get('daily_dir') or (base_dir / 'daily'))
        hourly_framerate = int(settings.get('hourly_framerate', 24))
        start_label = config.CAMERAS_ON_TIME.replace(':', '')
        coverage_end_label = end_dt.strftime('%H:%M')
        end_label = coverage_end_label.replace(':', '')

        all_sent = True
        sent_any = False
        for camera in camera_names:
            token = f'{date_key}:{camera}'
            if token in sent_cameras_set:
                sent_any = True
                continue
            attempt_count = attempts.get(token, 0)
            if attempt_count >= max_attempts:
                all_sent = False
                continue
            attempts[token] = attempt_count + 1
            self._save_attempts(cc.DAILY_TIMELAPSE_SUMMARY_ATTEMPTS, attempts)
            hourly_files = []
            for hour in range(start_hour, end_hour):
                clip_path = self._ensure_hourly_clip(date_key, hour, camera, captures_dir, hourly_dir,
                                                     hourly_framerate)
                if clip_path:
                    hourly_files.append(clip_path)
            if not hourly_files:
                self.logger.info('Timelapse summary: no hourly clips available for camera %s', camera)
                all_sent = False
                continue

            output_path = daily_dir / camera / f'{date_key}_summary_{start_label}-{end_label}.mp4'
            if self._stitch_hourly_clips(hourly_files, output_path):
                caption = f'Timelapse ({camera}) {date_key} {config.CAMERAS_ON_TIME}-{coverage_end_label}'
                resp = utils.send_telegram_video(str(output_path),
                                                 caption=caption,
                                                 timeout=config.TIMELAPSE_SEND_TIMEOUT)
                if resp is not None and resp.ok:
                    sent_any = True
                    sent_cameras_set.add(token)
                    cache.set(cc.DAILY_TIMELAPSE_SUMMARY_SENT_CAMERAS, list(sent_cameras_set))
                    attempts.pop(token, None)
                    self._save_attempts(cc.DAILY_TIMELAPSE_SUMMARY_ATTEMPTS, attempts)
                else:
                    all_sent = False
                    self.logger.warning('Timelapse summary telegram send failed for camera %s on %s',
                                        camera, date_key)
            else:
                all_sent = False

        return all_sent and sent_any

    @schedule_method
    def timelapse_capture_watchdog(self):
        if not config.TIMELAPSE_CAPTURE_WATCHDOG_ENABLE or not config.TIMELAPSE_ENABLE:
            return
        now = time.time()
        hb = cache.get(cc.TIMELAPSE_CAPTURE_HEARTBEAT)
        try:
            hb = float(hb) if hb else None
        except (TypeError, ValueError):
            hb = None

        stale_seconds = config.TIMELAPSE_CAPTURE_STALE_SECONDS
        if hb is None:
            if now - self.timelapse_watchdog_started_at < stale_seconds:
                return
            if now - self.last_timelapse_capture_restart < config.TIMELAPSE_CAPTURE_WATCHDOG_COOLDOWN:
                return
            self._restart_timelapse_capture('no heartbeat')
            return

        if now - hb < stale_seconds:
            return
        if now - self.last_timelapse_capture_restart < config.TIMELAPSE_CAPTURE_WATCHDOG_COOLDOWN:
            return
        self._restart_timelapse_capture(f'stale heartbeat ({now - hb:.0f}s)')

    def _restart_timelapse_capture(self, reason):
        if self._timelapse_restart_inflight:
            return
        self._timelapse_restart_inflight = True

        def _run():
            try:
                supervisorctl = shutil.which('supervisorctl')
                if not supervisorctl:
                    self.logger.error('Timelapse watchdog: supervisorctl not found; cannot restart capture loop')
                    return
                cmd = [supervisorctl, 'restart', config.TIMELAPSE_CAPTURE_SUPERVISOR_NAME]
                subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
                self.logger.warning('Timelapse watchdog restarted capture loop (%s)', reason)
            except Exception as exc:
                self.logger.error('Timelapse watchdog restart failed: %s', exc)
            finally:
                self.last_timelapse_capture_restart = time.time()
                self._timelapse_restart_inflight = False

        threading.Thread(target=_run, daemon=True).start()

    def _send_full_day_timelapse(self, target_date, settings):
        camera_names = self._get_timelapse_camera_names(settings)
        if not camera_names:
            return False
        base_dir = Path(settings['base_dir'])
        hourly_dir = Path(settings.get('hourly_dir') or (base_dir / 'hourly'))
        daily_dir = Path(settings.get('daily_dir') or (base_dir / 'daily'))
        date_key = target_date.strftime('%Y%m%d')
        sent_cameras = cache.get(cc.DAILY_TIMELAPSE_SENT_CAMERAS) or []
        sent_cameras_set = set(sent_cameras)
        attempts = self._load_attempts(cc.DAILY_TIMELAPSE_ATTEMPTS)
        max_attempts = config.TIMELAPSE_MAX_SEND_ATTEMPTS
        all_sent = True
        sent_any = False
        for camera in camera_names:
            token = f'{date_key}:{camera}'
            if token in sent_cameras_set:
                sent_any = True
                continue
            attempt_count = attempts.get(token, 0)
            if attempt_count >= max_attempts:
                all_sent = False
                continue
            attempts[token] = attempt_count + 1
            self._save_attempts(cc.DAILY_TIMELAPSE_ATTEMPTS, attempts)
            clip_path = daily_dir / camera / f'{date_key}_timelapse.mp4'
            if not clip_path.exists():
                hourly_files = sorted((hourly_dir / camera).glob(f'{date_key}_*.mp4'))
                hourly_files = [f for f in hourly_files if f.is_file()]
                if not hourly_files:
                    self.logger.info('Timelapse daily push: no hourly clips for camera %s on %s', camera, date_key)
                    all_sent = False
                    continue
                if not self._stitch_hourly_clips(hourly_files, clip_path):
                    all_sent = False
                    continue
            caption = f'Timelapse ({camera}) {date_key} full day'
            resp = utils.send_telegram_video(str(clip_path),
                                             caption=caption,
                                             timeout=config.TIMELAPSE_SEND_TIMEOUT)
            if resp is not None and resp.ok:
                sent_any = True
                sent_cameras_set.add(token)
                cache.set(cc.DAILY_TIMELAPSE_SENT_CAMERAS, list(sent_cameras_set))
                attempts.pop(token, None)
                self._save_attempts(cc.DAILY_TIMELAPSE_ATTEMPTS, attempts)
            else:
                all_sent = False
                self.logger.warning('Timelapse daily push telegram send failed for camera %s on %s',
                                    camera, date_key)
        return all_sent and sent_any

    @staticmethod
    def _get_timelapse_camera_names(settings):
        names = settings.get('camera_names') or []
        if isinstance(names, str):
            names = [names]
        fallback = settings.get('camera_name')
        if not names and fallback:
            names = [fallback]
        cleaned = [name.strip() for name in names if name and name.strip()]
        # preserve order, drop duplicates
        seen = set()
        unique = []
        for name in cleaned:
            if name not in seen:
                seen.add(name)
                unique.append(name)
        return unique

    def _ensure_hourly_clip(self, date_key, hour, camera, captures_dir, hourly_dir, framerate):
        hour_label = f'{hour:02d}'
        hourly_path = hourly_dir / camera / f'{date_key}_{hour_label}.mp4'
        if hourly_path.exists():
            return hourly_path

        frames_dir = captures_dir / camera / date_key / hour_label
        if not frames_dir.exists():
            return None
        images = sorted(frames_dir.glob('*.jpg'))
        if not images:
            return None
        hourly_path.parent.mkdir(parents=True, exist_ok=True)
        pattern = str(frames_dir / '*.jpg')
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(framerate),
            '-pattern_type', 'glob',
            '-i', pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(hourly_path)
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=config.TIMELAPSE_FFMPEG_TIMEOUT)
        except subprocess.TimeoutExpired:
            self.logger.error('Timed out building hourly timelapse for %s %s %s',
                              camera, date_key, hour_label)
            return None
        except subprocess.CalledProcessError as exc:
            self.logger.error('Failed building hourly timelapse for %s %s %s: %s',
                              camera, date_key, hour_label, exc)
            if exc.stderr:
                self.logger.debug('ffmpeg stderr: %s', exc.stderr.decode(errors='ignore'))
            return None
        return hourly_path

    def _stitch_hourly_clips(self, hourly_files, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile('w', delete=False) as concat_file:
            for clip in hourly_files:
                concat_file.write(f"file '{self._escape_ffmpeg_path(clip)}'\n")
            concat_name = concat_file.name

        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_name,
            '-c', 'copy',
            str(output_path)
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=config.TIMELAPSE_FFMPEG_TIMEOUT)
        except subprocess.TimeoutExpired:
            self.logger.error('Timed out stitching timelapse summary for %s', output_path)
            return False
        except subprocess.CalledProcessError as exc:
            self.logger.error('Failed stitching timelapse summary for %s: %s', output_path, exc)
            if exc.stderr:
                self.logger.debug('ffmpeg stderr: %s', exc.stderr.decode(errors='ignore'))
            return False
        finally:
            Path(concat_name).unlink(missing_ok=True)
        return True

    @staticmethod
    def _escape_ffmpeg_path(path):
        return str(path).replace("'", r"'\''")

    @schedule_method
    def compress_videos(self):
        if (self.is_in_range('cameras_on') or not self.is_compression_thread_available() or config.DISABLE_DB or
                config.IS_COMPRESSION_DISABLED or cache.get_current_experiment()):
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
        if not self.is_in_range('run_pose') or cache.get_current_experiment() or self.dlc_on.is_set() or \
                not config.IS_RUN_NIGHTLY_POSE_ESTIMATION:
            return

        multiprocessing.Process(target=_run_pose_callback, args=(self.dlc_on, self.dlc_errors_cache)).start()
        self.dlc_on.set()

    @schedule_method
    def tracking_pose(self):
        if predict_tracking is None or not self.is_in_range('tracking_pose') or \
                cache.get_current_experiment() or self.dlc_on.is_set() or \
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

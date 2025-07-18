import re
import json
import redis
import time
import inspect
import queue
import threading
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import multiprocessing as mp
import redis.exceptions
import websockets
import asyncio
from websockets.sync.client import connect

from cache import CacheColumns as cc, RedisCache
import config
from loggers import get_logger, get_process_logger
from utils import Serializer, run_in_thread, run_command, send_telegram_message
from db_models import ORM
from periphery_integration import PeripheryIntegrator, ArenaListener, MQTTListener


class DoubleEvent(Exception):
    """Double event"""


class Subscriber(threading.Thread):
    sub_name = ''

    def __init__(self, stop_event: threading.Event, log_queue=None, channel=None, callback=None):
        super().__init__()
        self.cache = RedisCache()
        self.channel = channel or config.subscription_topics[self.sub_name]
        self.name = self.sub_name or self.channel.split('/')[-1]
        if log_queue is None:
            self.logger = get_logger(str(self))
        else:
            self.logger = get_process_logger(str(self), log_queue)
        self.stop_event = stop_event
        self.callback = callback

    def __str__(self):
        return self.name

    def run(self):
        try:
            p = self.cache._redis.pubsub()
            p.psubscribe(self.channel)
            self.logger.debug(f'start listening on {self.channel}')
            while not self.stop_event.is_set():
                message_dict = p.get_message(ignore_subscribe_messages=True, timeout=1)
                if message_dict:
                    channel, data = self.parse_message(message_dict)
                    self._run(channel, data)
                    if self.name == 'arena_shutdown':
                        return
                time.sleep(0.01)
            p.punsubscribe()
        except BrokenPipeError:
            print(f'{self.name} process is down!')
            return
        except Exception as exc:
            print(f'error in {self.name}; {exc}')
        finally:
            self.close()
            # self.logger.exception(f'Error in subscriber {self.name}')

    def _run(self, channel, data):
        if self.callback is not None:
            self.callback(channel, data)

    def close(self):
        pass

    @staticmethod
    def parse_message(msg: dict):
        channel = msg.get('channel')
        payload = msg.get('data')
        if isinstance(channel, bytes):
            channel = channel.decode()
        if isinstance(payload, bytes):
            payload = payload.decode()
        return channel, payload


class ExperimentLogger(Subscriber):
    def __init__(self, stop_event: threading.Event, log_queue, channel=None, callback=None):
        super().__init__(stop_event, log_queue, channel, callback)
        self.config = config.experiment_metrics[self.name]
        self.orm = ORM()

        self.ws = None
        for _ in range(3):
            try:
                self.ws = connect(config.WEBSOCKET_URL)
            except ConnectionRefusedError:
                time.sleep(2)
        if self.ws is None:
            self.logger.error('Unable to connect to websocket server')

    def __str__(self):
        return f'{self.name}-Logger'

    def run(self):
        self.logger.debug(f'start listening using websockets on {self.channel}')
        while not self.stop_event.is_set():
            try:
                message = self.ws.recv(timeout=1)
                message_dict = json.loads(message)
                if message_dict.get('channel') == self.channel:
                    try:
                        payload = json.loads(message_dict.get('payload', '{}'))
                        payload = self.convert_time_fields(payload)
                        self.payload_action(payload)

                        if self.config.get('is_write_csv'):
                            self.save_to_csv(payload)
                        if self.config.get('is_write_db'):
                            self.commit_to_db(payload)
                    except DoubleEvent:
                        pass
                    except BrokenPipeError:
                        print(f'{self.name} process is down!')
                        return
                    except Exception as exc:
                        self.logger.exception(f'Unable to parse log payload of {self.name}: {exc}')
            except TimeoutError:
                continue

    def payload_action(self, payload):
        pass

    def ms2datetime(self, x, to_string=True):
        try:
            x = pd.to_datetime(x, unit='ms').tz_localize('utc').tz_convert('Asia/Jerusalem')
            if to_string:
                x = x.isoformat()
        except Exception as exc:
            self.logger.exception(f'Unable to convert ms time to local; {exc}')
        return x

    @property
    def time_fields(self):
        return ['time', 'start_time', 'end_time']

    def convert_time_fields(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            return payload
        for k, v in payload.copy().items():
            if k in self.time_fields:
                payload[k] = self.ms2datetime(v)
            elif isinstance(v, list):
                payload[k] = [self.convert_time_fields(x) for x in v]
            elif isinstance(v, dict):
                payload[k] = self.convert_time_fields(v)

        return payload

    def commit_to_db(self, payload):
        pass

    def save_to_csv(self, payload, filename=None):
        df = self.to_dataframe(payload)
        try:
            filename = self.get_csv_filename(filename)
            if filename.exists():
                df.to_csv(filename, mode='a', header=False)
            else:
                df.to_csv(filename)
                self.logger.debug(f'Creating analysis log: {filename}')
        except Exception as exc:
            self.logger.exception(f'ERROR saving event to csv; {exc}')

    def get_csv_filename(self, filename=None) -> Path:
        if self.cache.get_current_experiment():
            if self.config.get('is_overall_experiment'):
                parent = self.cache.get(cc.EXPERIMENT_PATH)
            else:
                parent = self.cache.get(cc.EXPERIMENT_BLOCK_PATH)
        else:
            parent = f'events/{datetime.today().strftime("%Y%m%d")}'
            Path(parent).mkdir(parents=True, exist_ok=True)

        return Path(f'{parent}/{filename or self.config["csv_file"]}')

    @staticmethod
    def to_dataframe(payload) -> pd.DataFrame:
        if not isinstance(payload, (list, tuple)):
            payload = [payload]
        return pd.DataFrame(payload)


class TouchLogger(ExperimentLogger):
    def __init__(self, *args, **kwargs):
        super(TouchLogger, self).__init__(*args, **kwargs)
        self.periphery = PeripheryIntegrator()
        self.touches_queue = queue.Queue()
        self.start_touches_receiver_thread()

    def payload_action(self, payload):
        try:
            self.touches_queue.put_nowait(payload)
        except queue.Full:
            pass
        except Exception as exc:
            self.logger.error(f'Error in image sink; {exc}')

    def start_touches_receiver_thread(self):
        def loop(q):
            self.logger.info('touch listener has started')
            last_touch_ts = None
            while not self.stop_event.is_set():
                try:
                    payload = q.get_nowait()
                    ts = payload.get('time')
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts)

                    dt = (ts - last_touch_ts).total_seconds() if last_touch_ts else 0
                    if last_touch_ts and ts and dt < 0.2:
                        continue

                    last_touch_ts = ts
                    self.logger.info(f'Received touch event; timestamp={ts}; '
                                     f'time passed from last reward: {dt:.1f} seconds')
                    self.handle_hit(payload)
                except queue.Empty:
                    pass
            self.logger.debug('touches receiver thread is closed')

        t = threading.Thread(target=loop, args=(self.touches_queue,))
        t.start()

    def handle_hit(self, payload):
        if self.cache.get(cc.IS_ALWAYS_REWARD) and payload.get('is_reward_bug') and \
                (payload.get('is_hit') or payload.get('is_reward_any_touch')):
            self.periphery.feed()
            return True

    @run_in_thread
    def commit_to_db(self, payload):
        self.orm.commit_strike(payload)


class TrialDataLogger(ExperimentLogger):
    def save_to_csv(self, payload, filename=None):
        for key, csv_path in self.config["csv_file"].items():
            if key == 'trials_data':
                payload_ = {k: v for k, v in payload.items() if k not in ['bug_trajectory', 'video_frames', 'app_events']}
            else:
                payload_ = payload.get(key)
                
            if payload_:
                super().save_to_csv(payload_, filename=csv_path)

    def commit_to_db(self, payload):
        self.orm.update_trial_data(payload)


class TemperatureLogger(Subscriber):
    def __init__(self, stop_event: threading.Event, log_queue, **kwargs):
        super().__init__(stop_event, log_queue, channel=config.subscription_topics['temperature'])
        self.n_tries = 5
        self.orm = ORM()

    def run(self):
        def callback(payload):
            if payload:
                self.commit_to_db(payload)
            # self.cache.publish(config.subscription_topics['temperature'], payload)

        try:
            listener = ArenaListener(is_debug=False, stop_event=self.stop_event, callback=callback)
        except BrokenPipeError:
            print(f'{self.name} process is down!')
            return
        except Exception as exc:
            self.logger.error(f'Error loading temperature listener; {exc}')
            return
        self.logger.debug('read_temp started')
        listener.loop()

    def commit_to_db(self, payload):
        try:
            self.orm.commit_temperature(payload)
        except:
            self.logger.exception('Error committing temperature to DB')


class AppHealthCheck(Subscriber):
    sub_name = 'app_healthcheck'

    def run(self):
        try:
            p = self.cache._redis.pubsub()
            p.psubscribe(self.channel)
            self.logger.debug(f'start listening on {self.channel}')
            while not self.stop_event.is_set():
                self.cache.publish_command(self.sub_name)
                time.sleep(0.01)
                open_apps_hosts = set()
                for _ in range(3):
                    try:
                        message_dict = p.get_message(ignore_subscribe_messages=True, timeout=1)
                    except redis.exceptions.ConnectionError:
                        message_dict = None
                    if message_dict and message_dict.get('data'):
                        try:
                            message_dict = json.loads(message_dict.get('data').decode('utf-8'))
                            open_apps_hosts.add(message_dict['host'])
                        except:
                            pass

                if open_apps_hosts:
                    if len(open_apps_hosts) > 1:
                        self.logger.warning(f'more than 1 pogona hunter apps are open: {open_apps_hosts}')
                    self.cache.set(cc.OPEN_APP_HOST, list(open_apps_hosts)[0])
                else:
                    if self.cache.get(cc.OPEN_APP_HOST):
                        self.cache.delete(cc.OPEN_APP_HOST)
                time.sleep(2)
            p.punsubscribe()
        except BrokenPipeError:
            print(f'{self.name} process is down!')
            return
        except:
            self.logger.exception(f'Error in subscriber {self.name}')


class PeripheryHealthCheck(Subscriber):
    sub_name = 'periphery_healthcheck'
    def __init__(self, stop_event: threading.Event, log_queue=None, channel=None, callback=None):
        super().__init__(stop_event, log_queue, channel, callback)
        self.last_health_check_time = time.time()
        self.last_publish_error_time = 0
        self.last_action_time = 0
        self.periphery = PeripheryIntegrator()


        cfg = config.PERIPHERY_HEALTHCHECK
        self.check_interval    = cfg['CHECK_INTERVAL']
        self.max_check_delay   = cfg['MAX_CHECK_DELAY']  # if there's no new healthcheck message for 10 seconds log error
        self.max_publish_delay = cfg['MAX_PUBLISH_DELAY']
        self.max_action_delay  = cfg['MAX_ACTION_DELAY']

    def run(self):
        try:
            def hc_callback(payload):
                now = time.time()
                toggles_state = self.periphery.check_toggles_states()
                if any("light" in key for key in toggles_state):
                    self.last_health_check_time = now

            listener = MQTTListener(topics=['arena/listening', 'arena/value'], is_debug=False, callback=hc_callback)
            self.logger.debug('periphery_healthcheck started')
            while not self.stop_event.is_set():
                listener.loop()
                time.sleep(self.check_interval)
                if time.time() - self.last_health_check_time > self.max_check_delay:
                    if self.last_publish_error_time and time.time() - self.last_publish_error_time < self.max_publish_delay:
                        # self.logger.debug(f"No healthcheck received for {time.time() - self.last_health_check_time:.1f} seconds, last action was {time.time() - self.last_action_time:.1f} seconds ago")
                        continue
                    hc = self.periphery.check_periphery_healthcheck()
                    tc = self.periphery.check_toggles_states()
                    self.logger.error('Arena periphery MQTT bridge is down')
                    tel_message = f'Arena periphery MQTT bridge is down; ' \
                                  f'last healthcheck received {time.time() - self.last_health_check_time:.1f} seconds ago; ' \
                                  f'last action was {time.time() - self.last_action_time:.1f} seconds ago'
                    if hc:
                        tel_message += f'; periphery_healthcheck={hc}'
                    if tc:
                        tel_message += f'; toggles_healthcheck={tc}'
                    send_telegram_message(tel_message)
                    if not self.last_action_time or time.time() - self.last_action_time > self.max_action_delay:
                        self.logger.warning('Running restart for arena periphery process')
                        send_telegram_message('Restarting arena periphery process')
                        next(run_command('supervisorctl restart reptilearn_arena'))
                        self.last_action_time = time.time()
                        time.sleep(4)

                    self.last_publish_error_time = time.time()

        except:
            self.logger.exception(f'Error in subscriber {self.name}')

class WebSocketServer(mp.Process):
    def __init__(self, stop_event: threading.Event):
        self.ws = None
        self.sub = None
        self.connections = set()
        self.stop_event = stop_event
        super().__init__()

    def run(self):
        try:
            asyncio.run(self.main_loop())
        except Exception as e:
            print(f'websocket server stopped running!; {e}')
            send_telegram_message('WebSocket server stopped running!')

    async def echo(self, websocket):
        self.connections.add(websocket)
        try:
            async for message in websocket:
                websockets.broadcast(self.connections, message)
                # Handle disconnecting clients
        except websockets.exceptions.ConnectionClosed as e:
            print(f"A client just disconnected from websocket: {e}")
        finally:
            self.connections.remove(websocket)

    async def main_loop(self):
        host, port = config.WEBSOCKET_URL.replace('ws://', '').split(':')
        async with websockets.serve(self.echo, host, int(port), ping_interval=None, max_size=2**24):
            while not self.stop_event.is_set():
                await asyncio.sleep(0.1)


class WebSocketPublisher(Subscriber):
    def __init__(self, stop_event: threading.Event, log_queue, **kwargs):
        logging.getLogger("websockets").addHandler(logging.NullHandler())
        logging.getLogger("websockets").propagate = False
        super().__init__(stop_event, log_queue, channel='cmd/visual_app/*')
        self.ws = None
        for _ in range(3):
            try:
                self.ws = connect(config.WEBSOCKET_URL)
            except ConnectionRefusedError:
                time.sleep(2)
        if self.ws is None:
            self.logger.error('Unable to connect to websocket server')

    def _run(self, channel, data):
        payload_ = json.dumps({'channel': channel, 'payload': data})
        if self.ws is not None:
            self.ws.send(payload_)

    def close(self):
        if self.ws is not None:
            self.ws.close()


def start_management_subscribers(arena_shutdown_event, log_queue, subs_dict):
    """Start all subscribers that must listen as long as an arena management instance initiated"""
    threads = dict()
    threads['websocket_server'] = WebSocketServer(arena_shutdown_event)
    threads['websocket_server'].start()
    threads['websocket_publisher'] = WebSocketPublisher(arena_shutdown_event, log_queue)
    threads['websocket_publisher'].start()
    for topic, callback in subs_dict.items():
        threads[topic] = Subscriber(arena_shutdown_event, log_queue,
                                    config.subscription_topics[topic], callback)
        threads[topic].start()

    # if config.IS_USE_REDIS:
    #     threads['app_healthcheck'] = AppHealthCheck(arena_shutdown_event, log_queue)
    #     threads['app_healthcheck'].start()
    if not config.DISABLE_PERIPHERY:
        threads['temperature'] = TemperatureLogger(arena_shutdown_event, log_queue)
        threads['temperature'].start()
        if config.SUBSCRIBE_TO_MQTT:
            threads['periphery_healthcheck'] = PeripheryHealthCheck(arena_shutdown_event, log_queue, channel='mqtt')
            threads['periphery_healthcheck'].start()
        # threads['periphery_healthcheck'] = PeripheryHealthCheckPassive(
        #     arena_shutdown_event,
        #     log_queue
        # )
        # threads['periphery_healthcheck'].start()
    return threads


def start_experiment_subscribers(arena_shutdown_event, log_queue):
    """Start the subscribers for a running experiment"""
    threads = {}
    for channel_name, d in config.experiment_metrics.items():
        thread_name = f'metric_{channel_name}'
        if channel_name == 'touch':
            logger_cls = TouchLogger
        elif channel_name == 'trial_data':
            logger_cls = TrialDataLogger
        else:
            logger_cls = ExperimentLogger

        threads[thread_name] = logger_cls(arena_shutdown_event, log_queue,
                                          channel=config.subscription_topics[channel_name])
        threads[thread_name].start()
    return threads

import json
import threading
import time
from datetime import datetime
import paho.mqtt.client as mqtt
import paho.mqtt.subscribe as subscribe
from pathlib import Path
import utils
from cache import RedisCache, CacheColumns as cc
from loggers import get_logger
from db_models import ORM
import config


CONFIG_PATH = 'configurations/periphery_config.json'


class PeripheryIntegrator:
    """class for communicating with reptilearn's arena.py"""
    def __init__(self):
        self.logger = get_logger('Periphery')
        self.logger.debug('periphery integration created')
        self.cache = RedisCache()
        self.mqtt_client = mqtt.Client()
        self.orm = ORM()
        self.periphery_config = self.read_config()
        if self.periphery_config and 'arena' in self.periphery_config:
            self.devices = self.periphery_config['arena']['interfaces']
        else:
            self.devices = []

    @staticmethod
    def read_config() -> dict:
        d = {}
        if Path(CONFIG_PATH).exists():
            with open(CONFIG_PATH, 'r') as f:
                d = json.load(f)
        return d

    def save_config_to_file(self):
        with open(CONFIG_PATH, 'w') as f:
            json.dump(self.periphery_config, f, indent=4)

    def switch(self, name, state):
        assert state in [0, 1]
        self.mqtt_publish(config.mqtt['publish_topic'], f'["set","{name}",{state}]')

    def cam_trigger(self, state):
        assert state in [0, 1]
        self.mqtt_publish(config.mqtt['publish_topic'], f'["set","Camera Trigger",{state}]')
        self.cache.set(cc.CAM_TRIGGER_STATE, state)

    def publish_cam_trigger_state(self):
        self.mqtt_publish(config.mqtt['publish_topic'], f'["get","Camera Trigger"]')

    def change_trigger_fps(self, new_fps):

        new_duration = round(1000 / new_fps)
        trig_inters = self.periphery_config['camera trigger']['interfaces'][0]
        trig_inters['pulse_len'] = new_duration
        self.save_config_to_file()
        next(utils.run_command('cd ../docker && docker-compose restart periphery'))

        # self.mqtt_publish('change_cam_trigger_duration', new_duration)

        time.sleep(5)
        self.cam_trigger(1)
        self.logger.info(f'Published cam trigger FPS change to {new_fps}')

    def feed(self, is_manual=False):
        if self.cache.get(cc.IS_REWARD_TIMEOUT):
            return
        self.cache.set(cc.IS_REWARD_TIMEOUT, True)

        feed_counts = self.get_feeders_counts()
        if all([c == 0 for c in feed_counts.values()]):
            self.logger.warning('No reward left in feeders')
            return

        for feeder_name, count in self.get_feeders_counts().items():
            if count == 0:
                continue

            self.mqtt_publish(config.mqtt['publish_topic'], f'["dispense","{feeder_name}"]')
            self.update_reward_count(feeder_name, count - 1)
            self.logger.info(f'Reward was given by {feeder_name}')
            self.orm.commit_reward(datetime.now(), is_manual=is_manual)
            break

    def mqtt_publish(self, topic, payload):
        self.mqtt_client.connect(config.mqtt['host'], config.mqtt['port'], keepalive=60)
        self.mqtt_client.publish(topic, payload)

    def get_feeders_counts(self) -> dict:
        counts = self.cache.get(cc.REWARD_LEFT)
        if counts is None:
            counts = [0 for _ in self.feeders]
        return {n: int(c) for n, c in zip(self.feeders, counts)}

    def update_reward_count(self, feeder_name, reward_count):
        c = self.get_feeders_counts()
        c[feeder_name] = reward_count
        new_counts = [str(c.get(feeder, 0)) for feeder in self.feeders]
        self.cache.set(cc.REWARD_LEFT, new_counts)

    @property
    def toggles(self) -> list:
        return [dev['name'] for dev in self.devices if dev['type'] == 'line']

    @property
    def feeders(self) -> list:
        feeds = [(dev['name'], dev['order']) for dev in self.devices if dev['type'] == 'feeder']
        return [x[0] for x in sorted(feeds, key=lambda x: x[1])]


class MQTTListener:
    topics = []

    def __init__(self, topics=None, callback=None, is_debug=True, stop_event=None, is_loop_forever=False):
        self.client = mqtt.Client()
        self.callback = callback
        self.topics = topics or self.topics
        self.topics = self.topics if isinstance(self.topics, (tuple, list)) else [self.topics]
        self.stop_event = stop_event
        self.is_debug = is_debug
        self.is_loop_forever = is_loop_forever
        self.is_initiated = False

    def init(self):
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(config.mqtt['host'], config.mqtt['port'])
        self.is_initiated = True

    def loop(self):
        if not self.is_initiated:
            self.init()
        if self.stop_event is not None:
            while not self.stop_event.is_set():
                self.client.loop()
        elif self.is_loop_forever:
            self.client.loop_forever()
        else:
            self.client.loop()

    def on_connect(self, client, userdata, flags, rc):
        if self.is_debug:
            print(f'MQTT connecting to host: {config.mqtt["host"]}; rc: {rc}')
        client.subscribe([(topic, 0) for topic in self.topics])

    def parse_payload(self, payload):
        return payload

    def on_message(self, client, userdata, msg):
        payload = self.parse_payload(msg.payload.decode('utf-8'))
        if self.callback is not None:
            self.callback(payload)
        if self.is_debug:
            print(f'received message with topic {msg.topic}: {payload}')


class ArenaListener(MQTTListener):
    topics = ['arena/value']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = RedisCache()

    def parse_payload(self, payload):
        payload = json.loads(payload)
        if not isinstance(payload, dict):
            return payload

        for name, value in payload.items():
            if name in config.mqtt['temperature_sensors']:
                payload = self.parse_temperature(name, value)
            elif name == 'Camera Trigger':
                payload = self.parse_trigger_state(value)
        return payload

    @staticmethod
    def parse_temperature(name, data):
        res = {}
        if not data:
            return
        elif len(data) == 1:
            res[name] = data[0]
        else:
            res.update({f'{name}{i}': v for i, v in enumerate(data)})
        return res

    def parse_trigger_state(self, value):
        if value in [0, 1]:
            self.cache.set(cc.CAM_TRIGGER_STATE, value)
        return


if __name__ == "__main__":
    def hc_callback(payload):
        print(payload)


    e = threading.Event()
    listener = ArenaListener(is_debug=False, callback=hc_callback, is_loop_forever=True, stop_event=e)
    t = threading.Thread(target=listener.loop)
    t.start()

    pi = PeripheryIntegrator()
    pi.mqtt_publish(config.mqtt['publish_topic'], f'["set","Camera Trigger",1]')
    time.sleep(1)
    pi.mqtt_publish(config.mqtt['publish_topic'], f'["get","Camera Trigger"]')
    time.sleep(1)
    pi.mqtt_publish(config.mqtt['publish_topic'], f'["set","Camera Trigger",0]')
    time.sleep(1)
    pi.mqtt_publish(config.mqtt['publish_topic'], f'["get","Camera Trigger"]')
    time.sleep(10)
    e.set()
    # listener.loop()
    # while True:
    #     listener.loop()
    #     time.sleep(0.1)


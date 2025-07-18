import json
import sys
import time

import pandas as pd
from tqdm.auto import tqdm
from functools import wraps
import numpy as np
from datetime import datetime, timedelta, date
from sqlalchemy import (Column, Integer, String, DateTime, Float, ForeignKey, Boolean, create_engine, cast, Date, and_,
                        desc, func, alias, not_)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.dialects.postgresql import JSON
import config
from cache import RedisCache, CacheColumns as cc
from loggers import get_logger

Base = declarative_base()


class User(Base):
    __tablename__ = 'users'

    name = Column(String, primary_key=True)
    password = Column(String)
    authenticated = Column(Boolean, default=False)

    def is_active(self):
        """True, as all users are active."""
        return True

    def get_id(self):
        return self.name

    def is_authenticated(self):
        """Return True if the user is authenticated."""
        return self.authenticated

    def is_anonymous(self):
        """False, as anonymous users aren't supported."""
        return False


class Animal(Base):
    __tablename__ = 'animals'

    id = Column(Integer, primary_key=True)
    animal_id = Column(String)
    start_time = Column(DateTime)
    end_time = Column(DateTime, nullable=True)
    sex = Column(String)
    arena = Column(String)
    dwh_key = Column(Integer, nullable=True)


class Schedule(Base):
    __tablename__ = 'schedules'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime)
    animal_id = Column(String)
    arena = Column(String)
    experiment_name = Column(String)


class ModelGroup(Base):
    __tablename__ = 'model_groups'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    versions = relationship('ModelVersion')


class ModelVersion(Base):
    __tablename__ = 'model_versions'

    id = Column(Integer, primary_key=True)
    create_date = Column(DateTime)
    version = Column(String)
    folder = Column(String)
    model_group_id = Column(Integer, ForeignKey('model_groups.id'))
    is_active = Column(Boolean, default=True)


class Experiment(Base):
    __tablename__ = 'experiments'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    start_time = Column(DateTime)
    end_time = Column(DateTime, nullable=True, default=None)
    animal_id = Column(String)
    cameras = Column(String)  # list
    num_blocks = Column(Integer)
    extra_time_recording = Column(Integer)
    time_between_blocks = Column(Integer)
    experiment_path = Column(String)
    arena = Column(String)
    blocks = relationship('Block')
    dwh_key = Column(Integer, nullable=True)


class Block(Base):
    __tablename__ = 'blocks'

    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime, nullable=True, default=None)
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    block_id = Column(Integer)  # the ID of the block inside the experiment
    num_trials = Column(Integer)
    trial_duration = Column(Integer)
    iti = Column(Integer)
    block_type = Column(String)
    bug_types = Column(String)  # originally a list
    bug_speed = Column(Integer)
    bug_size = Column(Integer)
    is_default_bug_size = Column(Boolean)
    exit_hole = Column(String)
    reward_bugs = Column(String)  # originally a list
    reward_any_touch_prob = Column(Float)
    media_url = Column(String, nullable=True)
    movement_type = Column(String)
    holes_height_scale = Column(Float, nullable=True)
    background_color = Column(String, nullable=True)
    agent_label = Column(String, nullable=True)  # The key in the trials dict of the agent_config
    tags = Column(String, nullable=True, default='')
    strikes = relationship('Strike')
    trials = relationship('Trial')
    videos = relationship('Video')
    dwh_key = Column(Integer, nullable=True)


class Trial(Base):
    """Trial refers to a single trajectory of the bug/media on the screen"""
    __tablename__ = 'trials'

    id = Column(Integer, primary_key=True)
    in_block_trial_id = Column(Integer)  # The ID of the trial inside the block. Starts from 1
    block_id = Column(Integer, ForeignKey('blocks.id'))
    start_time = Column(DateTime)
    end_time = Column(DateTime, nullable=True, default=None)
    duration = Column(Float, nullable=True, default=None)
    bug_trajectory = Column(JSON, nullable=True)
    exit_hole = Column(String, nullable=True)
    bug_speed = Column(Float, nullable=True)
    trial_bugs = Column(String, nullable=True)
    bug_sizes = Column(String, nullable=True)
    extra = Column(JSON, nullable=True)
    tags = Column(String, nullable=True, default='')
    strikes = relationship('Strike')
    dwh_key = Column(Integer, nullable=True)


class Temperature(Base):
    __tablename__ = 'temperatures'

    id = Column(Integer, primary_key=True)
    time = Column(DateTime)
    value = Column(Float)
    arena = Column(String)
    sensor = Column(String)
    block_id = Column(Integer, ForeignKey('blocks.id'), nullable=True)


class Strike(Base):
    __tablename__ = 'strikes'

    id = Column(Integer, primary_key=True)
    time = Column(DateTime)
    is_hit = Column(Boolean)
    is_reward_bug = Column(Boolean)
    is_climbing = Column(Boolean)
    x = Column(Float)
    y = Column(Float)
    bug_x = Column(Float)
    bug_y = Column(Float)
    bug_type = Column(String)
    bug_size = Column(Integer)
    in_block_trial_id = Column(Integer, nullable=True)  # The ID of the trial inside the block. Starts from 1
    prediction_distance = Column(Float, nullable=True)
    calc_speed = Column(Float, nullable=True)
    projected_strike_coords = Column(JSON, nullable=True)
    projected_leap_coords = Column(JSON, nullable=True)
    max_acceleration = Column(Float, nullable=True)
    strike_frame = Column(Integer, nullable=True)
    leap_frame = Column(Integer, nullable=True)
    arena = Column(String)
    tags = Column(String, nullable=True, default='')
    block_id = Column(Integer, ForeignKey('blocks.id'), nullable=True)
    trial_id = Column(Integer, ForeignKey('trials.id'), nullable=True)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=True)
    dwh_key = Column(Integer, nullable=True)
    analysis_error = Column(String, nullable=True)


class Video(Base):
    __tablename__ = 'videos'

    id = Column(Integer, primary_key=True)
    cam_name = Column(String)
    path = Column(String)
    start_time = Column(DateTime, nullable=True)
    fps = Column(Float)
    calc_fps = Column(Float, nullable=True)
    num_frames = Column(Integer, nullable=True)
    animal_id = Column(String, nullable=True)
    frames = Column(JSON, nullable=True)
    compression_status = Column(Integer, default=0)  # 0: no compression, 1: compressed, 2: error
    block_id = Column(Integer, ForeignKey('blocks.id'), nullable=True)
    predictions = relationship('VideoPrediction')
    dwh_key = Column(Integer, nullable=True)


class VideoPrediction(Base):
    __tablename__ = 'video_predictions'

    id = Column(Integer, primary_key=True)
    model = Column(String, nullable=True)
    animal_id = Column(String, nullable=True)
    start_time = Column(DateTime)
    arena = Column(String)
    data = Column(JSON)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=True)
    dwh_key = Column(Integer, nullable=True)


class PoseEstimation(Base):
    __tablename__ = 'pose_estimations'

    id = Column(Integer, primary_key=True)
    cam_name = Column(String)
    start_time = Column(DateTime)
    x = Column(Float)
    y = Column(Float)
    prob = Column(Float, nullable=True)
    bodypart = Column(String, nullable=True)
    model = Column(String, nullable=True)
    animal_id = Column(String, nullable=True)
    angle = Column(Float, nullable=True)
    engagement = Column(Float, nullable=True)
    frame_id = Column(Integer, nullable=True)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=True)
    block_id = Column(Integer, ForeignKey('blocks.id'), nullable=True)
    dwh_key = Column(Integer, nullable=True)


class Reward(Base):
    __tablename__ = 'rewards'

    id = Column(Integer, primary_key=True)
    time = Column(DateTime)
    animal_id = Column(String)
    arena = Column(String)
    is_manual = Column(Boolean, default=False)
    block_id = Column(Integer, ForeignKey('blocks.id'), nullable=True)


def commit_func(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        if config.DISABLE_DB:
            return
        return method(*args, **kwargs)
    return wrapped


class ORM:
    def __init__(self):
        self.engine = get_engine()
        self.session = sessionmaker(bind=self.engine)
        self.current_experiment_id = None
        self.cache = RedisCache()
        self.logger = get_logger('orm')

    @commit_func
    def commit_experiment(self, exp):
        with self.session() as s:
            kwargs = {c.name: getattr(exp, c.name)
                      for c in Experiment.__table__.columns if c.name not in ['id', 'end_time', 'cameras', 'arena', 'dwh_key']}
            kwargs['arena'] = config.ARENA_NAME
            exp_model = Experiment(**kwargs)
            exp_model.cameras = ','.join(list(exp.cameras.keys()))
            s.add(exp_model)
            s.commit()
            self.current_experiment_id = exp_model.id

    @commit_func
    def commit_block(self, blk, is_cache_set=True):
        with self.session() as s:
            kwargs = {c.name: getattr(blk, c.name)
                      for c in Block.__table__.columns if c.name not in ['id', 'end_time', 'dwh_key', 'tags']
                      and not c.foreign_keys}
            kwargs['experiment_id'] = self.current_experiment_id
            for k in ['reward_bugs', 'bug_types']:  # convert lists to strings
                if isinstance(kwargs[k], list):
                    kwargs[k] = ','.join(kwargs[k])
            # fix for blocks with multiple bug speeds. Set 0 if it's list
            if isinstance(kwargs['bug_speed'], list):
                kwargs['bug_speed'] = 0
            b = Block(**kwargs)
            s.add(b)
            s.commit()
            block_id = b.id
            if is_cache_set:
                self.cache.set(cc.CURRENT_BLOCK_DB_INDEX, block_id, timeout=blk.overall_block_duration)
        return block_id

    @commit_func
    def commit_trial(self, trial_dict):
        kwargs = {c.name: trial_dict.get(c.name)
                  for c in Trial.__table__.columns if c.name not in ['id', 'dwh_key', 'tags'] and not c.foreign_keys}
        kwargs['block_id'] = trial_dict.get('block_id') or self.cache.get(cc.CURRENT_BLOCK_DB_INDEX)
        with self.session() as s:
            trial = Trial(**kwargs)
            s.add(trial)
            s.commit()
            trial_id = trial.id
        return trial_id

    @commit_func
    def update_trial_data(self, trial_dict):
        trial_id = trial_dict.get('trial_db_id')
        with self.session() as s:
            trial_model = s.query(Trial).filter_by(id=trial_id).first()
            if trial_model is None:
                self.logger.warning(f'Trial DB id: {trial_id} was not found in DB; cancel update.')
                return
            model_cols = [c.name for c in Trial.__table__.columns]
            for k, v in trial_dict.items():
                if k in model_cols and k not in ['id', 'trial_db_id', 'dwh_key']:
                    setattr(trial_model, k, v)
            s.commit()

    @commit_func
    def update_block_end_time(self, block_id=None, end_time=None):
        block_id = block_id or self.cache.get(cc.CURRENT_BLOCK_DB_INDEX)
        with self.session() as s:
            block_model = s.query(Block).filter_by(id=block_id).first()
            if block_model is None:
                self.logger.warning(f'No block ID found for end_time update')
                return
            block_model.end_time = end_time or datetime.now()
            s.commit()
            self.cache.delete(cc.CURRENT_BLOCK_DB_INDEX)

    @commit_func
    def update_experiment_end_time(self, end_time=None):
        end_time = end_time or datetime.now()
        with self.session() as s:
            exp_model = s.query(Experiment).filter_by(id=self.current_experiment_id).first()
            exp_model.end_time = end_time
            s.commit()

    @commit_func
    def commit_temperature(self, temps):
        self.cache.set(cc.TEMPERATURE, json.dumps(temps))
        with self.session() as s:
            for sensor_name, temp in temps.items():
                t = Temperature(time=datetime.now(), value=temp, block_id=self.cache.get(cc.CURRENT_BLOCK_DB_INDEX),
                                arena=config.ARENA_NAME, sensor=sensor_name)
                s.add(t)
            s.commit()

    def get_temperature(self):
        """return the last temperature value from the last 2 minutes, if none return None"""
        with self.session() as s:
            since = datetime.now() - timedelta(minutes=2)
            temp = s.query(Temperature).filter(and_(Temperature.time > since,
                                               Temperature.arena == config.ARENA_NAME)).order_by(
                Temperature.time.desc()
            ).all()

            res = {}
            for t in temp:
                if t.sensor not in res:
                    res[t.sensor] = t.value
            return res

    @commit_func
    def commit_strike(self, strike_dict):
        kwargs = {c.name: strike_dict.get(c.name)
                  for c in Strike.__table__.columns if c.name not in ['id', 'arena', 'dwh_key', 'tags'] and not c.foreign_keys}
        kwargs['arena'] = config.ARENA_NAME
        kwargs['block_id'] = strike_dict.get('block_id') or self.cache.get(cc.CURRENT_BLOCK_DB_INDEX)
        kwargs['trial_id'] = strike_dict.get('trial_id')

        with self.session() as s:
            strike = Strike(**kwargs)
            s.add(strike)
            s.commit()

    @commit_func
    def commit_video(self, path, fps, cam_name, start_time, animal_id=None, block_id=None):
        animal_id = animal_id or self.cache.get(cc.CURRENT_ANIMAL_ID)
        vid = Video(path=path, fps=fps, cam_name=cam_name, start_time=start_time,
                    block_id=block_id or self.cache.get(cc.CURRENT_BLOCK_DB_INDEX), animal_id=animal_id,
                    compression_status=0 if not self.cache.get(cc.IS_COMPRESSED_LONG_RECORDING) else 1)
        with self.session() as s:
            s.add(vid)
            s.commit()
            vid_id = vid.id
        return vid_id

    @commit_func
    def commit_video_frames(self, timestamps: list, video_id: int):
        with self.session() as s:
            video_model = s.query(Video).filter_by(id=video_id).first()
            if video_model is None:
                self.logger.warning(f'No video found in DB for frames timestamps commit; video id={video_id}')
                return
            video_model.frames = {i: ts for i, ts in enumerate(timestamps)}
            video_model.num_frames = len(timestamps)
            video_model.calc_fps = 1 / np.diff(timestamps).mean()
            s.commit()

    @commit_func
    def commit_video_predictions(self, model: str, data: pd.DataFrame, video_id: int, start_time: datetime,
                                 animal_id=None, arena=config.ARENA_NAME):
        vid_pred = VideoPrediction(model=model, data=data.to_json(), animal_id=animal_id, arena=arena,
                                   video_id=video_id, start_time=start_time)
        with self.session() as s:
            s.add(vid_pred)
            s.commit()

    def update_video_prediction(self, video_stem: str, model: str, data: pd.DataFrame):
        with self.session() as s:
            vid = s.query(Video).filter(Video.path.contains(video_stem)).first()
            if not vid:
                self.logger.warning(f'No video model found for update: {video_stem}')
                return
            elif not vid.predictions:
                self.logger.warning(f'No video predictions found for update: {video_stem}')
                return

            vps = [vp for vp in vid.predictions if vp.model == model]
            if not vps:
                self.logger.warning(f'No video predictions found for update: {video_stem}, model: {model}')
                return
            elif len(vps) > 1:
                self.logger.warning(f'Multiple video predictions found for update: {video_stem}, model: {model}')
                return

            vp = vps[0]
            vp.data = data.to_json()
            s.commit()

    @commit_func
    def commit_pose_estimation(self, cam_name, start_time, x, y, angle, engagement, video_id, model,
                               bodypart, prob, frame_id, animal_id=None, block_id=None):
        animal_id = animal_id or self.cache.get(cc.CURRENT_ANIMAL_ID)
        pe = PoseEstimation(cam_name=cam_name, start_time=start_time, x=x, y=y, angle=angle, animal_id=animal_id,
                            engagement=engagement, video_id=video_id, model=model, bodypart=bodypart, prob=prob,
                            frame_id=frame_id, block_id=block_id or self.cache.get(cc.CURRENT_BLOCK_DB_INDEX)
        )
        with self.session() as s:
            s.add(pe)
            s.commit()

    @commit_func
    def commit_animal_id(self, **data):
        with self.session() as s:
            data['arena'] = config.ARENA_NAME
            animal = Animal(start_time=datetime.now(), **data)
            s.add(animal)
            s.commit()
            self.cache.set(cc.CURRENT_ANIMAL_ID, data['animal_id'])
            self.cache.set(cc.CURRENT_ANIMAL_ID_DB_INDEX, animal.id)

    @commit_func
    def update_animal_id(self, **kwargs):
        with self.session() as s:
            db_index = self.cache.get(cc.CURRENT_ANIMAL_ID_DB_INDEX)
            if db_index is None:
                return
            animal_model = s.query(Animal).filter_by(id=db_index).first()
            if animal_model is None:
                return
            for k, v in kwargs.items():
                setattr(animal_model, k, v)
            s.commit()

        if 'end_time' in kwargs:
            self.cache.delete(cc.CURRENT_ANIMAL_ID)
            self.cache.delete(cc.CURRENT_ANIMAL_ID_DB_INDEX)

    def get_animal_data(self, animal_id):
        with self.session() as s:
            animal = s.query(Animal).filter_by(animal_id=animal_id, arena=config.ARENA_NAME).order_by(
                desc(Animal.start_time)).first()
            if animal is not None:
                animal_dict = {k: v for k, v in animal.__dict__.items() if not k.startswith('_')}
            else:
                if animal_id != 'test':
                    self.logger.error('No Animal was found')
                animal_dict = {}
        return animal_dict

    def get_upcoming_schedules(self):
        with self.session() as s:
            animal_id = self.cache.get(cc.CURRENT_ANIMAL_ID)
            schedules = s.query(Schedule).filter(Schedule.date >= datetime.now(),
                                                 Schedule.animal_id == animal_id,
                                                 Schedule.arena == config.ARENA_NAME).order_by(Schedule.date)
        return schedules

    def commit_multiple_schedules(self, start_date, experiment_name, end_date=None, every=None):
        if not end_date:
            hour, minute = [int(x) for x in config.SCHEDULE_EXPERIMENTS_END_TIME.split(':')]
            end_date = start_date.replace(hour=hour, minute=minute)
        if every:
            curr_date = start_date
            while curr_date < end_date:
                self.commit_schedule(curr_date, experiment_name)
                curr_date += timedelta(minutes=every)
        else:
            self.commit_schedule(start_date, experiment_name)

    def commit_schedule(self, date, experiment_name):
        with self.session() as s:
            animal_id = self.cache.get(cc.CURRENT_ANIMAL_ID)
            sch = Schedule(date=date, experiment_name=experiment_name, animal_id=animal_id, arena=config.ARENA_NAME)
            s.add(sch)
            s.commit()

    def delete_schedule(self, schedule_id):
        with self.session() as s:
            s.query(Schedule).filter_by(id=int(schedule_id)).delete()
            s.commit()

    def commit_reward(self, time, is_manual=False):
        with self.session() as s:
            rwd = Reward(time=time,
                         animal_id=self.cache.get(cc.CURRENT_ANIMAL_ID),
                         block_id=self.cache.get(cc.CURRENT_BLOCK_DB_INDEX),
                         is_manual=is_manual,
                         arena=config.ARENA_NAME)
            s.add(rwd)
            s.commit()

    @staticmethod
    def _parse_day_string(day_string):
        if not day_string:
            day = date.today()
        else:
            day = datetime.strptime(day_string, '%Y-%m-%d').date()
        return day

    def get_rewards_for_day(self, day_string=None, animal_id=None) -> dict:
        """
        @param day_string: string format="%Y-%m-%d", if None use today.
        @param animal_id: string of animal_id
        @return: dict with 'manual' and 'auto' counts
        """
        day = self._parse_day_string(day_string)
        with self.session() as s:
            rewards = s.query(Reward).filter(and_(cast(Reward.time, Date) == day,
                                                  Reward.arena == config.ARENA_NAME))
            if animal_id:
                rewards = rewards.filter_by(animal_id=animal_id)
        return {'manual': rewards.filter_by(is_manual=True).count(),
                'auto': rewards.filter_by(is_manual=False).count()}

    def get_strikes_for_day(self, day_string=None, animal_id=None) -> pd.DataFrame:
        day = self._parse_day_string(day_string)
        filters = [cast(Strike.time, Date) == day,
                   not_(Experiment.animal_id.ilike('%test%'))]
        if animal_id:
            filters.append(Experiment.animal_id == animal_id)
        cols = ['id', 'time', 'bug_type', 'movement_type', 'bug_speed', 'tags', 'x', 'y', 'bug_x', 'bug_y', 'bug_size',
                'in_block_trial_id', 'is_hit', 'is_climbing', 'analysis_error', 'block_id', 'trial_id', 'video_id']
        df = []
        with self.session() as s:
            orm_res = s.query(Strike, Block, Experiment).join(
                Block, Block.id == Strike.block_id).join(
                Experiment, Experiment.id == Block.experiment_id).filter(*filters).all()
            for strk, blk, exp in orm_res:
                d = {c: strk.__dict__.get(c) if c in strk.__dict__ else blk.__dict__.get(c) for c in cols}
                d['miss_distance'] = np.sqrt((strk.x - strk.bug_x) ** 2 + (strk.y - strk.bug_y) ** 2)
                df.append(d)
            df = pd.DataFrame(df)
            if not df.empty:
                df = df.sort_values(by='time')
                df.insert(6, 'miss_distance', df.pop('miss_distance'))
        return df

    def get_trials_for_day(self, day_string=None, animal_id=None) -> pd.DataFrame:
        day = self._parse_day_string(day_string)
        filters = [cast(Trial.start_time, Date) == day,
                   not_(Experiment.animal_id.ilike('%test%'))]
        if animal_id:
            filters.append(Experiment.animal_id == animal_id)
        cols = ['id', 'start_time', 'end_time', 'duration', 'movement_type', 'bug_speed', 'exit_hole', 'trial_bugs',
                'block_id', 'in_block_trial_id']
        df = []
        with self.session() as s:
            orm_res = s.query(Trial, Block, Experiment).join(
                Block, Block.id == Trial.block_id).join(
                Experiment, Experiment.id == Block.experiment_id).filter(*filters).all()
            for tr, blk, exp in orm_res:
                d = {c: tr.__dict__.get(c) if c in tr.__dict__ else blk.__dict__.get(c) for c in cols}
                d['n_strikes'] = len(tr.strikes)
                df.append(d)
        df = pd.DataFrame(df)
        if not df.empty:
            df = df.sort_values(by='start_time')
        return df

    def get_animal_ids_for_summary(self) -> dict:
        """return dict with animal ids that had experiments as keys, and an experiment days list as values """
        res = []
        current_animal_id = self.cache.get(cc.CURRENT_ANIMAL_ID)
        with self.session() as s:
            orm_res = s.query(Block, Experiment).join(
                Experiment, Experiment.id == Block.experiment_id).filter(
                Block.block_type == 'bugs',
                not_(Experiment.animal_id.ilike('%test%')),
                Experiment.animal_id.isnot(None)
            ).all()

            if not orm_res:
                return {}

            for blk, exp in orm_res:
                if exp.animal_id and exp.arena:
                    res.append({'block_id': blk.id, 'date': exp.start_time, 'animal_id': exp.animal_id, 'arena': exp.arena})

            res = pd.DataFrame(res)
            res['exp_day'] = res.date.dt.strftime('%Y-%m-%d')

        res = res.groupby(['arena', 'animal_id']).exp_day.apply(lambda x: sorted(np.unique(x), reverse=True))
        res = {key: group.droplevel(0).to_dict() for key, group in res.groupby(level=0)}
        today = date.today().strftime('%Y-%m-%d')
        if current_animal_id and current_animal_id != 'test':
            arena_dict = res.get(config.ARENA_NAME, {})
            if current_animal_id in arena_dict.keys():
                days = arena_dict.pop(current_animal_id)
                if today not in days:
                    days = [today] + days
            else:
                days = [today]
            d = {current_animal_id: days}
            d.update(arena_dict)
            res[config.ARENA_NAME] = d
        return res

    def today_summary(self):
        summary = {}
        with self.session() as s:
            exps = s.query(Experiment).filter(and_(cast(Experiment.start_time, Date) == date.today(),
                                                   Experiment.arena == config.ARENA_NAME)).all()
            for exp in exps:
                summary.setdefault(exp.animal_id, {'total_trials': 0, 'total_strikes': 0, 'blocks': {}})
                for blk in exp.blocks:
                    block_dict = summary[exp.animal_id]['blocks'].setdefault(blk.movement_type, {'hits': 0, 'misses': 0})
                    for tr in blk.trials:
                        summary[exp.animal_id]['total_trials'] += 1
                    for strk in blk.strikes:
                        summary[exp.animal_id]['total_strikes'] += 1
                        if strk.is_hit:
                            block_dict['hits'] += 1
                        else:
                            block_dict['misses'] += 1
        for animal_id, d in summary.items():
            rewards_dict = self.get_rewards_for_day(animal_id=animal_id)
            d['total_rewards'] = f'{rewards_dict["auto"]} ({rewards_dict["manual"]})'
        return summary


class DWH:
    commit_models = [Animal, Experiment, Block, Trial, Strike, Video]  # VideoPrediction

    def __init__(self):
        self.logger = get_logger('dwh')
        self.local_session = sessionmaker(bind=get_engine())
        self.dwh_session = sessionmaker(bind=create_engine(config.DWH_URL))
        self.keys_table = {}

    def commit(self, n_retries_dwh=3):
        self.logger.info('start DWH commit')
        with self.local_session() as local_s:
            with self.dwh_session() as dwh_s:
                for model in self.commit_models:
                    mappings = []
                    j = 0
                    recs = local_s.query(model).filter(model.dwh_key.is_(None)).all()
                    for rec in tqdm(recs, desc=model.__name__):
                        try:
                            kwargs = {}
                            for c in model.__table__.columns:
                                if c.name in ['id']:
                                    continue
                                value = getattr(rec, c.name)
                                if c.foreign_keys:
                                    fk = list(c.foreign_keys)[0]
                                    dwh_fk = self.keys_table.get(fk.column.table.name, {}).get(value)
                                    if value and not dwh_fk:
                                        # this happened probably due to previously failed runs of DWH commit
                                        dwh_fk = self.get_prev_committed_dwh_fk(local_s, value, fk.column.table)
                                    kwargs[c.name] = dwh_fk if value else None
                                else:
                                    kwargs[c.name] = value

                            r = model(**kwargs)
                            dwh_s.add(r)
                            dwh_s.commit()
                            self.keys_table.setdefault(model.__table__.name, {})[rec.id] = r.id

                            if model == PoseEstimation:
                                mappings.append({'id': rec.id, 'dwh_key': r.id})
                                j += 1
                                if j % 10000 == 0:
                                    local_s.bulk_update_mappings(model, mappings)
                                    local_s.flush()
                                    local_s.commit()
                                    mappings[:] = []
                            else:
                                rec.dwh_key = r.id
                                local_s.commit()
                        except Exception as exc:
                            print(f'Error committing {model.__name__} {rec.id}: {exc}')
                            dwh_s.rollback()

                    if model == PoseEstimation:
                        local_s.bulk_update_mappings(model, mappings)

        self.logger.info('Finished DWH commit')

    def update_model(self, db_model, columns=(), **filters):
        assert isinstance(columns, (list, tuple)), 'columns must be list or tuple'
        with self.local_session() as local_s:
            with self.dwh_session() as dwh_s:
                recs = local_s.query(db_model)
                if filters:
                    recs = recs.filter_by(**filters)
                columns = columns or [c.name for c in db_model.__table__.columns if c.name in ['id', 'dwh_key'] or c.foreign_keys]
                # for rec in tqdm(recs):
                q = recs.filter(db_model.dwh_key.is_not(None))
                total = q.count()
                for rec in tqdm(q.yield_per(10), total=total):
                    dwh_rec = dwh_s.query(db_model).filter_by(id=rec.dwh_key).first()
                    if dwh_rec is None:  # object does not exist on dwh
                        rec.dwh_key = None
                    else:
                        for c in columns:
                            setattr(dwh_rec, c, getattr(rec, c))
                local_s.commit()
                dwh_s.commit()
                print(f'Finished updating columns={columns} for {db_model.__name__}; Total rows updated: {total}')

    @staticmethod
    def get_prev_committed_dwh_fk(s, local_fk, table):
        try:
            return s.query(table).filter_by(id=local_fk).first().dwh_key
        except Exception:
            return


def get_engine():
    return create_engine(config.sqlalchemy_url, pool_size=20, max_overflow=30)


def delete_duplicates(model, col):
    """
    Delete duplicates from a given model based on a given column.

    Args:
        model (SQLAlchemy model): The model to delete duplicates from.
        col (str): The name of the column used to determine duplicates.

    Returns:
        None
    """
    engine = get_engine()
    session = sessionmaker(bind=engine)
    with session() as session:
        inner_q = session.query(func.min(model.id)).group_by(getattr(model, col))
        aliased = alias(inner_q)
        q = session.query(model).filter(~model.id.in_(aliased))
        num_deleted = 0
        for domain in q:
            session.delete(domain)
            num_deleted += 1
        session.commit()
        print(f'Deleted {num_deleted} duplicates for {model.__name__}')


if __name__ == '__main__':
    # delete_duplicates(VideoPrediction, 'video_id')
    # DWH().commit()
    # DWH().update_model(Strike, ['prediction_distance', 'leap_frame'])
    # DWH().update_model(VideoPrediction, ['data'], model='front_head_only_resnet_152')
    DWH().commit()
    sys.exit(0)

    # create all models
    engine = get_engine()
    if not database_exists(engine.url):
        print(f'Database {config.db_name} was created')
        create_database(engine.url)

    # Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    # run "alembic stamp head" to create a versions table in the db

    # Updating
    # If you change something in SQLAlchemy models, you need to create a migration file using:
    # alembic revision --autogenerate -m "migration name"
    #
    # and then to upgrade (make sure there are no open sessions before):
    # alembic upgrade head

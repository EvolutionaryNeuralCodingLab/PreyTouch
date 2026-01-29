import datetime
import math
import time
import pickle
import re
import yaml
import cv2
import traceback
import importlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from pathlib import Path
from scipy.spatial import distance
from scipy.signal import savgol_filter
from scipy.stats import ttest_ind
from multiprocessing.pool import ThreadPool
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import os
if Path('.').resolve().name != 'Arena':
    os.chdir('..')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re
import config
from calibration import CharucoEstimator
from loggers import get_logger
from utils import run_in_thread, Kalman
from sqlalchemy import cast, Date
from db_models import ORM, Experiment, Block, Video, VideoPrediction, Strike, PoseEstimation, Trial
from image_handlers.video_writers import OpenCVWriter, ImageIOWriter
from analysis.pose_utils import put_text, flatten


MOVE_MIN_DISTANCE = 5  # cm
COMMIT_INTERVAL = 2  # seconds


class MissingFile(Exception):
    """"""


class NoFramesVideo(Exception):
    """"""


class ArenaPose:
    """
    ArenaPose is a class that provides a high-level interface for interacting with the pose estimation model. It handles loading and saving data from and to the database, as well as providing methods for predicting poses and analyzing data.

    Args:
        cam_name (str): The name of the camera from which the pose is being estimated.
        predictor (str or object): The pose estimation model to use. If a string, it should be the name of a model from the config file. If an object, it should be an instantiated model.
        is_use_db (bool, optional): Whether to use the database for loading and saving data. Defaults to True.
        orm (object, optional): The ORM object to use for interacting with the database. Defaults to None.
        model_path (str, optional): The path to the model file if it is not located in the default location. Defaults to None.
        is_raise_no_caliber (bool, optional): Whether to raise an exception if no calibration data is available. Defaults to True.

    Attributes:
        cam_name (str): The name of the camera from which the pose is being estimated.
        predictor (object): The pose estimation model used for prediction.
        model_path (str): The path to the model file if it is not located in the default location.
        is_use_db (bool): Whether to use the database for loading and saving data.
        orm (object): The ORM object used for interacting with the database.
        last_commit (tuple): The timestamp and position of the last commit to the database.
        caliber (object): The calibrator used for converting pixel coordinates to real-world coordinates.
        predictions (list): A list of tuples containing the timestamp and position of each prediction.
        current_position (tuple): The current position of the object, calculated using a Kalman filter.
        current_velocity (tuple): The current velocity of the object, calculated using a Kalman filter.
        is_initialized (bool): Whether the calibrator and predictor have been initialized.
        example_writer (object): A video writer used for creating an example video with predictions.

    """

    def __init__(self, cam_name, predictor, is_use_db=True, orm=None, model_path=None, commit_bodypart='head',
                 is_dwh=False, is_raise_no_caliber=True):
        self.cam_name = cam_name
        self.predictor = predictor
        self.model_path = model_path
        self.is_use_db = is_use_db
        self.is_dwh = is_dwh
        self.is_raise_no_caliber = is_raise_no_caliber
        self.load_predictor()
        self.last_commit = None
        self.caliber = None
        self.orm = orm if orm is not None else ORM()
        self.commit_bodypart = commit_bodypart
        self.kinematic_cols = ['x', 'y', 'vx', 'vy', 'ax', 'ay']
        self.time_col = ('time', '')
        self.kalman = None
        self.predictions = []
        self.current_position = (None, None)
        self.current_velocity = None
        self.is_initialized = False
        self.example_writer = None
        self.logger = get_logger('ArenaPose')

    def init(self, img, caliber_only=False):
        """
        Initialize the pose estimator and caliber.

        Args:
            img (numpy.ndarray): The image to use for initialization.
            caliber_only (bool, optional): Whether to only initialize the calibrator and not the predictor. Defaults to False.

        Raises:
            Exception: If the caliber could not be initialized, an exception is raised.

        """
        if not caliber_only:
            self.predictor.init(img)
        if self.cam_name:
            self.caliber = CharucoEstimator(self.cam_name, is_debug=False)
            self.caliber.init(img)
            if not self.caliber.is_on:
                msg = 'Could not initiate caliber; closing ArenaPose'
                if self.is_raise_no_caliber:
                    raise Exception(msg)
                else:
                    self.caliber = None
                    print('Caliber could no loaded. Working without pose calibration')
        self.is_initialized = True

    def init_from_video(self, video_path: [str, Path], caliber_only=False):
        """
        Initialize the pose estimator from a video file.

        Args:
            video_path (str or Path): The path to the video file.
            caliber_only (bool, optional): Whether to only initialize the calibrator and not the predictor. Defaults to False.

        Raises:
            Exception: If the video file does not exist or has 0 frames, an exception is raised.

        """
        if isinstance(video_path, Path):
            video_path = video_path.as_posix()
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if frame is None:
            raise Exception('Video has 0 frames')
        self.init(frame, caliber_only=caliber_only)
        cap.release()

    def change_aruco_markers(self, vid_date=None, video_path: str = None):
        """
        Change the loaded markers to match the video date.

        Args:
            video_path (str): The path to the video file.

        """
        if video_path:
            if not re.match(r'\w+_\d{8}T\d{6}', Path(video_path).stem):
                return
            cam_name, vid_date = Path(video_path).stem.split('_')
        try:
            self.caliber.set_image_date_and_load(vid_date)
        except Exception as e:
            if self.is_raise_no_caliber:
                raise(e)
            else:
                print(f'No caliber loaded: {e}')
                self.caliber = None

    def load(self, video_path=None, video_db_id=None, only_load=False, prefix=''):
        """
        Load the pose data for a video.

        Args:
            video_path (str or Path, optional): The path to the video file. If not provided, the video will be loaded from the database.
            video_db_id (int, optional): The database ID of the video. If not provided, the video will be loaded from the path.
            only_load (bool, optional): Whether to only load the data without analyzing or saving it. Defaults to False.
            prefix (str, optional): A prefix to use for the tqdm progress bar. Defaults to ''.

        Returns:
            pandas.DataFrame: The pose data for the video.

        """
        if self.is_use_db:
            assert video_db_id, 'must provide video_db_id if is_use_db=True'
            try:
                return self._load_from_db(video_db_id)
            except MissingFile:
                if video_path is not None:
                    return self._load_from_local_files(video_path, only_load, prefix)
        else:
            assert video_path, 'must provide video_path if is_use_db=False'
            return self._load_from_local_files(video_path, only_load, prefix)

    def _load_from_db(self, video_db_id):
        with self.orm.session() as s:
            vp = s.query(VideoPrediction).filter_by(video_id=video_db_id,
                                                    model=self.predictor.model_name).order_by(VideoPrediction.id.desc()).first()
            vid = s.query(Video).filter_by(id=video_db_id).first()
            block_id = vid.block_id

        if vp is None:
            raise MissingFile(f'Video prediction was not found for video db id: {video_db_id}')
        df = pd.read_json(vp.data)
        df["('block_id', '')"] = block_id
        df["('animal_id', '')"] = vp.animal_id
        df.columns = pd.MultiIndex.from_tuples([eval(c) for c in df.columns])
        df = df.sort_values(by='time')
        return df

    def _load_from_local_files(self, video_path: Path, only_load=False, prefix=''):
        if isinstance(video_path, str):
            video_path = Path(video_path)
        # if not self.is_initialized:
        #     self.init_from_video(video_path, caliber_only=True)
        cache_path = self.get_predicted_cache_path(video_path)
        if cache_path.exists():
            pose_df = pd.read_parquet(cache_path)
        else:
            if not only_load:
                pose_df = self.predict_video(video_path=video_path, prefix=prefix)
            else:
                raise MissingFile(f'Pose cache file does not exist')
        return pose_df

    def start_new_session(self, fps):
        self.kalman = Kalman(dt=1/fps)
        self.predictions = []

    def load_predictor(self):
        if isinstance(self.predictor, str):
            pred_config = config.load_configuration('predict')[self.predictor]
            prd_class = pred_config['predictor_name']
            prd_module = config.predictors_map[prd_class]
            # prd_module, prd_class = config.arena_modules['predictors'][self.predictor]
            prd_module = importlib.import_module(prd_module)
            self.predictor = getattr(prd_module, prd_class)(self.cam_name, self.model_path)

    def predict_video(self, db_video_id=None, video_path=None, is_save_cache=True, is_create_example_video=False,
                      prefix='', is_tqdm=True):
        """
        predict pose for a given video
        @param db_video_id: The DB index of the video in the videos table
        @param video_path: The path of the video
        @param is_save_cache: save predicted dataframe as parquet file
        @param is_create_example_video: create annotated video with predictions
        @param prefix: to be displayed before the tqdm desc
        @return:
        """
        db_video_id, video_path = self.check_video_inputs(db_video_id, video_path)
        frames_times = self.load_frames_times(db_video_id, video_path)
        bug_traj = self.load_bug_trajectory(db_video_id, video_path)

        pose_df = []
        cap = cv2.VideoCapture(video_path)
        n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(frames_times))
        if n_frames == 0:
            self.tag_error_video(video_path, 'video has 0 frames')
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.start_new_session(fps)
        iters = range(n_frames)
        if not is_tqdm:
            self.logger.info(f'Start video prediction of {video_path}')
        for frame_id in (tqdm(iters, desc=f'{prefix}{Path(video_path).stem}') if is_tqdm else iters):
            ret, frame = cap.read()
            if not self.is_initialized:
                self.init(frame)
                if self.caliber is not None:
                    self.change_aruco_markers(video_path=video_path)

            timestamp = frames_times.loc[frame_id, 'time'].timestamp()
            pred_row, _ = self.predictor.predict(frame, frame_id)
            pred_row = self.analyze_frame(timestamp, pred_row, db_video_id)
            if bug_traj is not None:
                pred_row = self.add_bug_traj(pred_row, bug_traj, timestamp)
            if is_create_example_video:
                self.write_to_example_video(frame, frame_id, pred_row, fps, video_path)
            pose_df.append(pred_row)
        cap.release()

        if not pose_df:
            return
        pose_df = pd.concat(pose_df)
        if is_save_cache:
            self.save_predicted_video(pose_df, video_path)
            self.logger.info(f'Video prediction of {video_path} was saved successfully')
        self.close_example_writer()
        return pose_df

    def predict_frame(self, frame, frame_date=None, is_plot_preds=False) -> pd.DataFrame:
        self.init(frame)
        pred_row, frame = self.predictor.predict(frame, 0, is_plot_preds=is_plot_preds)

        if frame_date is not None and self.caliber is not None:
            self.change_aruco_markers(frame_date)
            bodyparts = pred_row.columns.get_level_values(0).unique().tolist()
            for bp in bodyparts:
                row = pred_row[bp].iloc[0]
                x, y = self.caliber.get_location(row['cam_x'], row['cam_y'])
                pred_row[(bp, 'x')] = x
                pred_row[(bp, 'y')] = y
        
        return frame, pred_row

    def tag_error_video(self, vid_path, msg):
        with self.get_predicted_cache_path(vid_path).with_suffix('.txt').open('w') as f:
            f.write(msg)

    def analyze_frame(self, timestamp: float, pred_row: pd.DataFrame, db_video_id=None):
        """
        Convert pose in pixels to real-world coordinates using the calibrator, and smooth using Kalman.
        In addition, if is_commit is enabled this method would commit to DB the predictions for body parts in
        self.commit_bodypart
        """
        pred_row[self.time_col] = timestamp
        for col in self.kinematic_cols:
            pred_row[(col, '')] = np.nan

        for bodypart in self.predictor.bodyparts:
            if self.is_ready_to_commit(timestamp) and bodypart == self.commit_bodypart:
                # if predictions stack has reached the COMMIT_INTERVAL, and it's the commit_bodypart,
                # then calculate aggregated metrics (e.g, velocity).
                self.aggregate_to_commit(self.predictions.copy(), db_video_id, timestamp)

            cam_x, cam_y = pred_row[(bodypart, 'cam_x')].iloc[0], pred_row[(bodypart, 'cam_y')].iloc[0]
            x, y = np.nan, np.nan
            if self.caliber is not None and not np.isnan(cam_x) and not np.isnan(cam_y):
                x, y = self.caliber.get_location(cam_x, cam_y)

            if bodypart == self.commit_bodypart:
                if not self.kalman.is_initiated:
                    self.kalman.init(x, y)
                x, y, vx, vy, ax, ay = self.kalman.get_filtered(x, y)
                for col in self.kinematic_cols:
                    pred_row.loc[pred_row.index[0], (col, '')] = locals()[col]
                if not np.isnan(x) or not np.isnan(y):
                    self.predictions.append((timestamp, x, y))
            else:
                pred_row.loc[pred_row.index[0], (bodypart, 'x')] = x
                pred_row.loc[pred_row.index[0], (bodypart, 'y')] = y

        pred_row = self.after_analysis_actions(pred_row)
        return pred_row

    def check_video_inputs(self, db_video_id, video_path):
        if self.is_use_db and not db_video_id and video_path:
            db_video_id = self.get_video_db_id(video_path)
        elif not video_path and db_video_id:
            if not self.is_use_db:
                raise Exception('must use DB to get video path but is_use_db=False')
            video_path = self.get_video_path(db_video_id)
        video_path = Path(video_path).as_posix()
        return db_video_id, video_path

    def aggregate_to_commit(self, predictions, db_video_id, timestamp):
        predictions = np.array(predictions)
        x, y = [round(z) for z in predictions[:, 1:3].mean(axis=0)]
        if self.is_use_db and self.is_moved(x, y):
            self.commit_to_db(timestamp, x, y, db_video_id)
        self.predictions = []

    @run_in_thread
    def commit_to_db(self, timestamp, x, y, db_video_id):
        start_time = datetime.datetime.fromtimestamp(timestamp)
        self.orm.commit_pose_estimation(self.cam_name, start_time, x, y, None, None,
                                        db_video_id, model='deeplabcut_v1')
        self.last_commit = (timestamp, x, y)

    def after_analysis_actions(self, pred_row):
        return pred_row

    def load_frames_times(self, db_video_id: int, video_path: str) -> pd.DataFrame:
        if self.is_use_db:
            with self.orm.session() as s:
                vid = s.query(Video).filter_by(id=db_video_id).first()
                if vid.frames is None:
                    raise MissingFile(f'unable to find frames_timestamps in DB for video_id: {db_video_id}')
                frames_ts = pd.DataFrame(vid.frames.items(), columns=['frame_id', 'time']).set_index('frame_id')
        else:
            frames_output_dir = Path(video_path).parent / config.FRAMES_TIMESTAMPS_DIR
            csv_path = frames_output_dir / Path(video_path).with_suffix('.csv').name
            if not csv_path.exists():
                raise MissingFile(f'unable to find frames_timestamps in {csv_path}')
            frames_ts = pd.read_csv(csv_path, names=['frame_id', 'time'], header=0).set_index('frame_id')

        frames_ts['time'] = pd.to_datetime(frames_ts.time, unit='s', utc=True).dt.tz_convert(
            'Asia/Jerusalem').dt.tz_localize(None)
        frames_ts.index = frames_ts.index.astype(int)
        return frames_ts

    def rename_bug_columns(self, df):
        df = df.copy()
        rename_map = {}
        for col in df.columns:
            # match axis + numeric suffix
            m = re.fullmatch(r'([xy])(\d+)$', col)
            if m:
                axis, idx = m.groups()
                rename_map[col] = f'bug{idx}_{axis}'
            # axis
            elif col in ('x', 'y'):
                rename_map[col] = f'bug_{col}'
        return df.rename(columns=rename_map)

    def load_bug_trajectory(self, db_video_id: int, video_path: str):
        if self.is_use_db:
            bug_trajs = []
            with self.orm.session() as s:
                vid = s.query(Video).filter_by(id=db_video_id).first()
                if vid.block_id is None:
                    raise MissingFile(f'unable to find block_id in DB for video_id: {db_video_id}')
                blk = s.query(Block).filter_by(id=vid.block_id).first()
                for tr in blk.trials:
                    if tr.bug_trajectory is not None:
                        bt = pd.DataFrame(tr.bug_trajectory)
                        bt['trial_id'] = tr.id
                        bug_trajs.append(bt)
            if bug_trajs:
                bug_trajs = pd.concat(bug_trajs)
            else:
                if video_path:
                    bug_trajs = self._load_bug_trajectory_from_file(video_path)
        else:
            assert video_path, 'must provide video_path for loading bug trajectory'
            bug_trajs = self._load_bug_trajectory_from_file(video_path)

        if bug_trajs is None or len(bug_trajs) == 0:
            return

        bug_trajs = self.rename_bug_columns(bug_trajs)
        bug_trajs['datetime'] = pd.to_datetime(bug_trajs['time']).dt.tz_localize(None)
        bug_trajs['timestamp'] = bug_trajs.datetime.astype(int).div(10**9)
        bug_trajs = bug_trajs.sort_values(by='datetime').reset_index(drop=True)
        return bug_trajs

    def _load_bug_trajectory_from_file(self, video_path):
        frames_output_dir = Path(video_path).parent.parent
        csv_path = frames_output_dir / 'bug_trajectory.csv'
        if not csv_path.exists():
            return None
        return pd.read_csv(csv_path, index_col=0)

    def add_bug_traj(self, pred_row, bug_traj, timestamp):
        dt = (bug_traj.timestamp - timestamp).abs()
        cols = [c if isinstance(c, str) else c[0] for c in bug_traj.columns]   # flatten any MultiIndex columns down to their string names
        bug_x_cols = [nm for nm in cols if re.match(r'bug(?:\d+)?_x$', nm)]    # find all “bug…_x” columns and extract their indices
        bug_indices = sorted(
            int(m.group(1)) if (m := re.match(r'bug(\d+)_x', nm)) and m.group(1).isdigit() else 0
            for nm in bug_x_cols
        )

        for i in bug_indices:
            single = (len(bug_indices) == 1 and i == 0)
            if single:
                x_name, y_name, trial_name = 'bug_x',   'bug_y',   'trial_id'
                xcm_name, ycm_name           = 'bug_x_cm','bug_y_cm'
            else:
                x_name, y_name               = f'bug{i}_x',   f'bug{i}_y'
                trial_name                   = 'trial_id'
                xcm_name, ycm_name           = f'bug{i}_x_cm', f'bug{i}_y_cm'

            for nm in (x_name, y_name, trial_name, xcm_name, ycm_name):
                pred_row[(nm, '')] = np.nan

            if dt.min() < 0.03:  # in case the diff is bigger than 30 msec, it means that this frame is not with bug.
                idx = dt.idxmin()

                for col in (x_name, y_name, trial_name):
                    if col in bug_traj.columns:
                        val = bug_traj.loc[idx, col]
                        pred_row[(col, '')] = val

                        if col == x_name and config.IS_SCREEN_CONFIGURED_FOR_POSE:
                            pred_row[(xcm_name, '')] = (
                                config.SCREEN_START_X_CM + val * config.SCREEN_PIX_CM
                            )
                            raw_y = bug_traj.loc[idx, y_name]
                            pred_row[(ycm_name, '')] = (
                                config.SCREEN_Y_CM       + raw_y * config.SCREEN_PIX_CM
                            )

        return pred_row

    def get_video_path(self, db_video_id: int) -> str:
        with self.orm.session() as s:
            vid = s.query(Video).filter_by(id=db_video_id).first()
            if vid is None:
                raise Exception(f'unable to find video_id: {db_video_id}')
            return vid.path

    def get_video_db_id(self, video_path: Path):
        video_path = Path(video_path)
        with self.orm.session() as s:
            vid = s.query(Video).filter(Video.path.contains(video_path.stem)).first()
            if vid is None:
                raise Exception(f'unable to find video in DB; video path: {video_path}')
            return vid.id

    def load_predicted_video(self, video_path):
        cache_path = self.get_predicted_cache_path(video_path)
        if not cache_path.exists():
            raise MissingFile(f'No prediction cache found under: {cache_path}')
        pose_df = pd.read_parquet(cache_path)
        return pose_df
    
    def save_predicted_video(self, pose_df: pd.DataFrame, video_path: str) -> None:
        """
        Save predicted pose data as a parquet file.

        Args:
            pose_df (pd.DataFrame): The predicted pose data.
            video_path (str): The path of the video.

        Returns:
            None

        """
        cache_path = self.get_predicted_cache_path(video_path)
        pose_df.to_parquet(cache_path)

    def write_to_example_video(self, frame, frame_id, pred_row, fps, video_path=None, example_path=None,
                               is_plot_preds=True):
        if self.example_writer is None:
            example_path = example_path or self.get_predicted_cache_path(video_path).with_suffix('.example.mp4').as_posix()
            # self.example_writer = OpenCVWriter(frame, fps, is_color=True, full_path=example_path)
            self.example_writer = ImageIOWriter(frame, fps, is_color=True, full_path=example_path)

        if is_plot_preds:
            frame = self.predictor.plot_predictions(frame, frame_id, pred_row)
        x, y = 40, 200
        for col in self.kinematic_cols:
            frame = put_text(f'{col}={pred_row[col].iloc[0]:.1f}', frame, x, y)
            y += 30
        if 'angle' in pred_row.columns:
            frame = put_text(f'angle={math.degrees(pred_row["angle"].iloc[0]):.1f}', frame, x, y)
        self.example_writer.write(frame)

    def close_example_writer(self):
        if self.example_writer is not None:
            self.example_writer.close()
            self.example_writer = None

    def get_predicted_cache_path(self, video_path) -> Path:
        preds_dir = Path(video_path).parent / 'predictions'
        preds_dir.mkdir(exist_ok=True)
        vid_name = Path(video_path).with_suffix('.parquet').name
        return preds_dir / f'{self.predictor.model_name}__{vid_name}'

    def is_moved(self, x, y):
        return not self.last_commit or distance.euclidean(self.last_commit[1:], (x, y)) < MOVE_MIN_DISTANCE

    def is_ready_to_commit(self, timestamp):
        return self.predictions and (timestamp - self.predictions[0][0]) > COMMIT_INTERVAL


class DLCArenaPose(ArenaPose):
    def __init__(self, cam_name, predictor_name='deeplabcut', is_use_db=True, orm=None, commit_bodypart='mid_ears', **kwargs):
        super().__init__(cam_name, predictor_name, is_use_db, orm, commit_bodypart=commit_bodypart, **kwargs)
        self.lizard_head_bodyparts = ['nose', 'right_ear', 'left_ear']
        self.is_lizard_head = all(bp in self.predictor.bodyparts for bp in self.lizard_head_bodyparts)
        self.pose_df = pd.DataFrame()
        self.angle_col = ('angle', '')

    def after_analysis_actions(self, pred_row):
        if self.is_lizard_head:
            angle = self.calc_head_angle(pred_row.iloc[0])
        else:
            angle = np.nan

        pred_row.loc[pred_row.index[0], self.angle_col] = angle
        return pred_row

    @staticmethod
    def calc_head_angle(row):
        x_nose, y_nose = row.nose.x, row.nose.y
        x_ears = (row.right_ear.x + row.left_ear.x) / 2
        y_ears = (row.right_ear.y + row.left_ear.y) / 2
        dy = y_ears - y_nose
        dx = x_ears - x_nose
        if dx != 0.0:
            theta = np.arctan(abs(dy) / abs(dx))
        else:
            theta = np.pi / 2
        if dx > 0:  # looking south
            theta = np.pi - theta
        if dy < 0:  # looking opposite the screen
            theta = -1 * theta
        if theta < 0:
            theta = 2 * np.pi + theta
        return theta

    def add_bug_traj(self, pred_row, bug_traj, timestamp):
        pred_row = super().add_bug_traj(pred_row, bug_traj, timestamp)
        p = pred_row.iloc[0] if isinstance(pred_row, pd.DataFrame) else pred_row
        
        # find all bug_x_cm columns ( ('bug_x_cm','') or ('bug1_x_cm',''))
        bug_x_cols = [col for col in pred_row.columns if re.match(r"bug\d*_x_cm", col[0])]
        
        for bug_x_col in bug_x_cols:
            m = re.search(r"bug(\d*)_x_cm", bug_x_col[0])   # extract the bug index (default to 0 if none)
            i = int(m.group(1)) if m and m.group(1).isdigit() else 0
            # 'dev_angle' if only one bug, otherwise 'dev_angle{i}'
            dev_name = ('dev_angle', '') if (i == 0 and len(bug_x_cols) == 1) else (f'dev_angle{i}', '')

            # compute deviation only if conditions are met
            if ( self.is_lizard_head
                and all(p[(bp, 'prob')] >= 0.8 for bp in self.lizard_head_bodyparts)
                and not np.isnan(p[bug_x_col])
                and not np.isnan(p[('nose','x')])
                and not np.isnan(p[('nose','y')])
                and config.SCREEN_Y_CM is not None
            ):                
                pred_row.loc[pred_row.index, dev_name] = self.calc_gaze_deviation_angle(
                    ang=p[('angle', '')],
                    bug_x=p[bug_x_col],
                    x=p[('nose', 'x')],
                    y=p[('nose', 'y')]
                )
            else:
                pred_row.loc[pred_row.index, dev_name] = np.nan

        return pred_row

    def calc_gaze_deviation_angle(self, ang: float, bug_x: float, x: float, y: float):
        """
        Calculates the gaze deviation angle between an animal and a moving object.

        Args:
            ang (float): The angle between the animal's nose and the object [radians].
            bug_x (float): The x-coordinate of the bug [cm].
            x (float): The x-coordinate of the animal's nose [cm].
            y (float): The y-coordinate of the animal's nose [cm].

        Returns:
            tuple[float, float]: A tuple containing the gaze deviation angle and the x-coordinate of the object.

        """
        m_exp = (y - config.SCREEN_Y_CM) / (x - bug_x)
        if ang == np.pi / 2:
            x_obs = x
            dev_ang = np.math.degrees(self.calc_angle_between_lines(1, 0, m_exp, -1))
        else:
            m_obs = np.tan(np.pi - ang)
            n_obs = y - m_obs * x
            x_obs = (config.SCREEN_Y_CM - n_obs) / m_obs
            a = distance.euclidean((bug_x, config.SCREEN_Y_CM), (x, y))
            b = distance.euclidean((x_obs, config.SCREEN_Y_CM), (x, y))
            c = np.abs(x_obs - bug_x)
            dev_ang = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
            if ang > np.pi:
                dev_ang = np.pi - dev_ang
            dev_ang = np.math.degrees(dev_ang)

        sgn = np.sign(x_obs - bug_x) if ang < np.pi else -np.sign(x_obs - bug_x)
        return sgn * dev_ang

    @staticmethod
    def calc_angle_between_lines(a1, b1, a2, b2):
        return np.arccos((a1*a2 + b1*b2) / (np.sqrt(a1**2 + b1**2) * np.sqrt(a2**2 + b2**2)))

    @property
    def body_parts(self):
        return [b for b in self.pose_df.columns.get_level_values(0).unique()
                if b and isinstance(self.pose_df[b], pd.DataFrame)]


class PogonaHeadPose(DLCArenaPose):
    def __init__(self, cam_name, predictor_name='pogona_head', is_use_db=True, orm=None, commit_bodypart='mid_ears',
                 **kwargs):
        super().__init__(cam_name, predictor_name, is_use_db, orm, commit_bodypart=commit_bodypart, **kwargs)


class SpatialAnalyzer:
    """
    A class for analyzing the spatial data of the animals in the arena.

    Args:
        animal_ids (list, optional): A list of animal IDs to analyze. If None, all animals will be analyzed.
        day (date, optional): The day of the experiment to analyze. If None, all experiments on the specified date range will be analyzed.
        start_date (date, optional): The start date of the experiment to analyze. If None, all experiments on the specified date range will be analyzed.
        cam_name (str, optional): The camera name to use for analysis. Defaults to 'front'.
        bodypart (str, optional): The body part to analyze. Defaults to 'mid_ears'.
        split_by (list, optional): A list of columns to split the data by. If None, no splitting will be performed. Defaults to None.
        orm (ORM, optional): The ORM object to use for database access. If None, a new ORM object will be created. Defaults to None.
        is_use_db (bool, optional): Whether to use the database for loading data. If False, the video paths will be used for loading data. Defaults to False.
        cache_dir (str, optional): The directory to use for caching data. If None, no caching will be performed. Defaults to None.
        arena_name (str, optional): The name of the arena to analyze. If None, all arenas will be analyzed. Defaults to None.
        excluded_animals (list, optional): A list of animal IDs to exclude from analysis. If None, no animals will be excluded. Defaults to None.
        **block_kwargs: Additional keyword arguments to filter the blocks to analyze.

    Attributes:
        animal_ids (list): A list of animal IDs to analyze.
        day (date): The day of the experiment to analyze.
        start_date (date): The start date of the experiment to analyze.
        cam_name (str): The camera name to use for analysis.
        bodypart (str): The body part to analyze.
        split_by (list): A list of columns to split the data by.
        orm (ORM): The ORM object to use for database access.
        is_use_db (bool): Whether to use the database for loading data.
        cache_dir (str): The directory to use for caching data.
        arena_name (str): The name of the arena to analyze.
        excluded_animals (list): A list of animal IDs to exclude from analysis.
        block_kwargs (dict): Additional keyword arguments to filter the blocks to analyze.
        pose_dict (dict): A dictionary of pandas dataframes, where the keys are the group names and the values are the pose data for the animals in the group.
    """
    splits_table = {
        'animal_id': 'experiment',
        'arena': 'experiment',
        'exit_hole': 'block',
        'bug_speed': 'block',
        'movement_type': 'block'
    }

    def __init__(self, animal_ids=None, day=None, start_date=None, cam_name='front', bodypart='mid_ears', split_by=None,
                 orm=None, is_use_db=False, cache_dir=None, arena_name=None, excluded_animals=None, max_y_arena=20, **block_kwargs):
        if animal_ids and not isinstance(animal_ids, list):
            animal_ids = [animal_ids]
        self.animal_ids = animal_ids
        self.day = day
        self.start_date = start_date
        self.cam_name = cam_name
        self.bodypart = bodypart
        self.arena_name = arena_name
        self.excluded_animals = excluded_animals or []
        assert split_by is None or isinstance(split_by, list), 'split_by must be a list of strings'
        self.split_by = split_by
        self.block_kwargs = block_kwargs
        self.is_use_db = is_use_db
        self.cache_dir = cache_dir
        self.orm = orm if orm is not None else ORM()
        self.dlc = DLCArenaPose('front', is_use_db=is_use_db, orm=self.orm)
        self.max_x_arena = 70
        self.max_y_arena = max_y_arena
        self.coords = {
            'arena': np.array([(0, 0), (self.max_x_arena, self.max_y_arena)]),
            'screen': np.array([(10, 0), (60, 1)])
        }
        self.pose_dict = self.get_pose()
        # fix for arenas
        for k, pf in self.pose_dict.items():
            # align msi-regev arean to reptilearn
            idx = pf.animal_id.isin(['PV80', 'PV42'])
            pf.loc[idx, 'x'] = pf.loc[idx, 'x'] * (self.max_x_arena/50)
            # move x-y coordinates in reptilearn arenas to start from 0
            pf['x'] = pf['x'] + 3
            # pf['y'] = pf['y'] + 2

    def get_pose(self) -> dict:
        """
        Load the pose data for the animals.

        Returns:
            dict: A dictionary of pandas dataframes, where the keys are the group names and the values are the pose data for the animals in the group.

        """
        cache_path = f'{self.cache_dir}/spatial_{"_".join(self.animal_ids) if self.animal_ids else "all"}.pkl'
        if self.cache_dir:
            cache_path = Path(cache_path)
            if cache_path.exists():
                with cache_path.open('rb') as f:
                    res = pickle.load(f)
                    return res

        res = {}
        for group_name, vids in self.get_videos_to_load().items():
            for video_path in vids:
                try:
                    pose_df = self._load_pose(video_path)
                    res.setdefault(group_name, []).append(pose_df)
                except MissingFile:
                    continue
                except Exception as exc:
                    ident = f'video DB ID: {video_path}' if self.is_use_db else f'video path: {video_path}'
                    print(f'Error loading {ident}; {exc}')

        for group_name in res.copy().keys():
            res[group_name] = pd.concat(res[group_name])

        # sort results by first split value
        if not self.split_by:
            pass
        elif len(self.split_by) == 2:
            res = dict(sorted(res.items(), key=lambda x: (x[0].split(',')[0].split('=')[1], x[0].split(',')[1].split('=')[1])))
        elif len(self.split_by) == 1:
            res = dict(sorted(res.items(), key=lambda x: x[0].split(',')[0].split('=')[1]))

        if self.cache_dir:
            with Path(cache_path).open('wb') as f:
                pickle.dump(res, f)
        return res

    def _load_pose(self, video_path):
        load_key = 'video_db_id' if self.is_use_db else 'video_path'
        pose_df = self.dlc.load(only_load=True, **{load_key: video_path})
        if pose_df is None:
            raise MissingFile('')
        if self.bodypart == 'mid_ears':
            for c in ['x', 'y']:
                pose_df[('mid_ears', c)] = pose_df[[('right_ear', c), ('left_ear', c)]].mean(axis=1)
            pose_df[('mid_ears', 'prob')] = pose_df[[('right_ear', 'prob'), ('left_ear', 'prob')]].min(axis=1)
        l = [
            pd.to_datetime(pose_df['time'], unit='s'),
            pose_df[self.bodypart]
        ]
        if self.is_use_db:
            l.extend([pose_df['block_id'], pose_df['animal_id']])
        return pd.concat(l, axis=1)

    def get_videos_to_load(self, is_add_block_video_id=False) -> dict:
        video_paths = {}
        with self.orm.session() as s:
            exps = s.query(Experiment).filter(Experiment.animal_id.not_in(['test', '']))
            if self.animal_ids:
                exps = exps.filter(Experiment.animal_id.in_(self.animal_ids))
            if self.excluded_animals:
                exps = exps.filter(Experiment.animal_id.not_in(self.excluded_animals))
            if self.arena_name:
                exps = exps.filter_by(arena=self.arena_name)
            if self.day:
                exps = exps.filter(cast(Experiment.start_time, Date) == self.day)
            elif self.start_date:
                exps = exps.filter(Experiment.start_time >= self.start_date)
            for exp in exps.all():
                for blk in exp.blocks:
                    if exp.animal_id == 'PV163' and self.block_kwargs.get(
                            'movement_type') == 'low_horizontal' and blk.movement_type == 'rect_tunnel':
                        blk.movement_type = 'low_horizontal'
                        blk.exit_hole = 'bottomRight'

                    if self.block_kwargs and any(getattr(blk, k) != v for k, v in self.block_kwargs.items()):
                        continue

                    group_name = self._get_split_values(exp, blk)
                    if self.split_by and not group_name:
                        continue
                    for vid in blk.videos:
                        if self.cam_name and vid.cam_name != self.cam_name:
                            continue

                        if not is_add_block_video_id:
                            v = vid.id if self.is_use_db else vid.path
                        else:
                            v = (vid.path, blk.id, vid.id)
                        video_paths.setdefault(group_name, []).append(v)

        return video_paths

    def _get_split_values(self, exp, blk):
        if not self.split_by:
            return ''

        s = []
        for c in self.split_by:
            assert c in self.splits_table, f'unknown split: {c}; possible splits: {",".join(self.splits_table.keys())}'
            if self.splits_table[c] == 'block':
                val = getattr(blk, c, None)
            elif self.splits_table[c] == 'experiment':
                val = getattr(exp, c, None)
            else:
                raise Exception(f'bad target for {c}: {self.splits_table[c]}')
            if val is None:
                continue
            s.append(f"{c}={val}")
        return ','.join(s)

    def get_out_of_experiment_pose(self):
        groups_pose = {}
        for group_name, vids in self.get_videos_to_load().items():
            day_paths = list(set([Path(*Path(v).parts[:-3]) for v in vids]))
            pose_ = []
            for day_p in day_paths:
                tracking_dir = day_p / 'tracking' / 'predictions'
                if not tracking_dir.exists():
                    continue

                for p in tracking_dir.rglob('*.csv'):
                    df = pd.read_csv(p, index_col=0)
                    if df.empty:
                        continue
                    df = df[~df.x.isna()]
                    pose_.extend(df[['x', 'y']].to_records(index=False).tolist())
            groups_pose[group_name] = pose_
        return groups_pose

    def plot_out_of_experiment_pose(self, axes=None):
        groups_pose = self.get_out_of_experiment_pose()
        axes = self.get_axes(4, len(groups_pose), axes, is_cbar=False)
        for i, (group_name, pose_list) in enumerate(groups_pose.items()):
            df = pd.DataFrame(pose_list, columns=['x', 'y'])
            sns.histplot(data=df, x='x', y='y', ax=axes[i], bins=(30, 30), cmap='Greens', stat='probability')
            # axes[i].set_xlim([0, 50])
        plt.show()

    def plot_spatial_hist(self, single_animal, pose_dict=None, cols=4, axes=None, is_title=True, animal_colors=None):
        if pose_dict is None:
            pose_dict = self.pose_dict

        axes_ = self.get_axes(cols, len(pose_dict), axes=axes)
        for i, (group_name, pose_df) in enumerate(pose_dict.items()):
            print(f'group_name: {group_name}')
            if not group_name:
                continue
            cbar_ax = None
            if i == len(pose_dict) - 1 and len(pose_dict) > 1:
                cbar_ax = inset_axes(
                    axes_[i], width="3%", height="70%", loc="lower left",
                    bbox_to_anchor=(1.1, 0., 3, 1),  # x_offset, y_offset, width, height in ax coords
                    bbox_transform=axes_[i].transAxes, borderpad=0
                )
            df_ = pose_df.query(f'0<=x<={self.max_x_arena} and 0<=y<={self.max_y_arena}')
            self.plot_hist2d(df_, axes_[i], single_animal, cbar_ax=cbar_ax)
            self.plot_arena(axes_[i])
            if len(self.split_by) == 1 and self.split_by[0] == 'exit_hole':
                group_name = r'Left $\rightarrow$ Right' if 'right' in group_name else r'Left $\leftarrow$ Right'
            if is_title:
                axes_[i].set_title(group_name)
        if axes is None:
            plt.tight_layout()
            plt.show()

    def plot_hist2d(self, df, ax, single_animal, cbar_ax=None):
        df_ = df.query(f'animal_id == "{single_animal}"')
        sns.histplot(data=df_, x='x', y='y', ax=ax,
                     bins=(np.arange(0, self.max_x_arena+2, 2), np.arange(0, self.max_y_arena+1, 1)), cmap='Greens', stat='probability',
                     cbar=cbar_ax is not None, cbar_kws=dict(shrink=.75, label='', orientation='vertical', ticks=[0, 0.04]),
                     cbar_ax=cbar_ax)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(None)
        ax.set_xlabel(None)
        # scalebar
        fontprops = fm.FontProperties(size=14)
        scalebar = AnchoredSizeBar(ax.transData, 10, '10cm', 'lower right', pad=1, color='black', frameon=False,
                                   size_vertical=0.7, fontproperties=fontprops)
        ax.add_artist(scalebar)
        # upper 1D histogram for x values
        hist_x_ax = ax.inset_axes([0, 1, 1, 0.3])
        sns.histplot(data=df_, x='x', ax=hist_x_ax, bins=30)
        hist_x_ax.axis('off')

    def plot_arena(self, ax):
        # plot screen
        c = self.coords['screen']
        rect = patches.Rectangle(c[0, :], *(c[1, :] - c[0, :]).tolist(), linewidth=1, edgecolor='k',
                                 facecolor='k')
        ax.add_patch(rect)
        # set arena bounds
        ax.set_xlim(self.coords['arena'][:, 0])
        ax.set_ylim(self.coords['arena'][:, 1])
        ax.invert_yaxis()

    def plot_spatial_x_kde(self, axes=None, cols=4, animal_colors=None, pose_dict=None):
        if pose_dict is None:
            pose_dict = self.pose_dict

        axes_ = self.get_axes(cols, len(pose_dict), axes=axes)
        for i, (group_name, pose_df) in enumerate(pose_dict.items()):
            df = pose_df.query(f'0 <= x <= {self.max_x_arena} and y<20')
            sns.violinplot(data=df, x='x', y='animal_id', hue='animal_id', ax=axes_[i], palette=animal_colors, order=list(animal_colors.keys()), linewidth=1)
            axes_[i].set_xlim([0, self.max_x_arena])
            axes_[i].axvline(self.max_x_arena/2, linestyle='--', color='k')
            axes_[i].set_yticks([])
            axes_[i].set_ylabel('Animals')

    def plot_trajectories(self, cols=2, axes=None, only_to_screen=False, is_title=True, animal_colors=None):
        axes_ = self.get_axes(cols, len(self.pose_dict), axes, is_cbar=False)

        for i, (group_name, pose_df) in enumerate(self.pose_dict.items()):
            trajs = self.cluster_trajectories(pose_df, only_to_screen=only_to_screen, cross_y_val=10)
            if is_title:
                axes_[i].set_title(group_name.replace(',', '\n'))
            if not trajs:
                continue

            # x_values = {}
            traj_df = []
            for (block_id, frame_id, animal_id), traj in trajs.items():
                traj_df.append({'animal_id': animal_id, 'x': traj.x.values[-1]})
                # x_values.setdefault(animal_id, []).append(traj.x.values[-1])
            traj_df = pd.DataFrame(traj_df)

            inner_ax = axes_[i]
            sns.boxplot(data=traj_df, x='x', y='animal_id', hue='animal_id', ax=inner_ax, palette=animal_colors, order=list(animal_colors.keys()))
            # for animal_id, x_ in x_values.items():
                # color_kwargs = {'color': animal_colors[animal_id] if animal_colors else None}
                # sns.kdeplot(x=x_, ax=inner_ax, label=animal_id, **color_kwargs) # clip=[0, 40], bw_adjust=.4,
            inner_ax.axvline(self.max_x_arena/2, linestyle='--', color='k')
            inner_ax.set_xticks([0, 20, 40, 60])
            inner_ax.set_xlim([0, self.max_x_arena])
            inner_ax.set_yticks([])
            # inner_ax.set_ylabel('Probability')
            # inner_ax.set_ylim([0, 0.12])
            # inner_ax.legend()

        if axes is None:
            plt.tight_layout()
            plt.show()

    def play_trajectories(self, video_path: str, only_to_screen=False):
        pose_df = self._load_pose(video_path)
        cap = cv2.VideoCapture(video_path)
        trajs = self.cluster_trajectories(pose_df, only_to_screen=only_to_screen)

        for start_frame, traj in trajs.items():
            total_distance = self.calc_traj_distance(traj)
            self.play_segment(cap, start_frame, len(traj), f'Traj{start_frame} ({total_distance:.1f})')

        cap.release()

    @staticmethod
    def play_segment(cap, start_frame, n_frames, frames_text=None):
        assert frames_text is None or isinstance(frames_text, str) or len(frames_text) == n_frames, \
            'frames_text must be a string or a list in the length of n_frames'
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(n_frames):
            ret, frame = cap.read()
            frame = cv2.resize(frame, None, None, fx=0.5, fy=0.5)
            if frames_text:
                put_text(frames_text[i] if isinstance(frames_text, list) else frames_text, frame, 30, 20)
            cv2.imshow('Pogona Pose', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    @staticmethod
    def get_cross_limit(cross_id: int, s: pd.Series, grace=10, dist_threshold=0.01):
        assert s.index[0] == cross_id or s.index[-1] == cross_id, 'cross_id must be the last or first element of the series'
        if len(s) < 2:
            return cross_id
        if s.index[-1] == cross_id:
            s = s.copy().iloc[::-1]
        i = 1  # start after cross_id
        grace_count = 0
        while grace_count <= grace and i < len(s)-1:
            if s.iloc[i] < dist_threshold:
                grace_count += 1
            else:
                grace_count = 0
            i += 1
        return s.index[i]

    def cluster_trajectories(self, pose: pd.DataFrame, cross_y_val=20, frames_around_cross=300, window_length=31,
                             min_traj_len=2, only_to_screen=False, is_plot=False):
        """
        This function clusters animal trajectories based on the spatial proximity of crossing points.

        Args:
            pose (pd.DataFrame): A pandas dataframe containing the animal poses.
            cross_y_val (float, optional): The y-value at which to consider a crossing point. Defaults to 20.
            frames_around_cross (int, optional): The number of frames to consider around a crossing point. Defaults to 300.
            window_length (int, optional): The window length for the Savitzky-Golay filter. Defaults to 31.
            min_traj_len (int, optional): The minimum trajectory length. Defaults to 2.
            only_to_screen (bool, optional): Whether to only consider trajectories that end up on the screen. Defaults to False.
            is_plot (bool, optional): Whether to plot the trajectories. Defaults to False.

        Returns:
            dict: A dictionary of trajectories, where the key is a tuple of (block_id, frame_id, animal_id) and the value is a pandas dataframe containing the trajectory.

        """
        trajs = {}
        dist_df = pose[['time', 'x', 'y', 'prob', 'block_id', 'animal_id', 'bug_x'
                        ]].reset_index().copy().rename(columns={'index': 'frame_id'})
        dist_df = dist_df.drop(index=dist_df[(dist_df.prob < 0.5) | (dist_df.x.isna())].index, errors='ignore')
        if len(dist_df) < window_length:
            return trajs
        dist_df['time_diff'] = dist_df.time.diff().dt.total_seconds()
        dists = dist_df.y.diff().abs()
        dist_df['distance'] = savgol_filter(dists.values, window_length=window_length, polyorder=0, mode='interp')
        # g = dist_df.groupby(dist_df['time_diff'].ge(1).cumsum())
        blocks_groups = dist_df.groupby('block_id')

        if is_plot:
            cols = 5
            rows = int(np.ceil(len(blocks_groups)/cols))
            fig, axes = plt.subplots(rows, cols, figsize=(25, 3*rows))
            axes = axes.flatten()
        for i, (block_id, xf) in enumerate(blocks_groups):
            xf = xf.query('y>-1')
            df = xf.query(f'{cross_y_val - 1}<=y<={cross_y_val + 1}')
            g2 = df.groupby(df.index.to_series().diff().ge(60).cumsum())
            crosses = []
            if is_plot:
                axes[i].plot(xf.y, color='k')
                axes[i].set_ylim([-1, 85])
            for n, gf in g2:
                cross_id = (gf.y - cross_y_val).abs().idxmin()
                lower_lim = self.get_cross_limit(cross_id, xf.loc[cross_id-frames_around_cross:cross_id, 'distance'])
                upper_lim = self.get_cross_limit(cross_id, xf.loc[cross_id:cross_id+frames_around_cross, 'distance'])
                dy_traj = np.abs(xf.loc[upper_lim, 'y'] - xf.loc[lower_lim, 'y'])
                if dy_traj < min_traj_len or (only_to_screen and xf.loc[upper_lim, 'y'] > 8):
                    continue
                crosses.append(cross_id)
                frame_id = xf.loc[lower_lim, 'frame_id']
                animal_id = gf.animal_id.unique()[0]
                trajs[(block_id, frame_id, animal_id)] = xf.loc[lower_lim:upper_lim, ['x', 'y', 'bug_x', 'time']].copy()
                if is_plot:
                    axes[i].plot(xf.y.loc[lower_lim:upper_lim])
            if is_plot:
                axes[i].set_title(f'# crosses: {len(crosses)}')

        if is_plot:
            fig.tight_layout()

        return trajs

    def find_crosses(self, video_path=None, y_value=10, is_play=True, axes=None, cols=3):
        if video_path is not None:
            pose_dict = {'video_crosses': self._load_pose(video_path)}
        else:
            pose_dict = self.get_pose()

        axes_ = self.get_axes(cols, len(pose_dict), axes)
        x_values = {}
        for i, (group_name, pose_df) in enumerate(pose_dict.items()):
            df_ = pose_df.query(f'{y_value-0.1} <= y <= {y_value+0.1} and 0 <= x <= 40').copy()
            m = df_.index.to_series().diff().ne(1).cumsum()
            idx_ = df_.index.to_series().groupby(m).agg(list)
            idx2drop = flatten([idx_[j][1:] for j in idx_[idx_.map(lambda x: len(x)) > 1].index])
            df_.drop(index=idx2drop, inplace=True)

            if is_play and video_path is not None:
                cap = cv2.VideoCapture(video_path)
                for cross_id in df_.index:
                    self.play_segment(cap, cross_id-60, 120, f'cross index {cross_id}')
                cap.release()

            x_values[group_name] = df_.x.values
            sns.kdeplot(x=df_.x.values, ax=axes_[i])
            # axes_[i].hist(df_.x.values, label=f'mean = {df_.x.values.mean():.2f}')
            axes_[i].set_title(group_name)
            axes_[i].legend()

        if len(x_values) == 2:
            groups = list(x_values.values())
            t_stat, p_value = ttest_ind(groups[0], groups[1], equal_var=False)
            p_text = f'p-value={p_value:.3f}' if p_value >= 0.001 else 'p-value<0.001'
            plt.suptitle(f'T={t_stat:.1f}, {p_text}')

        if axes is None:
            plt.tight_layout()
            plt.show()

    @staticmethod
    def calc_traj_distance(traj):
        try:
            traj = np.array(traj)
            traj_no_nan = traj[~np.isnan(traj).any(axis=2), :]
            return distance.euclidean(traj_no_nan[0, :], traj_no_nan[-1, :])

        except:
            return 0
        # return np.sum(np.sqrt(np.sum(np.diff(traj_no_nan, axis=0) ** 2, axis=1)))

    @staticmethod
    def get_axes(cols, n, axes=None, is_cbar=True):
        cols = min(cols, n)
        rows = int(np.ceil(n / cols))
        if axes is None:
            width_ratios = [15 for _ in range(cols)]
            if is_cbar:
                width_ratios += [1]
            fig, axes = plt.subplots(rows, cols+(1 if is_cbar else 0), figsize=(cols * 4, rows * 3),
                                     gridspec_kw={'width_ratios': width_ratios})

        if n > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        return axes


########################################################################################################################


def get_bug_exit_hole(day) -> list:
    orm = ORM()
    with orm.session() as s:
        q = s.query(Block).filter(cast(Block.start_time, Date) == day)
        res = q.all()
    return list(set([r.exit_hole for r in res if r.exit_hole]))


def get_day_from_path(p):
    return p.stem.split('_')[1].split('T')[0]


def get_screen_coords(name):
    s = yaml.load(Path('/analysis/strikes/screen_coords.yaml').open(), Loader=yaml.FullLoader)
    cnt = s['screens'][name]
    return np.array(cnt)


def load_pose_from_videos(animal_id, cam_name):
    orm = ORM()
    dp = ArenaPose(cam_name, 'deeplabcut', is_use_db=True)
    with orm.session() as s:
        for exp in s.query(Experiment).filter_by(animal_id=animal_id).all():
            for blk in exp.blocks:
                for vid in blk.videos:
                    if vid.cam_name != cam_name:
                        continue
                    try:
                        dp.predict_pred_cache(video_path=vid.path)
                    except Exception as exc:
                        print(f'ERROR; {vid.path}; {exc}')


def compare_sides(animal_id='PV80'):
    with ORM().session() as s:
        exps = s.query(Experiment).filter_by(animal_id=animal_id).all()
        days = list(set([e.start_time.strftime('%Y-%m-%d') for e in exps]))

    res = {}
    for day in days.copy():
        exit_holes = [x for x in get_bug_exit_hole(day) if x]
        if len(exit_holes) == 1:
            exit_hole = exit_holes[0]
        else:
            days.remove(day)
            continue
        ps = SpatialAnalyzer(animal_id, day=day).get_pose()
        res.setdefault(exit_hole, []).append(ps)

    res = {k: pd.concat(v).query('y<20') for k, v in res.items()}
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sa = SpatialAnalyzer(animal_id)
    for i, (k, v) in enumerate(res.items()):
        sa.plot_spatial_hist(v, axes[i])
        axes[i].set_title(k)
    fig.tight_layout()
    plt.show()


def run_predict_process(pred_name, images, cam_name, image_date, res_dict, is_base64):
    import importlib
    import base64
    import cv2
    from PIL import Image
    from io import BytesIO
    import numpy as np

    pred_config = config.load_configuration('predict')
    assert pred_name in pred_config, f'Unknown predictor: {pred_name}'
    prd_class = pred_config[pred_name]['predictor_name']

    ap_class = DLCArenaPose if prd_class == 'DLCPose' else ArenaPose
    ap = ap_class(cam_name, pred_name)
    l = []
    for img in images:
        if is_base64:
            imgdata = img.split(',')[1]
            img = base64.b64decode(imgdata)
            img = np.array(Image.open(BytesIO(img)))
        
        img, pdf = ap.predict_frame(img, image_date, is_plot_preds=True)
        l.append((img, pdf))
    res_dict['images'] = l


def run_predict(pred_name, images, cam_name='', image_date=None, is_base64=False):
    import torch.multiprocessing as mp

    manager = mp.Manager()
    return_dict = manager.dict()
    p = mp.Process(target=run_predict_process, args=(pred_name, images, cam_name, image_date, return_dict, is_base64))
    p.start()
    p.join()
    return return_dict


class VideoPoseScanner:
    def __init__(self, cam_name=config.NIGHT_POSE_CAMERA, is_use_db=True, model_path=None, animal_ids=None, is_replace_exp_dir=True,
                 only_strikes_vids=False):
        self.cam_name = cam_name
        self.is_use_db = is_use_db
        self.model_path = model_path
        if animal_ids and isinstance(animal_ids, str):
            animal_ids = [animal_ids]
        self.animal_ids = animal_ids
        self.is_replace_exp_dir = is_replace_exp_dir
        self.only_strikes_vids = only_strikes_vids
        self.logger = get_logger('Video-Pose-Scanner')
        self.orm = ORM() if is_use_db else None
        self.dlc = DLCArenaPose(cam_name, is_use_db=is_use_db, model_path=model_path, orm=self.orm, commit_bodypart=None)

    def predict_all(self, is_tqdm=True, max_videos=None, errors_cache=None, skip_committed=True):
        videos = self.get_videos_to_predict(skip_committed)
        if not videos:
            return
        self.logger.info(f'found {len(videos)} to predict for pose.')
        success_count = 0
        for i, video_path in enumerate(videos):
            try:
                if self.dlc.get_predicted_cache_path(video_path).exists():
                    continue
                pose_df = self.predict_video(video_path, prefix=f'({i+1}/{len(videos)}) ', is_tqdm=is_tqdm)
                success_count += 1

                if self.is_use_db: # commit the video predictions to the database
                    self.commit_video_prediction(video_path, pose_df)
                if max_videos and success_count >= max_videos:
                    return
            except MissingFile as exc:
                self.print_cache(exc, errors_cache)
            except Exception as exc:
                self.print_cache(exc, errors_cache)
            finally:
                torch.cuda.empty_cache()

    def predict_video(self, video_path, prefix='', is_tqdm=True):
        pose_df = self.dlc.predict_video(video_path=video_path, is_create_example_video=False, 
                                         prefix=prefix, is_tqdm=is_tqdm)
        return pose_df
        
    
    def commit_video_prediction(self, video_path, pose_df):
        pose_df = pose_df.dropna(subset=[('nose', 'x')])
        animal_id_ = Path(video_path).parts[-5]
        video_id, _ = self.dlc.check_video_inputs(None, video_path)
        start_time = datetime.datetime.fromtimestamp(pose_df.iloc[0][('time', '')])
        self.orm.commit_video_predictions(self.dlc.predictor.model_name, pose_df, video_id, start_time, animal_id_)

    def get_videos_to_predict(self, skip_committed=True):
        if self.is_use_db:
            videos = self._get_videos_to_predict_from_db(skip_committed)
        else:
            videos = self._get_videos_to_predict_from_files(is_skip_predicted=skip_committed)
        videos = sorted(videos, key=lambda x: x.name, reverse=True)
        return videos

    def _get_videos_to_predict_from_db(self, skip_committed=True):
        """Get from the DB all the videos that have not been predicted yet"""
        # self.logger.info('Start scan of video predictions in the database')
        with self.orm.session() as s:
            pred_kwargs = {'predictions': None} if skip_committed else {}
            vids = s.query(Video).filter_by(cam_name=self.cam_name, **pred_kwargs).all()
            videos = []
            for vid in vids:
                if not vid.path.endswith('.mp4') or not vid.animal_id.startswith('PV') \
                    or vid.animal_id in ['test'] \
                    or (self.animal_ids and vid.animal_id not in self.animal_ids):
                    continue
                
                blk = s.query(Block).filter_by(id=vid.block_id).first()
                if config.NIGHT_POSE_RUN_ONLY_BUG_SESSIONS:
                    if blk is None or blk.block_type != 'bugs':
                        continue

                if blk is None or (self.only_strikes_vids and not blk.strikes):
                    continue

                video_path = Path(vid.path)
                if self.is_replace_exp_dir: # replace the experiment directory with the experiment directory of the video
                    vid_exp_dir = os.path.join(*video_path.parts[:-5])
                    if vid_exp_dir != config.EXPERIMENTS_DIR:
                        video_path = Path(video_path.as_posix().replace(vid_exp_dir, config.EXPERIMENTS_DIR))
                
                if video_path.exists():
                    if skip_committed and self.dlc.get_predicted_cache_path(video_path).exists():
                        # commit if the prediction file exists, but not committed yet
                        self.logger.info(f'{video_path.stem},{vid.animal_id} has already been predicted, but was not written to the database. Committing now.')
                        self.dlc.is_use_db = False
                        try:
                            pose_df = self.dlc.load(video_path=video_path, only_load=True)
                        except Exception as exc:
                            self.logger.error(f'Unable to load video prediction for {video_path}; {exc}')
                            self.dlc.get_predicted_cache_path(video_path).unlink()
                        else:
                            self.dlc.is_use_db = self.is_use_db
                            self.commit_video_prediction(video_path, pose_df)
                            continue

                    videos.append(video_path)

        return videos
    
    def _get_videos_to_predict_from_files(self, experiments_dir=None, is_skip_predicted=True):
        experiments_dir = experiments_dir or config.EXPERIMENTS_DIR
        exp_path = Path(experiments_dir)
        self.logger.info(f'Start scan of video files in {exp_path}')
        if self.animal_ids:
            all_videos = []
            for animal_id in self.animal_ids:
                p_ = exp_path / animal_id
                all_videos.extend(list(p_.rglob(f'*{self.cam_name}*.mp4')))
        else:
            all_videos = exp_path.rglob(f'*{self.cam_name}*.mp4')

        videos = []
        for vid_path in all_videos:
            pred_path = self.dlc.get_predicted_cache_path(vid_path)
            if (is_skip_predicted and (pred_path.exists() or pred_path.with_suffix('.txt').exists())) or \
                    (len(pred_path.parts) >= 6 and pred_path.parts[-6] == 'test'):
                continue
            # skip blocks without strikes if only_strikes_vids=True
            if self.only_strikes_vids and not (Path(vid_path).parent.parent / config.experiment_metrics['touch']['csv_file']).exists():
                continue
            videos.append(vid_path)
        return videos

    def scan_video_predictions(self):
        """search for uncommitted video predictions in the database"""
        assert self.is_use_db
        with self.orm.session() as s:
            vids = s.query(Video).filter_by(cam_name=self.cam_name, predictions=None).all()
            for vid in vids:
                if not vid.path.endswith('.mp4') or not vid.animal_id.startswith('PV') \
                    or (self.animal_ids and vid.animal_id not in self.animal_ids):
                    continue
                    
                pose_df = self.dlc.load(vid.path)
                self.commit_video_prediction()

    def add_bug_trajectory(self, videos=None):
        if videos is None:
            videos = self.get_videos_to_predict(skip_committed=False)
        if not videos:
            self.logger.info('No videos found; aborting')
            return
        self.logger.info(f'found {len(videos)} to add bug traj')
        for i, video_path in enumerate(videos):
            new_df = []
            try:
                if self.dlc.get_predicted_cache_path(video_path).exists():
                    animal_id = Path(video_path).parts[-5]
                    self.dlc.is_use_db = False
                    pose_df = self.dlc.load(video_path=video_path, only_load=True)
                    if ('bug_x_cm', '') in pose_df.columns:
                        self.dlc.is_use_db = self.is_use_db
                        continue
                    bug_traj = self.dlc.load_bug_trajectory(None, video_path)
                    self.dlc.is_use_db = self.is_use_db
                    for i, row in tqdm(pose_df.iterrows(), desc=f'({i+1}/{len(videos)}) {animal_id} {video_path.stem}', total=len(pose_df)):
                        new_df.append(self.dlc.add_bug_traj(row, bug_traj, row[('time', '')]))
                    new_df = pd.DataFrame(new_df)
                    self.dlc.save_predicted_video(new_df, video_path)
                    self.orm.update_video_prediction(video_path.stem, self.dlc.predictor.model_name, new_df.dropna(subset=[('nose', 'x')]))
            except Exception as exc:
                self.logger.error(f'{video_path}, {exc}')

    def fix_calibrations(self, experiments_dir=None, calibration_dir=None):
        """redo calibration for all video predictions in the database and files"""
        # assert self.is_use_db, 'must set is_use_db to True'
        if calibration_dir is not None:
            config.CALIBRATION_DIR = calibration_dir
        videos = self._get_videos_to_predict_from_files(experiments_dir=experiments_dir, is_skip_predicted=False)
        if not videos:
            self.logger.info('No videos found; aborting')
            return
        self.logger.info(f'found {len(videos)}/{len(videos)} to fix calibration')
        is_initialized = False
        for i, video_path in enumerate(videos):
            self.dlc.start_new_session(60)
            if not is_initialized:
                self.dlc.init_from_video(video_path, caliber_only=True)
                is_initialized = True
            try:
                self.dlc.caliber.set_image_date_and_load(video_path.stem.split('_')[1])
                cache_path = self.dlc.get_predicted_cache_path(video_path)
                zf = pd.read_parquet(cache_path)
                for i in tqdm(zf.index, desc=f'({i+1}/{len(videos)}) {video_path.stem}'):
                    row = zf.loc[i:i].copy()
                    new_row = self.dlc.analyze_frame(row['time'].iloc[0], row.copy())
                    zf.loc[i] = new_row.iloc[0]
                self.dlc.save_predicted_video(zf, video_path)
                if self.is_use_db:
                    self.orm.update_video_prediction(video_path.stem, self.dlc.predictor.model_name, zf.dropna(subset=[('nose', 'x')]))
            except MissingFile as exc:
                self.logger.error(f'Missing File Error; {exc}')
            except Exception:
                self.logger.error(f'\n\n{traceback.format_exc()}\n')

    def print_cache(self, exc, errors_cache):
        if errors_cache is None or str(exc) not in errors_cache:
            self.logger.error(exc)
        if errors_cache and str(exc) not in errors_cache:
            errors_cache.append(str(exc))


if __name__ == '__main__':
    PogonaHeadPose('top').predict_video(video_path='/data/PreyTouch/output/experiments/PV51/20250130/block2/videos/top_20250130T163032.mp4')
    # VideoPoseScanner(only_strikes_vids=True).predict_all(is_tqdm=True)
    # VideoPoseScanner(animal_id='PV163').add_bug_trajectory(videos=[Path('/media/reptilearn4/experiments/PV163/20240201/block10/videos/front_20240201T173016.mp4')])
    # img = cv2.imread('/data/Pogona_Pursuit/output/calibrations/Archive/front/20221205T093815_front.png', 0)
    # print(run_predict('pogona_head', [img]))
    # DLCArenaPose('front', is_use_db=True).predict_frame(img)
    # plt.imshow(img)
    # plt.show()
    # sa = SpatialAnalyzer(animal_ids=None, movement_type='low_horizontal', start_date='2023-04-18',
    #                      split_by=['exit_hole'], bodypart='nose', is_use_db=True, excluded_animals=['PV85'])
    # load_pose_from_videos('PV80', 'front', is_exit_agg=True) #, day='20221211')h
    # SpatialAnalyzer(animal_ids=['PV91'], bodypart='nose', is_use_db=True).plot_spatial_hist('PV91')
    # SpatialAnalyzer(movement_type='low_horizontal', split_by=['exit_hole'], bodypart='nose').find_crosses(y_value=5)
    # SpatialAnalyzer(animal_ids=['PV42', 'PV91'], movement_type='low_horizontal',
    #                 split_by=['animal_id', 'exit_hole'], bodypart='nose',
    #                 is_use_db=True).plot_trajectories(only_to_screen=True)
    # SpatialAnalyzer(animal_ids=['PV91'], split_by=['exit_hole'], bodypart='nose').plot_out_of_experiment_pose()
    # fix_calibrations('PV95')
    # for vid in sa.get_videos_paths()['exit_hole=left']:
    #     sa.play_trajectories(vid, only_to_screen=True)
    # compare_sides(animal_id='PV80')

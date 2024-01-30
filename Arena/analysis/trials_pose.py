import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from scipy.spatial import distance
from scipy.signal import savgol_filter
from scipy.stats import ttest_ind
from multiprocessing.pool import ThreadPool
import os
if Path('.').resolve().name != 'Arena':
    os.chdir('..')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import config
from calibration import CharucoEstimator
from loggers import get_logger
from utils import run_in_thread, Kalman
from sqlalchemy import cast, Date
from db_models import ORM, Experiment, Block, Video, VideoPrediction, Strike, PoseEstimation, Trial
from image_handlers.video_writers import OpenCVWriter, ImageIOWriter
from analysis.pose_utils import put_text, flatten
from analysis.pose import DLCArenaPose


class TrialPose:
    def __init__(self, trial_id, orm=None, is_dwh=False):
        self.trial_id = trial_id
        self.is_dwh = is_dwh
        self.orm = orm if orm is not None else ORM()
        self.animal_id = None  # extracted from DB
    
    def play_trial(self, cam_name='back', is_save_video=False):
        ap = DLCArenaPose(cam_name, is_use_db=True, orm=self.orm, commit_bodypart=None)
        df = self.load_from_db(ap, cam_name)
        for video_path in df['video_path'].unique():
            pf_ = df[df['video_path'] == video_path].copy()
            video_path = f'{config.EXPERIMENTS_DIR}/{video_path}'
            if not Path(video_path).exists():
                print(f'{video_path} does not exist')
                continue

            cap = cv2.VideoCapture(video_path)
            start_frame, end_frame = pf_.index[0], pf_.index[-1]
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            fps = cap.get(cv2.CAP_PROP_FPS)
            for i in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                put_text(f'Angle={np.math.degrees(pf_.loc[i].angle):.0f}', frame, frame.shape[1]-250, 30)
                frame = self.plot_on_frame(frame, i, pf_)
                if is_save_video:
                    example_path = f'{config.OUTPUT_DIR}/trial_videos/{self.animal_id}_trial_{self.trial_id}_{cam_name}.mp4'
                    Path(example_path).parent.mkdir(parents=True, exist_ok=True)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ap.write_to_example_video(frame, i, pd.DataFrame(pf_.loc[i]).transpose(), fps, example_path=example_path, is_plot_preds=False)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    frame = cv2.resize(frame, None, None, fx=0.5, fy=0.5)
                    cv2.imshow(f'Trial {self.trial_id}', frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
            ap.close_example_writer()
            cv2.destroyAllWindows()
            cap.release()

    def load_from_db(self, ap: DLCArenaPose, cam_name: str):
        frames_df, pose_df = [], []
        with self.orm.session() as s:
            tr_filter = {'id' if not self.is_dwh else 'dwh_key': self.trial_id}
            tr = s.query(Trial).filter_by(**tr_filter).first()
            blk = s.query(Block).filter_by(id=tr.block_id).first()
            exp = s.query(Experiment).filter_by(id=blk.experiment_id).first()
            self.animal_id = exp.animal_id
            for vid in blk.videos:
                if vid.cam_name == cam_name:
                    frames_ = pd.DataFrame(pd.Series(vid.frames, name='time'))
                    frames_['video_path'] = '/'.join(Path(vid.path).parts[-5:])
                    frames_df.append(frames_)
                try:
                    pdf_ = ap.load(video_db_id=vid.id)
                    pose_df.append(pdf_)
                except Exception:
                    pass
            strikes_df = self.load_strikes(tr)
        if not frames_df:
            raise Exception('No frames data was loaded')
        elif not pose_df:
            raise Exception('No pose data was loaded')
        elif tr.bug_trajectory is None:
            raise Exception(f'Trial {self.trial_id} has no bug trajectory')
        else:
            print(f'Found {len(frames_df)} pose videos data')
        bug_trajs = pd.DataFrame(tr.bug_trajectory)
        pose_df = self.align_pose(pose_df, frames_df)
        df = self.merge_all(pose_df, bug_trajs, strikes_df)
        return df

    def load_strikes(self, tr):
        strikes_list = []
        for strk in tr.strikes:
            strikes_list.append({'strike_id': strk.id, 'time': strk.time, 'is_hit': strk.is_hit, 'strike_x': strk.x, 'strike_y': strk.y, 
                               'bug_strike_x': strk.bug_x, 'bug_strike_y': strk.bug_y})
        return strikes_list

    def align_pose(self, pose_df, frames_df):
        pose_df = pd.concat(pose_df)
        frames_df = pd.concat(frames_df)
        pose_df.columns = ['_'.join(c) if c[1] else c[0] for c in pose_df.columns]
        pose_df['time'] = pd.to_datetime(pose_df['time'], unit='s')
        frames_df['time'] = pd.to_datetime(frames_df.time, unit='s', utc=True).dt.tz_convert(
                'Asia/Jerusalem').dt.tz_localize(None)

        pose_df = pd.merge_asof(left=frames_df, right=pose_df, left_on='time', right_on='time', 
                                direction='nearest', tolerance=pd.Timedelta('100 ms'))
        return pose_df

    def merge_all(self, pose_df, bug_trajs, strikes_list):
        bug_trajs = bug_trajs.rename(columns={'x': 'bug_x', 'y': 'bug_y'})
        bug_trajs['time'] = pd.to_datetime(bug_trajs['time']).dt.tz_localize(None)
        bug_trajs = bug_trajs.sort_values(by='time')

        df = pd.merge_asof(left=pose_df, right=bug_trajs, left_on='time', right_on='time', 
                                direction='nearest', tolerance=pd.Timedelta('100 ms'))
        df = df.dropna(subset='bug_x')

        df['strikes'] = None
        for d in strikes_list:
            frame_id = (df.time - d['time']).dt.total_seconds().abs().idxmin()
            d['time'] = d['time'].isoformat()
            df.loc[frame_id, 'strikes'] = json.dumps(d)

        return df
    
    def plot_on_frame(self, frame, frame_id, df):
        df_ = df.loc[:frame_id]
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        axes[0].plot(df_.nose_y, linewidth=4)
        axes[0].set_xlim([df.index[0], df.index[-1]])
        axes[0].set_ylim([-3, 20])
        axes[0].set_title('Y-Nose', fontsize=16)

        axes[1].plot(df_.bug_x, df_.bug_y, linewidth=4)
        for i, row in df_[~df_.strikes.isna()].iterrows():
            d = json.loads(row.strikes)
            axes[1].scatter(d['strike_x'], d['strike_y'], marker='*', color='r', s=90)
        axes[1].set_ylim([0, 1080])
        axes[1].set_xlim([0, 1920])
        axes[1].invert_yaxis()
        axes[1].set_title('Angle', fontsize=16)
        fig.tight_layout()

        frame = self.insert_figure_to_frame(fig, frame, fig_size=(400, 200), top_left=(1000, 100))
        plt.close(fig)
        return frame


    @staticmethod
    def insert_figure_to_frame(fig, frame, fig_size=None, top_left=None):
        fig.canvas.draw()
        fig_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        fig_img  = fig_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        fig_img = cv2.cvtColor(fig_img, cv2.COLOR_RGB2BGR)
        if fig_size:
            fig_img = cv2.resize(fig_img, dsize=fig_size, interpolation=cv2.INTER_CUBIC)
        else:
            fig_size = fig_img.shape[:2]
        if top_left:
            frame[top_left[1]:top_left[1]+fig_size[1], top_left[0]:top_left[0]+fig_size[0], :] = fig_img
        else:
            frame[-fig_size[1]:, -fig_size[0]:, :] = fig_img
        return frame


def get_trials_ids(animal_id, movement_type=None, orm=None):
    orm = orm if orm is not None else ORM()
    with orm.session() as s:
        filters = [Experiment.animal_id == animal_id]
        if movement_type is not None:
            filters.append(Block.movement_type == movement_type)
        trs = s.query(Trial, Block, Experiment).join(
            Block, Block.id == Trial.block_id).join(
            Experiment, Experiment.id == Block.experiment_id).filter(*filters).all()
        return [tr.id for tr, blk, exp in trs]
    

if __name__ == '__main__':
    # print(get_trials_ids('PV91', movement_type='jump_up_old'))
    TrialPose(26269, is_dwh=True).play_trial(cam_name='back', is_save_video=True)

    # orm = ORM()
    # for tid in get_trials_ids('PV163', movement_type='accelerate', orm=orm):
    #     try:
    #         TrialPose(tid, orm=orm).play_trial(cam_name='back', is_save_video=True)
    #     except Exception as exc:
    #         print(f'Error: {exc}; {tid}')
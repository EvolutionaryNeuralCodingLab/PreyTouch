import time
from pprint import pprint
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import math
if Path('.').resolve().name != 'Arena':
    import os
    os.chdir('../..')
import config
from analysis.pose import ArenaPose
from analysis.pose_utils import put_text
from image_handlers.video_writers import ImageIOWriter
from db_models import ORM, Block, Strike, Trial, Temperature, Experiment

DEFAULT_OUTPUT_DIR = '/data/Pogona_Pursuit/output'


class MissingStrikeData(Exception):
    """Could not find timestamps for video frames"""


class Loader:
    def __init__(self, db_id, cam_name, raise_no_pose=False, is_use_db=True, is_debug=True, orm=None,
                 sec_before=3, sec_after=2, is_dwh=False, is_trial=False):
        self.db_id = db_id
        self.is_trial = is_trial
        self.cam_name = cam_name
        self.raise_no_pose = raise_no_pose  # raise exception if no pose data found
        self.is_use_db = is_use_db
        self.is_debug = is_debug
        self.sec_before = sec_before
        self.sec_after = sec_after
        self.is_dwh = is_dwh
        self.orm = orm if orm is not None else ORM()
        self.frames_delta = None
        self.n_frames_back = None
        self.n_frames_forward = None
        self.bug_traj_strike_id = None
        self.bug_traj_before_strike = None
        self.relevant_video_frames = (None, None)
        self.strike_frame_id = None
        self.video_path = None
        self.arena_name = None
        self.avg_temperature = None
        self.strike_video_writer = None
        self.dlc_pose = ArenaPose(cam_name, 'deeplabcut', is_use_db=is_use_db, orm=orm)
        self.frames_df: pd.DataFrame = pd.DataFrame()
        self.traj_df: pd.DataFrame = pd.DataFrame(columns=['time', 'x', 'y'])
        self.info = {}
        self.load()

    def __str__(self):
        return f'Strike-Loader:{self.db_id}' if not self.is_trial else f'Trial-Loader:{self.db_id}'

    def load(self):
        with self.orm.session() as s:
            if not self.is_trial:
                n_tries = 3
                for i in range(n_tries):
                    try:
                        strk = s.query(Strike).filter_by(id=self.db_id).first()
                        break
                    except Exception as exc:
                        time.sleep(0.2)
                        if i >= n_tries - 1:
                            raise exc
                if strk is None:
                    raise MissingStrikeData(f'could not find strike id: {self.db_id}')
                self.info = {k: v for k, v in strk.__dict__.items() if not k.startswith('_')}
                trial_id = strk.trial_id
            else:
                trial_id, strk = self.db_id, None

            trial = s.query(Trial).filter_by(id=trial_id).first()
            blk = s.query(Block).filter_by(id=trial.block_id).first()
            exp = s.query(Experiment).filter_by(id=blk.experiment_id).first()
            self.arena_name = exp.arena
            if trial is None:
                raise MissingStrikeData('No trial found in DB')

            self.load_bug_trajectory_data(trial, strk)
            self.load_frames_data(s, trial, strk)
            self.load_temperature(s, trial.block_id)

    def load_bug_trajectory_data(self, trial, strk):
        self.traj_df = pd.DataFrame(trial.bug_trajectory)
        if self.traj_df.empty:
            raise MissingStrikeData('traj_df is empty')
        self.traj_df['time'] = pd.to_datetime(self.traj_df.time).dt.tz_localize(None)
        if not self.is_trial:
            self.bug_traj_strike_id = (strk.time - self.traj_df.time).dt.total_seconds().abs().idxmin()
            n = self.sec_before / self.traj_df['time'].diff().dt.total_seconds().mean()
            self.bug_traj_before_strike = self.traj_df.loc[self.bug_traj_strike_id-n:self.bug_traj_strike_id].copy()

    def get_bug_traj_around_strike(self, sec_before=None, sec_after=None):
        if self.traj_df.empty:
            return None

        n_before = (sec_before or self.sec_before) / self.traj_df['time'].diff().dt.total_seconds().mean()
        n_after = (sec_after or self.sec_after) / self.traj_df['time'].diff().dt.total_seconds().mean()
        return self.traj_df.loc[self.bug_traj_strike_id - n_before:self.bug_traj_strike_id+n_after].copy()

    def load_frames_data(self, s, trial, strk):
        block = s.query(Block).filter_by(id=trial.block_id).first()
        self.update_info_with_block_data(block)

        cam_vids = [vid for vid in block.videos if vid.cam_name == self.cam_name]
        if not cam_vids:
            raise MissingStrikeData(f'No {self.cam_name} camera videos were found in DB for block {block.id}')

        errors = {}
        for vid in cam_vids:
            frames_times = self.load_frames_times(vid)
            if frames_times.empty:
                errors[Path(vid.path).stem] = 'frame times not found'
                continue
            # in strike mode use the strike time, in trial mode use the mid-trial time, and check if this time
            # is included in this video's frame times.
            center_time = self.traj_df.time.iloc[len(self.traj_df) // 2] if self.is_trial else strk.time
            if not (frames_times.iloc[0].time <= center_time <= frames_times.iloc[-1].time):
                errors[Path(vid.path).stem] = 'center frame is not in the video'
                continue

            if self.is_trial:  # in trials, relevant frames are taken from trajectory times
                times = [self.traj_df['time'].iloc[i] for i in [0, -1]]
                self.relevant_video_frames = [(t - frames_times.time).dt.total_seconds().abs().idxmin() for t in times]
            else:  # in strikes, relevant frames are calculated using strike frame with sec_before and sec_after
                self.strike_frame_id = (strk.time - frames_times.time).dt.total_seconds().abs().idxmin()
                self.relevant_video_frames = [max(self.strike_frame_id - self.n_frames_back, frames_times.index[0]),
                                              min(self.strike_frame_id + self.n_frames_forward, frames_times.index[-1])]

            self.set_video_path(vid)  # save video path
            frames_df = frames_times
            try:
                frames_df = self.load_pose(vid)
            except Exception as exc:
                errors[Path(vid.path).stem] = f'load pose failed; {exc}'
                if self.raise_no_pose:
                    raise Exception('loader failed since no pose found and raise_no_pose=True')

            first_frame, last_frame = self.relevant_video_frames
            self.frames_df = frames_df.loc[first_frame:last_frame].copy()
            self.frames_df['time'] = pd.to_datetime(self.frames_df.time, unit='s')
            # break since the relevant video was found
            break

        if self.frames_df.empty:
            if self.is_debug:
                print(f'Frames load failed:')
                pprint(errors)
            raise MissingStrikeData('frames_df is empty after loading')

    def load_pose(self, vid):
        return self.dlc_pose.load(video_db_id=vid.id, video_path=self.video_path, only_load=True)

    def set_video_path(self, vid):
        video_path = Path(vid.path).resolve()
        # fix for cases in which the analysis runs from other servers
        if DEFAULT_OUTPUT_DIR != config.OUTPUT_DIR and video_path.as_posix().startswith(DEFAULT_OUTPUT_DIR):
            video_path = Path(video_path.as_posix().replace(DEFAULT_OUTPUT_DIR, config.OUTPUT_DIR))
        if self.is_dwh:
            # TODO: move to config
            video_path = Path(f'/media/sil2/Data/regev/experiments/{self.arena_name}/' + '/'.join(video_path.parts[-5:]))
        if not video_path.exists():
            if self.is_debug:
                print(f'Video path does not exist: {video_path}')
            if not self.is_use_db:  # raise an exception only if not on DB mode
                raise Exception(f'Video path {video_path} does not exist')
        self.video_path = video_path

    def update_info_with_block_data(self, blk: Block):
        fields = ['movement_type', 'exit_hole', 'bug_speed']
        self.info.update({k: blk.__dict__.get(k) for k in fields})

    def load_frames_times(self, vid):
        frames_times = self.dlc_pose.load_frames_times(vid.id, vid.path)
        if not frames_times.empty:
            self.frames_delta = np.mean(frames_times.time.diff().dt.total_seconds())
            self.n_frames_back = round(self.sec_before / self.frames_delta)
            self.n_frames_forward = round(self.sec_after / self.frames_delta)
        return frames_times

    def load_temperature(self, s, block_id):
        temps = s.query(Temperature).filter_by(block_id=block_id).all()
        if not temps:
            self.avg_temperature = np.nan
        else:
            self.avg_temperature = np.mean([t.value for t in temps if isinstance(t.value, (int, float))])

    def get_strike_frame(self) -> np.ndarray:
        for _, frame in self.gen_frames_around_strike(0, 1):
            return frame

    def gen_frames(self, frame_ids, video_path=None, cam_name=None, frames_map=None):
        cap = cv2.VideoCapture(video_path or self.video_path.as_posix())
        start_frame, end_frame = frame_ids[0], frame_ids[-1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if i not in frame_ids:
                continue
            if cam_name != 'back':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frames_map is not None:
                i = frames_map[i]
            yield i, frame
        cap.release()

    def gen_frames_around(self, start_frame=None, end_frame=None, cam_name=None, step=1):
        start_frame = start_frame if start_frame is not None else self.relevant_video_frames[0]
        end_frame = end_frame if end_frame is not None else self.relevant_video_frames[1] + 1
        frame_ids = list(range(start_frame, end_frame, step))
        video_path = self.video_path.as_posix()
        if cam_name is not None and cam_name != self.cam_name:
            video_path, frame_ids, frames_pairs = self.align_frames_to_other_cam(cam_name, video_path, frame_ids)
            return self.gen_frames(frame_ids, video_path=video_path, cam_name=cam_name, frames_map=frames_pairs)
        else:
            return self.gen_frames(frame_ids, video_path=video_path, cam_name=cam_name)

    def gen_frames_around_strike(self, cam_name=None, n_frames_back=None, n_frames_forward=None, step=1):
        if self.is_trial:
            raise Exception('Loader is working in trial mode, cannot generate strike frames')
        start_frame, end_frame = self.relevant_video_frames[0], self.relevant_video_frames[1] + 1
        if n_frames_back is not None:
            start_frame = max([start_frame, self.strike_frame_id - n_frames_back])
        if n_frames_forward is not None:
            end_frame = min([end_frame, self.strike_frame_id + n_frames_forward])
        return self.gen_frames_around(start_frame, end_frame, cam_name, step)

    def align_frames_to_other_cam(self, cam_name: str, video_path: str, frame_ids: list):
        vids = list(Path(video_path).parent.glob(f'{cam_name}*.mp4'))
        if not vids:
            raise Exception(f'No videos found for {cam_name} camera in {Path(video_path).parent}')
        
        frames_times_path = vids[0].parent / 'frames_timestamps' / vids[0].with_suffix('.csv').name
        assert frames_times_path.exists(), f'File {frames_times_path} does not exist'
        frames_times = pd.read_csv(frames_times_path, index_col=0).reset_index().rename(columns={'index': 'frame_id'})
        frames_times['time'] = pd.to_datetime(frames_times['0'], unit='s', utc=True).dt.tz_convert(
            'Asia/Jerusalem').dt.tz_localize(None)
        orig_frames = self.frames_df.loc[frame_ids].time
        orig_frames = self.frames_df.loc[frame_ids][['time']].reset_index().rename(columns={'index': 'orig_frame'})
        orig_frames.columns = [c[0] for c in orig_frames.columns]

        merged = pd.merge_asof(left=orig_frames, right=frames_times, left_on='time', right_on='time', 
                                 direction='nearest', tolerance=pd.Timedelta('100 ms'))
        new_frames_ids = sorted(merged.frame_id.unique().tolist())
        frames_pairs = merged[['orig_frame', 'frame_id']].set_index('frame_id').orig_frame.to_dict()
        return vids[0].as_posix(), new_frames_ids, frames_pairs

    def get_bodypart_pose(self, bodypart):
        return pd.concat([pd.to_datetime(self.frames_df['time'], unit='s'), self.frames_df[bodypart]], axis=1)

    def _play(self, start_frame, end_frame, cam_name=None, annotations=None, between_frames_delay=None,
              save_video=False):
        is_pose_loaded = 'nose' in self.frames_df.columns
        if is_pose_loaded:
            nose_df = self.frames_df['nose']
        for i, frame in self.gen_frames_around(start_frame, end_frame, cam_name):
            if len(frame.squeeze().shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            if not self.is_trial and i == self.strike_frame_id:
                put_text('Strike Frame', frame, 30, 20)
            if annotations and i in annotations:
                put_text(annotations[i], frame, 30, frame.shape[0] - 30)
            if is_pose_loaded:
                if cam_name == self.cam_name:
                    self.dlc_pose.predictor.plot_predictions(frame, i, self.frames_df)
                if i in nose_df.index and not np.isnan(nose_df['cam_x'][i]):
                    angle = self.frames_df.loc[i, [("angle", "")]]
                    put_text(f'Angle={math.degrees(angle):.0f}', frame, 1000, 30)
            if save_video:
                self.save_strike_video(frame, cam_name)
            frame = cv2.resize(frame, None, None, fx=0.5, fy=0.5)
            cv2.imshow(str(self), frame)
            if between_frames_delay:
                time.sleep(between_frames_delay)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        if save_video:
            self.strike_video_writer.close()
        self.strike_video_writer = None

    def play_trial(self, cam_name=None, annotations=None, between_frames_delay=None, save_video=False):
        self._play(self.relevant_video_frames[0], self.relevant_video_frames[1], cam_name=cam_name,
                   annotations=annotations, between_frames_delay=between_frames_delay, save_video=save_video)

    def play_strike(self, cam_name=None, n_frames_back=None, n_frames_forward=None, annotations=None,
                    between_frames_delay=None, save_video=False):
        n_frames_back = n_frames_back if n_frames_back is not None else self.n_frames_back
        n_frames_forward = n_frames_forward if n_frames_forward is not None else self.n_frames_forward
        start_frame = max([self.strike_frame_id - n_frames_back], self.relevant_video_frames[0])
        end_frame = min([self.strike_frame_id + n_frames_forward], self.relevant_video_frames[1])
        self._play(start_frame, end_frame, cam_name=cam_name, annotations=annotations,
                   between_frames_delay=between_frames_delay, save_video=save_video)

    def save_strike_video(self, frame, cam_name):
        output_dir = Path(config.OUTPUT_DIR) / 'strikes_videos'
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.strike_video_writer is None:
            is_color = cam_name == 'back'
            video_path = (output_dir / f'strike_{self.db_id}_{cam_name}.mp4').as_posix()
            self.strike_video_writer = ImageIOWriter(frame, 30, None, cam_name, is_color, full_path=video_path)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.strike_video_writer.write(frame)

    def get_block_info(self):
        with self.orm.session() as s:
            strk = s.query(Strike).filter_by(id=self.db_id).first()
            blk = s.query(Block).filter_by(id=strk.block_id).first()
            return blk.__dict__


if __name__ == "__main__":
    ld = Loader(5353, 'top', is_use_db=True, is_dwh=False, is_trial=True)
    # ld = Loader(611, 'top', is_use_db=True, is_dwh=False, is_trial=True)
    ld.play_trial(cam_name='right')

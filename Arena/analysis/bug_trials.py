import sys
import time
from pathlib import Path
import yaml
import pickle
import cv2
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import config
from db_models import ORM, Experiment, Block, Trial, not_
from analysis.strikes.loader import Loader
from analysis.predictors.tongue_out import TongueTrainer


class BugTrialsAnalyzer:
    def __init__(self, is_use_db=True, is_only_pose=False, is_tqdm=True, animal_ids=None, is_debug=True, is_dwh=False):
        self.is_use_db = is_use_db
        self.is_only_pose = is_only_pose  # only scan trials with pose data
        self.is_tqdm = is_tqdm
        self.is_debug = is_debug
        self.is_dwh = is_dwh
        self.animal_ids = animal_ids
        self.tongue_model = self.init_tongue_model()
        self.orm = ORM()

    def run(self, block_id=None, is_cache=True):
        block_ids = self.scan(block_id=block_id, is_cache=is_cache)
        if not block_ids:
            print('No blocks found in DB scan; aborting.')
            return

        print(f'Found {len(block_ids)} blocks for trials analysis.')
        errors = {}
        for block_id, (cache_path, trials) in block_ids.items():
            res = []
            for trial_id, strikes_times in (tqdm(trials.items(), desc=cache_path.as_posix()) if self.is_tqdm else trials.items()):
                try:
                    tdf = self.run_on_trial(trial_id, strikes_times)
                    res.append(tdf)
                except Exception as e:
                    errors.setdefault(str(e), []).append(str(trial_id))
            if self.is_debug:
                for err, trial_ids in errors.items():
                    print(f'{err}; Trials:{",".join(trial_ids)}')
            time.sleep(0.1)
            if not res:
                print(f'None of the {len(trials)} trials of block {block_id} were processed successfully.')
                continue
            res = pd.concat(res)
            res.to_parquet(cache_path)
            self.plot_cached_block_results(block_id, is_overwrite=True)

    def scan(self, is_cache=True, block_id=None):
        block_ids = {}
        with self.orm.session() as s:
            filters = [not_(Experiment.animal_id.ilike('%test%')), Block.block_type == 'bugs']
            if block_id is not None:
                filters.append(Block.id == block_id)
            if self.animal_ids:
                filters.append(Experiment.animal_id.in_(self.animal_ids))
            orm_res = s.query(Block, Experiment).join(
                Experiment, Experiment.id == Block.experiment_id).filter(*filters).all()
            for blk, exp in orm_res:
                parent_path = Path(f'{exp.experiment_path}/block{blk.block_id}')
                cache_path = self.get_trials_analysis_filename(parent_path)
                if (cache_path.exists() and is_cache) or len(blk.trials) == 0:
                    continue
                block_ids[int(blk.id)] = (cache_path, {int(tr.id): [strk.time for strk in tr.strikes] for tr in blk.trials})
        return block_ids

    def run_on_trial(self, trial_id, strikes_times=None):
        if strikes_times is None:
            strikes_times = self.get_strike_times_for_trial(trial_id)
        ld = Loader(trial_id, config.NIGHT_POSE_CAMERA, is_use_db=self.is_use_db, is_trial=True, raise_no_pose=True,
                    orm=self.orm, is_debug=False)
        pose_df = (pd.concat([ld.frames_df[['time', 'bug_x', 'bug_y', 'angle']].droplevel(1, axis=1),
                                   ld.frames_df['nose']],
                             axis=1).reset_index().rename(columns={'index': 'frame_id', 'prob': 'pose_prob'}))
        tdf = self.predict_tongues(ld)
        tdf = pose_df.merge(tdf, how='left', on='frame_id')
        tdf['is_strike'] = False
        for strike_time in strikes_times:
            closest_index = (tdf.time - strike_time).dt.total_seconds().abs().idxmin()
            tdf.loc[closest_index, 'is_strike'] = True
        tdf['trial_id'] = trial_id
        # add the trial number to the dataframe
        with self.orm.session() as s:
            tr = s.query(Trial).filter(Trial.id == trial_id).first()
            tdf['trial_num'] = tr.in_block_trial_id
        return tdf

    def plot_cached_block_results(self, block_id, is_overwrite=False, is_show=False):
        block_path, trials_data = self.get_block_path_and_trials(block_id)
        analysis_cache_path = self.get_trials_analysis_filename(block_path)
        cached_plot_path = analysis_cache_path.with_suffix('.png')

        if not cached_plot_path.exists() or is_overwrite:
            is_analysis = analysis_cache_path.exists()
            cols = 5 if is_analysis else 3
            fig, axes = plt.subplots(len(trials_data), cols, figsize=(25, 3 * len(trials_data)),
                                     gridspec_kw={'width_ratios': [1, 1, 1, 2, 2]})
            self.plot_trials_images(block_path, axes[:, :3])
            if is_analysis:
                tdf = pd.read_parquet(analysis_cache_path)
                self.plot_trials_tongues_and_pose(axes[:, 3:], tdf)
                # save only if analysis exists
                fig.savefig(cached_plot_path.as_posix(), bbox_inches='tight')
            if is_show:
                plt.show()
            else:
                plt.close(fig)
        return cv2.imread(cached_plot_path.as_posix())

    @staticmethod
    def plot_trials_tongues_and_pose(axes, tdf):
        trials = tdf.trial_id.unique().tolist()
        for i, trial_id in enumerate(trials):
            df = tdf.query(f'trial_id=={trial_id}').copy()
            time = (df.time - df.time.iloc[0]).dt.total_seconds().values
            bug_x = df.bug_x.values
            screen_x_lim = np.array([0, int(config.SCREEN_RESOLUTION.split(',')[0])])
            if config.SCREEN_PIX_CM:
                bug_x = bug_x * config.SCREEN_PIX_CM
                screen_x_lim = screen_x_lim * config.SCREEN_PIX_CM
            axes[0].imshow(np.expand_dims(df.tongue_prob.values, axis=0), cmap='coolwarm', vmin=0, vmax=1,
                           aspect='auto',
                           extent=[time[0], time[-1], *screen_x_lim])
            axes[0].plot(time, bug_x, color='k', linewidth=2)
            axes[0].margins(x=0)
            axes[0].set_xlabel('Trial Time [sec]')
            axes[0].set_ylabel('ScreenX ' + '[cm]' if config.SCREEN_PIX_CM else '[pixels]')

            axes[1].plot(time, df.y.values, linewidth=2)
            axes[1].margins(x=0)
            for strike_time in df.query('is_strike').time.values:
                strike_time = (strike_time - df.time.iloc[0]).total_seconds()
                for j in [0, 1]:
                    axes[j].axvline(strike_time, color='tab:green', ls='--')
            if config.SCREEN_Y_CM:
                axes[1].axhline(config.SCREEN_Y_CM, color='tab:orange', ls='--')

    @staticmethod
    def plot_trials_images(block_path, axes):
        trials_images_dir = block_path / 'trials_images'
        if not trials_images_dir.exists():
            raise Exception(f'Trials images do not exist in: {block_path}')

        for image_path in trials_images_dir.glob('*.png'):
            _, trial_num, img_num = image_path.stem.split('_')
            ax = axes[int(trial_num)-1, int(img_num)]
            ax.imshow(cv2.imread(image_path.as_posix()))
            ax.axis('off')

    def get_block_path_and_trials(self, block_id):
        trial_cols = ['id', 'in_block_trial_id', 'trial_bugs', 'duration', 'bug_speed', 'exit_hole']
        with self.orm.session() as s:
            blk = s.query(Block).filter_by(id=block_id).first()
            if blk is None:
                raise Exception(f'could not find block with id: {block_id}')
            exp = s.query(Experiment).filter_by(id=blk.experiment_id).first()
            block_path = Path(f'{exp.experiment_path}/block{blk.block_id}')
            trials_data = []
            for tr in blk.trials:
                for col in trial_cols:
                    trials_data.append(getattr(tr, col, None))
            trials_data = pd.DataFrame(trials_data)
        return block_path, trials_data

    def get_strike_times_for_trial(self, trial_id):
        with self.orm.session() as s:
            tr = s.query(Trial).filter_by(id=trial_id).first()
            return [strk.time for strk in tr.strikes]

    @staticmethod
    def init_tongue_model() -> TongueTrainer:
        pconfig = config.load_configuration('predict')
        if 'tongue_out' not in pconfig:
            raise Exception('Unable to load tongue_out PredictHandler, since no tongue_out configuration in predict_config')
        return TongueTrainer(model_path=pconfig['tongue_out']['model_path'])

    def predict_tongues(self, ld: Loader):
        res = []
        for i, frame in ld.gen_frames_around():
            label, prob = self.tongue_model.predict(frame)
            res.append({'frame_id': i, 'label': label, 'tongue_prob': prob})
        return pd.DataFrame(res)

    def load_bug_trajectory(self, tr, parent_path):
        traj = tr.bug_trajectory
        if traj is None:  # try loading from file, if not available in the database
            traj = self.load_bug_trajectory_from_file(parent_path, tr.start_time, tr.end_time)
        return traj

    @staticmethod
    def load_bug_trajectory_from_file(parent_dir, start_time, end_time):
        traj_path = parent_dir / 'bug_trajectory.csv'
        if traj_path.exists():
            traj_df = pd.read_csv(traj_path, index_col=0)
            traj_df = traj_df.query(f'time >= "{start_time}" and time <= "{end_time}"')
            if not traj_df.empty:
                return traj_df

    def load_trial_model(self, trial_id):
        with self.orm.session() as s:
            return s.query(Trial).filter(Trial.id == trial_id).first()

    @staticmethod
    def is_bugs_block(parent_dir):
        d = yaml.load((parent_dir / 'info.yaml').open(), Loader=yaml.FullLoader)
        return d.get('block_type') == 'bugs'

    @staticmethod
    def get_trials_analysis_filename(parent_dir) -> Path:
        return Path(parent_dir) / 'trials_analysis.parquet'


if __name__ == "__main__":
    ts = BugTrialsAnalyzer(animal_ids=['PV51'])
    ts.run()
    # ts.run(4909, is_cache=False)
    # ts.plot_cached_block_results(block_id=4909)
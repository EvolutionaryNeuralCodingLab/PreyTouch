import time
from pathlib import Path
import yaml
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import config
from db_models import ORM, Experiment, Block, Trial, not_
from analysis.strikes.loader import Loader
from analysis.predictors.tongue_out import TongueTrainer


class TrialScanner:
    def __init__(self, is_use_db=True, is_only_pose=False, is_tqdm=True, animal_ids=None):
        self.is_use_db = is_use_db
        self.is_only_pose = is_only_pose  # only scan trials with pose data
        self.is_tqdm = is_tqdm
        self.animal_ids = animal_ids
        self.tongue_model = self.init_tongue_model()
        self.orm = ORM()

    def run(self):
        block_ids = self.scan()
        if not block_ids:
            print('No blocks found in DB scan; aborting.')
            return

        print(f'Found {len(block_ids)} blocks for trials analysis.')
        for block_id, (cache_path, trials) in block_ids.items():
            res, trials_images = [], {}
            for trial_id, strikes_times in (tqdm(trials.items(), desc=cache_path.as_posix()) if self.is_tqdm else trials.items()):
                try:
                    tdf, images = self.run_on_trial(trial_id, strikes_times)
                    trials_images[trial_id] = images
                    res.append(tdf)
                except Exception as e:
                    continue
            time.sleep(0.1)
            if not res:
                print(f'None of the {len(trials)} trials of block {block_id} were processed successfully.')
                continue
            res = pd.concat(res)
            self.plot_trials_analysis(res, trials_images, cache_path)
            res.to_csv(cache_path)

    def scan(self, is_cache=True):
        block_ids = {}
        with self.orm.session() as s:
            filters = [not_(Experiment.animal_id.ilike('%test%')), Block.block_type == 'bugs']
            if self.animal_ids:
                filters.append(Experiment.animal_id.in_(self.animal_ids))
            orm_res = s.query(Block, Experiment).join(
                Experiment, Experiment.id == Block.experiment_id).filter(*filters).all()
            for blk, exp in orm_res:
                parent_path = Path(f'{exp.experiment_path}/block{blk.block_id}')
                cache_path = self.get_trial_analysis_filename(parent_path)
                if (cache_path.exists() and is_cache) or len(blk.trials) == 0:
                    continue
                block_ids[int(blk.id)] = (cache_path, {int(tr.id): [strk.time for strk in tr.strikes] for tr in blk.trials})
        return block_ids

    def run_on_trial(self, trial_id, strikes_times):
        ld = Loader(trial_id, config.NIGHT_POSE_CAMERA, is_use_db=self.is_use_db, is_trial=True, raise_no_pose=True)
        pose_df = (pd.concat([ld.frames_df[['time', 'bug_x', 'bug_y', 'angle']].droplevel(1, axis=1),
                                   ld.frames_df['nose']],
                             axis=1).reset_index().rename(columns={'index': 'frame_id', 'prob': 'pose_prob'}))
        tdf, images = self.predict_tongues(ld)
        tdf = pose_df.merge(tdf, how='left', on='frame_id')
        tdf['is_strike'] = False
        for strike_time in strikes_times:
            closest_index = (tdf.time - strike_time).dt.total_seconds().abs().idxmin()
            tdf.loc[closest_index, 'is_strike'] = True
        tdf['trial_id'] = trial_id
        return tdf, images

    def plot_trials_analysis(self, df, trials_images, cache_path):
        trials = df.trial_id.unique().tolist()
        fig, axes = plt.subplots(len(trials), 5, figsize=(25, 3*len(trials)))
        for i, trial_id in enumerate(trials):
            self.plot_tongues_and_bug_position(df.query(f'trial_id=={trial_id}'), axes[i, 3:])
            for j, image in enumerate(trials_images[trial_id]):
                axes[i, j].imshow(image, cmap='gray')
                axes[i, j].axis('off')
        fig.tight_layout()
        fig.savefig(cache_path.with_suffix('.png'), bbox_inches='tight')

    @staticmethod
    def plot_tongues_and_bug_position(tdf, axes):
        time = (tdf.time - tdf.time.iloc[0]).dt.total_seconds().values
        bug_x = tdf.bug_x.values
        screen_x_lim = np.array([0, int(config.SCREEN_RESOLUTION.split(',')[0])])
        if config.SCREEN_PIX_CM:
            bug_x = bug_x * config.SCREEN_PIX_CM
            screen_x_lim = screen_x_lim * config.SCREEN_PIX_CM
        axes[0].imshow(np.expand_dims(tdf.tongue_prob.values, axis=0), cmap='coolwarm', vmin=0, vmax=1, aspect='auto',
                       extent=[time[0], time[-1], *screen_x_lim])
        axes[0].plot(time, bug_x, color='k', linewidth=2)
        axes[0].margins(x=0)
        axes[0].set_xlabel('Trial Time [sec]')
        axes[0].set_ylabel('ScreenX ' + '[cm]' if config.SCREEN_PIX_CM else '[pixels]')

        axes[1].plot(time, tdf.y.values, linewidth=2)
        axes[1].margins(x=0)
        for strike_time in tdf.query('is_strike').time.values:
            strike_time = (strike_time - tdf.time.iloc[0]).total_seconds()
            for j in [0, 1]:
                axes[j].axvline(strike_time, color='tab:green', ls='--')
        if config.SCREEN_Y_CM:
            axes[1].axhline(config.SCREEN_Y_CM, color='tab:orange', ls='--')
        # axes[1].set_ylim([-5, 80])

    @staticmethod
    def init_tongue_model() -> TongueTrainer:
        pconfig = config.load_configuration('predict')
        if 'tongue_out' not in pconfig:
            raise Exception('Unable to load tongue_out PredictHandler, since no tongue_out configuration in predict_config')
        return TongueTrainer(model_path=pconfig['tongue_out']['model_path'])

    def predict_tongues(self, ld: Loader):
        res, images = [], []
        for i, frame in ld.gen_frames_around():
            if i in ld.relevant_video_frames + [round(np.mean(ld.relevant_video_frames))]:
                images.append(frame)
            label, prob = self.tongue_model.predict(frame)
            res.append({'frame_id': i, 'label': label, 'tongue_prob': prob})
        return pd.DataFrame(res), images

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
    def get_trial_analysis_filename(parent_dir) -> Path:
        return Path(parent_dir) / 'trials_analysis.csv'


if __name__ == "__main__":
    ts = TrialScanner(animal_ids=['PV157'])
    ts.run()
    # ts.run_on_trial(5353)
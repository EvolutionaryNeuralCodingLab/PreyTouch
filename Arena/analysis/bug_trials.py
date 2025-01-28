import sys
import tempfile
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
from sqlalchemy import func
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

    def run(self, block_id=None, block_path=None, is_cache=True):
        if block_path and not block_id:
            block_id = self.get_block_id_from_block_path(block_path)
        block_ids = self.scan(block_id=block_id, is_cache=is_cache)
        if not block_ids:
            print('No blocks found in DB scan; aborting.')
            return

        print(f'Found {len(block_ids)} blocks for trials analysis.')
        for i, (block_id, (cache_path, trials)) in enumerate(block_ids.items()):
            res, errors = [], {}
            for trial_id, strikes_times in (tqdm(trials.items(), desc=f'({i+1}/{len(block_ids)}) {cache_path.as_posix()}') if self.is_tqdm else trials.items()):
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
            res['time'] = res.time.astype('datetime64[us]')
            res.to_parquet(cache_path)
            self.plot_cached_block_results(block_id, is_overwrite=True)

    def scan(self, is_cache=True, block_id=None, drop_with_tags=True):
        block_ids = {}
        with self.orm.session() as s:
            filters = [not_(Experiment.animal_id.ilike('%test%')), Block.block_type == 'bugs']
            if block_id is not None:
                filters.append(Block.id == block_id)
            if self.animal_ids:
                filters.append(Experiment.animal_id.in_(self.animal_ids))
            if drop_with_tags:
                filters.append(func.coalesce(Block.tags, '') == '')
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
        ld = Loader(trial_id, self.get_tongue_camera(), is_use_db=self.is_use_db, is_trial=True, raise_no_pose=True,
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

    def plot_cached_block_results(self, block_id=None, block_path=None, is_overwrite=False, is_show=False):
        if block_path and not block_id:
            block_id = self.get_block_id_from_block_path(block_path)
            print(f'Block ID found: {block_id}')
        block_path, trials_data = self.get_block_path_and_trials(block_id)
        analysis_cache_path = self.get_trials_analysis_filename(block_path)
        cached_plot_path = analysis_cache_path.with_suffix('.png')
        return_img_path = cached_plot_path.as_posix()

        if not cached_plot_path.exists() or is_overwrite:
            is_analysis = analysis_cache_path.exists()
            ratios = [1, 2, 2, 2, 4, 4] if is_analysis else [1, 2, 2, 2]
            rows, cols = len(trials_data), len(ratios)
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), gridspec_kw={'width_ratios': ratios})
            self.plot_trials_images(block_path, trials_data, axes[:, :4])

            # if pose and tongues data is available, add them to the plot
            if is_analysis:
                tdf = pd.read_parquet(analysis_cache_path)
                self.plot_trials_tongues_and_pose(axes[:, 4:], tdf)
            else:
                return_img_path = f'{tempfile.NamedTemporaryFile(delete=False).name}.png'
            fig.tight_layout()
            fig.savefig(return_img_path, bbox_inches='tight')
            if not is_show:
                plt.close(fig)

        res_img = cv2.imread(return_img_path)
        if is_show:
            window_name = 'Block Results'
            # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # cv2.imshow(window_name, cv2.resize(res_img, (960, 1300)))
            cv2.imshow(window_name, res_img)
            cv2.moveWindow(window_name, 0, 0)
            while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
                keycode = cv2.waitKey(50)
                if keycode > 0:
                    break
            cv2.destroyAllWindows()
        return res_img

    @staticmethod
    def plot_trials_tongues_and_pose(axes, tdf):
        trials = tdf.trial_id.unique().tolist()
        for trial_id in trials:
            df = tdf.query(f'trial_id=={trial_id}').copy()
            i = df.trial_num.iloc[0] - 1
            time = (df.time - df.time.iloc[0]).dt.total_seconds().values
            bug_x = df.bug_x.values
            screen_x_lim = np.array([0, int(config.SCREEN_RESOLUTION.split(',')[0])])
            if config.SCREEN_PIX_CM:
                bug_x = bug_x * config.SCREEN_PIX_CM
                screen_x_lim = screen_x_lim * config.SCREEN_PIX_CM
            axes[i, 0].imshow(np.expand_dims(df.tongue_prob.values, axis=0), cmap='coolwarm', vmin=0, vmax=1,
                           aspect='auto',
                           extent=[time[0], time[-1], *screen_x_lim])
            axes[i, 0].plot(time, bug_x, color='k', linewidth=2)
            axes[i, 0].margins(x=0)
            axes[i, 0].set_xlabel('Trial Time [sec]')
            axes[i, 0].set_ylabel('ScreenX ' + '[cm]' if config.SCREEN_PIX_CM else '[pixels]')

            axes[i, 1].plot(time, df.y.values, linewidth=2)
            axes[i, 1].margins(x=0)
            for strike_time in df.query('is_strike').time.values:
                strike_time = (strike_time - df.time.iloc[0]).total_seconds()
                for j in [0, 1]:
                    axes[i, j].axvline(strike_time, color='tab:green', ls='--')
            if config.SCREEN_Y_CM:
                axes[i, 1].axhline(config.SCREEN_Y_CM, color='tab:orange', ls='--')

    @staticmethod
    def plot_trials_images(block_path, trials_data, axes):
        trials_images_dir = block_path / 'trials_images'
        if not trials_images_dir.exists():
            raise Exception(f'Trials images do not exist in: {block_path}')

        for _, row in trials_data.reset_index(drop=True).iterrows():
            i = row.in_block_trial_id - 1
            text = "\n".join([f'{k}: {v}' for k, v in row.to_dict().items()])
            axes[i, 0].text(-0.1, 1, text, va='top')
            axes[i, 0].axis('off')

        for image_path in trials_images_dir.glob('*.png'):
            _, trial_num, img_num = image_path.stem.split('_')
            if int(img_num)+1 >= axes.shape[1]:
                continue
            ax = axes[int(trial_num)-1, int(img_num)+1]
            img = cv2.imread(image_path.as_posix())
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
                row = {}
                for col in trial_cols:
                    row[col] = getattr(tr, col)
                row['num_strikes'] = len(tr.strikes)
                row['movement_type'] = blk.movement_type
                trials_data.append(row)
            trials_data = pd.DataFrame(trials_data)
        return block_path, trials_data

    def get_block_id_from_block_path(self, block_path):
        block_id = None
        exp_path = Path(block_path).parent.as_posix()
        block_number = int(Path(block_path).name.replace('block', ''))

        with self.orm.session() as s:
            for exp in s.query(Experiment).filter_by(experiment_path=exp_path).all():
                for blk in exp.blocks:
                    if blk.block_id == block_number:
                        block_id = blk.id
                        break
        if not block_id:
            raise Exception(f'could not find block with experiment path: {exp_path}')
        return block_id

    def get_strike_times_for_trial(self, trial_id):
        with self.orm.session() as s:
            tr = s.query(Trial).filter_by(id=trial_id).first()
            return [strk.time for strk in tr.strikes]

    def init_tongue_model(self) -> TongueTrainer:
        pconfig = config.load_configuration('predict')
        if 'tongue_out' not in pconfig:
            raise Exception('Unable to load tongue_out PredictHandler, since no tongue_out configuration in predict_config')
        return TongueTrainer(model_path=pconfig['tongue_out']['model_path'], is_debug=self.is_debug)

    def predict_tongues(self, ld: Loader):
        res = []
        for i, frame in ld.gen_frames_around():
            label, prob = self.tongue_model.predict(frame)
            res.append({'frame_id': i, 'label': label, 'tongue_prob': prob})
        return pd.DataFrame(res)

    @staticmethod
    def get_tongue_camera():
        tongue_cam = config.NIGHT_POSE_CAMERA  # default in case no tongue_out camera configuration
        conf = config.load_configuration('cameras')
        for cam_name, cam_dict in conf.items():
            if 'tongue_out' in cam_dict.get('predictors', {}):
                tongue_cam = cam_name
                break
        return tongue_cam

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


def create_trial_images_dir(animal_ids=None):
    orm = ORM()
    with orm.session() as s:
        exps = s.query(Experiment)
        if animal_ids is not None:
            exps = exps.filter(Experiment.animal_id.in_(animal_ids))
        blocks = []
        for exp in exps.all():
            for blk in exp.blocks:
                if blk.block_type != 'bugs':
                    continue
                block_path = Path(f'{exp.experiment_path}/block{blk.block_id}')
                trials_images_dir = block_path / 'trials_images'
                if trials_images_dir.exists() and [x for x in trials_images_dir.glob('*.png')]:
                    continue
                blocks.append((int(blk.id), trials_images_dir))
        
        print(f'Found {len(blocks)} blocks to create trial images for')
        for i, (block_id, trials_images_dir) in enumerate(blocks):
            blk = s.query(Block).filter_by(id=block_id).first()
            trials_images_dir.mkdir(parents=True, exist_ok=True)
            errors = {}
            for tr in tqdm(blk.trials, desc=f'({i+1}/{len(blocks)}){trials_images_dir.parent}'):
                trial_num = int(tr.in_block_trial_id)
                img_num = 0
                try:
                    ld = Loader(int(tr.id), config.TRIAL_IMAGE_CAMERA, is_debug=False, is_trial=True, orm=orm, raise_no_traj=False)
                    for frame_id, frame in ld.gen_frames_around():
                        if frame_id in ld.relevant_video_frames + [round(np.mean(ld.relevant_video_frames))]:
                            frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
                            cv2.imwrite(str(trials_images_dir / f'trial_{trial_num}_{img_num}.png'), frame)
                            img_num += 1
                except Exception as exc:
                    errors.setdefault(str(exc), []).append(str(tr.id))
            if errors:
                for err_text, trial_ids in errors.items():
                    print(f'ERROR: {err_text} for trials: {",".join(trial_ids)}')


if __name__ == "__main__":
    ts = BugTrialsAnalyzer(is_debug=True, animal_ids=['PV51'])
    # blk_path = '/data/PreyTouch/output/experiments/PV162/20240414/block1'
    ts.run(is_cache=True)
    # ts.run(892, is_cache=False)
    # ts.plot_cached_block_results(block_id=1983, is_show=True, is_overwrite=True)
    # create_trial_images_dir(['PV51'])

    # /data/PreyTouch/output/experiments/PV162/20240413/block1
    # /data/PreyTouch/output/experiments/PV162/20240414/block1
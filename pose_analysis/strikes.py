from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import re
from pose import Analyzer, BODY_PARTS
from scipy.signal import find_peaks
from loader import Loader

NUM_FRAMES_BACK = 200


class StrikesAnalyzer:
    def __init__(self, loader: Loader = None, experiment_name=None, trial_id=None, camera=None,
                 n_frames_back=None, n_frames_forward=10):
        self.loader = loader or Loader(experiment_name, trial_id, camera)
        self.pose_df = Analyzer(self.loader.video_path).run_pose(load_only=True)
        self.xfs = []
        self.run_strikes_pose(n_frames_back, n_frames_forward)

    def run_strikes_pose(self, n_frames_back=None, n_frames_forward=10):
        """Run pose estimation on hits frames"""
        self.xfs = []
        n_frames_back = n_frames_back or NUM_FRAMES_BACK

        def first_frame(frame_id):
            return frame_id - n_frames_back if frame_id >= n_frames_back else 0
        
        def last_frame(frame_id):
            return frame_id + n_frames_forward if frame_id < len(self.pose_df) - n_frames_forward else self.pose_df.index[-1]
        
        frames_groups = [list(range(first_frame(f), last_frame(f))) for f in self.loader.get_hits_frames()]
        for frame_group in frames_groups:
            # flat_frames = sorted([item for sublist in frame_group for item in sublist])
            self.xfs.append(self.pose_df.loc[frame_group, :])

    # def play_strike(self, xf: pd.DataFrame, n=20):
    #     frames2plot = xf.index[-n:]
    #     cols = 3
    #     rows = int(np.ceil(len(frames2plot) / cols))
    #     fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    #     axes = axes.flatten()
    #     for i, frame_id in enumerate(frames2plot):
    #         frame = self.pose_analyzer.saved_frames.get(frame_id)
    #         if frame is None:
    #             continue
    #         axes[i].imshow(frame, aspect="auto")
    #         for part in BODY_PARTS:
    #             axes[i].scatter(xf.loc[frame_id, part].x, xf.loc[frame_id, part].y)
    #         axes[i].set_title(frame_id)
    #     fig.tight_layout()
    #     plt.show()

    def ballistic_analysis(self, circle_only=False):
        if circle_only:
            if self.loader.info.get('movement_type') != 'circle':
                return
            x_center, y_center = self.loader.fit_circle()
            theta = lambda x1, y1: np.arctan2(x1, y1)

        res = []
        hits_df = self.loader.hits_df
        for i, xf in enumerate(self.xfs):
            start_frame = self.get_strike_start_frame(xf)
            if start_frame is not None:
                strike_start_time = self.loader.frames_ts[start_frame]
                traj_time = self.loader.traj_df.time.dt.tz_convert('utc').dt.tz_localize(None)
                closest_hit_index = (traj_time - pd.to_datetime(strike_start_time)).abs().argsort()[0]
                bug_start_pos = self.loader.traj_df.loc[closest_hit_index, :]
                if circle_only:
                    PD = theta(bug_start_pos.x - x_center, bug_start_pos.y - y_center) -\
                         theta(hits_df.loc[i, 'x'] - x_center, hits_df.loc[i, 'y'] - y_center)
                    MD = theta(hits_df.loc[i, 'bug_x'] - x_center, hits_df.loc[i, 'bug_y'] - y_center) -\
                         theta(hits_df.loc[i, 'x'] - x_center, hits_df.loc[i, 'y'] - y_center)
                else:
                    PD = distance(bug_start_pos.x, bug_start_pos.y, hits_df.loc[i, 'x'], hits_df.loc[i, 'y'])
                    MD = distance(hits_df.loc[i, 'bug_x'], hits_df.loc[i, 'bug_y'],
                                  hits_df.loc[i, 'x'], hits_df.loc[i, 'y'])
                res.append(PD / MD)
        return res

    def transform_circle(self) -> pd.DataFrame:
        """Transform the (x,y) coordinates into circular coordinates in which the bug position while hit is
        the center of axis. Return data frame with the headers: x,y - that represent the transformed hit position,
        start_x, start_y - bug position when lizard started leap towards bug."""
        x_center, y_center = self.loader.fit_circle()

        def project(cx, cy, x, y) -> np.ndarray:
            cx = cx - x_center
            x = x - x_center
            cy = -cy - y_center
            y = -y - y_center
            th = np.arctan2(cx, cy)
            th = th - np.pi / 2
            r = np.array(((np.cos(th), -np.sin(th)),
                          (np.sin(th), np.cos(th))))
            u = np.array([x, y]) - np.array([cx, cy])
            xr = r.dot(np.array([1, 0]))
            yr = r.dot(np.array([0, 1]))
            projection = lambda x1, y1: y1 * np.dot(y1, x1) / np.dot(y1, y1)
            v = np.array([np.dot(projection(u, xr), xr), np.dot(projection(u, yr), yr)])
            if np.abs(v[0]) > 1000 or np.abs(v[1]) > 1000:
                return np.array([np.nan, np.nan])
            return v

        def project2(cx, cy, x, y):
            cx = cx - x_center
            x = x - x_center
            cy = -cy - y_center
            y = -y - y_center
            theta = lambda x1, y1: np.arctan2(x1, y1)
            rho = lambda x1, y1: np.sqrt(x1**2 + y1**2)
            return np.array([theta(x, y) -theta(cx,cy), rho(x, y) - rho(cx, cy)])

        vs = []
        for i, row in self.loader.hits_df.iterrows():
            start_frame = self.get_strike_start_frame(self.xfs[i])
            s = self.loader.bug_data_for_frame(start_frame)
            # hit = np.array([row.x - x_center, -row.y - y_center])
            # bug_start = np.array([s.x - x_center, -s.y - y_center])
            # bug = np.array([row.bug_x - x_center, -row.bug_y - y_center])
            hit = project2(row.bug_x, row.bug_y, row.x, row.y)
            bug_start = project2(row.bug_x, row.bug_y, s.x, s.y)
            vs.append(np.concatenate([hit, bug_start]))

        vs = pd.DataFrame(vs, columns=['x', 'y', 'start_x', 'start_y'])
        return vs

    @staticmethod
    def get_strike_start_frame(xf: pd.DataFrame, th=1.5, grace=3, min_leap=20) -> (int, None):
        y = xf['nose'].y
        peaks, _ = find_peaks(y.to_numpy(), height=840, distance=10)
        hit_idx = y.index[peaks][-1] if len(peaks) > 0 else y.index[-1]
        dy = y.diff()
        grace_count = 0
        first_idx = y.index[0]
        for r in np.arange(hit_idx, dy.index[0], -1):
            if dy[r] < th:
                grace_count += 1
                if grace_count < grace:
                    continue
                first_idx = r
                break
            else:
                grace_count = 0
        if y[hit_idx] - y[first_idx] < min_leap:
            return
        return first_idx


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

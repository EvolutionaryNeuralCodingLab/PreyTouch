from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from scipy.stats import circmean

# Metric groups used when plotting trials
LINEAR_METRICS = [
    "nose_cam_x",
    "nose_cam_y",
    "left_ear_cam_x",
    "left_ear_cam_y",
    "right_ear_cam_x",
    "right_ear_cam_y",
    "head_angle_deg",
    "head_angle_n_shifted_deg",
]
LINEAR_METRICS_BASE = [
    "nose_x",
    "nose_y",
    "left_ear_x",
    "left_ear_y",
    "right_ear_x",
    "right_ear_y",
    "head_angle_deg",
    "head_angle_n_shifted_deg",
]


class MetricsMode(Enum):
    cam = "cam"
    parts = "parts"


MODE_METRICS = {
    MetricsMode.cam: LINEAR_METRICS,
    MetricsMode.parts: LINEAR_METRICS_BASE,
}


@dataclass
class AnimalData:
    key: str
    df: pd.DataFrame
    data_path: Path
    data_signature: Optional[Tuple[str, ...]] = None


_loaded_animals: Dict[str, AnimalData] = {}




def _normalize_key(animal_key: str) -> str:
    key = (animal_key or "").strip().lower()
    if not key:
        raise ValueError("animal_key must be non-empty")
    return key


def _resolve_data_path(data_path: str | Path) -> Path:
    if data_path is None:
        raise ValueError("data_path must be provided")
    path = Path(data_path).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if path.is_dir():
        raise IsADirectoryError(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _data_signature(data_path: Path) -> Tuple[str, ...]:
    return ("path", str(data_path))

def _coerce_mode(mode: MetricsMode | str | Iterable[str]) -> Iterable[str]:
    if isinstance(mode, MetricsMode):
        return MODE_METRICS[mode]
    if isinstance(mode, str):
        try:
            return MODE_METRICS[MetricsMode[mode]]
        except KeyError:
            try:
                return MODE_METRICS[MetricsMode(mode)]
            except (KeyError, ValueError):
                raise ValueError(f"Unknown MetricsMode: {mode!r}") from None
    return mode

def _format_day(day_val: Any) -> str:
    if isinstance(day_val, (datetime, date)):
        return day_val.strftime("%Y%m%d")
    day_str = str(day_val)
    if "T" in day_str:
        day_str = day_str.split("T", 1)[0]
    return day_str.replace("-", "")

nanmean_func = lambda x: np.nanmean(x)
circmean_func = lambda x, high=180, low=-180, nan_policy="omit": circmean(x, high=high, low=low, nan_policy=nan_policy)
mean_func = lambda x: np.nanmean(x)

# region df helpers
################################################################################################
#------------------------------------------- df helpers ---------------------------------------#
################################################################################################

def filter_missing_from_df(df: pd.DataFrame) -> pd.DataFrame:
    missing_time_Realtive_to_end = []
    for trial_id in df['trial_id'].unique():
        trial_data = df[df['trial_id'] == trial_id]
        if trial_data['time_relative_to_end'].max() <= 0:
            missing_time_Realtive_to_end.append(trial_id)
            # print(f"Trial {trial_id} is missing data after 5s")
    df = df[~df['trial_id'].isin(missing_time_Realtive_to_end)].copy()
    print(f"Filtered out trials with missing data after 0s: {missing_time_Realtive_to_end}")
    return df

# endregion


#region Loading trials data
################################################################################################
###################################### Loading trials data #####################################
################################################################################################


def load_df_from_path(path: str | Path) -> pd.DataFrame:
    """
    Load a dataframe from an explicit parquet or csv path.
    """
    path = _resolve_data_path(path)
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)
    except EmptyDataError:
        print(f"[load_df_from_path] Empty file: {path} - returning empty DataFrame")
        return pd.DataFrame()

def load_animal_df(
    animal_key: str,
    data_path: str | Path,
    force_reload: bool = False,
) -> AnimalData:
    """
    Load the dataframe for a given animal key, caching the result.
    Data is loaded from an explicit parquet or csv file path.
    """
    raw_key = (animal_key or "").strip()
    key = _normalize_key(raw_key)
    path = _resolve_data_path(data_path)
    signature = _data_signature(path)
    cached = _loaded_animals.get(key)
    if cached and not force_reload and cached.data_signature == signature:
        return cached

    df = load_df_from_path(path).copy(deep=True)
    animal_data = AnimalData(
        key=key,
        df=df,
        data_path=path,
        data_signature=signature,
    )
    _loaded_animals[key] = animal_data
    return animal_data

def get_trial_info(
    df: pd.DataFrame,
    trial_id: int,
    *,
    arena: str = "reptilearn4",
    cam: Optional[str] = "front",
    data_volume: str = "Data",
) -> Dict[str, Any]:
    """
    Gather metadata and file paths for a specific trial.
    """
    trial_rows = df.loc[df["trial_id"] == trial_id]
    if trial_rows.empty:
        raise ValueError(f"trial_id {trial_id!r} not found in df")

    row = trial_rows.iloc[0]
    animal_id = row["animal_id"]
    day = _format_day(row["day"])
    blk = row["in_exp_block_id"]
    trl = row["in_trial_id"]
    sec = row["block_sec"]
    start = row["start_time"]
    hit = row["time_of_hit"]
    hit_bug = row["hit_bug_type"] if "hit_bug_type" in row else None

    camera = (cam or "front").lower()
    arena_name = arena

    media_dir = Path(f"/Volumes/{data_volume}/Bareket/experiments/{arena_name}/{animal_id}/{day}/block{blk}/videos")
    sil_media_dir = Path.home() / f"media/sil3/Bareket/experiments/{arena_name}/{animal_id}/{day}/block{blk}/videos"
    prediction_path = Path(f"/Volumes/{data_volume}/Bareket/experiments/{arena_name}/{animal_id}/{day}/block{blk}/predictions")

    candidates = sorted(p for p in sil_media_dir.glob("*.mp4") if camera in p.name.lower())
    if not candidates:
        print(f"No .mp4 files containing '{camera}' found in {sil_media_dir}")

    video_path_sil = candidates[0] if candidates else Path()
    video_name = video_path_sil.name if candidates else ""
    video_path_media = media_dir / video_name if video_name else Path()

    print(f"{video_path_media} (trial {trl} at sec: {sec})")

    return {
        "media": str(media_dir),
        "sil_media": str(sil_media_dir),
        "prediction_path": str(prediction_path),
        "trial_id": trial_id,
        "in_trial_id": trl,
        "block_sec": sec,
        "start_time": start,
        "hit": hit,
        "day": day,
        "block_id": blk,
        "animal_id": animal_id,
        "video_name": video_name,
        "media_video_path": str(video_path_media),
        "sil_video_path": str(video_path_sil),
        "hit_bug_type": hit_bug,
    }


def get_trial(
    animal_key: str,
    trial_id: int,
    cam: Optional[str] = None,
    post_win: float = 10,
    mode: MetricsMode | str | Iterable[str] = MetricsMode.cam,
    *,
    data_path: str | Path,
    arena: str = "reptilearn4",
    data_volume: str = "Data",
) -> Dict[str, Any]:
    """
    Load an animal, plot the requested trial, and return its metadata.
    """
    animal = load_animal_df(animal_key, data_path=data_path)
    metrics = _coerce_mode(mode)
    plot_dlc_trial(animal.df, trial_id, post_win=post_win, linear_metrics=metrics)
    info = get_trial_info(
        animal.df,
        trial_id,
        arena=arena,
        cam=cam,
        data_volume=data_volume,
    )
    return info
#endregion



#region Plotting utilities for trials data
################################################################################################
######################## Plotting utilities for trials data ####################################
################################################################################################



def calc_metric_per_bin(_examples_trials: pd.DataFrame, metric_col: str, mean_func, observed: bool = True):
    # 1) average metric per trial per bin
    trial_bin_means = (
        _examples_trials.dropna(subset=[metric_col])
        .groupby(["trial_id", "time_bin"], observed=observed)[metric_col]
        .apply(lambda x: mean_func(x))
        .reset_index(name=f"{metric_col}_trial_bin_mean")
    )

    # 2) average across trials per bin -> each trial has equal weight
    mean_metric = (
        trial_bin_means.groupby("time_bin", observed=observed)[f"{metric_col}_trial_bin_mean"]
        .apply(lambda x: mean_func(x))
        .reset_index(name=f"mean_{metric_col}")
    )

    # 3) how many trials contributed to each bin
    n_trials_per_bin = (
        trial_bin_means.groupby("time_bin", observed=observed)["trial_id"].nunique().reset_index(name="n_trials_in_bin")
    )

    return mean_metric, n_trials_per_bin


def plot_means_only_multi_by_prefix(
    _examples_trials: pd.DataFrame,
    metric_cols: Iterable[str],
    mean_func=np.nanmean,
    observed: bool = True,
    ax=None,
    set_legend: bool = True,
):
    """
    Plot only the per-bin means for multiple metric columns on the same axes.
    Color is determined by metric prefix (nose / left / right),
    linestyle + marker distinguish x vs y.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    prefixes = sorted({m.split("_")[0] for m in metric_cols})
    cmap = [
        "blue"
        if "nose" in p
        else "green"
        if "right" in p
        else "orange"
        if "left" in p
        else "purple"
        if "mid" in p
        else "black"
        for p in prefixes
    ]
    prefix_to_color = dict(zip(prefixes, cmap))

    def suffix_style_and_marker(metric_name: str) -> Tuple[str, str]:
        suffix = metric_name.split("_")[-1]
        if suffix == "x":
            return "-", "x"
        elif suffix == "y":
            return "--", "$ y $"
        return "-.", "o"

    for metric_col in metric_cols:
        mean_metric, _ = calc_metric_per_bin(
            _examples_trials,
            metric_col,
            mean_func=mean_func,
            observed=observed,
        )

        time_centers = mean_metric["time_bin"].apply(lambda x: x.mid)
        prefix = metric_col.split("_")[0]
        color = prefix_to_color.get(prefix, "black")
        ls, marker = suffix_style_and_marker(metric_col)

        ax.plot(
            time_centers,
            mean_metric[f"mean_{metric_col}"],
            marker=marker,
            linewidth=2,
            color=color,
            linestyle=ls,
            label=metric_col,
        )

    ax.axvline(x=0, color="green", linestyle="--", label="End of Trial")
    ax.axvline(x=-2, color="red", linestyle="--", label="Hit Time")

    ax.set_title("Mean trajectories per metric")
    ax.set_xlabel("Time relative to end of trial (s)")
    ax.set_ylabel("Mean value")
    if set_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return fig, ax


def plot_dlc_trial(
    df: pd.DataFrame,
    trial_id: int,
    pre_win: float = -5,
    post_win: float = 10,
    bin_size: float = 0.3,
    linear_metrics: Iterable[str] = LINEAR_METRICS,
):
    one_trial = df.loc[df["trial_id"] == trial_id].copy()
    if "time_bin" not in one_trial.columns:
        one_trial["time_bin"] = pd.cut(
            x=one_trial["time_relative_to_end"],
            bins=np.arange(pre_win, post_win + bin_size, bin_size).tolist()
        )
    one_fig, one_ax = plot_means_only_multi_by_prefix(
        one_trial,
        metric_cols=linear_metrics,
        mean_func=mean_func,
        observed=True,
    )
    one_ax.set_title(f"trajectories per metric for trial {trial_id} ")
    one_fig.tight_layout()
    one_fig.show()
    return one_fig, one_ax


def plot_mean_scores_pp(
    scores_df,
    trial_type="reward",
    *,
    range_start=-4,
    range_end=6,
    roll_window_s=1.5,
    roll_win_by_animal={"pv163": 1.5, "pv129": 1.5},
    mean_smooth_window_s=2.5,
    ax: Optional[plt.Axes] = None,
    is_legend: bool = True,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    x_label_shift = 2.0
    hit_time = -2.0
    end_time = 0.0
    figsize = (14, 7)

    df = scores_df.copy()
    df = df[df["is_reward_bug"] == (trial_type == "reward")]

    df["time_s"] = df.get("time_s", df["time_bin"])
    df = df[(df["time_s"] >= range_start) & (df["time_s"] <= range_end)].copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.groupby("trial_id").filter(lambda g: g["score"].abs().sum() > 0)

    df["score_bin"] = df["score"]
    df["time_bin_use"] = df["time_bin"]

    mean_by_bin = (
        df.groupby(["animal_id", "time_bin_use"], as_index=False)["score_bin"]
          .mean()
          .rename(columns={"time_bin_use": "time_bin"})
    )
    counts_by_bin = (
        df.groupby(["animal_id", "time_bin_use"], as_index=False)["trial_id"]
          .nunique()
          .rename(columns={"time_bin_use": "time_bin", "trial_id": "n_trials"})
    )
    mean_by_bin = mean_by_bin.merge(counts_by_bin, on=["animal_id", "time_bin"], how="left")

    def _norm_minmax(g, col):
        y = g[col]
        lo, hi = y.min(), y.max()
        delta = hi - lo
        denom = np.where(np.isfinite(delta) & (delta != 0), delta, 1.0)
        g[col] = (y - lo) / denom
        return g

    def _estimate_dt(series):
        t = np.sort(pd.to_numeric(series, errors="coerce").dropna().unique())
        return float(np.nanmedian(np.diff(t)))

    dt_by_animal = mean_by_bin.groupby("animal_id")["time_bin"].apply(_estimate_dt)

    def _smooth_group(g):
        animal_id = g["animal_id"].iat[0]
        window_s = roll_win_by_animal.get(animal_id, roll_window_s)
        dt = dt_by_animal.get(animal_id, np.nan)
        ratio = np.divide(window_s, dt, out=np.array([1.0]), where=np.isfinite(dt) & (dt > 0))
        win = max(1, int(round(float(ratio))))
        g = g.sort_values("time_bin").copy()
        g["score_roll"] = g["score_bin"].rolling(win, center=True, min_periods=1).mean()
        return g

    rolling_df = mean_by_bin.groupby("animal_id", group_keys=False).apply(_smooth_group)
    rolling_df = rolling_df.groupby("animal_id", group_keys=False).apply(_norm_minmax, col="score_roll")

    mean_group = rolling_df.groupby("time_bin", as_index=False)

    def _score_mean(g):
        values = g["score_roll"].to_numpy(dtype=float)
        has_values = np.isfinite(values).any()
        score_roll = float(np.nanmean(values)) if has_values else np.nan
        out = {
            "score_roll": score_roll,
            "n_animals": g["animal_id"].nunique(),
        }
        out["n_trials"] = g["n_trials"].sum()
        return pd.Series(out)

    mean_all = mean_group.apply(_score_mean).reset_index(drop=True)

    mean_all = mean_all.sort_values("time_bin").copy()
    tuniq = np.sort(pd.to_numeric(mean_all["time_bin"], errors="coerce").dropna().unique())
    dt_mean = float(np.nanmedian(np.diff(tuniq)))
    ratio = np.divide(float(mean_smooth_window_s), dt_mean, out=np.array([1.0]), where=np.isfinite(dt_mean) & (dt_mean > 0))
    win_mean = max(1, int(round(float(ratio))))
    mean_all["score_roll"] = mean_all["score_roll"].rolling(win_mean, center=True, min_periods=1).mean()

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    for animal_id, g in rolling_df.groupby("animal_id"):
        ax.plot(g["time_bin"], g["score_roll"], label=animal_id, alpha=0.2)

    ax.axvline(hit_time, color="red", linestyle="--", linewidth=1, label="_nolegend_")
    ax.axvline(end_time, color="green", linestyle="--", linewidth=1, label="_nolegend_")

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x + x_label_shift:g}"))
    ax.set_xlabel("Time around hit [s]")

    ax.plot(
        mean_all["time_bin"],
        mean_all["score_roll"],
        color="black",
        linewidth=3,
        label="Avg(all animals)",
        zorder=10,
    )

    ax.set_ylabel("Turn score")
    if is_legend:
        ax.legend(title="animal_id")
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    # ax.grid(alpha=0.3)

    return fig, rolling_df






#endregion




#region saving functions
################################################################################################
#---------------------------------------- saving functions ------------------------------------#
################################################################################################

# from matplotlib.figure import Figure
def save_fig(fig: plt.Figure, save_dir: str, save_prefix: str, use_datestamp: bool = True):
    from matplotlib import rcParams
    plt.style.use('default')
    rcParams['pdf.fonttype'] = 42  # Ensure fonts are embedded and editable
    rcParams['ps.fonttype'] = 42  # Ensure compatibility with vector outputs
    
    from pathlib import Path
    save_dir_path = Path(save_dir)
    
    if use_datestamp:
        import datetime
        dt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        save_dir_path = Path(f"{save_dir_path}_{dt}")
    save_dir_path.mkdir(parents=True, exist_ok=True)

    out_path = save_dir_path / f"{save_prefix}.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    
#endregion

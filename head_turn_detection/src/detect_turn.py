from __future__ import annotations

import ast
import copy
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import (Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, TypedDict, cast)
import matplotlib as mpl
from matplotlib import pyplot as plt, colors as mpl_colors, cm 
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score


# ============================================================
# Defaults (overridable via cfg/params)
# ============================================================

TRIAL_COL = "trial_id"
TIME_COL = "time_relative_to_end"  # hit=-2, end=0
META_COLS = ["window_label", "is_reward_bug", "animal_id", "day", "hit_bug_type"]
ANGLE_CONF_COL = "angle_conf"
ANGLE_CONF_MIN = 0.49

SIGNALS = [
    "nose_cam_x",
    "nose_cam_y",
    "left_ear_cam_x",
    "left_ear_cam_y",
    "right_ear_cam_x",
    "right_ear_cam_y",
    "nose_x",
    "nose_y",
    "left_ear_x",
    "left_ear_y",
    "right_ear_x",
    "right_ear_y",
    "head_angle_deg",
]
POSE_COLS = list(dict.fromkeys(SIGNALS + ["head_angle_n_shifted_deg"]))

BIN_SIZE = 0.05
SMOOTH_ROLL_BINS_CLASSIC = 1

SIDE_BASELINE_WIN = (-4.0, -2.0)
SIDE_POST_WIN = (-2.0, 0.0)
SIDE_SMOOTH_BINS = 5
SIDE_DERIV_WIN_BINS = 11
SIDE_RIGHT_IS_POSITIVE = True
SIDE_USE_DERIV = True

CLASSIC_BASELINE_SUBTRACT = False
CLASSIC_SCORE_BASELINE_WIN = (-4.0, -2.0)
CLASSIC_CLIP_AT_ZERO = True

ML_SMOOTH_ROLL_BINS = 10
ML_DIFF_K_BINS = 10
ML_LAGS = 5

TRAIN_TIME_RANGE = (-5.0, 15.0)
POS_HALF_WIDTH_S = 0.20
ENFORCE_POS_AFTER = -2.0
ML_BASELINE_SUB_WIN = (-2.0, 0.0)

PLOT_TIME_RANGE = (-5.0, 15.0)
EVENT_LINES = ((-2.0, "red"), (0.0, "green"))
PLOT_ROW_HEIGHT = 0.10
POSTPROC_APPLY_TO_HEATMAPS = True
SORT_ROWS = True
SORT_TIME = -2.0
SORT_MIN_PEAK_FOR_ORDER = 0.05
HEATMAP_XTICK_STEP_S = 1.0
HEATMAP_LAYOUT = "vertical"
HEATMAP_ALIGN_COLUMNS = True
HEATMAP_NORMALIZE_FOR_PLOT = "none"  # "none" | "rowmax" | "globalmax"
HEATMAP_FORCE_WHITE_ZERO = True
PLOT_NORMALIZE_PER_MODEL = True
PLOT_ZERO_BELOW = None

CLASSIC_PP_BASELINE_MODE = "window_median"
ML_PP_BASELINE_MODE = "none"
CLASSIC_PP_BASELINE_FIXED = 0.2
ML_PP_BASELINE_FIXED = 0.15
CLASSIC_PP_BASELINE_PCT = 70
ML_PP_BASELINE_PCT = 70
CLASSIC_PP_BASELINE_WIN = (-4.0, -2.0)
ML_PP_BASELINE_WIN = (-4.0, -2.0)
CLASSIC_PP_THRESH_MODE = "percentile"
CLASSIC_PP_THRESH_FIXED = 0.1
CLASSIC_PP_THRESH_PCT = 60
ML_PP_THRESH_MODE = "fixed"
ML_PP_THRESH_FIXED = 0.61
ML_PP_THRESH_PCT = 60
POSTPROC_SMOOTH_BINS = 1

STRENGTH_W: Dict[str, float] = {
    "weak": 0.5,
    "noticable": 0.8,
    "moderate": 1.0,
    "nice": 1.2,
    "strong": 1.5,
    "full": 1.8,
}

SIG_WEIGHTS_LEFT = {
    "conv__nose_cam_x": 1.0,
    "conv__nose_cam_y": -20.0,
    "conv__left_ear_cam_x": -0.5,
    "conv__left_ear_cam_y": 0.5,
    "conv__right_ear_cam_x": 0.5,
    "feat__y_good": 0.8,
    "feat__y_bad": -0.8,
    "feat__x_good": 0.6,
    "feat__x_bad": -0.6,
    "feat__head_inc": 0.3,
    "feat__head_dec": -0.3,
    "feat__jitter": -1.0,
    "feat__ear_sep_bad": -1.0,
}
SIG_WEIGHTS_RIGHT = {
    "conv__nose_cam_x": -1.0,
    "conv__nose_cam_y": -20.0,
    "conv__left_ear_cam_x": 0.5,
    "conv__left_ear_cam_y": -0.5,
    "conv__right_ear_cam_x": -0.5,
    "feat__y_good": 0.8,
    "feat__y_bad": -0.8,
    "feat__x_good": 0.6,
    "feat__x_bad": -0.6,
    "feat__head_inc": 0.3,
    "feat__head_dec": -0.3,
    "feat__jitter": -1.0,
    "feat__ear_sep_bad": -1.0,
}
KERNEL = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=float)

ML_TURN_MODEL = dict(penalty="elasticnet", solver="saga", l1_ratio=0.6, C=0.6, max_iter=8000, random_state=0) #8000
ML_USE_CLASS_BALANCED = True


def _default_cfg() -> Dict[str, Any]:
    return {
        "TRIAL_COL": TRIAL_COL,
        "TIME_COL": TIME_COL,
        "META_COLS": tuple(META_COLS),
        "SIGNALS": tuple(SIGNALS),
        "POSE_COLS": tuple(POSE_COLS),
        "ANGLE_CONF_COL": ANGLE_CONF_COL,
        "ANGLE_CONF_MIN": ANGLE_CONF_MIN,
        "BIN_SIZE": BIN_SIZE,
        "SMOOTH_ROLL_BINS_CLASSIC": SMOOTH_ROLL_BINS_CLASSIC,
        "SIDE_BASELINE_WIN": tuple(SIDE_BASELINE_WIN),
        "SIDE_POST_WIN": tuple(SIDE_POST_WIN),
        "SIDE_SMOOTH_BINS": SIDE_SMOOTH_BINS,
        "SIDE_DERIV_WIN_BINS": SIDE_DERIV_WIN_BINS,
        "SIDE_RIGHT_IS_POSITIVE": SIDE_RIGHT_IS_POSITIVE,
        "SIDE_USE_DERIV": SIDE_USE_DERIV,
        "CLASSIC_BASELINE_SUBTRACT": CLASSIC_BASELINE_SUBTRACT,
        "CLASSIC_SCORE_BASELINE_WIN": tuple(CLASSIC_SCORE_BASELINE_WIN),
        "CLASSIC_CLIP_AT_ZERO": CLASSIC_CLIP_AT_ZERO,
        "ML_SMOOTH_ROLL_BINS": ML_SMOOTH_ROLL_BINS,
        "ML_DIFF_K_BINS": ML_DIFF_K_BINS,
        "ML_LAGS": ML_LAGS,
        "TRAIN_TIME_RANGE": tuple(TRAIN_TIME_RANGE),
        "POS_HALF_WIDTH_S": POS_HALF_WIDTH_S,
        "ENFORCE_POS_AFTER": ENFORCE_POS_AFTER,
        "ML_BASELINE_SUB_WIN": tuple(ML_BASELINE_SUB_WIN),
        "PLOT_TIME_RANGE": tuple(PLOT_TIME_RANGE),
        "EVENT_LINES": tuple(EVENT_LINES),
        "PLOT_ROW_HEIGHT": PLOT_ROW_HEIGHT,
        "POSTPROC_APPLY_TO_HEATMAPS": POSTPROC_APPLY_TO_HEATMAPS,
        "SORT_ROWS": SORT_ROWS,
        "SORT_TIME": SORT_TIME,
        "SORT_MIN_PEAK_FOR_ORDER": SORT_MIN_PEAK_FOR_ORDER,
        "HEATMAP_XTICK_STEP_S": HEATMAP_XTICK_STEP_S,
        "HEATMAP_LAYOUT": HEATMAP_LAYOUT,
        "HEATMAP_ALIGN_COLUMNS": HEATMAP_ALIGN_COLUMNS,
        "HEATMAP_NORMALIZE_FOR_PLOT": HEATMAP_NORMALIZE_FOR_PLOT,
        "HEATMAP_FORCE_WHITE_ZERO": HEATMAP_FORCE_WHITE_ZERO,
        "PLOT_NORMALIZE_PER_MODEL": PLOT_NORMALIZE_PER_MODEL,
        "PLOT_ZERO_BELOW": PLOT_ZERO_BELOW,
        "CLASSIC_PP_BASELINE_MODE": CLASSIC_PP_BASELINE_MODE,
        "ML_PP_BASELINE_MODE": ML_PP_BASELINE_MODE,
        "CLASSIC_PP_BASELINE_FIXED": CLASSIC_PP_BASELINE_FIXED,
        "ML_PP_BASELINE_FIXED": ML_PP_BASELINE_FIXED,
        "CLASSIC_PP_BASELINE_PCT": CLASSIC_PP_BASELINE_PCT,
        "ML_PP_BASELINE_PCT": ML_PP_BASELINE_PCT,
        "CLASSIC_PP_BASELINE_WIN": tuple(CLASSIC_PP_BASELINE_WIN),
        "ML_PP_BASELINE_WIN": tuple(ML_PP_BASELINE_WIN),
        "CLASSIC_PP_THRESH_MODE": CLASSIC_PP_THRESH_MODE,
        "CLASSIC_PP_THRESH_FIXED": CLASSIC_PP_THRESH_FIXED,
        "CLASSIC_PP_THRESH_PCT": CLASSIC_PP_THRESH_PCT,
        "ML_PP_THRESH_MODE": ML_PP_THRESH_MODE,
        "ML_PP_THRESH_FIXED": ML_PP_THRESH_FIXED,
        "ML_PP_THRESH_PCT": ML_PP_THRESH_PCT,
        "POSTPROC_SMOOTH_BINS": POSTPROC_SMOOTH_BINS,
        "STRENGTH_W": dict(STRENGTH_W),
        "SIG_WEIGHTS_LEFT": dict(SIG_WEIGHTS_LEFT),
        "SIG_WEIGHTS_RIGHT": dict(SIG_WEIGHTS_RIGHT),
        "KERNEL": np.array(KERNEL, float),
        "ML_TURN_MODEL": dict(ML_TURN_MODEL),
        "ML_USE_CLASS_BALANCED": ML_USE_CLASS_BALANCED,
    }


def resolve_cfg(overrides: Optional[Dict[str, Any] | TurnConfig] = None) -> Dict[str, Any]:
    cfg = _default_cfg()
    if overrides is None:
        return cfg

    if isinstance(overrides, TurnConfig):
        overrides = overrides.to_dict()
    elif not isinstance(overrides, dict):
        to_dict = getattr(overrides, "to_dict", None)
        if callable(to_dict):
            overrides = to_dict()  # type: ignore[assignment]
        else:
            raise TypeError("overrides must be a dict or TurnConfig")

    if not isinstance(overrides, dict):
        raise TypeError("overrides must be a dict or TurnConfig")

    for k, v in overrides.items():
        cfg[k.upper()] = v
    return cfg


def _cfg(cfg: Optional[Dict[str, Any]], key: str) -> Any:
    base = cfg if cfg is not None else _default_cfg()
    if cfg is None and key in globals():
        return copy.deepcopy(getattr(__import__(__name__), key))
    val = base.get(key, _default_cfg().get(key))
    return copy.deepcopy(val)


@dataclass
class TurnConfig:
    """
    Typed config helper; fill only what you want to override and call .to_dict().
    Unknown keys can be passed via extras.
    """

    # High-signal overrides
    bin_size: float | None = None
    plot_time_range: tuple[float, float] | None = None
    plot_row_height: float | None = None
    heatmap_layout: str | None = None
    heatmap_normalize_for_plot: str | None = None
    sort_rows: bool | None = None
    event_lines: tuple[tuple[float, str], ...] | None = None
    heatmap_xtick_step_s: float | None = None
    plot_zero_below: float | None = None

    # Columns / signals
    trial_col: str | None = None
    time_col: str | None = None
    signals: list[str] | None = None

    # Classic
    smooth_roll_bins_classic: int | None = None  # rolling median bins for classic features
    classic_pp_baseline_mode: str | None = None  # baseline mode for classic postproc ("window_median", "none", etc.)
    classic_pp_baseline_fixed: float | None = None  # fixed baseline value if mode uses it
    classic_pp_baseline_pct: float | None = None  # percentile baseline if mode uses pct
    classic_pp_baseline_win: tuple[float, float] | None = None  # time window for baseline calc (classic)
    classic_pp_thresh_mode: str | None = None  # thresholding mode for classic ("percentile", "fixed", etc.)
    classic_pp_thresh_fixed: float | None = None  # fixed threshold value (classic)
    classic_pp_thresh_pct: float | None = None  # percentile threshold (classic)

    # ML
    ml_smooth_roll_bins: int | None = None  # rolling median bins for ML features
    ml_diff_k_bins: int | None = None  # derivative window (bins) for ML
    ml_lags: int | None = None  # number of lagged features to add
    ml_pp_baseline_mode: str | None = None  # baseline mode for ML postproc
    ml_pp_baseline_fixed: float | None = None  # fixed baseline value (ML)
    ml_pp_baseline_pct: float | None = None  # percentile baseline (ML)
    ml_pp_baseline_win: tuple[float, float] | None = None  # baseline window (ML)
    ml_pp_thresh_mode: str | None = None  # threshold mode for ML
    ml_pp_thresh_fixed: float | None = None  # fixed threshold (ML)
    ml_pp_thresh_pct: float | None = None  # percentile threshold (ML)

    # Postproc
    postproc_apply_to_heatmaps: bool | None = None  # whether to apply postproc before plotting
    postproc_smooth_bins: int | None = None  # smoothing bins after postproc

    # Any extra keys to merge directly
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Merge provided overrides onto library defaults."""
        cfg = _default_cfg()
        mapping = {
            "bin_size": "BIN_SIZE",
            "plot_time_range": "PLOT_TIME_RANGE",
            "plot_row_height": "PLOT_ROW_HEIGHT",
            "heatmap_layout": "HEATMAP_LAYOUT",
            "heatmap_normalize_for_plot": "HEATMAP_NORMALIZE_FOR_PLOT",
            "sort_rows": "SORT_ROWS",
            "event_lines": "EVENT_LINES",
            "heatmap_xtick_step_s": "HEATMAP_XTICK_STEP_S",
            "plot_zero_below": "PLOT_ZERO_BELOW",
            "trial_col": "TRIAL_COL",
            "time_col": "TIME_COL",
            "signals": "SIGNALS",
            "smooth_roll_bins_classic": "SMOOTH_ROLL_BINS_CLASSIC",
            "classic_pp_baseline_mode": "CLASSIC_PP_BASELINE_MODE",
            "classic_pp_baseline_fixed": "CLASSIC_PP_BASELINE_FIXED",
            "classic_pp_baseline_pct": "CLASSIC_PP_BASELINE_PCT",
            "classic_pp_baseline_win": "CLASSIC_PP_BASELINE_WIN",
            "classic_pp_thresh_mode": "CLASSIC_PP_THRESH_MODE",
            "classic_pp_thresh_fixed": "CLASSIC_PP_THRESH_FIXED",
            "classic_pp_thresh_pct": "CLASSIC_PP_THRESH_PCT",
            "ml_smooth_roll_bins": "ML_SMOOTH_ROLL_BINS",
            "ml_diff_k_bins": "ML_DIFF_K_BINS",
            "ml_lags": "ML_LAGS",
            "ml_pp_baseline_mode": "ML_PP_BASELINE_MODE",
            "ml_pp_baseline_fixed": "ML_PP_BASELINE_FIXED",
            "ml_pp_baseline_pct": "ML_PP_BASELINE_PCT",
            "ml_pp_baseline_win": "ML_PP_BASELINE_WIN",
            "ml_pp_thresh_mode": "ML_PP_THRESH_MODE",
            "ml_pp_thresh_fixed": "ML_PP_THRESH_FIXED",
            "ml_pp_thresh_pct": "ML_PP_THRESH_PCT",
            "postproc_apply_to_heatmaps": "POSTPROC_APPLY_TO_HEATMAPS",
            "postproc_smooth_bins": "POSTPROC_SMOOTH_BINS",
        }
        for attr, key in mapping.items():
            val = getattr(self, attr)
            if val is not None:
                cfg[key] = val
        if self.extras:
            cfg.update(self.extras)
        return cfg

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TurnConfig":
        """Initialize from a partial dict; unknown keys land in extras."""
        data = {k.upper(): v for k, v in (data or {}).items()}
        reverse = {
            "BIN_SIZE": "bin_size",
            "PLOT_TIME_RANGE": "plot_time_range",
            "PLOT_ROW_HEIGHT": "plot_row_height",
            "HEATMAP_LAYOUT": "heatmap_layout",
            "HEATMAP_NORMALIZE_FOR_PLOT": "heatmap_normalize_for_plot",
            "SORT_ROWS": "sort_rows",
            "EVENT_LINES": "event_lines",
            "HEATMAP_XTICK_STEP_S": "heatmap_xtick_step_s",
            "PLOT_ZERO_BELOW": "plot_zero_below",
            "TRIAL_COL": "trial_col",
            "TIME_COL": "time_col",
            "SIGNALS": "signals",
            "SMOOTH_ROLL_BINS_CLASSIC": "smooth_roll_bins_classic",
            "CLASSIC_PP_BASELINE_MODE": "classic_pp_baseline_mode",
            "CLASSIC_PP_BASELINE_FIXED": "classic_pp_baseline_fixed",
            "CLASSIC_PP_BASELINE_PCT": "classic_pp_baseline_pct",
            "CLASSIC_PP_BASELINE_WIN": "classic_pp_baseline_win",
            "CLASSIC_PP_THRESH_MODE": "classic_pp_thresh_mode",
            "CLASSIC_PP_THRESH_FIXED": "classic_pp_thresh_fixed",
            "CLASSIC_PP_THRESH_PCT": "classic_pp_thresh_pct",
            "ML_SMOOTH_ROLL_BINS": "ml_smooth_roll_bins",
            "ML_DIFF_K_BINS": "ml_diff_k_bins",
            "ML_LAGS": "ml_lags",
            "ML_PP_BASELINE_MODE": "ml_pp_baseline_mode",
            "ML_PP_BASELINE_FIXED": "ml_pp_baseline_fixed",
            "ML_PP_BASELINE_PCT": "ml_pp_baseline_pct",
            "ML_PP_BASELINE_WIN": "ml_pp_baseline_win",
            "ML_PP_THRESH_MODE": "ml_pp_thresh_mode",
            "ML_PP_THRESH_FIXED": "ml_pp_thresh_fixed",
            "ML_PP_THRESH_PCT": "ml_pp_thresh_pct",
            "POSTPROC_APPLY_TO_HEATMAPS": "postproc_apply_to_heatmaps",
            "POSTPROC_SMOOTH_BINS": "postproc_smooth_bins",
        }
        known = {}
        extras = {}
        for k, v in data.items():
            attr = reverse.get(k)
            if attr:
                known[attr] = v
            else:
                extras[k] = v
        known["extras"] = extras
        return cls(**known)


def resolve_turn_config(cfg: TurnConfig | Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Convenience: accept TurnConfig, dict, or None and return a full cfg dict with defaults filled.
    """
    if cfg is None:
        return _default_cfg()
    if isinstance(cfg, TurnConfig):
        return cfg.to_dict()
    return resolve_cfg(cfg)


# ============================================================
# Label helpers
# ============================================================


@dataclass
class StrengthScale:
    weights: Dict[str, float] = field(default_factory=lambda: dict(STRENGTH_W))

    def resolve(self, strength: Optional[str], default: float = 1.0) -> float:
        if strength is None:
            return float(default)
        key = str(strength).strip().lower()
        return float(self.weights.get(key, default))


@dataclass
class TurnLabel:
    trial_id: float | int
    trial_side: Optional[str] = None
    turn_times_expect: Optional[Sequence[float]] = None
    strength: Optional[str] = None
    w: Optional[float] = None

    def to_dict(self, strength_scale: StrengthScale, trial_col: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            trial_col: self.trial_id,
            "trial_side": self.trial_side,
            "turn_times_expect": list(self.turn_times_expect or []),
        }
        weight = self.w if self.w is not None else strength_scale.resolve(self.strength, default=1.0)
        out["w"] = float(weight)
        return out


def make_labels_df(
    labels_list: Sequence[TurnLabel | Dict[str, Any]],
    *,
    cfg: Optional[Dict[str, Any]] = None,
    strength_scale: Optional[StrengthScale] = None,
) -> pd.DataFrame:
    trial_col = _cfg(cfg, "TRIAL_COL")
    strength_scale = strength_scale or StrengthScale(weights=_cfg(cfg, "STRENGTH_W"))

    if not labels_list:
        return pd.DataFrame(columns=[trial_col, "trial_side", "y_side", "w", "turn_times_expect"])

    normalized = []
    for item in labels_list:
        if isinstance(item, TurnLabel):
            normalized.append(item.to_dict(strength_scale, trial_col))
        else:
            normalized.append(dict(item))

    df = pd.DataFrame(normalized).copy()
    df[trial_col] = pd.to_numeric(df[trial_col], errors="coerce")

    if "trial_side" in df.columns:
        df["trial_side"] = df["trial_side"].astype(str)
        df["y_side"] = (df["trial_side"].str.lower() == "right").astype(int)
    else:
        df["trial_side"] = np.nan
        df["y_side"] = np.nan

    w_direct = pd.to_numeric(df["w"], errors="coerce") if "w" in df.columns else pd.Series(np.nan, index=df.index)
    w_strength = df["strength"].map(strength_scale.weights).astype(float) if "strength" in df.columns else pd.Series(
        np.nan, index=df.index
    )
    df["w"] = w_direct.fillna(w_strength).fillna(1.0).astype(float)

    if "turn_times_expect" in df.columns:
        df["turn_times_expect"] = df["turn_times_expect"].apply(_parse_turn_times)
    elif "turn_time_expect" in df.columns:
        df["turn_times_expect"] = df["turn_time_expect"].apply(_parse_turn_times)
    else:
        df["turn_times_expect"] = [[] for _ in range(len(df))]

    return df[[trial_col, "trial_side", "y_side", "w", "turn_times_expect"]].copy()


# ============================================================
# Basic transforms
# ============================================================


def _postprocess_series_per_trial(
    g: pd.DataFrame,
    *,
    value_col: str,
    time_col: str = "time_bin",
    baseline_mode: str = "none",
    baseline_fixed: float = 0.0,
    baseline_pct: float = 50.0,
    baseline_win: Tuple[float, float] = (-4.0, -2.0),
    thresh_mode: str = "none",
    thresh_fixed: float = 0.0,
    thresh_pct: float = 50.0,
    smooth_bins: int = 1,
    out_col: str = "turn_score_pp",
) -> pd.DataFrame:
    g = g.sort_values(time_col).copy()
    if value_col not in g.columns:
        g[out_col] = 0.0
        return g

    t = pd.to_numeric(g[time_col], errors="coerce").to_numpy()
    x = pd.to_numeric(g[value_col], errors="coerce").to_numpy()
    finite = np.isfinite(x)
    if not finite.any():
        g[out_col] = 0.0
        return g

    if baseline_mode == "fixed":
        base = float(baseline_fixed)
    elif baseline_mode == "percentile":
        base = float(np.nanpercentile(x[finite], float(baseline_pct)))
    elif baseline_mode == "window_median":
        m = np.isfinite(t) & np.isfinite(x) & (t >= baseline_win[0]) & (t <= baseline_win[1])
        base = float(np.nanmedian(x[m])) if m.any() else 0.0
    else:
        base = 0.0
    x = np.maximum(x - base, 0.0)

    finite2 = np.isfinite(x)
    if thresh_mode == "fixed":
        thr = float(thresh_fixed)
    elif thresh_mode == "percentile":
        thr = float(np.nanpercentile(x[finite2], float(thresh_pct))) if finite2.any() else 0.0
    else:
        thr = None
    if thr is not None and np.isfinite(thr):
        x = np.where(x >= thr, x, 0.0)

    sb = int(max(1, smooth_bins))
    if sb > 1:
        x = (
            pd.Series(x)
            .rolling(sb, center=True, min_periods=max(1, sb // 2))
            .median()
            .to_numpy()
        )

    g[out_col] = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return g


def _sanitize_window_stack(df: pd.DataFrame, *, cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("window_stack must be a pandas DataFrame")

    trial_col = _cfg(cfg, "TRIAL_COL")
    time_col = _cfg(cfg, "TIME_COL")

    work = df.copy()
    if work.columns.duplicated().any():
        work = work.loc[:, ~work.columns.duplicated()].copy()
    v = work.get(trial_col, None)
    if isinstance(v, pd.DataFrame):
        work[trial_col] = v.iloc[:, 0]
    work[trial_col] = pd.to_numeric(work[trial_col], errors="coerce")
    if time_col not in work.columns:
        raise ValueError(f"window_stack missing {time_col!r}")
    work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
    work = _filter_low_angle_conf(work, cfg=cfg)
    return work


def _filter_low_angle_conf(df: pd.DataFrame, *, cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    conf_col = _cfg(cfg, "ANGLE_CONF_COL")
    if not conf_col or conf_col not in df.columns:
        return df
    try:
        min_conf = float(_cfg(cfg, "ANGLE_CONF_MIN"))
    except Exception:
        min_conf = 0.49

    conf = pd.to_numeric(df[conf_col], errors="coerce")
    keep = conf.isna() | (conf >= min_conf)
    if not keep.any():
        return df.iloc[0:0].copy()
    return df.loc[keep].copy()


def _rolling_median_by_trial(
    df: pd.DataFrame,
    cols: List[str],
    win_bins: int,
    *,
    time_col: str = "time_bin",
    trial_col: str = TRIAL_COL,
) -> pd.DataFrame:
    if win_bins <= 1:
        return df.copy()
    out = []
    for tid, g in df.groupby(trial_col, observed=False):
        g = g.sort_values(time_col).copy()
        for c in cols:
            if c in g.columns:
                g[c] = (
                    pd.to_numeric(g[c], errors="coerce")
                    .rolling(win_bins, center=True, min_periods=max(1, win_bins // 2))
                    .median()
                )
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0].copy()


def bin_signals(
    df: pd.DataFrame,
    *,
    bin_size: float,
    cols: List[str],
    trial_col: str,
    time_col: str,
) -> pd.DataFrame:
    work = df.copy()
    work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
    work["time_bin"] = np.floor(work[time_col] / bin_size) * bin_size + (bin_size / 2.0)

    keep = [trial_col, "time_bin"] + [c for c in cols if c in work.columns]
    work = work[keep].copy()
    for c in keep:
        if c not in (trial_col, "time_bin"):
            work[c] = pd.to_numeric(work[c], errors="coerce")

    b = (
        work.groupby([trial_col, "time_bin"], observed=False)
        .mean(numeric_only=True)
        .reset_index()
        .sort_values([trial_col, "time_bin"])
        .reset_index(drop=True)
    )
    b[time_col] = b["time_bin"]
    return b


# ============================================================
# Numeric helpers
# ============================================================


def wrap_deg180(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, float)
    return (d + 180.0) % 360.0 - 180.0


def conv1d(v: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    k = np.asarray(kernel, float)
    if v.size < k.size:
        return np.full_like(v, np.nan, dtype=float)
    win = sliding_window_view(v, window_shape=k.size)
    conv = (win * k).sum(axis=-1)
    pad = np.full(k.size - 1, np.nan)
    return np.concatenate([pad, conv])


def conv_derivative(v: pd.Series | np.ndarray, *, win_bins: int, dt: float) -> np.ndarray:
    x = pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(dtype=float)
    w = int(max(3, win_bins))
    if w % 2 == 0:
        w += 1
    m = w // 2
    idx = np.arange(-m, m + 1, dtype=float)
    denom = np.sum(idx ** 2)
    if denom <= 0 or not np.isfinite(denom):
        return np.full_like(x, np.nan, dtype=float)
    kernel = (idx / denom) / float(dt)
    if x.size < w:
        return np.full_like(x, np.nan, dtype=float)
    win = sliding_window_view(x, window_shape=w)
    bad = ~np.isfinite(win).all(axis=1)
    y = (win * kernel).sum(axis=1)
    y[bad] = np.nan
    out = np.full_like(x, np.nan, dtype=float)
    out[m : m + y.size] = y
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out

def _shift_np(x: np.ndarray, lag: int) -> np.ndarray:
    x = np.asarray(x, float)
    if lag <= 0:
        return x.copy()
    out = np.empty_like(x, dtype=float)
    out[:lag] = np.nan
    out[lag:] = x[:-lag]
    return out

# ============================================================
# Mid-ear helpers (derived from left/right ears)
# ============================================================


def _add_mid_ear_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df
    pairs = [
        ("left_ear_cam_x", "right_ear_cam_x", "mid_ear_cam_x"),
        ("left_ear_cam_y", "right_ear_cam_y", "mid_ear_cam_y"),
        ("left_ear_x", "right_ear_x", "mid_ear_x"),
        ("left_ear_y", "right_ear_y", "mid_ear_y"),
    ]
    for left, right, mid in pairs:
        if mid in out.columns:
            continue
        if left in out.columns and right in out.columns:
            l = pd.to_numeric(out[left], errors="coerce")
            r = pd.to_numeric(out[right], errors="coerce")
            out[mid] = (l + r) / 2.0
    return out

# ============================================================
# Side model (simple heuristic/logistic)
# ============================================================


def _pick_pair_cols(g: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if "nose_cam_x" in g.columns and "mid_ear_cam_x" in g.columns:
        return "nose_cam_x", "mid_ear_cam_x"
    if "nose_x" in g.columns and "mid_ear_x" in g.columns:
        return "nose_x", "mid_ear_x"
    if "nose_cam_x" in g.columns and "left_ear_cam_x" in g.columns:
        return "nose_cam_x", "left_ear_cam_x"
    if "nose_x" in g.columns and "left_ear_x" in g.columns:
        return "nose_x", "left_ear_x"
    if "nose_cam_x" in g.columns:
        return "nose_cam_x", None
    if "nose_x" in g.columns:
        return "nose_x", None
    return None, None


class SideFeatConfig(TypedDict):
    bin_size: float
    baseline_win: Tuple[float, float]
    post_win: Tuple[float, float]
    smooth_bins: int
    deriv_win_bins: int
    use_derivative: bool
    right_is_positive: bool
    time_col: str


def _side_feature_simple(
    window_stack: pd.DataFrame,
    *,
    bin_size: float,
    baseline_win: Tuple[float, float],
    post_win: Tuple[float, float],
    smooth_bins: int,
    deriv_win_bins: int,
    use_derivative: bool,
    right_is_positive: bool,
    trial_col: str,
    time_col: str,
) -> pd.DataFrame:
    need = [
        c
        for c in [
            "nose_cam_x",
            "left_ear_cam_x",
            "right_ear_cam_x",
            "nose_x",
            "left_ear_x",
            "right_ear_x",
        ]
        if c in window_stack.columns
    ]
    if not need:
        return pd.DataFrame(columns=[trial_col, "side_feat"])

    b = bin_signals(window_stack, bin_size=float(bin_size), cols=need, trial_col=trial_col, time_col=time_col)
    if smooth_bins and int(smooth_bins) > 1:
        b = _rolling_median_by_trial(b, need, int(smooth_bins), time_col="time_bin", trial_col=trial_col)
    b = _add_mid_ear_cols(b)
    dt = float(bin_size)

    rows = []
    for tid, g in b.groupby(trial_col, observed=False):
        g = g.sort_values("time_bin").copy()
        nose_col, ear_col = _pick_pair_cols(g)
        if nose_col is None:
            rows.append({trial_col: tid, "side_feat": np.nan})
            continue

        nose = pd.to_numeric(g[nose_col], errors="coerce").to_numpy(dtype=float)
        if ear_col is not None and ear_col in g.columns:
            ear = pd.to_numeric(g[ear_col], errors="coerce").to_numpy(dtype=float)
            sig = nose - ear
        else:
            sig = nose

        t = pd.to_numeric(g["time_bin"], errors="coerce").to_numpy(dtype=float)
        base = np.isfinite(t) & (t >= baseline_win[0]) & (t <= baseline_win[1])
        post = np.isfinite(t) & (t >= post_win[0]) & (t <= post_win[1])
        if not base.any() or not post.any():
            rows.append({trial_col: tid, "side_feat": np.nan})
            continue

        if use_derivative:
            d1 = conv_derivative(sig, win_bins=int(deriv_win_bins), dt=dt)
            base0 = np.nanmedian(d1[base]) if np.isfinite(d1[base]).any() else 0.0
            feat = np.nanmean(d1[post] - base0) if np.isfinite(d1[post]).any() else np.nan
        else:
            base_med = np.nanmedian(sig[base]) if np.isfinite(sig[base]).any() else np.nan
            post_med = np.nanmedian(sig[post]) if np.isfinite(sig[post]).any() else np.nan
            feat = (post_med - base_med) if (np.isfinite(base_med) and np.isfinite(post_med)) else np.nan

        if not right_is_positive and np.isfinite(feat):
            feat = -feat
        rows.append({trial_col: tid, "side_feat": float(feat) if np.isfinite(feat) else np.nan})

    return pd.DataFrame(rows)


class SimpleSideModel:
    def __init__(self, *, alpha: float = 4.0, feat_cfg: dict, trial_col: str):
        self.alpha = float(alpha)
        self.feat_cfg = dict(feat_cfg)
        self.trial_col = trial_col

    def predict_from_window_stack(self, window_stack: pd.DataFrame, *, time_col: Optional[str] = None, **_) -> pd.DataFrame:
        cfg = dict(self.feat_cfg)
        cfg["trial_col"] = self.trial_col
        cfg["time_col"] = cfg.get("time_col", time_col)
        if cfg["time_col"] is None:
            cfg["time_col"] = _cfg(None, "TIME_COL")
        X = _side_feature_simple(window_stack, **cfg)
        if X.empty:
            return pd.DataFrame(columns=[self.trial_col, "p_right", "trial_side", "side_feat"])

        z = pd.to_numeric(X["side_feat"], errors="coerce").to_numpy(dtype=float)
        z = np.nan_to_num(z, nan=0.0)
        p_right = 1.0 / (1.0 + np.exp(-self.alpha * z))

        out = X[[self.trial_col]].copy()
        out["side_feat"] = X["side_feat"]
        out["p_right"] = p_right
        out["trial_side"] = np.where(out["p_right"] >= 0.5, "right", "left")
        return out


class SignSideModel:
    def __init__(self, *, feat_cfg: dict, trial_col: str):
        self.feat_cfg = dict(feat_cfg)
        self.trial_col = trial_col

    def predict_from_window_stack(self, window_stack: pd.DataFrame, *, time_col: Optional[str] = None, **_) -> pd.DataFrame:
        cfg = dict(self.feat_cfg)
        cfg["trial_col"] = self.trial_col
        cfg["time_col"] = cfg.get("time_col", time_col)
        if cfg["time_col"] is None:
            cfg["time_col"] = _cfg(None, "TIME_COL")
        X = _side_feature_simple(window_stack, **cfg)
        if X.empty:
            return pd.DataFrame(columns=[self.trial_col, "p_right", "trial_side", "side_feat"])

        z = pd.to_numeric(X["side_feat"], errors="coerce").to_numpy(dtype=float)
        p_right = np.where(z > 0, 1.0, np.where(z < 0, 0.0, 0.5))

        out = X[[self.trial_col]].copy()
        out["side_feat"] = X["side_feat"]
        out["p_right"] = p_right
        out["trial_side"] = np.where(out["p_right"] >= 0.5, "right", "left")
        return out


class TrainedSideModel:
    def __init__(self, pipe, feat_cfg, trial_col, time_col):
        self.pipe = pipe
        self.feat_cfg = feat_cfg
        self.trial_col = trial_col
        self.time_col = time_col

    def predict_from_window_stack(self, window_stack, *, time_col: Optional[str] = None, **_):
        cfg = dict(self.feat_cfg)
        cfg["trial_col"] = self.trial_col
        cfg["time_col"] = cfg.get("time_col", time_col or self.time_col)
        if cfg["time_col"] is None:
            cfg["time_col"] = _cfg(None, "TIME_COL")
        X = _side_feature_simple(window_stack, **cfg)
        if X.empty:
            return pd.DataFrame(
                columns=[self.trial_col, "p_right", "trial_side", "side_feat"]
            )
        p = self.pipe.predict_proba(X[["side_feat"]])[:, 1]
        out = X[[self.trial_col]].copy()
        out["side_feat"] = X["side_feat"]
        out["p_right"] = p
        out["trial_side"] = np.where(p >= 0.5, "right", "left")
        return out


class DummySideModel:
    def __init__(self, *, trial_col: str, trial_side: str = "right"):
        self.trial_col = trial_col
        self.trial_side = str(trial_side).lower()

    def predict_from_window_stack(self, window_stack, *, time_col: Optional[str] = None, **_):
        if self.trial_col not in window_stack.columns:
            return pd.DataFrame(columns=[self.trial_col, "p_right", "trial_side", "side_feat"])
        trial_ids = (
            pd.to_numeric(window_stack[self.trial_col], errors="coerce")
            .dropna()
            .unique()
            .tolist()
        )
        if not trial_ids:
            return pd.DataFrame(columns=[self.trial_col, "p_right", "trial_side", "side_feat"])
        out = pd.DataFrame({self.trial_col: sorted(trial_ids)})
        out["side_feat"] = np.nan
        out["p_right"] = 0.5
        out["trial_side"] = self.trial_side
        return out


class TrialClassificationSideModel:
    def __init__(
        self,
        *,
        trial_col: str,
        time_col: str,
        col: Optional[str] = None,
        time_window: Tuple[float, float] = (-2.0, -0.5),
        roll_window: int = 5,
        x_thresh: float = 8.0,
        min_samples_beyond: int = 3,
        min_samples_window: int = 5,
    ):
        self.trial_col = trial_col
        self.time_col = time_col
        self.col = col
        self.time_window = time_window
        self.roll_window = int(roll_window)
        self.x_thresh = float(x_thresh)
        self.min_samples_beyond = int(min_samples_beyond)
        self.min_samples_window = int(min_samples_window)

    def _pick_col(self, window_stack: pd.DataFrame) -> Optional[str]:
        if self.col and self.col in window_stack.columns:
            return self.col
        for cand in ["nose_cam_x", "nose_x"]:
            if cand in window_stack.columns:
                return cand
        return None

        

    def predict_from_window_stack(self, window_stack: pd.DataFrame, *, time_col: Optional[str] = None, **_):
        if self.trial_col not in window_stack.columns:
            return pd.DataFrame(columns=[self.trial_col, "p_right", "trial_side", "side_feat"])
        col = self._pick_col(window_stack)
        if col is None:
            return pd.DataFrame(columns=[self.trial_col, "p_right", "trial_side", "side_feat"])
        use_time_col = time_col or self.time_col

        from trial_classification import classify_trials_mean_displacement

        summary = classify_trials_mean_displacement(
            window_stack,
            col=col,
            time_col=use_time_col,
            return_col_name="side_label",
            time_window=self.time_window,
            roll_window=self.roll_window,
            x_thresh=self.x_thresh,
            min_samples_beyond=self.min_samples_beyond,
            min_samples_window=self.min_samples_window,
        )
        if summary.empty:
            return pd.DataFrame(columns=[self.trial_col, "p_right", "trial_side", "side_feat"])
        out = summary.rename(columns={self.trial_col: self.trial_col}).copy()
        out["trial_side"] = out["side_label"].astype(str).str.lower()
        out["trial_side"] = out["trial_side"].where(out["trial_side"].isin(["left", "right"]), "unknown")
        out["p_right"] = np.where(out["trial_side"] == "right", 1.0, np.where(out["trial_side"] == "left", 0.0, 0.5))
        x_mean = out["x_mean"] if "x_mean" in out.columns else pd.Series(np.nan, index=out.index)
        out["side_feat"] = pd.to_numeric(x_mean, errors="coerce")
        return out[[self.trial_col, "p_right", "trial_side", "side_feat"]].copy()


def fit_side_model_simple(
    window_stack: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    bin_size: float,
    cfg: Optional[Dict[str, Any]] = None,
) -> SimpleSideModel:
    trial_col = _cfg(cfg, "TRIAL_COL")
    time_col = _cfg(cfg, "TIME_COL")
    feat_cfg = dict(
        bin_size=float(bin_size),
        baseline_win=tuple(_cfg(cfg, "SIDE_BASELINE_WIN")),
        post_win=tuple(_cfg(cfg, "SIDE_POST_WIN")),
        smooth_bins=int(_cfg(cfg, "SIDE_SMOOTH_BINS")),
        deriv_win_bins=int(_cfg(cfg, "SIDE_DERIV_WIN_BINS")),
        use_derivative=bool(_cfg(cfg, "SIDE_USE_DERIV")),
        right_is_positive=bool(_cfg(cfg, "SIDE_RIGHT_IS_POSITIVE")),
        time_col=time_col,
    )

    X = _side_feature_simple(window_stack, trial_col=trial_col, **feat_cfg)
    ds = (
        X.merge(labels_df[[trial_col, "y_side", "w"]], on=trial_col, how="inner")
        .dropna(subset=["y_side", "side_feat"])
    )

    if not ds.empty and ds["y_side"].nunique() == 2:
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=2000, class_weight="balanced", random_state=0)),
            ]
        )
        w = pd.to_numeric(ds["w"], errors="coerce").fillna(1.0).to_numpy()
        pipe.fit(ds[["side_feat"]], ds["y_side"].astype(int), clf__sample_weight=w)
        return TrainedSideModel(pipe, feat_cfg, trial_col, time_col)

    return SimpleSideModel(alpha=4.0, feat_cfg=feat_cfg, trial_col=trial_col)


# ============================================================
# Classic features + scoring
# ============================================================


def build_classic_features(
    window_stack: pd.DataFrame,
    trial_sides_pred: Optional[pd.DataFrame],
    *,
    bin_size: float,
    smooth_roll_bins: int,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    trial_col = _cfg(cfg, "TRIAL_COL")
    time_col = _cfg(cfg, "TIME_COL")
    needed = list(dict.fromkeys(_cfg(cfg, "POSE_COLS")))

    b = bin_signals(window_stack, bin_size=bin_size, cols=needed, trial_col=trial_col, time_col=time_col)
    smooth_cols = [c for c in b.columns if c not in [trial_col, "time_bin", time_col]]
    b = _rolling_median_by_trial(b, smooth_cols, int(smooth_roll_bins), time_col="time_bin", trial_col=trial_col)

    if trial_sides_pred is None or trial_sides_pred.empty or "trial_side" not in trial_sides_pred.columns:
        b["trial_side"] = "right"
    else:
        side_map = (
            trial_sides_pred[[trial_col, "trial_side"]]
            .dropna(subset=[trial_col])
            .set_index(trial_col)["trial_side"]
            .astype(str)
            .str.lower()
            .to_dict()
        )
        if not side_map:
            b["trial_side"] = "right"
        else:
            b = b[b[trial_col].isin(side_map.keys())].copy()
            b["trial_side"] = b[trial_col].map(side_map)

    out = []
    for tid, g in b.groupby(trial_col, observed=False):
        g = g.sort_values("time_bin").copy()
        side = str(g["trial_side"].iloc[0]).lower()
        for s in ["nose_cam_x", "nose_cam_y", "left_ear_cam_x", "left_ear_cam_y", "right_ear_cam_x", "right_ear_cam_y"]:
            if s in g.columns:
                g[f"conv__{s}"] = conv1d(pd.to_numeric(g[s], errors="coerce").to_numpy(), _cfg(cfg, "KERNEL"))
            else:
                g[f"conv__{s}"] = np.nan

        nx = pd.to_numeric(g.get("nose_cam_x", np.nan), errors="coerce").to_numpy()
        ny = pd.to_numeric(g.get("nose_cam_y", np.nan), errors="coerce").to_numpy()
        lx = pd.to_numeric(g.get("left_ear_cam_x", np.nan), errors="coerce").to_numpy()
        ly = pd.to_numeric(g.get("left_ear_cam_y", np.nan), errors="coerce").to_numpy()
        rx = pd.to_numeric(g.get("right_ear_cam_x", np.nan), errors="coerce").to_numpy()
        ry = pd.to_numeric(g.get("right_ear_cam_y", np.nan), errors="coerce").to_numpy()

        ear_min_y = np.fmin(ly, ry)
        y_good = np.isfinite(ny) & np.isfinite(ear_min_y) & (ny < ear_min_y)
        y_bad = np.isfinite(ny) & np.isfinite(ear_min_y) & (~y_good)
        g["feat__y_good"] = y_good.astype(float)
        g["feat__y_bad"] = y_bad.astype(float)

        if side == "right":
            ref = lx
            x_good = np.isfinite(nx) & np.isfinite(ref) & (nx <= ref)
        else:
            ref = rx
            x_good = np.isfinite(nx) & np.isfinite(ref) & (nx >= ref)
        x_bad = np.isfinite(nx) & np.isfinite(ref) & (~x_good)
        g["feat__x_good"] = x_good.astype(float)
        g["feat__x_bad"] = x_bad.astype(float)

        dy = np.concatenate([[np.nan], np.abs(np.diff(ny))])
        g["feat__jitter"] = (np.isfinite(dy) & (dy > 0.4)).astype(float)
        sep = np.abs(lx - rx)
        g["feat__ear_sep_bad"] = (np.isfinite(sep) & (sep < 0.2)).astype(float)

        if "head_angle_deg" in g.columns:
            ang = pd.to_numeric(g["head_angle_deg"], errors="coerce").to_numpy()
            dang = np.concatenate([[np.nan], wrap_deg180(np.diff(ang))])
            g["feat__head_inc"] = (dang > 0).astype(float)
            g["feat__head_dec"] = (dang < 0).astype(float)
        else:
            g["feat__head_inc"] = 0.0
            g["feat__head_dec"] = 0.0

        out.append(g)

    feat = pd.concat(out, ignore_index=True) if out else b.iloc[0:0].copy()
    feat_cols = [c for c in feat.columns if c.startswith("conv__") or c.startswith("feat__")]
    feat[feat_cols] = feat[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    keep = [trial_col, "time_bin", time_col, "trial_side"] + feat_cols
    return feat[keep].copy()


def classic_score_rows(feat_rows: pd.DataFrame, side: str, *, w_left: dict, w_right: dict) -> np.ndarray:
    w = w_right if str(side).lower() == "right" else w_left
    x = np.zeros(len(feat_rows), dtype=float)
    for k, wk in w.items():
        if k in feat_rows.columns:
            x += wk * pd.to_numeric(feat_rows[k], errors="coerce").fillna(0.0).to_numpy()
    return x


def score_all_trials_classic(feat: pd.DataFrame, *, cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    trial_col = _cfg(cfg, "TRIAL_COL")
    time_col = _cfg(cfg, "TIME_COL")
    scored = feat[[trial_col, "time_bin", time_col, "trial_side"]].copy()

    side = scored["trial_side"].astype(str).str.lower().to_numpy()
    raw = np.zeros(len(scored), float)
    mL = side == "left"
    mR = side == "right"
    if mL.any():
        raw[mL] = classic_score_rows(
            feat.loc[mL],
            "left",
            w_left=_cfg(cfg, "SIG_WEIGHTS_LEFT"),
            w_right=_cfg(cfg, "SIG_WEIGHTS_RIGHT"),
        )
    if mR.any():
        raw[mR] = classic_score_rows(
            feat.loc[mR],
            "right",
            w_left=_cfg(cfg, "SIG_WEIGHTS_LEFT"),
            w_right=_cfg(cfg, "SIG_WEIGHTS_RIGHT"),
        )
    scored["turn_score_raw"] = raw

    out = []
    for tid, g in scored.groupby(trial_col, observed=False):
        g = g.sort_values("time_bin").copy()
        t = pd.to_numeric(g["time_bin"], errors="coerce").to_numpy()
        r = pd.to_numeric(g["turn_score_raw"], errors="coerce").to_numpy()
        if _cfg(cfg, "CLASSIC_BASELINE_SUBTRACT"):
            base_mask = np.isfinite(t) & (t >= _cfg(cfg, "CLASSIC_SCORE_BASELINE_WIN")[0]) & (
                t <= _cfg(cfg, "CLASSIC_SCORE_BASELINE_WIN")[1]
            )
            base0 = np.nanmedian(r[base_mask]) if base_mask.any() else 0.0
            s = r - base0
        else:
            s = r.copy()
        if _cfg(cfg, "CLASSIC_CLIP_AT_ZERO"):
            s = np.maximum(s, 0.0)
        g["turn_score_nms"] = s
        out.append(g)
    scored = pd.concat(out, ignore_index=True) if out else scored.iloc[0:0].copy()

    out2 = []
    for tid, g in scored.groupby(trial_col, observed=False):
        g = _postprocess_series_per_trial(
            g,
            value_col="turn_score_nms",
            baseline_mode=_cfg(cfg, "CLASSIC_PP_BASELINE_MODE"),
            baseline_fixed=_cfg(cfg, "CLASSIC_PP_BASELINE_FIXED"),
            baseline_pct=_cfg(cfg, "CLASSIC_PP_BASELINE_PCT"),
            baseline_win=tuple(_cfg(cfg, "CLASSIC_PP_BASELINE_WIN")),
            thresh_mode=_cfg(cfg, "CLASSIC_PP_THRESH_MODE"),
            thresh_fixed=_cfg(cfg, "CLASSIC_PP_THRESH_FIXED"),
            thresh_pct=_cfg(cfg, "CLASSIC_PP_THRESH_PCT"),
            smooth_bins=_cfg(cfg, "POSTPROC_SMOOTH_BINS"),
            out_col="turn_score_pp",
        )
        out2.append(g)
    return pd.concat(out2, ignore_index=True) if out2 else scored


# ============================================================
# ML features + labels + fit/score
# ============================================================

def build_ml_features_diff(
    window_stack: pd.DataFrame,
    trial_sides_pred: Optional[pd.DataFrame],
    *,
    bin_size: float,
    signals: list[str],
    smooth_roll_bins: int,
    diff_k_bins: int,
    lags: int,
    method: str = "convolution",  # "convolution" | "shift"
    cfg: Optional[Dict[str, Any]] = None,
    align_x_by_side: bool = True,  # flip x signals by trial side
    use_new_side_v: bool = False,
    flip_head_angle_aligned: Optional[bool] = True,
) -> pd.DataFrame:
    trial_col = _cfg(cfg, "TRIAL_COL")
    time_col = _cfg(cfg, "TIME_COL")

    cols = [c for c in signals if c in window_stack.columns]
    angle_src_col = None
    if use_new_side_v and "head_angle_deg" in cols and "head_angle_n_shifted_deg" in window_stack.columns:
        angle_src_col = "head_angle_n_shifted_deg"
    bin_cols = cols + ([angle_src_col] if angle_src_col and angle_src_col not in cols else [])
    if not cols:
        return pd.DataFrame(columns=[trial_col, "time_bin", time_col, "trial_side", "pred_turn_side"])

    # 1) bin + smooth
    b = bin_signals(window_stack, bin_size=float(bin_size), cols=bin_cols, trial_col=trial_col, time_col=time_col)
    b = _rolling_median_by_trial(b, bin_cols, int(max(1, smooth_roll_bins)), time_col="time_bin", trial_col=trial_col)
    b = _add_mid_ear_cols(b)

    # 2) map side
    if trial_sides_pred is None or trial_sides_pred.empty or "trial_side" not in trial_sides_pred.columns:
        b["trial_side"] = "right"
    else:
        side_map = (
            trial_sides_pred[[trial_col, "trial_side"]]
            .dropna(subset=[trial_col])
            .set_index(trial_col)["trial_side"]
            .astype(str).str.lower()
            .to_dict()
        )
        if not side_map:
            b["trial_side"] = "right"
        else:
            b = b[b[trial_col].isin(side_map.keys())].copy()
            b["trial_side"] = b[trial_col].map(side_map).astype(str).str.lower()

    # 3) add requested dynamics (computed AFTER binning, from binned signals)
    extra_cols: list[str] = []
    if "mid_ear_cam_x" in b.columns:
        extra_cols.append("mid_ear_cam_x")
    if "mid_ear_x" in b.columns:
        extra_cols.append("mid_ear_x")

    cols_plus = cols + [c for c in extra_cols if c not in cols]

    k = int(max(1, diff_k_bins))
    L = int(max(0, lags))
    dt = float(bin_size)

    out = []
    for tid, g in b.groupby(trial_col, observed=False):
        g = g.sort_values("time_bin").copy()

        # side sign: right=+1, left=-1 (default +1 if unknown)
        side = str(g["trial_side"].iloc[0]).lower() if "trial_side" in g.columns and len(g) else ""
        side_sign = 1.0 if side == "right" else (-1.0 if side == "left" else 1.0)
        g["pred_turn_side"] = float(side_sign)

        # helper: flip selected signals based on the requested side-alignment rules
        def _maybe_flip_signal(name: str, arr: np.ndarray) -> np.ndarray:
            name_str = str(name)
            if use_new_side_v and name_str == "head_angle_deg":
                if flip_head_angle_aligned is None:
                    return arr
                if flip_head_angle_aligned:
                    return arr * (side_sign if align_x_by_side else 1.0)
                return arr * (-side_sign if align_x_by_side else side_sign)
            if not align_x_by_side:
                return arr
            if name_str.endswith("_x"):
                return arr * side_sign
            # if name_str.endswith("_y") and ("left_ear" in name_str or "right_ear" in name_str):
                return arr * side_sign
            if use_new_side_v and name_str.endswith("_y") and not name_str.lower().startswith("nose"):
                return arr * side_sign
            return arr

        # build ALL derived columns here, then concat once.
        new_cols: Dict[str, np.ndarray] = {}

        if method == "convolution":
            win = k
            for s in cols_plus:
                if s not in g.columns:
                    continue

                if s == "head_angle_deg" and angle_src_col and angle_src_col in g.columns:
                    v = pd.to_numeric(g[angle_src_col], errors="coerce").to_numpy(dtype=float)
                else:
                    v = pd.to_numeric(g[s], errors="coerce").to_numpy(dtype=float)
                v = _maybe_flip_signal(s, v)

                d1 = conv_derivative(v, win_bins=win, dt=dt)
                absd1 = np.abs(d1)

                # derivative features (diff naming)
                new_cols[f"ml__{s}__diff{k}"] = d1
                new_cols[f"ml__{s}__absdiff{k}"] = absd1

                if L > 0:
                    lag_sources = {
                        f"ml__{s}__diff{k}": d1,
                        f"ml__{s}__absdiff{k}": absd1,
                    }
                    for lag in range(1, L + 1):
                        for base_name, arr in lag_sources.items():
                            new_cols[f"{base_name}__lag{lag}"] = _shift_np(arr, lag)

        elif method == "shift":
            for s in cols_plus:
                if s not in g.columns:
                    continue

                if s == "head_angle_deg" and angle_src_col and angle_src_col in g.columns:
                    v = pd.to_numeric(g[angle_src_col], errors="coerce").to_numpy(dtype=float)
                else:
                    v = pd.to_numeric(g[s], errors="coerce").to_numpy(dtype=float)
                v = _maybe_flip_signal(s, v)

                # shift-diff on the (possibly flipped) signal
                v_ser = pd.Series(v, index=g.index)
                d = (v_ser - v_ser.shift(k)) / (k * dt)
                d_np = pd.to_numeric(d, errors="coerce").to_numpy(dtype=float)
                absd_np = np.abs(d_np)

                new_cols[f"ml__{s}__diff{k}"] = d_np
                new_cols[f"ml__{s}__absdiff{k}"] = absd_np

                if L > 0:
                    ds = pd.Series(d_np, index=g.index)
                    absds = pd.Series(absd_np, index=g.index)
                    for lag in range(1, L + 1):
                        new_cols[f"ml__{s}__diff{k}__lag{lag}"] = ds.shift(lag).to_numpy()
                        new_cols[f"ml__{s}__absdiff{k}__lag{lag}"] = absds.shift(lag).to_numpy()
        else:
            raise ValueError("method must be 'convolution' or 'shift'")

        # concat once (prevents fragmentation)
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=g.index)
            g = pd.concat([g, new_df], axis=1)

        out.append(g)

    feat = pd.concat(out, ignore_index=True) if out else b.iloc[0:0].copy()

    # ensure numeric
    raw_cols = [c for c in cols_plus if c in feat.columns]
    feat[raw_cols] = feat[raw_cols].apply(pd.to_numeric, errors="coerce")
    ml_cols = [c for c in feat.columns if c.startswith("ml__")]
    feat[ml_cols] = feat[ml_cols].apply(pd.to_numeric, errors="coerce")
    feat["pred_turn_side"] = pd.to_numeric(feat.get("pred_turn_side", np.nan), errors="coerce")

    keep = [trial_col, "time_bin", time_col, "trial_side", "pred_turn_side"] + raw_cols + ml_cols
    return feat[keep].copy()


def _parse_turn_times(x: Any) -> List[float]:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        out = []
        for v in x:
            try:
                fv = float(v)
                if np.isfinite(fv):
                    out.append(fv)
            except Exception:
                pass
        return out
    if isinstance(x, (int, float, np.number)):
        fv = float(x)
        return [fv] if np.isfinite(fv) else []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = json.loads(s)
            return _parse_turn_times(v)
        except Exception:
            pass
        try:
            v = ast.literal_eval(s)
            return _parse_turn_times(v)
        except Exception:
            pass
        parts = [p.strip() for p in s.split(",")]
        out = []
        for p in parts:
            try:
                fv = float(p)
                if np.isfinite(fv):
                    out.append(fv)
            except Exception:
                pass
        return out
    return []


def _flatten_turn_times(items: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for item in items:
        out.extend(_parse_turn_times(item))
    return out


def make_turn_labels_multi_for_bins_v2(
    feat_ml: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    train_time_range: Tuple[float, float],
    pos_half_width_s: float,
    hit_time: float = -2.0,
    enforce_pos_after: float = -2.0,
    snap_if_missed: bool = True,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    if feat_ml.empty:
        trial_col = _cfg(cfg, "TRIAL_COL")
        return feat_ml.assign(y_turn=np.nan, y_phase=np.nan).iloc[0:0].copy()

    trial_col = _cfg(cfg, "TRIAL_COL")
    need_cols = [trial_col, "turn_times_expect", "w"]
    if not all(c in labels_df.columns for c in need_cols):
        raise ValueError(f"labels_df must include columns: {need_cols}")

    df = feat_ml.merge(labels_df[need_cols], on=trial_col, how="inner").copy()
    if df.empty:
        return df.assign(y_turn=np.nan, y_phase=np.nan)

    t = pd.to_numeric(df["time_bin"], errors="coerce")
    df = df[(t >= train_time_range[0]) & (t <= train_time_range[1])].copy()
    if df.empty:
        return df.assign(y_turn=np.nan, y_phase=np.nan)

    out = []
    for tid, g in df.groupby(trial_col, observed=False):
        g = g.sort_values("time_bin").copy()
        tt = pd.to_numeric(g["time_bin"], errors="coerce").to_numpy()
        y_turn = np.zeros(len(g), dtype=float)
        y_phase = np.zeros(len(g), dtype=float)
        pre = np.isfinite(tt) & (tt < float(hit_time))
        y_phase[pre] = -1.0

        gts = _parse_turn_times(g["turn_times_expect"].iloc[0])
        for gt in gts:
            if not np.isfinite(gt):
                continue
            allow_pre_for_this_gt = gt < float(enforce_pos_after)
            lo = gt - float(pos_half_width_s)
            hi = gt + float(pos_half_width_s)
            pos = np.isfinite(tt) & (tt >= lo) & (tt <= hi)
            if not allow_pre_for_this_gt:
                pos = pos & (tt >= float(enforce_pos_after))
            if pos.any():
                y_turn[pos] = 1.0
                y_phase[pos] = 1.0
            elif snap_if_missed and np.isfinite(tt).any():
                j = int(np.nanargmin(np.abs(tt - gt)))
                if allow_pre_for_this_gt or (tt[j] >= float(enforce_pos_after)):
                    y_turn[j] = 1.0
                    y_phase[j] = 1.0
        g["y_turn"] = y_turn
        g["y_phase"] = y_phase
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0].copy()


def _dummy_prob_model(feat_cols: List[str], p: float = 0.0):
    class _Dummy:
        def __init__(self, p, feat_cols):
            self._p = float(np.clip(p, 0.0, 1.0))
            self._feat_cols = list(feat_cols)

        def predict_proba(self, X):
            n = len(X)
            return np.column_stack([np.full(n, 1.0 - self._p), np.full(n, self._p)])

    d = _Dummy(p, feat_cols)
    d._feat_cols = list(feat_cols)
    return d


def fit_turn_model_ml_diff(
    feat_ml: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    train_time_range: Tuple[float, float],
    pos_half_width_s: float,
    enforce_pos_after: float,
    model_cfg: dict,
    class_balanced: bool,
) -> Any:
    ds = make_turn_labels_multi_for_bins_v2(
        feat_ml,
        labels_df,
        train_time_range=train_time_range,
        pos_half_width_s=pos_half_width_s,
        enforce_pos_after=enforce_pos_after,
    )
    ml_cols = [c for c in ds.columns if c.startswith("ml__")]
    if ds.empty or not ml_cols:
        return _dummy_prob_model(ml_cols or [c for c in feat_ml.columns if c.startswith("ml__")], p=0.0)

    uniq = pd.Series(ds["y_turn"]).dropna().unique()
    if len(uniq) < 2:
        p = float(np.clip(np.nanmean(ds["y_turn"]), 0.0, 1.0)) if len(uniq) == 1 else 0.0
        return _dummy_prob_model(ml_cols, p=p)

    clf = LogisticRegression(
        penalty=model_cfg.get("penalty", "elasticnet"),
        solver=model_cfg.get("solver", "saga"),
        l1_ratio=model_cfg.get("l1_ratio", 0.5),
        C=model_cfg.get("C", 1.0),
        max_iter=model_cfg.get("max_iter", 8000),
        random_state=model_cfg.get("random_state", 0),
        class_weight=("balanced" if class_balanced else None),
    )
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", clf),
        ]
    )
    w_series = ds["w"] if "w" in ds.columns else pd.Series(1.0, index=ds.index)
    w = pd.to_numeric(w_series, errors="coerce").fillna(1.0).to_numpy()
    pipe.fit(ds[ml_cols], ds["y_turn"].astype(int), clf__sample_weight=w)
    setattr(pipe, "_feat_cols", ml_cols)
    return pipe


def score_all_trials_from_turn_pipe_ml(
    feat_ml: pd.DataFrame,
    turn_pipe,
    *,
    baseline_win: Tuple[float, float],
    postproc_value_col: str = "turn_score_rawprob",
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    trial_col = _cfg(cfg, "TRIAL_COL")
    time_col = _cfg(cfg, "TIME_COL")

    if feat_ml.empty:
        return pd.DataFrame(
            columns=[trial_col, "time_bin", time_col, "trial_side", "turn_score_rawprob", "turn_score_nms", "turn_score_pp"]
        )

    feat_cols = getattr(turn_pipe, "_feat_cols", None) or [c for c in feat_ml.columns if c.startswith("ml__")]
    X = feat_ml[feat_cols]
    scored = feat_ml[[trial_col, "time_bin", time_col, "trial_side"]].copy()
    scored["turn_score_rawprob"] = turn_pipe.predict_proba(X)[:, 1]

    out = []
    for tid, g in scored.groupby(trial_col, observed=False):
        g = g.sort_values("time_bin").copy()
        t = pd.to_numeric(g["time_bin"], errors="coerce").to_numpy()
        r = pd.to_numeric(g["turn_score_rawprob"], errors="coerce").to_numpy()
        m = np.isfinite(t) & np.isfinite(r) & (t >= baseline_win[0]) & (t <= baseline_win[1])
        base0 = float(np.nanmedian(r[m])) if m.any() else 0.0
        g["turn_score_nms"] = np.maximum(r - base0, 0.0)
        g = _postprocess_series_per_trial(
            g,
            value_col=postproc_value_col,
            baseline_mode=_cfg(cfg, "ML_PP_BASELINE_MODE"),
            baseline_fixed=_cfg(cfg, "ML_PP_BASELINE_FIXED"),
            baseline_pct=_cfg(cfg, "ML_PP_BASELINE_PCT"),
            baseline_win=tuple(_cfg(cfg, "ML_PP_BASELINE_WIN")),
            thresh_mode=_cfg(cfg, "ML_PP_THRESH_MODE"),
            thresh_fixed=_cfg(cfg, "ML_PP_THRESH_FIXED"),
            thresh_pct=_cfg(cfg, "ML_PP_THRESH_PCT"),
            smooth_bins=_cfg(cfg, "POSTPROC_SMOOTH_BINS"),
            out_col="turn_score_pp",
        )
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else scored.iloc[0:0].copy()


# ============================================================
# Meta + pivots + heatmaps
# ============================================================


def build_trial_meta_hit_only(df: pd.DataFrame, *, cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    trial_col = _cfg(cfg, "TRIAL_COL")
    meta_cols = list(_cfg(cfg, "META_COLS"))
    work = df.copy()
    if "window_label" in work.columns:
        hit_df = work[work["window_label"].astype(str).str.lower().eq("hit")].copy()
    else:
        hit_df = work.copy()

    keep = [trial_col] + [c for c in meta_cols if c in hit_df.columns]
    meta = (
        hit_df[keep]
        .dropna(subset=[trial_col])
        .groupby(trial_col, observed=False)
        .agg({c: "first" for c in keep if c != trial_col})
        .reset_index()
    )
    if "is_reward_bug" not in meta.columns:
        meta["is_reward_bug"] = False
    return meta


def make_reward_pivots(
    scored_all: pd.DataFrame,
    meta: pd.DataFrame,
    trial_ids: List[Any],
    *,
    value_col: str,
    time_range: Tuple[float, float],
    trial_col: str,
    time_col: str,
) -> pd.DataFrame:
    s = scored_all.copy()
    s[time_col] = pd.to_numeric(s[time_col], errors="coerce")
    s = s[s[trial_col].isin(trial_ids)].copy()
    s = s[(s[time_col] >= time_range[0]) & (s[time_col] <= time_range[1])].copy()
    piv = s.pivot_table(index=trial_col, columns=time_col, values=value_col, aggfunc="mean")
    if piv.empty:
        return piv.copy()
    return piv.reindex(columns=sorted(piv.columns))


def _force_white_min_cmap(cmap_name: str):
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, 256))
    colors[0] = np.array([1, 1, 1, 1])
    return mpl_colors.LinearSegmentedColormap.from_list(f"{cmap_name}_white0", colors)


def sort_by_peak_time(
    pivot: pd.DataFrame,
    *,
    start_time: float = -2.0,
    min_peak: float = SORT_MIN_PEAK_FOR_ORDER,
) -> pd.DataFrame:
    cols = [c for c in pivot.columns if np.isfinite(c) and float(c) >= float(start_time)]
    if not cols:
        return pivot
    sub = pivot[cols]
    arr = sub.to_numpy()
    arr_masked = np.where(np.isfinite(arr), arr, -np.inf)
    peak_j = np.argmax(arr_masked, axis=1)
    peak_time = np.array(cols, dtype=float)[peak_j]
    peak_val = arr_masked[np.arange(arr_masked.shape[0]), peak_j]
    bad = (~np.isfinite(peak_val)) | (peak_val < float(min_peak))
    peak_val = np.where(bad, -np.inf, peak_val)
    peak_time = np.where(bad, np.inf, peak_time)
    order = np.lexsort((-peak_val, peak_time))
    return pivot.iloc[order]


def plot_reward_heatmap_single_no_side(
    pivot: pd.DataFrame,
    *,
    event_lines: tuple[tuple[float, str], ...] = EVENT_LINES,
    cmap: str = "Reds",
    color_range: Optional[tuple[float, float]] = None,
    legend_min_score: Optional[float] = None,
    row_height: float = 0.18,
    sort_rows: bool = True,
    sort_start_time: float = -2.0,
    normalize_for_plot: str = "none",
    force_white_zero: bool = True,
    xtick_step_s: float = 1.0,
    cbar_label: str = "Score",
    title: str = "Trials",
    ax: Optional[plt.Axes] = None,
    cax: Optional[plt.Axes] = None,
    show_colorbar: bool = True,
    is_legend: bool = True,
) -> Figure:
    piv = pivot.copy()
    if sort_rows:
        piv = sort_by_peak_time(piv, start_time=sort_start_time)

    normalize_for_plot = str(normalize_for_plot).lower()
    if normalize_for_plot == "rowmax":
        a = piv.to_numpy()
        m = np.nanmax(a, axis=1)
        m[m <= 0] = np.nan
        piv = piv.div(m, axis=0)
    elif normalize_for_plot == "globalmax":
        a = piv.to_numpy().ravel()
        finite = a[np.isfinite(a)]
        gmax = np.nanmax(finite) if finite.size else 1.0
        if gmax > 0:
            piv = piv / gmax

    if piv.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 2))
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, "No trials to plot", ha="center", va="center")
        ax.axis("off")
        return fig

    allv = piv.to_numpy().ravel()
    finite = allv[np.isfinite(allv)]
    if color_range is None:
        vmax = np.percentile(finite, 99) if finite.size else 1.0
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        vmin = 0.0
    else:
        vmin, vmax = color_range
    if legend_min_score is not None and np.isfinite(legend_min_score):
        vmin = float(legend_min_score)
    if vmin > vmax:
        vmin = vmax

    cmap_obj = _force_white_min_cmap(cmap) if force_white_zero else mpl.cm.get_cmap(cmap)
    height = max(0.4, len(piv.index) * row_height)
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(14, height + 1.2))
    else:
        fig = ax.figure
    show_cbar = (cax is not None or created_fig) and show_colorbar
    heatmap_kwargs = dict(
        ax=ax,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        cbar=show_cbar,
        xticklabels=False,
        yticklabels=True,
    )
    if show_cbar:
        if cax is not None:
            heatmap_kwargs["cbar_ax"] = cax
        heatmap_kwargs["cbar_kws"] = {"label": cbar_label if is_legend else ""}
    sns.heatmap(piv.fillna(0.0), **heatmap_kwargs)
    cols = np.asarray(list(piv.columns), float)

    def _time_to_x(t):
        if cols.size == 0:
            return np.nan
        return float(np.interp(t, cols, np.arange(len(cols)))) + 0.5

    for tt, color in event_lines:
        xpos = _time_to_x(tt)
        if np.isfinite(xpos):
            ax.axvline(x=xpos, color=color, linestyle="--", linewidth=1)

    if cols.size:
        tmin, tmax = float(np.nanmin(cols)), float(np.nanmax(cols))
        step = max(1e-9, float(xtick_step_s))
        tick_times = np.arange(np.floor(tmin / step) * step, np.ceil(tmax / step) * step + 0.5 * step, step)
        tick_times = tick_times[(tick_times >= tmin) & (tick_times <= tmax)]
        ax.set_xticks([_time_to_x(tt) for tt in tick_times])
        ax.set_xticklabels([f"{tt:g}" for tt in tick_times], rotation=0)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trials")
    ax.tick_params(axis="y", labelsize=6, length=0)
    if created_fig:
        fig.tight_layout(h_pad=0.8)
    return fig


def plot_reward_split_no_side(
    scored_all: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    value_col: str,
    time_range: Tuple[float, float],
    row_height: float,
    title_prefix: str = "",
    which: Literal["both", "reward", "no_reward"] = "both",
    normalize_for_plot: str = "none",
    sort_rows: bool = True,
    cmap: str = "Reds",
    xtick_step_s: float = 1.0,
    ax: Optional[plt.Axes] = None,
    cax: Optional[plt.Axes] = None,
    show_colorbar: bool = True,
    is_legend: bool = True,
    legend_min_score: Optional[float] = None,
    do_layout: bool = True,
    cfg: Optional[Dict[str, Any]] = None,
) -> Figure:
    trial_col = _cfg(cfg, "TRIAL_COL")
    time_col = _cfg(cfg, "TIME_COL")

    reward_ids = meta.loc[meta["is_reward_bug"] == True, trial_col].tolist()
    norew_ids = meta.loc[meta["is_reward_bug"] == False, trial_col].tolist()
    if which == "reward":
        trial_ids = reward_ids
        title = "Reward trials"
    elif which == "no_reward":
        trial_ids = norew_ids
        title = "No-reward trials"
    else:
        trial_ids = None

    def _pivot(ids):
        return make_reward_pivots(
            scored_all=scored_all,
            meta=meta,
            trial_ids=ids,
            value_col=value_col,
            time_range=time_range,
            trial_col=trial_col,
            time_col=time_col,
        )

    if which in ("reward", "no_reward"):
        piv = _pivot(trial_ids)
        allv = piv.to_numpy().ravel()
        finite = allv[np.isfinite(allv)]
        vmax = np.percentile(finite, 99) if finite.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        vmin = 0.0
        fig = plot_reward_heatmap_single_no_side(
            piv,
            event_lines=_cfg(cfg, "EVENT_LINES"),
            cmap=cmap,
            color_range=(vmin, vmax),
            legend_min_score=legend_min_score,
            row_height=row_height,
            sort_rows=sort_rows,
            sort_start_time=_cfg(cfg, "SORT_TIME"),
            normalize_for_plot=normalize_for_plot,
            force_white_zero=_cfg(cfg, "HEATMAP_FORCE_WHITE_ZERO"),
            xtick_step_s=xtick_step_s,
            cbar_label=value_col,
            title=title,
            ax=ax,
            cax=cax,
            show_colorbar=show_colorbar,
            is_legend=is_legend,
        )
        prefix = (title_prefix.strip()) if title_prefix.strip() else ""
        fig.suptitle(prefix, fontsize=14, y=1.02)
        if do_layout:
            fig.tight_layout(h_pad=0.8)
        return fig

    reward_piv = _pivot(reward_ids)
    norew_piv = _pivot(norew_ids)
    allv = np.concatenate([reward_piv.to_numpy().ravel(), norew_piv.to_numpy().ravel()])
    finite = allv[np.isfinite(allv)]
    vmax = np.percentile(finite, 99) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    vmin = 0.0
    if legend_min_score is not None and np.isfinite(legend_min_score):
        vmin = float(legend_min_score)
    if vmin > vmax:
        vmin = vmax

    h1 = max(0.6, len(reward_piv.index) * row_height)
    h2 = max(0.6, len(norew_piv.index) * row_height)
    figsize = (14, h1 + h2 + 2.0)
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=figsize)
    if show_colorbar:
        gs = GridSpec(2, 2, width_ratios=[40, 0.7], height_ratios=[h1, h2], figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        cax = fig.add_subplot(gs[:, 1])
    else:
        gs = GridSpec(2, 1, height_ratios=[h1, h2], figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        cax = None

    cmap_obj = _force_white_min_cmap(cmap) if _cfg(cfg, "HEATMAP_FORCE_WHITE_ZERO") else mpl.cm.get_cmap(cmap)

    def _time_to_x(cols, t):
        cols = np.asarray(list(cols), float)
        if cols.size == 0:
            return np.nan
        return float(np.interp(t, cols, np.arange(len(cols)))) + 0.5

    def _add_lines(ax, cols):
        for tt, color in _cfg(cfg, "EVENT_LINES"):
            xpos = _time_to_x(cols, tt)
            if np.isfinite(xpos):
                ax.axvline(x=xpos, color=color, linestyle="--", linewidth=1)

    def _set_time_xticks(ax, cols):
        cols = np.asarray(list(cols), float)
        if cols.size == 0:
            return
        tmin = np.nanmin(cols)
        tmax = np.nanmax(cols)
        step = max(1e-9, float(xtick_step_s))
        tick_times = np.arange(np.floor(tmin / step) * step, np.ceil(tmax / step) * step + 0.5 * step, step)
        tick_times = tick_times[(tick_times >= tmin) & (tick_times <= tmax)]
        xticks = [_time_to_x(cols, tt) for tt in tick_times]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{tt:g}" for tt in tick_times], rotation=0)

    if sort_rows:
        reward_piv = sort_by_peak_time(reward_piv, start_time=_cfg(cfg, "SORT_TIME"))
        norew_piv = sort_by_peak_time(norew_piv, start_time=_cfg(cfg, "SORT_TIME"))

    sns.heatmap(
        reward_piv.fillna(0.0),
        ax=ax1,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
    )
    _add_lines(ax1, reward_piv.columns)
    _set_time_xticks(ax1, reward_piv.columns)
    ax1.set_title("Reward trials")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Trials")
    ax1.tick_params(axis="y", labelsize=6, length=0)

    heatmap_kws = dict(
        ax=ax2,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        cbar=show_colorbar,
    )
    if show_colorbar:
        heatmap_kws["cbar_ax"] = cax
        heatmap_kws["cbar_kws"] = {"label": value_col if is_legend else ""}
    sns.heatmap(norew_piv.fillna(0.0), **heatmap_kws)
    _add_lines(ax2, norew_piv.columns)
    _set_time_xticks(ax2, norew_piv.columns)
    ax2.set_title("No-reward trials")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Trials")
    ax2.tick_params(axis="y", labelsize=6, length=0)

    prefix = (title_prefix.strip() + " -- ") if title_prefix.strip() else ""
    fig.suptitle(prefix + f"Reward split (all sides) -- {value_col}", fontsize=14, y=1.02)
    if do_layout:
        fig.tight_layout(h_pad=1.2)
    return fig


def plot_reward_split(
    *,
    scored_all: pd.DataFrame,
    meta: pd.DataFrame,
    trial_sides_pred: pd.DataFrame,
    value_col: str,
    time_range: Tuple[float, float],
    row_height: float,
    cmap: str = "Reds",
    title_prefix: str = "",
    which: Literal["both", "reward", "no_reward"] = "both",
    layout: Literal["vertical", "horizontal"] = "vertical",
    normalize_for_plot: str = "none",
    sort_rows: bool = True,
    xtick_step_s: float = 1.0,
    cfg: Optional[Dict[str, Any]] = None,
) -> List[plt.Figure]:
    """
    Reward vs No-reward are DIFFERENT figures.
    Inside each figure: LEFT/RIGHT panels can be stacked vertically (default) or side-by-side.
    Returns a list of figures: [reward_fig, no_reward_fig] (or only one).
    """
    trial_col = _cfg(cfg, "TRIAL_COL")
    time_col = _cfg(cfg, "TIME_COL")
    event_lines = _cfg(cfg, "EVENT_LINES")
    force_white_zero = _cfg(cfg, "HEATMAP_FORCE_WHITE_ZERO")
    sort_time = float(_cfg(cfg, "SORT_TIME"))
    min_peak = float(_cfg(cfg, "SORT_MIN_PEAK_FOR_ORDER"))
    align_columns = bool(_cfg(cfg, "HEATMAP_ALIGN_COLUMNS"))

    info = (
        meta[[trial_col, "is_reward_bug"]].copy()
        .merge(trial_sides_pred[[trial_col, "trial_side"]], on=trial_col, how="left")
    )
    info["trial_side"] = info["trial_side"].astype(str).str.lower()
    info["is_reward_bug"] = info["is_reward_bug"].astype(bool)

    def _ids(is_reward: bool, side: str) -> List[float]:
        m = info["trial_side"].eq(side) & info["is_reward_bug"].eq(bool(is_reward))
        ids =  cast(pd.Series, info.loc[m, trial_col])
        return ids.tolist()

    def _pivot(ids: List[float]) -> pd.DataFrame:
        return make_reward_pivots(
            scored_all=scored_all,
            meta=meta,
            trial_ids=ids,
            value_col=value_col,
            time_range=time_range,
            trial_col=trial_col,
            time_col=time_col,
        )

    if which == "reward":
        blocks = [(True, "Reward")]
    elif which == "no_reward":
        blocks = [(False, "No-reward")]
    else:
        blocks = [(True, "Reward"), (False, "No-reward")]

    pivs: Dict[Tuple[str, str], pd.DataFrame] = {}
    for is_reward, block_name in blocks:
        for side in ("left", "right"):
            piv = _pivot(_ids(is_reward, side))

            if sort_rows and not piv.empty:
                piv = sort_by_peak_time(piv, start_time=sort_time, min_peak=min_peak)

            norm = str(normalize_for_plot).lower()
            if norm == "rowmax" and not piv.empty:
                a = piv.to_numpy()
                m = np.nanmax(a, axis=1)
                m[m <= 0] = np.nan
                piv = piv.div(m, axis=0)
            elif norm == "globalmax" and not piv.empty:
                a = piv.to_numpy().ravel()
                finite = a[np.isfinite(a)]
                gmax = np.nanmax(finite) if finite.size else 1.0
                if gmax > 0:
                    piv = piv / gmax

            pivs[(block_name, side)] = piv

    # shared color range across all blocks/panels
    all_vals = []
    for piv in pivs.values():
        if piv is not None and not piv.empty:
            v = piv.to_numpy().ravel()
            all_vals.append(v[np.isfinite(v)])
    if all_vals:
        all_finite = np.concatenate(all_vals)
        vmax = float(np.percentile(all_finite, 99)) if all_finite.size else 1.0
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    else:
        vmax = 1.0
    vmin = 0.0

    cmap_obj = _force_white_min_cmap(cmap) if force_white_zero else mpl.cm.get_cmap(cmap)

    def _time_to_x(cols, t):
        cols = np.asarray(list(cols), float)
        if cols.size == 0:
            return np.nan
        return float(np.interp(t, cols, np.arange(len(cols)))) + 0.5

    def _add_lines(ax, cols):
        for tt, color in event_lines:
            xpos = _time_to_x(cols, tt)
            if np.isfinite(xpos):
                ax.axvline(x=xpos, color=color, linestyle="--", linewidth=1)

    def _set_time_xticks(ax, cols):
        cols = np.asarray(list(cols), float)
        if cols.size == 0:
            return
        tmin = float(np.nanmin(cols))
        tmax = float(np.nanmax(cols))
        step = max(1e-9, float(xtick_step_s))
        tick_times = np.arange(np.floor(tmin / step) * step, np.ceil(tmax / step) * step + 0.5 * step, step)
        tick_times = tick_times[(tick_times >= tmin) & (tick_times <= tmax)]
        ax.set_xticks([_time_to_x(cols, tt) for tt in tick_times])
        ax.set_xticklabels([f"{tt:g}" for tt in tick_times], rotation=0)

    figs: List[plt.Figure] = []
    layout_mode = "horizontal" if str(layout).lower().startswith("h") else "vertical"
    from matplotlib.gridspec import GridSpec

    for _, block_name in blocks:
        piv_L = pivs.get((block_name, "left"), pd.DataFrame())
        piv_R = pivs.get((block_name, "right"), pd.DataFrame())

        if align_columns:
            shared_cols = sorted(set(piv_L.columns).union(piv_R.columns))
            if shared_cols:
                piv_L = piv_L.reindex(columns=shared_cols)
                piv_R = piv_R.reindex(columns=shared_cols)

        hL = int(len(piv_L.index)) if not piv_L.empty else 4
        hR = int(len(piv_R.index)) if not piv_R.empty else 4

        if layout_mode == "horizontal":
            fig_h = max(6.0, max(hL, hR) * float(row_height) + 2.5)
            fig_w = 18.0
            fig = plt.figure(figsize=(fig_w, fig_h))
            gs = GridSpec(1, 3, width_ratios=[40, 40, 0.7], figure=fig)
            ax_left = fig.add_subplot(gs[0, 0])
            ax_right = fig.add_subplot(gs[0, 1])
            cax = fig.add_subplot(gs[0, 2])
            left_xticklabels = True
        else:
            fig_h = max(6.0, (hL + hR) * float(row_height) + 2.5)
            fig_w = 16.0
            fig = plt.figure(figsize=(fig_w, fig_h))
            gs = GridSpec(2, 2, width_ratios=[40, 0.7], height_ratios=[max(1, hL), max(1, hR)], figure=fig)
            ax_left = fig.add_subplot(gs[0, 0])
            ax_right = fig.add_subplot(gs[1, 0])
            cax = fig.add_subplot(gs[:, 1])
            left_xticklabels = False

        plotted_any = False
        cbar_drawn = False

        # LEFT heatmap (no seaborn colorbar)
        if piv_L.empty:
            ax_left.text(0.5, 0.5, f"No trials\n({block_name}, left)", ha="center", va="center")
            ax_left.axis("off")
        else:
            sns.heatmap(
                piv_L.fillna(0.0),
                ax=ax_left,
                cmap=cmap_obj,
                vmin=vmin,
                vmax=vmax,
                cbar=not cbar_drawn,
                cbar_ax=cax if not cbar_drawn else None,
                cbar_kws={"label": value_col} if not cbar_drawn else None,
                xticklabels=left_xticklabels,
                yticklabels=True,
            )
            plotted_any = True
            cbar_drawn = True if not cbar_drawn else cbar_drawn
            _add_lines(ax_left, piv_L.columns)
            _set_time_xticks(ax_left, piv_L.columns)
            ax_left.set_title(f"{block_name} — left")
            ax_left.set_xlabel("Time (s)")
            ax_left.set_ylabel("Trials")
            ax_left.tick_params(axis="y", labelsize=6, length=0)

        # RIGHT heatmap (no seaborn colorbar)
        if piv_R.empty:
            ax_right.text(0.5, 0.5, f"No trials\n({block_name}, right)", ha="center", va="center")
            ax_right.axis("off")
        else:
            sns.heatmap(
                piv_R.fillna(0.0),
                ax=ax_right,
                cmap=cmap_obj,
                vmin=vmin,
                vmax=vmax,
                cbar=not cbar_drawn,
                cbar_ax=cax if not cbar_drawn else None,
                cbar_kws={"label": value_col} if not cbar_drawn else None,
                xticklabels=True,
                yticklabels=True,
            )
            plotted_any = True
            cbar_drawn = True if not cbar_drawn else cbar_drawn
            _add_lines(ax_right, piv_R.columns)
            _set_time_xticks(ax_right, piv_R.columns)
            ax_right.set_title(f"{block_name} — right")
            ax_right.set_xlabel("Time (s)")
            ax_right.set_ylabel("Trials")
            ax_right.tick_params(axis="y", labelsize=6, length=0)

        # shared, consistent seaborn/mpl colorbar
        if not plotted_any:
            cax.axis("off")

        prefix = (title_prefix.strip() + " — ") if title_prefix.strip() else ""
        layout_label = "stacked" if layout_mode == "vertical" else "horizontal"
        fig.suptitle(prefix + f"{block_name}", fontsize=14, y=1.02)
        fig.tight_layout()
        figs.append(fig)

    return figs

# ============================================================
# Single-trial plotting
# ============================================================


def plot_one_trial_scores_all3(
    *,
    trial_id: int | float,
    scored_classic: pd.DataFrame,
    scored_ml: pd.DataFrame,
    labels_df: Optional[pd.DataFrame] = None,
    time_range: Tuple[float, float] = PLOT_TIME_RANGE,
    cfg: Optional[Dict[str, Any]] = None,
):
    trial_col = _cfg(cfg, "TRIAL_COL")

    tid = float(trial_id)

    def _get(scored):
        if scored is None or scored.empty or trial_col not in scored.columns:
            return pd.DataFrame(columns=scored.columns if isinstance(scored, pd.DataFrame) else [])
        g = scored[pd.to_numeric(scored[trial_col], errors="coerce") == tid].copy()
        if g.empty:
            return g
        g["time_bin"] = pd.to_numeric(g["time_bin"], errors="coerce")
        g = g.sort_values("time_bin")
        g = g[(g["time_bin"] >= time_range[0]) & (g["time_bin"] <= time_range[1])]
        return g

    c = _get(scored_classic)
    m = _get(scored_ml)

    if _cfg(cfg, "PLOT_NORMALIZE_PER_MODEL"):
        if not c.empty:
            denom = float(np.nanmax(pd.to_numeric(c.get("turn_score_raw", 0), errors="coerce").to_numpy()))
            denom = denom if np.isfinite(denom) and denom > 0 else 1.0
            for col in ["turn_score_raw", "turn_score_nms", "turn_score_pp"]:
                if col in c.columns:
                    c[col + "__plot"] = pd.to_numeric(c[col], errors="coerce") / denom
        if not m.empty:
            m["turn_score_rawprob__plot"] = pd.to_numeric(m["turn_score_rawprob"], errors="coerce")
            for col in ["turn_score_nms", "turn_score_pp"]:
                if col in m.columns:
                    mm = float(np.nanmax(pd.to_numeric(m[col], errors="coerce").to_numpy()))
                    mm = mm if np.isfinite(mm) and mm > 0 else 1.0
                    m[col + "__plot"] = pd.to_numeric(m[col], errors="coerce") / mm

    fig, ax = plt.subplots(figsize=(12, 4))
    if not c.empty:
        if _cfg(cfg, "PLOT_NORMALIZE_PER_MODEL"):
            ax.plot(c["time_bin"], c.get("turn_score_raw__plot"), label="classic raw (plot)", linewidth=1, alpha=0.6)
            ax.plot(c["time_bin"], c.get("turn_score_nms__plot"), label="classic base (plot)", linewidth=2, alpha=0.9)
            ax.plot(c["time_bin"], c.get("turn_score_pp__plot"), label="classic post (plot)", linewidth=2.5)
        else:
            ax.plot(c["time_bin"], c.get("turn_score_raw"), label="classic raw", linewidth=1, alpha=0.6)
            ax.plot(c["time_bin"], c.get("turn_score_nms"), label="classic base", linewidth=2, alpha=0.9)
            ax.plot(c["time_bin"], c.get("turn_score_pp"), label="classic post", linewidth=2.5)

    if not m.empty:
        if _cfg(cfg, "PLOT_NORMALIZE_PER_MODEL"):
            ax.plot(m["time_bin"], m.get("turn_score_rawprob__plot"), label="ml rawprob (plot)", linewidth=1, alpha=0.6)
            ax.plot(m["time_bin"], m.get("turn_score_nms__plot"), label="ml base (plot)", linewidth=2, alpha=0.9)
            ax.plot(m["time_bin"], m.get("turn_score_pp__plot"), label="ml post (plot)", linewidth=2.5)
        else:
            ax.plot(m["time_bin"], m.get("turn_score_rawprob"), label="ml rawprob", linewidth=1, alpha=0.6)
            ax.plot(m["time_bin"], m.get("turn_score_nms"), label="ml base", linewidth=2, alpha=0.9)
            ax.plot(m["time_bin"], m.get("turn_score_pp"), label="ml post", linewidth=2.5)

    for tt, col in _cfg(cfg, "EVENT_LINES"):
        ax.axvline(tt, linestyle="--", linewidth=1, color=col)

    if labels_df is not None and not labels_df.empty:
        row = labels_df[pd.to_numeric(labels_df[trial_col], errors="coerce") == tid]
        if not row.empty:
            gts_raw = row["turn_times_expect"].iloc[0]
            gts = _parse_turn_times(gts_raw)
            for gt in gts:
                ax.axvline(float(gt), linestyle=":", linewidth=1.5, color="k")

    ax.set_title(f"Trial {int(tid) if tid.is_integer() else tid}: classic+ml (raw/base/post)")
    ax.set_xlabel("time_relative_to_end (s)")
    ax.set_ylabel("score" + (" (plot-normalized)" if _cfg(cfg, "PLOT_NORMALIZE_PER_MODEL") else ""))
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_one_trial_raw_and_diffs(
    *,
    trial_id: int | float,
    feat_ml: pd.DataFrame,
    signals: List[str],
    diff_k_bins: int,
    time_range: Tuple[float, float] = PLOT_TIME_RANGE,
    event_lines: tuple[tuple[float, str], ...] = EVENT_LINES,
    use_twin_axes: bool = True,
    cfg: Optional[Dict[str, Any]] = None,
):
    trial_col = _cfg(cfg, "TRIAL_COL")
    tid = float(trial_id)
    k = int(max(1, diff_k_bins))

    g = feat_ml[pd.to_numeric(feat_ml[trial_col], errors="coerce") == tid].copy()
    if g.empty:
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.text(0.5, 0.5, f"No feat_ml rows for trial_id={trial_id}", ha="center", va="center")
        ax.axis("off")
        return fig

    g["time_bin"] = pd.to_numeric(g["time_bin"], errors="coerce")
    g = g.sort_values("time_bin")
    g = g[(g["time_bin"] >= time_range[0]) & (g["time_bin"] <= time_range[1])].copy()

    x_sigs = [s for s in signals if s.endswith("_x")]
    y_sigs = [s for s in signals if s.endswith("_y")]
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(14, 7))
    ax_x, ax_y = axes
    def _plot_panel(ax, sigs, title):
        ax2 = ax.twinx() if use_twin_axes else ax
        for s in sigs:
            if s not in g.columns:
                continue
            raw = pd.to_numeric(g[s], errors="coerce")
            diff_col = f"ml__{s}__diff{k}"
            d = pd.to_numeric(g[diff_col], errors="coerce") if diff_col in g.columns else None
            ax.plot(g["time_bin"], raw, label=f"{s} (raw)", linewidth=1.6, alpha=0.9)
            if d is not None:
                color = ax.lines[-1].get_color()
                ax2.plot(g["time_bin"], d, label=f"{s} (diff/deriv{k})", linewidth=1.2, linestyle="--", alpha=0.8, color=color)
        for tt, col in event_lines:
            ax.axvline(tt, linestyle="--", linewidth=1, color=col)
            if use_twin_axes:
                ax2.axvline(tt, linestyle="--", linewidth=1, color=col)
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        ax.set_ylabel("raw")
        if use_twin_axes:
            ax2.set_ylabel(f"diff/deriv{k}")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels() if use_twin_axes else ([], [])
        if l1 or l2:
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, ncol=2)

    _plot_panel(ax_x, x_sigs, f"Trial {trial_id} -- X signals: raw (solid) + diff/deriv{k} (dashed)")
    _plot_panel(ax_y, y_sigs, f"Trial {trial_id} -- Y signals: raw (solid) + diff/deriv{k} (dashed)")
    ax_y.set_xlabel("time_relative_to_end (s)")
    fig.tight_layout()
    return fig


# ============================================================
# Single-trial heatmap + ML feature traces
# ============================================================


def plot_one_trial_heatmap_row_from_results(
    *,
    results: Dict[str, Any],
    trial_id: int | float,
    value_col: Optional[str] = None,
    time_range: Optional[Tuple[float, float]] = None,
    normalize_for_plot: Optional[str] = None,
    row_height: Optional[float] = None,
    xtick_step_s: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    cax: Optional[plt.Axes] = None,
    is_legend: bool = True,
    cfg: Optional[Dict[str, Any]] = None,
) -> plt.Figure:
    cfg = resolve_cfg(cfg or results.get("cfg"))
    trial_col = cfg["TRIAL_COL"]
    time_col = cfg["TIME_COL"]
    time_range = time_range or tuple(cfg["PLOT_TIME_RANGE"])
    normalize_for_plot = normalize_for_plot or cfg["HEATMAP_NORMALIZE_FOR_PLOT"]
    row_height = float(row_height if row_height is not None else cfg["PLOT_ROW_HEIGHT"])
    xtick_step_s = float(xtick_step_s if xtick_step_s is not None else cfg["HEATMAP_XTICK_STEP_S"])
    event_lines = tuple(cfg["EVENT_LINES"])
    force_white_zero = bool(cfg["HEATMAP_FORCE_WHITE_ZERO"])

    scored = results.get("scored_ml")
    if not isinstance(scored, pd.DataFrame) or scored.empty:
        scored = results.get("scored_classic")

    if not isinstance(scored, pd.DataFrame) or scored.empty:
        fig, ax = plt.subplots(figsize=(10, 2)) if ax is None else (ax.figure, ax)
        ax.text(0.5, 0.5, "No scored data available", ha="center", va="center")
        ax.axis("off")
        return fig

    value_col = value_col or ("turn_score_pp" if cfg["POSTPROC_APPLY_TO_HEATMAPS"] else "turn_score_nms")
    tid = float(trial_id)

    s = scored[pd.to_numeric(scored[trial_col], errors="coerce") == tid].copy()
    if s.empty:
        fig, ax = plt.subplots(figsize=(10, 2)) if ax is None else (ax.figure, ax)
        ax.text(0.5, 0.5, f"No data for trial_id={trial_id}", ha="center", va="center")
        ax.axis("off")
        return fig

    s[time_col] = pd.to_numeric(s[time_col], errors="coerce")
    s = s[(s[time_col] >= time_range[0]) & (s[time_col] <= time_range[1])].copy()
    piv = s.pivot_table(index=trial_col, columns=time_col, values=value_col, aggfunc="mean")
    if piv.empty:
        fig, ax = plt.subplots(figsize=(10, 2)) if ax is None else (ax.figure, ax)
        ax.text(0.5, 0.5, f"No bins for trial_id={trial_id}", ha="center", va="center")
        ax.axis("off")
        return fig

    piv = piv.reindex(columns=sorted(piv.columns))
    norm = str(normalize_for_plot).lower()
    if norm == "rowmax":
        a = piv.to_numpy()
        m = np.nanmax(a, axis=1)
        m[m <= 0] = np.nan
        piv = piv.div(m, axis=0)
    elif norm == "globalmax":
        a = piv.to_numpy().ravel()
        finite = a[np.isfinite(a)]
        gmax = np.nanmax(finite) if finite.size else 1.0
        if gmax > 0:
            piv = piv / gmax

    allv = piv.to_numpy().ravel()
    finite = allv[np.isfinite(allv)]
    vmax = np.percentile(finite, 99) if finite.size else 1.0
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    vmin = 0.0

    cmap_obj = _force_white_min_cmap("Reds") if force_white_zero else mpl.cm.get_cmap("Reds")
    height = min(0.01, len(piv.index) * row_height)
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(14, height + 0.4))
        show_cbar = True
    else:
        fig = ax.figure
        show_cbar = cax is not None

    cols = np.asarray(list(piv.columns), float)
    data = piv.fillna(0.0).to_numpy()
    n_rows = data.shape[0]
    if cols.size == 0:
        x_edges = np.array([0.0, 1.0])
    elif cols.size == 1:
        default_step = float(cfg.get("BIN_SIZE", xtick_step_s))
        x_edges = np.array([cols[0] - default_step / 2.0, cols[0] + default_step / 2.0])
    else:
        mids = (cols[:-1] + cols[1:]) / 2.0
        first = cols[0] - (mids[0] - cols[0])
        last = cols[-1] + (cols[-1] - mids[-1])
        x_edges = np.concatenate([[first], mids, [last]])
    y_edges = np.arange(n_rows + 1)

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        data,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        linewidth=0.0,
        edgecolors="none",
    )
    if show_cbar:
        label = value_col if is_legend else ""
        if cax is None:
            fig.colorbar(mesh, ax=ax, label=label, pad=0.02)
        else:
            fig.colorbar(mesh, cax=cax, label=label)

    ax.set_ylim(0, n_rows)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([str(v) for v in piv.index])

    for tt, color in event_lines:
        ax.axvline(x=tt, color=color, linestyle="--", linewidth=1)

    tmin, tmax = (float(time_range[0]), float(time_range[1])) if time_range else (float(np.nanmin(cols)), float(np.nanmax(cols)))
    step = max(1e-9, float(xtick_step_s))
    tick_times = np.arange(np.floor(tmin / step) * step, np.ceil(tmax / step) * step + 0.5 * step, step)
    tick_times = tick_times[(tick_times >= tmin) & (tick_times <= tmax)]
    ax.set_xticks(tick_times)
    ax.set_xticklabels([f"{tt:g}" for tt in tick_times], rotation=0)
    ax.set_xlim(tmin, tmax)

    title_tid = int(tid) if float(tid).is_integer() else tid
    ax.set_title(f"Trial {title_tid} — {value_col} (heatmap row)")
    ax.set_xlabel("rel time to trial end (s)")
    ax.set_ylabel("Trials")
    ax.tick_params(axis="y", labelsize=6, length=0)
    if created_fig:
        fig.tight_layout()
    return fig


def _assign_dlc_group_colors(group_keys: Sequence[str]) -> Dict[str, Tuple[float, float, float, float]]:
    def _fixed_color(group: str) -> Optional[Tuple[float, float, float, float]]:
        key = group.lower()
        if "head_angle" in key or key.startswith("angle"):
            return mpl.colors.to_rgba("black")
        if key.startswith("nose"):
            return mpl.colors.to_rgba("blue")
        if key.startswith("mid_ear"):
            return mpl.colors.to_rgba("purple")
        if key.startswith("left_ear"):
            return mpl.colors.to_rgba("orange")
        if key.startswith("right_ear"):
            return mpl.colors.to_rgba("green")
        return None

    mapping: Dict[str, Tuple[float, float, float, float]] = {}
    fixed_rgba = []
    for group in group_keys:
        fixed = _fixed_color(group)
        if fixed is not None:
            mapping[group] = fixed
            fixed_rgba.append(fixed)

    palette = plt.cm.get_cmap("tab20", max(len(group_keys), 1))
    palette_colors = [palette(i) for i in range(palette.N)]
    palette_idx = 0
    for group in group_keys:
        if group in mapping:
            continue
        while palette_idx < len(palette_colors) and palette_colors[palette_idx] in fixed_rgba:
            palette_idx += 1
        if palette_idx >= len(palette_colors):
            color = palette_colors[len(mapping) % len(palette_colors)]
        else:
            color = palette_colors[palette_idx]
            palette_idx += 1
        mapping[group] = color
    return mapping


def plot_one_trial_feature_traces_from_results(
    *,
    results: Dict[str, Any],
    trial_id: int | float,
    feature_mode: Literal["diff", "all"] = "diff",
    time_range: Optional[Tuple[float, float]] = None,
    legend_outside: bool = True,
    ax: Optional[plt.Axes] = None,
    cfg: Optional[Dict[str, Any]] = None,
    is_legend: bool = True,
) -> plt.Figure:
    cfg = resolve_cfg(cfg or results.get("cfg"))
    trial_col = cfg["TRIAL_COL"]
    time_range = time_range or tuple(cfg["PLOT_TIME_RANGE"])
    event_lines = tuple(cfg["EVENT_LINES"])

    feat_ml = results.get("feat_ml")
    if not isinstance(feat_ml, pd.DataFrame) or feat_ml.empty:
        fig, ax = plt.subplots(figsize=(10, 2)) if ax is None else (ax.figure, ax)
        ax.text(0.5, 0.5, "No ML features available", ha="center", va="center")
        ax.axis("off")
        return fig

    tid = float(trial_id)
    g = feat_ml[pd.to_numeric(feat_ml[trial_col], errors="coerce") == tid].copy()
    if g.empty:
        fig, ax = plt.subplots(figsize=(10, 2)) if ax is None else (ax.figure, ax)
        ax.text(0.5, 0.5, f"No features for trial_id={trial_id}", ha="center", va="center")
        ax.axis("off")
        return fig

    g["time_bin"] = pd.to_numeric(g["time_bin"], errors="coerce")
    g = g.sort_values("time_bin")
    g = g[(g["time_bin"] >= time_range[0]) & (g["time_bin"] <= time_range[1])].copy()

    feature_cols = getattr(results.get("turn_pipe_ml"), "_feat_cols", None)
    if not feature_cols:
        feature_cols = [c for c in g.columns if c.startswith("ml__")]

    def _base_name(col: str) -> str:
        base = col.replace("ml__", "")
        return base.split("__")[0]

    if feature_mode == "diff":
        filtered = [c for c in feature_cols if "__diff" in c and "__abs" not in c and "__lag" not in c]
        feature_cols = filtered if filtered else feature_cols

    feature_cols = [c for c in feature_cols if c in g.columns]
    feature_cols = [c for c in feature_cols if ("_cam_" in _base_name(c)) or ("head_angle" in _base_name(c))]
    if not feature_cols:
        fig, ax = plt.subplots(figsize=(10, 2)) if ax is None else (ax.figure, ax)
        ax.text(0.5, 0.5, "No matching feature columns to plot", ha="center", va="center")
        ax.axis("off")
        return fig

    interp_missing = bool(cfg.get("PLOT_INTERP_MISSING", False))
    fill_trailing = bool(cfg.get("PLOT_FILL_TRAILING", False))
    
    # Create complete time grid if we need to fill trailing values
    if fill_trailing and not g.empty:
        bin_size = float(cfg.get("BIN_SIZE", 0.05))
        full_time_bins = np.arange(time_range[0], time_range[1] + bin_size/2, bin_size)
        g = g.set_index("time_bin")
        g = g.reindex(full_time_bins, method="nearest", tolerance=bin_size/2)
        g = g.reset_index()
        g.rename(columns={"index": "time_bin"}, inplace=True)
    
    if interp_missing:
        for col in feature_cols:
            g[col] = pd.to_numeric(g[col], errors="coerce").interpolate(
                method="linear",
                limit_direction="both",
            )
    if fill_trailing:
        for col in feature_cols:
            g[col] = g[col].ffill()

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(14, 4))
    else:
        fig = ax.figure

    def _label(col: str) -> str:
        label = _base_name(col)
        return label[:100] + ("..." if len(label) > 100 else "")

    def _split_axis(base: str) -> Tuple[str, Optional[str]]:
        parts = base.split("_")
        axis = parts[-1] if parts and parts[-1] in {"x", "y"} else None
        root = "_".join(parts[:-1]) if axis else base
        if root.endswith("_cam"):
            root = root[:-4]
        return root, axis

    base_lookup = {col: _base_name(col) for col in feature_cols}
    group_keys = []
    for col in feature_cols:
        group, _ = _split_axis(base_lookup[col])
        if group not in group_keys:
            group_keys.append(group)
    group_to_color = _assign_dlc_group_colors(group_keys)
    x_vals = g["time_bin"].to_numpy()
    if time_range:
        x_min, x_max = float(time_range[0]), float(time_range[1])
    else:
        finite_x = x_vals[np.isfinite(x_vals)]
        x_min = float(np.min(finite_x)) if finite_x.size else 0.0
        x_max = float(np.max(finite_x)) if finite_x.size else 1.0
    x_span = max(x_max - x_min, 1e-6)

    for col in feature_cols:
        y = pd.to_numeric(g[col], errors="coerce").to_numpy()
        base = base_lookup[col]
        group, axis_suffix = _split_axis(base)
        color = group_to_color.get(group, "black")
        linestyle = "-" if axis_suffix != "y" else "--"
        ax.plot(x_vals, y, label=_label(col), linewidth=1, alpha=0.9, color=color, linestyle=linestyle)
        if axis_suffix in {"x", "y"}:
            valid = np.isfinite(x_vals) & np.isfinite(y)
            if valid.any():
                label_idx = np.flatnonzero(valid)[-1]
                x_pos = x_vals[label_idx]
                y_pos = y[label_idx]
                x_offset = -8 if x_pos > x_max - 0.1 * x_span else 6
                y_offset = 6 if axis_suffix == "x" else -6
                ax.annotate(
                    axis_suffix,
                    (x_pos, y_pos),
                    textcoords="offset points",
                    xytext=(x_offset, y_offset),
                    ha="left" if x_offset > 0 else "right",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=color,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.2),
                    clip_on=True,
                )

    for tt, col in event_lines:
        ax.axvline(tt, linestyle="--", linewidth=1, color=col)

    title_tid = int(tid) if float(tid).is_integer() else tid
    ax.set_title(f"Trial {title_tid} — DLC FEATURES")
    ax.set_xlabel("rel time to trial end [s]")
    ax.set_ylabel("Feature diff [px]")
    ax.grid(True, alpha=0.2)
    if time_range:
        ax.set_xlim(float(time_range[0]), float(time_range[1]))
    if is_legend:
        if legend_outside:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
        else:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    if created_fig:
        fig.tight_layout()
    return fig


def plot_one_trial_raw_feature_traces_from_results(
    *,
    results: Dict[str, Any],
    trial_id: int | float,
    time_range: Optional[Tuple[float, float]] = None,
    legend_outside: bool = True,
    ax: Optional[plt.Axes] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> plt.Figure:
    cfg = resolve_cfg(cfg or results.get("cfg"))
    trial_col = cfg["TRIAL_COL"]
    time_range = time_range or tuple(cfg["PLOT_TIME_RANGE"])
    event_lines = tuple(cfg["EVENT_LINES"])

    feat_ml = results.get("feat_ml")
    if not isinstance(feat_ml, pd.DataFrame) or feat_ml.empty:
        fig, ax = plt.subplots(figsize=(10, 2)) if ax is None else (ax.figure, ax)
        ax.text(0.5, 0.5, "No ML features available", ha="center", va="center")
        ax.axis("off")
        return fig

    tid = float(trial_id)
    g = feat_ml[pd.to_numeric(feat_ml[trial_col], errors="coerce") == tid].copy()
    if g.empty:
        fig, ax = plt.subplots(figsize=(10, 2)) if ax is None else (ax.figure, ax)
        ax.text(0.5, 0.5, f"No features for trial_id={trial_id}", ha="center", va="center")
        ax.axis("off")
        return fig

    g["time_bin"] = pd.to_numeric(g["time_bin"], errors="coerce")
    g = g.sort_values("time_bin")
    g = g[(g["time_bin"] >= time_range[0]) & (g["time_bin"] <= time_range[1])].copy()

    signals = list(cfg.get("SIGNALS", []))
    feature_cols = [c for c in signals if c in g.columns]
    if not feature_cols:
        fig, ax = plt.subplots(figsize=(10, 2)) if ax is None else (ax.figure, ax)
        ax.text(0.5, 0.5, "No raw feature columns to plot", ha="center", va="center")
        ax.axis("off")
        return fig

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(14, 4))
    else:
        fig = ax.figure

    def _label(col: str) -> str:
        return col[:100] + ("..." if len(col) > 100 else "")

    def _split_axis(base: str) -> Tuple[str, Optional[str]]:
        parts = base.split("_")
        axis = parts[-1] if parts and parts[-1] in {"x", "y"} else None
        root = "_".join(parts[:-1]) if axis else base
        if root.endswith("_cam"):
            root = root[:-4]
        return root, axis

    group_keys = []
    for col in feature_cols:
        group, _ = _split_axis(col)
        if group not in group_keys:
            group_keys.append(group)
    group_to_color = _assign_dlc_group_colors(group_keys)

    x_vals = g["time_bin"].to_numpy()
    if time_range:
        x_min, x_max = float(time_range[0]), float(time_range[1])
    else:
        finite_x = x_vals[np.isfinite(x_vals)]
        x_min = float(np.min(finite_x)) if finite_x.size else 0.0
        x_max = float(np.max(finite_x)) if finite_x.size else 1.0
    x_span = max(x_max - x_min, 1e-6)

    for col in feature_cols:
        y = pd.to_numeric(g[col], errors="coerce").to_numpy()
        group, axis_suffix = _split_axis(col)
        color = group_to_color.get(group, "black")
        linestyle = "-" if axis_suffix != "y" else "--"
        ax.plot(x_vals, y, label=_label(col), linewidth=1, alpha=0.9, color=color, linestyle=linestyle)
        if axis_suffix in {"x", "y"}:
            valid = np.isfinite(x_vals) & np.isfinite(y)
            if valid.any():
                label_idx = np.flatnonzero(valid)[-1]
                x_pos = x_vals[label_idx]
                y_pos = y[label_idx]
                x_offset = -8 if x_pos > x_max - 0.1 * x_span else 6
                y_offset = 6 if axis_suffix == "x" else -6
                ax.annotate(
                    axis_suffix,
                    (x_pos, y_pos),
                    textcoords="offset points",
                    xytext=(x_offset, y_offset),
                    ha="left" if x_offset > 0 else "right",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=color,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.2),
                    clip_on=True,
                )

    for tt, col in event_lines:
        ax.axvline(tt, linestyle="--", linewidth=1, color=col)

    title_tid = int(tid) if float(tid).is_integer() else tid
    ax.set_title(f"Trial {title_tid} — raw features")
    ax.set_xlabel("rel time to trial end (s)")
    ax.set_ylabel("feature value")
    ax.grid(True, alpha=0.2)
    if time_range:
        ax.set_xlim(float(time_range[0]), float(time_range[1]))
    if legend_outside:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
    else:
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    if created_fig:
        fig.tight_layout()
    return fig


def shift_fig_axis(fig, shift, label):
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda x, pos: f"{x + shift:g}")
    for ax in fig.axes:
        if not ax.get_xaxis().get_visible() or len(ax.get_xticks()) == 0:
            continue
        ax.xaxis.set_major_formatter(fmt)
        if ax.get_xlabel():
            label_str = str(label)
            if label_str.lower() == "hit":
                ax.set_xlabel("Time around hit [s]")
            else:
                ax.set_xlabel(f"time aligned to {label_str} [s]")
    return fig


def plot_trial_heatmap_and_traces_from_results(
    *,
    results: Dict[str, Any],
    trial_id: int | float,
    value_col: Optional[str] = None,
    feature_mode: Literal["diff", "all"] = "diff",
    time_range: Optional[Tuple[float, float]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    hspace: float = 0.1,
    ax: Optional[plt.Axes] = None,
    is_legend: bool = True,
) -> plt.Figure:
    cfg = resolve_cfg(cfg or results.get("cfg"))
    time_range = time_range or tuple(cfg["PLOT_TIME_RANGE"])

    from matplotlib.gridspec import GridSpec

    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=(14, 7))
        gs = GridSpec(
            2,
            2,
            width_ratios=[40, 0.35],
            height_ratios=[0.3, 3],
            wspace=0.02,
            hspace=hspace,
            figure=fig,
        )
    else:
        fig = ax.figure
        try:
            parent_spec = ax.get_subplotspec()
        except Exception:
            parent_spec = None
        if parent_spec is None:
            raise ValueError("ax must be a subplot axis created via subplots/gridspec")
        ax.remove()
        gs = parent_spec.subgridspec(
            2,
            2,
            width_ratios=[40, 0.35],
            height_ratios=[0.3, 3],
            wspace=0.02,
            hspace=hspace,
        )
    ax_heat = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax_traces = fig.add_subplot(gs[1, 0])
    ax_blank = fig.add_subplot(gs[1, 1])
    ax_blank.axis("off")

    plot_one_trial_heatmap_row_from_results(
        results=results,
        trial_id=trial_id,
        value_col=value_col,
        time_range=time_range,
        ax=ax_heat,
        cax=cax,
        cfg=cfg,
    )
    plot_one_trial_feature_traces_from_results(
        results=results,
        trial_id=trial_id,
        feature_mode=feature_mode,
        time_range=time_range,
        legend_outside=True,
        ax=ax_traces,
        cfg=cfg,
        is_legend=is_legend,
    )
    if created_fig:
        fig.tight_layout()

    ax_heat.set_title("")
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    cax.set_ylabel("")  # remove colorbar label

    for line in ax_traces.get_lines():
        if line.get_label() == "mid_ear_cam_x":
            line.set_color("purple")

    # refresh legend so the handle matches
    if is_legend:
        ax_traces.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
    else:
        legend = ax_traces.get_legend()
        if legend is not None:
            legend.remove()

    ax_heat.tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax_heat.spines.values():
        spine.set_visible(True)

    ax_traces.set_title("")
    date_tag = "05"
    if created_fig:
        # bring the rows closer
        fig.subplots_adjust(hspace=0.0)
        fig = shift_fig_axis(fig, 2, "hit")
    return fig


def _resolve_scores_plot_cfg(
    scores_df: pd.DataFrame,
    *,
    trial_col: Optional[str] = None,
    time_col: Optional[str] = None,
    time_range: Optional[Tuple[float, float]] = None,
    normalize_for_plot: Optional[str] = None,
    row_height: Optional[float] = None,
    xtick_step_s: Optional[float] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg_use = resolve_cfg(cfg)

    if trial_col is None:
        trial_col = "trial_id" if "trial_id" in scores_df.columns else cfg_use.get("TRIAL_COL")
    if not trial_col or trial_col not in scores_df.columns:
        raise ValueError(f"trial_col '{trial_col}' not found in scores_df")

    if time_col is None:
        if "time_s" in scores_df.columns:
            time_col = "time_s"
        elif "time_bin" in scores_df.columns:
            time_col = "time_bin"
        else:
            time_col = cfg_use.get("TIME_COL")
    if not time_col or time_col not in scores_df.columns:
        raise ValueError(f"time_col '{time_col}' not found in scores_df")

    cfg_use["TRIAL_COL"] = trial_col
    cfg_use["TIME_COL"] = time_col
    if time_range is not None:
        cfg_use["PLOT_TIME_RANGE"] = tuple(time_range)
    if normalize_for_plot is not None:
        cfg_use["HEATMAP_NORMALIZE_FOR_PLOT"] = normalize_for_plot
    if row_height is not None:
        cfg_use["PLOT_ROW_HEIGHT"] = float(row_height)
    if xtick_step_s is not None:
        cfg_use["HEATMAP_XTICK_STEP_S"] = float(xtick_step_s)
    return cfg_use


def plot_one_trial_heatmap_row_from_scores(
    *,
    scores_df: pd.DataFrame,
    trial_id: int | float,
    value_col: str = "score",
    trial_col: Optional[str] = None,
    time_col: Optional[str] = None,
    animal_id: Optional[str] = None,
    time_range: Optional[Tuple[float, float]] = None,
    normalize_for_plot: Optional[str] = None,
    row_height: Optional[float] = None,
    xtick_step_s: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    cax: Optional[plt.Axes] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> plt.Figure:
    if animal_id is not None:
        if "animal_id" not in scores_df.columns:
            fig, ax = plt.subplots(figsize=(10, 2)) if ax is None else (ax.figure, ax)
            ax.text(0.5, 0.5, "scores_df missing animal_id column", ha="center", va="center")
            ax.axis("off")
            return fig
        scores_df = scores_df[scores_df["animal_id"] == animal_id].copy()
        if scores_df.empty:
            fig, ax = plt.subplots(figsize=(10, 2)) if ax is None else (ax.figure, ax)
            ax.text(0.5, 0.5, f"No scores for animal_id={animal_id}", ha="center", va="center")
            ax.axis("off")
            return fig

    cfg_use = _resolve_scores_plot_cfg(
        scores_df,
        trial_col=trial_col,
        time_col=time_col,
        time_range=time_range,
        normalize_for_plot=normalize_for_plot,
        row_height=row_height,
        xtick_step_s=xtick_step_s,
        cfg=cfg,
    )
    results = {"cfg": cfg_use, "scored_ml": scores_df}
    return plot_one_trial_heatmap_row_from_results(
        results=results,
        trial_id=trial_id,
        value_col=value_col,
        time_range=time_range,
        normalize_for_plot=normalize_for_plot,
        row_height=row_height,
        xtick_step_s=xtick_step_s,
        ax=ax,
        cax=cax,
        cfg=cfg_use,
    )


def plot_one_trial_score_trace_from_scores(
    *,
    scores_df: pd.DataFrame,
    trial_id: int | float,
    value_col: str = "score",
    trial_col: Optional[str] = None,
    time_col: Optional[str] = None,
    animal_id: Optional[str] = None,
    time_range: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> plt.Figure:
    if animal_id is not None:
        if "animal_id" not in scores_df.columns:
            fig, ax = plt.subplots(figsize=(10, 2)) if ax is None else (ax.figure, ax)
            ax.text(0.5, 0.5, "scores_df missing animal_id column", ha="center", va="center")
            ax.axis("off")
            return fig
        scores_df = scores_df[scores_df["animal_id"] == animal_id].copy()
        if scores_df.empty:
            fig, ax = plt.subplots(figsize=(10, 2)) if ax is None else (ax.figure, ax)
            ax.text(0.5, 0.5, f"No scores for animal_id={animal_id}", ha="center", va="center")
            ax.axis("off")
            return fig

    cfg_use = _resolve_scores_plot_cfg(
        scores_df,
        trial_col=trial_col,
        time_col=time_col,
        time_range=time_range,
        cfg=cfg,
    )
    trial_col = cfg_use["TRIAL_COL"]
    time_col = cfg_use["TIME_COL"]
    time_range = time_range or tuple(cfg_use["PLOT_TIME_RANGE"])
    event_lines = tuple(cfg_use["EVENT_LINES"])

    if value_col not in scores_df.columns:
        fig, ax = plt.subplots(figsize=(10, 2)) if ax is None else (ax.figure, ax)
        ax.text(0.5, 0.5, f"Missing score column '{value_col}'", ha="center", va="center")
        ax.axis("off")
        return fig

    tid = float(trial_id)
    s = scores_df[pd.to_numeric(scores_df[trial_col], errors="coerce") == tid].copy()
    if s.empty:
        fig, ax = plt.subplots(figsize=(10, 2)) if ax is None else (ax.figure, ax)
        ax.text(0.5, 0.5, f"No data for trial_id={trial_id}", ha="center", va="center")
        ax.axis("off")
        return fig

    s[time_col] = pd.to_numeric(s[time_col], errors="coerce")
    s[value_col] = pd.to_numeric(s[value_col], errors="coerce")
    s = s[(s[time_col] >= time_range[0]) & (s[time_col] <= time_range[1])].copy()
    if s.empty:
        fig, ax = plt.subplots(figsize=(10, 2)) if ax is None else (ax.figure, ax)
        ax.text(0.5, 0.5, f"No bins for trial_id={trial_id}", ha="center", va="center")
        ax.axis("off")
        return fig

    s = s.groupby(time_col, as_index=False)[value_col].mean()
    s = s.sort_values(time_col)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(14, 4))
    else:
        fig = ax.figure

    ax.plot(s[time_col].to_numpy(), s[value_col].to_numpy(), color="black", linewidth=1.5)
    for tt, col in event_lines:
        ax.axvline(tt, linestyle="--", linewidth=1, color=col)

    title_tid = int(tid) if float(tid).is_integer() else tid
    ax.set_title(f"Trial {title_tid} — {value_col} trace")
    ax.set_xlabel("rel time to trial end (s)")
    ax.set_ylabel(value_col)
    ax.grid(True, alpha=0.2)
    if time_range:
        ax.set_xlim(float(time_range[0]), float(time_range[1]))
    if created_fig:
        fig.tight_layout()
    return fig


def plot_trial_heatmap_and_score_trace_from_scores(
    *,
    scores_df: pd.DataFrame,
    trial_id: int | float,
    value_col: str = "score",
    trial_col: Optional[str] = None,
    time_col: Optional[str] = None,
    animal_id: Optional[str] = None,
    time_range: Optional[Tuple[float, float]] = None,
    normalize_for_plot: Optional[str] = None,
    row_height: Optional[float] = None,
    xtick_step_s: Optional[float] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> plt.Figure:
    cfg_use = _resolve_scores_plot_cfg(
        scores_df,
        trial_col=trial_col,
        time_col=time_col,
        time_range=time_range,
        normalize_for_plot=normalize_for_plot,
        row_height=row_height,
        xtick_step_s=xtick_step_s,
        cfg=cfg,
    )
    time_range = time_range or tuple(cfg_use["PLOT_TIME_RANGE"])

    fig = plt.figure(figsize=(14, 7))
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(2, 2, width_ratios=[40, 0.35], height_ratios=[0.3, 3], wspace=0.02, hspace=0.3, figure=fig)
    ax_heat = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax_trace = fig.add_subplot(gs[1, 0])
    ax_blank = fig.add_subplot(gs[1, 1])
    ax_blank.axis("off")

    plot_one_trial_heatmap_row_from_scores(
        scores_df=scores_df,
        trial_id=trial_id,
        value_col=value_col,
        trial_col=trial_col,
        time_col=time_col,
        animal_id=animal_id,
        time_range=time_range,
        normalize_for_plot=normalize_for_plot,
        row_height=row_height,
        xtick_step_s=xtick_step_s,
        ax=ax_heat,
        cax=cax,
        cfg=cfg_use,
    )
    plot_one_trial_score_trace_from_scores(
        scores_df=scores_df,
        trial_id=trial_id,
        value_col=value_col,
        trial_col=trial_col,
        time_col=time_col,
        animal_id=animal_id,
        time_range=time_range,
        ax=ax_trace,
        cfg=cfg_use,
    )
    fig.tight_layout()
    return fig


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_results_bundle(
    results: Dict[str, Any],
    out_dir: Path | str,
    *,
    include_models: bool = False,
    prefix: str = "",
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tables = {
        "scored_ml": results.get("scored_ml"),
        "scored_classic": results.get("scored_classic"),
        "feat_ml": results.get("feat_ml"),
        "feat_classic": results.get("feat_classic"),
        "meta": results.get("meta"),
        "trial_sides_pred": results.get("trial_sides_pred"),
    }
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_parquet(out_path / f"{prefix}{name}.parquet", index=False)

    cfg = results.get("cfg")
    if cfg is not None:
        with open(out_path / f"{prefix}cfg.json", "w") as f:
            json.dump(cfg, f, default=_json_default)

    bundle_meta = {
        "bin_size": results.get("bin_size"),
        "has_models": bool(include_models),
        "align_x_by_side": results.get("align_x_by_side"),
        "run_side": results.get("run_side"),
    }
    with open(out_path / f"{prefix}bundle.json", "w") as f:
        json.dump(bundle_meta, f, default=_json_default)

    if include_models:
        model_obj = {
            "side_pipe": results.get("side_pipe"),
            "turn_pipe_ml": results.get("turn_pipe_ml"),
            "turn_pipe_classic": results.get("turn_pipe_classic"),
            "cfg": results.get("cfg"),
            "align_x_by_side": results.get("align_x_by_side"),
            "run_side": results.get("run_side"),
        }
        try:
            import joblib

            joblib.dump(model_obj, out_path / f"{prefix}models.joblib")
        except Exception:
            with open(out_path / f"{prefix}models.pkl", "wb") as f:
                pickle.dump(model_obj, f)

    return out_path


def load_results_bundle(
    bundle_dir: Path | str,
    *,
    prefix: str = "",
    tables: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    in_path = Path(bundle_dir)
    results: Dict[str, Any] = {}

    cfg_path = in_path / f"{prefix}cfg.json"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            results["cfg"] = json.load(f)

    meta_path = in_path / f"{prefix}bundle.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        results["bin_size"] = meta.get("bin_size")
        if "align_x_by_side" in meta:
            results["align_x_by_side"] = meta.get("align_x_by_side")
        if "run_side" in meta:
            results["run_side"] = meta.get("run_side")

    default_tables = [
        "scored_ml",
        "scored_classic",
        "feat_ml",
        "feat_classic",
        "meta",
        "trial_sides_pred",
    ]
    table_names = list(default_tables if tables is None else tables)
    for name in table_names:
        fpath = in_path / f"{prefix}{name}.parquet"
        if fpath.exists():
            results[name] = pd.read_parquet(fpath)

    models_joblib = in_path / f"{prefix}models.joblib"
    models_pkl = in_path / f"{prefix}models.pkl"
    if models_joblib.exists():
        try:
            import joblib

            models = joblib.load(models_joblib)
            results.update(models if isinstance(models, dict) else {})
        except Exception:
            pass
    elif models_pkl.exists():
        try:
            with open(models_pkl, "rb") as f:
                models = pickle.load(f)
            results.update(models if isinstance(models, dict) else {})
        except Exception:
            pass

    return results

# ============================================================
# Runner helpers
# ============================================================


def _pick_example_trial(labels_df, trial_col, valid_trial_ids=None):
    if labels_df.empty:
        return None
    vals = pd.to_numeric(labels_df[trial_col], errors="coerce").dropna()
    if valid_trial_ids is not None:
        vals = vals[vals.isin(valid_trial_ids)]
    if vals.empty:
        return None
    return int(vals.sample(n=1).iloc[0])


def run_models(
    window_stack: pd.DataFrame,
    *,
    labels: Sequence[TurnLabel | Dict[str, Any]],
    run_side: bool = True,
    side_method: str = "simple",
    side_method_kwargs: Optional[Dict[str, Any]] = None,
    run_classic: bool = True,
    run_ml: bool = True,
    plot: bool = True,
    cfg: Optional[Dict[str, Any]] = None,
    example_trial_id: Optional[int] = None,
    signals: Optional[List[str]] = None,
    use_new_side_v: bool = False,
    flip_head_angle_aligned: Optional[bool] = True,
    align_x_by_side: bool = True,
    run_loto: bool = True,
    loto_out_dir: Optional[str | Path] = "evaluations",
    loto_prefix: Optional[str] = None,
    loto_score_col: Optional[str] = None,
    loto_min_score: float = 0.0,
    loto_tol_s: float = 0.5,
    loto_peak_threshold: Optional[float] = None,
    loto_true_window_s: float = 0.2,
    
) -> Dict[str, Any]:
    cfg = resolve_cfg(cfg)
    trial_col = cfg["TRIAL_COL"]
    time_col = cfg["TIME_COL"]
    signals = signals or list(cfg["SIGNALS"])

    ws = _sanitize_window_stack(window_stack, cfg=cfg)
    labels_df = make_labels_df(labels, cfg=cfg)
    if labels_df.empty:
        raise ValueError("labels_list produced empty labels_df")
    meta = build_trial_meta_hit_only(ws, cfg=cfg)

    if run_side:
        side_method_use = str(side_method).strip().lower()
        side_kwargs = dict(side_method_kwargs or {})
        if side_method_use == "simple":
            side_pipe = fit_side_model_simple(ws, labels_df, bin_size=float(cfg["BIN_SIZE"]), cfg=cfg)
        elif side_method_use in {"sigmoid", "simple_sigmoid"}:
            feat_cfg = dict(
                bin_size=float(cfg["BIN_SIZE"]),
                baseline_win=tuple(_cfg(cfg, "SIDE_BASELINE_WIN")),
                post_win=tuple(_cfg(cfg, "SIDE_POST_WIN")),
                smooth_bins=int(_cfg(cfg, "SIDE_SMOOTH_BINS")),
                deriv_win_bins=int(_cfg(cfg, "SIDE_DERIV_WIN_BINS")),
                use_derivative=bool(_cfg(cfg, "SIDE_USE_DERIV")),
                right_is_positive=bool(_cfg(cfg, "SIDE_RIGHT_IS_POSITIVE")),
                time_col=time_col,
            )
            alpha = float(side_kwargs.get("alpha", 4.0))
            side_pipe = SimpleSideModel(alpha=alpha, feat_cfg=feat_cfg, trial_col=trial_col)
        elif side_method_use in {"sign", "simple_sign"}:
            feat_cfg = dict(
                bin_size=float(cfg["BIN_SIZE"]),
                baseline_win=tuple(_cfg(cfg, "SIDE_BASELINE_WIN")),
                post_win=tuple(_cfg(cfg, "SIDE_POST_WIN")),
                smooth_bins=int(_cfg(cfg, "SIDE_SMOOTH_BINS")),
                deriv_win_bins=int(_cfg(cfg, "SIDE_DERIV_WIN_BINS")),
                use_derivative=bool(_cfg(cfg, "SIDE_USE_DERIV")),
                right_is_positive=bool(_cfg(cfg, "SIDE_RIGHT_IS_POSITIVE")),
                time_col=time_col,
            )
            side_pipe = SignSideModel(feat_cfg=feat_cfg, trial_col=trial_col)
        elif side_method_use == "trial_classification":
            side_pipe = TrialClassificationSideModel(
                trial_col=trial_col,
                time_col=time_col,
                **side_kwargs,
            )
        else:
            raise ValueError(f"Unknown side_method: {side_method!r}")
        trial_sides_pred = side_pipe.predict_from_window_stack(ws, time_col=time_col)
    else:
        align_x_by_side = False
        side_pipe = None
        trial_sides_pred = None

    feat_classic = scored_classic = None
    if run_classic:
        feat_classic = build_classic_features(
            ws,
            trial_sides_pred,
            bin_size=float(cfg["BIN_SIZE"]),
            smooth_roll_bins=int(cfg["SMOOTH_ROLL_BINS_CLASSIC"]),
            cfg=cfg,
        )
        scored_classic = score_all_trials_classic(feat_classic, cfg=cfg)

    feat_ml = scored_ml = turn_pipe_ml = None
    if run_ml:
        feat_ml = build_ml_features_diff(
            ws,
            trial_sides_pred,
            bin_size=float(cfg["BIN_SIZE"]),
            signals=signals,
            smooth_roll_bins=int(cfg["ML_SMOOTH_ROLL_BINS"]),
            diff_k_bins=int(cfg["ML_DIFF_K_BINS"]),
            lags=int(cfg["ML_LAGS"]),
            method="convolution",
            cfg=cfg,
            align_x_by_side=align_x_by_side,
            use_new_side_v=use_new_side_v,
            flip_head_angle_aligned=flip_head_angle_aligned,
        )
        turn_pipe_ml = fit_turn_model_ml_diff(
            feat_ml,
            labels_df,
            train_time_range=tuple(cfg["TRAIN_TIME_RANGE"]),
            pos_half_width_s=float(cfg["POS_HALF_WIDTH_S"]),
            enforce_pos_after=float(cfg["ENFORCE_POS_AFTER"]),
            model_cfg=cfg["ML_TURN_MODEL"],
            class_balanced=bool(cfg["ML_USE_CLASS_BALANCED"]),
        )
        scored_ml = score_all_trials_from_turn_pipe_ml(
            feat_ml,
            turn_pipe_ml,
            baseline_win=tuple(cfg["ML_BASELINE_SUB_WIN"]),
            postproc_value_col="turn_score_rawprob",
            cfg=cfg,
        )

    loto_metrics = loto_summary = None
    if run_loto and run_ml:
        if loto_prefix is None:
            if "animal_id" in meta.columns and len(meta):
                loto_prefix = str(meta["animal_id"].iloc[0])
            else:
                loto_prefix = "loto"
        loto_metrics, loto_summary = run_loto_turn_evaluation(
            ws,
            labels=labels,
            cfg=cfg,
            run_side=run_side,
            side_method=side_method,
            side_method_kwargs=side_method_kwargs,
            align_x_by_side=align_x_by_side,
            use_new_side_v=use_new_side_v,
            flip_head_angle_aligned=flip_head_angle_aligned,
            score_col=loto_score_col,
            min_score=loto_min_score,
            tol_s=loto_tol_s,
            peak_threshold=loto_peak_threshold,
            true_window_s=loto_true_window_s,
            save_dir=loto_out_dir,
            save_prefix=loto_prefix,
        )

    figs: List[plt.Figure] = []
    if plot:
        animal_id = meta["animal_id"].iloc[0] if "animal_id" in meta.columns and len(meta) else "unknown"
        classic_heat_col = "turn_score_pp" if cfg["POSTPROC_APPLY_TO_HEATMAPS"] else "turn_score_nms"
        ml_heat_col = "turn_score_pp" if cfg["POSTPROC_APPLY_TO_HEATMAPS"] else "turn_score_nms"
        if run_ml and scored_ml is not None:
            figs.append(
                plot_reward_split_no_side(
                    scored_all=scored_ml,
                    meta=meta,
                    value_col=ml_heat_col,
                    time_range=tuple(cfg["PLOT_TIME_RANGE"]),
                    row_height=cfg["PLOT_ROW_HEIGHT"],
                    cmap="Reds",
                    title_prefix=f"{animal_id} turn detection",
                    which="no_reward",
                    normalize_for_plot=cfg["HEATMAP_NORMALIZE_FOR_PLOT"],
                    sort_rows=cfg["SORT_ROWS"],
                    xtick_step_s=cfg["HEATMAP_XTICK_STEP_S"],
                    cfg=cfg,
                )
            )
            figs.append(
                plot_reward_split_no_side(
                    scored_all=scored_ml,
                    meta=meta,
                    value_col=ml_heat_col,
                    time_range=tuple(cfg["PLOT_TIME_RANGE"]),
                    row_height=cfg["PLOT_ROW_HEIGHT"],
                    cmap="Reds",
                    title_prefix=f"{animal_id} turn detection",
                    which="reward",
                    normalize_for_plot=cfg["HEATMAP_NORMALIZE_FOR_PLOT"],
                    sort_rows=cfg["SORT_ROWS"],
                    xtick_step_s=cfg["HEATMAP_XTICK_STEP_S"],
                    cfg=cfg,
                )
            )
        if run_classic and scored_classic is not None:
            if run_side:
                figs += plot_reward_split(
                    scored_all=scored_classic,
                    meta=meta,
                    trial_sides_pred=trial_sides_pred,
                    value_col=classic_heat_col,
                    time_range=tuple(cfg["PLOT_TIME_RANGE"]),
                    row_height=cfg["PLOT_ROW_HEIGHT"],
                    cmap="Reds",
                    title_prefix=f"{animal_id} CLASSIC ({classic_heat_col})",
                    layout=cfg["HEATMAP_LAYOUT"],
                    normalize_for_plot=cfg["HEATMAP_NORMALIZE_FOR_PLOT"],
                    sort_rows=cfg["SORT_ROWS"],
                    cfg=cfg,
                )
            else:
                figs.append(
                    plot_reward_split_no_side(
                        scored_all=scored_classic,
                        meta=meta,
                        value_col=classic_heat_col,
                        time_range=tuple(cfg["PLOT_TIME_RANGE"]),
                        row_height=cfg["PLOT_ROW_HEIGHT"],
                        cmap="Reds",
                        title_prefix=f"{animal_id} CLASSIC ({classic_heat_col})",
                        which="both",
                        normalize_for_plot=cfg["HEATMAP_NORMALIZE_FOR_PLOT"],
                        sort_rows=cfg["SORT_ROWS"],
                        xtick_step_s=cfg["HEATMAP_XTICK_STEP_S"],
                        cfg=cfg,
                    )
                )
        if run_ml and scored_ml is not None:
            if run_side:
                figs += plot_reward_split(
                    scored_all=scored_ml,
                    meta=meta,
                    trial_sides_pred=trial_sides_pred,
                    value_col=ml_heat_col,
                    time_range=tuple(cfg["PLOT_TIME_RANGE"]),
                    row_height=cfg["PLOT_ROW_HEIGHT"],
                    cmap="Reds",
                    title_prefix=f"{animal_id} ({ml_heat_col})",
                    layout=cfg["HEATMAP_LAYOUT"],
                    normalize_for_plot=cfg["HEATMAP_NORMALIZE_FOR_PLOT"],
                    sort_rows=cfg["SORT_ROWS"],
                    cfg=cfg,
                )

        example_trial_id = example_trial_id or _pick_example_trial(labels_df, trial_col)
        if example_trial_id is not None:
            figs.append(
                plot_one_trial_scores_all3(
                    trial_id=example_trial_id,
                    scored_classic=scored_classic if scored_classic is not None else pd.DataFrame(),
                    scored_ml=scored_ml if scored_ml is not None else pd.DataFrame(),
                    labels_df=labels_df,
                    time_range=tuple(cfg["PLOT_TIME_RANGE"]),
                    cfg=cfg,
                )
            )
            if feat_ml is not None:
                figs.append(
                    plot_one_trial_raw_and_diffs(
                        trial_id=example_trial_id,
                        feat_ml=feat_ml,
                        signals=signals,
                        diff_k_bins=cfg["ML_DIFF_K_BINS"],
                        time_range=tuple(cfg["PLOT_TIME_RANGE"]),
                        event_lines=tuple(cfg["EVENT_LINES"]),
                        use_twin_axes=True,
                        cfg=cfg,
                    )
                )

    return dict(
        labels_df=labels_df,
        meta=meta,
        side_pipe=side_pipe,
        trial_sides_pred=trial_sides_pred,
        feat_classic=feat_classic,
        scored_classic=scored_classic,
        feat_ml=feat_ml,
        turn_pipe_ml=turn_pipe_ml,
        scored_ml=scored_ml,
        figs=figs,
        bin_size=float(cfg["BIN_SIZE"]),
        cfg=cfg,
        run_side=run_side,
        align_x_by_side=align_x_by_side,
        loto_metrics=loto_metrics,
        loto_summary=loto_summary,
    )


# ============================================================
# Evaluation helpers
# ============================================================


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[keep].astype(int)
    y_pred = y_pred[keep].astype(int)
    if y_true.size == 0:
        return float("nan")
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    if not np.isfinite(tpr) and not np.isfinite(tnr):
        return float("nan")
    return float(np.nanmean([tpr, tnr]))


def evaluate_side_and_turn(
    *,
    labels: Sequence[TurnLabel | Dict[str, Any]],
    trial_sides_pred: Optional[pd.DataFrame],
    scored_ml: Optional[pd.DataFrame],
    cfg: Optional[Dict[str, Any]] = None,
    score_col: Optional[str] = None,
    time_col: Optional[str] = None,
    min_score: float = 0.0,
    tol_s: float = 0.5,
    peak_threshold: Optional[float] = None,
    true_window_s: float = 0.2,
    return_tables: bool = False,
) -> Dict[str, Any]:
    cfg = resolve_cfg(cfg)
    trial_col = _cfg(cfg, "TRIAL_COL")
    labels_df = make_labels_df(labels, cfg=cfg)
    out: Dict[str, Any] = {}

    # ---- side model metrics ----
    side_metrics = {"n": 0, "accuracy": float("nan"), "balanced_accuracy": float("nan")}
    side_table = pd.DataFrame()
    if trial_sides_pred is not None and not labels_df.empty:
        cols = [trial_col] + [c for c in ["trial_side", "p_right"] if c in trial_sides_pred.columns]
        side_table = trial_sides_pred[cols].merge(
            labels_df[[trial_col, "y_side"]],
            on=trial_col,
            how="inner",
        )
        if not side_table.empty:
            pred_side = side_table["trial_side"].astype(str).str.lower()
            pred_y = (pred_side == "right").astype(int)
            y_true = pd.to_numeric(side_table["y_side"], errors="coerce").astype(float)
            keep = np.isfinite(y_true)
            if keep.any():
                y_true = y_true[keep].astype(int)
                pred_y = pred_y[keep].to_numpy()
                acc = float((pred_y == y_true.to_numpy()).mean())
                bal = _balanced_accuracy(y_true.to_numpy(), pred_y)
                side_metrics = {"n": int(y_true.size), "accuracy": acc, "balanced_accuracy": bal}
    out["side"] = side_metrics

    # ---- turn score metrics ----
    turn_metrics = {
        "n": 0,
        "n_pred": 0,
        "mae": float("nan"),
        "median_ae": float("nan"),
        "within_tol": float("nan"),
        "within_tol_any_peak": float("nan"),
        "any_peak_rate": float("nan"),
        "false_alarm_rate": float("nan"),
        "precision": float("nan"),
        "bin_detection": float("nan"),
        "bin_precision": float("nan"),
        "bin_false_alarm_rate": float("nan"),
        "bin_pos": 0,
        "bin_neg": 0,
        "bin_pred": 0,
        "bin_tp": 0,
        "bin_fp": 0,
        "peak_threshold": float(peak_threshold) if peak_threshold is not None else float("nan"),
        "true_window_mean": float("nan"),
        "true_window_median": float("nan"),
        "true_window_s": float(true_window_s),
        "tol_s": float(tol_s),
    }
    turn_table = pd.DataFrame()
    if scored_ml is not None and not labels_df.empty and not scored_ml.empty:
        score_col = score_col or ("turn_score_pp" if "turn_score_pp" in scored_ml.columns else None)
        score_col = score_col or ("turn_score_nms" if "turn_score_nms" in scored_ml.columns else None)
        score_col = score_col or ("turn_score_rawprob" if "turn_score_rawprob" in scored_ml.columns else None)
        time_col = time_col or ("time_bin" if "time_bin" in scored_ml.columns else _cfg(cfg, "TIME_COL"))
        if score_col and score_col in scored_ml.columns and time_col in scored_ml.columns:
            labels_map = labels_df.groupby(trial_col, sort=False)["turn_times_expect"].apply(_flatten_turn_times)
            rows = []
            peak_thr = peak_threshold
            if peak_thr is None and np.isfinite(min_score):
                peak_thr = float(min_score)
            bin_counts = {"pos": 0, "neg": 0, "pred": 0, "tp": 0, "fp": 0}
            for tid, g in scored_ml.groupby(trial_col, observed=False):
                if tid not in labels_map.index:
                    continue
                expected = labels_map.loc[tid]
                t = pd.to_numeric(g[time_col], errors="coerce").to_numpy()
                s = pd.to_numeric(g[score_col], errors="coerce").to_numpy()
                keep = np.isfinite(t) & np.isfinite(s)
                if np.isfinite(min_score):
                    keep = keep & (s >= float(min_score))
                pred_t = float("nan")
                if keep.any():
                    t_keep = t[keep]
                    s_keep = s[keep]
                    pred_t = float(t_keep[int(np.nanargmax(s_keep))])
                exp_times = [float(x) for x in expected if np.isfinite(x)]
                if peak_thr is not None and np.isfinite(peak_thr):
                    valid = np.isfinite(t) & np.isfinite(s)
                    if valid.any():
                        pos_mask = np.zeros_like(t, dtype=bool)
                        for gt in exp_times:
                            pos_mask |= np.abs(t - gt) <= float(tol_s)
                        pos_mask = pos_mask & valid
                        neg_mask = valid & ~pos_mask
                        pred_mask = valid & (s >= float(peak_thr))
                        bin_counts["pos"] += int(pos_mask.sum())
                        bin_counts["neg"] += int(neg_mask.sum())
                        bin_counts["pred"] += int(pred_mask.sum())
                        bin_counts["tp"] += int((pred_mask & pos_mask).sum())
                        bin_counts["fp"] += int((pred_mask & neg_mask).sum())
                if not exp_times:
                    continue
                best_err = float("nan")
                best_gt = float("nan")
                if exp_times and np.isfinite(pred_t):
                    errs = [abs(pred_t - gt) for gt in exp_times]
                    best_err = float(min(errs))
                    best_gt = float(exp_times[int(np.argmin(errs))])
                true_means = []
                for gt in exp_times:
                    if not np.isfinite(gt):
                        continue
                    m = np.isfinite(t) & np.isfinite(s) & (np.abs(t - gt) <= float(true_window_s))
                    if m.any():
                        true_means.append(float(np.nanmean(s[m])))
                true_window_mean = float(np.nanmean(true_means)) if true_means else float("nan")
                any_peak_ok = float("nan")
                any_peak_flag = float("nan")
                false_alarm_flag = float("nan")
                if exp_times and peak_thr is not None and np.isfinite(peak_thr):
                    peak_keep = np.isfinite(t) & np.isfinite(s) & (s >= float(peak_thr))
                    has_peak = bool(peak_keep.any())
                    any_peak = False
                    if peak_keep.any():
                        t_hits = t[peak_keep]
                        for gt in exp_times:
                            if np.any(np.abs(t_hits - gt) <= float(tol_s)):
                                any_peak = True
                                break
                    any_peak_ok = bool(any_peak)
                    any_peak_flag = has_peak
                    false_alarm_flag = bool(has_peak and not any_peak_ok)
                rows.append(
                    {
                        trial_col: tid,
                        "pred_time": pred_t,
                        "best_expected": best_gt,
                        "abs_error": best_err,
                        "within_tol": bool(np.isfinite(best_err) and best_err <= tol_s),
                        "any_peak_within_tol": any_peak_ok,
                        "any_peak": any_peak_flag,
                        "false_alarm": false_alarm_flag,
                        "true_window_mean": true_window_mean,
                    }
                )
            if rows:
                turn_table = pd.DataFrame(rows)
                errs = pd.to_numeric(turn_table["abs_error"], errors="coerce")
                pred_ok = pd.to_numeric(turn_table["pred_time"], errors="coerce").notna()
                any_peak_vals = pd.to_numeric(turn_table.get("any_peak_within_tol"), errors="coerce")
                peak_vals = pd.to_numeric(turn_table.get("any_peak"), errors="coerce")
                false_alarm_vals = pd.to_numeric(turn_table.get("false_alarm"), errors="coerce")
                true_means = pd.to_numeric(turn_table.get("true_window_mean"), errors="coerce")
                precision = float("nan")
                if peak_vals.notna().any():
                    peak_mask = peak_vals.astype(bool)
                    if peak_mask.any():
                        precision = float(turn_table.loc[peak_mask, "any_peak_within_tol"].mean())
                if errs.notna().any():
                    turn_metrics = {
                        "n": int(errs.notna().sum()),
                        "n_pred": int(pred_ok.sum()),
                        "mae": float(errs.mean()),
                        "median_ae": float(errs.median()),
                        "within_tol": float(turn_table.loc[errs.notna(), "within_tol"].mean()),
                        "within_tol_any_peak": float(any_peak_vals.mean()) if any_peak_vals.notna().any() else float("nan"),
                        "any_peak_rate": float(peak_vals.mean()) if peak_vals.notna().any() else float("nan"),
                        "false_alarm_rate": float(false_alarm_vals.mean()) if false_alarm_vals.notna().any() else float("nan"),
                        "precision": precision,
                        "peak_threshold": float(peak_thr) if peak_thr is not None else float("nan"),
                        "true_window_mean": float(true_means.mean()) if true_means.notna().any() else float("nan"),
                        "true_window_median": float(true_means.median()) if true_means.notna().any() else float("nan"),
                        "true_window_s": float(true_window_s),
                        "tol_s": float(tol_s),
                    }
            if bin_counts:
                bin_pos = float(bin_counts["pos"])
                bin_neg = float(bin_counts["neg"])
                bin_pred = float(bin_counts["pred"])
                bin_tp = float(bin_counts["tp"])
                bin_fp = float(bin_counts["fp"])
                turn_metrics.update(
                    {
                        "bin_detection": float(bin_tp / bin_pos) if bin_pos > 0 else float("nan"),
                        "bin_precision": float(bin_tp / bin_pred) if bin_pred > 0 else float("nan"),
                        "bin_false_alarm_rate": float(bin_fp / bin_neg) if bin_neg > 0 else float("nan"),
                        "bin_pos": int(bin_counts["pos"]),
                        "bin_neg": int(bin_counts["neg"]),
                        "bin_pred": int(bin_counts["pred"]),
                        "bin_tp": int(bin_counts["tp"]),
                        "bin_fp": int(bin_counts["fp"]),
                    }
                )
    out["turn"] = turn_metrics

    if return_tables:
        out["side_table"] = side_table
        out["turn_table"] = turn_table
    return out


def _wavg(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w, errors="coerce")
    m = np.isfinite(x) & np.isfinite(w)
    if not m.any():
        return float("nan")
    return float(np.average(x[m], weights=w[m]))


def sweep_turn_thresholds_for_animals(
    animals: Sequence[Any],
    *,
    results_dir: Path | str,
    thresholds: Optional[np.ndarray] = None,
    score_col: str = "turn_score_rawprob",
    tol_s: float = 1.0,
    true_window_s: float = 0.2,
) -> pd.DataFrame:
    results_dir = Path(results_dir)
    payloads: List[Tuple[Any, Dict[str, Any]]] = []
    for spec in animals:
        bundle_dir = results_dir / str(spec.key)
        if not bundle_dir.exists():
            continue
        payload = load_results_bundle(bundle_dir, tables=["scored_ml", "trial_sides_pred", "cfg"])
        if payload.get("cfg") is None or payload.get("scored_ml") is None:
            continue
        payloads.append((spec, payload))

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)

    rows: List[Dict[str, Any]] = []
    for thr in thresholds:
        for spec, payload in payloads:
            metrics = evaluate_side_and_turn(
                labels=spec.labels,
                trial_sides_pred=payload.get("trial_sides_pred"),
                scored_ml=payload.get("scored_ml"),
                cfg=payload.get("cfg"),
                score_col=score_col,
                tol_s=tol_s,
                peak_threshold=float(thr),
                true_window_s=true_window_s,
            )
            row = {"animal_id": str(spec.key), "threshold": float(thr)}
            row.update({f"turn_{k}": v for k, v in metrics.get("turn", {}).items()})
            rows.append(row)

    return pd.DataFrame(rows)


def build_threshold_summary(df_sweep: pd.DataFrame) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    if df_sweep.empty:
        return pd.DataFrame(), None

    use_bin = {
        "turn_bin_detection",
        "turn_bin_false_alarm_rate",
        "turn_bin_pos",
        "turn_bin_neg",
    }.issubset(df_sweep.columns)

    def _summary_row(g: pd.DataFrame) -> pd.Series:
        if use_bin:
            detection = _wavg(g["turn_bin_detection"], g["turn_bin_pos"])
            false_alarm = _wavg(g["turn_bin_false_alarm_rate"], g["turn_bin_neg"])
        else:
            detection = _wavg(g["turn_within_tol_any_peak"], g["turn_n"])
            false_alarm = _wavg(g["turn_false_alarm_rate"], g["turn_n"])
        trial_detection = _wavg(g["turn_within_tol_any_peak"], g["turn_n"])
        pred_weight = pd.to_numeric(g["turn_any_peak_rate"], errors="coerce") * pd.to_numeric(
            g["turn_n"], errors="coerce"
        )
        precision = _wavg(g["turn_precision"], pred_weight)
        return pd.Series(
            {
                "detection": detection,
                "false_alarm": false_alarm,
                "trial_detection": trial_detection,
                "precision": precision,
            }
        )

    summary = df_sweep.groupby("threshold", sort=False).apply(_summary_row).reset_index()
    summary["balanced"] = summary["detection"] - summary["false_alarm"]
    denom = summary["precision"] + summary["trial_detection"]
    summary["f1"] = np.where(
        np.isfinite(denom) & (denom > 0),
        2 * summary["precision"] * summary["trial_detection"] / denom,
        np.nan,
    )

    finite = summary[np.isfinite(summary["balanced"])]
    if finite.empty:
        best = summary.iloc[0]
    else:
        best = finite.sort_values(["balanced", "detection"], ascending=False).iloc[0]
    return summary, best


def plot_threshold_sweep(summary: pd.DataFrame, best: Optional[pd.Series], *, out_path: Path | str) -> None:
    if summary.empty:
        print("Skip threshold plot: empty sweep summary.")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(5, 3))
    # ax.plot(summary["threshold"], summary["detection"], label="detection")  # Removed
    ax.plot(summary["threshold"], summary["false_alarm"], label="false_alarm")
    ax.plot(summary["threshold"], summary["f1"], label="f1")
    if best is not None and np.isfinite(best.get("threshold", np.nan)):
        ax.axvline(
            float(best["threshold"]),
            color="C1",
            linestyle="--",
            label=f"best={best['threshold']:.2f}",
        )
    ax.set_xlabel("Threshold")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved threshold plot to {out_path}")


def _labels_df_to_turn_label_list(labels_df: pd.DataFrame, trial_col: str) -> List[Dict[str, Any]]:
    if labels_df.empty:
        return []
    out: List[Dict[str, Any]] = []
    for row in labels_df[[trial_col, "trial_side", "turn_times_expect", "w"]].to_dict("records"):
        tid = pd.to_numeric(row.get(trial_col), errors="coerce")
        if not np.isfinite(tid):
            continue
        out.append(
            {
                trial_col: float(tid),
                "trial_side": row.get("trial_side"),
                "turn_times_expect": row.get("turn_times_expect", []),
                "w": float(row.get("w", 1.0)),
            }
        )
    return out


def run_loto_turn_evaluation(
    window_stack: pd.DataFrame,
    *,
    labels: Sequence[TurnLabel | Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    run_side: bool = True,
    side_method: str = "simple",
    side_method_kwargs: Optional[Dict[str, Any]] = None,
    align_x_by_side: bool = True,
    use_new_side_v: bool = False,
    flip_head_angle_aligned: Optional[bool] = True,
    score_col: Optional[str] = None,
    min_score: float = 0.0,
    tol_s: float = 0.5,
    peak_threshold: Optional[float] = None,
    true_window_s: float = 0.2,
    save_dir: Optional[str | Path] = "evaluations",
    save_prefix: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = resolve_cfg(cfg)
    trial_col = cfg["TRIAL_COL"]
    labels_df = make_labels_df(labels, cfg=cfg)
    if labels_df.empty:
        return pd.DataFrame(), {"n": 0}

    labels_list = _labels_df_to_turn_label_list(labels_df, trial_col)
    trial_ids = (
        pd.to_numeric(labels_df[trial_col], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    rows = []
    for tid in sorted(trial_ids):
        train_labels = [l for l in labels_list if int(l[trial_col]) != int(tid)]
        test_labels = [l for l in labels_list if int(l[trial_col]) == int(tid)]
        if not train_labels or not test_labels:
            continue

        results = run_models(
            window_stack,
            labels=train_labels,
            run_side=run_side,
            side_method=side_method,
            side_method_kwargs=side_method_kwargs,
            run_classic=False,
            run_ml=True,
            plot=False,
            cfg=cfg,
            use_new_side_v=use_new_side_v,
            flip_head_angle_aligned=flip_head_angle_aligned,
            align_x_by_side=align_x_by_side,
            run_loto=False,
            loto_out_dir=None,
        )

        mask = pd.to_numeric(window_stack[trial_col], errors="coerce") == tid
        trial_df = window_stack.loc[mask].copy()
        if trial_df.empty:
            continue

        applied = score_new_trials_with_ml_model(
            trial_df,
            results=results,
            cfg=cfg,
            use_new_side_v=use_new_side_v,
            flip_head_angle_aligned=flip_head_angle_aligned,
            align_x_by_side=align_x_by_side,
        )
        eval_out = evaluate_side_and_turn(
            labels=test_labels,
            trial_sides_pred=applied.get("trial_sides_pred"),
            scored_ml=applied.get("scored_ml"),
            cfg=cfg,
            score_col=score_col,
            min_score=min_score,
            tol_s=tol_s,
            peak_threshold=peak_threshold,
            true_window_s=true_window_s,
            return_tables=True,
        )

        row = {trial_col: int(tid)}
        side_table = eval_out.get("side_table")
        if isinstance(side_table, pd.DataFrame) and not side_table.empty:
            pred_side = str(side_table["trial_side"].iloc[0]).lower()
            y_side = pd.to_numeric(side_table["y_side"].iloc[0], errors="coerce")
            row["side_pred"] = pred_side
            row["side_p_right"] = float(side_table["p_right"].iloc[0]) if "p_right" in side_table.columns else np.nan
            row["side_true"] = "right" if y_side == 1 else ("left" if y_side == 0 else np.nan)
            row["side_correct"] = bool((pred_side == "right") == (y_side == 1)) if np.isfinite(y_side) else np.nan
        else:
            row["side_pred"] = np.nan
            row["side_p_right"] = np.nan
            row["side_true"] = np.nan
            row["side_correct"] = np.nan

        turn_table = eval_out.get("turn_table")
        if isinstance(turn_table, pd.DataFrame) and not turn_table.empty:
            r = turn_table.iloc[0]
            row["pred_time"] = float(r.get("pred_time", np.nan))
            row["best_expected"] = float(r.get("best_expected", np.nan))
            row["abs_error"] = float(r.get("abs_error", np.nan))
            row["within_tol"] = bool(r.get("within_tol", False))
            row["any_peak_within_tol"] = r.get("any_peak_within_tol", np.nan)
            row["true_window_mean"] = r.get("true_window_mean", np.nan)
        else:
            row["pred_time"] = np.nan
            row["best_expected"] = np.nan
            row["abs_error"] = np.nan
            row["within_tol"] = np.nan
            row["any_peak_within_tol"] = np.nan
            row["true_window_mean"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    summary = {
        "n": int(len(df)),
        "side_accuracy": float(df["side_correct"].mean()) if "side_correct" in df.columns else float("nan"),
        "turn_mae": float(df["abs_error"].mean()) if "abs_error" in df.columns else float("nan"),
        "turn_median_ae": float(df["abs_error"].median()) if "abs_error" in df.columns else float("nan"),
        "turn_within_tol": float(df["within_tol"].mean()) if "within_tol" in df.columns else float("nan"),
        "turn_within_tol_any_peak": float(df["any_peak_within_tol"].mean()) if "any_peak_within_tol" in df.columns else float("nan"),
        "peak_threshold": float(peak_threshold) if peak_threshold is not None else float("nan"),
        "turn_true_window_mean": float(df["true_window_mean"].mean()) if "true_window_mean" in df.columns else float("nan"),
        "turn_true_window_median": float(df["true_window_mean"].median()) if "true_window_mean" in df.columns else float("nan"),
        "true_window_s": float(true_window_s),
        "tol_s": float(tol_s),
    }

    if save_dir:
        eval_dir = Path(save_dir)
        if eval_dir.name != "evaluations":
            eval_dir = eval_dir / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{save_prefix}_loto" if save_prefix else "loto"
        df.to_parquet(eval_dir / f"{stem}.parquet", index=False)
        with open(eval_dir / f"{stem}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return df, summary


# Apply trained side/turn models to new trials without refitting.
def score_new_trials_with_ml_model(
    window_stack: pd.DataFrame,
    *,
    results: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    signals: Optional[List[str]] = None,
    use_new_side_v: bool = False,
    flip_head_angle_aligned: Optional[bool] = True,
    align_x_by_side: Optional[bool] = None,
) -> Dict[str, Any]:
    cfg = resolve_cfg(cfg or results.get("cfg"))
    ws = _sanitize_window_stack(window_stack, cfg=cfg)
    trial_col = cfg["TRIAL_COL"]
    time_col = cfg["TIME_COL"]

    side_pipe = results.get("side_pipe")
    turn_pipe_ml = results.get("turn_pipe_ml")
    if turn_pipe_ml is None:
        raise ValueError("results must include turn_pipe_ml (run_models with run_ml=True)")

    if align_x_by_side is None:
        align_x_by_side = bool(results.get("align_x_by_side", True))
    run_side = bool(results.get("run_side", True))
    if not run_side:
        align_x_by_side = False

    trial_sides_pred = None
    if run_side:
        if side_pipe is None:
            raise ValueError("results must include side_pipe when run_side=True")
        trial_sides_pred = side_pipe.predict_from_window_stack(ws, time_col=time_col)
    signals = signals or list(cfg["SIGNALS"])
    feat_ml = build_ml_features_diff(
        ws,
        trial_sides_pred,
        bin_size=float(cfg["BIN_SIZE"]),
        signals=signals,
        smooth_roll_bins=int(cfg["ML_SMOOTH_ROLL_BINS"]),
        diff_k_bins=int(cfg["ML_DIFF_K_BINS"]),
        lags=int(cfg["ML_LAGS"]),
        method="convolution",
        cfg=cfg,
        align_x_by_side=align_x_by_side,
        use_new_side_v=use_new_side_v,
        flip_head_angle_aligned=flip_head_angle_aligned,
    )
    scored_ml = score_all_trials_from_turn_pipe_ml(
        feat_ml,
        turn_pipe_ml,
        baseline_win=tuple(cfg["ML_BASELINE_SUB_WIN"]),
        postproc_value_col="turn_score_rawprob",
        cfg=cfg,
    )
    return dict(
        trial_sides_pred=trial_sides_pred,
        feat_ml=feat_ml,
        scored_ml=scored_ml,
        cfg=cfg,
    )


# wrappers for single-model runs
def run_ml_only(
    window_stack: pd.DataFrame,
    *,
    labels: Sequence[TurnLabel | Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    use_new_side_v: bool = False,
    flip_head_angle_aligned: Optional[bool] = True,
    align_x_by_side: bool = True,
    run_side: bool = True,
    side_method: str = "simple",
    side_method_kwargs: Optional[Dict[str, Any]] = None,
    run_loto: bool = True,
    loto_out_dir: Optional[str | Path] = "evaluations",
    loto_prefix: Optional[str] = None,
    loto_score_col: Optional[str] = None,
    loto_min_score: float = 0.0,
    loto_tol_s: float = 0.5,
    loto_peak_threshold: Optional[float] = None,
    loto_true_window_s: float = 0.2,
):
    return run_models(
        window_stack,
        labels=labels,
        run_side=run_side,
        side_method=side_method,
        side_method_kwargs=side_method_kwargs,
        run_classic=False,
        run_ml=True,
        plot=False,
        cfg=cfg,
        use_new_side_v=use_new_side_v,
        flip_head_angle_aligned=flip_head_angle_aligned,
        align_x_by_side=align_x_by_side,
        run_loto=run_loto,
        loto_out_dir=loto_out_dir,
        loto_prefix=loto_prefix,
        loto_score_col=loto_score_col,
        loto_min_score=loto_min_score,
        loto_tol_s=loto_tol_s,
        loto_peak_threshold=loto_peak_threshold,
        loto_true_window_s=loto_true_window_s,
    )


def run_classic_only(
    window_stack: pd.DataFrame,
    *,
    labels: Sequence[TurnLabel | Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    run_side: bool = True,
    side_method: str = "simple",
    side_method_kwargs: Optional[Dict[str, Any]] = None,
):
    return run_models(
        window_stack,
        labels=labels,
        run_side=run_side,
        side_method=side_method,
        side_method_kwargs=side_method_kwargs,
        run_classic=True,
        run_ml=False,
        plot=False,
        cfg=cfg,
        run_loto=False,
        loto_out_dir=None,
    )


def plot_from_results(
    *,
    scored_classic: Optional[pd.DataFrame],
    scored_ml: Optional[pd.DataFrame],
    trial_sides_pred: Optional[pd.DataFrame],
    meta: pd.DataFrame,
    labels_df: Optional[pd.DataFrame] = None,
    which: Literal["both", "reward", "no_reward"] = "both",
    trial_ids: Optional[Sequence[Any]] = None,
    split_by_side: bool = True,
    cfg: Optional[Dict[str, Any]] = None,
) -> List[plt.Figure]:
    cfg = resolve_cfg(cfg)
    figs: List[plt.Figure] = []
    trial_col = _cfg(cfg, "TRIAL_COL")
    classic_heat_col = "turn_score_pp" if cfg["POSTPROC_APPLY_TO_HEATMAPS"] else "turn_score_nms"
    ml_heat_col = "turn_score_pp" if cfg["POSTPROC_APPLY_TO_HEATMAPS"] else "turn_score_nms"
    plot_zero_below = cfg.get("PLOT_ZERO_BELOW", None)

    def _zero_below(scored: Optional[pd.DataFrame], value_col: str) -> Optional[pd.DataFrame]:
        if scored is None or not isinstance(scored, pd.DataFrame) or scored.empty:
            return scored
        if value_col not in scored.columns or plot_zero_below is None:
            return scored
        try:
            thr = float(plot_zero_below)
        except (TypeError, ValueError):
            return scored
        out = scored.copy()
        vals = pd.to_numeric(out[value_col], errors="coerce")
        mask = np.isfinite(vals) & (vals < thr)
        if mask.any():
            out.loc[mask, value_col] = 0.0
        return out

    if trial_ids is not None:
        trial_ids_set = set(trial_ids)
        if isinstance(scored_classic, pd.DataFrame) and trial_col in scored_classic.columns:
            scored_classic = scored_classic[scored_classic[trial_col].isin(trial_ids_set)].copy()
        if isinstance(scored_ml, pd.DataFrame) and trial_col in scored_ml.columns:
            scored_ml = scored_ml[scored_ml[trial_col].isin(trial_ids_set)].copy()
        if isinstance(meta, pd.DataFrame) and trial_col in meta.columns:
            meta = meta[meta[trial_col].isin(trial_ids_set)].copy()
        if isinstance(trial_sides_pred, pd.DataFrame) and trial_col in trial_sides_pred.columns:
            trial_sides_pred = trial_sides_pred[trial_sides_pred[trial_col].isin(trial_ids_set)].copy()
        if isinstance(labels_df, pd.DataFrame) and trial_col in labels_df.columns:
            labels_df = labels_df[labels_df[trial_col].isin(trial_ids_set)].copy()

    if plot_zero_below is not None:
        scored_classic = _zero_below(scored_classic, classic_heat_col)
        scored_ml = _zero_below(scored_ml, ml_heat_col)

    has_classic = isinstance(scored_classic, pd.DataFrame) and not scored_classic.empty
    has_ml = isinstance(scored_ml, pd.DataFrame) and not scored_ml.empty
    animal_id = meta["animal_id"].iloc[0] if "animal_id" in meta.columns and len(meta) else "unknown"
    if split_by_side:
        if not isinstance(trial_sides_pred, pd.DataFrame) or trial_sides_pred.empty:
            split_by_side = False
    
    if has_classic:
        if split_by_side:
            figs += plot_reward_split(
                scored_all=scored_classic,
                meta=meta,
                trial_sides_pred=trial_sides_pred,
                value_col=classic_heat_col,
                time_range=tuple(cfg["PLOT_TIME_RANGE"]),
                row_height=cfg["PLOT_ROW_HEIGHT"],
                cmap="Reds",
                title_prefix=f"CLASSIC ({classic_heat_col})",
                layout=cfg["HEATMAP_LAYOUT"],
                normalize_for_plot=cfg["HEATMAP_NORMALIZE_FOR_PLOT"],
                sort_rows=cfg["SORT_ROWS"],
                which=which,
                cfg=cfg,
            )
        else:
            figs.append(
                plot_reward_split_no_side(
                    scored_all=scored_classic,
                    meta=meta,
                    value_col=classic_heat_col,
                    time_range=tuple(cfg["PLOT_TIME_RANGE"]),
                    row_height=cfg["PLOT_ROW_HEIGHT"],
                    title_prefix=f"CLASSIC ({classic_heat_col})",
                    which=which,
                    normalize_for_plot=cfg["HEATMAP_NORMALIZE_FOR_PLOT"],
                    sort_rows=cfg["SORT_ROWS"],
                    cmap="Reds",
                    xtick_step_s=cfg["HEATMAP_XTICK_STEP_S"],
                    cfg=cfg,
                )
            )
    if has_ml:
        if split_by_side:
            figs += plot_reward_split(
                scored_all=scored_ml,
                meta=meta,
                trial_sides_pred=trial_sides_pred,
                value_col=ml_heat_col,
                time_range=tuple(cfg["PLOT_TIME_RANGE"]),
                row_height=cfg["PLOT_ROW_HEIGHT"],
                cmap="Reds",
                title_prefix=f"{animal_id} ({ml_heat_col})",
                layout=cfg["HEATMAP_LAYOUT"],
                normalize_for_plot=cfg["HEATMAP_NORMALIZE_FOR_PLOT"],
                sort_rows=cfg["SORT_ROWS"],
                which=which,
                cfg=cfg,
            )
        else:
            figs.append(
                plot_reward_split_no_side(
                    scored_all=scored_ml,
                    meta=meta,
                    value_col=ml_heat_col,
                    time_range=tuple(cfg["PLOT_TIME_RANGE"]),
                    row_height=cfg["PLOT_ROW_HEIGHT"],
                    title_prefix=f"{animal_id} ({ml_heat_col})",
                    which=which,
                    normalize_for_plot=cfg["HEATMAP_NORMALIZE_FOR_PLOT"],
                    sort_rows=cfg["SORT_ROWS"],
                    cmap="Reds",
                    xtick_step_s=cfg["HEATMAP_XTICK_STEP_S"],
                    cfg=cfg,
                )
            )
    if scored_classic is not None or scored_ml is not None:
        labels_df_nonnull = labels_df if labels_df is not None else pd.DataFrame()
        example_trial_id = _pick_example_trial(labels_df_nonnull, cfg["TRIAL_COL"])
        if example_trial_id is not None and scored_ml is not None:
            figs.append(
                plot_one_trial_scores_all3(
                    trial_id=example_trial_id,
                    scored_classic=scored_classic if scored_classic is not None else pd.DataFrame(),
                    scored_ml=scored_ml if scored_ml is not None else pd.DataFrame(),
                    labels_df=labels_df,
                    time_range=tuple(cfg["PLOT_TIME_RANGE"]),
                    cfg=cfg,
                )
            )
    return figs

def build_peak_aligned_mean_results(
    results,
    *,
    animal_id=None,
    time_range=(-4, 6),
    value_col=None,
    min_score=None,
    heatmap_min_score=None,
    mean_trial_id=-1,
    recenter_mean_peak=False,
    abs_features=False,
):
    cfg = resolve_cfg(results.get("cfg"))
    trial_col = cfg["TRIAL_COL"]

    scored = results.get("scored_ml")
    if scored is None or scored.empty:
        scored = results.get("scored_classic")
    if scored is None or scored.empty:
        raise ValueError("No scored data in results")

    feat = results.get("feat_ml")
    if feat is None or feat.empty:
        raise ValueError("No feat_ml data in results")

    time_col = "time_bin" if "time_bin" in scored.columns else cfg["TIME_COL"]
    if time_col not in scored.columns:
        raise ValueError(f"time column '{time_col}' not found in scored data")

    meta = results.get("meta")
    if animal_id is not None and isinstance(meta, pd.DataFrame) and "animal_id" in meta.columns:
        keep_ids = meta.loc[meta["animal_id"] == animal_id, trial_col].dropna().unique().tolist()
        if keep_ids:
            scored = scored[scored[trial_col].isin(keep_ids)].copy()
            feat = feat[feat[trial_col].isin(keep_ids)].copy()
        else:
            print(f"[build_peak_aligned_mean_results] animal_id={animal_id} not found in meta; skipping filter")

    def _make_score_table(score_col):
        if score_col not in scored.columns:
            return pd.DataFrame()
        s = scored[[trial_col, time_col, score_col]].copy()
        s[time_col] = pd.to_numeric(s[time_col], errors="coerce")
        s[score_col] = pd.to_numeric(s[score_col], errors="coerce")
        if min_score is not None:
            s = s[s[score_col] >= float(min_score)]
        s = s.dropna(subset=[trial_col, time_col, score_col])
        return s

    candidates = []
    if value_col:
        candidates.append(value_col)
    if cfg.get("POSTPROC_APPLY_TO_HEATMAPS"):
        candidates.append("turn_score_pp")
    candidates += ["turn_score_pp", "turn_score_nms", "turn_score_rawprob"]
    candidates = [c for i, c in enumerate(candidates) if c and c in scored.columns and c not in candidates[:i]]
    if not candidates:
        raise ValueError("No score column found in scored data")

    s = pd.DataFrame()
    for cand in candidates:
        s = _make_score_table(cand)
        if not s.empty:
            value_col = cand
            break
    if s.empty:
        raise ValueError("No valid scored rows after filtering; check animal_id and score columns")

    bin_size = float(results.get("bin_size") or cfg.get("BIN_SIZE") or 0.05)

    peaks = (
        s.loc[s.groupby(trial_col)[value_col].idxmax(), [trial_col, time_col]]
        .rename(columns={time_col: "peak_time"})
        .reset_index(drop=True)
    )

    def _align_time(df, tcol):
        df = df.merge(peaks, on=trial_col, how="inner")
        df["time_rel"] = df[tcol] - df["peak_time"]
        df["time_rel"] = np.round(df["time_rel"] / bin_size) * bin_size
        df = df[(df["time_rel"] >= time_range[0]) & (df["time_rel"] <= time_range[1])]
        return df

    s_aligned = _align_time(s, time_col)
    mean_scores = (
        s_aligned.groupby("time_rel", observed=False)[value_col]
        .mean()
        .reset_index()
        .rename(columns={"time_rel": "time_bin"})
    )
    mean_scores[trial_col] = mean_trial_id
    mean_scores[cfg["TIME_COL"]] = mean_scores["time_bin"]

    f_aligned = _align_time(feat, "time_bin")
    feat_cols = getattr(results.get("turn_pipe_ml"), "_feat_cols", None)
    if not feat_cols:
        feat_cols = [c for c in f_aligned.columns if c.startswith("ml__")]
    feat_cols = [c for c in feat_cols if c in f_aligned.columns]
    if not feat_cols:
        raise ValueError("No feature columns found after alignment")

    if abs_features:
        x_diff_cols = []
        for c in feat_cols:
            if "__diff" not in c or "__abs" in c:
                continue
            base = str(c).replace("ml__", "").split("__")[0]
            if base.endswith("_x"):
                x_diff_cols.append(c)
        if x_diff_cols:
            f_aligned[x_diff_cols] = f_aligned[x_diff_cols].abs()
    mean_feat = (
        f_aligned.groupby("time_rel", observed=False)[feat_cols]
        .mean()
        .reset_index()
        .rename(columns={"time_rel": "time_bin"})
    )
    mean_feat[trial_col] = mean_trial_id
    mean_feat[cfg["TIME_COL"]] = mean_feat["time_bin"]

    if recenter_mean_peak and not mean_scores.empty:
        peak_at = float(mean_scores.loc[mean_scores[value_col].idxmax(), "time_bin"])
        mean_scores["time_bin"] = mean_scores["time_bin"] - peak_at
        mean_scores[cfg["TIME_COL"]] = mean_scores[cfg["TIME_COL"]] - peak_at
        mean_feat["time_bin"] = mean_feat["time_bin"] - peak_at
        mean_feat[cfg["TIME_COL"]] = mean_feat[cfg["TIME_COL"]] - peak_at

    if heatmap_min_score is not None:
        thr = float(heatmap_min_score)
        mean_scores[value_col] = mean_scores[value_col].where(mean_scores[value_col] >= thr, 0.0)
    mean_results = dict(results)
    mean_results["scored_ml"] = mean_scores
    mean_results["feat_ml"] = mean_feat
    return mean_results, mean_trial_id, peaks





def build_turn_labels_and_scores(res, score_col="turn_score_rawprob"):
    cfg = resolve_cfg(res.get("cfg"))
    labels_df = res.get("labels_df")
    feat_ml = res.get("feat_ml")
    scored_ml = res.get("scored_ml")

    if labels_df is None or labels_df.empty:
        raise ValueError("labels_df missing/empty")
    if feat_ml is None or feat_ml.empty:
        raise ValueError("feat_ml missing/empty")
    if scored_ml is None or scored_ml.empty:
        raise ValueError("scored_ml missing/empty")
    if score_col not in scored_ml.columns:
        raise ValueError(f"{score_col} not in scored_ml")

    # build per-bin labels (same rule used for training)
    ds = make_turn_labels_multi_for_bins_v2(
        feat_ml,
        labels_df,
        train_time_range=tuple(cfg["TRAIN_TIME_RANGE"]),
        pos_half_width_s=float(cfg["POS_HALF_WIDTH_S"]),
        enforce_pos_after=float(cfg["ENFORCE_POS_AFTER"]),
        hit_time=-2.0,
        cfg=cfg,
    )

    trial_col = cfg["TRIAL_COL"]
    y = ds[[trial_col, "time_bin", "y_turn"]].copy()
    s = scored_ml[[trial_col, "time_bin", score_col]].copy()

    m = y.merge(s, on=[trial_col, "time_bin"], how="inner").dropna()
    y_true = m["y_turn"].astype(int).to_numpy()
    y_score = m[score_col].astype(float).to_numpy()

    if np.unique(y_true).size < 2:
        return None, None

    return y_true, y_score

def compute_turn_pr_auc(res, score_col="turn_score_pp"):
    y_true, y_score = build_turn_labels_and_scores(res, score_col=score_col)
    if y_true is None:
        return np.nan, None, None, None

    ap = average_precision_score(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return ap, precision, recall, thresholds



# ============================================================
# Usage 
# ============================================================
#
# labels = [
#     TurnLabel(trial_id=10087, trial_side="left", turn_times_expect=[3.0, 10.5], strength="strong"),
#     TurnLabel(trial_id=10083, trial_side="right", turn_times_expect=[5.0], strength="full"),
# ]
# cfg = {"BIN_SIZE": 0.05, "ML_DIFF_K_BINS": 8}
# # from another script (non-notebook) or notebook cell:
# # from expectations.detect_turn import TurnLabel, run_models, run_ml_only, run_classic_only, plot_from_results
# # results = run_models(window_stack, labels=labels, cfg=cfg, plot=True)
#
# # 1) Full run (side + classic + ml) with figures:
# # results = run_models(window_stack, labels=labels, cfg=cfg, plot=True)
# #
# # 2) ML only (no classic), later plot figs:
# # ml_only = run_ml_only(window_stack, labels=labels, cfg={"ML_LAGS": 3})
# # figs = plot_from_results(
# #     scored_classic=None,
# #     scored_ml=ml_only["scored_ml"],
# #     trial_sides_pred=ml_only["trial_sides_pred"],
# #     meta=ml_only["meta"],
# #     labels_df=ml_only["labels_df"],
# #     cfg=cfg,
# # )
# #
# # 3) Classic only:
# # classic_only = run_classic_only(window_stack, labels=labels, cfg={"SMOOTH_ROLL_BINS_CLASSIC": 3})
# # plt.show(classic_only["scored_classic"].head())  # inspect raw scores per bin
#

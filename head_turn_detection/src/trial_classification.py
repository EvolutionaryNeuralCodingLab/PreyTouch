"""
Utilities to classify per-trial lateral turning (left/right/forward) based on nose_x excursion.

The core logic looks at the *relative* lateral displacement in a specified window
around the end of a trial (time aligned by `time_relative_to_end`, where 0 is
trial end and negative values are earlier).

This variant uses the **mean relative displacement** (mean of values minus an
initial baseline within the window) as the main scalar for classification,
rather than the single max excursion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _smooth_series_per_trial(series: pd.Series, trial_ids: pd.Series, window: int) -> pd.Series:
    """Rolling-mean smoother applied per trial_id."""
    if window is None or window <= 1:
        return series
    return (
        series.groupby(trial_ids)
        .transform(lambda v: v.rolling(window, center=True, min_periods=1).mean())
    )


def classify_trials_mean_displacement(
    df: pd.DataFrame,
    col: str = "nose_x",
    time_col: str = "time_relative_to_end",
    return_col_name: str = "side_label",
    time_window: tuple[float, float] = (-2.0, -0.5),
    roll_window: int = 5,
    x_thresh: float = 8.0,
    min_samples_beyond: int = 3,
    min_samples_window: int = 5,
) -> pd.DataFrame:
    """
    Classify each trial as 'left', 'right', or 'forward' based on the mean RELATIVE
    lateral displacement of a column (default nose_x) within a time window
    before trial end.

    Steps (per trial):
      1) Smooth the column with a centered rolling mean (optional).
      2) Restrict to a time window (default -2.0s to -0.5s relative to end).
      3) Use the median of the earliest few samples in the window as baseline.
      4) Compute relative displacement: x_rel = smooth_col - baseline.
      5) Take the mean of x_rel over the window (x_mean).
      6) If |x_mean| < x_thresh -> 'forward'.
      7) Otherwise, treat the sign of x_mean as left/right candidate, and
         require at least `min_samples_beyond` samples with x_rel beyond
         ±x_thresh in that same direction. If insufficient, fall back to 'forward'.

      If there are fewer than `min_samples_window` samples in the window,
      the trial is labeled as 'insufficient'.

    Returns a dataframe with one row per trial_id containing:
      - x_mean: mean relative displacement in the window
      - side_label: 'left'/'right'/'forward'/'insufficient'
      - samples_in_window: number of samples in the time window
      - samples_beyond_thresh: samples supporting the chosen direction
      - baseline: baseline position used to define x_rel
    """
    required = {"trial_id", time_col, col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns for classification: {sorted(missing)}")

    # Work on a minimal copy
    work = df[[*required]].copy()
    work = work.sort_values(["trial_id", time_col])
    work[f"{col}_smooth"] = _smooth_series_per_trial(work[col], work["trial_id"], roll_window)

    # Restrict to time window
    mask_win = (work[time_col] >= time_window[0]) & (work[time_col] <= time_window[1])
    win = work.loc[mask_win].copy()

    def _summarise_trial(trial: pd.DataFrame) -> pd.Series:
        n_samples = trial.shape[0]
        if n_samples < min_samples_window:
            return pd.Series(
                {
                    "x_mean": np.nan,
                    "side_label": "insufficient",
                    "samples_in_window": n_samples,
                    "samples_beyond_thresh": 0,
                    "baseline": np.nan,
                }
            )

        # Baseline from earliest few samples in the window (by time)
        earliest = trial.nsmallest(min(5, n_samples), time_col)
        baseline_vals = pd.to_numeric(earliest[f"{col}_smooth"], errors="coerce")
        finite_baseline = baseline_vals[np.isfinite(baseline_vals)]
        if finite_baseline.empty:
            return pd.Series(
                {
                    "x_mean": np.nan,
                    "side_label": "insufficient",
                    "samples_in_window": n_samples,
                    "samples_beyond_thresh": 0,
                    "baseline": np.nan,
                }
            )
        baseline = float(np.nanmedian(finite_baseline))

        if not np.isfinite(baseline):
            return pd.Series(
                {
                    "x_mean": np.nan,
                    "side_label": "insufficient",
                    "samples_in_window": n_samples,
                    "samples_beyond_thresh": 0,
                    "baseline": baseline,
                }
            )

        x_rel = pd.to_numeric(trial[f"{col}_smooth"], errors="coerce").to_numpy() - baseline
        finite_rel = x_rel[np.isfinite(x_rel)]
        if finite_rel.size == 0:
            return pd.Series(
                {
                    "x_mean": np.nan,
                    "side_label": "insufficient",
                    "samples_in_window": n_samples,
                    "samples_beyond_thresh": 0,
                    "baseline": baseline,
                }
            )

        x_mean = float(np.nanmean(finite_rel))

        if not np.isfinite(x_mean):
            return pd.Series(
                {
                    "x_mean": np.nan,
                    "side_label": "insufficient",
                    "samples_in_window": n_samples,
                    "samples_beyond_thresh": 0,
                    "baseline": baseline,
                }
            )

        # Default: forward
        label = "forward"
        samples_beyond = 0

        if abs(x_mean) >= x_thresh:
            # Candidate direction from sign of mean displacement
            if x_mean > 0:
                beyond_mask = x_rel > x_thresh
                candidate = "right"
            else:
                beyond_mask = x_rel < -x_thresh
                candidate = "left"

            samples_beyond = int(np.sum(beyond_mask))

            if samples_beyond >= min_samples_beyond:
                label = candidate

        return pd.Series(
            {
                "x_mean": x_mean,
                return_col_name: label,
                "samples_in_window": n_samples,
                "samples_beyond_thresh": samples_beyond,
                "baseline": baseline,
            }
        )

    summary = win.groupby("trial_id", observed=False, group_keys=False).apply(
        _summarise_trial
    ).reset_index()
    return summary


@dataclass
class LateralSignalSpec:
    """
    Definition of a lateral signal to test for left/right separation.

    - col: main column to use
    - ref_col: optional reference column to subtract (col - ref_col)
    """
    name: str
    col: str
    ref_col: Optional[str] = None


DEFAULT_LATERAL_SIGNAL_SPECS: List[LateralSignalSpec] = [
    LateralSignalSpec("nose_cam_x", "nose_cam_x"),
    LateralSignalSpec("nose_cam_x_minus_left_ear_cam_x", "nose_cam_x", "left_ear_cam_x"),
    LateralSignalSpec("nose_cam_x_minus_right_ear_cam_x", "nose_cam_x", "right_ear_cam_x"),
    LateralSignalSpec("nose_x", "nose_x"),
    LateralSignalSpec("nose_x_minus_left_ear_x", "nose_x", "left_ear_x"),
    LateralSignalSpec("nose_x_minus_right_ear_x", "nose_x", "right_ear_x"),
]

# Broader set for stamp scanning (includes y and angles)
DEFAULT_STAMP_SIGNAL_SPECS: List[LateralSignalSpec] = [
    # cam coords
    LateralSignalSpec("nose_cam_x", "nose_cam_x"),
    LateralSignalSpec("nose_cam_y", "nose_cam_y"),
    LateralSignalSpec("left_ear_cam_x", "left_ear_cam_x"),
    LateralSignalSpec("left_ear_cam_y", "left_ear_cam_y"),
    LateralSignalSpec("right_ear_cam_x", "right_ear_cam_x"),
    LateralSignalSpec("right_ear_cam_y", "right_ear_cam_y"),
    LateralSignalSpec("nose_cam_x_minus_left_ear_cam_x", "nose_cam_x", "left_ear_cam_x"),
    LateralSignalSpec("nose_cam_x_minus_right_ear_cam_x", "nose_cam_x", "right_ear_cam_x"),
    LateralSignalSpec("nose_cam_y_minus_left_ear_cam_y", "nose_cam_y", "left_ear_cam_y"),
    LateralSignalSpec("nose_cam_y_minus_right_ear_cam_y", "nose_cam_y", "right_ear_cam_y"),
    # world coords
    LateralSignalSpec("nose_x", "nose_x"),
    LateralSignalSpec("nose_y", "nose_y"),
    LateralSignalSpec("left_ear_x", "left_ear_x"),
    LateralSignalSpec("left_ear_y", "left_ear_y"),
    LateralSignalSpec("right_ear_x", "right_ear_x"),
    LateralSignalSpec("right_ear_y", "right_ear_y"),
    LateralSignalSpec("nose_x_minus_left_ear_x", "nose_x", "left_ear_x"),
    LateralSignalSpec("nose_x_minus_right_ear_x", "nose_x", "right_ear_x"),
    LateralSignalSpec("nose_y_minus_left_ear_y", "nose_y", "left_ear_y"),
    LateralSignalSpec("nose_y_minus_right_ear_y", "nose_y", "right_ear_y"),
    # angles
    LateralSignalSpec("head_angle_deg", "head_angle_deg"),
    LateralSignalSpec("head_angle_n_shifted_deg", "head_angle_n_shifted_deg"),
]


def _effect_size_cohens_d(pos: np.ndarray, neg: np.ndarray) -> float:
    """Cohen's d between two groups; returns nan if insufficient data."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size < 2 or neg.size < 2:
        return np.nan
    diff = np.nanmean(pos) - np.nanmean(neg)
    pooled = np.sqrt(((pos.size - 1) * np.nanvar(pos, ddof=1) + (neg.size - 1) * np.nanvar(neg, ddof=1)) / (pos.size + neg.size - 2))
    if not np.isfinite(pooled) or pooled == 0:
        return np.nan
    return diff / pooled


def _pairwise_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """
    Rank-based AUC (equivalent to Mann-Whitney U / (n_pos*n_neg)).
    Uses a simple O(n_pos * n_neg) computation to avoid extra deps.
    """
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return np.nan
    # Count wins, with 0.5 for ties
    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return (wins + 0.5 * ties) / float(n_pos * n_neg)


def _best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """
    Pick threshold on scores (>= => right) that maximizes balanced accuracy.
    Returns (threshold, balanced_accuracy, accuracy).
    """
    if scores.size == 0 or labels.size == 0:
        return np.nan, np.nan, np.nan
    uniq = np.unique(scores[np.isfinite(scores)])
    if uniq.size == 0:
        return np.nan, np.nan, np.nan

    # add extremes to allow "all left" / "all right" cases
    deltas = uniq if uniq.size == 1 else np.diff(np.sort(uniq))
    step = np.nanmin(deltas) if deltas.size else 1.0
    extra = step if np.isfinite(step) and step > 0 else 1.0
    candidates = np.concatenate([uniq, [np.nanmin(uniq) - extra, np.nanmax(uniq) + extra]])

    best_ba = -np.inf
    best_acc = np.nan
    best_thr = np.nan
    for thr in candidates:
        preds = scores >= thr
        tp = np.sum((preds == 1) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        ba = 0.5 * (tpr + tnr)
        acc = (tp + tn) / max(1, labels.size)
        if ba > best_ba or (np.isclose(ba, best_ba) and acc > best_acc):
            best_ba = ba
            best_acc = acc
            best_thr = thr
    if best_ba == -np.inf:
        return np.nan, np.nan, np.nan
    return best_thr, best_ba, best_acc


def search_best_lateral_separator(
    df: pd.DataFrame,
    trial_sides: pd.DataFrame,
    *,
    trial_col: str = "trial_id",
    time_col: str = "time_relative_to_end",
    signal_specs: Optional[Iterable[LateralSignalSpec]] = None,
    baseline_windows: Iterable[Tuple[float, float]] = ((-2.5, -2.0), (-2.2, -1.8), (-2.0, -1.5)),
    eval_windows: Iterable[Tuple[float, float]] = ((-2.0, 0.0), (-2.0, 0.5), (-2.0, 1.5), (-1.5, 1.0)),
    smooth_rolls: Iterable[int] = (3, 5, 7),
    min_samples_baseline: int = 3,
    min_samples_eval: int = 3,
    allow_sign_mirror: bool = True,
    trial_filter: Optional[Iterable] = None,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Grid-search simple left vs right separation across candidate signals, windows, and smoothing.

    For each signal spec:
      - optionally smooth per trial
      - compute (eval_mean - baseline_median) per trial for each window pair
      - optionally mirror the sign (treat positive as either right or left)
      - pick the threshold that maximises balanced accuracy

    Returns
    -------
    results : pd.DataFrame
        One row per combination with metrics: balanced_accuracy, accuracy, auc,
        effect_size, threshold, sign_used, n_trials, etc.
    best_trials : pd.DataFrame | None
        Per-trial scores/predictions for the top row of `results`, or None if nothing scored.
    """
    specs = list(signal_specs) if signal_specs is not None else list(DEFAULT_LATERAL_SIGNAL_SPECS)
    label_series = trial_sides.set_index(trial_col)["trial_side"]
    label_series = label_series[label_series.isin(["left", "right"])]

    allowed_ids = None
    if trial_filter is not None:
        if isinstance(trial_filter, pd.Series):
            if trial_filter.dtype == bool:
                allowed_ids = set(trial_filter[trial_filter].index)
            else:
                allowed_ids = set(trial_filter.index)
        else:
            allowed_ids = set(trial_filter)
    if allowed_ids is not None:
        label_series = label_series[label_series.index.isin(allowed_ids)]
        df = df[df[trial_col].isin(label_series.index)]

    if label_series.empty:
        return pd.DataFrame(), None

    work = df[[trial_col, time_col]].copy()
    results: List[dict] = []
    best_trials: Optional[pd.DataFrame] = None
    best_key = -np.inf

    for spec in specs:
        if spec.col not in df.columns:
            continue
        base = pd.to_numeric(df[spec.col], errors="coerce")
        if spec.ref_col:
            if spec.ref_col not in df.columns:
                continue
            ref = pd.to_numeric(df[spec.ref_col], errors="coerce")
            raw_signal = base - ref
            expr = f"{spec.col} - {spec.ref_col}"
        else:
            raw_signal = base
            expr = spec.col

        for roll in smooth_rolls:
            if roll is None or roll <= 1:
                sig = raw_signal
            else:
                sig = _smooth_series_per_trial(raw_signal, df[trial_col], roll)
            work["signal"] = sig

            for base_win in baseline_windows:
                for eval_win in eval_windows:
                    feats: List[tuple] = []
                    for tid, g in work.groupby(trial_col, sort=False):
                        if tid not in label_series.index:
                            continue
                        base_mask = (g[time_col] >= base_win[0]) & (g[time_col] <= base_win[1])
                        eval_mask = (g[time_col] >= eval_win[0]) & (g[time_col] <= eval_win[1])
                        base_vals = pd.to_numeric(g.loc[base_mask, "signal"], errors="coerce")
                        eval_vals = pd.to_numeric(g.loc[eval_mask, "signal"], errors="coerce")
                        if base_vals.notna().sum() < min_samples_baseline or eval_vals.notna().sum() < min_samples_eval:
                            continue
                        baseline = base_vals.median()
                        delta = eval_vals.mean() - baseline
                        feats.append((tid, delta))

                    if not feats:
                        continue

                    feat_df = pd.DataFrame(feats, columns=[trial_col, "delta"])
                    feat_df["trial_side"] = feat_df[trial_col].map(label_series)
                    feat_df = feat_df[feat_df["trial_side"].isin(["left", "right"])]
                    if feat_df.empty:
                        continue

                    scores = feat_df["delta"].to_numpy()
                    labels = (feat_df["trial_side"] == "right").astype(int).to_numpy()

                    sign_options = (1, -1) if allow_sign_mirror else (1,)
                    best_local = None
                    for sign in sign_options:
                        signed = sign * scores
                        thr, ba, acc = _best_threshold(signed, labels)
                        if not np.isfinite(ba):
                            continue
                        if best_local is None or ba > best_local["balanced_accuracy"]:
                            auc = _pairwise_auc(signed[labels == 1], signed[labels == 0])
                            eff = _effect_size_cohens_d(signed[labels == 1], signed[labels == 0])
                            best_local = {
                                "sign_used": sign,
                                "threshold": thr,
                                "balanced_accuracy": ba,
                                "accuracy": acc,
                                "auc": auc,
                                "effect_size": eff,
                                "mean_right": float(np.nanmean(signed[labels == 1])) if (labels == 1).any() else np.nan,
                                "mean_left": float(np.nanmean(signed[labels == 0])) if (labels == 0).any() else np.nan,
                            }

                    if best_local is None:
                        continue

                    record = {
                        "signal": spec.name,
                        "expression": expr,
                        "smooth_roll": roll,
                        "baseline_window": base_win,
                        "eval_window": eval_win,
                        "n_trials": int(feat_df.shape[0]),
                        "n_right": int((feat_df["trial_side"] == "right").sum()),
                        "n_left": int((feat_df["trial_side"] == "left").sum()),
                        **best_local,
                    }
                    results.append(record)

                    score_key = best_local["balanced_accuracy"]
                    if np.isfinite(score_key) and score_key > best_key:
                        best_key = score_key
                        signed = best_local["sign_used"] * scores
                        preds = np.where(signed >= best_local["threshold"], "right", "left")
                        best_trials = feat_df[[trial_col, "trial_side"]].copy()
                        best_trials["score_signed"] = signed
                        best_trials["score_raw"] = scores
                        best_trials["predicted_side"] = preds
                        best_trials["baseline_window"] = str(base_win)
                        best_trials["eval_window"] = str(eval_win)
                        best_trials["signal"] = spec.name
                        best_trials["smooth_roll"] = roll
                        best_trials["threshold"] = best_local["threshold"]
                        best_trials["sign_used"] = best_local["sign_used"]

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)
    return results_df, best_trials


def apply_separator_to_trials(
    df: pd.DataFrame,
    separator_config: pd.Series | dict,
    *,
    signal_specs: Optional[Iterable[LateralSignalSpec]] = None,
    trial_col: str = "trial_id",
    time_col: str = "time_relative_to_end",
    min_samples_baseline: int = 3,
    min_samples_eval: int = 3,
) -> pd.DataFrame:
    """
    Apply a chosen separator config (typically a row from search_best_lateral_separator results) to all trials.

    Expected keys in separator_config:
      - signal: name matching a LateralSignalSpec
      - baseline_window: (start, end)
      - eval_window: (start, end)
      - smooth_roll: int
      - sign_used: float (1 or -1)
      - threshold: float
    """
    if separator_config is None:
        return pd.DataFrame()
    cfg = dict(separator_config)
    specs = {s.name: s for s in (signal_specs or DEFAULT_LATERAL_SIGNAL_SPECS)}
    if cfg.get("signal") not in specs:
        return pd.DataFrame()

    spec = specs[cfg["signal"]]
    bw = cfg.get("baseline_window")
    ew = cfg.get("eval_window")
    smooth_roll = int(cfg.get("smooth_roll", 1))
    sign_used = float(cfg.get("sign_used", 1.0))
    thr = float(cfg.get("threshold", 0.0))

    if not isinstance(bw, (tuple, list)) or not isinstance(ew, (tuple, list)):
        return pd.DataFrame()
    bw0, bw1 = float(bw[0]), float(bw[1])
    ew0, ew1 = float(ew[0]), float(ew[1])

    def _make_signal(group: pd.DataFrame) -> np.ndarray:
        if spec.col not in group.columns:
            return np.full(group.shape[0], np.nan)
        sig = pd.to_numeric(group[spec.col], errors="coerce")
        if spec.ref_col:
            if spec.ref_col not in group.columns:
                return np.full(group.shape[0], np.nan)
            sig = sig - pd.to_numeric(group[spec.ref_col], errors="coerce")
        arr = sig.to_numpy()
        if smooth_roll > 1:
            arr = (
                pd.Series(arr, index=group.index)
                .rolling(smooth_roll, center=True, min_periods=1)
                .median()
                .to_numpy()
            )
        return arr

    rows: List[dict] = []
    for tid, g in df.groupby(trial_col, sort=False):
        sig_arr = _make_signal(g)
        time_series = g[time_col] if time_col in g.columns else pd.Series(np.nan, index=g.index)
        t = pd.to_numeric(time_series, errors="coerce").to_numpy()
        base_mask = (t >= bw0) & (t <= bw1)
        eval_mask = (t >= ew0) & (t <= ew1)
        base_vals = sig_arr[base_mask]
        eval_vals = sig_arr[eval_mask]
        if np.sum(np.isfinite(base_vals)) < min_samples_baseline or np.sum(np.isfinite(eval_vals)) < min_samples_eval:
            continue
        baseline = np.nanmedian(base_vals)
        delta = np.nanmean(eval_vals) - baseline
        signed = sign_used * delta
        pred = "right" if signed >= thr else "left"
        rows.append(
            {
                trial_col: tid,
                "score_raw": delta,
                "score_signed": signed,
                "predicted_side": pred,
                "signal": spec.name,
                "baseline_window": (bw0, bw1),
                "eval_window": (ew0, ew1),
                "smooth_roll": smooth_roll,
                "sign_used": sign_used,
                "threshold": thr,
            }
        )

    return pd.DataFrame(rows)


def merge_predictions_into_windows(
    windows: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    trial_col: str = "trial_id",
    prefix: str = "pred_",
    keep_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Left-join predictions onto window data; trials without predictions get NaN.

    keep_cols: columns from predictions to bring over (default: predicted_side, score_signed, score_raw).
    """
    merged = windows.copy()
    if keep_cols is None:
        keep_cols = ["predicted_side", "score_signed", "score_raw"]
    keep_cols = list(keep_cols)

    if predictions is None or predictions.empty:
        for col in keep_cols:
            merged[f"{prefix}{col}"] = np.nan
        return merged

    avail_cols = [trial_col] + [c for c in keep_cols if c in predictions.columns]
    merged = merged.merge(predictions[avail_cols], on=trial_col, how="left")
    # add any missing requested columns as NaN
    for col in keep_cols:
        full_col = f"{col}" if col in merged.columns else col
        if full_col not in merged.columns:
            merged[full_col] = np.nan
    # prefix the kept columns (optional)
    for col in keep_cols:
        if col in merged.columns:
            merged.rename(columns={col: f"{prefix}{col}"}, inplace=True)
    return merged


def summarize_pre_post_deltas(
    df: pd.DataFrame,
    trial_sides: pd.DataFrame,
    *,
    features: list[str],
    pre_window: tuple[float, float] = (-3.0, 0.0),
    post_window: tuple[float, float] = (0.0, 5.0),
    trial_col: str = "trial_id",
    time_col: str = "time_relative_to_end",
    side_col: str = "trial_side",
) -> pd.DataFrame:
    """
    Quick data-driven scan of feature shifts pre->post. For each feature and side
    it reports mean/median delta (post_mean - pre_mean) per trial.
    Use this to decide which score_side_specific components are informative.
    """
    work = df[[trial_col, time_col] + [f for f in features if f in df.columns]].copy()
    t = pd.to_numeric(work[time_col], errors="coerce")
    pre_mask = t.between(*pre_window)
    post_mask = t.between(*post_window)

    rows = []
    side_lookup = trial_sides.set_index(trial_col)[side_col]

    for feat in features:
        if feat not in work.columns:
            continue
        pre_mean = work.loc[pre_mask].groupby(trial_col)[feat].mean()
        post_mean = work.loc[post_mask].groupby(trial_col)[feat].mean()
        delta = post_mean - pre_mean
        merged = pd.DataFrame({"delta": delta})
        merged[side_col] = merged.index.map(side_lookup)
        for side in ["left", "right"]:
            d = merged[merged[side_col] == side]["delta"].dropna()
            if d.empty:
                continue
            rows.append(
                {
                    "feature": feat,
                    "side": side,
                    "mean_delta": float(d.mean()),
                    "median_delta": float(d.median()),
                    "abs_mean_delta": float(np.abs(d.mean())),
                    "n_trials": int(d.shape[0]),
                }
            )
    return pd.DataFrame(rows)

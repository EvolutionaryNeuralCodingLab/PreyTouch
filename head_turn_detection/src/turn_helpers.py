from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import detect_turn
from detect_turn import TurnConfig, TurnLabel, load_results_bundle, resolve_turn_config, run_models
from trial_utils import filter_missing_from_df, load_animal_df, save_fig as save_fig_as_pdf


@dataclass(frozen=True)
class AnimalSpec:
    key: str
    labels: List[Dict[str, Any]]
    cfg: Optional[TurnConfig | Dict[str, Any]] = None
    cache_df_path: Optional[str] = None
    env_file: Optional[str] = None
    experiment: Optional[str] = None
    sil_subfolder: Optional[str] = None
    trial_ids: Optional[List[int]] = None
    exclude_trials: Optional[List[int]] = None
    exclude_trials_from_plots: Optional[List[int]] = None
    apply_score_to_trials: Optional[List[int]] = None
    title: Optional[str] = None


def _shift_axis(fig, shift, label: str = "hit"):
    for ax in fig.axes:
        labels = [t.get_text() for t in ax.get_xticklabels()]
        if not labels:
            continue
        try:
            new_labels = [f"{float(l) + shift:g}" if l else "" for l in labels]
        except ValueError:
            continue
        ax.set_xticklabels(new_labels)
        if ax.get_xlabel():
            ax.set_xlabel(f"time relative to {label} [s]")
    return fig


def plot_animal_heatmaps(
    *,
    scored_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    cfg: dict,
    score_col: str,
    title_prefix: str,
    exclude_plot_trials: Optional[set] = None,
    is_shift_req: bool = True,
    shift: float = 2,
    ax_no_reward: Optional[plt.Axes] = None,
    ax_reward: Optional[plt.Axes] = None,
    cax_no_reward: Optional[plt.Axes] = None,
    cax_reward: Optional[plt.Axes] = None,
    show_colorbar: bool = True,
    is_legend: bool = True,
    legend_min_score: Optional[float] = 0,
) -> List:
    if scored_df is None or scored_df.empty or meta_df is None or meta_df.empty:
        return []

    exclude_plot_trials = set(exclude_plot_trials or [])
    trial_col = cfg["TRIAL_COL"]

    def _filter_plot_data(scored_df: pd.DataFrame, meta_df: pd.DataFrame):
        if not exclude_plot_trials:
            return scored_df, meta_df
        if scored_df is not None and not scored_df.empty and trial_col in scored_df.columns:
            scored_df = scored_df[~scored_df[trial_col].isin(exclude_plot_trials)].copy()
        if meta_df is not None and not meta_df.empty and trial_col in meta_df.columns:
            meta_df = meta_df[~meta_df[trial_col].isin(exclude_plot_trials)].copy()
        return scored_df, meta_df

    use_shared_axes = any([ax_no_reward, ax_reward, cax_no_reward, cax_reward])
    if legend_min_score is None:
        legend_min_score = cfg.get("PLOT_ZERO_BELOW")
        if legend_min_score is None and cfg.get("POSTPROC_APPLY_TO_HEATMAPS"):
            if str(cfg.get("ML_PP_THRESH_MODE", "")).lower() == "fixed":
                legend_min_score = cfg.get("ML_PP_THRESH_FIXED")

    def _generate_fig(which, scored_df, meta_df, title_prefix, ax, cax):
        return detect_turn.plot_reward_split_no_side(
            scored_all=scored_df,
            meta=meta_df,
            value_col=score_col,
            time_range=tuple(cfg["PLOT_TIME_RANGE"]),
            row_height=cfg["PLOT_ROW_HEIGHT"],
            cmap="Reds",
            title_prefix=title_prefix,
            which=which,
            normalize_for_plot=cfg["HEATMAP_NORMALIZE_FOR_PLOT"],
            sort_rows=cfg["SORT_ROWS"],
            xtick_step_s=cfg["HEATMAP_XTICK_STEP_S"],
            cfg=cfg,
            ax=ax,
            cax=cax,
            show_colorbar=show_colorbar,
            is_legend=is_legend,
            do_layout=not use_shared_axes,
            legend_min_score=legend_min_score,
        )

    def _plot_reward_only(scored_df: pd.DataFrame, meta_df: pd.DataFrame, title_prefix: str):
        figs = []
        if scored_df is None or scored_df.empty or meta_df is None or meta_df.empty:
            return figs

        for t in ["no_reward", "reward"]:
            if t == "no_reward":
                ax = ax_no_reward
                cax = cax_no_reward
            else:
                ax = ax_reward
                cax = cax_reward
            fig = _generate_fig(t, scored_df, meta_df, title_prefix, ax, cax)
            if fig not in figs:
                figs.append(fig)
            if is_shift_req and not use_shared_axes:
                _shift_axis(fig, shift)
        if is_shift_req and use_shared_axes and figs:
            _shift_axis(figs[0], shift)
        return figs

    scored_plot, meta_plot = _filter_plot_data(scored_df, meta_df)
    return _plot_reward_only(scored_plot, meta_plot, title_prefix)


def run_model_for_animal(
    animal_id: str,
    df: pd.DataFrame,
    labels: list[TurnLabel],
    config: TurnConfig,
    filter_trial_ids: list[int] = None,
    filter_missing: bool = True,
    exclude_trials: list[int] = None,
    data_path: Optional[str | Path] = None,
    run_side: bool = True,
    run_loto: bool = True,
    side_method: str = "simple",
    side_method_kwargs: Optional[dict] = None,
    use_new_side_v: bool = False,
    flip_head_angle_aligned: Literal[False, True, None] = False,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    animal = None
    if df is None:
        if data_path is None:
            raise ValueError("data_path must be provided when df is None")
        animal = load_animal_df(animal_id, data_path=data_path)
        df = animal.df
    if filter_missing:
        df = filter_missing_from_df(df)
    config.update(
        {
            "ML_ALIGN_X_BY_SIDE": True,
            "ML_DIFF_PER_SEC": True,
            "PLOT_ALIGN_X_BY_SIDE": False,
        }
    )
    results = run_models(
        df,
        run_classic=False,
        labels=labels,
        cfg=config,
        plot=True,
        run_side=run_side,
        side_method=side_method,
        side_method_kwargs=side_method_kwargs,
        run_loto=run_loto,
        use_new_side_v=use_new_side_v,
        flip_head_angle_aligned=flip_head_angle_aligned,
    )
    animal_df = animal.df if animal is not None else df
    return results, df, animal_df


def run_all_animals(
    animal_ids: List[str],
    ANIMALS: List[AnimalSpec],
    out_dir: Path = None,
    use_new_side_v: bool = False,
    flip_head_angle_aligned: bool = False,
    run_side: bool = False,
    run_loto: bool = False,
    side_method: str = "simple",
    side_method_kwargs: Optional[dict] = None,
    is_shift_req: bool = True,
    shift: float = 2,
    save_models: bool = True,
    save_results: bool = True,
    force_regenerating_model: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    all_scores = []
    figs = {}
    all_animals_df = pd.DataFrame()
    animals_results: Dict[str, Any] = {}

    get_animal_spec = lambda key: next((animal for animal in ANIMALS if animal.key == key), None)

    for animal_id in animal_ids:
        ttle = f"=============Processing animal: {animal_id}=============="
        n_chars = len(ttle)
        print("=" * n_chars)
        print(ttle)

        animal_spec = get_animal_spec(animal_id)
        if animal_spec is None:
            raise ValueError(f"Unknown animal_id: {animal_id}")
        config = resolve_turn_config(animal_spec.cfg)
        data_path = getattr(animal_spec, "cache_df_path", None)

        results = None
        animal_df = None
        loaded_from_cache = False
        results_dir = out_dir / "results" / animal_id if out_dir is not None else None

        if not force_regenerating_model and results_dir is not None and results_dir.exists():
            try:
                results = load_results_bundle(results_dir)
                loaded_from_cache = True
                print(f"Loaded cached results for {animal_id}")
            except Exception as exc:
                print(f"Failed to load cached results for {animal_id}: {exc}")
                results = None
                loaded_from_cache = False

        if (
            results is None
            or not isinstance(results, dict)
            or "scored_ml" not in results
            or "meta" not in results
        ):
            results, _, animal_df = run_model_for_animal(
                animal_id,
                None,
                animal_spec.labels,
                config,
                filter_trial_ids=animal_spec.trial_ids,
                exclude_trials=animal_spec.exclude_trials,
                data_path=data_path,
                run_side=run_side,
                run_loto=run_loto,
                side_method=side_method,
                side_method_kwargs=side_method_kwargs,
                use_new_side_v=use_new_side_v,
                flip_head_angle_aligned=flip_head_angle_aligned,
            )
            loaded_from_cache = False
        else:
            if "cfg" not in results:
                results["cfg"] = config
            if "bin_size" not in results:
                try:
                    results["bin_size"] = float(results["cfg"]["BIN_SIZE"])
                except Exception:
                    results["bin_size"] = None
            if data_path is None:
                raise ValueError("data_path must be provided when loading cached results")
            animal = load_animal_df(animal_id, data_path=data_path)
            animal_df = animal.df

        all_animals_df = pd.concat([all_animals_df, animal_df], ignore_index=True)

        if results is None:
            continue

        animals_results[animal_id] = results
        scored = results["scored_ml"]
        if scored is None or scored.empty:
            continue

        cfg = results["cfg"]
        trial_col = cfg["TRIAL_COL"]
        time_col = cfg["TIME_COL"]
        score_col = "turn_score_pp" if cfg["POSTPROC_APPLY_TO_HEATMAPS"] else "turn_score_nms"
        exclude_plot_trials = set(getattr(animal_spec, "exclude_trials_from_plots", None) or [])

        def _scores_to_df(scored_df: pd.DataFrame, meta_df: pd.DataFrame, score_source: str) -> pd.DataFrame:
            meta = meta_df[[trial_col, "is_reward_bug"]].copy()
            meta = meta.rename(columns={trial_col: "trial_id"})

            s = scored_df[[trial_col, "time_bin", time_col, score_col]].copy()
            s = s.rename(columns={trial_col: "trial_id", time_col: "time_s", score_col: "score"})
            s = s.merge(meta, on="trial_id", how="left")
            s["is_reward_bug"] = s["is_reward_bug"].astype("boolean").fillna(False).astype(bool)
            s["trial_type"] = np.where(s["is_reward_bug"], "reward", "no_reward")
            s["animal_id"] = animal_id
            s["bin_size"] = results["bin_size"]
            s["score_source"] = score_source
            return s

        animal_scores = []
        s_train = _scores_to_df(scored, results["meta"], score_source="train")
        animal_scores.append(s_train)
        figs_for_animal = []
        if exclude_plot_trials:
            figs_for_animal.extend(
                plot_animal_heatmaps(
                    scored_df=scored,
                    meta_df=results["meta"],
                    cfg=cfg,
                    score_col=score_col,
                    title_prefix=f"{animal_id} turn detection (train)",
                    exclude_plot_trials=exclude_plot_trials,
                    is_shift_req=is_shift_req,
                    shift=shift,
                )
            )

        if save_models and out_dir is not None and not loaded_from_cache:
            models_dir = out_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            model_obj = {
                "side_pipe": results.get("side_pipe"),
                "turn_pipe_ml": results.get("turn_pipe_ml"),
                "cfg": results.get("cfg"),
                "align_x_by_side": results.get("align_x_by_side"),
                "run_side": results.get("run_side"),
            }
            try:
                import joblib

                joblib.dump(model_obj, models_dir / f"{animal_id}_models.joblib")
            except Exception:
                with open(models_dir / f"{animal_id}_models.pkl", "wb") as f:
                    import pickle

                    pickle.dump(model_obj, f)

        if save_results and out_dir is not None and not loaded_from_cache:
            results_dir = out_dir / "results" / animal_id
            detect_turn.save_results_bundle(results, results_dir, include_models=save_models)

        apply_trials = set(getattr(animal_spec, "apply_score_to_trials", None) or [])
        exclude_trials = set(animal_spec.exclude_trials or [])
        base_trials = set(animal_spec.trial_ids or [])
        if not base_trials:
            base_trials = set(pd.to_numeric(scored[trial_col], errors="coerce").dropna().astype(int).tolist())

        apply_trials = apply_trials - base_trials - exclude_trials
        if apply_trials and results.get("turn_pipe_ml") is None:
            print(f"Skipping apply trials for {animal_id} (missing model in results)")
        if apply_trials and results.get("turn_pipe_ml") is not None:
            apply_df = animal_df.copy()
            apply_df = filter_missing_from_df(apply_df)

            if not apply_df.empty:
                applied = detect_turn.score_new_trials_with_ml_model(
                    apply_df,
                    results=results,
                    cfg=cfg,
                )
                scored_apply = applied.get("scored_ml")
                if scored_apply is not None and not scored_apply.empty:
                    meta_apply = detect_turn.build_trial_meta_hit_only(apply_df, cfg=cfg)
                    s_apply = _scores_to_df(scored_apply, meta_apply, score_source="apply")
                    animal_scores.append(s_apply)
                    figs_for_animal.extend(
                        plot_animal_heatmaps(
                            scored_df=scored_apply,
                            meta_df=meta_apply,
                            cfg=cfg,
                            score_col=score_col,
                            title_prefix=f"{animal_id} turn detection (apply)",
                            exclude_plot_trials=exclude_plot_trials,
                            is_shift_req=is_shift_req,
                            shift=shift,
                        )
                    )

        if animal_scores:
            all_scores.extend(animal_scores)

        base_figs = results.get("figs", [])
        base_figs_from_cache = bool(base_figs)
        if not base_figs and not exclude_plot_trials:
            base_figs = plot_animal_heatmaps(
                scored_df=scored,
                meta_df=results["meta"],
                cfg=cfg,
                score_col=score_col,
                title_prefix=f"{animal_id} turn detection",
                exclude_plot_trials=exclude_plot_trials,
                is_shift_req=is_shift_req,
                shift=shift,
            )
            base_figs_from_cache = False
        if is_shift_req and base_figs_from_cache:
            for fig in base_figs:
                _shift_axis(fig, shift)
        figs[animal_id] = (base_figs if not exclude_plot_trials else []) + figs_for_animal
        if out_dir is not None:
            s_out = pd.concat(animal_scores, ignore_index=True)
            s_out.to_parquet(out_dir / f"{animal_id}_scores.parquet", index=False)
            figs_dir = out_dir / "figs" / animal_id
            for fig_idx, fig in enumerate(figs[animal_id]):
                save_fig_as_pdf(fig, figs_dir, save_prefix=f"{animal_id}_fig_{fig_idx:02d}", use_datestamp=True)

    scores_df = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    if out_dir is not None:
        scores_df.to_parquet(out_dir / "all_animals_scores.parquet", index=False)
    return scores_df, all_animals_df, animals_results


def _resolve_feat_names(turn_pipe, feat_cols, n_features: int):
    names = None
    imputer = None
    if hasattr(turn_pipe, "named_steps"):
        imputer = turn_pipe.named_steps.get("imputer")
    if imputer is not None:
        get_names = getattr(imputer, "get_feature_names_out", None)
        if callable(get_names):
            try:
                names = list(get_names(feat_cols))
            except Exception:
                names = None
    if names is None and feat_cols:
        names = list(feat_cols)
    if names is None:
        names = [f"feat_{i}" for i in range(n_features)]
    if len(names) < n_features:
        names += [f"extra_{i}" for i in range(len(names), n_features)]
    elif len(names) > n_features:
        names = names[:n_features]
    return names


def _load_model_payload(path: Path):
    try:
        import joblib
    except Exception as exc:
        print(f"joblib not available; can't load {path.name}: {exc}")
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        print(f"failed to load {path.name}: {exc}")
        return None


def collect_turn_model_weights(models_dir: Path | str, animals: Optional[List[str]] = None) -> pd.DataFrame:
    rows = []
    for model_path in sorted(Path(models_dir).glob("*_models.joblib")):
        animal_id = model_path.stem.replace("_models", "")
        if animals and animal_id not in animals:
            continue
        payload = _load_model_payload(model_path)
        if not isinstance(payload, dict):
            continue
        turn_pipe = payload.get("turn_pipe_ml")
        if turn_pipe is None or not hasattr(turn_pipe, "named_steps"):
            continue
        clf = turn_pipe.named_steps.get("clf")
        if clf is None or not hasattr(clf, "coef_"):
            continue
        coefs = np.ravel(clf.coef_)
        feat_cols = getattr(turn_pipe, "_feat_cols", None) or []
        names = _resolve_feat_names(turn_pipe, feat_cols, len(coefs))
        rows.extend(
            {"animal_id": animal_id, "feature": feat, "weight": float(w)}
            for feat, w in zip(names, coefs)
        )
    return pd.DataFrame(rows)


def _clean_feature_label(name: str) -> str:
    label = str(name)
    label = label.replace("ml__", "")
    label = label.replace("__abs", "__")
    label = label.replace("_abs", "_")
    label = label.replace("abs", "")
    label = re.sub(r"win\\d+", "", label)
    label = label.replace("_win", "_")
    label = re.sub(r"diff\\d+", "diff", label)
    label = re.sub(r"__+", "_", label)
    label = label.strip("_")
    return label


def _feature_group(name: str) -> str:
    base = str(name).replace("ml__", "")
    base = base.split("__")[0]
    parts = base.split("_")
    axis = parts[-1] if parts and parts[-1] in {"x", "y"} else None
    root = "_".join(parts[:-1]) if axis else base
    if root.endswith("_cam"):
        root = root[:-4]
    return root


def _feature_color_map(features: List[str]) -> Dict[str, str]:
    groups = []
    for feat in features:
        grp = _feature_group(feat)
        if grp not in groups:
            groups.append(grp)
    assign = getattr(detect_turn, "_assign_dlc_group_colors", None)
    if callable(assign):
        group_to_color = assign(groups)
    else:
        group_to_color = {
            "head_angle": "black",
            "nose": "green",
            "mid_ear": "green",
            "left_ear": "orange",
            "right_ear": "blue",
        }
    for grp in groups:
        if "mid_ear" in grp:
            group_to_color[grp] = "purple"
    return {feat: group_to_color.get(_feature_group(feat), "black") for feat in features}


def plot_model_weights_scatter(
    weights_df: pd.DataFrame,
    top_n: Optional[int] = 30,
    simplify_labels: bool = False,
):
    if weights_df.empty:
        raise ValueError("weights_df is empty; no model weights found")
    work = weights_df.copy()
    if simplify_labels:
        work["plot_feature"] = work["feature"].map(_clean_feature_label)
    else:
        work["plot_feature"] = work["feature"].str.replace("ml__", "", regex=False)

    if top_n is not None:
        top_features = (
            work.groupby("plot_feature")["weight"]
            .apply(lambda x: x.abs().mean())
            .sort_values(ascending=False)
            .head(int(top_n))
            .index
        )
        work = work[work["plot_feature"].isin(top_features)].copy()
        feature_order = list(top_features)
    else:
        feature_order = sorted(work["plot_feature"].unique().tolist())

    feature_group_map = {}
    for pf in feature_order:
        first_feat = work.loc[work["plot_feature"] == pf, "feature"].iloc[0]
        feature_group_map[pf] = _feature_group(first_feat)

    feature_order = sorted(feature_order, key=lambda f: (feature_group_map.get(f, ""), f))

    feature_to_color = {}
    for pf in feature_order:
        first_feat = work.loc[work["plot_feature"] == pf, "feature"].iloc[0]
        feature_to_color[pf] = _feature_color_map([first_feat])[first_feat]

    animals = sorted(work["animal_id"].unique().tolist())
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    animal_markers = {a: markers[i % len(markers)] for i, a in enumerate(animals)}

    feat_to_y = {f: i for i, f in enumerate(feature_order)}
    n_animals = max(len(animals), 1)
    jitter_step = 0.18 if n_animals > 1 else 0.0

    fig_h = max(4.0, 0.35 * len(feature_order))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    for idx, animal in enumerate(animals):
        sub = work[work["animal_id"] == animal]
        y = sub["plot_feature"].map(feat_to_y).astype(float).to_numpy()
        jitter = (idx - (n_animals - 1) / 2.0) * jitter_step
        colors = sub["plot_feature"].map(feature_to_color).to_numpy()
        ax.scatter(
            sub["weight"].to_numpy(),
            y + jitter,
            label=animal,
            alpha=0.8,
            s=24,
            marker=animal_markers[animal],
            color=colors,
        )

    ax.set_yticks(range(len(feature_order)))
    ax.set_yticklabels(feature_order)
    for tick in ax.get_yticklabels():
        tick.set_color(feature_to_color.get(tick.get_text(), "black"))

    ax.set_xlabel("model weight (logreg coef)")
    ax.set_ylabel("feature")
    ax.set_title("Turn model weights per animal")
    ax.grid(True, axis="x", alpha=0.2)

    handles = [
        Line2D([0], [0], marker=animal_markers[a], color="black", linestyle="None", markersize=6, label=a)
        for a in animals
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    return fig, ax


def plot_peak_aligned_mean_heatmaps(
    *,
    animal_ids: List[str],
    results_dir: Optional[Path | str] = None,
    results: Optional[Dict[str, Any]] = None,
    time_range: Tuple[float, float] = (-4, 6),
    heatmap_min_score: float = 0.6,
    feature_mode: Literal["diff", "all"] = "diff",
    row_height: float = 0.01,
    figsize: Optional[Tuple[float, float]] = None,
    fig: Optional[plt.Figure] = None,
    heat_axes: Optional[List[plt.Axes]] = None,
    trace_axes: Optional[List[plt.Axes]] = None,
    show_colorbar: bool = True,
    cbar_ax: Optional[plt.Axes] = None,
    cbar_width: float = 0.01,
    cbar_pad: float = 0.02,
    colorbar_label: Optional[str] = None,
    trace_legend: bool = True,
    trace_legend_last_only: bool = True,
    trace_legend_outside: bool = True,
    trace_legend_kwargs: Optional[Dict[str, Any]] = None,
    event_lines: Tuple[Tuple[float, str], ...] = (),
    nose_minus_left_ear_color: Optional[str] = "purple",
) -> Tuple[plt.Figure, List[plt.Axes], List[plt.Axes], Optional[str]]:
    if not animal_ids:
        raise ValueError("animal_ids must be non-empty")
    n_animals = len(animal_ids)

    if heat_axes is None or trace_axes is None:
        if fig is None:
            fig_w = 8 * n_animals
            fig_h = 6.0
            fig, axes = plt.subplots(
                nrows=2,
                ncols=n_animals,
                figsize=figsize or (fig_w, fig_h),
                gridspec_kw={"height_ratios": [0.35, 3]},
            )
        else:
            axes = fig.subplots(nrows=2, ncols=n_animals, gridspec_kw={"height_ratios": [0.35, 3]})
        if n_animals == 1:
            heat_axes = [axes[0]]
            trace_axes = [axes[1]]
        else:
            heat_axes = list(axes[0])
            trace_axes = list(axes[1])
    else:
        if fig is None:
            fig = heat_axes[0].figure
        if len(heat_axes) != n_animals or len(trace_axes) != n_animals:
            raise ValueError("heat_axes/trace_axes length must match animal_ids")

    if results is None and results_dir is None:
        raise ValueError("results_dir or results must be provided")

    mesh = None
    value_label = None
    trace_legend_kwargs = dict(trace_legend_kwargs or {})

    def _get_results(aid: str) -> Dict[str, Any]:
        if isinstance(results, dict):
            res_i = results.get(aid)
            if res_i is None:
                res_i = results.get(str(aid).lower())
            if res_i is None:
                res_i = results.get(str(aid))
            if res_i is not None:
                return res_i
        if results_dir is None:
            raise ValueError("results_dir not set")
        return detect_turn.load_results_bundle(Path(results_dir) / aid, tables=["scored_ml", "feat_ml", "meta"])

    for i, aid in enumerate(animal_ids):
        res_i = _get_results(aid)
        mean_res_i, mean_tid_i, _ = detect_turn.build_peak_aligned_mean_results(
            res_i,
            animal_id=None,
            time_range=time_range,
            recenter_mean_peak=True,
            heatmap_min_score=heatmap_min_score,
            abs_features=False,
        )

        if value_label is None:
            cfg_i = detect_turn.resolve_cfg(mean_res_i.get("cfg"))
            score_cols = [
                c
                for c in mean_res_i["scored_ml"].columns
                if c not in (cfg_i["TRIAL_COL"], cfg_i["TIME_COL"], "time_bin")
            ]
            value_label = score_cols[0] if score_cols else "score"

        detect_turn.plot_one_trial_heatmap_row_from_results(
            results=mean_res_i,
            trial_id=mean_tid_i,
            value_col=value_label,
            row_height=row_height,
            time_range=time_range,
            ax=heat_axes[i],
            cax=None,
            is_legend=False,
            cfg={**detect_turn.resolve_cfg(mean_res_i.get("cfg")), "EVENT_LINES": event_lines},
        )
        detect_turn.plot_one_trial_feature_traces_from_results(
            results=mean_res_i,
            trial_id=mean_tid_i,
            feature_mode=feature_mode,
            time_range=time_range,
            legend_outside=trace_legend_outside,
            ax=trace_axes[i],
            cfg={**detect_turn.resolve_cfg(mean_res_i.get("cfg")), "EVENT_LINES": event_lines},
            is_legend=trace_legend,
        )

        heat_axes[i].set_title(aid)
        heat_axes[i].set_xlabel("")
        if i == 0:
            heat_axes[i].set_yticklabels(["mean"])
            heat_axes[i].set_ylabel("")
        else:
            heat_axes[i].set_ylabel("")
            heat_axes[i].set_yticklabels([])

        trace_axes[i].set_title("")
        trace_axes[i].set_xlabel("time relative to peak [s]")
        if nose_minus_left_ear_color:
            for line_obj in trace_axes[i].get_lines():
                if line_obj.get_label() == "mid_ear_cam_x":
                    line_obj.set_color(nose_minus_left_ear_color)

        leg = trace_axes[i].get_legend()
        if trace_legend and trace_legend_last_only:
            if i != (n_animals - 1) and leg is not None:
                leg.remove()
            if i == (n_animals - 1):
                if leg is not None:
                    leg.remove()
                trace_axes[i].legend(**{
                    "loc": "upper left",
                    "bbox_to_anchor": (1.02, 1),
                    "borderaxespad": 0,
                    "fontsize": 8,
                    "frameon": True,
                    **trace_legend_kwargs,
                })
        elif not trace_legend and leg is not None:
            leg.remove()

        if heat_axes[i].collections:
            mesh = heat_axes[i].collections[-1]

    for ax in trace_axes:
        ax.set_xlabel("time relative to peak [s]")

    fig.tight_layout(rect=[0, 0, 0.86, 1])

    if show_colorbar and mesh is not None:
        if cbar_ax is None:
            heat_pos = heat_axes[-1].get_position()
            x0 = min(heat_pos.x1 + cbar_pad, 0.98 - cbar_width)
            cbar_ax = fig.add_axes([x0, heat_pos.y0, cbar_width, heat_pos.height])
        fig.colorbar(mesh, cax=cbar_ax, label=colorbar_label or value_label)

    return fig, heat_axes, trace_axes, value_label

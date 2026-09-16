"""Compare observation modalities: vector (state) vs. pixel observations.

Pixel runs are separated by the camera they render from

Able to plot both curves and bars, sharing the same loader and selection flags:

    # learning curves (default)
    python -m results.plotting_results.obs_comparison --envs safe_goal_point --algos ppo_lag

    # final-performance bars
    python -m results.plotting_results.obs_comparison --bars --envs safe_goal_point --algos ppo_lag

Currently, one figure per algorithm, since obs mode is already the line/bar
dimension and overlaying algorithms on top of it is unreadable.

Data layout (see `download.main_results.obs_mode_segment`):
    vector  -> data/<env>/level_<l>/<algo>/seed_<n>.parquet
    pixels  -> data/<env>/level_<l>/<algo>/vision_<camera>/seed_<n>.parquet
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results import cli
from results.common import (
    DEFAULT_METRIC_COLS as METRIC_COLS,
    DEFAULT_OBS_MODES,
    OBS_MODE_COLORS,
    TRANSLATIONS,
    align_and_stack,
    get_series,
    legend_ncol,
    moving_average,
    nice_grid,
    obs_mode_dir,
    results_path,
    set_mpl_style,
)

# (env, algo, obs_mode, metric) -> one DataFrame['_step', 'value'] per seed
RunStore = Dict[Tuple[str, str, str, str], List[pd.DataFrame]]


def load_runs(base: Path, env: str, level: int, algo: str, obs_mode: str,
              seeds: List[int], metrics: List[str]) -> RunStore:
    """Load every seed of one (env, level, algo, obs_mode) cell."""
    out: RunStore = {}
    folder = base / env / f"level_{level}" / algo / obs_mode_dir(obs_mode)
    for metric in metrics:
        key = (env, algo, obs_mode, metric)
        out[key] = []
        for seed in seeds:
            fp = folder / f"seed_{seed}.parquet"
            if not fp.exists():
                continue
            df = pd.read_parquet(fp, engine="pyarrow")
            if "_step" not in df:
                continue
            series = get_series(df, algo=algo, metric=metric,
                                metric_cols=METRIC_COLS, env_name=env)
            if series is None:
                continue
            out[key].append(pd.DataFrame({
                "_step": df["_step"].astype(np.int64),
                "value": series.astype(np.float32),
            }).dropna())
    return out


def load_all(args: argparse.Namespace) -> RunStore:
    base = results_path(args.input)
    store: RunStore = {}
    for env in args.envs:
        for algo in args.algos:
            for obs_mode in args.obs_modes:
                store.update(load_runs(base, env, args.level, algo, obs_mode,
                                       args.seeds, args.metrics))
    return store


def _present_obs_modes(store: RunStore, args: argparse.Namespace, algo: str) -> List[str]:
    """Obs modes that actually have data for this algo, in the requested order."""
    return [
        om for om in args.obs_modes
        if any(store.get((env, algo, om, metric))
               for env in args.envs for metric in args.metrics)
    ]


def _finalize(fig, handles: Dict[str, plt.Line2D], n_entries: int,
              args: argparse.Namespace, algo: str, kind: str) -> None:
    """Attach the shared bottom legend and write the figure out."""
    if handles:
        labels, hs = zip(*handles.items())
        labels = [TRANSLATIONS.get(lbl, lbl) for lbl in labels]
        fig.legend(hs, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                   ncol=legend_ncol(n_entries, len(labels)),
                   fancybox=True, shadow=True)

    out_dir = results_path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}_{kind}_level_{args.level}_{algo}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def plot_curves(store: RunStore, args: argparse.Namespace, algo: str) -> None:
    """Reward/cost training curves, one line per observation modality."""
    set_mpl_style()
    envs, metrics = args.envs, args.metrics
    m = len(metrics)
    nrows, ncols_env = nice_grid(len(envs), max_cols=args.max_cols)

    fig, axs = plt.subplots(nrows, ncols_env * m,
                            figsize=(args.panel_w * ncols_env * m, args.panel_h * nrows),
                            squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.12,
                        wspace=0.35, hspace=0.55)

    def get_ax(env_i: int, metric_i: int):
        return axs[env_i // ncols_env, (env_i % ncols_env) * m + metric_i]

    handles: Dict[str, plt.Line2D] = {}
    obs_modes = _present_obs_modes(store, args, algo)

    for env_i, env in enumerate(envs):
        for metric_i, metric in enumerate(metrics):
            ax = get_ax(env_i, metric_i)

            for obs_mode in obs_modes:
                runs = store.get((env, algo, obs_mode, metric), [])
                if not runs:
                    continue
                steps, vals = align_and_stack(runs)
                if vals.size == 0:
                    continue

                x = steps.astype(float)
                mean = vals.mean(axis=0)
                ci = 1.96 * vals.std(axis=0) / np.sqrt(max(vals.shape[0], 1))
                if args.smoothing_window:
                    mean = moving_average(mean, args.smoothing_window)
                    ci = moving_average(ci, args.smoothing_window)

                line, = ax.plot(x, mean, label=obs_mode,
                                color=OBS_MODE_COLORS.get(obs_mode))
                ax.fill_between(x, mean - ci, mean + ci, alpha=0.25,
                                color=line.get_color())
                handles.setdefault(obs_mode, line)

            ax.set_xlabel("Steps")
            ax.set_ylabel(TRANSLATIONS.get(metric, metric.capitalize()))
            if ax.get_ylim()[1] >= 1000:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.yaxis.get_major_formatter().set_useOffset(False)
            ax.set_xlim(0.0, args.x_max)

            if metric == "cost" and not args.no_threshold:
                thr = ax.axhline(args.threshold, linestyle="--", color="red", linewidth=1.8)
                handles.setdefault("Threshold", thr)
            if args.grid:
                ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.45)

        left = get_ax(env_i, 0).get_position()
        right = get_ax(env_i, m - 1).get_position()
        fig.text(0.5 * (left.x0 + right.x1), max(left.y1, right.y1) + 0.01,
                 TRANSLATIONS.get(env, env), ha="center", va="bottom", fontsize=14)

    for env_i in range(len(envs), nrows * ncols_env):
        for metric_i in range(m):
            get_ax(env_i, metric_i).axis("off")

    _finalize(fig, handles, len(obs_modes), args, algo, "curves")


def _final_value(runs: List[pd.DataFrame], last_k: int) -> Optional[Tuple[float, float, int]]:
    """(mean, 95% CI half-width, n_seeds) of the last `last_k` points, over seeds."""
    if not runs:
        return None
    _, vals = align_and_stack(runs)
    if vals.size == 0:
        return None
    k = min(last_k, vals.shape[1])
    per_seed = vals[:, -k:].mean(axis=1)
    ci = 1.96 * per_seed.std() / np.sqrt(max(per_seed.size, 1))
    return float(per_seed.mean()), float(ci), int(per_seed.size)


def plot_bars(store: RunStore, args: argparse.Namespace, algo: str) -> None:
    """Final-performance bars, one bar per observation modality."""
    set_mpl_style()
    envs, metrics = args.envs, args.metrics
    m = len(metrics)
    nrows, ncols_env = nice_grid(len(envs), max_cols=args.max_cols)

    fig, axs = plt.subplots(nrows, ncols_env * m,
                            figsize=(args.panel_w * ncols_env * m, args.panel_h * nrows),
                            squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.12,
                        wspace=0.35, hspace=0.55)

    def get_ax(env_i: int, metric_i: int):
        return axs[env_i // ncols_env, (env_i % ncols_env) * m + metric_i]

    handles: Dict[str, plt.Line2D] = {}
    obs_modes = _present_obs_modes(store, args, algo)

    for env_i, env in enumerate(envs):
        for metric_i, metric in enumerate(metrics):
            ax = get_ax(env_i, metric_i)

            positions, heights, errors, colors, drawn = [], [], [], [], []
            for i, obs_mode in enumerate(obs_modes):
                stat = _final_value(store.get((env, algo, obs_mode, metric), []), args.last_k)
                if stat is None:
                    continue
                mean, ci, n_seeds = stat
                positions.append(i)
                heights.append(mean)
                errors.append(ci)
                colors.append(OBS_MODE_COLORS.get(obs_mode))
                drawn.append(obs_mode)

            if positions:
                bars = ax.bar(positions, heights, yerr=errors, capsize=4,
                              color=colors, edgecolor="black", linewidth=0.6)
                for obs_mode, bar in zip(drawn, bars):
                    handles.setdefault(obs_mode, bar)

            ax.set_xticks(range(len(obs_modes)))
            # Bars are labelled by the legend; the tick marks only anchor them.
            ax.set_xticklabels([""] * len(obs_modes))
            ax.set_ylabel(TRANSLATIONS.get(metric, metric.capitalize()))
            ax.axhline(0.0, color="black", linewidth=0.8)

            if metric == "cost" and not args.no_threshold:
                thr = ax.axhline(args.threshold, linestyle="--", color="red", linewidth=1.8)
                handles.setdefault("Threshold", thr)
            if args.grid:
                ax.grid(True, axis="y", linestyle="--", linewidth=0.9, alpha=0.45)

        left = get_ax(env_i, 0).get_position()
        right = get_ax(env_i, m - 1).get_position()
        fig.text(0.5 * (left.x0 + right.x1), max(left.y1, right.y1) + 0.01,
                 TRANSLATIONS.get(env, env), ha="center", va="bottom", fontsize=14)

    for env_i in range(len(envs), nrows * ncols_env):
        for metric_i in range(m):
            get_ax(env_i, metric_i).axis("off")

    _finalize(fig, handles, len(obs_modes), args, algo, "bars")


def main(args: argparse.Namespace) -> None:
    store = load_all(args)
    if not any(store.values()):
        raise SystemExit(
            "No data loaded. Check --input/--envs/--algos/--level/--seeds, and that "
            "the pixel runs were downloaded (main_results.py --obs vision)."
        )

    for algo in args.algos:
        if not any(store.get((env, algo, om, metric))
                   for env in args.envs for om in args.obs_modes for metric in args.metrics):
            print(f"No data for algo '{algo}', skipping.")
            continue
        if args.bars:
            plot_bars(store, args, algo)
        else:
            plot_curves(store, args, algo)


def build_args() -> argparse.ArgumentParser:
    p = cli.plot_parser(
        "Compare vector vs. pixel observations (per camera) for CRAX runs.",
        level_arg="single",
        omit=("ci_method", "last_frac"),
        stats=True,
        out_name="obs_comparison",
        envs=["safe_goal_point", "safe_push_point", "safe_circle_point", "safe_reacher"],
        algos=["ppo", "ppo_lag", "p3o", "focops"],
        panel_h=3.0,
    )
    p.add_argument("--obs_modes", type=str, nargs="+", default=list(DEFAULT_OBS_MODES),
                   help="Observation modes to compare. 'vector' is the state-observation "
                        "baseline, 'vision_<camera>' are pixel runs.")
    p.add_argument("--bars", action="store_true",
                   help="Draw final-performance bars instead of training curves.")
    p.add_argument("--x_max", type=float, default=5e8, help="Curves only: upper x limit (env steps)")
    p.add_argument("--last_k", type=int, default=10,
                   help="--bars only: logged points averaged to get each run's final value.")
    p.add_argument("--no_threshold", action="store_true", help="Hide safety threshold lines.")
    return p


if __name__ == "__main__":
    main(build_args().parse_args())

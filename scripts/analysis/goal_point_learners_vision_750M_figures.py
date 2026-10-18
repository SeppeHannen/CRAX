"""The figures of docs/acl/experiments/2026-10-18_goal_point_learners_vision_750M.md.

    .venv/bin/python scripts/analysis/pull_group_history.py goal_point_learners_500M /tmp/learners
    .venv/bin/python scripts/analysis/pull_group_history.py goal_point_learners_vision_500M /tmp/vision_learners --samples 1000
    .venv/bin/python scripts/analysis/pull_group_history.py goal_point_learners_vision_750M /tmp/vision750 --samples 1500
    .venv/bin/python -m scripts.analysis.goal_point_learners_vision_750M_figures /tmp/learners /tmp/vision_learners /tmp/vision750

Six arms — uniform, PLR reward, staged x {scalar blind, additive told} — in three experiments:
sensors only at rate 1 per 655 360-step round (2026-10-15), pixels + sensors at rate 1 per
81 920-step round (2026-10-17, eight times the dual ascent per step), pixels + sensors at
rate 1/8 per 81 920-step round (this run; the 2026-10-15 dual ascent per step, 750 M steps).
Writes ``swing.png`` (how far the evaluations move between consecutive evaluations, per arm
and experiment, against the sampling error), ``level3.png`` (level-3 return and cost through
training, the three experiments overlaid per arm), ``multiplier.png`` (λ and training cost)
and ``trajectories.html`` (every evaluation through the return–cost plane, three experiments);
prints the tables the write-up quotes.
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
from typing import Dict, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from scripts.analysis.goal_point_500M_figures import BUDGET, moving_mean, series  # noqa: E402
from scripts.analysis.goal_point_learners_figures import DISTRIBUTIONS, LEARNERS, MOTIONLESS_RETURN, Arm, colour_of, marker_of  # noqa: E402
from scripts.analysis.trajectory_page import ArmStyle, TrajectoryPage  # noqa: E402

FIGURES = pathlib.Path(__file__).resolve().parents[2] / "docs/acl/experiments/figures/2026-10-18_goal_point_learners_vision_750M"
EVALUATION_EPISODES = 128
LAST_EVALUATIONS = 5
ARMS: Tuple[Arm, ...] = tuple(Arm(distribution, learner) for distribution in ("uniform", "plr", "staged123") for learner in ("scalar_blind", "additive_told"))
MARKER_NAMES = {"x": "cross", "s": "square", "o": "circle"}


@dataclasses.dataclass(frozen=True)
class Experiment:
    label: str
    short: str
    frames: Dict[Arm, pd.DataFrame]
    line_style: str


def load(data: pathlib.Path) -> Dict[Arm, pd.DataFrame]:
    return {arm: pd.read_parquet(data / f"{arm.stem}.parquet") for arm in ARMS}


def label_of(arm: Arm) -> str:
    return f"{dict((d, l) for d, l, _ in DISTRIBUTIONS)[arm.distribution]} · {dict((n, l) for n, l, _ in LEARNERS)[arm.learner]}"


def swing(frame: pd.DataFrame, key: str, last: int = 15, log: bool = False) -> float:
    """Mean absolute change between consecutive evaluations over the last ``last`` of them."""
    values = series(frame, key)[1][-last:]
    if log:
        values = np.log(values)
    return float(np.abs(np.diff(values)).mean())


def sampling_error(frame: pd.DataFrame, last: int = 15) -> float:
    """The standard error of a 128-episode mean return, from the evaluation's own spread."""
    return float(series(frame, "evaluation/deployment/episode_reward_std")[1][-last:].mean() / np.sqrt(EVALUATION_EPISODES))


def end_state(frame: pd.DataFrame, key: str) -> float:
    return float(series(frame, key)[1][-LAST_EVALUATIONS:].mean())


def lambda_oscillation(frame: pd.DataFrame) -> Tuple[float, float]:
    """(sd / mean of the batch-mean λ over the last fifth, dominant period in M steps)."""
    steps, lam = series(frame, "training/lambda_lagr")
    last = slice(-lam.size // 5, None)
    detrended = lam - np.convolve(lam, np.ones(51) / 51, mode="same")
    detrended, t = detrended[25:-25], steps[25:-25]
    spectrum = np.abs(np.fft.rfft(detrended - detrended.mean()))
    spectrum[0] = 0
    frequencies = np.fft.rfftfreq(detrended.size, d=float(np.mean(np.diff(t))))
    return float(lam[last].std() / lam[last].mean()), float(1.0 / frequencies[spectrum.argmax()])


def swing_table(experiments: Sequence[Experiment]) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        row: Dict[str, object] = {"arm": label_of(arm)}
        for experiment in experiments:
            frame = experiment.frames[arm]
            cv, period = lambda_oscillation(frame)
            row[f"{experiment.short} |Δ return|"] = swing(frame, "evaluation/deployment/episode_reward")
            row[f"{experiment.short} |Δ log cost|"] = swing(frame, "evaluation/deployment/episode_cost", log=True)
            row[f"{experiment.short} SE"] = sampling_error(frame)
            row[f"{experiment.short} λ CV"] = cv
            row[f"{experiment.short} λ period"] = period
        rows.append(row)
    return pd.DataFrame(rows)


def end_table(experiments: Sequence[Experiment]) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        row: Dict[str, object] = {"arm": label_of(arm)}
        for experiment in experiments:
            frame = experiment.frames[arm]
            costs = series(frame, "evaluation/deployment/episode_cost")[1]
            row[f"{experiment.short} L3 return"] = end_state(frame, "evaluation/deployment/episode_reward")
            row[f"{experiment.short} L3 cost"] = end_state(frame, "evaluation/deployment/episode_cost")
            row[f"{experiment.short} L3 ≤ budget (last 10)"] = int((costs[-10:] <= BUDGET).sum())
            row[f"{experiment.short} Ω return"] = end_state(frame, "evaluation/uniform/episode_reward")
            row[f"{experiment.short} Ω cost"] = end_state(frame, "evaluation/uniform/episode_cost")
            if "training/lambda_lagr/max" in frame:
                row[f"{experiment.short} λ hardest (last 50)"] = float(series(frame, "training/lambda_lagr/max")[1][-50:].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def swing_figure(experiments: Sequence[Experiment], figures: pathlib.Path) -> None:
    """Per arm, the evaluation-to-evaluation swing of level-3 return in each experiment, as bars,
    with the sampling error of one evaluation as a tick: a swing above the tick is the policy moving."""
    fig, axis = plt.subplots(figsize=(11, 4.2))
    width = 0.26
    for k, experiment in enumerate(experiments):
        xs = np.arange(len(ARMS)) + (k - 1) * width
        swings = [swing(experiment.frames[arm], "evaluation/deployment/episode_reward") for arm in ARMS]
        errors = [sampling_error(experiment.frames[arm]) for arm in ARMS]
        bars = axis.bar(xs, swings, width, color=[colour_of(arm.distribution) for arm in ARMS], alpha=(0.45, 0.75, 1.0)[k],
                        edgecolor="k" if k == 2 else "none", label=experiment.label)
        axis.scatter(xs, errors, marker="_", color="k", s=160, zorder=4, label="sampling error of one evaluation" if k == 0 else None)
        for bar, arm in zip(bars, ARMS):
            if arm.learner == "scalar_blind":
                bar.set_hatch("//")
    axis.set_xticks(np.arange(len(ARMS)))
    axis.set_xticklabels([label_of(arm).replace(" · ", "\n") for arm in ARMS], fontsize=8)
    axis.set_ylabel("mean |Δ level-3 return| between consecutive evaluations (last 15)")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=8)
    fig.suptitle("How far the frozen policy moves between evaluations 25 M steps apart: the three experiments per arm", y=1.0)
    fig.tight_layout()
    fig.savefig(figures / "swing.png", bbox_inches="tight")
    plt.close(fig)


def level3_figure(experiments: Sequence[Experiment], figures: pathlib.Path) -> None:
    """Level-3 return and cost through training, one column per arm, the three experiments overlaid."""
    fig, axes = plt.subplots(2, len(ARMS), figsize=(17, 6), sharex=True, sharey="row")
    for column, arm in enumerate(ARMS):
        for row, key in enumerate(("evaluation/deployment/episode_reward", "evaluation/deployment/episode_cost")):
            axis = axes[row, column]
            for experiment in experiments:
                steps, values = series(experiment.frames[arm], key)
                axis.plot(steps, values, experiment.line_style, ms=2.5, lw=1.0, color=colour_of(arm.distribution), alpha=(0.5, 0.75, 1.0)[experiments.index(experiment)], label=experiment.label)
            if "cost" in key:
                axis.axhline(BUDGET, color="k", ls="--", lw=0.8)
                axis.set_yscale("log")
                axis.set_ylim(1, 400)
            else:
                axis.axhspan(0, MOTIONLESS_RETURN, color="grey", alpha=0.12, lw=0)
            axis.grid(alpha=0.3)
        axes[0, column].set_title(label_of(arm).replace(" · ", "\n"), fontsize=9)
        axes[1, column].set_xlabel("environment steps (M)")
    axes[0, 0].set_ylabel("goals per episode")
    axes[1, 0].set_ylabel("cost per episode (log)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("The frozen policy on level 3 through training, 128 episodes per point, one seed; the three experiments overlaid (light → dark)", y=1.0)
    fig.tight_layout()
    fig.savefig(figures / "level3.png", bbox_inches="tight")
    plt.close(fig)


def multiplier_figure(experiments: Sequence[Experiment], figures: pathlib.Path) -> None:
    """λ (batch mean, 10-round moving mean) and the training cost per round, per arm, the three experiments overlaid."""
    fig, axes = plt.subplots(2, len(ARMS), figsize=(17, 6), sharex=True, sharey="row")
    window = 10
    for column, arm in enumerate(ARMS):
        for experiment in experiments:
            frame = experiment.frames[arm]
            alpha = (0.5, 0.75, 1.0)[experiments.index(experiment)]
            steps, lam = series(frame, "training/lambda_lagr")
            axes[0, column].plot(steps[window - 1:], moving_mean(lam, window), lw=1.0, color=colour_of(arm.distribution), alpha=alpha, label=experiment.label)
            steps, cost = series(frame, "episodic/cost")
            axes[1, column].plot(steps[window - 1:], moving_mean(cost, window), lw=1.0, color=colour_of(arm.distribution), alpha=alpha)
        axes[0, column].set_title(label_of(arm).replace(" · ", "\n"), fontsize=9)
        axes[0, column].set_yscale("log")
        axes[0, column].set_ylim(0.05, 40)
        axes[0, column].grid(alpha=0.3, which="both")
        axes[1, column].axhline(BUDGET, color="k", ls="--", lw=0.8)
        axes[1, column].set_ylim(0, 80)
        axes[1, column].grid(alpha=0.3)
        axes[1, column].set_xlabel("environment steps (M)")
    axes[0, 0].set_ylabel("λ, batch mean (log)")
    axes[1, 0].set_ylabel("training cost per episode")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"The multiplier and the training cost it reacts to, {window}-round moving mean; the three experiments overlaid", y=1.0)
    fig.tight_layout()
    fig.savefig(figures / "multiplier.png", bbox_inches="tight")
    plt.close(fig)


def trajectory_page(experiments: Sequence[Experiment], figures: pathlib.Path) -> None:
    page = TrajectoryPage(
        title="Six arms through the return–cost plane, one evaluation per step (128 episodes): sensors (2026-10-15), pixels at 8x the dual ascent (2026-10-17), pixels rate-matched (2026-10-18)",
        targets=(("deployment", "Level 3 (w): the deployment task"), ("uniform", "Uniform over Ω")),
    )
    styles = {label_of(arm): ArmStyle(colour_of(arm.distribution), MARKER_NAMES[marker_of(arm.learner)],
                                      dict((d, l) for d, l, _ in DISTRIBUTIONS)[arm.distribution], dict((n, l) for n, l, _ in LEARNERS)[arm.learner]) for arm in ARMS}
    for experiment in experiments:
        page.add_experiment(experiment.label, {label_of(arm): experiment.frames[arm] for arm in ARMS}, styles)
    page.write(figures / "trajectories.html")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sensors", type=pathlib.Path)
    parser.add_argument("pixels_rate1", type=pathlib.Path)
    parser.add_argument("pixels_matched", type=pathlib.Path)
    arguments = parser.parse_args()
    experiments = (
        Experiment("sensors, rate 1 / 655k-step round (2026-10-15, 500 M)", "S", load(arguments.sensors), "o-"),
        Experiment("pixels, rate 1 / 82k-step round = 8x (2026-10-17, 500 M)", "V8", load(arguments.pixels_rate1), "s-"),
        Experiment("pixels, rate 1/8 / 82k-step round = matched (2026-10-18, 750 M)", "V1", load(arguments.pixels_matched), "D-"),
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    swing_figure(experiments, FIGURES)
    level3_figure(experiments, FIGURES)
    multiplier_figure(experiments, FIGURES)
    trajectory_page(experiments, FIGURES)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 60)
    print(swing_table(experiments).round(2).to_string(index=False))
    print()
    print(end_table(experiments).round(2).to_string(index=False))
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()

"""The three figures of docs/acl/experiments/2026-10-11_goal_point_plr_scores_500M.md.

    .venv/bin/python scripts/analysis/pull_group_history.py goal_point_plr_scores_500M /tmp/plr_scores
    .venv/bin/python scripts/analysis/goal_point_plr_scores_figures.py /tmp/plr_scores

Reads one parquet per arm (uniform, staged123, plr, plrcost, plrlagrangian) and
writes evaluation.png, training.png and plr.png into the experiment's figures
directory. Shares its helpers with goal_point_500M_figures.py.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Dict, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from scripts.analysis.goal_point_500M_figures import BUDGET, DIMENSION_WORDS, LEVEL_3, moving_mean, series  # noqa: E402

ARMS: Tuple[Tuple[str, str, str], ...] = (
    ("uniform", "uniform", "#e377c2"),
    ("staged123", "staged 1,2,3", "#2ca02c"),
    ("plr", "PLR reward", "#1f77b4"),
    ("plrcost", "PLR cost", "#d62728"),
    ("plrlagrangian", "PLR Lagrangian", "#ff7f0e"),
)
PLR_ARMS = tuple(arm for arm in ARMS if arm[0].startswith("plr"))
FIGURES = pathlib.Path(__file__).resolve().parents[2] / "docs/acl/experiments/figures/2026-10-11_goal_point_plr_scores_500M"
Frames = Dict[str, pd.DataFrame]


def evaluation_figure(frames: Frames) -> None:
    """Return and cost of the frozen policy on the three targets; Ω₂ only exists for the two new arms."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    panels = (
        ("evaluation/deployment/episode_reward", "Return on level 3 (w)", axes[0, 0]),
        ("evaluation/uniform/episode_reward", "Return on Ω", axes[0, 1]),
        ("evaluation/out_of_distribution/episode_reward", "Return on Ω₂ (placeholder; new arms only)", axes[0, 2]),
        ("evaluation/deployment/episode_cost", "Cost on level 3 (w)", axes[1, 0]),
        ("evaluation/uniform/episode_cost", "Cost on Ω", axes[1, 1]),
        ("evaluation/out_of_distribution/episode_cost", "Cost on Ω₂", axes[1, 2]),
    )
    for key, title, axis in panels:
        for arm, label, colour in ARMS:
            if key not in frames[arm]:
                continue
            steps, values = series(frames[arm], key)
            axis.plot(steps, values, marker="o", ms=3, lw=1.2, color=colour, label=label)
        if "cost" in key:
            axis.axhline(BUDGET, color="k", ls="--", lw=0.8, label="budget 25")
            axis.set_yscale("log")
            axis.set_ylim(1, 400)
        axis.set_title(title)
        axis.grid(alpha=0.3)
    for axis in axes[1]:
        axis.set_xlabel("environment steps (M)")
    axes[0, 0].set_ylabel("goals per episode")
    axes[1, 0].set_ylabel("cost per episode (log)")
    axes[0, 0].legend(loc="lower right", fontsize=7)
    fig.suptitle("Evaluation of the frozen policy, 128 episodes per point, one seed", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES / "evaluation.png", bbox_inches="tight")
    plt.close(fig)


def training_figure(frames: Frames) -> None:
    """λ, training cost and training return per round: the controller decides every end state."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    panels = (
        ("training/lambda_lagr", "Lagrange multiplier λ", axes[0]),
        ("episodic/cost", "Training cost per episode (what λ sees)", axes[1]),
        ("episodic/sum_reward", "Training return per episode", axes[2]),
    )
    window = 10
    for key, title, axis in panels:
        for arm, label, colour in ARMS:
            steps, values = series(frames[arm], key)
            axis.plot(steps[window - 1:], moving_mean(values, window), lw=1.2, color=colour, label=label)
        if key == "episodic/cost":
            axis.axhline(BUDGET, color="k", ls="--", lw=0.8, label="budget 25")
            axis.set_ylim(0, 70)
        axis.set_title(title)
        axis.grid(alpha=0.3)
    axes[0].set_ylim(0, 42)
    axes[0].legend(loc="upper left", ncol=3)
    axes[2].set_xlabel("environment steps (M)")
    fig.suptitle(f"Training side, one point per round, {window}-round moving mean", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES / "training.png", bbox_inches="tight")
    plt.close(fig)


def plr_figure(frames: Frames) -> None:
    """Left: total active hazards each PLR variant intended, against Uniform's. Right: each variant's score spread."""
    fig, (mass_axis, score_axis) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    window = 20
    hazard_dimensions = [dimension for dimension, _ in DIMENSION_WORDS if dimension != "goal_size"]

    def total_hazards(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        steps, first = series(frame, f"training_curriculum/intended/{hazard_dimensions[0]}/mean")
        total = first.copy()
        for dimension in hazard_dimensions[1:]:
            total = total + series(frame, f"training_curriculum/intended/{dimension}/mean")[1]
        return steps, total

    _, uniform_total = total_hazards(frames["uniform"])
    mass_axis.axhline(uniform_total.mean(), color="#e377c2", lw=1.0, label=f"uniform ({uniform_total.mean():.1f})")
    mass_axis.axhline(sum(LEVEL_3[dimension] for dimension in hazard_dimensions), color="k", ls="--", lw=0.8, label="level 3 (20)")
    for arm, label, colour in PLR_ARMS:
        steps, total = total_hazards(frames[arm])
        mass_axis.plot(steps[window - 1:], moving_mean(total, window), lw=1.2, color=colour, label=label)
    mass_axis.set_ylabel("intended mean of active hazards (sum over the four groups)")
    mass_axis.set_xlabel("environment steps (M)")
    mass_axis.set_title("Where each PLR variant put its mass")
    mass_axis.legend(fontsize=7, loc="lower right")
    mass_axis.grid(alpha=0.3)

    for arm, label, colour in PLR_ARMS:
        steps, score_max = series(frames[arm], "training_curriculum/distribution/score/max")
        _, score_mean = series(frames[arm], "training_curriculum/distribution/score/mean")
        score_axis.plot(steps[window - 1:], moving_mean(score_max / score_mean, window), lw=1.2, color=colour, label=label)
    score_axis.set_yscale("log")
    score_axis.set_xlabel("environment steps (M)")
    score_axis.set_ylabel("max ÷ mean score in the buffer")
    score_axis.set_title("How much there is to rank")
    score_axis.legend(fontsize=7)
    score_axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "plr.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("history_dir", type=pathlib.Path, help="directory written by pull_group_history.py")
    arguments = parser.parse_args()
    frames = {arm: pd.read_parquet(arguments.history_dir / f"{arm}.parquet").sort_values("environment_steps") for arm, _, _ in ARMS}
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8, "figure.dpi": 150})
    FIGURES.mkdir(parents=True, exist_ok=True)
    evaluation_figure(frames)
    training_figure(frames)
    plr_figure(frames)
    print(f"wrote {FIGURES}")


if __name__ == "__main__":
    main()

"""The three figures of docs/acl/experiments/2026-10-08_goal_point_uniform_staged_plr_500M.md.

    .venv/bin/python scripts/analysis/pull_group_history.py goal_point_uniform_staged_plr_500M_v2 /tmp/v2
    .venv/bin/python scripts/analysis/goal_point_500M_figures.py /tmp/v2

Reads one parquet per arm (uniform, staged123, plr) and writes evaluation.png,
training.png and plr.png into the experiment's figures directory.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

ARMS = (("uniform", "uniform", "#e377c2"), ("staged123", "staged 1,2,3", "#2ca02c"), ("plr", "PLR", "#1f77b4"))
BUDGET = 25.0
STEPS_PER_ROUND = 655_360
STAGED_SWITCH_ROUNDS = (254, 509)
LEVEL_3 = {"active_cylinders": 6, "active_collidable_cylinders": 4, "active_cubes": 6, "active_collidable_cubes": 4, "goal_size": 0.16}
DIMENSION_WORDS = (
    ("active_cylinders", "flat discs"), ("active_collidable_cylinders", "solid cylinders"),
    ("active_cubes", "flat squares"), ("active_collidable_cubes", "solid cubes"), ("goal_size", "goal radius"),
)
FIGURES = pathlib.Path(__file__).resolve().parents[2] / "docs/acl/experiments/figures/2026-10-08_goal_point_500M"


def series(frame: pd.DataFrame, key: str) -> Tuple[np.ndarray, np.ndarray]:
    """``(environment steps in millions, values)`` of one logged key, rows where it is present."""
    rows = frame[["environment_steps", key]].dropna()
    return rows["environment_steps"].to_numpy() / 1e6, rows[key].astype(float).to_numpy()


def moving_mean(values: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(values, np.ones(window) / window, mode="valid")


def evaluation_figure(frames) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    panels = (
        ("evaluation/deployment/episode_reward", "Return on level 3 (w)", axes[0, 0]),
        ("evaluation/uniform/episode_reward", "Return on Ω", axes[0, 1]),
        ("evaluation/deployment/episode_cost", "Cost on level 3 (w)", axes[1, 0]),
        ("evaluation/uniform/episode_cost", "Cost on Ω", axes[1, 1]),
    )
    for key, title, axis in panels:
        for arm, label, colour in ARMS:
            steps, values = series(frames[arm], key)
            axis.plot(steps, values, marker="o", ms=3, lw=1.2, color=colour, label=label)
        if "cost" in key:
            axis.axhline(BUDGET, color="k", ls="--", lw=0.8, label="budget 25")
            axis.set_yscale("log")
            axis.set_ylim(0.1, 400)
        axis.set_title(title)
        axis.grid(alpha=0.3)
    axes[1, 0].set_xlabel("environment steps (M)")
    axes[1, 1].set_xlabel("environment steps (M)")
    axes[0, 0].set_ylabel("goals per episode")
    axes[1, 0].set_ylabel("cost per episode (log)")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Evaluation of the frozen policy, 128 episodes per point, one seed", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES / "evaluation.png", bbox_inches="tight")
    plt.close(fig)


def training_figure(frames) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
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
        for switch in STAGED_SWITCH_ROUNDS:
            axis.axvline(switch * STEPS_PER_ROUND / 1e6, color="#2ca02c", ls=":", lw=0.8)
        axis.set_title(title)
        axis.grid(alpha=0.3)
    axes[0].legend(loc="upper left")
    axes[2].set_xlabel("environment steps (M)  — dotted: staged's switches to level 2 and 3")
    fig.suptitle(f"Training side, one point per round, {window}-round moving mean", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES / "training.png", bbox_inches="tight")
    plt.close(fig)


def plr_figure(frames) -> None:
    fig, (mass_axis, score_axis) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [3, 2]})
    window = 20
    for dimension, words in DIMENSION_WORDS:
        key = f"training_curriculum/intended/{dimension}/mean"
        _, uniform_values = series(frames["uniform"], key)
        steps, plr_values = series(frames["plr"], key)
        uniform_mean = uniform_values.mean()
        mass_axis.plot(steps[window - 1:], moving_mean(plr_values / uniform_mean, window), lw=1.2,
                       label=f"{words} (level 3: {LEVEL_3[dimension] / uniform_mean:.2f}×)")
    mass_axis.axhline(1.0, color="k", lw=0.8)
    mass_axis.set_ylabel("PLR intended mean ÷ Uniform's mean")
    mass_axis.set_xlabel("environment steps (M)")
    mass_axis.set_title("Where PLR put its mass, per dimension of Ω")
    mass_axis.legend(fontsize=7, loc="upper left", ncol=2)
    mass_axis.grid(alpha=0.3)
    mass_axis.set_ylim(0.9, 1.25)

    steps, score_max = series(frames["plr"], "training_curriculum/distribution/score/max")
    _, score_mean = series(frames["plr"], "training_curriculum/distribution/score/mean")
    score_axis.plot(steps, score_max, lw=1, label="max score in buffer")
    score_axis.plot(steps, score_mean, lw=1, label="mean score in buffer")
    score_axis.set_yscale("log")
    score_axis.set_ylim(0.005, 0.2)
    score_axis.set_xlabel("environment steps (M)")
    score_axis.set_ylabel("mean |reward advantage| per episode")
    score_axis.set_title("PLR's score: little spread to rank")
    score_axis.legend()
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

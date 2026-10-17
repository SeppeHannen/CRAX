"""The figures of docs/acl/experiments/2026-10-15_goal_point_learners_500M.md.

    .venv/bin/python scripts/analysis/pull_group_history.py goal_point_learners_500M /tmp/learners
    .venv/bin/python -m scripts.analysis.goal_point_learners_figures /tmp/learners

Reads one parquet per arm (``<distribution>_<constraint>_<policy_context>.parquet``,
fifteen of them) and writes ``end_state.png``, ``level3.png`` and ``multiplier.png``
into the experiment's figures directory; prints the numbers the write-up quotes.
Shares its helpers with goal_point_500M_figures.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
from typing import Dict, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from scripts.analysis.goal_point_500M_figures import BUDGET, DIMENSION_WORDS, moving_mean, series  # noqa: E402

FIGURES = pathlib.Path(__file__).resolve().parents[2] / "docs/acl/experiments/figures/2026-10-15_goal_point_learners_500M"
LAST_EVALUATIONS = 5  # the end state is the mean of the last five evaluations (400-510 M)
MOTIONLESS_RETURN = 5.0  # below this many goals per episode on level 3 the policy is, in effect, standing still

DISTRIBUTIONS: Tuple[Tuple[str, str, str], ...] = (
    ("uniform", "uniform", "#e377c2"),
    ("staged123", "staged 1,2,3", "#2ca02c"),
    ("plr", "PLR reward", "#1f77b4"),
    ("plrcost", "PLR cost", "#d62728"),
    ("plrlagrangian", "PLR Lagrangian", "#ff7f0e"),
)
LEARNERS: Tuple[Tuple[str, str, str], ...] = (
    ("scalar_blind", "scalar λ, blind policy", "x"),
    ("network_told", "λ_ψ(ω) network, told policy", "s"),
    ("additive_told", "λ(ω) additive, told policy", "o"),
)
HAZARD_DIMENSIONS = tuple(dimension for dimension, _ in DIMENSION_WORDS if dimension != "goal_size")


@dataclasses.dataclass(frozen=True)
class Arm:
    distribution: str
    learner: str

    @property
    def stem(self) -> str:
        return f"{self.distribution}_{self.learner}"


ARMS = tuple(Arm(distribution, learner) for distribution, _, _ in DISTRIBUTIONS for learner, _, _ in LEARNERS)
Frames = Dict[Arm, pd.DataFrame]


def colour_of(distribution: str) -> str:
    return next(colour for name, _, colour in DISTRIBUTIONS if name == distribution)


def marker_of(learner: str) -> str:
    return next(marker for name, _, marker in LEARNERS if name == learner)


def end_state(frame: pd.DataFrame, key: str) -> float:
    """Mean of the last evaluations of one key."""
    return float(series(frame, key)[1][-LAST_EVALUATIONS:].mean())


def end_states(frames: Frames) -> pd.DataFrame:
    """One row per arm: return and cost on each target at the end, λ's peak, PLR's mass."""
    rows = []
    for arm, frame in frames.items():
        row = {
            "distribution": arm.distribution, "learner": arm.learner,
            "L3 return": end_state(frame, "evaluation/deployment/episode_reward"),
            "L3 cost": end_state(frame, "evaluation/deployment/episode_cost"),
            "Ω return": end_state(frame, "evaluation/uniform/episode_reward"),
            "Ω cost": end_state(frame, "evaluation/uniform/episode_cost"),
            "Ω₂ return": end_state(frame, "evaluation/out_of_distribution/episode_reward"),
            "Ω₂ cost": end_state(frame, "evaluation/out_of_distribution/episode_cost"),
            "Ω under budget": int((series(frame, "evaluation/uniform/episode_cost")[1] <= BUDGET).sum()),
            "L3 under budget": int((series(frame, "evaluation/deployment/episode_cost")[1] <= BUDGET).sum()),
            "λ peak": float(series(frame, "training/lambda_lagr")[1].max()),
            "training cost (last 100)": float(series(frame, "episodic/cost")[1][-100:].mean()),
            "training return (last 100)": float(series(frame, "episodic/sum_reward")[1][-100:].mean()),
        }
        if "training/lambda_lagr/max" in frame:
            row["λ max peak"] = float(series(frame, "training/lambda_lagr/max")[1].max())
            row["fit error (last 100)"] = float(series(frame, "training/lambda_lagr/fit_error")[1][-100:].mean())
            row["target spread (last 100)"] = float(series(frame, "training/lambda_lagr/target_spread")[1][-100:].mean())
        if arm.distribution.startswith("plr"):
            row["intended hazards (last 100)"] = float(total_hazards(frame)[1][-100:].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def total_hazards(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Sum over the four hazard groups of the intended mean count per round."""
    steps, total = series(frame, f"training_curriculum/intended/{HAZARD_DIMENSIONS[0]}/mean")
    for dimension in HAZARD_DIMENSIONS[1:]:
        total = total + series(frame, f"training_curriculum/intended/{dimension}/mean")[1]
    return steps, total


def end_state_figure(frames: Frames, table: pd.DataFrame) -> None:
    """Every arm as one point in the return–cost plane of level 3 and of Ω: colour says the
    distribution, marker says the learner."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for axis, target, title in ((axes[0], "L3", "Level 3 (w): the deployment task"), (axes[1], "Ω", "Uniform over Ω")):
        for _, row in table.iterrows():
            colour = colour_of(row["distribution"])
            edge = {} if row["learner"] == "scalar_blind" else {"edgecolors": "k"}
            axis.scatter(row[f"{target} cost"], row[f"{target} return"], s=70, color=colour, marker=marker_of(row["learner"]), linewidths=1.6, zorder=3, **edge)
        axis.axvline(BUDGET, color="k", ls="--", lw=0.8)
        axis.text(BUDGET * 1.03, 0.5, "budget 25", fontsize=8, rotation=90, va="bottom")
        axis.axhspan(0, MOTIONLESS_RETURN, color="grey", alpha=0.12, lw=0)
        axis.text(60, MOTIONLESS_RETURN / 2, "standing still", fontsize=8, ha="right", va="center", color="dimgrey")
        axis.set_xscale("log")
        axis.set_xlim(5, 110)
        axis.set_ylim(0, 27)
        axis.set_xlabel("cost per episode (log)")
        axis.set_title(title)
        axis.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("goals per episode")
    distribution_handles = [plt.Line2D([], [], color=colour, marker="o", ls="", ms=7, label=label) for _, label, colour in DISTRIBUTIONS]
    learner_handles = [
        plt.Line2D([], [], color="grey", marker=marker, ls="", ms=7, markeredgecolor="k" if marker != "x" else "grey", label=label)
        for _, label, marker in LEARNERS
    ]
    axes[1].legend(handles=distribution_handles + learner_handles, fontsize=7, loc="lower right")
    fig.suptitle(f"End state of the fifteen arms: mean of the last {LAST_EVALUATIONS} evaluations (400–510 M), 128 episodes each, one seed", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES / "end_state.png", bbox_inches="tight")
    plt.close(fig)


def level3_figure(frames: Frames) -> None:
    """Return and cost on level 3 through training, one column per learner, one line per distribution."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey="row")
    for column, (learner, learner_label, _) in enumerate(LEARNERS):
        for row, (key, ylabel) in enumerate((("evaluation/deployment/episode_reward", "goals per episode"), ("evaluation/deployment/episode_cost", "cost per episode (log)"))):
            axis = axes[row, column]
            for distribution, label, colour in DISTRIBUTIONS:
                steps, values = series(frames[Arm(distribution, learner)], key)
                axis.plot(steps, values, marker="o", ms=2.5, lw=1.1, color=colour, label=label)
            if "cost" in key:
                axis.axhline(BUDGET, color="k", ls="--", lw=0.8, label="budget 25")
                axis.set_yscale("log")
                axis.set_ylim(1, 400)
            else:
                axis.axhspan(0, MOTIONLESS_RETURN, color="grey", alpha=0.12, lw=0)
            axis.grid(alpha=0.3)
            if column == 0:
                axis.set_ylabel(ylabel)
        axes[0, column].set_title(learner_label)
        axes[1, column].set_xlabel("environment steps (M)")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("The frozen policy on level 3 through training, 128 episodes per point, one seed", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES / "level3.png", bbox_inches="tight")
    plt.close(fig)


def multiplier_figure(frames: Frames) -> None:
    """What the constraint did: the multiplier the batch was weighted by on average (solid) and at the
    round's hardest context (dashed; a per-context arm only), and the training cost it reacted to."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey="row")
    window = 10
    for column, (learner, learner_label, _) in enumerate(LEARNERS):
        lambda_axis, cost_axis = axes[0, column], axes[1, column]
        for distribution, label, colour in DISTRIBUTIONS:
            frame = frames[Arm(distribution, learner)]
            steps, lam = series(frame, "training/lambda_lagr")
            lambda_axis.plot(steps[window - 1:], moving_mean(lam, window), lw=1.2, color=colour, label=label)
            if "training/lambda_lagr/max" in frame:
                steps, high = series(frame, "training/lambda_lagr/max")
                lambda_axis.plot(steps[window - 1:], moving_mean(high, window), lw=0.9, ls="--", color=colour)
            steps, cost = series(frame, "episodic/cost")
            cost_axis.plot(steps[window - 1:], moving_mean(cost, window), lw=1.2, color=colour, label=label)
        lambda_axis.set_title(learner_label)
        lambda_axis.set_yscale("log")
        lambda_axis.set_ylim(0.05, 40)
        lambda_axis.grid(alpha=0.3, which="both")
        cost_axis.axhline(BUDGET, color="k", ls="--", lw=0.8)
        cost_axis.set_ylim(0, 70)
        cost_axis.grid(alpha=0.3)
        cost_axis.set_xlabel("environment steps (M)")
    axes[0, 0].set_ylabel("λ: batch mean (solid), hardest context (dashed)")
    axes[1, 0].set_ylabel("training cost per episode")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"The multiplier and the training cost it reacts to, {window}-round moving mean, one seed", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES / "multiplier.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=pathlib.Path, help="directory of <arm>.parquet files from pull_group_history.py")
    arguments = parser.parse_args()
    frames: Frames = {arm: pd.read_parquet(arguments.data / f"{arm.stem}.parquet") for arm in ARMS}
    FIGURES.mkdir(parents=True, exist_ok=True)
    table = end_states(frames)
    end_state_figure(frames, table)
    level3_figure(frames)
    multiplier_figure(frames)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(table.sort_values(["learner", "distribution"]).round(2).to_string(index=False))
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()

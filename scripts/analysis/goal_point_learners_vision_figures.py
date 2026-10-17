"""The figures of docs/acl/experiments/2026-10-17_goal_point_learners_vision_500M.md.

    .venv/bin/python scripts/analysis/pull_group_history.py goal_point_learners_500M /tmp/learners
    .venv/bin/python scripts/analysis/pull_group_history.py goal_point_learners_vision_500M /tmp/vision_learners
    .venv/bin/python -m scripts.analysis.goal_point_learners_vision_figures /tmp/learners /tmp/vision_learners

The same fifteen arms as the 2026-10-15 experiment, the policy now seeing pixels next to
the sensors. Writes ``paired.png`` (each arm's move in the return–cost plane from the
state-based to the vision run), and the experiment's own ``end_state.png``, ``level3.png``
and ``multiplier.png`` by the 2026-10-15 script's functions; prints the numbers the
write-up quotes, side by side for the two experiments.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Dict

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from scripts.analysis.goal_point_500M_figures import BUDGET, series  # noqa: E402
from scripts.analysis.goal_point_learners_figures import (  # noqa: E402
    ARMS, DISTRIBUTIONS, LAST_EVALUATIONS, LEARNERS, MOTIONLESS_RETURN, Frames,
    colour_of, end_state, end_state_figure, end_states, level3_figure, load_frames, marker_of, multiplier_figure,
)
from scripts.analysis.trajectory_page import ArmStyle, TrajectoryPage  # noqa: E402

FIGURES = pathlib.Path(__file__).resolve().parents[2] / "docs/acl/experiments/figures/2026-10-17_goal_point_learners_vision_500M"
RECOVERED_RETURN = 8.0  # the 2026-10-15 write-up's threshold for "moving again" on level 3
MARKER_NAMES = {"x": "cross", "s": "square", "o": "circle"}  # matplotlib marker -> the page's


def trajectory_page(state: Frames, vision: Frames, figures: pathlib.Path) -> None:
    """Every evaluation of every arm of both experiments, played through the return–cost plane."""
    page = TrajectoryPage(
        title="Fifteen arms through the return–cost plane, one evaluation per step (128 episodes); sensors-only (2026-10-15) and pixels+sensors (2026-10-17)",
        targets=(("deployment", "Level 3 (w): the deployment task"), ("uniform", "Uniform over Ω")),
    )
    labels = {arm: f"{dict((d, l) for d, l, _ in DISTRIBUTIONS)[arm.distribution]} · {dict((n, l) for n, l, _ in LEARNERS)[arm.learner]}" for arm in ARMS}
    styles = {labels[arm]: ArmStyle(colour_of(arm.distribution), MARKER_NAMES[marker_of(arm.learner)],
                                    dict((d, l) for d, l, _ in DISTRIBUTIONS)[arm.distribution], dict((n, l) for n, l, _ in LEARNERS)[arm.learner]) for arm in ARMS}
    page.add_experiment("sensors only (2026-10-15)", {labels[arm]: state[arm] for arm in ARMS}, styles)
    page.add_experiment("pixels + sensors (2026-10-17)", {labels[arm]: vision[arm] for arm in ARMS}, styles)
    page.write(figures / "trajectories.html")


def paired_figure(state: Frames, vision: Frames, figures: pathlib.Path) -> None:
    """Each arm as an arrow from its state-based end state (open marker) to its vision end state
    (filled marker) in the return–cost plane of level 3 and of Ω."""
    state_table = end_states(state).set_index(["distribution", "learner"])
    vision_table = end_states(vision).set_index(["distribution", "learner"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for axis, target, title in ((axes[0], "L3", "Level 3 (w): the deployment task"), (axes[1], "Ω", "Uniform over Ω")):
        for arm in ARMS:
            key = (arm.distribution, arm.learner)
            before = (state_table.loc[key, f"{target} cost"], state_table.loc[key, f"{target} return"])
            after = (vision_table.loc[key, f"{target} cost"], vision_table.loc[key, f"{target} return"])
            colour, marker = colour_of(arm.distribution), marker_of(arm.learner)
            axis.annotate("", xy=after, xytext=before, arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.0, alpha=0.8, shrinkA=4, shrinkB=4))
            axis.scatter(*before, s=55, facecolors="none", edgecolors=colour, marker=marker, linewidths=1.4, zorder=3)
            axis.scatter(*after, s=70, color=colour, marker=marker, linewidths=1.6, zorder=4, **({} if marker == "x" else {"edgecolors": "k"}))
        axis.axvline(BUDGET, color="k", ls="--", lw=0.8)
        axis.text(BUDGET * 1.03, 0.5, "budget 25", fontsize=8, rotation=90, va="bottom")
        axis.axhspan(0, MOTIONLESS_RETURN, color="grey", alpha=0.12, lw=0)
        axis.set_xscale("log")
        axis.set_xlim(5, 110)
        axis.set_ylim(0, 27)
        axis.set_xlabel("cost per episode (log)")
        axis.set_title(title)
        axis.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("goals per episode")
    handles = [plt.Line2D([], [], color=colour, marker="o", ls="", ms=7, label=label) for _, label, colour in DISTRIBUTIONS]
    handles += [
        plt.Line2D([], [], color="grey", marker=marker, ls="", ms=7, markeredgecolor="k" if marker != "x" else "grey", label=label)
        for _, label, marker in LEARNERS
    ]
    handles += [
        plt.Line2D([], [], color="grey", marker="o", ls="", ms=7, markerfacecolor="none", label="sensors only (2026-10-15)"),
        plt.Line2D([], [], color="grey", marker="o", ls="", ms=7, markeredgecolor="k", label="pixels + sensors (this run)"),
    ]
    axes[1].legend(handles=handles, fontsize=7, loc="lower right")
    fig.suptitle(f"The same fifteen arms, the observation changed: mean of the last {LAST_EVALUATIONS} evaluations, 128 episodes each, one seed", y=1.0)
    fig.tight_layout()
    fig.savefig(figures / "paired.png", bbox_inches="tight")
    plt.close(fig)


def first_recovery(frame: pd.DataFrame) -> float:
    """Environment steps (M) of the first evaluation at which level-3 return is back above
    RECOVERED_RETURN after having been below MOTIONLESS_RETURN; NaN if it never dips or never recovers."""
    steps, returns = series(frame, "evaluation/deployment/episode_reward")
    dipped = np.flatnonzero(returns < MOTIONLESS_RETURN)
    if dipped.size == 0:
        return float("nan")
    recovered = np.flatnonzero((returns > RECOVERED_RETURN) & (np.arange(returns.size) > dipped[0]))
    return float(steps[recovered[0]]) if recovered.size else float("nan")


def comparison_table(state: Frames, vision: Frames) -> pd.DataFrame:
    """One row per arm: level-3 and Ω end states of both experiments, the budget count, the
    evaluation's cost spread relative to its mean (the tail the mean hides), and recovery times."""
    rows = []
    for arm in ARMS:
        row: Dict[str, object] = {"distribution": arm.distribution, "learner": arm.learner}
        for label, frames in (("state", state), ("vision", vision)):
            frame = frames[arm]
            row[f"{label} L3 return"] = end_state(frame, "evaluation/deployment/episode_reward")
            row[f"{label} L3 cost"] = end_state(frame, "evaluation/deployment/episode_cost")
            row[f"{label} L3 ≤ budget"] = int((series(frame, "evaluation/deployment/episode_cost")[1][-10:] <= BUDGET).sum())
            row[f"{label} Ω return"] = end_state(frame, "evaluation/uniform/episode_reward")
            row[f"{label} Ω cost"] = end_state(frame, "evaluation/uniform/episode_cost")
            cost_sd = end_state(frame, "evaluation/deployment/episode_cost_std")
            row[f"{label} L3 cost CV"] = cost_sd / row[f"{label} L3 cost"]
            row[f"{label} recovered (M)"] = first_recovery(frame)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("state_data", type=pathlib.Path, help="<arm>.parquet files of goal_point_learners_500M")
    parser.add_argument("vision_data", type=pathlib.Path, help="<arm>.parquet files of goal_point_learners_vision_500M")
    arguments = parser.parse_args()
    state, vision = load_frames(arguments.state_data), load_frames(arguments.vision_data)
    FIGURES.mkdir(parents=True, exist_ok=True)
    paired_figure(state, vision, FIGURES)
    trajectory_page(state, vision, FIGURES)
    end_state_figure(vision, end_states(vision), FIGURES)
    level3_figure(vision, FIGURES)
    multiplier_figure(vision, FIGURES)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    table = comparison_table(state, vision)
    print(table.sort_values(["learner", "distribution"]).round(2).to_string(index=False))
    print()
    print(end_states(vision).sort_values(["learner", "distribution"]).round(2).to_string(index=False))
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()

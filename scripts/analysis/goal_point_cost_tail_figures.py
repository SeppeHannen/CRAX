"""The figures of docs/acl/experiments/2026-10-19_goal_point_cost_tail_500M.md.

    .venv/bin/python -m scripts.analysis.goal_point_cost_tail_figures

Pulls the five arms' evaluation histories from W&B (group ``goal_point_cost_tail_500M``) and
writes ``level3.png`` (return, mean cost, median cost and the share of episodes over 4× the
budget on level 3, over training, one line per arm) into the experiment's figures directory;
prints the end-state numbers the write-up quotes. The replay figures of the same directory are
``return_cost_mass_level{1,3}.html`` (``return_cost_mass_page.py``) and the table printed by
``goal_point_cost_tail.py``.
"""
from __future__ import annotations

import pathlib
from typing import Dict, Tuple

import matplotlib
import numpy as np
import pandas as pd
import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from training.config import WANDB_ENTITY, WANDB_PROJECT  # noqa: E402
from training.dashboard.view import ARM_COLOURS  # noqa: E402

GROUP = "goal_point_cost_tail_500M"
BUDGET = 25.0
FIGURES = pathlib.Path(__file__).resolve().parents[2] / "docs/acl/experiments/figures/2026-10-19_goal_point_cost_tail_500M"
LAST_EVALUATIONS = 5

# Arm label as the dashboard groups it -> the words the figure uses
ARM_WORDS: Dict[str, str] = {
    "uniform additive told": "control: uniform, centre lidar",
    "uniform additive told hazard_lidar=surface_by_kind": "uniform, surface lidar by kind",
    "uniform additive told memory32": "uniform, memory τ=32",
    "uniform additive told memory32 hazard_lidar=surface_by_kind": "uniform, surface lidar + memory",
    "plr additive told hazard_lidar=surface_by_kind": "PLR, surface lidar by kind",
}
PANELS: Tuple[Tuple[str, str, str], ...] = (
    ("episode_reward", "return (goals per episode)", "Return on level 3"),
    ("episode_cost", "cost per episode", "Mean cost on level 3"),
    ("episode_cost_median", "cost per episode", "Median cost on level 3"),
    ("episode_cost_over_4x_budget", "share of 128 episodes", "Episodes over 4× the budget on level 3"),
)


def histories() -> Dict[str, pd.DataFrame]:
    api = wandb.Api(timeout=120)
    frames: Dict[str, pd.DataFrame] = {}
    keys = ["environment_steps"] + [f"evaluation/{target}/{key}" for target in ("deployment", "uniform") for key, _, _ in PANELS]
    for run in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"group": GROUP, "jobType": "ppo_lag"}):
        frame = run.history(keys=keys, pandas=True, samples=2000)
        frames[run.config["arm"]] = frame.dropna(subset=["evaluation/deployment/episode_reward"]).sort_values("environment_steps")
    missing = set(ARM_WORDS) - set(frames)
    if missing:
        raise KeyError(f"group {GROUP} lacks arms {sorted(missing)}")
    return frames


def draw_level3(frames: Dict[str, pd.DataFrame], *, grid: Tuple[int, int] = (1, 4), name: str = "level3.png") -> None:
    """The four level-3 panels over training, laid out ``grid`` (1×4 for the document, 2×2 for a slide)."""
    rows, columns = grid
    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 4.2 * rows))
    for axis, (key, unit, title) in zip(axes.flat, PANELS):
        for arm, words in ARM_WORDS.items():
            frame = frames[arm]
            axis.plot(frame["environment_steps"] / 1e6, frame[f"evaluation/deployment/{key}"], marker="o", markersize=3, linewidth=1.4,
                      color=ARM_COLOURS[arm], label=words)
        if key in ("episode_cost", "episode_cost_median"):
            axis.axhline(BUDGET, color="#c7c7c7", linestyle="--", linewidth=1.2, label="budget")
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("environment steps (M)")
        axis.set_ylabel(unit)
        axis.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8, loc="upper left")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=140, bbox_inches="tight")
    print(f"wrote {FIGURES / name}")


def print_end_state(frames: Dict[str, pd.DataFrame]) -> None:
    print(f"\nmeans of the last {LAST_EVALUATIONS} evaluations")
    print(f"{'arm':<48} {'L3 ret':>7} {'L3 cost':>8} {'L3 med':>7} {'L3 >4x':>7} {'Ω ret':>6} {'Ω cost':>7}")
    for arm, words in ARM_WORDS.items():
        last = frames[arm].tail(LAST_EVALUATIONS)
        value = lambda target, key: float(last[f"evaluation/{target}/{key}"].mean())
        print(f"{words:<48} {value('deployment', 'episode_reward'):7.1f} {value('deployment', 'episode_cost'):8.1f} "
              f"{value('deployment', 'episode_cost_median'):7.1f} {value('deployment', 'episode_cost_over_4x_budget'):7.3f} "
              f"{value('uniform', 'episode_reward'):6.1f} {value('uniform', 'episode_cost'):7.1f}")


def main() -> None:
    frames = histories()
    draw_level3(frames)
    draw_level3(frames, grid=(2, 2), name="level3_2x2.png")
    print_end_state(frames)


if __name__ == "__main__":
    main()

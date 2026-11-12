"""What the runs of a generalization group did while they trained, from their W&B histories.

    .venv/bin/python -m scripts.analysis.generalization_training_figures --group goal_point_hazard_total_500M

``generalization_figures.py`` draws the result: the final policy in every cell of the
protocol. This script draws how each run got there, which says how much one final checkpoint
can be trusted. It reads every finished run of the group from W&B (a run that crashed or is
still training is named and left out) and writes into
``docs/acl/experiments/figures/<group>/``:

* ``evaluation_cost.png`` and ``evaluation_return.png``: the evaluations a run logs while it
  trains (``evaluation/<region>/*``, one mean over a whole region). One column per learner,
  one row per region of the protocol; in every panel the run with the regions held out and
  its reference run. The last evaluations, whose mean is the end state, are shaded.
* ``multiplier.png``: the multiplier and the training cost it reacts to. One column per
  learner; the multiplier the batch was weighted by on average, the multiplier of the
  round's hardest context (a Lagrange function only), and the cost per training episode.

It prints the end state per run, on every region and on the deployment task: the mean over the
last evaluations of the mean cost, the median cost, the standard deviation of an episode's
cost, the share of episodes over the budget and the return; the lowest, highest and final of
those mean costs, which say how far one evaluation lies from the end state; and the share of
episodes that left the arena, which should be zero (``docs/acl/known_defects.md``).
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
from typing import List, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.generalization_figures import (  # noqa: E402
    BAND, FIGURES, HELD_OUT_COLOUR, INK, MUTED, REFERENCE_COLOUR, legend_below, quiet_axes, save_figure,
)
from scripts.analysis.pull_group_history import sampled_rows  # noqa: E402
from training import contexts  # noqa: E402
from training.config import WANDB_ENTITY, WANDB_PROJECT  # noqa: E402
from training.dashboard.budget_run import JOB_TYPE as BUDGET_JOB_TYPE  # noqa: E402

END_STATE_EVALUATIONS = 5  # the end state is the mean of the last five evaluations (the last fifth of training)
HISTORY_ROWS = 2_000  # how many logged rows are asked of W&B per run; a run that has as many is refused, so every round is read
SMOOTHING_ROUNDS = 10  # the multiplier and the training cost are drawn as a moving mean over this many rounds
MULTIPLIER_LINEAR_BELOW = 0.1  # the multiplier axis is linear below this and logarithmic above: zero has a place on it


@dataclasses.dataclass(frozen=True)
class TrainedRun:
    """One finished run of the group: what it was, and every row it logged."""

    constraint: str
    policy_context: str
    reference: bool
    seed: int
    env_name: str
    evaluation_protocol: str
    budget: float
    history: pd.DataFrame

    @property
    def learner(self) -> str:
        return f"{self.constraint} {self.policy_context}"

    def series(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        """``(environment steps in millions, values)`` of one logged key, over the rows that hold it."""
        rows = self.history[["environment_steps", key]].dropna().sort_values("environment_steps")
        return rows["environment_steps"].to_numpy() / 1e6, rows[key].to_numpy(float)

    def has(self, key: str) -> bool:
        return key in self.history.columns


def finished_runs(group: str) -> List[TrainedRun]:
    """Every finished training run of ``group``, with its history; the others are named."""
    runs = []
    for run in wandb.Api(timeout=300).runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"group": group}):
        if run.job_type == BUDGET_JOB_TYPE:
            continue
        if run.state != "finished":
            print(f"{run.name}: {run.state}, left out")
            continue
        history = sampled_rows(run, HISTORY_ROWS)
        if len(history) >= HISTORY_ROWS:  # W&B thins a history longer than the rows asked for
            raise ValueError(f"{run.name} logged at least {HISTORY_ROWS} rows, the number read per run; "
                             f"raise HISTORY_ROWS, the figures draw every round")
        runs.append(TrainedRun(
            constraint=run.config["constraint"], policy_context=run.config["policy_context"], reference=run.config["reference"],
            seed=run.config["seed"], env_name=run.config["env_name"], evaluation_protocol=run.config["evaluation_protocol"],
            budget=float(run.config["safety_bound"]), history=history,
        ))
    if not runs:
        raise ValueError(f"no finished run in group {group!r}")
    for fact in ("env_name", "evaluation_protocol", "budget"):
        if len({getattr(run, fact) for run in runs}) != 1:
            raise ValueError(f"the runs of {group!r} differ in {fact}; one figure cannot hold them")
    return runs


def regions_of(runs: Sequence[TrainedRun]) -> List[str]:
    """The regions of the runs' protocol, the training region first."""
    protocol = contexts.suite_contexts(runs[0].env_name).protocol(runs[0].evaluation_protocol)
    return [contexts.TRAINING_REGION, *(region.name for region in protocol.held_out)]


def learners_of(runs: Sequence[TrainedRun]) -> List[str]:
    return sorted({run.learner for run in runs})  # the order of the columns in generalization_figures.py


# The two runs a panel holds, as (reference, colour, line style, legend words); as in generalization_figures.py.
ROLES: Tuple[Tuple[bool, str, str, str], ...] = (
    (False, HELD_OUT_COLOUR, "-", "bands held out of training"),
    (True, REFERENCE_COLOUR, "--", "reference: trained on everything"),
)


def draw_evaluations(runs: Sequence[TrainedRun], measure: str, unit: str, against_budget: bool, path: pathlib.Path) -> None:
    """``evaluation/<region>/<measure>`` through training: a column per learner, a row per region.
    ``against_budget`` draws the budget as a line on a logarithmic axis, for a cost."""
    learners, regions = learners_of(runs), regions_of(runs)
    figure, axes = plt.subplots(len(regions), len(learners), figsize=(3.4 * len(learners), 2.3 * len(regions)),
                                sharex=True, sharey=True, squeeze=False)
    for row, region in enumerate(regions):
        for column, learner in enumerate(learners):
            axis = axes[row][column]
            for reference, colour, style, words in ROLES:
                for run in runs:
                    if run.learner != learner or run.reference != reference:
                        continue
                    steps, values = run.series(f"evaluation/{region}/{measure}")
                    axis.axvspan(steps[-END_STATE_EVALUATIONS], steps[-1], color=BAND, zorder=0)
                    axis.plot(steps, values, color=colour, linestyle=style, linewidth=1.6, marker="o", markersize=2.5, label=words)
            if against_budget:
                axis.axhline(runs[0].budget, color=INK, linestyle=":", linewidth=1.0)
                axis.set_yscale("log")
                if column == len(learners) - 1:
                    axis.annotate(f"budget {runs[0].budget:g}", (1.0, runs[0].budget), xycoords=("axes fraction", "data"),
                                  xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
            if row == 0:
                axis.set_title(learner, fontsize=11, color=INK)
            if row == len(regions) - 1:
                axis.set_xlabel("environment steps (millions)", color=INK)
            if column == 0:
                axis.set_ylabel(f"{region}\n{unit}", color=INK)
            quiet_axes(axis)
    legend_below(figure, axes[0])
    figure.text(0.5, -0.06, f"shaded: the last {END_STATE_EVALUATIONS} evaluations, whose mean is the end state", fontsize=8, color=MUTED, ha="center")
    save_figure(figure, path)


def draw_multiplier(runs: Sequence[TrainedRun], path: pathlib.Path) -> None:
    """The multiplier (top) and the cost per training episode it reacts to (bottom), a column per learner."""
    learners = learners_of(runs)
    figure, axes = plt.subplots(2, len(learners), figsize=(3.4 * len(learners), 5.2), sharex=True, sharey="row", squeeze=False)
    for column, learner in enumerate(learners):
        multiplier_axis, cost_axis = axes[0][column], axes[1][column]
        for reference, colour, style, words in ROLES:
            for run in runs:
                if run.learner != learner or run.reference != reference:
                    continue
                multiplier_axis.plot(*_smoothed(run, "training/lambda_lagr"), color=colour, linestyle=style, linewidth=1.6, label=words)
                if run.has("training/lambda_lagr/max"):
                    multiplier_axis.plot(*_smoothed(run, "training/lambda_lagr/max"), color=colour, linestyle=style, linewidth=0.8)
                cost_axis.plot(*_smoothed(run, "episodic/cost"), color=colour, linestyle=style, linewidth=1.6)
        multiplier_axis.set_yscale("symlog", linthresh=MULTIPLIER_LINEAR_BELOW)
        multiplier_axis.set_title(learner, fontsize=11, color=INK)
        cost_axis.axhline(runs[0].budget, color=INK, linestyle=":", linewidth=1.0)
        cost_axis.set_ylim(0.0, 3.0 * runs[0].budget)
        cost_axis.set_xlabel("environment steps (millions)", color=INK)
        quiet_axes(multiplier_axis)
        quiet_axes(cost_axis)
    axes[0][0].set_ylim(0.0, 1.5 * max(line.get_ydata().max() for axis in axes[0] for line in axis.get_lines()))
    axes[0][0].set_ylabel("multiplier λ", color=INK)
    axes[1][0].set_ylabel("cost per training episode", color=INK)
    axes[1][-1].annotate(f"budget {runs[0].budget:g}", (1.0, runs[0].budget), xycoords=("axes fraction", "data"),
                         xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    legend_below(figure, axes[0])
    figure.text(0.5, -0.07, f"thick: the mean over the batch; thin: the round's hardest context (a Lagrange function only). "
                             f"Moving mean over {SMOOTHING_ROUNDS} rounds.", fontsize=8, color=MUTED, ha="center")
    save_figure(figure, path)


def _smoothed(run: TrainedRun, key: str) -> Tuple[np.ndarray, np.ndarray]:
    steps, values = run.series(key)
    return steps[SMOOTHING_ROUNDS - 1:], np.convolve(values, np.ones(SMOOTHING_ROUNDS) / SMOOTHING_ROUNDS, mode="valid")


def end_states(runs: Sequence[TrainedRun]) -> pd.DataFrame:
    """Per run and evaluation (the protocol's regions, then the deployment task): the mean over
    the last evaluations of each logged measure, the lowest, highest and final of their mean
    costs, and the largest share of episodes that left the arena, in evaluation and in training."""
    rows = []
    for run in sorted(runs, key=lambda run: (run.learner, run.reference, run.seed)):
        for evaluation in (*regions_of(runs), contexts.DEPLOYMENT_EVALUATION):
            last = lambda measure: run.series(f"evaluation/{evaluation}/{measure}")[1][-END_STATE_EVALUATIONS:]
            costs = last("episode_cost")
            rows.append({
                "learner": run.learner, "reference": run.reference, "seed": run.seed, "evaluation": evaluation,
                "cost": costs.mean(), "lowest": costs.min(), "highest": costs.max(), "final": costs[-1],
                "median cost": last("episode_cost_median").mean(), "cost deviation": last("episode_cost_std").mean(),
                "over budget": last("episode_cost_over_budget").mean(), "return": last("episode_reward").mean(),
                "left arena": max(run.series(f"evaluation/{evaluation}/episode_outside_arena")[1].max(), run.series("episodic/outside_arena")[1].max()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", required=True, help="W&B group of an experiment of the generalization evaluation")
    arguments = parser.parse_args()
    runs = finished_runs(arguments.group)
    directory = FIGURES / arguments.group
    draw_evaluations(runs, "episode_cost", "cost per episode (mean)", against_budget=True, path=directory / "evaluation_cost.png")
    draw_evaluations(runs, "episode_reward", "return (goals per episode)", against_budget=False, path=directory / "evaluation_return.png")
    draw_multiplier(runs, directory / "multiplier.png")
    pd.set_option("display.width", 200)
    print(f"\nend state: the mean of the last {END_STATE_EVALUATIONS} evaluations; budget {runs[0].budget:g}")
    print(end_states(runs).round(2).to_string(index=False))


if __name__ == "__main__":
    main()

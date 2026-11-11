"""The figures of the generalization evaluation, from the score tables of one or more groups.

    .venv/bin/python -m training.score --group goal_point_hazard_total_500M
    .venv/bin/python -m scripts.analysis.generalization_figures --groups goal_point_hazard_total_500M

Reads ``runs/scores/<group>/*.csv`` (``training/score.py``: one row per run and cell of the
evaluation protocol) and writes into ``docs/acl/experiments/figures/<first group>/``:

* a protocol with one factor (the hazard total, or the drag): ``generalization.png``. The
  factor on the horizontal axis, the held-out bands shaded. One column per learner; cost on
  the top row with the budget as a line, return on the bottom row. In every panel the run
  trained with the bands held out and its reference run, trained on everything: the mean over
  seeds as a line, the range over seeds as a band.
* a protocol with two factors (hazard total by drag): ``generalization_grid.png``. One column
  per learner; the top row is the cost in every cell, coloured by its distance from the budget,
  the bottom row the return minus the reference run's. The held-out corner is outlined.

It prints, per learner and region, the numbers a write-up quotes: mean cost, the share of
cells over the budget, the return as a share of the reference's, and the share of episodes
that left the arena (``docs/acl/known_defects.md``), which should be zero.

Several groups are pooled, so curriculum runs can be drawn next to the uniform baseline they
are compared with; the groups must share the suite and the evaluation protocol.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import List, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from training import contexts  # noqa: E402
from training.contexts.registry import DRAG_DIMENSION, HAZARD_TOTAL_FACTOR  # noqa: E402
from training.score import SCORES_DIRECTORY  # noqa: E402

FIGURES = pathlib.Path(__file__).resolve().parents[2] / "docs/acl/experiments/figures"
LEARNER_COLUMNS = ["training_distribution", "constraint", "policy_context"]
FACTOR_WORDS = {HAZARD_TOTAL_FACTOR: "total number of hazards", DRAG_DIMENSION: "drag (1 = the benchmark's)"}

# Colours: the first two categorical slots of the palette the dataviz guideline validates
# (colour-blind separation ΔE 24.7, contrast ≥ 3:1 on white); the diverging pair is its
# blue ↔ red around a neutral grey. Text and axes are ink, never a series colour.
HELD_OUT_COLOUR = "#2a78d6"
REFERENCE_COLOUR = "#eb6834"
INK, MUTED, HAIRLINE, BAND = "#0b0b0b", "#898781", "#e1e0d9", "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", ["#2a78d6", "#f0efec", "#d03b3b"])


def read_scores(groups: Sequence[str]) -> pd.DataFrame:
    """Every run's score table of ``groups`` as one frame; the groups share suite and protocol."""
    paths = [path for group in groups for path in sorted((SCORES_DIRECTORY / group).glob("*.csv"))]
    if not paths:
        raise FileNotFoundError(f"no score tables under {[str(SCORES_DIRECTORY / group) for group in groups]}; run training.score first")
    scores = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    for column in ("env_name", "evaluation_protocol", "budget"):
        if scores[column].nunique() != 1:
            raise ValueError(f"the groups differ in {column}: {sorted(scores[column].unique())}; one figure cannot hold them")
    return scores


def factors_of(scores: pd.DataFrame) -> Tuple[str, ...]:
    """The factor columns of the tables: the coordinates of the protocol's cells, from the registry."""
    protocol = contexts.suite_contexts(scores["env_name"].iloc[0]).protocol(scores["evaluation_protocol"].iloc[0])
    return tuple(name for name, _ in protocol.cells[0].coordinates)


def learners_of(scores: pd.DataFrame) -> List[Tuple[str, str, str]]:
    return sorted(scores[LEARNER_COLUMNS].drop_duplicates().itertuples(index=False, name=None))


def learner_words(learner: Tuple[str, str, str], scores: pd.DataFrame) -> str:
    """``additive told``, with the training distribution in front when the figure holds more than one."""
    distribution, constraint, policy_context = learner
    words = f"{constraint} {policy_context}"
    return words if scores["training_distribution"].nunique() == 1 else f"{distribution}: {words}"


def of_learner(scores: pd.DataFrame, learner: Tuple[str, str, str], reference: bool) -> pd.DataFrame:
    rows = (scores[LEARNER_COLUMNS] == learner).all(axis=1) & (scores["reference"] == reference)
    return scores[rows]


def held_out_spans(scores: pd.DataFrame, factor: str) -> List[Tuple[str, float, float]]:
    """Each held-out region as ``(name, first, last)`` cell coordinate along ``factor``."""
    held_out = scores[scores["region"] != contexts.TRAINING_REGION]
    return [(name, float(rows[factor].min()), float(rows[factor].max())) for name, rows in held_out.groupby("region")]


def draw_one_factor(scores: pd.DataFrame, factor: str, path: pathlib.Path) -> None:
    learners = learners_of(scores)
    budget = float(scores["budget"].iloc[0])
    coordinates = np.sort(scores[factor].unique())
    half_cell = float(np.min(np.diff(coordinates))) / 2.0
    figure, axes = plt.subplots(2, len(learners), figsize=(3.6 * len(learners), 6.0), sharex=True, sharey="row", squeeze=False)
    rows = (("cost_mean", "cost per episode (mean)"), ("return_mean", "return (goals per episode)"))
    for column, learner in enumerate(learners):
        for row, (measure, unit) in enumerate(rows):
            axis = axes[row][column]
            for name, first, last in held_out_spans(scores, factor):
                axis.axvspan(first - half_cell, last + half_cell, color=BAND, zorder=0)
                if row == 0:  # above the axes, clear of the curves
                    axis.annotate(name, ((first + last) / 2.0, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 2),
                                  textcoords="offset points", ha="center", va="bottom", fontsize=7, color=MUTED)
            series = (
                (False, HELD_OUT_COLOUR, "-", "o", "bands held out of training"),
                (True, REFERENCE_COLOUR, "--", "s", "reference: trained on everything"),
            )
            for reference, colour, style, marker, label in series:
                runs = of_learner(scores, learner, reference)
                if runs.empty:
                    continue
                per_seed = runs.pivot_table(index=factor, columns="run", values=measure)
                axis.fill_between(per_seed.index, per_seed.min(axis=1), per_seed.max(axis=1), color=colour, alpha=0.18, linewidth=0)
                axis.plot(per_seed.index, per_seed.mean(axis=1), color=colour, linestyle=style, linewidth=2.0, marker=marker,
                          markersize=3.5, label=f"{label} ({per_seed.shape[1]} seeds)")
            if measure == "cost_mean":
                axis.axhline(budget, color=INK, linestyle=":", linewidth=1.0)
                if column == len(learners) - 1:  # in the margin, clear of the curves
                    axis.annotate(f"budget {budget:g}", (1.0, budget), xycoords=("axes fraction", "data"), xytext=(4, 0),
                                  textcoords="offset points", va="center", fontsize=8, color=INK)
            if row == 0:
                axis.set_title(learner_words(learner, scores), fontsize=11, color=INK, pad=16)
            if row == len(rows) - 1:
                axis.set_xlabel(FACTOR_WORDS[factor], color=INK)
            if column == 0:
                axis.set_ylabel(unit, color=INK)
            _quiet(axis)
    figure.legend(*axes[0][0].get_legend_handles_labels(), fontsize=9, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    _save(figure, path)


def draw_two_factors(scores: pd.DataFrame, factors: Tuple[str, str], path: pathlib.Path) -> None:
    across, down = factors
    learners = learners_of(scores)
    budget = float(scores["budget"].iloc[0])
    figure, axes = plt.subplots(2, len(learners), figsize=(3.6 * len(learners), 6.4), sharex=True, sharey=True, squeeze=False)
    grids = []
    for learner in learners:
        mean = lambda reference, measure: of_learner(scores, learner, reference).pivot_table(index=down, columns=across, values=measure)
        cost, gap = mean(False, "cost_mean"), mean(False, "return_mean") - mean(True, "return_mean")
        if gap.isna().all().all():
            raise ValueError(f"{learner}: no reference run in the tables; the return panel compares with it")
        grids.append((cost, gap))
    highest_cost = max(float(cost.to_numpy().max()) for cost, _ in grids)
    gap_reach = max(float(np.abs(gap.to_numpy()).max()) for _, gap in grids)
    panels = (
        (0, TwoSlopeNorm(vcenter=budget, vmin=0.0, vmax=max(highest_cost, budget + 1.0)), f"cost per episode (grey = the budget, {budget:g})"),
        (1, TwoSlopeNorm(vcenter=0.0, vmin=-gap_reach, vmax=gap_reach), "return minus the reference run's (goals per episode)"),
    )
    held_out = scores[scores["region"] != contexts.TRAINING_REGION]
    for row, norm, words in panels:
        for column, learner in enumerate(learners):
            axis, grid = axes[row][column], grids[column][row]
            xs, ys = grid.columns.to_numpy(float), grid.index.to_numpy(float)
            dx, dy = float(np.min(np.diff(xs))), float(np.min(np.diff(ys)))
            image = axis.pcolormesh(xs, ys, grid.to_numpy(), cmap=DIVERGING, norm=norm, shading="nearest", edgecolors="white", linewidth=0.5)
            for _, rows in held_out.groupby("region"):
                corner = (rows[across].min() - dx / 2, rows[down].min() - dy / 2)
                axis.add_patch(Rectangle(corner, rows[across].max() - rows[across].min() + dx, rows[down].max() - rows[down].min() + dy,
                                         fill=False, edgecolor=INK, linewidth=1.6))
            if row == 0:
                axis.set_title(learner_words(learner, scores), fontsize=11, color=INK)
            if row == 1:
                axis.set_xlabel(FACTOR_WORDS[across], color=INK)
            if column == 0:
                axis.set_ylabel(FACTOR_WORDS[down], color=INK)
            _quiet(axis, grid=False)
        bar = figure.colorbar(image, ax=list(axes[row]), fraction=0.025, pad=0.02)
        bar.set_label(words, fontsize=8, color=INK)
        bar.ax.tick_params(colors=MUTED, labelsize=8)
        bar.outline.set_visible(False)
    figure.text(0.01, 0.005, "outlined: held out of training; every cell outside the outline is trained on", fontsize=8, color=MUTED)
    _save(figure, path, tight=False)


def _quiet(axis: plt.Axes, grid: bool = True) -> None:
    """Recessive axes: hairline grid, no box, muted ticks."""
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelsize=8)
    if grid:
        axis.grid(color=HAIRLINE, linewidth=0.6)
        axis.set_axisbelow(True)


def _save(figure: plt.Figure, path: pathlib.Path, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"wrote {path}")


def print_summary(scores: pd.DataFrame) -> None:
    """Per learner and region: the mean cost over the region's cells, the share of its cells whose
    mean cost is over the budget, the return as a share of the reference run's, and the share of
    episodes that left the arena. Seeds are averaged per cell first."""
    budget = float(scores["budget"].iloc[0])
    factors = list(factors_of(scores))
    print(f"\nbudget {budget:g}; seeds averaged per cell")
    print(f"{'learner':<28} {'region':<14} {'cells':>5} {'cost':>6} {'cells over':>10} {'return':>7} {'of reference':>12} {'left arena':>10}")
    for learner in learners_of(scores):
        per_cell = lambda reference: of_learner(scores, learner, reference).groupby(["region", *factors])[
            ["cost_mean", "return_mean", "share_outside_arena"]].mean()
        held_out, reference = per_cell(False), per_cell(True)
        for region, cells in held_out.groupby(level="region"):
            of_reference = (
                f"{cells['return_mean'].mean() / reference.loc[cells.index, 'return_mean'].mean():12.0%}" if not reference.empty else f"{'—':>12}"
            )
            print(f"{learner_words(learner, scores):<28} {region:<14} {len(cells):5d} {cells['cost_mean'].mean():6.1f} "
                  f"{(cells['cost_mean'] > budget).mean():10.0%} {cells['return_mean'].mean():7.1f} {of_reference} "
                  f"{cells['share_outside_arena'].mean():10.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="+", required=True, help="W&B groups scored with training.score; figures go under the first")
    arguments = parser.parse_args()
    scores = read_scores(arguments.groups)
    factors = factors_of(scores)
    directory = FIGURES / arguments.groups[0]
    if len(factors) == 1:
        draw_one_factor(scores, factors[0], directory / "generalization.png")
    elif len(factors) == 2:
        draw_two_factors(scores, (factors[1], factors[0]), directory / "generalization_grid.png")
    else:
        raise ValueError(f"a figure is drawn for one or two factors; this protocol's cells have {factors}")
    print_summary(scores)


if __name__ == "__main__":
    main()

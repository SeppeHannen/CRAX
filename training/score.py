"""Score a group's trained policies on the cells of their evaluation protocol.

    .venv/bin/python -m training.score --group goal_point_hazard_total_500M

For every finished run of the W&B group, the final checkpoint is fetched and the
policy is rolled out for ``--episodes`` whole episodes in every **cell** of the
run's evaluation protocol (:class:`~training.contexts.protocol.Cell`: one hazard
total, one band of drag, ...). The result is one table per run,
``runs/scores/<group>/<run>.csv``, with one row per cell: which region the cell
lies in (``training`` or a held-out region), the mean return, and the cost as
its mean, its median and the share of episodes over the budget. A run whose
table exists is not scored again. ``scripts/analysis/generalization_figures.py``
draws the figures from these tables.

The evaluations a run logs while it trains (``evaluation/<region>/*``) average
over a whole region. An average can hide one hazard total over the budget and
another under it; the cells are what the generalization evaluation reports
(``docs/acl/design/generalization_evaluation.md``, *Measurement*).

**One compiled program per run.** A context is a value, so every cell is rolled
out at once: the contexts are drawn here, uniformly within each cell, and
assigned one per slot (:class:`~training.contexts.AssignedContexts`). Every run
of a group is scored on the same contexts with the same keys. The rollout
therefore simulates ``cells × --episodes`` robots at once, which has to fit on
the GPU: 2,688 for the 21 hazard totals at the default 128 episodes; for the
combination's 210 cells pass ``--episodes 32`` (6,720, fewer than a training run's 8,192).

**Same policy, same code.** Checkpoint, networks and policy are a replay's
(``training/replay.py``): the run's own sampling or deterministic policy, and a
refusal when this checkout changes the code the policy is made of.

**An episode outside the arena is counted.** The fence can fail
(``docs/acl/known_defects.md``); outside it there is neither goal nor hazard, so
such an episode reads as cheap. ``share_outside_arena`` is the share of a cell's
episodes that spent at least one step there. The suite must report that metric
(``outside_arena``; the goal task does).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import functools
import pathlib
from typing import Dict, List, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import wandb

from training import acting, contexts, replay
from training.agents.ppo import networks as ppo_networks
from training.config import WANDB_ENTITY, WANDB_PROJECT
from training.contexts.wrapper import wrap_for_context_evaluation
from training.dashboard.budget_run import budget_run_id

SCORES_DIRECTORY = pathlib.Path("runs/scores")
CHECKPOINT_CACHE = pathlib.Path("runs/checkpoints")
# What a run's facts are called in a score table, and the W&B config key each is read from.
RUN_COLUMNS: Dict[str, str] = {
    "env_name": "env_name",
    "arm": "arm",
    "constraint": "constraint",
    "policy_context": "policy_context",
    "training_distribution": "training_distribution",
    "reference": "reference",
    "seed": "seed",
    "evaluation_protocol": "evaluation_protocol",
    "budget": "safety_bound",
}


@dataclasses.dataclass(frozen=True)
class CellScore:
    """What ``episodes`` episodes of one policy came to in one cell."""

    cell: contexts.Cell
    region: str
    episodes: int
    return_mean: float
    cost_mean: float
    cost_median: float
    share_over_budget: float
    share_outside_arena: float

    def row(self) -> Dict[str, Union[str, int, float]]:
        """The score as table columns: the region, the cell's coordinates under their factor names, the numbers."""
        numbers = {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self) if field.name not in ("cell", "region")
        }
        return {"region": self.region, **dict(self.cell.coordinates), **numbers}


def score_cells(
    run_config: argparse.Namespace, learner: replay.Learner, deterministic: bool, episodes: int, seed: int
) -> List[CellScore]:
    """Roll ``learner``'s policy out for ``episodes`` whole episodes in every cell of the run's
    evaluation protocol, all at once, and score each cell against the run's budget."""
    protocol = replay.protocol_of(run_config)
    context_key, rollout_key = jax.random.split(jax.random.PRNGKey(seed))
    cell_keys = jax.random.split(context_key, len(protocol.cells))
    assigned = jnp.concatenate([cell.space.sample_uniform(key, episodes) for cell, key in zip(protocol.cells, cell_keys)])
    env = wrap_for_context_evaluation(
        replay.environment_of(run_config), contexts.AssignedContexts(protocol.contexts, assigned), episode_length=run_config.episode_length
    )
    evaluator = acting.Evaluator(
        env, functools.partial(ppo_networks.make_inference_fn(learner.network), deterministic=deterministic),
        num_eval_envs=assigned.shape[0], episode_length=run_config.episode_length, action_repeat=1, key=rollout_key,
    )
    metrics = evaluator.run_evaluation(learner.policy_tuple, training_metrics={}, aggregate_episodes=False)

    def per_cell(name: str) -> np.ndarray:
        """The evaluator's per-slot metric as ``[cells, episodes]``: slot order is the assigned order."""
        return np.asarray(metrics[f"eval/episode_{name}"]).reshape(len(protocol.cells), episodes)

    returns, costs, steps_outside = per_cell("reward"), per_cell("cost"), per_cell("outside_arena")
    return [
        CellScore(
            cell=cell, region=protocol.region_of(cell), episodes=episodes,
            return_mean=float(returns[index].mean()), cost_mean=float(costs[index].mean()), cost_median=float(np.median(costs[index])),
            share_over_budget=float(np.mean(costs[index] > run_config.safety_bound)),
            share_outside_arena=float(np.mean(steps_outside[index] > 0)),
        )
        for index, cell in enumerate(protocol.cells)
    ]


def write_scores(path: pathlib.Path, run_name: str, run_config: argparse.Namespace, environment_steps: int, scores: Sequence[CellScore]) -> None:
    """One row per cell: the run and its facts, the checkpoint's step, then :meth:`CellScore.row`."""
    facts = {column: getattr(run_config, key) for column, key in RUN_COLUMNS.items()}
    rows = [{"run": run_name, **facts, "environment_steps": environment_steps, **score.row()} for score in scores]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", required=True, help="W&B group of context runs trained with --store_model true")
    parser.add_argument("--episodes", type=int, default=128,
                        help="episodes per cell; every cell of a run is rolled out at once, cells x episodes robots")
    parser.add_argument("--seed", type=int, default=0, help="of the contexts and the rollout; the same for every run")
    parser.add_argument("--entity", default=WANDB_ENTITY)
    parser.add_argument("--project", default=WANDB_PROJECT)
    arguments = parser.parse_args()

    repository = pathlib.Path(__file__).resolve().parents[1]
    runs = [
        run for run in wandb.Api().runs(f"{arguments.entity}/{arguments.project}", filters={"group": arguments.group})
        if run.id != budget_run_id(arguments.group)
    ]
    if not runs:
        raise ValueError(f"no runs in group {arguments.group!r} of {arguments.entity}/{arguments.project}")
    for run in sorted(runs, key=lambda run: run.name):
        path = SCORES_DIRECTORY / arguments.group / f"{run.name}.csv"
        if run.state != "finished":
            print(f"{run.name}: {run.state}, not scored")
            continue
        if path.exists():
            print(f"{run.name}: already scored, {path}")
            continue
        replay.require_the_runs_policy_code(run.commit, repository)
        run_config = argparse.Namespace(**run.config)
        checkpoint_directory = replay.fetch_latest_checkpoint(run, CHECKPOINT_CACHE)
        learner = replay.learner_from_checkpoint(checkpoint_directory, run_config)
        scores = score_cells(run_config, learner, bool(run_config.deterministic_eval), arguments.episodes, arguments.seed)
        write_scores(path, run.name, run_config, int(checkpoint_directory.name), scores)
        held_out = [score for score in scores if score.region != contexts.TRAINING_REGION]
        print(
            f"{run.name}: {len(scores)} cells x {arguments.episodes} episodes -> {path}; "
            f"mean cost {np.mean([score.cost_mean for score in scores]):.1f} over all cells"
            + (f", {np.mean([score.cost_mean for score in held_out]):.1f} over the held-out cells" if held_out else "")
            + f" (budget {run_config.safety_bound:g})"
        )


if __name__ == "__main__":
    main()

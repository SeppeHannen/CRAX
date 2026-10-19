"""The shape of an evaluation's episode-cost distribution beyond its mean.

A constraint on the mean cost is met by a policy whose failures are few and
large as readily as by one whose cost is spread evenly: on ``safe_goal_point``
the typical level-3 episode of the per-context learner costs half the budget
while one episode in two hundred costs ten times it, and that tail sets the
multiplier (``docs/acl/README.md``, plan task 1b). The evaluator logs each
evaluation's mean and standard deviation of cost; these three numbers say
where the mean comes from:

* ``episode_cost_median`` — the typical episode;
* ``episode_cost_over_budget`` — the fraction of episodes whose cost exceeds the
  budget: how often the constraint, read per episode, is violated;
* ``episode_cost_over_4x_budget`` — the fraction over four times the budget: the
  catastrophic tail.

The budget is the run's episodic ``safety_bound``; the agent that holds the
constraint builds a :class:`CostTail` and hands it to the trainer, which hands
it to every :class:`~training.acting.Evaluator`.
"""
from __future__ import annotations

import dataclasses
from typing import Dict

import numpy as np

CATASTROPHE_MULTIPLE = 4.0


@dataclasses.dataclass(frozen=True)
class CostTail:
    """Statistics of a population of episode costs, read against ``budget`` (per episode)."""

    budget: float

    def __post_init__(self) -> None:
        if self.budget < 0.0:
            raise ValueError(f"a cost tail is read against a nonnegative episode budget, got {self.budget}")

    def metrics(self, episode_costs: np.ndarray) -> Dict[str, float]:
        """The three statistics of ``episode_costs`` (``[episodes]``), keyed as the evaluator's
        other episode metrics are (``episode_cost_<statistic>``)."""
        costs = np.asarray(episode_costs, dtype=np.float64)
        if costs.ndim != 1 or costs.shape[0] == 0:
            raise ValueError(f"expected one cost per episode, got shape {costs.shape}")
        return {
            "episode_cost_median": float(np.median(costs)),
            "episode_cost_over_budget": float(np.mean(costs > self.budget)),
            "episode_cost_over_4x_budget": float(np.mean(costs > CATASTROPHE_MULTIPLE * self.budget)),
        }

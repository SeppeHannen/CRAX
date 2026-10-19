"""The cost tail of replayed goal-point episodes, and whether the policy avoids the trap or escapes it.

    .venv/bin/python -m scripts.analysis.goal_point_cost_tail <records root> [<records root> ...] [--budget 25]

Each root holds the episode records of one arm's replays (``python -m training.replay`` writes
``replays/<run>/seed<k>/episodes``). For every arm and every evaluation target found below it:

* the **tail**: mean, median, p99 and worst episode cost, and the share of episodes over the
  budget and over four times it — the numbers the constraint's meaning rests on
  (``docs/acl/README.md``, plan task 1b);
* the **traps**: a trap is repeated solid-hazard contact in one spot — at least ``MIN_TRAP_TOUCHES``
  touches on the same pair (or single) of solid hazards, each within ``--trap_gap`` steps of the
  previous; fewer touches are a brush. Entering a trap and leaving it are different skills; the
  table separates them: how many episodes enter at least one trap, how many touches a trap costs,
  and how many traps are *escaped* (the episode goes on for at least ``--trap_gap`` steps after
  the last touch without touching that set again).

A touch is a step with cost at least ``SOLID_CONTACT_COST`` (``crax/envs/safe_goal.py``,
``collision_cost``). Which solid hazards were touched is read from the scene: the active solid
hazards whose footprint the agent's centre is within reach of at that step.
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
from collections import defaultdict
from typing import Dict, FrozenSet, List, Sequence, Tuple

import numpy as np

from crax import envs
from training.evaluation import EpisodeRecord, records_under

SOLID_CONTACT_COST = 3.0
# How far from a solid hazard's footprint the agent's centre may be at a touch to count that hazard
AGENT_REACH = 0.25
# Fewer touches on one spot are a brush, not a trap
MIN_TRAP_TOUCHES = 3


@dataclasses.dataclass(frozen=True)
class SolidHazards:
    """What the trap counter needs of the suite: which hazards are solid, and their footprints."""

    mocap_ids: np.ndarray  # [H] the mocap body of each hazard
    is_solid: np.ndarray  # [H]
    footprint_radius: np.ndarray  # [H]

    @classmethod
    def of(cls, env_name: str) -> "SolidHazards":
        env = envs.get_environment(env_name, level=3).unwrapped
        hazards = env._hazard_manager.hazards
        return cls(
            mocap_ids=np.asarray(env._hazard_mocap_ids),
            is_solid=np.array([hazard.collidable for hazard in hazards]),
            footprint_radius=np.array([hazard.get_keepout_radius() for hazard in hazards], dtype=np.float64),
        )

    def touched(self, record: EpisodeRecord, step: int) -> FrozenSet[int]:
        """The solid hazards within reach of the agent at ``step``."""
        agent = record.qpos[step, :2]
        positions = record.mocap_pos[step, self.mocap_ids, :2]
        distance = np.linalg.norm(positions - agent, axis=1) - self.footprint_radius
        return frozenset(int(index) for index in np.flatnonzero(self.is_solid & (distance < AGENT_REACH)))


@dataclasses.dataclass(frozen=True)
class Trap:
    hazards: FrozenSet[int]
    first_touch: int
    last_touch: int
    touches: int
    escaped: bool  # the episode went on past the trap without touching these hazards again


def traps_in(record: EpisodeRecord, solids: SolidHazards, trap_gap: int) -> List[Trap]:
    """Consecutive touches on the same solid set within ``trap_gap`` steps of each other form one
    trap, if there are at least ``MIN_TRAP_TOUCHES`` of them."""
    touches = [(int(step), solids.touched(record, int(step))) for step in np.flatnonzero(record.cost >= SOLID_CONTACT_COST)]
    traps: List[Trap] = []
    current: List[Tuple[int, FrozenSet[int]]] = []

    def close(group: Sequence[Tuple[int, FrozenSet[int]]]) -> None:
        if len(group) < MIN_TRAP_TOUCHES:
            return
        last = group[-1][0]
        later_touch_on_same = any(step > last and hazards & group[0][1] for step, hazards in touches)
        traps.append(Trap(group[0][1], group[0][0], last, len(group), escaped=not later_touch_on_same and record.length - last > trap_gap))

    for step, hazards in touches:
        if current and (step - current[-1][0] > trap_gap or not (hazards & current[0][1])):
            close(current)
            current = []
        current.append((step, hazards))
    if current:
        close(current)
    return traps


@dataclasses.dataclass(frozen=True)
class TailSummary:
    arm: str
    target: str
    episodes: int
    mean: float
    median: float
    p99: float
    worst: float
    over_budget: float
    over_4x_budget: float
    goals: float
    episodes_with_a_trap: float
    traps_per_episode: float
    touches_per_trap: float
    traps_escaped: float

    @staticmethod
    def header() -> str:
        return (f"{'arm':<64} {'target':<10} {'n':>5} {'goals':>6} {'mean':>6} {'median':>7} {'p99':>6} {'worst':>6} "
                f"{'>budget':>8} {'>4x':>6} {'trapped':>8} {'traps/ep':>9} {'touch/trap':>11} {'escaped':>8}")

    def row(self) -> str:
        return (f"{self.arm:<64} {self.target:<10} {self.episodes:5d} {self.goals:6.1f} {self.mean:6.1f} {self.median:7.1f} "
                f"{self.p99:6.0f} {self.worst:6.0f} {self.over_budget:8.1%} {self.over_4x_budget:6.1%} {self.episodes_with_a_trap:8.1%} "
                f"{self.traps_per_episode:9.2f} {self.touches_per_trap:11.1f} {self.traps_escaped:8.1%}")


def summarise(records: Sequence[EpisodeRecord], solids: SolidHazards, budget: float, trap_gap: int) -> TailSummary:
    costs = np.array([record.episode_cost for record in records])
    traps = [traps_in(record, solids, trap_gap) for record in records]
    all_traps = [trap for per_episode in traps for trap in per_episode]
    return TailSummary(
        arm=records[0].arm, target=records[0].evaluation_name, episodes=len(records),
        mean=float(costs.mean()), median=float(np.median(costs)), p99=float(np.percentile(costs, 99)), worst=float(costs.max()),
        over_budget=float(np.mean(costs > budget)), over_4x_budget=float(np.mean(costs > 4 * budget)),
        goals=float(np.mean([record.episode_return for record in records])),
        episodes_with_a_trap=float(np.mean([bool(per_episode) for per_episode in traps])),
        traps_per_episode=len(all_traps) / len(records),
        touches_per_trap=float(np.mean([trap.touches for trap in all_traps])) if all_traps else 0.0,
        traps_escaped=float(np.mean([trap.escaped for trap in all_traps])) if all_traps else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=pathlib.Path)
    parser.add_argument("--budget", type=float, default=25.0)
    parser.add_argument("--trap_gap", type=int, default=100, help="touches further apart than this are different traps")
    arguments = parser.parse_args()

    by_arm_and_target: Dict[Tuple[str, str], List[EpisodeRecord]] = defaultdict(list)
    for root in arguments.roots:
        for record in records_under(root):
            by_arm_and_target[(record.arm, record.evaluation_name)].append(record)
    if not by_arm_and_target:
        raise FileNotFoundError(f"no episode records under {arguments.roots}")
    env_names = {record.env_name for records in by_arm_and_target.values() for record in records}
    if len(env_names) != 1:
        raise ValueError(f"records from several suites: {sorted(env_names)}")
    solids = SolidHazards.of(env_names.pop())

    print(TailSummary.header())
    for key in sorted(by_arm_and_target):
        print(summarise(by_arm_and_target[key], solids, arguments.budget, arguments.trap_gap).row())


if __name__ == "__main__":
    main()

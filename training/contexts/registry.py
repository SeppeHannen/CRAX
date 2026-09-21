"""Per-suite context spaces and the mapping from CRAX difficulty levels to contexts.

This is the single place that knows *what* varies in each suite. Environments
know how to *read* a context (see :func:`training.contexts.wrapper`), and
distributions know how to *choose* one; neither needs to know the suite.

Adding a suite: write a :class:`SuiteContexts` and register it. The level
contexts must reproduce ``crax/envs/difficulty.py`` exactly, so that
``FixedContext(level_context(env, 3))`` is the same task as
``get_environment(env, level=3)``.

Only value-typed suites are registered so far (their difficulty is a number an
existing operation reads). Count-typed suites (Goal, Reach, Circle) need the
pad-and-mask change to the environment first; see
``docs/acl/design/per_slot_constraints.md``.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict, Mapping, Tuple

import jax.numpy as jnp

from training.contexts.space import Context, ContextSpace, Dimension

LEVELS = (1, 2, 3)


@dataclasses.dataclass(frozen=True)
class TaskDescription:
    """The suite's task in words, for a reader of the dashboard who has not seen the environment.

    Each field is one or two sentences; the dashboard quotes them and refers
    to "the reward" and "the cost" defined here in every panel description.
    """

    agent: str
    reward: str
    cost: str


@dataclasses.dataclass(frozen=True)
class SuiteContexts:
    """Everything the context machinery needs to know about one environment."""

    env_name: str
    space: ContextSpace
    level_contexts: Mapping[int, Context]
    # Which keys of the decoded context the environment reads, and under what
    # name in state.info["context_values"]; identity by default.
    parameter_names: Tuple[str, ...]
    task: TaskDescription

    def level(self, level: int) -> Context:
        if level not in self.level_contexts:
            raise KeyError(f"{self.env_name}: no level {level}; have {sorted(self.level_contexts)}")
        return self.level_contexts[level]

    def levels(self) -> jnp.ndarray:
        """All difficulty levels as a ``[3, D]`` array (for StagedContexts)."""
        return jnp.stack([self.level(l) for l in LEVELS])


_REGISTRY: Dict[str, Callable[[], SuiteContexts]] = {}


def register(env_name: str):
    def decorator(factory: Callable[[], SuiteContexts]):
        _REGISTRY[env_name] = factory
        return factory

    return decorator


def suite_contexts(env_name: str) -> SuiteContexts:
    """The registered :class:`SuiteContexts` for ``env_name``."""
    try:
        return _REGISTRY[env_name]()
    except KeyError:
        raise KeyError(
            f"No context space registered for {env_name!r}. Registered: {sorted(_REGISTRY)}. "
            "Count-typed suites need pad-and-mask support first."
        ) from None


def registered_environments() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


# --------------------------------------------------------------------------- #
# Safe Velocity: one dimension, the speed limit.
# --------------------------------------------------------------------------- #

# Ω spans from a bit below the hardest level to the easiest level. The lower
# bound is deliberately not 0: a threshold of 0 makes every step a violation
# and the task degenerate. 0.4 × baseline sits below level 3 (0.5 ×) so the
# uniform distribution also covers "harder than any level".
_VELOCITY_LOW_FRACTION = 0.4
_VELOCITY_HIGH_FRACTION = 1.0

# How each agent's speed is measured for the constraint (SafeVelocity*.velocity_mode).
_VELOCITY_MEASURE = {
    "ant": "the torso's speed in the horizontal plane, √(vₓ² + vᵧ²)",
    "halfcheetah": "the forward velocity vₓ",
    "hopper": "the forward velocity vₓ",
    "humanoid": "the centre of mass's speed in the horizontal plane",
    "swimmer": "the forward velocity vₓ",
    "walker2d": "the forward velocity vₓ",
}


def _velocity_task(agent: str) -> TaskDescription:
    # The reward is the agent's stock MuJoCo/Brax locomotion reward; its weights differ per
    # agent (see crax/envs/<agent>.py), so it is described by structure, not by constants.
    return TaskDescription(
        agent=f"A MuJoCo {agent} that has to run forward along the x-axis. Actions are joint torques in [−1, 1]. "
        f"An episode ends either at the step limit or earlier when the agent falls (its torso leaves the healthy "
        f"height range); violating the constraint does not end the episode.",
        reward=f"The {agent}'s stock locomotion reward per step: forward velocity vₓ, plus a constant for staying "
        f"upright, minus a penalty proportional to ‖action‖² (large torques), with the agent's own weights, "
        f"multiplied by the suite's reward scale of 0.01. The return is the sum over the episode; higher means it "
        f"travelled further with less effort.",
        cost=f"Per step, 1 if the agent's speed — {_VELOCITY_MEASURE[agent]} — exceeds the episode's velocity_threshold, "
        f"else 0. The episode's cost is the number of steps over the limit; the budget bounds its mean.",
    )


def _velocity_suite(agent: str) -> Callable[[], SuiteContexts]:
    def factory() -> SuiteContexts:
        from crax.envs.safe_velocity import DEFAULT_THRESHOLDS, get_threshold_for_level

        baseline = DEFAULT_THRESHOLDS[agent]
        space = ContextSpace((
            Dimension(
                "velocity_threshold",
                low=_VELOCITY_LOW_FRACTION * baseline,
                high=_VELOCITY_HIGH_FRACTION * baseline,
                description=f"the speed above which a step counts as a violation (the {agent}'s speed limit)",
            ),
        ))
        level_contexts = {
            level: space.encode(velocity_threshold=get_threshold_for_level(agent, level)) for level in LEVELS
        }
        return SuiteContexts(
            env_name=f"safe_velocity_{agent}",
            space=space,
            level_contexts=level_contexts,
            parameter_names=("velocity_threshold",),
            task=_velocity_task(agent),
        )

    return factory


for _agent in ("ant", "halfcheetah", "hopper", "humanoid", "swimmer", "walker2d"):
    register(f"safe_velocity_{_agent}")(_velocity_suite(_agent))

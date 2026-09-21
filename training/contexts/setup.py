"""From two CLI strings — training distribution, deployment distribution — to
the keyword arguments ``train(...)`` needs.

A distribution is written as a short spec, the same grammar for both flags:

    uniform            r, Uniform(Ω)
    level:3            point mass on the suite's difficulty level 3
    staged:1,2,3       CRAX's manual curriculum: level 1, then 2, then 3, equal split of the rounds

The training distribution is what the student learns on. The deployment
distribution w is what it is *for*; the policy is evaluated on w and, always,
on r as well (metrics ``evaluation/deployment/...`` and ``evaluation/uniform/...``).
Nothing here is suite-specific: levels are resolved through the registry, and a
suite without levels simply rejects ``level:``/``staged:`` specs.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, Optional

from training.contexts.distribution import ContextDistribution
from training.contexts.distributions import FixedContext, StagedContexts, UniformDistribution
from training.contexts.registry import SuiteContexts, suite_contexts
from training.contexts.round_hook import ContextRoundHook
from training.contexts.wrapper import make_wrap_env_fn

NO_DISTRIBUTION = "none"
DEPLOYMENT_EVALUATION = "deployment"
UNIFORM_EVALUATION = "uniform"
SPEC_HELP = "'uniform', 'level:<n>', or 'staged:<n>,<n>,...'"


def parse_distribution(spec: str, suite: SuiteContexts, total_rounds: int) -> ContextDistribution:
    """A :class:`ContextDistribution` from its CLI spec (see module docstring)."""
    kind, _, argument = spec.partition(":")
    if kind == "uniform" and not argument:
        return UniformDistribution(suite.space)
    if kind == "level" and argument:
        return FixedContext(suite.space, suite.level(int(argument)))
    if kind == "staged" and argument:
        import jax.numpy as jnp

        levels = [int(level) for level in argument.split(",")]
        stages = jnp.stack([suite.level(level) for level in levels])
        return StagedContexts.equal_split(suite.space, stages, total_rounds)
    raise ValueError(f"cannot parse distribution spec {spec!r}; expected {SPEC_HELP}")


def spec_label(spec: str) -> str:
    """Run-name fragment: ``staged:1,2,3`` -> ``staged123``."""
    return spec.replace(":", "").replace(",", "")


@dataclasses.dataclass(frozen=True)
class ContextTrainingSetup:
    """Everything an entry point needs to run one arm of an experiment."""

    suite: SuiteContexts
    training_spec: str
    deployment_spec: str
    distribution: ContextDistribution
    deployment: ContextDistribution
    steps_per_round: int
    total_rounds: int

    def train_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for ``train(...)`` (any CRAX PPO-family trainer).

        ``training_metrics_steps`` makes the trainer report the training
        episodes' return, cost and length (``episodic/*``) once per round, the
        same cadence as every other training-side metric on the dashboard.
        """
        return {
            "wrap_env_fn": make_wrap_env_fn(self.distribution),
            "round_hook": ContextRoundHook(self.distribution),
            "evaluation_wrap_env_fns": {
                DEPLOYMENT_EVALUATION: make_wrap_env_fn(self.deployment),
                UNIFORM_EVALUATION: make_wrap_env_fn(UniformDistribution(self.suite.space)),
            },
            "training_metrics_steps": self.steps_per_round,
        }

    def wandb_config(self) -> Dict[str, Any]:
        """What the dashboard's text panels need to describe this run (see training/dashboard)."""
        return {
            "task": dataclasses.asdict(self.suite.task),
            "context_space": {
                dimension.name: {"low": dimension.low, "high": dimension.high, "description": dimension.description}
                for dimension in self.suite.space.dimensions
            },
            "training_distribution": self.training_spec,
            "deployment_distribution": self.deployment_spec,
            "num_rounds": self.total_rounds,
            "environment_steps_per_round": self.steps_per_round,
        }

    def describe(self) -> str:
        switches = getattr(self.distribution, "switch_rounds", None)
        staged_note = f" (stage switches at rounds {list(switches)})" if switches is not None else ""
        return (
            f"training on {self.training_spec}{staged_note}, deployment {self.deployment_spec}; "
            f"{self.total_rounds} rounds of {self.steps_per_round:,} env steps; "
            f"evaluated on [{DEPLOYMENT_EVALUATION}, {UNIFORM_EVALUATION}]\n{self.suite.space.describe()}"
        )


def context_training_setup(
    env_name: str,
    training_spec: str,
    deployment_spec: str,
    *,
    num_timesteps: int,
    batch_size: int,
    unroll_length: int,
    num_minibatches: int,
    action_repeat: int = 1,
) -> Optional[ContextTrainingSetup]:
    """The setup for one run, or ``None`` when ``training_spec`` is ``"none"``."""
    if training_spec == NO_DISTRIBUTION:
        return None
    suite = suite_contexts(env_name)
    steps_per_round = batch_size * unroll_length * num_minibatches * action_repeat
    total_rounds = math.ceil(num_timesteps / steps_per_round)  # a staged curriculum splits these equally
    return ContextTrainingSetup(
        suite=suite,
        training_spec=training_spec,
        deployment_spec=deployment_spec,
        distribution=parse_distribution(training_spec, suite, total_rounds),
        deployment=parse_distribution(deployment_spec, suite, total_rounds),
        steps_per_round=steps_per_round,
        total_rounds=total_rounds,
    )

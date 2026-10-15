"""From the CLI — training distribution, deployment distribution, and the two
learner switches — to the keyword arguments ``train(...)`` needs.

A distribution is written as a short spec, the same grammar for both flags:

    uniform            r, Uniform(Ω)
    level:3            point mass on the suite's difficulty level 3
    staged:1,2,3       CRAX's manual curriculum: level 1, then 2, then 3, equal split of the rounds
    plr                Prioritized Level Replay scored by the reward critic's error (the paper's;
                       docs/acl/design/prioritized_level_replay.md)
    plr:<signal>       the same, scored by that signal's magnitude: reward, cost, or lagrangian
                       (the advantage the policy loss optimises, A_r − λ A_c)

The training distribution is what the student learns on. The deployment
distribution w is what it is *for*; the policy is evaluated on w, on r, and on
Uniform(region) for every :class:`~training.contexts.registry.EvaluationRegion`
the suite registers (metrics ``evaluation/<name>/...``). The
:class:`~training.contexts.learner.Learner` says how the constraint is held and
what the policy observes. Nothing here is suite-specific: levels and regions
are resolved through the registry.

An **arm** of an experiment is a training distribution under a learner; the
label ``"<training spec> <constraint> <policy context>"`` names it in the run
name and in the W&B config (``arm``), which the dashboard groups lines by.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, Optional, Tuple

from training.contexts.distribution import ContextDistribution
from training.contexts.distributions import FixedContext, PrioritizedLevelReplay, StagedContexts, UniformDistribution
from training.contexts.learner import Learner
from training.contexts.registry import OUT_OF_DISTRIBUTION_REGION, SuiteContexts, suite_contexts
from training.contexts.round_hook import ContextRoundHook
from training.contexts.wrapper import make_evaluation_wrap_env_fn, make_wrap_env_fn
from training.rounds import Signal

NO_DISTRIBUTION = "none"
# The published PPO-Lagrange: one λ, a policy that does not see ω. Every other learner needs a context distribution.
DEFAULT_LEARNER = Learner(constraint="scalar", policy_context="blind")
# The two evaluations every context run has; a suite's regions add theirs under their own names.
DEPLOYMENT_EVALUATION = "deployment"
UNIFORM_EVALUATION = "uniform"
OUT_OF_DISTRIBUTION_EVALUATION = OUT_OF_DISTRIBUTION_REGION
SIGNAL_NAMES = tuple(signal.value for signal in Signal)
SPEC_HELP = f"'uniform', 'level:<n>', 'staged:<n>,<n>,...', 'plr', or 'plr:<signal>' with signal in {SIGNAL_NAMES}"


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """One distribution the policy is scored on: its W&B name (``evaluation/<name>/*``),
    the distribution, and the words the dashboard uses for it."""

    name: str
    distribution: ContextDistribution
    label: str


def parse_distribution(spec: str, suite: SuiteContexts, total_rounds: int) -> ContextDistribution:
    """A :class:`ContextDistribution` from its CLI spec (see module docstring)."""
    kind, _, argument = spec.partition(":")
    if kind == "uniform" and not argument:
        return UniformDistribution(suite.space)
    if kind == "plr" and not argument:
        return PrioritizedLevelReplay(suite.space)
    if kind == "plr" and argument in SIGNAL_NAMES:
        return PrioritizedLevelReplay(suite.space, score=Signal(argument))
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


def arm_label(training_spec: str, learner: Learner) -> str:
    """The arm as the dashboard legend shows it: ``"staged:1,2,3 additive told"``."""
    return f"{training_spec} {learner.label}"


def run_name_fragment(training_spec: str, learner: Learner) -> str:
    """The arm in a run name: ``staged123_additive_told``."""
    return f"{spec_label(training_spec)}_{learner.label.replace(' ', '_')}"


@dataclasses.dataclass(frozen=True)
class ContextTrainingSetup:
    """Everything an entry point needs to run one arm of an experiment."""

    suite: SuiteContexts
    training_spec: str
    deployment_spec: str
    distribution: ContextDistribution
    learner: Learner
    evaluations: Tuple[Evaluation, ...]
    num_envs: int
    steps_per_round: int
    total_rounds: int

    @property
    def arm(self) -> str:
        return arm_label(self.training_spec, self.learner)

    def train_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for ``train(...)`` (any CRAX PPO-family trainer; a per-context
        constraint needs PPO-Lagrange, the only agent that takes a ``constraint``).

        ``training_metrics_steps`` makes the trainer report the training
        episodes' return, cost and length (``episodic/*``) once per round, the
        same cadence as every other training-side metric on the dashboard.
        """
        return {
            "wrap_env_fn": make_wrap_env_fn(self.distribution),
            "round_hook": ContextRoundHook(self.distribution, num_slots=self.num_envs),
            "evaluation_wrap_env_fns": {
                evaluation.name: make_evaluation_wrap_env_fn(evaluation.distribution) for evaluation in self.evaluations
            },
            "training_metrics_steps": self.steps_per_round,
            **self.learner.train_kwargs(self.suite.space),
        }

    def wandb_config(self) -> Dict[str, Any]:
        """What the dashboard's text panels need to describe this run (see training/dashboard)."""
        return {
            "task": dataclasses.asdict(self.suite.task),
            "context_space": {
                dimension.name: {"low": dimension.low, "high": dimension.high, "description": dimension.description}
                for dimension in self.suite.space.dimensions
            },
            "context_space_integer_total_cap": self.suite.space.integer_total_cap,
            "training_distribution": self.training_spec,
            "deployment_distribution": self.deployment_spec,
            "constraint": self.learner.constraint,
            "policy_context": self.learner.policy_context,
            "arm": self.arm,
            "evaluations": {evaluation.name: evaluation.label for evaluation in self.evaluations},
            "num_rounds": self.total_rounds,
            "environment_steps_per_round": self.steps_per_round,
        }

    def describe(self) -> str:
        switches = getattr(self.distribution, "switch_rounds", None)
        staged_note = f" (stage switches at rounds {list(switches)})" if switches is not None else ""
        return (
            f"training on {self.training_spec}{staged_note}, deployment {self.deployment_spec}; "
            f"constraint {self.learner.constraint}, policy {self.learner.policy_context}, critics told; "
            f"{self.total_rounds} rounds of {self.steps_per_round:,} env steps; "
            f"evaluated on {[evaluation.name for evaluation in self.evaluations]}\n{self.suite.space.describe()}"
        )


def evaluations_for(suite: SuiteContexts, deployment_spec: str, total_rounds: int) -> Tuple[Evaluation, ...]:
    """The distributions every run of ``suite`` is scored on: w (``deployment_spec``), r,
    then Uniform(region) for each of the suite's evaluation regions. The one place that
    rule is written; ``train_kwargs``, the W&B config and the dashboard all read the result."""
    for region in suite.evaluation_regions:
        if region.name in (DEPLOYMENT_EVALUATION, UNIFORM_EVALUATION):
            raise ValueError(f"{suite.env_name}: evaluation region {region.name!r} collides with a built-in evaluation")
    return (
        Evaluation(DEPLOYMENT_EVALUATION, parse_distribution(deployment_spec, suite, total_rounds), f"the deployment task ({deployment_spec})"),
        Evaluation(UNIFORM_EVALUATION, UniformDistribution(suite.space), "all of Ω (uniform)"),
        *(Evaluation(region.name, UniformDistribution(region.space), region.label) for region in suite.evaluation_regions),
    )


def context_training_setup(
    env_name: str,
    training_spec: str,
    deployment_spec: str,
    learner: Learner,
    *,
    num_timesteps: int,
    num_envs: int,
    batch_size: int,
    unroll_length: int,
    num_minibatches: int,
    action_repeat: int = 1,
) -> Optional[ContextTrainingSetup]:
    """The setup for one run, or ``None`` when ``training_spec`` is ``"none"`` (a stock run,
    which has no context for the learner to be contextual about: refused unless the
    learner is the default, scalar and blind)."""
    if training_spec == NO_DISTRIBUTION:
        if learner != DEFAULT_LEARNER:
            raise ValueError(
                f"--constraint {learner.constraint} / --policy_context {learner.policy_context} need a context "
                f"distribution; a stock run (--context_distribution none) has no context"
            )
        return None
    suite = suite_contexts(env_name)
    steps_per_round = batch_size * unroll_length * num_minibatches * action_repeat
    total_rounds = math.ceil(num_timesteps / steps_per_round)  # a staged curriculum splits these equally
    return ContextTrainingSetup(
        suite=suite,
        training_spec=training_spec,
        deployment_spec=deployment_spec,
        distribution=parse_distribution(training_spec, suite, total_rounds),
        learner=learner,
        evaluations=evaluations_for(suite, deployment_spec, total_rounds),
        num_envs=num_envs,
        steps_per_round=steps_per_round,
        total_rounds=total_rounds,
    )

"""From the CLI — training distribution, deployment distribution, evaluation
protocol, and the two learner switches — to the keyword arguments ``train(...)`` needs.

A distribution is written as a short spec, the same grammar wherever one is named:

    uniform            uniform over the region it is given (the training contexts, or Ω)
    level:3            point mass on the suite's difficulty level 3
    staged:1,2,3       CRAX's manual curriculum: level 1, then 2, then 3, equal split of the rounds
    plr                Prioritized Level Replay scored by the reward critic's error (the paper's;
                       docs/acl/design/prioritized_level_replay.md)
    plr:<signal>       the same, scored by that signal's magnitude: reward, cost, or lagrangian
                       (the advantage the policy loss optimises, A_r − λ A_c)

The training distribution is what the student learns on. The **evaluation
protocol** (:class:`~training.contexts.protocol.EvaluationProtocol`,
``docs/acl/design/generalization_evaluation.md``) says where it may put mass:
on the protocol's training contexts, never on its held-out regions. A
**reference** run trains on all of the protocol's contexts instead. The policy
is evaluated on the deployment distribution w, uniformly on all of the
protocol's contexts, and, when the protocol holds contexts out, uniformly on
its training contexts and on each held-out region (metrics
``evaluation/<name>/...``). Every run of a protocol has the same evaluations,
reference or not. The :class:`~training.contexts.learner.Learner` says how the
constraint is held and what the policy observes; it is built over the
protocol's contexts, so a coordinate the protocol keeps fixed is no input to
the multiplier. Nothing here is suite-specific: levels and protocols are
resolved through the registry.

An **arm** of an experiment is everything that may differ between the runs of
one group: the training distribution, the learner, whether the run is a
reference, and any environment overrides the run was given (``--env_kwargs``,
e.g. a variant of the suite's sensors). The label
``"<training spec> <learner> [reference] <key>=<value> ..."`` names it in the
run name and in the W&B config (``arm``), which the dashboard groups lines by.
"""
from __future__ import annotations

import dataclasses
import math
import re
from typing import Any, Dict, Mapping, Optional, Tuple

from training.contexts.distribution import ContextDistribution
from training.contexts.distributions import FixedContext, PrioritizedLevelReplay, StagedContexts, UniformDistribution
from training.contexts.learner import Learner
from training.contexts.protocol import NO_PROTOCOL, TRAINING_REGION, EvaluationProtocol
from training.contexts.registry import SuiteContexts, suite_contexts
from training.contexts.space import Context, ContextSpace
from training.contexts.round_hook import ContextRoundHook
from training.contexts.wrapper import make_evaluation_wrap_env_fn, make_wrap_env_fn
from training.rounds import Signal

NO_DISTRIBUTION = "none"
# The published PPO-Lagrange: one λ, a policy that does not see ω. Every other learner needs a context distribution.
DEFAULT_LEARNER = Learner(constraint="scalar", policy_context="blind")
# The two evaluations every context run has. A protocol that holds contexts out adds one on its
# training contexts and one per held-out region, under the region's own name.
DEPLOYMENT_EVALUATION = "deployment"
UNIFORM_EVALUATION = "uniform"
TRAINING_EVALUATION = TRAINING_REGION
SIGNAL_NAMES = tuple(signal.value for signal in Signal)
SPEC_HELP = f"'uniform', 'level:<n>', 'staged:<n>,<n>,...', 'plr' or 'plr:<signal>' with signal in {SIGNAL_NAMES}"


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """One distribution the policy is scored on: its W&B name (``evaluation/<name>/*``),
    the distribution, and the words the dashboard uses for it."""

    name: str
    distribution: ContextDistribution
    label: str


def parse_distribution(spec: str, suite: SuiteContexts, region: ContextSpace, total_rounds: int) -> ContextDistribution:
    """A :class:`ContextDistribution` from its CLI spec (see module docstring) that puts mass
    only on ``region``, a region of the suite's Ω: a run's training contexts, or all of its
    protocol's contexts for an evaluation. A spec that names a context outside ``region`` (a
    held-out level) is refused."""
    kind, _, argument = spec.partition(":")
    if kind == "uniform" and not argument:
        return UniformDistribution(region)
    if kind == "plr" and not argument:
        return PrioritizedLevelReplay(region)
    if kind == "plr" and argument in SIGNAL_NAMES:
        return PrioritizedLevelReplay(region, score=Signal(argument))
    if kind == "level" and argument:
        return FixedContext(region, _level_within(suite, int(argument), region))
    if kind == "staged" and argument:
        import jax.numpy as jnp

        stages = jnp.stack([_level_within(suite, int(level), region) for level in argument.split(",")])
        return StagedContexts.equal_split(region, stages, total_rounds)
    raise ValueError(f"cannot parse distribution spec {spec!r}; expected {SPEC_HELP}")


def _level_within(suite: SuiteContexts, level: int, region: ContextSpace) -> Context:
    """The suite's difficulty level as a context, refused when ``region`` does not contain it."""
    context = suite.level(level)
    if not bool(region.contains(context[None])[0]):
        raise ValueError(
            f"{suite.env_name}: level {level} ({suite.space.decode(context)}) is not among the contexts this "
            f"distribution may sample under the run's evaluation protocol\n{region.describe()}"
        )
    return context


def spec_label(spec: str) -> str:
    """Run-name fragment: ``staged:1,2,3`` -> ``staged123``."""
    return spec.replace(":", "").replace(",", "")


EnvironmentOverrides = Mapping[str, Any]
"""The environment constructor arguments a run overrides (``--env_kwargs``); ``{}`` for the suite's defaults."""


REFERENCE_WORD = "reference"


def arm_label(training_spec: str, learner: Learner, reference: bool, environment_overrides: EnvironmentOverrides) -> str:
    """The arm as the dashboard legend shows it: ``"staged:1,2,3 additive told"``,
    ``"uniform additive told reference hazard_lidar=surface_by_kind"``."""
    overrides = "".join(f" {key}={value}" for key, value in sorted(environment_overrides.items()))
    return f"{training_spec} {learner.label}{' ' + REFERENCE_WORD if reference else ''}{overrides}"


def run_name_fragment(training_spec: str, learner: Learner, reference: bool, environment_overrides: EnvironmentOverrides) -> str:
    """The arm in a run name: ``staged123_additive_told``,
    ``uniform_additive_told_reference_hazard_lidar_surface_by_kind``."""
    overrides = "".join(f"_{key}_{value}" for key, value in sorted(environment_overrides.items()))
    learner_words = learner.label.replace(" ", "_") + (f"_{REFERENCE_WORD}" if reference else "")
    return re.sub(r"[^A-Za-z0-9_]", "", f"{spec_label(training_spec)}_{learner_words}{overrides}")


@dataclasses.dataclass(frozen=True)
class ContextTrainingSetup:
    """Everything an entry point needs to run one arm of an experiment."""

    suite: SuiteContexts
    protocol: EvaluationProtocol
    reference: bool  # trains on all of the protocol's contexts, its held-out regions included
    training_spec: str
    deployment_spec: str
    distribution: ContextDistribution
    learner: Learner
    environment_overrides: EnvironmentOverrides
    evaluations: Tuple[Evaluation, ...]
    num_envs: int
    steps_per_round: int
    total_rounds: int

    @property
    def arm(self) -> str:
        return arm_label(self.training_spec, self.learner, self.reference, self.environment_overrides)

    @property
    def run_name_fragment(self) -> str:
        return run_name_fragment(self.training_spec, self.learner, self.reference, self.environment_overrides)

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
            **self.learner.train_kwargs(self.protocol.contexts),
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
            "evaluation_protocol": self.protocol.name,
            "reference": self.reference,
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
        if not self.protocol.held_out:
            held_out = ""
        elif self.reference:
            held_out = f"protocol {self.protocol.name}, reference run: nothing held out of training; "
        else:
            held_out = f"protocol {self.protocol.name}, held out of training: {', '.join(region.name for region in self.protocol.held_out)}; "
        return (
            f"training on {self.training_spec}{staged_note}, deployment {self.deployment_spec}; {held_out}"
            f"constraint {self.learner.constraint}, policy {self.learner.policy_context}, critics told; "
            f"{self.total_rounds} rounds of {self.steps_per_round:,} env steps; "
            f"evaluated on {[evaluation.name for evaluation in self.evaluations]}\n{self.distribution.space.describe()}"
        )


def evaluations_for(suite: SuiteContexts, protocol: EvaluationProtocol, deployment_spec: str, total_rounds: int) -> Tuple[Evaluation, ...]:
    """The distributions every run of ``suite`` under ``protocol`` is scored on: w
    (``deployment_spec``, anywhere in the protocol's contexts), uniform over all of them, and, when
    the protocol holds contexts out, uniform over its training contexts and over each held-out
    region. The same for a reference run, so the two are compared evaluation by evaluation. The
    one place that rule is written; ``train_kwargs``, the W&B config and the dashboard all
    read the result."""
    for region in protocol.held_out:  # that none is called "training" is the protocol's own check
        if region.name in (DEPLOYMENT_EVALUATION, UNIFORM_EVALUATION):
            raise ValueError(f"{suite.env_name}: held-out region {region.name!r} collides with a built-in evaluation")
    always = (
        Evaluation(DEPLOYMENT_EVALUATION, parse_distribution(deployment_spec, suite, protocol.contexts, total_rounds), f"the deployment task ({deployment_spec})"),
        Evaluation(UNIFORM_EVALUATION, UniformDistribution(protocol.contexts), "all of the protocol's contexts (uniform)"),
    )
    if not protocol.held_out:
        return always
    return (
        *always,
        Evaluation(TRAINING_EVALUATION, UniformDistribution(protocol.training), "the training contexts (uniform; the held-out regions excluded)"),
        *(Evaluation(region.name, UniformDistribution(region.space), region.label) for region in protocol.held_out),
    )


def context_training_setup(
    env_name: str,
    training_spec: str,
    deployment_spec: str,
    learner: Learner,
    *,
    protocol_name: str,
    reference: bool,
    environment_overrides: EnvironmentOverrides,
    num_timesteps: int,
    num_envs: int,
    batch_size: int,
    unroll_length: int,
    num_minibatches: int,
    action_repeat: int = 1,
) -> Optional[ContextTrainingSetup]:
    """The setup for one run, or ``None`` when ``training_spec`` is ``"none"`` (a stock run,
    which has no context for the learner to be contextual about: refused unless the
    learner is the default, scalar and blind, and no protocol is named).
    ``protocol_name`` and ``reference`` are the run's ``--evaluation_protocol`` and
    ``--reference``: where the training distribution may put mass. ``environment_overrides``
    are the run's ``--env_kwargs``; they name the arm with the distribution and the learner."""
    if training_spec == NO_DISTRIBUTION:
        if learner != DEFAULT_LEARNER:
            raise ValueError(
                f"--constraint {learner.constraint} / --policy_context {learner.policy_context} need a context "
                f"distribution; a stock run (--context_distribution none) has no context"
            )
        if protocol_name != NO_PROTOCOL or reference:
            raise ValueError("--evaluation_protocol / --reference need a context distribution; a stock run has no context")
        return None
    suite = suite_contexts(env_name)
    protocol = suite.protocol(protocol_name)
    if reference and not protocol.held_out:
        raise ValueError(f"--reference means training on a protocol's held-out contexts too; protocol {protocol.name!r} holds none out")
    steps_per_round = batch_size * unroll_length * num_minibatches * action_repeat
    total_rounds = math.ceil(num_timesteps / steps_per_round)  # a staged curriculum splits these equally
    return ContextTrainingSetup(
        suite=suite,
        protocol=protocol,
        reference=reference,
        training_spec=training_spec,
        deployment_spec=deployment_spec,
        distribution=parse_distribution(training_spec, suite, protocol.training_contexts(reference), total_rounds),
        learner=learner,
        environment_overrides=dict(environment_overrides),
        evaluations=evaluations_for(suite, protocol, deployment_spec, total_rounds),
        num_envs=num_envs,
        steps_per_round=steps_per_round,
        total_rounds=total_rounds,
    )

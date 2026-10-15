"""The per-context multiplier as a network: λ_ψ(ω) = softplus(f_ψ(ω̃)), moved
each round by PPO-Lagrange's integrator.

A scalar λ is updated from the mean cost of whatever the curriculum sampled
this round, so it settles where the policy is on budget for the training
distribution and nowhere else (``research_design_brief.md`` §6). Here the
constraint ``C(π, ω) ≤ d`` is held for every ω through a multiplier that is a
function of the context: a small network ``f_ψ`` over the context, normalised to
Ω's box, through a positive map. The policy loss weights each transition by the
multiplier of the context it was generated in. The other per-context form,
monotone and additive in the factors of ω, is ``additive.py``; the shared
machinery (targets, metrics) is ``fitted.py``.

**The update is the integrator, projected onto the function class.** After a
round, every transition names where its context's multiplier should move by
PPO-Lagrange's own rule,

    λ*_t = relu( λ_ψ(ω_t) + rate · (c_t − d) ),

and ψ is fitted to those targets by a few Adam steps on the squared error,
the way a critic is fitted to its targets. One context, one cost: the
function's value there moves by exactly the scalar's step. Many contexts: each
moves by its own transitions' step, and the network's smoothness spreads the
movement to contexts near them. A context the curriculum has not sampled
therefore has a multiplier before its first episode, shaped by its neighbours,
so a stage switch does not start from zero
(``docs/acl/design/contextual_learner.md`` §1).

Why not gradient ascent on the dual objective directly: its step size is in
units of the hidden weights, not of λ, and at the integrator's rate the
function diverges within tens of rounds (measured 2026-10-14). The target form
keeps ``rate`` meaning what it means for the scalar.

The network starts as the scalar does: its output layer is zero so that
``λ_ψ ≡ initial λ`` for every ω, and the first rounds shape it from the data.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, Tuple

import flax.struct
import jax
import jax.numpy as jnp
import optax
from flax import linen

from training import types
from training.agents.ppo_lag.constraint import StateExtras
from training.contexts.constraints.fitted import integrator_targets, multiplier_metrics
from training.contexts.rollout import TRANSITION_CONTEXT_KEY
from training.contexts.space import ContextSpace

HIDDEN_LAYER_SIZES: Tuple[int, ...] = (64, 64)
# Adam steps towards the round's targets, and their size. Measured 2026-10-14 on goal-point's Ω against the
# integrator: at (32, 3e-3) the function tracks a scalar λ to within 0.06 over 40 rounds and reproduces
# the per-hazard-total integrator value under uniform sampling to within 0.02; fewer steps lag, a larger
# rate overshoots.
FIT_STEPS = 32
FIT_LEARNING_RATE = 3e-3


@flax.struct.dataclass
class MultiplierFunctionState:
    """What ``TrainingState.aux_state`` holds for a per-context constraint: ψ and its optimiser."""

    parameters: types.Params
    optimizer_state: optax.OptState


class _PreActivation(linen.Module):
    """``f_ψ``: an MLP whose output layer starts at zero weights and a chosen bias, so the
    function is a constant until the first update."""

    hidden_layer_sizes: Tuple[int, ...]
    initial_output: float

    @linen.compact
    def __call__(self, normalised_context: jax.Array) -> jax.Array:
        hidden = normalised_context
        for size in self.hidden_layer_sizes:
            hidden = linen.swish(linen.Dense(size, kernel_init=jax.nn.initializers.lecun_uniform())(hidden))
        output = linen.Dense(
            1, kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.constant(self.initial_output)
        )(hidden)
        return jnp.squeeze(output, axis=-1)


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


@dataclasses.dataclass(frozen=True)
class NetworkConstraint:
    """λ_ψ(ω) over ``space`` (a :class:`~training.agents.ppo_lag.constraint.LagrangeConstraint`).

    ``seed`` initialises the hidden layers; the output layer is deterministic (zero
    weights), so the initial function is the constant ``initial_lambda`` whatever the seed.
    """

    space: ContextSpace
    seed: int = 0

    @property
    def extra_fields(self) -> Tuple[str, ...]:
        return (TRANSITION_CONTEXT_KEY,)

    def initial_multipliers(self, initial_value: float) -> MultiplierFunctionState:
        """ψ such that ``λ_ψ(ω) = initial_value`` for every ω. Softplus is never exactly 0, so
        the initial value must be positive; a scalar λ may start at 0 because its projection
        can sit there, a function's cannot."""
        if not initial_value > 0.0:
            raise ValueError(
                f"a per-context multiplier λ_ψ(ω) = softplus(f_ψ(ω)) cannot start at {initial_value}; pass "
                f"--initial_lambda_lagr > 0 (it is the constant the function starts from)"
            )
        module = _PreActivation(hidden_layer_sizes=HIDDEN_LAYER_SIZES, initial_output=_inverse_softplus(initial_value))
        parameters = module.init(jax.random.PRNGKey(self.seed), jnp.zeros((1, self.space.size)))
        return MultiplierFunctionState(parameters=parameters, optimizer_state=_optimizer().init(parameters))

    def per_transition(self, multipliers: MultiplierFunctionState, state_extras: StateExtras) -> jax.Array:
        """``λ_ψ`` at each transition's context: ``[..., T]`` for contexts ``[..., T, D]``."""
        return self.multipliers_at(multipliers, state_extras[TRANSITION_CONTEXT_KEY])

    def update(
        self, multipliers: MultiplierFunctionState, costs: jax.Array, state_extras: StateExtras,
        budget_per_step: float, rate: float,
    ) -> Tuple[MultiplierFunctionState, Dict[str, jax.Array]]:
        """Move the function by the integrator: fit ψ to the targets of
        :func:`~training.contexts.constraints.fitted.integrator_targets` over the round's
        transitions by ``FIT_STEPS`` Adam steps; metrics as in :func:`multiplier_metrics`."""
        contexts = state_extras[TRANSITION_CONTEXT_KEY]
        targets = integrator_targets(self._apply(multipliers.parameters, contexts), costs, budget_per_step, rate)

        def squared_error(parameters: types.Params) -> jax.Array:
            return jnp.mean(jnp.square(self._apply(parameters, contexts) - targets))

        def fit_step(carry: Tuple[types.Params, optax.OptState], _: None) -> Tuple[Tuple[types.Params, optax.OptState], None]:
            parameters, optimizer_state = carry
            updates, optimizer_state = _optimizer().update(jax.grad(squared_error)(parameters), optimizer_state, parameters)
            return (optax.apply_updates(parameters, updates), optimizer_state), None

        (parameters, optimizer_state), _ = jax.lax.scan(
            fit_step, (multipliers.parameters, multipliers.optimizer_state), None, length=FIT_STEPS
        )
        per_transition = self._apply(parameters, contexts)
        return MultiplierFunctionState(parameters=parameters, optimizer_state=optimizer_state), multiplier_metrics(per_transition, targets)

    def multipliers_at(self, multipliers: MultiplierFunctionState, contexts: jax.Array) -> jax.Array:
        """``λ_ψ`` at arbitrary contexts ``[..., D]``: the function can be read anywhere in Ω."""
        return self._apply(multipliers.parameters, contexts)

    # ---- the function ---------------------------------------------------- #

    def _apply(self, parameters: types.Params, contexts: jax.Array) -> jax.Array:
        # `initial_output` only acts at init; from then on the stored bias carries it.
        module = _PreActivation(hidden_layer_sizes=HIDDEN_LAYER_SIZES, initial_output=0.0)
        return jax.nn.softplus(module.apply(parameters, self._normalise(contexts)))

    def _normalise(self, contexts: jax.Array) -> jax.Array:
        """ω mapped to [-1, 1] per dimension over Ω's box (0 on a degenerate dimension), so the
        network's inputs are of order one whatever a dimension's units."""
        middle = (self.space.high + self.space.low) / 2.0
        half_range = (self.space.high - self.space.low) / 2.0
        degenerate = half_range == 0.0
        return jnp.where(degenerate, 0.0, (contexts - middle) / jnp.where(degenerate, 1.0, half_range))


def _optimizer() -> optax.GradientTransformation:
    return optax.adam(FIT_LEARNING_RATE)

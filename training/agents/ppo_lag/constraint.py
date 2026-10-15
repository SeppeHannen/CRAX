"""How PPO-Lagrange holds its constraint: with one multiplier for the whole
training distribution, or with one per context.

The agent asks a :class:`LagrangeConstraint` three things and knows nothing
else about where λ lives: the λ each transition is weighted by in the loss, the
multiplier state after a round given the round's transitions, and which
``state.info`` keys the trainer must record for the first two to be answerable.
The multiplier state is whatever the constraint keeps in ``TrainingState.aux_state``:
a ``[1]`` array for a scalar λ, a network's parameters for λ(ω).

:class:`ScalarConstraint` is the published PPO-Lagrange: one λ, updated from the
batch's mean cost. The per-context forms λ(ω) are ``training.contexts.constraints``
(``docs/acl/design/contextual_learner.md`` §1).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Mapping, Protocol, Tuple, runtime_checkable

import jax
import jax.numpy as jnp

StateExtras = Mapping[str, jax.Array]
"""The recorded ``state.info`` fields of a batch of transitions, each ``[..., T]`` or ``[..., T, D]``."""

Multipliers = Any
"""A constraint's multiplier state: a pytree of arrays of fixed structure and shapes."""


@runtime_checkable
class LagrangeConstraint(Protocol):
    """One rule for the multipliers the Lagrangian loss uses."""

    @property
    def extra_fields(self) -> Tuple[str, ...]:
        """``state.info`` keys the trainer must record per transition for :meth:`per_transition`."""

    def initial_multipliers(self, initial_value: float) -> Multipliers:
        """The multiplier state before the first round: λ = ``initial_value`` everywhere."""

    def per_transition(self, multipliers: Multipliers, state_extras: StateExtras) -> jax.Array:
        """The λ each transition in ``state_extras`` is weighted by: broadcastable against the
        transitions' leading shape ``[..., T]``."""

    def update(
        self, multipliers: Multipliers, costs: jax.Array, state_extras: StateExtras, budget_per_step: float, rate: float
    ) -> Tuple[Multipliers, Dict[str, jax.Array]]:
        """The multiplier state after a round whose transitions had per-step ``costs`` (``[..., T]``),
        and the ``training/*`` metrics that describe it (``lambda_lagr`` among them)."""


@dataclasses.dataclass(frozen=True)
class ScalarConstraint:
    """One λ for the whole training distribution: ``λ ← relu(λ + rate · (mean cost − budget))``."""

    @property
    def extra_fields(self) -> Tuple[str, ...]:
        return ()

    def initial_multipliers(self, initial_value: float) -> jax.Array:
        return jnp.array([initial_value], dtype=jnp.float32)

    def per_transition(self, multipliers: jax.Array, state_extras: StateExtras) -> jax.Array:
        del state_extras
        return multipliers  # shape [1]: one value for every transition

    def update(
        self, multipliers: jax.Array, costs: jax.Array, state_extras: StateExtras, budget_per_step: float, rate: float
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        del state_extras
        updated = jax.nn.relu(multipliers + rate * (jnp.mean(costs) - budget_per_step))
        return updated, {"lambda_lagr": updated}

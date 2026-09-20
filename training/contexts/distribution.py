"""The context-distribution interface.

A :class:`ContextDistribution` is the object that decides which contexts the
student trains on. At training round ``k`` it is fully described by a pytree of
arrays, its *parameters* ``φ_k`` (the thesis' Φ_{m,k}); the distribution class
itself is stateless.

Two halves, matching the two places code can run (see
``docs/acl/design/training_round.md``):

* :meth:`sample` is **JAX** and runs *inside* the compiled training step,
  every time a slot's episode ends. It may only use ``params`` and a PRNG key.
  ``params`` are frozen for the whole round.
* :meth:`update` runs on the **host between rounds**, with the completed
  episodes of the round as feedback. It may do anything (Python, NumPy, call
  the policy) and returns the parameters for the next round.

Uniform sampling is the degenerate case: ``update`` returns ``params`` unchanged.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Mapping, Protocol, TypeVar, runtime_checkable

import jax
import jax.numpy as jnp

from training.contexts.space import ContextSpace, Contexts

Params = Any  # a pytree of jax arrays; checkpointed as part of the run state
P = TypeVar("P")


@dataclasses.dataclass(frozen=True)
class EpisodeFeedback:
    """Outcomes of the episodes that *completed* during one training round.

    All arrays have a leading axis of length ``num_completed``. This is what a
    distribution learns from; it deliberately contains episodic aggregates only.
    """

    contexts: Contexts  # [N, D] the context each episode was run in
    returns: jax.Array  # [N]    undiscounted episodic return R(τ)
    costs: jax.Array  # [N]      undiscounted episodic cost C(τ)
    lengths: jax.Array  # [N]    steps until termination or truncation
    round_index: int

    @property
    def num_completed(self) -> int:
        return int(self.contexts.shape[0])

    def safe(self, cost_threshold: float) -> jax.Array:
        """Boolean ``[N]``: episode satisfied the cost budget."""
        return self.costs <= cost_threshold


@runtime_checkable
class ContextDistribution(Protocol):
    """Interface every curriculum / task-distribution method implements."""

    space: ContextSpace

    def initialise(self, key: jax.Array) -> Params:
        """Parameters φ_0 before any training."""

    def sample(self, params: Params, key: jax.Array, n: int) -> Contexts:
        """Draw ``n`` contexts. JAX; called inside the compiled training step."""

    def update(self, params: Params, feedback: EpisodeFeedback) -> Params:
        """Host-side update from the round's completed episodes -> φ_{k+1}."""

    def log_probability(self, params: Params, contexts: Contexts) -> jax.Array:
        """log q(ω) for each row of ``contexts``, shape ``[N]``.

        Used for importance weights and for comparing the intended distribution
        with the realised one. May return ``-inf`` for contexts outside support.
        """

    def summary(self, params: Params) -> Dict[str, float]:
        """Low-dimensional scalars describing φ, for logging each round."""


def batched_log_probability_uniform(space: ContextSpace, contexts: Contexts) -> jax.Array:
    """log-density of Uniform(Ω) at each context: constant inside, -inf outside."""
    widths = space.high - space.low
    # Degenerate (fixed) dimensions contribute no volume.
    log_volume = jnp.sum(jnp.where(widths > 0, jnp.log(jnp.where(widths > 0, widths, 1.0)), 0.0))
    inside = space.contains(contexts)
    return jnp.where(inside, -log_volume, -jnp.inf)


def as_dict(params: Params) -> Mapping[str, Any]:
    """Best-effort flatten of a params pytree for logging."""
    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    return {"/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path): leaf for path, leaf in leaves}

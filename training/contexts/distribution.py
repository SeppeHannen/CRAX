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
from typing import Any, Dict, Protocol, Union, runtime_checkable

import jax
import numpy as np

from training.contexts.space import ContextSpace, Contexts

Params = Any  # a pytree of jax arrays; checkpointed as part of the run state
HostArray = Union[np.ndarray, jax.Array]  # feedback lives on the host; NumPy in practice


@dataclasses.dataclass(frozen=True)
class EpisodeFeedback:
    """Outcomes of the episodes that *completed* during one training round.

    All arrays have a leading axis of length ``num_completed``. This is what a
    distribution learns from; it deliberately contains episodic aggregates only.
    Built on the host by :func:`training.contexts.rollout.completed_episodes`.
    """

    contexts: HostArray  # [N, D] the context each episode was run in
    returns: HostArray  # [N]    undiscounted episodic return R(τ)
    costs: HostArray  # [N]      undiscounted episodic cost C(τ)
    lengths: HostArray  # [N]    steps until termination or truncation
    round_index: int

    @property
    def num_completed(self) -> int:
        return int(self.contexts.shape[0])

    def safe(self, cost_threshold: float) -> HostArray:
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

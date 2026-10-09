"""The context-distribution interface.

A :class:`ContextDistribution` is the object that decides which contexts the
student trains on. At training round ``k`` it is fully described by a pytree of
arrays, its *parameters* ``φ_k`` (the thesis' Φ_{m,k}); the distribution class
itself is stateless.

Two halves, matching the two places code can run (see
``docs/acl/design/training_round.md``):

* :meth:`sample` is **JAX** and runs *inside* the compiled training step,
  every time a slot's episode ends. It may only use ``parameters`` and a PRNG key.
  ``parameters`` are frozen for the whole round.
* :meth:`update` runs on the **host between rounds**, with the round's
  transitions as feedback (:class:`~training.contexts.rollout.RoundRollout`:
  every transition's context, the episode bookkeeping and the learner's reward
  advantage). It may do anything (Python, NumPy) and returns the parameters for
  the next round.

Uniform sampling is the degenerate case: ``update`` returns ``parameters`` unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

import jax

from training.contexts.rollout import RoundRollout
from training.contexts.space import ContextSpace, Contexts

DistributionParameters = Any
"""φ: everything a distribution needs to ``sample`` from, as a pytree of JAX arrays.

The wrapper carries φ inside the environment state between rounds; the host
replaces it after every ``update``. Three things follow for how a method lays
φ out:

* **Shapes are fixed for the whole run.** The training round is compiled once;
  a φ whose shapes change forces a recompile (~50 s). A buffer is a fixed number
  of rows plus an ``occupied`` mask, never a growing array.
* **No slot axis.** φ is one value for the whole population, not one per
  environment; the wrapper hands it to ``sample`` once per step. Every other
  array in the state has the slot axis first, which is why the wrapper lifts φ
  out before the vmapped stack steps.
"""


@runtime_checkable
class ContextDistribution(Protocol):
    """Interface every curriculum / task-distribution method implements."""

    space: ContextSpace

    def initialise(self) -> DistributionParameters:
        """Parameters φ_0 before any training."""

    def sample(self, parameters: DistributionParameters, key: jax.Array, n: int) -> Contexts:
        """Draw ``n`` contexts. JAX; called inside the compiled training step."""

    def update(self, parameters: DistributionParameters, rollout: RoundRollout) -> DistributionParameters:
        """Host-side update from the round's transitions -> φ_{k+1}."""

    def summary(self, parameters: DistributionParameters) -> Dict[str, float]:
        """Low-dimensional scalars describing φ, for logging each round
        (``training_curriculum/distribution/<key>``). Empty when φ has nothing
        to say beyond what sampling it shows."""

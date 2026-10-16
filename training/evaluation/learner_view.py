"""What the agent can say about an episode it did not train on: the values its
critics assign to every state, the multiplier it applies, and the advantages its
loss would compute.

The agent that trains and the recorder that keeps evaluation episodes are
independent pieces; what they agree on is :class:`LearnerView`. The agent hands
the trainer a :class:`LearnerViewFunction`, the way it hands it a
``LearningSignalsFunction`` for the round hook (``training/rounds.py``), and the
trainer applies it to each recorded evaluation unroll. Only the agent knows its
critics, where its multiplier lives and how its loss forms an advantage; the
recorder knows only this struct.

Everything here is the loss's own arithmetic, applied to a batch of whole
evaluation episodes instead of a training batch. One difference is stated: the
loss standardises advantages over a training minibatch before combining them
with λ; an evaluation episode has no minibatch, so the view holds them raw.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import flax.struct
import jax


@flax.struct.dataclass
class LearnerView:
    """Per-step quantities for a batch of episodes, every array ``[B, T]`` in the data's layout.

    Values and advantages are in the loss's units: the reward critic is trained on
    ``reward × reward_scaling`` and the cost critic on the raw cost.
    """

    value: jax.Array  # V(s_t): the reward critic's estimate
    cost_value: jax.Array  # V_c(s_t): the cost critic's estimate
    lagrange_multiplier: jax.Array  # λ applied at s_t (constant along an episode under a scalar λ)
    reward_advantage: jax.Array  # GAE of the scaled reward under the reward critic
    cost_advantage: jax.Array  # GAE of the cost under the cost critic
    lagrangian_advantage: jax.Array  # reward_advantage − λ · cost_advantage, unstandardised


@dataclasses.dataclass(frozen=True)
class LearnerViewFunction:
    """``compute(params, normalizer_params, data, aux_state, *, ppo_network, discounting,
    reward_scaling, gae_lambda)`` returns the :class:`LearnerView` of ``data`` (a
    ``Transition`` with leading dimensions ``[B, T]``) under the parameters given."""

    compute: Callable[..., LearnerView]

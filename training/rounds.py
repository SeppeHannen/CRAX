"""Training rounds and the hook that acts between them.

The problem this solves. A context distribution has two halves that run in two
places: it *samples* contexts inside the compiled training step, from parameters
φ that are frozen for the whole call, and it *learns* from the episodes that
completed — on the host, between calls (``docs/acl/design/training_round.md``).
For the host half to see every round's data and to hand back φ for the next
round, one compiled call must be exactly one training step (a **round**), and
the trainer must (a) ship the round's transitions to the host and (b) let
something edit the environment state before the next call.

:class:`RoundHook` is that something. The trainer knows nothing about contexts;
``training.contexts.ContextRoundHook`` is the one implementation today, and any
later curriculum method plugs in the same way.

Next to the recorded state fields the hook receives the round's **learning
signals**: per-transition quantities the learner derived from the rollout
before updating (``docs/acl/design/learning_signals.md``). The agent that
trains and the hook that teaches are independent plug-ins; what they agree on
is :class:`Signal`, the vocabulary of signals. The agent declares which it
*produces* (:class:`LearningSignalsFunction`), the hook declares which it
*requires* (:attr:`RoundHook.required_signals`), and the trainer checks the one
against the other before anything compiles.
"""
from __future__ import annotations

import dataclasses
import enum
from typing import Any, Callable, Dict, FrozenSet, Mapping, Protocol, Tuple, runtime_checkable

import jax
import numpy as np

from crax.envs.base import State


class Signal(str, enum.Enum):
    """A per-transition scalar the learner derives from its critics.

    Computed under the parameters a round was collected with, before the round's
    gradient steps, with the same layout as the recorded state fields: leading
    axes ``[..., unroll_length]`` over all transitions of the round. The
    ``value`` is the short name used in CLI specs (``plr:cost``) and metric keys
    (``training_curriculum/cost_value_loss/<d>``). ``str`` mixin so that a
    ``Mapping[Signal, …]`` is an ordinary JAX pytree.
    """

    REWARD_ADVANTAGE = "reward"
    """GAE of the (scaled) reward under the reward critic, as the PPO loss estimates it."""

    COST_ADVANTAGE = "cost"
    """GAE of the cost under the cost critic, as the constrained losses estimate it."""

    LAGRANGIAN_ADVANTAGE = "lagrangian"
    """The advantage the policy loss weights each transition by: the reward and cost
    advantages combined with the λ in force during the round, by the loss's own rule
    (standardisation included) — ``A_r − λ A_c``. In standardised units."""


LearningSignals = Mapping[Signal, jax.Array]
"""One array per produced :class:`Signal`, all of the same shape."""


@dataclasses.dataclass(frozen=True)
class LearningSignalsFunction:
    """What an agent hands the trainer so a round hook can see the learner's view of a round.

    ``compute(params, normalizer_params, data, aux_state, *, ppo_network,
    discounting, reward_scaling, gae_lambda, normalize_advantage)`` returns the
    signals in ``produces``, nothing more or less; the trainer checks the keys
    when it traces the round. ``produces`` is what the trainer checks a hook's
    :attr:`RoundHook.required_signals` against before compiling anything.
    """

    produces: FrozenSet[Signal]
    compute: Callable[..., LearningSignals]


@runtime_checkable
class RoundHook(Protocol):
    """Sees every round's rollout; may edit the environment state between rounds."""

    @property
    def extra_fields(self) -> Tuple[str, ...]:
        """``state.info`` keys the trainer must record for every transition."""

    @property
    def required_signals(self) -> FrozenSet[Signal]:
        """The signals this hook reads. The trainer refuses an agent that does not produce them."""

    def observe(self, rollout: Mapping[str, np.ndarray], learning_signals: Mapping[Signal, np.ndarray]) -> None:
        """Receive the round's transitions, on the host, while the round runs.

        ``rollout`` has one NumPy array per name in :attr:`extra_fields`, with
        leading axes ``[..., unroll_length]`` over all transitions of the round;
        ``learning_signals`` has the same layout, one array per signal the agent
        produces. Called from a ``jax.debug.callback``; store, do not compute here.
        """

    def on_round_end(self, round_index: int, env_state: State) -> Tuple[State, Dict[str, Any]]:
        """Host side, after round ``round_index`` finished.

        Return the environment state for the next round (same pytree structure
        and shapes, or the next call recompiles) and metrics to log (scalars or
        W&B media such as ``wandb.Histogram``).
        """

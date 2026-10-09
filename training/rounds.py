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

Next to the recorded state fields the hook receives the :class:`LearningSignals`
of the round: what the learner computed from the rollout before updating. A
curriculum that prioritises by learning potential (PLR's L1 value loss,
``docs/acl/design/prioritized_level_replay.md``) reads them; one that does not,
ignores them.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Protocol, Tuple, runtime_checkable

import flax
import jax
import numpy as np

from crax.envs.base import State


@flax.struct.dataclass
class LearningSignals:
    """Per-transition quantities the learner derives from the round's rollout.

    Same layout as the recorded fields: leading axes ``[..., unroll_length]``
    over all transitions of the round. Computed under the parameters the round
    was collected with, before the round's gradient steps.
    """

    reward_advantage: jax.Array  # GAE of the reward, as the PPO loss estimates it


@runtime_checkable
class RoundHook(Protocol):
    """Sees every round's rollout; may edit the environment state between rounds."""

    @property
    def extra_fields(self) -> Tuple[str, ...]:
        """``state.info`` keys the trainer must record for every transition."""

    def observe(self, rollout: Mapping[str, np.ndarray], learning_signals: LearningSignals) -> None:
        """Receive the round's transitions, on the host, while the round runs.

        ``rollout`` has one NumPy array per name in :attr:`extra_fields`, with
        leading axes ``[..., unroll_length]`` over all transitions of the round;
        ``learning_signals`` has the same layout. Called from a
        ``jax.debug.callback``; store, do not compute here.
        """

    def on_round_end(self, round_index: int, env_state: State) -> Tuple[State, Dict[str, Any]]:
        """Host side, after round ``round_index`` finished.

        Return the environment state for the next round (same pytree structure
        and shapes, or the next call recompiles) and metrics to log (scalars or
        W&B media such as ``wandb.Histogram``).
        """

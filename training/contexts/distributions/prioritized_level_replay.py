"""Prioritized Level Replay (Jiang, Grefenstette, Rocktäschel 2021) over a
continuous context space Ω.

The design, and every departure from the paper, is in
``docs/acl/design/prioritized_level_replay.md``. In one paragraph: a buffer of
``buffer_size`` contexts, each an exact point of Ω that was once drawn from
Uniform(Ω), carries a *score* (the mean |reward advantage| over the last
complete episode run in that context — PLR's L1 value loss) and the round that
score was set in. Each new episode replays a buffer row with probability
``replay_probability × occupancy``, drawn by rank of score mixed with
staleness, and otherwise draws a fresh context from Uniform(Ω). After each
round every context whose episode ended is re-scored; new contexts enter the
buffer while there is room, and afterwards only by displacing the row with the
least replay mass, if they out-score it.

The two halves of :class:`~training.contexts.distribution.ContextDistribution`:
``sample`` is a table lookup over the parameters (JAX, in-program); ``update``
is NumPy on the host, once per round, and precomputes the replay weights so
that ``sample`` has nothing to derive.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from training.contexts.rollout import CompletedEpisodes, RoundRollout
from training.contexts.space import Contexts, ContextSpace


@struct.dataclass
class PrioritizedLevelReplayParameters:
    """φ: the buffer, and the replay distribution derived from it for the coming round."""

    round_index: jax.Array  # scalar int32: the round these parameters are in force
    contexts: jax.Array  # [M, D] float32: buffer rows (exact points of Ω); rows of zeros where unoccupied
    scores: jax.Array  # [M] float32: mean |reward advantage| when the row was last trained on; 0 where unoccupied
    last_round: jax.Array  # [M] int32: the round the score was set in; -1 where unoccupied
    occupied: jax.Array  # [M] bool
    replay_weights: jax.Array  # [M] float32: P_replay over rows for round `round_index` (0 where unoccupied)
    replay_probability: jax.Array  # scalar float32: P(replay a row) for a new episode; 0 while the buffer is empty, else the method's p


@dataclasses.dataclass(frozen=True)
class PrioritizedLevelReplay:
    """PLR as a context distribution.

    ``replay_probability`` is the paper's P_D, a constant as in its Appendix B.3
    (0 only while the buffer is empty, i.e. in round 0, when there is nothing to
    replay). ``temperature`` is β: P_S ∝ rank(score)^(-1/β), β = 1 is Zipf over ranks
    (the paper's 0.1 puts all mass on the top rank, which is a point mass when
    hundreds of episodes start from one frozen φ). ``staleness_coefficient`` is
    ρ: the weight of the staleness distribution in P_replay.
    """

    space: ContextSpace
    buffer_size: int = 1000
    replay_probability: float = 0.5
    temperature: float = 1.0
    staleness_coefficient: float = 0.3

    def __post_init__(self) -> None:
        if self.buffer_size < 1:
            raise ValueError(f"buffer_size must be ≥ 1, got {self.buffer_size}")
        if not 0.0 <= self.replay_probability <= 1.0:
            raise ValueError(f"replay_probability must be in [0, 1], got {self.replay_probability}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if not 0.0 <= self.staleness_coefficient <= 1.0:
            raise ValueError(f"staleness_coefficient must be in [0, 1], got {self.staleness_coefficient}")

    # ---- JAX half ---------------------------------------------------------- #

    def initialise(self) -> PrioritizedLevelReplayParameters:
        size = self.buffer_size
        return PrioritizedLevelReplayParameters(
            round_index=jnp.asarray(0, jnp.int32),
            contexts=jnp.zeros((size, self.space.size), jnp.float32),
            scores=jnp.zeros((size,), jnp.float32),
            last_round=jnp.full((size,), -1, jnp.int32),
            occupied=jnp.zeros((size,), bool),
            replay_weights=jnp.zeros((size,), jnp.float32),
            replay_probability=jnp.asarray(0.0, jnp.float32),
        )

    def sample(self, parameters: PrioritizedLevelReplayParameters, key: jax.Array, n: int) -> Contexts:
        """Per slot: a buffer row (by ``replay_weights``) with probability
        ``replay_probability``, otherwise a fresh draw from Uniform(Ω). Replayed
        rows are copied exactly, so the host can match them to the buffer."""
        decision_key, row_key, fresh_key = jax.random.split(key, 3)
        fresh = self.space.sample_uniform(fresh_key, n)
        # log(0) = -inf gives unoccupied rows zero probability; with an empty buffer every
        # logit is -inf and categorical returns row 0, which the decision below never selects.
        rows = jax.random.categorical(row_key, jnp.log(parameters.replay_weights), shape=(n,))
        replay = jax.random.uniform(decision_key, (n,)) < parameters.replay_probability
        return jnp.where(replay[:, None], parameters.contexts[rows], fresh)

    # ---- host half --------------------------------------------------------- #

    def update(self, parameters: PrioritizedLevelReplayParameters, rollout: RoundRollout) -> PrioritizedLevelReplayParameters:
        buffer = _Buffer.from_params(self, parameters)
        next_round = rollout.round_index + 1

        round_contexts, round_scores = _score_per_context(rollout.completed_episodes)
        in_buffer = buffer.rows_of(round_contexts)  # -1 where the context is not a buffer row
        seen = in_buffer >= 0
        buffer.set_scores(in_buffer[seen], round_scores[seen], rollout.round_index)

        new_order = np.argsort(-round_scores[~seen], kind="stable")
        for context, score in zip(round_contexts[~seen][new_order], round_scores[~seen][new_order]):
            if not buffer.admit(context, score, rollout.round_index, next_round):
                break  # every later candidate scores lower against an unchanged buffer

        return buffer.to_parameters(next_round)

    def summary(self, parameters: PrioritizedLevelReplayParameters) -> Dict[str, float]:
        occupied = np.asarray(parameters.occupied)
        scores = np.asarray(parameters.scores)[occupied]
        staleness = int(parameters.round_index) - np.asarray(parameters.last_round)[occupied]
        replay_weights = np.sort(np.asarray(parameters.replay_weights, dtype=np.float64))[::-1]
        return {
            "replay_probability": float(parameters.replay_probability),
            "buffer_occupancy": float(occupied.mean()),
            "score/mean": float(scores.mean()) if scores.size else float("nan"),
            "score/max": float(scores.max()) if scores.size else float("nan"),
            "staleness/mean": float(staleness.mean()) if staleness.size else float("nan"),
            # How concentrated P_replay is: the share of replay mass on the most-replayed rows.
            # Marginals over one dimension of Ω cannot show this (1000 points in a 5-D Ω).
            "replay_mass/top_10": float(replay_weights[:10].sum()),
            "replay_mass/top_100": float(replay_weights[:100].sum()),
        }

    # ---- the replay distribution ------------------------------------------- #

    def replay_weights(self, scores: np.ndarray, last_round: np.ndarray, occupied: np.ndarray, round_index: int) -> np.ndarray:
        """P_replay = (1 − ρ) P_S + ρ P_C over the buffer rows, for a round ``round_index``.

        P_S ∝ rank(score)^(-1/β) (rank 1 = highest score, ties by row order);
        P_C ∝ how many rounds old the score will be in round ``round_index``
        (≥ 1, so P_C is a distribution even when every row was just refreshed).
        Zero where unoccupied; all zeros for an empty buffer.
        """
        weights = np.zeros(scores.shape[0], dtype=np.float64)
        rows = np.flatnonzero(occupied)
        if rows.size == 0:
            return weights
        order = np.argsort(-scores[rows], kind="stable")
        rank = np.empty(rows.size, dtype=np.float64)
        rank[order] = np.arange(1, rows.size + 1)
        by_score = rank ** (-1.0 / self.temperature)
        by_staleness = (round_index - last_round[rows]).astype(np.float64)
        if np.any(by_staleness < 1):
            raise ValueError("a buffer row's score is dated in the future: last_round ≥ round_index")
        weights[rows] = (1.0 - self.staleness_coefficient) * by_score / by_score.sum() + self.staleness_coefficient * by_staleness / by_staleness.sum()
        return weights

    def effective_replay_probability(self, occupied: np.ndarray) -> float:
        """The method's p, or 0 when there is nothing to replay."""
        return self.replay_probability if occupied.any() else 0.0


def _score_per_context(episodes: CompletedEpisodes) -> Tuple[np.ndarray, np.ndarray]:
    """The distinct contexts whose episodes ended this round, ``[K, D]``, and each one's
    score: its episodes' L1 value loss (``CompletedEpisodes.value_loss``), averaged when
    several episodes of the same context ended in the round."""
    keys = _row_keys(episodes.contexts)
    unique_keys, first_index, inverse = np.unique(keys, return_index=True, return_inverse=True)
    total = np.bincount(inverse, weights=episodes.value_loss.astype(np.float64), minlength=unique_keys.size)
    count = np.bincount(inverse, minlength=unique_keys.size)
    return episodes.contexts[first_index], total / count


def _row_keys(contexts: np.ndarray) -> np.ndarray:
    """One opaque key per row so rows can be grouped and matched bit for bit."""
    contiguous = np.ascontiguousarray(contexts, dtype=np.float32)
    return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).ravel()


class _Buffer:
    """The mutable host-side copy of φ that ``update`` edits; ``to_parameters`` freezes it."""

    def __init__(self, distribution: PrioritizedLevelReplay, contexts: np.ndarray, scores: np.ndarray,
                 last_round: np.ndarray, occupied: np.ndarray):
        self.distribution = distribution
        self.contexts = contexts
        self.scores = scores
        self.last_round = last_round
        self.occupied = occupied
        self._row_by_key = {key.tobytes(): int(row) for row, key in zip(np.flatnonzero(occupied), _row_keys(contexts[occupied]))}
        if len(self._row_by_key) != int(occupied.sum()):
            raise ValueError("the PLR buffer holds the same context in two rows")

    @classmethod
    def from_params(cls, distribution: PrioritizedLevelReplay, parameters: PrioritizedLevelReplayParameters) -> "_Buffer":
        return cls(
            distribution=distribution,
            contexts=np.array(parameters.contexts, dtype=np.float32),
            scores=np.array(parameters.scores, dtype=np.float64),
            last_round=np.array(parameters.last_round, dtype=np.int64),
            occupied=np.array(parameters.occupied, dtype=bool),
        )

    def rows_of(self, contexts: np.ndarray) -> np.ndarray:
        """Buffer row of each context, −1 where it is not in the buffer."""
        return np.asarray([self._row_by_key.get(key.tobytes(), -1) for key in _row_keys(contexts)], dtype=np.int64)

    def set_scores(self, rows: np.ndarray, scores: np.ndarray, round_index: int) -> None:
        self.scores[rows] = scores
        self.last_round[rows] = round_index

    def admit(self, context: np.ndarray, score: float, round_index: int, next_round: int) -> bool:
        """Put a new context in the buffer: into an empty row, or in place of the row
        with the least replay mass if the candidate out-scores it. False if neither."""
        empty = np.flatnonzero(~self.occupied)
        if empty.size:
            row = int(empty[0])
        else:
            weights = self.distribution.replay_weights(self.scores, self.last_round, self.occupied, next_round)
            row = int(np.argmin(weights))
            if self.scores[row] >= score:
                return False
            del self._row_by_key[_row_keys(self.contexts[row][None])[0].tobytes()]
        self.contexts[row] = context
        self.scores[row] = score
        self.last_round[row] = round_index
        self.occupied[row] = True
        self._row_by_key[_row_keys(context[None])[0].tobytes()] = row
        return True

    def to_parameters(self, next_round: int) -> PrioritizedLevelReplayParameters:
        weights = self.distribution.replay_weights(self.scores, self.last_round, self.occupied, next_round)
        return PrioritizedLevelReplayParameters(
            round_index=jnp.asarray(next_round, jnp.int32),
            contexts=jnp.asarray(self.contexts, jnp.float32),
            scores=jnp.asarray(self.scores, jnp.float32),
            last_round=jnp.asarray(self.last_round, jnp.int32),
            occupied=jnp.asarray(self.occupied),
            replay_weights=jnp.asarray(weights, jnp.float32),
            replay_probability=jnp.asarray(self.distribution.effective_replay_probability(self.occupied), jnp.float32),
        )

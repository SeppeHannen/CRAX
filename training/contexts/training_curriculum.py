"""The ``training_curriculum/*`` metrics: which contexts the student trained on
in one round, as *intended*, *sampled* and *experienced*.

``docs/acl/design/intended_vs_realised_curriculum.md`` distinguishes three
distributions over contexts for a round and asks for all of them side by side:

* **intended** — q as the distribution states it (its ``summary(φ_k)``: the
  stage and context of a staged curriculum; nothing for uniform).
* **sampled** — one count per episode that started (observed as the episodes
  that *completed* in the round): the empirical q.
* **experienced** — one count per *transition*. A context whose episodes last
  longer contributes more transitions, hence more gradient signal: q̂, the
  realised curriculum.

The gap between sampled and experienced is explained by the mean episode
length per bin, reported alongside. Everything here is NumPy on the host, once
per round, over fixed bins of Ω so values compare across rounds, arms and
seeds. This module is the only producer of keys under :data:`SECTION`; what
each key means for a reviewer is in ``training/dashboard/metrics.py``.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import wandb

from training.contexts.distribution import ContextDistribution, Params
from training.contexts.rollout import RoundRollout
from training.contexts.space import ContextSpace

# The W&B section. It describes the *training* rollouts, so it sits next to
# `episodic/*` (also training) and apart from `evaluation/*`.
SECTION = "training_curriculum/"

# Display resolution: Ω is cut into this many equal bins per dimension.
NUM_BINS = 12


def training_curriculum_metrics(
    distribution: ContextDistribution, parameters: Params, rollout: RoundRollout
) -> Dict[str, Any]:
    """All ``training_curriculum/*`` metrics of one round. Per context dimension ``d``:

    ``intended/<key>``                             ``distribution.summary(parameters)``, e.g. ``intended/stage``
    ``sampled/<d>``, ``experienced/<d>``           wandb.Histogram over Ω; W&B renders the sequence
                                                   over rounds as a heatmap
    ``sampled/<d>/mean``, ``experienced/<d>/mean``  the one-line summaries that overlay across arms
    ``sampled/<d>/std``, ``experienced/<d>/std``
    ``episode_length/<d>``                         wandb.Histogram: mean length of the completed
                                                   episodes per bin — why sampled ≠ experienced
    ``num_transitions``, ``num_completed_episodes`` the sample sizes behind the two histograms
    """
    space = distribution.space
    done = rollout.episode_done
    metrics: Dict[str, Any] = {
        "num_transitions": float(rollout.num_transitions),
        "num_completed_episodes": float(rollout.num_completed_episodes),
    }
    for key, value in distribution.summary(parameters).items():
        metrics[f"intended/{key}"] = value
    for dimension in space.names:
        experienced = rollout.contexts[:, space.index(dimension)]  # one value per transition
        sampled = experienced[done]  # one value per completed episode
        edges = _bin_edges(space, dimension)
        for name, values in (("sampled", sampled), ("experienced", experienced)):
            metrics[f"{name}/{dimension}"] = wandb.Histogram(np_histogram=(_mass_per_bin(values, edges), edges))
            metrics[f"{name}/{dimension}/mean"] = float(values.mean()) if values.size else float("nan")
            metrics[f"{name}/{dimension}/std"] = float(values.std()) if values.size else float("nan")
        metrics[f"episode_length/{dimension}"] = wandb.Histogram(
            np_histogram=(_mean_episode_length_per_bin(sampled, rollout.episode_length[done], edges), edges)
        )
    return {SECTION + key: value for key, value in metrics.items()}


def _bin_edges(space: ContextSpace, dimension: str) -> np.ndarray:
    index = space.index(dimension)
    low, high = float(space.low[index]), float(space.high[index])
    if high <= low:  # degenerate dimension: one bin around the value
        return np.asarray([low - 0.5, low + 0.5])
    return np.linspace(low, high, NUM_BINS + 1)


def _bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin of each value; values outside Ω fall into the outermost bins."""
    clipped = np.clip(values, edges[0], np.nextafter(edges[-1], -np.inf))
    return np.digitize(clipped, edges) - 1


def _mass_per_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Fraction of ``values`` in each bin (all zeros when there are none)."""
    counts = np.bincount(_bin_index(values, edges), minlength=len(edges) - 1).astype(float)
    return counts / counts.sum() if counts.sum() > 0 else counts


def _mean_episode_length_per_bin(contexts: np.ndarray, lengths: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean of ``lengths`` per bin of ``contexts``; 0 where no episode ended in the bin."""
    bins = _bin_index(contexts, edges)
    num_bins = len(edges) - 1
    total = np.bincount(bins, weights=lengths, minlength=num_bins)
    count = np.bincount(bins, minlength=num_bins)
    return np.divide(total, count, out=np.zeros(num_bins), where=count > 0)

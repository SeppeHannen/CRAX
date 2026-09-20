"""What the student trained on in one round: the curriculum as *sampled* and as
*experienced*.

``docs/acl/design/intended_vs_realised_curriculum.md`` distinguishes two
empirical distributions over contexts for a round, and asks for both:

* **sampled** — one count per episode that started (we observe it as the
  episodes that *completed* in the round). This is what the distribution chose:
  the empirical q.
* **experienced** — one count per *transition*. A context whose episodes last
  longer contributes more transitions, hence more gradient signal. This is q̂,
  the realised curriculum.

Their difference is the exposure-vs-selection gap; the mean episode length per
bin explains it. Everything here is NumPy on the host, once per round, over
fixed bins of Ω so that values compare across rounds, arms and seeds.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import wandb

from training.contexts.rollout import RoundRollout
from training.contexts.space import ContextSpace

# Display resolution: Ω is cut into this many equal bins per dimension.
NUM_BINS = 12


def bin_edges(space: ContextSpace, dimension: str, num_bins: int = NUM_BINS) -> np.ndarray:
    index = space.index(dimension)
    low, high = float(space.low[index]), float(space.high[index])
    if high <= low:  # degenerate dimension: one bin around the value
        return np.asarray([low - 0.5, low + 0.5])
    return np.linspace(low, high, num_bins + 1)


def bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin of each value; values outside Ω fall into the outermost bins."""
    clipped = np.clip(values, edges[0], np.nextafter(edges[-1], -np.inf))
    return np.digitize(clipped, edges) - 1


def mass_per_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Fraction of ``values`` in each bin (all zeros when there are none)."""
    counts = np.bincount(bin_index(values, edges), minlength=len(edges) - 1).astype(float)
    return counts / counts.sum() if counts.sum() > 0 else counts


def curriculum_metrics(space: ContextSpace, rollout: RoundRollout) -> Dict[str, Any]:
    """W&B metrics for one round. Per context dimension ``d``:

    ``sampled/<d>``  and  ``experienced/<d>``      wandb.Histogram over Ω — logged every round,
                                                   W&B renders the sequence as a heatmap over time
    ``sampled/<d>/bin_i``, ``experienced/<d>/bin_i`` the same masses as scalars (comparable
                                                   across runs; groupable by seed)
    ``sampled/<d>/mean``, ``experienced/<d>/mean``  one-line summaries for arm-vs-arm panels
    ``episode_length/<d>/bin_i``                   mean length of completed episodes per bin —
                                                   the mechanism behind sampled ≠ experienced
    """
    done = rollout.episode_done
    metrics: Dict[str, Any] = {
        "num_transitions": float(rollout.num_transitions),
        "num_completed_episodes": float(rollout.num_completed_episodes),
    }
    for dimension in space.names:
        experienced = rollout.contexts[:, space.index(dimension)]  # one value per transition
        sampled = experienced[done]  # one value per completed episode
        edges = bin_edges(space, dimension)
        for name, values in (("sampled", sampled), ("experienced", experienced)):
            mass = mass_per_bin(values, edges)
            metrics[f"{name}/{dimension}"] = wandb.Histogram(np_histogram=(mass, edges))
            metrics[f"{name}/{dimension}/mean"] = float(values.mean()) if values.size else float("nan")
            metrics[f"{name}/{dimension}/std"] = float(values.std()) if values.size else float("nan")
            for i, share in enumerate(mass):
                metrics[f"{name}/{dimension}/bin_{i:02d}"] = float(share)

        episode_bins = bin_index(sampled, edges)
        episode_lengths = rollout.episode_length[done]
        for i in range(len(edges) - 1):
            in_bin = episode_bins == i
            metrics[f"episode_length/{dimension}/bin_{i:02d}"] = (
                float(episode_lengths[in_bin].mean()) if in_bin.any() else float("nan")
            )
    return metrics

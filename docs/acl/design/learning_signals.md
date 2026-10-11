# Learning signals: how a teacher gets per-episode numbers out of the learner

Decision (2026-10-11, Giuseppe's review of the PLR-score change). The agent
that trains and the distribution that teaches are independent plug-ins; what
they must agree on is small and should be written in one place and checked at
assembly time, never by convention. This document states that contract, the
one limitation it has, and what to do when the limitation is hit.

## Terms

- **Signal** — a per-transition scalar the learner derives from its critics
  under the parameters a round was collected with: today the reward advantage,
  the cost advantage, and the Lagrangian advantage $A_r - \lambda A_c$ as the
  policy loss combines them. Produced on the device, inside the compiled round.
- **Episode statistic** — one number per *completed episode*, computed from
  one signal over all of that episode's transitions: PLR's L1 value loss is
  the mean $|A|$ over the episode. Consumed on the host, by a distribution.
- **Fold** — how a statistic is accumulated: a per-transition *map*, an
  associative *combine* with its *identity* (together a monoid), and a
  *finalise* step (`mean |x|` = map $|\cdot|$, combine $+$ with identity 0 and
  a count, finalise sum/count).

## The problem

Episodes are ~1000 steps; a round contributes ~80 steps per slot. An episode
therefore spans about twelve rounds, and the transitions of earlier rounds are
**not stored** — the trainer ships each round's transitions to the host once,
the host reduces them, and they are gone. Any per-episode statistic has to be
built from a **carry**: an accumulator per slot, updated with each round's
transitions and closed when the slot's episode ends
(`training/contexts/episodes.py`).

The first version hard-coded one signal (reward advantage) and one fold
(mean $|\cdot|$) into that carry. Adding the cost and Lagrangian signals grew
it to a `[2, …]` then `[3, …]` axis read by position, and the fold stayed
welded into the engine. Each new signal touched the device struct, the host
struct, the tracker and the diagnostics; each new fold would have meant a
second engine.

## The contract

Three parties, one vocabulary, capabilities declared on both ends and checked
before anything compiles.

```
 agent                         trainer                      teacher (distribution)
 ─────                         ───────                      ──────────────────────
 produces: {Signal, …}   ──►   requires ⊆ produces ?  ◄──   required_statistics:
 compute(...) ->               raise at setup if not          {EpisodeStatistic(signal, fold), …}
   Mapping[Signal, array]        │
                                 ▼  every round
                         hook → EpisodeTracker(statistics) → CompletedEpisodes.statistics[s]
                                (knows no names; applies each statistic's fold over its signal)
```

- `training/rounds.py: Signal` — a `str` enum, one member per signal, each with
  a docstring saying its units and meaning; the `value` is the short name used
  in CLI specs (`plr:cost`) and metric keys (`cost_value_loss/<d>`). **The only
  place the names live.** `rounds.py` is the trainer ↔ hook contract, so this is
  where the trainer finds the vocabulary.
- `training/rounds.py: LearningSignalsFunction(produces, compute)` — what an
  agent hands the trainer: the set of signals it `produces` and the function
  that computes them as `Mapping[Signal, jax.Array]`.
- `training/contexts/statistics.py: Fold` — a Protocol (`combine`: an
  associative NumPy ufunc; `identity`: its neutral element; `map`; `finalise`),
  with the implementations next to it. `MeanAbsolute` is the one PLR needs.
  `EpisodeStatistic` checks at construction that the identity really is neutral
  for the combine. `EpisodeStatistic(signal, fold)` —
  frozen, hashable: what a distribution asks for, and the key it reads the
  answer under. This is the distribution ↔ tracker contract, so it lives in the
  contexts package, one module for the concept.
- **Agent side** (`ppo_lag/train.py`, `ppo_pid/train.py`): both build a
  `LearningSignalsFunction(produces=ppo_lag_losses.LEARNING_SIGNALS, compute=…)`
  next to their `loss_fn`. An agent declares honestly what its critics let it
  compute; plain PPO would produce `{REWARD_ADVANTAGE}` only.
- **Teacher side** (`training/contexts/distribution.py`): `ContextDistribution`
  has `required_statistics: FrozenSet[EpisodeStatistic]`. PLR:
  `{EpisodeStatistic(score, MeanAbsolute())}`. Uniform, fixed, staged: `∅`.
  The hook (`round_hook.py`) derives `required_signals` from it and sizes the
  tracker with it.
- **Assembly** (`ppo/train.py`, argument validation, before any key, network
  or environment exists): `round_hook.required_signals ⊆
  learning_signals_fn.produces`, otherwise raise naming both sets. `--alg ppo
  --context_distribution plr:cost` fails in under a second, not after a
  50-second compile inside a callback. At trace time the trainer also checks
  that `compute` returned exactly the declared keys.
- **Middle** (`rollout.py`, `episodes.py`, `training_curriculum.py`): keyed by
  `Signal` / `EpisodeStatistic`, loops over whatever arrives, knows no names.
  `Transitions.signals` holds every signal the agent produced; the
  `<signal>_value_loss/<d>` heatmaps are one per *produced* signal (mean
  |signal| per bin, a diagnostic definition stated in the metric's
  description), so the dashboards show all three on every constrained arm
  regardless of which teacher runs, and a reward-only agent would show one.

**Adding a teacher** that wants an existing statistic: write the class, set
`required_statistics`, read `episodes.statistics[s]`. Nothing else changes.
**Adding a fold** (Robust PLR's positive value loss, `mean max(x, 0)`): one
small class next to `MeanAbsolute`. **Adding a signal**: one enum member, the
agent that can compute it adds it to `produces`. **Adding an agent**: it
declares `produces`; teachers it cannot serve are refused at setup.

## How the engine reduces (`episodes.py`)

`EpisodeEnds.find(episode_done)` computes the round's geometry once: each
completed episode's *segment* of transitions and each slot's *tail* (what
follows its last done, or the whole round if none). `reduce(per_transition,
combine, identity, carry)` then reduces every segment with `combine.reduceat`,
joins the carry onto each slot's first episode, and returns the next carry —
the tail, or the identity where the tail is empty. Because every carry starts
at the identity, there is no "is there a carry" case: the identity is what
makes the join unconditional. The tracker calls it once per statistic with
`fold.map(signal)`, `fold.combine`, `fold.identity`, and once with ones,
`np.add`, 0 for the transition count that `finalise` receives. Verified against
a brute-force per-episode reduction on 10 300 random episodes for `np.add`,
`np.maximum`, `np.minimum` and `np.multiply` (2026-10-11); a unit test runs a
maximum fold through the tracker across rounds.

## The limitation, stated

**A statistic must be expressible as a fold over a monoid** — an associative
combine with an identity — so that the answer does not depend on how the
episode's transitions happen to be split across rounds. Mean, sum, min, max, count, variance via moments, any
exponential average, quantile *sketches*: yes. Exact median, exact quantiles,
anything that ranks the raw sequence, anything that needs two passes: **no**.

This is forced by the architecture (episodes outlast rounds, transitions are
not buffered), not by this design. The design makes it visible — a non-fold
statistic cannot be written against the `Fold` protocol — rather than hiding
it.

## What to do when the limitation bites

If a method needs a non-associative per-episode statistic, do not bolt a
special case onto the tracker. The honest fix is to **buffer each slot's
in-progress episode on the host** and compute every statistic from the full
sequence at episode end. Size: `num_envs × episode_length × num_signals × 4 B`
= 8192 × 1000 × 3 × 4 ≈ 100 MB at the operating point, host RAM, trivial. Cost:
the carry arithmetic in `episodes.py` disappears (the buffer *is* the carry),
and every fold becomes an ordinary NumPy reduction over a `[length]` array —
the code gets *simpler*, not harder. The reason not to do it today is only that
nothing needs it; when something does, switch everything, not one statistic.

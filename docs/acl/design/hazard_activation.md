# Per-episode hazard activation: how a hazard *count* becomes a value

The four hazard suites (goal, button, reach, circle) vary difficulty by
*how many* hazards exist, of *which kinds*. A count is the size of MuJoCo
arrays and therefore a compiled-program property, not a value
(`what_can_vary_per_slot.md` §3). This document is the one mechanism that
turns it into a value, written before the code so the code can be checked
against it.

## The idea in one paragraph

Build the environment once with every hazard any context could need — the
**union model**, one MuJoCo body per hazard, the maximum count of every
(type, collidable) group. Give each episode a 0/1 **activation** per hazard,
read from the context like every other knob (`crax/envs/context.py`). An
inactive hazard is *parked* far outside the arena at reset and *ignored* by
everything that computes with hazards: it costs nothing, shows on no lidar,
is no compass target, blocks no placement. Every slot runs the same program;
the activations are data.

## The invariant

> **A hazard whose activation is 0 has no effect on the episode.** Not on the
> cost, not on the observation, not on where anything else is placed, not on
> the physics.

Its test (`tests/test_suites.py`, per hazard suite): an environment built at
level 1 and reset into the level-3 context is step-for-step identical to an
environment built at level 3 — the existing invariant test, unchanged. For a
hazard suite this only holds if the parked hazards are invisible; so the test
we already have is the test for this document.

## Where a hazard is *seen* — the four consumers

An audit (2026-09-22, `context_spaces_by_suite.md` "Plan for the full
benchmark") found exactly four places that look at hazards. Each gets the
activation vector and treats an inactive hazard as absent:

| consumer | today | with activation `a[H]` |
|---|---|---|
| **placement** (`env_utils.place_objects`, reset) | every hazard is placed, respecting keepouts of everything already placed | inactive: the scan iteration still runs (fixed length) but writes the parking position and a zero keepout, so later objects ignore it |
| **cost** (`hazards.compute_hazard_costs`, step) | Python loop over hazard objects, each adds its cost | each term × `a[i]` |
| **lidar** (`_get_obs`, step; goal/circle read `mocap_pos`, button/reach/pathway read `info["hazard_positions"]`) | nearest reading per ray over all hazards | reading × `a[i]` before the per-ray max — parking alone is not enough (a far hazard still gives ε) |
| **compass / closest-k** (`_get_obs`, step) | `argsort` of distances, take k | distance += ∞ where `a[i] == 0` before the sort; the existing `-1` sentinel handles a padded id |

Physics needs nothing: a parked collidable hazard is a mocap body 1 km away
that touches nothing. It still owns its contact slot in the solver — the price
of the union, measured in `what_can_vary_per_slot.md` §5, paid once per
collidable *group*, not per hazard.

## Parking

`PARKING_POSITION = (1000.0, 1000.0, hazard_height)`, written into `mocap_pos`
for inactive hazards at reset. Far enough that no lidar (`max_dist` 3 m), no
proximity cost (radius ≤ 0.4 m) and no contact can reach it; near enough that
float32 keeps millimetre precision. Walls (`fixed=True`) are never parked —
they are not part of Ω.

## The union model for goal

From `crax/envs/difficulty.py`, the `(type, collidable)` groups across levels
1–3 and their maximum counts:

| group | L1 | L2 | L3 | union |
|---|---|---|---|---|
| cylinder, non-collidable | 12 (size 0.4) | 8 (0.4) | 6 (0.35) | **12** |
| cylinder, collidable | — | 8 (0.3, h 0.4) | 4 (0.25, h 0.4) | **8** |
| cube, non-collidable | — | — | 6 (0.3) | **6** |
| cube, collidable | — | — | 4 (0.25, h 0.5) | **4** |

30 hazards plus the four walls. **Size and height within a group differ
between levels** (cylinder 0.4 vs 0.35; collidable cylinder 0.3 vs 0.25).
Both are JAX fields (`geom_size`, `body_pos` z) and could be per slot; for now
the union uses **one size per group** (the larger, so level 3's hazards are
slightly bigger than the paper's). Flagged to Tristan; a `size` dimension per
group is a later addition if the difference matters.

Ω for goal: one integer dimension per group (`active_cylinders`,
`active_collidable_cylinders`, `active_cubes`, `active_collidable_cubes`) and
`goal_size`. A count `n` for a group activates its first `n` hazards — which
`n` is a value, and since positions are random the identity of the hazard
does not matter. Levels: L1 = (12, 0, 0, 0, 0.20), L2 = (8, 8, 0, 0, 0.18),
L3 = (6, 4, 6, 4, 0.16).

## Where the code goes

- `crax/envs/hazards.py` — `HazardManager` gains the group structure: hazards
  are added in named groups; `activation_from_counts(counts)` maps per-group
  counts `[G]` to per-hazard `a[H]` (fixed hazards always 1). `compute_hazard_costs`
  takes `activation`.
- `crax/envs/env_utils.py` — `place_objects` takes `activation` (parking +
  zero keepout for inactive).
- `crax/envs/safe_goal.py` — reads counts and `goal_size` from the context;
  lidar and compass masked; `goal_size` into the SDF radius and keepout.
- `training/contexts/registry.py` — the goal entry.

Reach, circle and button then reuse the same three pieces; their own wiring
is the same three reads.

## What this does not do

- It does not make `collidable` or geom type a value (structure, §1 of the
  mechanics doc). A group is what it is; only its count varies.
- It does not vary positions through the context yet — placement stays
  random. Tristan's suggestion to make positions a context dimension is a
  separate step: `place_objects` would take target positions instead of
  sampling them, which is a smaller change than this one.
- Circle level 1 has no hazards and therefore no hazard lidar in its
  observation; the union gives it one (all zeros). That changes level 1's
  observation size for the benchmark — Tristan.

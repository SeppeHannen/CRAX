# What can and cannot differ between parallel environments

Reference for the context / curriculum design. Every parallel environment is a
*slot* on the `num_envs` axis of one compiled JAX program.

## The rule

**Every slot runs the same program. Slots may differ in data; they may not
differ in shape, structure, or amount of work.**

This is not a JAX quirk. A GPU executes one instruction stream on thousands of
threads in lockstep; a per-thread `if` is realised by running both branches and
letting each thread *not write* the one it does not need. Results can differ per
slot; time spent cannot.

## Can differ per slot (free)

| Thing | Mechanism | Examples |
|---|---|---|
| Any numeric value in the state | array with a leading `[num_envs]` axis | velocity threshold, max height, goal speed, hazard positions and radii |
| Which branch's *result* a slot keeps | `jnp.where(cond, a, b)`; both computed | reset vs continue; cost formula A vs B |
| Masks over a fixed-size set | boolean array | "hazards 0–11 active, 12–19 disabled" |
| Selection from a fixed-size table | `table[index]` with per-slot index | which of 3 levels; which of K goal types |
| Random draws | per-slot RNG key | layouts, poses, noise |
| Episode timing | `done` is per slot; auto-reset is per slot | episodes end whenever |
| The context ω | all of the above | one vector per slot in `state.info["context"]` |

## Cannot differ per slot

| Thing | Why | Workaround |
|---|---|---|
| Array shapes | one array per quantity | pad to the maximum, mask the rest |
| Number of objects in the MuJoCo model | model is compiled into the program | build with max count; park unused objects outside the arena with zero cost weight |
| Object types / geometry | baked into the model | include every type at max count; mask |
| Which code runs | single instruction stream | both branches run; `where` selects |
| Time a slot's step takes | lockstep | none: cost is the union of all paths, every step |
| Loop lengths (`scan` length, `n_frames`, `episode_length`) | compile-time constants | episodes can end *early* via `done`, never run longer |
| Observation / action dimensions | network shapes | fixed per env class |
| Agent morphology | different model | separate run |

## What costs time

| Change | Cost |
|---|---|
| New per-slot value read by an existing operation | 0 |
| New operation per step | ≈ 3 µs × steps per epoch (launch-bound); a handful is fine |
| `reset()` inside the step (reset-on-done via `where`) | one reset's instructions **every step**, done or not — measure |
| Padding hazards 12 → 20 | more work per op, no new ops — likely ≈ 0 while launch-bound |
| Any shape or structure change | new compiled program, ≈ 50 s |

## Design consequences

1. A context is a vector of numbers per slot, in `state.info["context"]`; every
   suite reads its parameters from there rather than from `self._x`.
2. Suites whose difficulty is a *count* (Goal, Reach, Circle) get a fixed
   max-count model and a per-slot active mask. The mask is the count dimension.
3. All object types are always present in the model; masks decide.
4. Reset-on-done happens inside the program via `where`. Semantically "fresh
   environment every episode"; its cost is the one open measurement.
5. One agent morphology per run.
6. Distributions decide values on the host; the program never changes. What a
   distribution hands the program is a `[num_envs, D]` float array.

See also: `../measurements/2026-09-20_num_envs_sweep_and_launch_bound.md`.

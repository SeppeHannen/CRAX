# What an environment can vary without recompiling — and what it costs

Companion to `per_slot_constraints.md` (the rule) and
`context_spaces_by_suite.md` (the suites). This document is the *mechanics*:
which knobs of a CRAX/MJX environment are plain values that can differ per
parallel slot for free, which are structure that forces a new compiled program,
and which sit in between — legal but paid for on every step. Written 2026-09-22
from the MJX documentation, the installed `mujoco-mjx` source, and a probe of
the `mjx.Model` fields on this machine; the facts are cited so they can be
re-checked when MJX moves.

## 1. The two kinds of fields in `mjx.Model`

MJX is explicit about this ([MJX docs, "Structs"](https://mujoco.readthedocs.io/en/stable/mjx.html#structs)):

> JAX arrays in `mjx.Model` and `mjx.Data` support adding batch dimensions
> [...] a natural way to express domain randomization. Numpy arrays in
> `mjx.Model` and `mjx.Data` are structural fields that control the output of
> JIT compilation. Modifying these arrays will force JAX to recompile.

Probed on the installed version (`mujoco 3.11.0`, `jax 0.10.1`):

| field | kind | meaning for us |
|---|---|---|
| `geom_size`, `geom_pos`, `geom_rgba`, `geom_margin`, `geom_friction` | JAX | a hazard's radius, visual colour, offset within its body — **values**, can be per slot |
| `body_pos`, `body_mass`, `jnt_range`, `dof_damping`, `actuator_gear`, `opt.timestep`, `opt.gravity` | JAX | physics parameters — values |
| `geom_type`, `geom_contype`, `geom_conaffinity`, `geom_bodyid`, `body_mocapid`, `jnt_limited` | numpy | **structure** — change ⇒ recompile |
| `nbody`, `ngeom`, `nmocap`, `nq`, `nv` | int | structure; the sizes of everything above |

CRAX's `System` subclasses `mjx.Model` (`crax/base.py:415`), so the same split
applies to `env.sys`. Two consequences the audit of the suites depends on:

- **`collidable` is structure.** MJX decides at trace time which geom pairs
  to test: `collision_driver.geom_pairs` iterates the model in Python and
  keeps a pair only if `contype & conaffinity` matches
  (`collision_driver.py:127-190`). A geom with `contype=conaffinity=0` is not in
  any pair, so it costs no contact work — and cannot be made collidable per slot.
- **Size is a value; type is structure.** A hazard's radius can differ per slot
  (`geom_size` is JAX) but cylinder vs cube cannot (`geom_type` is numpy, and
  the collision *function* is chosen by type at trace time).

## 2. Where a per-slot value can land

A per-slot number is read from `state.info` and can land in three places:
env code, `mjx.Data`, or a JAX field of `mjx.Model`. Everything a suite does
is one of these.

### 2a. From `state.info` — free, what we do today

`state.info["context"]` is a `[num_envs, D]` array; the env indexes it where
it used to read `self._knob` (`crax/envs/context.py` fixes the pattern: the
read is `context.parameter(self, state, name)`, always, with no fallback).
No physics involved. This covers every knob the *environment code* consumes:
thresholds, speeds, cost weights, masks, the bounds of a random draw in
`reset`. Cost: one gather per read, unmeasurable.

### 2b. From `pipeline_state` (i.e. `mjx.Data`) — free, already batched

`qpos`, `qvel`, `mocap_pos`, `mocap_quat`, `ctrl`, `xfrc_applied` are per-slot
by construction. Anything expressible as "put this body here" or "apply this
force" is a value. Hazards are mocap bodies (`hazards.py:138,196,238,301`), so
their positions are the canonical example: parking a hazard at x = 1000 is a
write to `mocap_pos` at reset and nothing else.

### 2c. Into a JAX field of `mjx.Model` — free, and still door 2a

MJX documents that the JAX fields of `mjx.Model` may carry a batch axis ("a
natural way to express domain randomization"). Brax's
`DomainRandomizationVmapWrapper` uses this, but with one draw per slot *at
construction*, fixed for the run — the frozen-layout defect we replaced
`AutoResetWrapper` to get rid of. Not what a context is.

We do not need the wrapper. `mjx.Model` is a pytree, so an env can read a
value from `state.info["context"]` and write it into the model *inside* `step`:

```python
sys = self.sys.tree_replace({"geom_size": self.sys.geom_size.at[self._goal_geom, 0].set(goal_radius)})
pipeline_state = self._pipeline.step(sys, state.pipeline_state, action, ...)
```

Per slot (under `vmap`), per episode (the context changes at reset), same
compiled program (`geom_size` is a JAX field, §1). Cost: one gather per field
per substep, launch-bound noise. This is how a per-slot `goal_size` or hazard
`size` would move the *visual* geom; the cost already uses its own radius
array, so correctness never depends on it.

Verified 2026-09-22 on `safe_lift_ant` (CPU): a jitted step that does
`tree_replace` on `opt.gravity`, `geom_size`, `body_pos`, `body_ipos` from two
runtime scalars (gravity, leg scale) compiles **one** program and gives
different torso trajectories for gravity −2/−9.81/−15/−20 and leg scale
0.7/1.0/1.1/1.3; `vmap` over both gives four slots with four physics in one
program. So **gravity and limb length are per-slot values** — Tristan's
question. Caveat for "longer legs": a consistent change moves four fields
together (geom size, the child body's `body_pos` so joints follow, `body_ipos`,
and mass/inertia if they are to scale); MuJoCo would derive the last from the
geometry at compile time, here it is on us. What cannot change: the number of
segments, a capsule becoming a box, which body a geom belongs to.
`PipelineEnv.pipeline_step` uses `self.sys`, so an env doing this calls
`self._pipeline.step(modified_sys, …)` directly or we add a `sys` parameter.

So there is really **one door**: `state.info["context"]`. The env decides
whether to read it into a comparison (velocity, height), a random bound
(pathway), a mask (lift), `mocap_pos` (layouts, parking), or a model field
(sizes). Nothing in the model that is *not* a JAX field can be reached this
way — that is §3.

## 3. What can never vary per slot

| what | why | evidence |
|---|---|---|
| number of bodies / geoms / mocaps | array sizes | `nbody` etc. are ints |
| geom type (cylinder vs cube) | collision function chosen per type at trace time | `_geom_groups`, `collision_driver.py:196` |
| collidable / non-collidable | pair list built from `contype/conaffinity` in Python | `geom_pairs`, `:127` |
| which bodies are mocap | `body_mocapid` is numpy | probe |
| joint limits on/off | `jnt_limited` numpy (`jnt_range` *values* are fine) | MJX docs |
| observation / action size | network shapes | — |
| `n_frames`, `episode_length`, unroll lengths | `scan` lengths | `crax/envs/base.py:148` |
| which Python branch runs | single instruction stream; `lax.cond` on a traced value runs both | `per_slot_constraints.md` |

For all of these the only move is the **union**: build the model with every
variant present at maximum count, and select per slot with values (masks,
positions). What a union costs is §5.

## 4. How a hazard is "present" in an env — the four consumers

An audit of the six hazard suites (2026-09-22) found that a hazard is looked
at by exactly four pieces of code. Making "active" a per-slot value means
making each of the four honour a mask. Nothing else needs to know.

| consumer | reads | today | with `active_mask[H]` |
|---|---|---|---|
| **physics** | `mocap_pos`, `geom_contype` | collidable hazards collide wherever they are | park inactive ones far away; collidable-ness stays what the XML says |
| **cost** | `info["hazard_positions"]`, `self.size`, `h.collidable` — a Python loop over hazard objects (`hazards.py:456-465`) | every hazard contributes | multiply each hazard's cost by `mask[h]` |
| **lidar** | `mocap_pos[hazard_ids]` (goal, circle) or `info["hazard_positions"]` (button, reach, pathway) | nearest hazard per ray, `exp(-d)`-style | zero the reading where `mask[h]==0` *before* the per-ray min; parking alone leaves a tiny non-zero |
| **compass / closest-k** | `mocap_pos[all_hz_ids]` → `top_k` | closest k hazards | set masked distances to +∞ before `top_k`; pad with the existing `-1` sentinel |
| **placement at reset** | `place_objects` scan, keepout radii | every hazard gets a position | inactive: skip the draw, write the parking position; exclude from keepouts |

Placement is the one that runs *inside* `reset`, which — under
`ContextualAutoResetWrapper` — runs every step for every slot. Its scan length
is the max count; its per-iteration work is the same whether the hazard is
active or parked (`where`). So a union model's reset is as expensive as the
largest level's reset. That is the number to measure (§5).

Two facts from the audit that make this simpler than it sounds:

- The cost loop branches on the *static* `h.collidable`, so a union model with
  collidable and non-collidable groups needs no change to the loop — each
  hazard object keeps its type; the mask is a multiplication at the end.
- Gremlins (button) already keep their `travel` in a per-object array fed to
  `compute_gremlin_positions`; that array can be per slot with no change.

## 5. What the union costs — and where the money actually goes

Measured facts about this GPU (`measurements/2026-09-20_num_envs_sweep_and_launch_bound.md`):
the training step is **launch-bound** — ~2 µs mean kernel, ~2 M launches per
epoch, the GPU mostly idle waiting for the next kernel. In that regime the
cost of a change is *the number of kernels it adds*, not the arithmetic
inside them. This reorders the intuitions:

| change | kernels added | expected cost |
|---|---|---|
| read a value from `state.info` | 0–1 gather | none |
| more *non-collidable* hazards | 0 pairs; a few kinematics ops | small (see probe below) |
| the first *collidable* hazard group | one contact slot per (hazard, agent geom) pair, carried through collision + constraint solver every step, touching or not | **real**: the solver's arrays get `ncon` rows, and the pair test runs for each |
| more hazards in an existing collidable group | 0 new ops (same program, wider arrays) | none while launch-bound |
| mask multiply in cost / lidar | ~3 elementwise kernels per step | ≈ 10 µs per step × 20 × 16 steps per epoch ≈ 3 ms per epoch — invisible |
| union-model `reset` every step | the reset's own kernel count, every step | **real**; today's velocity reset is trivial, Goal's placement scan is not |
| a `lax.cond` on a traced value | both branches, every step | fine if branches are small; never for physics |

Probe (CPU, `mjx.step` on a sphere + plane + N mocap cylinders, 2026-09-22;
HLO line count as a crude size proxy, `ncon` = static contact slots):

| hazards | collidable | `ncon` | HLO lines |
|---|---|---|---|
| 0 | — | 1 | 8 293 |
| 12 | no | 1 | 8 933 (+8 %) |
| 12 | yes | 13 | 11 529 (+39 %) |
| 20 | yes | 21 | 11 529 (+0 % over 12) |

So: **non-collidable hazards are nearly free and collidable hazards are not,
but the price is paid once per collidable *group*, not per hazard.** A union
model that adds collidable groups a level did not have (Goal level 1 has none;
the union has 8 + 4) makes *every* slot pay for contact machinery that only
some slots use. That is the honest cost of the union; whether it matters is
measured, not argued (§8). Parking does not reduce it — a parked collidable
hazard still owns its contact slot; it merely never fills it.

Levers if it does matter, from MJX ([Performance tuning](https://mujoco.readthedocs.io/en/stable/mjx.html#mjx-jax)):
`max_geom_pairs` and `max_contact_points` as `<custom><numeric>` in the XML
cull to the *k* closest pairs / deepest contacts with one `top_k`
(`collision_driver.py:368-397, 413-455`), so the solver sees a fixed small
`ncon` however many hazards exist. Explicit `<contact><pair>` lists are the
other lever the docs recommend. Both turn the structural cost back into a
value-typed one.

## 6. Decision table for a new knob

Given a difficulty knob, decide in this order:

1. **Is it read by env code only (cost, reward, done, obs, a bound in reset)?**
   → `state.info["context"]`, one line. Height, push, velocity, pathway, lift
   (after the all-feet rewrite), circle boundaries, goal `goal_size` for the
   SDF and keepout.
2. **Is it a position or pose?** → `mocap_pos` / `qpos` at reset. Hazard
   layouts; parking.
3. **Is it a JAX field of the model (size, mass, friction, colour)?**
   → `sys.tree_replace` from the context inside `step` (2c). Only for the
   visual radius of a per-slot-sized hazard; not needed for correctness yet.
4. **Is it a count, a type, or collidable?** → union model + `active_mask`,
   four consumers (§4). Reach, circle, goal, button.
5. **Does it change observation or action size?** → not a context. Circle
   level 1's missing lidar block is this: the union fixes the obs size at
   "lidar present", which changes level 1's observation (flag to Tristan).

## 7. What this rules out for Ω, and what it buys

Ruled out as dimensions of Ω (for one compiled program): geom type, collidable
flag, morphology, episode length, number of lidar rays. Each of these is a
*different environment*, not a context.

Bought: every remaining knob in `difficulty.py` — thresholds, speeds, gaps,
heights, foot masks, boundaries, hazard counts per group, hazard sizes,
gremlin travel, placement extents, goal size — is a value. So Ω for the full
benchmark is a box (with integer dimensions for counts and masks) exactly as
`space.py` already models it; no suite needs a new kind of `Dimension`.

## 8. Open measurements

Before building the union for goal/button:

1. Reach at 10 hazards, `--context_distribution level:3` vs stock `--difficulty 3`,
   idle GPU, `--measure_performance`: the cost of a masked placement scan in a
   reset that runs every step.
2. Same on goal with the union model (20 hazards + walls, 12 collidable):
   pair count in the trace; whether `max_geom_pairs` is needed.
3. Whether XLA fuses the mask multiplies into the existing cost kernels (it
   should; check the kernel count in XProf, not the wall-clock).

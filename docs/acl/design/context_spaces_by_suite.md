# Context spaces per suite: what the manual curriculum actually varies

Audit of `crax/envs/difficulty.py` against the per-slot rule ("slots may differ
in data, not in shape/structure/work"). For each suite: what the three levels
change, whether each knob is a **value** (free per slot) or **structure** (needs
pad-and-mask or is off the table), and the resulting Ω.

Legend: **V** = value, already or trivially per-slot · **M** = count, needs a
fixed max-count model + per-slot mask · **S** = structural, changes the MuJoCo
model or array shapes; only expressible as a *fixed union* over all levels.

## Value-typed suites (Ω definable now)

| suite | knob(s) across levels | kind | Ω | status |
|---|---|---|---|---|
| **velocity** (6 agents) | `velocity_threshold` = baseline × {1, 0.75, 0.5} | V | `threshold ∈ [0.4, 1.0] × baseline` | **registered**; env reads context |
| **height** (humanoid) | `max_height` ∈ {1.20, 1.10, 1.00} | V — but see note | `max_height ∈ [0.9, 1.3]` | env change needed: `max_height` is currently written into the XML (`HEIGHT_PLACEHOLDER`, safe_height.py:139), presumably for a visual ceiling geom; the *cost* uses `self._max_height` (line 213). Cost side → context; the geom either stays at max or moves per slot via mocap. |
| **push** (point) | `goal_velocity` ∈ {0, 0.3, 0.6} | V | `goal_velocity ∈ [0, 0.8]` | env change: read from context in `step` (safe_push.py:570, 635) |
| **pathway** (walker2d) | `max_gap` ∈ {6, 4, 2} | V | `max_gap ∈ [1.5, 8]` (with `min_gap` fixed) | env change: `max_gap` is used at *reset* to sample hazard placement (safe_pathway.py:254) → needs `reset_with_context` |
| **lift_ant / lift_spider** | `restricted_feet` = list of names | V as a mask | `feet_mask ∈ {0,1}^4` / `{0,1}^6` — 4 (6) integer dims | env change: cost = `mask · foot_contacts`; level contexts are specific masks; Ω = all masks (or those with ≥1 restricted foot) |

Five suites, all straightforward. Lift is the interesting one: a *discrete* Ω
(16 / 64 masks) where "uniform over Ω" means uniform over subsets of feet —
a genuinely different thing from the 3 levels.

## Count-typed suites (need pad-and-mask first)

| suite | knob(s) across levels | kind | plan |
|---|---|---|---|
| **reach** | `num_hazards` ∈ {4, 7, 10} | M | model with 10 (or more) hazards; per-slot `active_mask`; inactive ones parked out of the workspace with zero cost; Ω = `num_active ∈ {1..10}` (+ hazard radius as a V dimension if desired) |
| **circle** | `boundary_x, boundary_y` ∈ {(1.125, None), (1.05, 1.05), (0.975, 0.975)}; hazards 0 → 1 → 2, size 0.15 → 0.2 | V + M | boundaries: V (`boundary_y=None` ⇒ encode as "very large"). Hazards: model with 2; `active_mask`; `size` V per slot if hazard geoms are scaled via mocap/size arrays, else fixed at max |

## Structural suites (Ω is a fixed union; levels are points in it)

| suite | knob(s) across levels | kind |
|---|---|---|
| **goal** | hazard **type** (cylinder → +cube), **count** (12 → 16 → 20), **size** (0.4 / 0.3 / 0.35 / 0.25), **height** (0.01 / 0.4 / 0.5), **collidable** (F → mixed → mixed); `goal_size` 0.2 → 0.18 → 0.16 | S + M + V |
| **button** | hazard count 4 → 8 → 12, gremlin count 4 → 6 → 8, gremlin `travel` 0.35 → 0.45, `placement_extents` ±2 → ±2.5 → ±3 | M + V |

For these, Ω has to be built as the **union model**: every hazard group from
every level present at max count, with per-slot masks. Concretely for Goal:

- 6 groups in the union: cylinder-prox (12, size 0.4), cylinder-coll (8, 0.3,
  h 0.4), cube-prox (6, 0.3), cube-coll (4, 0.25, h 0.5), cylinder-prox-L3
  (6, 0.35), cylinder-coll-L3 (4, 0.25) — or merged by (type, collidable) with
  size made a per-slot value → 4 groups.
- Ω = per-group active counts (integer dims) + goal_size (V) + optionally
  per-group size (V).
- Level 1 = (12, 0, 0, 0, goal 0.2); level 2 = (8, 8, 0, 0, 0.18); level 3 =
  (6, 4, 6, 4, 0.16). All three are the same compiled program.
- `collidable` is a **contact-pair property in the MuJoCo model** — not a
  per-slot value. Hence collidable and non-collidable hazards are *different
  groups*, and "turning collision on" for a slot means activating a collidable
  group rather than flipping a flag.

Costs to check: (a) lidar and cost code must ignore inactive hazards
(masked); (b) inactive collidable hazards must be parked where they cannot be
touched; (c) extra geoms add work per physics op — likely free while
launch-bound, to be verified with `--measure_performance`.

## Recommended order

1. **Velocity** (done) → first experiment.
2. **Height, Push**: trivial `step`-side context reads; adds two more
   value-typed suites for almost nothing.
3. **Lift** (mask Ω) and **Pathway** (`reset_with_context`): still value-typed,
   slightly more work; Pathway is the first suite whose reset *uses* the
   context — a good test of that path.
4. **Reach, Circle**: first pad-and-mask, small (≤ 10 objects).
5. **Goal, Button**: union model; the largest change and the one that touches
   lidar/cost code. Also where the frozen-layout fix matters most.

Steps 4–5 are environment engineering; they should be measured for cost
before committing to them, and Tristan should probably know we're building a
union model of his levels.

## Plan for the full benchmark (2026-09-22, after a line-by-line audit)

Decision: support all nine suites, not only the value-typed ones. The audit
of every use of every difficulty knob (file:line) confirmed the table above
and added three facts:

1. **Every hazard is a MuJoCo mocap body; `collidable` is `contype/conaffinity`
   on its geom, and the cost path is a Python loop over hazard objects that
   branches on the static `collidable`** (`hazards.py:456-465`). So a union
   model works without touching the cost loop: each hazard object keeps its
   static type; a per-slot `active_mask[H]` multiplies each hazard's cost and
   zeroes its lidar/compass reading; inactive hazards are parked (in
   `mocap_pos`, which lidar reads — not only in `info`) far outside the arena.
2. **Circle level 1 has a different observation shape**: with 0 hazards the
   lidar and compass blocks are omitted (`safe_circle.py:341-342`). A union
   model has 2 hazards always, so level 1 under contexts observes an
   all-zero lidar block that stock level 1 does not. Behaviour change for the
   benchmark → agree with Tristan.
3. **Lift's cost is a Python loop over the *restricted* feet only**
   (`safe_lift.py:277-343`), so the array shape depends on the level. The fix
   is the natural one: compute contact for all 4/6 feet and dot with a per-slot
   mask — one shape for every level, and Ω = the mask.

Shared machinery to build once (in `crax/envs/hazards.py` / `env_utils.py`,
flagged to Tristan): an `active_mask` on the hazard manager consumed by cost,
lidar, compass and placement (inactive → parked, excluded from keepouts), and
a `reset_with_context` on every suite whose reset consumes a knob. The
mechanics — why these four consumers and no others, which model fields are
values, and what a union model costs (probed: non-collidable hazards ~free,
each collidable *group* adds contact slots to every slot's solver) — are in
`what_can_vary_per_slot.md`.

| # | suite | env change | Ω | risk |
|---|---|---|---|---|
| 1 | height | cost reads `max_height` from context (`safe_height.py:213`); ceiling geom stays visual at Ω max | `max_height ∈ [0.9, 1.3]` | none |
| 2 | push | `goal_velocity` from context at :570; `lax.cond` on a Python float (:634) → `jp.where` | `goal_velocity ∈ [0, 0.8]` | none |
| 3 | lift ant / spider | all-feet contact vector × mask | `feet_mask ∈ {0,1}^4 / {0,1}^6`, integer dims | none; Ω is discrete |
| 4 | pathway | `reset_with_context`: `max_gap` bounds the gap `uniform` at :254; shape `(100, 3)` unchanged | `max_gap ∈ [1.5, 8]` | first reset-side context |
| 5 | reach | model with 10 hazards; `active_mask`; placement parks inactive | `num_active ∈ {1..10}` | first pad-and-mask |
| 6 | circle | model with 2 hazards, lidar always on (fact 2); boundaries from context, `None` → the arena half-width 3.0; visual walls fixed | `active_cylinders ∈ {0,1,2}`, `boundary_x ∈ [0.9, 1.2]`, `boundary_y ∈ [0.9, 3.0]` | obs-shape change at L1 (28-d → 60-d; L2/L3 were 60-d) — **done** |
| 7 | goal | union of the level-3 groups + L1/L2 cylinder groups (4 groups by (type, collidable)); `active_count` per group; `goal_size` → SDF radius and keepout from context, geom at Ω max | 4 integer dims + `goal_size ∈ [0.14, 0.22]` | largest change; touches lidar/cost |
| 8 | button | 12 blocks + 8 gremlins; `active_count` per group; gremlin keepout from the context's travel; no walls in this arena (extents ⇒ layout only) | 2 integer dims + `gremlin_travel ∈ [0.3, 0.5]` + `placement_extent ∈ [2, 3]` | **done**, but the box has an infeasible corner (full crowd in the 2 m square); see README 4c |

Order: 1–4 are each an afternoon; 5 builds the mask machinery on the smallest
suite; 6–8 reuse it. Each suite lands with its `SuiteContexts`, and
`tests/test_suites.py` (parametrised over the registry) checks it for free:
keys registered, levels reproduce `difficulty.py`, env reads its context.

Per-slot cost to measure at step 5, before 6–8: extra parked geoms in every
physics step (probably free while launch-bound) and the masked placement loop.

What "uniform over Ω" means for the structural suites is a design choice, not
a derivation: uniform over per-group active counts gives layouts no level has
(e.g. 12 cylinders *and* 4 cubes). That is the point of Ω, but it should be
said in the thesis and agreed with Tristan.

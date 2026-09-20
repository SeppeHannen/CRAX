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

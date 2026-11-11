# Standing decisions about the environment and the trainer

Decisions taken once that every experiment since rests on. Each says what was
decided, by whom and when, and what it changed. A decision that closed a whole
line of work has its own file in this folder.

- **Ω is the environment's physics and layout, not its cost threshold** (with
  Tristan, 2026-09-22). A run keeps one threshold. The threshold, `max_height`,
  `max_gap` and foot-mask spaces stay as pipeline smoke tests; experiments use
  layout and physics Ω. Physics dimensions (friction, mass, gravity, limb
  length) are per-slot values via `sys.tree_replace` in `step` — verified,
  `design/what_can_vary_per_slot.md` §2c.
- **First experimental suite: `safe_goal_point`** (Tristan: low training
  variance, so a curriculum effect is detectable with few seeds).
- **Contexts are values, never shapes.** A count becomes pad-and-mask;
  structure becomes a union model (`design/per_slot_constraints.md`).
- **Uniform over Ω for the hazard suites produces layouts no level has** (12
  cylinders *and* 4 cubes). Intended; say so in the thesis.
- **Goal-point has one goal, not two** (Giuseppe, 2026-10-06). Stock
  `difficulty.py` sets `goal_count: 2` at every level, but the observation's
  goal compass reads only the first goal (`safe_goal.py` `TODO handle multiple
  goals`); the second is visible only through the coarse goal lidar. One goal
  is what the agent actually sees. Change: `goal_count: 1` at every level
  (before the first goal-point run; flag to Tristan).
- **The arena walls are a fence, not hazards** (Giuseppe, 2026-10-06). Stock
  CRAX builds the four walls of goal and push through the hazard pipeline
  (`outer_wall` → four `fixed` `rect` hazards), so they were scanned into the
  hazard lidar as four centre-points, competed for the eight compass slots, and
  cost 3 per contact step — verified on the running environment (circle and
  button have no walls). Safety-Gymnasium's Goal has no walls; its placement
  extents bound the arena. Decision: walls are static collidable geometry in the
  model and nothing else — not observed, not costed, not a hazard
  (`crax/envs/arena.py`). Invariant: *every hazard the environment knows is one
  of its hazard specs'*; `fixed` has left the hazard pipeline. Verified: full
  thrust into the fence for 400 steps — robot stays in, cost 0, hazard lidar
  dark. Observation shape unchanged; goal's union model has 30 hazard bodies.
- **Goal's Ω is capped at 20 active hazards in total** (Giuseppe, 2026-10-06).
  The box's corner (30 hazards) does not fit the 5 m square; goal's placement
  nudges instead of raising, so the failure was silent. Measured on CPU: the
  packing quantity is total keepout area — L1 6.0 m², L2 6.3 m², L3 9.1 m²; at
  L3's area 4 % of resets have two hazards overlapping, one kind more and half
  do; `Uniform(box)` hit an overlap in 11.5 % of draws. `ContextSpace` gained
  `integer_total_cap`: Ω = {counts : Σ ≤ 20} × goal_size, so level 3 is the
  densest point of Ω and every level stays inside Ω (the invariant
  `SuiteContexts` enforces; a cap of 16 would have put w outside Ω). Uniform
  samples one admissible integer tuple from an explicit table — which also fixes
  a pre-existing bias: rounding a uniform float gave the endpoints half weight.
  Residual overlap under the cap ≈ 2 %, the same crowding level 3 has.
- **Trainer contract.** `training/rounds.py` `RoundHook` (`extra_fields`,
  `observe`, `on_round_end`) and `evaluation_wrap_env_fns` are the only two
  trainer additions; `ppo_lag` forwards them, and `ppo_pid`. Any other PPO-family trainer needs the same two lines.

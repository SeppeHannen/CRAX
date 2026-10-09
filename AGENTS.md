# Agent instructions for this repository

This is Giuseppe Hannen's fork of CRAX (Constrained RL Accelerated with JAX),
extended with automated curriculum learning (ACL) for his TU/e graduation project.

## Before doing anything

1. **Read `docs/acl/README.md`.** It is the task board. Find the task being
   worked on, then read the documents listed under its *read first* column and
   open the files under *touches*. Do not start from the code alone.
2. Skim `/memories/repo/project.md` if available for the condensed history.
   When it disagrees with `docs/acl/README.md`, the README wins.
3. **Before calling work done, review the whole `git diff` as a stranger.**
   Not the parts you remember writing — all of it, file by file. For every
   line ask: who reads this value, who calls this, does this check guard a
   case that can happen? Dead state, a defensive check on our own data, an
   assumption living in a comment instead of a `raise` — these are found by
   reading, not by tests. Then run the CPU tests.
4. At the end of a piece of work, update the task board (status, new docs,
   commit hash) so the next session is not in the dark.

## Standing rules

- **Never start a GPU run without asking.** CPU tests are fine:
  `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/ -q -p no:cacheprovider`.
- Use `.venv/bin/python`; bare `python` is not on PATH.
- W&B is mandatory (entity/project pinned in `training/config.py`). Never add a
  way to run without it.
- Contexts are **values on the `num_envs` axis, never shapes**. A new compiled
  program costs ~50 s; a per-slot value costs nothing.
- Vocabulary: *context* $\omega \in \Omega$, *distribution* (not "teacher"),
  $r$ = uniform over $\Omega$, $w$ = deployment, *round* = one training step,
  $q$ intended vs $\hat q$ realised curriculum. Defined in `docs/acl/README.md`.
- Anything that changes the upstream benchmark's behaviour (baselines, wrappers,
  models) must be flagged so it can be raised with Tristan (upstream author).
- Leave code uncommitted until Giuseppe has reviewed it; he asks for the
  commit. Then commit in small, described steps. Never push unasked.

## Code quality

Every change must raise the average quality of the repository. Concretely:

- **Fully typed.** Every function signature and every public attribute has a
  type. Prefer `Protocol`s and small dataclasses over dicts of dicts.
- **Encapsulate complexity.** A caller should read as a sentence; the
  machinery lives behind a named function or class with a docstring that says
  what, not how.
- **Intuitive names, no abbreviations.** `environment_steps`, not `env_steps`;
  `performance`, not `perf`. A name should make the comment unnecessary.
- **No fallbacks, no backward compatibility, not defensive.** No
  `if old_format ... else ...`, no `.get(key, default)` to paper over a
  missing value, no `if run is None: return`. Validate once at the boundary
  (CLI, W&B config, file input), then trust the types. When something is not
  as expected, **raise** with a message that says what was expected; a loud
  failure tells us what to improve, a silent default hides it.
- **One door per effect.** For each side effect (a metric reaching W&B, a
  context reaching an environment, a file being written) there is exactly one
  code path, and a component that needs the effect is *handed* that path
  (a callback, a sink) rather than reaching for the global itself. Before
  adding a rule to a path, grep for every other place that produces the same
  effect; a second path makes the rule a lie.
- **State the invariant, then look for its violations.** When a design rests
  on a sentence like "every metric passes the registry", write the sentence
  down and check the whole repository against it before calling the work
  done. Most of our real bugs are a true sentence with one exception.
- **Label data at the source.** If a consumer needs a fact the producer has
  (which slot a row came from, which round a value belongs to), the producer
  writes it into the data. Recovering it downstream from layout, order or
  arithmetic is an assumption about someone else's code.
- **One module per concept** (e.g. one file per context distribution); no
  utility grab-bags.
- Tests are CPU-only and assert behaviour, not implementation.

## Pull-request write-ups

Giuseppe reviews by following the data through the code, not file by file.
`docs/acl/PR_prioritized_level_replay.md` is the template. Its spine is **the
walk**: a call tree from the entry point (a CLI flag) to the last effect (a
point on the dashboard), through changed and unchanged files alike. Rules,
from program-comprehension and code-review research (Letovsky; Pennington;
Baum et al.; Knuth's literate programming):

- **Call tree, not file list.** Each entry is `file:line — what the line
  does`; what it calls is indented beneath it. "Where does this change enter
  the flow" is always the parent entry.
- **One sentence, one model.** Control flow ("line N calls X") and meaning
  ("X is where the score is computed") in separate sentences, never mixed.
- **Beacon first.** Each phase opens with one sentence saying what the phase
  does for the flow, before any file name.
- **Tags on the callee**: `[changed]` (what, in one line), `[new]`,
  `[unchanged]` (one or two sentences that connect the changed parts; nothing
  to review). A reader of only the `[changed]` entries must understand the
  change.
- **Verified line numbers** (grep them; never from memory), a verification
  table at the top, per-file detail after the walk, "open, not in this
  change" at the end.
- **One logical change per PR.** If the walk has more than one spine, the PR
  should have been split; say so before letting it grow, not after.

## Experiment write-ups

`docs/acl/experiments/2026-10-08_goal_point_uniform_staged_plr_500M.md` is the
template: question → setup (one table, the exact command) → results as two or
three figures, each followed by a few bullets saying what they show → what
this changes in the plan. About a hundred lines. Figures come from a kept,
typed script in `scripts/analysis/` and live in `figures/<experiment>/`. The
document states the result, not how it was arrived at: no superseded runs, no
narration of the analysis; a bug found on the way goes in the README's *Known
defects*. Say what a number means (a motionless policy "meets the budget").

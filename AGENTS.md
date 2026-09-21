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
- **One module per concept** (e.g. one file per context distribution); no
  utility grab-bags.
- Tests are CPU-only and assert behaviour, not implementation.

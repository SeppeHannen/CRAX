# Agent instructions for this repository

This is Giuseppe Hannen's fork of CRAX (Constrained RL Accelerated with JAX),
extended with automated curriculum learning (ACL) for his TU/e graduation project.

## Before doing anything

1. **Read `docs/acl/README.md`.** It is the task board. Find the task being
   worked on, then read the documents listed under its *read first* column and
   open the files under *touches*. Do not start from the code alone.
2. Skim `/memories/repo/project.md` if available for the condensed history.
   When it disagrees with `docs/acl/README.md`, the README wins.
3. At the end of a piece of work, update the task board (status, new docs,
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
- Code style: clear types, protocols/interfaces, encapsulate complexity, no
  abbreviations in identifiers, one module per distribution.
- Anything that changes the upstream benchmark's behaviour (baselines, wrappers,
  models) must be flagged so it can be raised with Tristan (upstream author).
- Commit in small, described steps; do not push without being asked.

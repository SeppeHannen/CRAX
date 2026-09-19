"""Pull XProf traces from Weights & Biases to this machine and open them.

Cluster jobs upload their trace as a W&B artifact (type ``xprof-trace``); this
tool downloads one or many into a single XProf log directory and, unless told
otherwise, starts ``xprof`` on it.  Every downloaded run becomes one entry in
XProf's "Sessions" dropdown, so several runs can be compared in one UI.

Examples::

    # one run, by W&B run name (as shown in the UI) or run id
    python -m training.performance.fetch_traces safe_goal_point_Level_1_ppo_lag_seed0_20260920_000133_017387

    # everything tagged 'perf' in the project, don't start the UI
    python -m training.performance.fetch_traces --filter tags=perf --no-open

    # a different project / destination
    python -m training.performance.fetch_traces --project crax --entity my-team --output runs/traces <run> ...

Downloads are cached by W&B, so re-running is cheap.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional

from training.config import WANDB_ENTITY, WANDB_PROJECT
from training.performance.sinks import TRACE_ARTIFACT_TYPE

DEFAULT_OUTPUT = "runs/traces"


def _api():
    import wandb

    return wandb.Api()


def _parse_filter(spec: Optional[str]) -> Dict[str, Any]:
    """Turn 'tags=perf' / 'config.env_name=safe_goal_point' into a MongoDB-style filter."""
    if not spec:
        return {}
    key, _, value = spec.partition("=")
    if key == "tags":
        return {"tags": {"$in": [value]}}
    return {key: value}


def _resolve_runs(api, project_path: str, run_specs: Iterable[str], filter_spec: Optional[str]) -> List[Any]:
    runs: List[Any] = []
    if filter_spec:
        runs.extend(api.runs(project_path, filters=_parse_filter(filter_spec)))
    for spec in run_specs:
        # Accept a full path, a run id, or a display name.
        if "/" in spec:
            runs.append(api.run(spec))
            continue
        try:
            runs.append(api.run(f"{project_path}/{spec}"))
            continue
        except Exception:
            pass
        matches = list(api.runs(project_path, filters={"display_name": spec}))
        if not matches:
            print(f"[fetch_traces] no run named or with id '{spec}' in {project_path}", file=sys.stderr)
            continue
        runs.extend(matches)
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for run in runs:
        if run.id not in seen:
            seen.add(run.id)
            unique.append(run)
    return unique


def _trace_artifacts(run) -> List[Any]:
    return [a for a in run.logged_artifacts() if a.type == TRACE_ARTIFACT_TYPE]


def fetch(
    run_specs: Iterable[str],
    *,
    project: str,
    entity: Optional[str],
    output: str,
    filter_spec: Optional[str] = None,
) -> List[str]:
    """Download traces for the given runs into ``output/plugins/profile/<run_name>/``.

    Returns the list of session directories that were populated.
    """
    api = _api()
    entity = entity or api.default_entity
    project_path = f"{entity}/{project}"
    runs = _resolve_runs(api, project_path, run_specs, filter_spec)
    if not runs:
        print(f"[fetch_traces] nothing to fetch from {project_path}", file=sys.stderr)
        return []

    sessions_root = os.path.join(output, "plugins", "profile")
    os.makedirs(sessions_root, exist_ok=True)
    populated: List[str] = []
    for run in runs:
        artifacts = _trace_artifacts(run)
        if not artifacts:
            print(f"[fetch_traces] {run.name}: no {TRACE_ARTIFACT_TYPE} artifact (was it run with --profile_epochs?)")
            continue
        for artifact in artifacts:
            session_dir = os.path.join(sessions_root, run.name)
            # The artifact holds XProf's `<timestamp>/` session folder(s); download
            # into a scratch dir next to the target and flatten to one session per
            # run so the UI lists runs, not timestamps.
            scratch_dir = session_dir + ".download"
            if os.path.isdir(scratch_dir):
                shutil.rmtree(scratch_dir)
            artifact.download(root=scratch_dir)
            if os.path.isdir(session_dir):
                shutil.rmtree(session_dir)
            os.makedirs(session_dir)
            for root, _dirs, files in os.walk(scratch_dir):
                for name in files:
                    shutil.move(os.path.join(root, name), os.path.join(session_dir, name))
            shutil.rmtree(scratch_dir)
            size_mb = sum(os.path.getsize(os.path.join(session_dir, f)) for f in os.listdir(session_dir)) / 2**20
            print(f"[fetch_traces] {run.name}: {artifact.name} -> {session_dir} ({size_mb:.0f} MB)")
            populated.append(session_dir)
    return populated


def open_xprof(logdir: str, port: int) -> int:
    executable = shutil.which("xprof")
    if executable is None:
        print("[fetch_traces] xprof is not installed: pip install xprof", file=sys.stderr)
        return 1
    print(f"[fetch_traces] xprof --logdir {logdir} --port {port}   (http://localhost:{port}, Ctrl-C to stop)")
    return subprocess.call([executable, "--logdir", logdir, "--port", str(port)])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="W&B run names, run ids, or entity/project/run_id paths")
    parser.add_argument("--filter", dest="filter_spec", default=None,
                        help="select runs by 'tags=<tag>' or '<field>=<value>' (e.g. config.env_name=safe_goal_point)")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", WANDB_PROJECT))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", WANDB_ENTITY))
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="XProf log directory to populate")
    parser.add_argument("--no-open", dest="open_ui", action="store_false", help="download only; do not start xprof")
    parser.add_argument("--port", type=int, default=8791)
    args = parser.parse_args(argv)

    if not args.runs and not args.filter_spec:
        parser.error("give at least one run, or --filter")

    populated = fetch(
        args.runs,
        project=args.project,
        entity=args.entity,
        output=args.output,
        filter_spec=args.filter_spec,
    )
    if not populated:
        return 1
    if args.open_ui:
        return open_xprof(args.output, args.port)
    print(f"[fetch_traces] done. View with: xprof --logdir {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

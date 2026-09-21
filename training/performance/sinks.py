"""Output sinks for performance records.

The per-round scalars are *not* handled here: the tracker reports them through
the same progress callback as the trainer's metrics, so one path (and the
dashboard registry on it) covers everything that reaches the run history.

What is left is W&B-specific and not a metric: the run summary (columns in
the runs table), the JSON file attached to the run, and the profiler trace as
an artifact of type ``xprof-trace``. A JSON copy is always written locally as
well; it makes a run inspectable without network access (e.g. on a cluster
compute node before ``wandb sync``).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping

import wandb

TRACE_ARTIFACT_TYPE = "xprof-trace"


class JsonSink:
    """Accumulates everything into one JSON document written on ``flush``."""

    def __init__(self, path: str):
        self.path = path
        self._document: Dict[str, Any] = {"history": [], "summary": {}}

    def log_scalars(self, scalars: Mapping[str, float], environment_steps: int) -> None:
        entry = dict(scalars)
        entry["environment_steps"] = environment_steps
        self._document["history"].append(entry)

    def set_summary(self, summary: Mapping[str, Any]) -> None:
        self._document["summary"].update(summary)

    def set_section(self, name: str, content: Any) -> None:
        self._document[name] = content

    def flush(self) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w") as handle:
            json.dump(self._document, handle, indent=2, default=_json_default)
        return self.path


class WandbSink:
    """Summary, files and trace artifact of the active ``wandb`` run.

    Every CRAX run is tracked in W&B and the tracker is installed after
    ``wandb.init``, so a missing run is a programming error and raises.
    """

    PREFIX = "performance/"

    @staticmethod
    def run() -> "wandb.Run":
        if wandb.run is None:
            raise RuntimeError("the performance tracker needs an active W&B run; install it after wandb.init")
        return wandb.run

    def set_summary(self, summary: Mapping[str, Any]) -> None:
        run = self.run()
        for key, value in summary.items():
            if value is not None:
                run.summary[self.PREFIX + key] = value

    def save_file(self, path: str) -> None:
        self.run().save(path, base_path=os.path.dirname(os.path.abspath(path)), policy="now")

    def log_trace_artifact(self, sessions_dir: str, run_name: str, metadata: Mapping[str, Any]) -> str:
        """Upload ``sessions_dir`` (XProf's ``plugins/profile`` folder) as one artifact; returns its name."""
        name = f"trace-{_artifact_safe(run_name)}"
        artifact = wandb.Artifact(name=name, type=TRACE_ARTIFACT_TYPE, metadata=dict(metadata))
        artifact.add_dir(sessions_dir)
        self.run().log_artifact(artifact)
        return name


def _artifact_safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


def _json_default(value: Any) -> Any:
    """Make numpy / jax scalars and paths serialisable."""
    for attribute in ("item", "__fspath__"):
        method = getattr(value, attribute, None)
        if callable(method):
            try:
                return method()
            except Exception:
                pass
    return str(value)

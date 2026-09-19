"""Output sinks for performance records.

Weights & Biases is the system of record: per-epoch scalars go to the run
history (on their own x-axis, ``performance/environment_steps``), the summary
goes to ``run.summary`` so it shows up as columns in the runs table, and the
profiler trace is uploaded as an artifact of type ``xprof-trace``.

A JSON copy is always written next to the run's local output as well and
uploaded to the run's files.  It costs nothing and makes a run inspectable
without network access (e.g. on a cluster compute node before ``wandb sync``).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Optional

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
    """Forwards to the active ``wandb`` run.

    Silently does nothing when no run is active, so unit tests and library code
    never need to care.  The entry points are responsible for guaranteeing a
    run exists (see ``run_utils.require_wandb_login``).
    """

    PREFIX = "performance/"
    STEP_METRIC = PREFIX + "environment_steps"

    def __init__(self):
        self._defined_step_metric = False

    @staticmethod
    def run():
        try:
            import wandb
        except ImportError:
            return None
        return wandb.run

    def log_scalars(self, scalars: Mapping[str, float], environment_steps: int) -> None:
        run = self.run()
        if run is None:
            return
        if not self._defined_step_metric:
            # Own x-axis so these never collide with the trainer's own step counter.
            run.define_metric(self.STEP_METRIC)
            run.define_metric(self.PREFIX + "*", step_metric=self.STEP_METRIC)
            self._defined_step_metric = True
        payload = {self.PREFIX + key: value for key, value in scalars.items()}
        payload[self.STEP_METRIC] = environment_steps
        run.log(payload)

    def set_summary(self, summary: Mapping[str, Any]) -> None:
        run = self.run()
        if run is None:
            return
        for key, value in summary.items():
            if value is not None:
                run.summary[self.PREFIX + key] = value

    def save_file(self, path: str) -> None:
        run = self.run()
        if run is None:
            return
        run.save(path, base_path=os.path.dirname(os.path.abspath(path)), policy="now")

    def log_trace_artifact(
        self, sessions_dir: str, run_name: str, metadata: Optional[Mapping[str, Any]] = None
    ) -> Optional[str]:
        """Upload ``sessions_dir`` (XProf's ``plugins/profile`` folder) as one artifact.

        Returns the artifact name, or None if no run is active.
        """
        run = self.run()
        if run is None:
            return None
        import wandb

        name = f"trace-{_artifact_safe(run_name)}"
        artifact = wandb.Artifact(name=name, type=TRACE_ARTIFACT_TYPE, metadata=dict(metadata or {}))
        artifact.add_dir(sessions_dir)
        run.log_artifact(artifact)
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

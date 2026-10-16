"""The HTTP server behind the review page. It reads records and renders frames;
it computes nothing about an episode (``docs/acl/design/episode_review.md`` §1).

Routes:

    GET /                                  the page
    GET /static/<file>                     its script and style
    GET /api/library                       the group, every episode's summary, each environment's control dt and cameras,
                                           and the policy views (image parts) the records hold
    GET /api/episode/<id>                  one episode's per-step traces and facts
    GET /frame/<id>/<step>?camera=<name>   one frame: the scene from a model camera (JPEG), or, when ``name`` is an
                                           image part of the record, what the policy saw (PNG)
"""
from __future__ import annotations

import functools
import http.server
import json
import pathlib
import urllib.parse
from typing import Any, Dict, Tuple

from training.evaluation.record import EpisodeRecord
from training.review.library import Library
from training.review.policy_view import policy_view
from training.review.renderer import DEFAULT_CAMERA, Renderers

STATIC = pathlib.Path(__file__).parent / "static"
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}
RECORD_CACHE_SIZE = 8  # whole records held in memory; a record is ~1 MB
FRAME_CACHE_SIZE = 4096  # JPEG frames, ~10 KB each


class ReviewServer(http.server.ThreadingHTTPServer):
    """Owns the library and the renderers; the handler reads them."""

    daemon_threads = True

    def __init__(self, address: Tuple[str, int], library: Library):
        super().__init__(address, _Handler)
        self.library = library
        self.renderers = Renderers()
        self.record = functools.lru_cache(maxsize=RECORD_CACHE_SIZE)(library.record)
        self.frame = functools.lru_cache(maxsize=FRAME_CACHE_SIZE)(self._render)

    def _render(self, index: int, step: int, camera: str) -> Tuple[bytes, str]:
        """The frame and its content type: the policy's own view if ``camera`` names one of the
        record's image parts, else the scene from the model's camera."""
        record: EpisodeRecord = self.record(index)
        if camera in record.images:
            return policy_view(record, step, camera), "image/png"
        return self.renderers.for_environment(record.env_name).frame(record, step, camera), "image/jpeg"

    def library_json(self) -> Dict[str, Any]:
        environments = {}
        for env_name in self.library.env_names:
            renderer = self.renderers.for_environment(env_name)
            environments[env_name] = {"control_dt": renderer.control_dt, "cameras": list(renderer.cameras)}
        return {
            "group": self.library.group,
            "episodes": [episode.as_json() for episode in self.library.episodes],
            "environments": environments,
            "policy_views": sorted({part for episode in self.library.episodes for part in episode.image_parts}),
            "default_camera": DEFAULT_CAMERA,
        }

    def episode_json(self, index: int) -> Dict[str, Any]:
        record: EpisodeRecord = self.record(index)
        per_step = {
            name: getattr(record, name).tolist()
            for name in ("reward", "cost", "value", "cost_value", "lagrange_multiplier",
                         "reward_advantage", "cost_advantage", "lagrangian_advantage")
        }
        return {
            "index": index,
            "env_name": record.env_name,
            "run_name": record.run_name,
            "arm": record.arm,
            "evaluation_name": record.evaluation_name,
            "environment_steps": record.environment_steps,
            "episode_index": record.episode_index,
            "length": record.length,
            "context": record.context.tolist(),
            "episode_return": record.episode_return,
            "episode_cost": record.episode_cost,
            "action": record.action.tolist(),
            **per_step,
        }


class _Handler(http.server.BaseHTTPRequestHandler):
    server: ReviewServer

    def do_GET(self) -> None:  # noqa: N802  (http.server's name)
        url = urllib.parse.urlparse(self.path)
        parts = [part for part in url.path.split("/") if part]
        query = urllib.parse.parse_qs(url.query)
        try:
            if not parts:
                self._send_file(STATIC / "index.html")
            elif parts[0] == "static" and len(parts) == 2:
                self._send_file(STATIC / parts[1])
            elif parts == ["api", "library"]:
                self._send_json(self.server.library_json())
            elif parts[:2] == ["api", "episode"] and len(parts) == 3:
                self._send_json(self.server.episode_json(int(parts[2])))
            elif parts[0] == "frame" and len(parts) == 3:
                camera = query.get("camera", [DEFAULT_CAMERA])[0]
                self._send_bytes(*self.server.frame(int(parts[1]), int(parts[2]), camera))
            else:
                self.send_error(404, f"no route for {url.path}")
        except (IndexError, ValueError, FileNotFoundError) as error:
            self.send_error(400, str(error))

    def _send_file(self, path: pathlib.Path) -> None:
        if not path.is_file() or path.parent != STATIC:
            raise FileNotFoundError(f"no static file {path.name}")
        self._send_bytes(path.read_bytes(), CONTENT_TYPES[path.suffix])

    def _send_json(self, payload: Dict[str, Any]) -> None:
        self._send_bytes(json.dumps(payload).encode(), "application/json")

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002  (http.server's signature)
        pass  # frames arrive many per second; the console stays for errors

"""``python -m training.review --group <wandb group>``: serve the review page for a group's episode records."""
from __future__ import annotations

import argparse
import pathlib
import webbrowser

from training.config import WANDB_ENTITY, WANDB_PROJECT
from training.review.library import Library, group_record_directories
from training.review.server import ReviewServer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, help="the W&B group whose runs' episode records to review")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--wandb_dir", type=pathlib.Path, default=pathlib.Path("wandb"),
                        help="the local W&B directory; a run that trained on this machine is read from here, live")
    parser.add_argument("--cache_dir", type=pathlib.Path, default=pathlib.Path("runs/review"),
                        help="where records of runs not on this machine are downloaded to")
    parser.add_argument("--entity", default=WANDB_ENTITY)
    parser.add_argument("--project", default=WANDB_PROJECT)
    parser.add_argument("--no_browser", action="store_true")
    arguments = parser.parse_args()

    directories = group_record_directories(arguments.entity, arguments.project, arguments.group, arguments.wandb_dir, arguments.cache_dir)
    library = Library.from_directories(arguments.group, directories)
    print(f"{len(library.episodes)} recorded episodes from {len(directories)} runs of {arguments.group!r}")
    server = ReviewServer(("127.0.0.1", arguments.port), library)
    url = f"http://127.0.0.1:{arguments.port}/"
    print(f"review: {url}  (Ctrl-C to stop)")
    if not arguments.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.renderers.close()


if __name__ == "__main__":
    main()

import argparse
import os
from pathlib import Path
from typing import Optional, Set

import wandb
from wandb.apis.public import Run

from results import cli
from results.common import apply_max_age_filter, get_metrics_for_env, results_path


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    filters = build_filters(args)
    runs = api.runs(args.project, filters=filters, order="-created_at", per_page=200)

    # Runs come newest-first. The `written` field keeps a re-run of the same cell from
    # being overwritten by an older duplicate of itself.
    written: Set[Path] = set()
    for run in runs:
        store_data(run, args, written)


def build_filters(args: argparse.Namespace) -> dict:
    """Server-side filters for wandb.Api().runs()."""
    f = {"state": {"$in": list(args.states)}}

    # config.* filters
    if args.algos:
        f["config.alg"] = {"$in": args.algos}
    if args.envs:
        f["config.env_name"] = {"$in": args.envs}
    if args.levels:
        f["config.difficulty"] = {"$in": args.levels}
    if args.seeds:
        f["config.seed"] = {"$in": args.seeds}

    # Observation modality. Runs predating the --vision flag have no `vision` key at all,
    # so "vector" has to match a missing key as well as an explicit False. $ne does both.
    obs = getattr(args, "obs", "any")
    if obs == "vision":
        f["config.vision"] = True
    elif obs == "vector":
        f["config.vision"] = {"$ne": True}
    if getattr(args, "vision_cameras", None):
        f["config.vision_camera"] = {"$in": args.vision_cameras}

    # only runs created within the last `max_age_days` days
    apply_max_age_filter(f, args)

    # tags live on the run, not in config
    if args.wandb_tags:
        f["tags"] = {"$in": args.wandb_tags}

    # include specific runs by name (display_name) as an OR clause
    # if include_runs is set, we *add* them even if they don't match other filters
    if args.include_runs:
        ors = [{"display_name": {"$in": args.include_runs}}]
        # $or coexists with the ANDed top-level filters:
        f = {"$or": [f, *ors]}

    return f


def obs_mode_segment(config: dict) -> str:
    """Path segment separating a run's observation modality, '' for vector obs."""
    if not config.get("vision"):
        return ""
    return f"vision_{config.get('vision_camera') or 'unknown'}"


def store_data(run: Run, args: argparse.Namespace,
               written: Optional[Set[Path]] = None) -> None:
    """Write one run's history to its per-seed parquet."""
    config = run.config
    run_id = run.id
    seed = config['seed']
    env = config['env_name']
    level = config['difficulty']
    algo = config['alg']
    extra_attribute = ''

    metrics = get_metrics_for_env(env, args.metrics)

    attribute_key = args.extra_attribute
    if attribute_key:
        if attribute_key not in config:
            raise ValueError(f"Extra attribute_key '{attribute_key}' not found in run config.")
        attribute_val = config[attribute_key]
        extra_attribute = f"{attribute_key}_{attribute_val}"

    # Construct folder path for each configuration
    folder_path = results_path(
        args.output, env, f"level_{level}", algo, obs_mode_segment(config), extra_attribute
    )
    os.makedirs(folder_path, exist_ok=True)  # Ensure the directory exists

    file_path = folder_path / f"seed_{seed}.parquet"

    # An earlier run for the given criteria already exists
    if written is not None and file_path in written:
        print(f"Skipping older duplicate of {file_path.name}: run {run_id}")
        return

    # Skip if file already exists and overwrite flag not set
    if file_path.exists() and not args.overwrite:
        print(f"Skipping existing file: {file_path}")
        return

    try:
        df = run.history(keys=metrics)
        if df is None or df.empty:
            print(f"No history for run {run_id}")
            return
        df.to_parquet(file_path)
        if written is not None:
            written.add(file_path)
        print(f"Saved data for run {run_id} to {file_path}")
    except Exception as e:
        print(f"Error downloading data for run: {run_id}: {e}")
        return


def common_dl_args() -> argparse.ArgumentParser:
    """Parser shared by the download scripts that reuse `build_filters`/`store_data`."""
    parser = cli.download_parser("Download benchmark results from WandB.")
    parser.add_argument("--extra_attribute", type=str, default=None,
                        help="Config attribute to store data by")
    parser.add_argument("--obs", type=str, default="any", choices=("any", "vector", "vision"),
                        help="Observation modality to download")
    parser.add_argument("--vision_cameras", type=str, nargs="+", default=None,
                        help="Restrict pixel runs to these cameras (e.g. vision fixedfar track).")
    return parser


if __name__ == "__main__":
    parser = common_dl_args()
    main(parser.parse_args())

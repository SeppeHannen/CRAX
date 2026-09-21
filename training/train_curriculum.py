#!/usr/bin/env python
"""Curriculum training across difficulty levels.

This script trains an agent sequentially across multiple difficulty levels,
with the policy warm-started from the previous level.

"""

import functools
import os
from datetime import datetime
from pathlib import Path

import wandb

from training import curriculum
from training.config import build_base_parser
from training.run_utils import (
    setup_gpu_environment, get_algorithm_train_fn, filter_kwargs_for_fn,
    wandb_progress_fn, record_episode_video, make_vision_network_factory,
    morphology_override, VISION_CAMERA_OVERRIDES, install_performance_tracker,
    require_wandb_login,
)


def main():
    parser = build_base_parser(description='Curriculum training across difficulty levels')
    parser.add_argument('--levels', type=int, nargs='+', default=[1, 2, 3], help='Difficulty levels to train on')
    config = parser.parse_args()

    alg_name = config.alg
    env_name = config.env_name

    # Fill in the morphology-specific pixel-obs training camera
    if config.vision_camera is None:
        config.vision_camera = morphology_override(env_name, VISION_CAMERA_OVERRIDES) or 'vision'

    # Setup GPU environment
    setup_gpu_environment(vision=config.vision)
    # Every run is tracked in W&B; abort before compiling anything if credentials are missing
    require_wandb_login()

    # Run training for each seed
    for seed in config.seeds:
        # Create curriculum stages
        stages = curriculum.create_difficulty_curriculum(
            env_name=env_name,
            levels=config.levels,
            steps_per_level=int(config.num_timesteps // len(config.levels)),
        )

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        cli_cfg = vars(config)
        runtime_cfg = {"seed": seed, "timestamp": timestamp}
        cfg = {**cli_cfg, **runtime_cfg}
        run_name = f"{env_name}_curriculum_{alg_name}_seed{seed}_{timestamp}"

        print(f"\n{'=' * 50}")
        print(f"Running experiment {run_name}")
        print(f"{'=' * 50}\n")
        print(f"\nCurriculum stages:")
        for i, stage in enumerate(stages):
            print(f"  {i + 1}. {stage.env_name}: {stage.num_steps:,} steps")

        wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=run_name,
            id=run_name,
            config=cfg.copy(),
            group=config.wandb_group if config.wandb_group else env_name,
            job_type=alg_name,
            tags=config.wandb_tags,
        )

        if config.store_model:
            root_dir = Path(__file__).parent.parent.resolve()  # repo root, not training/
            ckpt_root = root_dir / config.model_dir / run_name
            os.makedirs(ckpt_root, exist_ok=True)
            cfg["save_checkpoint_path"] = ckpt_root

        progress_fn = wandb_progress_fn(safety_bound=config.safety_bound, verbose=not config.quiet)

        # Get the appropriate training function
        train_fn_base = get_algorithm_train_fn(alg_name)
        train_kwargs = filter_kwargs_for_fn(train_fn_base, cfg)

        # Add vision support
        if config.vision:
            vision_kwargs = dict(
                cameras=config.vision_cameras,
                height=config.vision_height,
                width=config.vision_width,
                obs_mode=config.vision_obs_mode,
                frame_stack=config.vision_frame_stack,
                grayscale=config.vision_grayscale,
                num_render_workers=config.vision_render_workers,
            )
            train_kwargs['vision'] = True
            train_kwargs['vision_kwargs'] = vision_kwargs
            state_obs_key = 'state' if config.vision_obs_mode == 'pixels+state' else ''
            train_kwargs['network_factory'] = make_vision_network_factory(
                alg_name,
                policy_obs_key=state_obs_key,
                value_obs_key=state_obs_key,
            )
            train_kwargs['augment_pixels'] = config.vision_augment

        # Optional performance measurement (--measure_performance / --profile_epochs).
        # One tracker spans all stages, so recompiles at stage boundaries show up
        # in the compile count and per-epoch records.
        performance_tracker = install_performance_tracker(config, run_name, progress_fn)

        # Train with curriculum
        policy_fn, final_params, results, eval_env = curriculum.train_curriculum(
            stages=stages,
            train_fn=train_fn_base,
            train_kwargs=train_kwargs,
            progress_fn=progress_fn,
            seed=seed,
        )
        performance_tracker.finish()

        # Print final summary
        print("\n" + "=" * 60)
        print("CURRICULUM TRAINING SUMMARY")
        print("=" * 60)
        for result in results:
            print(f"\nStage {result.stage_idx + 1}: {result.env_name}")
            print(f"  Steps: {result.num_steps:,}")
            print(f"  Time: {result.training_time:.1f}s")
            if 'eval/episode_reward' in result.final_metrics:
                print(f"  Final reward: {result.final_metrics['eval/episode_reward']:.2f}")
            if 'eval/episode_cost' in result.final_metrics:
                print(f"  Final cost: {result.final_metrics['eval/episode_cost']:.2f}")

        if not config.skip_video:
            video_length = config.video_length if config.video_length else config.episode_length
            if video_length is None:
                video_length = getattr(eval_env, 'episode_length', None)
            record_episode_video(
                env=eval_env,
                make_inference_fn=policy_fn,
                params=final_params,
                steps=video_length,
                cameras=config.cameras,
                width=config.video_width,
                height=config.video_height,
                fps=config.video_fps,
                frame_stride=config.video_frame_stride,
                out_name=run_name,
                seed=seed,
                num_episodes=config.num_video_episodes,
            )

        wandb.finish()

    print("\nAll experiments completed!")


if __name__ == '__main__':
    main()

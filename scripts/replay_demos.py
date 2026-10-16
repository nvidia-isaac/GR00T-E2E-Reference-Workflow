#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replay demonstrations recorded by ``record_demos.py``.

Loads an Isaac Lab HDF5 dataset and plays each recorded episode back in the same
static-apple scene for visual review -- no teleop device or headset needed.
Recorded-state playback renders the saved scene snapshots without advancing
controllers or physics. Action replay re-applies commands through the live
controller and physics pipeline, so it may diverge in contact-rich scenes.

Run with ``--viz kit`` to open a viewport; see ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from loguru import logger

# Register Pinocchio's eigenpy bindings before Isaac Sim loads (Isaac Lab #4090).
from e2e_robot_workflows.compat import warmup_pinocchio_bindings
from e2e_robot_workflows.data.dataset_manifest import (
    ACTION_PROFILES,
    NATIVE_AGILE_PROFILE,
    NATIVE_FIXED_PROFILE,
    SCENE_RANDOMIZATION_FIELD,
    DatasetContractError,
    resolve_recording_lane,
    validate_dataset,
)

warmup_pinocchio_bindings()

from isaaclab.app import AppLauncher

from e2e_robot_workflows.data.artifact_paths import atomic_write_text, require_distinct_artifact_paths
from e2e_robot_workflows.data.replay import (
    assert_recorded_state_restored,
    make_state_renderer,
    replay_recorded_states,
    reset_replay_episode,
)
from e2e_robot_workflows.tasks.static_apple.spec import GYM_ID as DEFAULT_TASK
from e2e_robot_workflows.tasks.static_apple.table_height import HEIGHT_TOLERANCE_M

parser = argparse.ArgumentParser(description="Replay recorded E2E Robot Workflows demonstrations.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Gym task id to replay in.")
parser.add_argument("--dataset_file", type=str, required=True, help="Recorded HDF5 dataset to replay.")
parser.add_argument(
    "--lane",
    choices=("native", "arena"),
    default=None,
    help="Raw replay owner. Defaults to native; the dataset manifest must agree.",
)
parser.add_argument("--arena", action="store_true", help="Compatible alias for --lane arena.")
parser.add_argument(
    "--wbc-mode",
    choices=("fixed", "agile"),
    default=None,
    help="Lower-body mode; must match the mode the dataset was recorded with.",
)
parser.add_argument(
    "--select_episodes",
    type=str,
    default=None,
    help="Comma-separated episode names (e.g. demo_0,demo_3) or 0-based indices. Defaults to all.",
)
parser.add_argument("--num_replays", type=int, default=1, help="Times to loop through the selected episodes.")
parser.add_argument(
    "--seed", type=int, default=None, help="Override the recorded reset seed; scene state is still restored."
)
parser.add_argument(
    "--expected_episodes",
    type=int,
    help="Fail unless the dataset contains exactly this many episodes.",
)
parser.add_argument(
    "--replay-mode",
    choices=("action", "state"),
    default="action",
    help="Replay live actions or render recorded scene snapshots. Defaults to action.",
)
parser.add_argument(
    "--strict",
    action="store_true",
    help="Fail unless every selected episode's requested replay stream is consumed in full.",
)
parser.add_argument("--result_file", type=Path, default=None, help="Optional structured replay result JSON.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

dataset_path = Path(args_cli.dataset_file).expanduser()
if not dataset_path.is_file():
    parser.error(f"Dataset file does not exist: {dataset_path}")
if args_cli.result_file is not None:
    try:
        require_distinct_artifact_paths(dataset_path, args_cli.result_file, destination_label="Replay result file")
    except ValueError as error:
        parser.error(str(error))
try:
    args_cli.lane = resolve_recording_lane(args_cli.lane, arena_alias=args_cli.arena)
    dataset_manifest = validate_dataset(dataset_path, require_lane=args_cli.lane)
except DatasetContractError as error:
    parser.error(str(error))
action_profile = ACTION_PROFILES[dataset_manifest["action"]["layout"]]
# Admission guarantees a non-null block with complete table state.
RECORDED_MOVABLE_SUPPORT = SCENE_RANDOMIZATION_FIELD in dataset_manifest["environment"]
replay_seed = args_cli.seed if args_cli.seed is not None else dataset_manifest["seed"]
if replay_seed is None:
    replay_seed = 42
if replay_seed < 0:
    parser.error("--seed must be non-negative")
if args_cli.lane == "arena" and args_cli.task != DEFAULT_TASK:
    parser.error("Arena replay composes the static-apple environment itself; drop --task.")
if args_cli.lane == "arena" and args_cli.wbc_mode is not None:
    parser.error("--wbc-mode configures the native lane and cannot be used with --lane arena.")
if args_cli.task != DEFAULT_TASK and args_cli.wbc_mode is not None:
    parser.error("--wbc-mode is supported only for the built-in static-apple task.")
if args_cli.lane == "native":
    mode_by_layout = {
        NATIVE_FIXED_PROFILE.layout: "fixed",
        NATIVE_AGILE_PROFILE.layout: "agile",
    }
    inferred_mode = mode_by_layout.get(action_profile.layout)
    if inferred_mode is None:
        parser.error(f"Unsupported native replay action layout: {action_profile.layout}")
    if args_cli.wbc_mode is not None and args_cli.wbc_mode != inferred_mode:
        parser.error(
            f"Dataset manifest selects native {inferred_mode}, but --wbc-mode {args_cli.wbc_mode} was requested."
        )
    args_cli.wbc_mode = inferred_mode
if args_cli.num_replays < 1:
    parser.error("--num_replays must be a positive integer.")
if args_cli.expected_episodes is not None and args_cli.expected_episodes < 1:
    parser.error("--expected_episodes must be a positive integer.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.datasets import HDF5DatasetFileHandler

import e2e_robot_workflows.tasks.static_apple  # noqa: F401 - registers the gym task
from e2e_robot_workflows.tasks.static_apple.env_cfg import make_env_cfg


def build_env(wbc_mode: str | None) -> gym.Env:
    """Build the manifest-selected raw replay owner with recording disabled."""

    env_kwargs = {}
    task_id = args_cli.task
    if args_cli.lane == "arena":
        if not action_profile.autodata_compatible or (
            action_profile.controller.startswith("h2_") and args_cli.replay_mode == "state"
        ):
            from e2e_robot_workflows.arena.composition import compose

            composed = compose(
                embodiment=action_profile.controller,
                device=args_cli.device,
                num_envs=1,
                enable_cameras=bool(args_cli.enable_cameras),
                seed=replay_seed,
                movable_support=RECORDED_MOVABLE_SUPPORT,
            )
        else:
            from e2e_robot_workflows.arena.mimic import compose_arena_mimic_environment

            composed = compose_arena_mimic_environment(
                device=args_cli.device,
                num_envs=1,
                enable_cameras=bool(args_cli.enable_cameras),
                manipulation_hand=dataset_manifest["task"]["manipulation_hand"],
                seed=replay_seed,
                embodiment=action_profile.controller,
                movable_support=RECORDED_MOVABLE_SUPPORT,
            )
        task_id, env_cfg, env_kwargs = composed.task_id, composed.env_cfg, composed.env_kwargs
    elif args_cli.task == DEFAULT_TASK:
        env_cfg = make_env_cfg(wbc_mode=wbc_mode, device=args_cli.device, num_envs=1)
    else:
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = replay_seed
    # Replay only reads back recorded actions; unlike record_demos.py it attaches no
    # dataset recorder, so no HDF5 is written. Set explicitly rather than relying on
    # the config default.
    env_cfg.recorders = None
    # Replay drives the scene directly; disable auto-termination so an episode is
    # never cut short while we step through it.
    for term, value in vars(env_cfg.terminations).items():
        if isinstance(value, TerminationTermCfg):
            setattr(env_cfg.terminations, term, None)
    env_cfg.observations.policy.concatenate_terms = False
    if args_cli.lane == "arena" and action_profile.controller == "h2_agile_joint" and args_cli.replay_mode == "action":
        from e2e_robot_workflows.arena.mimic import H2ReplayEnv

        return H2ReplayEnv(cfg=env_cfg, **env_kwargs)
    return gym.make(task_id, cfg=env_cfg, **env_kwargs).unwrapped


def select_episode_names(all_names: list[str]) -> list[str]:
    """Resolve --select_episodes into concrete episode names, preserving order."""

    if not args_cli.select_episodes:
        return all_names
    chosen: list[str] = []
    for token in (t.strip() for t in args_cli.select_episodes.split(",") if t.strip()):
        if token in all_names:
            chosen.append(token)
        elif token.isdigit() and int(token) < len(all_names):
            chosen.append(all_names[int(token)])
        else:
            if args_cli.strict:
                raise ValueError(f"Episode {token!r} was not found in the dataset.")
            logger.warning("Episode '{}' not found in the dataset; skipping.", token)
    if not chosen:
        logger.warning("--select_episodes matched no episodes in the dataset.")
    return chosen


def assert_support_restored(env: gym.Env, initial_state) -> None:
    """A recorded support is kinematic and must stay at its restored pose with zero velocity."""

    if RECORDED_MOVABLE_SUPPORT:
        support = dataset_manifest["environment"][SCENE_RANDOMIZATION_FIELD]["support_asset"]
        assert_recorded_state_restored(
            env.scene.get_state(is_relative=True),
            {"rigid_object": {support: initial_state["rigid_object"][support]}},
            atol=HEIGHT_TOLERANCE_M,
        )


def replay_action(env: gym.Env, episode) -> tuple[int, int]:
    """Reset and re-apply recorded actions; return consumed and expected steps."""

    initial_state = episode.get_initial_state()
    if initial_state is None:
        raise RuntimeError("Loaded episode has no initial state")
    reset_replay_episode(env, initial_state)
    assert_support_restored(env, initial_state)
    actions = episode.data.get("actions")
    if actions is None:
        raise RuntimeError("Loaded episode has no actions")
    expected_steps = len(actions)
    steps = 0
    with torch.inference_mode():
        action = episode.get_next_action()
        while action is not None and simulation_app.is_running():
            env.step(action.reshape(1, -1).to(env.device))
            assert_support_restored(env, initial_state)
            steps += 1
            action = episode.get_next_action()
    return steps, expected_steps


def main() -> None:
    logger.info(
        "Replaying {} on the {} lane ({}) in {} mode",
        dataset_path,
        args_cli.lane,
        action_profile.layout,
        args_cli.replay_mode,
    )
    if args_cli.replay_mode == "state" and args_cli.enable_cameras:
        logger.warning(
            "State mode restores scene snapshots for viewport review; it does not replay the dataset's saved pixels."
        )
    handler = HDF5DatasetFileHandler()
    env = None
    result: dict[str, object] = {
        "status": "error",
        "dataset": str(dataset_path),
        "lane": args_cli.lane,
        "action_layout": action_profile.layout,
        "replay_mode": args_cli.replay_mode,
        "seed": replay_seed,
        "episodes": [],
        "skipped_episodes": [],
    }
    # Guarantee the dataset handle, env, and Sim app are torn down on every path
    # (a failed open, no-episodes exit, or an unexpected error in the replay loop).
    try:
        handler.open(str(dataset_path))
        all_names = list(handler.get_episode_names())
        result["dataset_episode_count"] = len(all_names)
        if args_cli.expected_episodes is not None and len(all_names) != args_cli.expected_episodes:
            raise RuntimeError(f"Dataset contains {len(all_names)} episodes, expected {args_cli.expected_episodes}.")
        if not all_names:
            raise SystemExit(f"No episodes found in {dataset_path}")
        names = select_episode_names(all_names)
        if not names:
            raise SystemExit("No episodes selected to replay.")
        logger.info("Dataset env: {} | {} episode(s), replaying {}", handler.get_env_name(), len(all_names), len(names))

        env = build_env(args_cli.wbc_mode)
        env.reset()
        render_frame = make_state_renderer(env) if args_cli.replay_mode == "state" else None

        with contextlib.suppress(KeyboardInterrupt):
            for loop in range(args_cli.num_replays):
                for name in names:
                    if not simulation_app.is_running():
                        break
                    episode = handler.load_episode(name, args_cli.device)
                    if episode is None:
                        logger.warning("Could not load episode '{}'; skipping.", name)
                        result["skipped_episodes"].append(name)
                        continue
                    if args_cli.replay_mode == "state":
                        count, expected_count = replay_recorded_states(
                            env, episode, simulation_app.is_running, render_frame
                        )
                    else:
                        count, expected_count = replay_action(env, episode)
                    assert_support_restored(env, episode.get_initial_state())
                    result["episodes"].append(
                        {
                            "name": name,
                            "pass": loop + 1,
                            "frames": count,
                            "expected_frames": expected_count,
                            "complete": count == expected_count,
                        }
                    )
                    logger.info(
                        "Replayed {} ({}/{} frames) [pass {}/{}]",
                        name,
                        count,
                        expected_count,
                        loop + 1,
                        args_cli.num_replays,
                    )
        expected_replays = len(names) * args_cli.num_replays
        result["selected_episode_count"] = len(names)
        result["expected_replays"] = expected_replays
        result["completed_replays"] = len(result["episodes"])
        result["total_frames"] = sum(episode["frames"] for episode in result["episodes"])
        if args_cli.strict and result["completed_replays"] != expected_replays:
            raise RuntimeError(
                f"Strict replay completed {result['completed_replays']} of {expected_replays} selected episode passes."
            )
        incomplete = [episode for episode in result["episodes"] if not episode["complete"]]
        if args_cli.strict and incomplete:
            first = incomplete[0]
            raise RuntimeError(
                f"Strict replay stopped after {first['frames']} of {first['expected_frames']} frames "
                f"for {first['name']!r}."
            )
        result["status"] = "ok"
    except BaseException as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        cleanup_errors = []
        try:
            handler.close()
        except Exception as error:  # noqa: BLE001 - preserve cleanup evidence in the structured result
            cleanup_errors.append(f"dataset: {type(error).__name__}: {error}")
        if env is not None:
            try:
                env.close()
            except Exception as error:  # noqa: BLE001 - preserve cleanup evidence in the structured result
                cleanup_errors.append(f"environment: {type(error).__name__}: {error}")
        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
            result["status"] = "error"
        if args_cli.result_file is not None:
            atomic_write_text(args_cli.result_file, json.dumps(result, indent=2, sort_keys=True) + "\n")
        if cleanup_errors and sys.exc_info()[0] is None:
            raise RuntimeError(f"Replay cleanup failed: {cleanup_errors}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

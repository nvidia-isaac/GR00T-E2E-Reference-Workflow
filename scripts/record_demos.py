#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 The Isaac Lab Project Developers. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
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

"""Record demos for the standalone E2E Robot Workflows tasks.

This is recording-first and supports both the built-in and Arena runtimes. When the task registers a
completion policy (its ``completion_policy_entry_point``), recording runs
continuously: each episode is auto-saved with the policy's measured outcome and
the scene resets without stopping the session. Pass ``--no-auto_success`` for
purely operator-accepted episodes. The recorder itself stays task-agnostic -- the
success/failure semantics live with the task. NVIDIA Isaac AutoData and richer
evaluation metrics can be layered later without being required for teleop data
collection.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import platform
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from loguru import logger

# Register Pinocchio's eigenpy bindings before Isaac Sim loads (Isaac Lab #4090).
# Must run before `from isaaclab.app import AppLauncher`.
from e2e_robot_workflows.compat import warmup_pinocchio_bindings
from e2e_robot_workflows.data.artifact_paths import (
    recorder_hdf5_output_path,
    require_distinct_artifact_paths,
)
from e2e_robot_workflows.data.dataset_manifest import (
    DatasetContractError,
    default_recording_dataset_file,
    resolve_recording_lane,
)
from e2e_robot_workflows.data.replay import make_state_renderer
from e2e_robot_workflows.devices.recording_runtime import (
    RecordingControls,
    XrRuntimeTelemetry,
    admit_h2_recording_episode,
    align_h2_recording_action,
    close_recording_dataset,
    refresh_reset_view,
    require_arena_teleop_mode,
    require_h2_recording_cameras,
    require_xr_runtime_metrics_mode,
    step_recording_environment,
    xr_recording_completion_notice,
)
from e2e_robot_workflows.tasks.static_apple.table_height import parse_offset_range, scene_randomization_manifest

warmup_pinocchio_bindings()

from isaaclab.app import AppLauncher

from e2e_robot_workflows.tasks.static_apple.spec import DEFAULT_WBC_MODE, GYM_ID as DEFAULT_TASK

XR_DEVICE_ALIASES = {
    "xr": "handtracking",
    "openxr": "handtracking",
    "handtracking": "handtracking",
    "manusvive": "manusvive",
    "manus_vive": "manusvive",
    "controllers": "motion_controllers",
    "motion_controllers": "motion_controllers",
}

parser = argparse.ArgumentParser(description="Record E2E Robot Workflows demonstrations with NVIDIA Isaac Teleop.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Gym task id to record.")
parser.add_argument(
    "--wbc-mode",
    choices=("fixed", "agile"),
    default=None,
    help="Lower-body mode for the built-in static-apple task. Defaults to the task config.",
)
parser.add_argument(
    "--recording-lane",
    choices=("native", "arena"),
    default=None,
    help="Explicit recording backend. Defaults to native; --arena remains a compatible alias.",
)
parser.add_argument(
    "--arena",
    action="store_true",
    help="Compatible alias for --recording-lane arena.",
)
parser.add_argument(
    "--arena_embodiment",
    type=str,
    default=None,
    help="Arena embodiment to compose with --arena. Defaults to the teleoperation-capable Pink variant.",
)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="handtracking",
    help="Teleop device: handtracking/xr/openxr, motion_controllers/controllers, manusvive, keyboard, or spacemouse.",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default=None,
    help="Output HDF5 path. Default: ./datasets/e2e_ra_dataset_<lane>_<timestamp>.hdf5.",
)
parser.add_argument("--step_hz", type=int, default=30, help="Interactive stepping rate in Hz.")
parser.add_argument("--seed", type=int, default=42, help="Environment and dataset provenance seed.")
parser.add_argument(
    "--manipulation-hand",
    choices=("auto", "left", "right"),
    default="auto",
    help="Hand responsible for the transfer. Native auto verifies it from the saved trajectory; Arena defaults right.",
)
parser.add_argument("--num_demos", type=int, default=0, help="Number of demos to record. Use 0 for infinite.")
parser.add_argument("--start_paused", action="store_true", help="Start with physics rendering but recording paused.")
parser.add_argument(
    "--h2-station-keep",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Bounded XY/yaw assistance with neutral sticks during H2 XR recording (enabled by default).",
)
parser.add_argument(
    "--sample_steps",
    type=int,
    default=0,
    help="Launch the recorder, execute this many deterministic sample steps, and exit.",
)
parser.add_argument(
    "--sample_action",
    choices=("hold", "wrist_sine"),
    default="hold",
    help="Action used by --sample_steps. wrist_sine adds a small deterministic wrist motion.",
)
parser.add_argument(
    "--sample_export",
    action="store_true",
    help="Export the finite sample as one unsuccessful smoke-test episode.",
)
parser.add_argument(
    "--sample_performance_result",
    type=Path,
    default=None,
    help="Optional JSON timing report for a finite sample; excludes startup, reset, and export from step rate.",
)
parser.add_argument(
    "--xr_runtime_metrics",
    type=Path,
    default=None,
    help="Optional atomic host-wall timing report for an interactive XR recording session.",
)
parser.add_argument(
    "--environment_snapshot",
    type=Path,
    default=None,
    help="Optional JSON path for the shared contract and realized recording environment.",
)
parser.add_argument(
    "--controller_profiles_config",
    type=Path,
    default=None,
    help="Controller profile YAML config.",
)
parser.add_argument(
    "--controller_profile",
    type=str,
    default=None,
    help="Controller profile to use. Defaults to default_record_profile in the profile config.",
)
parser.add_argument(
    "--dataset_export_mode",
    choices=("all", "succeeded"),
    default="succeeded",
    help="Export only successful episodes by default; pass 'all' to also keep failed episodes.",
)
parser.add_argument(
    "--auto_success",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
        "Auto-end and label an episode from the task's completion policy (success/failure), instead of the "
        "operator asserting success on save. Defaults to the task policy's setting; no effect for tasks without one."
    ),
)
parser.add_argument(
    "--completion_override",
    action="append",
    metavar="KEY=VALUE",
    default=None,
    help="Task-namespaced completion-policy tuning, e.g. --completion_override far_distance_m=0.4 (repeatable).",
)
parser.add_argument(
    "--table_height_offset_range_m",
    nargs=2,
    type=float,
    metavar=("MIN", "MAX"),
    default=None,
    help=(
        "Randomize the tabletop height per episode by MIN..MAX metres around the robot's task variant. "
        "Omitted (or '0 0') records the nominal fixed-height scene."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

try:
    args_cli.recording_lane = resolve_recording_lane(args_cli.recording_lane, arena_alias=args_cli.arena)
except DatasetContractError as error:
    parser.error(str(error))
args_cli.arena = args_cli.recording_lane == "arena"
# Arena needs one active hand while composing its mimic cfg, but the direct recorder
# captures both pickup signals. Keep the operator's `auto` request intact so the
# exported trajectory, rather than this provisional runtime choice, owns the manifest.
arena_runtime_hand = "right" if args_cli.arena and args_cli.manipulation_hand == "auto" else args_cli.manipulation_hand

if args_cli.sample_steps < 0:
    parser.error("--sample_steps must be non-negative")
if args_cli.sample_export and not args_cli.sample_steps:
    parser.error("--sample_export requires --sample_steps")
if args_cli.sample_performance_result is not None and not args_cli.sample_steps:
    parser.error("--sample_performance_result requires --sample_steps")
if args_cli.task != DEFAULT_TASK and args_cli.wbc_mode is not None:
    parser.error("--wbc-mode is supported only for the built-in static-apple task.")
if args_cli.arena and args_cli.task != DEFAULT_TASK:
    parser.error("--arena composes the static-apple environment itself; drop --task.")
if args_cli.arena and args_cli.wbc_mode is not None:
    parser.error("--wbc-mode configures the built-in task; with --arena the embodiment selects the control stack.")
if args_cli.arena_embodiment is not None and not args_cli.arena:
    parser.error("--arena_embodiment requires --arena.")
if args_cli.sample_steps and args_cli.task != DEFAULT_TASK:
    parser.error("--sample_steps currently supports only the built-in static-apple task.")
if args_cli.arena and args_cli.sample_steps and args_cli.sample_action != "hold":
    parser.error("--sample_action wrist_sine drives the built-in task's Pink layout; with --arena use 'hold'.")
if args_cli.environment_snapshot is not None and args_cli.task != DEFAULT_TASK:
    parser.error("--environment_snapshot is supported only for the built-in static-apple task.")
if args_cli.table_height_offset_range_m is not None:
    if not args_cli.arena:
        parser.error("--table_height_offset_range_m composes the Arena scene's table; it requires --arena.")
    try:
        args_cli.table_height_offset_range_m = parse_offset_range(args_cli.table_height_offset_range_m)
    except ValueError as error:
        parser.error(str(error))
# Parse the namespaced --completion_override KEY=VALUE pairs (the task's policy
# interprets and range-checks them; here we only validate the KEY=VALUE format).
completion_overrides: dict[str, float] = {}
for _item in args_cli.completion_override or []:
    _key, _sep, _value = _item.partition("=")
    _key = _key.strip()
    if not _sep or not _key:
        parser.error(f"--completion_override must be KEY=VALUE, got: {_item!r}")
    try:
        completion_overrides[_key] = float(_value)
    except ValueError:
        parser.error(f"--completion_override {_key!r} value must be a number, got: {_value!r}")
H2_ARENA_RECORDING = args_cli.arena and (args_cli.arena_embodiment or "").startswith("h2_")
if args_cli.h2_station_keep is not None and (not H2_ARENA_RECORDING or args_cli.sample_steps):
    parser.error("--h2-station-keep is only for interactive H2 Arena recording")
try:
    require_h2_recording_cameras(
        h2_arena_recording=H2_ARENA_RECORDING,
        cameras_enabled=args_cli.enable_cameras,
    )
except ValueError as error:
    parser.error(str(error))
if H2_ARENA_RECORDING and "success_hold_frames" in completion_overrides:
    parser.error("H2 qualification fixes success_hold_frames at the manifested 20-frame contract")
if H2_ARENA_RECORDING and args_cli.auto_success is False:
    parser.error("H2 qualification requires measured auto-success and does not support --no-auto_success")

from e2e_robot_workflows.controllers import (
    load_controller_profiles,
    require_recording_ready_profile,
    select_controller_profile,
)

controller_profiles = load_controller_profiles(args_cli.controller_profiles_config)
controller_profile = select_controller_profile(controller_profiles, args_cli.controller_profile)
try:
    require_recording_ready_profile(controller_profile)
except ValueError as exc:
    raise SystemExit(str(exc)) from exc

if args_cli.dataset_file is None:
    args_cli.dataset_file = default_recording_dataset_file(args_cli.recording_lane)
args_cli.dataset_file = str(recorder_hdf5_output_path(args_cli.dataset_file))
dataset_output_path = Path(args_cli.dataset_file).resolve()
protected_result_paths = {dataset_output_path}
if args_cli.environment_snapshot is not None:
    args_cli.environment_snapshot = args_cli.environment_snapshot.expanduser().resolve()
    try:
        require_distinct_artifact_paths(
            dataset_output_path,
            args_cli.environment_snapshot,
            destination_label="Environment snapshot",
        )
    except ValueError as error:
        parser.error(str(error))
    protected_result_paths.add(args_cli.environment_snapshot)
if args_cli.sample_performance_result is not None:
    args_cli.sample_performance_result = args_cli.sample_performance_result.expanduser().resolve()
    if args_cli.sample_performance_result in protected_result_paths:
        parser.error("--sample_performance_result must differ from the dataset and environment snapshot")
if args_cli.xr_runtime_metrics is not None:
    args_cli.xr_runtime_metrics = args_cli.xr_runtime_metrics.expanduser().resolve()
    if args_cli.xr_runtime_metrics in protected_result_paths:
        parser.error("--xr_runtime_metrics must differ from the dataset and environment snapshot")

args_cli.teleop_device = XR_DEVICE_ALIASES.get(args_cli.teleop_device.lower(), args_cli.teleop_device.lower())
try:
    require_arena_teleop_mode(
        arena_recording=args_cli.arena,
        h2_arena_recording=H2_ARENA_RECORDING,
        sample_steps=args_cli.sample_steps,
        teleop_device=args_cli.teleop_device,
    )
except ValueError as error:
    parser.error(str(error))
if args_cli.teleop_device in XR_DEVICE_ALIASES.values():
    args_cli.xr = True
try:
    require_xr_runtime_metrics_mode(
        metrics_enabled=args_cli.xr_runtime_metrics is not None,
        xr_enabled=args_cli.xr,
        sample_steps=args_cli.sample_steps,
    )
except ValueError as error:
    parser.error(str(error))
if args_cli.xr and platform.machine() not in {"x86_64", "AMD64"}:
    parser.error("NVIDIA Isaac Teleop XR requires Linux x86_64; use a non-XR device on this platform.")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_teleop import remove_camera_configs

import e2e_robot_workflows.tasks.static_apple  # noqa: F401
from e2e_robot_workflows.data.completion import (
    CompletionTransition,
    EpisodeOutcome,
    advance_success_hold,
    resolve_completion_policy,
)
from e2e_robot_workflows.data.recorders import WorkflowRecorderManagerCfg
from e2e_robot_workflows.environments import STATIC_APPLE_ENVIRONMENT, resolve_role_to_scene_entity
from e2e_robot_workflows.environments.runtime_snapshot import (
    ENVIRONMENT_SNAPSHOT_SETTLE_STEPS,
    collect_environment_snapshot,
    write_environment_snapshot,
)
from e2e_robot_workflows.tasks.static_apple.env_cfg import make_env_cfg

DATASET_EXPORT_MODES = {
    "all": DatasetExportMode.EXPORT_ALL,
    "succeeded": DatasetExportMode.EXPORT_SUCCEEDED_ONLY,
}

# Arena embodiment this run composed, resolved once the Arena imports are available and
# reported with the environment snapshot. None on the built-in task's path.
arena_embodiment: str | None = None

# The task owns its completion policy (registered as completion_policy_entry_point
# in its gym kwargs); the recorder resolves and consumes it generically. None for
# tasks that provide no policy.
_policy_entry = gym.spec(args_cli.task).kwargs.get("completion_policy_entry_point")
if _policy_entry:
    try:
        COMPLETION_POLICY = resolve_completion_policy(_policy_entry, completion_overrides)
    except ValueError as exc:
        raise SystemExit(f"Invalid --completion_override: {exc}") from exc
else:
    COMPLETION_POLICY = None
    if completion_overrides:
        raise SystemExit(f"--completion_override given, but task '{args_cli.task}' has no completion policy.")


class RateLimiter:
    """Small render-aware rate limiter for interactive teleop."""

    def __init__(self, hz: int):
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)
        self.last_time = time.time()

    def sleep(self, env: gym.Env) -> None:
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time += self.sleep_duration
        while self.last_time < time.time():
            self.last_time += self.sleep_duration


def split_dataset_path(dataset_file: str) -> tuple[str, str]:
    """Return Isaac Lab recorder export dir and extension-free filename."""

    dataset_path = recorder_hdf5_output_path(dataset_file)
    output_dir = str(dataset_path.parent if dataset_path.parent != Path("") else Path("."))
    output_file_name = dataset_path.stem
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, output_file_name


def disable_terminations(terminations: object) -> list[str]:
    """Silence every termination term and return the names that were live.

    The operator, or the task's completion policy, decides when a demonstration ends.
    Every term is dropped rather than a known list because the composed environment's
    terms depend on who built it: Arena's pick-and-place task adds ``object_dropped``,
    which would end a demonstration the moment the apple left the operator's hand.
    """

    disabled = []
    for name, term in list(vars(terminations).items()):
        if not isinstance(term, TerminationTermCfg):
            continue
        setattr(terminations, name, None)
        disabled.append(name)
    return disabled


def build_arena_env() -> tuple[str, ManagerBasedRLEnvCfg, dict[str, object]]:
    """Compose and register the Arena realization, returning what ``gym.make`` needs."""

    global arena_embodiment
    from e2e_robot_workflows.arena.embodiment_capabilities import (
        G1_ARENA_TELEOP_EMBODIMENT,
        capability_for_embodiment,
    )

    # Recording means teleoperation, and only the Pink layout has a retargeter.
    arena_embodiment = args_cli.arena_embodiment or G1_ARENA_TELEOP_EMBODIMENT
    logger.info("Arena embodiment: {}", arena_embodiment)
    capability = capability_for_embodiment(arena_embodiment)
    if arena_embodiment == capability.teleop_embodiment:
        from e2e_robot_workflows.arena.mimic import compose_arena_mimic_environment

        composed = compose_arena_mimic_environment(
            device=args_cli.device,
            num_envs=1,
            enable_cameras=bool(args_cli.enable_cameras),
            manipulation_hand=arena_runtime_hand,
            seed=args_cli.seed,
            embodiment=arena_embodiment,
            # Only XR reaches Arena's device registry (checked above); the finite sample run
            # steps the env itself and must not configure a device.
            teleop_device="openxr" if args_cli.xr and not args_cli.sample_steps else None,
            table_height_offset_range_m=args_cli.table_height_offset_range_m,
        )
    else:
        from e2e_robot_workflows.arena.composition import compose

        composed = compose(
            embodiment=arena_embodiment,
            device=args_cli.device,
            enable_cameras=bool(args_cli.enable_cameras),
            teleop_device="openxr" if args_cli.xr and not args_cli.sample_steps else None,
            table_height_offset_range_m=args_cli.table_height_offset_range_m,
        )
    return composed.task_id, composed.env_cfg, composed.env_kwargs


def build_env_cfg(output_dir: str, output_file_name: str) -> tuple[str, ManagerBasedRLEnvCfg | DirectRLEnvCfg, dict]:
    """Parse and adapt the task config for manual demo recording."""

    env_kwargs: dict[str, object] = {}
    if args_cli.arena:
        task_id, env_cfg, env_kwargs = build_arena_env()
    elif args_cli.task == DEFAULT_TASK:
        task_id = args_cli.task
        env_cfg = make_env_cfg(
            wbc_mode=args_cli.wbc_mode or DEFAULT_WBC_MODE,
            device=args_cli.device,
            num_envs=1,
        )
    else:
        task_id = args_cli.task
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = task_id.split(":")[-1]
    disabled = disable_terminations(env_cfg.terminations)
    logger.info("Terminations disabled for recording: {}", ", ".join(disabled) if disabled else "none")
    env_cfg.observations.policy.concatenate_terms = False

    if args_cli.xr:
        if not args_cli.enable_cameras:
            env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    if args_cli.arena:
        from e2e_robot_workflows.arena.recorders import make_recorder_manager_cfg

        env_cfg.recorders = make_recorder_manager_cfg(env_cfg)
    else:
        env_cfg.recorders = WorkflowRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    # Default keeps only successful episodes. The finite --sample_export smoke
    # episode is intentionally unsuccessful, so it must persist regardless.
    export_mode = "all" if args_cli.sample_export else args_cli.dataset_export_mode
    env_cfg.recorders.dataset_export_mode = DATASET_EXPORT_MODES[export_mode]
    env_cfg.recorders.export_in_close = False
    # Export only through the explicit save path below -- the default
    # export-on-reset would double-export every episode (and export ghost
    # episodes on stray resets).
    env_cfg.recorders.export_in_record_pre_reset = False
    env_cfg.seed = args_cli.seed
    return task_id, env_cfg, env_kwargs


def create_env(task_id: str, env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, env_kwargs: dict) -> gym.Env:
    """Instantiate the configured Gym environment.

    ``env_kwargs`` carries what the builder asks the constructor for; Arena passes its
    ``variation_recorder`` there, so forwarding it keeps variation sampling recorded.
    """

    return gym.make(task_id, cfg=env_cfg, **env_kwargs).unwrapped


def create_device(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, callbacks: dict[str, Callable]) -> object:
    """Create a configured NVIDIA Isaac Teleop device or a built-in fallback."""

    if args_cli.xr and getattr(env_cfg, "isaac_teleop", None) is not None:
        try:
            from isaaclab_teleop import create_isaac_teleop_device
        except ImportError as exc:
            raise RuntimeError("This task uses IsaacTeleop for XR, but isaaclab_teleop is not installed.") from exc
        return create_isaac_teleop_device(
            env_cfg.isaac_teleop,
            sim_device=env_cfg.sim.device,
            callbacks=callbacks,
        )

    if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
        return create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)

    if args_cli.xr:
        available = sorted(getattr(getattr(env_cfg, "teleop_devices", None), "devices", {}).keys())
        raise ValueError(
            f"XR teleop device '{args_cli.teleop_device}' is not configured for this task. "
            f"Available configured devices: {available}"
        )

    if args_cli.teleop_device.lower() == "keyboard":
        device = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
    elif args_cli.teleop_device.lower() == "spacemouse":
        device = Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
    else:
        raise ValueError(f"Unsupported teleop device: {args_cli.teleop_device}")

    for key, callback in callbacks.items():
        device.add_callback(key, callback)
    return device


def exported_episode_count(env: gym.Env) -> int:
    """Number of episodes actually written to the dataset.

    In succeeded-only mode the RecorderManager still increments its failed-episode
    counter for dropped failures, so count failures only when the export mode
    actually writes them; otherwise a failed episode would wrongly advance --num_demos.
    """

    manager = env.recorder_manager
    count = manager.exported_successful_episode_count
    if manager.cfg.dataset_export_mode == DatasetExportMode.EXPORT_ALL:
        count += manager.exported_failed_episode_count
    return count


def export_current_episode(env: gym.Env, success: bool = True) -> None:
    """Finalize and export the current episode as dataset data."""

    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
    success_tensor = torch.tensor([[success]], dtype=torch.bool, device=env.device)
    env.recorder_manager.set_success_to_episodes([0], success_tensor)
    env.recorder_manager.export_episodes([0])


def evaluate_outcome_or_none(env: gym.Env) -> EpisodeOutcome | None:
    """Return the current episode outcome, or ``None`` when detection failed.

    Detection must never crash a live episode, so read errors are logged and the
    caller continues recording.
    """

    try:
        return COMPLETION_POLICY.evaluate(env)
    except Exception as exc:  # noqa: BLE001 - detection is best-effort; keep recording
        logger.warning("Auto success/failure detection failed: {}; continuing episode.", exc)
        return None


def check_completion_policy_readable(env: gym.Env) -> None:
    """Fail before recording starts if the task's policy cannot read the composed scene.

    Which names the scene carries depends on who composed it, so this is the check that
    holds the built-in task and Arena to the same contract roles -- and it runs on every
    recording path, including the finite ``--sample_steps`` run that CI drives.
    """

    if COMPLETION_POLICY is None:
        return
    COMPLETION_POLICY.check_readable(env)
    logger.info("Completion policy reads the composed scene.")


def maybe_auto_finish(env: gym.Env, controls: RecordingControls) -> None:
    """Auto-save and reset the scene when the task's completion policy reports a terminal outcome.

    Continuous hands-free recording: the episode is saved with the measured label and
    the scene resets, *without stopping the session*. Because the recorder cannot
    drive the XR panel (IsaacTeleop has no host-to-panel channel), keeping the
    session running means the panel's "running" state stays truthful between episodes.
    """

    hold_frames = COMPLETION_POLICY.hold_frames
    previous_remaining = controls.success_hold_remaining
    outcome = evaluate_outcome_or_none(env)
    transition, controls.success_hold_remaining = advance_success_hold(
        outcome,
        hold_remaining=previous_remaining,
        hold_frames=hold_frames,
    )
    if transition is CompletionTransition.CONTINUE:
        return
    if transition is CompletionTransition.START_HOLD:
        controls.pending_outcome = outcome
        logger.info("SUCCESS detected; recording {} more frame(s) before saving.", hold_frames)
        return
    if transition is CompletionTransition.INTERRUPT_HOLD:
        controls.pending_outcome = None
        logger.info("SUCCESS follow-through interrupted; continuing the current episode.")
        return
    if transition is CompletionTransition.COMPLETE_FAILURE:
        controls.pending_outcome = None
        controls.finish(outcome, success=False)
        logger.info("Auto-detected FAILURE ({}); saving and resetting for the next episode.", outcome.value)
        return
    completed_outcome = controls.pending_outcome or outcome
    controls.pending_outcome = None
    controls.finish(completed_outcome, success=True)
    logger.info("SUCCESS follow-through complete ({} frames); requesting export before reset.", hold_frames)


def finish_recording_episode(env: gym.Env, controls: RecordingControls, *, auto_success_enabled: bool) -> int | None:
    """Admit retained interactive episodes before writing or counting them."""

    success = controls.pending_success
    if auto_success_enabled and not controls.auto_reset:
        if H2_ARENA_RECORDING:
            success = False
            logger.info("Manual H2 finish has no 20-frame success hold; labelling FAILURE.")
        else:
            outcome = evaluate_outcome_or_none(env)
            success = bool(outcome and outcome.is_success)
            if outcome is None:
                logger.warning("Outcome unreadable on manual finish; labelling FAILURE.")
    if H2_ARENA_RECORDING and (success or args_cli.dataset_export_mode == "all"):
        if int(env.episode_length_buf[0]) == 0:
            controls.discard_episode()
            controls.auto_reset = controls.running
            logger.info("Empty H2 take discarded; resetting without export.")
            return None
        if not admit_h2_recording_episode(
            env.recorder_manager.get_episode(0).data, controls, manipulation_hand=args_cli.manipulation_hand
        ):
            return None
    export_current_episode(env, success=success)
    if controls.auto_reset and controls.pending_outcome is not None:
        controls.auto_outcome_counts[controls.pending_outcome.value] += 1
        if success:
            controls.auto_success_count += 1
    recorded_count = exported_episode_count(env)
    if not success and args_cli.dataset_export_mode == "succeeded":
        logger.info("Episode not a success; discarded. Recorded {} demonstration(s).", recorded_count)
    else:
        logger.info("Recorded {} demonstration(s).", recorded_count)
    controls.should_export = False
    controls.should_reset = True
    controls.pending_success = True
    controls.pending_outcome = None
    return recorded_count


def reset_episode(env: gym.Env) -> None:
    """Reset sim/env/recorder state for the next recording attempt."""

    env.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    if H2_ARENA_RECORDING:
        from e2e_robot_workflows.devices.h2_ready import prepare_h2_recording, prime_h2_controller

        if args_cli.xr and not args_cli.sample_steps:
            prepare_h2_recording(env, embodiment=arena_embodiment, is_running=simulation_app.is_running)
        else:
            prime_h2_controller(env, arena_embodiment)
            env.recorder_manager.reset()
            env.recorder_manager.record_post_reset([0])
    # Refresh the viewport without moving physics beyond the reset state represented
    # by env.obs_buf, which is the first recorder frame.
    refresh_reset_view(env)


def complete_recording_session(env: gym.Env, controls: RecordingControls, recorded_count: int) -> None:
    """Reset the final scene and let the operator see completion before XR closes."""

    controls.complete()
    reset_episode(env)
    # Reset/settling can create a new recorder buffer. It is not another take;
    # the retained exports are already flushed by export_current_episode.
    env.recorder_manager.reset()
    render_frame = make_state_renderer(env)
    render_frame()
    logger.info("Target reached: {} demo(s). Recording is complete.", recorded_count)
    if args_cli.xr:
        with xr_recording_completion_notice(recorded_count):
            env.sim.render()
            deadline = time.monotonic() + 5.0
            while simulation_app.is_running() and not env.sim.is_stopped() and time.monotonic() < deadline:
                env.sim.render()


def run_recording_loop(env: gym.Env, env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> int:
    """Run teleop loop until requested demo count is recorded or the app exits."""

    recorded_count = 0
    is_xr_session = args_cli.xr
    # Auto-detection runs only when the task provides a completion policy. Whether
    # it is on defaults to that policy; --auto_success/--no-auto_success overrides.
    if COMPLETION_POLICY is None:
        auto_success_enabled = False
        if args_cli.auto_success:
            logger.warning("--auto_success requested, but task '{}' has no completion policy.", args_cli.task)
    elif args_cli.auto_success is None:
        auto_success_enabled = COMPLETION_POLICY.default_enabled
    else:
        auto_success_enabled = args_cli.auto_success
    # XR sessions wait for START on the panel; keyboard flow keeps the old
    # default (running unless --start_paused).
    controls = RecordingControls(running=False if is_xr_session else not args_cli.start_paused)
    pose_alignment = None
    if H2_ARENA_RECORDING and is_xr_session:
        from e2e_robot_workflows.devices.h2_reset_alignment import H2ResetAlignment

        pose_alignment = H2ResetAlignment()

    if is_xr_session:
        # The IsaacTeleop client sends STOP followed by RESET when the operator
        # presses Reset. STOP owns the episode boundary; RESET only acknowledges
        # device state reinitialization during controller-tracking recovery.
        callbacks = {
            "START": controls.resume,
            "STOP": controls.request_export,
            "RESET": controls.observe_device_reset,
        }
    else:
        callbacks = {
            "M": controls.request_export,
            "R": controls.request_reset,
            "P": controls.pause,
            "O": controls.resume,
            "STOP": controls.pause,
            "START": controls.resume,
            "RESET": controls.request_reset,
        }
    teleop_interface = create_device(env_cfg, callbacks)
    station_keeping = None
    if H2_ARENA_RECORDING and is_xr_session and args_cli.h2_station_keep is not False:
        from e2e_robot_workflows.devices.h2_station_keeping import H2StationKeeping, apply_h2_station_keeping

        station_keeping = H2StationKeeping()
        logger.info(
            "H2 neutral-stick station keeping enabled (max {} m/s, {} rad/s); navigation overrides it. "
            "Effective commands are recorded. Use --no-h2-station-keep to disable.",
            station_keeping.config.max_linear_speed,
            station_keeping.config.max_yaw_speed,
        )
    rate_limiter = None if is_xr_session else RateLimiter(args_cli.step_hz)
    telemetry = None
    if args_cli.xr_runtime_metrics is not None:
        sim_cfg = env.cfg.sim
        telemetry = XrRuntimeTelemetry(
            path=args_cli.xr_runtime_metrics,
            runtime={
                "device": str(env.device),
                "teleop_device": args_cli.teleop_device,
                "physics_dt_s": float(sim_cfg.dt),
                "decimation": int(env.cfg.decimation),
                "render_interval": int(sim_cfg.render_interval),
                "control_period_s": float(env.step_dt),
                "h2_station_keeping": station_keeping is not None,
                "h2_reset_alignment": pose_alignment is not None,
            },
        )
        logger.info("XR runtime metrics: {}", args_cli.xr_runtime_metrics)

    logger.info("Teleop device: {}", teleop_interface)
    if auto_success_enabled:
        export_note = (
            "Only successful episodes reach the dataset (--dataset_export_mode succeeded, the default); "
            "pass 'all' to keep labeled failures too."
            if args_cli.dataset_export_mode == "succeeded"
            else "Successful and failed episodes both reach the dataset, each labeled (--dataset_export_mode all)."
        )
        logger.info(
            "Auto success/failure ON (task completion policy): episodes record continuously -- each one "
            "auto-saves with its measured outcome and the scene resets, without stopping the session. "
            "Keep performing the task; press Stop/Reset to end. {} Every auto-decision is logged, and a "
            "per-outcome tally prints at the end.",
            export_note,
        )
    if is_xr_session:
        logger.info(
            "XR panel: Play starts the first episode; Reset finishes the current episode. "
            "Automatic outcomes reset and continue without another button press."
        )
        logger.info("Waiting for START on the XR panel...")
    else:
        logger.info("Controls: M=finish/save episode, R=reset/skip, P=pause, O=resume.")
    reset_episode(env)
    teleop_interface.reset()

    device_context = (
        teleop_interface
        if is_xr_session and hasattr(teleop_interface, "__enter__") and hasattr(teleop_interface, "__exit__")
        else contextlib.nullcontext(teleop_interface)
    )
    with device_context:
        if telemetry is not None:
            telemetry.write_running()
        telemetry_status = "ok"
        exit_reason = "application_stopped"
        telemetry_error_type = None
        try:
            with torch.inference_mode():
                while simulation_app.is_running():
                    cycle_started_ns = time.perf_counter_ns()
                    cycle_was_active = controls.running
                    controls.begin_advance()
                    advance_started_ns = time.perf_counter_ns()
                    action = teleop_interface.advance()
                    if pose_alignment is not None and not cycle_was_active and controls.running:
                        pose_alignment.reset()
                    session_lost = bool(action is None and cycle_was_active and controls.recover_tracking())
                    if telemetry is not None:
                        advance_phase = "teleop_advance_active" if cycle_was_active else "teleop_advance_idle"
                        telemetry.observe(advance_phase, time.perf_counter_ns() - advance_started_ns)
                        telemetry.increment("loop_iterations")
                        if session_lost:
                            telemetry.increment("tracking_loss_resets")
                    tracking_recovered = controls.finish_tracking_recovery(
                        action_is_finite=bool(
                            controls.waiting_for_tracking and action is not None and torch.isfinite(action).all()
                        )
                    )

                    if action is None or tracking_recovered:
                        render_generation = env.sim.render_generation if telemetry is not None else 0
                        env.sim.render()
                        if telemetry is not None:
                            if tracking_recovered:
                                telemetry.increment("idle_renders")
                            elif cycle_was_active:
                                telemetry.increment("action_none_active")
                                telemetry.increment(
                                    "active_render_calls", env.sim.render_generation - render_generation
                                )
                            else:
                                telemetry.increment("action_none_idle")
                                telemetry.increment("idle_renders")
                    else:
                        if controls.running:
                            validity_started_ns = time.perf_counter_ns()
                            action_is_finite = bool(torch.isfinite(action).all())
                            if telemetry is not None:
                                telemetry.observe("action_validity", time.perf_counter_ns() - validity_started_ns)
                                telemetry.increment("valid_actions" if action_is_finite else "nonfinite_actions")
                            if not action_is_finite:
                                controls.recover_tracking()
                                if telemetry is not None:
                                    telemetry.increment("tracking_loss_resets")

                        if controls.running:
                            if pose_alignment is not None:
                                action = align_h2_recording_action(action, env, pose_alignment)
                            if station_keeping is not None:
                                action = apply_h2_station_keeping(action, env.scene["robot"].data, station_keeping)
                            render_generation = env.sim.render_generation if telemetry is not None else 0
                            step_started_ns = time.perf_counter_ns()
                            stepped = step_recording_environment(
                                lambda action=action: env.step(action.repeat(env.num_envs, 1)),
                                controls,
                                recover_h2_joint_limits=H2_ARENA_RECORDING and is_xr_session,
                            )
                            if telemetry is not None:
                                telemetry.observe("env_step", time.perf_counter_ns() - step_started_ns)
                                telemetry.increment("active_env_steps" if stepped else "joint_limit_resets")
                                telemetry.increment(
                                    "active_render_calls", env.sim.render_generation - render_generation
                                )
                            if stepped and auto_success_enabled:
                                completion_started_ns = time.perf_counter_ns()
                                maybe_auto_finish(env, controls)
                                if telemetry is not None:
                                    telemetry.observe(
                                        "completion_policy", time.perf_counter_ns() - completion_started_ns
                                    )
                                    telemetry.increment("completion_calls")
                        else:
                            env.sim.render()
                            if telemetry is not None:
                                telemetry.increment("idle_renders")

                    if tracking_recovered:
                        controls.running = True
                        logger.info("Controller tracking recovered; recording resumes on the next sample.")

                    if telemetry is not None and (cycle_was_active or controls.running):
                        telemetry.observe("active_cycle", time.perf_counter_ns() - cycle_started_ns)

                    if controls.should_export:
                        exported_count = finish_recording_episode(
                            env, controls, auto_success_enabled=auto_success_enabled
                        )
                        if exported_count is not None:
                            recorded_count = exported_count
                            if telemetry is not None:
                                telemetry.increment("exports")

                    if controls.should_reset and not (args_cli.num_demos > 0 and recorded_count >= args_cli.num_demos):
                        reset_episode(env)
                        if pose_alignment is not None:
                            pose_alignment.reset()
                        if station_keeping is not None:
                            station_keeping.reset()
                        if telemetry is not None:
                            telemetry.increment("episode_resets")
                        controls.should_reset = False
                        controls.success_hold_remaining = None
                        controls.pending_outcome = None
                        was_auto_reset = controls.auto_reset
                        controls.auto_reset = False
                        teleop_interface.reset()
                        if controls.waiting_for_tracking and is_xr_session:
                            logger.info("Scene reset. Waiting for controller tracking to recover.")
                        elif was_auto_reset and is_xr_session:
                            logger.info("Scene reset. Ready for the next take.")
                        elif not was_auto_reset and is_xr_session:
                            controls.running = False
                            logger.info("Scene reset. Press START to begin the next episode.")

                    if args_cli.num_demos > 0 and recorded_count >= args_cli.num_demos:
                        exit_reason = "target_reached"
                        complete_recording_session(env, controls, recorded_count)
                        break

                    if env.sim.is_stopped():
                        exit_reason = "simulation_stopped"
                        break

                    if rate_limiter:
                        rate_limiter.sleep(env)
        except KeyboardInterrupt:
            exit_reason = "keyboard_interrupt"
        except BaseException as error:
            telemetry_status = "failed"
            exit_reason = "error"
            telemetry_error_type = type(error).__name__
            raise
        finally:
            if telemetry is not None:
                try:
                    telemetry.write_final(
                        status=telemetry_status,
                        exit_reason=exit_reason,
                        recorded_count=recorded_count,
                        auto_outcome_counts=controls.auto_outcome_counts,
                        admission_rejections=controls.admission_rejections,
                        error_type=telemetry_error_type,
                    )
                except Exception:
                    logger.exception("Failed to finalize XR runtime metrics: {}", telemetry.path)
                    if telemetry_status == "ok":
                        raise

    if controls.admission_rejections:
        logger.info("{} episode(s) rejected by handover admission in every export mode.", controls.admission_rejections)
    if controls.auto_outcome_counts:
        total = sum(controls.auto_outcome_counts.values())
        breakdown = ", ".join(f"{name}={count}" for name, count in sorted(controls.auto_outcome_counts.items()))
        logger.info("Auto-detected outcomes this session ({} episode(s)): {}", total, breakdown)
        if args_cli.dataset_export_mode == "succeeded":
            dropped = total - controls.auto_success_count
            if dropped:
                logger.info(
                    "{} failed episode(s) were dropped by --dataset_export_mode succeeded; "
                    "re-run with 'all' to retain failures that pass dataset admission.",
                    dropped,
                )

    return recorded_count


def make_fixed_pink_hold_action(env: gym.Env) -> torch.Tensor:
    """Build a valid Pink command that holds the live wrists and hand joints."""

    robot = env.scene["robot"]
    action_term = env.action_manager.get_term("upper_body_ik")
    try:
        left_wrist_id = robot.body_names.index("left_wrist_yaw_link")
        right_wrist_id = robot.body_names.index("right_wrist_yaw_link")
    except ValueError as exc:
        raise RuntimeError("The recorder sample requires the configured G1 wrist links.") from exc
    hand_joint_ids, _ = robot.find_joints(action_term.cfg.hand_joint_names)
    action = torch.zeros((env.num_envs, action_term.action_dim), device=env.device)
    action[:, 0:7] = robot.data.body_link_pose_w.torch[:, left_wrist_id, :7]
    action[:, 7:14] = robot.data.body_link_pose_w.torch[:, right_wrist_id, :7]
    action[:, 14:] = robot.data.joint_pos.torch[:, hand_joint_ids]
    return action


def make_hold_action(env: gym.Env) -> torch.Tensor:
    """Build the sample's safe hold command for whichever runtime composed ``env``."""

    if args_cli.arena:
        from e2e_robot_workflows.arena.g1_actions import hold_action

        assert arena_embodiment is not None
        return hold_action(env, arena_embodiment)
    return make_fixed_pink_hold_action(env)


def run_recorder_sample(env: gym.Env, steps: int) -> int:
    """Exercise the live recorder manager and optionally export its episode."""

    env_device = torch.device(env.device)
    reset_episode(env)
    previous_exports = exported_episode_count(env)
    hold_action = make_hold_action(env)
    warmup_steps = 0
    reset_wall_s = 0.0
    if args_cli.sample_performance_result is not None:
        warmup_steps = min(30, steps)
        with torch.inference_mode():
            for _ in range(warmup_steps):
                env.step(hold_action)
                if COMPLETION_POLICY is not None:
                    COMPLETION_POLICY.evaluate(env)
        if env_device.type == "cuda":
            torch.cuda.synchronize(env_device)
        reset_started = time.perf_counter()
        reset_episode(env)
        if env_device.type == "cuda":
            torch.cuda.synchronize(env_device)
        reset_wall_s = time.perf_counter() - reset_started
        hold_action = make_hold_action(env)

    if env_device.type == "cuda":
        torch.cuda.synchronize(env_device)
    step_started = time.perf_counter()
    episode_ended = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    with torch.inference_mode():
        for step in range(steps):
            action = hold_action.clone()
            if args_cli.sample_action == "wrist_sine":
                offset_m = 0.005 * math.sin(2.0 * math.pi * step / steps)
                action[:, 2] += offset_m
                action[:, 9] -= offset_m
            _, _, terminated, truncated, _ = env.step(action)
            episode_ended |= terminated | truncated
            if args_cli.sample_performance_result is not None and COMPLETION_POLICY is not None:
                COMPLETION_POLICY.evaluate(env)
    if env_device.type == "cuda":
        torch.cuda.synchronize(env_device)
    stepping_wall_s = time.perf_counter() - step_started
    if bool(episode_ended.any().item()):
        raise RuntimeError("The recorder sample episode ended before the requested safe hold steps completed.")

    export_wall_s = 0.0
    if args_cli.sample_export:
        export_started = time.perf_counter()
        export_current_episode(env, success=False)
        export_wall_s = time.perf_counter() - export_started
        exported_count = exported_episode_count(env)
        if exported_count != previous_exports + 1:
            raise RuntimeError(f"Expected one new sample episode, found {exported_count - previous_exports}.")
        logger.info(
            "Recorder sample completed {} {} step(s) and exported one unsuccessful smoke-test episode.",
            steps,
            args_cli.sample_action,
        )
    else:
        exported_count = 0
        logger.info("Recorder sample completed {} {} step(s); no episode was exported.", steps, args_cli.sample_action)

    control_period_s = float(env.step_dt)
    effective_hz = steps / stepping_wall_s
    realtime_factor = steps * control_period_s / stepping_wall_s
    logger.info(
        "Recorder stepping performance: steps={} wall={:.6f}s effective={:.3f}Hz realtime_factor={:.4f}",
        steps,
        stepping_wall_s,
        effective_hz,
        realtime_factor,
    )
    if args_cli.sample_performance_result is not None:
        import json

        from e2e_robot_workflows.data.artifact_paths import atomic_write_text

        sim_cfg = env.cfg.sim
        performance = {
            "status": "ok",
            "steps": steps,
            "warmup_steps": warmup_steps,
            "device": str(env.device),
            "physics_dt_s": float(sim_cfg.dt),
            "decimation": int(env.cfg.decimation),
            "render_interval": int(sim_cfg.render_interval),
            "control_period_s": control_period_s,
            "simulated_duration_s": steps * control_period_s,
            "stepping_wall_s": stepping_wall_s,
            "effective_hz": effective_hz,
            "realtime_factor": realtime_factor,
            "reset_wall_s": reset_wall_s,
            "export_wall_s": export_wall_s,
            "completion_policy_evaluated": COMPLETION_POLICY is not None,
            "teleop_advance_included": False,
        }
        atomic_write_text(args_cli.sample_performance_result, json.dumps(performance, indent=2, sort_keys=True) + "\n")
    return exported_count


def snapshot_mode(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> dict[str, object]:
    """Describe how this run realized the environment, for the snapshot comparison."""

    if args_cli.arena:
        return {
            "name": "recording",
            "runtime": "arena",
            "arena_embodiment": arena_embodiment,
            "enable_waist": False,
        }
    return {
        "name": "recording",
        "runtime": "isaaclab",
        "controller_profile": controller_profile.name,
        "wbc_mode": getattr(env_cfg, "wbc_mode", "task-defined"),
        "enable_waist": getattr(env_cfg, "enable_waist", None),
    }


def finalize_recording_dataset(env_cfg, exported_dataset: Path) -> None:
    """Stamp closed, nonempty exports without inferring a hand from the CLI alone."""

    from e2e_robot_workflows.data.dataset_manifest import stamp_recording_dataset

    manifest_hand = args_cli.manipulation_hand
    if args_cli.arena:
        from e2e_robot_workflows.data.dataset_manifest import detect_arena_manipulation_hand

        try:
            detected_hand = detect_arena_manipulation_hand(exported_dataset)
        except DatasetContractError:
            if not args_cli.sample_export:
                raise
            detected_hand = "right"
            logger.warning("Finite unsuccessful Arena sample cannot verify its hand; marking the fixture right.")
    else:
        from e2e_robot_workflows.autodata.dataset_mode import DatasetModeError, detect_manipulation_hand

        try:
            detected_hand = detect_manipulation_hand(exported_dataset)
        except DatasetModeError:
            if not args_cli.sample_export:
                raise
            detected_hand = "right"
            logger.warning("Finite unsuccessful native sample cannot verify its hand; marking the fixture right.")
    if manifest_hand != "auto" and manifest_hand != detected_hand:
        raise RuntimeError(
            f"Recorded trajectory uses the {detected_hand} hand, but --manipulation-hand {manifest_hand} was requested."
        )
    randomization = getattr(env_cfg, "e2e_table_height_randomization", None)
    manifest = stamp_recording_dataset(
        exported_dataset,
        lane=args_cli.recording_lane,
        wbc_mode=None if args_cli.arena else getattr(env_cfg, "wbc_mode", args_cli.wbc_mode or DEFAULT_WBC_MODE),
        arena_embodiment=arena_embodiment if args_cli.arena else None,
        seed=args_cli.seed,
        manipulation_hand=detected_hand,
        scene_randomization=(
            None if randomization is None else scene_randomization_manifest(randomization, seed=args_cli.seed)
        ),
    )
    logger.info(
        "Dataset manifest: lane={} layout={} producer={}",
        manifest["lane"],
        manifest["action"]["layout"],
        manifest["producer"],
    )


def main() -> None:
    environment = STATIC_APPLE_ENVIRONMENT if args_cli.task == DEFAULT_TASK else None
    if environment is not None and args_cli.arena:
        from e2e_robot_workflows.arena.embodiment_capabilities import (
            G1_ARENA_TELEOP_EMBODIMENT,
            capability_for_embodiment,
        )
        from e2e_robot_workflows.tasks.static_apple.config import load_static_apple_environment

        selected_embodiment = args_cli.arena_embodiment or G1_ARENA_TELEOP_EMBODIMENT
        capability = capability_for_embodiment(selected_embodiment)
        environment = load_static_apple_environment(capability.environment_variant)
    if environment is not None:
        logger.info(
            "Environment: {} ({})",
            environment.environment_id,
            environment.contract_digest,
        )
        logger.info("Qualification boundary: {}", environment.qualification_note)
    else:
        logger.warning("Task {} has no E2E-managed environment definition.", args_cli.task)
    output_dir, output_file_name = split_dataset_path(args_cli.dataset_file)
    logger.info(f"Dataset file: {args_cli.dataset_file}")
    if args_cli.arena:
        # Arena's embodiment brings its own controllers, so this app's profile does not
        # describe what is running.
        logger.info("Control stack: composed by Arena (this app's controller profiles do not apply).")
    else:
        logger.info(f"Controller profile: {controller_profile.name}")
        logger.info(
            "Controller stack: "
            f"{controller_profile.upper_body.name} + {controller_profile.lower_body.name} "
            f"→ {controller_profile.composer.name}"
        )
    task_id, env_cfg, env_kwargs = build_env_cfg(output_dir, output_file_name)
    logger.info(f"WBC mode: {getattr(env_cfg, 'wbc_mode', 'task-defined')}")
    env = create_env(task_id, env_cfg, env_kwargs)
    recording_error = None
    try:
        check_completion_policy_readable(env)
        if args_cli.sample_steps:
            for _ in range(max(1, args_cli.num_demos)):
                run_recorder_sample(env, args_cli.sample_steps)
        else:
            run_recording_loop(env, env_cfg)
        if args_cli.environment_snapshot is not None:
            assert environment is not None
            # Compare settled reset scenes for both interactive and sample runs.
            reset_episode(env)
            with torch.inference_mode():
                hold_action = make_hold_action(env)
                for _ in range(ENVIRONMENT_SNAPSHOT_SETTLE_STEPS):
                    env.step(hold_action)
            # Snapshot preparation is not another recorded demonstration.
            env.recorder_manager.reset()
            snapshot = collect_environment_snapshot(
                definition=environment,
                mode=snapshot_mode(env_cfg),
                env=env,
                role_to_scene_entity=resolve_role_to_scene_entity(env.scene),
            )
            write_environment_snapshot(args_cli.environment_snapshot, snapshot)
            logger.info("Environment snapshot: {}", args_cli.environment_snapshot)
    except BaseException as error:
        recording_error = error
        raise
    finally:
        recorded_count = close_recording_dataset(
            lambda: exported_episode_count(env),
            env.close,
            lambda: (
                finalize_recording_dataset(env_cfg, Path(output_dir) / f"{output_file_name}.hdf5")
                if environment is not None
                else None
            ),
            original_error=recording_error,
        )
    if args_cli.sample_steps:
        logger.info("Recorder acceptance sample completed with {} exported test episode(s).", recorded_count)
    else:
        logger.info(f"Recording session completed with {recorded_count} saved demo(s).")
    logger.info(f"Dataset target: {args_cli.dataset_file}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

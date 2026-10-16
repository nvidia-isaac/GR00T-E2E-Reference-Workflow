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

"""Generate a bounded Arena dataset with public AutoData interfaces."""

from __future__ import annotations

import argparse
import atexit
import math
import shutil
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher

from e2e_robot_workflows.data.artifact_paths import file_sha256, require_distinct_artifact_paths
from e2e_robot_workflows.tasks.static_apple.table_height import parse_offset_range, scene_randomization_manifest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input_file", required=True)
parser.add_argument("--output_file", required=True)
parser.add_argument("--result_file", required=True)
parser.add_argument("--run_id", required=True)
parser.add_argument("--task_descriptor", required=True)
parser.add_argument("--manipulation_hand", choices=("left", "right"), required=True)
parser.add_argument("--requested_successes", type=int, default=1)
parser.add_argument("--max_attempts", type=int, default=3)
parser.add_argument("--generation_seed", type=int, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--table_height_offset_range_m",
    nargs=2,
    type=float,
    metavar=("MIN", "MAX"),
    default=None,
    help=(
        "Generate each attempt on a tabletop MIN..MAX metres from the robot's task variant. "
        "Omitted (or '0 0') generates on the source recording's nominal fixed-height scene."
    ),
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True)
args_cli = parser.parse_args()

try:
    args_cli.table_height_offset_range_m = parse_offset_range(args_cli.table_height_offset_range_m)
except ValueError as error:
    parser.error(str(error))
if args_cli.num_envs != 1:
    parser.error("Arena Pink generation is qualified sequentially with exactly one environment")
if args_cli.requested_successes <= 0:
    parser.error("--requested_successes must be positive")
if args_cli.max_attempts < args_cli.requested_successes:
    parser.error("--max_attempts must be at least --requested_successes")


ORIGINAL_SOURCE_PATH = Path(args_cli.input_file).expanduser().resolve()
ORIGINAL_DESCRIPTOR_PATH = Path(args_cli.task_descriptor).expanduser().resolve()
OUTPUT_PATH = Path(args_cli.output_file).expanduser().resolve()
RESULT_PATH = Path(args_cli.result_file).expanduser().resolve()
if OUTPUT_PATH.suffix != ".hdf5" or not OUTPUT_PATH.stem.strip(".") or OUTPUT_PATH.stem.endswith(".hdf5"):
    parser.error("--output_file must have exactly one .hdf5 suffix and a basename before it")
FAILED_OUTPUT_PATH = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_failed{OUTPUT_PATH.suffix}")
for input_path in (ORIGINAL_SOURCE_PATH, ORIGINAL_DESCRIPTOR_PATH):
    for output_path, output_label in (
        (OUTPUT_PATH, "Arena generated dataset"),
        (FAILED_OUTPUT_PATH, "Arena failed dataset"),
        (RESULT_PATH, "Arena result file"),
    ):
        require_distinct_artifact_paths(input_path, output_path, destination_label=output_label)
require_distinct_artifact_paths(OUTPUT_PATH, FAILED_OUTPUT_PATH, destination_label="Arena failed dataset")
require_distinct_artifact_paths(OUTPUT_PATH, RESULT_PATH, destination_label="Arena result file")
require_distinct_artifact_paths(FAILED_OUTPUT_PATH, RESULT_PATH, destination_label="Arena result file")
SOURCE_SHA256 = file_sha256(ORIGINAL_SOURCE_PATH)
DESCRIPTOR_SHA256 = file_sha256(ORIGINAL_DESCRIPTOR_PATH)
SNAPSHOT_ROOT = Path(tempfile.mkdtemp(prefix="e2e-arena-autodata-"))
atexit.register(shutil.rmtree, SNAPSHOT_ROOT, ignore_errors=True)
SNAPSHOT_SOURCE_PATH = SNAPSHOT_ROOT / "source" / SOURCE_SHA256 / ORIGINAL_SOURCE_PATH.name
SNAPSHOT_DESCRIPTOR_PATH = SNAPSHOT_ROOT / "descriptor" / DESCRIPTOR_SHA256 / ORIGINAL_DESCRIPTOR_PATH.name
try:
    SNAPSHOT_SOURCE_PATH.parent.mkdir(parents=True)
    SNAPSHOT_DESCRIPTOR_PATH.parent.mkdir(parents=True)
    shutil.copyfile(ORIGINAL_SOURCE_PATH, SNAPSHOT_SOURCE_PATH)
    shutil.copyfile(ORIGINAL_DESCRIPTOR_PATH, SNAPSHOT_DESCRIPTOR_PATH)
    if file_sha256(SNAPSHOT_SOURCE_PATH) != SOURCE_SHA256:
        raise RuntimeError("Arena source snapshot changed while being copied")
    if file_sha256(SNAPSHOT_DESCRIPTOR_PATH) != DESCRIPTOR_SHA256:
        raise RuntimeError("Arena task descriptor snapshot changed while being copied")
    SNAPSHOT_SOURCE_PATH.chmod(0o444)
    SNAPSHOT_DESCRIPTOR_PATH.chmod(0o444)
except BaseException:
    shutil.rmtree(SNAPSHOT_ROOT, ignore_errors=True)
    raise
args_cli.input_file = str(SNAPSHOT_SOURCE_PATH)
args_cli.task_descriptor = str(SNAPSHOT_DESCRIPTOR_PATH)
args_cli.output_file = str(OUTPUT_PATH)
args_cli.result_file = str(RESULT_PATH)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import asyncio  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import traceback  # noqa: E402
from typing import Any  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaac_autodata_core import DataGenerator, get_algorithm  # noqa: E402
from isaac_autodata_interfaces.env import env_loop, setup_env_config, setup_output_paths  # noqa: E402
from isaac_autodata_interfaces.tasks.task_descriptor import TaskDescriptor  # noqa: E402

from e2e_robot_workflows.arena.mimic import compose_arena_mimic_environment  # noqa: E402
from e2e_robot_workflows.arena.recorders import make_recorder_manager_cfg  # noqa: E402
from e2e_robot_workflows.autodata.arena import (  # noqa: E402
    finalize_arena_generation,
    make_arena_datastream,
    qualify_and_export_final_state,
    run_generation_with_recovery,
    validate_generated_episode_counts,
)
from e2e_robot_workflows.data.artifact_paths import atomic_write_text  # noqa: E402
from e2e_robot_workflows.data.dataset_manifest import (  # noqa: E402
    stamp_autodata_dataset,
    validate_arena_recording_contract,
)
from e2e_robot_workflows.environments import STATIC_APPLE_ENVIRONMENT  # noqa: E402

# Bound contact settling without weakening the shared success predicate.
GENERATION_SETTLE_TIMEOUT_SECONDS = 5.0


def _write_result(result: dict[str, Any]) -> None:
    atomic_write_text(RESULT_PATH, json.dumps(result, indent=2, sort_keys=True) + "\n")


def _record_artifacts(result: dict[str, Any]) -> None:
    """Validate, stamp, and hash artifacts only after the recorder has closed."""

    output_path = OUTPUT_PATH
    if result.get("status") == "ok":
        validate_generated_episode_counts(
            output_path,
            successful_expected=result["num_success"],
            failed_expected=result["num_failures"],
        )
        stamp_autodata_dataset(
            output_path,
            source_dataset=SNAPSHOT_SOURCE_PATH,
            stage="generation",
            seed=result["generation_seed"],
            scene_randomization=result.get("scene_randomization"),
        )
    candidates = {
        "generated": output_path,
        "failed": FAILED_OUTPUT_PATH,
    }
    result["artifacts"] = {
        name: {"path": str(path), "sha256": file_sha256(path)} for name, path in candidates.items() if path.is_file()
    }


async def _run_generator(
    *,
    env_id: int,
    reset_queue: asyncio.Queue,
    action_queue: asyncio.Queue,
    generator: DataGenerator,
    success_term: Any,
    stats: dict[str, int],
    result_state: dict[str, Any],
    requested_successes: int,
    max_attempts: int,
    max_settle_frames: int,
) -> None:
    """Run until the success target or bounded attempt limit is reached."""

    while True:
        if stats["num_success"] >= requested_successes or stats["num_attempts_started"] >= max_attempts:
            return
        stats["num_attempts_started"] += 1
        try:
            await generator.generate(
                env_id=env_id,
                success_term=success_term,
                env_reset_queue=reset_queue,
                env_action_queue=action_queue,
                export_demo=False,
            )
        except Exception:
            traceback.print_exc()
            raise
        final_success = await qualify_and_export_final_state(
            env=generator.datastream.get_env(),
            env_id=env_id,
            success_term=success_term,
            action_queue=action_queue,
            hold_frames=STATIC_APPLE_ENVIRONMENT.success_hold_frames,
            max_settle_frames=max_settle_frames,
        )
        stats["num_attempts_completed"] += 1
        stats["num_success" if final_success else "num_failures"] += 1
        stats["num_attempts"] += 1
        result_state.update(stats)
        _write_result(result_state)


def main() -> int:
    """Compose Arena locally, run a finite number of attempts, and retain a result."""

    result: dict[str, Any] = {
        "status": "running",
        "run_id": args_cli.run_id,
        "lane": "arena",
        "requested_successes": args_cli.requested_successes,
        "max_attempts": args_cli.max_attempts,
        "settle_timeout_seconds": GENERATION_SETTLE_TIMEOUT_SECONDS,
        "source_sha256": SOURCE_SHA256,
        "source_layout": None,
        "task_descriptor_sha256": DESCRIPTOR_SHA256,
        "num_success": 0,
        "num_failures": 0,
        "num_aborted": 0,
        "num_attempts": 0,
        "num_attempts_started": 0,
        "num_attempts_completed": 0,
    }
    env = None
    event_loop = None
    exit_code = 2
    stats = {
        "num_success": 0,
        "num_failures": 0,
        "num_aborted": 0,
        "num_attempts": 0,
        "num_attempts_started": 0,
        "num_attempts_completed": 0,
    }
    try:
        source_manifest = validate_arena_recording_contract(
            args_cli.input_file,
            manipulation_hand=args_cli.manipulation_hand,
            require_success_boundary=True,
            qualification=True,
        )
        result["source_layout"] = source_manifest["action"]["layout"]
        result["embodiment"] = source_manifest["action"]["controller"]
        descriptor_contract = source_manifest["task"].get("autodata_descriptor")
        if descriptor_contract is not None and (
            descriptor_contract.get("sha256") != DESCRIPTOR_SHA256
            or Path(descriptor_contract.get("path", "")).name != ORIGINAL_DESCRIPTOR_PATH.name
        ):
            raise RuntimeError("selected Arena task descriptor does not match the source manifest")
        _write_result(result)
        descriptor = TaskDescriptor.from_yaml(args_cli.task_descriptor)
        policy = descriptor.get_generation_policy()
        if args_cli.generation_seed is not None:
            policy.seed = args_cli.generation_seed
        result["generation_seed"] = policy.seed
        policy.num_trials = args_cli.requested_successes
        policy.guarantee_success = True
        policy.keep_failed = True
        policy.use_skillgen = False

        random.seed(policy.seed)
        np.random.seed(policy.seed)
        torch.manual_seed(policy.seed)

        composed = compose_arena_mimic_environment(
            device=args_cli.device,
            num_envs=1,
            enable_cameras=bool(args_cli.enable_cameras),
            manipulation_hand=args_cli.manipulation_hand,
            seed=policy.seed,
            embodiment=source_manifest["action"]["controller"],
            table_height_offset_range_m=args_cli.table_height_offset_range_m,
        )
        randomization = getattr(composed.env_cfg, "e2e_table_height_randomization", None)
        result["scene_randomization"] = (
            None if randomization is None else scene_randomization_manifest(randomization, seed=policy.seed)
        )
        if result["scene_randomization"] is not None:
            print("Experimental table-height retargeting: qualify generation yield for the selected robot and range.")
        output_dir, output_name = setup_output_paths(args_cli.output_file)
        recorder_cfg = make_recorder_manager_cfg(composed.env_cfg)
        env_cfg, success_term = setup_env_config(
            env_name=composed.task_id,
            output_dir=output_dir,
            output_file_name=output_name,
            num_envs=1,
            device=args_cli.device,
            generation_policy_params=policy,
            recorder_cfg=recorder_cfg,
        )
        env_cfg.recorders.export_in_close = False
        env_cfg.e2e_h2_prepare_on_reset = True
        control_period_seconds = env_cfg.sim.dt * env_cfg.decimation
        max_settle_frames = math.ceil(GENERATION_SETTLE_TIMEOUT_SECONDS / control_period_seconds)
        result["settle_timeout_frames"] = max_settle_frames
        env = gym.make(composed.task_id, cfg=env_cfg, **composed.env_kwargs).unwrapped
        env.reset()

        datastream = make_arena_datastream(
            env=env,
            task_descriptor=descriptor,
            source_dataset_path=args_cli.input_file,
            uses_start_signals=False,
        )
        print(f"Loaded {datastream.num_source_demos} Arena source episodes")

        generator = DataGenerator(datastream=datastream, algorithm=get_algorithm("dexmimicgen"))
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)

        def worker_factory(reset_queue, action_queue):
            return _run_generator(
                env_id=0,
                reset_queue=reset_queue,
                action_queue=action_queue,
                generator=generator,
                success_term=success_term,
                stats=stats,
                result_state=result,
                requested_successes=args_cli.requested_successes,
                max_attempts=args_cli.max_attempts,
                max_settle_frames=max_settle_frames,
            )

        run_generation_with_recovery(
            env=env,
            event_loop=event_loop,
            env_loop=env_loop,
            worker_factory=worker_factory,
            policy=policy,
            stats=stats,
            result=result,
            write_result=_write_result,
        )
        # The bounded worker returns after reaching the success target or exhausting
        # the attempt budget. AutoData's env loop may report that the worker finished
        # while the action queue is idle, so the shared counters own completion.
        if stats["num_attempts_started"] != stats["num_attempts_completed"]:
            raise RuntimeError("generation ended with an incomplete attempt")
        if stats["num_attempts"] != stats["num_success"] + stats["num_failures"] + stats["num_aborted"]:
            raise RuntimeError("generation attempt outcomes do not reconcile")
        if stats["num_success"] < args_cli.requested_successes and stats["num_attempts"] < args_cli.max_attempts:
            raise RuntimeError("generation ended before reaching either bounded completion condition")
        if stats["num_success"] < args_cli.requested_successes:
            raise RuntimeError(
                f"generated {stats['num_success']} successes in {stats['num_attempts']} bounded attempts"
            )
        if file_sha256(ORIGINAL_SOURCE_PATH) != SOURCE_SHA256:
            raise RuntimeError("Arena source dataset changed during generation")
        if file_sha256(ORIGINAL_DESCRIPTOR_PATH) != DESCRIPTOR_SHA256:
            raise RuntimeError("Arena task descriptor changed during generation")
        result["status"] = "ok"
        exit_code = 0
    except Exception as error:  # noqa: BLE001 - preserve structured failure evidence
        result["status"] = "error"
        result["failure"] = repr(error)
    finally:
        exit_code = finalize_arena_generation(
            event_loop=event_loop,
            environment=env,
            result=result,
            stats=stats,
            exit_code=exit_code,
            record_artifacts=_record_artifacts,
            write_result=_write_result,
        )
    return exit_code


if __name__ == "__main__":
    exit_code = 2
    try:
        exit_code = main()
    finally:
        shutil.rmtree(SNAPSHOT_ROOT, ignore_errors=True)
        simulation_app.close(exit_code=exit_code)
    raise SystemExit(exit_code)

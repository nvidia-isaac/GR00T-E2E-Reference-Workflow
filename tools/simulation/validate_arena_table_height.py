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

"""Validate seeded height resets, physical support, and live-height scoring."""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
from pathlib import Path
from typing import Any

from validation_runner import run_validation
from validation_support import assert_finite, assert_not_done, collect_runtime_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_PATH = "/tmp/e2e_arena_table_height_validation.json"

PROBE_OFFSETS_M = (0.0, 0.03, -0.03, 0.03)
SETTLE_STEPS = 60
PLACEMENT_SETTLE_STEPS = 100
# Upper bound: placement stops when both success predicates agree.
TOTAL_CONTROL_STEPS = SETTLE_STEPS * len(PROBE_OFFSETS_M) + PLACEMENT_SETTLE_STEPS
SETTLE_TOLERANCE_M = 0.02  # Spawn airgap and the plate's dish.
STATIONARY_TOLERANCE_M = 1e-4  # Float32 pose round-trips.


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", default=os.environ.get("RESULT_PATH", DEFAULT_RESULT_PATH))
    return parser.parse_args()


def _portable_error_message(error: BaseException) -> str:
    """Keep structured artifacts useful without persisting host/container paths."""

    message = str(error)
    for path in (str(REPO_ROOT), os.environ.get("E2E_ARENA_ROOT", "")):
        if path and Path(path).is_absolute():
            message = message.replace(path, "<repository>" if path == str(REPO_ROOT) else "<arena>")
    return message


def _local_height(env: Any, name: str) -> float:
    """Return one scene body's env-local root height."""

    return float(env.scene[name].data.root_pose_w.torch[0, 2] - env.scene.env_origins[0, 2])


def _local_position(env: Any, name: str) -> list[float]:
    """Return one scene body's env-local root position."""

    return (env.scene[name].data.root_pose_w.torch[0, :3] - env.scene.env_origins[0]).tolist()


def _never_done(env: Any, **_: Any) -> Any:
    """Termination stand-in that never ends an episode."""

    import torch

    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _run_simulation(result: dict[str, Any]) -> Any:
    result["stage"] = "simulation"

    import gymnasium as gym
    import torch
    from isaaclab.assets import RigidObjectCfg

    from e2e_robot_workflows.arena.composition import compose
    from e2e_robot_workflows.arena.embodiment_capabilities import G1_ARENA_TELEOP_EMBODIMENT
    from e2e_robot_workflows.arena.g1_actions import hold_action, set_navigate_command
    from e2e_robot_workflows.environments import STATIC_APPLE_ENVIRONMENT
    from e2e_robot_workflows.tasks.static_apple.observations import object_held_by_link
    from e2e_robot_workflows.tasks.static_apple.success import Outcome, evaluate
    from e2e_robot_workflows.tasks.static_apple.table_height import support_height_offset_value
    from e2e_robot_workflows.tasks.static_apple.terminations import object_placed_on_target

    if not torch.cuda.is_available():
        raise AssertionError("CUDA is unavailable to the Arena table-height validation")
    result["runtime"] = collect_runtime_metadata(torch)

    definition = STATIC_APPLE_ENVIRONMENT
    support_name = definition.asset("support").name
    apple_name = definition.asset("pick_object").name
    plate_name = definition.asset("destination").name
    nominal = {
        support_name: definition.support_position[2],
        apple_name: definition.spawn_position("pick_object")[2],
        plate_name: definition.spawn_position("destination")[2],
    }
    apple_origin_m = definition.asset("pick_object").origin_above_bottom_m

    result["stage"] = "compose_environment"
    embodiment = G1_ARENA_TELEOP_EMBODIMENT
    composed = compose(
        embodiment=embodiment,
        device="cuda:0",
        disable_task_terminations=False,
        table_height_offset_range_m=(-0.03, 0.03),
    )
    env_cfg = composed.env_cfg
    env_cfg.seed = 42
    randomization = env_cfg.e2e_table_height_randomization
    support_cfg = getattr(env_cfg.scene, support_name)
    for disabled_range in (None, [0.0, 0.0]):
        nominal_cfg = compose(
            embodiment=embodiment, device="cuda:0", table_height_offset_range_m=disabled_range
        ).env_cfg
        nominal_support = getattr(nominal_cfg.scene, support_name)
        assert not isinstance(nominal_support, RigidObjectCfg)
        assert not hasattr(nominal_cfg, "e2e_table_height_randomization")
        assert nominal_support.spawn == support_cfg.spawn
        assert nominal_support.init_state.pos == support_cfg.init_state.pos
        assert nominal_support.init_state.rot == support_cfg.init_state.rot
    success_params = dict(env_cfg.terminations.success.params)
    result["composition"] = {
        "embodiment": embodiment,
        "declared_range_m": list(randomization.offset_range_m),
        "nominal_heights_m": nominal,
        "support_asset": randomization.support_asset_name,
        "support_surface_z_m": definition.support_surface_z,
    }

    result["stage"] = "make_environment"
    env = composed.make(gym)
    try:
        if support_name not in env.scene.rigid_objects:
            raise AssertionError(
                f"the randomized table must be a rigid body to be recorded and restored; "
                f"scene rigid objects are {sorted(env.scene.rigid_objects)}"
            )
        height_term = env.event_manager.get_term_cfg("randomize_support_height")
        sequences = []
        for seed in (42, 42, 99):
            env.seed(seed)
            offsets = []
            for _ in range(4):
                env.reset()
                offsets.append(support_height_offset_value(env))
            assert all(-0.03 <= offset <= 0.03 for offset in offsets)
            assert len(set(offsets)) > 1, offsets
            sequences.append(offsets)
        assert sequences[0] == sequences[1] and sequences[0] != sequences[2], sequences
        result["sampled_offsets_m"] = sequences
        env.seed(42)
        action = set_navigate_command(hold_action(env, embodiment), 0.0, 0.0, 0.0)
        probes: list[dict[str, Any]] = []
        result["probes"] = probes
        completed_steps = 0

        for index, offset in enumerate(PROBE_OFFSETS_M):
            result["stage"] = f"probe_{index}_offset_{offset:+.3f}"
            height_term.params["randomization"] = dataclasses.replace(randomization, offset_range_m=(offset, offset))
            observations, _ = env.reset()
            assert_finite("reset_observations", observations, torch)
            reset_heights = {name: _local_height(env, name) for name in nominal}
            reported_offset = support_height_offset_value(env)
            reset_apple_position = _local_position(env, apple_name)
            reset_outcome = evaluate(env)

            for _ in range(SETTLE_STEPS):
                _, _, terminated, truncated, _ = env.step(action)
                completed_steps += 1
                assert_not_done(terminated, truncated, context=f"{offset:+.3f} m settle")
            settled_heights = {name: _local_height(env, name) for name in nominal}
            table_motion = abs(settled_heights[support_name] - reset_heights[support_name])
            expected_surface = definition.support_surface_z + offset
            resting_error = settled_heights[apple_name] - (expected_surface + apple_origin_m)
            held = bool(
                object_held_by_link(
                    env,
                    task_link_name=success_params["task_link_names"][0],
                    grasp_distance_m=definition.release_clearance_m,
                    min_object_height_m=nominal[apple_name] + definition.pickup_lift_clearance_m,
                    object_cfg=success_params["object_cfg"],
                )[0]
            )
            placed = bool(object_placed_on_target(env, **success_params)[0])
            settled_apple_position = _local_position(env, apple_name)
            probes.append(
                {
                    "offset_m": offset,
                    "reset_heights_m": reset_heights,
                    "reset_outcome": str(reset_outcome),
                    "reported_offset_m": reported_offset,
                    "resting_apple_error_m": resting_error,
                    "settled_apple_position_m": settled_apple_position,
                    "settled_heights_m": settled_heights,
                    "settled_outcome": str(evaluate(env)),
                    "table_motion_m": table_motion,
                    "untouched_apple_held": held,
                    "untouched_apple_placed": placed,
                    "untouched_xy_drift_m": math.dist(reset_apple_position[:2], settled_apple_position[:2]),
                }
            )

            for name, nominal_height in nominal.items():
                written = reset_heights[name] - nominal_height
                if abs(written - offset) > STATIONARY_TOLERANCE_M:
                    raise AssertionError(
                        f"{name} reset to {written:+.4f} m from nominal, expected the table's {offset:+.4f} m"
                    )
            if abs(reported_offset - offset) > STATIONARY_TOLERANCE_M:
                raise AssertionError(
                    f"the live table reads {reported_offset:+.4f} m from nominal, expected {offset:+.4f} m"
                )
            if table_motion > STATIONARY_TOLERANCE_M:
                raise AssertionError(f"the kinematic table moved {table_motion:.4g} m while the scene settled")
            if abs(resting_error) > SETTLE_TOLERANCE_M:
                raise AssertionError(
                    f"the apple settled {resting_error:+.4f} m from a tabletop at {expected_surface:.4f} m, "
                    "so the collision geometry did not follow the recorded root"
                )
            # The holding robot can nudge a settled apple into a failure; it must not lift or place it.
            if reset_outcome is not Outcome.RUNNING:
                raise AssertionError(f"a freshly reset {offset:+.3f} m table scene scored as {reset_outcome}")
            if held or placed:
                raise AssertionError(
                    f"an untouched apple on a {offset:+.3f} m table scored as held={held} placed={placed}"
                )

        result["stage"] = "placement_on_a_raised_table"
        height_term.params["randomization"] = dataclasses.replace(randomization, offset_range_m=(0.03, 0.03))
        env.reset()
        # Suppress auto-reset so both predicates observe the successful placement, not a fresh scene.
        env.termination_manager.get_term_cfg("success").func = _never_done
        apple = env.scene[apple_name]
        plate_pose = env.scene[plate_name].data.root_pose_w.torch[:, :3].clone()
        apple_pose = apple.data.root_pose_w.torch.clone()
        apple_pose[:, :2] = plate_pose[:, :2]
        apple_pose[:, 2] = plate_pose[:, 2] + apple_origin_m + 0.01
        apple.write_root_pose_to_sim(apple_pose, env_ids=torch.tensor([0], device=env.device))
        apple.write_root_velocity_to_sim(
            torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device)
        )
        placed = False
        outcome = Outcome.RUNNING
        for _ in range(PLACEMENT_SETTLE_STEPS):
            _, _, _, truncated, _ = env.step(action)
            completed_steps += 1
            if bool(truncated.any().item()):
                raise AssertionError("the placement episode timed out before the apple settled")
            outcome = evaluate(env)
            placed = bool(object_placed_on_target(env, **success_params)[0])
            if placed and outcome is Outcome.SUCCESS:
                break
        result["placement"] = {
            "apple_position_m": _local_position(env, apple_name),
            "episode_end_suppressed": True,
            "outcome": str(outcome),
            "placed": placed,
            "plate_position_m": _local_position(env, plate_name),
            "table_height_m": _local_height(env, support_name),
            "table_offset_m": support_height_offset_value(env),
        }
        placement = result["placement"]
        if not placed or outcome is not Outcome.SUCCESS:
            raise AssertionError(
                f"an apple resting in the plate on a table {placement['table_offset_m']:+.4f} m from nominal "
                f"scored as placed={placed} outcome={outcome}"
            )

        result["completed_steps"] = completed_steps
        return env
    except BaseException:
        try:
            env.close()
        except BaseException as cleanup_error:
            result.setdefault("cleanup_errors", []).append(
                f"environment: {type(cleanup_error).__name__}: {_portable_error_message(cleanup_error)}"
            )
        raise


def main() -> int:
    args = _parse_args()

    return run_validation(
        result_path=args.result_path,
        task_id="e2e_static_apple",
        requested_steps=TOTAL_CONTROL_STEPS,
        execute=_run_simulation,
        log_label="arena-table-height-validation",
        error_message=_portable_error_message,
    )


if __name__ == "__main__":
    raise SystemExit(main())

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

"""Exercise the fixed-base static-apple environment through real GPU simulation.

This validation intentionally checks behavior, not only environment construction:
Pink must produce a joint target for a small reachable wrist command, the fixed
robot root must remain stationary, and the apple must remain supported.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from pink_validation import make_live_pink_action, resolve_g1_pink_handles, run_daqp_sanity
from validation_runner import run_validation
from validation_support import (
    assert_action_manager_reset,
    assert_finite as _assert_finite,
    assert_not_done,
    assert_zero_tensor as _assert_zero_tensor,
    collect_runtime_metadata,
    quaternion_distance_rad,
)

DEFAULT_RESULT_PATH = "/tmp/e2e_static_apple_validation.json"
DEFAULT_TASK_ID = "Isaac-E2E-G1-StaticApple-PickPlace-Abs-v0"
DEFAULT_STEPS = 64
WRIST_DISPLACEMENT_M = 0.02
MIN_WRIST_RESPONSE_M = 0.001
MIN_PINK_TARGET_DELTA_RAD = 1.0e-5
MAX_ROOT_TRANSLATION_DRIFT_M = 1.0e-4
MAX_ROOT_ROTATION_DRIFT_RAD = 1.0e-4
MAX_OBJECT_XY_DRIFT_M = 0.02
OBJECT_SUPPORT_TOLERANCE_M = 0.01
# How far the declared radius may sit inside the measured rim before it reads as stale;
# exceeding the rim is always an error (the gate would reach off the plate).
PLACEMENT_RADIUS_SLACK_M = 0.02
RIM_SWEEP_DIRECTIONS = 360


def _assert_plate_placement_radius_matches_usd(result: dict[str, Any]) -> None:
    """Check the declared plate ``placement_rim_radius_m`` against the plate USD geometry.

    The gate measures from the prim origin and the mesh can sit off-centre from it, so what
    matters is the largest circle about the origin still inside the footprint: for a convex
    footprint, ``min over directions d of (max over points of p.d)``. A bounding-box extent
    is translation-invariant and would pass an off-centre asset.
    """

    import math as _math

    from isaaclab.utils.assets import retrieve_file_path
    from pxr import Gf, Usd, UsdGeom

    from e2e_robot_workflows.environments.static_apple import STATIC_APPLE_ENVIRONMENT

    plate = STATIC_APPLE_ENVIRONMENT.asset("destination")
    stage = Usd.Stage.Open(retrieve_file_path(plate.usd_path))
    default_prim = stage.GetDefaultPrim()
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)

    # Mesh points in the default prim's frame, i.e. relative to the prim origin the gate
    # measures from.
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    origin_inverse = xform_cache.GetLocalToWorldTransform(default_prim).GetInverse()
    points_xy: list[tuple[float, float]] = []
    for prim in Usd.PrimRange(default_prim):
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            continue
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        to_origin = xform_cache.GetLocalToWorldTransform(prim) * origin_inverse
        for point in points:
            local = to_origin.Transform(Gf.Vec3d(point[0], point[1], point[2]))
            points_xy.append((local[0], local[1]))
    if not points_xy:
        raise AssertionError(f"No mesh points found in {plate.usd_path!r}; cannot verify the placement radius")

    inradius_units = min(
        max(x * _math.cos(angle) + y * _math.sin(angle) for x, y in points_xy)
        for angle in (2.0 * _math.pi * index / RIM_SWEEP_DIRECTIONS for index in range(RIM_SWEEP_DIRECTIONS))
    )
    measured_radius = inradius_units * meters_per_unit
    centroid_offset = (
        _math.hypot(sum(x for x, _ in points_xy) / len(points_xy), sum(y for _, y in points_xy) / len(points_xy))
        * meters_per_unit
    )
    declared = plate.placement_rim_radius_m
    result["plate_placement_radius"] = {
        "declared_m": declared,
        "usd_min_rim_m": measured_radius,
        "usd_centroid_offset_from_origin_m": centroid_offset,
        "meters_per_unit": meters_per_unit,
    }
    if declared > measured_radius:
        raise AssertionError(
            f"plate placement_rim_radius_m={declared:.4g} m exceeds the measured minimum rim distance "
            f"{measured_radius:.4g} m, so the placement gate reaches past the plate"
        )
    if measured_radius - declared > PLACEMENT_RADIUS_SLACK_M:
        raise AssertionError(
            f"plate placement_rim_radius_m={declared:.4g} m is {measured_radius - declared:.4g} m inside the "
            f"measured rim {measured_radius:.4g} m (slack {PLACEMENT_RADIUS_SLACK_M} m); the gate is stale"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", default=os.environ.get("RESULT_PATH", DEFAULT_RESULT_PATH))
    parser.add_argument("--task-id", default=os.environ.get("TASK_ID", DEFAULT_TASK_ID))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", str(DEFAULT_STEPS))))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    return args


def _make_pink_action(env: Any, torch: Any, left_wrist_z_offset_m: float) -> tuple[Any, dict[str, Any]]:
    """Build an absolute Pink action from the current wrist poses and hand state."""

    robot = env.scene["robot"]
    action_term = env.action_manager.get_term("upper_body_ik")
    action_dim = env.action_manager.total_action_dim
    if env.action_manager.active_terms != ["upper_body_ik"]:
        raise AssertionError(f"unexpected fixed-mode action terms: {env.action_manager.active_terms}")
    if action_dim != action_term.action_dim:
        raise AssertionError(
            f"fixed mode must expose only the Pink action term: total={action_dim}, Pink={action_term.action_dim}"
        )
    if action_term.hand_joint_dim != 14 or action_dim - action_term.hand_joint_dim != 14:
        raise AssertionError(
            f"unexpected Pink action layout: action_dim={action_dim}, hand_dim={action_term.hand_joint_dim}"
        )

    handles = resolve_g1_pink_handles(robot, action_term)
    action = make_live_pink_action(env, action_term, handles, torch)
    left_wrist_initial_position = action[0, :3].clone()
    action[:, 2] += left_wrist_z_offset_m

    pink_joint_ids, pink_joint_names = robot.find_joints(action_term.cfg.pink_controlled_joint_names)
    action_context = {
        "action_term": action_term,
        "left_wrist_id": handles["left_wrist_id"],
        "left_wrist_initial_position": left_wrist_initial_position,
        "left_wrist_target_position": action[0, :3].clone(),
        "pink_joint_ids": pink_joint_ids,
        "pink_joint_names": pink_joint_names,
        "hand_joint_names": handles["hand_joint_names"],
    }
    return action, action_context


def _assert_runtime_state(
    *,
    env: Any,
    observations: Any,
    reward: Any,
    terminated: Any,
    truncated: Any,
    initial_root_state: Any,
    initial_object_xy: Any,
    support_z: float,
    torch: Any,
) -> dict[str, float]:
    robot = env.scene["robot"]
    pick_object = env.scene["object"]
    robot_root_state = robot.data.root_state_w.torch
    object_root_state = pick_object.data.root_state_w.torch

    _assert_finite("observations", observations, torch)
    _assert_finite("reward", reward, torch)
    _assert_finite("robot.root_state_w", robot_root_state, torch)
    _assert_finite("robot.joint_pos", robot.data.joint_pos.torch, torch)
    _assert_finite("robot.joint_vel", robot.data.joint_vel.torch, torch)
    _assert_finite("object.root_state_w", object_root_state, torch)
    assert_not_done(terminated, truncated, context="fixed-base behavior validation")

    root_translation_drift = float(
        torch.linalg.vector_norm(robot_root_state[:, :3] - initial_root_state[:, :3], dim=-1).max().item()
    )
    root_rotation_drift = quaternion_distance_rad(robot_root_state[:, 3:7], initial_root_state[:, 3:7], torch)
    object_xy_drift = float(torch.linalg.vector_norm(object_root_state[:, :2] - initial_object_xy, dim=-1).max().item())
    object_local_z = float((object_root_state[:, 2] - env.scene.env_origins[:, 2]).min().item())

    if root_translation_drift > MAX_ROOT_TRANSLATION_DRIFT_M:
        raise AssertionError(
            f"fixed robot root translated {root_translation_drift:.6g} m (limit {MAX_ROOT_TRANSLATION_DRIFT_M:.6g} m)"
        )
    if root_rotation_drift > MAX_ROOT_ROTATION_DRIFT_RAD:
        raise AssertionError(
            f"fixed robot root rotated {root_rotation_drift:.6g} rad (limit {MAX_ROOT_ROTATION_DRIFT_RAD:.6g} rad)"
        )
    if object_xy_drift > MAX_OBJECT_XY_DRIFT_M:
        raise AssertionError(f"pick object drifted {object_xy_drift:.6g} m in XY (limit {MAX_OBJECT_XY_DRIFT_M:.6g} m)")
    if object_local_z < support_z:
        raise AssertionError(
            f"pick object fell below its support surface: local_z={object_local_z:.6g} m, limit={support_z:.6g} m"
        )

    return {
        "object_xy_drift_m": object_xy_drift,
        "object_local_z_m": object_local_z,
        "root_rotation_drift_rad": root_rotation_drift,
        "root_translation_drift_m": root_translation_drift,
    }


def _run_simulation(task_id: str, steps: int, result: dict[str, Any]) -> Any:
    import gymnasium as gym
    import torch

    import e2e_robot_workflows.tasks.static_apple  # noqa: F401
    from e2e_robot_workflows.tasks.static_apple.env_cfg import SHELF_SURFACE_Z, make_env_cfg

    if not torch.cuda.is_available():
        raise AssertionError("CUDA is unavailable to the static-apple validation")

    result["runtime"] = collect_runtime_metadata(torch)

    result["stage"] = "build_env_cfg"
    cfg = make_env_cfg(wbc_mode="fixed", device="cuda:0", num_envs=1)
    cfg.seed = 42
    if cfg.wbc_mode != "fixed":
        raise AssertionError(f"static-apple validation requires fixed WBC mode, got {cfg.wbc_mode!r}")
    if not cfg.scene.robot.spawn.articulation_props.fix_root_link:
        raise AssertionError("static-apple validation received a free-base robot configuration")
    if cfg.actions.lower_body_joint_pos is not None:
        raise AssertionError("static-apple validation unexpectedly includes the lower-body action")

    result["stage"] = "plate_placement_radius"
    _assert_plate_placement_radius_matches_usd(result)

    result["stage"] = "make_environment"
    env = gym.make(task_id, cfg=cfg).unwrapped
    try:
        if env.num_envs != 1 or not str(env.device).startswith("cuda"):
            raise AssertionError(f"expected one CUDA environment, got num_envs={env.num_envs}, device={env.device}")

        result["stage"] = "initial_reset"
        observations, _ = env.reset()
        _assert_finite("reset_observations", observations, torch)

        robot = env.scene["robot"]
        pick_object = env.scene["object"]
        initial_root_state = robot.data.root_state_w.torch.clone()
        initial_object_state = pick_object.data.root_state_w.torch.clone()
        action, action_context = _make_pink_action(env, torch, WRIST_DISPLACEMENT_M)
        initial_pink_joint_positions = robot.data.joint_pos.torch[:, action_context["pink_joint_ids"]].clone()
        support_z = SHELF_SURFACE_Z - OBJECT_SUPPORT_TOLERANCE_M

        max_metrics = {
            "object_xy_drift_m": 0.0,
            "pink_joint_target_delta_rad": 0.0,
            "root_rotation_drift_rad": 0.0,
            "root_translation_drift_m": 0.0,
        }
        min_object_z = float("inf")
        result["stage"] = "behavior_steps"
        for step in range(steps):
            observations, reward, terminated, truncated, _ = env.step(action)
            step_metrics = _assert_runtime_state(
                env=env,
                observations=observations,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                initial_root_state=initial_root_state,
                initial_object_xy=initial_object_state[:, :2],
                support_z=support_z,
                torch=torch,
            )
            for metric_name in (
                "object_xy_drift_m",
                "root_rotation_drift_rad",
                "root_translation_drift_m",
            ):
                max_metrics[metric_name] = max(max_metrics[metric_name], step_metrics[metric_name])
            min_object_z = min(min_object_z, step_metrics["object_local_z_m"])

            processed_actions = action_context["action_term"].processed_actions
            _assert_finite("upper_body_ik.processed_actions", processed_actions, torch)
            pink_target_delta = float(
                (processed_actions[:, : len(action_context["pink_joint_ids"])] - initial_pink_joint_positions)
                .abs()
                .max()
                .item()
            )
            max_metrics["pink_joint_target_delta_rad"] = max(
                max_metrics["pink_joint_target_delta_rad"], pink_target_delta
            )
            result["completed_steps"] = step + 1

        if max_metrics["pink_joint_target_delta_rad"] <= MIN_PINK_TARGET_DELTA_RAD:
            raise AssertionError(
                "Pink did not produce a changed joint target for the reachable wrist command: "
                f"delta={max_metrics['pink_joint_target_delta_rad']:.6g} rad"
            )

        final_left_wrist_position = robot.data.body_link_state_w.torch[0, action_context["left_wrist_id"], :3]
        command_vector = action_context["left_wrist_target_position"] - action_context["left_wrist_initial_position"]
        command_direction = command_vector / torch.linalg.vector_norm(command_vector)
        wrist_response = float(
            torch.dot(
                final_left_wrist_position - action_context["left_wrist_initial_position"], command_direction
            ).item()
        )
        if wrist_response <= MIN_WRIST_RESPONSE_M:
            raise AssertionError(
                f"left wrist did not respond along the Pink command: {wrist_response:.6g} m "
                f"(minimum {MIN_WRIST_RESPONSE_M:.6g} m)"
            )

        result["stage"] = "second_reset"
        reset_observations, _ = env.reset()
        assert_action_manager_reset(env, reset_observations, torch)
        _assert_zero_tensor("upper_body_ik.raw_actions", action_context["action_term"].raw_actions, torch)
        second_root_state = robot.data.root_state_w.torch.clone()
        second_object_state = pick_object.data.root_state_w.torch.clone()
        hold_action, _ = _make_pink_action(env, torch, 0.0)
        observations, reward, terminated, truncated, _ = env.step(hold_action)
        if not bool((env.episode_length_buf == 1).all().item()):
            raise AssertionError(f"episode length did not advance after reset: {env.episode_length_buf.tolist()}")
        post_reset_metrics = _assert_runtime_state(
            env=env,
            observations=observations,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            initial_root_state=second_root_state,
            initial_object_xy=second_object_state[:, :2],
            support_z=support_z,
            torch=torch,
        )
        post_reset_metrics["episode_length"] = int(env.episode_length_buf[0].item())

        result["simulation"] = {
            "action_dim": env.action_manager.total_action_dim,
            "control_steps": steps,
            "decimation": cfg.decimation,
            "device": str(env.device),
            "env_type": type(env).__name__,
            "hand_joint_count": len(action_context["hand_joint_names"]),
            "left_wrist_command_m": WRIST_DISPLACEMENT_M,
            "left_wrist_response_m": wrist_response,
            "max_object_xy_drift_m": max_metrics["object_xy_drift_m"],
            "max_root_rotation_drift_rad": max_metrics["root_rotation_drift_rad"],
            "max_root_translation_drift_m": max_metrics["root_translation_drift_m"],
            "min_object_local_z_m": min_object_z,
            "num_envs": env.num_envs,
            "physics_dt_s": cfg.sim.dt,
            "pink_joint_count": len(action_context["pink_joint_names"]),
            "pink_joint_target_delta_rad": max_metrics["pink_joint_target_delta_rad"],
            "post_reset_step": post_reset_metrics,
            "scene_entities": sorted(env.scene.keys()),
            "support_limit_local_z_m": support_z,
            "wbc_mode": cfg.wbc_mode,
        }
        return env
    except BaseException:
        try:
            env.close()
        except BaseException as cleanup_error:
            result.setdefault("cleanup_errors", []).append(
                f"environment: {type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise


def main() -> int:
    args = _parse_args()

    def execute(result: dict[str, Any]) -> Any:
        result["stage"] = "daqp_sanity"
        result["daqp"] = run_daqp_sanity()
        result["stage"] = "simulation"
        return _run_simulation(args.task_id, args.steps, result)

    return run_validation(
        result_path=args.result_path,
        task_id=args.task_id,
        requested_steps=args.steps,
        execute=execute,
        log_label="static-apple-validation",
        include_traceback=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())

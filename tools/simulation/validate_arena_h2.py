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

"""Smoke the E2E-owned H2 embodiment through the pinned Arena runtime."""

from __future__ import annotations

import argparse
import math
import os
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from h2_teleop_validation import H2_TELEOP_CONTROL_STEPS, qualify_h2_teleop_control
from validation_runner import run_validation
from validation_support import assert_finite, collect_runtime_metadata

from e2e_robot_workflows.data.artifact_paths import file_sha256
from e2e_robot_workflows.tasks.static_apple.grasp_contract import H2_GRASP_CRITERIA

DEFAULT_RESULT_PATH = "/tmp/e2e_arena_h2_validation.json"

_CAMERA_ALIGNMENT_MIN_REFERENCE_SEPARATION = 8.0
_CAMERA_ALIGNMENT_MAX_NOISE_RATIO = 0.15
_CAMERA_ALIGNMENT_MIN_CURRENT_SCORE = 0.5
_CAMERA_ALIGNMENT_MAX_CURRENT_DISTANCE_RATIO = 0.35
_STATION_KEEPING_STEPS = 1000
_STATION_KEEPING_SETTLE_STEPS = 200
_GRASP_CONTACT_PHASE_STEPS = H2_GRASP_CRITERIA.contact_history_samples + 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embodiment", choices=("h2_agile_joint", "h2_agile_pink"), default="h2_agile_joint")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--enable-cameras", action="store_true")
    parser.add_argument("--mimic", action="store_true", help="Compose the H2 Pink Mimic adapter and test round trips")
    parser.add_argument("--startup-only", action="store_true", help="Qualify the guarded CPU-physics XR startup")
    parser.add_argument("--station-keeping", action="store_true", help="Qualify prolonged H2 Pink neutral navigation")
    parser.add_argument("--result-path", default=os.environ.get("RESULT_PATH", DEFAULT_RESULT_PATH))
    args = parser.parse_args()
    if args.startup_only and (args.embodiment != "h2_agile_pink" or not args.mimic):
        parser.error("--startup-only requires H2 Pink Mimic")
    if args.station_keeping and args.embodiment != "h2_agile_pink":
        parser.error("--station-keeping requires --embodiment h2_agile_pink")
    if args.startup_only and args.station_keeping:
        parser.error("--startup-only cannot be combined with --station-keeping")
    return args


def _qualify_mimic_task_configs(composed_cfg: Any) -> dict[str, Any]:
    """Match both H2 handover configs to the consumer-owned AutoData descriptors."""

    import yaml
    from isaaclab.envs.mimic_env_cfg import SubTaskConstraintCoordinationScheme, SubTaskConstraintType

    from e2e_robot_workflows.arena.mimic import E2EStaticAppleMimicEnvCfg

    repo_root = Path(__file__).resolve().parents[2]
    config_by_hand = {
        "left": composed_cfg,
        "right": E2EStaticAppleMimicEnvCfg(
            pick_object_name="pick_object",
            placement_target_name="placement_target",
            manipulation_hand="right",
            embodiment="h2_agile_pink",
        ),
    }
    report: dict[str, Any] = {}
    for hand, config in config_by_hand.items():
        support = "right" if hand == "left" else "left"
        descriptor_name = (
            "static_apple_arena_h2_left_task.yaml" if hand == "left" else "static_apple_arena_h2_task.yaml"
        )
        descriptor_path = repo_root / "configs/stages/autodata" / descriptor_name
        descriptor = yaml.safe_load(descriptor_path.read_text(encoding="utf-8"))
        for eef in (hand, support):
            actual_subtasks = [
                {
                    "object_ref": subtask.object_ref,
                    "subtask_term_signal": subtask.subtask_term_signal,
                    "num_interpolation_steps": subtask.num_interpolation_steps,
                }
                for subtask in config.subtask_configs[eef]
            ]
            expected_subtasks = [
                {
                    "object_ref": subtask["object_ref"],
                    "subtask_term_signal": subtask.get("subtask_term_signal"),
                    "num_interpolation_steps": subtask["num_interpolation_steps"],
                }
                for subtask in descriptor["subtasks"][eef]
            ]
            if actual_subtasks != expected_subtasks:
                raise AssertionError(f"H2 {hand}-hand {eef} Mimic subtasks disagree with {descriptor_name}")

        body_subtasks = config.subtask_configs.get("body", [])
        if len(body_subtasks) != 1 or (
            body_subtasks[0].object_ref != "pick_object" or body_subtasks[0].subtask_term_signal is not None
        ):
            raise AssertionError("H2 Mimic body passthrough must remain one signal-free full-demo subtask")

        actual_constraints = []
        runtime_constraints: dict[tuple[str, int], dict[str, Any]] = {}
        for constraint in config.task_constraint_configs:
            pairs = [list(pair) for pair in constraint.eef_subtask_constraint_tuple]
            if any(pair[0] == "body" for pair in pairs):
                raise AssertionError("H2 Mimic coordination must not treat AGILE navigation as a pose EEF")
            actual_constraints.append(
                {
                    "constraint_type": constraint.constraint_type.name.lower(),
                    "eef_subtask_constraint_tuple": pairs,
                    "coordination_scheme": constraint.coordination_scheme.name.lower(),
                    "coordination_scheme_pos_noise_scale": constraint.coordination_scheme_pos_noise_scale,
                    "coordination_scheme_rot_noise_scale": constraint.coordination_scheme_rot_noise_scale,
                    "coordination_synchronize_start": constraint.coordination_synchronize_start,
                }
            )
            expanded = constraint.generate_runtime_subtask_constraints()
            overlap = runtime_constraints.keys() & expanded.keys()
            if overlap:
                raise AssertionError(f"H2 Mimic subtasks have overlapping constraints: {sorted(overlap)}")
            runtime_constraints.update(expanded)
        if actual_constraints != descriptor["constraints"]:
            raise AssertionError(f"H2 {hand}-hand Mimic constraints disagree with {descriptor_name}")
        expected_runtime_keys = {(hand, 1), (support, 0), (hand, 2), (support, 1)}
        if set(runtime_constraints) != expected_runtime_keys:
            raise AssertionError(f"H2 {hand}-hand runtime constraint keys changed: {sorted(runtime_constraints)}")
        if any(
            entry["type"] != SubTaskConstraintType.COORDINATION
            or entry["coordination_scheme"] != SubTaskConstraintCoordinationScheme.TRANSFORM
            for entry in runtime_constraints.values()
        ):
            raise AssertionError("H2 Mimic runtime constraints must use transform coordination")
        report[hand] = {
            "descriptor": str(descriptor_path.relative_to(repo_root)),
            "arm_subtask_counts": {eef: len(config.subtask_configs[eef]) for eef in (hand, support)},
            "body_subtask_count": len(body_subtasks),
            "runtime_constraint_keys": [list(key) for key in sorted(runtime_constraints)],
        }

    g1_config = E2EStaticAppleMimicEnvCfg(
        manipulation_hand="right",
        embodiment="g1_wbc_agile_pink",
    )
    if {name: len(subtasks) for name, subtasks in g1_config.subtask_configs.items()} != {
        "right": 2,
        "left": 1,
        "body": 1,
    } or g1_config.task_constraint_configs:
        raise AssertionError("H2-specific segmentation changed the G1 Mimic topology")
    return report


def _snapshot_tensor_tree(value: Any, torch: Any, *, path: str) -> dict[str, Any]:
    """Clone every tensor below a nested state/observation value."""

    if isinstance(value, torch.Tensor):
        return {path: value.detach().clone()}
    if isinstance(value, Mapping):
        snapshot: dict[str, Any] = {}
        for key, child in value.items():
            snapshot.update(_snapshot_tensor_tree(child, torch, path=f"{path}.{key}"))
        return snapshot
    if isinstance(value, (list, tuple)):
        snapshot = {}
        for index, child in enumerate(value):
            snapshot.update(_snapshot_tensor_tree(child, torch, path=f"{path}[{index}]"))
        return snapshot
    return {}


def _assert_tensor_snapshot_unchanged(before: dict[str, Any], after: dict[str, Any], *, label: str) -> None:
    """Fail when a render-only refresh changes a captured tensor tree."""

    if before.keys() != after.keys():
        raise AssertionError(
            f"{label} tensor structure changed during render-only refresh: "
            f"before={sorted(before)}, after={sorted(after)}"
        )
    changed = [path for path in before if not before[path].equal(after[path])]
    if changed:
        raise AssertionError(f"{label} changed during render-only refresh: {changed[:5]}")


def _score_camera_temporal_alignment(
    *,
    reference_separation: float,
    repeat_noise: float,
    candidate_to_previous: float,
    candidate_to_current: float,
) -> dict[str, float | str]:
    """Classify a production camera frame against independently forced pose references."""

    values = (reference_separation, repeat_noise, candidate_to_previous, candidate_to_current)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise AssertionError(f"H2 camera temporal alignment received invalid pixel distances: {values}")
    if reference_separation < _CAMERA_ALIGNMENT_MIN_REFERENCE_SEPARATION:
        raise AssertionError(
            f"H2 camera temporal alignment references are not visually distinct: separation={reference_separation:.6f}"
        )
    noise_limit = _CAMERA_ALIGNMENT_MAX_NOISE_RATIO * reference_separation
    if repeat_noise > noise_limit:
        raise AssertionError(
            "H2 camera temporal alignment references are not repeatable: "
            f"noise={repeat_noise:.6f}, limit={noise_limit:.6f}"
        )

    score = (candidate_to_previous - candidate_to_current) / reference_separation
    current_distance_limit = _CAMERA_ALIGNMENT_MAX_CURRENT_DISTANCE_RATIO * reference_separation + 3.0 * repeat_noise
    if score <= -_CAMERA_ALIGNMENT_MIN_CURRENT_SCORE:
        raise AssertionError(
            "H2 camera RGB is aligned to the previous head pose: "
            f"score={score:.6f}, previous={candidate_to_previous:.6f}, current={candidate_to_current:.6f}"
        )
    if score < _CAMERA_ALIGNMENT_MIN_CURRENT_SCORE or candidate_to_current > current_distance_limit:
        raise AssertionError(
            "H2 camera RGB temporal alignment is ambiguous: "
            f"score={score:.6f}, current={candidate_to_current:.6f}, limit={current_distance_limit:.6f}"
        )
    return {
        "status": "aligned_to_current_pose",
        "current_alignment_score": score,
        "current_distance_limit": current_distance_limit,
        "noise_limit": noise_limit,
    }


def _qualify_render_only_refresh(
    env: Any,
    torch: Any,
    refresh: Callable[[Any], None],
) -> dict[str, Any]:
    """Prove a viewport refresh renders once without advancing or mutating env state."""

    physics_step_before = int(env.sim.get_physics_step_count())
    environment_step_before = int(env._sim_step_counter)
    render_generation_before = int(env.sim.render_generation)
    scene_before = _snapshot_tensor_tree(env.scene.get_state(), torch, path="scene")
    observations_before = _snapshot_tensor_tree(env.obs_buf, torch, path="observations")
    if not scene_before or not observations_before:
        raise AssertionError("render-only refresh qualification captured no scene or observation tensors")

    refresh(env)

    physics_step_after = int(env.sim.get_physics_step_count())
    environment_step_after = int(env._sim_step_counter)
    render_generation_after = int(env.sim.render_generation)
    scene_after = _snapshot_tensor_tree(env.scene.get_state(), torch, path="scene")
    observations_after = _snapshot_tensor_tree(env.obs_buf, torch, path="observations")
    if physics_step_after != physics_step_before or environment_step_after != environment_step_before:
        raise AssertionError(
            "render-only refresh advanced simulation: "
            f"physics={physics_step_before}->{physics_step_after}, "
            f"environment={environment_step_before}->{environment_step_after}"
        )
    if render_generation_after <= render_generation_before:
        raise AssertionError(
            "render-only refresh did not execute a render: "
            f"generation={render_generation_before}->{render_generation_after}"
        )
    _assert_tensor_snapshot_unchanged(scene_before, scene_after, label="scene state")
    _assert_tensor_snapshot_unchanged(observations_before, observations_after, label="observation buffer")
    return {
        "environment_step_counter_after": environment_step_after,
        "environment_step_counter_before": environment_step_before,
        "observation_tensor_count": len(observations_before),
        "observations_unchanged": True,
        "physics_step_count_after": physics_step_after,
        "physics_step_count_before": physics_step_before,
        "render_generation_after": render_generation_after,
        "render_generation_before": render_generation_before,
        "render_generation_delta": render_generation_after - render_generation_before,
        "scene_state_unchanged": True,
        "scene_tensor_count": len(scene_before),
    }


def _pelvis_relative_pose(robot: Any, frame_name: str, torch: Any) -> Any:
    from isaaclab.utils import math as math_utils

    pelvis = robot.data.body_link_pose_w.torch[:, robot.body_names.index("pelvis")]
    frame = robot.data.body_link_pose_w.torch[:, robot.body_names.index(frame_name)]
    relative_pos, relative_quat = math_utils.subtract_frame_transforms(
        pelvis[:, :3],
        pelvis[:, 3:7],
        frame[:, :3],
        frame[:, 3:7],
    )
    return torch.cat((relative_pos, relative_quat), dim=-1)


def _task_object_bounding_radii(env: Any) -> dict[str, float]:
    """Measure conservative task-object radii from their composed render bounds."""

    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    radii: dict[str, float] = {}
    env_path = env.scene.env_prim_paths[0]
    for object_name in ("pick_object", "placement_target"):
        prim = env.sim.stage.GetPrimAtPath(f"{env_path}/{object_name}")
        if not prim.IsValid():
            raise AssertionError(f"H2 camera visibility cannot find {object_name!r}")
        size = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetSize()
        radius = 0.5 * sum(float(component) ** 2 for component in size) ** 0.5
        if not 0.0 < radius < 0.5:
            raise AssertionError(f"H2 camera visibility measured invalid {object_name} radius {radius:.6f} m")
        radii[object_name] = radius
    return radii


def _camera_task_visibility(
    env: Any,
    torch: Any,
    object_radii: dict[str, float],
    *,
    sample: str,
) -> dict[str, Any]:
    """Project conservative live task-object bounds into the head-camera image."""

    from isaaclab.utils import math as math_utils

    camera = env.scene["robot_head_cam"]
    camera.update(0.0, force_recompute=True)
    position = camera.data.pos_w.torch[0]
    orientation = camera.data.quat_w_world.torch[0]
    intrinsics = camera.data.intrinsic_matrices.torch[0]
    height, width = camera.data.image_shape
    margin = 24.0
    robot = env.scene["robot"]
    head_position = robot.data.body_link_pose_w.torch[0, robot.body_names.index("head_yaw_link"), :3]
    expected_mount_distance = sum(float(value) ** 2 for value in camera.cfg.offset.pos) ** 0.5
    mount_distance = float(torch.linalg.vector_norm(position - head_position).item())
    mount_error = abs(mount_distance - expected_mount_distance)
    if mount_error > 1e-3:
        raise AssertionError(
            "H2 head camera is not mounted to the live head rigid body: "
            f"distance={mount_distance:.6f} m, expected={expected_mount_distance:.6f} m, "
            f"camera_position={position.tolist()}, head_position={head_position.tolist()}"
        )

    projections: dict[str, Any] = {
        "sample": sample,
        "camera_position_m": position.tolist(),
        "camera_orientation_xyzw": orientation.tolist(),
        "head_mount_position_error_m": mount_error,
        "objects": {},
    }
    for object_name in ("pick_object", "placement_target"):
        object_pose = env.scene[object_name].data.root_link_pose_w.torch
        camera_position = math_utils.quat_apply_inverse(orientation, object_pose[0, :3] - position)
        forward = float(camera_position[0].item())
        radius_m = object_radii[object_name]
        if forward <= radius_m:
            raise AssertionError(
                f"H2 head camera points away from {object_name}: depth={forward:.6f} m, radius={radius_m:.6f} m"
            )
        pixel_x = float((intrinsics[0, 2] - intrinsics[0, 0] * camera_position[1] / forward).item())
        pixel_y = float((intrinsics[1, 2] - intrinsics[1, 1] * camera_position[2] / forward).item())
        pixel_radius = max(float(intrinsics[0, 0]), float(intrinsics[1, 1])) * radius_m / (forward - radius_m)
        if not (
            margin <= pixel_x - pixel_radius
            and pixel_x + pixel_radius < width - margin
            and margin <= pixel_y - pixel_radius
            and pixel_y + pixel_radius < height - margin
        ):
            raise AssertionError(
                f"H2 head camera does not frame the full {object_name} bound at {sample}: "
                f"pixel=({pixel_x:.2f}, {pixel_y:.2f}), radius={pixel_radius:.2f}px, "
                f"image={width}x{height}, margin={margin:.0f}, "
                f"camera_position={position.tolist()}, camera_orientation_xyzw={orientation.tolist()}, "
                f"object_in_camera={camera_position.tolist()}, intrinsics={intrinsics.tolist()}"
            )
        projections["objects"][object_name] = {
            "bounding_radius_m": radius_m,
            "camera_depth_m": forward,
            "pixel_radius": pixel_radius,
            "pixel_xy": [pixel_x, pixel_y],
        }
    return projections


def _urdf_position_limits(path: Path) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for joint in ET.parse(path).getroot().findall("joint"):
        if joint.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            raise AssertionError(f"URDF joint {joint.get('name')!r} has no bounded position limit")
        limits[str(joint.get("name"))] = (float(limit.get("lower")), float(limit.get("upper")))
    return limits


def _qualify_kinematics(env: Any, robot: Any, contract: Any, torch: Any) -> dict[str, Any]:
    """Compare live limits and pelvis-relative flange FK against the pinned URDF."""

    import numpy as np
    import pinocchio as pin

    urdf_path = contract.resolve_urdf()
    urdf_limits = _urdf_position_limits(urdf_path)
    live_limits_tensor = robot.data.joint_pos_limits
    live_limits = getattr(live_limits_tensor, "torch", live_limits_tensor)[0].detach().cpu().numpy()
    limit_error = 0.0
    for joint_index, joint_name in enumerate(robot.joint_names):
        expected = np.asarray(urdf_limits[joint_name])
        limit_error = max(limit_error, float(np.max(np.abs(live_limits[joint_index] - expected))))
    if limit_error > 1e-6:
        raise AssertionError(f"PhysX/URDF position limits differ by {limit_error:.9f} rad")

    model = pin.buildModelFromUrdf(str(urdf_path))
    model_data = model.createData()
    pin_q_indices = {name: model.joints[model.getJointId(name)].idx_q for name in contract.expected_joint_order}
    frame_ids = {name: model.getFrameId(name) for name in contract.flange_frames}
    generator = torch.Generator(device="cpu").manual_seed(20260902)
    neutral = robot.data.default_joint_pos[0].detach().cpu()
    samples = [neutral]
    for _ in range(32):
        unit = torch.rand(len(robot.joint_names), generator=generator)
        midpoint = 0.5 * (torch.from_numpy(live_limits[:, 0]) + torch.from_numpy(live_limits[:, 1]))
        half_range = 0.1 * (torch.from_numpy(live_limits[:, 1]) - torch.from_numpy(live_limits[:, 0]))
        samples.append(midpoint + (2.0 * unit - 1.0) * half_range)

    max_position_error = 0.0
    max_orientation_error = 0.0
    worst_position: dict[str, Any] = {}
    zero_velocity = torch.zeros_like(robot.data.joint_vel.torch)
    for sample_index, sample in enumerate(samples):
        live_q = sample.to(device=env.device, dtype=robot.data.joint_pos.torch.dtype).unsqueeze(0)
        robot.write_joint_state_to_sim(live_q, zero_velocity)
        env.sim.forward()
        robot.update(0.0)

        pin_q = np.zeros(model.nq)
        sample_np = sample.numpy()
        for live_index, joint_name in enumerate(robot.joint_names):
            pin_q[pin_q_indices[joint_name]] = sample_np[live_index]
        pin.forwardKinematics(model, model_data, pin_q)
        pin.updateFramePlacements(model, model_data)

        for frame_name in contract.flange_frames:
            live_pose = _pelvis_relative_pose(robot, frame_name, torch)[0].detach().cpu().numpy()
            pin_pose = model_data.oMf[frame_ids[frame_name]]
            pin_quat = pin.Quaternion(pin_pose.rotation).coeffs()
            position_error = float(np.linalg.norm(live_pose[:3] - pin_pose.translation))
            if position_error > max_position_error:
                max_position_error = position_error
                worst_position = {
                    "frame": frame_name,
                    "live": live_pose[:3].tolist(),
                    "pinocchio": pin_pose.translation.tolist(),
                    "sample": sample_index,
                }
            live_quat = live_pose[3:7] / np.linalg.norm(live_pose[3:7])
            pin_quat = pin_quat / np.linalg.norm(pin_quat)
            quaternion_dot = float(abs(np.dot(live_quat, pin_quat)))
            orientation_error = 2.0 * np.arccos(np.clip(quaternion_dot, -1.0, 1.0))
            max_orientation_error = max(max_orientation_error, float(orientation_error))

    env.reset()
    position_threshold_m = 0.001
    orientation_threshold_rad = np.deg2rad(0.1)
    if max_position_error > position_threshold_m or max_orientation_error > orientation_threshold_rad:
        raise AssertionError(
            "PhysX/Pinocchio flange FK exceeds tolerance: "
            f"position={max_position_error:.9f} m, orientation={np.rad2deg(max_orientation_error):.9f} deg, "
            f"worst={worst_position}"
        )
    return {
        "limit_max_abs_error_rad": limit_error,
        "max_orientation_error_deg": float(np.rad2deg(max_orientation_error)),
        "max_position_error_m": max_position_error,
        "sample_count": len(samples),
    }


def _qualify_hand_limits(robot: Any, contract: Any) -> dict[str, Any]:
    live_limits_tensor = robot.data.joint_pos_limits
    live_limits = getattr(live_limits_tensor, "torch", live_limits_tensor)[0].detach().cpu().numpy()
    minimum_margin = float("inf")
    for targets in (contract.hand_open_targets, contract.hand_closed_targets):
        for name, target in targets.items():
            lower, upper = live_limits[robot.joint_names.index(name)]
            if target < lower - 1e-6 or target > upper + 1e-6:
                raise AssertionError(f"H2 Sharpa target for {name}={target} is outside [{lower}, {upper}]")
            minimum_margin = min(minimum_margin, float(target - lower), float(upper - target))
    return {"target_count": 2 * len(contract.hand_joint_order), "minimum_limit_margin_rad": minimum_margin}


def _physics_material_binding_count(env: Any, material_path: str) -> int:
    from pxr import UsdShade

    count = 0
    robot_root = f"{env.scene.env_prim_paths[0]}/Robot"
    for prim in env.sim.stage.Traverse():
        if str(prim.GetPath()).startswith(robot_root):
            binding = UsdShade.MaterialBindingAPI(prim).GetDirectBinding("physics")
            count += int(str(binding.GetMaterialPath()) == material_path)
    if count == 0:
        raise AssertionError(f"no H2 collision prim is bound to {material_path}")
    return count


def _step_finite(env: Any, robot: Any, action: Any, count: int, torch: Any, label: str) -> int:
    for index in range(count):
        observations, _, _, _, _ = env.step(action)
        assert_finite(f"{label}.observations[{index}]", observations, torch)
        assert_finite(f"{label}.joint_pos[{index}]", robot.data.joint_pos.torch, torch)
    return count


def _qualify_camera_reset_alignment(
    env: Any,
    robot: Any,
    embodiment: str,
    object_radii: dict[str, float],
    torch: Any,
) -> dict[str, Any]:
    """Reproduce recorder warmup/reset and qualify its first recorder-visible RGB frame."""

    from e2e_robot_workflows.arena.h2_actions import h2_hold_action
    from e2e_robot_workflows.devices.recording_runtime import refresh_reset_view

    warmup_steps = 30
    _step_finite(env, robot, h2_hold_action(env, embodiment), warmup_steps, torch, "camera_reset_warmup")
    before = _camera_task_visibility(env, torch, object_radii, sample="before_reset")
    camera = env.scene["robot_head_cam"]
    live_position = camera.data.pos_w.torch.clone()
    live_orientation = camera.data.quat_w_world.torch.clone()
    pre_poison_rgb = camera.data.output["rgb"].torch.clone()
    poison_position = live_position + live_position.new_tensor([[0.0, 0.0, 1.0]])
    camera.set_world_poses(poison_position, live_orientation, convention="world")
    camera.reset()
    poison_refresh = _qualify_render_only_refresh(env, torch, refresh_reset_view)
    poison_rgb = camera.data.output["rgb"].torch.clone()
    pre_poison_difference = float(torch.mean(torch.abs(poison_rgb.float() - pre_poison_rgb.float())).item())
    if pre_poison_difference < 1.0:
        raise AssertionError(
            "H2 camera poison pose did not produce a distinct rendered frame: "
            f"mean_absolute_difference={pre_poison_difference:.6f}"
        )

    # Match reset_episode(): reset the simulator and recorder, reset the env,
    # then refresh the waiting viewport without taking a physics step.
    env.sim.reset()
    env.recorder_manager.reset()
    observations, _ = env.reset()
    reset_refresh = _qualify_render_only_refresh(env, torch, refresh_reset_view)
    camera_obs = observations.get("camera_obs")
    if not isinstance(camera_obs, dict) or not isinstance(camera_obs.get("robot_head_cam_rgb"), torch.Tensor):
        raise AssertionError("H2 reset did not return robot_head_cam_rgb")
    reset_rgb = camera_obs["robot_head_cam_rgb"]
    assert_finite("camera_reset_rgb", reset_rgb, torch)
    recorder_rgb = env.obs_buf["camera_obs"]["robot_head_cam_rgb"]
    if not torch.equal(reset_rgb, recorder_rgb):
        raise AssertionError("H2 viewport refresh replaced the first recorder-visible reset RGB frame")
    env.recorder_manager.record_pre_step()
    recorded_rgb = env.recorder_manager.get_episode(0).data["camera_obs"]["robot_head_cam_rgb"][0]
    if not torch.equal(reset_rgb[0], recorded_rgb):
        raise AssertionError("H2 camera recorder did not store the reset observation as its first frame")
    env.recorder_manager.reset()
    reset_rgb_float = reset_rgb.float()
    dynamic_range = float((torch.max(reset_rgb_float) - torch.min(reset_rgb_float)).item())
    standard_deviation = float(torch.std(reset_rgb_float).item())
    if dynamic_range < 1.0 or standard_deviation < 1e-3:
        raise AssertionError(
            f"H2 first reset RGB frame is blank: range={dynamic_range:.6f}, standard_deviation={standard_deviation:.6f}"
        )
    poison_difference = float(torch.mean(torch.abs(reset_rgb_float - poison_rgb.float())).item())
    if poison_difference < 1.0:
        raise AssertionError(
            "H2 first recorder-visible RGB retained the poisoned pre-reset view: "
            f"mean_absolute_difference={poison_difference:.6f}"
        )
    after = _camera_task_visibility(env, torch, object_radii, sample="first_frame_after_reset")
    poison_to_reset_displacement = float(
        torch.linalg.vector_norm(torch.tensor(after["camera_position_m"]) - poison_position[0].detach().cpu()).item()
    )
    if poison_to_reset_displacement < 0.5:
        raise AssertionError(
            "H2 camera reset probe did not create enough pose change to expose a stale frame: "
            f"displacement={poison_to_reset_displacement:.6f} m"
        )
    return {
        "before_reset": before,
        "first_frame_after_reset": after,
        "first_frame_dynamic_range": dynamic_range,
        "first_frame_mean_absolute_difference_from_poison": poison_difference,
        "first_frame_shape": list(reset_rgb.shape),
        "first_frame_standard_deviation": standard_deviation,
        "first_recorder_frame_equals_reset_observation": True,
        "poison_camera_position_m": poison_position[0].tolist(),
        "poison_frame_mean_absolute_difference_from_pre_poison": pre_poison_difference,
        "poison_render_refresh": poison_refresh,
        "poison_to_reset_pose_change_m": poison_to_reset_displacement,
        "reset_render_refresh": reset_refresh,
        "warmup_steps": warmup_steps,
    }


def _h2_head_camera_world_pose(robot: Any, camera: Any, torch: Any) -> tuple[Any, Any]:
    """Compute the configured camera pose from the live H2 head pose."""

    from isaaclab.utils import math as math_utils

    head_pose = robot.data.body_link_pose_w.torch[:, robot.body_names.index("head_yaw_link")]
    offset_position = head_pose.new_tensor(camera.cfg.offset.pos).expand(head_pose.shape[0], -1)
    offset_orientation = head_pose.new_tensor(camera.cfg.offset.rot).expand(head_pose.shape[0], -1)
    return math_utils.combine_frame_transforms(
        head_pose[:, :3],
        head_pose[:, 3:7],
        offset_position,
        offset_orientation,
    )


def _camera_frame_number(camera: Any) -> int:
    frame = getattr(camera.frame, "torch", camera.frame)
    return int(frame[0].item())


def _camera_rgb_crop(rgb: Any) -> tuple[Any, list[int]]:
    """Return a fixed scene crop; no semantic labels influence pixel scoring."""

    if rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[-1] < 3:
        raise AssertionError(f"H2 camera temporal alignment expected [1,H,W,C] RGB, got {list(rgb.shape)}")
    height, width = int(rgb.shape[1]), int(rgb.shape[2])
    row_end = int(0.85 * height)
    column_start = int(0.05 * width)
    column_end = int(0.95 * width)
    if row_end <= 0 or column_end <= column_start:
        raise AssertionError(f"H2 camera temporal alignment cannot crop RGB shape {list(rgb.shape)}")
    return rgb[0, :row_end, column_start:column_end, :3].float(), [0, row_end, column_start, column_end]


def _qualify_camera_temporal_alignment(
    env: Any,
    robot: Any,
    embodiment: str,
    torch: Any,
) -> dict[str, Any]:
    """Distinguish current-pose pixels from one-control-step-stale pixels."""

    from e2e_robot_workflows.arena.h2_actions import h2_hold_action
    from e2e_robot_workflows.arena.h2_events import follow_h2_head_camera
    from e2e_robot_workflows.devices.recording_runtime import refresh_reset_view

    camera = env.scene["robot_head_cam"]
    render_interval = int(env.cfg.sim.render_interval)
    decimation = int(env.cfg.decimation)
    alignment_steps = 0
    while env._sim_step_counter % render_interval:
        if alignment_steps >= render_interval:
            raise AssertionError(
                "H2 camera temporal alignment could not reach a render boundary from the control cadence"
            )
        env.step(h2_hold_action(env, embodiment))
        alignment_steps += 1

    env.recorder_manager.reset()
    observations, _ = env.reset()
    camera_obs = observations.get("camera_obs")
    if not isinstance(camera_obs, dict) or not isinstance(camera_obs.get("robot_head_cam_rgb"), torch.Tensor):
        raise AssertionError("H2 temporal alignment reset did not return robot_head_cam_rgb")
    baseline_frame = _camera_frame_number(camera)
    if baseline_frame != 1:
        raise AssertionError(f"H2 temporal alignment expected reset camera frame 1, got {baseline_frame}")

    lead_steps = 3
    hold_action = h2_hold_action(env, embodiment)
    for index in range(lead_steps):
        observations, _, _, _, _ = env.step(hold_action)
        assert_finite(f"camera_temporal_lead.observations[{index}]", observations, torch)
        if _camera_frame_number(camera) != baseline_frame:
            raise AssertionError("H2 camera updated before its qualified 25 Hz capture boundary")

    expected_pre_render_modulo = (render_interval - decimation) % render_interval
    pre_render_modulo = int(env._sim_step_counter % render_interval)
    if pre_render_modulo != expected_pre_render_modulo:
        raise AssertionError(
            "H2 temporal probe is not one control step before a render boundary: "
            f"modulo={pre_render_modulo}, expected={expected_pre_render_modulo}"
        )

    head_joint_index = robot.joint_names.index("head_yaw_joint")
    position_limits = robot.data.joint_pos_limits[0, head_joint_index]
    previous_yaw_target = -0.12
    current_yaw_target = 0.12
    if previous_yaw_target <= float(position_limits[0]) or current_yaw_target >= float(position_limits[1]):
        raise AssertionError(
            f"H2 head-yaw limits cannot support the temporal camera probe: limits={position_limits.tolist()}"
        )

    previous_joint_positions = robot.data.joint_pos.torch.clone()
    previous_joint_velocities = robot.data.joint_vel.torch.clone()
    previous_joint_positions[:, head_joint_index] = previous_yaw_target
    previous_joint_velocities[:, head_joint_index] = 0.0
    robot.write_joint_state_to_sim(previous_joint_positions, previous_joint_velocities)
    env.sim.forward()
    robot.update(0.0)
    previous_yaw = float(robot.data.joint_pos.torch[0, head_joint_index].item())
    previous_camera_position, previous_camera_orientation = _h2_head_camera_world_pose(robot, camera, torch)
    follow_h2_head_camera(env, None)

    candidate_action = h2_hold_action(env, embodiment)
    current_joint_positions = previous_joint_positions.clone()
    current_joint_velocities = previous_joint_velocities.clone()
    current_joint_positions[:, head_joint_index] = current_yaw_target
    robot.write_joint_state_to_sim(current_joint_positions, current_joint_velocities)

    candidate_physics_before = int(env.sim.get_physics_step_count())
    candidate_environment_before = int(env._sim_step_counter)
    candidate_render_before = int(env.sim.render_generation)
    candidate_frame_before = _camera_frame_number(camera)
    observations, _, _, _, _ = env.step(candidate_action)
    candidate_physics_after = int(env.sim.get_physics_step_count())
    candidate_environment_after = int(env._sim_step_counter)
    candidate_render_after = int(env.sim.render_generation)
    candidate_frame_after = _camera_frame_number(camera)
    candidate_rgb = observations["camera_obs"]["robot_head_cam_rgb"].clone()
    assert_finite("camera_temporal_candidate_rgb", candidate_rgb, torch)
    if (
        candidate_physics_after - candidate_physics_before != decimation
        or candidate_environment_after - candidate_environment_before != decimation
        or candidate_render_after - candidate_render_before != 1
        or candidate_frame_after - candidate_frame_before != 1
        or candidate_environment_after % render_interval != 0
    ):
        raise AssertionError(
            "H2 temporal probe missed the shared render/camera boundary: "
            f"physics={candidate_physics_before}->{candidate_physics_after}, "
            f"environment={candidate_environment_before}->{candidate_environment_after}, "
            f"renders={candidate_render_before}->{candidate_render_after}, "
            f"frames={candidate_frame_before}->{candidate_frame_after}"
        )

    current_yaw = float(robot.data.joint_pos.torch[0, head_joint_index].item())
    current_camera_position, current_camera_orientation = _h2_head_camera_world_pose(robot, camera, torch)
    quaternion_dot = float(
        torch.abs(torch.sum(previous_camera_orientation[0] * current_camera_orientation[0])).clamp(max=1.0).item()
    )
    camera_pose_separation = 2.0 * math.acos(quaternion_dot)
    if camera_pose_separation < 0.15:
        raise AssertionError(
            f"H2 temporal probe did not create enough camera rotation: separation={camera_pose_separation:.6f} rad"
        )

    reference_physics_before = int(env.sim.get_physics_step_count())
    reference_environment_before = int(env._sim_step_counter)
    reference_scene = _snapshot_tensor_tree(env.scene.get_state(), torch, path="scene")
    reference_observations = _snapshot_tensor_tree(env.obs_buf, torch, path="observations")
    reference_refreshes: list[dict[str, Any]] = []

    def capture_reference(position: Any, orientation: Any) -> Any:
        camera.set_world_poses(position, orientation, convention=camera.cfg.offset.convention)
        camera.reset()
        reference_refreshes.append(_qualify_render_only_refresh(env, torch, refresh_reset_view))
        image = camera.data.output["rgb"].torch.clone()
        if _camera_frame_number(camera) != 1:
            raise AssertionError("H2 forced camera reference did not capture exactly one frame")
        return image

    try:
        previous_reference_1 = capture_reference(previous_camera_position, previous_camera_orientation)
        current_reference_1 = capture_reference(current_camera_position, current_camera_orientation)
        previous_reference_2 = capture_reference(previous_camera_position, previous_camera_orientation)
        current_reference_2 = capture_reference(current_camera_position, current_camera_orientation)

        reference_physics_after = int(env.sim.get_physics_step_count())
        reference_environment_after = int(env._sim_step_counter)
        if (
            reference_physics_after != reference_physics_before
            or reference_environment_after != reference_environment_before
        ):
            raise AssertionError(
                "H2 forced camera references advanced simulation: "
                f"physics={reference_physics_before}->{reference_physics_after}, "
                f"environment={reference_environment_before}->{reference_environment_after}"
            )
        _assert_tensor_snapshot_unchanged(
            reference_scene,
            _snapshot_tensor_tree(env.scene.get_state(), torch, path="scene"),
            label="temporal reference scene state",
        )
        _assert_tensor_snapshot_unchanged(
            reference_observations,
            _snapshot_tensor_tree(env.obs_buf, torch, path="observations"),
            label="temporal reference observation buffer",
        )

        candidate_crop, crop_bounds = _camera_rgb_crop(candidate_rgb)
        previous_crop_1, _ = _camera_rgb_crop(previous_reference_1)
        previous_crop_2, _ = _camera_rgb_crop(previous_reference_2)
        current_crop_1, _ = _camera_rgb_crop(current_reference_1)
        current_crop_2, _ = _camera_rgb_crop(current_reference_2)
        previous_reference = 0.5 * (previous_crop_1 + previous_crop_2)
        current_reference = 0.5 * (current_crop_1 + current_crop_2)

        previous_repeat_noise = float(torch.mean(torch.abs(previous_crop_1 - previous_crop_2)).item())
        current_repeat_noise = float(torch.mean(torch.abs(current_crop_1 - current_crop_2)).item())
        repeat_noise = max(previous_repeat_noise, current_repeat_noise)
        reference_separation = float(torch.mean(torch.abs(previous_reference - current_reference)).item())
        candidate_to_previous = float(torch.mean(torch.abs(candidate_crop - previous_reference)).item())
        candidate_to_current = float(torch.mean(torch.abs(candidate_crop - current_reference)).item())
        score = _score_camera_temporal_alignment(
            reference_separation=reference_separation,
            repeat_noise=repeat_noise,
            candidate_to_previous=candidate_to_previous,
            candidate_to_current=candidate_to_current,
        )
        return {
            **score,
            "alignment_steps": alignment_steps,
            "lead_control_steps": lead_steps,
            "crop_bounds_y0_y1_x0_x1": crop_bounds,
            "candidate_to_current_mean_absolute_difference": candidate_to_current,
            "candidate_to_previous_mean_absolute_difference": candidate_to_previous,
            "current_reference_repeat_noise": current_repeat_noise,
            "previous_reference_repeat_noise": previous_repeat_noise,
            "repeat_noise": repeat_noise,
            "reference_separation_mean_absolute_difference": reference_separation,
            "camera_pose_separation_rad": camera_pose_separation,
            "previous_head_yaw_rad": previous_yaw,
            "current_head_yaw_rad": current_yaw,
            "previous_camera_position_m": previous_camera_position[0].tolist(),
            "previous_camera_orientation_xyzw": previous_camera_orientation[0].tolist(),
            "current_camera_position_m": current_camera_position[0].tolist(),
            "current_camera_orientation_xyzw": current_camera_orientation[0].tolist(),
            "candidate_cadence": {
                "physics_step_count_before": candidate_physics_before,
                "physics_step_count_after": candidate_physics_after,
                "environment_step_counter_before": candidate_environment_before,
                "environment_step_counter_after": candidate_environment_after,
                "render_generation_before": candidate_render_before,
                "render_generation_after": candidate_render_after,
                "camera_frame_before": candidate_frame_before,
                "camera_frame_after": candidate_frame_after,
            },
            "reference_render_refreshes": reference_refreshes,
            "reference_physics_unchanged": True,
            "reference_environment_counter_unchanged": True,
            "reference_scene_state_unchanged": True,
            "reference_observations_unchanged": True,
        }
    finally:
        env.reset()
        env.recorder_manager.reset()


def _qualify_pink_motion(
    env: Any,
    robot: Any,
    contract: Any,
    steps: int,
    torch: Any,
    visibility_probe: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], int]:
    """Move both flanges and exercise left, right, and simultaneous whole-hand commands."""

    from e2e_robot_workflows.arena.h2_actions import h2_pink_hold_action

    phase_steps = max(steps, 20)
    settle_steps = 100
    results: dict[str, Any] = {"flanges": {}, "hands": {}, "settle_steps": settle_steps}
    completed = 0

    def flange_trial(frame_names: tuple[str, str], pose_offset: int, offset_m: float, label: str):
        nonlocal completed
        env.reset()
        completed += _step_finite(
            env,
            robot,
            h2_pink_hold_action(env),
            settle_steps,
            torch,
            f"{label}_settle",
        )
        if visibility_probe is not None:
            visibility_probe(f"{label}_settled")
        before = [_pelvis_relative_pose(robot, name, torch)[0, :3].clone() for name in frame_names]
        action = h2_pink_hold_action(env)
        action[:, pose_offset + 2] += offset_m
        completed += _step_finite(env, robot, action, phase_steps, torch, label)
        if visibility_probe is not None:
            visibility_probe(f"{label}_complete")
        after = [_pelvis_relative_pose(robot, name, torch)[0, :3].clone() for name in frame_names]
        return tuple(end - start for start, end in zip(before, after, strict=True))

    for side, pose_offset, frame_name, other_frame in (
        ("left", 0, contract.flange_frames[0], contract.flange_frames[1]),
        ("right", 7, contract.flange_frames[1], contract.flange_frames[0]),
    ):
        trials: list[dict[str, Any]] = []
        for repeat in range(3):
            frame_names = (frame_name, other_frame)
            baseline = flange_trial(frame_names, pose_offset, 0.0, f"{side}_baseline_{repeat}")
            commanded = flange_trial(frame_names, pose_offset, 0.03, f"{side}_command_{repeat}")
            response = commanded[0] - baseline[0]
            other_response = commanded[1] - baseline[1]
            baseline_drift = max(float(torch.linalg.vector_norm(value).item()) for value in baseline)
            signed_z = float(response[2].item())
            orthogonal = float(torch.linalg.vector_norm(response[:2]).item())
            other_motion = float(torch.linalg.vector_norm(other_response).item())
            tracking_error = abs(0.03 - signed_z)
            if baseline_drift > 0.02:
                raise AssertionError(f"H2 {side} hold did not converge: drift={baseline_drift:.6f} m")
            if signed_z < 0.015 or tracking_error > 0.015 or orthogonal > 0.01 or other_motion > 0.0075:
                raise AssertionError(
                    f"H2 {side} +Z flange isolation failed: response={response.tolist()}, "
                    f"other={other_response.tolist()}, baseline={baseline_drift:.6f}"
                )
            trials.append(
                {
                    "baseline_max_drift_m": baseline_drift,
                    "command_response_xyz_m": response.tolist(),
                    "other_response_xyz_m": other_response.tolist(),
                    "orthogonal_error_m": orthogonal,
                    "signed_z_response_m": signed_z,
                    "tracking_error_m": tracking_error,
                }
            )
        results["flanges"][side] = {"command_z_m": 0.03, "trials": trials}

    hand_ids = [robot.joint_names.index(name) for name in contract.hand_joint_order]
    closed = contract.hand_closed_targets
    hand_trials = (
        ("left", contract.left_hand_indices, contract.right_hand_indices),
        ("right", contract.right_hand_indices, contract.left_hand_indices),
        ("both", contract.left_hand_indices + contract.right_hand_indices, ()),
    )
    for label, selected, other_indices in hand_trials:
        env.reset()
        settle_action = h2_pink_hold_action(env)
        completed += _step_finite(env, robot, settle_action, settle_steps, torch, f"{label}_hand_settle")
        if visibility_probe is not None:
            visibility_probe(f"{label}_hand_settled")
        before = robot.data.joint_pos.torch[:, hand_ids].clone()
        action = h2_pink_hold_action(env)
        for index in selected:
            action[:, 14 + index] = closed[contract.hand_joint_order[index]]
        completed += _step_finite(env, robot, action, phase_steps, torch, f"{label}_hand_close")
        if visibility_probe is not None:
            visibility_probe(f"{label}_hand_closed")
        after_close = robot.data.joint_pos.torch[:, hand_ids].clone()
        target = action[0, 14:58]
        active = [index for index in selected if abs(float(target[index] - before[0, index])) > 0.01]
        if not active:
            raise AssertionError(f"H2 {label} hand has no joint away from its target")
        progress = (after_close[0, active] - before[0, active]) * torch.sign(target[active] - before[0, active])
        minimum_progress = float(torch.min(progress).item())
        other_motion = (
            float(torch.max(torch.abs(after_close[0, list(other_indices)] - before[0, list(other_indices)])).item())
            if other_indices
            else 0.0
        )
        joint_target_tensor = robot.data.joint_pos_target
        joint_targets = getattr(joint_target_tensor, "torch", joint_target_tensor)[0, hand_ids]
        target_error = float(torch.max(torch.abs(joint_targets[list(selected)] - target[list(selected)])).item())
        if minimum_progress < 0.015 or other_motion > 0.02 or target_error > 1e-6:
            raise AssertionError(
                f"H2 {label} hand close failed: min_progress={minimum_progress:.6f}, "
                f"other={other_motion:.6f}, target_error={target_error:.6g}"
            )

        open_action = h2_pink_hold_action(env)
        open_action[:, [14 + index for index in selected]] = contract.hand_open_position
        completed += _step_finite(env, robot, open_action, phase_steps, torch, f"{label}_hand_open")
        if visibility_probe is not None:
            visibility_probe(f"{label}_hand_opened")
        after_open = robot.data.joint_pos.torch[:, hand_ids]
        return_progress = (after_close[0, active] - after_open[0, active]) * torch.sign(
            after_close[0, active] - contract.hand_open_position
        )
        minimum_return_progress = float(torch.min(return_progress).item())
        open_error = float(torch.max(torch.abs(after_open[0, active] - contract.hand_open_position)).item())
        if minimum_return_progress < 0.015 or open_error > 0.03:
            raise AssertionError(
                f"H2 {label} hand open failed: min_progress={minimum_return_progress:.6f}, error={open_error:.6f}"
            )
        results["hands"][label] = {
            "selected_joint_count": len(selected),
            "active_joint_count": len(active),
            "minimum_selected_progress_rad": minimum_progress,
            "max_other_hand_motion_rad": other_motion,
            "max_command_target_error_rad": target_error,
            "minimum_open_return_progress_rad": minimum_return_progress,
            "max_open_error_rad": open_error,
        }
    return results, completed


def _qualify_agile_walk(env: Any, robot: Any, contract: Any, steps: int, torch: Any) -> tuple[dict[str, Any], int]:
    """Exercise a bounded velocity envelope and prove recurrent state clears on reset."""

    import math

    import numpy as np

    from e2e_robot_workflows.arena.h2_actions import h2_joint_hold_action, set_h2_navigate_command

    commands = ((0.0, 0.0, 0.0), (0.15, 0.0, 0.0), (0.0, 0.10, 0.0), (0.0, 0.0, 0.20))
    phase_steps = max(steps // len(commands), 20)
    pelvis_index = robot.body_names.index(contract.base_frame)
    initial_pose = robot.data.body_link_pose_w.torch[:, pelvis_index].clone()
    minimum_height = float("inf")
    maximum_tilt_deg = 0.0
    completed = 0
    responses: list[dict[str, Any]] = []
    # Keep a fixed upper-body command during the locomotion probe. Recapturing
    # measured joints every step makes gravity sag the next commanded target.
    hold_action = h2_joint_hold_action(env)

    def yaw(pose: Any) -> float:
        x, y, z, w = (float(value) for value in pose[3:7])
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    for command in commands:
        start_pose = robot.data.body_link_pose_w.torch[0, pelvis_index].clone()
        for index in range(phase_steps):
            action = set_h2_navigate_command(hold_action.clone(), *command)
            completed += _step_finite(env, robot, action, 1, torch, f"walk_{command}_{index}")
            if command == commands[0] and bool(torch.any(env.termination_manager.get_term("success")).item()):
                raise AssertionError("untouched H2 reset satisfied the static-apple success predicate")
            pose = robot.data.body_link_pose_w.torch[:, pelvis_index]
            minimum_height = min(minimum_height, float(pose[0, 2].item()))
            x, y = float(pose[0, 3].item()), float(pose[0, 4].item())
            up_z = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
            maximum_tilt_deg = max(maximum_tilt_deg, math.degrees(math.acos(up_z)))
        end_pose = robot.data.body_link_pose_w.torch[0, pelvis_index]
        displacement = end_pose[:2] - start_pose[:2]
        start_yaw = yaw(start_pose)
        forward = float(displacement[0]) * math.cos(start_yaw) + float(displacement[1]) * math.sin(start_yaw)
        lateral = -float(displacement[0]) * math.sin(start_yaw) + float(displacement[1]) * math.cos(start_yaw)
        yaw_change = wrap_to_pi(yaw(end_pose) - start_yaw)
        responses.append(
            {
                "command": list(command),
                "forward_displacement_m": forward,
                "lateral_displacement_m": lateral,
                "yaw_change_rad": yaw_change,
            }
        )
    if minimum_height < 0.75 or maximum_tilt_deg > 45.0:
        raise AssertionError(
            f"H2 left the stand/walk envelope: min pelvis z={minimum_height:.6f}, max tilt={maximum_tilt_deg:.3f}"
        )

    response_limits = {
        "zero_max_planar_drift_m": 0.035,
        "zero_max_yaw_drift_rad": 0.05,
        "vx_min_forward_m": 0.04,
        "vx_max_cross_axis_m": 0.06,
        "vy_min_lateral_m": 0.01,
        "vy_max_cross_axis_m": 0.04,
        "wz_min_yaw_rad": 0.06,
        "wz_max_planar_drift_m": 0.06,
    }
    zero, vx, vy, wz = responses
    if (
        math.hypot(zero["forward_displacement_m"], zero["lateral_displacement_m"])
        > response_limits["zero_max_planar_drift_m"]
        or abs(zero["yaw_change_rad"]) > response_limits["zero_max_yaw_drift_rad"]
    ):
        raise AssertionError(f"H2 AGILE zero command drifted outside its envelope: {zero}")
    if (
        vx["forward_displacement_m"] < response_limits["vx_min_forward_m"]
        or abs(vx["lateral_displacement_m"]) > response_limits["vx_max_cross_axis_m"]
    ):
        raise AssertionError(f"H2 AGILE vx response left its directional envelope: {vx}")
    if (
        vy["lateral_displacement_m"] < response_limits["vy_min_lateral_m"]
        or abs(vy["forward_displacement_m"]) > response_limits["vy_max_cross_axis_m"]
    ):
        raise AssertionError(f"H2 AGILE vy response left its directional envelope: {vy}")
    if (
        wz["yaw_change_rad"] < response_limits["wz_min_yaw_rad"]
        or math.hypot(wz["forward_displacement_m"], wz["lateral_displacement_m"])
        > response_limits["wz_max_planar_drift_m"]
    ):
        raise AssertionError(f"H2 AGILE wz response left its directional envelope: {wz}")

    lower = env.action_manager.get_term("lower_body_joint_pos")
    feedback_before = max(float(np.max(np.abs(value))) for value in lower._feedback.values())
    final_pose = robot.data.body_link_pose_w.torch[:, pelvis_index]
    displacement = torch.linalg.vector_norm(final_pose[:, :2] - initial_pose[:, :2]).item()
    env.reset()
    feedback_after = max(float(np.max(np.abs(value))) for value in lower._feedback.values())
    if feedback_before == 0.0 or feedback_after != 0.0:
        raise AssertionError(
            f"H2 AGILE feedback reset failed: before={feedback_before:.6g}, after={feedback_after:.6g}"
        )
    return (
        {
            "commands": [list(command) for command in commands],
            "phase_steps": phase_steps,
            "minimum_pelvis_height_m": minimum_height,
            "maximum_pelvis_tilt_deg": maximum_tilt_deg,
            "planar_displacement_before_reset_m": displacement,
            "feedback_max_before_reset": feedback_before,
            "feedback_max_after_reset": feedback_after,
            "response_limits": response_limits,
            "responses": responses,
        },
        completed,
    )


def _qualify_h2_grasp_contacts(env: Any, robot: Any, contract: Any, torch: Any, result: dict[str, Any]) -> int:
    """Qualify real per-hand PhysX contacts with an explicitly forced contact fixture.

    The apple is placed inside a named palm collider before each physics substep.
    This proves sensor filtering and label semantics, not a self-sustaining grasp.
    No contact-buffer values are authored to obtain positive evidence.
    """

    from isaaclab.utils import math as math_utils

    from e2e_robot_workflows.arena.h2_actions import h2_pink_hold_action
    from e2e_robot_workflows.tasks.static_apple.h2_grasp import h2_hand_body_names, object_held_by_h2_hand

    report: dict[str, Any] = {
        "qualification_scope": "forced_contact_sensor_and_predicate_fixture",
        "autonomous_grasp_qualified": False,
        "phase_control_steps": _GRASP_CONTACT_PHASE_STEPS,
        "requested_steps": 4 * _GRASP_CONTACT_PHASE_STEPS,
        "completed_steps": 0,
        "phases": [],
        "sensors": {},
    }
    result["h2_grasp_contacts"] = report
    terms = {
        hand: getattr(env.cfg.observations.subtask_terms, f"{hand}_hand_holds_object") for hand in ("left", "right")
    }
    sensors = {}
    for hand, term in terms.items():
        if term.func is not object_held_by_h2_hand:
            raise AssertionError(f"H2 {hand} pickup signal does not use the contact-aware predicate")
        sensor = env.scene[term.params["contact_sensor_cfg"].name]
        report["sensors"][hand] = {
            "name": term.params["contact_sensor_cfg"].name,
            "update_period_s": sensor.cfg.update_period,
            "history_length": sensor.cfg.history_length,
            "filter_count": sensor.contact_view.filter_count,
            "filter_body_names": [path.rsplit("/", 1)[-1] for path in sensor.cfg.filter_prim_paths_expr],
        }
        if (
            not math.isclose(sensor.cfg.update_period, env.step_dt, rel_tol=0.0, abs_tol=1e-12)
            or sensor.cfg.history_length != H2_GRASP_CRITERIA.contact_history_samples
            or sensor.contact_view.filter_count != H2_GRASP_CRITERIA.contact_bodies_per_hand
            or tuple(report["sensors"][hand]["filter_body_names"]) != h2_hand_body_names(hand)
        ):
            raise AssertionError(f"H2 {hand} grasp sensor cadence or per-body filter contract changed")
        sensors[hand] = sensor
    apple = env.scene[terms["left"].params["object_cfg"].name]
    initial_completed = result.get("completed_steps", 0)
    completed = 0

    def assert_reset_clean(phase: dict[str, Any]) -> None:
        env.reset()
        env.recorder_manager.reset()
        phase["reset_history_max_force_n"] = {
            hand: float(torch.abs(sensor.data.force_matrix_w_history.torch).max().item())
            for hand, sensor in sensors.items()
        }
        phase["reset_labels"] = {hand: bool(term.func(env, **term.params)[0].item()) for hand, term in terms.items()}
        if any(value != 0.0 for value in phase["reset_history_max_force_n"].values()) or any(
            phase["reset_labels"].values()
        ):
            raise AssertionError("H2 grasp sensor history or labels survived an episode reset")

    for hand in ("left", "right"):
        opposite = "right" if hand == "left" else "left"
        palm_name = f"{hand}_hand_C_MC"
        palm_index = robot.body_names.index(palm_name)
        palm_filter_index = h2_hand_body_names(hand).index(palm_name)
        hand_indices = [robot.joint_names.index(name) for name in contract.hand_joint_order]
        for closed in (False, True):
            phase: dict[str, Any] = {"hand": hand, "close_intent": closed, "contact_body": palm_name, "samples": []}
            report["phases"].append(phase)
            assert_reset_clean(phase)
            joints = robot.data.joint_pos.torch.clone()
            for joint_index, name in zip(hand_indices, contract.hand_joint_order, strict=True):
                joints[:, joint_index] = (
                    0.25 * contract.hand_closed_targets[name]
                    if closed and name.startswith(f"{hand}_")
                    else contract.hand_open_targets[name]
                )
            robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
            env.sim.forward()
            robot.update(0.0)
            action = h2_pink_hold_action(env)
            for index, name in enumerate(contract.hand_joint_order):
                action[:, 14 + index] = (
                    contract.hand_closed_targets[name]
                    if closed and name.startswith(f"{hand}_")
                    else contract.hand_open_targets[name]
                )
            # The pinned palm collider spans local Z=[0.0295, 0.11591] m;
            # this interior point establishes intentional overlap without a
            # machine-local trajectory or a fabricated contact-force tensor.
            palm_contact_point = joints.new_tensor([[0.0, 0.0, 0.075]])
            phase["palm_local_contact_point_m"] = palm_contact_point[0].tolist()
            for control_step in range(_GRASP_CONTACT_PHASE_STEPS):
                env.action_manager.process_action(action)
                for _ in range(env.cfg.decimation):
                    env.action_manager.apply_action()
                    env.scene.write_data_to_sim()
                    palm_pose = robot.data.body_link_pose_w.torch[:, palm_index]
                    apple_pose = apple.data.root_link_pose_w.torch.clone()
                    apple_pose[:, :3] = palm_pose[:, :3] + math_utils.quat_apply(palm_pose[:, 3:7], palm_contact_point)
                    apple.write_root_pose_to_sim(apple_pose)
                    apple.write_root_velocity_to_sim(torch.zeros_like(apple.data.root_vel_w.torch))
                    env.sim.step(render=False)
                    env._sim_step_counter += 1
                    env.scene.update(env.cfg.sim.dt)
                completed += 1
                report["completed_steps"] = completed
                result["completed_steps"] = initial_completed + completed
                histories = {side: sensor.data.force_matrix_w_history.torch for side, sensor in sensors.items()}
                labels = {side: bool(term.func(env, **term.params)[0].item()) for side, term in terms.items()}
                intended_forces = torch.linalg.vector_norm(histories[hand][0, :, 0, palm_filter_index], dim=-1)
                other_force = float(torch.linalg.vector_norm(histories[opposite], dim=-1).max().item())
                sample = {
                    "control_step": control_step + 1,
                    "palm_contact_history_n": intended_forces.tolist(),
                    "other_hand_max_contact_n": other_force,
                    "labels": labels,
                    "apple_position_m": apple.data.root_link_pos_w.torch[0].tolist(),
                }
                phase["samples"].append(sample)
                assert_finite("h2_grasp_contacts.forces", histories, torch)
                assert_finite("h2_grasp_contacts.joints", robot.data.joint_pos.torch, torch)
                assert_finite("h2_grasp_contacts.apple_pose", apple.data.root_link_pose_w.torch, torch)
                if other_force > H2_GRASP_CRITERIA.contact_force_threshold_n or labels[opposite]:
                    raise AssertionError(f"H2 {hand} palm fixture leaked contact or a held label to {opposite}")
                if not closed and labels[hand]:
                    raise AssertionError(f"H2 open {hand} palm contact was mislabeled as a grasp")
            last = phase["samples"][-1]
            if min(last["palm_contact_history_n"]) <= H2_GRASP_CRITERIA.contact_force_threshold_n:
                raise AssertionError(f"H2 {hand} contact fixture did not produce sustained measured palm contact")
            if last["labels"][hand] != closed:
                raise AssertionError(f"H2 {hand} contact fixture did not produce the expected close-aware held label")
            phase["status"] = "ok"
            # Check immediately after positive evidence so stale nonzero history
            # cannot be concealed by a later zero-contact warmup.
            reset_after: dict[str, Any] = {}
            phase["after_reset"] = reset_after
            assert_reset_clean(reset_after)
    report["status"] = "ok"
    return completed


def _qualify_station_keeping(env: Any, robot: Any, torch: Any, result: dict[str, Any]) -> int:
    """Hold one world station through settling and small bimanual arm motion."""

    from e2e_robot_workflows.arena.h2_actions import h2_pink_hold_action
    from e2e_robot_workflows.devices.h2_station_keeping import H2StationKeeping, apply_h2_station_keeping

    helper = H2StationKeeping()
    bounds = {"peak_xy_m": 0.15, "settled_xy_m": 0.05, "settled_yaw_rad": 0.1}
    report: dict[str, Any] = {
        "requested_steps": _STATION_KEEPING_STEPS,
        "completed_steps": 0,
        "settle_steps": _STATION_KEEPING_SETTLE_STEPS,
        "arm_amplitude_m": 0.005,
        "arm_period_s": 2.0,
        "bounds": bounds,
        "command_caps": {
            "linear_speed_m_s": helper.config.max_linear_speed,
            "yaw_speed_rad_s": helper.config.max_yaw_speed,
        },
        "peak_xy_error_m": 0.0,
        "peak_yaw_error_rad": 0.0,
        "settled_peak_xy_error_m": 0.0,
        "settled_peak_yaw_error_rad": 0.0,
        "peak_linear_command_m_s": 0.0,
        "peak_yaw_command_rad_s": 0.0,
        "minimum_pelvis_height_m": None,
        "maximum_pelvis_tilt_deg": 0.0,
        "recorded_command_match_steps": 0,
        "resets": 0,
    }
    result["station_keeping"] = report
    completed_before = result.get("completed_steps", 0)
    env.reset()
    env.recorder_manager.reset()
    hold_action = h2_pink_hold_action(env)
    initial_pose = robot.data.root_link_pose_w.torch[0].clone()
    assert_finite("station_keeping.initial_pose", initial_pose, torch)
    report["initial_pose_xyzw"] = initial_pose.tolist()
    report["minimum_pelvis_height_m"] = float(initial_pose[2])
    lower = env.action_manager.get_term("lower_body_joint_pos")

    def yaw(pose: Any) -> float:
        x, y, z, w = (float(value) for value in pose[3:7])
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def pose_error(pose: Any, reference: Any) -> tuple[float, float]:
        xy = float(torch.linalg.vector_norm(pose[:2] - reference[:2]).item())
        yaw_delta = yaw(pose) - yaw(reference)
        return xy, abs(math.atan2(math.sin(yaw_delta), math.cos(yaw_delta)))

    for index in range(_STATION_KEEPING_STEPS):
        report["attempted_step"] = index + 1
        action = hold_action.clone()
        motion_steps = max(index + 1 - _STATION_KEEPING_SETTLE_STEPS, 0)
        offset = report["arm_amplitude_m"] * math.sin(
            2.0 * math.pi * motion_steps * env.step_dt / report["arm_period_s"]
        )
        action[:, 2] += offset
        action[:, 9] -= offset
        action = apply_h2_station_keeping(action, robot.data, helper)
        report["last_command"] = action[0, -3:].tolist()
        report["last_arm_offset_m"] = offset
        assert_finite("station_keeping.action", action, torch)
        observations, _, terminated, truncated, _ = env.step(action)
        report["completed_steps"] = index + 1
        result["completed_steps"] = completed_before + index + 1
        report["resets"] += int(bool(torch.any(terminated | truncated).item()))
        if report["resets"] or int(env.episode_length_buf[0].item()) != index + 1:
            raise AssertionError(f"H2 station keeping reset at control step {index + 1}")
        pose = robot.data.root_link_pose_w.torch[0]
        report["final_pose_xyzw"] = pose.tolist()
        assert_finite("station_keeping.pose", pose, torch)
        assert_finite("station_keeping.joint_pos", robot.data.joint_pos.torch, torch)
        assert_finite("station_keeping.joint_vel", robot.data.joint_vel.torch, torch)
        assert_finite("station_keeping.observations", observations, torch)
        xy_error, yaw_error = pose_error(pose, initial_pose)
        x, y = float(pose[3]), float(pose[4])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))
        report["final_xy_error_m"] = xy_error
        report["final_yaw_error_rad"] = yaw_error
        report["peak_xy_error_m"] = max(report["peak_xy_error_m"], xy_error)
        report["peak_yaw_error_rad"] = max(report["peak_yaw_error_rad"], yaw_error)
        report["minimum_pelvis_height_m"] = min(report["minimum_pelvis_height_m"], float(pose[2]))
        report["maximum_pelvis_tilt_deg"] = max(report["maximum_pelvis_tilt_deg"], tilt)
        linear_command = float(torch.linalg.vector_norm(action[0, -3:-1]).item())
        yaw_command = abs(float(action[0, -1]))
        report["peak_linear_command_m_s"] = max(report["peak_linear_command_m_s"], linear_command)
        report["peak_yaw_command_rad_s"] = max(report["peak_yaw_command_rad_s"], yaw_command)
        if index + 1 >= _STATION_KEEPING_SETTLE_STEPS:
            report["settled_peak_xy_error_m"] = max(report["settled_peak_xy_error_m"], xy_error)
            report["settled_peak_yaw_error_rad"] = max(report["settled_peak_yaw_error_rad"], yaw_error)
        recorded_actions = env.recorder_manager.get_episode(0).data.get("actions", [])
        report["recorded_action_count"] = len(recorded_actions)
        if len(recorded_actions) != index + 1 or lower.locomotion_command is None:
            raise AssertionError("H2 station keeping did not retain every effective action/AGILE command")
        recorded = recorded_actions[-1].reshape(1, -1)
        report["last_recorded_command"] = recorded[0, -3:].tolist()
        report["last_agile_command"] = lower.locomotion_command[0].tolist()
        if (
            not torch.equal(recorded, action)
            or not torch.equal(env.action_manager.action, action)
            or not torch.equal(lower.locomotion_command, action[:, -3:])
        ):
            raise AssertionError("H2 station-keeping recorded action differs from the effective AGILE input")
        report["recorded_command_match_steps"] += 1
        if linear_command > helper.config.max_linear_speed + 1e-6 or yaw_command > helper.config.max_yaw_speed + 1e-6:
            raise AssertionError(f"H2 station-keeping command exceeded its caps: {report['last_command']}")
        if abs(offset) > report["arm_amplitude_m"] + 1e-9:
            raise AssertionError("H2 station-keeping arm motion exceeded the configured amplitude")
        if report["minimum_pelvis_height_m"] < 0.75 or report["maximum_pelvis_tilt_deg"] > 45.0:
            raise AssertionError("H2 station keeping left the existing upright envelope")
        if report["peak_xy_error_m"] > bounds["peak_xy_m"]:
            raise AssertionError(f"H2 station keeping exceeded its peak XY safety bound: {xy_error:.6f} m")
        if (
            report["settled_peak_xy_error_m"] > bounds["settled_xy_m"]
            or report["settled_peak_yaw_error_rad"] > bounds["settled_yaw_rad"]
        ):
            raise AssertionError(
                f"H2 station keeping left its settled envelope: XY={xy_error:.6f} m, yaw={yaw_error:.6f} rad"
            )
    report["simulated_duration_s"] = _STATION_KEEPING_STEPS * env.step_dt
    report["status"] = "ok"
    return _STATION_KEEPING_STEPS


def _run(
    embodiment: str,
    steps: int,
    enable_cameras: bool,
    mimic: bool,
    result: dict[str, Any],
    station_keeping: bool = False,
    startup_only: bool = False,
) -> Any:
    import gymnasium as gym
    import torch
    from isaaclab_teleop.xr_cfg import XrAnchorRotationMode

    from e2e_robot_workflows.arena.composition import compose
    from e2e_robot_workflows.arena.h2_embodiments import make_h2_xr_cfg
    from e2e_robot_workflows.robots.h2 import load_h2_pink_contract, load_h2_robot_config

    if not torch.cuda.is_available():
        raise AssertionError("CUDA is unavailable to the Arena H2 validation")
    contract = load_h2_pink_contract()
    robot_config = load_h2_robot_config()
    result["runtime"] = collect_runtime_metadata(torch)
    result["embodiment"] = embodiment
    result["phase_steps_argument"] = steps
    result["asset"] = {
        "arena_usd_sha256": contract.arena_usd_sha256,
        "source_contract_sha256": contract.source_contract_sha256,
        "usd_sha256": contract.simulation_usd.sha256,
        "urdf_sha256": contract.urdf_sha256,
        "variants": contract.simulation_usd.variant_selection,
    }
    policy_path = robot_config.resolve_policy()
    result["policy"] = {
        "filename": robot_config.policy_filename,
        "repository": robot_config.policy_repository,
        "revision": robot_config.policy_revision,
        "sha256": file_sha256(policy_path),
    }
    xr_cfg = make_h2_xr_cfg()
    if (
        xr_cfg.anchor_pos != (0.0, 0.0, -1.0)
        or xr_cfg.anchor_rot != (0.0, 0.0, 0.0, 1.0)
        or xr_cfg.anchor_prim_path != "/World/envs/env_0/Robot/pelvis"
        or xr_cfg.anchor_rotation_mode != XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED
        or xr_cfg.fixed_anchor_height is not True
    ):
        raise AssertionError(f"H2 XR anchor changed: {xr_cfg}")
    result["xr_anchor"] = {
        "position": list(xr_cfg.anchor_pos),
        "rotation_xyzw": list(xr_cfg.anchor_rot),
        "prim_path": xr_cfg.anchor_prim_path,
        "rotation_mode": "follow_prim_smoothed",
        "fixed_height": xr_cfg.fixed_anchor_height,
    }

    result["stage"] = "compose"
    if mimic:
        if embodiment != "h2_agile_pink":
            raise AssertionError("H2 Mimic validation requires h2_agile_pink")
        from e2e_robot_workflows.arena.mimic import compose_arena_mimic_environment

        composed = compose_arena_mimic_environment(
            embodiment=embodiment,
            device="cpu" if startup_only else "cuda:0",
            num_envs=1,
            enable_cameras=enable_cameras,
            manipulation_hand="left",
            seed=42,
        )
    else:
        composed = compose(
            embodiment=embodiment,
            device="cuda:0",
            enable_cameras=enable_cameras,
            disable_task_terminations=False,
        )
    if mimic:
        result["mimic_task_configs"] = _qualify_mimic_task_configs(composed.env_cfg)
    if startup_only:
        composed.env_cfg.episode_length_s = 30.0
    if station_keeping:
        # The scene's normal six-second timeout is shorter than this ten-second
        # probe. Keep all termination terms, but allow the entire probe to run.
        control_dt = composed.env_cfg.sim.dt * composed.env_cfg.decimation
        composed.env_cfg.episode_length_s = max(
            composed.env_cfg.episode_length_s, (_STATION_KEEPING_STEPS + 1) * control_dt
        )
        result["station_keeping_episode_length_s"] = composed.env_cfg.episode_length_s
    expected_render_interval = 4
    if (
        composed.env_cfg.sim.dt != 0.005
        or composed.env_cfg.decimation != 2
        or composed.env_cfg.sim.render_interval != expected_render_interval
    ):
        raise AssertionError(
            f"H2 requires 200 Hz physics / 100 Hz control, got dt={composed.env_cfg.sim.dt}, "
            f"decimation={composed.env_cfg.decimation}, render_interval={composed.env_cfg.sim.render_interval}"
        )
    if enable_cameras:
        camera_period = composed.env_cfg.scene.robot_head_cam.update_period
        camera_focal_length = composed.env_cfg.scene.robot_head_cam.spawn.focal_length
        camera_follow_period = composed.env_cfg.events.follow_head_camera.interval_range_s
        if camera_period != 1.0 / robot_config.camera_hz:
            raise AssertionError(
                f"H2 head camera must update at {robot_config.camera_hz} Hz, got period={camera_period}"
            )
        if camera_focal_length != 12.0:
            raise AssertionError(f"H2 head camera must use the qualified 12 mm lens, got {camera_focal_length}")
        expected_follow_period = 1.0 / robot_config.control_hz
        if camera_follow_period != (expected_follow_period, expected_follow_period):
            raise AssertionError(
                f"H2 head camera follower must run at {robot_config.control_hz} Hz, got {camera_follow_period}"
            )
        if composed.env_cfg.scene.robot_head_cam.update_latest_camera_pose is not False:
            raise AssertionError("H2 head camera must avoid redundant production FrameView pose refreshes")
        # The production interval event updates the renderer explicitly. Pose
        # refresh is enabled only here so projection evidence reads that pose.
        composed.env_cfg.scene.robot_head_cam.update_latest_camera_pose = True
        result["camera_update_hz"] = robot_config.camera_hz
        result["camera_focal_length_mm"] = camera_focal_length
        result["camera_follow_hz"] = robot_config.control_hz
    if enable_cameras or station_keeping or mimic:
        from isaaclab.managers import DatasetExportMode

        from e2e_robot_workflows.arena.recorders import make_recorder_manager_cfg

        camera_recorder_cfg = make_recorder_manager_cfg(composed.env_cfg)
        retained_terms = set()
        if enable_cameras:
            retained_terms.add("record_pre_step_flat_camera_observations")
        if station_keeping or mimic:
            retained_terms.add("record_pre_step_actions")
        if startup_only:
            retained_terms.update(("record_initial_state", "record_post_reset_h2_feedback"))
        for term_name in vars(camera_recorder_cfg):
            if term_name.startswith("record_") and term_name not in retained_terms:
                setattr(camera_recorder_cfg, term_name, None)
        camera_recorder_cfg.dataset_export_mode = DatasetExportMode.EXPORT_NONE
        camera_recorder_cfg.export_in_record_pre_reset = False
        camera_recorder_cfg.export_in_close = False
        composed.env_cfg.recorders = camera_recorder_cfg
    result["rates_hz"] = {
        "physics": round(1.0 / composed.env_cfg.sim.dt),
        "control": round(1.0 / (composed.env_cfg.sim.dt * composed.env_cfg.decimation)),
        "render": round(1.0 / (composed.env_cfg.sim.dt * composed.env_cfg.sim.render_interval)),
        "camera": robot_config.camera_hz if enable_cameras else None,
    }
    if embodiment == "h2_agile_joint":
        from e2e_robot_workflows.arena.mimic import H2ReplayEnv

        env = H2ReplayEnv(cfg=composed.env_cfg, **composed.env_kwargs)
    else:
        env = composed.make(gym)
    try:
        if enable_cameras:
            reset_event_terms = env.event_manager.active_terms.get("reset", [])
            if not reset_event_terms or reset_event_terms[-1] != "follow_head_camera_after_reset":
                raise AssertionError(
                    f"H2 reset camera follower must run after every robot/task reset term, got {reset_event_terms}"
                )
            result["camera_reset_event_order"] = reset_event_terms
        result["stage"] = "reset"
        observations, _ = env.reset()
        assert_finite("reset_observations", observations, torch)
        if embodiment == "h2_agile_joint":
            from e2e_robot_workflows.devices.h2_ready import prime_h2_controller

            prime_h2_controller(env, embodiment)
            feedback = env.action_manager.get_term("lower_body_joint_pos")
            snapshot = env.scene.get_state(is_relative=True)
            snapshot["h2_agile_feedback"] = feedback.get_feedback_state()
            snapshot["h2_actuator_targets"] = {
                "position": env.scene["robot"].data.joint_pos_target.torch.clone(),
                "velocity": env.scene["robot"].data.joint_vel_target.torch.clone(),
            }
            env.reset_to(snapshot, is_relative=True)
            for name, expected in snapshot["h2_agile_feedback"].items():
                torch.testing.assert_close(feedback.get_feedback_state()[name], expected, rtol=0, atol=0)
            robot = env.scene["robot"]
            torch.testing.assert_close(robot.data.joint_pos_target.torch, snapshot["h2_actuator_targets"]["position"])
            torch.testing.assert_close(robot.data.joint_vel_target.torch, snapshot["h2_actuator_targets"]["velocity"])
            result["joint_replay_feedback_restored"] = True
        if startup_only:
            from e2e_robot_workflows.arena.h2_actions import h2_pink_hold_action
            from e2e_robot_workflows.devices.h2_ready import READY_STEPS, prepare_h2_recording

            result["stage"] = "guarded_startup"
            result["startup"] = []
            for cycle in range(2):
                if cycle:
                    env.sim.reset()
                    env.recorder_manager.reset()
                    env.cfg.e2e_h2_prepare_on_reset = True
                    observations, _ = env.reset(env_ids=torch.tensor([0], device=env.device))
                    assert_finite("generation_reset_observations", observations, torch)
                    result["startup"].append(env.h2_reset_preparation)
                    env.cfg.e2e_h2_prepare_on_reset = False
                    result["generation_reset_prepared"] = True
                else:
                    result["startup"].append(prepare_h2_recording(env))
                episode = env.recorder_manager.get_episode(0).data
                if "actions" in episode:
                    raise AssertionError("Startup steps leaked into the demonstration")
                if "h2_agile_feedback" not in episode.get("initial_state", {}):
                    raise AssertionError("Ready initial state is missing AGILE feedback")
            feedback = env.action_manager.get_term("lower_body_joint_pos")
            snapshot = env.scene.get_state(is_relative=True)
            snapshot["h2_agile_feedback"] = feedback.get_feedback_state()
            snapshot["h2_actuator_targets"] = {
                "position": env.scene["robot"].data.joint_pos_target.torch.clone(),
                "velocity": env.scene["robot"].data.joint_vel_target.torch.clone(),
            }
            action = h2_pink_hold_action(env)
            traces = []
            for replay in range(2):
                if replay:
                    env.reset_to(snapshot, is_relative=True)
                    for name, expected in snapshot["h2_agile_feedback"].items():
                        torch.testing.assert_close(feedback.get_feedback_state()[name], expected, rtol=0, atol=0)
                targets = []
                for _ in range(20):
                    _, _, terminated, truncated, _ = env.step(action)
                    if (terminated | truncated).any().item():
                        raise AssertionError("Ready hold unexpectedly reset")
                    targets.append(feedback.processed_actions.clone())
                traces.append(torch.stack(targets))
            error = float((traces[0] - traces[1]).abs().max().item())
            result["startup_replay_max_target_error_rad"] = error
            # The recurrent controller must reproduce its first inference from
            # restored observations/history. Subsequent physical trajectories
            # include PhysX contact-cache state absent from scene snapshots;
            # report that drift separately rather than claim exact physics replay.
            result["startup_replay_first_target_error_rad"] = float((traces[0][0] - traces[1][0]).abs().max().item())
            torch.testing.assert_close(traces[0][0], traces[1][0], rtol=0, atol=1e-6)
            result["completed_steps"] = 2 * READY_STEPS + 40
            return env
        visibility_probe: Callable[[str], None] | None = None
        if enable_cameras:
            camera_obs = observations.get("camera_obs")
            if not isinstance(camera_obs, dict) or not camera_obs:
                raise AssertionError("camera-enabled H2 observations contain no camera_obs group")
            camera_shapes = {
                name: list(value.shape) for name, value in camera_obs.items() if isinstance(value, torch.Tensor)
            }
            if not camera_shapes:
                raise AssertionError("camera-enabled H2 observations contain no camera tensor")
            expected_camera_shape = [1, 480, 640, 3]
            if camera_shapes.get("robot_head_cam_rgb") != expected_camera_shape:
                raise AssertionError(
                    "H2 head camera must have shape "
                    f"{expected_camera_shape}, got {camera_shapes.get('robot_head_cam_rgb')}"
                )
            assert_finite("camera_obs", camera_obs, torch)
            result["cameras"] = camera_shapes
            object_radii = _task_object_bounding_radii(env)
            visibility_samples: list[dict[str, Any]] = []

            def probe_visibility(sample: str) -> None:
                visibility_samples.append(_camera_task_visibility(env, torch, object_radii, sample=sample))

            visibility_probe = probe_visibility
            result["camera_task_visibility"] = {
                "required_margin_px": 24,
                "samples": visibility_samples,
            }
        robot = env.scene["robot"]
        urdf_info = contract.validate_urdf(contract.resolve_urdf())
        if tuple(robot.joint_names) != contract.expected_joint_order:
            raise AssertionError("live H2 joint order differs from the pinned 75-joint contract")
        if len(robot.body_names) != contract.expected_body_count or set(robot.body_names) != set(urdf_info.link_names):
            missing = sorted(set(urdf_info.link_names) - set(robot.body_names))
            unexpected = sorted(set(robot.body_names) - set(urdf_info.link_names))
            raise AssertionError(
                f"live H2 body set differs from the pinned {contract.expected_body_count}-body contract: "
                f"missing={missing}, unexpected={unexpected}"
            )
        missing_frames = sorted(set(contract.flange_frames + (contract.base_frame,)) - set(robot.body_names))
        if missing_frames:
            raise AssertionError(f"live H2 articulation is missing frames: {missing_frames}")
        if env.action_manager.total_action_dim != 61:
            raise AssertionError(f"H2 Arena action must be 61-D, got {env.action_manager.total_action_dim}")
        from e2e_robot_workflows.arena.embodiment_capabilities import capability_for_embodiment
        from e2e_robot_workflows.tasks.static_apple.config import load_static_apple_environment
        from e2e_robot_workflows.tasks.static_apple.terminations import object_placed_on_target

        capability = capability_for_embodiment(embodiment)
        binding = capability.load_binding()
        definition = load_static_apple_environment(capability.environment_variant)
        success_cfg = env.termination_manager.get_term_cfg("success")
        if success_cfg.func is not object_placed_on_target:
            raise AssertionError("H2 Arena did not install the E2E static-apple placement predicate")
        support_surface_z = definition.support_surface_z
        expected_values = {
            "placement_radius_m": definition.success_placement_xy_m,
            "placement_z_m": definition.success_placement_z_m,
            "min_object_height_above_target_m": definition.min_object_height_above_target_m,
            "max_object_height_m": support_surface_z + definition.max_object_height_above_support_m,
            "min_object_height_m": support_surface_z - definition.min_supported_height_below_support_m,
            "min_target_height_m": support_surface_z - definition.min_supported_height_below_support_m,
            "min_contact_force": definition.success_force_threshold,
            "release_clearance_m": definition.release_clearance_m,
            "max_object_speed": definition.success_velocity_threshold,
            "task_link_names": binding.task_link_names,
        }
        expected_entities = {
            "object_cfg": "pick_object",
            "target_cfg": "placement_target",
            "contact_sensor_cfg": "contact_sensor_pick_object",
        }
        expected_keys = set(expected_values) | set(expected_entities)
        if set(success_cfg.params) != expected_keys:
            raise AssertionError(
                "H2 Arena placement predicate parameter set changed: "
                f"expected={sorted(expected_keys)}, got={sorted(success_cfg.params)}"
            )
        for name, expected in expected_values.items():
            actual = success_cfg.params[name]
            matches = (
                math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
                if isinstance(expected, float)
                else tuple(actual) == tuple(expected)
            )
            if not matches:
                raise AssertionError(f"H2 Arena placement predicate {name} changed: expected={expected}, got={actual}")
        for name, expected in expected_entities.items():
            if success_cfg.params[name].name != expected:
                raise AssertionError(
                    f"H2 Arena placement predicate {name} changed: "
                    f"expected={expected}, got={success_cfg.params[name].name}"
                )
        result["success_contract"] = {
            "function": f"{success_cfg.func.__module__}:{success_cfg.func.__name__}",
            **{name: list(value) if name == "task_link_names" else value for name, value in expected_values.items()},
            **expected_entities,
        }
        result["kinematics"] = _qualify_kinematics(env, robot, contract, torch)
        result["hands"] = _qualify_hand_limits(robot, contract)
        material = binding.finger_friction
        result["hands"]["physics_material_binding_count"] = _physics_material_binding_count(
            env, str(material["material_path"])
        )
        if enable_cameras:
            result["camera_reset_alignment"] = _qualify_camera_reset_alignment(
                env,
                robot,
                embodiment,
                object_radii,
                torch,
            )
            result["camera_temporal_alignment"] = _qualify_camera_temporal_alignment(
                env,
                robot,
                embodiment,
                torch,
            )

        result["stage"] = "behavior"
        grasp_contact_steps = 0
        teleop_control_steps = 0
        if mimic:
            pickup_terms = env.cfg.observations.subtask_terms
            pickup_contract = {}
            for hand, task_link_name in zip(("left", "right"), binding.task_link_names, strict=True):
                term = getattr(pickup_terms, f"{hand}_hand_holds_object")
                if term.params["task_link_name"] != task_link_name:
                    raise AssertionError(f"H2 Arena {hand} pickup predicate uses the wrong task link")
                if term.params["grasp_distance_m"] != definition.release_clearance_m:
                    raise AssertionError(f"H2 Arena {hand} pickup predicate uses the wrong grasp clearance")
                pickup_contract[hand] = {
                    "task_link_name": task_link_name,
                    "grasp_distance_m": term.params["grasp_distance_m"],
                    "min_object_height_m": term.params["min_object_height_m"],
                }
            result["pickup_contract"] = pickup_contract
            eef_poses = {name: env.get_robot_eef_pose(name) for name in ("left", "right", "body")}
            expected_body_pose = torch.eye(4, device=env.device).expand(eef_poses["body"].shape[0], -1, -1)
            if not torch.allclose(eef_poses["body"], expected_body_pose, atol=1e-6, rtol=0.0):
                raise AssertionError("H2 Mimic body pseudo-EEF is not identity in the pelvis frame")
            gripper_actions = {
                "left": torch.linspace(0.01, 0.22, 22, device=env.device),
                "right": torch.linspace(0.23, 0.44, 22, device=env.device),
                "body": torch.tensor([0.1, -0.05, 0.02], device=env.device),
            }
            roundtrip_action = env.target_eef_pose_to_action(eef_poses, gripper_actions).reshape(1, -1)
            recovered = env.action_to_target_eef_pose(roundtrip_action)
            for name in ("left", "right", "body"):
                if not torch.allclose(recovered[name], eef_poses[name], atol=1e-5, rtol=0.0):
                    raise AssertionError(f"H2 Mimic {name} pose/action round trip changed its target")
            unpacked = env.actions_to_gripper_actions(roundtrip_action)
            for name in ("left", "right", "body"):
                expected = gripper_actions[name].reshape(1, -1)
                if not torch.allclose(unpacked[name], expected, atol=1e-6, rtol=0.0):
                    raise AssertionError(f"H2 Mimic changed the {name} passthrough action")
            object_poses = env.get_object_poses()
            if "pick_object" not in object_poses:
                raise AssertionError("H2 Mimic did not expose the task object pose")
            assert_finite("mimic_object_poses", object_poses, torch)
            result["mimic"] = {
                "action_dim": roundtrip_action.shape[-1],
                "eef_names": sorted(eef_poses),
                "hand_dims": [unpacked["left"].shape[-1], unpacked["right"].shape[-1]],
                "object_names": sorted(object_poses),
            }
            result["stage"] = "h2_grasp_contacts"
            grasp_contact_steps = _qualify_h2_grasp_contacts(env, robot, contract, torch, result)
            result["stage"] = "h2_teleop_control"
            teleop_control_steps = qualify_h2_teleop_control(env, robot, contract, torch, result)
            result["stage"] = "behavior"

        if embodiment == "h2_agile_joint":
            result["agile_walk"], completed_steps = _qualify_agile_walk(env, robot, contract, steps, torch)
        else:
            result["pink_motion"], completed_steps = _qualify_pink_motion(
                env,
                robot,
                contract,
                steps,
                torch,
                visibility_probe=visibility_probe,
            )
            if enable_cameras and not result["camera_task_visibility"]["samples"]:
                raise AssertionError("H2 camera visibility validation collected no settled samples")

        completed_steps += grasp_contact_steps + teleop_control_steps
        result["completed_steps"] = completed_steps
        if station_keeping:
            result["stage"] = "station_keeping"
            completed_steps += _qualify_station_keeping(env, robot, torch, result)
        expected_completed_steps = _expected_completed_steps(embodiment, steps, station_keeping, mimic)
        if completed_steps != expected_completed_steps:
            raise AssertionError(
                f"H2 validation completed {completed_steps} steps, expected {expected_completed_steps}"
            )
        result["completed_steps"] = completed_steps
        result["simulation"] = {
            "action_dim": env.action_manager.total_action_dim,
            "action_terms": list(env.action_manager.active_terms),
            "body_count": len(robot.body_names),
            "control_period_s": composed.env_cfg.sim.dt * composed.env_cfg.decimation,
            "joint_count": len(robot.joint_names),
            "joint_order": list(robot.joint_names),
        }
        return env
    except BaseException as error:
        try:
            env.close()
            result["environment_close_after_failure"] = "completed"
        except BaseException as close_error:
            result.setdefault("cleanup_errors", []).append(f"environment: {type(close_error).__name__}: {close_error}")
            error.add_note(f"environment cleanup also failed: {type(close_error).__name__}: {close_error}")
        raise


def _expected_completed_steps(embodiment: str, steps: int, station_keeping: bool = False, mimic: bool = False) -> int:
    additional_steps = _STATION_KEEPING_STEPS if station_keeping else 0
    if mimic:
        additional_steps += 4 * _GRASP_CONTACT_PHASE_STEPS + H2_TELEOP_CONTROL_STEPS
    if embodiment == "h2_agile_joint":
        return 4 * max(steps // 4, 20) + additional_steps
    phase_steps = max(steps, 20)
    return 12 * (100 + phase_steps) + 3 * (100 + 2 * phase_steps) + additional_steps


def main() -> int:
    args = _parse_args()
    return run_validation(
        result_path=args.result_path,
        task_id="e2e_static_apple",
        requested_steps=640
        if args.startup_only
        else _expected_completed_steps(args.embodiment, args.steps, args.station_keeping, args.mimic),
        execute=lambda result: _run(
            args.embodiment,
            args.steps,
            args.enable_cameras,
            args.mimic,
            result,
            args.station_keeping,
            args.startup_only,
        ),
        log_label="arena-h2-validation",
        enable_cameras=args.enable_cameras,
    )


if __name__ == "__main__":
    raise SystemExit(main())

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

"""Live reset-alignment and numeric posture probes for the canonical H2 gate."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from validation_support import assert_finite

H2_TELEOP_CONTROL_STEPS = 360
_PHASE_STEPS = H2_TELEOP_CONTROL_STEPS // 2
_MOTION_START = 120
_MOTION_END = 150
_MOTION_Z_M = 0.03


def _angular_distance(first: Any, second: Any) -> float:
    """Shortest XYZW rotation distance, independent of quaternion sign."""

    from scipy.spatial.transform import Rotation

    return float((Rotation.from_quat(first).inv() * Rotation.from_quat(second)).magnitude())


def _first_alignment_metrics(aligned: Any, reference: Any) -> dict[str, Any]:
    """Measure physical pose agreement without comparing quaternion scale/sign."""

    aligned = np.asarray(aligned, dtype=np.float64).reshape(2, 7)
    reference = np.asarray(reference, dtype=np.float64).reshape(2, 7)
    return {
        "first_pose_max_error": float(np.max(np.abs(aligned - reference))),
        "reference_quaternion_norms": np.linalg.norm(reference[:, 3:7], axis=1).tolist(),
        "first_position_errors_m": np.linalg.norm(aligned[:, :3] - reference[:, :3], axis=1).tolist(),
        "first_rotation_errors_rad": [
            _angular_distance(actual[3:7], expected[3:7]) for actual, expected in zip(aligned, reference, strict=True)
        ],
    }


def _posture_preference_probe(env: Any, robot: Any, report: dict[str, Any]) -> None:
    """Compare two real Pink solves without advancing or changing the plant.

    Both solves receive the same bounded shoulder/elbow perturbation and the
    flange poses at that configuration. Only the initialized null-space task
    differs. Restore the controller's task list, frame targets, and kinematics
    even when a solver or an assertion fails.
    """

    from isaaclab.controllers.pink_ik.null_space_posture_task import NullSpacePostureTask
    from isaaclab.controllers.pink_ik.pink_tasks import LocalFrameTask

    controller = env.action_manager.get_term("upper_body_ik")._ik_controllers[0]
    tasks = controller._variable_input_tasks
    posture_tasks = [task for task in tasks if isinstance(task, NullSpacePostureTask)]
    if len(posture_tasks) != 1:
        raise AssertionError("H2 teleop requires exactly one initialized null-space posture task")
    posture = posture_tasks[0]
    configuration = controller.pink_configuration
    if posture.target_q is None:
        raise AssertionError("H2 preferred posture was not initialized")
    np.testing.assert_array_equal(posture.target_q, controller.init_joint_positions)
    controlled_names = list(controller.cfg.joint_names)
    expected_names = [name for name in controlled_names if "shoulder" in name or "elbow" in name]
    if set(posture.controlled_joints) != set(expected_names):
        raise AssertionError("H2 posture preference must select shoulders and elbows, not wrists or fingers")
    selected = [controlled_names.index(name) for name in expected_names]
    elbow_ids = [controlled_names.index(name) for name in expected_names if "elbow" in name]
    if posture.target_q.shape != (configuration.model.nq,):
        raise AssertionError("H2 initialized posture must use the reduced Pink configuration order")
    preferred = np.asarray(
        [
            posture.target_q[configuration.model.joints[configuration.model.getJointId(name)].idx_q]
            for name in controlled_names
        ]
    )
    joint_ids = np.asarray(controller.controlled_joint_indices)
    current = robot.data.joint_pos.torch[0].detach().cpu().numpy().copy()
    current[joint_ids] = preferred
    for index, name in enumerate(controlled_names):
        side = 1.0 if name.startswith("left_") else -1.0
        if "shoulder_roll" in name:
            current[joint_ids[index]] += side * 0.25
        elif "shoulder_yaw" in name:
            current[joint_ids[index]] -= side * 0.12
        elif "elbow" in name:
            current[joint_ids[index]] -= 0.10

    saved_configuration = configuration.full_q.copy()
    frame_tasks = [task for task in tasks if isinstance(task, LocalFrameTask)]
    if len(frame_tasks) != 2:
        raise AssertionError("H2 posture comparison requires both real pelvis-relative flange tasks")
    saved_targets = [task.transform_target_to_base.copy() for task in frame_tasks]
    saved_posture = posture.target_q.copy()
    saved_mask = posture._joint_mask
    saved_frames = posture._frame_names
    report.update(
        {
            "selected_joints": expected_names,
            "preferred_joint_positions_rad": preferred[selected].tolist(),
            "input_joint_positions_rad": current[joint_ids][selected].tolist(),
            "same_input_state_and_flange_targets": True,
            "simulator_steps": 0,
        }
    )
    try:
        configuration.update(current[controller.isaac_lab_to_pink_ordering])
        for task in frame_tasks:
            task.set_target_from_configuration(configuration)
        controller._variable_input_tasks = [task for task in tasks if task is not posture]
        unbiased = controller.compute(current.copy(), env.step_dt).detach().cpu().numpy()
        controller._variable_input_tasks = tasks
        biased = controller.compute(current.copy(), env.step_dt).detach().cpu().numpy()
        if not np.isfinite(unbiased).all() or not np.isfinite(biased).all():
            raise AssertionError("H2 posture comparison returned non-finite IK targets")
        for label, target in (("without_preference", unbiased), ("with_preference", biased)):
            report[label] = {
                "joint_positions_rad": target[selected].tolist(),
                "preferred_posture_error_rad": float(np.linalg.norm(target[selected] - preferred[selected])),
                "elbow_angle_error_rad": float(np.linalg.norm(target[elbow_ids] - preferred[elbow_ids])),
            }
            configuration_state = current.copy()
            configuration_state[joint_ids] = target
            configuration.update(configuration_state[controller.isaac_lab_to_pink_ordering])
            report[label]["flange_task_error_norm"] = max(
                float(np.linalg.norm(task.compute_error(configuration))) for task in frame_tasks
            )
        improvement = (
            report["without_preference"]["preferred_posture_error_rad"]
            - report["with_preference"]["preferred_posture_error_rad"]
        )
        report["preferred_posture_improvement_rad"] = improvement
        if improvement <= 1e-7:
            raise AssertionError(f"H2 initialized posture task did not improve the real IK solution: {improvement:.9g}")
        if report["with_preference"]["flange_task_error_norm"] > 0.002:
            raise AssertionError("H2 posture preference displaced the higher-priority flange pose")
        report["status"] = "ok"
    finally:
        controller._variable_input_tasks = tasks
        for task, target in zip(frame_tasks, saved_targets, strict=True):
            task.set_target(target)
        posture.set_target(saved_posture)
        posture._joint_mask = saved_mask
        posture._frame_names = saved_frames
        configuration.update(saved_configuration)
        report["controller_state_restored"] = True


def qualify_h2_teleop_control(env: Any, robot: Any, contract: Any, torch: Any, result: dict[str, Any]) -> int:
    """Exercise the recorder's alignment adapter, real IK, and recorded action."""

    from scipy.spatial.transform import Rotation

    from e2e_robot_workflows.arena.h2_actions import h2_pink_hold_action
    from e2e_robot_workflows.devices.h2_reset_alignment import H2ResetAlignment
    from e2e_robot_workflows.devices.h2_station_keeping import H2StationKeeping, apply_h2_station_keeping
    from e2e_robot_workflows.devices.recording_runtime import align_h2_recording_action

    report: dict[str, Any] = {
        "requested_steps": H2_TELEOP_CONTROL_STEPS,
        "completed_steps": 0,
        "unexpected_resets": 0,
        "phases": [],
        "posture_preference": {},
        "recorded_action_match_steps": 0,
        "minimum_pelvis_height_m": None,
        "maximum_pelvis_tilt_deg": 0.0,
    }
    result["h2_teleop_control"] = report
    completed_before = result.get("completed_steps", 0)
    env.reset()
    env.recorder_manager.reset()
    _posture_preference_probe(env, robot, report["posture_preference"])
    alignment = H2ResetAlignment()
    station = H2StationKeeping()
    lower = env.action_manager.get_term("lower_body_joint_pos")

    for phase_index in range(2):
        if phase_index:
            env.reset()
            env.recorder_manager.reset()
        alignment.reset()
        station.reset()
        reference = h2_pink_hold_action(env)[0].clone()
        raw_start = reference.clone()
        # Exercise downward controller offsets and distinct orientations across resets.
        raw_start[:3] += raw_start.new_tensor((0.03 * phase_index, -0.04 * phase_index, -0.245))
        raw_start[7:10] += raw_start.new_tensor((-0.04, 0.05 * phase_index, -0.18 - 0.04 * phase_index))
        for offset, direction in ((0, 1.0), (7, -1.0)):
            raw_rotation = Rotation.from_rotvec((direction * (0.3 + 0.2 * phase_index), 0.1, 0.0))
            original = Rotation.from_quat(raw_start[offset + 3 : offset + 7].cpu().numpy())
            raw_start[offset + 3 : offset + 7] = raw_start.new_tensor((raw_rotation * original).as_quat())
        selected_hand = "left" if phase_index == 0 else "right"
        for index, name in enumerate(contract.hand_joint_order):
            targets = (
                contract.hand_closed_targets if name.startswith(f"{selected_hand}_") else contract.hand_open_targets
            )
            raw_start[14 + index] = targets[name]
        raw_start[-3:] = 0.0
        phase: dict[str, Any] = {
            "index": phase_index,
            "reference_flange_poses_xyzw": reference[:14].reshape(2, 7).tolist(),
            "initial_raw_position_error_m": [
                float(torch.linalg.vector_norm(raw_start[offset : offset + 3] - reference[offset : offset + 3]).item())
                for offset in (0, 7)
            ],
            "initial_raw_rotation_error_rad": [
                _angular_distance(
                    raw_start[offset + 3 : offset + 7].cpu().numpy(),
                    reference[offset + 3 : offset + 7].cpu().numpy(),
                )
                for offset in (0, 7)
            ],
            "max_position_increment_m": 0.0,
            "max_rotation_increment_rad": 0.0,
        }
        report["phases"].append(phase)
        previous = None
        before_motion = None
        for step in range(_PHASE_STEPS):
            raw = raw_start.clone()
            if 4 <= step < 8:
                raw[-3:] = raw.new_tensor((0.02, -0.015, 0.025))
            if _MOTION_START <= step < _MOTION_END:
                raw[2] += _MOTION_Z_M
                raw[9] += _MOTION_Z_M
                for offset in (0, 7):
                    moved = Rotation.from_rotvec((0.0, 0.04, 0.0)) * Rotation.from_quat(
                        raw[offset + 3 : offset + 7].cpu().numpy()
                    )
                    raw[offset + 3 : offset + 7] = raw.new_tensor(moved.as_quat())
            aligned = align_h2_recording_action(raw, env, alignment)
            if not torch.equal(aligned[14:], raw[14:]):
                raise AssertionError("H2 reset alignment changed a hand or navigation command")
            if step == 0:
                phase.update(_first_alignment_metrics(aligned[:14].cpu().numpy(), reference[:14].cpu().numpy()))
                if max(phase["first_position_errors_m"]) > 1e-6 or max(phase["first_rotation_errors_rad"]) > 1e-6:
                    raise AssertionError("H2 first aligned sample did not match both measured flange poses")
            if previous is not None:
                for offset in (0, 7):
                    distance = float(
                        torch.linalg.vector_norm(aligned[offset : offset + 3] - previous[offset : offset + 3])
                    )
                    angle = _angular_distance(
                        aligned[offset + 3 : offset + 7].cpu().numpy(),
                        previous[offset + 3 : offset + 7].cpu().numpy(),
                    )
                    phase["max_position_increment_m"] = max(phase["max_position_increment_m"], distance)
                    phase["max_rotation_increment_rad"] = max(phase["max_rotation_increment_rad"], angle)
                    if distance > alignment.config.max_linear_speed * env.step_dt + 1e-6:
                        raise AssertionError("H2 aligned flange target exceeded its linear slew bound")
                    if angle > alignment.config.max_angular_speed * env.step_dt + 1e-6:
                        raise AssertionError("H2 aligned flange target exceeded its angular slew bound")
            previous = aligned.clone()
            action = apply_h2_station_keeping(aligned, robot.data, station).unsqueeze(0)
            observations, _, terminated, truncated, _ = env.step(action)
            report["completed_steps"] += 1
            result["completed_steps"] = completed_before + report["completed_steps"]
            report["unexpected_resets"] += int(bool(torch.any(terminated | truncated).item()))
            if report["unexpected_resets"] or int(env.episode_length_buf[0].item()) != step + 1:
                raise AssertionError("H2 aligned teleop motion unexpectedly reset its episode")
            for name, value in (
                ("action", action),
                ("observations", observations),
                ("joint_position", robot.data.joint_pos.torch),
                ("joint_velocity", robot.data.joint_vel.torch),
                ("root_pose", robot.data.root_link_pose_w.torch),
            ):
                assert_finite(f"h2_teleop_control.{name}", value, torch)
            pose = robot.data.root_link_pose_w.torch[0]
            x, y = float(pose[3]), float(pose[4])
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))
            minimum_height = report["minimum_pelvis_height_m"]
            report["minimum_pelvis_height_m"] = (
                float(pose[2]) if minimum_height is None else min(minimum_height, float(pose[2]))
            )
            report["maximum_pelvis_tilt_deg"] = max(report["maximum_pelvis_tilt_deg"], tilt)
            if float(pose[2]) < 0.75 or tilt > 45.0:
                raise AssertionError("H2 aligned teleop motion left the existing upright envelope")
            recorded = env.recorder_manager.get_episode(0).data.get("actions", [])
            if len(recorded) != step + 1:
                raise AssertionError("H2 teleop recorder did not retain every aligned action")
            if (
                not torch.equal(recorded[-1].reshape(1, -1), action)
                or not torch.equal(env.action_manager.action, action)
                or not torch.equal(lower.locomotion_command, action[:, -3:])
            ):
                raise AssertionError("H2 recorded action differs from the final aligned/station-keeping command")
            report["recorded_action_match_steps"] += 1
            if step == _MOTION_START - 1:
                before_motion = h2_pink_hold_action(env)[0].clone()
            elif step == _MOTION_END - 1:
                assert before_motion is not None
                measured = h2_pink_hold_action(env)[0]
                phase["measured_flange_z_response_m"] = [
                    float(measured[offset + 2] - before_motion[offset + 2]) for offset in (0, 7)
                ]
                if any(value < 0.005 for value in phase["measured_flange_z_response_m"]):
                    raise AssertionError("H2 aligned controller motion did not move both live flanges upward")
        phase["status"] = "ok"
    report["status"] = "ok"
    return H2_TELEOP_CONTROL_STEPS

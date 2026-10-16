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

"""Action and phase mechanics for the AGILE locomotion validations.

Two runtimes drive the same schedule here. The built-in task takes a four-number
``[vx, vy, wz, hip_height]`` action term, and Arena's whole-body controller takes the
seven-number locomotion tail its layouts end with; they also run different policy
artifacts. Everything that is a statement about *locomotion behavior* rather than about
either layout -- the phases, the pelvis band, the object staying supported, the settle
and turn measures -- is therefore shared, and each runtime supplies only what is
genuinely its own through :class:`LocomotionRuntime`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pink_validation import make_live_pink_action
from validation_support import assert_finite, assert_not_done

MIN_PELVIS_HEIGHT_M = 0.60
MAX_PELVIS_HEIGHT_M = 0.90
MAX_PELVIS_TILT_RAD = 0.45

# The command every phase is expressed in, for both runtimes.
Command = tuple[float, float, float, float]

# The drive both runtimes are held to: settle, reverse, stop, turn, then reset and settle
# again. Shared so a behavior difference between the runtimes cannot be a difference in
# what they were asked to do.
HIP_HEIGHT_M = 0.72
EXPECTED_CONTROL_PERIOD_S = 0.02
SETTLE_PHASE = ("settle", 32, (0.0, 0.0, 0.0, HIP_HEIGHT_M))
BACKWARD_PHASE = ("backward", 64, (-0.2, 0.0, 0.0, HIP_HEIGHT_M))
STOP_PHASE = ("stop", 32, (0.0, 0.0, 0.0, HIP_HEIGHT_M))
TURN_PHASE = ("turn", 64, (0.0, 0.0, 0.4, HIP_HEIGHT_M))
PHASES = (SETTLE_PHASE, BACKWARD_PHASE, STOP_PHASE, TURN_PHASE)
RESET_SETTLE_STEPS = 48
TOTAL_CONTROL_STEPS = sum(steps for _, steps, _ in PHASES) + RESET_SETTLE_STEPS

OBJECT_SUPPORT_TOLERANCE_M = 0.01
MIN_BACKWARD_PROGRESS_M = 0.05
MAX_BACKWARD_LATERAL_DRIFT_M = 0.10
MAX_STOP_TAIL_PLANAR_SPEED_M_S = 0.05
MAX_STOP_TAIL_ABS_YAW_RATE_RAD_S = 0.20
MAX_TURN_PLANAR_DISPLACEMENT_M = 0.12
STOP_TAIL_STEPS = 16
MIN_TURN_YAW_DELTA_RAD = 0.08
MAX_RESET_ROOT_POSITION_ERROR_M = 0.02
MAX_RESET_ROOT_ROTATION_ERROR_RAD = 0.05


@dataclass
class LocomotionRuntime:
    """What a runtime contributes to the shared phase mechanics.

    Attributes:
        make_action: Build the action for one step from ``(vx, vy, wz, base_height)``.
        pick_object_name: Scene entity name of the object that must stay supported.
        assert_policy_outputs: Check the locomotion policy actually produced usable
            output this step, returning its magnitude. Runtime-specific because the
            two stacks expose their policy differently.
        assert_upper_body: Check whatever holds the arms still, when the runtime has
            something to check.
    """

    make_action: Callable[[Command], Any]
    pick_object_name: str
    assert_policy_outputs: Callable[[], float]
    assert_upper_body: Callable[[], None] = field(default=lambda: None)


@dataclass
class PhaseOutcome:
    """Metrics collected while executing one AGILE command phase."""

    name: str
    summary: dict[str, Any]
    final_state: dict[str, float]
    min_object_z: float
    planar_speeds: list[float]
    absolute_yaw_rates: list[float]


def derive_action_layout(env: Any) -> tuple[dict[str, slice], Any, Any]:
    """Validate and return the Pink and AGILE action layout."""

    expected_terms = ["upper_body_ik", "lower_body_joint_pos"]
    if env.action_manager.active_terms != expected_terms:
        raise AssertionError(
            f"unexpected AGILE action terms: {env.action_manager.active_terms}; expected {expected_terms}"
        )

    slices: dict[str, slice] = {}
    cursor = 0
    for term_name in env.action_manager.active_terms:
        term = env.action_manager.get_term(term_name)
        slices[term_name] = slice(cursor, cursor + term.action_dim)
        cursor += term.action_dim

    upper_term = env.action_manager.get_term("upper_body_ik")
    lower_term = env.action_manager.get_term("lower_body_joint_pos")
    if upper_term.action_dim != 28 or lower_term.action_dim != 4 or cursor != 32:
        raise AssertionError(
            "AGILE must expose Pink 28-D plus lower-body 4-D actions: "
            f"Pink={upper_term.action_dim}, lower={lower_term.action_dim}, total={cursor}"
        )
    if env.action_manager.total_action_dim != cursor:
        raise AssertionError(
            f"action manager dimensions disagree: terms={cursor}, total={env.action_manager.total_action_dim}"
        )
    if upper_term.hand_joint_dim != 14:
        raise AssertionError(f"unexpected Pink hand action dimension: {upper_term.hand_joint_dim}")
    return slices, upper_term, lower_term


def make_builtin_action(
    env: Any,
    action_slices: dict[str, slice],
    upper_term: Any,
    context: dict[str, Any],
    command: Command,
    torch: Any,
) -> Any:
    """Hold the upper body at its live state while commanding the built-in AGILE term."""

    action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    upper_slice = action_slices["upper_body_ik"]
    lower_slice = action_slices["lower_body_joint_pos"]
    action[:, upper_slice] = make_live_pink_action(env, upper_term, context, torch)
    action[:, lower_slice] = torch.tensor(command, device=env.device)
    return action


def assert_builtin_policy_outputs(lower_term: Any, lower_policy_obs: Any, torch: Any) -> float:
    assert_finite("lower_body_policy.observations", lower_policy_obs, torch)
    assert_finite("lower_body_joint_pos.raw_actions", lower_term.raw_actions, torch)
    assert_finite("lower_body_joint_pos.processed_actions", lower_term.processed_actions, torch)
    if tuple(lower_policy_obs.shape) != (1, 79):
        raise AssertionError(f"unexpected AGILE lower-policy observation shape: {tuple(lower_policy_obs.shape)}")
    if tuple(lower_term.raw_actions.shape) != (1, 12):
        raise AssertionError(f"unexpected AGILE raw policy output shape: {tuple(lower_term.raw_actions.shape)}")
    if tuple(lower_term.processed_actions.shape) != (1, 12):
        raise AssertionError(
            f"unexpected AGILE processed policy output shape: {tuple(lower_term.processed_actions.shape)}"
        )
    if lower_term.raw_actions.requires_grad or lower_term.processed_actions.requires_grad:
        raise AssertionError("AGILE policy outputs must be detached inference tensors before Warp actuation")
    return float(lower_term.raw_actions.abs().max().item())


def _assert_runtime_state(
    *,
    env: Any,
    observations: Any,
    reward: Any,
    terminated: Any,
    truncated: Any,
    support_z: float,
    runtime: LocomotionRuntime,
    torch: Any,
    quat_apply: Any,
) -> dict[str, float]:
    robot = env.scene["robot"]
    pick_object = env.scene[runtime.pick_object_name]
    robot_root_state = robot.data.root_state_w.torch
    object_root_state = pick_object.data.root_state_w.torch

    assert_finite("observations", observations, torch)
    assert_finite("reward", reward, torch)
    assert_finite("robot.root_state_w", robot_root_state, torch)
    assert_finite("robot.root_lin_vel_b", robot.data.root_lin_vel_b.torch, torch)
    assert_finite("robot.root_ang_vel_b", robot.data.root_ang_vel_b.torch, torch)
    assert_finite("robot.joint_pos", robot.data.joint_pos.torch, torch)
    assert_finite("robot.joint_vel", robot.data.joint_vel.torch, torch)
    assert_finite("object.root_state_w", object_root_state, torch)
    assert_not_done(terminated, truncated, context="AGILE locomotion validation")

    pelvis_height = float((robot_root_state[:, 2] - env.scene.env_origins[:, 2]).item())
    root_quaternion = torch.nn.functional.normalize(robot_root_state[:, 3:7], dim=-1)
    local_up = torch.zeros((env.num_envs, 3), device=env.device)
    local_up[:, 2] = 1.0
    world_up = quat_apply(root_quaternion, local_up)
    pelvis_tilt = float(torch.acos(world_up[:, 2].clamp(-1.0, 1.0)).item())
    object_local_z = float((object_root_state[:, 2] - env.scene.env_origins[:, 2]).item())

    if not MIN_PELVIS_HEIGHT_M <= pelvis_height <= MAX_PELVIS_HEIGHT_M:
        raise AssertionError(
            f"pelvis height {pelvis_height:.6g} m is outside [{MIN_PELVIS_HEIGHT_M:.6g}, {MAX_PELVIS_HEIGHT_M:.6g}] m"
        )
    if pelvis_tilt >= MAX_PELVIS_TILT_RAD:
        raise AssertionError(f"pelvis tilt {pelvis_tilt:.6g} rad reached the {MAX_PELVIS_TILT_RAD:.6g} rad limit")
    if object_local_z < support_z:
        raise AssertionError(
            f"pick object fell below its support surface: local_z={object_local_z:.6g} m, limit={support_z:.6g} m"
        )

    policy_output_magnitude = runtime.assert_policy_outputs()
    return {
        "body_planar_speed_m_s": float(torch.linalg.vector_norm(robot.data.root_lin_vel_b.torch[0, :2]).item()),
        "body_vx_m_s": float(robot.data.root_lin_vel_b.torch[0, 0].item()),
        "body_yaw_rate_rad_s": float(robot.data.root_ang_vel_b.torch[0, 2].item()),
        "object_local_z_m": object_local_z,
        "pelvis_height_m": pelvis_height,
        "pelvis_tilt_rad": pelvis_tilt,
        "policy_output_magnitude": policy_output_magnitude,
    }


def run_phase(
    name: str,
    steps: int,
    command: Command,
    *,
    stage: str,
    env: Any,
    runtime: LocomotionRuntime,
    support_z: float,
    result: dict[str, Any],
    torch: Any,
    quat_apply: Any,
    check_upper_action: bool = True,
) -> PhaseOutcome:
    """Execute one command phase and collect its behavior metrics."""

    if steps < 1:
        raise AssertionError(f"AGILE phase {name!r} must contain at least one step")

    result["stage"] = stage
    policy_output = 0.0
    min_object_z = float("inf")
    min_height = float("inf")
    max_height = float("-inf")
    max_tilt = 0.0
    planar_speeds: list[float] = []
    absolute_yaw_rates: list[float] = []
    final_state = None
    for _ in range(steps):
        observations, reward, terminated, truncated, _ = env.step(runtime.make_action(command))
        final_state = _assert_runtime_state(
            env=env,
            observations=observations,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            support_z=support_z,
            runtime=runtime,
            torch=torch,
            quat_apply=quat_apply,
        )
        if check_upper_action:
            runtime.assert_upper_body()
        policy_output = max(policy_output, final_state["policy_output_magnitude"])
        min_object_z = min(min_object_z, final_state["object_local_z_m"])
        min_height = min(min_height, final_state["pelvis_height_m"])
        max_height = max(max_height, final_state["pelvis_height_m"])
        max_tilt = max(max_tilt, final_state["pelvis_tilt_rad"])
        planar_speeds.append(final_state["body_planar_speed_m_s"])
        absolute_yaw_rates.append(abs(final_state["body_yaw_rate_rad_s"]))
        result["completed_steps"] += 1

    if final_state is None:
        raise AssertionError(f"AGILE phase {name!r} did not execute")
    return PhaseOutcome(
        name=name,
        summary={
            "command": list(command),
            "max_pelvis_height_m": max_height,
            "max_pelvis_tilt_rad": max_tilt,
            "max_policy_output_magnitude": policy_output,
            "min_pelvis_height_m": min_height,
            "steps": steps,
        },
        final_state=final_state,
        min_object_z=min_object_z,
        planar_speeds=planar_speeds,
        absolute_yaw_rates=absolute_yaw_rates,
    )


def tail_motion(outcome: PhaseOutcome, tail_steps: int) -> tuple[float, float]:
    """Return the mean planar speed and absolute yaw rate over a phase tail."""

    if tail_steps < 1:
        raise AssertionError("an AGILE phase tail must contain at least one step")
    if len(outcome.planar_speeds) < tail_steps:
        raise AssertionError(f"AGILE phase {outcome.name!r} is too short for a {tail_steps}-step tail")
    tail = slice(-tail_steps, None)
    return (
        sum(outcome.planar_speeds[tail]) / tail_steps,
        sum(outcome.absolute_yaw_rates[tail]) / tail_steps,
    )


def heading_xy(root_quaternion: Any, torch: Any, quat_apply: Any) -> Any:
    """Return the robot's unit heading projected onto the ground plane."""

    root_quaternion = torch.nn.functional.normalize(root_quaternion, dim=-1)
    forward = torch.zeros((root_quaternion.shape[0], 3), device=root_quaternion.device)
    forward[:, 0] = 1.0
    heading = quat_apply(root_quaternion, forward)[:, :2]
    norm = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
    if bool((norm < 1.0e-6).any().item()):
        raise AssertionError("the projected robot heading is degenerate")
    return heading / norm


def signed_yaw_delta(initial_heading: Any, final_heading: Any, torch: Any) -> float:
    """Return the signed yaw change between two ground-plane headings, in radians."""

    dot = torch.sum(initial_heading * final_heading, dim=-1)
    cross_z = initial_heading[:, 0] * final_heading[:, 1] - initial_heading[:, 1] * final_heading[:, 0]
    return float(torch.atan2(cross_z, dot).item())


def assert_walked_backward(
    start_position: Any, end_position: Any, start_heading: Any, torch: Any
) -> tuple[float, float]:
    """Check the reverse phase moved the robot backward without drifting sideways.

    Returns:
        The signed progress along the initial heading and the absolute lateral drift, in
        metres.
    """

    displacement_xy = end_position[:, :2] - start_position[:, :2]
    left_heading = torch.stack((-start_heading[:, 1], start_heading[:, 0]), dim=-1)
    progress = float(torch.sum(displacement_xy * start_heading, dim=-1).item())
    lateral_drift = float(torch.sum(displacement_xy * left_heading, dim=-1).abs().item())
    if progress > -MIN_BACKWARD_PROGRESS_M:
        raise AssertionError(
            f"backward progress was {progress:.6g} m; expected at most "
            f"{-MIN_BACKWARD_PROGRESS_M:.6g} m along the initial heading"
        )
    if lateral_drift > MAX_BACKWARD_LATERAL_DRIFT_M:
        raise AssertionError(
            f"lateral drift was {lateral_drift:.6g} m during reverse (limit {MAX_BACKWARD_LATERAL_DRIFT_M:.6g} m)"
        )
    return progress, lateral_drift


def assert_settled(outcome: PhaseOutcome, described_as: str) -> tuple[float, float]:
    """Check a phase ended with the robot standing still.

    Returns:
        The tail mean planar speed (m/s) and absolute yaw rate (rad/s).
    """

    mean_planar_speed, mean_abs_yaw_rate = tail_motion(outcome, STOP_TAIL_STEPS)
    if mean_planar_speed >= MAX_STOP_TAIL_PLANAR_SPEED_M_S:
        raise AssertionError(
            f"{described_as}: tail mean planar speed {mean_planar_speed:.6g} m/s "
            f"(limit {MAX_STOP_TAIL_PLANAR_SPEED_M_S:.6g} m/s)"
        )
    if mean_abs_yaw_rate >= MAX_STOP_TAIL_ABS_YAW_RATE_RAD_S:
        raise AssertionError(
            f"{described_as}: tail mean |yaw rate| {mean_abs_yaw_rate:.6g} rad/s "
            f"(limit {MAX_STOP_TAIL_ABS_YAW_RATE_RAD_S:.6g} rad/s)"
        )
    return mean_planar_speed, mean_abs_yaw_rate


def assert_turned_in_place(
    initial_heading: Any,
    final_heading: Any,
    initial_position_xy: Any,
    final_position_xy: Any,
    torch: Any,
) -> tuple[float, float]:
    """Check the turn phase rotated the robot without walking it away.

    Returns:
        The signed yaw delta (rad) and the planar displacement (m).
    """

    yaw_delta = signed_yaw_delta(initial_heading, final_heading, torch)
    if yaw_delta < MIN_TURN_YAW_DELTA_RAD:
        raise AssertionError(
            f"the turn changed yaw by only {yaw_delta:.6g} rad (minimum signed delta {MIN_TURN_YAW_DELTA_RAD:.6g} rad)"
        )
    displacement = float(torch.linalg.vector_norm(final_position_xy - initial_position_xy, dim=-1).item())
    if displacement > MAX_TURN_PLANAR_DISPLACEMENT_M:
        raise AssertionError(
            f"the robot translated {displacement:.6g} m during the in-place turn "
            f"(limit {MAX_TURN_PLANAR_DISPLACEMENT_M:.6g} m)"
        )
    return yaw_delta, displacement


def assert_reset_to_initial_pose(
    reset_root_state: Any, initial_root_state: Any, torch: Any, quaternion_distance_rad: Any
) -> tuple[float, float]:
    """Check a reset returned the robot root to where it started.

    Returns:
        The position error (m) and rotation error (rad).
    """

    position_error = float(torch.linalg.vector_norm(reset_root_state[:, :3] - initial_root_state[:, :3], dim=-1).item())
    rotation_error = quaternion_distance_rad(reset_root_state[:, 3:7], initial_root_state[:, 3:7], torch)
    if position_error > MAX_RESET_ROOT_POSITION_ERROR_M:
        raise AssertionError(
            f"reset root position differs from its initial state by {position_error:.6g} m "
            f"(limit {MAX_RESET_ROOT_POSITION_ERROR_M:.6g} m)"
        )
    if rotation_error > MAX_RESET_ROOT_ROTATION_ERROR_RAD:
        raise AssertionError(
            f"reset root rotation differs from its initial state by {rotation_error:.6g} rad "
            f"(limit {MAX_RESET_ROOT_ROTATION_ERROR_RAD:.6g} rad)"
        )
    return position_error, rotation_error

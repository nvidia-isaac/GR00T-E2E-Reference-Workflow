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

"""Name-based translation from GR00T H2 groups to the Arena 61-D joint command."""

from __future__ import annotations

from typing import Any

import numpy as np

H2_UPPER_BODY_ACTION_DIM = 58
H2_NAVIGATION_DIM = 3
H2_ARENA_JOINT_ACTION_DIM = H2_UPPER_BODY_ACTION_DIM + H2_NAVIGATION_DIM
H2_ACTION_HORIZON = 40


def validate_h2_action_schedule(action_horizon: int, action_chunk_length: int) -> None:
    """Require the exact action schedule used to train the H2 N1.7 policy."""

    if (
        type(action_horizon) is not int
        or type(action_chunk_length) is not int
        or action_horizon != H2_ACTION_HORIZON
        or action_chunk_length != H2_ACTION_HORIZON
    ):
        raise ValueError("H2 evaluation requires the N1.7 40-step horizon and a 40-step chunk")


def validate_h2_live_joint_limits(
    live_limits: dict[str, tuple[float, float]],
    pinned_limits: dict[str, tuple[float, float]],
    *,
    atol: float = 1e-6,
) -> None:
    """Reject evaluation when the live 58-joint plant drifts from its pinned sidecar."""

    if atol < 0 or not np.isfinite(atol):
        raise ValueError("H2 live-limit tolerance must be finite and non-negative")
    for joint, expected in pinned_limits.items():
        actual = live_limits.get(joint)
        if actual is None or len(actual) != 2 or not np.isfinite(actual).all():
            raise ValueError(f"H2 live articulation has no valid position limits for joint {joint!r}")
        if not np.allclose(actual, expected, atol=atol, rtol=0.0):
            raise ValueError(
                f"H2 live position limits for joint {joint!r} drift from the pinned sidecar: "
                f"expected {expected}, got {actual}"
            )


def validate_h2_joint_configs(
    policy_joints: dict[str, list[str]],
    action_joints: dict[str, int],
    state_joints: dict[str, int],
) -> None:
    """Require complete, non-overlapping 75-D state and 58-D learned-action maps."""

    flattened = [joint for group in policy_joints.values() for joint in group]
    if len(flattened) != 75 or len(set(flattened)) != 75:
        raise ValueError("H2 policy groups must cover 75 distinct joints")
    if set(flattened) != set(state_joints) or sorted(state_joints.values()) != list(range(75)):
        raise ValueError("H2 state map must be a bijection over the 75 policy joints")
    if len(action_joints) != H2_UPPER_BODY_ACTION_DIM or sorted(action_joints.values()) != list(
        range(H2_UPPER_BODY_ACTION_DIM)
    ):
        raise ValueError("H2 learned action map must be a 58-joint bijection")
    allowed_groups = {"left_arm", "right_arm", "left_hand", "right_hand"}
    expected_actions = {joint for group, joints in policy_joints.items() if group in allowed_groups for joint in joints}
    if set(action_joints) != expected_actions:
        raise ValueError("H2 learned action map must contain exactly both arms and both hands")


def build_h2_gr00t_action_np(
    policy_action: dict[str, Any],
    policy_joints: dict[str, list[str]],
    action_joints: dict[str, int],
    *,
    joint_position_limits: dict[str, tuple[float, float]],
    navigation_limits: tuple[tuple[float, float], ...],
) -> np.ndarray:
    """Build ``58 named joint targets + [vx, vy,wz]`` without pose-action reuse."""

    required_groups = ("left_arm", "right_arm", "left_hand", "right_hand")
    missing = [group for group in (*required_groups, "navigate_command") if group not in policy_action]
    if missing:
        raise ValueError(f"H2 GR00T action is missing groups: {missing}")

    arrays = {group: np.asarray(policy_action[group], dtype=np.float64) for group in required_groups}
    reference_shape = arrays[required_groups[0]].shape[:2]
    if arrays[required_groups[0]].ndim != 3:
        raise ValueError("H2 GR00T joint groups must have shape (batch, horizon, joints)")
    output = np.zeros((*reference_shape, H2_ARENA_JOINT_ACTION_DIM), dtype=np.float64)

    locations: dict[str, tuple[str, int]] = {}
    for group in required_groups:
        joints = policy_joints.get(group, [])
        values = arrays[group]
        if values.ndim != 3 or values.shape[:2] != reference_shape or values.shape[2] != len(joints):
            raise ValueError(
                f"H2 GR00T group {group!r} has shape {values.shape}, expected {(*reference_shape, len(joints))}"
            )
        for index, joint in enumerate(joints):
            locations[joint] = (group, index)

    for joint, output_index in action_joints.items():
        try:
            group, group_index = locations[joint]
        except KeyError as error:
            raise ValueError(f"H2 GR00T policy has no value for action joint {joint!r}") from error
        output[..., output_index] = arrays[group][..., group_index]

    navigation = np.asarray(policy_action["navigate_command"], dtype=np.float64)
    if navigation.shape != (*reference_shape, H2_NAVIGATION_DIM):
        raise ValueError(
            f"H2 GR00T navigate_command has shape {navigation.shape}, expected {(*reference_shape, H2_NAVIGATION_DIM)}"
        )
    output[..., H2_UPPER_BODY_ACTION_DIM:] = navigation
    if not np.isfinite(output).all():
        raise ValueError("H2 GR00T action contains non-finite values")
    for joint, output_index in action_joints.items():
        limits = joint_position_limits.get(joint)
        if limits is None or len(limits) != 2 or not np.isfinite(limits).all() or limits[0] > limits[1]:
            raise ValueError(f"H2 GR00T action has no valid live position limits for joint {joint!r}")
        values = output[..., output_index]
        if np.any(values < limits[0] - 1e-5) or np.any(values > limits[1] + 1e-5):
            raise ValueError(f"H2 GR00T action for joint {joint!r} is outside live position limits {limits}")
    if len(navigation_limits) != H2_NAVIGATION_DIM:
        raise ValueError(f"H2 navigation limits must contain {H2_NAVIGATION_DIM} ranges")
    for index, limits in enumerate(navigation_limits):
        if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] > limits[1]:
            raise ValueError(f"H2 navigate_command[{index}] has invalid configured limits {limits}")
        values = navigation[..., index]
        if np.any(values < limits[0] - 1e-5) or np.any(values > limits[1] + 1e-5):
            raise ValueError(f"H2 navigate_command[{index}] is outside configured limits {limits}")
    return output


__all__ = [
    "H2_ACTION_HORIZON",
    "H2_ARENA_JOINT_ACTION_DIM",
    "H2_NAVIGATION_DIM",
    "H2_UPPER_BODY_ACTION_DIM",
    "build_h2_gr00t_action_np",
    "validate_h2_live_joint_limits",
    "validate_h2_joint_configs",
    "validate_h2_action_schedule",
]

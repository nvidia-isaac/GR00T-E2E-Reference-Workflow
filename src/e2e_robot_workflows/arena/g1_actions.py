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

"""Commands that hold the scene still on Arena's two G1 action layouts.

Both layouts end with the same seven locomotion numbers: the navigate command
``[vx, vy, wz]``, the base height, and the torso orientation ``[roll, pitch, yaw]``.
They differ in what precedes them -- joint targets for ``g1_wbc_agile_joint``, hand
states and wrist poses for ``g1_wbc_agile_pink``.
"""

from __future__ import annotations

from typing import Any

import torch

# Navigate (3), base height (1), and torso orientation (3) follow the joint targets.
LOCOMOTION_COMMAND_DIM = 7
NAVIGATE_COMMAND_DIM = 3
# Hip height the AGILE policy holds while standing.
DEFAULT_BASE_HEIGHT_M = 0.75
_BASE_HEIGHT_INDEX_FROM_END = 4


def joint_hold_action(env: Any, base_height_m: float = DEFAULT_BASE_HEIGHT_M) -> torch.Tensor:
    """Return the joint-embodiment command that holds the spawned pose while standing.

    Commanding zeros instead would swing the arms down from the open-arm start pose and
    sweep them across the table, so the spawned joint positions are commanded verbatim.

    Args:
        env: The unwrapped Arena-composed environment.
        base_height_m: Hip height to hold.

    Raises:
        RuntimeError: The environment's action term is not the joint layout.
    """

    robot = env.scene["robot"]
    action_dim = env.action_manager.total_action_dim
    joint_count = len(robot.joint_names)
    if action_dim != joint_count + LOCOMOTION_COMMAND_DIM:
        raise RuntimeError(
            f"This command needs the {joint_count}-joint layout plus {LOCOMOTION_COMMAND_DIM} locomotion "
            f"numbers, but the environment takes {action_dim} numbers. The Pink layout takes end-effector "
            "poses instead, which only the teleoperation retargeter produces."
        )
    action = torch.zeros((env.num_envs, action_dim), device=env.device)
    action[:, :joint_count] = robot.data.default_joint_pos.torch
    action[:, -_BASE_HEIGHT_INDEX_FROM_END] = base_height_m
    return action


def pink_hold_action(env: Any, base_height_m: float = DEFAULT_BASE_HEIGHT_M) -> torch.Tensor:
    """Return the Pink command that targets each wrist where it already is.

    The Pink layout's wrist targets are poses in the pelvis frame with XYZW rotations,
    passed straight to the IK solver, whose model is rooted at the pelvis. Commanding
    the wrists' current poses therefore asks the solver to stay put. Hand states of zero
    hold both hands open.

    Args:
        env: The unwrapped Arena-composed environment.
        base_height_m: Hip height to hold.

    Raises:
        RuntimeError: The environment's action term is not the Pink layout.
    """

    from isaaclab_arena.terms import transforms
    from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy import action_constants

    action_dim = env.action_manager.total_action_dim
    if action_dim != action_constants.TORSO_ORIENTATION_RPY_CMD_END_IDX:
        raise RuntimeError(
            f"This command needs Arena's {action_constants.TORSO_ORIENTATION_RPY_CMD_END_IDX}-number Pink layout, "
            f"but the environment takes {action_dim} numbers."
        )
    action = torch.zeros((env.num_envs, action_dim), device=env.device)
    for link_name, position_slice, rotation_slice in (
        (
            action_constants.LEFT_WRIST_LINK_NAME,
            slice(action_constants.LEFT_WRIST_POS_START_IDX, action_constants.LEFT_WRIST_POS_END_IDX),
            slice(action_constants.LEFT_WRIST_QUAT_START_IDX, action_constants.LEFT_WRIST_QUAT_END_IDX),
        ),
        (
            action_constants.RIGHT_WRIST_LINK_NAME,
            slice(action_constants.RIGHT_WRIST_POS_START_IDX, action_constants.RIGHT_WRIST_POS_END_IDX),
            slice(action_constants.RIGHT_WRIST_QUAT_START_IDX, action_constants.RIGHT_WRIST_QUAT_END_IDX),
        ),
    ):
        action[:, position_slice] = transforms.get_target_link_position_in_target_frame(env, target_link_name=link_name)
        rotation_xyzw = transforms.get_target_link_quaternion_in_target_frame(env, target_link_name=link_name)
        action[:, rotation_slice] = rotation_xyzw
    action[:, -_BASE_HEIGHT_INDEX_FROM_END] = base_height_m
    return action


def hold_action(env: Any, embodiment: str, base_height_m: float = DEFAULT_BASE_HEIGHT_M) -> torch.Tensor:
    """Return the command that holds the scene still on ``embodiment``'s layout.

    Args:
        env: The unwrapped Arena-composed environment.
        embodiment: The embodiment the environment was composed with.
        base_height_m: Hip height to hold.
    """

    # Imported here, not at module scope: the embodiment names live beside the adapter
    # that imports Arena, and this module stays importable without an Arena checkout.
    from e2e_robot_workflows.arena.external_environment import G1_ARENA_EMBODIMENT

    if embodiment.startswith("h2_"):
        from e2e_robot_workflows.arena.h2_actions import h2_hold_action

        return h2_hold_action(env, embodiment)
    if embodiment == G1_ARENA_EMBODIMENT:
        return joint_hold_action(env, base_height_m=base_height_m)
    return pink_hold_action(env, base_height_m=base_height_m)


def set_navigate_command(action: torch.Tensor, vx: float, vy: float, wz: float) -> torch.Tensor:
    """Write the navigate command ``[vx, vy, wz]`` into ``action``, in place.

    The navigate command opens the seven-number locomotion tail both layouts end with, so
    it is addressed from the end of the action and needs no embodiment-specific indices.
    Everything ahead of that tail is left as the caller built it, which is how a hold
    command becomes a walk command without disturbing the arms.

    Returns:
        ``action``, for chaining onto a hold command.
    """

    start = -LOCOMOTION_COMMAND_DIM
    action[:, start : start + NAVIGATE_COMMAND_DIM] = torch.tensor(
        [vx, vy, wz], device=action.device, dtype=action.dtype
    )
    return action


__all__ = [
    "DEFAULT_BASE_HEIGHT_M",
    "LOCOMOTION_COMMAND_DIM",
    "NAVIGATE_COMMAND_DIM",
    "hold_action",
    "joint_hold_action",
    "pink_hold_action",
    "set_navigate_command",
]

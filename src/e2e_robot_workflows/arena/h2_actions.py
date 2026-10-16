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

"""Safe hold commands for the two 61-D Arena H2 embodiments."""

from __future__ import annotations

from typing import Any

import torch

from e2e_robot_workflows.robots.h2 import H2_ARENA_ACTION_DIM, H2_PINK_ACTION_DIM, load_h2_pink_contract


def _pelvis_relative_pose(env: Any, frame_name: str) -> torch.Tensor:
    from isaaclab.utils import math as math_utils

    robot = env.scene["robot"]
    pelvis = robot.data.body_link_pose_w.torch[:, robot.body_names.index("pelvis")]
    frame = robot.data.body_link_pose_w.torch[:, robot.body_names.index(frame_name)]
    position, quaternion = math_utils.subtract_frame_transforms(
        pelvis[:, :3],
        pelvis[:, 3:7],
        frame[:, :3],
        frame[:, 3:7],
    )
    return torch.cat((position, quaternion), dim=-1)


def h2_joint_hold_action(env: Any) -> torch.Tensor:
    """Hold all learned upper-body targets and command zero velocity."""

    contract = load_h2_pink_contract()
    robot = env.scene["robot"]
    target_names = contract.arm_joint_names + contract.hand_joint_order
    target_ids = [robot.joint_names.index(name) for name in target_names]
    action = torch.zeros((env.num_envs, H2_ARENA_ACTION_DIM), device=env.device)
    action[:, :H2_PINK_ACTION_DIM] = robot.data.joint_pos.torch[:, target_ids]
    return action


def h2_pink_hold_action(env: Any) -> torch.Tensor:
    """Hold both live flange poses and hand targets with zero velocity."""

    contract = load_h2_pink_contract()
    robot = env.scene["robot"]
    hand_ids = [robot.joint_names.index(name) for name in contract.hand_joint_order]
    action = torch.zeros((env.num_envs, H2_ARENA_ACTION_DIM), device=env.device)
    action[:, :7] = _pelvis_relative_pose(env, contract.flange_frames[0])
    action[:, 7:14] = _pelvis_relative_pose(env, contract.flange_frames[1])
    action[:, 14:H2_PINK_ACTION_DIM] = robot.data.joint_pos.torch[:, hand_ids]
    return action


def h2_hold_action(env: Any, embodiment: str) -> torch.Tensor:
    if embodiment == "h2_agile_joint":
        return h2_joint_hold_action(env)
    if embodiment == "h2_agile_pink":
        return h2_pink_hold_action(env)
    raise ValueError(f"unsupported H2 Arena embodiment {embodiment!r}")


def set_h2_navigate_command(action: torch.Tensor, vx: float, vy: float, wz: float) -> torch.Tensor:
    """Write the H2 AGILE command into the final three action values."""

    action[:, -3:] = torch.tensor([vx, vy, wz], device=action.device, dtype=action.dtype)
    return action


__all__ = ["h2_hold_action", "h2_joint_hold_action", "h2_pink_hold_action", "set_h2_navigate_command"]

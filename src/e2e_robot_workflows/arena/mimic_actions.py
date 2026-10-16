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

"""Pure action-shape helpers for Arena mimic integration.

The pinned Arena and Isaac Lab mimic boundary is XYZW end to end. This module deliberately has no
quaternion reorder helper; semantic rotation tests guard that decision.
"""

from __future__ import annotations

from typing import Any

from e2e_robot_workflows.data.dataset_manifest import ARENA_PINK_PROFILE, H2_ARENA_PINK_PROFILE


def normalize_mimic_action_inputs(
    target_eef_pose_dict: dict[str, Any],
    passthrough_action_dict: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove AutoData's singleton batch axis before Arena assembles one action."""

    def element_count(value: Any) -> int:
        numel = getattr(value, "numel", None)
        return int(numel() if numel is not None else value.size)

    target_poses = dict(target_eef_pose_dict)
    passthrough = dict(passthrough_action_dict)
    for hand in ("left", "right"):
        pose = target_poses[hand]
        if pose.ndim == 3 and pose.shape[0] == 1:
            pose = pose[0]
        if tuple(pose.shape) != (4, 4):
            raise ValueError(f"Expected one {hand} target pose with shape (4, 4), got {tuple(pose.shape)}")
        target_poses[hand] = pose

        hand_command = passthrough[hand].reshape(-1)
        if element_count(hand_command) != 1:
            raise ValueError(f"Expected one {hand} hand command, got shape {tuple(passthrough[hand].shape)}")
        passthrough[hand] = hand_command[0]

    body_command = passthrough["body"].reshape(-1)
    if element_count(body_command) != 7:
        raise ValueError(f"Expected a 7-D Arena body command, got shape {tuple(passthrough['body'].shape)}")
    passthrough["body"] = body_command
    return target_poses, passthrough


def require_action_width(action: Any, *, rank: int | None = None) -> None:
    """Fail before treating an incompatible action as Arena Pink input."""

    if action.ndim < 1 or action.shape[-1] != ARENA_PINK_PROFILE.width:
        raise ValueError(
            f"Expected Arena Pink action width {ARENA_PINK_PROFILE.width}, got shape {tuple(action.shape)}"
        )
    if rank is not None and action.ndim != rank:
        raise ValueError(f"Expected rank-{rank} Arena Pink action, got shape {tuple(action.shape)}")


def normalize_h2_mimic_action_inputs(
    target_eef_pose_dict: dict[str, Any],
    passthrough_action_dict: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize one H2 target: two matrices, two 22-D hands, and 3-D AGILE."""

    target_poses = dict(target_eef_pose_dict)
    passthrough = dict(passthrough_action_dict)
    for hand in ("left", "right"):
        pose = target_poses[hand]
        if pose.ndim == 3 and pose.shape[0] == 1:
            pose = pose[0]
        if tuple(pose.shape) != (4, 4):
            raise ValueError(f"Expected one {hand} target pose with shape (4, 4), got {tuple(pose.shape)}")
        target_poses[hand] = pose
        command = passthrough[hand].reshape(-1)
        if command.shape[0] != 22:
            raise ValueError(f"Expected a 22-D H2 {hand} hand command, got {tuple(passthrough[hand].shape)}")
        passthrough[hand] = command

    body = passthrough["body"].reshape(-1)
    if body.shape[0] != 3:
        raise ValueError(f"Expected a 3-D H2 AGILE command, got {tuple(passthrough['body'].shape)}")
    passthrough["body"] = body
    return target_poses, passthrough


def require_h2_action_width(action: Any, *, rank: int | None = None) -> None:
    """Fail before treating an incompatible action as H2 Arena Pink input."""

    if action.ndim < 1 or action.shape[-1] != H2_ARENA_PINK_PROFILE.width:
        raise ValueError(
            f"Expected H2 Arena Pink action width {H2_ARENA_PINK_PROFILE.width}, got shape {tuple(action.shape)}"
        )
    if rank is not None and action.ndim != rank:
        raise ValueError(f"Expected rank-{rank} H2 Arena Pink action, got shape {tuple(action.shape)}")


def h2_body_pose_in_pelvis_frame(action: Any) -> Any:
    """Return the pelvis-to-pelvis pose for Mimic's AGILE body pseudo-EEF.

    The final three action values are a velocity passthrough, not a target pose.
    Mimic nevertheless requires a homogeneous pose for every configured EEF.
    """

    require_h2_action_width(action, rank=2)
    pose = action.new_zeros((action.shape[0], 4, 4))
    for axis in range(4):
        pose[:, axis, axis] = 1
    return pose


__all__ = [
    "h2_body_pose_in_pelvis_frame",
    "normalize_h2_mimic_action_inputs",
    "normalize_mimic_action_inputs",
    "require_action_width",
    "require_h2_action_width",
]

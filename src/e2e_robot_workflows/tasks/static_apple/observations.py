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

"""Subtask boundary signal for the static apple transfer task.

Isaac AutoData's automatic annotation replays each source demonstration and samples one boolean
observation term per subtask boundary the task descriptor declares, taking the first
False -> True transition as the boundary. Publishing that term is what lets annotation run
headless instead of pausing for someone to mark the boundary in the viewer.
"""

from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from e2e_robot_workflows.tasks.static_apple.table_height import support_height_offset

DEFAULT_OBJECT_CFG = SceneEntityCfg("object")
DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


def object_held_by_link(
    env: ManagerBasedRLEnv,
    task_link_name: str,
    grasp_distance_m: float,
    min_object_height_m: float,
    object_cfg: SceneEntityCfg = DEFAULT_OBJECT_CFG,
    robot_cfg: SceneEntityCfg = DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
    """Whether the link holds the object clear of the surface it started on.

    Both conditions are needed to place the rising edge at the end of the grasp: the hand passes
    close to the object while reaching for it, and the object leaves the surface only once the
    grasp holds.

    Args:
        env: The environment being stepped.
        task_link_name: Robot link near the selected gripper's grasp centre.
        grasp_distance_m: How close that link must be to the object to count as holding it.
        min_object_height_m: Height above the environment origin the object must exceed to be
            clear of the surface, given for the variant's nominal tabletop. A scene with a
            randomized table shifts it by that episode's live height offset, so a taller
            table does not read an untouched apple as already lifted.
        object_cfg: The object being held.
        robot_cfg: The robot holding it.

    Returns:
        Boolean tensor, one entry per environment.
    """

    obj: RigidObject = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]

    object_pos = obj.data.root_pos_w.torch - env.scene.env_origins
    link_index = robot.data.body_names.index(task_link_name)
    link_pos = robot.data.body_pos_w.torch[:, link_index, :] - env.scene.env_origins

    in_hand = torch.linalg.vector_norm(link_pos - object_pos, dim=1) < grasp_distance_m
    lifted = object_pos[:, 2] > min_object_height_m + support_height_offset(env)

    return in_hand & lifted

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

"""Arena events owned by the E2E H2 embodiment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedEnv


def apply_high_friction_to_h2_fingers(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    material_path: str,
    static_friction: float,
    dynamic_friction: float,
    prim_name_markers: Sequence[str],
) -> None:
    """Bind the configured contact material to H2 Sharpa collision prims."""

    del env_ids
    from isaaclab.sim import bind_physics_material
    from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
    from pxr import UsdPhysics, UsdShade

    stage = env.sim.stage
    material_cfg = RigidBodyMaterialCfg(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        friction_combine_mode="max",
    )
    material_cfg.func(material_path, material_cfg)

    bound_count = 0
    for env_prim_path in env.scene.env_prim_paths:
        robot_prim_path = f"{env_prim_path}/Robot"
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if not prim_path.startswith(robot_prim_path):
                continue
            if not any(marker in prim_path.lower() for marker in prim_name_markers):
                continue
            if not (prim.HasAPI(UsdPhysics.CollisionAPI) or "PhysicsCollisionAPI" in prim.GetAppliedSchemas()):
                continue
            bind_physics_material(prim_path, material_path, stage=stage)
            binding = UsdShade.MaterialBindingAPI(prim).GetDirectBinding("physics")
            if str(binding.GetMaterialPath()) == material_path:
                bound_count += 1

    if bound_count == 0:
        raise RuntimeError("no H2 Sharpa collision prims received the configured contact material")


def follow_h2_head_camera(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    *,
    camera_name: str = "robot_head_cam",
    robot_name: str = "robot",
    head_body_name: str = "head_yaw_link",
) -> None:
    """Move the H2 RGB camera to its configured offset from the live head pose."""

    from isaaclab.utils import math as math_utils

    camera = env.scene[camera_name]
    robot = env.scene[robot_name]
    body_poses = robot.data.body_link_pose_w.torch
    head_index = robot.body_names.index(head_body_name)
    head_pose = body_poses[:, head_index] if env_ids is None else body_poses[env_ids, head_index]
    offset_position = head_pose.new_tensor(camera.cfg.offset.pos).expand(head_pose.shape[0], -1)
    offset_orientation = head_pose.new_tensor(camera.cfg.offset.rot).expand(head_pose.shape[0], -1)
    position, orientation = math_utils.combine_frame_transforms(
        head_pose[:, :3],
        head_pose[:, 3:7],
        offset_position,
        offset_orientation,
    )
    camera.set_world_poses(
        position,
        orientation,
        env_ids=env_ids,
        convention=camera.cfg.offset.convention,
    )
    # This event runs after the environment's scheduled render. Force the
    # camera observation that follows it to resynchronize renderer transforms
    # at the same physics step instead of reusing the pre-event transform.
    env.sim.render_context.reset_transform_cadence()


def reset_h2_head_camera(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
) -> None:
    """Synchronize the H2 camera after reset pose writes and before reset rendering."""

    # Reset events write root and joint poses directly into PhysX. Refresh the
    # resulting link kinematics before reading the live head pose; the normal
    # environment reset render happens after this event returns.
    env.sim.forward()
    follow_h2_head_camera(env, env_ids)


__all__ = ["apply_high_friction_to_h2_fingers", "follow_h2_head_camera", "reset_h2_head_camera"]

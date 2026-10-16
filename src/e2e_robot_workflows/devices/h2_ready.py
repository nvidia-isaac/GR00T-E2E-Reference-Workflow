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

"""Guarded H2 startup, before controller alignment or demonstration recording."""

from __future__ import annotations

from itertools import product

import numpy as np
from loguru import logger

from e2e_robot_workflows.tasks.static_apple.table_height import support_height_offset_value

READY_STEPS = 300
STABLE_STEPS = 40
MIN_CLEARANCE_M = 0.04
MAX_DOWNWARD_SPEED_M_S = 0.5


class H2StartupGuard:
    """Fail closed on fast descent, low geometry, or an unsettled final window."""

    def __init__(self, dt: float, support_z: float):
        if not np.isfinite([dt, support_z]).all() or dt <= 0:
            raise ValueError("H2 startup needs a finite positive timestep and table height")
        self.dt = dt
        self.support_z = support_z
        self.previous_z = None
        self.stable_steps = 0
        self.minimum_clearance_m = float("inf")
        self.maximum_downward_speed_m_s = 0.0

    def observe(self, points, root_velocity, arm_velocity) -> None:
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError("H2 startup requires nonempty world collision bounds")
        if not all(np.isfinite(value).all() for value in (points, root_velocity, arm_velocity)):
            raise RuntimeError("H2 startup encountered nonfinite robot state")
        z = points[:, 2]
        clearance = float(z.min() - self.support_z)
        downward = 0.0 if self.previous_z is None else float(np.maximum((self.previous_z - z) / self.dt, 0).max())
        self.minimum_clearance_m = min(self.minimum_clearance_m, clearance)
        self.maximum_downward_speed_m_s = max(self.maximum_downward_speed_m_s, downward)
        self.previous_z = z.copy()
        if clearance < MIN_CLEARANCE_M or downward > MAX_DOWNWARD_SPEED_M_S:
            raise RuntimeError(
                f"H2 startup blocked: arm/hand clearance={clearance:.4f} m, downward speed={downward:.3f} m/s; "
                "teleoperation has not been enabled"
            )
        stable = (
            np.linalg.norm(root_velocity[:3]) < 0.08
            and np.linalg.norm(root_velocity[3:]) < 0.15
            and np.max(np.abs(arm_velocity)) < 0.35
        )
        self.stable_steps = self.stable_steps + 1 if stable else 0

    def finish(self) -> dict:
        if self.stable_steps < STABLE_STEPS:
            raise RuntimeError(f"H2 startup did not settle: {self.stable_steps}/{STABLE_STEPS} stable samples")
        return {
            "minimum_clearance_m": self.minimum_clearance_m,
            "maximum_downward_speed_m_s": self.maximum_downward_speed_m_s,
            "stable_steps": self.stable_steps,
        }


def _collision_bounds(env):
    """Conservative body-local corners of every enabled forearm/hand collider."""

    from pxr import Usd, UsdGeom, UsdPhysics

    robot = env.scene["robot"]
    stage = env.sim.stage
    root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"], False, True)
    ids, corners = [], []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if (
            not prim.HasAPI(UsdPhysics.CollisionAPI)
            or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False
        ):
            continue
        if not any(f"/{side}_elbow_link/" in str(prim.GetPath()) for side in ("left", "right")):
            continue
        body = prim
        while body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body.IsValid() or body.GetName() not in robot.body_names:
            raise RuntimeError(f"Unresolved H2 startup collision body: {prim.GetPath()}")
        bounds = cache.ComputeRelativeBound(prim, body).ComputeAlignedRange()
        if bounds.IsEmpty():
            raise RuntimeError(f"Empty H2 startup collision bounds: {prim.GetPath()}")
        lower, upper = np.array(bounds.GetMin()), np.array(bounds.GetMax())
        corners.extend([np.where(mask, upper, lower) for mask in product((False, True), repeat=3)])
        ids.extend([robot.body_names.index(body.GetName())] * 8)
    covered = {robot.body_names[index] for index in ids}
    if not all(f"{side}_elbow_link" in covered for side in ("left", "right")):
        raise RuntimeError("H2 startup collision bounds omit one or both forearms")
    return np.asarray(ids), np.asarray(corners)


def prime_h2_controller(env, embodiment: str):
    """Initialize recurrent observations without advancing physics or recording frames."""

    from e2e_robot_workflows.arena.h2_actions import h2_hold_action

    action = h2_hold_action(env, embodiment)
    lower_body = env.action_manager.get_term("lower_body_joint_pos")
    for _ in range(2 * lower_body.cfg.history_length):
        lower_body.process_actions(action[:, -3:])
    return action


def prepare_h2_recording(env, *, embodiment="h2_agile_pink", is_running=lambda: True) -> dict:
    """Settle without recorder frames, then capture the actual ready initial state.

    No robot/controller reset occurs at readiness. Recurrent history is recorded
    with the physical snapshot so replay and AutoData start from the same state.
    """

    from isaaclab.managers import RecorderManager, RecorderManagerBaseCfg
    from scipy.spatial.transform import Rotation

    from e2e_robot_workflows.robots.h2 import load_h2_pink_contract

    if env.num_envs != 1:
        raise ValueError("H2 startup supports exactly one environment")
    robot = env.scene["robot"]
    ids, corners = _collision_bounds(env)
    arm_ids = [robot.joint_names.index(name) for name in load_h2_pink_contract().arm_joint_names]
    guard = H2StartupGuard(
        float(env.step_dt), float(env.cfg.e2e_support_surface_z_m) + support_height_offset_value(env)
    )
    action = prime_h2_controller(env, embodiment)
    recorder = env.recorder_manager
    logger.info("H2 startup: holding raised hands and waiting for a stable stance; controls disabled.")
    try:
        env.recorder_manager = RecorderManager(RecorderManagerBaseCfg(), env)
        for step in range(READY_STEPS + 1):
            if not is_running() or env.sim.is_stopped():
                raise RuntimeError("H2 startup interrupted before readiness")
            if step:
                _, _, terminated, truncated, _ = env.step(action)
                if (terminated | truncated).any().item():
                    raise RuntimeError("H2 startup unexpectedly reset the scene")
            poses = robot.data.body_link_pose_w.torch[0].detach().cpu().numpy()[ids]
            points = Rotation.from_quat(poses[:, 3:7]).apply(corners) + poses[:, :3]
            root_velocity = robot.data.root_link_vel_w.torch[0].detach().cpu().numpy()
            arm_velocity = robot.data.joint_vel.torch[0, arm_ids].detach().cpu().numpy()
            try:
                guard.observe(points, root_velocity, arm_velocity)
            except RuntimeError as error:
                error.add_note(f"startup step={step}, root velocity={root_velocity}, arm velocity={arm_velocity}")
                raise
        report = guard.finish()
    finally:
        env.recorder_manager = recorder
    env.episode_length_buf[:] = 0
    env.observation_manager.reset()
    env.obs_buf = env.observation_manager.compute(update_history=True)
    recorder.reset()
    recorder.record_post_reset([0])
    logger.info("H2 READY: {}. Align controllers from this pose; no further robot reset.", report)
    return report

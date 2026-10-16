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

"""Pelvis-relative 58-D Pink/Sharpa action for the Arena H2 embodiment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import numpy as np
import torch
from isaaclab.controllers.pink_ik.pink_tasks import LocalFrameTask
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.configclass import configclass
from pink.tasks import FrameTask

from e2e_robot_workflows.controllers.errors import H2MeasuredJointLimitViolation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class H2PinkInverseKinematicsAction(PinkInverseKinematicsAction):
    """Interpret the two commanded poses in the pelvis frame and preserve live order."""

    cfg: H2PinkInverseKinematicsActionCfg

    def __init__(self, cfg: H2PinkInverseKinematicsActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        live_joint_names = tuple(self._asset.data.joint_names)
        expected_arms = tuple(name for name in live_joint_names if name in cfg.pink_controlled_joint_names)
        expected_hands = tuple(name for name in live_joint_names if name in cfg.hand_joint_names)
        if tuple(self._isaaclab_controlled_joint_names) != expected_arms:
            raise ValueError(f"H2 Pink arm projection changed: {self._isaaclab_controlled_joint_names}")
        if tuple(self._hand_joint_names) != expected_hands:
            raise ValueError(f"H2 Sharpa projection changed: {self._hand_joint_names}")
        guard_values = (cfg.joint_limit_guard_margin, cfg.joint_limit_guard_tolerance)
        if any(isinstance(value, (bool, np.bool_)) or not np.isfinite(value) for value in guard_values):
            raise ValueError("H2 Pink joint-limit guard margin and tolerance must be finite numbers")
        if not 0.0 < cfg.joint_limit_guard_margin < cfg.joint_limit_guard_tolerance:
            raise ValueError("H2 Pink joint-limit guard requires 0 < margin < tolerance")
        if (
            isinstance(cfg.joint_target_limit_margin, (bool, np.bool_))
            or not np.isfinite(cfg.joint_target_limit_margin)
            or cfg.joint_target_limit_margin <= 0.0
        ):
            raise ValueError("H2 Pink joint-target limit margin must be finite and positive")
        if cfg.joint_target_limit_margin <= cfg.joint_limit_guard_tolerance:
            raise ValueError("H2 Pink joint-target limit margin must exceed the guard tolerance")
        for ik_controller in self._ik_controllers:
            lower, upper = self._controlled_limits_in_isaaclab_order(ik_controller)
            if np.any(upper - lower <= 2.0 * cfg.joint_target_limit_margin):
                raise ValueError("H2 Pink joint-target limit margin collapses a controlled joint interval")
        self._control_dt = env.step_dt
        self._has_cached_joint_position_target = False

    def process_actions(self, actions: torch.Tensor) -> None:
        """Resolve one joint target per 100 Hz H2 control step."""

        self._has_cached_joint_position_target = False
        if not torch.isfinite(actions).all().item():
            raise ValueError("H2 Pink actions must contain only finite values")
        self._raw_actions[:] = actions
        self._target_hand_joint_positions = actions[:, -self.hand_joint_dim :]

        # H2's two 7-D flange targets are already pelvis-relative. The upstream
        # world-frame path builds and immediately dismantles 4x4 transforms and
        # reads the live pelvis transform even though none of those values are
        # used here. Convert the two quaternions once and set the local tasks.
        pose_actions = actions[:, : self._num_frame_tasks * self.pose_dim].reshape(
            self.num_envs,
            self._num_frame_tasks,
            self.pose_dim,
        )
        positions = pose_actions[..., : self.position_dim].transpose(0, 1)
        quaternions = pose_actions[..., self.position_dim :].transpose(0, 1)
        self._set_task_targets((positions, math_utils.matrix_from_quat(quaternions)))

        ik_joint_positions = self._compute_ik_solutions(dt=self._control_dt)
        self._processed_actions[:] = torch.cat((ik_joint_positions, self._target_hand_joint_positions), dim=1)
        self._has_cached_joint_position_target = True

    def _set_task_targets(self, transformed_poses: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Set both flange targets after one device-to-host transfer."""

        positions, rotation_matrices = transformed_poses
        packed_targets = torch.cat((positions, rotation_matrices.flatten(-2)), dim=-1).detach().cpu().numpy()
        for env_index, ik_controller in enumerate(self._ik_controllers):
            for frame_task_index, task in enumerate(ik_controller.cfg.variable_input_tasks):
                if isinstance(task, LocalFrameTask):
                    target = task.transform_target_to_base
                elif isinstance(task, FrameTask):
                    target = task.transform_target_to_world
                else:
                    continue
                target.translation = packed_targets[frame_task_index, env_index, :3]
                target.rotation = packed_targets[frame_task_index, env_index, 3:].reshape(3, 3)
                task.set_target(target)

    def apply_actions(self) -> None:
        """Apply the cached target during each 200 Hz physics substep."""

        if not self._has_cached_joint_position_target:
            raise RuntimeError("H2 Pink actions must be processed before their cached target can be applied")
        if self.cfg.enable_gravity_compensation:
            self._apply_gravity_compensation()
        self._asset.set_joint_position_target_index(
            target=self._processed_actions,
            joint_ids=self._controlled_joint_ids,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Invalidate the cached target so reset state cannot receive a stale command."""

        super().reset(env_ids)
        self._has_cached_joint_position_target = False

    def _compute_ik_solutions(self, *, dt: float | None = None) -> torch.Tensor:
        """Solve after guarding states and keeping every target inside URDF limits."""

        integration_dt = self._sim_dt if dt is None else dt
        ik_solutions = []
        current_joint_positions = self._asset.data.joint_pos.torch.cpu().numpy()
        for env_index, ik_controller in enumerate(self._ik_controllers):
            lower, upper = self._controlled_limits_in_isaaclab_order(ik_controller)
            joint_ids = np.asarray(ik_controller.controlled_joint_indices)
            current = current_joint_positions[env_index].copy()
            current[joint_ids] = self._guard_joint_limits(
                current[joint_ids],
                lower,
                upper,
                joint_names=ik_controller.cfg.joint_names,
                env_index=env_index,
                stage="current state",
            )

            target = ik_controller.compute(current, integration_dt)
            guarded_target = self._guard_joint_limits(
                target.detach().cpu().numpy(),
                lower,
                upper,
                joint_names=ik_controller.cfg.joint_names,
                env_index=env_index,
                stage="IK target",
            )
            guarded_target = np.clip(
                guarded_target,
                lower + self.cfg.joint_target_limit_margin,
                upper - self.cfg.joint_target_limit_margin,
            )
            ik_solutions.append(target.new_tensor(guarded_target))

        return torch.stack(ik_solutions)

    @staticmethod
    def _controlled_limits_in_isaaclab_order(ik_controller) -> tuple[np.ndarray, np.ndarray]:
        model = ik_controller.pink_configuration.controlled_model
        lower_pink = np.asarray(model.lowerPositionLimit, dtype=np.float64)
        upper_pink = np.asarray(model.upperPositionLimit, dtype=np.float64)
        pink_to_isaaclab = np.asarray(ik_controller.isaac_lab_to_pink_controlled_ordering)
        joint_count = len(ik_controller.controlled_joint_indices)
        if lower_pink.shape != (joint_count,) or upper_pink.shape != (joint_count,):
            raise RuntimeError("H2 Pink joint-limit guard requires one position coordinate per controlled joint")
        if not np.isfinite(lower_pink).all() or not np.isfinite(upper_pink).all():
            raise RuntimeError("H2 Pink controlled joint limits must be finite")
        if np.any(lower_pink >= upper_pink):
            raise RuntimeError("H2 Pink controlled joint limits must be strictly ordered")
        if sorted(pink_to_isaaclab.tolist()) != list(range(joint_count)):
            raise RuntimeError("H2 Pink joint-limit ordering is not a permutation of the controlled joints")

        lower = np.empty(joint_count, dtype=np.float64)
        upper = np.empty(joint_count, dtype=np.float64)
        lower[pink_to_isaaclab] = lower_pink
        upper[pink_to_isaaclab] = upper_pink
        return lower, upper

    def _guard_joint_limits(
        self,
        positions: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        joint_names: list[str],
        env_index: int,
        stage: str,
    ) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != lower.shape or len(joint_names) != positions.size:
            raise RuntimeError("H2 Pink joint-limit guard received an inconsistent joint projection")
        if not np.isfinite(positions).all():
            raise RuntimeError(f"H2 Pink {stage} contains a non-finite joint position in environment {env_index}")

        below = lower - positions
        above = positions - upper
        excursions = np.maximum(below, above)
        material = excursions > self.cfg.joint_limit_guard_tolerance
        if material.any():
            joint_index = int(np.flatnonzero(material)[0])
            error_type = H2MeasuredJointLimitViolation if stage == "current state" else RuntimeError
            raise error_type(
                f"H2 Pink {stage} materially violates {joint_names[joint_index]} limits in environment {env_index}: "
                f"value={positions[joint_index]:.9f}, limits=[{lower[joint_index]:.9f}, "
                f"{upper[joint_index]:.9f}], excursion={excursions[joint_index]:.3g} rad"
            )

        guarded = positions.copy()
        microscopic_lower = below > 0.0
        microscopic_upper = above > 0.0
        guarded[microscopic_lower] = lower[microscopic_lower] + self.cfg.joint_limit_guard_margin
        guarded[microscopic_upper] = upper[microscopic_upper] - self.cfg.joint_limit_guard_margin
        return guarded


@configclass
class H2PinkInverseKinematicsActionCfg(PinkInverseKinematicsActionCfg):
    class_type: type[ActionTerm] = H2PinkInverseKinematicsAction
    joint_limit_guard_tolerance: float = 5e-5
    """Largest URDF-limit excursion treated as numerical noise, in radians."""

    joint_limit_guard_margin: float = 5e-6
    """Interior margin used when correcting a numerical limit excursion, in radians."""

    joint_target_limit_margin: float = 5e-4
    """Headroom kept between every H2 Pink target and its URDF limit, in radians."""


__all__ = ["H2PinkInverseKinematicsAction", "H2PinkInverseKinematicsActionCfg"]

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

"""Capture measured H2 wrist poses when XR recording starts, then limit target slew.

This changes only the first 14 pose entries of the H2 Arena action, before recording.
It does not move the XR anchor, rotate translation axes, or alter hand/navigation
commands. Callers must reset the helper on robot reset and recording resume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from e2e_robot_workflows.robots.h2 import H2_ARENA_ACTION_DIM


@dataclass(frozen=True)
class H2ResetAlignmentConfig:
    """Per-hand target speed limits in meters/second and radians/second."""

    max_linear_speed: float = 0.75
    max_angular_speed: float = 3.0

    def __post_init__(self) -> None:
        for value in (self.max_linear_speed, self.max_angular_speed):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError("H2 reset-alignment speed limits must be positive and finite")


class H2ResetAlignment:
    """Align each controller to its measured wrist pose on the first valid sample.

    The captured orientation offset is right-multiplied, so controller rotation
    deltas remain in pelvis axes: ``R_target = R_raw * inverse(R_raw_start) *
    R_robot_start``. Translation deltas stay in those same axes without rotation.
    Subsequent targets take the shortest angular path with bounded per-step slew.
    """

    def __init__(self, config: H2ResetAlignmentConfig | None = None) -> None:
        self.config = config if config is not None else H2ResetAlignmentConfig()
        self.reset()

    @property
    def is_aligned(self) -> bool:
        return self._raw_reference is not None

    def reset(self) -> None:
        self._raw_reference: np.ndarray | None = None
        self._robot_reference: np.ndarray | None = None
        self._previous_pose: np.ndarray | None = None

    @staticmethod
    def _normalized_action(action, name: str) -> np.ndarray:
        import torch

        if (
            not isinstance(action, torch.Tensor)
            or action.shape != (H2_ARENA_ACTION_DIM,)
            or not action.is_floating_point()
        ):
            raise ValueError(
                f"H2 reset alignment requires a floating {name} tensor with shape ({H2_ARENA_ACTION_DIM},)"
            )
        values = action.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
        if not np.isfinite(values).all():
            raise ValueError(f"H2 reset alignment requires a finite {name}")
        poses = values[:14].reshape(2, 7)
        norms = np.linalg.norm(poses[:, 3:7], axis=1)
        if not np.isfinite(norms).all() or np.any(norms < 1e-8):
            raise ValueError(f"H2 reset alignment requires nonzero finite XYZW quaternions in {name}")
        poses[:, 3:7] /= norms[:, None]
        return values

    def apply(self, action, reference_action=None, *, dt: float):
        """Copy one torch action and replace its poses with aligned, bounded targets.

        ``reference_action`` is required only before capture and is ignored after
        alignment. The first output matches its measured poses, with normalized
        XYZW quaternions; its hand/navigation suffix always comes from ``action``.
        Invalid inputs raise ValueError without capturing or advancing state.
        ``dt`` must be the simulation control period, not elapsed wall time.
        """

        from scipy.spatial.transform import Rotation

        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("H2 reset alignment requires a positive finite control dt")
        raw_pose = self._normalized_action(action, "action")[:14].reshape(2, 7)
        if not self.is_aligned:
            robot_pose = self._normalized_action(reference_action, "reference action")[:14].reshape(2, 7)
            target_pose = robot_pose.copy()
        else:
            assert self._raw_reference is not None
            assert self._robot_reference is not None
            assert self._previous_pose is not None
            target_pose = self._previous_pose.copy()
            for hand in range(2):
                desired_position = self._robot_reference[hand, :3] + (
                    raw_pose[hand, :3] - self._raw_reference[hand, :3]
                )
                displacement = desired_position - self._previous_pose[hand, :3]
                distance = float(np.linalg.norm(displacement))
                if not math.isfinite(distance):
                    raise ValueError("H2 reset-alignment position displacement must be finite")
                if distance > 0.0:
                    displacement *= min(1.0, self.config.max_linear_speed * dt / distance)
                target_pose[hand, :3] += displacement

                desired_rotation = (
                    Rotation.from_quat(raw_pose[hand, 3:7])
                    * Rotation.from_quat(self._raw_reference[hand, 3:7]).inv()
                    * Rotation.from_quat(self._robot_reference[hand, 3:7])
                )
                previous_rotation = Rotation.from_quat(self._previous_pose[hand, 3:7])
                rotation_delta = (desired_rotation * previous_rotation.inv()).as_rotvec()
                angle = float(np.linalg.norm(rotation_delta))
                if angle > 0.0:
                    rotation_delta *= min(1.0, self.config.max_angular_speed * dt / angle)
                quaternion = (Rotation.from_rotvec(rotation_delta) * previous_rotation).as_quat()
                if np.dot(quaternion, self._previous_pose[hand, 3:7]) < 0.0:
                    quaternion *= -1.0
                target_pose[hand, 3:7] = quaternion

        adjusted = action.clone()
        adjusted[:14] = action.new_tensor(target_pose.reshape(14))
        # Validate representability in the caller's dtype before committing state.
        emitted_pose = self._normalized_action(adjusted, "aligned action")[:14].reshape(2, 7)
        if not self.is_aligned:
            self._raw_reference = raw_pose.copy()
            self._robot_reference = robot_pose.copy()
        self._previous_pose = emitted_pose.copy()
        return adjusted

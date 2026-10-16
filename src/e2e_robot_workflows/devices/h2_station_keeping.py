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

"""Bounded H2 XR station keeping through explicit recorded velocity commands.

This assists a free-standing robot; it neither pins its base nor moves the XR
anchor. Apply the returned command to the action tail *before* ``env.step`` so
recording, replay, AutoData, and training all see the actual AGILE input. The
bounded gains require GPU behavior validation and a physical PICO comfort check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from e2e_robot_workflows.robots.h2 import H2_ARENA_ACTION_DIM, load_h2_robot_config


@dataclass(frozen=True)
class H2StationKeepingConfig:
    """Conservative planar position/yaw feedback for neutral thumbsticks."""

    position_gain: float = 1.2
    yaw_gain: float = 0.6
    max_linear_speed: float = 0.10
    max_yaw_speed: float = 0.15
    position_deadband_m: float = 0.005
    yaw_deadband_rad: float = 0.01
    navigate_command_limits: tuple[tuple[float, float], ...] = field(
        default_factory=lambda: load_h2_robot_config().navigate_command_limits
    )

    def __post_init__(self) -> None:
        positive = (self.position_gain, self.yaw_gain, self.max_linear_speed, self.max_yaw_speed)
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("H2 station-keeping gains and speed caps must be positive and finite")
        deadbands = (self.position_deadband_m, self.yaw_deadband_rad)
        if not all(math.isfinite(value) and value >= 0.0 for value in deadbands):
            raise ValueError("H2 station-keeping deadbands must be non-negative and finite")
        limits = np.asarray(self.navigate_command_limits, dtype=np.float64)
        if (
            limits.shape != (3, 2)
            or not np.isfinite(limits).all()
            or np.any(limits[:, 0] > 0.0)
            or np.any(limits[:, 1] < 0.0)
        ):
            raise ValueError("H2 station keeping requires three finite command ranges containing zero")
        caps = np.asarray((self.max_linear_speed, self.max_linear_speed, self.max_yaw_speed))
        if np.any(-caps < limits[:, 0]) or np.any(caps > limits[:, 1]):
            raise ValueError("H2 station-keeping caps must fit inside the navigation-command envelope")


class H2StationKeeping:
    """Hold the last neutral XY/yaw reference, yielding immediately to navigation.

    Any nonzero requested velocity is deliberate input and passes through
    unchanged. Apply controller deadzones upstream. Returning to zero latches
    the current pose, so assistance never pulls an operator back to the old
    station. Reset on each episode boundary and tracking-loss recovery.
    """

    def __init__(self, config: H2StationKeepingConfig | None = None) -> None:
        self.config = config if config is not None else H2StationKeepingConfig()
        self._reference_xy: np.ndarray | None = None
        self._reference_yaw: float | None = None

    def reset(self) -> None:
        self._reference_xy = None
        self._reference_yaw = None

    @staticmethod
    def _finite_vector(value, size: int, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (size,) or not np.isfinite(vector).all():
            raise ValueError(f"H2 station keeping requires finite {name} with shape ({size},)")
        return vector

    def command(self, requested, position_xy, orientation_xyzw) -> np.ndarray:
        """Return a float32 ``[vx, vy, wz]`` action without changing the inputs.

        Invalid measurements/commands clear the reference and raise ValueError;
        the caller must not step physics using stale feedback in that case.
        """

        try:
            requested = self._finite_vector(requested, 3, "requested command")
            position_xy = self._finite_vector(position_xy, 2, "world XY position")
            quaternion = self._finite_vector(orientation_xyzw, 4, "XYZW orientation")
            norm = float(np.linalg.norm(quaternion))
            if not math.isfinite(norm) or norm < 1e-8:
                raise ValueError("H2 station keeping requires a nonzero finite quaternion norm")
            x, y, z, w = quaternion / norm
            yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            limits = np.asarray(self.config.navigate_command_limits)
            if np.any(requested < limits[:, 0]) or np.any(requested > limits[:, 1]):
                raise ValueError("H2 requested command exceeds the navigation-command envelope")

            if np.any(requested != 0.0):
                self.reset()
                return requested.astype(np.float32)
            if self._reference_xy is None:
                self._reference_xy = position_xy.copy()
                self._reference_yaw = yaw
                return np.zeros(3, dtype=np.float32)

            error_xy = self._reference_xy - position_xy
            distance = float(np.linalg.norm(error_xy))
            if not math.isfinite(distance):
                raise ValueError("H2 station-keeping position error must be finite")
            command = np.zeros(3, dtype=np.float32)
            if distance > self.config.position_deadband_m:
                speed = min(
                    self.config.position_gain * (distance - self.config.position_deadband_m),
                    self.config.max_linear_speed,
                )
                velocity_world = error_xy * (speed / distance)
                cosine, sine = math.cos(yaw), math.sin(yaw)
                command[0] = cosine * velocity_world[0] + sine * velocity_world[1]
                command[1] = -sine * velocity_world[0] + cosine * velocity_world[1]

            assert self._reference_yaw is not None
            yaw_error = math.atan2(math.sin(self._reference_yaw - yaw), math.cos(self._reference_yaw - yaw))
            yaw_speed = min(
                self.config.yaw_gain * max(abs(yaw_error) - self.config.yaw_deadband_rad, 0.0),
                self.config.max_yaw_speed,
            )
            command[2] = math.copysign(yaw_speed, yaw_error)
            return command
        except (TypeError, ValueError):
            self.reset()
            raise


def apply_h2_station_keeping(action, robot_data, helper: H2StationKeeping):
    """Copy one H2 action and write the effective navigation command before recording."""

    if tuple(action.shape) not in ((H2_ARENA_ACTION_DIM,), (1, H2_ARENA_ACTION_DIM)):
        raise ValueError(f"H2 station keeping requires one {H2_ARENA_ACTION_DIM}-D action")
    flat_action = action.reshape(H2_ARENA_ACTION_DIM)
    command = helper.command(
        flat_action[-3:].detach().cpu().numpy(),
        robot_data.root_link_pos_w.torch[0, :2].detach().cpu().numpy(),
        robot_data.root_link_quat_w.torch[0].detach().cpu().numpy(),
    )
    adjusted = action.clone()
    adjusted.reshape(H2_ARENA_ACTION_DIM)[-3:] = action.new_tensor(command)
    return adjusted

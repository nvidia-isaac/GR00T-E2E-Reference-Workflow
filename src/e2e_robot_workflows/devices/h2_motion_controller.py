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

"""PICO/OpenXR retargeters for the Arena H2 Pink embodiment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from isaacteleop.retargeters import Se3AbsRetargeter
from isaacteleop.retargeting_engine.interface import BaseRetargeter, RetargeterIOType
from isaacteleop.retargeting_engine.interface.retargeter_core_types import RetargeterIO
from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType, TensorGroupType
from isaacteleop.retargeting_engine.tensor_types import (
    ControllerInput,
    ControllerInputIndex,
    DLDataType,
    NDArrayType,
    RobotHandJoints,
)

from e2e_robot_workflows.robots.h2 import load_h2_robot_config


@dataclass(frozen=True)
class H2VelocityCommandConfig:
    """Thumbstick scaling for H2 AGILE ``[vx, vy, wz]``."""

    movement_scale: float = 0.5
    rotation_scale: float = 0.35
    thumbstick_deadzone: float = 0.1
    navigate_command_limits: tuple[tuple[float, float], ...] = field(
        default_factory=lambda: load_h2_robot_config().navigate_command_limits
    )

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not np.isfinite(value) or value <= 0
            for value in (self.movement_scale, self.rotation_scale)
        ):
            raise ValueError("H2 locomotion scales must be positive")
        if (
            isinstance(self.thumbstick_deadzone, bool)
            or not np.isfinite(self.thumbstick_deadzone)
            or not 0.0 <= self.thumbstick_deadzone < 1.0
        ):
            raise ValueError("H2 thumbstick deadzone must be finite and in [0, 1)")
        if len(self.navigate_command_limits) != 3 or any(
            len(limits) != 2 or not all(np.isfinite(value) for value in limits) or limits[0] > 0.0 or limits[1] < 0.0
            for limits in self.navigate_command_limits
        ):
            raise ValueError("H2 locomotion requires three finite navigation-command ranges containing zero")


class H2VelocityCommandRetargeter(BaseRetargeter):
    """Map left XY and right X thumbsticks to three AGILE commands."""

    def __init__(self, config: H2VelocityCommandConfig, name: str) -> None:
        self._config = config
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        return {
            "controller_left": OptionalType(ControllerInput()),
            "controller_right": OptionalType(ControllerInput()),
        }

    def output_spec(self) -> RetargeterIOType:
        return {
            "velocity": TensorGroupType(
                "h2_velocity",
                [NDArrayType("command", shape=(3,), dtype=DLDataType.FLOAT, dtype_bits=32)],
            )
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        left_x = left_y = right_x = 0.0
        left = inputs["controller_left"]
        if not left.is_none and bool(left[ControllerInputIndex.GRIP_IS_VALID]):
            left_x = float(left[ControllerInputIndex.THUMBSTICK_X])
            left_y = float(left[ControllerInputIndex.THUMBSTICK_Y])
        right = inputs["controller_right"]
        if not right.is_none and bool(right[ControllerInputIndex.GRIP_IS_VALID]):
            right_x = float(right[ControllerInputIndex.THUMBSTICK_X])
        axes = np.asarray((left_y, -left_x, -right_x), dtype=np.float32)
        # A malformed navigation sample must not count as deliberate navigation
        # or release the station-keeping latch. Pose validity is handled separately.
        if not np.isfinite(axes).all():
            axes[:] = 0.0
        axes = np.clip(axes, -1.0, 1.0)
        deadzone = self._config.thumbstick_deadzone
        axes = np.sign(axes) * np.maximum(np.abs(axes) - deadzone, 0.0) / (1.0 - deadzone)
        command = np.asarray(
            (
                axes[0] * self._config.movement_scale,
                axes[1] * self._config.movement_scale,
                axes[2] * self._config.rotation_scale,
            ),
            dtype=np.float32,
        )
        limits = np.asarray(self._config.navigate_command_limits, dtype=np.float32)
        outputs["velocity"][0] = np.clip(command, limits[:, 0], limits[:, 1])


@dataclass(frozen=True)
class H2SharpaControllerHandsConfig:
    """Binary grasp targets in the qualified live Sharpa order."""

    joint_names: tuple[str, ...]
    open_joint_pos_by_name: dict[str, float]
    closed_joint_pos_by_name: dict[str, float]
    # Invalid samples that retain the last grasp before the hand fails open.
    tracking_loss_hold_samples: int

    close_threshold: float = 0.55
    open_threshold: float = 0.35

    def __post_init__(self) -> None:
        expected = set(self.joint_names)
        if len(expected) != len(self.joint_names):
            raise ValueError("H2 Sharpa controller joint names must be distinct")
        for label, values in (("open", self.open_joint_pos_by_name), ("closed", self.closed_joint_pos_by_name)):
            if set(values) != expected:
                raise ValueError(f"H2 Sharpa {label} targets must cover exactly the configured joints")
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError(f"H2 Sharpa {label} targets must be finite")
        if not 0.0 <= self.open_threshold < self.close_threshold <= 1.0:
            raise ValueError("H2 Sharpa thresholds must satisfy 0 <= open < close <= 1")
        if (
            isinstance(self.tracking_loss_hold_samples, bool)
            or not isinstance(self.tracking_loss_hold_samples, int)
            or self.tracking_loss_hold_samples < 0
        ):
            raise ValueError("H2 Sharpa tracking-loss hold must be a non-negative integer sample count")


class H2SharpaControllerHandsRetargeter(BaseRetargeter):
    """Map trigger or squeeze to independent binary grasps that fail open."""

    def __init__(self, config: H2SharpaControllerHandsConfig, name: str) -> None:
        self._config = config
        self._closed_by_side = {"left": False, "right": False}
        self._invalid_samples_by_side = {"left": 0, "right": 0}
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        return {
            "controller_left": OptionalType(ControllerInput()),
            "controller_right": OptionalType(ControllerInput()),
        }

    def output_spec(self) -> RetargeterIOType:
        return {"hand_joints": RobotHandJoints("h2_sharpa_controller_hands", list(self._config.joint_names))}

    @staticmethod
    def _controller_grasp_value(controller) -> float | None:
        if controller.is_none or not bool(controller[ControllerInputIndex.GRIP_IS_VALID]):
            return None
        values = (
            float(controller[ControllerInputIndex.TRIGGER_VALUE]),
            float(controller[ControllerInputIndex.SQUEEZE_VALUE]),
        )
        if not all(np.isfinite(value) for value in values):
            return None
        return float(np.clip(max(values), 0.0, 1.0))

    def _update_grasp_state(self, side: str, controller) -> None:
        grasp_value = self._controller_grasp_value(controller)
        if grasp_value is None:
            invalid_samples = min(
                self._invalid_samples_by_side[side] + 1,
                self._config.tracking_loss_hold_samples + 1,
            )
            self._invalid_samples_by_side[side] = invalid_samples
            if invalid_samples > self._config.tracking_loss_hold_samples:
                self._closed_by_side[side] = False
            return
        self._invalid_samples_by_side[side] = 0
        if self._closed_by_side[side]:
            if grasp_value <= self._config.open_threshold:
                self._closed_by_side[side] = False
        elif grasp_value >= self._config.close_threshold:
            self._closed_by_side[side] = True

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        if context.execution_events.reset:
            self._closed_by_side = {"left": False, "right": False}
            self._invalid_samples_by_side = {"left": 0, "right": 0}
        else:
            self._update_grasp_state("left", inputs["controller_left"])
            self._update_grasp_state("right", inputs["controller_right"])

        for index, joint_name in enumerate(self._config.joint_names):
            side = joint_name.split("_", 1)[0]
            targets = (
                self._config.closed_joint_pos_by_name
                if self._closed_by_side[side]
                else self._config.open_joint_pos_by_name
            )
            outputs["hand_joints"][index] = targets[joint_name]


class H2TrackingSafeSe3AbsRetargeter(Se3AbsRetargeter):
    """Emit NaNs on pose-tracking loss so the recorder discards the take."""

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        controller = inputs[self._config.input_device]
        if controller.is_none or not bool(controller[ControllerInputIndex.GRIP_IS_VALID]):
            outputs["ee_pose"][0] = np.full((7,), np.nan, dtype=np.float32)
            return
        super()._compute_fn(inputs, outputs, context)


__all__ = [
    "H2SharpaControllerHandsConfig",
    "H2SharpaControllerHandsRetargeter",
    "H2TrackingSafeSe3AbsRetargeter",
    "H2VelocityCommandConfig",
    "H2VelocityCommandRetargeter",
]

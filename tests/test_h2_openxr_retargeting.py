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

"""Behavioral checks for H2 PICO/OpenXR retargeting."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("isaacteleop", reason="Isaac Teleop is not installed in the CPU-only test environment")

from isaacteleop.retargeters import Se3RetargeterConfig
from isaacteleop.retargeting_engine.interface import ComputeContext, ExecutionEvents
from isaacteleop.retargeting_engine.interface.tensor_group import OptionalTensorGroup
from isaacteleop.retargeting_engine.tensor_types import ControllerInput, ControllerInputIndex

from e2e_robot_workflows.devices.h2_motion_controller import (
    H2SharpaControllerHandsConfig,
    H2SharpaControllerHandsRetargeter,
    H2TrackingSafeSe3AbsRetargeter,
    H2VelocityCommandConfig,
    H2VelocityCommandRetargeter,
)
from e2e_robot_workflows.robots.h2 import load_h2_pink_contract


def _controller(
    *,
    stick_x: float = 0.0,
    stick_y: float = 0.0,
    trigger: float = 0.0,
    squeeze: float = 0.0,
    valid: bool = True,
):
    group = OptionalTensorGroup(ControllerInput())
    group[ControllerInputIndex.GRIP_IS_VALID] = valid
    group[ControllerInputIndex.THUMBSTICK_X] = stick_x
    group[ControllerInputIndex.THUMBSTICK_Y] = stick_y
    group[ControllerInputIndex.TRIGGER_VALUE] = trigger
    group[ControllerInputIndex.SQUEEZE_VALUE] = squeeze
    return group


def _hands_retargeter(contract) -> H2SharpaControllerHandsRetargeter:
    return H2SharpaControllerHandsRetargeter(
        H2SharpaControllerHandsConfig(
            joint_names=contract.hand_joint_order,
            open_joint_pos_by_name=contract.hand_open_targets,
            closed_joint_pos_by_name=contract.hand_closed_targets,
            tracking_loss_hold_samples=5,
        ),
        name="hands",
    )


def _assert_hand_targets(contract, output, *, left_closed: bool, right_closed: bool) -> None:
    states = {"left": left_closed, "right": right_closed}
    for index, joint_name in enumerate(contract.hand_joint_order):
        target_by_name = (
            contract.hand_closed_targets if states[joint_name.split("_", 1)[0]] else contract.hand_open_targets
        )
        assert float(output[index]) == pytest.approx(target_by_name[joint_name])


def test_h2_velocity_maps_each_stick_axis_and_zeros_invalid_tracking() -> None:
    retargeter = H2VelocityCommandRetargeter(
        H2VelocityCommandConfig(movement_scale=0.5, rotation_scale=0.25), name="velocity"
    )
    output = retargeter(
        {
            "controller_left": _controller(stick_x=0.4, stick_y=-0.8),
            "controller_right": _controller(stick_x=0.6),
        }
    )["velocity"][0]
    assert np.allclose(output, (-7 / 18, -1 / 6, -5 / 36))

    invalid = retargeter(
        {
            "controller_left": _controller(stick_x=1.0, stick_y=1.0, valid=False),
            "controller_right": _controller(stick_x=1.0, valid=False),
        }
    )["velocity"][0]
    assert np.array_equal(invalid, np.zeros(3, dtype=np.float32))


@pytest.mark.parametrize("stick", [0.0, 0.05, -0.05, 0.1, -0.1, np.nan, np.inf])
def test_h2_neutral_and_malformed_sticks_do_not_request_navigation(stick: float) -> None:
    retargeter = H2VelocityCommandRetargeter(H2VelocityCommandConfig(), name="velocity")
    output = retargeter(
        {
            "controller_left": _controller(stick_x=stick, stick_y=stick),
            "controller_right": _controller(stick_x=stick),
        }
    )["velocity"][0]
    assert np.array_equal(output, np.zeros(3, dtype=np.float32))


@pytest.mark.parametrize("deadzone", [-0.1, 1.0, np.nan, np.inf, True])
def test_h2_velocity_rejects_invalid_deadzone(deadzone: float) -> None:
    with pytest.raises(ValueError, match="deadzone"):
        H2VelocityCommandConfig(thumbstick_deadzone=deadzone)


def test_h2_velocity_deadzone_is_continuous_and_preserves_full_scale() -> None:
    retargeter = H2VelocityCommandRetargeter(H2VelocityCommandConfig(), name="velocity")
    just_outside = retargeter({"controller_left": _controller(stick_y=0.1001), "controller_right": _controller()})[
        "velocity"
    ][0].copy()
    full_scale = retargeter(
        {"controller_left": _controller(stick_y=1.0), "controller_right": _controller(stick_x=-1.0)}
    )["velocity"][0]
    assert 0 < just_outside[0] < 0.001
    assert np.allclose(full_scale, (0.5, 0.0, 0.35))


def test_h2_hands_are_binary_independent_hysteretic_and_fail_open() -> None:
    contract = load_h2_pink_contract()
    retargeter = _hands_retargeter(contract)

    initial = retargeter(
        {
            "controller_left": _controller(trigger=0.8),
            "controller_right": _controller(squeeze=0.2),
        }
    )["hand_joints"]
    _assert_hand_targets(contract, initial, left_closed=True, right_closed=False)

    malformed_grip = retargeter(
        {
            "controller_left": _controller(trigger=np.nan),
            "controller_right": _controller(squeeze=0.8),
        }
    )["hand_joints"]
    _assert_hand_targets(contract, malformed_grip, left_closed=True, right_closed=True)

    hysteresis_band = retargeter(
        {
            "controller_left": _controller(trigger=0.45),
            "controller_right": _controller(squeeze=0.45),
        }
    )["hand_joints"]
    _assert_hand_targets(contract, hysteresis_band, left_closed=True, right_closed=True)

    independent_release = retargeter(
        {
            "controller_left": _controller(trigger=0.2),
            "controller_right": _controller(squeeze=np.nan),
        }
    )["hand_joints"]
    _assert_hand_targets(contract, independent_release, left_closed=False, right_closed=True)

    reset = retargeter(
        {
            "controller_left": _controller(trigger=1.0),
            "controller_right": _controller(squeeze=1.0),
        },
        context=ComputeContext(execution_events=ExecutionEvents(reset=True)),
    )["hand_joints"]
    _assert_hand_targets(contract, reset, left_closed=False, right_closed=False)


def test_h2_hands_open_independently_after_five_malformed_grip_samples() -> None:
    contract = load_h2_pink_contract()
    retargeter = _hands_retargeter(contract)

    both_closed = retargeter(
        {
            "controller_left": _controller(trigger=1.0),
            "controller_right": _controller(squeeze=1.0),
        }
    )["hand_joints"]
    _assert_hand_targets(contract, both_closed, left_closed=True, right_closed=True)

    for _ in range(5):
        held = retargeter(
            {
                "controller_left": _controller(trigger=np.nan),
                "controller_right": _controller(squeeze=1.0),
            }
        )["hand_joints"]
        _assert_hand_targets(contract, held, left_closed=True, right_closed=True)

    expired = retargeter(
        {
            "controller_left": _controller(trigger=np.nan),
            "controller_right": _controller(squeeze=1.0),
        }
    )["hand_joints"]
    _assert_hand_targets(contract, expired, left_closed=False, right_closed=True)

    reset = retargeter(
        {
            "controller_left": _controller(trigger=1.0),
            "controller_right": _controller(squeeze=1.0),
        },
        context=ComputeContext(execution_events=ExecutionEvents(reset=True)),
    )["hand_joints"]
    _assert_hand_targets(contract, reset, left_closed=False, right_closed=False)

    retargeter(
        {
            "controller_left": _controller(trigger=1.0),
            "controller_right": _controller(),
        }
    )
    first_invalid_after_reset = retargeter(
        {
            "controller_left": _controller(trigger=np.nan),
            "controller_right": _controller(),
        }
    )["hand_joints"]
    _assert_hand_targets(contract, first_invalid_after_reset, left_closed=True, right_closed=False)


def test_h2_pose_tracking_loss_is_non_finite() -> None:
    retargeter = H2TrackingSafeSe3AbsRetargeter(
        Se3RetargeterConfig(
            input_device="controller_left",
            zero_out_xy_rotation=False,
            target_offset_roll=0.0,
            target_offset_pitch=0.0,
            target_offset_yaw=0.0,
        ),
        name="left_flange",
    )
    tracked = _controller()
    tracked[ControllerInputIndex.GRIP_POSITION] = np.asarray((0.2, 0.3, 1.0), dtype=np.float32)
    tracked[ControllerInputIndex.GRIP_ORIENTATION] = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
    assert np.allclose(
        retargeter({"controller_left": tracked})["ee_pose"][0],
        (0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    assert np.isnan(retargeter({"controller_left": _controller(valid=False)})["ee_pose"][0]).all()

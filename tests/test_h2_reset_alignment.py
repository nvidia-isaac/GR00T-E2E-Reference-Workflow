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

"""CPU regression tests for H2 XR reset capture and bounded pose targets."""

from __future__ import annotations

import math

import numpy as np
import pytest

from e2e_robot_workflows.devices.h2_reset_alignment import H2ResetAlignment, H2ResetAlignmentConfig

torch = pytest.importorskip("torch")
Rotation = pytest.importorskip("scipy.spatial.transform").Rotation


def _action(*, dtype=torch.float64):
    action = torch.linspace(-0.5, 0.5, 61, dtype=dtype)
    action[:3] = torch.tensor((0.3, 0.2, 0.1), dtype=dtype)
    action[7:10] = torch.tensor((0.2, -0.3, -0.2), dtype=dtype)
    action[3:7] = torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=dtype)
    action[10:14] = torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=dtype)
    return action


def _reference(*, dtype=torch.float64):
    action = _action(dtype=dtype)
    action[:3] = torch.tensor((0.45, 0.3, 0.18), dtype=dtype)
    action[7:10] = torch.tensor((0.45, -0.3, 0.18), dtype=dtype)
    action[3:7] = torch.tensor(Rotation.from_euler("xyz", (0.3, -0.5, 0.2)).as_quat(), dtype=dtype)
    action[10:14] = torch.tensor(Rotation.from_euler("xyz", (-0.4, 0.2, -0.6)).as_quat(), dtype=dtype)
    action[14:] = 0.0
    return action


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_first_sample_holds_measured_poses_preserves_suffix_and_does_not_alias(dtype) -> None:
    helper = H2ResetAlignment()
    raw = _action(dtype=dtype)
    reference = _reference(dtype=dtype)
    original_raw, original_reference = raw.clone(), reference.clone()
    assert not helper.is_aligned

    result = helper.apply(raw, reference, dt=0.01)

    assert helper.is_aligned
    assert result.dtype == raw.dtype
    assert result.device == raw.device
    assert result.shape == (61,)
    torch.testing.assert_close(result[:14], reference[:14])
    assert torch.equal(result[14:], raw[14:])
    assert torch.equal(raw, original_raw)
    assert torch.equal(reference, original_reference)
    result[:] = 0.0
    raw[0] += 0.5
    reference[0] += 0.5
    held = helper.apply(original_raw, dt=0.01)
    torch.testing.assert_close(held[:14], original_reference[:14])


def test_stationary_input_and_alternating_quaternion_signs_do_not_move_targets() -> None:
    helper = H2ResetAlignment()
    raw, reference = _action(), _reference()
    raw[3:7] = torch.tensor(Rotation.from_euler("xyz", (0.7, -0.4, 0.9)).as_quat())
    raw[10:14] = torch.tensor(Rotation.from_euler("xyz", (-0.3, 0.5, 0.8)).as_quat())
    reference[3:7] *= -1.0
    first = helper.apply(raw, reference, dt=0.01)
    for _ in range(20):
        raw[3:7] *= -1.0
        raw[10:14] *= -1.0
        actual = helper.apply(raw, dt=0.01)
        torch.testing.assert_close(actual[:14], first[:14], atol=1e-12, rtol=0.0)
        assert torch.equal(actual[14:], raw[14:])


def test_position_deltas_keep_pelvis_axes_despite_nontrivial_orientation_alignment() -> None:
    helper = H2ResetAlignment()
    raw, reference = _action(), _reference()
    raw[3:7] = torch.tensor(Rotation.from_euler("z", math.pi / 2.0).as_quat())
    raw[10:14] = torch.tensor(Rotation.from_euler("y", math.pi / 2.0).as_quat())
    helper.apply(raw, reference, dt=0.01)
    shifted = raw.clone()
    shifted[:3] += torch.tensor((0.03, -0.02, 0.01))
    shifted[7:10] += torch.tensor((-0.04, 0.01, -0.02))

    result = helper.apply(shifted, dt=1.0)

    torch.testing.assert_close(result[:3], reference[:3] + shifted[:3] - raw[:3])
    torch.testing.assert_close(result[7:10], reference[7:10] + shifted[7:10] - raw[7:10])
    torch.testing.assert_close(result[3:7], reference[3:7])
    torch.testing.assert_close(result[10:14], reference[10:14])


def test_noncommuting_controller_rotation_delta_is_applied_in_pelvis_axes() -> None:
    helper = H2ResetAlignment()
    raw, reference = _action(), _reference()
    start = Rotation.from_euler("xyz", (0.7, -0.3, 0.4))
    raw[3:7] = torch.tensor(start.as_quat())
    first = helper.apply(raw, reference, dt=0.01)
    pelvis_delta = Rotation.from_euler("xyz", (0.2, 0.4, -0.3))
    rotated = raw.clone()
    rotated[3:7] = torch.tensor((pelvis_delta * start).as_quat())

    result = helper.apply(rotated, dt=1.0)

    expected = pelvis_delta.as_matrix() @ Rotation.from_quat(reference[3:7]).as_matrix()
    np.testing.assert_allclose(Rotation.from_quat(result[3:7]).as_matrix(), expected, atol=1e-12)
    torch.testing.assert_close(result[7:14], first[7:14])
    torch.testing.assert_close(result[:3], first[:3])
    assert torch.equal(result[14:], raw[14:])


@pytest.mark.parametrize("dt", (0.01, 0.025))
def test_independent_hand_translation_and_angular_slew_bounds_and_eventual_arrival(dt: float) -> None:
    config = H2ResetAlignmentConfig()
    helper = H2ResetAlignment(config)
    raw, reference = _action(), _reference()
    previous = helper.apply(raw, reference, dt=dt)
    jumped = raw.clone()
    jumped[:3] += torch.tensor((0.3, 0.4, 0.0))
    jumped[7:10] += torch.tensor((0.0, 0.0, -0.25))
    rotations = (Rotation.from_rotvec((0.0, 0.0, 1.5)), Rotation.from_rotvec((0.0, -0.8, 0.0)))
    for offset, rotation in zip((0, 7), rotations, strict=True):
        jumped[offset + 3 : offset + 7] = torch.tensor(rotation.as_quat())
    for step in range(math.ceil(1.0 / dt)):
        result = helper.apply(jumped, dt=dt)
        for offset in (0, 7):
            distance = torch.linalg.vector_norm(result[offset : offset + 3] - previous[offset : offset + 3]).item()
            angle = (
                Rotation.from_quat(result[offset + 3 : offset + 7])
                * Rotation.from_quat(previous[offset + 3 : offset + 7]).inv()
            ).magnitude()
            assert distance <= config.max_linear_speed * dt + 1e-12
            assert angle <= config.max_angular_speed * dt + 1e-12
            if step == 0:
                assert distance == pytest.approx(config.max_linear_speed * dt)
                assert angle == pytest.approx(config.max_angular_speed * dt)
        assert torch.equal(result[14:], raw[14:])
        previous = result
    for offset, rotation in zip((0, 7), rotations, strict=True):
        expected_position = reference[offset : offset + 3] + jumped[offset : offset + 3] - raw[offset : offset + 3]
        torch.testing.assert_close(result[offset : offset + 3], expected_position)
        expected_rotation = rotation * Rotation.from_quat(reference[offset + 3 : offset + 7])
        np.testing.assert_allclose(
            Rotation.from_quat(result[offset + 3 : offset + 7]).as_matrix(),
            expected_rotation.as_matrix(),
            atol=1e-12,
        )


def test_angular_slew_uses_shortest_arc_across_quaternion_sign_boundary() -> None:
    helper = H2ResetAlignment()
    raw = _action()
    reference = _reference()
    reference[3:7] = torch.tensor(Rotation.from_euler("z", math.radians(179.0)).as_quat())
    first = helper.apply(raw, reference, dt=0.01)
    raw[3:7] = torch.tensor(-Rotation.from_euler("z", math.radians(2.0)).as_quat())

    result = helper.apply(raw, dt=0.1)

    expected = Rotation.from_euler("z", math.radians(181.0))
    np.testing.assert_allclose(Rotation.from_quat(result[3:7]).as_matrix(), expected.as_matrix(), atol=1e-12)
    assert torch.dot(result[3:7], first[3:7]).item() > 0.0


def test_reset_discards_both_capture_and_in_progress_slew() -> None:
    helper = H2ResetAlignment()
    raw, reference = _action(), _reference()
    helper.apply(raw, reference, dt=0.01)
    raw[:3] += 1.0
    helper.apply(raw, dt=0.01)
    helper.reset()
    assert not helper.is_aligned
    with pytest.raises(ValueError, match="reference action"):
        helper.apply(raw, dt=0.01)
    new_reference = _reference()
    new_reference[:3] += 0.2
    new_reference[7:10] -= 0.3
    new_reference[10:14] *= -1.0

    recaptured = helper.apply(raw, new_reference, dt=0.01)

    torch.testing.assert_close(recaptured[:14], new_reference[:14])
    torch.testing.assert_close(helper.apply(raw, dt=0.01)[:14], new_reference[:14])
    assert torch.equal(recaptured[14:], raw[14:])


def test_nonunit_xyzw_quaternions_are_normalized_without_mutating_inputs() -> None:
    helper = H2ResetAlignment()
    raw, reference = _action(), _reference()
    raw[3:7] *= 3.0
    raw[10:14] *= 2.0
    reference[3:7] *= 4.0
    reference[10:14] *= 5.0
    original_raw, original_reference = raw.clone(), reference.clone()

    result = helper.apply(raw, reference, dt=0.01)

    for offset in (0, 7):
        assert torch.linalg.vector_norm(result[offset + 3 : offset + 7]).item() == pytest.approx(1.0)
        np.testing.assert_allclose(
            Rotation.from_quat(result[offset + 3 : offset + 7]).as_matrix(),
            Rotation.from_quat(reference[offset + 3 : offset + 7]).as_matrix(),
            atol=1e-12,
        )
    assert torch.equal(raw, original_raw)
    assert torch.equal(reference, original_reference)


@pytest.mark.parametrize("target", ("action", "reference"))
@pytest.mark.parametrize("index", (0, 3, 7, 10, 14, 35, 58, 60))
@pytest.mark.parametrize("invalid", (np.nan, np.inf, -np.inf))
def test_nonfinite_inputs_do_not_capture(target: str, index: int, invalid: float) -> None:
    helper = H2ResetAlignment()
    raw, reference = _action(), _reference()
    (raw if target == "action" else reference)[index] = invalid
    with pytest.raises(ValueError, match="finite"):
        helper.apply(raw, reference, dt=0.01)
    assert not helper.is_aligned


@pytest.mark.parametrize("target", ("action", "reference"))
@pytest.mark.parametrize("bad_input", (None, [0.0] * 61, torch.zeros(60), torch.zeros((1, 61)), torch.zeros(61).long()))
def test_invalid_action_layout_or_type_does_not_capture(target, bad_input) -> None:
    helper = H2ResetAlignment()
    raw, reference = _action(), _reference()
    with pytest.raises(ValueError, match="shape"):
        helper.apply(
            bad_input if target == "action" else raw, bad_input if target == "reference" else reference, dt=0.01
        )
    assert not helper.is_aligned


@pytest.mark.parametrize("target", ("action", "reference"))
@pytest.mark.parametrize("offset", (3, 10))
def test_zero_quaternions_do_not_capture(target: str, offset: int) -> None:
    helper = H2ResetAlignment()
    raw, reference = _action(), _reference()
    (raw if target == "action" else reference)[offset : offset + 4] = 0.0
    with pytest.raises(ValueError, match="XYZW"):
        helper.apply(raw, reference, dt=0.01)
    assert not helper.is_aligned


def test_rejected_sample_does_not_advance_existing_slew_state() -> None:
    helper, uninterrupted = H2ResetAlignment(), H2ResetAlignment()
    raw, reference = _action(), _reference()
    for instance in (helper, uninterrupted):
        instance.apply(raw, reference, dt=0.01)
    jumped = raw.clone()
    jumped[:3] += 0.5
    invalid = jumped.clone()
    invalid[60] = np.nan
    with pytest.raises(ValueError, match="finite"):
        helper.apply(invalid, dt=0.01)
    assert helper.is_aligned
    torch.testing.assert_close(helper.apply(jumped, dt=0.01), uninterrupted.apply(jumped, dt=0.01))


@pytest.mark.parametrize("dt", (0.0, -0.01, np.nan, np.inf, True, None))
def test_invalid_control_period_does_not_capture(dt) -> None:
    helper = H2ResetAlignment()
    with pytest.raises(ValueError, match="control dt"):
        helper.apply(_action(), _reference(), dt=dt)
    assert not helper.is_aligned


@pytest.mark.parametrize("name", ("max_linear_speed", "max_angular_speed"))
@pytest.mark.parametrize("value", (0.0, -1.0, np.nan, np.inf, True, None))
def test_invalid_speed_limits_are_rejected(name, value) -> None:
    with pytest.raises(ValueError, match="speed limits"):
        H2ResetAlignmentConfig(**{name: value})

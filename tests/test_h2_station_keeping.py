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

"""CPU checks for explicit H2 XR station-keeping commands."""

from __future__ import annotations

import math

import numpy as np
import pytest

from e2e_robot_workflows.devices.h2_station_keeping import H2StationKeeping, H2StationKeepingConfig


def _quaternion(yaw: float) -> np.ndarray:
    return np.asarray((0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)))


def test_station_keeping_latches_without_moving_and_does_not_alias_inputs() -> None:
    helper = H2StationKeeping()
    position = np.asarray((2.0, 3.0))
    requested = np.zeros(3, dtype=np.float32)
    result = helper.command(requested, position, _quaternion(0.0))
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, requested)
    position[0] += 0.1
    corrected = helper.command(requested, position, _quaternion(0.0))
    np.testing.assert_allclose(corrected, (-0.1, 0.0, 0.0), atol=1e-7)
    np.testing.assert_array_equal(position, (2.1, 3.0))
    np.testing.assert_array_equal(requested, (0.0, 0.0, 0.0))


def test_station_keeping_rotates_world_error_into_current_body_frame() -> None:
    helper = H2StationKeeping()
    quarter_turn = _quaternion(math.pi / 2.0)
    helper.command(np.zeros(3), (0.0, 0.0), quarter_turn)
    corrected = helper.command(np.zeros(3), (0.1, 0.0), quarter_turn)
    np.testing.assert_allclose(corrected, (0.0, 0.1, 0.0), atol=1e-7)


def test_station_keeping_caps_planar_norm_and_yaw_and_uses_shortest_yaw_arc() -> None:
    helper = H2StationKeeping()
    helper.command(np.zeros(3), (0.0, 0.0), _quaternion(0.0))
    corrected = helper.command(np.zeros(3), (10.0, 10.0), _quaternion(math.pi / 2.0))
    assert np.linalg.norm(corrected[:2]) == pytest.approx(0.1)
    assert corrected[2] == pytest.approx(-0.15)

    helper.reset()
    helper.command(np.zeros(3), (0.0, 0.0), _quaternion(math.radians(179.0)))
    corrected = helper.command(np.zeros(3), (0.0, 0.0), _quaternion(math.radians(-179.0)))
    assert corrected[2] == pytest.approx(-0.6 * (math.radians(2.0) - 0.01))


def test_station_keeping_has_neutral_deadbands_and_accepts_scaled_quaternions() -> None:
    helper = H2StationKeeping()
    helper.command(np.zeros(3), (0.0, 0.0), _quaternion(0.0))
    corrected = helper.command(np.zeros(3), (0.002, -0.002), 2.0 * _quaternion(0.005))
    np.testing.assert_array_equal(corrected, (0.0, 0.0, 0.0))


@pytest.mark.parametrize("requested", [(0.2, 0.0, 0.0), (0.0, 0.0, -0.3), (1e-6, 0.0, 0.0)])
def test_station_keeping_yields_to_navigation_and_relatches_at_new_stop(requested) -> None:
    helper = H2StationKeeping()
    helper.command(np.zeros(3), (0.0, 0.0), _quaternion(0.0))
    actual = helper.command(requested, (0.2, 0.3), _quaternion(0.5))
    np.testing.assert_array_equal(actual, np.asarray(requested, dtype=np.float32))
    stopped = helper.command(np.zeros(3), (1.0, 2.0), _quaternion(0.8))
    np.testing.assert_array_equal(stopped, (0.0, 0.0, 0.0))
    helper.reset()
    reset = helper.command(np.zeros(3), (10.0, 20.0), _quaternion(-0.7))
    np.testing.assert_array_equal(reset, (0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    ("requested", "position", "quaternion"),
    [
        ((np.nan, 0.0, 0.0), (0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ((0.6, 0.0, 0.0), (0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (np.inf, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0), (0.0, np.nan, 0.0, 1.0)),
    ],
)
def test_station_keeping_rejects_invalid_samples_and_clears_reference(requested, position, quaternion) -> None:
    helper = H2StationKeeping()
    helper.command(np.zeros(3), (0.0, 0.0), _quaternion(0.0))
    with pytest.raises(ValueError):
        helper.command(requested, position, quaternion)
    fresh = helper.command(np.zeros(3), (2.0, 3.0), _quaternion(0.5))
    np.testing.assert_array_equal(fresh, (0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    "overrides",
    [
        {"position_gain": 0.0},
        {"yaw_gain": np.inf},
        {"position_deadband_m": -0.1},
        {"yaw_deadband_rad": np.nan},
        {"max_linear_speed": 0.6},
        {"max_yaw_speed": 0.6},
        {"navigate_command_limits": ((-0.5, 0.5),)},
        {"navigate_command_limits": ((0.1, 0.5),) * 3},
    ],
)
def test_station_keeping_rejects_invalid_or_out_of_envelope_configuration(overrides) -> None:
    with pytest.raises(ValueError):
        H2StationKeepingConfig(**overrides)


@pytest.mark.parametrize("batched", (False, True))
def test_station_keeping_writes_effective_action_tail_without_mutating_pose_or_grip(batched: bool) -> None:
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from e2e_robot_workflows.devices.h2_station_keeping import apply_h2_station_keeping

    data = SimpleNamespace(
        root_link_pos_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 1.0]])),
        root_link_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]])),
    )
    helper = H2StationKeeping()
    action = torch.arange(61, dtype=torch.float32)
    action[-3:] = 0.0
    if batched:
        action = action.unsqueeze(0)
    original = action.clone()
    apply_h2_station_keeping(action, data, helper)
    data.root_link_pos_w.torch[0, 0] = 0.1

    effective = apply_h2_station_keeping(action, data, helper)

    assert effective.shape == action.shape
    assert torch.equal(action, original)
    assert torch.equal(effective.reshape(61)[:58], original.reshape(61)[:58])
    assert effective.reshape(61)[-3:].tolist() == pytest.approx((-0.1, 0.0, 0.0))

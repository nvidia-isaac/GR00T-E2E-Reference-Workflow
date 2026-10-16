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

"""Host checks for the numeric conventions in H2 teleop GPU qualification."""

from __future__ import annotations

import importlib
import math
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def validation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools" / "simulation"))
    return importlib.import_module("h2_teleop_validation")


def test_teleop_validation_declares_every_live_control_step(validation) -> None:
    assert validation.H2_TELEOP_CONTROL_STEPS == 360
    assert 2 * validation._PHASE_STEPS == validation.H2_TELEOP_CONTROL_STEPS
    assert validation._PHASE_STEPS == 180
    assert validation._MOTION_START == 120
    assert validation._MOTION_END == 150
    assert 0 < validation._MOTION_START < validation._MOTION_END < validation._PHASE_STEPS


def test_teleop_validation_rotation_distance_is_xyzw_and_sign_invariant(validation) -> None:
    pytest.importorskip("scipy")
    identity = np.array((0.0, 0.0, 0.0, 1.0))
    assert validation._angular_distance(identity, -identity) == pytest.approx(0.0)
    quarter_turn = np.array((0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)))
    assert validation._angular_distance(identity, quarter_turn) == pytest.approx(math.pi / 2.0)
    assert validation._angular_distance(identity, -quarter_turn) == pytest.approx(math.pi / 2.0)


def test_teleop_validation_rotation_distance_uses_the_short_arc(validation) -> None:
    pytest.importorskip("scipy")
    angle = math.radians(179.0)
    first = (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0))
    second = (0.0, 0.0, -math.sin(angle / 2.0), math.cos(angle / 2.0))
    assert validation._angular_distance(first, second) == pytest.approx(math.radians(2.0))


def test_first_aligned_pose_accepts_equivalent_nonunit_reference_quaternions(validation) -> None:
    pytest.importorskip("scipy")
    reference = np.array(((0.3, 0.2, 0.1, 0.0, 0.0, 0.6, 0.8), (0.4, -0.2, 0.2, 0.0, 0.6, 0.0, 0.8)))
    aligned = reference.copy()
    reference[0, 3:] *= 1.000005
    reference[1, 3:] *= -1.000004
    metrics = validation._first_alignment_metrics(aligned, reference)
    assert metrics["first_pose_max_error"] > 1e-6
    assert metrics["reference_quaternion_norms"] == pytest.approx((1.000005, 1.000004))
    assert metrics["first_position_errors_m"] == pytest.approx((0.0, 0.0))
    assert metrics["first_rotation_errors_rad"] == pytest.approx((0.0, 0.0), abs=1e-12)


def test_first_aligned_pose_preserves_sensitivity_to_physical_position_and_rotation_errors(validation) -> None:
    pytest.importorskip("scipy")
    reference = np.array(((0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 1.0),) * 2)
    aligned = reference.copy()
    aligned[0, 0] += 2e-6
    aligned[1, 3:] = (0.0, 0.0, math.sin(3e-6 / 2.0), math.cos(3e-6 / 2.0))
    metrics = validation._first_alignment_metrics(aligned, reference)
    assert metrics["first_position_errors_m"] == pytest.approx((2e-6, 0.0))
    assert metrics["first_rotation_errors_rad"] == pytest.approx((0.0, 3e-6))

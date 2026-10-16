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

"""Behavioral H2 training/evaluation joint-map tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from e2e_robot_workflows.gr00t.h2_mapping import (
    build_h2_gr00t_action_np,
    validate_h2_action_schedule,
    validate_h2_joint_configs,
    validate_h2_live_joint_limits,
)
from e2e_robot_workflows.robots.h2 import load_h2_pink_contract

REPO_ROOT = Path(__file__).resolve().parents[1]


def _joints(relative_path: str):
    return yaml.safe_load((REPO_ROOT / relative_path).read_text(encoding="utf-8"))["joints"]


def _wide_joint_limits(action_joints: dict[str, int]) -> dict[str, tuple[float, float]]:
    return dict.fromkeys(action_joints, (-100.0, 100.0))


@pytest.mark.parametrize(
    ("action_horizon", "action_chunk_length"),
    ((39, 40), (40, 1), (40, 39), (40, 41), (41, 40), (40.0, 40), (40, 40.0), (True, 40)),
)
def test_h2_evaluation_rejects_any_non_training_action_schedule(
    action_horizon: int,
    action_chunk_length: int,
) -> None:
    with pytest.raises(ValueError, match="40-step horizon and a 40-step chunk"):
        validate_h2_action_schedule(action_horizon, action_chunk_length)


def test_h2_evaluation_accepts_the_exact_training_action_schedule() -> None:
    validate_h2_action_schedule(40, 40)


def test_h2_training_and_evaluation_maps_cover_the_live_contract() -> None:
    policy = _joints("configs/stages/gr00t/gr00t_h2_75dof_joint_space.yaml")
    action = _joints("configs/stages/gr00t/h2_58dof_action_space.yaml")
    state = _joints("configs/stages/gr00t/h2_75dof_joint_space.yaml")

    validate_h2_joint_configs(policy, action, state)

    contract = load_h2_pink_contract()
    assert tuple(name for name, _ in sorted(state.items(), key=lambda item: item[1])) == contract.expected_joint_order
    assert tuple(name for name, _ in sorted(action.items(), key=lambda item: item[1])) == (
        contract.arm_joint_names + contract.hand_joint_order
    )


def test_h2_policy_groups_become_the_joint_embodiments_live_61_values() -> None:
    policy = _joints("configs/stages/gr00t/gr00t_h2_75dof_joint_space.yaml")
    action = _joints("configs/stages/gr00t/h2_58dof_action_space.yaml")
    state = _joints("configs/stages/gr00t/h2_75dof_joint_space.yaml")
    batch, horizon = 2, 4
    policy_action = {}
    for group in ("left_arm", "right_arm", "left_hand", "right_hand"):
        policy_action[group] = np.stack(
            [np.full((batch, horizon), state[joint] + 0.25, dtype=np.float32) for joint in policy[group]],
            axis=-1,
        )
    navigation = np.arange(batch * horizon * 3, dtype=np.float32).reshape(batch, horizon, 3)
    policy_action["navigate_command"] = navigation

    result = build_h2_gr00t_action_np(
        policy_action,
        policy,
        action,
        joint_position_limits=_wide_joint_limits(action),
        navigation_limits=((-100.0, 100.0),) * 3,
    )

    assert result.shape == (batch, horizon, 61)
    for joint, action_index in action.items():
        assert np.all(result[..., action_index] == state[joint] + 0.25)
    assert np.array_equal(result[..., 58:], navigation)


def test_h2_mapping_rejects_incomplete_or_non_finite_policy_outputs() -> None:
    policy = _joints("configs/stages/gr00t/gr00t_h2_75dof_joint_space.yaml")
    action = _joints("configs/stages/gr00t/h2_58dof_action_space.yaml")
    values = {
        group: np.zeros((1, 2, len(policy[group])), dtype=np.float32)
        for group in ("left_arm", "right_arm", "left_hand", "right_hand")
    }
    with pytest.raises(ValueError, match="navigate_command"):
        build_h2_gr00t_action_np(
            values,
            policy,
            action,
            joint_position_limits=_wide_joint_limits(action),
            navigation_limits=((-1.0, 1.0),) * 3,
        )

    values["navigate_command"] = np.zeros((1, 2, 3), dtype=np.float32)
    values["left_arm"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        build_h2_gr00t_action_np(
            values,
            policy,
            action,
            joint_position_limits=_wide_joint_limits(action),
            navigation_limits=((-1.0, 1.0),) * 3,
        )


@pytest.mark.parametrize("outside", ["joint", "navigation"])
def test_h2_mapping_rejects_policy_outputs_outside_the_runtime_envelope(outside: str) -> None:
    policy = _joints("configs/stages/gr00t/gr00t_h2_75dof_joint_space.yaml")
    action = _joints("configs/stages/gr00t/h2_58dof_action_space.yaml")
    values = {
        group: np.zeros((1, 2, len(policy[group])), dtype=np.float32)
        for group in ("left_arm", "right_arm", "left_hand", "right_hand")
    }
    values["navigate_command"] = np.zeros((1, 2, 3), dtype=np.float32)
    limits = _wide_joint_limits(action)
    expected = "live position limits"
    if outside == "joint":
        values["left_arm"][0, 0, 0] = 1.1
        limits[policy["left_arm"][0]] = (-1.0, 1.0)
    else:
        values["navigate_command"][0, 0, 2] = 0.6
        expected = "configured limits"

    with pytest.raises(ValueError, match=expected):
        build_h2_gr00t_action_np(
            values,
            policy,
            action,
            joint_position_limits=limits,
            navigation_limits=((-0.5, 0.5),) * 3,
        )


def test_h2_evaluation_rejects_live_limit_drift_from_the_pinned_sidecar() -> None:
    pinned = load_h2_pink_contract().learned_joint_limits
    live = dict(pinned)

    validate_h2_live_joint_limits(live, pinned)

    joint = next(iter(pinned))
    lower, upper = live[joint]
    live[joint] = (lower, upper + 2e-6)
    with pytest.raises(ValueError, match="drift from the pinned sidecar"):
        validate_h2_live_joint_limits(live, pinned)

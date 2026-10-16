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

"""Host-safe contracts for H2's secondary shoulder/elbow posture objective."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from e2e_robot_workflows.robots.h2 import load_h2_pink_contract, load_h2_robot_config

SOURCE = Path(__file__).resolve().parents[1] / "src/e2e_robot_workflows/arena/h2_embodiments.py"
POSTURE_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)


class LocalFrameTaskCfg(SimpleNamespace):
    pass


class NullSpacePostureTaskCfg(SimpleNamespace):
    pass


@pytest.fixture
def controller_cfg() -> SimpleNamespace:
    """Evaluate the production configuration without importing or booting Isaac."""

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    embodiment = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "H2AgilePinkEmbodiment"
    )
    calls = [
        node
        for node in ast.walk(embodiment)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "PinkIKControllerCfg"
    ]
    assert len(calls) == 1
    contract = load_h2_pink_contract()
    # URDF resolution is an external-asset check; these tests exercise the actual
    # task configuration expression using the checked-in contract's values.
    config_contract = SimpleNamespace(
        arm_joint_names=contract.arm_joint_names,
        flange_frames=contract.flange_frames,
        base_frame=contract.base_frame,
        hand_joint_order=contract.hand_joint_order,
        resolve_urdf=lambda: Path("h2-test.urdf"),
    )
    namespace = {
        "contract": config_contract,
        "PinkIKControllerCfg": SimpleNamespace,
        "DampingTaskCfg": SimpleNamespace,
        "LocalFrameTaskCfg": LocalFrameTaskCfg,
        "NullSpacePostureTaskCfg": NullSpacePostureTaskCfg,
    }
    return eval(compile(ast.Expression(calls[0]), str(SOURCE), "eval"), namespace)


def test_posture_follows_both_flange_tasks_without_adding_action_dimensions(controller_cfg) -> None:
    tasks = controller_cfg.variable_input_tasks

    assert [type(task) for task in tasks] == [LocalFrameTaskCfg, LocalFrameTaskCfg, NullSpacePostureTaskCfg]
    assert [task.frame for task in tasks[:2]] == list(load_h2_pink_contract().flange_frames)
    assert tasks[2].controlled_frames == [task.frame for task in tasks[:2]]
    # The pinned action counts frame tasks only; the existing pose + hand +
    # navigation contract remains 14 + 44 + 3, despite the third IK objective.
    frame_count = sum(isinstance(task, LocalFrameTaskCfg) for task in tasks)
    assert frame_count * 7 + controller_cfg.num_hand_joints + load_h2_robot_config().command_dim == 61


def test_posture_masks_shoulders_and_elbows_but_not_wrists(controller_cfg) -> None:
    posture = controller_cfg.variable_input_tasks[-1]

    assert tuple(posture.controlled_joints) == POSTURE_JOINTS
    assert set(posture.controlled_joints) < set(load_h2_pink_contract().arm_joint_names)
    # Pink's NullSpacePostureTask initializes its target from this robot reset
    # configuration, not the startup measurement or an all-zero joint vector.
    reset_positions = load_h2_robot_config().initial_joint_pos
    assert [reset_positions[name] for name in posture.controlled_joints] == pytest.approx(
        [-0.95, 0.05, 0.16, 0.7, -0.95, -0.05, -0.16, 0.7]
    )


def test_posture_weights_remain_secondary_and_existing_frame_objectives_unchanged(controller_cfg) -> None:
    left, right, posture = controller_cfg.variable_input_tasks

    assert (posture.cost, posture.gain, posture.lm_damping) == (0.1, 0.1, 0.0)
    for frame in (left, right):
        assert frame.base_link_frame_name == load_h2_pink_contract().base_frame
        assert (frame.position_cost, frame.orientation_cost, frame.lm_damping, frame.gain) == (8.0, 4.0, 5.0, 0.25)
        assert posture.cost < min(frame.position_cost, frame.orientation_cost)
        assert posture.gain < frame.gain
    assert len(controller_cfg.fixed_input_tasks) == 1
    assert controller_cfg.fixed_input_tasks[0].cost == 1e-3
    assert controller_cfg.fail_on_joint_limit_violation is True

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

"""Behavioral checks for the H2 AGILE ONNX action term."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

SOURCE = Path(__file__).resolve().parents[1] / "src/e2e_robot_workflows/controllers/h2_agile_action.py"
JOINT_NAMES = tuple(f"joint_{index}" for index in range(31))
DRIVEN_JOINTS = JOINT_NAMES[:14]
FEEDBACK_PORTS = (
    "last_actions",
    "base_ang_vel_history",
    "projected_gravity_history",
    "velocity_commands_history",
    "controlled_joint_pos_history",
    "controlled_joint_vel_history",
    "actions_history",
)


class FakeSession:
    input_names = (
        "root_link_quat_w",
        "root_ang_vel_b",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        *(f"{name}_in" for name in FEEDBACK_PORTS),
    )
    output_names = (
        "action_joint_pos",
        "action_joint_pos_kp_gains",
        "action_joint_pos_kd_gains",
        *(f"{name}_out" for name in FEEDBACK_PORTS),
    )

    def __init__(self, policy_path, *, providers) -> None:
        self.policy_path = policy_path
        self.providers = providers
        self.calls: list[tuple[list[str], dict[str, np.ndarray]]] = []

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in self.input_names]

    def get_outputs(self):
        return [SimpleNamespace(name=name) for name in self.output_names]

    def run(self, names, inputs):
        copied = {name: value.copy() for name, value in inputs.items()}
        self.calls.append((list(names), copied))
        step = np.float32(len(self.calls))
        values = {
            "action_joint_pos": np.full((1, 14), step, dtype=np.float32),
            **{
                f"{name}_out": np.full_like(inputs[f"{name}_in"], 10 * step + index)
                for index, name in enumerate(FEEDBACK_PORTS)
            },
        }
        return [values[name] for name in names]


class FakeAsset:
    joint_names = list(JOINT_NAMES)

    def __init__(self) -> None:
        self.data = SimpleNamespace(
            root_link_quat_w=SimpleNamespace(torch=torch.tensor([[1.0, 2.0, 3.0, 4.0]])),
            root_ang_vel_b=SimpleNamespace(torch=torch.tensor([[5.0, 6.0, 7.0]])),
            joint_pos=SimpleNamespace(torch=torch.arange(31, dtype=torch.float32).reshape(1, 31)),
            joint_vel=SimpleNamespace(torch=(100 + torch.arange(31, dtype=torch.float32)).reshape(1, 31)),
        )
        self.targets: list[tuple[torch.Tensor, tuple[int, ...]]] = []

    def find_joints(self, names, *, preserve_order):
        assert preserve_order is True
        resolved = list(names)
        return [self.joint_names.index(name) for name in resolved], resolved

    def set_joint_position_target_index(self, *, target, joint_ids) -> None:
        self.targets.append((target.clone(), tuple(joint_ids)))


@pytest.fixture
def agile_module(monkeypatch: pytest.MonkeyPatch):
    class ActionTerm:
        def __init__(self, cfg, env) -> None:
            self.cfg = cfg
            self.num_envs = env.num_envs
            self.device = env.device
            self._asset = env.asset

    class ActionTermCfg:
        pass

    modules = {
        "isaaclab": ModuleType("isaaclab"),
        "isaaclab.assets": ModuleType("isaaclab.assets"),
        "isaaclab.assets.articulation": ModuleType("isaaclab.assets.articulation"),
        "isaaclab.managers": ModuleType("isaaclab.managers"),
        "isaaclab.managers.action_manager": ModuleType("isaaclab.managers.action_manager"),
        "isaaclab.managers.manager_term_cfg": ModuleType("isaaclab.managers.manager_term_cfg"),
        "isaaclab.utils": ModuleType("isaaclab.utils"),
        "isaaclab.utils.configclass": ModuleType("isaaclab.utils.configclass"),
    }
    modules["isaaclab.assets.articulation"].Articulation = object
    modules["isaaclab.managers.action_manager"].ActionTerm = ActionTerm
    modules["isaaclab.managers.manager_term_cfg"].ActionTermCfg = ActionTermCfg
    modules["isaaclab.utils.configclass"].configclass = lambda cls: cls
    for name, module in modules.items():
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    fake_onnxruntime = ModuleType("onnxruntime")

    def session_factory(*args, **kwargs):
        fake_onnxruntime.session = FakeSession(*args, **kwargs)
        return fake_onnxruntime.session

    fake_onnxruntime.InferenceSession = session_factory
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)

    spec = importlib.util.spec_from_file_location("_test_h2_agile_action", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.test_onnxruntime = fake_onnxruntime
    return module


def _config():
    return SimpleNamespace(
        policy_path="policy.onnx",
        command_dim=3,
        history_length=5,
        driven_joint_names=DRIVEN_JOINTS,
        proprio_joint_names=JOINT_NAMES,
        expected_joint_order=JOINT_NAMES,
    )


def test_feedback_snapshot_is_complete_independent_and_atomic(agile_module) -> None:
    env = SimpleNamespace(num_envs=1, device=torch.device("cpu"), asset=FakeAsset())
    action = agile_module.H2AgileVelocityAction(_config(), env)
    action.process_actions(torch.zeros(1, 3))
    saved = action.get_feedback_state()
    action.reset()
    action.restore_feedback_state(saved)
    for name, value in saved.items():
        torch.testing.assert_close(action.get_feedback_state()[name], value)
    saved["last_actions"].zero_()
    assert action.get_feedback_state()["last_actions"].count_nonzero() > 0
    before = action.get_feedback_state()
    for invalid in ({}, {**before, "actions_history": torch.full((1, 5, 14), float("nan"))}):
        with pytest.raises(ValueError):
            action.restore_feedback_state(invalid)
        for name, value in before.items():
            torch.testing.assert_close(action.get_feedback_state()[name], value)


def test_agile_action_batches_observations_and_carries_recurrent_state(agile_module) -> None:
    asset = FakeAsset()
    env = SimpleNamespace(num_envs=1, device=torch.device("cpu"), asset=asset)
    action = agile_module.H2AgileVelocityAction(_config(), env)
    command = torch.tensor([[0.1, -0.2, 0.3]])

    action.process_actions(command)
    command.zero_()
    session = agile_module.test_onnxruntime.session
    requested, inputs = session.calls[0]

    assert session.providers == ["CPUExecutionProvider"]
    assert requested == ["action_joint_pos", *(f"{name}_out" for name in FEEDBACK_PORTS)]
    assert inputs["root_link_quat_w"].tolist() == [[4.0, 1.0, 2.0, 3.0]]
    assert inputs["root_ang_vel_b"].tolist() == [[5.0, 6.0, 7.0]]
    np.testing.assert_allclose(inputs["velocity_commands"], [[0.1, -0.2, 0.3]])
    assert inputs["joint_pos"].tolist() == [list(range(31))]
    assert inputs["joint_vel"].tolist() == [list(range(100, 131))]
    assert all(not np.count_nonzero(inputs[f"{name}_in"]) for name in FEEDBACK_PORTS)
    assert action.locomotion_command is not None
    assert torch.allclose(action.locomotion_command, torch.tensor([[0.1, -0.2, 0.3]]))
    assert action.processed_actions.tolist() == [[1.0] * 14]

    action.process_actions(torch.tensor([[0.4, 0.5, 0.6]]))
    _, recurrent_inputs = session.calls[1]
    for index, name in enumerate(FEEDBACK_PORTS):
        assert np.all(recurrent_inputs[f"{name}_in"] == 10.0 + index)
    assert action.processed_actions.tolist() == [[2.0] * 14]
    action.apply_actions()
    assert len(asset.targets) == 1
    assert torch.equal(asset.targets[0][0], torch.full((1, 14), 2.0))
    assert asset.targets[0][1] == tuple(range(14))

    action.reset()
    assert action.locomotion_command is None
    action.process_actions(torch.zeros((1, 3)))
    _, reset_inputs = session.calls[2]
    assert all(not np.count_nonzero(reset_inputs[f"{name}_in"]) for name in FEEDBACK_PORTS)


def test_agile_action_rejects_a_static_batch_export_for_multiple_environments(agile_module) -> None:
    env = SimpleNamespace(num_envs=2, device=torch.device("cpu"), asset=FakeAsset())

    with pytest.raises(ValueError, match="batch dimension of 1"):
        agile_module.H2AgileVelocityAction(_config(), env)

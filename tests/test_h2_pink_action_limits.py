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

"""Behavioral checks for H2 Pink's numerical joint-limit guard."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from e2e_robot_workflows.controllers.errors import H2MeasuredJointLimitViolation

torch = pytest.importorskip("torch")

SOURCE = Path(__file__).resolve().parents[1] / "src/e2e_robot_workflows/controllers/h2_pink_action.py"


@pytest.fixture
def h2_pink_action(monkeypatch: pytest.MonkeyPatch):
    converted_quaternions: list[torch.Tensor] = []

    class Transform:
        def __init__(self) -> None:
            self.translation = np.zeros(3)
            self.rotation = np.eye(3)

        def copy(self):
            copied = Transform()
            copied.translation = self.translation.copy()
            copied.rotation = self.rotation.copy()
            return copied

    class FrameTask:
        def __init__(self) -> None:
            self.transform_target_to_world = Transform()
            self.received_target = None

        def set_target(self, target) -> None:
            self.received_target = target.copy()

    class LocalFrameTask(FrameTask):
        def __init__(self) -> None:
            super().__init__()
            self.transform_target_to_base = Transform()

    class PinkInverseKinematicsAction:
        def __init__(self, cfg, env) -> None:
            self.cfg = cfg
            self._env = env
            self.num_envs = 1
            self._asset = env.asset
            self._sim_dt = env.sim.get_physics_dt()
            self._ik_controllers = [env.controller]
            self._num_frame_tasks = 2
            self._isaaclab_controlled_joint_names = list(cfg.pink_controlled_joint_names)
            self._hand_joint_names = list(cfg.hand_joint_names)
            self._controlled_joint_ids = list(
                range(len(self._isaaclab_controlled_joint_names) + len(self._hand_joint_names))
            )
            self._raw_actions = torch.zeros((1, self._num_frame_tasks * self.pose_dim + self.hand_joint_dim))
            self._processed_actions = torch.zeros((1, len(self._controlled_joint_ids)))
            self.task_targets = None

        @property
        def hand_joint_dim(self) -> int:
            return len(self._hand_joint_names)

        @property
        def position_dim(self) -> int:
            return 3

        @property
        def pose_dim(self) -> int:
            return 7

        def _set_task_targets(self, targets) -> None:
            self.task_targets = tuple(value.clone() for value in targets)

        def _apply_gravity_compensation(self) -> None:
            self._asset.gravity_compensation_calls += 1

        def reset(self, env_ids=None) -> None:
            del env_ids
            self._raw_actions.zero_()

    class PinkInverseKinematicsActionCfg:
        pass

    class ActionTerm:
        pass

    modules = {
        "isaaclab": ModuleType("isaaclab"),
        "isaaclab.envs": ModuleType("isaaclab.envs"),
        "isaaclab.envs.mdp": ModuleType("isaaclab.envs.mdp"),
        "isaaclab.envs.mdp.actions": ModuleType("isaaclab.envs.mdp.actions"),
        "isaaclab.envs.mdp.actions.pink_actions_cfg": ModuleType("isaaclab.envs.mdp.actions.pink_actions_cfg"),
        "isaaclab.envs.mdp.actions.pink_task_space_actions": ModuleType(
            "isaaclab.envs.mdp.actions.pink_task_space_actions"
        ),
        "isaaclab.controllers": ModuleType("isaaclab.controllers"),
        "isaaclab.controllers.pink_ik": ModuleType("isaaclab.controllers.pink_ik"),
        "isaaclab.controllers.pink_ik.pink_tasks": ModuleType("isaaclab.controllers.pink_ik.pink_tasks"),
        "isaaclab.managers": ModuleType("isaaclab.managers"),
        "isaaclab.managers.action_manager": ModuleType("isaaclab.managers.action_manager"),
        "isaaclab.utils": ModuleType("isaaclab.utils"),
        "isaaclab.utils.math": ModuleType("isaaclab.utils.math"),
        "isaaclab.utils.configclass": ModuleType("isaaclab.utils.configclass"),
        "pink": ModuleType("pink"),
        "pink.tasks": ModuleType("pink.tasks"),
    }
    modules[
        "isaaclab.envs.mdp.actions.pink_actions_cfg"
    ].PinkInverseKinematicsActionCfg = PinkInverseKinematicsActionCfg
    modules[
        "isaaclab.envs.mdp.actions.pink_task_space_actions"
    ].PinkInverseKinematicsAction = PinkInverseKinematicsAction
    modules["isaaclab.controllers.pink_ik.pink_tasks"].LocalFrameTask = LocalFrameTask
    modules["isaaclab.managers.action_manager"].ActionTerm = ActionTerm
    modules["isaaclab.utils.configclass"].configclass = lambda cls: cls
    modules["pink.tasks"].FrameTask = FrameTask

    def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
        converted_quaternions.append(quaternions.clone())
        return torch.eye(3).expand(*quaternions.shape[:-1], 3, 3).clone()

    modules["isaaclab.utils.math"].matrix_from_quat = matrix_from_quat
    modules["isaaclab.utils"].math = modules["isaaclab.utils.math"]
    for name, module in modules.items():
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "_test_h2_pink_action"
    spec = importlib.util.spec_from_file_location(module_name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    module.converted_quaternions = converted_quaternions
    return module


class FakeController:
    def __init__(self, target: tuple[float, float]) -> None:
        self.controlled_joint_indices = [2, 0]
        self.isaac_lab_to_pink_controlled_ordering = np.asarray((1, 0))
        self.pink_configuration = SimpleNamespace(
            controlled_model=SimpleNamespace(
                lowerPositionLimit=np.asarray((-2.49400003, -0.576)),
                upperPositionLimit=np.asarray((0.517, 0.576)),
            )
        )
        self.cfg = SimpleNamespace(joint_names=["left_wrist_pitch_joint", "right_shoulder_roll_joint"])
        self.target = target
        self.received: np.ndarray | None = None
        self.received_dts: list[float] = []

    def compute(self, positions: np.ndarray, dt: float):
        self.received = positions.copy()
        self.received_dts.append(dt)
        return torch.tensor(self.target, dtype=torch.float32)


def _action(module, controller: FakeController, positions: tuple[float, float, float]):
    action = object.__new__(module.H2PinkInverseKinematicsAction)
    action.cfg = SimpleNamespace(
        joint_limit_guard_tolerance=5e-5,
        joint_limit_guard_margin=5e-6,
        joint_target_limit_margin=5e-4,
    )
    action._sim_dt = 0.01
    action._ik_controllers = [controller]
    action._asset = SimpleNamespace(
        data=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.tensor([positions], dtype=torch.float64)))
    )
    return action


def test_microscopic_state_and_target_excursions_are_moved_inside_urdf_limits(h2_pink_action) -> None:
    controller = FakeController(target=(-0.576049, 0.517049))
    action = _action(h2_pink_action, controller, positions=(0.517049, 0.0, -0.576049))

    result = action._compute_ik_solutions()

    assert controller.received is not None
    assert controller.received[2] == pytest.approx(-0.576 + 5e-6)
    assert controller.received[0] == pytest.approx(0.517 - 5e-6)
    assert result[0].tolist() == pytest.approx((-0.576 + 5e-4, 0.517 - 5e-4))


@pytest.mark.parametrize(
    "target",
    [
        (-0.576, 0.517),
        (-0.576 + 1e-5, 0.517 - 1e-5),
    ],
)
def test_exact_and_near_limit_targets_keep_replay_headroom(
    h2_pink_action,
    target: tuple[float, float],
) -> None:
    action = _action(h2_pink_action, FakeController(target=target), positions=(0.0, 0.0, 0.0))

    result = action._compute_ik_solutions()

    assert result[0].tolist() == pytest.approx((-0.576 + 5e-4, 0.517 - 5e-4))


@pytest.mark.parametrize(
    ("positions", "target", "stage"),
    [
        ((0.5172, 0.0, 0.0), (0.0, 0.0), "current state"),
        ((0.0, 0.0, 0.0), (-0.5762, 0.0), "IK target"),
    ],
)
def test_material_limit_violations_remain_hard_failures(
    h2_pink_action,
    positions: tuple[float, float, float],
    target: tuple[float, float],
    stage: str,
) -> None:
    action = _action(h2_pink_action, FakeController(target=target), positions=positions)

    with pytest.raises(RuntimeError, match=rf"{stage} materially violates"):
        action._compute_ik_solutions()


@pytest.mark.parametrize("measured", [1.220078349, 1.220068574])
def test_reported_wrist_faults_remain_rejected_with_a_specific_measured_state_error(
    h2_pink_action, measured: float
) -> None:
    controller = FakeController(target=(0.0, 0.0))
    controller.pink_configuration.controlled_model.lowerPositionLimit[1] = -1.219999950
    controller.pink_configuration.controlled_model.upperPositionLimit[1] = 1.219999950
    controller.cfg.joint_names[0] = "left_wrist_yaw_joint"
    action = _action(h2_pink_action, controller, positions=(0.0, 0.0, measured))

    with pytest.raises(H2MeasuredJointLimitViolation, match="left_wrist_yaw_joint"):
        action._compute_ik_solutions()
    assert controller.received is None


def test_invalid_ik_target_is_not_a_recoverable_measured_state_fault(h2_pink_action) -> None:
    action = _action(h2_pink_action, FakeController(target=(-0.5762, 0.0)), positions=(0.0, 0.0, 0.0))

    with pytest.raises(RuntimeError, match="IK target materially violates") as raised:
        action._compute_ik_solutions()
    assert not isinstance(raised.value, H2MeasuredJointLimitViolation)


def test_in_range_state_and_target_are_not_modified(h2_pink_action) -> None:
    controller = FakeController(target=(0.25, -1.0))
    action = _action(h2_pink_action, controller, positions=(-1.0, 0.0, 0.25))

    result = action._compute_ik_solutions()

    assert controller.received is not None
    assert np.array_equal(controller.received, np.asarray((-1.0, 0.0, 0.25)))
    assert result[0].tolist() == pytest.approx((0.25, -1.0))


def test_controlled_limit_mapping_uses_the_pink_to_isaaclab_direction(h2_pink_action) -> None:
    controller = SimpleNamespace(
        controlled_joint_indices=[0, 1, 2],
        isaac_lab_to_pink_controlled_ordering=np.asarray((1, 2, 0)),
        pink_configuration=SimpleNamespace(
            controlled_model=SimpleNamespace(
                lowerPositionLimit=np.asarray((10.0, 20.0, 30.0)),
                upperPositionLimit=np.asarray((11.0, 21.0, 31.0)),
            )
        ),
    )

    lower, upper = h2_pink_action.H2PinkInverseKinematicsAction._controlled_limits_in_isaaclab_order(controller)

    assert lower.tolist() == [30.0, 10.0, 20.0]
    assert upper.tolist() == [31.0, 11.0, 21.0]


@pytest.mark.parametrize(
    ("lower", "upper", "message"),
    [
        ((-2.49400003, np.nan), (0.517, 0.576), "must be finite"),
        ((-2.49400003, 0.576), (0.517, 0.576), "must be strictly ordered"),
    ],
)
def test_controlled_limits_reject_invalid_urdf_bounds(
    h2_pink_action,
    lower: tuple[float, float],
    upper: tuple[float, float],
    message: str,
) -> None:
    controller = FakeController(target=(0.0, 0.0))
    controller.pink_configuration.controlled_model.lowerPositionLimit = np.asarray(lower)
    controller.pink_configuration.controlled_model.upperPositionLimit = np.asarray(upper)

    with pytest.raises(RuntimeError, match=message):
        h2_pink_action.H2PinkInverseKinematicsAction._controlled_limits_in_isaaclab_order(controller)


class FakeAsset:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            joint_names=("right_shoulder_roll_joint", "unused_joint", "left_wrist_pitch_joint", "hand_joint"),
            joint_pos=SimpleNamespace(torch=torch.tensor([(-0.5, 0.0, 0.1, 0.0)], dtype=torch.float64)),
        )
        self.gravity_compensation_calls = 0
        self.position_target_writes: list[tuple[torch.Tensor, tuple[int, ...]]] = []

    def set_joint_position_target_index(self, *, target, joint_ids) -> None:
        self.position_target_writes.append((target.clone(), tuple(joint_ids)))


def _constructor_inputs(controller: FakeController, target_margin: float):
    controller.cfg.variable_input_tasks = []
    env = SimpleNamespace(
        asset=FakeAsset(),
        controller=controller,
        step_dt=0.01,
        sim=SimpleNamespace(get_physics_dt=lambda: 0.005),
    )
    cfg = SimpleNamespace(
        pink_controlled_joint_names=("right_shoulder_roll_joint", "left_wrist_pitch_joint"),
        hand_joint_names=("hand_joint",),
        joint_limit_guard_tolerance=5e-5,
        joint_limit_guard_margin=5e-6,
        joint_target_limit_margin=target_margin,
        enable_gravity_compensation=False,
    )
    return cfg, env


def _live_action(module, *, enable_gravity_compensation: bool = False):
    controller = FakeController(target=(0.25, -1.0))
    cfg, env = _constructor_inputs(controller, 5e-4)
    controller.cfg.variable_input_tasks = [module.LocalFrameTask(), module.LocalFrameTask()]
    cfg.enable_gravity_compensation = enable_gravity_compensation
    action = module.H2PinkInverseKinematicsAction(cfg, env)
    return action, controller, env.asset


@pytest.mark.parametrize("margin", (0.0, -1e-4, np.inf, np.nan))
def test_constructor_rejects_invalid_target_limit_margin(h2_pink_action, margin: float) -> None:
    cfg, env = _constructor_inputs(FakeController(target=(0.0, 0.0)), margin)

    with pytest.raises(ValueError, match="finite and positive"):
        h2_pink_action.H2PinkInverseKinematicsAction(cfg, env)


def test_constructor_rejects_infinite_joint_limit_guard_tolerance(h2_pink_action) -> None:
    cfg, env = _constructor_inputs(FakeController(target=(0.0, 0.0)), 5e-4)
    cfg.joint_limit_guard_tolerance = np.inf

    with pytest.raises(ValueError, match="must be finite numbers"):
        h2_pink_action.H2PinkInverseKinematicsAction(cfg, env)


@pytest.mark.parametrize(
    "field",
    ("joint_limit_guard_margin", "joint_limit_guard_tolerance", "joint_target_limit_margin"),
)
def test_constructor_rejects_boolean_joint_limit_values(h2_pink_action, field: str) -> None:
    cfg, env = _constructor_inputs(FakeController(target=(0.0, 0.0)), 5e-4)
    setattr(cfg, field, True)

    with pytest.raises(ValueError, match="must be finite"):
        h2_pink_action.H2PinkInverseKinematicsAction(cfg, env)


def test_constructor_requires_target_headroom_beyond_guard_tolerance(h2_pink_action) -> None:
    cfg, env = _constructor_inputs(FakeController(target=(0.0, 0.0)), 5e-5)

    with pytest.raises(ValueError, match="must exceed the guard tolerance"):
        h2_pink_action.H2PinkInverseKinematicsAction(cfg, env)


def test_default_target_limit_margin_preserves_guard_headroom(h2_pink_action) -> None:
    cfg_type = h2_pink_action.H2PinkInverseKinematicsActionCfg

    assert cfg_type.joint_target_limit_margin == pytest.approx(5e-4)
    assert cfg_type.joint_target_limit_margin > cfg_type.joint_limit_guard_tolerance


def test_constructor_rejects_target_margin_that_collapses_a_joint_interval(h2_pink_action) -> None:
    cfg, env = _constructor_inputs(FakeController(target=(0.0, 0.0)), 0.6)

    with pytest.raises(ValueError, match="collapses"):
        h2_pink_action.H2PinkInverseKinematicsAction(cfg, env)


def test_control_step_solves_once_and_physics_substeps_reuse_the_cached_target(h2_pink_action) -> None:
    action, controller, asset = _live_action(h2_pink_action, enable_gravity_compensation=True)
    command = torch.tensor(
        ((1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9, 4.0, 5.0, 6.0, 0.4, 0.5, 0.6, 0.7, 0.8),),
        dtype=torch.float32,
    )

    action.process_actions(command)
    command[0, -1] = -0.7
    asset.data.joint_pos.torch[:] = 0.0
    action.apply_actions()
    action.apply_actions()

    assert controller.received_dts == [0.01]
    assert controller.received is not None
    assert controller.received.tolist() == pytest.approx([-0.5, 0.0, 0.1, 0.0])
    task_targets = [task.received_target for task in controller.cfg.variable_input_tasks]
    assert all(target is not None for target in task_targets)
    assert np.allclose([target.translation for target in task_targets], ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
    assert all(target.rotation.shape == (3, 3) for target in task_targets)
    assert np.allclose(
        h2_pink_action.converted_quaternions[0].reshape(-1, 4).numpy(),
        ((0.1, 0.2, 0.3, 0.9), (0.4, 0.5, 0.6, 0.7)),
    )
    assert asset.gravity_compensation_calls == 2
    assert len(asset.position_target_writes) == 2
    expected_target = (0.25, -1.0, 0.8)
    for target, joint_ids in asset.position_target_writes:
        assert target[0].tolist() == pytest.approx(expected_target)
        assert joint_ids == (0, 1, 2)


@pytest.mark.parametrize(("index", "value"), ((0, np.nan), (-1, np.inf)))
def test_nonfinite_commands_invalidate_the_cached_target(
    h2_pink_action,
    index: int,
    value: float,
) -> None:
    action, controller, asset = _live_action(h2_pink_action)
    command = torch.tensor(
        ((1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9, 4.0, 5.0, 6.0, 0.4, 0.5, 0.6, 0.7, 0.8),),
        dtype=torch.float32,
    )
    action.process_actions(command)
    command[0, index] = value

    with pytest.raises(ValueError, match="only finite values"):
        action.process_actions(command)
    with pytest.raises(RuntimeError, match="must be processed"):
        action.apply_actions()

    assert controller.received_dts == [0.01]
    assert asset.position_target_writes == []


def test_reset_invalidates_the_cached_pink_target(h2_pink_action) -> None:
    action, controller, asset = _live_action(h2_pink_action)
    action.process_actions(
        torch.tensor(
            ((1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9, 4.0, 5.0, 6.0, 0.4, 0.5, 0.6, 0.7, 0.8),),
            dtype=torch.float32,
        )
    )

    action.reset()

    with pytest.raises(RuntimeError, match="must be processed"):
        action.apply_actions()
    assert controller.received_dts == [0.01]
    assert asset.position_target_writes == []

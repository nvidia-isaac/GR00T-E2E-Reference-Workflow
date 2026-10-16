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

"""Host-safe behavioral contracts for the Arena-only H2 AGILE binding."""

from __future__ import annotations

import ast
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from e2e_robot_workflows.arena.embodiment_capabilities import (
    G1_ARENA_EMBODIMENT,
    H2_ARENA_EMBODIMENT,
    capability_for_embodiment,
)
from e2e_robot_workflows.robots.h2 import (
    DRIVEN_JOINT_COUNT,
    EXPECTED_JOINT_COUNT,
    load_h2_pink_contract,
    load_h2_robot_config,
)
from e2e_robot_workflows.tasks.static_apple.config import H2_STATIC_APPLE_VARIANT, load_static_apple_environment

DRIVEN_INDICES_IN_BODY_ORDER = (0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 19, 20)
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_external_factory_registers_h2_before_resolving_the_embodiment() -> None:
    """Arena's evaluator calls the factory directly, without the compose helper."""
    source = REPO_ROOT / "src/e2e_robot_workflows/arena/external_environment.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    factory = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "E2EArenaExternalEnvironment"
    )
    build = next(node for node in factory.body if isinstance(node, ast.FunctionDef) and node.name == "build")
    registration = next(
        node
        for node in ast.walk(build)
        if isinstance(node, ast.Import)
        and any(alias.name == "e2e_robot_workflows.arena.h2_embodiments" for alias in node.names)
    )
    lookup = next(
        node
        for node in ast.walk(build)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get_asset_by_name"
    )
    assert registration.lineno < lookup.lineno


def test_h2_keeps_the_pelvis_following_xr_frame() -> None:
    assert capability_for_embodiment(G1_ARENA_EMBODIMENT).load_binding().fixed_xr_anchor_rotation is True
    assert capability_for_embodiment(H2_ARENA_EMBODIMENT).load_binding().fixed_xr_anchor_rotation is False


def test_arena_binding_preserves_configured_supported_reset_stance() -> None:
    robot = load_h2_robot_config()
    binding = capability_for_embodiment(H2_ARENA_EMBODIMENT).load_binding()
    for name, value in robot.initial_joint_pos.items():
        assert binding.joint_initial_pos[name] == value
    assert binding.joint_initial_pos["left_knee_joint"] != robot.reference_joint_pos["left_knee_joint"]


def test_arena_robot_rates_preserve_g1_and_decouple_h2_rendering() -> None:
    g1 = capability_for_embodiment(G1_ARENA_EMBODIMENT).load_binding()
    h2 = capability_for_embodiment(H2_ARENA_EMBODIMENT).load_binding()

    assert (g1.control_hz, g1.render_hz) == (50, 50)
    assert (h2.control_hz, h2.render_hz, load_h2_robot_config().camera_hz) == (100, 50, 25)


def test_h2_workcell_puts_pick_left_and_destination_right_of_robot() -> None:
    robot = load_h2_robot_config()
    task = load_static_apple_environment(H2_STATIC_APPLE_VARIANT)
    x, y, z, w = robot.initial_rotation_xyzw
    quaternion_norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = (value / quaternion_norm for value in (x, y, z, w))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    forward = np.array((math.cos(yaw), math.sin(yaw)))
    left = np.array((-math.sin(yaw), math.cos(yaw)))
    root = np.array(robot.initial_position_xy)
    pick_delta = np.array(task.pick_spawn_xy) - root
    destination_delta = np.array(task.destination_spawn_xy) - root

    assert np.dot(pick_delta, forward) == pytest.approx(np.dot(destination_delta, forward), abs=1e-6)
    assert np.dot(pick_delta, forward) > 0.0
    assert np.dot(pick_delta, left) == pytest.approx(-np.dot(destination_delta, left), abs=1e-6)
    assert np.dot(pick_delta, left) > 0.0
    assert np.dot(destination_delta, left) < 0.0
    assert task.support_surface_z == pytest.approx(0.96)
    assert task.release_clearance_m == pytest.approx(0.20)
    assert task.pickup_lift_clearance_m == pytest.approx(0.08)


def test_h2_controller_ownership_is_disjoint_and_covers_the_live_plant() -> None:
    robot = load_h2_robot_config()
    plant = load_h2_pink_contract()
    agile = set(robot.driven_joint_names)
    arms = set(plant.arm_joint_names)
    hands = set(plant.hand_joint_order)
    held = {"waist_yaw_joint", "head_pitch_joint", "head_yaw_joint"}

    owners = (agile, arms, hands, held)
    assert all(not left & right for index, left in enumerate(owners) for right in owners[index + 1 :])
    assert set().union(*owners) == set(plant.expected_joint_order)
    assert (
        tuple(plant.body_joint_order.index(name) for name in robot.driven_joint_names) == DRIVEN_INDICES_IN_BODY_ORDER
    )
    assert (len(robot.expected_joint_order), len(robot.driven_joints), robot.command_dim) == (
        EXPECTED_JOINT_COUNT,
        DRIVEN_JOINT_COUNT,
        3,
    )
    assert robot.task_link_names == plant.flange_frames


def test_h2_agile_gains_reference_pose_and_rate_keep_the_recovered_contract() -> None:
    robot = load_h2_robot_config()
    stiffness, damping = robot.driven_joint_stiffness, robot.driven_joint_damping
    for name in robot.driven_joint_names:
        if name.startswith("left_"):
            twin = "right_" + name.removeprefix("left_")
            assert (stiffness[twin], damping[twin]) == (stiffness[name], damping[name])

    reference = robot.reference_joint_pos
    assert {name for name, value in reference.items() if value != 0.0} == {
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
    }
    assert (robot.control_hz, robot.render_hz, robot.camera_hz) == (100, 50, 25)
    assert robot.navigate_command_limits == ((-0.5, 0.5),) * 3
    assert robot.history_length == 5


def test_h2_policy_resolution_rejects_substituted_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    robot = load_h2_robot_config()
    impostor = tmp_path / robot.policy_filename
    impostor.write_bytes(b"not the pinned H2 AGILE export")
    monkeypatch.setenv(robot.policy_env_var, str(impostor))

    with pytest.raises(ValueError, match="unexpected checkpoint"):
        robot.resolve_policy()


@pytest.mark.parametrize(
    "task_link_names",
    [["same", "same"], ["left"], ["", "right"], [1, "right"], "lr"],
)
def test_h2_robot_config_rejects_malformed_task_links(
    tmp_path: Path,
    task_link_names: object,
) -> None:
    config = yaml.safe_load((REPO_ROOT / "configs/robots/h2.yaml").read_text(encoding="utf-8"))
    config["task_link_names"] = task_link_names
    config_path = tmp_path / "h2.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="two distinct non-empty robot links"):
        load_h2_robot_config(str(config_path))


def test_h2_robot_config_rejects_malformed_direct_task_links() -> None:
    with pytest.raises(ValueError, match="two distinct non-empty robot links"):
        replace(load_h2_robot_config(), task_link_names="lr")


def _session_or_skip():
    onnxruntime = pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")
    robot = load_h2_robot_config()
    try:
        value = robot.resolve_policy()
    except RuntimeError as error:
        pytest.skip(str(error))
    return onnxruntime.InferenceSession(value, providers=["CPUExecutionProvider"]), robot


def _run_once(session, robot, *, joint_pos: np.ndarray | None) -> dict:
    driven, depth = DRIVEN_JOINT_COUNT, robot.history_length
    feeds = {
        "root_link_quat_w": np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
        "root_ang_vel_b": np.zeros((1, 3), np.float32),
        "velocity_commands": np.zeros((1, robot.command_dim), np.float32),
        "joint_pos": np.zeros((1, EXPECTED_JOINT_COUNT), np.float32) if joint_pos is None else joint_pos,
        "joint_vel": np.zeros((1, EXPECTED_JOINT_COUNT), np.float32),
        "last_actions_in": np.zeros((1, driven), np.float32),
        "base_ang_vel_history_in": np.zeros((1, depth, 3), np.float32),
        "projected_gravity_history_in": np.zeros((1, depth, 3), np.float32),
        "velocity_commands_history_in": np.zeros((1, depth, robot.command_dim), np.float32),
        "controlled_joint_pos_history_in": np.zeros((1, depth, driven), np.float32),
        "controlled_joint_vel_history_in": np.zeros((1, depth, driven), np.float32),
        "actions_history_in": np.zeros((1, depth, driven), np.float32),
    }
    names = [output.name for output in session.get_outputs()]
    return dict(zip(names, session.run(names, feeds), strict=True))


def test_h2_export_ports_gains_and_absolute_joint_observation() -> None:
    session, robot = _session_or_skip()
    shapes = {model_input.name: model_input.shape for model_input in session.get_inputs()}
    assert shapes["velocity_commands"] == [1, 3]
    assert shapes["joint_pos"] == shapes["joint_vel"] == [1, EXPECTED_JOINT_COUNT]
    assert shapes["root_link_quat_w"] == [1, 4]

    at_reference = np.zeros((1, EXPECTED_JOINT_COUNT), np.float32)
    for name, value in robot.reference_joint_pos.items():
        at_reference[0, robot.expected_joint_order.index(name)] = value
    output = _run_once(session, robot, joint_pos=at_reference)

    assert output["action_joint_pos_kp_gains"][0].tolist() == [entry.kp for entry in robot.driven_joints]
    assert output["action_joint_pos_kd_gains"][0].tolist() == [entry.kd for entry in robot.driven_joints]
    assert output["controlled_joint_pos_history_out"][0, -1] == pytest.approx(np.zeros(DRIVEN_JOINT_COUNT), abs=1e-6)
    assert np.isfinite(output["action_joint_pos"]).all()

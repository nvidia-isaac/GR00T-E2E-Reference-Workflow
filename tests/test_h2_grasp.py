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

"""Grasp labels must distinguish actual closed-hand contact from a nearby open palm."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from e2e_robot_workflows.robots.h2 import load_h2_pink_contract
from e2e_robot_workflows.tasks.static_apple.grasp_contract import H2_GRASP_CRITERIA
from e2e_robot_workflows.tasks.static_apple.h2_grasp import (
    h2_hand_body_names,
    h2_hand_contact_filter_paths,
    object_held_by_h2_hand,
    reset_h2_grasp_history,
)


def _grasp_fixture(*, num_envs: int = 1):
    torch = pytest.importorskip("torch")
    contract = load_h2_pink_contract()
    action = torch.zeros((num_envs, 61))
    joints = torch.zeros((num_envs, 75))
    object_position = torch.tensor([[0.0, 0.0, 1.2]]).repeat(num_envs, 1)
    links = torch.tensor([[[0.0, 0.1, 1.2], [0.0, -0.1, 1.2]]]).repeat(num_envs, 1, 1)
    scene = {
        "pick_object": SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=object_position))),
        "robot": SimpleNamespace(
            data=SimpleNamespace(
                joint_names=list(contract.expected_joint_order),
                joint_pos=SimpleNamespace(torch=joints),
                body_names=["left_hand_flange", "right_hand_flange"],
                body_pos_w=SimpleNamespace(torch=links),
            )
        ),
    }
    for hand in ("left", "right"):
        forces = torch.zeros((num_envs, 3, 1, 24, 3))
        scene[f"contact_{hand}"] = SimpleNamespace(
            data=SimpleNamespace(force_matrix_w_history=SimpleNamespace(torch=forces))
        )

    class Scene(dict):
        env_origins = torch.zeros((num_envs, 3))

    env = SimpleNamespace(cfg=SimpleNamespace(), scene=Scene(scene), action_manager=SimpleNamespace(action=action))
    return env


def _close(env, hand: str, *, commanded: float = 1.0, measured: float = 0.25) -> None:
    contract = load_h2_pink_contract()
    for finger in H2_GRASP_CRITERIA.primary_flexion_fingers:
        name = f"{hand}_{finger}_MCP_FE"
        closed = contract.hand_closed_targets[name]
        env.action_manager.action[:, 14 + contract.hand_joint_order.index(name)] = commanded * closed
        joint_index = contract.expected_joint_order.index(name)
        env.scene["robot"].data.joint_pos.torch[:, joint_index] = measured * closed


def _held(env, hand: str):
    return object_held_by_h2_hand(
        env,
        hand=hand,
        task_link_name=f"{hand}_hand_flange",
        grasp_distance_m=0.20,
        min_object_height_m=1.1,
        contact_sensor_cfg=SimpleNamespace(name=f"contact_{hand}"),
    )


@pytest.mark.parametrize("hand", ("left", "right"))
def test_open_palm_contact_does_not_count_as_a_grasp(hand: str) -> None:
    env = _grasp_fixture()
    env.scene[f"contact_{hand}"].data.force_matrix_w_history.torch[:, :, :, 1, 2] = 0.5
    assert not _held(env, hand).any().item()
    _close(env, hand, commanded=0.0, measured=0.5)
    assert not _held(env, hand).any().item()


@pytest.mark.parametrize("hand", ("left", "right"))
def test_closed_hand_near_apple_without_contact_is_not_holding(hand: str) -> None:
    env = _grasp_fixture()
    _close(env, hand)
    assert not _held(env, hand).any().item()


def test_overlapping_handover_allows_both_actual_grasps() -> None:
    env = _grasp_fixture()
    for hand in ("left", "right"):
        _close(env, hand)
        # Two opposing finger forces cancel as vectors but still prove contact.
        forces = env.scene[f"contact_{hand}"].data.force_matrix_w_history.torch
        forces[:, :, :, 6, 0] = 0.05
        forces[:, :, :, 10, 0] = -0.05
    assert _held(env, "left").all().item()
    assert _held(env, "right").all().item()
    _close(env, "left", commanded=0.0)
    assert not _held(env, "left").any().item()
    assert _held(env, "right").all().item()


def test_commanded_close_requires_measured_partial_closure() -> None:
    env = _grasp_fixture()
    _close(env, "right", measured=0.0)
    env.scene["contact_right"].data.force_matrix_w_history.torch[:, :, :, 6, 0] = 0.05
    assert not _held(env, "right").any().item()


def test_contact_must_persist_in_all_three_control_samples() -> None:
    env = _grasp_fixture()
    _close(env, "left")
    forces = env.scene["contact_left"].data.force_matrix_w_history.torch
    forces[:, :2, :, 6, 0] = 0.05
    for _ in range(4):
        assert not _held(env, "left").any().item()
    forces[:, 2, :, 6, 0] = 0.05
    assert _held(env, "left").all().item()


def test_contact_does_not_replace_lift_and_proximity_requirements() -> None:
    env = _grasp_fixture(num_envs=3)
    _close(env, "left")
    env.scene["contact_left"].data.force_matrix_w_history.torch[:, :, :, 6, 0] = 0.05
    env.scene["pick_object"].data.root_pos_w.torch[1, 2] = 1.05
    env.scene["robot"].data.body_pos_w.torch[2, 0, 0] = 1.0
    assert _held(env, "left").tolist() == [True, False, False]


def test_grasp_lift_threshold_follows_each_table_height() -> None:
    torch = pytest.importorskip("torch")
    env = _grasp_fixture(num_envs=3)
    _close(env, "left")
    env.scene["contact_left"].data.force_matrix_w_history.torch[:, :, :, 6, 0] = 0.05
    env.scene.env_origins[:, 2] = torch.tensor([0.0, 2.0, 4.0])
    env.scene["pick_object"].data.root_pos_w.torch += env.scene.env_origins
    env.scene["robot"].data.body_pos_w.torch += env.scene.env_origins[:, None, :]
    env.cfg.e2e_support_asset_name = "packing_table"
    env.cfg.e2e_support_root_z_m = 0.2
    table_poses = torch.zeros((3, 7))
    table_poses[:, 2] = env.scene.env_origins[:, 2] + 0.2 + torch.tensor([0.0, 0.15, -0.15])
    table_poses[:, 6] = 1.0
    env.scene["packing_table"] = SimpleNamespace(data=SimpleNamespace(root_pose_w=SimpleNamespace(torch=table_poses)))
    assert _held(env, "left").tolist() == [True, False, True]
    env.scene["pick_object"].data.root_pos_w.torch[:, 2] -= 0.15
    assert _held(env, "left").tolist() == [False, False, True]


@pytest.mark.parametrize("selected", (None, [1]))
def test_reset_clears_only_selected_hand_contact_histories(selected) -> None:
    torch = pytest.importorskip("torch")
    env = _grasp_fixture(num_envs=2)
    for hand in ("left", "right"):
        _close(env, hand)
        env.scene[f"contact_{hand}"].data.force_matrix_w_history.torch.fill_(1.0)
    env_ids = None if selected is None else torch.tensor(selected)
    reset_h2_grasp_history(env, env_ids, ("contact_left", "contact_right"))
    for hand in ("left", "right"):
        assert _held(env, hand).tolist() == ([False, False] if selected is None else [True, False])
        forces = env.scene[f"contact_{hand}"].data.force_matrix_w_history.torch
        for frame in range(3):
            forces[1, frame, 0, 6, 0] = 0.05
            assert _held(env, hand)[1].item() == (frame == 2)


@pytest.mark.parametrize("bad_input", ("command", "joint", "contact", "missing_sensor", "contact_shape"))
def test_invalid_grasp_evidence_fails_closed(bad_input: str) -> None:
    env = _grasp_fixture()
    _close(env, "left")
    contract = load_h2_pink_contract()
    if bad_input == "command":
        env.action_manager.action[0, 14 + contract.hand_joint_order.index("left_index_MCP_FE")] = float("nan")
    elif bad_input == "joint":
        env.scene["robot"].data.joint_pos.torch[0, contract.expected_joint_order.index("left_index_MCP_FE")] = float(
            "nan"
        )
    elif bad_input == "contact":
        env.scene["contact_left"].data.force_matrix_w_history.torch[0, 0, 0, 0, 0] = float("nan")
    elif bad_input == "missing_sensor":
        del env.scene["contact_left"]
    else:
        env.scene["contact_left"].data.force_matrix_w_history.torch = env.scene[
            "contact_left"
        ].data.force_matrix_w_history.torch[:, :, :, :1]
    with pytest.raises(ValueError, match="H2 left grasp evidence"):
        _held(env, "left")


def _hand_usd_fixture(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    path = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/H2").GetPrim()
    stage.SetDefaultPrim(root)
    for hand in ("left", "right"):
        flange = f"/H2/Geometry/{hand}_hand_flange"
        for name in h2_hand_body_names(hand):
            body_path = flange if name == f"{hand}_hand_flange" else f"{flange}/{name}"
            UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, body_path).GetPrim())
    stage.GetRootLayer().Save()
    return stage, path


def test_contact_filters_are_distinct_exact_rigid_body_paths(tmp_path) -> None:
    _, path = _hand_usd_fixture(tmp_path)
    filters = h2_hand_contact_filter_paths(path, "{ENV_REGEX_NS}/Robot")
    assert set(filters) == {"left", "right"}
    assert len(set(filters["left"] + filters["right"])) == 48
    for hand, paths in filters.items():
        assert len(paths) == 24
        assert all(path.startswith(f"{{ENV_REGEX_NS}}/Robot/Geometry/{hand}_hand_flange") for path in paths)
        assert {path.rsplit("/", 1)[-1] for path in paths} == set(h2_hand_body_names(hand))
        assert all(".*" not in path and "*" not in path for path in paths)


@pytest.mark.parametrize("change", ("missing", "extra"))
def test_contact_filters_reject_changed_robot_body_contract(tmp_path, change: str) -> None:
    stage, path = _hand_usd_fixture(tmp_path)
    from pxr import UsdGeom, UsdPhysics

    if change == "missing":
        stage.RemovePrim("/H2/Geometry/left_hand_flange/left_thumb_DP")
    else:
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/H2/Geometry/left_hand_flange/unexpected").GetPrim())
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="H2 left grasp contact body contract changed"):
        h2_hand_contact_filter_paths(path, "{ENV_REGEX_NS}/Robot")

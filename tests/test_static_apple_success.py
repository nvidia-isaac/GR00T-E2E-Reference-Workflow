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

"""Host-side tests for automatic success/failure detection of recording episodes.

Success requires a settled apple on the plate and both hands released; failure
stays geometric.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from e2e_robot_workflows.environments import STATIC_APPLE_ENVIRONMENT
from e2e_robot_workflows.tasks.static_apple.completion import build_policy
from e2e_robot_workflows.tasks.static_apple.success import (
    DEFAULT_CRITERIA,
    Outcome,
    PlacementCriteria,
    check_readable,
    classify,
    evaluate,
)
from e2e_robot_workflows.tasks.static_apple.table_height import resolve_randomization

PLATE = (0.35, 0.4)  # destination spawn XY
PLATE_Z = 0.75  # ~plate resting height
ON_TABLE_Z = 0.77  # apple resting height
CONTACT = DEFAULT_CRITERIA.contact_force_threshold + 1.0  # a clear contact force
RELEASED = (0.2, 0.2)
G1_TASK_LINK_NAMES = ("left_hand_thumb_2_link", "right_hand_thumb_2_link")


def test_in_contact_and_settled_is_success() -> None:
    assert classify(PLATE, ON_TABLE_Z, 0.0, PLATE, PLATE_Z, CONTACT, RELEASED) is Outcome.SUCCESS


def test_nominal_resting_weight_is_enough_contact_for_success() -> None:
    apple_mass = STATIC_APPLE_ENVIRONMENT.asset("pick_object").mass_kg
    assert apple_mass is not None
    nominal_static_force = apple_mass * 9.81

    assert nominal_static_force > DEFAULT_CRITERIA.contact_force_threshold
    assert classify(PLATE, ON_TABLE_Z, 0.0, PLATE, PLATE_Z, nominal_static_force, RELEASED) is Outcome.SUCCESS


def test_contact_on_an_elevated_target_is_not_supported_success() -> None:
    elevated_z = DEFAULT_CRITERIA.support_surface_z_m + DEFAULT_CRITERIA.max_object_above_support_m + 0.2
    assert classify(PLATE, elevated_z, 0.0, PLATE, elevated_z - 0.02, CONTACT, RELEASED) is Outcome.RUNNING


def test_contact_below_the_target_root_is_not_success() -> None:
    apple_z = PLATE_Z + DEFAULT_CRITERIA.min_object_height_above_target_m - 0.01
    assert classify(PLATE, apple_z, 0.0, PLATE, PLATE_Z, CONTACT, RELEASED) is Outcome.RUNNING


def test_contact_without_release_is_not_success() -> None:
    assert classify(PLATE, ON_TABLE_Z, 0.0, PLATE, PLATE_Z, CONTACT) is Outcome.RUNNING


def test_no_contact_is_not_success() -> None:
    # Apple resting exactly where the plate is, but not touching it -> not placed.
    assert (
        classify(PLATE, ON_TABLE_Z, 0.0, PLATE, PLATE_Z, contact_force=0.0, task_link_distances=RELEASED)
        is Outcome.RUNNING
    )


def test_force_below_threshold_is_not_success() -> None:
    weak = DEFAULT_CRITERIA.contact_force_threshold - 0.01
    assert (
        classify(PLATE, ON_TABLE_Z, 0.0, PLATE, PLATE_Z, contact_force=weak, task_link_distances=RELEASED)
        is Outcome.RUNNING
    )


def test_in_contact_but_moving_is_not_success() -> None:
    fast = DEFAULT_CRITERIA.settle_speed_mps + 0.5
    assert (
        classify(PLATE, ON_TABLE_Z, fast, PLATE, PLATE_Z, contact_force=CONTACT, task_link_distances=RELEASED)
        is Outcome.RUNNING
    )


def test_dropped_apple_is_failure_even_in_contact() -> None:
    assert classify(PLATE, 0.3, 0.0, PLATE, PLATE_Z, contact_force=CONTACT) is Outcome.FAILURE_DROPPED


def test_dropped_plate_is_failure() -> None:
    assert classify(PLATE, ON_TABLE_Z, 0.0, PLATE, 0.3, contact_force=CONTACT) is Outcome.FAILURE_PLATE_DROPPED


def test_far_from_plate_is_failure() -> None:
    far_xy = (PLATE[0] + DEFAULT_CRITERIA.far_distance_m + 0.1, PLATE[1])
    assert classify(far_xy, ON_TABLE_Z, 0.0, PLATE, PLATE_Z, contact_force=0.0) is Outcome.FAILURE_FAR


def test_side_touch_beyond_the_plate_footprint_is_not_success() -> None:
    # Apple touching the plate from the side -- its center past the plate footprint --
    # must not count as placed, even in contact and settled.
    side_xy = (PLATE[0] + DEFAULT_CRITERIA.placement_xy_m + 0.03, PLATE[1])
    assert classify(side_xy, ON_TABLE_Z, 0.0, PLATE, PLATE_Z, CONTACT, RELEASED) is Outcome.RUNNING


def test_diagonal_touch_outside_the_circular_footprint_is_not_success() -> None:
    offset = 0.8 * DEFAULT_CRITERIA.placement_xy_m
    diagonal_xy = (PLATE[0] + offset, PLATE[1] + offset)

    assert classify(diagonal_xy, ON_TABLE_Z, 0.0, PLATE, PLATE_Z, CONTACT, RELEASED) is Outcome.RUNNING


def test_contact_outside_the_placement_height_is_not_success() -> None:
    high_z = PLATE_Z + DEFAULT_CRITERIA.placement_z_m + 0.01
    height_only = PlacementCriteria(max_object_above_support_m=1.0)

    assert classify(PLATE, high_z, 0.0, PLATE, PLATE_Z, CONTACT, RELEASED, height_only) is Outcome.RUNNING


@pytest.mark.parametrize(
    ("height_delta", "expected"),
    [
        (0.0625, Outcome.SUCCESS),
        (0.0626, Outcome.RUNNING),
        (-0.0078125, Outcome.SUCCESS),
        (-0.015625, Outcome.RUNNING),
    ],
)
def test_recording_placement_height_boundaries(height_delta: float, expected: Outcome) -> None:
    height_only = PlacementCriteria(
        placement_z_m=0.0625,
        min_object_height_above_target_m=-0.015625,
        max_object_above_support_m=1.0,
    )

    assert classify(PLATE, PLATE_Z + height_delta, 0.0, PLATE, PLATE_Z, CONTACT, RELEASED, height_only) is expected


def test_apple_resting_near_the_plate_edge_is_success() -> None:
    # Center still within the footprint (not fully centered) -> placed.
    edge_xy = (PLATE[0] + DEFAULT_CRITERIA.placement_xy_m - 0.01, PLATE[1])
    assert classify(edge_xy, ON_TABLE_Z, 0.0, PLATE, PLATE_Z, CONTACT, RELEASED) is Outcome.SUCCESS


def test_record_gates_on_the_contract_placement_radius() -> None:
    """Recording, generation and evaluation all size the gate from one contract value."""

    from e2e_robot_workflows.tasks.static_apple.spec import ENVIRONMENT

    assert DEFAULT_CRITERIA.placement_xy_m == pytest.approx(float(ENVIRONMENT.success_placement_xy_m))


def test_success_criteria_are_respected() -> None:
    # A stricter success force threshold means the same contact no longer counts.
    strict = PlacementCriteria(contact_force_threshold=100.0)
    assert (
        classify(
            PLATE,
            ON_TABLE_Z,
            0.0,
            PLATE,
            PLATE_Z,
            contact_force=CONTACT,
            task_link_distances=RELEASED,
            criteria=strict,
        )
        is Outcome.RUNNING
    )


class _RootState:
    """Mimics an Isaac Lab proxy array exposing a ``.torch`` view (numpy stands in)."""

    def __init__(self, array: np.ndarray) -> None:
        self.torch = array


class _Data:
    def __init__(self, state: object) -> None:
        self.root_state_w = state


class _Object:
    def __init__(self, state: object) -> None:
        self.data = _Data(state)


class _RobotData:
    def __init__(self, poses: np.ndarray) -> None:
        self.body_pos_w = _RootState(poses[..., :3])


class _Robot:
    body_names = list(G1_TASK_LINK_NAMES)

    def __init__(self, poses: np.ndarray) -> None:
        self.data = _RobotData(poses)


class _ContactData:
    def __init__(self, force_matrix: object) -> None:
        self.force_matrix_w = force_matrix


class _ContactSensor:
    def __init__(self, force_matrix: object) -> None:
        self.data = _ContactData(force_matrix)


# The native Isaac Lab task inherits the locomanipulation scene's entity names and
# registers the contact sensor under a fixed name; the Arena adapter spawns each asset
# under the name the contract declares and names the sensor after the object it watches.
NATIVE_SCENE_NAMES = ("object", "destination", "object_contact_sensor")
ARENA_SCENE_NAMES = ("pick_object", "placement_target", "contact_sensor_pick_object")


class _Scene:
    def __init__(
        self,
        apple: _Object,
        plate: _Object,
        sensor: _ContactSensor,
        origins: np.ndarray,
        names: tuple[str, str, str] = NATIVE_SCENE_NAMES,
    ) -> None:
        self._entities = {names[0]: apple, names[1]: plate, names[2]: sensor}
        self.env_origins = origins

    def __getitem__(self, key: str) -> object:
        return self._entities[key]

    def keys(self) -> object:
        return self._entities.keys()


class _Env:
    def __init__(
        self,
        scene: _Scene,
        support_surface_z_m: float | None = None,
        release_clearance_m: float | None = None,
    ) -> None:
        self.scene = scene
        values = {}
        values["e2e_task_link_names"] = G1_TASK_LINK_NAMES
        if support_surface_z_m is not None:
            values["e2e_support_surface_z_m"] = support_surface_z_m
        if release_clearance_m is not None:
            values["e2e_release_clearance_m"] = release_clearance_m
        self.cfg = SimpleNamespace(**values)


def _state(local_xy: tuple[float, float], z: float, speed: float, origin: tuple[float, float, float]) -> np.ndarray:
    row = np.zeros((1, 13), dtype=np.float32)
    row[0, 0] = local_xy[0] + origin[0]  # world position = local + env origin
    row[0, 1] = local_xy[1] + origin[1]
    row[0, 2] = z + origin[2]
    row[0, 7] = speed  # lin_vel_x
    return row


def _env(
    apple_z: float,
    apple_speed: float,
    contact_force: float,
    names: tuple[str, str, str] = NATIVE_SCENE_NAMES,
    support_surface_z_m: float | None = None,
    release_clearance_m: float | None = None,
) -> _Env:
    origin = (5.0, -2.0, 0.0)
    apple = _Object(_RootState(_state(PLATE, apple_z, apple_speed, origin)))
    plate = _Object(_RootState(_state(PLATE, PLATE_Z, 0.0, origin)))
    force_matrix = np.zeros((1, 1, 1, 3), dtype=np.float32)  # (N, B, M, 3)
    force_matrix[0, 0, 0, 2] = contact_force  # Fz
    sensor = _ContactSensor(_RootState(force_matrix))
    origins = np.array([list(origin)], dtype=np.float32)
    link_poses = np.zeros((1, 2, 7), dtype=np.float32)
    link_poses[0, 0, :3] = (origin[0] + PLATE[0] - 0.2, origin[1] + PLATE[1], ON_TABLE_Z)
    link_poses[0, 1, :3] = (origin[0] + PLATE[0] + 0.2, origin[1] + PLATE[1], ON_TABLE_Z)
    scene = _Scene(apple, plate, sensor, origins, names)
    scene._entities["robot"] = _Robot(link_poses)
    return _Env(scene, support_surface_z_m, release_clearance_m)


def _randomized_env(offset: float, offset_range: tuple[float, float] = (-0.1, 0.1)) -> _Env:
    randomization = resolve_randomization(definition=STATIC_APPLE_ENVIRONMENT, offset_range_m=offset_range)
    env = _env(
        ON_TABLE_Z + offset,
        0.0,
        CONTACT,
        names=ARENA_SCENE_NAMES,
        support_surface_z_m=STATIC_APPLE_ENVIRONMENT.support_surface_z,
    )
    env.scene[ARENA_SCENE_NAMES[1]].data.root_state_w.torch[:, 2] += offset
    env.cfg.e2e_table_height_randomization = randomization
    env.cfg.e2e_support_asset_name = randomization.support_asset_name
    env.cfg.e2e_support_root_z_m = randomization.nominal_support_root_z_m
    pose = np.array([[0, 0, randomization.nominal_support_root_z_m + offset, 0, 0, 0, 1]], dtype=np.float32)
    pose[:, :3] += env.scene.env_origins
    env.scene._entities[randomization.support_asset_name] = SimpleNamespace(
        data=SimpleNamespace(root_pose_w=_RootState(pose))
    )
    return env


@pytest.mark.parametrize("offset", [-0.1, 0.1])
def test_evaluate_follows_the_live_table_without_moving_the_failure_plane(offset: float) -> None:
    env = _randomized_env(offset)
    assert evaluate(env) is Outcome.SUCCESS
    table = env.scene[env.cfg.e2e_support_asset_name].data.root_pose_w.torch
    table[:, 2] -= offset
    assert evaluate(env) is Outcome.RUNNING
    table[:, 2] += offset
    assert evaluate(env) is Outcome.SUCCESS
    env.scene[ARENA_SCENE_NAMES[0]].data.root_state_w.torch[:, 2] = DEFAULT_CRITERIA.drop_height_m - 0.01
    assert evaluate(env) is Outcome.FAILURE_DROPPED


@pytest.mark.parametrize("missing", ["asset", "pose", "nominal_root"])
def test_preflight_requires_the_live_support_read(missing: str) -> None:
    env = _randomized_env(0.03)
    name = env.cfg.e2e_support_asset_name
    if missing == "asset":
        del env.scene._entities[name]
    elif missing == "pose":
        del env.scene[name].data.root_pose_w
    else:
        del env.cfg.e2e_support_root_z_m
    with pytest.raises((KeyError, AttributeError)):
        check_readable(env)


@pytest.mark.parametrize(
    "offset_range, drop_height, rejects",
    [
        ((-0.1, 0.1), 0.72, True),
        ((-0.1, 0.1), 0.60, False),
        ((0.1, 0.1), 0.72, False),
    ],
)
def test_preflight_checks_the_whole_range_against_the_effective_failure_plane(
    offset_range: tuple[float, float],
    drop_height: float,
    rejects: bool,
) -> None:
    env = _randomized_env(0.1, offset_range)
    policy = build_policy({"drop_height_m": drop_height})
    assert policy.evaluate(env) is Outcome.SUCCESS
    if rejects:
        with pytest.raises(ValueError, match="drop_height_m"):
            policy.check_readable(env)
    else:
        policy.check_readable(env)


def test_nominal_recording_keeps_its_failure_override() -> None:
    build_policy({"drop_height_m": 0.72}).check_readable(_env(ON_TABLE_Z, 0.0, CONTACT))


def test_evaluate_reads_contact_force_and_env_local_pose() -> None:
    # Apple on the plate, in contact, settled -> success.
    assert evaluate(_env(ON_TABLE_Z, 0.0, CONTACT)) is Outcome.SUCCESS
    # Same pose but no contact force -> not placed.
    assert evaluate(_env(ON_TABLE_Z, 0.0, 0.0)) is Outcome.RUNNING
    # Dropped apple -> failure regardless of contact.
    assert evaluate(_env(0.2, 0.0, CONTACT)) is Outcome.FAILURE_DROPPED


def test_autodata_success_term_matches_motion_and_placement_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resting apple may roll in the plate; only translation gates settling."""

    torch = pytest.importorskip("torch")

    class SceneEntityCfg:
        def __init__(self, name: str) -> None:
            self.name = name

    modules = {
        "isaaclab": ModuleType("isaaclab"),
        "isaaclab.assets": ModuleType("isaaclab.assets"),
        "isaaclab.envs": ModuleType("isaaclab.envs"),
        "isaaclab.managers": ModuleType("isaaclab.managers"),
    }
    modules["isaaclab.assets"].RigidObject = object
    modules["isaaclab.envs"].ManagerBasedRLEnv = object
    modules["isaaclab.managers"].SceneEntityCfg = SceneEntityCfg
    for name, module in modules.items():
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    source = Path(__file__).resolve().parents[1] / "src/e2e_robot_workflows/tasks/static_apple/terminations.py"
    spec = importlib.util.spec_from_file_location("_test_static_apple_terminations", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    object_position = torch.tensor([[0.0, 0.0, 0.77]])
    object_velocity = torch.tensor([[0.0, 0.0, 0.0, 100.0, -100.0, 50.0]])
    target_position = torch.tensor([[0.0, 0.0, 0.75]])
    link_positions = torch.tensor([[[-0.4, 0.0, 0.77], [0.4, 0.0, 0.77]]])
    entities = {
        "object": SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=SimpleNamespace(torch=object_position),
                root_vel_w=SimpleNamespace(torch=object_velocity),
            )
        ),
        "target": SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=target_position))),
        "robot": SimpleNamespace(
            data=SimpleNamespace(
                body_names=["left", "right"],
                body_pos_w=SimpleNamespace(torch=link_positions[..., :3]),
            )
        ),
        "contact": SimpleNamespace(
            data=SimpleNamespace(force_matrix_w=SimpleNamespace(torch=torch.tensor([[[[0.0, 0.0, 2.0]]]])))
        ),
    }

    class Scene:
        env_origins = torch.zeros((1, 3))

        def __getitem__(self, name: str):
            return entities[name]

    scene = Scene()
    env = SimpleNamespace(scene=scene)
    arguments = {
        "placement_radius_m": 0.125,
        "placement_z_m": 0.0625,
        "min_object_height_above_target_m": -0.015625,
        "max_object_height_m": 0.9,
        "min_object_height_m": 0.7,
        "min_target_height_m": 0.7,
        "min_contact_force": 1.0,
        "release_clearance_m": 0.1,
        "max_object_speed": 0.1,
        "task_link_names": ("left", "right"),
        "object_cfg": SceneEntityCfg("object"),
        "robot_cfg": SceneEntityCfg("robot"),
        "target_cfg": SceneEntityCfg("target"),
        "contact_sensor_cfg": SceneEntityCfg("contact"),
    }

    assert module.object_placed_on_target(env, **arguments).item() is True
    object_position[0, 0] = 0.125
    assert module.object_placed_on_target(env, **arguments).item() is True
    object_position[0, 0] = 0.126
    assert module.object_placed_on_target(env, **arguments).item() is False
    object_position[0, 0] = 0.09
    object_position[0, 1] = 0.09
    assert module.object_placed_on_target(env, **arguments).item() is False
    object_position[0, :2] = 0.0
    object_position[0, 2] = 0.8125
    assert module.object_placed_on_target(env, **arguments).item() is True
    object_position[0, 2] = 0.8126
    assert module.object_placed_on_target(env, **arguments).item() is False
    object_position[0, 2] = 0.7421875
    assert module.object_placed_on_target(env, **arguments).item() is True
    object_position[0, 2] = 0.734375
    assert module.object_placed_on_target(env, **arguments).item() is False
    object_position[0, 0] = 0.0
    object_position[0, 2] = 0.77
    object_velocity[0, :3] = 0.09
    assert module.object_placed_on_target(env, **arguments).item() is False
    object_velocity[0, :3] = 0.0
    object_velocity[0, 0] = 1.0
    assert module.object_placed_on_target(env, **arguments).item() is False

    for invalid_links in (
        ("left",),
        ("left", "left"),
        ("", "right"),
        (1, "right"),
        "lr",
    ):
        with pytest.raises(ValueError, match="two distinct non-empty robot links"):
            module.object_placed_on_target(env, **{**arguments, "task_link_names": invalid_links})


def test_evaluate_uses_the_composed_support_surface_height() -> None:
    assert evaluate(_env(ON_TABLE_Z, 0.0, CONTACT, names=ARENA_SCENE_NAMES)) is Outcome.SUCCESS
    assert (
        evaluate(_env(ON_TABLE_Z, 0.0, CONTACT, names=ARENA_SCENE_NAMES, support_surface_z_m=0.96)) is Outcome.RUNNING
    )


def test_evaluate_uses_the_composed_task_link_clearance() -> None:
    assert evaluate(_env(ON_TABLE_Z, 0.0, CONTACT, release_clearance_m=0.10)) is Outcome.SUCCESS
    assert evaluate(_env(ON_TABLE_Z, 0.0, CONTACT, release_clearance_m=0.25)) is Outcome.RUNNING


def test_evaluate_uses_configured_links_when_multiple_robot_pairs_exist() -> None:
    env = _env(ON_TABLE_Z, 0.0, CONTACT)
    robot = env.scene["robot"]
    thumb_positions = robot.data.body_pos_w.torch.copy()
    thumb_positions[0, :, :3] = (5.35, -1.6, ON_TABLE_Z)  # both thumb links remain on the apple
    flange_positions = robot.data.body_pos_w.torch
    robot.body_names = [*G1_TASK_LINK_NAMES, "left_hand_flange", "right_hand_flange"]
    robot.data.body_pos_w = _RootState(np.concatenate((thumb_positions, flange_positions), axis=1))
    env.cfg.e2e_task_link_names = ("left_hand_flange", "right_hand_flange")

    assert evaluate(env) is Outcome.SUCCESS


def test_evaluate_rejects_a_missing_configured_task_link() -> None:
    env = _env(ON_TABLE_Z, 0.0, CONTACT)
    env.cfg.e2e_task_link_names = ("missing_left", "missing_right")

    with pytest.raises(KeyError, match="Configured static-apple task links"):
        evaluate(env)


@pytest.mark.parametrize(
    "task_link_names",
    [("left",), ("left", "left"), ("", "right"), (1, "right"), "lr"],
)
def test_evaluate_rejects_malformed_configured_task_links(task_link_names: object) -> None:
    env = _env(ON_TABLE_Z, 0.0, CONTACT)
    env.cfg.e2e_task_link_names = task_link_names

    with pytest.raises(KeyError, match="two distinct non-empty robot links"):
        evaluate(env)


EMPTY_FORCE_MATRIX = np.zeros((1, 0, 0, 3), dtype=np.float32)  # (N, B=0, M=0, 3)
UNWATCHED_SENSORS = [_RootState(EMPTY_FORCE_MATRIX), None]  # filters match no bodies; no filters at all


@pytest.mark.parametrize("force_matrix", UNWATCHED_SENSORS)
def test_evaluate_treats_a_sensor_watching_nothing_as_no_contact(force_matrix: object) -> None:
    # A mis-wired sensor must not crash a live episode -- report no contact.
    env = _env(ON_TABLE_Z, 0.0, CONTACT)
    env.scene[NATIVE_SCENE_NAMES[2]].data.force_matrix_w = force_matrix
    assert evaluate(env) is Outcome.RUNNING


@pytest.mark.parametrize("names", [NATIVE_SCENE_NAMES, ARENA_SCENE_NAMES])
def test_evaluate_reads_either_runtimes_scene_entity_names(names: tuple[str, str, str]) -> None:
    """The same detector serves the native task and the Arena-composed scene."""

    assert evaluate(_env(ON_TABLE_Z, 0.0, CONTACT, names=names)) is Outcome.SUCCESS
    assert evaluate(_env(ON_TABLE_Z, 0.0, 0.0, names=names)) is Outcome.RUNNING


def test_evaluate_rejects_a_scene_naming_neither_runtimes_entities() -> None:
    """A scene naming neither variant is a wiring bug, not a failed episode."""

    with pytest.raises(KeyError, match="pick_object"):
        evaluate(_env(ON_TABLE_Z, 0.0, CONTACT, names=("apple", "plate", "sensor")))


@pytest.mark.parametrize("names", [NATIVE_SCENE_NAMES, ARENA_SCENE_NAMES])
def test_a_readable_scene_passes_the_check(names: tuple[str, str, str]) -> None:
    check_readable(_env(ON_TABLE_Z, 0.0, 0.0, names=names))


def test_the_check_rejects_a_scene_naming_neither_runtimes_entities() -> None:
    with pytest.raises(KeyError, match="pick_object"):
        check_readable(_env(ON_TABLE_Z, 0.0, 0.0, names=("apple", "plate", "sensor")))


@pytest.mark.parametrize("force_matrix", UNWATCHED_SENSORS)
def test_the_check_rejects_a_sensor_watching_no_contact_pairs(force_matrix: object) -> None:
    """What :func:`evaluate` has to swallow to stay alive, the pre-session check raises on.

    Left to the per-step poll, either mis-wiring reads as "no contact" all session.
    """

    env = _env(ON_TABLE_Z, 0.0, CONTACT)
    env.scene[NATIVE_SCENE_NAMES[2]].data.force_matrix_w = force_matrix
    with pytest.raises(RuntimeError, match="no contact pairs"):
        check_readable(env)

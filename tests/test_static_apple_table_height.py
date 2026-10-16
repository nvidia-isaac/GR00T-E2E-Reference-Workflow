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

"""CPU reset, scoring, and admission contracts for table-height randomization."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pytest

from e2e_robot_workflows.data.dataset_manifest import (
    ARENA_PINK_PROFILE,
    H2_ARENA_PINK_PROFILE,
    NATIVE_FIXED_PROFILE,
    SCENE_RANDOMIZATION_FIELD,
    DatasetContractError,
    build_manifest,
    read_manifest,
    stamp_autodata_dataset,
    validate_dataset,
    write_manifest,
)
from e2e_robot_workflows.environments import STATIC_APPLE_ENVIRONMENT
from e2e_robot_workflows.tasks.static_apple.config import H2_STATIC_APPLE_VARIANT, load_static_apple_environment
from e2e_robot_workflows.tasks.static_apple.table_height import (
    SUPPORT_REPRESENTATION,
    parse_offset_range,
    randomize_support_height,
    resolve_randomization,
    scene_randomization_manifest,
    support_height_offset,
    support_height_offset_value,
)

if importlib.util.find_spec("torch") is not None:
    import torch
else:
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="Torch is in the optional runtime dependency group")

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_NAME = STATIC_APPLE_ENVIRONMENT.asset("support").name
APPLE_NAME = STATIC_APPLE_ENVIRONMENT.asset("pick_object").name
PLATE_NAME = STATIC_APPLE_ENVIRONMENT.asset("destination").name
SUPPORT_ROOT_Z = STATIC_APPLE_ENVIRONMENT.support_position[2]
APPLE_SPAWN_Z = STATIC_APPLE_ENVIRONMENT.spawn_position("pick_object")[2]
PLATE_SPAWN_Z = STATIC_APPLE_ENVIRONMENT.spawn_position("destination")[2]


def _randomization(offset_range: tuple[float, float] = (-0.02, 0.02)) -> Any:
    randomization = resolve_randomization(
        definition=STATIC_APPLE_ENVIRONMENT,
        offset_range_m=offset_range,
    )
    assert randomization is not None
    return randomization


class _Body:
    def __init__(self, position: tuple[float, float, float], num_envs: int) -> None:
        pose = torch.zeros((num_envs, 7))
        pose[:, :3] = torch.tensor(position)
        pose[:, 6] = 1.0
        self.data = SimpleNamespace(root_pose_w=SimpleNamespace(torch=pose))
        self.velocity_writes: list[torch.Tensor] = []

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.data.root_pose_w.torch[env_ids] = root_pose

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.velocity_writes.append(root_velocity.clone())


class _Scene:
    def __init__(self, bodies: dict[str, _Body], origins: torch.Tensor) -> None:
        self._bodies = bodies
        self.env_origins = origins
        self.num_envs = origins.shape[0]

    def __getitem__(self, name: str) -> _Body:
        return self._bodies[name]


def _env(num_envs: int = 1, *, origins: torch.Tensor | None = None, movable: bool = True) -> Any:
    resolved_origins = torch.zeros((num_envs, 3)) if origins is None else origins
    bodies = {
        SUPPORT_NAME: _Body((0.0, 0.55, SUPPORT_ROOT_Z), num_envs),
        APPLE_NAME: _Body((-0.15, 0.30, APPLE_SPAWN_Z), num_envs),
        PLATE_NAME: _Body((0.05, 0.30, PLATE_SPAWN_Z), num_envs),
    }
    for body in bodies.values():
        body.data.root_pose_w.torch[:, :3] += resolved_origins
    cfg = SimpleNamespace(e2e_support_surface_z_m=STATIC_APPLE_ENVIRONMENT.support_surface_z)
    if movable:
        cfg.e2e_support_asset_name = SUPPORT_NAME
        cfg.e2e_support_root_z_m = SUPPORT_ROOT_Z
    return SimpleNamespace(scene=_Scene(bodies, resolved_origins), cfg=cfg)


def _heights(env: Any, env_index: int = 0) -> dict[str, float]:
    origin_z = float(env.scene.env_origins[env_index, 2])
    return {
        name: float(env.scene[name].data.root_pose_w.torch[env_index, 2]) - origin_z
        for name in (SUPPORT_NAME, APPLE_NAME, PLATE_NAME)
    }


def test_an_absent_or_zero_range_keeps_the_nominal_scene() -> None:
    assert parse_offset_range(None) is None
    assert parse_offset_range([0.0, 0.0]) is None
    # Equal non-zero bounds are a fixed-height probe, not a disabled feature.
    assert parse_offset_range([0.05, 0.05]) == (0.05, 0.05)
    assert (
        resolve_randomization(
            definition=STATIC_APPLE_ENVIRONMENT,
            offset_range_m=[0.0, 0.0],
        )
        is None
    )


@pytest.mark.parametrize("values", [[0.01], [float("nan"), 0.01], [0.02, -0.02]])
def test_a_malformed_range_is_rejected(values: object) -> None:
    with pytest.raises(ValueError, match="table height offset range"):
        parse_offset_range(values)  # type: ignore[arg-type]


def test_a_range_that_drops_placements_through_the_failure_plane_is_rejected() -> None:
    # Supported placements sit within min_supported_height_below_support_m of the tabletop;
    # below failure_drop_height_m the recorder calls the same apple dropped.
    fatal_offset = -(
        STATIC_APPLE_ENVIRONMENT.support_surface_z
        - STATIC_APPLE_ENVIRONMENT.min_supported_height_below_support_m
        - STATIC_APPLE_ENVIRONMENT.failure_drop_height_m
    )
    with pytest.raises(ValueError, match="failure plane"):
        resolve_randomization(
            definition=STATIC_APPLE_ENVIRONMENT,
            offset_range_m=[fatal_offset, 0.02],
        )


@requires_torch
def test_a_reset_moves_the_table_apple_and_plate_by_the_same_offset() -> None:
    env = _env()
    randomize_support_height(env, torch.tensor([0]), _randomization())

    heights = _heights(env)
    offset = heights[SUPPORT_NAME] - SUPPORT_ROOT_Z
    assert -0.02 <= offset <= 0.02
    assert heights[APPLE_NAME] == pytest.approx(APPLE_SPAWN_Z + offset)
    assert heights[PLATE_NAME] == pytest.approx(PLATE_SPAWN_Z + offset)


@requires_torch
def test_a_reset_preserves_the_freshly_sampled_object_xy_and_rotation() -> None:
    env = _env()
    apple = env.scene[APPLE_NAME]
    apple.data.root_pose_w.torch[0, :2] = torch.tensor([-0.17, 0.32])
    apple.data.root_pose_w.torch[0, 3:] = torch.tensor([0.0, 0.0, 0.3827, 0.9239])

    randomize_support_height(env, torch.tensor([0]), _randomization())

    assert apple.data.root_pose_w.torch[0, :2].tolist() == pytest.approx([-0.17, 0.32])
    assert apple.data.root_pose_w.torch[0, 3:].tolist() == pytest.approx([0.0, 0.0, 0.3827, 0.9239])


@requires_torch
def test_repeated_resets_do_not_accumulate_height() -> None:
    env = _env()
    randomization = _randomization((0.02, 0.02))

    randomize_support_height(env, torch.tensor([0]), randomization)
    randomize_support_height(env, torch.tensor([0]), randomization)

    assert _heights(env)[SUPPORT_NAME] == pytest.approx(SUPPORT_ROOT_Z + 0.02)
    assert _heights(env)[APPLE_NAME] == pytest.approx(APPLE_SPAWN_Z + 0.02)


@requires_torch
def test_a_reset_selects_environments_and_respects_their_origins() -> None:
    origins = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.5]])
    env = _env(num_envs=2, origins=origins)
    table = env.scene[SUPPORT_NAME].data.root_pose_w.torch

    randomize_support_height(env, torch.tensor([1]), _randomization((0.02, 0.02)))

    assert _heights(env, 0)[SUPPORT_NAME] == pytest.approx(SUPPORT_ROOT_Z)
    assert _heights(env, 1)[SUPPORT_NAME] == pytest.approx(SUPPORT_ROOT_Z + 0.02)
    assert float(table[1, 2]) == pytest.approx(0.5 + SUPPORT_ROOT_Z + 0.02)


@requires_torch
def test_a_reset_zeroes_dynamic_velocities_but_never_commands_kinematic_velocity() -> None:
    env = _env()
    randomize_support_height(env, torch.tensor([0]), _randomization())

    assert env.scene[SUPPORT_NAME].velocity_writes == []
    for name in (APPLE_NAME, PLATE_NAME):
        writes = env.scene[name].velocity_writes
        assert len(writes) == 1
        assert torch.count_nonzero(writes[0]) == 0


@requires_torch
def test_none_selects_all_environments_and_an_empty_selection_writes_nothing() -> None:
    empty = _env()
    randomize_support_height(empty, torch.tensor([], dtype=torch.int64), _randomization())
    assert _heights(empty) == pytest.approx(
        {SUPPORT_NAME: SUPPORT_ROOT_Z, APPLE_NAME: APPLE_SPAWN_Z, PLATE_NAME: PLATE_SPAWN_Z}
    )
    assert empty.scene[SUPPORT_NAME].velocity_writes == []

    unselected = _env(num_envs=2)
    randomize_support_height(unselected, None, _randomization((-0.03, -0.03)))
    assert _heights(unselected, 0)[SUPPORT_NAME] == pytest.approx(SUPPORT_ROOT_Z - 0.03)
    assert _heights(unselected, 1)[SUPPORT_NAME] == pytest.approx(SUPPORT_ROOT_Z - 0.03)


@requires_torch
def test_the_live_reader_reports_the_table_the_episode_actually_has() -> None:
    env = _env()
    randomize_support_height(env, torch.tensor([0]), _randomization((0.02, 0.02)))

    assert support_height_offset(env).tolist() == pytest.approx([0.02], abs=1e-6)
    assert support_height_offset_value(env) == pytest.approx(0.02, abs=1e-6)


@requires_torch
def test_the_live_reader_reports_no_offset_without_a_recorded_table() -> None:
    unconfigured = _env(movable=False)
    assert support_height_offset(unconfigured) == 0.0
    assert support_height_offset_value(unconfigured) == 0.0

    static_fixture = _env()
    static_fixture.scene[SUPPORT_NAME].data = None
    with pytest.raises(AttributeError):
        support_height_offset(static_fixture)


def _predicate_modules(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Load the scoring predicates against stub Isaac Lab types, which need no simulator."""

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

    loaded = {}
    for name in ("terminations", "observations"):
        source = REPO_ROOT / f"src/e2e_robot_workflows/tasks/static_apple/{name}.py"
        spec = importlib.util.spec_from_file_location(f"_test_static_apple_{name}", source)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded[name] = module
    return SimpleNamespace(**loaded, scene_entity_cfg=SceneEntityCfg)


def _scored_env(apple_z: float, table_offset: float) -> Any:
    """Stub scoring inputs; zero offset uses an unrecorded nominal table."""

    env = _env(movable=table_offset != 0.0)
    if table_offset != 0.0:
        randomize_support_height(env, torch.tensor([0]), _randomization((table_offset, table_offset)))
    apple = env.scene[APPLE_NAME]
    plate = env.scene[PLATE_NAME]
    apple.data.root_pose_w.torch[0, :3] = torch.tensor([0.05, 0.30, apple_z])
    apple.data.root_vel_w = SimpleNamespace(torch=torch.zeros((1, 6)))
    entities = {
        "object": apple,
        "target": plate,
        "robot": SimpleNamespace(
            data=SimpleNamespace(
                body_names=["left", "right"],
                body_pos_w=SimpleNamespace(torch=torch.tensor([[[-0.4, 0.0, 1.0], [0.4, 0.0, 1.0]]])),
            )
        ),
        "contact": SimpleNamespace(
            data=SimpleNamespace(force_matrix_w=SimpleNamespace(torch=torch.tensor([[[[0.0, 0.0, 2.0]]]])))
        ),
    }
    entities["object"].data.root_pos_w = SimpleNamespace(torch=apple.data.root_pose_w.torch[:, :3])
    entities["target"].data.root_pos_w = SimpleNamespace(torch=plate.data.root_pose_w.torch[:, :3])
    scene = env.scene
    scene._bodies.update(entities)  # noqa: SLF001 - the fake scene's own storage
    return env


@requires_torch
def test_supported_placement_follows_the_table_it_rests_on(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _predicate_modules(monkeypatch)
    nominal = STATIC_APPLE_ENVIRONMENT.support_surface_z
    arguments = {
        "placement_radius_m": STATIC_APPLE_ENVIRONMENT.success_placement_xy_m,
        "placement_z_m": STATIC_APPLE_ENVIRONMENT.success_placement_z_m,
        "min_object_height_above_target_m": STATIC_APPLE_ENVIRONMENT.min_object_height_above_target_m,
        "max_object_height_m": nominal + STATIC_APPLE_ENVIRONMENT.max_object_height_above_support_m,
        "min_object_height_m": nominal - STATIC_APPLE_ENVIRONMENT.min_supported_height_below_support_m,
        "min_target_height_m": nominal - STATIC_APPLE_ENVIRONMENT.min_supported_height_below_support_m,
        "min_contact_force": 1.0,
        "release_clearance_m": 0.1,
        "max_object_speed": 0.1,
        "task_link_names": ("left", "right"),
        "object_cfg": module.scene_entity_cfg("object"),
        "robot_cfg": module.scene_entity_cfg("robot"),
        "target_cfg": module.scene_entity_cfg("target"),
        "contact_sensor_cfg": module.scene_entity_cfg("contact"),
    }

    raised = 0.15
    # An apple resting on a table raised past the nominal resting band still counts.
    env = _scored_env(PLATE_SPAWN_Z + raised + 0.01, raised)
    assert module.terminations.object_placed_on_target(env, **arguments).item() is True
    # The same apple over a nominal table is held too high to be resting on it.
    env = _scored_env(PLATE_SPAWN_Z + raised + 0.01, 0.0)
    assert module.terminations.object_placed_on_target(env, **arguments).item() is False


@requires_torch
def test_a_taller_table_does_not_read_an_untouched_apple_as_lifted(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _predicate_modules(monkeypatch)
    lift_threshold = APPLE_SPAWN_Z + STATIC_APPLE_ENVIRONMENT.pickup_lift_clearance_m
    raised = 0.10
    arguments = {
        "task_link_name": "left",
        "grasp_distance_m": 1.0,
        "min_object_height_m": lift_threshold,
        "object_cfg": module.scene_entity_cfg("object"),
        "robot_cfg": module.scene_entity_cfg("robot"),
    }

    env = _scored_env(APPLE_SPAWN_Z + raised, raised)
    assert module.observations.object_held_by_link(env, **arguments).item() is False
    env = _scored_env(APPLE_SPAWN_Z + raised + STATIC_APPLE_ENVIRONMENT.pickup_lift_clearance_m + 0.01, raised)
    assert module.observations.object_held_by_link(env, **arguments).item() is True


def _write_randomized_dataset(path: Path, *, offset: float | None = 0.02) -> None:
    steps = 3
    definition = STATIC_APPLE_ENVIRONMENT
    roles = ("pick_object", "destination") + (() if offset is None else ("support",))
    with h5py.File(path, "w") as dataset:
        episode = dataset.create_group("data/demo_0")
        episode.create_dataset("actions", shape=(steps, ARENA_PINK_PROFILE.width), dtype="f4")
        for role in roles:
            asset = definition.asset(role)
            position = definition.support_position if role == "support" else definition.spawn_position(role)
            pose = np.array([[*position, *asset.rotation_xyzw]], dtype=np.float32)
            pose[:, 2] += offset or 0.0
            for prefix, count in (("initial_state", 1), ("states", steps)):
                root = f"{prefix}/rigid_object/{asset.name}"
                episode.create_dataset(f"{root}/root_pose", data=np.repeat(pose, count, axis=0))
                episode.create_dataset(f"{root}/root_velocity", data=np.zeros((count, 6), dtype=np.float32))
    write_manifest(
        path,
        build_manifest(
            producer="recording",
            profile=ARENA_PINK_PROFILE,
            seed=42,
            scene_randomization=None if offset is None else scene_randomization_manifest(_randomization(), seed=42),
        ),
    )


@pytest.mark.parametrize("offset", [None, -0.02, 0.02])
def test_nominal_and_randomized_recordings_round_trip(tmp_path: Path, offset: float | None) -> None:
    path = tmp_path / "recording.hdf5"
    _write_randomized_dataset(path, offset=offset)
    manifest = validate_dataset(path, require_lane="arena")
    block = manifest["environment"].get(SCENE_RANDOMIZATION_FIELD)
    if offset is None:
        assert block is None
    else:
        assert block == scene_randomization_manifest(_randomization(), seed=42)
        assert block["support_representation"] == SUPPORT_REPRESENTATION
        with h5py.File(path, "r+") as dataset:
            dataset[f"data/demo_0/states/rigid_object/{SUPPORT_NAME}/root_pose"][:, 3:] *= -1
        validate_dataset(path)  # Quaternion sign does not change the pose.


@pytest.mark.parametrize(
    "mutation",
    [
        {"support_representation": "visual_only"},
        {"support_asset": "other_table"},
        {"nominal_support_root_z_m": 0.1},
        {"nominal_support_surface_z_m": 123.0},
        {"nominal_support_root_z_m": False},
        {"object_spawn_z_m": {}},
        {"table_height_offset_range_m": [0.02, -0.02]},
        {"table_height_offset_range_m": [0.0, 0.0]},
        {"table_height_offset_range_m": ["-0.02", "0.02"]},
        {"table_height_offset_range_m": [False, True]},
        {"seed": 7},
        {"hand_tuned": True},
    ],
)
def test_admission_rejects_inconsistent_metadata(tmp_path: Path, mutation: dict[str, Any]) -> None:
    path = tmp_path / "recording.hdf5"
    _write_randomized_dataset(path)
    manifest = read_manifest(path)
    manifest["environment"][SCENE_RANDOMIZATION_FIELD].update(mutation)
    write_manifest(path, manifest)
    with pytest.raises(DatasetContractError, match="scene randomization"):
        validate_dataset(path)


@pytest.mark.parametrize("omit", [False, True])
def test_unseeded_randomization_preserves_nullable_run_seed(tmp_path: Path, omit: bool) -> None:
    path = tmp_path / "recording.hdf5"
    _write_randomized_dataset(path)
    manifest = read_manifest(path)
    manifest["seed"] = None
    manifest["environment"][SCENE_RANDOMIZATION_FIELD]["seed"] = None
    if omit:
        del manifest["seed"]
    write_manifest(path, manifest)
    assert validate_dataset(path)["environment"][SCENE_RANDOMIZATION_FIELD]["seed"] is None


@pytest.mark.parametrize("omit", [False, True])
def test_seeded_randomization_rejects_an_unknown_run_seed(tmp_path: Path, omit: bool) -> None:
    path = tmp_path / "recording.hdf5"
    _write_randomized_dataset(path)
    manifest = read_manifest(path)
    if omit:
        del manifest["seed"]
    else:
        manifest["seed"] = None
    write_manifest(path, manifest)
    with pytest.raises(DatasetContractError, match="scene randomization"):
        validate_dataset(path)


@pytest.mark.parametrize("omit", [False, True])
def test_a_recorded_table_requires_nonnull_metadata(tmp_path: Path, omit: bool) -> None:
    path = tmp_path / "recording.hdf5"
    _write_randomized_dataset(path)
    manifest = read_manifest(path)
    if omit:
        del manifest["environment"][SCENE_RANDOMIZATION_FIELD]
    else:
        manifest["environment"][SCENE_RANDOMIZATION_FIELD] = None
    write_manifest(path, manifest)
    with pytest.raises(DatasetContractError, match="scene randomization"):
        validate_dataset(path)


@pytest.mark.parametrize("profile", [H2_ARENA_PINK_PROFILE, NATIVE_FIXED_PROFILE])
def test_the_height_contract_is_bound_to_its_lane_and_variant(profile: Any) -> None:
    with pytest.raises(DatasetContractError, match="scene randomization"):
        build_manifest(
            producer="recording",
            profile=profile,
            seed=42,
            scene_randomization=scene_randomization_manifest(_randomization(), seed=42),
        )
    if profile == H2_ARENA_PINK_PROFILE:
        randomization = resolve_randomization(
            definition=load_static_apple_environment(H2_STATIC_APPLE_VARIANT), offset_range_m=[-0.02, 0.02]
        )
        manifest = build_manifest(
            producer="recording",
            profile=profile,
            seed=42,
            scene_randomization=scene_randomization_manifest(randomization, seed=42),
        )
        assert manifest["environment"][SCENE_RANDOMIZATION_FIELD]["seed"] == 42


@pytest.mark.parametrize(
    "prefix, field, count",
    [
        ("initial_state", "root_pose", None),
        ("initial_state", "root_pose", 2),
        ("initial_state", "root_velocity", None),
        ("states", "root_pose", 1),
        ("states", "root_velocity", None),
        ("states", "root_velocity", 1),
    ],
)
def test_table_state_must_be_complete_and_aligned(tmp_path: Path, prefix: str, field: str, count: int | None) -> None:
    path = tmp_path / "recording.hdf5"
    _write_randomized_dataset(path)
    with h5py.File(path, "r+") as dataset:
        key = f"data/demo_0/{prefix}/rigid_object/{SUPPORT_NAME}/{field}"
        value = dataset[key][:1]
        del dataset[key]
        if count is not None:
            dataset.create_dataset(key, data=np.repeat(value, count, axis=0))
    with pytest.raises(DatasetContractError, match="must have shape"):
        validate_dataset(path)


@pytest.mark.parametrize(
    "prefix, name, field, column, value, error",
    [
        ("states", SUPPORT_NAME, "root_pose", 0, 1.0, "Z-only"),
        ("states", SUPPORT_NAME, "root_pose", 2, 0.03, "Z-only"),
        ("initial_state", SUPPORT_NAME, "root_pose", 0, 1.0, "Z-only"),
        ("initial_state", SUPPORT_NAME, "root_pose", 2, 0.05, "outside the declared"),
        ("states", SUPPORT_NAME, "root_pose", slice(3, 7), [0, 0, 1, 0], "Z-only"),
        ("states", SUPPORT_NAME, "root_pose", slice(3, 7), [0, 0, 0, 0], "normalized"),
        ("states", SUPPORT_NAME, "root_pose", 2, float("nan"), "finite"),
        ("states", SUPPORT_NAME, "root_velocity", 0, 1.0, "velocity must stay zero"),
        ("initial_state", APPLE_NAME, "root_pose", 2, APPLE_SPAWN_Z - 0.02, "initial height"),
    ],
)
def test_admission_rejects_invalid_rigid_state(tmp_path: Path, prefix, name, field, column, value, error) -> None:
    path = tmp_path / "recording.hdf5"
    _write_randomized_dataset(path)
    with h5py.File(path, "r+") as dataset:
        dataset[f"data/demo_0/{prefix}/rigid_object/{name}/{field}"][-1, column] = value
    with pytest.raises(DatasetContractError, match=error):
        validate_dataset(path)


@pytest.mark.parametrize("offset", [None, -0.01])
def test_generation_owns_its_distribution_and_seed(tmp_path: Path, offset: float | None) -> None:
    source, target = tmp_path / "source.hdf5", tmp_path / "target.hdf5"
    _write_randomized_dataset(source)
    _write_randomized_dataset(target, offset=offset)
    block = None if offset is None else scene_randomization_manifest(_randomization((-0.01, -0.01)), seed=7)
    manifest = stamp_autodata_dataset(
        target, source_dataset=source, stage="generation", seed=7, scene_randomization=block
    )
    assert manifest["seed"] == 7
    assert manifest["environment"].get(SCENE_RANDOMIZATION_FIELD) == block

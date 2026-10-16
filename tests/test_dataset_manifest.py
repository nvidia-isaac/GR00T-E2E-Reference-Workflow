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

"""CPU contracts for explicit native and Arena dataset routing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pytest

import e2e_robot_workflows.data.dataset_manifest as dataset_manifest_module
from e2e_robot_workflows.data.dataset_manifest import (
    ARENA_JOINT_PROFILE,
    ARENA_PINK_PROFILE,
    H2_ARENA_JOINT_PROFILE,
    H2_ARENA_PINK_PROFILE,
    MANIFEST_ATTRIBUTE,
    NATIVE_AGILE_PROFILE,
    NATIVE_FIXED_PROFILE,
    ArenaEpisodeRejection,
    DatasetContractError,
    build_manifest,
    default_recording_dataset_file,
    detect_arena_manipulation_hand,
    file_sha256,
    main,
    manifest_control_hz,
    migrate_legacy_recording,
    read_manifest,
    resolve_recording_lane,
    resolve_recording_profile,
    stamp_autodata_dataset,
    stamp_recording_dataset,
    validate_arena_recording_contract,
    validate_dataset,
    write_manifest,
)
from e2e_robot_workflows.environments import STATIC_APPLE_ENVIRONMENT
from e2e_robot_workflows.tasks.static_apple.config import H2_STATIC_APPLE_VARIANT, load_static_apple_environment
from e2e_robot_workflows.tasks.static_apple.table_height import resolve_randomization, scene_randomization_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_dataset(path: Path, widths: list[int]) -> None:
    with h5py.File(path, "w") as dataset:
        data = dataset.create_group("data")
        for index, width in enumerate(widths):
            episode = data.create_group(f"demo_{index}")
            episode.create_dataset("actions", shape=(3, width), dtype="f4")


def _write_arena_source(
    path: Path,
    *,
    pickup_signal: tuple[bool, ...] = (False, True, True),
    add_manifest: bool = True,
) -> None:
    steps = len(pickup_signal)
    with h5py.File(path, "w") as dataset:
        episode = dataset.create_group("data/demo_0")
        episode.attrs["success"] = True
        actions = np.zeros((steps, ARENA_PINK_PROFILE.width), dtype=np.float32)
        actions[:, 8] = 1.0
        actions[:, 15] = 1.0
        episode.create_dataset("actions", data=actions)
        datagen = episode.create_group("obs/datagen_info")
        for hand in ("left", "right"):
            datagen.create_dataset(f"eef_pose/{hand}", data=np.tile(np.eye(4), (steps, 1, 1)))
            datagen.create_dataset(f"target_eef_pose/{hand}", data=np.tile(np.eye(4), (steps, 1, 1)))
            signal = pickup_signal if hand == "right" else (False,) * steps
            datagen.create_dataset(f"subtask_term_signals/{hand}_hand_holds_object", data=signal)
        datagen.create_dataset("object_pose/pick_object", data=np.tile(np.eye(4), (steps, 1, 1)))
    if add_manifest:
        write_manifest(
            path,
            build_manifest(producer="recording", profile=ARENA_PINK_PROFILE, repo_root=REPO_ROOT),
        )


def _add_h2_initial_state(episode):
    from e2e_robot_workflows.controllers.h2_feedback import feedback_shapes
    from e2e_robot_workflows.robots.h2 import H2_LIVE_JOINT_COUNT, load_h2_robot_config

    cfg = load_h2_robot_config()
    shapes = feedback_shapes(
        num_envs=1, driven=len(cfg.driven_joint_names), depth=cfg.history_length, command_dim=cfg.command_dim
    )
    for name, shape in shapes.items():
        episode.create_dataset(f"initial_state/h2_agile_feedback/{name}", data=np.zeros(shape, dtype=np.float32))
    for name in ("position", "velocity"):
        episode.create_dataset(
            f"initial_state/h2_actuator_targets/{name}", data=np.zeros((1, H2_LIVE_JOINT_COUNT), dtype=np.float32)
        )


def _write_h2_dataset(path: Path, *, steps: int = 2, manipulation_hand: str = "left") -> None:
    with h5py.File(path, "w") as dataset:
        episode = dataset.create_group("data/demo_0")
        _add_h2_initial_state(episode)
        actions = np.zeros((steps, H2_ARENA_PINK_PROFILE.width), dtype=np.float32)
        actions[:, 6] = 1.0
        actions[:, 13] = 1.0
        actions[:, -3:] = (0.1, -0.2, 0.3)
        episode.create_dataset("actions", data=actions)
        episode.create_dataset("joint_pos_target", shape=(steps, 75), dtype="f4")
        episode.create_dataset("navigate_cmd", data=actions[:, -3:])
        episode.create_dataset("obs/robot_joint_pos", shape=(steps, 75), dtype="f4")
        episode.create_dataset("camera_obs/robot_head_cam_rgb", shape=(steps, 480, 640, 3), dtype="u1")
        pick_state = np.zeros((steps, 7), dtype=np.float32)
        pick_state[:, 2] = 1.00
        target_state = np.zeros((steps, 7), dtype=np.float32)
        target_state[:, 2] = 0.96
        episode.create_dataset("states/rigid_object/pick_object/root_pose", data=pick_state)
        episode.create_dataset("states/rigid_object/placement_target/root_pose", data=target_state)
    write_manifest(
        path,
        build_manifest(
            producer="recording",
            profile=H2_ARENA_PINK_PROFILE,
            manipulation_hand=manipulation_hand,
            repo_root=REPO_ROOT,
        ),
    )


def _add_randomized_h2_table(path: Path, *, offset: float, settling_displacement: float = 0.0) -> None:
    definition = load_static_apple_environment(H2_STATIC_APPLE_VARIANT)
    with h5py.File(path, "r+") as dataset:
        episode = dataset["data/demo_0"]
        steps = episode["actions"].shape[0]
        for role in ("support", "pick_object", "destination"):
            asset = definition.asset(role)
            position = definition.support_position if role == "support" else definition.spawn_position(role)
            pose = np.array([[*position, *asset.rotation_xyzw]], dtype=np.float32)
            pose[:, 2] += offset + (settling_displacement if role == "pick_object" else 0.0)
            for prefix, count in (("initial_state", 1), ("states", steps)):
                key = f"{prefix}/rigid_object/{asset.name}/root_pose"
                if key in episode:
                    del episode[key]
                episode.create_dataset(key, data=np.repeat(pose, count, axis=0))
                episode.create_dataset(
                    f"{prefix}/rigid_object/{asset.name}/root_velocity", data=np.zeros((count, 6), dtype=np.float32)
                )
    write_manifest(
        path,
        build_manifest(
            producer="recording",
            profile=H2_ARENA_PINK_PROFILE,
            manipulation_hand=read_manifest(path)["task"]["manipulation_hand"],
            seed=42,
            scene_randomization=scene_randomization_manifest(
                resolve_randomization(definition=definition, offset_range_m=[-0.03, 0.03]), seed=42
            ),
        ),
    )


@pytest.mark.parametrize("settling_displacement", [0.0, -0.015, -0.2, 0.02])
def test_randomized_h2_admits_supported_ready_poses(tmp_path: Path, settling_displacement: float) -> None:
    path = tmp_path / "ready.hdf5"
    _write_h2_dataset(path)
    _add_randomized_h2_table(path, offset=0.03, settling_displacement=settling_displacement)
    if settling_displacement in (0.0, -0.015):
        validate_dataset(path, qualification=True)
    else:
        with pytest.raises(DatasetContractError, match="height.*table"):
            validate_dataset(path, qualification=True)


def _write_h2_joint_dataset(path: Path, *, steps: int = 2) -> None:
    from e2e_robot_workflows.robots.h2 import load_h2_pink_contract

    contract = load_h2_pink_contract()
    target_names = contract.arm_joint_names + contract.hand_joint_order
    target_indices = [contract.expected_joint_order.index(name) for name in target_names]
    actions = np.zeros((steps, H2_ARENA_JOINT_PROFILE.width), dtype=np.float32)
    actions[:, :58] = np.linspace(0.0, 0.1, 58, dtype=np.float32)
    actions[:, -3:] = (0.1, -0.2, 0.3)
    joint_targets = np.zeros((steps, 75), dtype=np.float32)
    joint_targets[:, target_indices] = actions[:, :58]
    with h5py.File(path, "w") as dataset:
        episode = dataset.create_group("data/demo_0")
        _add_h2_initial_state(episode)
        episode.create_dataset("actions", data=actions)
        episode.create_dataset("joint_pos_target", data=joint_targets)
        episode.create_dataset("navigate_cmd", data=actions[:, -3:])
        episode.create_dataset("obs/robot_joint_pos", shape=(steps, 75), dtype="f4")
        episode.create_dataset("camera_obs/robot_head_cam_rgb", shape=(steps, 480, 640, 3), dtype="u1")
    write_manifest(
        path,
        build_manifest(
            producer="recording",
            profile=H2_ARENA_JOINT_PROFILE,
            manipulation_hand="left",
            repo_root=REPO_ROOT,
        ),
    )


def _add_h2_transfer_datagen(path: Path, *, manipulation_hand: str = "left") -> None:
    with h5py.File(path, "r+") as dataset:
        episode = dataset["data/demo_0"]
        episode.attrs["success"] = True
        steps = episode["actions"].shape[0]
        if steps < 4:
            raise ValueError("H2 handover fixture needs pickup, receive, and placement frames")
        datagen = episode.create_group("obs/datagen_info")
        identity = np.tile(np.eye(4, dtype=np.float32), (steps, 1, 1))
        for hand, x_position in (("left", 0.25), ("right", -0.25)):
            eef_pose = identity.copy()
            eef_pose[:, 0, 3] = x_position
            datagen.create_dataset(f"eef_pose/{hand}", data=eef_pose)
            datagen.create_dataset(f"target_eef_pose/{hand}", data=identity)
            signal = np.zeros(steps, dtype=bool)
            if hand == manipulation_hand:
                signal[1:] = True
            else:
                signal[2:] = True
            datagen.create_dataset(f"subtask_term_signals/{hand}_hand_holds_object", data=signal)
            datagen.create_dataset(f"grasp_signals/{hand}_hand_holds_object", data=signal)
        pick_pose = identity.copy()
        pick_pose[:, 2, 3] = 0.04
        datagen.create_dataset("object_pose/pick_object", data=pick_pose)
        target_pose = identity.copy()
        target_pose[:, 2, 3] = 0.0
        datagen.create_dataset("object_pose/placement_target", data=target_pose)


@pytest.mark.parametrize("offset", [-0.03, 0.03])
def test_h2_successful_source_admission_follows_its_recorded_table(tmp_path: Path, offset: float) -> None:
    path = tmp_path / "randomized-transfer.hdf5"
    _write_h2_dataset(path, steps=4)
    _add_randomized_h2_table(path, offset=offset)
    _add_h2_transfer_datagen(path)
    definition = load_static_apple_environment(H2_STATIC_APPLE_VARIANT)
    surface = definition.support_surface_z + offset
    minimum = surface - definition.min_supported_height_below_support_m
    maximum = surface + definition.max_object_height_above_support_m
    valid_pick_z = maximum - 0.01 if offset > 0 else minimum + 0.03
    invalid_pick_z = minimum + 0.015 if offset > 0 else maximum + 0.005
    for valid, pick_z in ((True, valid_pick_z), (False, invalid_pick_z)):
        with h5py.File(path, "r+") as dataset:
            for name, z in (("pick_object", pick_z), ("placement_target", pick_z - 0.02)):
                dataset[f"data/demo_0/states/rigid_object/{name}/root_pose"][-1, :3] = [0, 0, z]
        if valid:
            validate_arena_recording_contract(path, qualification=True, require_success_boundary=True)
        else:
            with pytest.raises(DatasetContractError, match="on the placement target"):
                validate_arena_recording_contract(path, qualification=True, require_success_boundary=True)


@pytest.mark.parametrize(
    ("requested", "alias", "expected"),
    [(None, False, "native"), ("native", False, "native"), ("arena", False, "arena"), (None, True, "arena")],
)
def test_recording_lane_preserves_native_default_and_arena_alias(
    requested: str | None,
    alias: bool,
    expected: str,
) -> None:
    assert resolve_recording_lane(requested, arena_alias=alias) == expected


def test_recording_lane_rejects_conflicting_alias() -> None:
    with pytest.raises(DatasetContractError, match="conflicts"):
        resolve_recording_lane("native", arena_alias=True)


@pytest.mark.parametrize("lane", ["native", "arena"])
def test_default_recording_filename_includes_the_resolved_lane(lane: str) -> None:
    recorded_at = datetime(2026, 9, 1, 22, 4, 5)

    assert default_recording_dataset_file(lane, timestamp=recorded_at) == (
        f"./datasets/e2e_ra_dataset_{lane}_20260901_220405.hdf5"
    )


def test_default_recording_filename_rejects_an_unknown_lane() -> None:
    with pytest.raises(DatasetContractError, match="unsupported recording lane"):
        default_recording_dataset_file("other", timestamp=datetime(2026, 9, 1))


@pytest.mark.parametrize(
    ("lane", "wbc_mode", "embodiment", "expected"),
    [
        ("native", "fixed", None, NATIVE_FIXED_PROFILE),
        ("native", "agile", None, NATIVE_AGILE_PROFILE),
        ("arena", None, "g1_wbc_agile_pink", ARENA_PINK_PROFILE),
        ("arena", None, "g1_wbc_agile_joint", ARENA_JOINT_PROFILE),
    ],
)
def test_profiles_are_selected_from_explicit_lane_details(lane, wbc_mode, embodiment, expected) -> None:
    assert resolve_recording_profile(lane, wbc_mode=wbc_mode, arena_embodiment=embodiment) == expected


@pytest.mark.parametrize(
    "profile",
    [NATIVE_FIXED_PROFILE, NATIVE_AGILE_PROFILE, ARENA_PINK_PROFILE, ARENA_JOINT_PROFILE],
)
def test_manifest_routes_and_validates_each_recording_profile(tmp_path: Path, profile) -> None:
    dataset_path = tmp_path / "source.hdf5"
    _write_dataset(dataset_path, [profile.width, profile.width])
    manifest = build_manifest(producer="recording", profile=profile, repo_root=REPO_ROOT)
    write_manifest(dataset_path, manifest)

    assert validate_dataset(dataset_path, require_lane=profile.lane, require_producer="recording") == manifest


def test_missing_manifest_requires_explicit_migration(tmp_path: Path) -> None:
    dataset_path = tmp_path / "legacy.hdf5"
    _write_dataset(dataset_path, [28])

    with pytest.raises(DatasetContractError, match="migrate it with an explicit lane"):
        read_manifest(dataset_path)


def test_manifest_rejects_an_empty_data_group(tmp_path: Path) -> None:
    dataset_path = tmp_path / "empty.hdf5"
    with h5py.File(dataset_path, "w") as dataset:
        dataset.create_group("data")
    write_manifest(
        dataset_path,
        build_manifest(producer="recording", profile=NATIVE_FIXED_PROFILE, repo_root=REPO_ROOT),
    )

    with pytest.raises(DatasetContractError, match="contains no episodes"):
        validate_dataset(dataset_path)


def test_manifest_rejects_lane_layout_disagreement(tmp_path: Path) -> None:
    dataset_path = tmp_path / "wrong-lane.hdf5"
    _write_dataset(dataset_path, [ARENA_PINK_PROFILE.width])
    manifest = build_manifest(producer="recording", profile=ARENA_PINK_PROFILE, repo_root=REPO_ROOT)
    manifest["lane"] = "native"
    write_manifest(dataset_path, manifest)

    with pytest.raises(DatasetContractError, match="lane and action profile disagree"):
        validate_dataset(dataset_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest.__setitem__("environment", {}), "environment"),
        (lambda manifest: manifest.__setitem__("runtime", {}), "runtime pins"),
        (lambda manifest: manifest["repository"].__setitem__("revision", "not-a-sha"), "repository state"),
        (lambda manifest: manifest.__setitem__("provenance", {}), "provenance"),
        (lambda manifest: manifest["provenance"].__setitem__("status", "derived"), "stage and provenance"),
        (lambda manifest: manifest.__setitem__("seed", True), "seed"),
    ],
)
def test_manifest_rejects_malformed_provenance_fields(tmp_path: Path, mutate, message: str) -> None:
    dataset_path = tmp_path / "malformed.hdf5"
    _write_dataset(dataset_path, [NATIVE_FIXED_PROFILE.width])
    manifest = build_manifest(producer="recording", profile=NATIVE_FIXED_PROFILE, repo_root=REPO_ROOT)
    mutate(manifest)
    write_manifest(dataset_path, manifest)

    with pytest.raises(DatasetContractError, match=message):
        validate_dataset(dataset_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest["environment"].__setitem__("id", "unsupported"), "environment"),
        (lambda manifest: manifest["environment"].__setitem__("digest", "invalid"), "environment"),
        (lambda manifest: manifest["runtime"].__setitem__("torch", 1), "runtime snapshot"),
        (lambda manifest: manifest["task"].pop("success_contract"), "task snapshot"),
        (lambda manifest: manifest["task"]["pickup_contract"].__setitem__("sha256", "0" * 64), "self-digest"),
    ],
    ids=("environment-id", "environment-digest", "runtime-value", "task-shape", "task-digest"),
)
def test_conversion_admission_validates_historical_g1_snapshots(tmp_path: Path, mutate, message: str) -> None:
    dataset_path = tmp_path / "historical.hdf5"
    _write_dataset(dataset_path, [ARENA_PINK_PROFILE.width])
    manifest = build_manifest(producer="recording", profile=ARENA_PINK_PROFILE, repo_root=REPO_ROOT)
    manifest["environment"]["digest"] = "sha256:" + "a" * 64
    manifest["runtime"]["torch"] = "historical-runtime"
    manifest["task"]["autodata_descriptor"] = {"path": "historical/task.yaml", "sha256": "b" * 64}
    for name in ("pickup_contract", "success_contract"):
        contract = {
            "function": f"historical.task:{name}",
            "implementation_sha256": "c" * 64,
            "version": 1,
            "parameters": {},
        }
        if name == "success_contract":
            contract["hold_frames"] = 20
        contract["sha256"] = dataset_manifest_module._snapshot_sha256(contract).removeprefix("sha256:")
        manifest["task"][name] = contract
    write_manifest(dataset_path, manifest)

    assert validate_dataset(dataset_path, conversion=True) == manifest
    mutate(manifest)
    write_manifest(dataset_path, manifest)

    with pytest.raises(DatasetContractError, match=message):
        validate_dataset(dataset_path, conversion=True)


def test_manifest_rejects_mixed_episode_widths(tmp_path: Path) -> None:
    dataset_path = tmp_path / "mixed.hdf5"
    _write_dataset(dataset_path, [28, 32])
    write_manifest(
        dataset_path,
        build_manifest(producer="recording", profile=NATIVE_FIXED_PROFILE, repo_root=REPO_ROOT),
    )

    with pytest.raises(DatasetContractError, match=r"demo_1/actions.*28.*\(3, 32\)"):
        validate_dataset(dataset_path)


def test_h2_dataset_admission_requires_the_complete_recorded_contract(tmp_path: Path) -> None:
    dataset_path = tmp_path / "h2.hdf5"
    _write_h2_dataset(dataset_path)

    assert validate_dataset(dataset_path)["action"]["layout"] == H2_ARENA_PINK_PROFILE.layout


def test_h2_generated_data_may_perturb_hands_within_the_recording_synergy_envelope(tmp_path: Path) -> None:
    source_path = tmp_path / "h2-source.hdf5"
    generated_path = tmp_path / "h2-generated.hdf5"
    _write_h2_dataset(source_path)
    _write_h2_dataset(generated_path)

    from e2e_robot_workflows.robots.h2 import load_h2_pink_contract

    contract = load_h2_pink_contract()
    index = contract.left_hand_indices[0]
    joint_name = contract.hand_joint_order[index]
    with h5py.File(generated_path, "r+") as dataset:
        target = 0.5 * (contract.hand_open_targets[joint_name] + contract.hand_closed_targets[joint_name])
        dataset["data/demo_0/actions"][0, 14 + index] = target
        joint_index = contract.expected_joint_order.index(joint_name)
        dataset["data/demo_0/joint_pos_target"][0, joint_index] = target

    stamp_autodata_dataset(generated_path, source_dataset=source_path, stage="generation")


def test_h2_autodata_stamp_rejects_an_incompatible_source_runtime(tmp_path: Path) -> None:
    source_path = tmp_path / "h2-source.hdf5"
    generated_path = tmp_path / "h2-generated.hdf5"
    _write_h2_dataset(source_path)
    _write_h2_dataset(generated_path)
    manifest = read_manifest(source_path)
    manifest["runtime"]["isaac_lab"] = "incompatible"
    manifest["runtime_digest"] = dataset_manifest_module._snapshot_sha256(manifest["runtime"])
    write_manifest(source_path, manifest)

    with pytest.raises(DatasetContractError, match="current H2 runtime contract"):
        stamp_autodata_dataset(generated_path, source_dataset=source_path, stage="generation")


@pytest.mark.parametrize("mutation", ["arm_target", "navigation"])
def test_h2_generation_rejects_values_outside_the_training_action_contract(
    tmp_path: Path,
    mutation: str,
) -> None:
    source_path = tmp_path / "h2-source.hdf5"
    generated_path = tmp_path / "h2-generated.hdf5"
    _write_h2_dataset(source_path)
    _write_h2_dataset(generated_path)
    stamp_autodata_dataset(generated_path, source_dataset=source_path, stage="generation")

    with h5py.File(generated_path, "r+") as dataset:
        episode = dataset["data/demo_0"]
        if mutation == "arm_target":
            from e2e_robot_workflows.robots.h2 import load_h2_pink_contract

            contract = load_h2_pink_contract()
            joint_index = contract.expected_joint_order.index(contract.arm_joint_names[0])
            episode["joint_pos_target"][0, joint_index] = 999.0
        else:
            episode["actions"][0, -1] = 100.0
            episode["navigate_cmd"][0, -1] = 100.0

    with pytest.raises(DatasetContractError, match="outside"):
        validate_dataset(generated_path, require_lane="arena", require_stage="generation")


def test_h2_joint_dataset_admission_uses_joint_targets_not_pose_quaternions(tmp_path: Path) -> None:
    dataset_path = tmp_path / "h2-joint.hdf5"
    _write_h2_joint_dataset(dataset_path)

    assert validate_dataset(dataset_path)["action"]["layout"] == H2_ARENA_JOINT_PROFILE.layout

    with h5py.File(dataset_path, "r+") as dataset:
        dataset["data/demo_0/joint_pos_target"][0, 12] += 1.0
    with pytest.raises(DatasetContractError, match="joint action targets"):
        validate_dataset(dataset_path)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_h2_manifest_pins_the_matching_autodata_descriptor(hand: str) -> None:
    manifest = build_manifest(
        producer="recording",
        profile=H2_ARENA_PINK_PROFILE,
        manipulation_hand=hand,
        repo_root=REPO_ROOT,
    )

    task = manifest["task"]
    h2_environment = load_static_apple_environment(H2_STATIC_APPLE_VARIANT)
    descriptor_path = REPO_ROOT / task["autodata_descriptor"]["path"]
    expected_name = "static_apple_arena_h2_left_task.yaml" if hand == "left" else "static_apple_arena_h2_task.yaml"

    assert descriptor_path.name == expected_name
    assert task["autodata_descriptor"]["sha256"] == file_sha256(descriptor_path)
    pickup_parameters = task["pickup_contract"]["parameters"]
    assert task["pickup_contract"]["version"] == 3
    assert pickup_parameters["mode"] == "h2_contact_and_closure_v1"
    assert pickup_parameters["contact_update_period_s"] == pytest.approx(0.01)
    assert pickup_parameters["held_object_qualification"]["contact_history_samples"] == 3
    assert pickup_parameters["criteria_implementation_sha256"] == file_sha256(
        REPO_ROOT / "src/e2e_robot_workflows/tasks/static_apple/grasp_contract.py"
    )
    assert pickup_parameters["grasp_distance_m"] == pytest.approx(0.20)
    assert pickup_parameters["lift_clearance_m"] == pytest.approx(0.08)
    success_parameters = task["success_contract"]["parameters"]
    assert success_parameters["placement_radius_m"] == pytest.approx(h2_environment.success_placement_xy_m)
    assert success_parameters["placement_z_m"] == pytest.approx(h2_environment.success_placement_z_m)
    assert success_parameters["release_clearance_m"] > success_parameters["placement_radius_m"]
    assert task["success_contract"]["hold_frames"] == 20
    assert manifest["environment"]["digest"] == h2_environment.contract_digest
    assert manifest["environment"]["variant_values"]["release_clearance_m"] == pytest.approx(0.20)


def test_h2_manifest_and_admission_ignore_ambient_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "h2-source.hdf5"
    _write_h2_dataset(dataset_path, steps=4)
    _add_h2_transfer_datagen(dataset_path)
    expected = read_manifest(dataset_path)

    monkeypatch.setenv("E2E_H2_ROBOT_CONFIG", "/dev/null")
    monkeypatch.setenv("E2E_STATIC_APPLE_TASK_CONFIG", "/dev/null")

    manifest = build_manifest(
        producer="recording",
        profile=H2_ARENA_PINK_PROFILE,
        manipulation_hand="left",
        repo_root=REPO_ROOT,
    )
    from e2e_robot_workflows.arena.embodiment_capabilities import H2_ARENA_EMBODIMENT, capability_for_embodiment

    binding = capability_for_embodiment(H2_ARENA_EMBODIMENT).load_binding()
    runtime_environment = load_static_apple_environment(H2_STATIC_APPLE_VARIANT)

    assert manifest == expected
    assert list(binding.task_link_names) == manifest["task"]["success_contract"]["parameters"]["task_link_names"]
    assert runtime_environment.contract_digest == manifest["environment"]["digest"]
    validate_arena_recording_contract(
        dataset_path,
        manipulation_hand="left",
        require_success_boundary=True,
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest["task"]["autodata_descriptor"].pop("path"), "task descriptor"),
        (lambda manifest: manifest["environment"]["robot"].pop("source_url"), "robot snapshot"),
        (lambda manifest: manifest["environment"]["robot"]["agile_policy"].pop("repository"), "robot snapshot"),
        (lambda manifest: manifest["environment"].__setitem__("unknown", True), "environment snapshot"),
        (lambda manifest: manifest["task"]["pickup_contract"].__setitem__("parameters", []), "pickup contract"),
        (lambda manifest: manifest["environment"].__setitem__("variant_values", {}), "variant snapshot"),
    ],
    ids=(
        "descriptor-path",
        "robot-source",
        "policy-repository",
        "unknown-environment-field",
        "malformed-task-contract",
        "empty-variant",
    ),
)
def test_h2_manifest_rejects_malformed_embedded_snapshot(tmp_path: Path, mutate, message: str) -> None:
    dataset_path = tmp_path / "h2-malformed-manifest.hdf5"
    _write_h2_dataset(dataset_path)
    manifest = read_manifest(dataset_path)
    mutate(manifest)
    for name in ("pickup_contract", "success_contract"):
        contract = manifest["task"][name]
        payload = {key: value for key, value in contract.items() if key != "sha256"}
        contract["sha256"] = dataset_manifest_module._snapshot_sha256(payload).removeprefix("sha256:")
    variant = manifest["environment"]["variant_values"]
    manifest["environment"]["variant_digest"] = dataset_manifest_module._snapshot_sha256(variant)
    write_manifest(dataset_path, manifest)

    with pytest.raises(DatasetContractError, match=message):
        validate_dataset(dataset_path)


def test_h2_manifest_accepts_historical_descriptor_digest_but_qualification_requires_current(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "h2-mutated-task.hdf5"
    _write_h2_dataset(dataset_path)
    manifest = read_manifest(dataset_path)
    manifest["task"]["autodata_descriptor"]["sha256"] = "0" * 64
    write_manifest(dataset_path, manifest)

    assert validate_dataset(dataset_path) == manifest
    with pytest.raises(DatasetContractError, match="current H2 task contract"):
        validate_dataset(dataset_path, qualification=True)


def test_proximity_only_h2_pickup_manifest_remains_readable_but_cannot_qualify(tmp_path: Path) -> None:
    dataset_path = tmp_path / "h2-proximity-only.hdf5"
    _write_h2_dataset(dataset_path)
    manifest = read_manifest(dataset_path)
    pickup = manifest["task"]["pickup_contract"]
    pickup["function"] = "e2e_robot_workflows.tasks.static_apple.observations:object_held_by_link"
    pickup["implementation_sha256"] = file_sha256(
        REPO_ROOT / "src/e2e_robot_workflows/tasks/static_apple/observations.py"
    )
    pickup["version"] = 1
    pickup["parameters"] = {
        key: pickup["parameters"][key] for key in ("task_link_names", "grasp_distance_m", "lift_clearance_m")
    }
    payload = {key: value for key, value in pickup.items() if key != "sha256"}
    pickup["sha256"] = dataset_manifest_module._snapshot_sha256(payload).removeprefix("sha256:")
    write_manifest(dataset_path, manifest)

    assert validate_dataset(dataset_path) == manifest
    with pytest.raises(DatasetContractError, match="current H2 task contract"):
        validate_dataset(dataset_path, qualification=True)


def test_pre_timing_h2_manifest_remains_readable_but_cannot_qualify(tmp_path: Path) -> None:
    dataset_path = tmp_path / "h2-pre-timing.hdf5"
    _write_h2_dataset(dataset_path)
    manifest = read_manifest(dataset_path)
    manifest["environment"].pop("timing")
    manifest.pop("runtime_digest")
    write_manifest(dataset_path, manifest)

    assert validate_dataset(dataset_path) == manifest
    assert manifest_control_hz(manifest) == 100

    manifest["runtime_digest"] = None
    write_manifest(dataset_path, manifest)
    with pytest.raises(DatasetContractError, match="pre-timing H2 manifest"):
        validate_dataset(dataset_path)
    manifest.pop("runtime_digest")
    write_manifest(dataset_path, manifest)

    with pytest.raises(DatasetContractError, match="pre-timing H2 data"):
        validate_dataset(dataset_path, qualification=True)


def test_h2_timing_snapshot_is_self_validating_and_qualification_requires_current(tmp_path: Path) -> None:
    dataset_path = tmp_path / "h2-timing.hdf5"
    _write_h2_dataset(dataset_path)
    manifest = read_manifest(dataset_path)
    timing = manifest["environment"]["timing"]

    assert {key: timing[key] for key in ("physics_hz", "control_hz", "render_hz", "camera_hz")} == {
        "physics_hz": 200,
        "control_hz": 100,
        "render_hz": 50,
        "camera_hz": 25,
    }
    validate_dataset(dataset_path, qualification=True)

    timing["control_hz"] = 50
    write_manifest(dataset_path, manifest)
    with pytest.raises(DatasetContractError, match="timing self-digest"):
        validate_dataset(dataset_path)

    timing_values = {key: timing[key] for key in ("physics_hz", "control_hz", "render_hz", "camera_hz")}
    timing["sha256"] = dataset_manifest_module._snapshot_sha256(timing_values)
    write_manifest(dataset_path, manifest)

    assert validate_dataset(dataset_path) == manifest
    with pytest.raises(DatasetContractError, match="current H2 environment contract"):
        validate_dataset(dataset_path, qualification=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("nonfinite_action", "actions contains non-finite"),
        ("unnormalized_quaternion", "quaternions are not normalized"),
        ("hand_outside_envelope", "hand target.*outside"),
        ("hand_midpoint", "complete open or closed synergy"),
        ("hand_mixed_synergy", "complete open or closed synergy"),
        ("wrong_state_width", "robot_joint_pos must have shape"),
        ("missing_joint_target", "joint_pos_target must have shape"),
        ("joint_hand_mismatch", "H2 Pink hand action targets"),
        ("navigation_mismatch", "navigate_cmd disagrees"),
        ("joint_target_outside_limits", "joint_pos_target.*outside the pinned"),
        ("navigation_outside_envelope", r"navigate_cmd\[0\].*outside the configured"),
        ("wrong_camera_shape", "robot_head_cam_rgb must have shape"),
        ("wrong_camera_dtype", "robot_head_cam_rgb must contain raw uint8"),
    ],
)
def test_h2_dataset_admission_rejects_incomplete_or_inconsistent_fields(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    dataset_path = tmp_path / "h2-invalid.hdf5"
    _write_h2_dataset(dataset_path)
    with h5py.File(dataset_path, "r+") as dataset:
        episode = dataset["data/demo_0"]
        if mutation == "nonfinite_action":
            episode["actions"][0, 0] = np.nan
        elif mutation == "unnormalized_quaternion":
            episode["actions"][0, 3:7] = 0.0
        elif mutation == "hand_outside_envelope":
            episode["actions"][0, 14] = 10.0
        elif mutation == "hand_midpoint":
            from e2e_robot_workflows.robots.h2 import load_h2_pink_contract

            contract = load_h2_pink_contract()
            joint_name = contract.hand_joint_order[contract.left_hand_indices[0]]
            episode["actions"][0, 14 + contract.left_hand_indices[0]] = 0.5 * (
                contract.hand_open_targets[joint_name] + contract.hand_closed_targets[joint_name]
            )
        elif mutation == "hand_mixed_synergy":
            from e2e_robot_workflows.robots.h2 import load_h2_pink_contract

            contract = load_h2_pink_contract()
            index = contract.left_hand_indices[0]
            joint_name = contract.hand_joint_order[index]
            episode["actions"][0, 14 + index] = contract.hand_closed_targets[joint_name]
        elif mutation == "wrong_state_width":
            del episode["obs/robot_joint_pos"]
            episode.create_dataset("obs/robot_joint_pos", shape=(2, 74), dtype="f4")
        elif mutation == "missing_joint_target":
            del episode["joint_pos_target"]
        elif mutation == "joint_hand_mismatch":
            from e2e_robot_workflows.robots.h2 import load_h2_pink_contract

            contract = load_h2_pink_contract()
            joint_index = contract.expected_joint_order.index(contract.hand_joint_order[0])
            episode["joint_pos_target"][0, joint_index] = 0.1
        elif mutation == "navigation_mismatch":
            episode["navigate_cmd"][0, 0] = 0.5
        elif mutation == "joint_target_outside_limits":
            from e2e_robot_workflows.robots.h2 import load_h2_pink_contract

            contract = load_h2_pink_contract()
            joint_index = contract.expected_joint_order.index(contract.arm_joint_names[0])
            episode["joint_pos_target"][0, joint_index] = 999.0
        elif mutation == "navigation_outside_envelope":
            episode["actions"][0, -3] = 0.6
            episode["navigate_cmd"][0, 0] = 0.6
        elif mutation == "wrong_camera_shape":
            del episode["camera_obs/robot_head_cam_rgb"]
            episode.create_dataset("camera_obs/robot_head_cam_rgb", shape=(2, 480, 639, 3), dtype="u1")
        elif mutation == "wrong_camera_dtype":
            del episode["camera_obs/robot_head_cam_rgb"]
            episode.create_dataset("camera_obs/robot_head_cam_rgb", shape=(2, 480, 640, 3), dtype="f4")
        else:  # pragma: no cover - the parameter table owns all cases
            raise AssertionError(mutation)

    with pytest.raises(DatasetContractError, match=message):
        validate_dataset(dataset_path)


@pytest.mark.parametrize("command_index", range(3))
@pytest.mark.parametrize("bound_index", range(2))
def test_h2_dataset_admission_accepts_navigation_values_at_each_exact_boundary(
    tmp_path: Path,
    command_index: int,
    bound_index: int,
) -> None:
    from e2e_robot_workflows.robots.h2 import load_h2_robot_config

    dataset_path = tmp_path / "h2-boundary.hdf5"
    _write_h2_dataset(dataset_path)
    boundary = load_h2_robot_config().navigate_command_limits[command_index][bound_index]
    with h5py.File(dataset_path, "r+") as dataset:
        dataset["data/demo_0/actions"][0, -3 + command_index] = boundary
        dataset["data/demo_0/navigate_cmd"][0, command_index] = boundary

    validate_dataset(dataset_path)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_h2_autodata_source_requires_a_placed_and_released_final_state(tmp_path: Path, hand: str) -> None:
    dataset_path = tmp_path / "h2-source.hdf5"
    _write_h2_dataset(dataset_path, steps=4)
    _add_h2_transfer_datagen(dataset_path)

    validate_arena_recording_contract(dataset_path, manipulation_hand="left", require_success_boundary=True)

    with h5py.File(dataset_path, "r+") as dataset:
        eef_pose = dataset[f"data/demo_0/obs/datagen_info/eef_pose/{hand}"]
        eef_pose[-1, :3, 3] = (0.0, 0.0, 0.04)
    with pytest.raises(DatasetContractError, match="hands released"):
        validate_arena_recording_contract(dataset_path, manipulation_hand="left", require_success_boundary=True)


@pytest.mark.parametrize(
    ("receiving_signal", "message_suffix"),
    [
        ((False, False, False, False), "handover boundary"),
        ((False, False, False, True), "handover boundary leaves no placement samples"),
    ],
)
@pytest.mark.parametrize(("manipulation_hand", "receiving_hand"), [("left", "right"), ("right", "left")])
def test_h2_autodata_source_requires_a_usable_receiving_hand_boundary(
    tmp_path: Path,
    receiving_signal: tuple[bool, ...],
    message_suffix: str,
    manipulation_hand: str,
    receiving_hand: str,
) -> None:
    dataset_path = tmp_path / "h2-source.hdf5"
    _write_h2_dataset(dataset_path, steps=4, manipulation_hand=manipulation_hand)
    _add_h2_transfer_datagen(dataset_path, manipulation_hand=manipulation_hand)

    with h5py.File(dataset_path, "r+") as dataset:
        dataset[f"data/demo_0/obs/datagen_info/subtask_term_signals/{receiving_hand}_hand_holds_object"][:] = (
            receiving_signal
        )

    with pytest.raises(ArenaEpisodeRejection, match=rf"{receiving_hand}-hand {message_suffix}"):
        validate_arena_recording_contract(
            dataset_path,
            manipulation_hand=manipulation_hand,
            require_success_boundary=True,
        )


def test_h2_file_admission_reuses_shared_array_boundary_checks(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "h2-source.hdf5"
    _write_h2_dataset(dataset_path, steps=4)
    _add_h2_transfer_datagen(dataset_path)
    original = dataset_manifest_module.validate_arena_episode_boundaries
    calls: list[dict] = []

    def check(signals, **kwargs):
        assert set(signals) == {"left", "right"}
        assert all(isinstance(value, np.ndarray) and value.shape == (4,) for value in signals.values())
        calls.append(kwargs)
        return original(signals, **kwargs)

    monkeypatch.setattr(dataset_manifest_module, "validate_arena_episode_boundaries", check)

    validate_arena_recording_contract(dataset_path, manipulation_hand="left", require_success_boundary=True)

    assert [call["require_handover"] for call in calls] == [False, True]
    assert calls[-1]["manipulation_hand"] == "left"


def test_h2_file_admission_malformed_signals_remain_fatal(tmp_path: Path) -> None:
    dataset_path = tmp_path / "h2-source.hdf5"
    _write_h2_dataset(dataset_path, steps=4)
    _add_h2_transfer_datagen(dataset_path)
    signal_path = "data/demo_0/obs/datagen_info/subtask_term_signals/right_hand_holds_object"
    with h5py.File(dataset_path, "r+") as dataset:
        del dataset[signal_path]
        dataset.create_dataset(signal_path, data=np.asarray((0.0, np.nan, 1.0, 1.0)))

    with pytest.raises(DatasetContractError, match="only boolean or 0/1 pickup samples") as error:
        validate_arena_recording_contract(dataset_path, manipulation_hand="left", require_success_boundary=True)
    assert not isinstance(error.value, ArenaEpisodeRejection)


def test_h2_autodata_source_rejects_an_elevated_plate_and_apple(tmp_path: Path) -> None:
    dataset_path = tmp_path / "h2-elevated-source.hdf5"
    _write_h2_dataset(dataset_path, steps=4)
    _add_h2_transfer_datagen(dataset_path)

    with h5py.File(dataset_path, "r+") as dataset:
        dataset["data/demo_0/states/rigid_object/pick_object/root_pose"][-1, 2] = 1.30
        dataset["data/demo_0/states/rigid_object/placement_target/root_pose"][-1, 2] = 1.28

    with pytest.raises(DatasetContractError, match="does not end with the H2 apple on the placement target"):
        validate_arena_recording_contract(dataset_path, manipulation_hand="left", require_success_boundary=True)


def test_h2_autodata_source_enforces_the_vertical_placement_contract(tmp_path: Path) -> None:
    dataset_path = tmp_path / "h2-vertical-separation-source.hdf5"
    _write_h2_dataset(dataset_path, steps=4)
    _add_h2_transfer_datagen(dataset_path)

    with h5py.File(dataset_path, "r+") as dataset:
        dataset["data/demo_0/states/rigid_object/pick_object/root_pose"][-1, 2] = 1.04
        dataset["data/demo_0/states/rigid_object/placement_target/root_pose"][-1, 2] = 0.92

    with pytest.raises(DatasetContractError, match="does not end with the H2 apple on the placement target"):
        validate_arena_recording_contract(dataset_path, manipulation_hand="left", require_success_boundary=True)


@pytest.mark.parametrize(
    ("pick_xyz", "target_xyz", "accepted"),
    [
        (
            (0.0, 0.0, 0.95 + STATIC_APPLE_ENVIRONMENT.success_placement_z_m),
            (0.0, 0.0, 0.95),
            True,
        ),
        (
            (0.0, 0.0, 0.9501 + STATIC_APPLE_ENVIRONMENT.success_placement_z_m),
            (0.0, 0.0, 0.95),
            False,
        ),
        ((0.0, 0.0, 0.975), (0.0, 0.0, 0.98), True),
        ((0.0, 0.0, 0.97), (0.0, 0.0, 0.98), False),
        (
            (
                0.8 * STATIC_APPLE_ENVIRONMENT.success_placement_xy_m,
                0.8 * STATIC_APPLE_ENVIRONMENT.success_placement_xy_m,
                1.0,
            ),
            (0.0, 0.0, 0.96),
            False,
        ),
    ],
)
def test_h2_autodata_source_placement_boundaries(
    tmp_path: Path,
    pick_xyz: tuple[float, float, float],
    target_xyz: tuple[float, float, float],
    accepted: bool,
) -> None:
    dataset_path = tmp_path / "h2-placement-boundary.hdf5"
    _write_h2_dataset(dataset_path, steps=4)
    _add_h2_transfer_datagen(dataset_path)

    with h5py.File(dataset_path, "r+") as dataset:
        positions = {"pick_object": pick_xyz, "placement_target": target_xyz}
        for object_name, xyz in positions.items():
            path = f"data/demo_0/states/rigid_object/{object_name}/root_pose"
            values = np.asarray(dataset[path], dtype=np.float64)
            values[-1, :3] = xyz
            del dataset[path]
            dataset.create_dataset(path, data=values)

    if accepted:
        validate_arena_recording_contract(dataset_path, manipulation_hand="left", require_success_boundary=True)
    else:
        with pytest.raises(DatasetContractError, match="does not end with the H2 apple on the placement target"):
            validate_arena_recording_contract(dataset_path, manipulation_hand="left", require_success_boundary=True)


def test_h2_autodata_source_uses_world_state_for_final_placement(tmp_path: Path) -> None:
    dataset_path = tmp_path / "h2-world-placement-source.hdf5"
    _write_h2_dataset(dataset_path, steps=4)
    _add_h2_transfer_datagen(dataset_path)

    with h5py.File(dataset_path, "r+") as dataset:
        dataset["data/demo_0/states/rigid_object/pick_object/root_pose"][-1, 0] = 2.0

    with pytest.raises(DatasetContractError, match="does not end with the H2 apple on the placement target"):
        validate_arena_recording_contract(dataset_path, manipulation_hand="left", require_success_boundary=True)


def test_autodata_manifest_requires_exact_source_hash(tmp_path: Path) -> None:
    source_path = tmp_path / "source.hdf5"
    annotated_path = tmp_path / "annotated.hdf5"
    generated_path = tmp_path / "generated.hdf5"
    _write_dataset(source_path, [NATIVE_FIXED_PROFILE.width])
    write_manifest(
        source_path,
        build_manifest(producer="recording", profile=NATIVE_FIXED_PROFILE, repo_root=REPO_ROOT),
    )
    _write_dataset(annotated_path, [NATIVE_FIXED_PROFILE.width])
    annotated_manifest = stamp_autodata_dataset(
        annotated_path,
        source_dataset=source_path,
        stage="annotation",
    )
    _write_dataset(generated_path, [NATIVE_FIXED_PROFILE.width])
    generated_manifest = stamp_autodata_dataset(
        generated_path,
        source_dataset=annotated_path,
        stage="generation",
    )

    assert annotated_manifest["source"]["stage"] == "recording"
    assert generated_manifest["source"]["stage"] == "annotation"
    assert validate_dataset(generated_path, source_dataset=annotated_path) == generated_manifest
    with h5py.File(annotated_path, "r+") as dataset:
        dataset.attrs["changed"] = True
    with pytest.raises(DatasetContractError, match="source SHA-256"):
        validate_dataset(generated_path, source_dataset=annotated_path)


def test_generation_rejects_unannotated_source(tmp_path: Path) -> None:
    source_path = tmp_path / "source.hdf5"
    _write_dataset(source_path, [NATIVE_FIXED_PROFILE.width])
    write_manifest(
        source_path,
        build_manifest(producer="recording", profile=NATIVE_FIXED_PROFILE, repo_root=REPO_ROOT),
    )

    with pytest.raises(DatasetContractError, match="generation output requires an annotation source"):
        build_manifest(
            producer="autodata",
            profile=NATIVE_FIXED_PROFILE,
            source_dataset=source_path,
            stage="generation",
            repo_root=REPO_ROOT,
        )


@pytest.mark.parametrize("seed", [None, 0, 731])
def test_arena_generation_preserves_seed_and_direct_recording_lineage(tmp_path: Path, seed: int | None) -> None:
    source_path = tmp_path / "arena-source.hdf5"
    generated_path = tmp_path / "arena-generated.hdf5"
    _write_dataset(source_path, [ARENA_PINK_PROFILE.width])
    write_manifest(
        source_path,
        build_manifest(producer="recording", profile=ARENA_PINK_PROFILE, seed=17, repo_root=REPO_ROOT),
    )
    source_sha256 = file_sha256(source_path)
    _write_dataset(generated_path, [ARENA_PINK_PROFILE.width])

    seed_arguments = {} if seed is None else {"seed": seed}
    manifest = stamp_autodata_dataset(
        generated_path,
        source_dataset=source_path,
        stage="generation",
        **seed_arguments,
    )

    assert manifest["lane"] == "arena"
    assert manifest["stage"] == "generation"
    assert manifest["seed"] == seed
    assert manifest["source"]["stage"] == "recording"
    assert manifest["source"]["sha256"] == source_sha256
    assert read_manifest(generated_path) == manifest
    assert validate_dataset(generated_path, source_dataset=source_path) == manifest
    assert read_manifest(source_path)["seed"] == 17
    assert file_sha256(source_path) == source_sha256


def test_arena_source_admission_checks_direct_xyzw_semantics_and_pickup_boundary(tmp_path: Path) -> None:
    source_path = tmp_path / "arena-source.hdf5"
    _write_arena_source(source_path)

    manifest = validate_arena_recording_contract(
        source_path,
        manipulation_hand="right",
        require_success_boundary=True,
    )

    assert manifest["action"]["layout"] == ARENA_PINK_PROFILE.layout


def test_arena_recording_detects_the_hand_with_the_pickup_boundary(tmp_path: Path) -> None:
    source_path = tmp_path / "arena-source.hdf5"
    _write_arena_source(source_path)

    assert detect_arena_manipulation_hand(source_path) == "right"

    with h5py.File(source_path, "r+") as dataset:
        right = dataset["data/demo_0/obs/datagen_info/subtask_term_signals/right_hand_holds_object"]
        left = dataset["data/demo_0/obs/datagen_info/subtask_term_signals/left_hand_holds_object"]
        left[...] = right[...]
        right[...] = False
    assert detect_arena_manipulation_hand(source_path) == "left"


@pytest.mark.parametrize(
    "malformed_signal",
    [
        np.asarray([0.0, np.nan, 1.0]),
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([0, 2, 1]),
    ],
    ids=["nan", "fractional", "non-binary"],
)
def test_arena_recording_rejects_malformed_pickup_samples(
    tmp_path: Path,
    malformed_signal: np.ndarray,
) -> None:
    source_path = tmp_path / "arena-source.hdf5"
    _write_arena_source(source_path)
    signal_path = "data/demo_0/obs/datagen_info/subtask_term_signals/right_hand_holds_object"
    with h5py.File(source_path, "r+") as dataset:
        del dataset[signal_path]
        dataset.create_dataset(signal_path, data=malformed_signal)

    with pytest.raises(DatasetContractError, match="only boolean or 0/1 pickup samples"):
        detect_arena_manipulation_hand(source_path)


def test_arena_source_admission_revalidates_pickup_samples_before_cast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "arena-source.hdf5"
    _write_arena_source(source_path)
    signal_path = "data/demo_0/obs/datagen_info/subtask_term_signals/right_hand_holds_object"
    with h5py.File(source_path, "r+") as dataset:
        del dataset[signal_path]
        dataset.create_dataset(signal_path, data=np.asarray([0.0, 0.5, 1.0]))
    monkeypatch.setattr(dataset_manifest_module, "detect_arena_manipulation_hand", lambda _path: "right")

    with pytest.raises(DatasetContractError, match="only boolean or 0/1 pickup samples"):
        validate_arena_recording_contract(
            source_path,
            manipulation_hand="right",
            require_success_boundary=True,
        )


def test_arena_recording_accepts_numeric_binary_pickup_samples(tmp_path: Path) -> None:
    source_path = tmp_path / "arena-source.hdf5"
    _write_arena_source(source_path)
    signal_path = "data/demo_0/obs/datagen_info/subtask_term_signals/right_hand_holds_object"
    with h5py.File(source_path, "r+") as dataset:
        del dataset[signal_path]
        dataset.create_dataset(signal_path, data=np.asarray([0.0, 1.0, 1.0]))

    assert detect_arena_manipulation_hand(source_path) == "right"


def test_arena_recording_uses_the_first_boundary_before_a_handover(tmp_path: Path) -> None:
    source_path = tmp_path / "arena-handover-source.hdf5"
    _write_arena_source(source_path, pickup_signal=(False, False, True, True))
    with h5py.File(source_path, "r+") as dataset:
        dataset["data/demo_0/obs/datagen_info/subtask_term_signals/left_hand_holds_object"][:] = (
            False,
            True,
            True,
            True,
        )

    assert detect_arena_manipulation_hand(source_path) == "left"


def test_arena_source_admission_requires_manifest_hand_to_own_first_boundary(tmp_path: Path) -> None:
    source_path = tmp_path / "arena-handover-source.hdf5"
    _write_arena_source(source_path, pickup_signal=(False, False, True, True))
    with h5py.File(source_path, "r+") as dataset:
        dataset["data/demo_0/obs/datagen_info/subtask_term_signals/left_hand_holds_object"][:] = (
            False,
            True,
            True,
            True,
        )

    with pytest.raises(DatasetContractError, match="first pickup boundary belongs to left"):
        validate_arena_recording_contract(source_path, manipulation_hand="right", require_success_boundary=True)


@pytest.mark.parametrize("left_signal", [(False, False, False), (False, True, True)])
def test_arena_recording_rejects_missing_or_ambiguous_pickup_hands(tmp_path: Path, left_signal) -> None:
    source_path = tmp_path / "arena-source.hdf5"
    _write_arena_source(source_path)
    with h5py.File(source_path, "r+") as dataset:
        dataset["data/demo_0/obs/datagen_info/subtask_term_signals/left_hand_holds_object"][:] = left_signal
        if not any(left_signal):
            dataset["data/demo_0/obs/datagen_info/subtask_term_signals/right_hand_holds_object"][:] = False

    with pytest.raises(DatasetContractError, match="one unambiguous Arena pickup boundary"):
        detect_arena_manipulation_hand(source_path)


def test_arena_source_admission_rejects_semantic_quaternion_mismatch(tmp_path: Path) -> None:
    source_path = tmp_path / "arena-source.hdf5"
    _write_arena_source(source_path)
    with h5py.File(source_path, "r+") as dataset:
        dataset["data/demo_0/actions"][:, 5:9] = np.asarray([1.0, 0.0, 0.0, 0.0])

    with pytest.raises(DatasetContractError, match="left target_eef_pose disagrees"):
        validate_arena_recording_contract(source_path)


def test_failed_arena_recording_stamp_rolls_back_the_manifest(tmp_path: Path) -> None:
    source_path = tmp_path / "invalid-arena-source.hdf5"
    _write_dataset(source_path, [ARENA_PINK_PROFILE.width])

    with pytest.raises(DatasetContractError, match="no direct obs/datagen_info"):
        stamp_recording_dataset(
            source_path,
            lane="arena",
            arena_embodiment=ARENA_PINK_PROFILE.controller,
        )

    with h5py.File(source_path, "r") as dataset:
        assert MANIFEST_ATTRIBUTE not in dataset.attrs


def test_arena_autodata_source_requires_a_pickup_boundary(tmp_path: Path) -> None:
    source_path = tmp_path / "arena-source.hdf5"
    _write_arena_source(source_path, pickup_signal=(False, False, False))

    with pytest.raises(DatasetContractError, match="must contain one unambiguous Arena pickup boundary"):
        validate_arena_recording_contract(
            source_path,
            manipulation_hand="right",
            require_success_boundary=True,
        )


def test_qualification_accepts_foreign_repository_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "foreign.hdf5"
    _write_h2_dataset(dataset_path, steps=4)
    _add_h2_transfer_datagen(dataset_path)
    manifest = read_manifest(dataset_path)
    manifest["repository"] = {"revision": "a" * 40, "dirty": True}
    write_manifest(dataset_path, manifest)
    monkeypatch.setattr(
        dataset_manifest_module,
        "_repository_state",
        lambda _root: pytest.fail("dataset admission must not inspect the ambient Git checkout"),
    )

    assert (
        validate_arena_recording_contract(
            dataset_path,
            manipulation_hand="left",
            require_success_boundary=True,
            qualification=True,
        )
        == manifest
    )


def test_qualification_accepts_foreign_repository_source_lineage(tmp_path: Path) -> None:
    source_path = tmp_path / "h2-source.hdf5"
    generated_path = tmp_path / "h2-generated.hdf5"
    _write_h2_dataset(source_path, steps=4)
    _add_h2_transfer_datagen(source_path)
    source_manifest = read_manifest(source_path)
    source_manifest["repository"] = {"revision": "a" * 40, "dirty": True}
    write_manifest(source_path, source_manifest)
    _write_h2_dataset(generated_path, steps=4)
    stamp_autodata_dataset(generated_path, source_dataset=source_path, stage="generation")

    manifest = validate_dataset(generated_path, qualification=True)
    assert manifest["source"]["repository"] == source_manifest["repository"]


def test_legacy_source_lineage_remains_usable_but_unqualified(tmp_path: Path) -> None:
    source_path = tmp_path / "native-source.hdf5"
    annotated_path = tmp_path / "native-annotated.hdf5"
    generated_path = tmp_path / "native-generated.hdf5"
    _write_dataset(source_path, [NATIVE_FIXED_PROFILE.width])
    source_manifest = build_manifest(producer="recording", profile=NATIVE_FIXED_PROFILE, repo_root=REPO_ROOT)
    source_manifest["provenance"] = {"status": "legacy_unverified", "original_sha256": "b" * 64}
    write_manifest(source_path, source_manifest)
    _write_dataset(annotated_path, [NATIVE_FIXED_PROFILE.width])
    stamp_autodata_dataset(annotated_path, source_dataset=source_path, stage="annotation")
    _write_dataset(generated_path, [NATIVE_FIXED_PROFILE.width])
    manifest = stamp_autodata_dataset(generated_path, source_dataset=annotated_path, stage="generation")

    assert manifest["source"]["source"]["provenance"] == source_manifest["provenance"]
    with pytest.raises(DatasetContractError, match="legacy-unverified"):
        validate_dataset(generated_path, qualification=True)


def test_arena_joint_recording_is_not_admitted_to_autodata(tmp_path: Path) -> None:
    source_path = tmp_path / "source.hdf5"
    _write_dataset(source_path, [ARENA_JOINT_PROFILE.width])
    write_manifest(
        source_path,
        build_manifest(producer="recording", profile=ARENA_JOINT_PROFILE, repo_root=REPO_ROOT),
    )

    with pytest.raises(DatasetContractError, match="not AutoData-compatible"):
        build_manifest(
            producer="autodata",
            profile=ARENA_JOINT_PROFILE,
            source_dataset=source_path,
            stage="annotation",
            repo_root=REPO_ROOT,
        )


def test_legacy_migration_requires_confirmation_and_matching_layout(tmp_path: Path) -> None:
    dataset_path = tmp_path / "legacy.hdf5"
    _write_dataset(dataset_path, [NATIVE_FIXED_PROFILE.width])

    with pytest.raises(DatasetContractError, match="confirm-legacy-migration"):
        migrate_legacy_recording(
            dataset_path,
            output_path=tmp_path / "migrated.hdf5",
            lane="native",
            layout=NATIVE_FIXED_PROFILE.layout,
            manipulation_hand="right",
            confirmed=False,
        )
    with pytest.raises(DatasetContractError, match="does not match selected 32-D"):
        migrate_legacy_recording(
            dataset_path,
            output_path=tmp_path / "migrated.hdf5",
            lane="native",
            layout=NATIVE_AGILE_PROFILE.layout,
            manipulation_hand="right",
            confirmed=True,
        )
    with h5py.File(dataset_path, "r") as dataset:
        assert MANIFEST_ATTRIBUTE not in dataset.attrs


def test_legacy_migration_cli_stamps_operator_selected_contract(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = tmp_path / "legacy.hdf5"
    migrated_path = tmp_path / "migrated.hdf5"
    _write_arena_source(dataset_path, add_manifest=False)

    assert (
        main(
            [
                "migrate-recording",
                "--input-file",
                str(dataset_path),
                "--output-file",
                str(migrated_path),
                "--lane",
                "arena",
                "--layout",
                ARENA_PINK_PROFILE.layout,
                "--manipulation-hand",
                "right",
                "--confirm-legacy-migration",
            ]
        )
        == 0
    )
    assert capsys.readouterr().err == ""
    with h5py.File(dataset_path, "r") as dataset:
        assert MANIFEST_ATTRIBUTE not in dataset.attrs
    manifest = validate_dataset(migrated_path, require_lane="arena", require_stage="recording")
    assert manifest["provenance"] == {
        "status": "legacy_unverified",
        "original_sha256": __import__("hashlib").sha256(dataset_path.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize("writer", [_write_h2_dataset, _write_h2_joint_dataset])
def test_h2_qualification_requires_snapshot_but_legacy_inspection_remains_possible(tmp_path, writer):
    path = tmp_path / "h2.hdf5"
    writer(path)
    validate_dataset(path, qualification=True)
    with h5py.File(path, "r+") as dataset:
        del dataset["data/demo_0/initial_state"]
    validate_dataset(path)
    with pytest.raises(DatasetContractError, match="requires both"):
        validate_dataset(path, qualification=True)


@pytest.mark.parametrize("damage", ["missing_targets", "missing_feedback", "missing_port", "shape", "nonfinite"])
def test_h2_snapshot_damage_is_rejected_even_without_qualification(tmp_path, damage):
    path = tmp_path / "h2.hdf5"
    _write_h2_dataset(path)
    with h5py.File(path, "r+") as dataset:
        state = dataset["data/demo_0/initial_state"]
        if damage == "missing_targets":
            del state["h2_actuator_targets"]
        elif damage == "missing_feedback":
            del state["h2_agile_feedback"]
        elif damage == "missing_port":
            del state["h2_agile_feedback/last_actions"]
        elif damage == "shape":
            del state["h2_actuator_targets/position"]
            state.create_dataset("h2_actuator_targets/position", data=np.zeros((75,)))
        else:
            state["h2_agile_feedback/last_actions"][0, 0] = np.nan
    with pytest.raises(DatasetContractError, match="H2"):
        validate_dataset(path)


def test_h2_source_admission_rejects_disjoint_pickups_and_missing_raw_evidence(tmp_path):
    path = tmp_path / "h2.hdf5"
    _write_h2_dataset(path, steps=8)
    _add_h2_transfer_datagen(path)
    with h5py.File(path, "r+") as dataset:
        datagen = dataset["data/demo_0/obs/datagen_info"]
        for hand, raw in {"left": [0, 1, 1, 0, 0, 0, 0, 0], "right": [0, 0, 0, 0, 1, 1, 0, 0]}.items():
            datagen[f"grasp_signals/{hand}_hand_holds_object"][:] = raw
            datagen[f"subtask_term_signals/{hand}_hand_holds_object"][:] = np.logical_or.accumulate(raw)
    with pytest.raises(ArenaEpisodeRejection, match="continuous handover"):
        validate_arena_recording_contract(path, require_success_boundary=True, qualification=True)
    with h5py.File(path, "r+") as dataset:
        del dataset["data/demo_0/obs/datagen_info/grasp_signals"]
    with pytest.raises(DatasetContractError, match="raw grasp_signals"):
        validate_arena_recording_contract(path, require_success_boundary=True, qualification=True)

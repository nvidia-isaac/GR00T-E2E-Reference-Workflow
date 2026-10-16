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

"""Behavioral tests for the HDF5-to-LeRobot inspection CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import e2e_robot_workflows.data.dataset_manifest as dataset_manifest_module
from e2e_robot_workflows.data.dataset_manifest import (
    ARENA_PINK_PROFILE,
    H2_ARENA_JOINT_PROFILE,
    H2_ARENA_PINK_PROFILE,
    ActionProfile,
    build_manifest,
    file_sha256,
    stamp_autodata_dataset,
    write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_SCRIPT = REPO_ROOT / "scripts/convert_hdf5_to_lerobot.py"
DATASET_CONFIG = REPO_ROOT / "configs/stages/gr00t_lerobot_dataset.yaml"
H2_DATASET_CONFIG = REPO_ROOT / "configs/stages/gr00t_h2_lerobot_dataset.yaml"
TEST_G1_ADMISSION = """\
manifest_admission:
  required: false
  accepted_profiles:
    - {lane: arena, controller: g1_wbc_agile_pink, layout: arena_g1_wbc_agile_pink_23d_v1}
  unmanifested_action_widths: [23]
"""


def write_dataset(
    path: Path,
    *,
    include_episode: bool = True,
    include_fields: bool = True,
    include_video: bool = True,
) -> None:
    """Write the smallest HDF5 hierarchy needed to exercise field discovery."""

    with h5py.File(path, "w") as hdf5_file:
        data_group = hdf5_file.create_group("data")
        data_group.attrs["total"] = int(include_episode)
        if not include_episode:
            return

        episode_group = data_group.create_group("demo_0")
        episode_group.attrs["num_samples"] = 2
        episode_group.attrs["success"] = True
        episode_group.create_dataset("actions", shape=(2, ARENA_PINK_PROFILE.width), dtype="f4")
        if not include_fields:
            return

        observation_group = episode_group.create_group("obs")
        observation_group.create_dataset("robot_joint_pos", shape=(2, 3), dtype="f4")
        episode_group.create_dataset("joint_pos_target", shape=(2, 3), dtype="f4")
        if include_video:
            camera_group = episode_group.create_group("camera_obs")
            camera_group.create_dataset("robot_head_cam_rgb", shape=(2, 4, 5, 3), dtype="u1")


def write_h2_dataset(path: Path, *, profile: ActionProfile, manifested: bool = True) -> None:
    actions = np.zeros((2, profile.width), dtype=np.float32)
    if profile == H2_ARENA_PINK_PROFILE:
        actions[:, 6] = 1.0
        actions[:, 13] = 1.0
    with h5py.File(path, "w") as hdf5_file:
        episode = hdf5_file.create_group("data/demo_0")
        episode.create_dataset("actions", data=actions)
        episode.create_dataset("joint_pos_target", shape=(2, 75), dtype="f4")
        episode.create_dataset("navigate_cmd", shape=(2, 3), dtype="f4")
        obs = episode.create_group("obs")
        obs.create_dataset("robot_joint_pos", shape=(2, 75), dtype="f4")
        camera = episode.create_group("camera_obs")
        camera.create_dataset("robot_head_cam_rgb", shape=(2, 480, 640, 3), dtype="u1")
    if manifested:
        write_manifest(
            path,
            build_manifest(
                producer="recording",
                profile=profile,
                manipulation_hand="left",
                repo_root=REPO_ROOT,
            ),
        )


def run_converter(
    dataset_path: Path,
    report_path: Path,
    *,
    cwd: Path,
    config_path: Path = DATASET_CONFIG,
) -> tuple[subprocess.CompletedProcess[str], dict]:
    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER_SCRIPT),
            "--config",
            str(config_path),
            "--input",
            str(dataset_path),
            "--inspect",
            "--strict",
            "--report",
            str(report_path),
        ],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert report_path.is_file(), result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert json.loads(result.stdout) == report
    return result, report


def required_mappings(report: dict) -> dict[str, dict]:
    required_fields = set(report["mapping_report"]["required_fields"])
    return {
        mapping["semantic_key"]: mapping
        for mapping in report["mapping_report"]["mappings"]
        if mapping["semantic_key"] in required_fields
    }


def test_converter_cli_reports_resolved_required_fields(tmp_path: Path) -> None:
    dataset_path = tmp_path / "complete.hdf5"
    report_path = tmp_path / "report.json"
    write_dataset(dataset_path)

    result, report = run_converter(dataset_path, report_path, cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["status"] == "ok"
    assert report["inspection"]["episode_count"] == 1
    assert required_mappings(report) == {
        "state": {
            "semantic_key": "state",
            "hdf5_name": "robot_joint_pos",
            "resolved_hdf5_path": "obs/robot_joint_pos",
            "lerobot_key": "observation.state",
        },
        "action": {
            "semantic_key": "action",
            "hdf5_name": "joint_pos_target",
            "resolved_hdf5_path": "joint_pos_target",
            "lerobot_key": "action",
        },
        "video": {
            "semantic_key": "video",
            "hdf5_name": "robot_head_cam_rgb",
            "resolved_hdf5_path": "camera_obs/robot_head_cam_rgb",
            "lerobot_key": "observation.images.ego_view",
        },
    }
    assert report["mapping_report"]["missing_required"] == []
    assert report["mapping_report"]["missing_required_source_mappings"] == []
    assert report["mapping_report"]["missing_required_target_mappings"] == []
    assert report["mapping_report"]["ready_for_writer"] is True


def test_converter_derives_h2_fps_from_manifest_outside_repository(tmp_path: Path) -> None:
    expanded_home = tmp_path / "expanded-home"
    expanded_home.mkdir()
    dataset_path = expanded_home / "h2.hdf5"
    report_path = tmp_path / "report.json"
    output_path = expanded_home / "lerobot"
    state_row = np.linspace(-0.1, 0.1, 75, dtype=np.float32)
    action_row = np.zeros(75, dtype=np.float32)
    from e2e_robot_workflows.robots.h2 import load_h2_pink_contract

    contract = load_h2_pink_contract()
    for joint_name, (lower, upper) in contract.learned_joint_position_limits:
        action_row[contract.expected_joint_order.index(joint_name)] = 0.5 * (lower + upper)
    hand_targets = np.asarray(
        [contract.hand_open_targets[name] for name in contract.hand_joint_order], dtype=np.float32
    )
    hand_joint_indices = [contract.expected_joint_order.index(name) for name in contract.hand_joint_order]
    action_row[hand_joint_indices] = hand_targets
    with h5py.File(dataset_path, "w") as hdf5_file:
        episode = hdf5_file.create_group("data/demo_0")
        actions = np.zeros((2, H2_ARENA_PINK_PROFILE.width), dtype=np.float32)
        actions[:, 6] = 1.0
        actions[:, 13] = 1.0
        actions[:, 14:58] = hand_targets
        episode.create_dataset("actions", data=actions)
        episode.create_dataset("joint_pos_target", data=np.repeat(action_row[None], 2, axis=0))
        episode.create_dataset("navigate_cmd", shape=(2, 3), dtype="f4")
        obs = episode.create_group("obs")
        obs.create_dataset("robot_joint_pos", data=np.repeat(state_row[None], 2, axis=0))
        camera = episode.create_group("camera_obs")
        camera.create_dataset("robot_head_cam_rgb", shape=(2, 480, 640, 3), dtype="u1")
    manifest = build_manifest(
        producer="recording",
        profile=H2_ARENA_PINK_PROFILE,
        manipulation_hand="left",
        repo_root=REPO_ROOT,
    )
    timing = manifest["environment"]["timing"]
    timing["control_hz"] = 50
    timing_values = {key: timing[key] for key in ("physics_hz", "control_hz", "render_hz", "camera_hz")}
    timing["sha256"] = dataset_manifest_module._snapshot_sha256(timing_values)
    write_manifest(dataset_path, manifest)

    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER_SCRIPT),
            "--input",
            "~/h2.hdf5",
            "--output",
            "~/lerobot",
            "--strict",
            "--report",
            str(report_path),
        ],
        cwd=tmp_path,
        env={**os.environ, "HOME": str(expanded_home)},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stderr
    assert Path(report["config"]).name == "gr00t_h2_lerobot_dataset.yaml"
    assert report["mapping_report"]["ready_for_writer"] is True
    assert report["mapping_report"]["dropped_modality_columns"] == []
    assert report["write_result"]["status"] == "ok"
    assert report["write_result"]["warnings"] == []
    assert (output_path / "meta/modality.json").is_file()
    assert json.loads((output_path / "meta/info.json").read_text(encoding="utf-8"))["fps"] == 50


@pytest.mark.parametrize("stage", ["recording", "generation"])
def test_converter_preserves_historical_g1_manifest_without_qualifying_it(tmp_path: Path, stage: str) -> None:
    source_path = tmp_path / "source.hdf5"
    dataset_path = source_path if stage == "recording" else tmp_path / "generated.hdf5"
    for path in {source_path, dataset_path}:
        with h5py.File(path, "w") as dataset:
            episode = dataset.create_group("data/demo_0")
            episode.create_dataset("actions", shape=(2, ARENA_PINK_PROFILE.width), dtype="f4")
            episode.create_dataset("obs/robot_joint_pos", shape=(2, 43), dtype="f4")
            episode.create_dataset("joint_pos_target", shape=(2, 43), dtype="f4")
            episode.create_dataset("navigate_cmd", shape=(2, 3), dtype="f4")
            episode.create_dataset("base_height_cmd", shape=(2, 1), dtype="f4")
            episode.create_dataset("camera_obs/robot_head_cam_rgb", shape=(2, 480, 640, 3), dtype="u1")
    source_manifest = build_manifest(producer="recording", profile=ARENA_PINK_PROFILE, repo_root=REPO_ROOT)
    write_manifest(source_path, source_manifest)
    manifest = (
        source_manifest
        if stage == "recording"
        else build_manifest(
            producer="autodata", profile=ARENA_PINK_PROFILE, source_dataset=source_path, repo_root=REPO_ROOT
        )
    )
    manifest["environment"]["digest"] = "sha256:" + "a" * 64
    manifest["runtime"]["torch"] = "2.11.0+cu126"
    write_manifest(dataset_path, manifest)
    original_sha = file_sha256(dataset_path)
    output = tmp_path / "lerobot"
    report_path = tmp_path / "conversion.json"
    command = [
        sys.executable,
        str(CONVERTER_SCRIPT),
        "--input",
        str(dataset_path),
        "--output",
        str(output),
        "--report",
        str(report_path),
        "--strict",
    ]
    if stage == "generation":
        command.extend(["--lane", "arena"])
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False, timeout=30)

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["write_result"]["status"] == "ok"
    assert json.loads((output / "meta/info.json").read_text())["total_frames"] == 2
    assert next((output / "data").rglob("*.parquet")).is_file()
    assert next((output / "videos").rglob("*.mp4")).is_file()
    assert json.loads((output / "meta/e2e_workflow_lineage.json").read_text()) == {
        "lane": "arena",
        "source_hdf5_sha256": original_sha,
        "source_manifest": manifest,
    }
    assert file_sha256(dataset_path) == original_sha
    with pytest.raises(dataset_manifest_module.DatasetContractError, match="environment"):
        dataset_manifest_module.validate_dataset(dataset_path)
    with pytest.raises(dataset_manifest_module.DatasetContractError, match="environment"):
        dataset_manifest_module.validate_arena_recording_contract(dataset_path, qualification=True)
    with pytest.raises(dataset_manifest_module.DatasetContractError, match="cannot qualify"):
        dataset_manifest_module.validate_dataset(dataset_path, conversion=True, qualification=True)


def test_lane_qualified_converter_rejects_generated_h2_targets_outside_the_pinned_plant(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "h2-source.hdf5"
    generated_path = tmp_path / "h2-generated.hdf5"
    report_path = tmp_path / "report.json"
    write_h2_dataset(source_path, profile=H2_ARENA_PINK_PROFILE)
    write_h2_dataset(generated_path, profile=H2_ARENA_PINK_PROFILE)
    from test_dataset_manifest import _add_h2_initial_state

    for path in (source_path, generated_path):
        with h5py.File(path, "r+") as dataset:
            _add_h2_initial_state(dataset["data/demo_0"])
    stamp_autodata_dataset(generated_path, source_dataset=source_path, stage="generation")

    from e2e_robot_workflows.robots.h2 import load_h2_pink_contract

    contract = load_h2_pink_contract()
    joint_index = contract.expected_joint_order.index(contract.arm_joint_names[0])
    with h5py.File(generated_path, "r+") as dataset:
        dataset["data/demo_0/joint_pos_target"][0, joint_index] = 999.0

    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER_SCRIPT),
            "--config",
            str(H2_DATASET_CONFIG),
            "--input",
            str(generated_path),
            "--lane",
            "arena",
            "--inspect",
            "--strict",
            "--report",
            str(report_path),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "joint_pos_target" in result.stderr
    assert "outside the pinned" in result.stderr
    assert not report_path.exists()


@pytest.mark.parametrize(
    ("profile", "manifested", "config_path", "error"),
    (
        (
            H2_ARENA_PINK_PROFILE,
            True,
            DATASET_CONFIG,
            "controller='h2_agile_pink'",
        ),
        (
            H2_ARENA_JOINT_PROFILE,
            True,
            H2_DATASET_CONFIG,
            "controller='h2_agile_joint'",
        ),
        (
            H2_ARENA_PINK_PROFILE,
            False,
            H2_DATASET_CONFIG,
            "requires a workflow manifest",
        ),
        (
            H2_ARENA_PINK_PROFILE,
            False,
            DATASET_CONFIG,
            "unmanifested action widths [61]",
        ),
    ),
    ids=("h2-pink-with-g1-config", "h2-joint-with-h2-config", "unmanifested-h2", "unmanifested-h2-as-g1"),
)
def test_converter_rejects_dataset_outside_config_admission(
    tmp_path: Path,
    profile: ActionProfile,
    manifested: bool,
    config_path: Path,
    error: str,
) -> None:
    dataset_path = tmp_path / "h2.hdf5"
    write_h2_dataset(dataset_path, profile=profile, manifested=manifested)

    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER_SCRIPT),
            "--config",
            str(config_path),
            "--input",
            str(dataset_path),
            "--inspect",
            "--strict",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert result.stderr.startswith("Dataset config admission failed:")
    assert error in result.stderr
    assert "Traceback" not in result.stderr


def test_converter_rejects_ambiguous_suffix_field_matches(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ambiguous.hdf5"
    report_path = tmp_path / "report.json"
    write_dataset(dataset_path)
    with h5py.File(dataset_path, "r+") as dataset:
        episode = dataset["data/demo_0"]
        del episode["obs/robot_joint_pos"]
        episode.create_dataset("first/robot_joint_pos", shape=(2, 3), dtype="f4")
        episode.create_dataset("second/robot_joint_pos", shape=(2, 3), dtype="f4")

    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER_SCRIPT),
            "--config",
            str(DATASET_CONFIG),
            "--input",
            str(dataset_path),
            "--inspect",
            "--strict",
            "--report",
            str(report_path),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert result.stderr.startswith("Dataset field admission failed:")
    assert "first/robot_joint_pos" in result.stderr
    assert "second/robot_joint_pos" in result.stderr
    assert not report_path.exists()


def test_converter_rejects_missing_nested_config_file_without_a_traceback(tmp_path: Path) -> None:
    dataset_path = tmp_path / "complete.hdf5"
    config_path = tmp_path / "config.yaml"
    write_dataset(dataset_path)
    config_path.write_text(
        """\
data_root: .
hdf5_fields:
  state: robot_joint_pos
  action: joint_pos_target
  video: robot_head_cam_rgb
lerobot_fields:
  state: observation.state
  action: action
  video: observation.images.ego_view
required_fields: [state, action, video]
paths:
  modality_template_path: missing-modality.json
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER_SCRIPT),
            "--config",
            str(config_path),
            "--input",
            str(dataset_path),
            "--inspect",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert result.stderr.startswith("Dataset config admission failed:")
    assert str(tmp_path / "missing-modality.json") in result.stderr
    assert "Traceback" not in result.stderr


def test_converter_rejects_malformed_yaml_without_a_traceback(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("paths: [\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(CONVERTER_SCRIPT), "--config", str(config_path), "--inspect"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert result.stderr.startswith("Dataset config admission failed: invalid YAML")
    assert "Traceback" not in result.stderr


def test_converter_rejects_config_without_manifest_admission_contract(tmp_path: Path) -> None:
    dataset_path = tmp_path / "complete.hdf5"
    config_path = tmp_path / "config.yaml"
    write_dataset(dataset_path)
    config_path.write_text(
        """\
data_root: .
hdf5_fields:
  state: robot_joint_pos
  action: joint_pos_target
  video: robot_head_cam_rgb
lerobot_fields:
  state: observation.state
  action: action
  video: observation.images.ego_view
required_fields: [state, action, video]
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER_SCRIPT),
            "--config",
            str(config_path),
            "--input",
            str(dataset_path),
            "--inspect",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert result.stderr.startswith("Dataset config admission failed: config manifest_admission must be a mapping")
    assert "Traceback" not in result.stderr


def test_converter_rejects_a_present_invalid_h2_manifest_without_a_lane_flag(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid-h2.hdf5"
    with h5py.File(dataset_path, "w") as hdf5_file:
        hdf5_file.create_dataset("data/demo_0/actions", shape=(2, H2_ARENA_PINK_PROFILE.width), dtype="f4")
    manifest = build_manifest(
        producer="recording",
        profile=H2_ARENA_PINK_PROFILE,
        manipulation_hand="left",
        repo_root=REPO_ROOT,
    )
    manifest["action"]["width"] = ARENA_PINK_PROFILE.width
    write_manifest(dataset_path, manifest)

    result = subprocess.run(
        [sys.executable, str(CONVERTER_SCRIPT), "--input", str(dataset_path), "--inspect"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert result.stderr.startswith("Dataset manifest admission failed:")
    assert "Traceback" not in result.stderr


def test_converter_cli_rejects_a_missing_required_field(tmp_path: Path) -> None:
    dataset_path = tmp_path / "missing-video.hdf5"
    report_path = tmp_path / "report.json"
    write_dataset(dataset_path, include_video=False)

    result, report = run_converter(dataset_path, report_path, cwd=tmp_path)

    assert result.returncode == 1
    assert report["status"] == "not_ready"
    assert report["mapping_report"]["ready_for_writer"] is False
    assert report["mapping_report"]["missing_required"] == [
        {
            "semantic_key": "video",
            "hdf5_name": "robot_head_cam_rgb",
            "resolved_hdf5_path": None,
            "lerobot_key": "observation.images.ego_view",
        }
    ]


def test_converter_cli_reports_manifest_admission_failure_without_a_traceback(tmp_path: Path) -> None:
    dataset_path = tmp_path / "unmanifested.hdf5"
    write_dataset(dataset_path)

    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER_SCRIPT),
            "--config",
            str(DATASET_CONFIG),
            "--input",
            str(dataset_path),
            "--inspect",
            "--lane",
            "arena",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert result.stderr.startswith("Dataset manifest admission failed:")
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("include_episode", [False, True], ids=["no-episodes", "empty-episode"])
def test_converter_cli_rejects_datasets_without_fields(tmp_path: Path, include_episode: bool) -> None:
    dataset_path = tmp_path / "empty.hdf5"
    report_path = tmp_path / "report.json"
    write_dataset(dataset_path, include_episode=include_episode, include_fields=False)

    result, report = run_converter(dataset_path, report_path, cwd=tmp_path)

    assert result.returncode == 1
    assert report["inspection"]["episode_count"] == int(include_episode)
    assert report["mapping_report"]["first_episode_checked"] == ("demo_0" if include_episode else None)
    assert report["mapping_report"]["ready_for_writer"] is False
    missing_required = {item["semantic_key"] for item in report["mapping_report"]["missing_required"]}
    assert missing_required == ({"state", "action", "video"} if include_episode else set())


def test_converter_cli_rejects_missing_required_config_mappings(tmp_path: Path) -> None:
    dataset_path = tmp_path / "complete.hdf5"
    report_path = tmp_path / "report.json"
    config_path = tmp_path / "incomplete-config.yaml"
    write_dataset(dataset_path)
    config_path.write_text(
        TEST_G1_ADMISSION
        + """\
data_root: .
hdf5_fields:
  state: robot_joint_pos
  action: joint_pos_target
lerobot_fields:
  state: observation.state
  action: ""
  video: observation.images.ego_view
required_fields: [state, action, video]
""",
        encoding="utf-8",
    )

    result, report = run_converter(dataset_path, report_path, cwd=tmp_path, config_path=config_path)

    assert result.returncode == 1
    assert report["mapping_report"]["missing_required"] == []
    assert report["mapping_report"]["missing_required_source_mappings"] == ["video"]
    assert report["mapping_report"]["missing_required_target_mappings"] == ["action"]
    assert report["mapping_report"]["ready_for_writer"] is False


def test_converter_cli_default_write_invokes_configured_writer(tmp_path: Path) -> None:
    """Conversion admits generated lane data and preserves its exact lineage."""

    source_path = tmp_path / "source.hdf5"
    dataset_path = tmp_path / "complete.hdf5"
    report_path = tmp_path / "report.json"
    config_path = tmp_path / "config" / "stub-config.yaml"
    config_path.parent.mkdir()
    write_dataset(source_path)
    write_manifest(
        source_path,
        build_manifest(
            producer="recording",
            profile=ARENA_PINK_PROFILE,
            manipulation_hand="left",
            repo_root=REPO_ROOT,
        ),
    )
    write_dataset(dataset_path)
    generation_manifest = build_manifest(
        producer="autodata",
        profile=ARENA_PINK_PROFILE,
        source_dataset=source_path,
        manipulation_hand="left",
        repo_root=REPO_ROOT,
    )
    write_manifest(dataset_path, generation_manifest)
    # A tiny stub writer, importable from tmp_path via the config's writer entry point.
    (tmp_path / "stub_writer.py").write_text(
        "from pathlib import Path\n\n\n"
        "def write(hdf5_path, config, output_root):\n"
        '    (Path(output_root) / "meta").mkdir(parents=True, exist_ok=True)\n'
        '    (Path(output_root) / "sentinel").write_text("ok")\n'
        '    return {"status": "ok", "output_root": str(output_root), "episodes": 1}\n',
        encoding="utf-8",
    )
    config_path.write_text(
        TEST_G1_ADMISSION
        + """\
data_root: ./outputs
writer: stub_writer:write
hdf5_fields:
  state: robot_joint_pos
  action: joint_pos_target
  video: robot_head_cam_rgb
lerobot_fields:
  state: observation.state
  action: action
  video: observation.images.ego_view
required_fields: [state, action, video]
""",
        encoding="utf-8",
    )
    out_root = config_path.parent / "outputs" / "lerobot"
    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER_SCRIPT),
            "--config",
            str(config_path),
            "--input",
            str(dataset_path),
            "--lane",
            "arena",
            "--report",
            str(report_path),
        ],
        cwd=caller_dir,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(tmp_path), str(REPO_ROOT / "src")))},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stderr
    assert report["mapping_report"]["ready_for_writer"] is True
    lineage_path = out_root / "meta/e2e_workflow_lineage.json"
    assert report["write_result"] == {
        "status": "ok",
        "output_root": str(out_root),
        "episodes": 1,
        "lineage_path": str(lineage_path),
    }
    assert json.loads(lineage_path.read_text(encoding="utf-8")) == {
        "lane": "arena",
        "source_hdf5_sha256": file_sha256(dataset_path),
        "source_manifest": generation_manifest,
    }
    assert (out_root / "sentinel").is_file()
    assert "stats.json" in report["next_step"]


def test_converter_report_cannot_replace_input_or_dataset_content(tmp_path: Path) -> None:
    dataset_path = tmp_path / "complete.hdf5"
    write_dataset(dataset_path)
    original = dataset_path.read_bytes()

    input_collision = subprocess.run(
        [
            sys.executable,
            str(CONVERTER_SCRIPT),
            "--config",
            str(DATASET_CONFIG),
            "--input",
            str(dataset_path),
            "--inspect",
            "--report",
            str(dataset_path),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert input_collision.returncode != 0
    assert "Report path must differ from every conversion input" in input_collision.stderr
    assert dataset_path.read_bytes() == original

    output_root = tmp_path / "lerobot"
    output_collision = subprocess.run(
        [
            sys.executable,
            str(CONVERTER_SCRIPT),
            "--config",
            str(DATASET_CONFIG),
            "--input",
            str(dataset_path),
            "--output",
            str(output_root),
            "--report",
            str(output_root / "meta" / "info.json"),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert output_collision.returncode != 0
    assert "Report path must remain outside LeRobot output directory" in output_collision.stderr
    assert not output_root.exists()
    assert dataset_path.read_bytes() == original


def test_converter_rejects_output_below_report_path_before_writing(tmp_path: Path) -> None:
    dataset_path = tmp_path / "complete.hdf5"
    report_path = tmp_path / "report.json"
    output_root = report_path / "lerobot"
    write_dataset(dataset_path)
    original = dataset_path.read_bytes()

    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER_SCRIPT),
            "--config",
            str(DATASET_CONFIG),
            "--input",
            str(dataset_path),
            "--output",
            str(output_root),
            "--report",
            str(report_path),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "LeRobot output directory must remain outside report path" in result.stderr
    assert not report_path.exists()
    assert not output_root.exists()
    assert dataset_path.read_bytes() == original


def test_converter_does_not_publish_before_lineage_is_complete(tmp_path: Path) -> None:
    source_path = tmp_path / "source.hdf5"
    dataset_path = tmp_path / "generated.hdf5"
    report_path = tmp_path / "report.json"
    output_root = tmp_path / "lerobot"
    config_path = tmp_path / "config.yaml"
    write_dataset(source_path)
    write_manifest(
        source_path,
        build_manifest(
            producer="recording",
            profile=ARENA_PINK_PROFILE,
            manipulation_hand="left",
            repo_root=REPO_ROOT,
        ),
    )
    write_dataset(dataset_path)
    write_manifest(
        dataset_path,
        build_manifest(
            producer="autodata",
            profile=ARENA_PINK_PROFILE,
            source_dataset=source_path,
            manipulation_hand="left",
            repo_root=REPO_ROOT,
        ),
    )
    (tmp_path / "blocking_writer.py").write_text(
        "from pathlib import Path\n\n\n"
        "def write(hdf5_path, config, output_root):\n"
        "    root = Path(output_root)\n"
        "    root.mkdir(parents=True)\n"
        '    (root / "meta").write_text("blocks lineage directory")\n'
        '    return {"status": "ok", "output_root": str(root)}\n',
        encoding="utf-8",
    )
    config_path.write_text(
        TEST_G1_ADMISSION
        + """\
data_root: .
writer: blocking_writer:write
hdf5_fields:
  state: robot_joint_pos
  action: joint_pos_target
  video: robot_head_cam_rgb
lerobot_fields:
  state: observation.state
  action: action
  video: observation.images.ego_view
required_fields: [state, action, video]
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER_SCRIPT),
            "--config",
            str(config_path),
            "--input",
            str(dataset_path),
            "--output",
            str(output_root),
            "--lane",
            "arena",
            "--report",
            str(report_path),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(tmp_path), str(REPO_ROOT / "src")))},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "write_failed"
    assert not output_root.exists()
    assert not list(tmp_path.glob(".lerobot.conversion-*"))


@pytest.mark.parametrize("protected_name", ("config.yaml", "modality.json"))
def test_converter_report_cannot_replace_config_inputs(tmp_path: Path, protected_name: str) -> None:
    dataset_path = tmp_path / "complete.hdf5"
    config_path = tmp_path / "config.yaml"
    modality_path = tmp_path / "modality.json"
    write_dataset(dataset_path)
    modality_path.write_text("{}\n", encoding="utf-8")
    config_path.write_text(
        TEST_G1_ADMISSION
        + """\
data_root: .
hdf5_fields:
  state: robot_joint_pos
  action: joint_pos_target
  video: robot_head_cam_rgb
lerobot_fields:
  state: observation.state
  action: action
  video: observation.images.ego_view
required_fields: [state, action, video]
paths:
  modality_template_path: modality.json
""",
        encoding="utf-8",
    )
    protected_path = tmp_path / protected_name
    original = protected_path.read_bytes()

    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER_SCRIPT),
            "--config",
            str(config_path),
            "--input",
            str(dataset_path),
            "--inspect",
            "--report",
            str(protected_path),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "Report path must differ from every conversion input" in result.stderr
    assert protected_path.read_bytes() == original

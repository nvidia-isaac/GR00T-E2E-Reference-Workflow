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

"""H2 LeRobot conversion preserves full joints and the separate navigation head."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from e2e_robot_workflows.gr00t.lerobot_writer import _staged_output_path, write_lerobot_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_h2_lerobot_writer_remaps_all_75_joints_and_keeps_navigation(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "configs/stages/gr00t_h2_lerobot_dataset.yaml").read_text(encoding="utf-8"))
    source = tmp_path / "h2.hdf5"
    state_row = np.arange(75, dtype=np.float32)
    action_row = state_row + 100.0
    navigation = np.asarray([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]], dtype=np.float32)
    with h5py.File(source, "w") as dataset:
        episode = dataset.create_group("data/demo_0")
        obs = episode.create_group("obs")
        obs.create_dataset("robot_joint_pos", data=np.repeat(state_row[None], 2, axis=0))
        episode.create_dataset("joint_pos_target", data=np.repeat(action_row[None], 2, axis=0))
        episode.create_dataset("navigate_cmd", data=navigation)
        camera = episode.create_group("camera_obs")
        camera.create_dataset("robot_head_cam_rgb", data=np.zeros((2, 4, 6, 3), dtype=np.uint8))

    output = tmp_path / "lerobot"
    summary = write_lerobot_dataset(source, config, output)

    assert summary["status"] == "ok"
    assert (summary["episodes"], summary["frames"]) == (1, 2)
    assert (summary["state_dim"], summary["action_dim"]) == (75, 75)
    assert summary["optional_columns"] == {"teleop.navigate_command": 3}
    assert summary["dropped_modality_groups"] == []
    assert summary["warnings"] == []
    table = pq.read_table(output / "data/chunk-000/episode_000000.parquet").to_pydict()
    sim_order = yaml.safe_load((REPO_ROOT / config["robot"]["state_joints_config_path"]).read_text(encoding="utf-8"))[
        "joints"
    ]
    groups = yaml.safe_load((REPO_ROOT / config["robot"]["policy_joints_config_path"]).read_text(encoding="utf-8"))[
        "joints"
    ]
    grouped_names = [joint for joints in groups.values() for joint in joints]
    expected_indices = [sim_order[joint] for joint in grouped_names]

    assert table["observation.state"][0] == state_row[expected_indices].tolist()
    assert table["action"][0] == action_row[expected_indices].tolist()
    assert table["teleop.navigate_command"] == navigation.tolist()
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert info["features"]["observation.state"]["names"] == grouped_names
    assert info["features"]["action"]["names"] == grouped_names


def test_h2_lerobot_writer_rejects_missing_required_navigation(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "configs/stages/gr00t_h2_lerobot_dataset.yaml").read_text(encoding="utf-8"))
    source = tmp_path / "h2-without-navigation.hdf5"
    with h5py.File(source, "w") as dataset:
        episode = dataset.create_group("data/demo_0")
        episode.create_dataset("obs/robot_joint_pos", shape=(2, 75), dtype="f4")
        episode.create_dataset("joint_pos_target", shape=(2, 75), dtype="f4")
        episode.create_dataset("camera_obs/robot_head_cam_rgb", shape=(2, 4, 6, 3), dtype="u1")

    with pytest.raises(ValueError, match="missing required field 'teleop.navigate_command'"):
        write_lerobot_dataset(source, config, tmp_path / "lerobot")


def test_h2_lerobot_writer_rejects_ambiguous_required_navigation(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "configs/stages/gr00t_h2_lerobot_dataset.yaml").read_text(encoding="utf-8"))
    source = tmp_path / "h2-ambiguous-navigation.hdf5"
    with h5py.File(source, "w") as dataset:
        episode = dataset.create_group("data/demo_0")
        episode.create_dataset("obs/robot_joint_pos", shape=(2, 75), dtype="f4")
        episode.create_dataset("joint_pos_target", shape=(2, 75), dtype="f4")
        episode.create_dataset("camera_obs/robot_head_cam_rgb", shape=(2, 4, 6, 3), dtype="u1")
        episode.create_dataset("first/navigate_cmd", shape=(2, 3), dtype="f4")
        episode.create_dataset("second/navigate_cmd", shape=(2, 3), dtype="f4")
    output = tmp_path / "lerobot"

    with pytest.raises(ValueError, match="field 'navigate_cmd' is ambiguous"):
        write_lerobot_dataset(source, config, output)
    assert not output.exists()


def test_lerobot_writer_refuses_destructive_output_roots(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "configs/stages/gr00t_h2_lerobot_dataset.yaml").read_text(encoding="utf-8"))
    source = tmp_path / "source.hdf5"
    source.touch()
    sibling = tmp_path / "keep.txt"
    sibling.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="must not contain its input"):
        write_lerobot_dataset(source, config, tmp_path)

    existing_output = tmp_path / "existing"
    existing_output.mkdir()
    sentinel = existing_output / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_lerobot_dataset(source, config, existing_output)

    assert source.is_file()
    assert sibling.read_text(encoding="utf-8") == "keep"
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("template", ("../escaped.parquet", "/tmp/escaped.parquet", r"..\escaped.parquet"))
def test_configured_dataset_paths_cannot_escape_staging(tmp_path: Path, template: str) -> None:
    work = tmp_path / "work"
    work.mkdir()

    with pytest.raises(ValueError, match="escapes the dataset staging directory"):
        _staged_output_path(work, template, label="data_path", required_root="data", episode_index=0)


def test_configured_dataset_paths_cannot_overwrite_metadata(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()

    with pytest.raises(ValueError, match="must remain below data"):
        _staged_output_path(
            work,
            "meta/info.json",
            label="data_path",
            required_root="data",
            episode_index=0,
        )

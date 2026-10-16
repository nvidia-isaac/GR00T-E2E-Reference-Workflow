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

"""Safety checks for workflow artifact paths."""

import argparse
import os
import runpy
import sys
import types
from pathlib import Path

import pytest

from e2e_robot_workflows.data.artifact_paths import (
    atomic_write_text,
    recorder_hdf5_output_path,
    require_distinct_artifact_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_recorder_output_path_matches_isaac_lab_hdf5_suffix_rewrite(tmp_path: Path) -> None:
    assert recorder_hdf5_output_path(tmp_path / "recording.h5") == tmp_path / "recording.hdf5"
    assert recorder_hdf5_output_path(tmp_path / "recording.hdf5") == tmp_path / "recording.hdf5"


def test_recorder_rejects_snapshot_at_normalized_dataset_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    isaaclab = types.ModuleType("isaaclab")
    isaaclab.__path__ = []
    app = types.ModuleType("isaaclab.app")

    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser: argparse.ArgumentParser) -> None:
            parser.add_argument("--enable_cameras", action="store_true")
            parser.add_argument("--xr", action="store_true")

        def __init__(self, _args: object) -> None:
            raise AssertionError("artifact path admission must run before AppLauncher")

    app.AppLauncher = AppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", app)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_demos.py",
            "--dataset_file",
            str(tmp_path / "recording.h5"),
            "--environment_snapshot",
            str(tmp_path / "recording.hdf5"),
        ],
    )

    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(REPO_ROOT / "scripts/record_demos.py"), run_name="__main__")

    assert error.value.code == 2
    assert "Environment snapshot must differ from input dataset" in capsys.readouterr().err


def test_result_path_cannot_replace_its_input_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "recording.hdf5"
    dataset.touch()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="must differ from input dataset"):
        require_distinct_artifact_paths(dataset, Path(".") / dataset.name, destination_label="Replay result file")


def test_distinct_result_path_is_accepted(tmp_path: Path) -> None:
    require_distinct_artifact_paths(
        tmp_path / "recording.hdf5",
        tmp_path / "replay.json",
        destination_label="Replay result file",
    )


def test_artifact_paths_reject_a_hardlink_alias(tmp_path: Path) -> None:
    source = tmp_path / "recording.hdf5"
    alias = tmp_path / "generated.hdf5"
    source.write_bytes(b"recording")
    os.link(source, alias)

    with pytest.raises(ValueError, match="must differ from input dataset"):
        require_distinct_artifact_paths(source, alias, destination_label="Generated dataset")


def test_artifact_paths_reject_a_dangling_symlink_alias(tmp_path: Path) -> None:
    target = tmp_path / "generated.hdf5"
    alias = tmp_path / "result.json"
    alias.symlink_to(target)

    with pytest.raises(ValueError, match="must differ from input dataset"):
        require_distinct_artifact_paths(target, alias, destination_label="Result file")


def test_atomic_text_write_preserves_a_preexisting_temp_like_path(tmp_path: Path) -> None:
    source = tmp_path / ".replay.json.tmp"
    source.write_bytes(b"source dataset")
    destination = tmp_path / "replay.json"

    atomic_write_text(destination, '{"status": "ok"}\n')

    assert source.read_bytes() == b"source dataset"
    assert destination.read_text(encoding="utf-8") == '{"status": "ok"}\n'
    assert not list(tmp_path.glob(".replay.json.*.tmp"))


def test_atomic_text_write_expands_user_like_path_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    literal_home = tmp_path / "~"
    literal_home.mkdir()
    source = literal_home / "result.json"
    source.write_text("source", encoding="utf-8")
    expanded_home = tmp_path / "home"
    expanded_home.mkdir()
    monkeypatch.setenv("HOME", str(expanded_home))
    monkeypatch.chdir(tmp_path)

    destination = Path("~/result.json")
    require_distinct_artifact_paths(source, destination, destination_label="Result file")
    atomic_write_text(destination, "result\n")

    assert source.read_text(encoding="utf-8") == "source"
    assert (expanded_home / "result.json").read_text(encoding="utf-8") == "result\n"

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

"""Behavioral tests for the Arena recording GPU validation mode."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = REPO_ROOT / "tools/simulation/check_arena_recording.py"
VALIDATION_SCRIPT = REPO_ROOT / "tools/validate_arena_recording.sh"


def write_recording(
    path: Path,
    *,
    steps: int = 2,
    joints: int = 43,
    camera_group: str | None = "camera_obs",
    target_width: int | None = None,
    include_target: bool = True,
    camera_frames: int | None = None,
    include_datagen: bool = True,
) -> None:
    """Write an exported recording shaped like the ones the Arena path produces."""

    with h5py.File(path, "w") as recording:
        data = recording.create_group("data")
        episode = data.create_group("demo_0")
        episode.attrs["num_samples"] = steps
        actions = np.zeros((steps, 23), dtype=np.float32)
        actions[:, 8] = 1.0
        actions[:, 15] = 1.0
        episode.create_dataset("actions", data=actions)
        if include_target:
            episode.create_dataset("joint_pos_target", shape=(steps, target_width or joints), dtype="f4")
        episode.create_group("obs").create_dataset("robot_joint_pos", shape=(steps, joints), dtype="f4")
        if camera_group is not None:
            frames = steps if camera_frames is None else camera_frames
            episode.require_group(camera_group).create_dataset(
                "robot_head_cam_rgb", shape=(frames, 4, 5, 3), dtype="u1"
            )
        if include_datagen:
            datagen = episode.require_group("obs/datagen_info")
            for hand in ("left", "right"):
                datagen.create_dataset(f"eef_pose/{hand}", data=np.tile(np.eye(4), (steps, 1, 1)))
                datagen.create_dataset(f"target_eef_pose/{hand}", data=np.tile(np.eye(4), (steps, 1, 1)))
                datagen.create_dataset(
                    f"subtask_term_signals/{hand}_hand_holds_object",
                    shape=(steps,),
                    dtype="?",
                )
            datagen.create_dataset("object_pose/pick_object", data=np.tile(np.eye(4), (steps, 1, 1)))


def run_check(*datasets: tuple[str, Path], result_path: Path) -> tuple[subprocess.CompletedProcess[str], dict]:
    arguments = [argument for name, path in datasets for argument in ("--dataset", f"{name}={path}")]
    completed = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--result-path", str(result_path), *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed, json.loads(result_path.read_text(encoding="utf-8"))


def test_a_complete_recording_passes_for_every_embodiment(tmp_path: Path) -> None:
    """The check must accept what the recording path is supposed to produce."""

    pink = tmp_path / "pink.hdf5"
    joint = tmp_path / "joint.hdf5"
    write_recording(pink)
    write_recording(joint)

    completed, result = run_check(
        ("g1_wbc_agile_pink", pink),
        ("g1_wbc_agile_joint", joint),
        result_path=tmp_path / "metrics.json",
    )

    assert completed.returncode == 0, completed.stderr
    assert result["status"] == "ok"
    assert result["failures"] == {}
    assert sorted(result["recordings"]) == ["g1_wbc_agile_joint", "g1_wbc_agile_pink"]
    assert result["recordings"]["g1_wbc_agile_pink"]["joint_pos_target"]["shape"] == [2, 43]
    assert result["recordings"]["g1_wbc_agile_pink"]["camera"]["path"] == "camera_obs/robot_head_cam_rgb"
    assert result["recordings"]["g1_wbc_agile_pink"]["datagen_info"]["direct"] is True
    assert result["recordings"]["g1_wbc_agile_pink"]["datagen_info"]["pose_action_errors"]["left"] == {
        "position_max_abs": 0.0,
        "rotation_max_abs": 0.0,
    }
    assert result["recordings"]["g1_wbc_agile_pink"]["datagen_info"]["first_hold_pose_max_abs"] == {
        "left": 0.0,
        "right": 0.0,
    }


def test_the_camera_is_found_wherever_the_environment_publishes_it(tmp_path: Path) -> None:
    """Arena nests the camera under its own group, and the built-in task does not.

    The converter resolves the field by suffix, so a check keyed to one layout would fail
    a recording the fine-tuning stage would have consumed happily.
    """

    dataset = tmp_path / "recording.hdf5"
    write_recording(dataset, camera_group="obs")

    completed, result = run_check(("g1_wbc_agile_joint", dataset), result_path=tmp_path / "metrics.json")

    assert completed.returncode == 0, completed.stderr
    assert result["recordings"]["g1_wbc_agile_joint"]["camera"]["path"] == "obs/robot_head_cam_rgb"


@pytest.mark.parametrize(
    ("recording_arguments", "expected_problem"),
    [
        pytest.param({"include_target": False}, "no joint_pos_target dataset", id="missing-action-column"),
        pytest.param({"target_width": 23}, "width 23 != robot_joint_pos width 43", id="wrong-action-width"),
        pytest.param({"camera_group": None}, "no robot_head_cam_rgb frames", id="missing-camera"),
        pytest.param({"camera_frames": 0}, "robot_head_cam_rgb has 0 frames", id="empty-camera"),
        pytest.param({"steps": 0}, "actions has 0 steps", id="empty-actions"),
    ],
)
def test_an_unusable_recording_fails_the_check(
    tmp_path: Path, recording_arguments: dict[str, object], expected_problem: str
) -> None:
    """Each field the downstream stage needs must be reported when it is absent.

    ``joint_pos_target`` is width-checked against the recorded joint state because it is
    the one action column that means the same thing under either embodiment: a recording
    whose action column is the raw 23-D Pink command instead is silently untrainable.
    """

    dataset = tmp_path / "recording.hdf5"
    write_recording(dataset, **recording_arguments)  # type: ignore[arg-type]
    result_path = tmp_path / "metrics.json"

    completed, result = run_check(("g1_wbc_agile_pink", dataset), result_path=result_path)

    assert completed.returncode == 1
    assert result["status"] == "error"
    problems = " ".join(result["recordings"]["g1_wbc_agile_pink"]["problems"])
    assert expected_problem in problems
    assert expected_problem in " ".join(result["failures"]["g1_wbc_agile_pink"])


def test_a_recording_that_was_never_written_fails_the_check(tmp_path: Path) -> None:
    """A launcher that exits before exporting must not read as a passing run."""

    completed, result = run_check(
        ("g1_wbc_agile_pink", tmp_path / "absent.hdf5"),
        result_path=tmp_path / "metrics.json",
    )

    assert completed.returncode == 1
    assert result["recordings"]["g1_wbc_agile_pink"]["problems"] == ["recording was not written"]


def test_pink_recording_rejects_quaternion_semantic_mismatch(tmp_path: Path) -> None:
    dataset = tmp_path / "wrong-quaternion.hdf5"
    write_recording(dataset)
    with h5py.File(dataset, "r+") as recording:
        actions = recording["data/demo_0/actions"]
        actions[:, 5:9] = np.asarray([1.0, 0.0, 0.0, 0.0])

    completed, result = run_check(
        ("g1_wbc_agile_pink", dataset),
        result_path=tmp_path / "metrics.json",
    )

    assert completed.returncode == 1
    problems = " ".join(result["recordings"]["g1_wbc_agile_pink"]["problems"])
    assert "left target_eef_pose disagrees with the recorded XYZW action" in problems


def test_the_mode_records_each_embodiment_before_checking_the_dataset() -> None:
    """A GPU-and-Arena-only run is asserted on as text, since host CI cannot execute it.

    What matters is the shape of the run: a finite scripted take per embodiment, exported
    to a file, then the dataset contract and the conversion contract checked against it.
    """

    script = VALIDATION_SCRIPT.read_text(encoding="utf-8")

    assert "EMBODIMENTS=(g1_wbc_agile_pink g1_wbc_agile_joint)" in script
    assert "./tools/launch_recorder.sh" in script
    assert "--sample_steps" in script
    assert "--sample_export" in script
    assert '--dataset_file "${dataset}"' in script
    assert "tools/simulation/check_arena_recording.py" in script
    assert "scripts/convert_hdf5_to_lerobot.py" in script
    assert "--strict" in script
    assert script.index("./tools/launch_recorder.sh") < script.index("tools/simulation/check_arena_recording.py")

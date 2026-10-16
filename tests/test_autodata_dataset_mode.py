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

"""CPU-only tests for AutoData dataset mode selection."""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from e2e_robot_workflows.autodata.dataset_mode import (
    DatasetModeError,
    detect_manipulation_hand,
    detect_wbc_mode,
    main,
    resolve_autodata_profile,
    resolve_manipulation_hand,
    resolve_wbc_mode,
)
from e2e_robot_workflows.data.dataset_manifest import (
    ARENA_JOINT_PROFILE,
    ARENA_PINK_PROFILE,
    NATIVE_AGILE_PROFILE,
    build_manifest,
    write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_dataset(path: Path, action_widths: list[int], hands: list[str] | None = None) -> None:
    with h5py.File(path, "w") as dataset:
        data = dataset.create_group("data")
        for index, width in enumerate(action_widths):
            episode = data.create_group(f"demo_{index}")
            episode.create_dataset("actions", shape=(4, width), dtype="f4")
            if hands is None:
                continue
            obs = episode.create_group("obs")
            obs.create_dataset(
                "object_pos",
                data=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.01], [0.0, 0.0, 0.04], [0.0, 0.0, 0.06]],
            )
            near, far = [0.1, 0.0, 0.05], [0.4, 0.0, 0.05]
            obs.create_dataset("left_eef_pos", data=[near if hands[index] == "left" else far] * 4)
            obs.create_dataset("right_eef_pos", data=[near if hands[index] == "right" else far] * 4)


@pytest.mark.parametrize(("action_width", "expected_mode"), [(28, "fixed"), (32, "agile")])
def test_detects_mode_from_every_episode(tmp_path: Path, action_width: int, expected_mode: str) -> None:
    dataset_path = tmp_path / "source.hdf5"
    _write_dataset(dataset_path, [action_width, action_width])

    assert detect_wbc_mode(dataset_path) == expected_mode
    assert resolve_wbc_mode(dataset_path, expected_mode) == expected_mode


def test_rejects_explicit_mode_that_disagrees_with_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "source.hdf5"
    _write_dataset(dataset_path, [32])

    with pytest.raises(DatasetModeError, match=r"32-D agile actions.*fixed.*28-D"):
        resolve_wbc_mode(dataset_path, "fixed")


def test_rejects_mixed_action_widths(tmp_path: Path) -> None:
    dataset_path = tmp_path / "source.hdf5"
    _write_dataset(dataset_path, [28, 32])

    with pytest.raises(DatasetModeError, match="inconsistent action widths"):
        detect_wbc_mode(dataset_path)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_detects_manipulation_hand_from_lifted_object_trajectory(tmp_path: Path, hand: str) -> None:
    dataset_path = tmp_path / "source.hdf5"
    _write_dataset(dataset_path, [32, 32], [hand, hand])

    assert detect_manipulation_hand(dataset_path) == hand
    assert resolve_manipulation_hand(dataset_path, hand) == hand


def test_rejects_explicit_hand_that_disagrees_with_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "source.hdf5"
    _write_dataset(dataset_path, [32], ["left"])

    with pytest.raises(DatasetModeError, match=r"left-hand manipulation.*right.*right-hand task descriptor"):
        resolve_manipulation_hand(dataset_path, "right")


def test_rejects_datasets_that_switch_manipulation_hands(tmp_path: Path) -> None:
    dataset_path = tmp_path / "source.hdf5"
    _write_dataset(dataset_path, [32, 32], ["left", "right"])

    with pytest.raises(DatasetModeError, match="different manipulation hands"):
        detect_manipulation_hand(dataset_path)


def test_cli_prints_resolved_profile(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset_path = tmp_path / "source.hdf5"
    _write_dataset(dataset_path, [32], ["left"])
    write_manifest(
        dataset_path,
        build_manifest(
            producer="recording",
            profile=NATIVE_AGILE_PROFILE,
            manipulation_hand="left",
            repo_root=REPO_ROOT,
        ),
    )

    assert (
        main(
            [
                "--input-file",
                str(dataset_path),
                "--requested-lane",
                "native",
                "--required-stage",
                "recording",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "native agile left"


def test_legacy_inspection_rejects_unknown_action_width(tmp_path: Path) -> None:
    dataset_path = tmp_path / "source.hdf5"
    _write_dataset(dataset_path, [31])

    with pytest.raises(DatasetModeError, match=r"unsupported 31-D actions.*28-D \(fixed\).*32-D \(agile\)"):
        detect_wbc_mode(dataset_path)


@pytest.mark.parametrize(
    ("profile", "lane", "controller"),
    [
        (NATIVE_AGILE_PROFILE, "native", "agile"),
        (ARENA_PINK_PROFILE, "arena", "g1_wbc_agile_pink"),
    ],
)
def test_manifest_selects_supported_autodata_route(tmp_path: Path, profile, lane: str, controller: str) -> None:
    dataset_path = tmp_path / "source.hdf5"
    _write_dataset(dataset_path, [profile.width])
    write_manifest(
        dataset_path,
        build_manifest(producer="recording", profile=profile, repo_root=REPO_ROOT),
    )

    selected = resolve_autodata_profile(
        dataset_path,
        requested_lane=lane,
        required_stage="recording",
    )

    assert selected.lane == lane
    assert selected.controller == controller
    assert selected.manipulation_hand == "right"


def test_manifest_route_rejects_missing_manifest(tmp_path: Path) -> None:
    dataset_path = tmp_path / "legacy.hdf5"
    _write_dataset(dataset_path, [NATIVE_AGILE_PROFILE.width])

    with pytest.raises(DatasetModeError, match="migrate it with an explicit lane"):
        resolve_autodata_profile(dataset_path, requested_lane="native", required_stage="recording")


def test_manifest_route_rejects_requested_lane_mismatch(tmp_path: Path) -> None:
    dataset_path = tmp_path / "source.hdf5"
    _write_dataset(dataset_path, [ARENA_PINK_PROFILE.width])
    write_manifest(
        dataset_path,
        build_manifest(producer="recording", profile=ARENA_PINK_PROFILE, repo_root=REPO_ROOT),
    )

    with pytest.raises(DatasetModeError, match="does not match requested 'native'"):
        resolve_autodata_profile(dataset_path, requested_lane="native", required_stage="recording")


def test_manifest_route_rejects_arena_joint_profile(tmp_path: Path) -> None:
    dataset_path = tmp_path / "source.hdf5"
    _write_dataset(dataset_path, [ARENA_JOINT_PROFILE.width])
    write_manifest(
        dataset_path,
        build_manifest(producer="recording", profile=ARENA_JOINT_PROFILE, repo_root=REPO_ROOT),
    )

    with pytest.raises(DatasetModeError, match="not AutoData-compatible"):
        resolve_autodata_profile(dataset_path, requested_lane="arena", required_stage="recording")


def test_manifest_route_rejects_native_mode_for_arena(tmp_path: Path) -> None:
    dataset_path = tmp_path / "source.hdf5"
    _write_dataset(dataset_path, [ARENA_PINK_PROFILE.width])
    write_manifest(
        dataset_path,
        build_manifest(producer="recording", profile=ARENA_PINK_PROFILE, repo_root=REPO_ROOT),
    )

    with pytest.raises(DatasetModeError, match="only valid for native"):
        resolve_autodata_profile(
            dataset_path,
            requested_lane="arena",
            requested_mode="fixed",
            required_stage="recording",
        )

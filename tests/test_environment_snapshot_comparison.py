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

"""Tests for the cross-runtime realized-environment comparison."""

from __future__ import annotations

from copy import deepcopy

import pytest

from tools.compare_environment_snapshots import compare_snapshots


def _snapshot(runtime: str) -> dict:
    static_entity = {
        "usd": "asset.usd",
        "scale": [1.0, 1.0, 1.0],
        "initial_position": [0.0, 0.0, 0.0],
        "initial_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    robot = {
        "usd": "g1.usd",
        "scale": [1.0, 1.0, 1.0],
        "initial_position": [0.25, 0.08, 0.792],
        "initial_rotation_xyzw": [0.0, 0.0, 0.70711, 0.70711],
        "runtime_root_pose_wxyz": [0.25, 0.08, 0.765, 0.70711, 0.0, 0.0, 0.70711],
    }
    return {
        "contract": {
            "environment_id": "static_apple_v0",
            "environment_contract_digest": "sha256:test",
            "pick_spawn_position": [0.15, 0.4, 0.7718],
            "pick_spawn_xy_range_m": 0.05,
            "destination_spawn_position": [0.35, 0.4, 0.7547],
            "destination_spawn_xy_range_m": 0.03,
            "task": {
                "min_object_height_above_target_m": -0.01,
                "max_object_height_above_support_m": 0.10,
                "min_supported_height_below_support_m": 0.05,
                "release_clearance_m": 0.10,
            },
        },
        "mode": {"runtime": runtime},
        "realized": {
            "physics_dt_s": 0.005,
            "decimation": 4,
            "episode_length_s": 6.0,
            "entities": {
                "background": deepcopy(static_entity),
                "support": deepcopy(static_entity),
                "pick_object": deepcopy(static_entity),
                "destination": deepcopy(static_entity),
                "robot": robot,
            },
        },
    }


def test_comparison_includes_robot_initial_and_runtime_height() -> None:
    result = compare_snapshots(_snapshot("isaaclab"), _snapshot("arena"))

    assert result["status"] == "ok"
    assert result["runtime_positions_compared"] == ["robot"]


def test_comparison_rejects_robot_spawned_on_the_floor() -> None:
    recording = _snapshot("isaaclab")
    evaluation = _snapshot("arena")
    evaluation["realized"]["entities"]["robot"]["initial_position"][2] = 0.0
    evaluation["realized"]["entities"]["robot"]["runtime_root_pose_wxyz"][2] = 0.0

    with pytest.raises(AssertionError, match=r"realized\.entities\.robot\.initial_position"):
        compare_snapshots(recording, evaluation)


def test_comparison_rejects_placement_threshold_drift() -> None:
    recording = _snapshot("isaaclab")
    evaluation = _snapshot("arena")
    evaluation["contract"]["task"]["release_clearance_m"] = 0.20

    with pytest.raises(AssertionError, match=r"contract\.task\.release_clearance_m"):
        compare_snapshots(recording, evaluation)


def test_comparison_rejects_live_robot_height_drift() -> None:
    recording = _snapshot("isaaclab")
    evaluation = _snapshot("arena")
    evaluation["realized"]["entities"]["robot"]["runtime_root_pose_wxyz"][2] = 0.5

    with pytest.raises(AssertionError, match=r"realized\.entities\.robot\.runtime_height"):
        compare_snapshots(recording, evaluation)


def test_comparison_rejects_scene_entity_runtime_pose_presence_mismatch() -> None:
    recording = _snapshot("isaaclab")
    evaluation = _snapshot("arena")
    evaluation["realized"]["entities"]["pick_object"]["runtime_root_pose_wxyz"] = [
        0.0,
        0.0,
        0.5,
        1.0,
        0.0,
        0.0,
        0.0,
    ]

    with pytest.raises(AssertionError, match=r"pick_object\.runtime_root_pose_wxyz"):
        compare_snapshots(recording, evaluation)


def test_comparison_allows_arena_only_runtime_pose_for_static_destination() -> None:
    recording = _snapshot("isaaclab")
    evaluation = _snapshot("arena")
    evaluation["realized"]["entities"]["destination"]["runtime_root_pose_wxyz"] = [
        0.0,
        0.0,
        0.5,
        1.0,
        0.0,
        0.0,
        0.0,
    ]

    assert compare_snapshots(recording, evaluation)["status"] == "ok"


def test_comparison_rejects_robot_runtime_pose_presence_mismatch() -> None:
    recording = _snapshot("isaaclab")
    evaluation = _snapshot("arena")
    evaluation["realized"]["entities"]["robot"].pop("runtime_root_pose_wxyz")

    with pytest.raises(AssertionError, match=r"robot\.runtime_root_pose_wxyz"):
        compare_snapshots(recording, evaluation)


def _with_pick_runtime(snapshot: dict, xy: tuple[float, float], z: float = 0.7718) -> dict:
    snapshot["realized"]["entities"]["pick_object"]["runtime_root_pose_wxyz"] = [
        xy[0],
        xy[1],
        z,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    return snapshot


def test_comparison_allows_different_but_in_window_pick_samples() -> None:
    # Randomized apple XY: the two runtimes draw different samples, both inside the
    # 0.15/0.4 +/- (0.05 + 0.02) window, so the comparison must accept them.
    recording = _with_pick_runtime(_snapshot("isaaclab"), (0.11, 0.42))
    evaluation = _with_pick_runtime(_snapshot("arena"), (0.19, 0.37))

    result = compare_snapshots(recording, evaluation)

    assert "pick_object" in result["runtime_positions_compared"]


def test_comparison_rejects_pick_sample_outside_spawn_window() -> None:
    recording = _with_pick_runtime(_snapshot("isaaclab"), (0.15, 0.4))
    evaluation = _with_pick_runtime(_snapshot("arena"), (0.30, 0.4))  # 0.15 m > 0.07 m window

    with pytest.raises(AssertionError, match=r"pick_object\.runtime_position"):
        compare_snapshots(recording, evaluation)


def test_comparison_rejects_pick_runtime_height_drift() -> None:
    # Z is not randomized; the two runtimes must still settle to the same height.
    recording = _with_pick_runtime(_snapshot("isaaclab"), (0.15, 0.4))
    evaluation = _with_pick_runtime(_snapshot("arena"), (0.15, 0.4), z=0.9)

    with pytest.raises(AssertionError, match=r"pick_object\.runtime_height"):
        compare_snapshots(recording, evaluation)

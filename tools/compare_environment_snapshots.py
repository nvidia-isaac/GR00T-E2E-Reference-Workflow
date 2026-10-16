#!/usr/bin/env python3
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

"""Compare one declared environment and its realizations in two runtimes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_ABS_TOLERANCE = 1.0e-5
RUNTIME_POSITION_TOLERANCE_M = 0.02
ROBOT_RUNTIME_HEIGHT_TOLERANCE_M = 0.05
REQUIRED_RUNTIME_POSE_ROLES = frozenset({"pick_object"})

# Roles whose spawn XY is randomized per episode (see the environment contract).
# Their realized runtime pose is a per-run sample, so it cannot be compared for
# cross-runtime equality on XY; instead each runtime must land within the declared
# spawn window (nominal +/- half-range, plus the physics-settle tolerance). Z is not
# randomized and is still compared exactly across runtimes. Maps role -> (contract
# nominal-position key, contract XY half-range key).
_RANDOMIZED_SPAWN_ROLES = {
    "pick_object": ("pick_spawn_position", "pick_spawn_xy_range_m"),
    "destination": ("destination_spawn_position", "destination_spawn_xy_range_m"),
}


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Snapshot {path} must contain a JSON object")
    return data


def _assert_close(left: Any, right: Any, context: str, tolerance: float = DEFAULT_ABS_TOLERANCE) -> None:
    if isinstance(left, int | float) and isinstance(right, int | float):
        if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance):
            raise AssertionError(f"{context}: {left!r} != {right!r} (absolute tolerance {tolerance})")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise AssertionError(f"{context}: list lengths differ ({len(left)} != {len(right)})")
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
            _assert_close(left_value, right_value, f"{context}[{index}]", tolerance)
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise AssertionError(
                f"{context}: mapping keys differ "
                f"(left-only={sorted(left.keys() - right.keys())}, right-only={sorted(right.keys() - left.keys())})"
            )
        for key in left:
            _assert_close(left[key], right[key], f"{context}.{key}", tolerance)
        return
    if left != right:
        raise AssertionError(f"{context}: {left!r} != {right!r}")


def _assert_matching_presence(left: Any, right: Any, context: str) -> None:
    if (left is None) != (right is None):
        raise AssertionError(f"{context}: field is present in only one snapshot")


def _assert_within_spawn_xy_window(
    position: list[float], nominal_position: list[float], xy_half_range: float, context: str
) -> None:
    """Assert a realized XY lies inside the declared spawn window (nominal +/- range)."""
    allowed = float(xy_half_range) + RUNTIME_POSITION_TOLERANCE_M
    for index in (0, 1):
        delta = abs(float(position[index]) - float(nominal_position[index]))
        if delta > allowed:
            raise AssertionError(
                f"{context}.runtime_position[{index}]: {position[index]!r} is {delta:.4f} m from "
                f"nominal {nominal_position[index]!r} (allowed {allowed} m)"
            )


def compare_snapshots(recording: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    """Assert shared contracts and realized static-scene configuration agree."""

    _assert_close(recording["contract"], evaluation["contract"], "contract")

    recording_realized = recording["realized"]
    evaluation_realized = evaluation["realized"]
    for field in ("physics_dt_s", "decimation", "episode_length_s"):
        _assert_close(recording_realized[field], evaluation_realized[field], f"realized.{field}")

    compared_runtime_roles: list[str] = []
    for role in ("background", "support", "pick_object", "destination"):
        recording_entity = recording_realized["entities"][role]
        evaluation_entity = evaluation_realized["entities"][role]
        for field in ("usd", "scale", "initial_position", "initial_rotation_xyzw"):
            _assert_close(recording_entity[field], evaluation_entity[field], f"realized.entities.{role}.{field}")
        recording_pose = recording_entity.get("runtime_root_pose_wxyz")
        evaluation_pose = evaluation_entity.get("runtime_root_pose_wxyz")
        if role in REQUIRED_RUNTIME_POSE_ROLES:
            _assert_matching_presence(
                recording_pose,
                evaluation_pose,
                f"realized.entities.{role}.runtime_root_pose_wxyz",
            )
        if recording_pose is not None and evaluation_pose is not None:
            if role in _RANDOMIZED_SPAWN_ROLES:
                position_key, range_key = _RANDOMIZED_SPAWN_ROLES[role]
                nominal_position = recording["contract"][position_key]
                xy_half_range = recording["contract"][range_key]
                # XY is a per-episode random draw: assert each runtime lands inside the
                # shared spawn window rather than requiring the two draws to match.
                _assert_within_spawn_xy_window(
                    recording_pose[:3], nominal_position, xy_half_range, f"recording.entities.{role}"
                )
                _assert_within_spawn_xy_window(
                    evaluation_pose[:3], nominal_position, xy_half_range, f"evaluation.entities.{role}"
                )
                # Z is not randomized; both runtimes must still settle to the same height.
                _assert_close(
                    recording_pose[2],
                    evaluation_pose[2],
                    f"realized.entities.{role}.runtime_height",
                    RUNTIME_POSITION_TOLERANCE_M,
                )
            else:
                _assert_close(
                    recording_pose[:3],
                    evaluation_pose[:3],
                    f"realized.entities.{role}.runtime_position",
                    RUNTIME_POSITION_TOLERANCE_M,
                )
            compared_runtime_roles.append(role)

    recording_robot = recording_realized["entities"]["robot"]
    evaluation_robot = evaluation_realized["entities"]["robot"]
    for field in ("usd", "scale", "initial_position", "initial_rotation_xyzw"):
        _assert_close(recording_robot[field], evaluation_robot[field], f"realized.entities.robot.{field}")
    recording_robot_pose = recording_robot.get("runtime_root_pose_wxyz")
    evaluation_robot_pose = evaluation_robot.get("runtime_root_pose_wxyz")
    _assert_matching_presence(
        recording_robot_pose,
        evaluation_robot_pose,
        "realized.entities.robot.runtime_root_pose_wxyz",
    )
    if recording_robot_pose is not None and evaluation_robot_pose is not None:
        _assert_close(
            recording_robot_pose[2],
            evaluation_robot_pose[2],
            "realized.entities.robot.runtime_height",
            ROBOT_RUNTIME_HEIGHT_TOLERANCE_M,
        )
        compared_runtime_roles.append("robot")

    return {
        "status": "ok",
        "environment_id": recording["contract"]["environment_id"],
        "environment_contract_digest": recording["contract"]["environment_contract_digest"],
        "recording_runtime": recording["mode"]["runtime"],
        "evaluation_runtime": evaluation["mode"]["runtime"],
        "runtime_positions_compared": compared_runtime_roles,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_snapshot", type=Path)
    parser.add_argument("evaluation_snapshot", type=Path)
    args = parser.parse_args()
    result = compare_snapshots(_load(args.recording_snapshot), _load(args.evaluation_snapshot))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

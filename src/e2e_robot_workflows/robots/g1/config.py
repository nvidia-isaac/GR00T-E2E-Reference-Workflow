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

"""G1 robot configuration, loaded from ``configs/robots/g1.yaml``.

Physical and kinematic defaults are separate from robot-independent
environment definitions. Override the path with ``E2E_G1_ROBOT_CONFIG``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from e2e_robot_workflows.common import load_yaml_mapping

_DEFAULT_ROBOT_CONFIG = Path(__file__).resolve().parents[4] / "configs" / "robots" / "g1.yaml"


@dataclass(frozen=True)
class G1RobotConfig:
    """Parsed G1 physical defaults consumed by the runtime adapters."""

    usd: str | None
    recording_wbc_mode: str
    recording_enable_waist: bool
    xr_anchor_position: tuple[float, float, float]
    xr_anchor_rotation_xyzw: tuple[float, float, float, float]
    initial_position_xy: tuple[float, float]
    initial_rotation_xyzw: tuple[float, float, float, float]
    task_link_names: tuple[str, str]
    pelvis_height_m: float
    standing_leg_joint_pos: dict[str, float]
    base_leg_joint_patterns: tuple[str, ...]
    open_arm_joint_pos: dict[str, float]
    finger_friction: dict[str, Any]
    head_camera: dict[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.task_link_names, tuple)
            or len(self.task_link_names) != 2
            or not all(isinstance(name, str) and name for name in self.task_link_names)
            or len(set(self.task_link_names)) != 2
        ):
            raise ValueError("task_link_names must contain two distinct non-empty robot links")


def load_g1_robot_config(path: str | None = None) -> G1RobotConfig:
    """Load G1 defaults from an explicit path, environment override, or repository config."""

    data = load_yaml_mapping(path or os.environ.get("E2E_G1_ROBOT_CONFIG") or str(_DEFAULT_ROBOT_CONFIG))
    recording = data["recording"]
    pose = data["initial_pose"]
    standing = data["standing"]
    task_link_names = data["task_link_names"]
    if not isinstance(task_link_names, list | tuple) or any(
        not isinstance(name, str) or not name for name in task_link_names
    ):
        raise ValueError("task_link_names must contain two distinct non-empty robot links")
    return G1RobotConfig(
        usd=data.get("usd"),
        recording_wbc_mode=str(recording["wbc_mode"]),
        recording_enable_waist=bool(recording["enable_waist"]),
        xr_anchor_position=tuple(float(value) for value in recording["xr_anchor_position"]),
        xr_anchor_rotation_xyzw=tuple(float(value) for value in recording["xr_anchor_rotation_xyzw"]),
        initial_position_xy=tuple(float(value) for value in pose["position_xy"]),
        initial_rotation_xyzw=tuple(float(value) for value in pose["rotation_xyzw"]),
        task_link_names=tuple(task_link_names),
        pelvis_height_m=float(standing["pelvis_height_m"]),
        standing_leg_joint_pos={str(name): float(position) for name, position in standing["leg_joint_pos"].items()},
        base_leg_joint_patterns=tuple(str(pattern) for pattern in standing["base_leg_joint_patterns"]),
        open_arm_joint_pos={str(name): float(position) for name, position in data["open_arm_joint_pos"].items()},
        finger_friction=dict(data["finger_friction"]),
        head_camera=dict(data.get("head_camera", {})),
    )

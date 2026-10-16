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

"""Arena-only H2 Plus AGILE configuration."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from e2e_robot_workflows.common import load_yaml_mapping
from e2e_robot_workflows.data.artifact_paths import file_sha256

_DEFAULT_ROBOT_CONFIG = Path(__file__).resolve().parents[4] / "configs" / "robots" / "h2.yaml"

# Widths the ONNX graph fixes, not values the YAML gets to choose: the export's
# ports are shaped (1, 31) and (1, 14) and its command is 3 wide. They are stated
# here rather than counted from the file on purpose -- a check derived from the
# thing it checks passes for any file, including one that silently lost a joint.
# The graph itself is the authority. GPU qualification holds these values against
# the export; the constants let a bad YAML fail on a CPU host where the export is
# not installed and cannot be consulted.
EXPECTED_JOINT_COUNT = 31
DRIVEN_JOINT_COUNT = 14
EXPECTED_COMMAND_DIM = 3


@dataclass(frozen=True)
class H2AgileDrivenJoint:
    """One joint the AGILE export drives, at its position in the export's output."""

    joint: str
    kp: float
    kd: float
    reference_pos: float


@dataclass(frozen=True)
class H2RobotConfig:
    """Parsed H2 Plus defaults consumed by the runtime adapters."""

    policy_env_var: str
    policy_repository: str
    policy_revision: str
    policy_filename: str
    policy_sha256: str
    initial_position_xy: tuple[float, float]
    initial_rotation_xyzw: tuple[float, float, float, float]
    initial_joint_pos: dict[str, float]
    pelvis_height_m: float
    task_link_names: tuple[str, str]
    control_hz: int
    render_hz: int
    camera_hz: int
    command_dim: int
    navigate_command_limits: tuple[tuple[float, float], ...]
    history_length: int
    expected_joint_order: tuple[str, ...]
    driven_joints: tuple[H2AgileDrivenJoint, ...]

    def __post_init__(self) -> None:
        order = self.expected_joint_order
        if len(order) != EXPECTED_JOINT_COUNT or len(set(order)) != len(order):
            raise ValueError(
                f"expected_joint_order must list {EXPECTED_JOINT_COUNT} distinct joints, "
                f"got {len(order)} ({len(set(order))} distinct)"
            )
        if len(self.driven_joints) != DRIVEN_JOINT_COUNT:
            raise ValueError(f"the export drives {DRIVEN_JOINT_COUNT} joints, got {len(self.driven_joints)}")
        unknown = [entry.joint for entry in self.driven_joints if entry.joint not in set(order)]
        if unknown:
            raise ValueError(f"driven joints absent from expected_joint_order: {unknown}")
        if len(set(self.driven_joint_names)) != len(self.driven_joint_names):
            raise ValueError("driven_joints must not repeat a joint")
        if self.command_dim != EXPECTED_COMMAND_DIM:
            raise ValueError(
                f"the H2 AGILE export takes a {EXPECTED_COMMAND_DIM}-wide velocity command, got {self.command_dim}"
            )
        if len(self.navigate_command_limits) != self.command_dim or any(
            len(limits) != 2 or not all(math.isfinite(value) for value in limits) or limits[0] > 0.0 or limits[1] < 0.0
            for limits in self.navigate_command_limits
        ):
            raise ValueError("navigate_command_limits must contain one finite range around zero per command")
        if self.control_hz <= 0 or self.render_hz <= 0 or self.camera_hz <= 0 or self.history_length <= 0:
            raise ValueError("control_hz, render_hz, camera_hz and history_length must be positive")
        if self.control_hz % self.render_hz or self.render_hz % self.camera_hz:
            raise ValueError("render_hz must divide control_hz and camera_hz must divide render_hz")
        if self.pelvis_height_m <= 0:
            raise ValueError("standing pelvis height must be positive")
        if (
            not isinstance(self.task_link_names, tuple)
            or len(self.task_link_names) != 2
            or not all(isinstance(name, str) and name for name in self.task_link_names)
            or len(set(self.task_link_names)) != 2
        ):
            raise ValueError("task_link_names must contain two distinct non-empty robot links")
        if not self.policy_repository or not self.policy_filename:
            raise ValueError("H2 AGILE repository and filename must be pinned")
        if len(self.policy_revision) != 40 or any(char not in "0123456789abcdef" for char in self.policy_revision):
            raise ValueError("policy_revision must be a lowercase 40-character commit")
        if len(self.policy_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.policy_sha256):
            raise ValueError("policy_sha256 must be a lowercase hex sha256 digest")
        if len(self.initial_position_xy) != 2:
            raise ValueError("H2 initial XY position must contain two values")

    @property
    def driven_joint_names(self) -> tuple[str, ...]:
        """The driven joints in the export's output order."""

        return tuple(entry.joint for entry in self.driven_joints)

    @property
    def driven_joint_stiffness(self) -> dict[str, float]:
        """Per-joint stiffness the export emits, keyed by joint name."""

        return {entry.joint: entry.kp for entry in self.driven_joints}

    @property
    def driven_joint_damping(self) -> dict[str, float]:
        """Per-joint damping the export emits, keyed by joint name."""

        return {entry.joint: entry.kd for entry in self.driven_joints}

    @property
    def reference_joint_pos(self) -> dict[str, float]:
        """The crouch the export has already added to its own output."""

        return {entry.joint: entry.reference_pos for entry in self.driven_joints}

    @property
    def initial_base_position(self) -> tuple[float, float, float]:
        """Spawn position: XY from the config, Z from the standing pelvis height."""

        return (self.initial_position_xy[0], self.initial_position_xy[1], self.pelvis_height_m)

    def resolve_policy(self) -> str:
        """Return the AGILE export path, verified by content digest."""

        path = _require_existing_path(
            self.policy_env_var,
            "the H2 AGILE ONNX export",
            f"{self.policy_repository}@{self.policy_revision}/{self.policy_filename}",
        )
        digest = file_sha256(path)
        if digest != self.policy_sha256:
            raise ValueError(
                f"{self.policy_env_var} points at an unexpected checkpoint.\n"
                f"  path:     {path}\n"
                f"  expected: sha256:{self.policy_sha256}\n"
                f"  found:    sha256:{digest}\n"
                f"Expected {self.policy_repository}@{self.policy_revision}/{self.policy_filename}. "
                "The joint mapping, gains and reference pose in configs/robots/h2.yaml were recovered "
                "from the expected export by probing its graph. Applying them to a different checkpoint "
                "would drive the wrong joints without raising."
            )
        return path


def _require_existing_path(env_var: str, what: str, hint: str) -> str:
    """Resolve a required path from the environment, failing with what to set."""

    value = os.environ.get(env_var)
    if not value:
        raise RuntimeError(f"Set {env_var} to {what} ({hint}). It is not redistributable from this repository.")
    if not Path(value).exists():
        raise FileNotFoundError(f"{env_var} points at a missing path for {what}: {value}")
    return value


def load_h2_robot_config(path: str | None = None) -> H2RobotConfig:
    """Load H2 Plus defaults from an explicit path or the repository config."""

    data = load_yaml_mapping(path or str(_DEFAULT_ROBOT_CONFIG))
    pose = data["initial_pose"]
    agile = data["agile"]
    task_link_names = data["task_link_names"]
    if not isinstance(task_link_names, list | tuple) or any(
        not isinstance(name, str) or not name for name in task_link_names
    ):
        raise ValueError("task_link_names must contain two distinct non-empty robot links")
    return H2RobotConfig(
        policy_env_var=str(data["policy_env_var"]),
        policy_repository=str(data["policy_repository"]),
        policy_revision=str(data["policy_revision"]).lower(),
        policy_filename=str(data["policy_filename"]),
        policy_sha256=str(data["policy_sha256"]).lower(),
        initial_position_xy=tuple(float(value) for value in pose["position_xy"]),
        initial_rotation_xyzw=tuple(float(value) for value in pose["rotation_xyzw"]),
        initial_joint_pos={str(name): float(value) for name, value in pose["joint_pos"].items()},
        pelvis_height_m=float(data["standing"]["pelvis_height_m"]),
        task_link_names=tuple(task_link_names),
        control_hz=int(agile["control_hz"]),
        render_hz=int(agile["render_hz"]),
        camera_hz=int(agile["camera_hz"]),
        command_dim=int(agile["command_dim"]),
        navigate_command_limits=tuple(
            tuple(float(value) for value in limits) for limits in agile["navigate_command_limits"]
        ),
        history_length=int(agile["history_length"]),
        expected_joint_order=tuple(str(name) for name in agile["expected_joint_order"]),
        driven_joints=tuple(
            H2AgileDrivenJoint(
                joint=str(entry["joint"]),
                kp=float(entry["kp"]),
                kd=float(entry["kd"]),
                reference_pos=float(entry["reference_pos"]),
            )
            for entry in agile["driven_joints"]
        ),
    )

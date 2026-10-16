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

"""Robot-specific choices consumed by the shared Arena environment builder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from e2e_robot_workflows.robots.g1.config import load_g1_robot_config
from e2e_robot_workflows.robots.h2 import load_h2_robot_config
from e2e_robot_workflows.tasks.static_apple.config import H2_STATIC_APPLE_VARIANT

G1_ARENA_EMBODIMENT = "g1_wbc_agile_joint"
G1_ARENA_TELEOP_EMBODIMENT = "g1_wbc_agile_pink"
H2_ARENA_EMBODIMENT = "h2_agile_joint"
H2_ARENA_TELEOP_EMBODIMENT = "h2_agile_pink"


@dataclass(frozen=True)
class ArenaRobotBinding:
    """Resolved spawn and contact settings for one robot family."""

    initial_position_xy: tuple[float, float]
    initial_rotation_xyzw: tuple[float, float, float, float]
    pelvis_height_m: float
    control_hz: int
    render_hz: int
    joint_initial_pos: dict[str, float]
    finger_friction: dict[str, Any]
    task_link_names: tuple[str, str]
    fixed_xr_anchor_rotation: bool = False
    usd_override: str | None = None


@dataclass(frozen=True)
class ArenaEmbodimentCapability:
    """The supported controller roles for one Arena robot family."""

    embodiments: tuple[str, ...]
    teleop_embodiment: str
    retargeted_devices: tuple[str, ...]
    load_binding: Callable[[], ArenaRobotBinding]
    environment_variant: str | None = None

    def validate(self, *, embodiment: str, teleop_device: str | None, mimic: bool) -> None:
        if embodiment not in self.embodiments:
            raise ValueError(f"unsupported Arena embodiment {embodiment!r}")
        if teleop_device in self.retargeted_devices and embodiment != self.teleop_embodiment:
            raise ValueError(
                f"Teleoperation device {teleop_device!r} has no retargeter for {embodiment!r}; "
                f"use {self.teleop_embodiment!r}."
            )
        if mimic and embodiment != self.teleop_embodiment:
            raise ValueError(f"Mimic generation requires {self.teleop_embodiment!r}")


def _load_g1_binding() -> ArenaRobotBinding:
    robot = load_g1_robot_config()
    return ArenaRobotBinding(
        initial_position_xy=robot.initial_position_xy,
        initial_rotation_xyzw=robot.initial_rotation_xyzw,
        pelvis_height_m=robot.pelvis_height_m,
        control_hz=50,
        render_hz=50,
        joint_initial_pos=robot.open_arm_joint_pos,
        finger_friction=robot.finger_friction,
        task_link_names=robot.task_link_names,
        fixed_xr_anchor_rotation=True,
        usd_override=robot.usd,
    )


def _load_h2_binding() -> ArenaRobotBinding:
    robot = load_h2_robot_config()
    return ArenaRobotBinding(
        initial_position_xy=robot.initial_position_xy,
        initial_rotation_xyzw=robot.initial_rotation_xyzw,
        pelvis_height_m=robot.pelvis_height_m,
        control_hz=robot.control_hz,
        render_hz=robot.render_hz,
        joint_initial_pos={**robot.reference_joint_pos, **robot.initial_joint_pos},
        finger_friction={
            "material_path": "/World/Materials/h2_static_object_transfer_high_friction_fingers",
            "static_friction": 1.0,
            "dynamic_friction": 0.8,
            "prim_name_markers": ("hand", "thumb", "index", "middle", "ring", "pinky"),
        },
        task_link_names=robot.task_link_names,
    )


_CAPABILITIES = (
    ArenaEmbodimentCapability(
        embodiments=(G1_ARENA_EMBODIMENT, G1_ARENA_TELEOP_EMBODIMENT),
        teleop_embodiment=G1_ARENA_TELEOP_EMBODIMENT,
        retargeted_devices=("openxr",),
        load_binding=_load_g1_binding,
    ),
    ArenaEmbodimentCapability(
        embodiments=(H2_ARENA_EMBODIMENT, H2_ARENA_TELEOP_EMBODIMENT),
        teleop_embodiment=H2_ARENA_TELEOP_EMBODIMENT,
        retargeted_devices=("openxr",),
        load_binding=_load_h2_binding,
        environment_variant=H2_STATIC_APPLE_VARIANT,
    ),
)


def capability_for_embodiment(name: str) -> ArenaEmbodimentCapability:
    for capability in _CAPABILITIES:
        if name in capability.embodiments:
            return capability
    supported = tuple(name for capability in _CAPABILITIES for name in capability.embodiments)
    raise ValueError(f"unsupported Arena embodiment {name!r}; expected one of {supported!r}")


def resolve_locomotion_embodiment(
    *,
    robot: str,
    arena_requested: bool,
    requested_embodiment: str | None,
) -> str | None:
    """Resolve the locomotion CLI route without importing either simulator runtime."""

    if robot == "h2":
        embodiment = requested_embodiment or H2_ARENA_EMBODIMENT
        if embodiment not in (H2_ARENA_EMBODIMENT, H2_ARENA_TELEOP_EMBODIMENT):
            raise ValueError("--robot h2 requires an H2 Arena embodiment")
        return embodiment
    if robot != "g1":
        raise ValueError(f"unsupported locomotion robot {robot!r}")
    if requested_embodiment is not None and not arena_requested:
        raise ValueError("--arena_embodiment requires --arena")
    if not arena_requested:
        return None
    embodiment = requested_embodiment or G1_ARENA_EMBODIMENT
    if embodiment not in (G1_ARENA_EMBODIMENT, G1_ARENA_TELEOP_EMBODIMENT):
        raise ValueError("--robot g1 requires a G1 Arena embodiment")
    return embodiment


__all__ = [
    "G1_ARENA_EMBODIMENT",
    "G1_ARENA_TELEOP_EMBODIMENT",
    "H2_ARENA_EMBODIMENT",
    "H2_ARENA_TELEOP_EMBODIMENT",
    "ArenaEmbodimentCapability",
    "ArenaRobotBinding",
    "capability_for_embodiment",
    "resolve_locomotion_embodiment",
]

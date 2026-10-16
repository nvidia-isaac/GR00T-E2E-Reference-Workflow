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

"""Generic controller profile contracts.

These specs describe how upper-body controllers, lower-body stabilizers, and
whole-body composers fit together. They intentionally do not import Isaac Lab,
NVIDIA Isaac GR00T, NVIDIA Isaac AutoData, or any concrete controller runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ControllerAdapterSpec:
    """A named controller or stabilizer role inside a workflow profile."""

    name: str
    role: str
    kind: str
    command_space: str
    output_space: str
    status: str = "planned"
    runtime_dependency: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ControllerAdapterSpec:
        """Build an adapter spec from a YAML/JSON mapping."""

        return cls(
            name=str(data["name"]),
            role=str(data["role"]),
            kind=str(data["kind"]),
            command_space=str(data["command_space"]),
            output_space=str(data["output_space"]),
            status=str(data.get("status", "planned")),
            runtime_dependency=data.get("runtime_dependency"),
            parameters=dict(data.get("parameters", {}) or {}),
        )


@dataclass(frozen=True)
class ControllerComposerSpec:
    """How adapter outputs are combined into the robot action contract."""

    name: str
    output_action_space: str
    waist_owner: str = "lower_body"
    conflict_resolution: str = "lower_body_wins"
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ControllerComposerSpec:
        """Build a composer spec from a YAML/JSON mapping."""

        return cls(
            name=str(data["name"]),
            output_action_space=str(data["output_action_space"]),
            waist_owner=str(data.get("waist_owner", "lower_body")),
            conflict_resolution=str(data.get("conflict_resolution", "lower_body_wins")),
            parameters=dict(data.get("parameters", {}) or {}),
        )


@dataclass(frozen=True)
class ControllerProfileSpec:
    """A concrete upper/lower/composer controller combination."""

    name: str
    robot: str
    status: str
    upper_body: ControllerAdapterSpec
    lower_body: ControllerAdapterSpec
    composer: ControllerComposerSpec
    description: str = ""
    tags: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ControllerProfileSpec:
        """Build a profile spec from a YAML/JSON mapping."""

        return cls(
            name=str(data["name"]),
            robot=str(data["robot"]),
            status=str(data.get("status", "planned")),
            description=str(data.get("description", "")),
            upper_body=ControllerAdapterSpec.from_mapping(data["upper_body"]),
            lower_body=ControllerAdapterSpec.from_mapping(data["lower_body"]),
            composer=ControllerComposerSpec.from_mapping(data["composer"]),
            tags=tuple(data.get("tags", ()) or ()),
        )


@dataclass(frozen=True)
class ControllerProfilesConfig:
    """A set of selectable controller profiles for a task or robot."""

    default_record_profile: str
    default_eval_profile: str
    profiles: tuple[ControllerProfileSpec, ...]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ControllerProfilesConfig:
        """Build a controller profiles config from a YAML/JSON mapping."""

        return cls(
            default_record_profile=str(data["default_record_profile"]),
            default_eval_profile=str(data.get("default_eval_profile", data["default_record_profile"])),
            profiles=tuple(ControllerProfileSpec.from_mapping(profile) for profile in data.get("profiles", ())),
        )

    def get_profile(self, name: str) -> ControllerProfileSpec:
        """Return a profile by name."""

        for profile in self.profiles:
            if profile.name == name:
                return profile
        raise KeyError(f"Unknown controller profile: {name}")

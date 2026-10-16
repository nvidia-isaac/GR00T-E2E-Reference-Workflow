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

"""Controller profile loading and runtime selection helpers."""

from __future__ import annotations

from pathlib import Path

from e2e_robot_workflows.common import load_yaml_mapping
from e2e_robot_workflows.controllers.controller_specs import ControllerProfilesConfig, ControllerProfileSpec

# Resolve relative to the repo root (CWD-independent).
DEFAULT_CONTROLLER_PROFILES_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "robots" / "g1_controller_profiles.yaml"
)
ACTIVE_PROFILE_STATUS = "active_today"


def load_controller_profiles(path: Path | None = None) -> ControllerProfilesConfig:
    """Load controller profiles from a YAML config (default: the G1 profiles)."""

    return ControllerProfilesConfig.from_mapping(load_yaml_mapping(path or DEFAULT_CONTROLLER_PROFILES_PATH))


def select_controller_profile(
    config: ControllerProfilesConfig,
    profile_name: str | None = None,
) -> ControllerProfileSpec:
    """Select a named profile or the default recording profile."""

    return config.get_profile(profile_name or config.default_record_profile)


def require_recording_ready_profile(profile: ControllerProfileSpec) -> None:
    """Fail if a profile is only a design target and not wired for recording yet."""

    if profile.status != ACTIVE_PROFILE_STATUS:
        raise ValueError(
            f"Controller profile '{profile.name}' has status '{profile.status}', not '{ACTIVE_PROFILE_STATUS}'. "
            "It documents a future controller stack but is not wired into the recorder action path yet."
        )

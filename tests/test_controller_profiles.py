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

from __future__ import annotations

from pathlib import Path

import yaml

from e2e_robot_workflows.controllers import (
    ACTIVE_PROFILE_STATUS,
    ControllerProfilesConfig,
    load_controller_profiles,
    require_recording_ready_profile,
    select_controller_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PROFILES_PATH = REPO_ROOT / "configs/robots/g1_controller_profiles.yaml"
G1_ROBOT_CONFIG_PATH = REPO_ROOT / "configs/robots/g1.yaml"
STATIC_APPLE_TASK_CONFIG_PATH = REPO_ROOT / "configs/tasks/static_apple.yaml"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    assert isinstance(data, dict)
    return data


def load_profiles() -> ControllerProfilesConfig:
    return ControllerProfilesConfig.from_mapping(load_yaml(CONTROLLER_PROFILES_PATH))


def test_robot_config_references_existing_controller_profiles() -> None:
    robot_config = load_yaml(G1_ROBOT_CONFIG_PATH)
    profiles = load_profiles()
    profile_names = {profile.name for profile in profiles.profiles}

    assert robot_config["controller_profiles_path"] == "configs/robots/g1_controller_profiles.yaml"
    assert robot_config["default_record_profile"] in profile_names
    assert robot_config["default_eval_profile"] in profile_names


def test_controller_profile_names_are_unique() -> None:
    profiles = load_profiles()
    profile_names = [profile.name for profile in profiles.profiles]

    assert len(profile_names) == len(set(profile_names))


def test_active_profile_matches_current_isaaclab_runtime_path() -> None:
    profiles = load_profiles()
    profile = profiles.get_profile(profiles.default_record_profile)

    assert profile.name == "g1_isaaclab_pink_ik_static"
    assert profile.status == "active_today"
    assert profile.upper_body.name == "isaaclab_pink_ik"
    assert profile.upper_body.runtime_dependency == "isaaclab"
    assert profile.lower_body.name == "fixed_standing_posture"
    assert profile.lower_body.parameters["navigation_command"] == [0.0, 0.0, 0.0]
    assert profile.composer.output_action_space == "pink_ik_task_space_action"


def test_default_task_wbc_mode_matches_recording_profile() -> None:
    task_config = load_yaml(STATIC_APPLE_TASK_CONFIG_PATH)
    profiles = load_profiles()
    profile = profiles.get_profile(profiles.default_record_profile)

    assert task_config["recording_adapter"] == "g1"
    assert load_yaml(G1_ROBOT_CONFIG_PATH)["recording"]["wbc_mode"] == "fixed"
    assert profile.lower_body.name == "fixed_standing_posture"


def test_planned_agile_profile_keeps_lower_body_stabilization_separate() -> None:
    profile = load_profiles().get_profile("g1_pink_ik_agile_stabilized")

    assert profile.status == "planned"
    assert profile.upper_body.name == "isaaclab_pink_ik"
    assert profile.lower_body.name == "g1_agile_stabilizer"
    assert profile.lower_body.runtime_dependency == "isaaclab"
    assert profile.lower_body.parameters["navigation_command"] == [0.0, 0.0, 0.0]
    assert profile.lower_body.parameters["mode_setting"] == "configs/robots/g1.yaml:recording.wbc_mode"
    assert profile.composer.output_action_space == "g1_43d_joint_targets"
    assert profile.composer.parameters["canonical_dataset_action"] == "joint_pos_target"


def test_profile_loader_selects_default_record_profile() -> None:
    profiles = load_controller_profiles(CONTROLLER_PROFILES_PATH)
    profile = select_controller_profile(profiles)

    assert profile.name == profiles.default_record_profile
    assert profile.status == ACTIVE_PROFILE_STATUS


def test_active_profile_is_recording_ready() -> None:
    profile = load_controller_profiles(CONTROLLER_PROFILES_PATH).get_profile("g1_isaaclab_pink_ik_static")

    require_recording_ready_profile(profile)


def test_planned_profile_is_not_recording_ready() -> None:
    profile = load_controller_profiles(CONTROLLER_PROFILES_PATH).get_profile("g1_pink_ik_agile_stabilized")

    try:
        require_recording_ready_profile(profile)
    except ValueError as exc:
        assert "not wired into the recorder action path yet" in str(exc)
    else:
        raise AssertionError("planned controller profile should not be recording-ready")

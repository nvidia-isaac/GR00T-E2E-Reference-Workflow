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

"""Compatibility constants for the G1 adapter of the static apple environment."""

from __future__ import annotations

from e2e_robot_workflows.environments import STATIC_APPLE_ENVIRONMENT
from e2e_robot_workflows.robots.g1.config import load_g1_robot_config

ENVIRONMENT = STATIC_APPLE_ENVIRONMENT
_G1 = load_g1_robot_config()

# G1-specific Isaac Lab registration and recording defaults.
TASK_NAME = "g1_static_apple_transfer"
GYM_ID = "Isaac-E2E-G1-StaticApple-PickPlace-Abs-v0"
AUTODATA_GYM_ID = "Isaac-E2E-G1-StaticApple-PickPlace-AutoData-v0"
AUTODATA_AGILE_GYM_ID = "Isaac-E2E-G1-StaticApple-PickPlace-AutoData-Agile-v0"
DEFAULT_WBC_MODE = _G1.recording_wbc_mode
ENABLE_WAIST = _G1.recording_enable_waist
XR_ANCHOR_POS = _G1.xr_anchor_position
XR_ANCHOR_ROT_XYZW = _G1.xr_anchor_rotation_xyzw

# Shared task semantics and timing.
LANGUAGE_INSTRUCTION = ENVIRONMENT.language_instruction
TASK_DESCRIPTION = ENVIRONMENT.description
EPISODE_LENGTH_S = ENVIRONMENT.episode_length_s
PHYSICS_DT_HZ = ENVIRONMENT.physics_hz
CONTROL_HZ = ENVIRONMENT.control_hz

# Shared scene values exposed under the names used by the existing Isaac Lab config.
PICK_OBJECT_NAME = ENVIRONMENT.asset("pick_object").name
DESTINATION_NAME = ENVIRONMENT.asset("destination").name
BACKGROUND_NAME = ENVIRONMENT.asset("background").name
PACKING_TABLE_NATIVE_TOP_Z = ENVIRONMENT.support_native_top_z
PACKING_TABLE_SCALE = ENVIRONMENT.asset("support").scale[2]
SHELF_SURFACE_Z = ENVIRONMENT.support_surface_z
SHELF_AIRGAP = ENVIRONMENT.support_airgap_m
PICK_SPAWN_XY = ENVIRONMENT.pick_spawn_xy
DESTINATION_SPAWN_XY = ENVIRONMENT.destination_spawn_xy
PICK_SPAWN_XY_RANGE_M = ENVIRONMENT.pick_spawn_xy_range_m
PICK_SPAWN_YAW_RANGE_RAD = ENVIRONMENT.pick_spawn_yaw_range_rad
DESTINATION_SPAWN_XY_RANGE_M = ENVIRONMENT.destination_spawn_xy_range_m
BACKGROUND_POS = ENVIRONMENT.background_position
BACKGROUND_PRIMS_TO_DEACTIVATE = ENVIRONMENT.deactivate_background_prims
PACKING_TABLE_PRIMS_TO_DEACTIVATE = ENVIRONMENT.deactivate_support_prims
PICK_OBJECT_MASS_KG = ENVIRONMENT.asset("pick_object").mass_kg
if PICK_OBJECT_MASS_KG is None:
    raise ValueError("The static apple environment must declare a pick-object mass")
DESTINATION_MASS_KG = ENVIRONMENT.asset("destination").mass_kg
if DESTINATION_MASS_KG is None:
    raise ValueError("The static apple environment must declare a plate (destination) mass")
TUNED_ASSET_SCALES = {
    asset.name: asset.scale for asset in (ENVIRONMENT.asset("pick_object"), ENVIRONMENT.asset("destination"))
}


def shelf_spawn_z(asset_name: str) -> float:
    """Return the initial Z for a task asset."""

    if asset_name == PICK_OBJECT_NAME:
        return ENVIRONMENT.spawn_position("pick_object")[2]
    if asset_name == DESTINATION_NAME:
        return ENVIRONMENT.spawn_position("destination")[2]
    return SHELF_SURFACE_Z + SHELF_AIRGAP

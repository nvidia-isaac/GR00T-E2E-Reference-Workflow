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

"""Controller composition contracts for robot workflow tasks."""

from e2e_robot_workflows.controllers.controller_specs import (
    ControllerAdapterSpec,
    ControllerComposerSpec,
    ControllerProfilesConfig,
    ControllerProfileSpec,
)
from e2e_robot_workflows.controllers.profiles import (
    ACTIVE_PROFILE_STATUS,
    DEFAULT_CONTROLLER_PROFILES_PATH,
    load_controller_profiles,
    require_recording_ready_profile,
    select_controller_profile,
)

__all__ = [
    "ACTIVE_PROFILE_STATUS",
    "ControllerAdapterSpec",
    "ControllerComposerSpec",
    "ControllerProfileSpec",
    "ControllerProfilesConfig",
    "DEFAULT_CONTROLLER_PROFILES_PATH",
    "load_controller_profiles",
    "require_recording_ready_profile",
    "select_controller_profile",
]

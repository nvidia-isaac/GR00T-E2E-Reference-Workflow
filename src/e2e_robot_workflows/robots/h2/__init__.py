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

"""Arena-only Unitree H2 Plus contracts."""

from e2e_robot_workflows.robots.h2.config import (
    DRIVEN_JOINT_COUNT,
    EXPECTED_JOINT_COUNT,
    H2AgileDrivenJoint,
    H2RobotConfig,
    load_h2_robot_config,
)
from e2e_robot_workflows.robots.h2.pink_contract import (
    H2_ARENA_ACTION_DIM,
    H2_LIVE_BODY_COUNT,
    H2_LIVE_JOINT_COUNT,
    H2_NAVIGATE_COMMAND_DIM,
    H2_PINK_ACTION_DIM,
    H2PinkContract,
    load_h2_pink_contract,
    materialize_published_usd,
)

__all__ = [
    "DRIVEN_JOINT_COUNT",
    "EXPECTED_JOINT_COUNT",
    "H2AgileDrivenJoint",
    "H2_ARENA_ACTION_DIM",
    "H2_LIVE_BODY_COUNT",
    "H2_LIVE_JOINT_COUNT",
    "H2_NAVIGATE_COMMAND_DIM",
    "H2RobotConfig",
    "H2PinkContract",
    "H2_PINK_ACTION_DIM",
    "load_h2_robot_config",
    "load_h2_pink_contract",
    "materialize_published_usd",
]

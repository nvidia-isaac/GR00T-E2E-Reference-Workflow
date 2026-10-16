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

"""Static apple task instance and optional Isaac Lab registration."""

from e2e_robot_workflows.tasks.static_apple.spec import (
    AUTODATA_AGILE_GYM_ID,
    AUTODATA_GYM_ID,
    GYM_ID,
    TASK_NAME,
)

try:
    import gymnasium as gym
except ImportError:
    gym = None

if gym is not None:
    gym.register(
        id=GYM_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": "e2e_robot_workflows.tasks.static_apple.env_cfg:G1StaticAppleEnvCfg",
            "completion_policy_entry_point": "e2e_robot_workflows.tasks.static_apple.completion:build_policy",
        },
        disable_env_checker=True,
    )
    # Same task with a success condition and a randomized apple pose, which is what Isaac
    # AutoData needs to score and retarget generated attempts. Built through our own factory,
    # which restores the recorder terms AutoData's env configuration drops.
    gym.register(
        id=AUTODATA_GYM_ID,
        entry_point="e2e_robot_workflows.tasks.static_apple.autodata_env:make_autodata_env",
        kwargs={
            "env_cfg_entry_point": (
                "e2e_robot_workflows.tasks.static_apple.autodata_env_cfg:G1StaticAppleAutoDataEnvCfg"
            ),
        },
        disable_env_checker=True,
    )
    gym.register(
        id=AUTODATA_AGILE_GYM_ID,
        entry_point="e2e_robot_workflows.tasks.static_apple.autodata_env:make_autodata_env",
        kwargs={
            "env_cfg_entry_point": (
                "e2e_robot_workflows.tasks.static_apple.autodata_env_cfg:G1StaticAppleAutoDataAgileEnvCfg"
            ),
        },
        disable_env_checker=True,
    )

__all__ = ["AUTODATA_AGILE_GYM_ID", "AUTODATA_GYM_ID", "GYM_ID", "TASK_NAME"]

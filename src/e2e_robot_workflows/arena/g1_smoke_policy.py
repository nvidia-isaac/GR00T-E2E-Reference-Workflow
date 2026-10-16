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

"""Small G1 action source for validating the external Arena environment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from e2e_robot_workflows.arena.external_environment import G1_ARENA_EMBODIMENT
from e2e_robot_workflows.arena.g1_actions import DEFAULT_BASE_HEIGHT_M, joint_hold_action
from e2e_robot_workflows.environments import STATIC_APPLE_ENVIRONMENT, resolve_role_to_scene_entity
from e2e_robot_workflows.environments.runtime_snapshot import (
    ENVIRONMENT_SNAPSHOT_SETTLE_STEPS,
    collect_environment_snapshot,
    write_environment_snapshot,
)


@dataclass
class G1StandingSineSmokePolicyCfg(PolicyCfg):
    """Configure the finite-action architecture check."""

    base_height_m: float = DEFAULT_BASE_HEIGHT_M
    sine_amplitude_rad: float = 0.05
    sine_period_steps: int = 100
    environment_snapshot: str | None = None
    snapshot_after_steps: int = ENVIRONMENT_SNAPSHOT_SETTLE_STEPS

    def __post_init__(self) -> None:
        if self.sine_period_steps <= 0:
            raise ValueError("sine_period_steps must be positive")


@register_policy
class G1StandingSineSmokePolicy(PolicyBase[G1StandingSineSmokePolicyCfg]):
    """Hold G1 standing while moving one shoulder through a small sine wave."""

    name = "e2e_g1_standing_sine"

    def __init__(self, config: G1StandingSineSmokePolicyCfg):
        super().__init__(config)
        self._step = 0
        self._snapshot_written = False

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._step = 0

    def get_action(self, env, observation) -> torch.Tensor:
        """Return one safe, deterministic G1 joint-control action."""

        del observation
        robot = env.unwrapped.scene["robot"]
        # Holding the spawned pose keeps the arms off the table, so the apple and the
        # plate stay in the poses this run exists to compare.
        action = joint_hold_action(env.unwrapped, base_height_m=self.config.base_height_m)
        shoulder_joint_id = robot.joint_names.index("left_shoulder_roll_joint")
        action[:, shoulder_joint_id] += self.config.sine_amplitude_rad * math.sin(
            2.0 * math.pi * self._step / self.config.sine_period_steps
        )
        if (
            self.config.environment_snapshot is not None
            and not self._snapshot_written
            and self._step >= self.config.snapshot_after_steps
        ):
            definition = STATIC_APPLE_ENVIRONMENT
            snapshot = collect_environment_snapshot(
                definition=definition,
                mode={
                    "name": "evaluation",
                    "runtime": "arena",
                    "arena_embodiment": G1_ARENA_EMBODIMENT,
                    "enable_waist": False,
                },
                env=env,
                role_to_scene_entity=resolve_role_to_scene_entity(env.unwrapped.scene),
            )
            write_environment_snapshot(self.config.environment_snapshot, snapshot)
            self._snapshot_written = True
        self._step += 1
        return action


__all__ = ["G1StandingSineSmokePolicy", "G1StandingSineSmokePolicyCfg"]

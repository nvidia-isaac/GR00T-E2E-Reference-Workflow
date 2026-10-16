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

"""H2 recurrent-state shapes and initial-snapshot validation without simulator imports."""

from __future__ import annotations

import numpy as np


def feedback_shapes(*, num_envs: int, driven: int, depth: int, command_dim: int) -> dict[str, tuple[int, ...]]:
    return {
        "last_actions": (num_envs, driven),
        "base_ang_vel_history": (num_envs, depth, 3),
        "projected_gravity_history": (num_envs, depth, 3),
        "velocity_commands_history": (num_envs, depth, command_dim),
        "controlled_joint_pos_history": (num_envs, depth, driven),
        "controlled_joint_vel_history": (num_envs, depth, driven),
        "actions_history": (num_envs, depth, driven),
    }


def validate_h2_initial_state(state, *, required: bool = True) -> bool:
    """Validate the complete controller/actuator pair before restoring or qualifying it.

    Legacy inspection may omit both groups. Partial or malformed snapshots never
    qualify, and action replay requires the pair even for a legacy recording.
    """

    from e2e_robot_workflows.robots.h2 import H2_LIVE_JOINT_COUNT, load_h2_robot_config

    if not hasattr(state, "keys"):
        raise ValueError("H2 initial state must be a mapping")
    groups = ("h2_agile_feedback", "h2_actuator_targets")
    present = [name in state for name in groups]
    if not any(present) and not required:
        return False
    if not all(present):
        raise ValueError("H2 initial state requires both h2_agile_feedback and h2_actuator_targets")
    cfg = load_h2_robot_config()
    shapes = feedback_shapes(
        num_envs=1, driven=len(cfg.driven_joint_names), depth=cfg.history_length, command_dim=cfg.command_dim
    )
    for group, expected in (
        (groups[0], shapes),
        (groups[1], {"position": (1, H2_LIVE_JOINT_COUNT), "velocity": (1, H2_LIVE_JOINT_COUNT)}),
    ):
        values = state[group]
        if not hasattr(values, "keys") or set(values.keys()) != set(expected):
            raise ValueError(f"H2 {group} has missing or unexpected fields")
        for name, shape in expected.items():
            value = values[name]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            value = np.asarray(value)
            if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"H2 {group}/{name} must contain finite numeric values with shape {shape}")
    return True

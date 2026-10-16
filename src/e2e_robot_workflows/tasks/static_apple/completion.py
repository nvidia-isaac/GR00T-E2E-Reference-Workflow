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

"""Static-apple episode-completion policy consumed by the recorder.

Owns this task's completion semantics -- the geometric success/failure detector
(:mod:`.success`), its thresholds (from the environment contract), the
follow-through length, and whether auto-detection is on by default -- so the
generic recorder stays task-agnostic. Registered as this task's
``completion_policy_entry_point`` in :mod:`.__init__`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from e2e_robot_workflows.tasks.static_apple.spec import ENVIRONMENT
from e2e_robot_workflows.tasks.static_apple.success import Outcome, PlacementCriteria, check_readable, evaluate

# Task-namespaced --completion_override keys -> PlacementCriteria fields. Only the
# geometric failure thresholds are session-tunable; the success thresholds (placement
# radius / contact force / velocity) come from the contract and stay contract-only, so
# recording cannot silently drift from the criteria evaluation scores with.
_CRITERIA_OVERRIDES = {
    "drop_height_m": "drop_height_m",
    "far_distance_m": "far_distance_m",
}
_HOLD_OVERRIDE = "success_hold_frames"
_VALID_OVERRIDES = frozenset(_CRITERIA_OVERRIDES) | {_HOLD_OVERRIDE}


@dataclass(frozen=True)
class StaticAppleCompletionPolicy:
    """Apple-on-plate completion: settle detection plus a follow-through hold."""

    criteria: PlacementCriteria = field(default_factory=PlacementCriteria)
    hold_frames: int = ENVIRONMENT.success_hold_frames
    default_enabled: bool = True

    def evaluate(self, env: Any) -> Outcome:
        return evaluate(env, self.criteria)

    def check_readable(self, env: Any) -> None:
        check_readable(env)
        randomization = getattr(env.cfg, "e2e_table_height_randomization", None)
        if randomization is not None:
            lowest_supported_height = (
                float(env.cfg.e2e_support_surface_z_m)
                + randomization.offset_range_m[0]
                - self.criteria.min_supported_height_below_support_m
            )
            if lowest_supported_height <= self.criteria.drop_height_m:
                raise ValueError(
                    "table height offset range puts supported placements at or below the effective drop_height_m"
                )


def build_policy(overrides: dict[str, float] | None = None) -> StaticAppleCompletionPolicy:
    """Build the policy, applying task-namespaced overrides onto the contract defaults."""

    overrides = overrides or {}
    unknown = sorted(set(overrides) - _VALID_OVERRIDES)
    if unknown:
        raise ValueError(f"unknown completion override(s) {unknown}; valid keys: {sorted(_VALID_OVERRIDES)}")
    for key in _CRITERIA_OVERRIDES:
        if key in overrides and (not math.isfinite(overrides[key]) or overrides[key] <= 0):
            raise ValueError(f"completion override '{key}' must be positive and finite")
    criteria = PlacementCriteria(
        **{field_name: overrides[key] for key, field_name in _CRITERIA_OVERRIDES.items() if key in overrides}
    )
    hold_frames = ENVIRONMENT.success_hold_frames
    if _HOLD_OVERRIDE in overrides:
        value = overrides[_HOLD_OVERRIDE]
        if value != int(value) or value < 0:
            raise ValueError(f"completion override '{_HOLD_OVERRIDE}' must be a non-negative integer")
        hold_frames = int(value)
    return StaticAppleCompletionPolicy(criteria=criteria, hold_frames=hold_frames)

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

"""Load task-owned static-apple environment variants."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from e2e_robot_workflows.common import load_yaml_mapping
from e2e_robot_workflows.environments import STATIC_APPLE_ENVIRONMENT, StaticAppleEnvironmentDefinition

_DEFAULT_TASK_CONFIG = Path(__file__).resolve().parents[4] / "configs" / "tasks" / "static_apple.yaml"
H2_STATIC_APPLE_VARIANT = "h2_reachable_v1"
_VARIANT_FIELDS = frozenset(
    {
        "support_surface_z_m",
        "pick_spawn_xy",
        "destination_spawn_xy",
        "release_clearance_m",
        "pickup_lift_clearance_m",
    }
)


def _finite_xy(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{name} must contain two coordinates")
    result = tuple(float(coordinate) for coordinate in value)
    if not all(math.isfinite(coordinate) for coordinate in result):
        raise ValueError(f"{name} must contain finite coordinates")
    return result


def load_static_apple_environment(
    variant: str | None = None,
    path: str | Path | None = None,
) -> StaticAppleEnvironmentDefinition:
    """Return the base task contract or one task-configured embodiment variant."""

    if variant is None:
        return STATIC_APPLE_ENVIRONMENT

    config_path = path or _DEFAULT_TASK_CONFIG
    data = load_yaml_mapping(str(config_path))
    variants = data.get("environment_variants")
    if not isinstance(variants, dict) or variant not in variants:
        raise ValueError(f"static-apple environment variant {variant!r} is not configured")
    values = variants[variant]
    if not isinstance(values, dict) or set(values) != _VARIANT_FIELDS:
        missing = sorted(_VARIANT_FIELDS - set(values) if isinstance(values, dict) else _VARIANT_FIELDS)
        unexpected = sorted(set(values) - _VARIANT_FIELDS) if isinstance(values, dict) else []
        raise ValueError(f"static-apple variant {variant!r} has missing={missing}, unexpected={unexpected}")

    support_surface_z = float(values["support_surface_z_m"])
    if not math.isfinite(support_surface_z) or support_surface_z < 0:
        raise ValueError("support_surface_z_m must be non-negative and finite")
    release_clearance = float(values["release_clearance_m"])
    pickup_lift_clearance = float(values["pickup_lift_clearance_m"])
    if not math.isfinite(release_clearance) or release_clearance <= 0:
        raise ValueError("release_clearance_m must be positive and finite")
    if not math.isfinite(pickup_lift_clearance) or pickup_lift_clearance <= 0:
        raise ValueError("pickup_lift_clearance_m must be positive and finite")

    definition = STATIC_APPLE_ENVIRONMENT
    support = definition.asset("support")
    support_position = (
        definition.support_position[0],
        definition.support_position[1],
        support_surface_z - definition.support_native_top_z * support.scale[2],
    )
    return replace(
        definition,
        support_position=support_position,
        pick_spawn_xy=_finite_xy(values["pick_spawn_xy"], "pick_spawn_xy"),
        destination_spawn_xy=_finite_xy(values["destination_spawn_xy"], "destination_spawn_xy"),
        release_clearance_m=release_clearance,
        pickup_lift_clearance_m=pickup_lift_clearance,
    )


__all__ = ["H2_STATIC_APPLE_VARIANT", "load_static_apple_environment"]

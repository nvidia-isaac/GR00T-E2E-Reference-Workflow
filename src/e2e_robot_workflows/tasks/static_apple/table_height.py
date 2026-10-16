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

"""Height-only resets and the recorded support contract for the static apple scene."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from e2e_robot_workflows.environments.static_apple import StaticAppleEnvironmentDefinition

SUPPORT_REPRESENTATION = "kinematic_rigid"
HEIGHT_TOLERANCE_M = 1e-4  # Float32 pose round-trips.


@dataclass
class TableHeightRandomization:
    """Environment-local nominal geometry and the per-reset offset range.

    Mutable because Isaac Lab's configclass reassigns nested dataclass fields.
    """

    support_asset_name: str
    nominal_support_root_z_m: float
    nominal_support_surface_z_m: float
    object_spawn_z_m: dict[str, float]
    offset_range_m: tuple[float, float]


def parse_offset_range(values: Sequence[float] | None) -> tuple[float, float] | None:
    """Validate metre bounds; omission and ``[0, 0]`` disable sampling and table state."""

    if values is None:
        return None
    if not isinstance(values, Sequence) or isinstance(values, str) or len(values) != 2:
        raise ValueError("table height offset range must contain two values: MIN MAX")
    if any(isinstance(value, bool) or not isinstance(value, int | float) for value in values):
        raise ValueError("table height offset range must contain numbers")
    low, high = values
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError("table height offset range must be finite and ordered")
    return None if low == high == 0 else (float(low), float(high))


def resolve_randomization(
    *, definition: StaticAppleEnvironmentDefinition, offset_range_m: Sequence[float] | None
) -> TableHeightRandomization | None:
    """Resolve reset geometry from the same definition that composes the scene."""

    offset_range = parse_offset_range(offset_range_m)
    if offset_range is None:
        return None
    lowest_supported_height = (
        definition.support_surface_z + offset_range[0] - definition.min_supported_height_below_support_m
    )
    if lowest_supported_height <= definition.failure_drop_height_m:
        raise ValueError("table height offset range puts supported placements at or below the failure plane")
    return TableHeightRandomization(
        support_asset_name=definition.asset("support").name,
        nominal_support_root_z_m=definition.support_position[2],
        nominal_support_surface_z_m=definition.support_surface_z,
        object_spawn_z_m={
            definition.asset(role).name: definition.spawn_position(role)[2] for role in ("pick_object", "destination")
        },
        offset_range_m=offset_range,
    )


def _write_height(env: Any, asset_name: str, env_ids: Any, origin_z: Any, local_z: Any) -> None:
    asset = env.scene[asset_name]
    pose = asset.data.root_pose_w.torch[env_ids].clone()
    pose[:, 2] = origin_z + local_z
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)


def randomize_support_height(env: Any, env_ids: Any, randomization: TableHeightRandomization) -> None:
    """Run after the ordinary resets, preserving sampled XY/yaw and never accumulating Z.

    Height uses the seeded Torch stream; existing XY/yaw sampling uses Python random.
    """

    if env_ids is None:
        import torch

        env_ids = torch.arange(env.scene.num_envs, device=env.scene.env_origins.device)
    if len(env_ids) == 0:
        return
    origin_z = env.scene.env_origins[env_ids, 2]
    offsets = origin_z.clone().uniform_(*randomization.offset_range_m)
    _write_height(
        env, randomization.support_asset_name, env_ids, origin_z, offsets + randomization.nominal_support_root_z_m
    )
    for asset_name, nominal_z in randomization.object_spawn_z_m.items():
        _write_height(env, asset_name, env_ids, origin_z, offsets + nominal_z)
        env.scene[asset_name].write_root_velocity_to_sim(offsets.new_zeros((len(env_ids), 6)), env_ids=env_ids)


def support_height_offset(env: Any) -> Any:
    """Read per-environment offsets from the live table, or zero for nominal scenes."""

    cfg = getattr(env, "cfg", None)
    asset_name = getattr(cfg, "e2e_support_asset_name", None)
    if asset_name is None:
        return 0.0
    return env.scene[asset_name].data.root_pose_w.torch[:, 2] - env.scene.env_origins[:, 2] - cfg.e2e_support_root_z_m


def support_height_offset_value(env: Any, index: int = 0) -> float:
    """Return one environment's live offset as a host float."""

    offset = support_height_offset(env)
    return offset if isinstance(offset, float) else float(offset[index])


def scene_randomization_manifest(randomization: TableHeightRandomization, *, seed: int | None) -> dict[str, Any]:
    """Describe the run's distribution; realized offsets live in recorded table poses."""

    return {
        "support_asset": randomization.support_asset_name,
        "support_representation": SUPPORT_REPRESENTATION,
        "nominal_support_root_z_m": randomization.nominal_support_root_z_m,
        "nominal_support_surface_z_m": randomization.nominal_support_surface_z_m,
        "object_spawn_z_m": dict(randomization.object_spawn_z_m),
        "table_height_offset_range_m": list(randomization.offset_range_m),
        "seed": seed,
    }


def validate_scene_randomization(
    block: Any, *, definition: StaticAppleEnvironmentDefinition, seed: int | None
) -> dict[str, Any]:
    """Require the declared geometry and seed to match the scene and run contracts."""

    if not isinstance(block, Mapping) or not isinstance(block.get("object_spawn_z_m"), Mapping):
        raise ValueError("scene randomization block is malformed")
    randomization = resolve_randomization(
        definition=definition, offset_range_m=block.get("table_height_offset_range_m")
    )
    if randomization is None:
        raise ValueError("scene randomization requires a nonzero offset range")
    heights = [block.get(key) for key in ("nominal_support_root_z_m", "nominal_support_surface_z_m")]
    heights.extend(block["object_spawn_z_m"].values())
    if any(isinstance(value, bool) or not isinstance(value, int | float) for value in heights):
        raise ValueError("scene randomization heights must be numbers")
    expected = scene_randomization_manifest(randomization, seed=seed)
    if block != expected or type(block.get("seed")) is not type(seed):
        raise ValueError("scene randomization does not match the environment geometry and run seed")
    return expected

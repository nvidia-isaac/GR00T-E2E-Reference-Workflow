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

"""Concrete, robot-independent definition of the static apple environment."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any

type Vec2 = tuple[float, float]
type Vec3 = tuple[float, float, float]

# Asset library root; override E2E_ASSET_ROOT to point at a mirror. Assets store a
# root-relative path and the root is excluded from contract_digest, so mirroring the
# library does not change environment identity (record/eval parity holds).
_DEFAULT_ASSET_ROOT = "omniverse://isaac-dev.ov.nvidia.com/Projects/GR00T_E2E_RA/assets"


def _asset_root() -> str:
    """Resolve the asset library root, honoring E2E_ASSET_ROOT (empty falls back)."""

    return (os.environ.get("E2E_ASSET_ROOT") or _DEFAULT_ASSET_ROOT).rstrip("/")


_REQUIRED_ASSET_ROLES = {"background", "support", "pick_object", "destination"}
# Both static-apple adapters leave Isaac Lab's default gravity unchanged at
# ``(0.0, 0.0, -9.81)`` m/s^2. Keep the validation assumption explicit here;
# an adapter that configures gravity must source this guard from the same value.
_STANDARD_GRAVITY_MPS2 = 9.81


@dataclass(frozen=True)
class SceneAssetDefinition:
    """One asset in the static apple workcell."""

    role: str
    name: str
    prim_path: str
    usd_relpath: str  # relative to the asset root; see _asset_root
    scale: Vec3 = (1.0, 1.0, 1.0)
    rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    origin_above_bottom_m: float = 0.0
    # Minimum horizontal distance from the prim origin to the footprint rim, unscaled.
    # Success gates on the root position (the prim origin), and geometry can sit off-centre
    # from it, so only the *minimum* rim distance stays inside the footprint in every
    # direction.
    placement_rim_radius_m: float | None = None
    mass_kg: float | None = None
    kinematic: bool = False

    @property
    def usd_path(self) -> str:
        """Full USD path: the env-resolved asset root joined with the relative path."""

        return f"{_asset_root()}/{self.usd_relpath}"


@dataclass(frozen=True)
class StaticAppleEnvironmentDefinition:
    """Scene composition and task semantics shared by recording and evaluation."""

    environment_id: str
    assets: tuple[SceneAssetDefinition, ...]
    background_position: Vec3
    support_position: Vec3
    support_native_top_z: float
    support_airgap_m: float
    pick_spawn_xy: Vec2
    destination_spawn_xy: Vec2
    # Per-episode spawn randomization (half-ranges; 0 pins the object to its nominal
    # spawn). Applied on reset by tasks/static_apple/env_cfg.py.
    pick_spawn_xy_range_m: float
    pick_spawn_yaw_range_rad: float
    destination_spawn_xy_range_m: float
    deactivate_background_prims: tuple[str, ...]
    deactivate_support_prims: tuple[str, ...]
    task_name: str
    family: str
    language_instruction: str
    description: str
    episode_length_s: float
    physics_hz: int
    control_hz: int
    # Automatic episode success/failure detection; see tasks/static_apple/success.py.
    # SUCCESS is eval-shared (placement footprint + contact force + settled velocity);
    # FAILURE is recorder-only and geometric (drop / far / plate-drop).
    success_force_threshold: float
    success_velocity_threshold: float
    max_object_height_above_support_m: float
    min_supported_height_below_support_m: float
    # The plate asset's root can sit slightly above its contact surface. Allowing
    # the apple root to be at most 1 cm lower tolerates that origin offset; contact,
    # footprint, and support checks still reject placements below the plate/table.
    min_object_height_above_target_m: float
    release_clearance_m: float
    pickup_lift_clearance_m: float
    failure_drop_height_m: float
    failure_far_distance_m: float
    # Extra frames to keep recording after success is detected (follow-through)
    # before the episode is saved and the scene resets. 0 saves immediately.
    success_hold_frames: int
    metrics: tuple[str, ...]
    qualification_note: str

    def __post_init__(self) -> None:
        roles = [asset.role for asset in self.assets]
        if set(roles) != _REQUIRED_ASSET_ROLES or len(roles) != len(set(roles)):
            raise ValueError("static apple assets must define each required role exactly once")
        if any(
            len(asset.scale) != 3 or not all(math.isfinite(value) and value > 0 for value in asset.scale)
            for asset in self.assets
        ):
            raise ValueError("asset scales must contain three positive finite values")
        if self.physics_hz <= 0 or self.control_hz <= 0 or self.physics_hz % self.control_hz:
            raise ValueError("physics_hz must be a positive integer multiple of control_hz")
        for role in ("pick_object", "destination"):
            asset = self.asset(role)
            if asset.kinematic:
                continue  # kinematic bodies need no mass
            mass = asset.mass_kg
            if mass is None or not math.isfinite(mass) or mass <= 0:
                raise ValueError(f"non-kinematic '{role}' must declare a positive finite mass")
        for name in (
            "success_force_threshold",
            "success_velocity_threshold",
            "max_object_height_above_support_m",
            "release_clearance_m",
            "pickup_lift_clearance_m",
            "failure_drop_height_m",
            "failure_far_distance_m",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        pick_object_mass = self.asset("pick_object").mass_kg
        if pick_object_mass is None:
            raise ValueError("pick_object must declare a mass for the contact-force success contract")
        if self.success_force_threshold >= pick_object_mass * _STANDARD_GRAVITY_MPS2:
            raise ValueError("success_force_threshold must be below the pick object's nominal static weight")
        if (
            not math.isfinite(self.min_supported_height_below_support_m)
            or self.min_supported_height_below_support_m < 0
        ):
            raise ValueError("min_supported_height_below_support_m must be non-negative and finite")
        # Spawn half-ranges may be 0 (pins the object), but a negative or non-finite
        # value makes the reset min/max interval invalid on both runtimes.
        for name in ("pick_spawn_xy_range_m", "pick_spawn_yaw_range_rad", "destination_spawn_xy_range_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")
        # bool is a subclass of int; True/False are not a frame count.
        if isinstance(self.success_hold_frames, bool) or not isinstance(self.success_hold_frames, int):
            raise ValueError("success_hold_frames must be a non-negative integer")
        if self.success_hold_frames < 0:
            raise ValueError("success_hold_frames must be a non-negative integer")
        # Placement success derives from geometry: the plate needs a positive, XY-uniform
        # rim radius (one radius, both axes) and the apple a positive origin height (sets
        # the z tolerance).
        plate = self.asset("destination")
        if (
            plate.placement_rim_radius_m is None
            or not math.isfinite(plate.placement_rim_radius_m)
            or plate.placement_rim_radius_m <= 0
        ):
            raise ValueError("destination must declare a positive finite placement_rim_radius_m")
        if plate.scale[0] != plate.scale[1]:
            raise ValueError("destination placement radius requires a uniform XY scale")
        apple_origin = self.asset("pick_object").origin_above_bottom_m
        if not math.isfinite(apple_origin) or apple_origin <= 0:
            raise ValueError("pick_object must declare a positive origin_above_bottom_m for placement success")
        if (
            not math.isfinite(self.min_object_height_above_target_m)
            or abs(self.min_object_height_above_target_m) >= self.success_placement_z_m
        ):
            raise ValueError(
                "min_object_height_above_target_m must be finite and inside the placement-height tolerance"
            )

    def asset(self, role: str) -> SceneAssetDefinition:
        """Return the asset assigned to a semantic role."""

        for asset in self.assets:
            if asset.role == role:
                return asset
        raise KeyError(f"Environment {self.environment_id!r} has no asset role {role!r}")

    @property
    def support_surface_z(self) -> float:
        """World-local height of the scaled support surface."""

        support = self.asset("support")
        return self.support_position[2] + self.support_native_top_z * support.scale[2]

    def spawn_position(self, role: str) -> Vec3:
        """Return the initial position for a task object."""

        if role == "pick_object":
            xy = self.pick_spawn_xy
        elif role == "destination":
            xy = self.destination_spawn_xy
        else:
            raise KeyError(f"Asset role {role!r} has no task-object spawn position")
        asset = self.asset(role)
        return (
            xy[0],
            xy[1],
            self.support_surface_z + self.support_airgap_m + asset.origin_above_bottom_m,
        )

    @property
    def success_placement_xy_m(self) -> float:
        """Max apple-center-to-plate-origin XY distance counted as placed.

        A circle at the scaled minimum rim distance, so it stays inside the footprint in
        every direction and rejects side/rim touches. The recorder and AutoData gate on
        this radius; Arena's per-axis box uses ``success_placement_box_half_width_m``.
        """

        plate = self.asset("destination")
        assert plate.placement_rim_radius_m is not None  # enforced in __post_init__
        return plate.placement_rim_radius_m * plate.scale[0]

    @property
    def success_placement_box_half_width_m(self) -> float:
        """Half-width of the axis-aligned box inscribed in the circular placement gate.

        Arena's ``objects_in_proximity`` is per-axis, and an r-half-width box reaches
        r*sqrt(2) at its corners -- past the plate, re-admitting corner side-touches.
        Inscribing it keeps evaluation leak-free.
        """

        return self.success_placement_xy_m / math.sqrt(2.0)

    @property
    def success_placement_z_m(self) -> float:
        """Vertical allowance for the placement check (apple diameter, scaled).

        Generous enough that an apple resting on the plate passes; the horizontal
        gate is what rejects side touches, so this only bounds gross height error.
        """

        apple = self.asset("pick_object")
        return 2.0 * apple.origin_above_bottom_m * apple.scale[2]

    @property
    def contract_digest(self) -> str:
        """Return a stable digest of the declared environment and task values."""

        payload = json.dumps(asdict(self), ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def contract_snapshot(self) -> dict[str, Any]:
        """Return the mode-neutral values compared across the two runtimes."""

        return {
            "environment_id": self.environment_id,
            "environment_contract_digest": self.contract_digest,
            "assets": {
                asset.role: {
                    "name": asset.name,
                    "prim_path": asset.prim_path,
                    "usd_relpath": asset.usd_relpath,
                    "scale": list(asset.scale),
                    "rotation_xyzw": list(asset.rotation_xyzw),
                    "origin_above_bottom_m": asset.origin_above_bottom_m,
                    "mass_kg": asset.mass_kg,
                    "kinematic": asset.kinematic,
                }
                for asset in self.assets
            },
            "background_position": list(self.background_position),
            "support_position": list(self.support_position),
            "support_surface_z": self.support_surface_z,
            "support_airgap_m": self.support_airgap_m,
            "pick_spawn_position": list(self.spawn_position("pick_object")),
            "destination_spawn_position": list(self.spawn_position("destination")),
            "pick_spawn_xy_range_m": self.pick_spawn_xy_range_m,
            "pick_spawn_yaw_range_rad": self.pick_spawn_yaw_range_rad,
            "destination_spawn_xy_range_m": self.destination_spawn_xy_range_m,
            "deactivate_background_prims": list(self.deactivate_background_prims),
            "deactivate_support_prims": list(self.deactivate_support_prims),
            "task": {
                "name": self.task_name,
                "family": self.family,
                "language_instruction": self.language_instruction,
                "description": self.description,
                "episode_length_s": self.episode_length_s,
                "physics_hz": self.physics_hz,
                "control_hz": self.control_hz,
                "success_force_threshold": self.success_force_threshold,
                "success_velocity_threshold": self.success_velocity_threshold,
                "success_placement_xy_m": self.success_placement_xy_m,
                "success_placement_box_half_width_m": self.success_placement_box_half_width_m,
                "success_placement_z_m": self.success_placement_z_m,
                "max_object_height_above_support_m": self.max_object_height_above_support_m,
                "min_supported_height_below_support_m": self.min_supported_height_below_support_m,
                "min_object_height_above_target_m": self.min_object_height_above_target_m,
                "release_clearance_m": self.release_clearance_m,
                "pickup_lift_clearance_m": self.pickup_lift_clearance_m,
                "failure_drop_height_m": self.failure_drop_height_m,
                "failure_far_distance_m": self.failure_far_distance_m,
                "success_hold_frames": self.success_hold_frames,
                "metrics": list(self.metrics),
            },
        }


STATIC_APPLE_ENVIRONMENT = StaticAppleEnvironmentDefinition(
    environment_id="static_apple_v0",
    assets=(
        SceneAssetDefinition(
            role="background",
            name="scene_background",
            prim_path="{ENV_REGEX_NS}/scene_background",
            usd_relpath="background_library/warehouse/warehouse.usd",
        ),
        SceneAssetDefinition(
            role="support",
            name="packing_table",
            prim_path="{ENV_REGEX_NS}/packing_table",
            usd_relpath="object_library/table/table.usd",
            kinematic=True,
        ),
        SceneAssetDefinition(
            role="pick_object",
            name="pick_object",
            prim_path="{ENV_REGEX_NS}/pick_object",
            usd_relpath="object_library/apple/apple.usd",
            origin_above_bottom_m=0.0514,
            mass_kg=0.05,
        ),
        SceneAssetDefinition(
            role="destination",
            name="placement_target",
            prim_path="{ENV_REGEX_NS}/placement_target",
            usd_relpath="object_library/clay_plate/clay_plate.usd",
            # Geometry is off-centre from the prim origin; the nearest rim measures
            # ~0.128 m from it unscaled (~0.09 m after scale).
            scale=(0.7, 0.7, 0.7),
            placement_rim_radius_m=0.1276,
            mass_kg=0.2,
            kinematic=False,
        ),
    ),
    background_position=(0.0, 0.0, 0.0),
    support_position=(0.0, 0.55, 0.0),
    support_native_top_z=0.7581,
    support_airgap_m=0.005,
    pick_spawn_xy=(-0.15, 0.3),
    destination_spawn_xy=(0.05, 0.3),
    # Apple XY jitter only (0.02 m, kept small because the plate shares the shelf); no
    # apple rotation and a fixed plate keep the place target deterministic.
    pick_spawn_xy_range_m=0.02,
    pick_spawn_yaw_range_rad=0.0,
    destination_spawn_xy_range_m=0.0,
    deactivate_background_prims=(),
    deactivate_support_prims=(),
    task_name="static_apple_transfer",
    family="object_transfer",
    language_instruction="move the apple to the plate",
    description=(
        "The robot stands in front of a table, picks up an apple, and places it onto a plate on the same table."
    ),
    episode_length_s=6.0,
    physics_hz=200,
    control_hz=50,
    # The 0.05 kg apple weighs about 0.4905 N. Keep a 10% solver margin so a
    # genuinely resting contact can pass the consecutive-frame success hold.
    success_force_threshold=0.44,
    success_velocity_threshold=0.1,
    max_object_height_above_support_m=0.10,
    min_supported_height_below_support_m=0.05,
    min_object_height_above_target_m=-0.01,
    release_clearance_m=0.10,
    pickup_lift_clearance_m=0.04,
    failure_drop_height_m=0.5,
    failure_far_distance_m=0.35,
    success_hold_frames=20,
    metrics=("success_rate", "object_moved_rate"),
    qualification_note=(
        "Architecture-validation environment preserving the current E2E recording scene. "
        "It does not claim equivalence with Arena's first-party Galileo scene or any "
        "released checkpoint performance."
    ),
)

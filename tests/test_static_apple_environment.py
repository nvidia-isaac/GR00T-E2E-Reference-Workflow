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

"""Host-side contracts for the shared static apple environment definition."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import yaml

from e2e_robot_workflows.environments import STATIC_APPLE_ENVIRONMENT
from e2e_robot_workflows.robots.g1 import load_g1_robot_config
from e2e_robot_workflows.tasks.static_apple.config import H2_STATIC_APPLE_VARIANT, load_static_apple_environment

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_definition_owns_scene_and_task_values_without_a_robot_binding() -> None:
    environment = STATIC_APPLE_ENVIRONMENT

    assert environment.environment_id == "static_apple_v0"
    assert environment.task_name == "static_apple_transfer"
    assert environment.family == "object_transfer"
    assert not hasattr(environment, "robot")
    assert not hasattr(environment, "embodiment")
    assert environment.contract_digest.startswith("sha256:")
    assert len(environment.contract_digest) == len("sha256:") + 64
    assert "does not claim equivalence" in environment.qualification_note


def test_current_e2e_scene_geometry_is_preserved() -> None:
    environment = STATIC_APPLE_ENVIRONMENT

    assert environment.support_surface_z == 0.7581
    assert environment.spawn_position("pick_object") == (-0.15, 0.3, 0.8145)
    assert environment.spawn_position("destination") == (0.05, 0.3, 0.7631)
    # Apple USD is pre-sized (1.0); plate USD is ~30 cm, scaled to ~21 cm.
    assert environment.asset("pick_object").scale == (1.0, 1.0, 1.0)
    assert environment.asset("destination").scale == (0.7, 0.7, 0.7)
    # The plate is a dynamic (movable) body that falls and rests on the table.
    assert environment.asset("destination").kinematic is False
    assert environment.asset("destination").mass_kg == 0.2
    assert environment.asset("destination").placement_rim_radius_m == 0.1276
    assert environment.asset("support").prim_path == "{ENV_REGEX_NS}/packing_table"
    assert environment.deactivate_support_prims == ()
    assert environment.physics_hz == 200
    assert environment.control_hz == 50


def test_success_detection_thresholds_are_declared_and_validated() -> None:
    environment = STATIC_APPLE_ENVIRONMENT

    # Success thresholds are eval-shared (used during evaluation).
    assert environment.success_force_threshold == 0.44
    pick_object_mass = environment.asset("pick_object").mass_kg
    assert pick_object_mass is not None
    static_weight = pick_object_mass * 9.81
    assert environment.success_force_threshold < static_weight
    assert environment.success_velocity_threshold == 0.1
    assert environment.max_object_height_above_support_m == 0.10
    assert environment.min_supported_height_below_support_m == 0.05
    assert environment.min_object_height_above_target_m == -0.01
    assert environment.release_clearance_m == 0.10
    assert environment.pickup_lift_clearance_m == 0.04
    assert environment.failure_drop_height_m == 0.5
    assert environment.failure_far_distance_m == 0.35
    # Placement footprint is computed from the plate/apple sizes, not hardcoded.
    assert environment.success_placement_xy_m == pytest.approx(0.1276 * 0.7)  # min rim x scale
    # Arena's per-axis box is inscribed in that circle, so its corners stay on the plate.
    assert environment.success_placement_box_half_width_m == pytest.approx(environment.success_placement_xy_m / 2**0.5)
    assert environment.success_placement_z_m == pytest.approx(2 * 0.0514)  # apple diameter
    # Tunable follow-through count; assert it stays a valid non-negative int.
    assert isinstance(environment.success_hold_frames, int) and environment.success_hold_frames >= 0
    for name in (
        "success_force_threshold",
        "success_velocity_threshold",
        "max_object_height_above_support_m",
        "release_clearance_m",
        "pickup_lift_clearance_m",
        "failure_drop_height_m",
        "failure_far_distance_m",
    ):
        with pytest.raises(ValueError, match="positive and finite"):
            replace(environment, **{name: 0.0})
        with pytest.raises(ValueError, match="positive and finite"):
            replace(environment, **{name: float("nan")})
    with pytest.raises(ValueError, match="below the pick object's nominal static weight"):
        replace(environment, success_force_threshold=static_weight)
    with pytest.raises(ValueError, match="non-negative and finite"):
        replace(environment, min_supported_height_below_support_m=-0.01)
    with pytest.raises(ValueError, match="inside the placement-height tolerance"):
        replace(environment, min_object_height_above_target_m=float("nan"))
    with pytest.raises(ValueError, match="inside the placement-height tolerance"):
        replace(environment, min_object_height_above_target_m=environment.success_placement_z_m)
    with pytest.raises(ValueError, match="non-negative integer"):
        replace(environment, success_hold_frames=-1)
    # bool is an int subclass; True/False are not a frame count.
    with pytest.raises(ValueError, match="non-negative integer"):
        replace(environment, success_hold_frames=True)


def test_placement_footprint_requires_the_plate_radius() -> None:
    """The placement success box is derived from the plate footprint, so it must exist."""

    with pytest.raises(ValueError, match="placement_rim_radius_m"):
        _replace_asset(STATIC_APPLE_ENVIRONMENT, "destination", placement_rim_radius_m=None)


def test_placement_requires_apple_origin_and_uniform_plate_scale() -> None:
    # The z tolerance derives from the apple origin height; a zero height is invalid.
    with pytest.raises(ValueError, match="origin_above_bottom_m"):
        _replace_asset(STATIC_APPLE_ENVIRONMENT, "pick_object", origin_above_bottom_m=0.0)
    # One footprint radius covers both axes, so the plate must be uniformly scaled in XY.
    with pytest.raises(ValueError, match="uniform XY scale"):
        _replace_asset(STATIC_APPLE_ENVIRONMENT, "destination", scale=(0.7, 0.9, 0.7))


def test_spawn_ranges_are_non_negative_and_finite() -> None:
    environment = STATIC_APPLE_ENVIRONMENT

    for name in ("pick_spawn_xy_range_m", "pick_spawn_yaw_range_rad", "destination_spawn_xy_range_m"):
        # 0 is allowed (pins the object to its nominal spawn).
        assert replace(environment, **{name: 0.0}).contract_digest.startswith("sha256:")
        with pytest.raises(ValueError, match="non-negative and finite"):
            replace(environment, **{name: -0.01})
        with pytest.raises(ValueError, match="non-negative and finite"):
            replace(environment, **{name: float("nan")})


def test_environment_digest_is_stable_and_cwd_independent(tmp_path: Path) -> None:
    expected = STATIC_APPLE_ENVIRONMENT.contract_digest
    command = [
        sys.executable,
        "-c",
        (
            "from e2e_robot_workflows.environments import STATIC_APPLE_ENVIRONMENT;"
            "print(STATIC_APPLE_ENVIRONMENT.contract_digest)"
        ),
    ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == expected


def test_asset_mirror_preserves_layout_and_identity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    environment = STATIC_APPLE_ENVIRONMENT
    monkeypatch.delenv("E2E_ASSET_ROOT", raising=False)
    default_paths = [asset.usd_path for asset in environment.assets]
    digest = environment.contract_digest
    snapshot = environment.contract_snapshot()
    for asset in environment.assets:
        path = tmp_path / asset.usd_relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(asset.role, encoding="utf-8")

    monkeypatch.setenv("E2E_ASSET_ROOT", f"{tmp_path}/")
    for asset in environment.assets:
        assert Path(asset.usd_path).read_text(encoding="utf-8") == asset.role
    assert environment.contract_digest == digest
    assert environment.contract_snapshot() == snapshot

    monkeypatch.setenv("E2E_ASSET_ROOT", "")
    assert [asset.usd_path for asset in environment.assets] == default_paths


def test_placement_thresholds_are_in_the_digest_and_snapshot() -> None:
    environment = STATIC_APPLE_ENVIRONMENT
    snapshot = environment.contract_snapshot()["task"]

    for name in (
        "max_object_height_above_support_m",
        "min_supported_height_below_support_m",
        "min_object_height_above_target_m",
        "release_clearance_m",
        "pickup_lift_clearance_m",
    ):
        assert snapshot[name] == getattr(environment, name)
        assert replace(environment, **{name: getattr(environment, name) + 0.001}).contract_digest != (
            environment.contract_digest
        )


def test_definition_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        STATIC_APPLE_ENVIRONMENT.environment_id = "changed"  # type: ignore[misc]


def _replace_asset(environment: object, role: str, **changes: object) -> object:
    """Return the environment with one asset's fields replaced."""

    updated = replace(environment.asset(role), **changes)
    return replace(
        environment,
        assets=tuple(updated if asset.role == role else asset for asset in environment.assets),
    )


def test_definition_rejects_invalid_required_values() -> None:
    environment = STATIC_APPLE_ENVIRONMENT
    with pytest.raises(ValueError, match="required role"):
        replace(environment, assets=tuple(asset for asset in environment.assets if asset.role != "destination"))
    with pytest.raises(ValueError, match="physics_hz"):
        replace(environment, control_hz=0)
    # Both task objects are simulated, so both must declare a usable mass.
    for role in ("pick_object", "destination"):
        for invalid_mass in (-1.0, 0.0, None, float("nan")):
            with pytest.raises(ValueError, match="positive finite mass"):
                _replace_asset(environment, role, mass_kg=invalid_mass)


def test_kinematic_assets_are_exempt_from_the_mass_requirement() -> None:
    """PhysX ignores a kinematic body's mass, so the contract must not demand one."""

    kinematic_plate = _replace_asset(STATIC_APPLE_ENVIRONMENT, "destination", kinematic=True, mass_kg=None)

    assert kinematic_plate.asset("destination").mass_kg is None
    assert STATIC_APPLE_ENVIRONMENT.asset("support").kinematic is True
    assert STATIC_APPLE_ENVIRONMENT.asset("support").mass_kg is None


def test_discovery_config_owns_h2_task_geometry() -> None:
    task_pointer = yaml.safe_load((REPO_ROOT / "configs/tasks/static_apple.yaml").read_text(encoding="utf-8"))
    h2_robot = yaml.safe_load((REPO_ROOT / "configs/robots/h2.yaml").read_text(encoding="utf-8"))

    assert {key: task_pointer[key] for key in ("environment", "definition_uri", "recording_adapter")} == {
        "environment": "static_apple_v0",
        "definition_uri": ("package://e2e_robot_workflows/environments/static_apple.py:STATIC_APPLE_ENVIRONMENT"),
        "recording_adapter": "g1",
    }
    assert task_pointer["environment_variants"][H2_STATIC_APPLE_VARIANT] == {
        "support_surface_z_m": 0.96,
        "pick_spawn_xy": [0.19, 0.35],
        "destination_spawn_xy": [0.39, 0.35],
        "release_clearance_m": 0.20,
        "pickup_lift_clearance_m": 0.08,
    }
    assert "workcell" not in h2_robot
    assert h2_robot["task_link_names"] == ["left_hand_flange", "right_hand_flange"]
    h2_environment = load_static_apple_environment(H2_STATIC_APPLE_VARIANT)
    assert h2_environment.contract_digest != STATIC_APPLE_ENVIRONMENT.contract_digest
    assert h2_environment.support_surface_z == pytest.approx(0.96)
    assert h2_environment.spawn_position("pick_object") == pytest.approx((0.19, 0.35, 1.0164))
    assert h2_environment.spawn_position("destination") == pytest.approx((0.39, 0.35, 0.965))
    h2_task_snapshot = h2_environment.contract_snapshot()["task"]
    assert h2_task_snapshot["release_clearance_m"] == pytest.approx(0.20)
    assert h2_task_snapshot["pickup_lift_clearance_m"] == pytest.approx(0.08)


@pytest.mark.parametrize(
    "task_link_names",
    [["same", "same"], ["left"], ["", "right"], [1, "right"], "lr"],
)
def test_g1_robot_config_rejects_malformed_task_links(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task_link_names: object,
) -> None:
    config = yaml.safe_load((REPO_ROOT / "configs/robots/g1.yaml").read_text(encoding="utf-8"))
    config["task_link_names"] = task_link_names
    config_path = tmp_path / "g1.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("E2E_G1_ROBOT_CONFIG", "/dev/null")

    with pytest.raises(ValueError, match="two distinct non-empty robot links"):
        load_g1_robot_config(str(config_path))


def test_g1_robot_config_rejects_malformed_direct_task_links() -> None:
    with pytest.raises(ValueError, match="two distinct non-empty robot links"):
        replace(load_g1_robot_config(), task_link_names="lr")


def test_g1_robot_config_uses_the_environment_override_without_an_explicit_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = yaml.safe_load((REPO_ROOT / "configs/robots/g1.yaml").read_text(encoding="utf-8"))
    config["task_link_names"] = ["custom_left", "custom_right"]
    config_path = tmp_path / "g1.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("E2E_G1_ROBOT_CONFIG", str(config_path))

    assert load_g1_robot_config().task_link_names == ("custom_left", "custom_right")


def test_task_variant_preserves_a_zero_support_surface(tmp_path: Path) -> None:
    config_path = tmp_path / "task.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "environment_variants": {
                    "test": {
                        "support_surface_z_m": 0.0,
                        "pick_spawn_xy": [0.0, 0.0],
                        "destination_spawn_xy": [0.1, 0.0],
                        "release_clearance_m": 0.1,
                        "pickup_lift_clearance_m": 0.04,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert load_static_apple_environment("test", config_path).support_surface_z == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"destination_spawn_xy": None}, "two coordinates"),
        ({"support_surface_z_m": float("nan")}, "non-negative and finite"),
        ({"support_surface_z_m": -0.01}, "non-negative and finite"),
        ({"release_clearance_m": 0.0}, "positive and finite"),
    ],
)
def test_task_variant_rejects_malformed_values(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    values = {
        "support_surface_z_m": 0.96,
        "pick_spawn_xy": [0.19, 0.35],
        "destination_spawn_xy": [0.39, 0.35],
        "release_clearance_m": 0.20,
        "pickup_lift_clearance_m": 0.08,
        **mutation,
    }
    config_path = tmp_path / "task.yaml"
    config_path.write_text(
        yaml.safe_dump({"environment_variants": {"test": values}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_static_apple_environment("test", config_path)

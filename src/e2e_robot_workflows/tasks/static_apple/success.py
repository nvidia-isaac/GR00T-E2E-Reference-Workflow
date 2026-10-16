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

"""Automatic success/failure detection for static-apple recording episodes.

The **success** condition is aligned with the E2E Arena and AutoData placement
predicate: the apple must be inside the plate's contract-owned circular footprint,
close to its height, supported, in contact, settled, and released by both hands.
Contact alone is not sufficient because the reset scene can settle the apple against
the plate rim without any robot action.

The apple, the plate, and the contact sensor are looked up by contract role, because
the built-in task and the Arena adapter spawn them under different scene names.

The **failure** condition is recorder-only and stays geometric -- a failed episode
is just a discard signal, so it does not need to match eval: the apple drops off
the table, ends up far from the plate, or the (dynamic) plate is knocked off.

``scripts/record_demos.py`` polls :func:`evaluate` each step to auto-end and label
a teleop episode hands-free. Thresholds live in the environment contract; tune them
there and calibrate once on hardware (see the recording guide).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from e2e_robot_workflows.environments.scene_keys import resolve_contact_sensor_name, resolve_scene_entity_name
from e2e_robot_workflows.tasks.static_apple.spec import ENVIRONMENT
from e2e_robot_workflows.tasks.static_apple.table_height import support_height_offset_value

# Roles of the entities read from the live env, resolved to scene entity names against
# the scene itself. The contact sensor watches the apple, filtered against the plate.
_APPLE_ROLE = "pick_object"
_PLATE_ROLE = "destination"


class Outcome(StrEnum):
    """Result of evaluating an episode at one step."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILURE_DROPPED = "failure_dropped"
    FAILURE_FAR = "failure_far"
    FAILURE_PLATE_DROPPED = "failure_plate_dropped"

    @property
    def is_terminal(self) -> bool:
        return self is not Outcome.RUNNING

    @property
    def is_success(self) -> bool:
        return self is Outcome.SUCCESS


@dataclass(frozen=True)
class PlacementCriteria:
    """Thresholds distinguishing success, failure, and an in-progress episode.

    Defaults come from the shared environment contract (single source of truth);
    override per-instance only for tests. Distances are metres in the env-local
    frame; ``apple_speed`` is the apple's linear speed (m/s); ``contact_force`` is
    the apple<->plate contact-force magnitude (N).
    """

    contact_force_threshold: float = float(ENVIRONMENT.success_force_threshold)
    settle_speed_mps: float = float(ENVIRONMENT.success_velocity_threshold)
    # Placement footprint: the apple center must be within this radius of the plate's
    # origin, so a side/rim touch is not a placement. The contract radius is already the
    # plate's minimum rim distance, so it needs no extra margin here.
    placement_xy_m: float = float(ENVIRONMENT.success_placement_xy_m)
    placement_z_m: float = float(ENVIRONMENT.success_placement_z_m)
    min_object_height_above_target_m: float = ENVIRONMENT.min_object_height_above_target_m
    support_surface_z_m: float = ENVIRONMENT.support_surface_z
    max_object_above_support_m: float = ENVIRONMENT.max_object_height_above_support_m
    min_supported_height_below_support_m: float = ENVIRONMENT.min_supported_height_below_support_m
    release_clearance_m: float = ENVIRONMENT.release_clearance_m
    # Failure (recorder-only, geometric):
    drop_height_m: float = float(ENVIRONMENT.failure_drop_height_m)
    far_distance_m: float = float(ENVIRONMENT.failure_far_distance_m)


DEFAULT_CRITERIA = PlacementCriteria()


def classify(
    apple_xy: tuple[float, float],
    apple_z: float,
    apple_speed: float,
    plate_xy: tuple[float, float],
    plate_z: float,
    contact_force: float,
    task_link_distances: tuple[float, ...] = (),
    criteria: PlacementCriteria = DEFAULT_CRITERIA,
) -> Outcome:
    """Classify an episode from the apple/plate poses, speed, and contact force.

    Failure (geometric, recorder-only) is checked first so a dropped/abandoned
    apple is never mislabeled; the plate-dropped check precedes "too far" because a
    plate on the floor makes its XY a meaningless reference. Success additionally
    requires the contract-owned placement footprint, target height, support, and robot
    release, so an untouched reset or still-held apple cannot be labeled as a transfer.
    """

    if apple_z < criteria.drop_height_m:
        return Outcome.FAILURE_DROPPED
    if plate_z < criteria.drop_height_m:
        return Outcome.FAILURE_PLATE_DROPPED
    xy_distance = math.hypot(apple_xy[0] - plate_xy[0], apple_xy[1] - plate_xy[1])
    if xy_distance > criteria.far_distance_m:
        return Outcome.FAILURE_FAR
    # Radial, not a box: an r-half-width box reaches r*sqrt(2) at its corners, past a
    # round plate of radius r, which admits diagonal side/rim touches.
    on_footprint = xy_distance <= criteria.placement_xy_m and abs(apple_z - plate_z) <= criteria.placement_z_m
    min_supported_height = criteria.support_surface_z_m - criteria.min_supported_height_below_support_m
    supported = apple_z > min_supported_height and plate_z > min_supported_height
    above_target = apple_z - plate_z > criteria.min_object_height_above_target_m
    resting_height = apple_z < criteria.support_surface_z_m + criteria.max_object_above_support_m
    released = bool(task_link_distances) and all(
        distance > criteria.release_clearance_m for distance in task_link_distances
    )
    on_plate = (
        on_footprint
        and supported
        and above_target
        and resting_height
        and released
        and contact_force > criteria.contact_force_threshold
        and apple_speed < criteria.settle_speed_mps
    )
    if on_plate:
        return Outcome.SUCCESS
    return Outcome.RUNNING


def _element_count(array: Any) -> int:
    """Number of elements in ``array``, across the numpy (tests) and torch (sim) backends."""

    count = 1
    for dim in array.shape:
        count *= dim
    return count


def _contact_magnitudes(scene: Any, key: str) -> Any | None:
    """Return env 0's contact-force magnitudes, or ``None`` if the sensor watches nothing.

    Mirrors the evaluation's ``object_on_destination``: the contact sensor's
    ``force_matrix_w`` holds per-filtered-body contact forces (shape ``(N, B, M, 3)``),
    so the result is shaped ``(B, M)``. A sensor configured without filter prims leaves
    the matrix unset, and one whose filters resolve to no bodies leaves it empty; both
    mean the same thing here.
    """

    matrix_source = scene[key].data.force_matrix_w
    if matrix_source is None:
        return None
    matrix = getattr(matrix_source, "torch", matrix_source)
    forces = matrix[0]  # (B, M, 3)
    magnitudes = ((forces * forces).sum(-1)) ** 0.5
    return magnitudes if _element_count(magnitudes) else None


def _task_link_indices(scene: Any, task_link_names: tuple[str, str]) -> tuple[int, int]:
    robot = scene[resolve_scene_entity_name(scene, "robot")]
    missing = [name for name in task_link_names if name not in robot.body_names]
    if missing:
        raise KeyError(f"Configured static-apple task links are absent from the robot: {missing}")
    return tuple(robot.body_names.index(name) for name in task_link_names)


def _configured_task_link_names(env: Any) -> tuple[str, str]:
    names = getattr(getattr(env, "cfg", None), "e2e_task_link_names", None)
    if names is None:
        raise KeyError("The environment does not configure e2e_task_link_names")
    if not isinstance(names, list | tuple):
        raise KeyError("e2e_task_link_names must contain two distinct non-empty robot links")
    result = tuple(names)
    if len(result) != 2 or any(not isinstance(name, str) or not name for name in result) or len(set(result)) != 2:
        raise KeyError("e2e_task_link_names must contain two distinct non-empty robot links")
    return result


def _measurements(
    scene: Any,
    task_link_names: tuple[str, str],
) -> tuple[
    tuple[float, float],
    float,
    float,
    tuple[float, float],
    float,
    float,
    tuple[float, ...],
]:
    """Read one environment's completion inputs with one CUDA-to-host transfer."""

    apple_name = resolve_scene_entity_name(scene, _APPLE_ROLE)
    plate_name = resolve_scene_entity_name(scene, _PLATE_ROLE)
    robot = scene[resolve_scene_entity_name(scene, "robot")]
    apple_source = scene[apple_name].data.root_state_w
    plate_source = scene[plate_name].data.root_state_w
    pose_source = robot.data.body_pos_w
    apple_state = getattr(apple_source, "torch", apple_source)
    plate_state = getattr(plate_source, "torch", plate_source)
    poses = getattr(pose_source, "torch", pose_source)
    origins = getattr(scene, "env_origins", None)
    if origins is not None:
        origins = getattr(origins, "torch", origins)
    link_indices = _task_link_indices(scene, task_link_names)
    contact_magnitudes = _contact_magnitudes(scene, resolve_contact_sensor_name(scene))

    if hasattr(apple_state, "detach"):
        import torch

        origin = origins[0] if origins is not None else apple_state.new_zeros(3)
        apple_position = apple_state[0, :3] - origin
        plate_position = plate_state[0, :3] - origin
        apple_speed = torch.linalg.vector_norm(apple_state[0, 7:10])
        link_positions = poses[0, list(link_indices), :3] - origin
        link_distances = torch.linalg.vector_norm(link_positions - apple_position, dim=-1)
        contact_force = contact_magnitudes.max() if contact_magnitudes is not None else apple_position.new_zeros(())
        values = torch.cat(
            (apple_position, apple_speed.reshape(1), plate_position, contact_force.reshape(1), link_distances)
        )
        apple_x, apple_y, apple_z, speed, plate_x, plate_y, plate_z, force, left_distance, right_distance = (
            values.detach().cpu().tolist()
        )
    else:
        origin = origins[0] if origins is not None else (0.0, 0.0, 0.0)
        apple_position = apple_state[0, :3] - origin
        plate_position = plate_state[0, :3] - origin
        speed = math.sqrt(sum(float(value) ** 2 for value in apple_state[0, 7:10]))
        distances = []
        for index in link_indices:
            link_position = poses[0, index, :3] - origin
            distances.append(
                math.sqrt(sum((float(link_position[axis]) - float(apple_position[axis])) ** 2 for axis in range(3)))
            )
        apple_x, apple_y, apple_z = (float(value) for value in apple_position)
        plate_x, plate_y, plate_z = (float(value) for value in plate_position)
        force = 0.0 if contact_magnitudes is None else float(contact_magnitudes.max())
        left_distance, right_distance = distances

    return (
        (apple_x, apple_y),
        apple_z,
        speed,
        (plate_x, plate_y),
        plate_z,
        force,
        (left_distance, right_distance),
    )


def evaluate(env: Any, criteria: PlacementCriteria = DEFAULT_CRITERIA) -> Outcome:
    """Classify the current episode from a live env (single recording environment).

    Reads the apple and plate root states plus the apple<->plate contact force, all
    for the first (recording) env, and delegates to :func:`classify`.
    """

    scene = env.scene
    support_surface_z_m = getattr(getattr(env, "cfg", None), "e2e_support_surface_z_m", None)
    release_clearance_m = getattr(getattr(env, "cfg", None), "e2e_release_clearance_m", None)
    overrides = {}
    # A randomized table reports where it actually stands; a fixed one reports no offset.
    support_offset_m = support_height_offset_value(env)
    if support_surface_z_m is not None or support_offset_m:
        nominal_support_z_m = (
            criteria.support_surface_z_m if support_surface_z_m is None else float(support_surface_z_m)
        )
        overrides["support_surface_z_m"] = nominal_support_z_m + support_offset_m
    if release_clearance_m is not None:
        overrides["release_clearance_m"] = float(release_clearance_m)
    if overrides:
        criteria = replace(criteria, **overrides)
    return classify(*_measurements(scene, _configured_task_link_names(env)), criteria=criteria)


def check_readable(env: Any) -> None:
    """Raise unless ``env`` exposes everything :func:`evaluate` reads.

    Every way this detector can be wrong is quiet by design: the recorder logs a read
    error and keeps recording, and a sensor watching nothing reads as no contact. Either
    one reaches an operator as a session in which no episode ever succeeds, long after
    the scene was composed. Called once before recording starts, so a mis-wired scene
    surfaces where it was built.

    Raises:
        KeyError: The scene names the apple, the plate, or their contact sensor as
            neither runtime does, or its configured support is absent.
        AttributeError: The configured support does not expose its root pose or nominal height.
        RuntimeError: The contact sensor watches no contact pairs.
    """

    scene = env.scene
    support_height_offset_value(env)
    _measurements(scene, _configured_task_link_names(env))
    sensor_name = resolve_contact_sensor_name(scene)
    if _contact_magnitudes(scene, sensor_name) is None:
        raise RuntimeError(
            f"Contact sensor {sensor_name!r} reports no contact pairs, so the apple would never "
            "register as placed. Check the sensor's prim path and its filter prim path."
        )

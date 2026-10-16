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

"""Versioned lane and provenance contract for static-apple HDF5 datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import h5py

from e2e_robot_workflows.data.artifact_paths import file_sha256
from e2e_robot_workflows.environments import STATIC_APPLE_ENVIRONMENT
from e2e_robot_workflows.tasks.static_apple.config import H2_STATIC_APPLE_VARIANT, load_static_apple_environment
from e2e_robot_workflows.tasks.static_apple.table_height import HEIGHT_TOLERANCE_M, validate_scene_randomization

MANIFEST_ATTRIBUTE = "e2e_workflow_manifest"
MANIFEST_VERSION = 1
SCHEMA_ID = "e2e.static-apple.dataset.v1"
# An optional environment field makes older readers reject unfamiliar scene state.
SCENE_RANDOMIZATION_FIELD = "scene_randomization"

RecordingLane = Literal["native", "arena"]
DatasetProducer = Literal["recording", "autodata"]
DatasetStage = Literal["recording", "annotation", "generation"]
ManipulationHand = Literal["left", "right"]


class DatasetContractError(ValueError):
    """Raised when a dataset cannot satisfy the declared workflow contract."""


class ArenaEpisodeRejection(DatasetContractError):
    """Well-formed episode signals do not describe an admissible source demonstration.

    Live recording may discard this episode and continue. Missing or malformed
    recorder fields still raise DatasetContractError, not this recoverable type.
    """


def _validated_pickup_signal(value: Any, *, path: str) -> Any:
    """Return a pickup signal as boolean after rejecting ambiguous samples."""

    import numpy as np

    signal = np.asarray(value)
    is_boolean = signal.dtype == np.dtype(np.bool_)
    is_binary_number = signal.dtype.kind in {"i", "u", "f"} and bool(
        np.isfinite(signal).all() and np.logical_or(signal == 0, signal == 1).all()
    )
    if not is_boolean and not is_binary_number:
        raise DatasetContractError(f"{path} must contain only boolean or 0/1 pickup samples")
    return signal.astype(bool, copy=False)


def validate_arena_episode_boundaries(
    signals: Mapping[str, Any],
    *,
    steps: int,
    manipulation_hand: ManipulationHand | None = None,
    require_handover: bool = True,
    episode_name: str = "pending",
    grasp_signals: Mapping[str, Any] | None = None,
) -> ManipulationHand:
    """Check the same pickup/handover boundaries for file and live array inputs.

    ``signals`` maps ``left`` and ``right`` to one-dimensional boolean or binary
    arrays. Live callers need stack only these two small recorder fields, not
    camera frames or the entire episode. The returned first-pickup hand can be
    latched after admission when the session's manipulation hand is automatic.
    """

    import numpy as np

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise DatasetContractError("Arena episode sample count must be a non-negative integer")
    if manipulation_hand not in (None, "left", "right"):
        raise DatasetContractError(f"unsupported Arena manipulation hand: {manipulation_hand!r}")
    first_boundaries: dict[ManipulationHand, int] = {}
    for hand in ("left", "right"):
        path = f"data/{episode_name}/obs/datagen_info/subtask_term_signals/{hand}_hand_holds_object"
        if hand not in signals:
            raise DatasetContractError(f"{path} is missing")
        signal = np.asarray(signals[hand])
        if signal.shape != (steps,):
            raise DatasetContractError(f"{path} must have shape ({steps},), got {signal.shape}")
        signal = _validated_pickup_signal(signal, path=path)
        boundaries = np.flatnonzero(~signal[:-1] & signal[1:])
        if boundaries.size:
            first_boundaries[hand] = int(boundaries[0]) + 1

    if not first_boundaries:
        raise ArenaEpisodeRejection(
            f"data/{episode_name} must contain one unambiguous Arena pickup boundary; found none"
        )
    first_step = min(first_boundaries.values())
    detected = [hand for hand, step in first_boundaries.items() if step == first_step]
    if len(detected) != 1:
        raise ArenaEpisodeRejection(
            f"data/{episode_name} must contain one unambiguous Arena pickup boundary; "
            f"found {detected} at frame {first_step}"
        )
    selected_hand = detected[0]
    if manipulation_hand is not None and selected_hand != manipulation_hand:
        raise ArenaEpisodeRejection(
            f"Arena recording's first pickup boundary belongs to {selected_hand}, "
            f"not manifest-selected {manipulation_hand} (data/{episode_name})"
        )
    if require_handover:
        receiving_hand = "right" if selected_hand == "left" else "left"
        if receiving_hand not in first_boundaries:
            raise ArenaEpisodeRejection(
                f"data/{episode_name} has no false-to-true {receiving_hand}-hand handover boundary"
            )
        if first_boundaries[receiving_hand] + 1 >= steps:
            raise ArenaEpisodeRejection(
                f"data/{episode_name} {receiving_hand}-hand handover boundary leaves no placement samples"
            )
    if require_handover and grasp_signals is not None:
        raw = {}
        for hand in ("left", "right"):
            path = f"data/{episode_name}/obs/datagen_info/grasp_signals/{hand}_hand_holds_object"
            if hand not in grasp_signals or np.shape(grasp_signals[hand]) != (steps,):
                raise DatasetContractError(f"{path} must have shape ({steps},)")
            raw[hand] = _validated_pickup_signal(grasp_signals[hand], path=path)
            if not np.array_equal(np.logical_or.accumulate(raw[hand]), np.asarray(signals[hand], dtype=bool)):
                raise DatasetContractError(f"{path} disagrees with latched subtask signals")
        receiver_step = first_boundaries[receiving_hand]
        # The giver must retain measured grasp through the receiver's first
        # grasp sample. Later release for placement is expected and allowed.
        if not raw[selected_hand][first_step : receiver_step + 1].all():
            raise ArenaEpisodeRejection(f"data/{episode_name} loses grasp before continuous handover")
    return selected_hand


@dataclass(frozen=True)
class ActionProfile:
    """One supported raw HDF5 action schema."""

    lane: RecordingLane
    layout: str
    width: int
    environment_runtime: str
    controller: str
    autodata_compatible: bool


NATIVE_FIXED_PROFILE = ActionProfile(
    lane="native",
    layout="native_g1_pink_fixed_28d_v1",
    width=28,
    environment_runtime="isaaclab",
    controller="pink_fixed",
    autodata_compatible=True,
)
NATIVE_AGILE_PROFILE = ActionProfile(
    lane="native",
    layout="native_g1_pink_agile_32d_v1",
    width=32,
    environment_runtime="isaaclab",
    controller="pink_agile",
    autodata_compatible=True,
)
ARENA_PINK_PROFILE = ActionProfile(
    lane="arena",
    layout="arena_g1_wbc_agile_pink_23d_v1",
    width=23,
    environment_runtime="arena",
    controller="g1_wbc_agile_pink",
    autodata_compatible=True,
)
ARENA_JOINT_PROFILE = ActionProfile(
    lane="arena",
    layout="arena_g1_wbc_agile_joint_50d_v1",
    width=50,
    environment_runtime="arena",
    controller="g1_wbc_agile_joint",
    autodata_compatible=False,
)
H2_ARENA_PINK_PROFILE = ActionProfile(
    lane="arena",
    layout="arena_h2_agile_pink_61d_v1",
    width=61,
    environment_runtime="arena",
    controller="h2_agile_pink",
    autodata_compatible=True,
)
H2_ARENA_JOINT_PROFILE = ActionProfile(
    lane="arena",
    layout="arena_h2_agile_joint_61d_v1",
    width=61,
    environment_runtime="arena",
    controller="h2_agile_joint",
    autodata_compatible=False,
)
ACTION_PROFILES = {
    profile.layout: profile
    for profile in (
        NATIVE_FIXED_PROFILE,
        NATIVE_AGILE_PROFILE,
        ARENA_PINK_PROFILE,
        ARENA_JOINT_PROFILE,
        H2_ARENA_PINK_PROFILE,
        H2_ARENA_JOINT_PROFILE,
    )
}
_ARENA_PINK_POSE_OFFSETS = {
    ARENA_PINK_PROFILE.layout: (("left", 2, 5), ("right", 9, 12)),
    H2_ARENA_PINK_PROFILE.layout: (("left", 0, 3), ("right", 7, 10)),
}
_H2_CAMERA_FRAME_SHAPE = (480, 640, 3)
_H2_TIMING_KEYS = ("physics_hz", "control_hz", "render_hz", "camera_hz")
_LEGACY_H2_PINK_CONTROL_HZ = 100


def _snapshot_sha256(values: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(values), sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _is_sha256(value: Any, *, prefixed: bool = False) -> bool:
    prefix = "sha256:" if prefixed else ""
    return isinstance(value, str) and re.fullmatch(rf"{prefix}[0-9a-f]{{64}}", value) is not None


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)


def _is_finite_xy(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_is_finite_number(item) for item in value)


def _resolve_episode_dataset(episode: h5py.Group, name: str) -> h5py.Dataset | None:
    """Resolve one recorder field without accepting ambiguous suffix matches."""

    for candidate in (name, f"obs/{name}", f"camera_obs/{name}", f"obs/camera_obs/{name}"):
        value = episode.get(candidate)
        if isinstance(value, h5py.Dataset):
            return value
    matches: list[h5py.Dataset] = []

    def collect(path: str, value: Any) -> None:
        if isinstance(value, h5py.Dataset) and path.endswith(f"/{name}"):
            matches.append(value)

    episode.visititems(collect)
    return matches[0] if len(matches) == 1 else None


def _episode_rigid_state(episode: h5py.Group, path: str, shape: tuple[int, int]) -> Any:
    """Read an aligned, finite rigid pose or velocity; poses must have unit XYZW quaternions."""

    import numpy as np

    value = episode.get(path)
    if not isinstance(value, h5py.Dataset) or value.shape != shape:
        raise DatasetContractError(
            f"{episode.name}/{path} must have shape {shape}, got {getattr(value, 'shape', None)}"
        )
    values = np.asarray(value)
    if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise DatasetContractError(f"{episode.name}/{path} must contain finite numbers")
    if shape[1] == 7 and not np.allclose(np.linalg.norm(values[:, 3:], axis=1), 1.0, atol=1e-4, rtol=0):
        raise DatasetContractError(f"{episode.name}/{path} quaternions must be normalized XYZW")
    return values


def _validate_scene_randomization_episode(
    episode: h5py.Group, randomization: Mapping[str, Any], definition: Any, steps: int, *, settled_objects: bool = False
) -> None:
    """Require a stationary, Z-translated table and compatible initial object heights."""

    import numpy as np

    support = randomization["support_asset"]
    low, high = randomization["table_height_offset_range_m"]
    initial = _episode_rigid_state(episode, f"initial_state/rigid_object/{support}/root_pose", (1, 7))
    offset = float(initial[0, 2]) - randomization["nominal_support_root_z_m"]
    if not low - HEIGHT_TOLERANCE_M <= offset <= high + HEIGHT_TOLERANCE_M:
        raise DatasetContractError(
            f"{episode.name} table offset {offset:+.4f} m is outside the declared [{low}, {high}]"
        )
    position = np.asarray(definition.support_position) + [0.0, 0.0, offset]
    rotation = np.asarray(definition.asset("support").rotation_xyzw)
    for prefix, count in (("initial_state", 1), ("states", steps)):
        path = f"{prefix}/rigid_object/{support}"
        poses = initial if prefix == "initial_state" else _episode_rigid_state(episode, f"{path}/root_pose", (count, 7))
        velocities = _episode_rigid_state(episode, f"{path}/root_velocity", (count, 6))
        # q and -q describe the same rotation.
        rotation_error = np.minimum(
            np.max(np.abs(poses[:, 3:] - rotation), axis=1), np.max(np.abs(poses[:, 3:] + rotation), axis=1)
        )
        if not np.allclose(poses[:, :3], position, atol=HEIGHT_TOLERANCE_M, rtol=0) or np.any(rotation_error > 1e-4):
            raise DatasetContractError(f"{episode.name}/{path} table pose must stay at its Z-only reset transform")
        if np.any(np.abs(velocities) > 1e-4):
            raise DatasetContractError(f"{episode.name}/{path} table velocity must stay zero")
    for name, nominal_z in randomization["object_spawn_z_m"].items():
        pose = _episode_rigid_state(episode, f"initial_state/rigid_object/{name}/root_pose", (1, 7))
        spawn_z = nominal_z + offset
        # H2 records after its ready hold, during which dynamic objects can settle.
        minimum_z = (
            definition.support_surface_z - definition.min_supported_height_below_support_m + offset
            if settled_objects
            else spawn_z
        )
        if not minimum_z - HEIGHT_TOLERANCE_M <= float(pose[0, 2]) <= spawn_z + HEIGHT_TOLERANCE_M:
            raise DatasetContractError(f"{episode.name} {name} initial height disagrees with the table")


def _validate_h2_episode_fields(
    episode_name: str,
    episode: h5py.Group,
    actions: h5py.Dataset,
    profile: ActionProfile,
    stage: DatasetStage,
) -> None:
    """Validate the complete recorded H2 state, command, and camera contract."""

    import numpy as np

    from e2e_robot_workflows.robots.h2 import (
        H2_LIVE_JOINT_COUNT,
        H2_NAVIGATE_COMMAND_DIM,
        H2_PINK_ACTION_DIM,
        load_h2_pink_contract,
    )

    steps = actions.shape[0]
    action_values = np.asarray(actions)
    if not np.isfinite(action_values).all():
        raise DatasetContractError(f"data/{episode_name}/actions contains non-finite values")

    contract = load_h2_pink_contract()
    if profile == H2_ARENA_PINK_PROFILE:
        for hand, quaternion_start in (("left", 3), ("right", 10)):
            norms = np.linalg.norm(action_values[:, quaternion_start : quaternion_start + 4], axis=1)
            if not np.allclose(norms, 1.0, atol=1e-4, rtol=0.0):
                raise DatasetContractError(f"data/{episode_name} {hand} action quaternions are not normalized XYZW")

        hand_values = action_values[:, 14:H2_PINK_ACTION_DIM]
        for index, joint_name in enumerate(contract.hand_joint_order):
            open_target = contract.hand_open_targets[joint_name]
            closed_target = contract.hand_closed_targets[joint_name]
            lower, upper = sorted((open_target, closed_target))
            if np.any(hand_values[:, index] < lower - 1e-4) or np.any(hand_values[:, index] > upper + 1e-4):
                raise DatasetContractError(
                    f"data/{episode_name}/actions hand target for {joint_name!r} is outside [{lower}, {upper}]"
                )
        if stage == "recording":
            open_targets = np.asarray(
                [contract.hand_open_targets[name] for name in contract.hand_joint_order], dtype=hand_values.dtype
            )
            closed_targets = np.asarray(
                [contract.hand_closed_targets[name] for name in contract.hand_joint_order], dtype=hand_values.dtype
            )
            for hand, indices in (
                ("left", contract.left_hand_indices),
                ("right", contract.right_hand_indices),
            ):
                values = hand_values[:, indices]
                open_rows = np.isclose(values, open_targets[list(indices)], atol=1e-4, rtol=0.0).all(axis=1)
                closed_rows = np.isclose(values, closed_targets[list(indices)], atol=1e-4, rtol=0.0).all(axis=1)
                if not np.all(open_rows | closed_rows):
                    raise DatasetContractError(
                        f"data/{episode_name}/actions recorded {hand} hand must match the complete open "
                        "or closed synergy"
                    )

    expected_fields = {
        "robot_joint_pos": (steps, H2_LIVE_JOINT_COUNT),
        "joint_pos_target": (steps, H2_LIVE_JOINT_COUNT),
        "navigate_cmd": (steps, H2_NAVIGATE_COMMAND_DIM),
        "robot_head_cam_rgb": (steps, *_H2_CAMERA_FRAME_SHAPE),
    }
    resolved: dict[str, h5py.Dataset] = {}
    for name, expected_shape in expected_fields.items():
        value = _resolve_episode_dataset(episode, name)
        if value is None or value.shape != expected_shape:
            shape = getattr(value, "shape", None)
            raise DatasetContractError(f"data/{episode_name}/{name} must have shape {expected_shape}, got {shape}")
        if name == "robot_head_cam_rgb":
            if value.dtype != np.dtype(np.uint8):
                raise DatasetContractError(f"data/{episode_name}/{name} must contain raw uint8 RGB frames")
        elif not np.isfinite(value[...]).all():
            raise DatasetContractError(f"data/{episode_name}/{name} contains non-finite values")
        resolved[name] = value

    navigate = np.asarray(resolved["navigate_cmd"])
    if not np.allclose(navigate, action_values[:, -H2_NAVIGATE_COMMAND_DIM:], atol=1e-6, rtol=0.0):
        raise DatasetContractError(f"data/{episode_name}/navigate_cmd disagrees with the H2 action tail")
    for index, (lower, upper) in enumerate(_repository_h2_robot_config().navigate_command_limits):
        if np.any(navigate[:, index] < lower - 1e-6) or np.any(navigate[:, index] > upper + 1e-6):
            raise DatasetContractError(
                f"data/{episode_name}/navigate_cmd[{index}] is outside the configured [{lower}, {upper}] envelope"
            )

    joint_targets = np.asarray(resolved["joint_pos_target"])
    for joint_name, (lower, upper) in contract.learned_joint_position_limits:
        joint_index = contract.expected_joint_order.index(joint_name)
        values = joint_targets[:, joint_index]
        if np.any(values < lower - 1e-5) or np.any(values > upper + 1e-5):
            raise DatasetContractError(
                f"data/{episode_name}/joint_pos_target for {joint_name!r} is outside the pinned "
                f"[{lower}, {upper}] position limits"
            )
    if profile == H2_ARENA_PINK_PROFILE:
        hand_target_indices = [contract.expected_joint_order.index(name) for name in contract.hand_joint_order]
        recorded_hand_targets = joint_targets[:, hand_target_indices]
        if not np.allclose(recorded_hand_targets, hand_values, atol=1e-5, rtol=0.0):
            raise DatasetContractError(
                f"data/{episode_name}/joint_pos_target disagrees with the H2 Pink hand action targets"
            )
    elif profile == H2_ARENA_JOINT_PROFILE:
        target_names = contract.arm_joint_names + contract.hand_joint_order
        target_indices = [contract.expected_joint_order.index(name) for name in target_names]
        recorded_targets = joint_targets[:, target_indices]
        if not np.allclose(recorded_targets, action_values[:, :H2_PINK_ACTION_DIM], atol=1e-5, rtol=0.0):
            raise DatasetContractError(
                f"data/{episode_name}/joint_pos_target disagrees with the H2 joint action targets"
            )


def resolve_recording_lane(requested_lane: str | None, *, arena_alias: bool) -> RecordingLane:
    """Resolve the additive lane selector while preserving the existing defaults."""

    if requested_lane not in {None, "native", "arena"}:
        raise DatasetContractError(f"unsupported recording lane {requested_lane!r}")
    if arena_alias and requested_lane == "native":
        raise DatasetContractError("--arena conflicts with --recording-lane native")
    if arena_alias or requested_lane == "arena":
        return "arena"
    return "native"


def default_recording_dataset_file(lane: RecordingLane, *, timestamp: datetime | None = None) -> str:
    """Return the lane-qualified default path for one recording session."""

    if lane not in {"native", "arena"}:
        raise DatasetContractError(f"unsupported recording lane {lane!r}")
    recorded_at = timestamp or datetime.now()
    return f"./datasets/e2e_ra_dataset_{lane}_{recorded_at:%Y%m%d_%H%M%S}.hdf5"


def detect_arena_manipulation_hand(dataset_path: str | Path) -> ManipulationHand:
    """Infer the Arena pickup hand from each episode's first direct boundary.

    A bimanual handover can legitimately produce a later boundary for the
    receiving hand. The manipulation hand is the hand that picked the object
    up first; simultaneous first boundaries remain ambiguous and fail closed.
    """

    import numpy as np

    episodes_by_hand: dict[ManipulationHand, list[str]] = {"left": [], "right": []}
    with h5py.File(dataset_path, "r") as dataset:
        episodes = dataset.get("data")
        if not isinstance(episodes, h5py.Group) or len(episodes) == 0:
            raise DatasetContractError("Arena recording contains no episodes")
        for episode_name, episode in episodes.items():
            signals: dict[str, Any] = {}
            for hand in ("left", "right"):
                path = f"obs/datagen_info/subtask_term_signals/{hand}_hand_holds_object"
                signal_dataset = episode.get(path) if isinstance(episode, h5py.Group) else None
                if not isinstance(signal_dataset, h5py.Dataset) or signal_dataset.ndim != 1:
                    raise DatasetContractError(f"data/{episode_name}/{path} must be a one-dimensional dataset")
                signals[hand] = np.asarray(signal_dataset)
            detected = validate_arena_episode_boundaries(
                signals,
                steps=len(signals["left"]),
                require_handover=False,
                episode_name=episode_name,
            )
            episodes_by_hand[detected].append(episode_name)

    selected = [hand for hand, episodes in episodes_by_hand.items() if episodes]
    if len(selected) != 1:
        summary = ", ".join(f"{hand}: {', '.join(episodes_by_hand[hand])}" for hand in selected)
        raise DatasetContractError(f"Arena episodes use different manipulation hands ({summary})")
    return selected[0]


def resolve_recording_profile(
    lane: RecordingLane,
    *,
    wbc_mode: str | None = None,
    arena_embodiment: str | None = None,
) -> ActionProfile:
    """Return the explicit raw-action profile for a recording run."""

    if lane == "native":
        if arena_embodiment is not None:
            raise DatasetContractError("native recording cannot declare an Arena embodiment")
        if wbc_mode == "fixed":
            return NATIVE_FIXED_PROFILE
        if wbc_mode == "agile":
            return NATIVE_AGILE_PROFILE
        raise DatasetContractError(f"native recording requires wbc_mode fixed or agile, got {wbc_mode!r}")
    if wbc_mode is not None:
        raise DatasetContractError("Arena recording cannot declare a native wbc_mode")
    if arena_embodiment in {None, ARENA_PINK_PROFILE.controller}:
        return ARENA_PINK_PROFILE
    if arena_embodiment == ARENA_JOINT_PROFILE.controller:
        return ARENA_JOINT_PROFILE
    if arena_embodiment == H2_ARENA_PINK_PROFILE.controller:
        return H2_ARENA_PINK_PROFILE
    if arena_embodiment == H2_ARENA_JOINT_PROFILE.controller:
        return H2_ARENA_JOINT_PROFILE
    raise DatasetContractError(f"unsupported Arena recording embodiment {arena_embodiment!r}")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _repository_h2_robot_config(repo_root: str | Path | None = None) -> Any:
    from e2e_robot_workflows.robots.h2 import load_h2_robot_config

    root = Path(repo_root) if repo_root is not None else _repository_root()
    return load_h2_robot_config(str(root / "configs" / "robots" / "h2.yaml"))


def _repository_h2_environment(repo_root: str | Path | None = None) -> Any:
    root = Path(repo_root) if repo_root is not None else _repository_root()
    return load_static_apple_environment(
        H2_STATIC_APPLE_VARIANT,
        path=root / "configs" / "tasks" / "static_apple.yaml",
    )


def _repository_runtime(repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else _repository_root()
    with (root / "runtime.toml").open("rb") as stream:
        return tomllib.load(stream)["tested"]


def _task_manifest(
    profile: ActionProfile,
    manipulation_hand: ManipulationHand,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    task: dict[str, Any] = {"manipulation_hand": manipulation_hand}
    if profile != H2_ARENA_PINK_PROFILE:
        return task

    root = Path(repo_root) if repo_root is not None else _repository_root()
    descriptor_name = (
        "static_apple_arena_h2_left_task.yaml" if manipulation_hand == "left" else "static_apple_arena_h2_task.yaml"
    )
    descriptor_path = root / "configs" / "stages" / "autodata" / descriptor_name
    robot = _repository_h2_robot_config(root)
    definition = _repository_h2_environment(root)
    success_values = {
        "function": "e2e_robot_workflows.tasks.static_apple.terminations:object_placed_on_target",
        "implementation_sha256": file_sha256(root / "src/e2e_robot_workflows/tasks/static_apple/terminations.py"),
        "version": 5,
        "hold_frames": definition.success_hold_frames,
        "parameters": {
            "placement_radius_m": definition.success_placement_xy_m,
            "placement_z_m": definition.success_placement_z_m,
            "min_object_height_above_target_m": definition.min_object_height_above_target_m,
            "max_object_height_m": definition.support_surface_z + definition.max_object_height_above_support_m,
            "min_object_height_m": definition.support_surface_z - definition.min_supported_height_below_support_m,
            "min_target_height_m": definition.support_surface_z - definition.min_supported_height_below_support_m,
            "min_contact_force": definition.success_force_threshold,
            "release_clearance_m": definition.release_clearance_m,
            "max_object_speed": definition.success_velocity_threshold,
            "task_link_names": list(robot.task_link_names),
        },
    }
    success_digest = hashlib.sha256(
        json.dumps(success_values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    from e2e_robot_workflows.tasks.static_apple.grasp_contract import H2_GRASP_CRITERIA, H2_GRASP_MODE

    grasp_criteria = asdict(H2_GRASP_CRITERIA)
    grasp_criteria["primary_flexion_fingers"] = list(H2_GRASP_CRITERIA.primary_flexion_fingers)
    pickup_values = {
        "function": "e2e_robot_workflows.tasks.static_apple.h2_grasp:object_held_by_h2_hand",
        "implementation_sha256": file_sha256(root / "src/e2e_robot_workflows/tasks/static_apple/h2_grasp.py"),
        "version": 3,
        "parameters": {
            "mode": H2_GRASP_MODE,
            "criteria_implementation_sha256": file_sha256(
                root / "src/e2e_robot_workflows/tasks/static_apple/grasp_contract.py"
            ),
            "task_link_names": list(robot.task_link_names),
            "grasp_distance_m": definition.release_clearance_m,
            "lift_clearance_m": definition.pickup_lift_clearance_m,
            "held_object_qualification": grasp_criteria,
            "contact_update_period_s": 1.0 / robot.control_hz,
        },
    }
    pickup_digest = hashlib.sha256(
        json.dumps(pickup_values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    task["autodata_descriptor"] = {
        "path": f"configs/stages/autodata/{descriptor_name}",
        "sha256": file_sha256(descriptor_path),
    }
    task["pickup_contract"] = {**pickup_values, "sha256": pickup_digest}
    task["success_contract"] = {**success_values, "sha256": success_digest}
    return task


def _environment_definition(profile: ActionProfile, repo_root: str | Path | None = None) -> Any:
    return _repository_h2_environment(repo_root) if profile.controller.startswith("h2_") else STATIC_APPLE_ENVIRONMENT


def _environment_manifest(profile: ActionProfile, repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else _repository_root()
    definition = _environment_definition(profile, root)
    environment: dict[str, Any] = {
        "id": definition.environment_id,
        "digest": definition.contract_digest,
    }
    if profile.controller.startswith("h2_"):
        from e2e_robot_workflows.robots.h2 import load_h2_pink_contract

        robot = _repository_h2_robot_config(root)
        plant = load_h2_pink_contract(root / "configs" / "robots" / "h2_pink.yaml")
        values = {
            "support_surface_z_m": definition.support_surface_z,
            "pick_spawn_xy": list(definition.pick_spawn_xy),
            "destination_spawn_xy": list(definition.destination_spawn_xy),
            "release_clearance_m": definition.release_clearance_m,
            "pickup_lift_clearance_m": definition.pickup_lift_clearance_m,
        }
        environment["variant"] = H2_STATIC_APPLE_VARIANT
        environment["variant_digest"] = _snapshot_sha256(values)
        environment["variant_values"] = values
        timing = {
            "physics_hz": definition.physics_hz,
            "control_hz": robot.control_hz,
            "render_hz": robot.render_hz,
            "camera_hz": robot.camera_hz,
        }
        environment["timing"] = {**timing, "sha256": _snapshot_sha256(timing)}
        environment["robot"] = {
            "source_url": plant.simulation_usd.url,
            "source_usd_sha256": plant.simulation_usd.sha256,
            "source_contract_sha256": plant.source_contract_sha256,
            "variants": plant.simulation_usd.variant_selection,
            "arena_usd_sha256": plant.arena_usd_sha256,
            "pink_urdf_sha256": plant.urdf_sha256,
            "agile_policy": {
                "repository": robot.policy_repository,
                "revision": robot.policy_revision,
                "filename": robot.policy_filename,
                "sha256": robot.policy_sha256,
            },
        }
    return environment


def _split_scene_randomization(environment: Any, *, label: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Split an environment snapshot into its nominal contract and randomization record."""

    if not isinstance(environment, Mapping):
        raise DatasetContractError(f"manifest {label} environment snapshot is malformed")
    nominal = dict(environment)
    block = nominal.pop(SCENE_RANDOMIZATION_FIELD, None)
    return nominal, block


def _scene_randomization(
    manifest: Mapping[str, Any], profile: ActionProfile, repo_root: str | Path | None = None
) -> dict[str, Any] | None:
    environment = manifest.get("environment")
    if not isinstance(environment, Mapping) or SCENE_RANDOMIZATION_FIELD not in environment:
        return None
    if profile.lane != "arena":
        raise DatasetContractError("scene randomization requires the Arena lane")
    definition = _environment_definition(profile, repo_root)
    if environment.get("digest") != definition.contract_digest:
        raise DatasetContractError("scene randomization requires the current environment geometry")
    try:
        return validate_scene_randomization(
            environment[SCENE_RANDOMIZATION_FIELD], definition=definition, seed=manifest.get("seed")
        )
    except ValueError as error:
        raise DatasetContractError(f"manifest scene randomization: {error}") from error


def _repository_state(repo_root: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"revision": revision, "dirty": dirty}


def build_manifest(
    *,
    producer: DatasetProducer,
    profile: ActionProfile,
    source_dataset: str | Path | None = None,
    seed: int | None = None,
    stage: DatasetStage | None = None,
    manipulation_hand: ManipulationHand = "right",
    scene_randomization: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a manifest with this run's seed/distribution and optional source lineage."""

    if producer not in {"recording", "autodata"}:
        raise DatasetContractError(f"unsupported producer {producer!r}")
    resolved_stage: DatasetStage = stage or ("recording" if producer == "recording" else "generation")
    if resolved_stage not in {"recording", "annotation", "generation"}:
        raise DatasetContractError(f"unsupported dataset stage {resolved_stage!r}")
    if manipulation_hand not in {"left", "right"}:
        raise DatasetContractError(f"unsupported manipulation hand {manipulation_hand!r}")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise DatasetContractError("seed must be a non-negative integer or null")
    if producer == "recording" and resolved_stage != "recording":
        raise DatasetContractError("the recording producer requires the recording stage")
    if producer == "autodata" and resolved_stage == "recording":
        raise DatasetContractError("the AutoData producer cannot claim the recording stage")
    if resolved_stage == "recording" and source_dataset is not None:
        raise DatasetContractError("recording data cannot claim source lineage")
    if resolved_stage != "recording" and source_dataset is None:
        raise DatasetContractError(f"{resolved_stage} output requires source lineage")
    if producer == "autodata" and not profile.autodata_compatible:
        raise DatasetContractError(f"action layout {profile.layout!r} is not AutoData-compatible")

    root = Path(repo_root) if repo_root is not None else _repository_root()
    runtime = _repository_runtime(root)

    manifest: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "manifest_version": MANIFEST_VERSION,
        "lane": profile.lane,
        "producer": producer,
        "stage": resolved_stage,
        "seed": seed,
        "action": asdict(profile),
        "task": _task_manifest(profile, manipulation_hand, root),
        "environment": _environment_manifest(profile, root),
        "repository": _repository_state(root),
        "runtime": runtime,
        "provenance": {"status": "recorded" if producer == "recording" else "derived"},
        "source": None,
    }
    if scene_randomization is not None:
        manifest["environment"][SCENE_RANDOMIZATION_FIELD] = scene_randomization
        manifest["environment"][SCENE_RANDOMIZATION_FIELD] = _scene_randomization(manifest, profile, root)
    if profile.controller.startswith("h2_"):
        manifest["runtime_digest"] = _snapshot_sha256(runtime)
    if source_dataset is not None:
        source_path = Path(source_dataset)
        source_manifest = validate_dataset(source_path)
        if source_manifest["lane"] != profile.lane or source_manifest["action"]["layout"] != profile.layout:
            raise DatasetContractError("generated profile does not match its source dataset")
        required_source_stage = (
            "recording" if resolved_stage == "annotation" or profile.lane == "arena" else "annotation"
        )
        if source_manifest["stage"] != required_source_stage:
            article = "an" if required_source_stage == "annotation" else "a"
            raise DatasetContractError(
                f"{resolved_stage} output requires {article} {required_source_stage} source, got "
                f"{source_manifest['stage']}"
            )
        if source_manifest["task"]["manipulation_hand"] != manipulation_hand:
            raise DatasetContractError("generated manipulation hand does not match its source dataset")
        manifest["source"] = {
            "name": source_path.name,
            "sha256": file_sha256(source_path),
            "schema_id": source_manifest["schema_id"],
            "stage": source_manifest["stage"],
            "repository": source_manifest["repository"],
            "provenance": source_manifest["provenance"],
            "source": source_manifest["source"],
        }
    return manifest


def write_manifest(dataset_path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Embed a manifest in an existing HDF5 file."""

    path = Path(dataset_path)
    with h5py.File(path, "r+") as dataset:
        dataset.attrs[MANIFEST_ATTRIBUTE] = json.dumps(dict(manifest), sort_keys=True, separators=(",", ":"))


def read_manifest(dataset_path: str | Path, *, conversion: bool = False) -> dict[str, Any]:
    """Read the manifest, admitting historical G1 snapshots only for conversion."""

    path = Path(dataset_path)
    try:
        with h5py.File(path, "r") as dataset:
            encoded = dataset.attrs.get(MANIFEST_ATTRIBUTE)
    except OSError as error:
        raise DatasetContractError(f"cannot open {path}: {error}") from error
    if encoded is None:
        raise DatasetContractError(f"{path} has no {MANIFEST_ATTRIBUTE!r}; migrate it with an explicit lane")
    try:
        manifest = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as error:
        raise DatasetContractError(f"{path} has an invalid {MANIFEST_ATTRIBUTE!r}") from error
    if not isinstance(manifest, dict):
        raise DatasetContractError("dataset manifest must be a JSON object")
    _validate_manifest(manifest, conversion=conversion)
    return manifest


def _validate_embedded_contract(value: Any, *, label: str, hold_frames: bool = False) -> None:
    fields = {"function", "implementation_sha256", "version", "parameters", "sha256"}
    if hold_frames:
        fields.add("hold_frames")
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or not isinstance(value.get("function"), str)
        or not value["function"]
        or isinstance(value.get("version"), bool)
        or not isinstance(value.get("version"), int)
        or value["version"] <= 0
        or not isinstance(value.get("parameters"), Mapping)
        or not _is_sha256(value.get("sha256"))
    ):
        raise DatasetContractError(f"manifest {label} is malformed")
    if hold_frames and (
        isinstance(value.get("hold_frames"), bool)
        or not isinstance(value.get("hold_frames"), int)
        or value["hold_frames"] <= 0
    ):
        raise DatasetContractError(f"manifest {label} is malformed")
    payload = dict(value)
    claimed = payload.pop("sha256")
    actual = _snapshot_sha256(payload).removeprefix("sha256:")
    if claimed != actual:
        raise DatasetContractError(f"manifest {label} self-digest does not match its embedded values")
    if not _is_sha256(payload.get("implementation_sha256")):
        raise DatasetContractError(f"manifest {label} implementation digest is malformed")


def _validate_task_snapshot(task: Mapping[str, Any], *, label: str, allow_hand_only: bool = False) -> None:
    if allow_hand_only and set(task) == {"manipulation_hand"}:
        return
    if set(task) != {"manipulation_hand", "autodata_descriptor", "pickup_contract", "success_contract"}:
        raise DatasetContractError(f"manifest {label} task snapshot is malformed")
    descriptor = task.get("autodata_descriptor")
    if (
        not isinstance(descriptor, Mapping)
        or set(descriptor) != {"path", "sha256"}
        or not isinstance(descriptor.get("path"), str)
        or not descriptor["path"]
        or not _is_sha256(descriptor.get("sha256"))
    ):
        raise DatasetContractError(f"manifest {label} AutoData task descriptor is malformed")
    _validate_embedded_contract(task.get("pickup_contract"), label=f"{label} pickup contract")
    _validate_embedded_contract(task.get("success_contract"), label=f"{label} success contract", hold_frames=True)


def _validate_h2_snapshots(manifest: Mapping[str, Any], profile: ActionProfile) -> None:
    task = manifest["task"]
    if profile == H2_ARENA_PINK_PROFILE:
        _validate_task_snapshot(task, label="H2")
    elif set(task) != {"manipulation_hand"}:
        raise DatasetContractError("manifest H2 joint task snapshot is malformed")

    environment, _ = _split_scene_randomization(manifest.get("environment"), label="H2")
    legacy_timing = "timing" not in environment
    environment_fields = {"id", "digest", "variant", "variant_digest", "variant_values", "robot"}
    if set(environment) != (environment_fields if legacy_timing else environment_fields | {"timing"}) or (
        legacy_timing and profile != H2_ARENA_PINK_PROFILE
    ):
        raise DatasetContractError("manifest H2 environment snapshot is malformed")
    if (
        not isinstance(environment.get("id"), str)
        or not environment["id"]
        or not _is_sha256(environment.get("digest"), prefixed=True)
        or environment.get("variant") != H2_STATIC_APPLE_VARIANT
        or not isinstance(environment.get("robot"), Mapping)
    ):
        raise DatasetContractError("manifest H2 environment identity is malformed")

    variant_values = environment.get("variant_values")
    core_variant_fields = {"support_surface_z_m", "pick_spawn_xy", "destination_spawn_xy"}
    current_variant_fields = core_variant_fields | {"release_clearance_m", "pickup_lift_clearance_m"}
    if (
        not isinstance(variant_values, Mapping)
        or set(variant_values) not in (core_variant_fields, current_variant_fields)
        or not _is_finite_number(variant_values.get("support_surface_z_m"))
        or not _is_finite_xy(variant_values.get("pick_spawn_xy"))
        or not _is_finite_xy(variant_values.get("destination_spawn_xy"))
        or any(
            not _is_finite_number(variant_values.get(field)) or variant_values[field] <= 0
            for field in current_variant_fields - core_variant_fields
            if field in variant_values
        )
        or environment.get("variant_digest") != _snapshot_sha256(variant_values)
    ):
        raise DatasetContractError("manifest H2 environment variant snapshot is malformed")
    robot = environment["robot"]
    robot_fields = {
        "source_url",
        "source_usd_sha256",
        "source_contract_sha256",
        "variants",
        "arena_usd_sha256",
        "pink_urdf_sha256",
        "agile_policy",
    }
    robot_digests = ("source_usd_sha256", "source_contract_sha256", "arena_usd_sha256", "pink_urdf_sha256")
    policy = robot.get("agile_policy")
    if (
        set(robot) != robot_fields
        or not isinstance(robot.get("source_url"), str)
        or not robot["source_url"]
        or any(not _is_sha256(robot.get(field)) for field in robot_digests)
        or not isinstance(policy, Mapping)
        or set(policy) != {"repository", "revision", "filename", "sha256"}
        or not all(isinstance(policy.get(field), str) and policy[field] for field in ("repository", "filename"))
        or re.fullmatch(r"[0-9a-f]{40}", str(policy.get("revision", ""))) is None
        or not _is_sha256(policy.get("sha256"))
    ):
        raise DatasetContractError("manifest H2 robot snapshot is malformed")
    variants = robot.get("variants")
    if (
        not isinstance(variants, Mapping)
        or not variants
        or not all(isinstance(key, str) and key and isinstance(value, str) and value for key, value in variants.items())
    ):
        raise DatasetContractError("manifest H2 robot variants are malformed")

    if not legacy_timing:
        timing = environment["timing"]
        if not isinstance(timing, Mapping) or set(timing) != {*_H2_TIMING_KEYS, "sha256"}:
            raise DatasetContractError("manifest H2 timing snapshot is malformed")
        values = {key: timing.get(key) for key in _H2_TIMING_KEYS}
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values.values()):
            raise DatasetContractError("manifest H2 timing rates must be positive integers")
        if values["physics_hz"] % values["control_hz"] or values["control_hz"] % values["render_hz"]:
            raise DatasetContractError("manifest H2 physics, control, and render rates must divide exactly")
        if values["render_hz"] % values["camera_hz"]:
            raise DatasetContractError("manifest H2 camera rate must divide the render rate")
        if timing.get("sha256") != _snapshot_sha256(values):
            raise DatasetContractError("manifest H2 timing self-digest does not match its embedded values")

    runtime = manifest.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or not runtime
        or not all(isinstance(key, str) and key and isinstance(value, str) and value for key, value in runtime.items())
    ):
        raise DatasetContractError("manifest H2 runtime snapshot is malformed")
    if legacy_timing:
        if "runtime_digest" in manifest:
            raise DatasetContractError("pre-timing H2 manifest cannot declare a runtime self-digest")
    elif not _is_sha256(manifest.get("runtime_digest"), prefixed=True) or manifest[
        "runtime_digest"
    ] != _snapshot_sha256(runtime):
        raise DatasetContractError("manifest H2 runtime self-digest does not match its embedded values")


def manifest_control_hz(manifest: Mapping[str, Any]) -> int:
    """Return the recorded H2 sample cadence, including the sole v1 fallback."""

    action = manifest.get("action")
    environment = manifest.get("environment")
    if not isinstance(action, Mapping) or not isinstance(environment, Mapping):
        raise DatasetContractError("manifest has no H2 timing contract")
    timing = environment.get("timing")
    if isinstance(timing, Mapping):
        value = timing.get("control_hz")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    if (
        "timing" not in environment
        and action.get("layout") == H2_ARENA_PINK_PROFILE.layout
        and "runtime_digest" not in manifest
    ):
        return _LEGACY_H2_PINK_CONTROL_HZ
    raise DatasetContractError("manifest has no H2 control cadence")


def _validate_manifest(manifest: Mapping[str, Any], *, conversion: bool = False) -> None:
    if manifest.get("schema_id") != SCHEMA_ID or manifest.get("manifest_version") != MANIFEST_VERSION:
        raise DatasetContractError("unsupported dataset manifest schema or version")
    lane = manifest.get("lane")
    producer = manifest.get("producer")
    stage = manifest.get("stage")
    if lane not in {"native", "arena"}:
        raise DatasetContractError(f"unsupported manifest lane {lane!r}")
    if producer not in {"recording", "autodata"}:
        raise DatasetContractError(f"unsupported manifest producer {producer!r}")
    if stage not in {"recording", "annotation", "generation"}:
        raise DatasetContractError(f"unsupported manifest stage {stage!r}")
    if (producer == "recording") != (stage == "recording"):
        raise DatasetContractError("manifest producer and stage disagree")
    seed = manifest.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise DatasetContractError("manifest seed must be a non-negative integer or null")
    action = manifest.get("action")
    if not isinstance(action, Mapping):
        raise DatasetContractError("manifest action must be an object")
    layout = action.get("layout")
    profile = ACTION_PROFILES.get(layout)
    if profile is None:
        raise DatasetContractError(f"unsupported action layout {layout!r}")
    if dict(action) != asdict(profile) or profile.lane != lane:
        raise DatasetContractError("manifest lane and action profile disagree")
    task = manifest.get("task")
    if not isinstance(task, Mapping) or task.get("manipulation_hand") not in {"left", "right"}:
        raise DatasetContractError("manifest task must declare a supported manipulation hand")
    environment = manifest.get("environment")
    _scene_randomization(manifest, profile)
    runtime = manifest.get("runtime")
    h2_profile = profile.controller.startswith("h2_")
    if h2_profile:
        _validate_h2_snapshots(manifest, profile)
    elif conversion:
        nominal_environment, _ = _split_scene_randomization(environment, label="G1")
        if (
            set(nominal_environment) != {"id", "digest"}
            or nominal_environment.get("id") != STATIC_APPLE_ENVIRONMENT.environment_id
            or not _is_sha256(nominal_environment.get("digest"), prefixed=True)
        ):
            raise DatasetContractError("manifest G1 environment snapshot is malformed")
        if (
            not isinstance(runtime, Mapping)
            or not runtime
            or not all(
                isinstance(key, str) and key and isinstance(value, str) and value for key, value in runtime.items()
            )
        ):
            raise DatasetContractError("manifest G1 runtime snapshot is malformed")
        _validate_task_snapshot(task, label="G1", allow_hand_only=True)
    else:
        nominal_environment, _ = _split_scene_randomization(environment, label="static-apple")
        if nominal_environment != _environment_manifest(profile, _repository_root()):
            raise DatasetContractError("manifest environment does not match the static-apple contract")
        if runtime != _repository_runtime():
            raise DatasetContractError("manifest runtime pins do not match this repository")
        if dict(task) != _task_manifest(profile, task["manipulation_hand"], _repository_root()):
            raise DatasetContractError("manifest task descriptor or success contract does not match this repository")
    repository = manifest.get("repository")
    _validate_repository_state(repository, label="manifest")
    provenance = manifest.get("provenance")
    _validate_provenance(provenance, label="manifest")
    provenance_status = provenance["status"]
    if (stage == "recording") != (provenance_status in {"recorded", "legacy_unverified"}):
        raise DatasetContractError("manifest stage and provenance disagree")
    source = manifest.get("source")
    if stage == "recording" and source is not None:
        raise DatasetContractError("recording data cannot claim source lineage")
    if stage != "recording" and not isinstance(source, Mapping):
        raise DatasetContractError("AutoData output requires source lineage")
    if isinstance(source, Mapping):
        expected_source_stage = "recording" if stage == "annotation" or lane == "arena" else "annotation"
        if (
            not isinstance(source.get("name"), str)
            or not source["name"]
            or not isinstance(source.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is None
            or source.get("schema_id") != SCHEMA_ID
            or source.get("stage") != expected_source_stage
        ):
            raise DatasetContractError("manifest source lineage is malformed")
        lineage_fields = {"repository", "provenance", "source"}
        present_lineage_fields = lineage_fields.intersection(source)
        if present_lineage_fields and present_lineage_fields != lineage_fields:
            raise DatasetContractError("manifest source qualification lineage is incomplete")
        if present_lineage_fields:
            _validate_source_qualification_lineage(source)


def _validate_repository_state(repository: Any, *, label: str) -> None:
    if (
        not isinstance(repository, Mapping)
        or re.fullmatch(r"[0-9a-f]{40}", str(repository.get("revision", ""))) is None
        or not isinstance(repository.get("dirty"), bool)
    ):
        raise DatasetContractError(f"{label} repository state is malformed")


def _validate_provenance(provenance: Any, *, label: str) -> None:
    if not isinstance(provenance, Mapping) or provenance.get("status") not in {
        "recorded",
        "derived",
        "legacy_unverified",
    }:
        raise DatasetContractError(f"{label} provenance is malformed")
    if (
        provenance.get("status") == "legacy_unverified"
        and re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("original_sha256", ""))) is None
    ):
        raise DatasetContractError("legacy manifest must preserve the original SHA-256")


def _validate_source_qualification_lineage(source: Mapping[str, Any]) -> None:
    generation = 0
    current: Any = source
    while current is not None:
        if not isinstance(current, Mapping):
            raise DatasetContractError("manifest source qualification lineage is malformed")
        if (
            not isinstance(current.get("name"), str)
            or not current["name"]
            or re.fullmatch(r"[0-9a-f]{64}", str(current.get("sha256", ""))) is None
            or current.get("schema_id") != SCHEMA_ID
            or current.get("stage") not in {"recording", "annotation", "generation"}
        ):
            raise DatasetContractError("manifest source qualification lineage is malformed")
        _validate_repository_state(current.get("repository"), label="manifest source")
        _validate_provenance(current.get("provenance"), label="manifest source")
        stage = current.get("stage")
        provenance_status = current["provenance"].get("status")
        if (stage == "recording") != (provenance_status in {"recorded", "legacy_unverified"}):
            raise DatasetContractError("manifest source stage and provenance disagree")
        ancestor = current.get("source")
        if stage == "recording" and ancestor is not None:
            raise DatasetContractError("recording source lineage cannot have an ancestor")
        if stage == "annotation" and (not isinstance(ancestor, Mapping) or ancestor.get("stage") != "recording"):
            raise DatasetContractError("annotation source lineage must preserve its recording ancestor")
        current = ancestor
        generation += 1
        if generation > 8:
            raise DatasetContractError("manifest source qualification lineage is unexpectedly deep")


def _require_qualified_lineage(source: Any) -> None:
    generation = 0
    while source is not None:
        if not isinstance(source, Mapping) or not {"repository", "provenance", "source"} <= set(source):
            raise DatasetContractError("qualification requires complete source qualification lineage")
        provenance = source["provenance"]
        if provenance.get("status") == "legacy_unverified":
            raise DatasetContractError("legacy-unverified source data cannot qualify a workflow run")
        source = source["source"]
        generation += 1
        if generation > 8:
            raise DatasetContractError("source qualification lineage is unexpectedly deep")


def _require_current_h2_contract(manifest: Mapping[str, Any], profile: ActionProfile) -> None:
    environment, _ = _split_scene_randomization(manifest["environment"], label="H2")
    if "timing" not in environment or "runtime_digest" not in manifest:
        raise DatasetContractError("pre-timing H2 data cannot qualify a workflow run")
    root = _repository_root()
    if environment != _environment_manifest(profile, root):
        raise DatasetContractError("qualification requires the current H2 environment contract")
    task = manifest["task"]
    if dict(task) != _task_manifest(profile, task["manipulation_hand"], root):
        raise DatasetContractError("qualification requires the current H2 task contract")
    runtime = _repository_runtime(root)
    if manifest["runtime"] != runtime or manifest["runtime_digest"] != _snapshot_sha256(runtime):
        raise DatasetContractError("qualification requires the current H2 runtime contract")


def validate_dataset(
    dataset_path: str | Path,
    *,
    require_lane: RecordingLane | None = None,
    require_producer: DatasetProducer | None = None,
    require_stage: DatasetStage | None = None,
    source_dataset: str | Path | None = None,
    qualification: bool = False,
    conversion: bool = False,
) -> dict[str, Any]:
    """Validate manifest routing, action width, and optional workflow compatibility."""

    if conversion and qualification:
        raise DatasetContractError("conversion admission cannot qualify a workflow run")
    path = Path(dataset_path)
    manifest = read_manifest(path, conversion=conversion)
    if require_lane is not None and manifest["lane"] != require_lane:
        raise DatasetContractError(f"dataset lane {manifest['lane']!r} does not match requested {require_lane!r}")
    if require_producer is not None and manifest["producer"] != require_producer:
        raise DatasetContractError(
            f"dataset producer {manifest['producer']!r} does not match required {require_producer!r}"
        )
    if require_stage is not None and manifest["stage"] != require_stage:
        raise DatasetContractError(f"dataset stage {manifest['stage']!r} does not match required {require_stage!r}")
    profile = ACTION_PROFILES[manifest["action"]["layout"]]
    if qualification:
        if profile.controller.startswith("h2_"):
            _require_current_h2_contract(manifest, profile)
        if manifest["provenance"]["status"] == "legacy_unverified":
            raise DatasetContractError("legacy-unverified data cannot qualify a workflow run")
        _require_qualified_lineage(manifest["source"])

    expected_width = int(manifest["action"]["width"])
    definition = _environment_definition(profile)
    _, scene_randomization = _split_scene_randomization(manifest["environment"], label="dataset")
    with h5py.File(path, "r") as dataset:
        episodes = dataset.get("data")
        if not isinstance(episodes, h5py.Group) or len(episodes) == 0:
            raise DatasetContractError("dataset contains no episodes")
        for episode_name, episode in episodes.items():
            actions = episode.get("actions") if isinstance(episode, h5py.Group) else None
            if not isinstance(actions, h5py.Dataset) or actions.ndim != 2:
                raise DatasetContractError(f"data/{episode_name}/actions must have shape (T, {expected_width})")
            if actions.shape[0] <= 0 or actions.shape[1] != expected_width:
                raise DatasetContractError(
                    f"data/{episode_name}/actions must have shape (T, {expected_width}), got {actions.shape}"
                )
            if scene_randomization is not None:
                _validate_scene_randomization_episode(
                    episode,
                    scene_randomization,
                    definition,
                    actions.shape[0],
                    settled_objects=profile.controller.startswith("h2_"),
                )
            elif any(
                f"{prefix}/rigid_object/{definition.asset('support').name}" in episode
                for prefix in ("initial_state", "states")
            ):
                raise DatasetContractError(f"{episode.name} recorded table state requires scene randomization metadata")
            if profile.controller.startswith("h2_"):
                _validate_h2_episode_fields(episode_name, episode, actions, profile, manifest["stage"])
                from e2e_robot_workflows.controllers.h2_feedback import validate_h2_initial_state

                try:
                    validate_h2_initial_state(episode.get("initial_state", {}), required=qualification)
                except ValueError as error:
                    raise DatasetContractError(f"data/{episode_name}: {error}") from error

    if source_dataset is not None:
        source = manifest.get("source")
        actual_sha = file_sha256(source_dataset)
        if not isinstance(source, Mapping) or source.get("sha256") != actual_sha:
            raise DatasetContractError("manifest source SHA-256 does not match the supplied source dataset")
    return manifest


def _xyzw_rotation_matrix(quaternion: Any) -> Any:
    """Return rotation matrices from normalized XYZW quaternions without Isaac helpers."""

    import numpy as np

    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def validate_arena_recording_contract(
    dataset_path: str | Path,
    *,
    manipulation_hand: ManipulationHand | None = None,
    require_success_boundary: bool = False,
    qualification: bool = False,
) -> dict[str, Any]:
    """Validate direct Arena datagen fields and their independent XYZW semantics."""

    import numpy as np

    manifest = validate_dataset(
        dataset_path,
        require_lane="arena",
        require_producer="recording",
        require_stage="recording",
        qualification=qualification,
    )
    layout = manifest["action"]["layout"]
    if layout not in _ARENA_PINK_POSE_OFFSETS:
        raise DatasetContractError("Arena AutoData requires the Pink recording action layout")
    selected_hand = manipulation_hand or manifest["task"]["manipulation_hand"]
    if selected_hand != manifest["task"]["manipulation_hand"]:
        raise DatasetContractError("requested manipulation hand disagrees with the Arena recording manifest")
    if require_success_boundary:
        detected_hand = detect_arena_manipulation_hand(dataset_path)
        if detected_hand != selected_hand:
            raise ArenaEpisodeRejection(
                "Arena recording's first pickup boundary belongs to "
                f"{detected_hand}, not manifest-selected {selected_hand}"
            )

    _, scene_randomization = _split_scene_randomization(manifest["environment"], label="dataset")
    with h5py.File(dataset_path, "r") as dataset:
        for episode_name, episode in dataset["data"].items():
            support_offset = 0.0
            if scene_randomization is not None:
                support = scene_randomization["support_asset"]
                support_offset = (
                    float(episode[f"initial_state/rigid_object/{support}/root_pose"][0, 2])
                    - scene_randomization["nominal_support_root_z_m"]
                )
            actions = np.asarray(episode["actions"])
            steps = actions.shape[0]
            datagen = episode.get("obs/datagen_info")
            if not isinstance(datagen, h5py.Group):
                raise DatasetContractError(f"data/{episode_name} has no direct obs/datagen_info")

            pose_paths = [
                "eef_pose/left",
                "eef_pose/right",
                "object_pose/pick_object",
                "target_eef_pose/left",
                "target_eef_pose/right",
            ]
            if layout == H2_ARENA_PINK_PROFILE.layout:
                pose_paths.append("object_pose/placement_target")
            for path in pose_paths:
                value = datagen.get(path)
                if not isinstance(value, h5py.Dataset) or value.shape != (steps, 4, 4):
                    shape = getattr(value, "shape", None)
                    raise DatasetContractError(
                        f"data/{episode_name}/obs/datagen_info/{path} must have shape ({steps}, 4, 4), got {shape}"
                    )
                if not np.isfinite(value[...]).all():
                    raise DatasetContractError(
                        f"data/{episode_name}/obs/datagen_info/{path} contains non-finite values"
                    )

            for hand, position_start, quaternion_start in _ARENA_PINK_POSE_OFFSETS[layout]:
                quaternion = actions[:, quaternion_start : quaternion_start + 4]
                norms = np.linalg.norm(quaternion, axis=1)
                if not np.allclose(norms, 1.0, atol=1e-4, rtol=0.0):
                    raise DatasetContractError(f"data/{episode_name} {hand} action quaternions are not normalized XYZW")
                normalized_quaternion = quaternion / norms[:, None]
                target = np.asarray(datagen[f"target_eef_pose/{hand}"])
                position_error = np.max(np.abs(target[:, :3, 3] - actions[:, position_start : position_start + 3]))
                rotation_error = np.max(np.abs(target[:, :3, :3] - _xyzw_rotation_matrix(normalized_quaternion)))
                if position_error > 1e-6 or rotation_error > 1e-5:
                    raise DatasetContractError(
                        f"data/{episode_name} {hand} target_eef_pose disagrees with its XYZW action "
                        f"(position={position_error:.3g}, rotation={rotation_error:.3g})"
                    )

            for hand in ("left", "right"):
                path = f"subtask_term_signals/{hand}_hand_holds_object"
                value = datagen.get(path)
                if not isinstance(value, h5py.Dataset) or value.shape != (steps,):
                    shape = getattr(value, "shape", None)
                    raise DatasetContractError(
                        f"data/{episode_name}/obs/datagen_info/{path} must have shape ({steps},), got {shape}"
                    )
            if require_success_boundary:
                if layout == H2_ARENA_PINK_PROFILE.layout:
                    for hand in ("left", "right"):
                        if f"grasp_signals/{hand}_hand_holds_object" not in datagen:
                            raise DatasetContractError(
                                f"data/{episode_name} requires raw grasp_signals for H2 handover"
                            )
                episode_success = episode.attrs.get("success")
                if not isinstance(episode_success, bool | np.bool_) or not bool(episode_success):
                    raise DatasetContractError(f"data/{episode_name} is not labeled as a successful episode")
                validate_arena_episode_boundaries(
                    {
                        hand: np.asarray(datagen[f"subtask_term_signals/{hand}_hand_holds_object"])
                        for hand in ("left", "right")
                    },
                    steps=steps,
                    manipulation_hand=selected_hand,
                    require_handover=layout == H2_ARENA_PINK_PROFILE.layout,
                    episode_name=episode_name,
                    grasp_signals=(
                        {
                            hand: np.asarray(datagen[f"grasp_signals/{hand}_hand_holds_object"])
                            for hand in ("left", "right")
                        }
                        if layout == H2_ARENA_PINK_PROFILE.layout
                        else None
                    ),
                )
                if layout == H2_ARENA_PINK_PROFILE.layout:
                    pick_position_relative = np.asarray(datagen["object_pose/pick_object"])[-1, :3, 3]
                    h2_environment = _repository_h2_environment()
                    # Supported-placement bounds follow this episode's own table height.
                    support_surface_z = h2_environment.support_surface_z + support_offset
                    min_supported_height = support_surface_z - h2_environment.min_supported_height_below_support_m
                    max_object_height = support_surface_z + h2_environment.max_object_height_above_support_m
                    world_pose_paths = {
                        "pick_object": "states/rigid_object/pick_object/root_pose",
                        "placement_target": "states/rigid_object/placement_target/root_pose",
                    }
                    world_positions = {}
                    for object_name, pose_path in world_pose_paths.items():
                        pose = episode.get(pose_path)
                        if not isinstance(pose, h5py.Dataset) or pose.shape != (steps, 7):
                            shape = getattr(pose, "shape", None)
                            raise DatasetContractError(
                                f"data/{episode_name}/{pose_path} must have shape ({steps}, 7), got {shape}"
                            )
                        values = np.asarray(pose)
                        if not np.isfinite(values).all():
                            raise DatasetContractError(f"data/{episode_name}/{pose_path} contains non-finite values")
                        world_positions[object_name] = values[-1, :3]
                    pick_position_world = world_positions["pick_object"]
                    target_position_world = world_positions["placement_target"]
                    xy_separation = float(np.linalg.norm(pick_position_world[:2] - target_position_world[:2]))
                    height_delta = float(pick_position_world[2] - target_position_world[2])
                    release_distances = {
                        hand: float(
                            np.linalg.norm(np.asarray(datagen[f"eef_pose/{hand}"])[-1, :3, 3] - pick_position_relative)
                        )
                        for hand in ("left", "right")
                    }
                    if (
                        xy_separation > h2_environment.success_placement_xy_m
                        or not (
                            h2_environment.min_object_height_above_target_m
                            < height_delta
                            <= h2_environment.success_placement_z_m
                        )
                        or world_positions["pick_object"][2] <= min_supported_height
                        or world_positions["pick_object"][2] >= max_object_height
                        or world_positions["placement_target"][2] <= min_supported_height
                    ):
                        raise DatasetContractError(
                            f"data/{episode_name} does not end with the H2 apple on the placement target"
                        )
                    if any(distance <= h2_environment.release_clearance_m for distance in release_distances.values()):
                        raise DatasetContractError(
                            f"data/{episode_name} does not end with both H2 hands released from the apple"
                        )
    return manifest


def stamp_recording_dataset(
    dataset_path: str | Path,
    *,
    lane: RecordingLane,
    wbc_mode: str | None = None,
    arena_embodiment: str | None = None,
    seed: int | None = None,
    manipulation_hand: ManipulationHand = "right",
    scene_randomization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stamp and immediately validate a newly recorded dataset."""

    profile = resolve_recording_profile(
        lane,
        wbc_mode=wbc_mode,
        arena_embodiment=arena_embodiment,
    )
    manifest = build_manifest(
        producer="recording",
        profile=profile,
        seed=seed,
        manipulation_hand=manipulation_hand,
        scene_randomization=scene_randomization,
    )
    write_manifest(dataset_path, manifest)
    try:
        validated = validate_dataset(dataset_path, require_lane=lane, require_producer="recording")
        if profile.layout in _ARENA_PINK_POSE_OFFSETS:
            validate_arena_recording_contract(dataset_path, manipulation_hand=manipulation_hand)
        return validated
    except Exception:
        with h5py.File(dataset_path, "r+") as dataset:
            if MANIFEST_ATTRIBUTE in dataset.attrs:
                del dataset.attrs[MANIFEST_ATTRIBUTE]
        raise


def stamp_autodata_dataset(
    dataset_path: str | Path,
    *,
    source_dataset: str | Path,
    stage: Literal["annotation", "generation"],
    seed: int | None = None,
    scene_randomization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stamp derived data with the source identity and the target run's seed/distribution."""

    source_manifest = validate_dataset(source_dataset)
    profile = ACTION_PROFILES[source_manifest["action"]["layout"]]
    if profile.controller.startswith("h2_"):
        source_manifest = validate_dataset(source_dataset, qualification=True)
    manifest = build_manifest(
        producer="autodata",
        profile=profile,
        source_dataset=source_dataset,
        stage=stage,
        seed=seed,
        manipulation_hand=source_manifest["task"]["manipulation_hand"],
        scene_randomization=scene_randomization,
    )
    write_manifest(dataset_path, manifest)
    return validate_dataset(
        dataset_path,
        require_lane=profile.lane,
        require_producer="autodata",
        require_stage=stage,
        source_dataset=source_dataset,
    )


def migrate_legacy_recording(
    dataset_path: str | Path,
    *,
    output_path: str | Path,
    lane: RecordingLane,
    layout: str,
    manipulation_hand: ManipulationHand,
    confirmed: bool,
) -> dict[str, Any]:
    """Stamp old recording data only after an operator explicitly selects its contract."""

    if not confirmed:
        raise DatasetContractError("legacy migration requires --confirm-legacy-migration")
    profile = ACTION_PROFILES.get(layout)
    if profile is None:
        raise DatasetContractError(f"unsupported action layout {layout!r}")
    if profile.lane != lane:
        raise DatasetContractError("selected legacy lane and action layout disagree")

    # Validate the selected width before copying the immutable legacy evidence.
    path = Path(dataset_path)
    with h5py.File(path, "r") as dataset:
        if MANIFEST_ATTRIBUTE in dataset.attrs:
            raise DatasetContractError("legacy migration refuses a dataset that already has a manifest")
        episodes = dataset.get("data")
        if not isinstance(episodes, h5py.Group) or len(episodes) == 0:
            raise DatasetContractError("dataset contains no episodes")
        for episode_name, episode in episodes.items():
            actions = episode.get("actions") if isinstance(episode, h5py.Group) else None
            if not isinstance(actions, h5py.Dataset) or actions.ndim != 2 or actions.shape[1] != profile.width:
                shape = getattr(actions, "shape", None)
                raise DatasetContractError(
                    f"data/{episode_name}/actions does not match selected {profile.width}-D layout; got {shape}"
                )

    destination = Path(output_path)
    if destination.resolve() == path.resolve():
        raise DatasetContractError("legacy migration requires a distinct output file")
    if destination.exists():
        raise DatasetContractError(f"legacy migration output already exists: {destination}")
    if profile.lane == "native":
        from e2e_robot_workflows.autodata.dataset_mode import DatasetModeError, detect_manipulation_hand

        try:
            detected_hand = detect_manipulation_hand(path)
        except DatasetModeError as error:
            raise DatasetContractError(f"legacy native hand evidence is invalid: {error}") from error
        if detected_hand != manipulation_hand:
            raise DatasetContractError(
                f"legacy trajectory uses the {detected_hand} hand, not the selected {manipulation_hand} hand"
            )

    manifest = build_manifest(
        producer="recording",
        profile=profile,
        manipulation_hand=manipulation_hand,
    )
    manifest["provenance"] = {
        "status": "legacy_unverified",
        "original_sha256": file_sha256(path),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    try:
        write_manifest(destination, manifest)
        validated = validate_dataset(destination, require_lane=lane, require_stage="recording")
        if profile.layout in _ARENA_PINK_POSE_OFFSETS:
            validate_arena_recording_contract(destination, manipulation_hand=manipulation_hand)
        return validated
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stamp = subparsers.add_parser("stamp-autodata", help="stamp one completed AutoData output")
    stamp.add_argument("--input-file", required=True, type=Path)
    stamp.add_argument("--source-file", required=True, type=Path)
    stamp.add_argument("--stage", required=True, choices=("annotation", "generation"))

    migrate = subparsers.add_parser("migrate-recording", help="explicitly stamp one legacy recording")
    migrate.add_argument("--input-file", required=True, type=Path)
    migrate.add_argument("--output-file", required=True, type=Path)
    migrate.add_argument("--lane", required=True, choices=("native", "arena"))
    migrate.add_argument("--layout", required=True, choices=tuple(ACTION_PROFILES))
    migrate.add_argument("--manipulation-hand", required=True, choices=("left", "right"))
    migrate.add_argument("--confirm-legacy-migration", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Stamp derived or explicitly migrated datasets for shell runners."""

    args = _parse_args(argv)
    try:
        if args.command == "stamp-autodata":
            stamp_autodata_dataset(args.input_file, source_dataset=args.source_file, stage=args.stage)
        else:
            migrate_legacy_recording(
                args.input_file,
                output_path=args.output_file,
                lane=args.lane,
                layout=args.layout,
                manipulation_hand=args.manipulation_hand,
                confirmed=args.confirm_legacy_migration,
            )
    except (DatasetContractError, OSError) as error:
        print(f"Dataset manifest update failed: {error}", file=sys.stderr)
        return 2
    return 0


__all__ = [
    "ACTION_PROFILES",
    "ARENA_JOINT_PROFILE",
    "ARENA_PINK_PROFILE",
    "H2_ARENA_JOINT_PROFILE",
    "H2_ARENA_PINK_PROFILE",
    "ArenaEpisodeRejection",
    "DatasetContractError",
    "MANIFEST_ATTRIBUTE",
    "NATIVE_AGILE_PROFILE",
    "NATIVE_FIXED_PROFILE",
    "SCENE_RANDOMIZATION_FIELD",
    "SCHEMA_ID",
    "build_manifest",
    "default_recording_dataset_file",
    "detect_arena_manipulation_hand",
    "file_sha256",
    "main",
    "migrate_legacy_recording",
    "read_manifest",
    "resolve_recording_lane",
    "resolve_recording_profile",
    "stamp_recording_dataset",
    "stamp_autodata_dataset",
    "validate_arena_episode_boundaries",
    "validate_arena_recording_contract",
    "validate_dataset",
    "write_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())

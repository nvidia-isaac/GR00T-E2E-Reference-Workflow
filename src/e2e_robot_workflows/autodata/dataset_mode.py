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

"""Resolve the G1 AutoData profile from an explicit dataset manifest.

New workflow data is routed by its versioned manifest, never by its filename or action width.
The width and trajectory helpers remain available only to inspect data during explicit legacy
migration.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from e2e_robot_workflows.data.dataset_manifest import (
    ACTION_PROFILES,
    NATIVE_AGILE_PROFILE,
    NATIVE_FIXED_PROFILE,
    DatasetContractError,
    DatasetStage,
    RecordingLane,
    validate_dataset,
)

WbcMode = Literal["fixed", "agile"]
RequestedWbcMode = Literal["auto", "fixed", "agile"]
ManipulationHand = Literal["left", "right"]
RequestedManipulationHand = Literal["auto", "left", "right"]

ACTION_WIDTH_BY_MODE: dict[WbcMode, int] = {
    "fixed": 28,
    "agile": 32,
}
MODE_BY_ACTION_WIDTH = {width: mode for mode, width in ACTION_WIDTH_BY_MODE.items()}
MIN_LIFT_M = 0.02
MIN_HAND_SCORE_MARGIN_M = 0.05


class DatasetModeError(ValueError):
    """Raised when a dataset cannot select one supported AutoData profile."""


@dataclass(frozen=True)
class AutoDataProfile:
    """Manifest-selected environment and task profile."""

    lane: RecordingLane
    controller: str
    manipulation_hand: ManipulationHand


def resolve_autodata_profile(
    dataset_path: str | Path,
    *,
    requested_lane: RecordingLane,
    requested_mode: RequestedWbcMode = "auto",
    requested_hand: RequestedManipulationHand = "auto",
    required_stage: DatasetStage,
) -> AutoDataProfile:
    """Validate a manifested input and return its explicit AutoData route."""

    try:
        manifest = validate_dataset(dataset_path, require_lane=requested_lane, require_stage=required_stage)
    except DatasetContractError as error:
        raise DatasetModeError(str(error)) from error

    if not manifest["action"]["autodata_compatible"]:
        raise DatasetModeError(f"action layout {manifest['action']['layout']!r} is not AutoData-compatible")

    layout = manifest["action"]["layout"]
    if requested_lane == "native":
        mode_by_layout: dict[str, WbcMode] = {
            NATIVE_FIXED_PROFILE.layout: "fixed",
            NATIVE_AGILE_PROFILE.layout: "agile",
        }
        try:
            controller = mode_by_layout[layout]
        except KeyError as error:
            raise DatasetModeError(f"unsupported native AutoData action layout {layout!r}") from error
        if requested_mode != "auto" and requested_mode != controller:
            raise DatasetModeError(
                f"manifest selects native {controller}, but --wbc-mode {requested_mode} was requested"
            )
    else:
        if requested_mode != "auto":
            raise DatasetModeError("--requested-mode is only valid for native datasets")
        profile = ACTION_PROFILES.get(layout)
        if profile is None or profile.lane != "arena" or not profile.autodata_compatible:
            raise DatasetModeError(f"unsupported Arena AutoData action layout {layout!r}")
        controller = profile.controller

    manifest_hand: ManipulationHand = manifest["task"]["manipulation_hand"]
    if requested_hand != "auto" and requested_hand != manifest_hand:
        raise DatasetModeError(
            f"manifest selects {manifest_hand}-hand manipulation, but --manipulation-hand "
            f"{requested_hand} was requested"
        )
    return AutoDataProfile(
        lane=requested_lane,
        controller=controller,
        manipulation_hand=manifest_hand,
    )


def detect_wbc_mode(dataset_path: str | Path) -> WbcMode:
    """Return the whole-body mode encoded by every episode's action width."""

    path = Path(dataset_path)
    try:
        dataset = h5py.File(path, "r")
    except OSError as error:
        raise DatasetModeError(f"cannot open {path}: {error}") from error

    with dataset:
        data = dataset.get("data")
        if not isinstance(data, h5py.Group):
            raise DatasetModeError(f"{path} has no HDF5 'data' group")

        episode_names = sorted(data.keys())
        if not episode_names:
            raise DatasetModeError(f"{path} contains no episodes")

        episodes_by_width: dict[int, list[str]] = {}
        for episode_name in episode_names:
            episode = data[episode_name]
            if not isinstance(episode, h5py.Group) or "actions" not in episode:
                raise DatasetModeError(f"episode {episode_name!r} has no 'actions' dataset")
            actions = episode["actions"]
            if not isinstance(actions, h5py.Dataset) or len(actions.shape) != 2:
                raise DatasetModeError(
                    f"episode {episode_name!r} actions must have shape (steps, width), got {actions.shape}"
                )
            episodes_by_width.setdefault(int(actions.shape[1]), []).append(episode_name)

    if len(episodes_by_width) != 1:
        summary = ", ".join(f"{width}-D: {', '.join(names)}" for width, names in sorted(episodes_by_width.items()))
        raise DatasetModeError(f"episodes use inconsistent action widths ({summary})")

    action_width = next(iter(episodes_by_width))
    try:
        return MODE_BY_ACTION_WIDTH[action_width]
    except KeyError as error:
        supported = ", ".join(f"{width}-D ({mode})" for mode, width in ACTION_WIDTH_BY_MODE.items())
        raise DatasetModeError(f"unsupported {action_width}-D actions; expected {supported}") from error


def resolve_wbc_mode(dataset_path: str | Path, requested_mode: RequestedWbcMode = "auto") -> WbcMode:
    """Resolve ``auto`` or verify that an explicit mode matches the dataset."""

    detected_mode = detect_wbc_mode(dataset_path)
    if requested_mode == "auto" or requested_mode == detected_mode:
        return detected_mode

    detected_width = ACTION_WIDTH_BY_MODE[detected_mode]
    requested_width = ACTION_WIDTH_BY_MODE[requested_mode]
    raise DatasetModeError(
        f"dataset contains {detected_width}-D {detected_mode} actions, but --wbc-mode "
        f"{requested_mode} selects a {requested_width}-D environment"
    )


def detect_manipulation_hand(dataset_path: str | Path) -> ManipulationHand:
    """Return the wrist that stays closest to the object while it is lifted."""

    path = Path(dataset_path)
    try:
        dataset = h5py.File(path, "r")
    except OSError as error:
        raise DatasetModeError(f"cannot open {path}: {error}") from error

    episode_hands: dict[ManipulationHand, list[str]] = {"left": [], "right": []}
    with dataset:
        data = dataset.get("data")
        if not isinstance(data, h5py.Group) or len(data) == 0:
            raise DatasetModeError(f"{path} contains no episodes")

        for episode_name in sorted(data.keys()):
            episode = data[episode_name]
            fields = {name: episode.get(f"obs/{name}") for name in ("object_pos", "left_eef_pos", "right_eef_pos")}
            missing = [name for name, field in fields.items() if not isinstance(field, h5py.Dataset)]
            if missing:
                raise DatasetModeError(
                    f"episode {episode_name!r} cannot infer the manipulation hand; missing "
                    f"{', '.join(f'obs/{name}' for name in missing)}"
                )

            object_pos = np.asarray(fields["object_pos"])
            left_pos = np.asarray(fields["left_eef_pos"])
            right_pos = np.asarray(fields["right_eef_pos"])
            if object_pos.ndim != 2 or object_pos.shape[1] != 3:
                raise DatasetModeError(
                    f"episode {episode_name!r} obs/object_pos must have shape (steps, 3), got {object_pos.shape}"
                )
            if left_pos.shape != object_pos.shape or right_pos.shape != object_pos.shape:
                raise DatasetModeError(
                    f"episode {episode_name!r} wrist and object observations must share shape "
                    f"{object_pos.shape}, got left={left_pos.shape}, right={right_pos.shape}"
                )

            lifted = object_pos[:, 2] > float(object_pos[:, 2].min()) + MIN_LIFT_M
            if not lifted.any():
                raise DatasetModeError(
                    f"episode {episode_name!r} never lifts the object by {MIN_LIFT_M:.2f} m, "
                    "so its manipulation hand cannot be verified"
                )

            scores = {
                "left": float(np.median(np.linalg.norm(left_pos[lifted] - object_pos[lifted], axis=1))),
                "right": float(np.median(np.linalg.norm(right_pos[lifted] - object_pos[lifted], axis=1))),
            }
            if abs(scores["left"] - scores["right"]) < MIN_HAND_SCORE_MARGIN_M:
                raise DatasetModeError(
                    f"episode {episode_name!r} has ambiguous manipulation-hand distances "
                    f"(left={scores['left']:.3f} m, right={scores['right']:.3f} m)"
                )
            hand: ManipulationHand = "left" if scores["left"] < scores["right"] else "right"
            episode_hands[hand].append(episode_name)

    selected_hands = [hand for hand, names in episode_hands.items() if names]
    if len(selected_hands) != 1:
        summary = ", ".join(f"{hand}: {', '.join(episode_hands[hand])}" for hand in selected_hands)
        raise DatasetModeError(f"episodes use different manipulation hands ({summary})")
    return selected_hands[0]


def resolve_manipulation_hand(
    dataset_path: str | Path,
    requested_hand: RequestedManipulationHand = "auto",
) -> ManipulationHand:
    """Resolve ``auto`` or verify that an explicit hand matches the dataset."""

    detected_hand = detect_manipulation_hand(dataset_path)
    if requested_hand == "auto" or requested_hand == detected_hand:
        return detected_hand
    raise DatasetModeError(
        f"dataset uses {detected_hand}-hand manipulation, but --manipulation-hand "
        f"{requested_hand} selects the {requested_hand}-hand task descriptor"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--requested-lane", required=True, choices=("native", "arena"))
    parser.add_argument("--requested-mode", choices=("auto", "fixed", "agile"), default="auto")
    parser.add_argument("--requested-hand", choices=("auto", "left", "right"), default="auto")
    parser.add_argument("--required-stage", required=True, choices=("recording", "annotation"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print the resolved lane, controller, and manipulation hand for the shell launcher."""

    args = _parse_args(argv)
    try:
        profile = resolve_autodata_profile(
            args.input_file,
            requested_lane=args.requested_lane,
            requested_mode=args.requested_mode,
            requested_hand=args.requested_hand,
            required_stage=args.required_stage,
        )
    except DatasetModeError as error:
        print(f"AutoData dataset preflight failed: {error}", file=sys.stderr)
        return 2
    print(profile.lane, profile.controller, profile.manipulation_hand)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

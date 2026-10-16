#!/usr/bin/env python3
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

"""Load every H2 LeRobot episode and frame through GR00T/TorchCodec."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_REPO_SRC))

from e2e_robot_workflows.data.artifact_paths import atomic_write_text  # noqa: E402

_EXPECTED_STATISTIC_WIDTHS = {
    "state": {
        "left_leg": 6,
        "right_leg": 6,
        "waist": 3,
        "head": 2,
        "left_arm": 7,
        "right_arm": 7,
        "left_hand": 22,
        "right_hand": 22,
    },
    "action": {
        "left_arm": 7,
        "right_arm": 7,
        "left_hand": 22,
        "right_hand": 22,
        "navigate_command": 3,
    },
}
_REQUIRED_STATISTICS = {"mean", "std", "min", "max", "q01", "q99"}


def _load_modality_config(path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("e2e_h2_modality_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load modality config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.h2_static_apple_config


def _write_result(path: Path, result: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(result, indent=2, sort_keys=True) + "\n")


def _dataset_totals(dataset: Path) -> tuple[int, int]:
    """Read the episode and frame totals the LeRobot writer committed."""

    metadata_path = dataset / "meta" / "info.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read LeRobot metadata {metadata_path}: {error}") from error
    if not isinstance(metadata, dict):
        raise RuntimeError(f"LeRobot metadata must be a JSON object: {metadata_path}")
    totals = metadata.get("total_episodes"), metadata.get("total_frames")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in totals):
        raise RuntimeError("LeRobot metadata must contain positive integer total_episodes and total_frames")
    return totals


def _validate_statistics(statistics: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Validate the exact normalization slices consumed by H2 training."""

    widths: dict[str, dict[str, int]] = {}
    for modality, expected_groups in _EXPECTED_STATISTIC_WIDTHS.items():
        groups = statistics.get(modality)
        if not isinstance(groups, dict) or set(groups) != set(expected_groups):
            raise RuntimeError(
                f"GR00T {modality} statistics groups {sorted(groups) if isinstance(groups, dict) else groups!r} "
                f"do not match {sorted(expected_groups)}"
            )
        widths[modality] = {}
        for group, expected_width in expected_groups.items():
            group_statistics = groups[group]
            if not isinstance(group_statistics, dict) or set(group_statistics) != _REQUIRED_STATISTICS:
                raise RuntimeError(f"GR00T {modality}.{group} statistics are incomplete")
            actual_shapes = {statistic: np.asarray(values).shape for statistic, values in group_statistics.items()}
            invalid_shapes = {
                statistic: shape for statistic, shape in actual_shapes.items() if shape != (expected_width,)
            }
            if invalid_shapes:
                raise RuntimeError(
                    f"GR00T {modality}.{group} statistic shapes {invalid_shapes} must all equal ({expected_width},)"
                )
            if not all(np.isfinite(np.asarray(values)).all() for values in group_statistics.values()):
                raise RuntimeError(f"GR00T {modality}.{group} statistics contain non-finite values")
            widths[modality][group] = expected_width
    return widths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--modality-config", required=True, type=Path)
    parser.add_argument(
        "--expected-episodes",
        type=int,
        help="Optional independent episode-count assertion; defaults to LeRobot metadata.",
    )
    parser.add_argument(
        "--expected-frames",
        type=int,
        help="Optional independent frame-count assertion; defaults to LeRobot metadata.",
    )
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()

    args.dataset = args.dataset.expanduser().resolve()
    args.modality_config = args.modality_config.expanduser().resolve()
    args.result = args.result.expanduser().resolve()
    if args.result == args.modality_config:
        parser.error(f"--result must differ from --modality-config: {args.result}")
    if args.result == args.dataset or args.result.is_relative_to(args.dataset):
        parser.error(f"--result must remain outside --dataset: {args.result}")

    result: dict[str, Any] = {"status": "error", "dataset": str(args.dataset)}
    try:
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset, extract_step_data
        from gr00t.data.embodiment_tags import EmbodimentTag

        metadata_episodes, metadata_frames = _dataset_totals(args.dataset)
        modality_config = _load_modality_config(args.modality_config)
        loader = LeRobotEpisodeLoader(
            args.dataset,
            modality_config,
            video_backend="torchcodec",
        )
        if len(loader) != metadata_episodes:
            raise RuntimeError(f"loaded {len(loader)} episodes, metadata declares {metadata_episodes}")
        if args.expected_episodes is not None and len(loader) != args.expected_episodes:
            raise RuntimeError(f"loaded {len(loader)} episodes, expected {args.expected_episodes}")

        total_frames = 0
        video_frames = 0
        for episode_index, expected_length in enumerate(loader.episode_lengths):
            episode = loader[episode_index]
            if len(episode) != expected_length:
                raise RuntimeError(f"episode {episode_index} loaded {len(episode)} frames, expected {expected_length}")
            state_width = sum(np.asarray(episode[key].iloc[0]).size for key in episode if key.startswith("state."))
            action_width = sum(np.asarray(episode[key].iloc[0]).size for key in episode if key.startswith("action."))
            if state_width != 75 or action_width != 61:
                raise RuntimeError(f"episode {episode_index} has state/action widths {state_width}/{action_width}")
            for key in episode:
                if key.startswith(("state.", "action.")):
                    if not all(np.isfinite(np.asarray(value)).all() for value in episode[key]):
                        raise RuntimeError(f"episode {episode_index} column {key!r} contains non-finite values")
            for frame in episode["video.ego_view"]:
                pixels = np.asarray(frame)
                if pixels.shape != (480, 640, 3) or not np.isfinite(pixels).all():
                    raise RuntimeError(f"episode {episode_index} has an invalid video frame")
                video_frames += 1
            total_frames += len(episode)

        if total_frames != metadata_frames or video_frames != metadata_frames:
            raise RuntimeError(
                f"loaded {total_frames} data frames and {video_frames} video frames, "
                f"metadata declares {metadata_frames}"
            )
        if args.expected_frames is not None and total_frames != args.expected_frames:
            raise RuntimeError(
                f"loaded {total_frames} data frames and {video_frames} video frames, expected {args.expected_frames}"
            )
        training_dataset = ShardedSingleStepDataset(
            dataset_path=args.dataset,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            modality_configs=modality_config,
            video_backend="torchcodec",
            shard_size=1024,
        )
        training_steps = int(sum(training_dataset.shard_lengths))
        if len(training_dataset) == 0 or training_steps <= 0:
            raise RuntimeError("GR00T training loader produced no usable shards or steps")
        training_sample = extract_step_data(
            loader[0],
            step_index=0,
            modality_configs=modality_config,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        )
        sample_state_width = sum(value.shape[-1] for value in training_sample.states.values())
        sample_action_shape = (
            next(iter(training_sample.actions.values())).shape[0],
            sum(value.shape[-1] for value in training_sample.actions.values()),
        )
        sample_image = training_sample.images["ego_view"]
        if (
            sample_state_width != 75
            or sample_action_shape != (40, 61)
            or any(value.shape[0] != 1 for value in training_sample.states.values())
            or any(value.shape[0] != 40 for value in training_sample.actions.values())
            or len(sample_image) != 1
            or np.asarray(sample_image[0]).shape != (480, 640, 3)
            or not training_sample.text
        ):
            raise RuntimeError("GR00T training sample differs from the 75-state/40x61-action/video/language contract")
        statistic_widths = _validate_statistics(training_dataset.get_dataset_statistics())
        result.update(
            status="ok",
            episodes=len(loader),
            frames=total_frames,
            video_frames=video_frames,
            state_width=75,
            action_width=61,
            training_shards=len(training_dataset),
            training_steps=training_steps,
            training_sample_action_shape=list(sample_action_shape),
            training_sample_state_width=sample_state_width,
            training_statistic_widths=statistic_widths,
        )
    except Exception as error:  # noqa: BLE001 - retain structured loader evidence
        result["failure"] = f"{type(error).__name__}: {error}"
        _write_result(args.result, result)
        return 1
    _write_result(args.result, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

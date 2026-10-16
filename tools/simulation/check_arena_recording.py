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

"""Check demonstrations recorded on the Arena-composed environment.

``tools/validate_arena_recording.sh`` records one short take per Arena embodiment and
then runs this check, which reopens each exported HDF5 and asserts the dataset contract
the recording path exists to satisfy:

- an episode was exported, with per-step ``actions``;
- ``joint_pos_target`` is present, one row per step, and as wide as the recorded joint
  state -- it is the canonical action column because it means the same thing under
  either embodiment, so a recording lacking it is untrainable whatever else it has;
- the head camera frames arrived.

Fields are matched by leaf name rather than by path: Arena nests the camera under its
``camera_obs`` group while the built-in task publishes it under ``obs``, and the
converter resolves either.

Writes a JSON result to ``$RESULT_PATH`` with ``"status": "ok"`` only when every
recording satisfies the contract.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validation_support import write_result  # noqa: E402

DEFAULT_RESULT_PATH = "test-results/arena-recording/arena-recording-metrics.json"
ACTIONS_LEAF = "actions"
CAMERA_LEAF = "robot_head_cam_rgb"
STATE_LEAF = "robot_joint_pos"
ACTION_TARGET_LEAF = "joint_pos_target"
ARENA_PINK_EMBODIMENT = "g1_wbc_agile_pink"
ARENA_PINK_WIDTH = 23
DATAGEN_PATHS = (
    "obs/datagen_info/eef_pose/left",
    "obs/datagen_info/eef_pose/right",
    "obs/datagen_info/object_pose/pick_object",
    "obs/datagen_info/target_eef_pose/left",
    "obs/datagen_info/target_eef_pose/right",
    "obs/datagen_info/subtask_term_signals/left_hand_holds_object",
    "obs/datagen_info/subtask_term_signals/right_hand_holds_object",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", default=os.environ.get("RESULT_PATH", DEFAULT_RESULT_PATH))
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="EMBODIMENT=PATH",
        help="An exported recording to check. Repeat once per Arena embodiment.",
    )
    return parser.parse_args()


def _leaf_datasets(group: Any) -> dict[str, Any]:
    """Return every dataset under ``group``, keyed by its full path within the episode."""

    found: dict[str, Any] = {}

    def visit(name: str, item: Any) -> None:
        if hasattr(item, "shape"):
            found[name] = item

    group.visititems(visit)
    return found


def _first_matching(datasets: dict[str, Any], leaf: str) -> tuple[str, Any] | tuple[None, None]:
    """Return the dataset whose path ends in ``leaf``, or ``(None, None)``."""

    for path, item in sorted(datasets.items()):
        if path.rsplit("/", 1)[-1] == leaf:
            return path, item
    return None, None


def _xyzw_rotation_matrix(quaternion: Any) -> Any:
    """Independent NumPy oracle for the pinned Arena Pink XYZW convention."""

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


def check_recording(dataset_path: Path, *, embodiment: str) -> dict[str, Any]:
    """Return the measured shape of one recording, plus any contract violations."""

    import h5py

    measured: dict[str, Any] = {"dataset": str(dataset_path)}
    problems: list[str] = []
    if not dataset_path.is_file():
        return {**measured, "problems": ["recording was not written"]}

    with h5py.File(dataset_path, "r") as recording:
        episodes = sorted(recording["data"].keys()) if "data" in recording else []
        measured["episodes"] = len(episodes)
        if not episodes:
            return {**measured, "problems": ["no episodes were exported"]}

        episode = recording["data"][episodes[0]]
        datasets = _leaf_datasets(episode)
        measured["fields"] = sorted(datasets)

        steps = None
        actions_path, actions = _first_matching(datasets, ACTIONS_LEAF)
        if actions is None:
            problems.append(f"no {ACTIONS_LEAF} dataset")
        else:
            steps = int(actions.shape[0])
            measured["actions"] = {"path": actions_path, "shape": list(actions.shape)}
            if steps == 0:
                problems.append(f"{ACTIONS_LEAF} has 0 steps")

        target_path, target = _first_matching(datasets, ACTION_TARGET_LEAF)
        state_path, state = _first_matching(datasets, STATE_LEAF)
        if target is None:
            problems.append(f"no {ACTION_TARGET_LEAF} dataset")
        else:
            measured["joint_pos_target"] = {"path": target_path, "shape": list(target.shape)}
            if steps is not None and target.shape[0] != steps:
                problems.append(f"{ACTION_TARGET_LEAF} has {target.shape[0]} rows for {steps} steps")
            if state is None:
                problems.append(f"no {STATE_LEAF} observation to size the action column against")
            elif target.ndim == 2 and target.shape[1] != state.shape[1]:
                problems.append(f"{ACTION_TARGET_LEAF} width {target.shape[1]} != {STATE_LEAF} width {state.shape[1]}")
        if state is not None:
            measured["robot_joint_pos"] = {"path": state_path, "shape": list(state.shape)}

        camera_path, camera = _first_matching(datasets, CAMERA_LEAF)
        if camera is None:
            problems.append(f"no {CAMERA_LEAF} frames")
        else:
            measured["camera"] = {"path": camera_path, "shape": list(camera.shape)}
            if camera.shape[0] == 0:
                problems.append(f"{CAMERA_LEAF} has 0 frames")

        if embodiment == ARENA_PINK_EMBODIMENT:
            import numpy as np

            missing = [path for path in DATAGEN_PATHS if path not in datasets]
            if missing:
                problems.append(f"missing direct Arena datagen_info fields: {', '.join(missing)}")
            elif actions is not None:
                if actions.ndim != 2 or actions.shape[1] != ARENA_PINK_WIDTH:
                    problems.append(f"Arena Pink actions must have shape (T, {ARENA_PINK_WIDTH}), got {actions.shape}")
                elif actions.shape[0] > 0:
                    action_values = np.asarray(actions)
                    semantic_errors: dict[str, dict[str, float]] = {}
                    for hand, position_start, quaternion_start in (("left", 2, 5), ("right", 9, 12)):
                        target_pose = np.asarray(datasets[f"obs/datagen_info/target_eef_pose/{hand}"])
                        position_error = float(
                            np.max(
                                np.abs(target_pose[:, :3, 3] - action_values[:, position_start : position_start + 3])
                            )
                        )
                        rotation_error = float(
                            np.max(
                                np.abs(
                                    target_pose[:, :3, :3]
                                    - _xyzw_rotation_matrix(action_values[:, quaternion_start : quaternion_start + 4])
                                )
                            )
                        )
                        semantic_errors[hand] = {
                            "position_max_abs": position_error,
                            "rotation_max_abs": rotation_error,
                        }
                        if position_error > 1e-6 or rotation_error > 1e-5:
                            problems.append(
                                f"{hand} target_eef_pose disagrees with the recorded XYZW action "
                                f"(position={position_error:.3g}, rotation={rotation_error:.3g})"
                            )
                    measured["datagen_info"] = {
                        "direct": True,
                        "pose_action_errors": semantic_errors,
                    }
                    first_hold_errors = {}
                    for hand in ("left", "right"):
                        target_pose = np.asarray(datasets[f"obs/datagen_info/target_eef_pose/{hand}"])[0]
                        eef_pose = np.asarray(datasets[f"obs/datagen_info/eef_pose/{hand}"])[0]
                        hold_error = float(np.max(np.abs(target_pose - eef_pose)))
                        first_hold_errors[hand] = hold_error
                        if hold_error > 1e-5:
                            problems.append(
                                f"{hand} first hold target disagrees with the current wrist pose ({hold_error:.3g})"
                            )
                    measured["datagen_info"]["first_hold_pose_max_abs"] = first_hold_errors

    return {**measured, "problems": problems}


def main() -> int:
    arguments = _parse_args()
    result: dict[str, Any] = {"status": "error", "recordings": {}}

    try:
        if not arguments.dataset:
            raise SystemExit("Pass at least one --dataset EMBODIMENT=PATH.")
        for entry in arguments.dataset:
            embodiment, separator, path = entry.partition("=")
            if not separator or not path:
                raise SystemExit(f"Expected --dataset EMBODIMENT=PATH, got {entry!r}.")
            result["recordings"][embodiment] = check_recording(Path(path), embodiment=embodiment)
        failed = {name: measured["problems"] for name, measured in result["recordings"].items() if measured["problems"]}
        result["failures"] = failed
        result["status"] = "ok" if not failed else "error"
    except Exception as exc:  # noqa: BLE001 - the result file is the report
        result["failure"] = repr(exc)
    finally:
        write_result(arguments.result_path, result)

    if result["status"] != "ok":
        print(f"Arena recording check failed: {result.get('failures') or result.get('failure')}", file=sys.stderr)
        return 1
    print(f"Arena recording check passed for {', '.join(sorted(result['recordings']))}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

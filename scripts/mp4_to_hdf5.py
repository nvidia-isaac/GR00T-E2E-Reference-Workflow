#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 The Isaac Lab Project Developers. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
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

"""Build an augmented HDF5 dataset by swapping one camera stream for MP4 frames.

Takes a ``record_demos.py`` HDF5 dataset and a directory of MP4 videos that are
visually augmented versions of a recorded camera (e.g. relit or recolored), and
writes a new dataset that appends, for each video, a copy of its source demo with
only the chosen camera observation replaced by the video frames. Actions, robot
state, and every other observation are preserved, so the augmented demos stay
trajectory-consistent.

Adapted from Isaac Lab's ``scripts/tools/mp4_to_hdf5.py``: it uses ``imageio`` (the
``converter`` extra) instead of OpenCV and replaces one configurable camera key via
a recursive copy, rather than hardcoding a fixed obs schema.

Videos must be named ``demo_<id>_*.mp4`` (as produced by ``scripts/hdf5_to_mp4.py``)
so each maps back to its source demo. The augmented video must have the same frame
count as the source episode.

Example
-------
.. code-block:: bash

    uv run --extra converter scripts/mp4_to_hdf5.py \\
        --input_file datasets/session.hdf5 \\
        --videos_dir /tmp/augmented \\
        --output_file datasets/session_augmented.hdf5
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import os
import re

import h5py
import imageio.v2 as imageio
import numpy as np

DEFAULT_CAMERA_KEY = "robot_head_cam_rgb"
_DEMO_ID_RE = re.compile(r"demo_(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an augmented dataset by swapping a camera stream for MP4s.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the source HDF5 dataset.")
    parser.add_argument("--videos_dir", type=str, required=True, help="Directory of augmented demo_<id>_*.mp4 videos.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to write the augmented HDF5 dataset.")
    parser.add_argument(
        "--camera_key",
        type=str,
        default=DEFAULT_CAMERA_KEY,
        help="Observation key under obs/ to overwrite with the video frames.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists.")
    return parser.parse_args()


def _read_mp4_frames(video_path: str) -> np.ndarray:
    """Read an MP4 into an (T, H, W, 3) uint8 array."""
    reader = imageio.get_reader(video_path)
    try:
        frames = [np.asarray(frame) for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"{video_path}: no frames decoded")
    return np.stack(frames).astype(np.uint8)


def _augment_demo(
    f_in: h5py.File, f_out: h5py.File, video_path: str, orig_demo_id: int, new_demo_id: int, camera_key: str
) -> None:
    """Copy demo_<orig> into demo_<new> and replace its camera stream with the video."""
    src_name = f"demo_{orig_demo_id}"
    if f"data/{src_name}" not in f_in:
        raise ValueError(f"{video_path}: source {src_name} not found in the input dataset")

    obs_rel = f"obs/{camera_key}"
    src_demo = f_in[f"data/{src_name}"]
    if obs_rel not in src_demo:
        raise ValueError(f"{src_name}: no '{obs_rel}' to replace (present keys: {list(src_demo['obs'].keys())})")

    frames = _read_mp4_frames(video_path)
    original = src_demo[obs_rel]
    if frames.shape[0] != original.shape[0]:
        raise ValueError(
            f"{video_path}: {frames.shape[0]} frame(s) but source {src_name} has {original.shape[0]}; "
            "the augmented video must match the recorded episode length."
        )
    if frames.shape[1:] != original.shape[1:] or frames.dtype != original.dtype:
        raise ValueError(
            f"{video_path}: frame shape/dtype {tuple(frames.shape[1:])}/{frames.dtype} does not match source "
            f"{src_name} '{obs_rel}' {tuple(original.shape[1:])}/{original.dtype}; the augmented video must "
            "preserve the recorded camera resolution and format."
        )

    dst_name = f"demo_{new_demo_id}"
    f_in.copy(f"data/{src_name}", f_out["data"], name=dst_name)
    dst_demo = f_out[f"data/{dst_name}"]
    del dst_demo[obs_rel]
    dst_demo.create_dataset(obs_rel, data=frames, compression="gzip")
    print(f"{os.path.basename(video_path)}: {src_name} -> {dst_name} (replaced {camera_key})")


def _sorted_demo_ids(data_group: h5py.Group) -> list[int]:
    ids = []
    for key in data_group.keys():
        suffix = key[len("demo_") :] if key.startswith("demo_") else ""
        if suffix.isdigit():
            ids.append(int(suffix))
    return sorted(ids)


def main() -> None:
    args = parse_args()

    if os.path.exists(args.output_file) and not args.overwrite:
        raise SystemExit(f"Output file {args.output_file} already exists. Pass --overwrite to replace.")

    video_paths = sorted(glob.glob(os.path.join(args.videos_dir, "*.mp4")))
    if not video_paths:
        raise SystemExit(f"No .mp4 files found in {args.videos_dir}")
    print(f"Found {len(video_paths)} MP4 video(s) in {args.videos_dir}")

    output_dir = os.path.dirname(os.path.abspath(args.output_file)) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Atomic write: stream into <output>.tmp and rename on success, so a mid-run
    # failure (bad MP4, OOM, interrupt) never truncates or destroys the prior output
    # under --overwrite.
    tmp_path = args.output_file + ".tmp"
    success = False
    try:
        with h5py.File(args.input_file, "r") as f_in, h5py.File(tmp_path, "w") as f_out:
            if "data" not in f_in:
                raise SystemExit(f"{args.input_file}: missing top-level 'data' group; not a record_demos HDF5 file")
            f_in.copy("data", f_out)
            for attr_name, attr_value in f_in.attrs.items():
                f_out.attrs[attr_name] = attr_value

            next_demo_id = (max(_sorted_demo_ids(f_in["data"])) + 1) if _sorted_demo_ids(f_in["data"]) else 0
            for video_path in video_paths:
                match = _DEMO_ID_RE.search(os.path.basename(video_path))
                if match is None:
                    print(f"WARNING: {os.path.basename(video_path)} is not named demo_<id>_*.mp4; skipping.")
                    continue
                _augment_demo(f_in, f_out, video_path, int(match.group(1)), next_demo_id, args.camera_key)
                next_demo_id += 1

            total = 0
            untracked: list[str] = []
            for name in f_out["data"].keys():
                demo = f_out[f"data/{name}"]
                if "num_samples" in demo.attrs:
                    total += int(demo.attrs["num_samples"])
                else:
                    untracked.append(name)
            f_out["data"].attrs["total"] = total
            if untracked:
                shown = ", ".join(untracked[:3]) + (", ..." if len(untracked) > 3 else "")
                print(
                    f"WARNING: {len(untracked)} demo(s) without 'num_samples' contributed 0 to "
                    f"data.attrs['total'] ({shown})."
                )
        os.replace(tmp_path, args.output_file)
        success = True
    finally:
        if not success and os.path.exists(tmp_path):
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

    print(f"Augmented dataset written to: {args.output_file}")


if __name__ == "__main__":
    main()

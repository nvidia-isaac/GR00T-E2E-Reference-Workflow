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

"""Export recorded camera frames from an HDF5 dataset to MP4 videos for quick review.

Reads image observations (e.g. ``robot_head_cam_rgb``) out of a ``record_demos.py``
HDF5 dataset and writes one MP4 per (demo, camera) into an output directory, without
launching the simulator. Adapted from Isaac Lab's ``scripts/tools/hdf5_to_mp4.py`` to
use ``imageio`` (the ``converter`` extra) instead of OpenCV, and to default to this
project's head camera rather than the Arena table/wrist cameras.

Videos are named ``demo_<id>_<camera>.mp4`` so ``scripts/mp4_to_hdf5.py`` can match
them back to their source demo.

Example
-------
.. code-block:: bash

    uv run --extra converter scripts/hdf5_to_mp4.py \\
        --input_file datasets/e2e_ra_dataset_native_YYYYMMDD_HHMMSS.hdf5 \\
        --output_dir /tmp/e2e_videos
"""

from __future__ import annotations

import argparse
import os

import h5py
import imageio.v2 as imageio
import numpy as np

DEFAULT_CAMERA_KEYS = ["robot_head_cam_rgb"]
DEFAULT_FRAMERATE = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert recorded HDF5 camera frames to MP4 videos.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input HDF5 dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write the MP4 files.")
    parser.add_argument(
        "--input_keys",
        type=str,
        nargs="+",
        default=DEFAULT_CAMERA_KEYS,
        help="Camera observation keys under obs/ to export (default: robot_head_cam_rgb).",
    )
    parser.add_argument(
        "--framerate", type=int, default=DEFAULT_FRAMERATE, help="Frames per second for the output video."
    )
    parser.add_argument(
        "--demo_id",
        type=int,
        default=None,
        help="If provided, only export this demo index; otherwise export every demo.",
    )
    return parser.parse_args()


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    """Coerce one recorded frame to an (H, W, 3) uint8 RGB image.

    Handles the common recorder outputs: uint8 RGB passes through; a single-channel
    or 2-D frame is broadcast to three channels; a floating-point frame (e.g. depth or
    normalized RGB) is min-max normalized per frame before quantizing.
    """
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]
    if frame.dtype != np.uint8:
        finite = frame[np.isfinite(frame)]
        lo = float(finite.min()) if finite.size else 0.0
        hi = float(finite.max()) if finite.size else 1.0
        span = hi - lo if hi > lo else 1.0
        frame = np.clip((np.nan_to_num(frame, nan=lo) - lo) / span, 0.0, 1.0)
        frame = (frame * 255.0).astype(np.uint8)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    return frame


def write_demo_to_mp4(hdf5_file: h5py.File, demo_name: str, input_key: str, output_dir: str, framerate: int) -> bool:
    """Write one demo's frames for ``input_key`` to an MP4. Returns False if absent."""
    obs_path = f"data/{demo_name}/obs/{input_key}"
    if obs_path not in hdf5_file:
        return False

    frames = hdf5_file[obs_path]
    output_path = os.path.join(output_dir, f"{demo_name}_{input_key}.mp4")
    with imageio.get_writer(output_path, fps=framerate, macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(_to_uint8_rgb(frame))
    print(f"Wrote {output_path} ({len(frames)} frame(s))")
    return True


def _sorted_demo_names(data_group: h5py.Group) -> list[str]:
    names = [k for k in data_group.keys() if k.startswith("demo_")]

    def _key(name: str) -> tuple[int, str]:
        suffix = name[len("demo_") :]
        return (0, f"{int(suffix):020d}") if suffix.isdigit() else (1, name)

    return sorted(names, key=_key)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with h5py.File(args.input_file, "r") as f:
        if "data" not in f:
            raise SystemExit(f"{args.input_file}: missing top-level 'data' group; not a record_demos HDF5 file")
        demo_names = _sorted_demo_names(f["data"])
        if args.demo_id is not None:
            wanted = f"demo_{args.demo_id}"
            if wanted not in demo_names:
                raise SystemExit(f"{args.input_file}: no {wanted} (found {len(demo_names)} demo(s))")
            demo_names = [wanted]
        print(f"Found {len(demo_names)} demonstration(s) in {args.input_file}")

        wrote_any = False
        for demo_name in demo_names:
            for input_key in args.input_keys:
                wrote_any |= write_demo_to_mp4(f, demo_name, input_key, args.output_dir, args.framerate)

    if not wrote_any:
        raise SystemExit(f"No frames written; none of {args.input_keys} exist under any demo's obs group.")


if __name__ == "__main__":
    main()

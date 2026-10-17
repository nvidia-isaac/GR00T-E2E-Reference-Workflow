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

"""Behavioral tests for the offline HDF5 dataset tools (merge / mp4 round-trip).

These tools have no simulator dependency; they only need ``h5py`` + ``imageio``
(the ``converter`` extra), so they run in host CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MERGE = REPO_ROOT / "scripts/merge_demos.py"
HDF5_TO_MP4 = REPO_ROOT / "scripts/hdf5_to_mp4.py"
MP4_TO_HDF5 = REPO_ROOT / "scripts/mp4_to_hdf5.py"

CAMERA_KEY = "robot_head_cam_rgb"


def write_dataset(
    path: Path, *, num_demos: int = 2, episode_len: int = 4, action_dim: int = 8, format_version: int = 1
) -> None:
    """Write a minimal record_demos-style HDF5 file with an RGB camera stream."""
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = format_version
        data = f.create_group("data")
        data.attrs["env_args"] = "{}"
        for i in range(num_demos):
            demo = data.create_group(f"demo_{i}")
            demo.attrs["num_samples"] = episode_len
            demo.attrs["success"] = True
            demo.create_dataset("actions", data=np.zeros((episode_len, action_dim), dtype=np.float32))
            obs = demo.create_group("obs")
            frames = np.full((episode_len, 16, 16, 3), i + 1, dtype=np.uint8)
            obs.create_dataset(CAMERA_KEY, data=frames)


def _run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, cwd=REPO_ROOT)


def _demo_names(path: Path) -> list[str]:
    with h5py.File(path, "r") as f:
        return sorted(f["data"].keys())


def test_merge_combines_and_renumbers(tmp_path: Path) -> None:
    a, b, out = tmp_path / "a.hdf5", tmp_path / "b.hdf5", tmp_path / "out.hdf5"
    write_dataset(a, num_demos=3, episode_len=5)
    write_dataset(b, num_demos=2, episode_len=7)

    result = _run(MERGE, "-o", str(out), str(a), str(b))
    assert result.returncode == 0, result.stderr

    assert _demo_names(out) == [f"demo_{i}" for i in range(5)]
    with h5py.File(out, "r") as f:
        assert int(f.attrs["format_version"]) == 1
        assert int(f["data"].attrs["total"]) == 3 * 5 + 2 * 7


def test_merge_dry_run_writes_nothing(tmp_path: Path) -> None:
    a, out = tmp_path / "a.hdf5", tmp_path / "out.hdf5"
    write_dataset(a)
    result = _run(MERGE, "-o", str(out), "--dry_run", str(a))
    assert result.returncode == 0, result.stderr
    assert not out.exists()


def test_merge_rejects_schema_mismatch(tmp_path: Path) -> None:
    a, b, out = tmp_path / "a.hdf5", tmp_path / "b.hdf5", tmp_path / "out.hdf5"
    write_dataset(a, action_dim=8)
    write_dataset(b, action_dim=6)
    result = _run(MERGE, "-o", str(out), str(a), str(b))
    assert result.returncode == 1
    assert not out.exists()


def test_mp4_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("imageio")
    src, vids, out = tmp_path / "src.hdf5", tmp_path / "vids", tmp_path / "aug.hdf5"
    write_dataset(src, num_demos=2, episode_len=6)

    export = _run(HDF5_TO_MP4, "--input_file", str(src), "--output_dir", str(vids))
    assert export.returncode == 0, export.stderr
    mp4s = sorted(vids.glob("*.mp4"))
    assert [p.name for p in mp4s] == [f"demo_{i}_{CAMERA_KEY}.mp4" for i in range(2)]

    reimport = _run(MP4_TO_HDF5, "--input_file", str(src), "--videos_dir", str(vids), "--output_file", str(out))
    assert reimport.returncode == 0, reimport.stderr

    # Original 2 demos preserved plus 2 augmented copies, each with a same-length stream.
    assert _demo_names(out) == [f"demo_{i}" for i in range(4)]
    with h5py.File(out, "r") as f:
        assert f[f"data/demo_2/obs/{CAMERA_KEY}"].shape[0] == 6
        assert "actions" in f["data/demo_2"]


def test_mp4_to_hdf5_preserves_output_on_failure(tmp_path: Path) -> None:
    pytest.importorskip("imageio")
    import imageio.v2 as imageio

    src, vids, out = tmp_path / "src.hdf5", tmp_path / "vids", tmp_path / "out.hdf5"
    write_dataset(src, num_demos=1, episode_len=6)
    vids.mkdir()
    # A video whose frame count (3) mismatches the source episode (6): augmentation fails.
    with imageio.get_writer(vids / f"demo_0_{CAMERA_KEY}.mp4", fps=30, macro_block_size=None) as writer:
        for _ in range(3):
            writer.append_data(np.zeros((16, 16, 3), dtype=np.uint8))
    # Pre-existing output that --overwrite must not destroy on a mid-run failure.
    with h5py.File(out, "w") as f:
        f.attrs["sentinel"] = "original"

    result = _run(
        MP4_TO_HDF5, "--input_file", str(src), "--videos_dir", str(vids), "--output_file", str(out), "--overwrite"
    )
    assert result.returncode != 0

    # The prior output survives intact and no temp file is left behind.
    with h5py.File(out, "r") as f:
        assert f.attrs["sentinel"] == "original"
    assert not (tmp_path / "out.hdf5.tmp").exists()


def test_mp4_to_hdf5_rejects_resized_video(tmp_path: Path) -> None:
    pytest.importorskip("imageio")
    import imageio.v2 as imageio

    src, vids, out = tmp_path / "src.hdf5", tmp_path / "vids", tmp_path / "out.hdf5"
    write_dataset(src, num_demos=1, episode_len=6)  # camera frames are 16x16x3
    vids.mkdir()
    # Right frame count but wrong resolution (32x32): must be rejected as a schema change.
    with imageio.get_writer(vids / f"demo_0_{CAMERA_KEY}.mp4", fps=30, macro_block_size=None) as writer:
        for _ in range(6):
            writer.append_data(np.zeros((32, 32, 3), dtype=np.uint8))

    result = _run(MP4_TO_HDF5, "--input_file", str(src), "--videos_dir", str(vids), "--output_file", str(out))
    assert result.returncode != 0
    assert "does not match source" in result.stderr
    assert not out.exists()
    assert not (tmp_path / "out.hdf5.tmp").exists()


def test_merge_rejects_output_path_as_input(tmp_path: Path) -> None:
    a, out = tmp_path / "a.hdf5", tmp_path / "out.hdf5"
    write_dataset(a)
    # 'out' does not exist yet; passing it as an input must still be caught as a collision.
    result = _run(MERGE, "-o", str(out), str(a), str(out))
    assert result.returncode == 2
    assert "equals an input path" in result.stderr


@pytest.mark.parametrize("camera_group", ["obs", "camera_obs"])
def test_mp4_export_camera_layouts(tmp_path: Path, camera_group: str) -> None:
    pytest.importorskip("imageio")
    import imageio.v2 as imageio

    src, vids = tmp_path / "src.hdf5", tmp_path / "vids"
    with h5py.File(src, "w") as f:
        # Match the reported Arena recording's frame count and resolution.
        f.create_dataset(
            f"data/demo_0/{camera_group}/{CAMERA_KEY}",
            data=np.full((129, 480, 640, 3), 90, dtype=np.uint8),
        )

    result = _run(HDF5_TO_MP4, "--input_file", str(src), "--output_dir", str(vids))
    assert result.returncode == 0, result.stderr
    with imageio.get_reader(vids / f"demo_0_{CAMERA_KEY}.mp4") as reader:
        assert reader.count_frames() == 129
        frame = reader.get_data(0)
        assert frame.shape == (480, 640, 3)
        np.testing.assert_allclose(frame, 90, atol=3)


def test_mp4_export_mixed_layouts_and_custom_key(tmp_path: Path) -> None:
    pytest.importorskip("imageio")
    import imageio.v2 as imageio

    src, vids = tmp_path / "src.hdf5", tmp_path / "vids"
    with h5py.File(src, "w") as f:
        for index, group in enumerate(("obs", "camera_obs")):
            f.create_dataset(
                f"data/demo_{index}/{group}/wrist_rgb",
                data=np.full((4, 16, 16, 3), 90 + 30 * index, dtype=np.uint8),
            )
        # Another camera in obs must not hide the requested Arena stream.
        f.create_dataset(f"data/demo_1/obs/{CAMERA_KEY}", shape=(4, 16, 16, 3), dtype="u1")
        # Preserve the existing obs precedence when both layouts have the key.
        f.create_dataset("data/demo_0/camera_obs/wrist_rgb", shape=(7, 16, 16, 3), dtype="u1")

    result = _run(HDF5_TO_MP4, "--input_file", str(src), "--output_dir", str(vids), "--input_keys", "wrist_rgb")
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in vids.glob("*.mp4")) == [f"demo_{i}_wrist_rgb.mp4" for i in range(2)]
    for index in range(2):
        with imageio.get_reader(vids / f"demo_{index}_wrist_rgb.mp4") as reader:
            assert reader.count_frames() == 4
            np.testing.assert_allclose(reader.get_data(0), 90 + 30 * index, atol=3)


def test_mp4_export_missing_camera_reports_both_layouts(tmp_path: Path) -> None:
    pytest.importorskip("imageio")
    src, vids = tmp_path / "src.hdf5", tmp_path / "vids"
    with h5py.File(src, "w") as f:
        f.create_group("data/demo_0/obs")
        f.create_group("data/demo_0/camera_obs")

    result = _run(HDF5_TO_MP4, "--input_file", str(src), "--output_dir", str(vids))
    assert result.returncode != 0
    assert "obs or camera_obs" in result.stderr
    assert not list(vids.glob("*.mp4"))

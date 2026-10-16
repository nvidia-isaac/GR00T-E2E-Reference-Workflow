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

"""Safety checks for workflow input and output artifacts."""

from __future__ import annotations

import ctypes
import hashlib
import os
import tempfile
from pathlib import Path

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def file_sha256(path: str | Path) -> str:
    """Return a file's SHA-256 without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recorder_hdf5_output_path(path: str | Path) -> Path:
    """Return the path Isaac Lab's recorder will create for an output argument."""

    requested = Path(path).expanduser()
    return requested.parent / f"{requested.stem}.hdf5"


def require_distinct_artifact_paths(source: str | Path, destination: str | Path, *, destination_label: str) -> None:
    """Reject an output path that would replace its source artifact."""

    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    same_existing_file = source_path.exists() and destination_path.exists() and source_path.samefile(destination_path)
    if source_path == destination_path or same_existing_file:
        raise ValueError(f"{destination_label} must differ from input dataset: {source_path}")


def atomic_write_text(destination: str | Path, text: str) -> None:
    """Replace one text artifact through an exclusive sibling temporary file."""

    destination_path = Path(destination).expanduser()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination_path.parent,
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = Path(stream.name)
        os.replace(temporary_path, destination_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def publish_directory_without_replace(source: Path, destination: Path) -> None:
    """Atomically publish a completed directory without replacing one."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory publication requires Linux renameat2")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
        )
        != 0
    ):
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))


__all__ = [
    "atomic_write_text",
    "file_sha256",
    "publish_directory_without_replace",
    "recorder_hdf5_output_path",
    "require_distinct_artifact_paths",
]

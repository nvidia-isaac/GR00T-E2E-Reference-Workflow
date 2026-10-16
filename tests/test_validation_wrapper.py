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

"""Behavioral tests for the repository validation entry point."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def test_simulation_wrapper_propagates_tee_failure(tmp_path: Path) -> None:
    """A logging failure must fail validation even when simulation succeeds."""

    fake_python = tmp_path / "python"
    _write_executable(
        fake_python,
        """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

if sys.argv[1] == "-c":
    raise SystemExit(0)

Path(os.environ["RESULT_PATH"]).write_text('{"status": "ok"}\\n', encoding="utf-8")
""",
    )
    _write_executable(
        tmp_path / "tee",
        """#!/bin/sh
while IFS= read -r line; do
    :
done
exit 7
""",
    )

    result_path = tmp_path / "metrics.json"
    log_path = tmp_path / "validation.log"
    junit_path = tmp_path / "junit.xml"
    environment = os.environ.copy()
    environment.update(
        {
            "JUNIT_XML": str(junit_path),
            "PATH": f"{tmp_path}{os.pathsep}{environment['PATH']}",
            "PYTHON_BIN": str(fake_python),
            "STATIC_APPLE_VALIDATION_LOG": str(log_path),
            "STATIC_APPLE_VALIDATION_RESULT": str(result_path),
        }
    )

    completed = subprocess.run(
        [str(REPO_ROOT / "tools" / "validate_repository.sh"), "sim-static-apple"],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 7
    assert result_path.is_file()
    assert not log_path.exists()
    assert 'failures="1"' in junit_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("missing_stats", (False, True), ids=("not-a-directory", "missing-stats-entrypoint"))
def test_h2_validation_rejects_an_invalid_gr00t_root_before_creating_results(
    tmp_path: Path,
    missing_stats: bool,
) -> None:
    gr00t_root = tmp_path / "Isaac-GR00T"
    if missing_stats:
        gr00t_root.mkdir()
    result_root = tmp_path / "results"
    environment = os.environ.copy()
    environment.update(
        {
            "ARENA_H2_RESULT_DIR": str(result_root),
            "ISAAC_GR00T_ROOT": str(gr00t_root),
        }
    )

    completed = subprocess.run(
        [str(REPO_ROOT / "tools" / "validate_arena_h2.sh")],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    expected = "missing gr00t/data/stats.py" if missing_stats else "is not a directory"
    assert expected in completed.stderr
    assert not result_root.exists()

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

"""The standalone GR00T launcher must not sync into an inherited E2E environment."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("arguments", [["--check"], ["--", "python", "example.py"]])
@pytest.mark.parametrize("relative_root", [False, True])
def test_gr00t_environment_overrides_stay_inside_gr00t(tmp_path, arguments, relative_root):
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "gr00t"
    runtime_bin = root / ".venv/bin"
    runtime_bin.mkdir(parents=True)
    (root / "pyproject.toml").write_text("")
    (runtime_bin / "python").write_text("#!/bin/sh\nexit 0\n")
    (runtime_bin / "python").chmod(0o755)
    tools = tmp_path / "bin"
    tools.mkdir()
    uv = tools / "uv"
    uv.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$UV_PROJECT_ENVIRONMENT" >> "$GR00T_TEST_CAPTURE"\n'
        'case "$*" in *tomllib*) printf "test-revision\\n";; esac\n'
    )
    uv.chmod(0o755)
    git = tools / "git"
    git.write_text('#!/bin/sh\ncase "$*" in *rev-parse*) printf "test-revision\\n";; esac\n')
    git.chmod(0o755)
    capture = tmp_path / "environments.txt"
    inherited = tmp_path / "e2e-runtime"
    result = subprocess.run(
        [
            "bash",
            str(repo / "tools/gr00t_runtime.sh"),
            "--gr00t-root",
            "gr00t" if relative_root else str(root),
            *arguments,
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(tools) + os.pathsep + os.environ["PATH"],
            "UV_BIN": str(uv),
            "UV_PROJECT_ENVIRONMENT": str(inherited),
            "GR00T_TEST_CAPTURE": str(capture),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    # The repository pin check uses E2E; only the GR00T command uses GR00T's env.
    assert capture.read_text().splitlines() == [str(inherited), str(root / ".venv")]

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

"""Exercise published inference admission without a simulator or model weights."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def server_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    """Keep the real launcher; isolate external Git, Hub, and evaluator boundaries."""
    repo = tmp_path / "repo"
    tools = repo / "tools"
    tools.mkdir(parents=True)
    shutil.copy(REPO_ROOT / "tools/launch_gr00t_server.sh", tools)
    pins = (REPO_ROOT / "runtime.toml").read_text()
    revision = tomllib.loads(pins)["tested"]["isaac_gr00t_revision"]
    (repo / "runtime.toml").write_text(pins)
    gr00t = tmp_path / "gr00t"
    (gr00t / "gr00t/eval").mkdir(parents=True)
    (gr00t / "gr00t/eval/run_gr00t_server.py").touch()
    (gr00t / "pyproject.toml").touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gr00t_bin = gr00t / ".venv/bin"
    gr00t_bin.mkdir(parents=True)

    def executable(path: Path, content: str) -> None:
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + content)
        path.chmod(0o755)

    executable(bin_dir / "git", f'if [[ "$3" == "rev-parse" ]]; then echo {revision}; fi\n')
    executable(bin_dir / "uv", f'shift 6\nexec "{sys.executable}" "$@"\n')
    executable(gr00t_bin / "hf", "exit 0\n")
    # Only the Hub-access probe uses the external interpreter in --check mode.
    executable(gr00t_bin / "python", "exit 0\n")
    executable(tools / "launch_gr00t_eval.sh", 'echo "reached modality preflight"\n')
    model = tmp_path / "model"
    model.mkdir()
    files = {
        "config.json": {"model_type": "Gr00tN1d7", "action_horizon": 40},
        "processor_config.json": {
            "processor_kwargs": {"modality_configs": {"new_embodiment": {"action": {"delta_indices": list(range(40))}}}}
        },
        "statistics.json": {},
        "embodiment_id.json": {},
        "model.safetensors.index.json": {"weight_map": {"example": "model-00001-of-00001.safetensors"}},
    }
    for name, value in files.items():
        (model / name).write_text(json.dumps(value))
    (model / "model-00001-of-00001.safetensors").touch()
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", UV_BIN=str(bin_dir / "uv"))
    return repo, model, env


def run_server(fixture: tuple[Path, Path, dict[str, str]], robot: str, source: str) -> subprocess.CompletedProcess[str]:
    repo, model, env = fixture
    return subprocess.run(
        [
            str(repo / "tools/launch_gr00t_server.sh"),
            "--arena-root",
            str(repo.parent / "arena"),
            "--gr00t-root",
            str(repo.parent / "gr00t"),
            "--model-path",
            str(model),
            "--robot",
            robot,
            "--checkpoint-source",
            source,
            "--check",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def stamp_download(fixture: tuple[Path, Path, dict[str, str]], robot: str) -> None:
    repo, model, _env = fixture
    pin = tomllib.loads((repo / "runtime.toml").read_text())["artifacts"][robot]["model_revision"]
    download = model / ".cache/huggingface/download"
    download.mkdir(parents=True)
    for file in model.iterdir():
        if file.is_file():
            (download / f"{file.name}.metadata").write_text(f"{pin}\netag\n0\n")


@pytest.mark.parametrize("robot", ["g1", "h2"])
def test_published_inference_checkpoint_needs_no_training_resume_state(server_fixture, robot: str) -> None:
    stamp_download(server_fixture, robot)
    result = run_server(server_fixture, robot, "published")
    assert result.returncode == 0, result.stderr
    assert "reached modality preflight" in result.stdout
    result = run_server(server_fixture, robot, "trained")
    assert result.returncode == 2
    assert "trainer_state.json and experiment_cfg/config.yaml" in result.stderr


def test_published_checkpoint_rejects_other_robot_revision(server_fixture) -> None:
    stamp_download(server_fixture, "g1")
    result = run_server(server_fixture, "h2", "published")
    assert result.returncode == 2
    assert "invalid file: config.json" in result.stderr
    assert "reached modality preflight" not in result.stdout


@pytest.mark.parametrize("defect", ["missing-metadata", "wrong-revision", "missing-shard", "wrong-horizon"])
def test_published_checkpoint_preserves_file_and_model_checks(server_fixture, defect: str) -> None:
    stamp_download(server_fixture, "h2")
    _repo, model, _env = server_fixture
    if defect == "missing-metadata":
        (model / ".cache/huggingface/download/config.json.metadata").unlink()
    elif defect == "wrong-revision":
        (model / ".cache/huggingface/download/config.json.metadata").write_text("wrong\netag\n0\n")
    elif defect == "missing-shard":
        (model / "model-00001-of-00001.safetensors").unlink()
    else:
        (model / "config.json").write_text(json.dumps({"model_type": "Gr00tN1d7", "action_horizon": 50}))
    result = run_server(server_fixture, "h2", "published")
    assert result.returncode != 0
    assert "reached modality preflight" not in result.stdout
    assert any(
        message in result.stderr for message in ("invalid file", "missing required file", "Expected action_horizon 40")
    )

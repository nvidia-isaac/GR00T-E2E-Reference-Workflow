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

"""Contracts for the self-contained uv Isaac Lab runtime."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_project_declares_the_pinned_uv_isaac_runtime() -> None:
    """A clone must resolve the complete simulator stack without Docker."""

    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime_dependencies = project["dependency-groups"]["sim"]
    dependency_text = "\n".join(runtime_dependencies)

    assert project["project"]["requires-python"] == ">=3.12,<3.13"
    assert "isaaclab[isaacsim,all]==3.0.0b2" in dependency_text
    assert "torch==2.11.0" in dependency_text
    assert "torchvision==0.26.0" in dependency_text
    assert "pin==4.0.0" in dependency_text
    assert "pin-pink==3.3.0" in dependency_text
    assert "qpsolvers==4.12.0" in dependency_text
    assert "daqp==0.8.5" in dependency_text
    assert project["tool"]["uv"]["default-groups"] == ["sim", "host-tools", "arena", "docs"]

    indexes = {entry["name"]: entry for entry in project["tool"]["uv"]["index"]}
    assert indexes["nvidia"]["url"] == "https://pypi.nvidia.com"
    assert indexes["nvidia"]["explicit"] is False
    assert indexes["pytorch-cu128"]["url"] == "https://download.pytorch.org/whl/cu128"
    assert indexes["pytorch-cu128"]["explicit"] is True
    for package in ("torch", "torchvision"):
        assert project["tool"]["uv"]["sources"][package] == {"index": "pytorch-cu128"}
    assert project["tool"]["uv"]["prerelease"] == "allow"


def test_uv_runtime_replaces_docker_and_external_checkout_contracts() -> None:
    """Runtime helpers must execute with the uv interpreter from a fresh clone."""

    assert (REPO_ROOT / "uv.lock").is_file()
    assert not (REPO_ROOT / "docker").exists()
    assert not (REPO_ROOT / ".dockerignore").exists()

    validation = (REPO_ROOT / "tools/validate_repository.sh").read_text(encoding="utf-8")
    verifier = (REPO_ROOT / "tools/verify_runtime.py").read_text(encoding="utf-8")
    locomotion = (REPO_ROOT / "scripts/locomotion_with_agile.py").read_text(encoding="utf-8")
    runtime = tomllib.loads((REPO_ROOT / "runtime.toml").read_text(encoding="utf-8"))["tested"]
    assert "/isaac-sim/python.sh" not in validation
    assert "PYTHON_BIN=${PYTHON_BIN:-python}" in validation
    assert "ISAACLAB_ROOT" not in validation
    assert "linux x86_64" in verifier
    assert "torch.cuda.is_available" in verifier
    assert runtime["torch"] == "2.11.0+cu128"
    assert runtime["torchvision"] == "0.26.0+cu128"
    assert 'make_env_cfg(wbc_mode="agile"' in locomotion


def test_built_in_task_validation_modes_remain_available() -> None:
    """The built-in task remains available as the Arena comparison oracle."""

    validation = (REPO_ROOT / "tools/validate_repository.sh").read_text(encoding="utf-8")
    for mode in ("sim-static-apple", "sim-agile-locomotion", "sim-recorder"):
        assert f"  {mode})" in validation

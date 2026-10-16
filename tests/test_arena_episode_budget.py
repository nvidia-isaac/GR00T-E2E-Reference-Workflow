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

"""Test Arena budget and artifact handling without loading Isaac Sim on the host."""

import ast
import math
from pathlib import Path
from typing import Any

import h5py
import pytest

from e2e_robot_workflows.autodata.arena import validate_generated_episode_counts
from e2e_robot_workflows.data.dataset_manifest import (
    ARENA_PINK_PROFILE,
    MANIFEST_ATTRIBUTE,
    build_manifest,
    file_sha256,
    read_manifest,
    stamp_autodata_dataset,
    write_manifest,
)


@pytest.fixture
def resolve_budget():
    source = Path(__file__).resolve().parents[1] / "src/e2e_robot_workflows/arena/external_environment.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_resolve_episode_length_s"
    )
    namespace = {"math": math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name]


def test_default_episode_budget_is_unchanged(resolve_budget):
    assert resolve_budget(None, 6.0) == 6.0


def test_explicit_episode_budget(resolve_budget):
    assert resolve_budget(30.0, 6.0) == 30.0


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "30"])
def test_invalid_episode_budget_is_rejected(resolve_budget, value):
    with pytest.raises(ValueError, match="positive and finite"):
        resolve_budget(value, 6.0)


@pytest.mark.parametrize("status", ["ok", "error"])
def test_generation_artifacts_preserve_effective_seed_and_final_hash(tmp_path: Path, status: str) -> None:
    source_path = tmp_path / "source.hdf5"
    output_path = tmp_path / "generated.hdf5"
    for path in (source_path, output_path):
        with h5py.File(path, "w") as dataset:
            episode = dataset.create_group("data/demo_0")
            episode.attrs["success"] = True
            episode.create_dataset("actions", shape=(3, ARENA_PINK_PROFILE.width), dtype="f4")
    write_manifest(source_path, build_manifest(producer="recording", profile=ARENA_PINK_PROFILE, seed=17))
    unstamped_hash = file_sha256(output_path)

    source = Path(__file__).resolve().parents[1] / "scripts/generate_arena_dataset.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_record_artifacts"
    )
    namespace = {
        "Any": Any,
        "OUTPUT_PATH": output_path,
        "FAILED_OUTPUT_PATH": tmp_path / "generated_failed.hdf5",
        "SNAPSHOT_SOURCE_PATH": source_path,
        "validate_generated_episode_counts": validate_generated_episode_counts,
        "stamp_autodata_dataset": stamp_autodata_dataset,
        "file_sha256": file_sha256,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    result = {"status": status, "num_success": 1, "num_failures": 0}
    if status == "ok":
        result["generation_seed"] = 731

    namespace[function.name](result)

    if status == "ok":
        assert read_manifest(output_path)["seed"] == 731
        assert file_sha256(output_path) != unstamped_hash
    else:
        with h5py.File(output_path, "r") as dataset:
            assert MANIFEST_ATTRIBUTE not in dataset.attrs
        assert file_sha256(output_path) == unstamped_hash
    assert result["artifacts"] == {"generated": {"path": str(output_path), "sha256": file_sha256(output_path)}}

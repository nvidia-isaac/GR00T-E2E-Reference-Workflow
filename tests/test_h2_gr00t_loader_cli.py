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

"""Artifact admission for the H2 GR00T loader check."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LOADER_CHECK = REPO_ROOT / "tools/check_h2_gr00t_dataset.py"
SPEC = importlib.util.spec_from_file_location("_test_check_h2_gr00t_dataset", LOADER_CHECK)
assert SPEC is not None and SPEC.loader is not None
LOADER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOADER_MODULE)
_validate_statistics = LOADER_MODULE._validate_statistics
_dataset_totals = LOADER_MODULE._dataset_totals

EXPECTED_STATISTIC_WIDTHS = {
    "state": {
        "left_leg": 6,
        "right_leg": 6,
        "waist": 3,
        "head": 2,
        "left_arm": 7,
        "right_arm": 7,
        "left_hand": 22,
        "right_hand": 22,
    },
    "action": {
        "left_arm": 7,
        "right_arm": 7,
        "left_hand": 22,
        "right_hand": 22,
        "navigate_command": 3,
    },
}


def _valid_statistics() -> dict:
    return {
        modality: {
            group: {
                statistic: np.zeros(width, dtype=np.float32)
                for statistic in ("mean", "std", "min", "max", "q01", "q99")
            }
            for group, width in groups.items()
        }
        for modality, groups in EXPECTED_STATISTIC_WIDTHS.items()
    }


def test_h2_training_statistics_validate_every_group_width() -> None:
    assert _validate_statistics(_valid_statistics()) == EXPECTED_STATISTIC_WIDTHS


@pytest.mark.parametrize("invalid", ("truncated-navigation", "wrong-rank", "non-finite"))
def test_h2_training_statistics_reject_invalid_values(invalid: str) -> None:
    statistics = _valid_statistics()
    if invalid == "truncated-navigation":
        statistics["action"]["navigate_command"]["mean"] = np.zeros(2, dtype=np.float32)
    elif invalid == "wrong-rank":
        statistics["action"]["navigate_command"]["mean"] = np.zeros((1, 3), dtype=np.float32)
    else:
        statistics["state"]["left_leg"]["std"][0] = np.nan

    with pytest.raises(RuntimeError):
        _validate_statistics(statistics)


@pytest.mark.parametrize(
    "payload",
    (
        "not JSON",
        "{}",
        "[]",
        '{"total_episodes": true, "total_frames": 17}',
        '{"total_episodes": 2, "total_frames": 0}',
    ),
)
def test_dataset_totals_reject_invalid_metadata(tmp_path: Path, payload: str) -> None:
    metadata = tmp_path / "meta" / "info.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(payload, encoding="utf-8")

    with pytest.raises(RuntimeError):
        _dataset_totals(tmp_path)


@pytest.mark.parametrize("collision", ("dataset", "config"))
def test_loader_result_cannot_replace_an_input(tmp_path: Path, collision: str) -> None:
    dataset = tmp_path / "dataset"
    metadata = dataset / "meta" / "info.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("dataset metadata", encoding="utf-8")
    modality_config = tmp_path / "modality.py"
    modality_config.write_text("modality config", encoding="utf-8")
    result_path = metadata if collision == "dataset" else modality_config

    result = subprocess.run(
        [
            sys.executable,
            str(LOADER_CHECK),
            "--dataset",
            str(dataset),
            "--modality-config",
            str(modality_config),
            "--expected-episodes",
            "1",
            "--expected-frames",
            "1",
            "--result",
            str(result_path),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2
    assert "--result must" in result.stderr
    assert metadata.read_text(encoding="utf-8") == "dataset metadata"
    assert modality_config.read_text(encoding="utf-8") == "modality config"

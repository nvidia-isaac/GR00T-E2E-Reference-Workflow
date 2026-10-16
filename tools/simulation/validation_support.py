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

"""Shared assertions and metadata for GPU simulation validations."""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def package_version(distribution: str) -> str:
    """Return an installed distribution version or ``unknown``."""

    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def write_result(path: str, result: dict[str, Any]) -> None:
    """Atomically write a structured validation result."""

    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = result_path.with_name(f".{result_path.name}.tmp")
    temporary_path.write_text(f"{json.dumps(result, indent=2, sort_keys=True)}\n", encoding="utf-8")
    temporary_path.replace(result_path)


def assert_finite(name: str, value: Any, torch: Any) -> None:
    """Recursively reject non-finite tensors in an observation or state tree."""

    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all().item()):
            raise AssertionError(f"{name} contains a non-finite tensor")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            assert_finite(f"{name}.{key}", child, torch)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            assert_finite(f"{name}[{index}]", child, torch)


def assert_not_done(terminated: Any, truncated: Any, *, context: str) -> None:
    """Reject unexpected termination or truncation during validation."""

    if bool(terminated.any().item()):
        raise AssertionError(f"the environment terminated during the {context}")
    if bool(truncated.any().item()):
        raise AssertionError(f"the environment timed out during the {context}")


def assert_zero_tensor(name: str, value: Any, torch: Any) -> None:
    """Require a finite tensor to be exactly zero."""

    assert_finite(name, value, torch)
    maximum = float(value.abs().max().item())
    if maximum != 0.0:
        raise AssertionError(f"{name} was not cleared by reset (maximum magnitude {maximum:.6g})")


def assert_action_manager_reset(env: Any, observations: Any, torch: Any) -> None:
    """Require a reset environment and its action manager to be clear."""

    assert_finite("second_reset_observations", observations, torch)
    if not bool((env.episode_length_buf == 0).all().item()):
        raise AssertionError(f"episode length was not cleared by reset: {env.episode_length_buf.tolist()}")
    assert_zero_tensor("action_manager.action", env.action_manager.action, torch)
    assert_zero_tensor("action_manager.prev_action", env.action_manager.prev_action, torch)


def collect_runtime_metadata(torch: Any) -> dict[str, Any]:
    """Collect portable runtime versions after the Simulation App starts."""

    from isaaclab.utils.version import get_isaac_sim_version

    return {
        "architecture": platform.machine(),
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "isaacsim_distribution_version": package_version("isaacsim"),
        "isaac_sim_version": str(get_isaac_sim_version()),
        "isaaclab_version": package_version("isaaclab"),
        "isaaclab_tasks_version": package_version("isaaclab-tasks"),
        "pin_pink_version": package_version("pin-pink"),
        "torch_version": torch.__version__,
    }


def quaternion_distance_rad(quaternion: Any, reference: Any, torch: Any) -> float:
    """Return the maximum sign-invariant angular distance between quaternions."""

    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    reference = torch.nn.functional.normalize(reference, dim=-1)
    quaternion_dot = torch.sum(quaternion * reference, dim=-1)
    aligned = torch.where((quaternion_dot < 0.0).unsqueeze(-1), -quaternion, quaternion)
    chord = torch.linalg.vector_norm(aligned - reference, dim=-1)
    return float((4.0 * torch.asin((chord / 2.0).clamp(max=1.0))).max().item())

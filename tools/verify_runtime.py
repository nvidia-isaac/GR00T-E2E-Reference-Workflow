#!/usr/bin/env python3
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

"""Validate the uv-managed Isaac Lab runtime needed by this application."""

from __future__ import annotations

import inspect
import json
import os
import platform
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

from check_arena_pin import ArenaPinError, verify_arena_pin

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    contract = tomllib.loads((REPO_ROOT / "runtime.toml").read_text(encoding="utf-8"))["tested"]
    if sys.platform != "linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise SystemExit("This uv runtime supports linux x86_64 only.")
    if f"{sys.version_info.major}.{sys.version_info.minor}" != contract["python"]:
        raise SystemExit(f"Python {contract['python']} is required by the locked Isaac Sim runtime.")

    expected_distributions = {
        "isaaclab": contract["isaaclab"],
        "isaacsim": contract["isaacsim"],
        "torch": contract["torch"],
        "torchvision": contract["torchvision"],
        "pin": contract["pin"],
        "pin-pink": contract["pin_pink"],
        "qpsolvers": contract["qpsolvers"],
        "daqp": contract["daqp"],
    }
    actual = {distribution: version(distribution) for distribution in expected_distributions}
    mismatches = {
        distribution: {"expected": expected, "actual": actual[distribution]}
        for distribution, expected in expected_distributions.items()
        if actual[distribution] != expected
    }
    if mismatches:
        raise SystemExit(f"Locked Isaac Lab runtime version mismatch: {json.dumps(mismatches, sort_keys=True)}")

    import daqp
    import isaaclab  # noqa: F401
    import isaaclab_teleop  # noqa: F401
    import pinocchio  # noqa: F401
    import torch
    from isaaclab.app import AppLauncher  # noqa: F401
    from qpsolvers import available_solvers

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Install a compatible NVIDIA driver and run on a supported GPU host.")
    if "daqp" not in available_solvers or "primal_start" not in inspect.signature(daqp.solve).parameters:
        raise SystemExit("The locked DAQP/Pink solver stack is unavailable.")
    # Arena is an external checkout rather than a locked distribution, so it can only be
    # verified where it is present. Absent E2E_ARENA_ROOT the Arena composition and
    # evaluation paths cannot run at all, so reporting "skipped" is honest, not lenient.
    arena_root = os.environ.get("E2E_ARENA_ROOT")
    arena_status: str | dict[str, str] = "skipped"
    if arena_root:
        try:
            arena_status = verify_arena_pin(Path(arena_root).expanduser().resolve())
        except ArenaPinError as exc:
            raise SystemExit(str(exc)) from exc
    print(
        json.dumps({"status": "ok", "platform": contract["platform"], "arena": arena_status, **actual}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

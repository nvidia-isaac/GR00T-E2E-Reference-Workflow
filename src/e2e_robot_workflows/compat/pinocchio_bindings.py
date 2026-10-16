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

"""Pinocchio eigenpy binding warm-up (Isaac Lab issue #4090).

Isaac Lab's ``PinkIKController`` builds a Pinocchio model *after* the sim/scene
loads. The scene / USD->URDF importer native libraries can clobber Pinocchio's
eigenpy type registry, so a later ``model.names.tolist()`` inside
``PinkKinematicsConfiguration`` raises::

    TypeError: No Python class registered for C++ class std::vector<std::string>

Importing Pinocchio and exercising that ``std::vector<std::string>`` conversion
*before* Isaac Sim loads registers the converter so it survives scene loading.

Call :func:`warmup_pinocchio_bindings` at the very top of every entry point that
launches Isaac Sim -- crucially, **before** ``from isaaclab.app import
AppLauncher`` -- so the registration happens first. See
https://github.com/isaac-sim/IsaacLab/issues/4090.
"""

from __future__ import annotations

from loguru import logger


def warmup_pinocchio_bindings(*, verbose: bool = True) -> bool:
    """Register Pinocchio's ``std::vector<std::string>`` converter early.

    Best-effort and side-effect-only: returns ``True`` if the warm-up ran,
    ``False`` if Pinocchio is absent or the warm-up failed. Never raises, so it
    can be called unconditionally at the top of an entry point without risking
    that it blocks startup.
    """

    try:
        import pinocchio

        pinocchio.buildSampleModelManipulator().names.tolist()
    except Exception as exc:  # noqa: BLE001 - diagnostic warm-up only
        if verbose:
            logger.info(f"[e2e] Pinocchio binding warm-up skipped: {exc}")
        return False
    if verbose:
        logger.info("[e2e] Pinocchio binding warm-up ok")
    return True

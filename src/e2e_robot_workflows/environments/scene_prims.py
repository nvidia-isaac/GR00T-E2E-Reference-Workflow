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

"""Startup edits to referenced scene prims, shared by the tasks that compose a cell.

A referenced asset arrives whole. The packing table brings a container prop with
it and the warehouse brings its own fixtures, and a task that wants the cell
without them has to switch the prims off on the stage rather than ask the
reference not to bring them. Which prims those are is declared per environment;
this is only the mechanism.

Deliberately free of Isaac Lab imports so it stays host-importable: it is handed
an environment at startup and reads the stage off it.
"""

from __future__ import annotations

from loguru import logger


def deactivate_scene_prims(
    env,
    env_ids,
    asset_root_name: str,
    prim_relative_paths: tuple[str, ...],
) -> None:
    """Deactivate selected asset-relative referenced prims before startup."""

    del env_ids
    stage = env.sim.stage
    for env_prim_path in env.scene.env_prim_paths:
        for prim_relative_path in prim_relative_paths:
            prim_path = f"{env_prim_path}/{asset_root_name}/{prim_relative_path}"
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                stage.OverridePrim(prim_path).SetActive(False)
            else:
                logger.warning(
                    f"deactivate_scene_prims: prim not found at '{prim_path}'; the asset may still be visible."
                )


__all__ = ["deactivate_scene_prims"]

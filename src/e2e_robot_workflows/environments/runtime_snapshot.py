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

"""Runtime snapshots used to compare two realizations of one environment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from e2e_robot_workflows.environments.static_apple import StaticAppleEnvironmentDefinition

ENVIRONMENT_SNAPSHOT_SETTLE_STEPS = 50


def _plain_tensor(value: Any) -> list[float] | None:
    tensor = getattr(value, "torch", value)
    try:
        row = tensor[0, :7].detach().cpu().tolist()
    except (AttributeError, IndexError, TypeError):
        return None
    return [float(item) for item in row]


def _entity_cfg_snapshot(entity_cfg: Any) -> dict[str, Any]:
    spawn = getattr(entity_cfg, "spawn", None)
    initial_state = getattr(entity_cfg, "init_state", None)
    scale = getattr(spawn, "scale", None)
    if scale is None and getattr(spawn, "usd_path", None) is not None:
        scale = (1.0, 1.0, 1.0)
    return {
        "usd": getattr(spawn, "usd_path", None),
        "scale": [float(value) for value in scale] if scale is not None else None,
        "initial_position": (
            [float(value) for value in initial_state.pos]
            if initial_state is not None and getattr(initial_state, "pos", None) is not None
            else None
        ),
        "initial_rotation_xyzw": (
            [float(value) for value in initial_state.rot]
            if initial_state is not None and getattr(initial_state, "rot", None) is not None
            else None
        ),
    }


def collect_environment_snapshot(
    *,
    definition: StaticAppleEnvironmentDefinition,
    mode: dict[str, Any],
    env: Any,
    role_to_scene_entity: dict[str, str],
) -> dict[str, Any]:
    """Collect normalized declared and realized values from a live environment."""

    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    cfg = unwrapped.cfg
    entities: dict[str, Any] = {}
    for role, entity_name in role_to_scene_entity.items():
        entity_cfg = getattr(cfg.scene, entity_name)
        entity_snapshot = _entity_cfg_snapshot(entity_cfg)
        try:
            entity = unwrapped.scene[entity_name]
        except (KeyError, TypeError):
            entity = None
        data = getattr(entity, "data", None)
        runtime_state = _plain_tensor(getattr(data, "root_state_w", None))
        if runtime_state is not None:
            entity_snapshot["runtime_root_pose_wxyz"] = runtime_state
        entities[role] = entity_snapshot

    return {
        "contract": definition.contract_snapshot(),
        "mode": mode,
        "realized": {
            "physics_dt_s": float(cfg.sim.dt),
            "decimation": int(cfg.decimation),
            "episode_length_s": float(cfg.episode_length_s),
            "action_dim": int(unwrapped.action_manager.total_action_dim),
            "scene_entities": sorted(str(name) for name in unwrapped.scene.keys()),
            "entities": entities,
        },
    }


def write_environment_snapshot(path: str | Path, snapshot: dict[str, Any]) -> None:
    """Write a deterministic, human-readable environment snapshot."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")

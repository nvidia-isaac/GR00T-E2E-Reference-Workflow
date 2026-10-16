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

"""Replay helpers shared by simulator entry points."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


def recorded_state_count(states: Mapping[str, Any]) -> int:
    """Return the common frame count across a nested recorded-state tree."""

    lengths: set[int] = set()

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            if not value:
                raise ValueError("Recorded state contains an empty mapping")
            for child in value.values():
                collect(child)
            return
        try:
            lengths.add(len(value))
        except TypeError as error:
            raise ValueError("Recorded state leaves must be frame sequences") from error

    collect(states)
    if len(lengths) != 1:
        raise ValueError(f"Recorded state leaves have inconsistent frame counts: {sorted(lengths)}")
    count = lengths.pop()
    if count < 1:
        raise ValueError("Recorded state contains no frames")
    return count


def reset_replay_episode(env: Any, initial_state: Mapping[str, Any]) -> None:
    """Reset simulator/controller state, then restore a relative scene snapshot."""

    env.sim.reset()
    env.reset_to(initial_state, env_ids=None, is_relative=True)


def _as_numpy(value: Any) -> np.ndarray:
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def assert_recorded_state_restored(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, atol: float = 1e-5
) -> None:
    """Fail unless the live scene contains the recorded state within tolerance."""

    def compare(actual_value: Any, expected_value: Any, path: str) -> None:
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping):
                raise RuntimeError(f"Restored state {path} is not a mapping")
            for key, expected_child in expected_value.items():
                if key not in actual_value:
                    raise RuntimeError(f"Restored state is missing {path}/{key}")
                compare(actual_value[key], expected_child, f"{path}/{key}")
            return

        actual_array = _as_numpy(actual_value)
        expected_array = _as_numpy(expected_value)
        if actual_array.shape != expected_array.shape:
            raise RuntimeError(f"Restored state {path} has shape {actual_array.shape}; expected {expected_array.shape}")
        if not np.allclose(actual_array, expected_array, rtol=0.0, atol=atol, equal_nan=False):
            max_error = float(np.max(np.abs(actual_array - expected_array)))
            raise RuntimeError(f"Restored state {path} differs from the recording (max error {max_error:.3g})")

    compare(actual, expected, "state")


def make_state_renderer(env: Any) -> Callable[[], None]:
    """Create a retained renderer for restored scene states, without stepping physics."""

    views = []
    if env.sim.is_rendering and env.sim.cfg.use_fabric:
        import usdrt
        import warp as wp
        from isaaclab.sim.utils import get_current_stage_id
        from isaaclab.sim.views import FrameView

        sources = [
            (asset, [path for paths in asset.root_view.link_paths for path in paths], "get_link_transforms")
            for asset in env.scene.articulations.values()
        ]
        sources.extend(
            (asset, asset.root_view.prim_paths, "get_transforms") for asset in env.scene.rigid_objects.values()
        )
        fabric_stage = usdrt.Usd.Stage.Attach(get_current_stage_id())
        # FrameView initialization replaces existing Fabric transforms with USD transforms.
        scales_by_path = {
            path: np.linalg.norm(
                np.asarray(
                    usdrt.Rt.Xformable(fabric_stage.GetPrimAtPath(path)).GetFabricHierarchyWorldMatrixAttr().Get()
                )[:3, :3],
                axis=1,
            )
            for _, paths, _ in sources
            for path in paths
        }
        for asset, paths, getter in sources:
            groups: dict[str, dict[str, int]] = {}
            for index, path in enumerate(paths):
                parent = path.rsplit("/", 1)[0]
                groups.setdefault(parent, {})[path] = index
            for parent, indices in groups.items():
                names = "|".join(re.escape(path.rsplit("/", 1)[1]) for path in indices)
                view = FrameView(
                    f"{re.escape(parent)}/({names})", device=env.device, validate_xform_ops=False, stage=env.sim.stage
                )
                scales = wp.array(
                    [scales_by_path[path] for path in view.prim_paths], dtype=wp.float32, device=env.device
                )
                views.append((view, asset, getter, [indices[path] for path in view.prim_paths], scales))

    def render_frame() -> None:
        # PhysX tensor writes do not publish Fabric transforms without a physics step.
        for view, asset, getter, indices, scales in views:
            poses = wp.to_torch(getattr(asset.root_view, getter)()).reshape(-1, 7)[indices]
            view.set_world_poses(poses[:, :3].contiguous(), poses[:, 3:].contiguous())
            # Repeated matrix decomposition must not accumulate roundoff in mesh scales.
            view.set_scales(scales)
        env.sim.render_context.reset_transform_cadence()
        env.sim.render()

    return render_frame


def replay_recorded_states(
    env: Any, episode: Any, is_running: Callable[[], bool], render_frame: Callable[[], None]
) -> tuple[int, int]:
    """Render stored post-step scene states without re-running controllers or physics."""

    states = episode.data.get("states")
    if not isinstance(states, Mapping):
        raise RuntimeError("Loaded episode has no recorded states")
    expected_steps = recorded_state_count(states)
    actions = episode.data.get("actions")
    if actions is None:
        raise RuntimeError("Loaded episode has no actions to align with recorded states")
    if len(actions) != expected_steps:
        raise RuntimeError(
            f"Recorded action/state frame counts differ: actions={len(actions)}, states={expected_steps}"
        )
    initial_state = episode.get_initial_state()
    if not isinstance(initial_state, Mapping):
        raise RuntimeError("Loaded episode has no initial state")
    reset_replay_episode(env, initial_state)

    steps = 0
    final_state = None
    while is_running():
        state = episode.get_next_state()
        if state is None:
            break
        env.scene.reset_to(state, env_ids=None, is_relative=True)
        env.sim.forward()
        render_frame()
        final_state = state
        steps += 1
    if steps == expected_steps and final_state is not None:
        assert_recorded_state_restored(env.scene.get_state(is_relative=True), final_state)
    return steps, expected_steps


__all__ = [
    "assert_recorded_state_restored",
    "make_state_renderer",
    "recorded_state_count",
    "replay_recorded_states",
    "reset_replay_episode",
]

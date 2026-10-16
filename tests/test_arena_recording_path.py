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

"""Contracts for recording demonstrations on the Arena-composed environment.

``scripts/record_demos.py`` and ``arena`` cannot be imported here: the script
launches Isaac Sim at import time and ``isaaclab_arena`` is an external source checkout
that host CI does not provision. Their wiring is therefore asserted as source text, the
same way the rest of the Arena integration is checked. What does not need Isaac at all
-- resolving contract roles to scene entity names -- is exercised directly.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from e2e_robot_workflows.environments import STATIC_APPLE_ENVIRONMENT, resolve_role_to_scene_entity
from e2e_robot_workflows.environments.scene_keys import (
    NATIVE_SCENE_ENTITY_NAMES,
    candidate_scene_entity_names,
    resolve_scene_entity_name,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORD_DEMOS = (REPO_ROOT / "scripts/record_demos.py").read_text(encoding="utf-8")
ARENA_RECORDERS = (REPO_ROOT / "src/e2e_robot_workflows/arena/recorders.py").read_text(encoding="utf-8")


class _Scene:
    """Minimal stand-in for a live ``InteractiveScene`` keyed by entity name."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = names

    def keys(self) -> tuple[str, ...]:
        return self._names


def test_scene_entity_candidates_cover_both_runtimes() -> None:
    """Every role offers the native scene's name and the contract's declared name."""

    for role, native_name in NATIVE_SCENE_ENTITY_NAMES.items():
        candidates = candidate_scene_entity_names(role)
        assert candidates[0] == native_name
        if role != "robot":
            assert STATIC_APPLE_ENVIRONMENT.asset(role).name in candidates

    with pytest.raises(KeyError):
        candidate_scene_entity_names("gripper")


def test_roles_resolve_against_either_runtimes_scene() -> None:
    """The same role map is recovered from the native scene and the Arena scene."""

    native = _Scene(("background", "packing_table", "object", "destination", "robot"))
    arena = _Scene(
        (
            STATIC_APPLE_ENVIRONMENT.asset("background").name,
            STATIC_APPLE_ENVIRONMENT.asset("support").name,
            STATIC_APPLE_ENVIRONMENT.asset("pick_object").name,
            STATIC_APPLE_ENVIRONMENT.asset("destination").name,
            "robot",
        )
    )

    assert resolve_role_to_scene_entity(native) == {
        "background": "background",
        "support": "packing_table",
        "pick_object": "object",
        "destination": "destination",
        "robot": "robot",
    }
    assert resolve_role_to_scene_entity(arena) == {
        "background": STATIC_APPLE_ENVIRONMENT.asset("background").name,
        "support": STATIC_APPLE_ENVIRONMENT.asset("support").name,
        "pick_object": STATIC_APPLE_ENVIRONMENT.asset("pick_object").name,
        "destination": STATIC_APPLE_ENVIRONMENT.asset("destination").name,
        "robot": "robot",
    }

    with pytest.raises(KeyError, match="pick_object"):
        resolve_scene_entity_name(_Scene(("robot",)), "pick_object")


def test_recording_composes_the_arena_environment_through_the_builder() -> None:
    """The Arena path registers its own environment rather than parsing a gym task.

    Recording, locomotion, and the validations all compose through one seam, so the
    builder call and Arena's constructor arguments are asserted where they live.
    """

    composition = (REPO_ROOT / "src/e2e_robot_workflows/arena/composition.py").read_text(encoding="utf-8")

    assert "task_id, env_cfg, env_kwargs = build_arena_env()" in RECORD_DEMOS
    assert "from e2e_robot_workflows.arena.composition import compose" in RECORD_DEMOS
    assert "task_id, env_cfg, env_kwargs = builder.build_registered()" in composition
    # Arena asks for constructor arguments (its variation recorder) alongside the cfg.
    assert "gym.make(self.task_id, cfg=self.env_cfg, **self.env_kwargs).unwrapped" in composition
    assert "gym.make(task_id, cfg=env_cfg, **env_kwargs).unwrapped" in RECORD_DEMOS


def test_the_composition_seam_imports_without_an_arena_checkout() -> None:
    """Arena is a path-discovered checkout, so importing must not require one.

    The adapter subclasses Arena's own base classes, which makes importing it importing
    Arena; a module-scope import here would take down every caller that never chooses the
    Arena path, this test file included.
    """

    importlib.import_module("e2e_robot_workflows.arena.composition")

    assert "isaaclab_arena" not in sys.modules


def test_recording_disables_every_termination_including_arenas() -> None:
    """No termination term may end an episode while an operator is demonstrating.

    Arena's pick-and-place task adds ``object_dropped`` on top of ``time_out`` and
    ``success``, so the recorder drops whatever terms the composed environment has
    instead of naming the ones the built-in task happens to define.
    """

    assert "disabled = disable_terminations(env_cfg.terminations)" in RECORD_DEMOS
    assert "for name, term in list(vars(terminations).items()):" in RECORD_DEMOS
    assert "setattr(terminations, name, None)" in RECORD_DEMOS
    assert "object_dropped" in RECORD_DEMOS
    # The superseded special case must not come back: it left Arena's terms live.
    assert 'hasattr(env_cfg.terminations, "success")' not in RECORD_DEMOS


def test_recording_keeps_arenas_demonstration_recorder_terms() -> None:
    """Recording extends Arena's demo recorder instead of replacing it wholesale.

    The app's terms arrive through the same ``add_workflow_recorder_terms`` seam that
    AutoData generation uses, so declaring a term once reaches both paths.
    """

    assert "add_workflow_recorder_terms(recorders)" in ARENA_RECORDERS
    assert "env_cfg.recorders = make_recorder_manager_cfg(env_cfg)" in RECORD_DEMOS
    # `None` is how a recorder term is switched off, and the seam leaves those alone.
    assert "recorders.record_post_step_navigate_cmd = None" in ARENA_RECORDERS
    assert "recorders.record_post_step_base_height_cmd = None" in ARENA_RECORDERS
    # Arena's camera recorder reads a group that camera-free embodiments never publish.
    assert 'if not hasattr(env_cfg.observations, "camera_obs"):' in ARENA_RECORDERS
    assert "recorders.record_pre_step_flat_camera_observations = None" in ARENA_RECORDERS


def test_arena_sample_commands_read_arenas_own_action_layouts() -> None:
    """The finite sample must hold the scene still on whichever layout was composed.

    Both commands are keyed to Arena's own definitions rather than to copied indices:
    the joint layout is checked against the articulation's joint count, and the Pink
    layout's slices come from Arena's ``action_constants``.
    """

    actions = (REPO_ROOT / "src/e2e_robot_workflows/arena/g1_actions.py").read_text(encoding="utf-8")

    assert "if action_dim != joint_count + LOCOMOTION_COMMAND_DIM:" in actions
    assert "from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy import action_constants" in actions
    assert "action_constants.LEFT_WRIST_QUAT_START_IDX" in actions
    # Which of the two a run holds with is decided next to them, for every entry point.
    assert "if embodiment == G1_ARENA_EMBODIMENT:" in actions
    assert "return pink_hold_action(env, base_height_m=base_height_m)" in actions
    assert "return hold_action(env, arena_embodiment)" in RECORD_DEMOS


def test_recording_checks_the_composed_scene_before_it_records() -> None:
    """Every recording path resolves the task's entities up front, Arena's included.

    The per-step poll logs read errors and keeps recording, so a scene naming its assets
    differently would only surface as "no episode ever succeeds". Checking right after
    the environment is built puts the failure where the composition happened, and puts it
    inside the finite sample run that ``sim-arena-recording`` drives.
    """

    assert "COMPLETION_POLICY.check_readable(env)" in RECORD_DEMOS
    assert RECORD_DEMOS.index("check_completion_policy_readable(env)") < RECORD_DEMOS.index("if args_cli.sample_steps:")


def test_recording_routes_the_validated_arena_selector_to_openxr() -> None:
    """The behavioral selector contract runs before Arena receives its device."""

    assert RECORD_DEMOS.index("require_arena_teleop_mode(") < RECORD_DEMOS.index("app_launcher = AppLauncher")
    assert 'teleop_device="openxr" if args_cli.xr and not args_cli.sample_steps else None' in RECORD_DEMOS

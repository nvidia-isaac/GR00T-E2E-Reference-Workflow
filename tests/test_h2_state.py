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

"""H2 reset restores a complete snapshot before exposing it to the next action."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from e2e_robot_workflows.arena.h2_state import H2ResetStateMixin
from e2e_robot_workflows.controllers.h2_feedback import feedback_shapes, validate_h2_initial_state
from e2e_robot_workflows.robots.h2 import H2_LIVE_JOINT_COUNT, load_h2_robot_config


def snapshot():
    cfg = load_h2_robot_config()
    shapes = feedback_shapes(
        num_envs=1, driven=len(cfg.driven_joint_names), depth=cfg.history_length, command_dim=cfg.command_dim
    )
    return {
        "h2_agile_feedback": {name: np.ones(shape, dtype=np.float32) for name, shape in shapes.items()},
        "h2_actuator_targets": {
            name: np.ones((1, H2_LIVE_JOINT_COUNT), dtype=np.float32) for name in ("position", "velocity")
        },
    }


def test_snapshot_validation_is_independent_of_simulator_and_rejects_partial_pairs():
    assert validate_h2_initial_state(snapshot())
    assert not validate_h2_initial_state({}, required=False)
    for value in ({}, {"h2_agile_feedback": snapshot()["h2_agile_feedback"]}):
        with pytest.raises(ValueError, match="requires both"):
            validate_h2_initial_state(value)


def test_reset_restores_feedback_and_both_motor_targets_before_snapshot():
    torch = pytest.importorskip("torch")
    state = {
        group: {name: torch.from_numpy(value) for name, value in fields.items()} for group, fields in snapshot().items()
    }
    events = []

    class Parent:
        def reset_to(self, state, **kwargs):
            events.append("parent_reset")
            return {"old": True}, {"reset": True}

    class Environment(H2ResetStateMixin, Parent):
        device = "cpu"
        action_manager = SimpleNamespace(
            get_term=lambda name: SimpleNamespace(
                restore_feedback_state=lambda value: events.append(("feedback", value))
            )
        )
        observation_manager = SimpleNamespace(
            reset=lambda ids: events.append("observations_reset"), compute=lambda **kwargs: {"restored": True}
        )
        recorder_manager = SimpleNamespace(
            reset=lambda ids: events.append("recorder_reset"), record_post_reset=lambda ids: events.append("snapshot")
        )

    class Scene(dict):
        def write_data_to_sim(self):
            events.append("write")

    env = Environment()
    env.scene = Scene(
        robot=SimpleNamespace(
            set_joint_position_target_index=lambda **kwargs: events.append(("position", kwargs["target"])),
            set_joint_velocity_target_index=lambda **kwargs: events.append(("velocity", kwargs["target"])),
        )
    )
    assert env.reset_to(state, is_relative=True) == ({"restored": True}, {"reset": True})
    assert [event[0] if isinstance(event, tuple) else event for event in events] == [
        "parent_reset",
        "feedback",
        "position",
        "velocity",
        "write",
        "observations_reset",
        "recorder_reset",
        "snapshot",
    ]
    torch.testing.assert_close(events[2][1], state["h2_actuator_targets"]["position"])
    events.clear()
    del state["h2_actuator_targets"]
    with pytest.raises(ValueError, match="requires both"):
        env.reset_to(state)
    assert not events


def test_generation_prepares_each_public_reset_before_returning_observations(monkeypatch):
    # Execute the production override with a parent that does not start Isaac Sim.
    source = Path(__file__).resolve().parents[1] / "src/e2e_robot_workflows/arena/mimic.py"
    node = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "H2MimicEnv"
    )
    node.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    node.body = [method for method in node.body if isinstance(method, ast.FunctionDef) and method.name == "reset"]
    events = []

    class Parent:
        cfg = SimpleNamespace(e2e_h2_prepare_on_reset=True)

        def reset(self, **kwargs):
            events.append(("reset", kwargs))
            return {"cold": True}, {}

    def prepare(env, *, embodiment):
        events.append(("prepare", embodiment))
        env.obs_buf = {"ready": True}

    monkeypatch.setattr("e2e_robot_workflows.devices.h2_ready.prepare_h2_recording", prepare)
    namespace = {"Parent": Parent}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec"), namespace)
    env = namespace["H2MimicEnv"]()
    for _ in range(2):
        assert env.reset(env_ids=[0]) == ({"ready": True}, {})
    assert events == [("reset", {"env_ids": [0]}), ("prepare", "h2_agile_pink")] * 2

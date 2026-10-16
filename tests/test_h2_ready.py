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

"""Startup gates must reject dangerous motion rather than hide it as warmup."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from e2e_robot_workflows.devices import h2_ready
from e2e_robot_workflows.devices.h2_ready import READY_STEPS, STABLE_STEPS, H2StartupGuard


def test_ready_requires_consecutive_stable_samples():
    guard = H2StartupGuard(0.01, 0.96)
    for _ in range(STABLE_STEPS - 1):
        guard.observe([[0, 0, 1.3]], np.zeros(6), np.zeros(14))
    with pytest.raises(RuntimeError, match="did not settle"):
        guard.finish()
    guard.observe([[0, 0, 1.3]], np.zeros(6), np.zeros(14))
    assert guard.finish()["stable_steps"] == STABLE_STEPS
    guard.observe([[0, 0, 1.3]], [0.1, 0, 0, 0, 0, 0], np.zeros(14))
    assert guard.stable_steps == 0


def test_fast_downward_motion_is_rejected_despite_safe_height():
    guard = H2StartupGuard(0.01, 0.96)
    guard.observe([[0, 0, 1.5]], np.zeros(6), np.zeros(14))
    with pytest.raises(RuntimeError, match="downward speed=1.000"):
        guard.observe([[0, 0, 1.49]], np.zeros(6), np.zeros(14))


@pytest.mark.parametrize("height", [0.95, 0.999, float("nan")])
def test_low_or_nonfinite_geometry_blocks_readiness(height):
    with pytest.raises(RuntimeError):
        H2StartupGuard(0.01, 0.96).observe([[0, 0, height]], np.zeros(6), np.zeros(14))


@pytest.mark.parametrize("dt", [0, -0.01, float("nan")])
def test_invalid_timestep_rejected(dt):
    with pytest.raises(ValueError):
        H2StartupGuard(dt, 0.96)


@pytest.mark.parametrize("interrupt", [False, True])
@pytest.mark.parametrize("table_offset", [None, 0.32])
def test_preparation_excludes_warmup_and_restores_recorder_on_interrupt(monkeypatch, interrupt, table_offset):
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    empty_recorder = object()
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.managers",
        SimpleNamespace(RecorderManager=lambda cfg, env: empty_recorder, RecorderManagerBaseCfg=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "e2e_robot_workflows.arena.h2_actions",
        SimpleNamespace(h2_hold_action=lambda env, embodiment: torch.zeros(1, 61)),
    )
    monkeypatch.setattr(
        "e2e_robot_workflows.robots.h2.load_h2_pink_contract", lambda: SimpleNamespace(arm_joint_names=["elbow"])
    )
    monkeypatch.setattr(h2_ready, "_collision_bounds", lambda env: (np.array([0]), np.zeros((1, 3))))
    events = []
    recorder = SimpleNamespace(
        reset=lambda: events.append("recorder_reset"),
        record_post_reset=lambda ids: events.append(("snapshot", ids)),
    )
    robot = SimpleNamespace(
        joint_names=["elbow"],
        data=SimpleNamespace(
            body_link_pose_w=SimpleNamespace(torch=torch.tensor([[[0.0, 0.0, 1.3, 0.0, 0.0, 0.0, 1.0]]])),
            root_link_vel_w=SimpleNamespace(torch=torch.zeros(1, 6)),
            joint_vel=SimpleNamespace(torch=torch.zeros(1, 1)),
        ),
    )
    lower = SimpleNamespace(cfg=SimpleNamespace(history_length=5), process_actions=lambda action: None)
    env = SimpleNamespace(
        num_envs=1,
        step_dt=0.01,
        scene={"robot": robot},
        recorder_manager=recorder,
        cfg=SimpleNamespace(e2e_support_surface_z_m=0.96),
        action_manager=SimpleNamespace(get_term=lambda name: lower),
        sim=SimpleNamespace(is_stopped=lambda: False),
        episode_length_buf=torch.zeros(1),
        observation_manager=SimpleNamespace(reset=lambda: None, compute=lambda **kw: {"ready": True}),
    )

    def step(action):
        assert env.recorder_manager is empty_recorder
        events.append("step")
        env.episode_length_buf[:] += 1
        return None, None, torch.tensor([False]), torch.tensor([False]), None

    env.step = step
    if table_offset is not None:

        class Scene(dict):
            env_origins = torch.zeros((1, 3))

        env.scene = Scene(env.scene)
        env.cfg.e2e_support_asset_name = "packing_table"
        env.cfg.e2e_support_root_z_m = 0.0
        env.scene["packing_table"] = SimpleNamespace(
            data=SimpleNamespace(root_pose_w=SimpleNamespace(torch=torch.tensor([[0, 0, table_offset, 0, 0, 0, 1]])))
        )
    if interrupt:
        with pytest.raises(RuntimeError, match="interrupted"):
            h2_ready.prepare_h2_recording(env, is_running=lambda: False)
        assert not events
    elif table_offset is not None:
        with pytest.raises(RuntimeError, match="clearance=0.0200"):
            h2_ready.prepare_h2_recording(env)
        assert not events
    else:
        h2_ready.prepare_h2_recording(env)
        assert events == ["step"] * READY_STEPS + ["recorder_reset", ("snapshot", [0])]
        assert env.episode_length_buf.item() == 0
        assert env.obs_buf == {"ready": True}
    assert env.recorder_manager is recorder


@pytest.mark.parametrize("embodiment", ["h2_agile_pink", "h2_agile_joint"])
def test_priming_dispatches_hold_without_advancing_physics(monkeypatch, embodiment):
    action = np.arange(61, dtype=np.float32)[None, :]
    seen = []
    monkeypatch.setitem(
        sys.modules,
        "e2e_robot_workflows.arena.h2_actions",
        SimpleNamespace(
            h2_hold_action=lambda env, selected: seen.append(selected) or action,
        ),
    )
    processed = []
    term = SimpleNamespace(
        cfg=SimpleNamespace(history_length=3), process_actions=lambda value: processed.append(value.copy())
    )
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: term))
    assert h2_ready.prime_h2_controller(env, embodiment) is action
    assert seen == [embodiment]
    np.testing.assert_array_equal(processed, np.repeat(action[None, :, -3:], 6, axis=0))

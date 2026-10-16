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

from __future__ import annotations

from types import SimpleNamespace

import pytest

from e2e_robot_workflows.data.replay import recorded_state_count, replay_recorded_states


class FakeEpisode:
    def __init__(self, states: list[dict[str, object]]) -> None:
        self.data = {
            "actions": list(range(len(states))),
            "states": {
                "articulation": {"robot": {"joint_position": list(range(len(states)))}},
                "rigid_object": {"pick_object": {"root_pose": list(range(len(states)))}},
            },
        }
        self._states = iter(states)

    def get_initial_state(self):
        return {"initial": "state"}

    def get_next_state(self):
        return next(self._states, None)


class FakeScene:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls
        self.state = None

    def reset_to(self, state, *, env_ids, is_relative):
        self.state = state
        self.calls.append(("scene_reset", state, env_ids, is_relative))

    def get_state(self, *, is_relative):
        self.calls.append(("scene_get_state", is_relative))
        return self.state


class FakeSimulation:
    def __init__(self, calls: list[tuple], scene: FakeScene) -> None:
        self.calls = calls
        self.scene = scene
        self.rendered_states = []

    def reset(self):
        self.calls.append(("sim_reset",))

    def forward(self):
        self.calls.append(("forward",))

    def render(self):
        self.calls.append(("render",))
        self.rendered_states.append(self.scene.state)


class FakeEnv:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.scene = FakeScene(self.calls)
        self.sim = FakeSimulation(self.calls, self.scene)

    def reset_to(self, state, *, env_ids, is_relative):
        self.calls.append(("env_reset", state, env_ids, is_relative))

    def step(self, _action):
        raise AssertionError("state playback must not step controllers or physics")


def test_recorded_state_count_rejects_inconsistent_streams() -> None:
    with pytest.raises(ValueError, match="inconsistent frame counts"):
        recorded_state_count({"robot": {"position": [1, 2]}, "object": {"pose": [1]}})


def test_recorded_state_count_rejects_empty_streams() -> None:
    with pytest.raises(ValueError, match="contains no frames"):
        recorded_state_count({"robot": {"position": []}})


def test_state_replay_restores_every_frame_without_stepping() -> None:
    states = [{"frame": 0}, {"frame": 1}]
    episode = FakeEpisode(states)
    env = FakeEnv()

    steps, expected = replay_recorded_states(env, episode, lambda: True, env.sim.render)

    assert (steps, expected) == (2, 2)
    assert env.sim.rendered_states == states


def test_state_replay_reports_early_application_stop() -> None:
    episode = FakeEpisode([{"frame": 0}, {"frame": 1}])
    env = FakeEnv()
    checks = iter((True, False))

    steps, expected = replay_recorded_states(env, episode, lambda: next(checks), env.sim.render)

    assert (steps, expected) == (1, 2)
    assert env.sim.rendered_states == [{"frame": 0}]


def test_state_replay_requires_recorded_states() -> None:
    episode = SimpleNamespace(data={}, get_initial_state=lambda: {"initial": "state"})

    with pytest.raises(RuntimeError, match="no recorded states"):
        replay_recorded_states(FakeEnv(), episode, lambda: True, lambda: None)


def test_state_replay_rejects_action_state_length_mismatch() -> None:
    episode = FakeEpisode([{"frame": 0}])
    episode.data["actions"] = [0, 1]

    with pytest.raises(RuntimeError, match="actions=2, states=1"):
        replay_recorded_states(FakeEnv(), episode, lambda: True, lambda: None)


def test_state_replay_rejects_a_live_state_that_was_not_restored() -> None:
    episode = FakeEpisode([{"frame": 0}])
    env = FakeEnv()
    env.scene.get_state = lambda *, is_relative: {"frame": 1}

    with pytest.raises(RuntimeError, match="differs from the recording"):
        replay_recorded_states(env, episode, lambda: True, env.sim.render)


def test_state_replay_accepts_isaac_lab_episode_batch_shape() -> None:
    torch = pytest.importorskip("torch")
    datasets = pytest.importorskip("isaaclab.utils.datasets")
    episode = datasets.EpisodeData()
    episode.data = {
        "actions": torch.zeros((2, 1)),
        "initial_state": {"articulation": {"robot": {"root_pose": torch.zeros((1, 7))}}},
        "states": {"articulation": {"robot": {"root_pose": torch.zeros((2, 7))}}},
    }
    env = FakeEnv()

    steps, expected = replay_recorded_states(env, episode, lambda: True, env.sim.render)

    assert (steps, expected) == (2, 2)
    restored = [call[1] for call in env.calls if call[0] == "scene_reset"]
    assert [state["articulation"]["robot"]["root_pose"].shape for state in restored] == [(1, 7), (1, 7)]

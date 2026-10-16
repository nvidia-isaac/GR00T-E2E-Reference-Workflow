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

"""Host contracts for the consumer-owned Arena AutoData seam."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np
import pytest

from e2e_robot_workflows.arena.mimic_actions import (
    h2_body_pose_in_pelvis_frame,
    normalize_mimic_action_inputs,
    require_action_width,
)
from e2e_robot_workflows.autodata.arena import (
    finalize_arena_generation,
    make_arena_datastream,
    make_arena_embodiment_adapter,
    qualify_and_export_final_state,
    validate_generated_episode_counts,
)


class _FakeEmbodimentAdapter:
    env = None

    def bind_env(self, env):
        self.env = env
        return self


class _FakeDatastream:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_h2_body_pseudo_eef_is_a_valid_pelvis_frame_pose() -> None:
    torch = pytest.importorskip("torch")
    action = torch.zeros((3, 61), dtype=torch.float64)
    action[:, -3:] = torch.tensor((0.4, -0.2, 0.1), dtype=action.dtype)

    pose = h2_body_pose_in_pelvis_frame(action)

    assert pose.shape == (3, 4, 4)
    assert pose.dtype == action.dtype
    torch.testing.assert_close(pose, torch.eye(4, dtype=action.dtype).expand(3, -1, -1))


@pytest.fixture
def fake_autodata_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    embodiment_module = ModuleType("isaac_autodata_interfaces.embodiments.embodiment_adapter")
    embodiment_module.EmbodimentAdapter = _FakeEmbodimentAdapter
    datastream_module = ModuleType("isaac_autodata_interfaces.datastream")
    datastream_module.Datastream = _FakeDatastream
    monkeypatch.setitem(sys.modules, embodiment_module.__name__, embodiment_module)
    monkeypatch.setitem(sys.modules, datastream_module.__name__, datastream_module)


def test_adapter_delegates_pose_action_contract_without_reordering(fake_autodata_modules: None) -> None:
    torch = pytest.importorskip("torch")
    calls: list[tuple] = []
    expected_poses = {"left": torch.eye(4), "right": torch.eye(4), "body": torch.zeros(4, 4)}
    expected_action = torch.arange(23, dtype=torch.float32)
    expected_passthrough = {
        "left": torch.tensor([0.1]),
        "right": torch.tensor([0.2]),
        "body": torch.arange(7),
    }

    class Env:
        def get_robot_eef_pose(self, name, env_ids=None):
            calls.append(("get", name, env_ids))
            return expected_poses[name]

        def action_to_target_eef_pose(self, action):
            calls.append(("to_pose", action))
            return expected_poses

        def target_eef_pose_to_action(self, poses, passthrough, *, action_noise_dict, env_id):
            calls.append(("to_action", poses, passthrough, action_noise_dict, env_id))
            return expected_action

        def actions_to_gripper_actions(self, actions):
            calls.append(("passthrough", actions))
            return expected_passthrough

    adapter = make_arena_embodiment_adapter().bind_env(Env())
    source_action = torch.arange(23, dtype=torch.float32).reshape(1, 23)

    current = adapter.get_eef_poses(env_ids=[0])
    assert current["left"] is expected_poses["left"]
    assert current["right"] is expected_poses["right"]
    targets = adapter.action_to_target_eef_pose(source_action)
    assert targets["left"] is expected_poses["left"]
    assert targets["right"] is expected_poses["right"]
    assert (
        adapter.target_eef_pose_to_action(
            expected_poses,
            expected_passthrough,
            action_noise_dict={"left": 0.0, "right": 0.0},
            env_id=2,
        )
        is expected_action
    )
    assert adapter.actions_to_passthrough_actions(source_action) is expected_passthrough
    assert calls[3][1] is expected_poses
    assert calls[3][2] is expected_passthrough


def test_datastream_delegates_arena_object_poses(fake_autodata_modules: None) -> None:
    torch = pytest.importorskip("torch")
    expected = {"pick_object": torch.eye(4).reshape(1, 4, 4)}

    class Env:
        device = "cpu"

        def get_object_poses(self, env_ids=None):
            assert env_ids == [0]
            return expected

    stream = make_arena_datastream(
        env=Env(),
        task_descriptor=object(),
        source_dataset_path="source.hdf5",
    )

    assert stream.get_object_poses(env_ids=[0]) is expected


@pytest.mark.parametrize(
    ("outcomes", "max_settle_frames", "expected_success", "expected_steps"),
    [
        ([False, True, False, True, True, True], 5, True, 5),
        ([False, False, False, False, False], 4, False, 4),
    ],
)
def test_generation_waits_for_a_consecutive_success_hold_within_its_settle_budget(
    outcomes: list[bool],
    max_settle_frames: int,
    expected_success: bool,
    expected_steps: int,
) -> None:
    torch = pytest.importorskip("torch")
    observed = iter(outcomes)

    class Recorder:
        labels: list[bool] = []
        exports = 0

        def set_success_to_episodes(self, _env_ids, success) -> None:
            self.labels.append(bool(success.item()))

        def export_episodes(self, _env_ids) -> None:
            self.exports += 1

    class Env:
        device = "cpu"
        action_manager = type("ActionManager", (), {"action": torch.zeros((1, 3))})()
        recorder_manager = Recorder()

    class SuccessTerm:
        params = {}

        @staticmethod
        def func(_env):
            return torch.tensor([next(observed)])

    class ActionQueue:
        actions: list[tuple] = []

        async def put(self, action) -> None:
            self.actions.append(action)

        async def join(self) -> None:
            pass

    queue = ActionQueue()
    success = asyncio.run(
        qualify_and_export_final_state(
            env=Env(),
            env_id=0,
            success_term=SuccessTerm(),
            action_queue=queue,
            hold_frames=2,
            max_settle_frames=max_settle_frames,
        )
    )

    assert success is expected_success
    assert len(queue.actions) == expected_steps
    assert Env.recorder_manager.labels == [expected_success]
    assert Env.recorder_manager.exports == 1


def test_worker_cleanup_errors_do_not_skip_remaining_generation_cleanup() -> None:
    events: list[str] = []
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)

    class Environment:
        def close(self) -> None:
            events.append("environment.close")

    result = {"status": "error", "cleanup_errors": ["generator_task: RuntimeError: generation failed"]}

    def record_artifacts(state: dict) -> None:
        events.append("record_artifacts")
        state["artifacts"] = {}

    def write_result(state: dict) -> None:
        events.append("write_result")
        assert state["status"] == "error"

    exit_code = finalize_arena_generation(
        event_loop=event_loop,
        environment=Environment(),
        result=result,
        stats={"num_success": 0},
        exit_code=0,
        record_artifacts=record_artifacts,
        write_result=write_result,
    )

    assert exit_code == 2
    assert events == ["environment.close", "record_artifacts", "write_result"]
    assert event_loop.is_closed()
    assert result["num_success"] == 0
    assert result["cleanup_errors"] == ["generator_task: RuntimeError: generation failed"]


def _write_outcome_artifact(path: Path, outcomes: list[bool]) -> None:
    with h5py.File(path, "w") as dataset:
        data = dataset.create_group("data")
        for index, success in enumerate(outcomes):
            data.create_group(f"demo_{index}").attrs["success"] = success


def test_generated_artifact_labels_must_match_attempt_results(tmp_path: Path) -> None:
    output_path = tmp_path / "generated.hdf5"
    _write_outcome_artifact(output_path, [True, True])
    _write_outcome_artifact(tmp_path / "generated_failed.hdf5", [False])

    validate_generated_episode_counts(output_path, successful_expected=2, failed_expected=1)

    with h5py.File(output_path, "r+") as dataset:
        dataset["data/demo_1"].attrs["success"] = False
    with pytest.raises(RuntimeError, match="success=False"):
        validate_generated_episode_counts(output_path, successful_expected=2, failed_expected=1)


def test_generated_artifact_counts_must_match_attempt_results(tmp_path: Path) -> None:
    output_path = tmp_path / "generated.hdf5"
    _write_outcome_artifact(output_path, [True])

    with pytest.raises(RuntimeError, match="successful artifact contains 1 episodes, expected 2"):
        validate_generated_episode_counts(output_path, successful_expected=2, failed_expected=0)


def test_positive_generation_count_requires_an_artifact(tmp_path: Path) -> None:
    output_path = tmp_path / "generated.hdf5"

    with pytest.raises(RuntimeError, match="missing successful generation artifact"):
        validate_generated_episode_counts(output_path, successful_expected=1, failed_expected=0)


def test_empty_generated_artifact_is_valid_when_zero_episodes_are_expected(tmp_path: Path) -> None:
    output_path = tmp_path / "generated.hdf5"
    _write_outcome_artifact(output_path, [])

    validate_generated_episode_counts(output_path, successful_expected=0, failed_expected=0)


@pytest.mark.parametrize("invalid_label", ["successful", "failed"])
def test_expected_zero_artifact_requires_a_valid_data_group(tmp_path: Path, invalid_label: str) -> None:
    output_path = tmp_path / "generated.hdf5"
    failed_path = tmp_path / "generated_failed.hdf5"
    _write_outcome_artifact(output_path, [])
    invalid_path = output_path if invalid_label == "successful" else failed_path
    if invalid_label == "successful":
        output_path.unlink()
    with h5py.File(invalid_path, "w"):
        pass

    with pytest.raises(RuntimeError, match=rf"{invalid_label} artifact does not contain a valid /data group"):
        validate_generated_episode_counts(output_path, successful_expected=0, failed_expected=0)


def test_normalizes_only_autodata_batch_axes() -> None:
    poses = {"left": np.eye(4)[None], "right": np.eye(4)[None]}
    passthrough = {
        "left": np.asarray([[0.1]]),
        "right": np.asarray([[0.2]]),
        "body": np.arange(7)[None],
    }

    normalized_poses, normalized_passthrough = normalize_mimic_action_inputs(poses, passthrough)

    assert normalized_poses["left"].shape == (4, 4)
    assert normalized_poses["right"].shape == (4, 4)
    assert np.asarray(normalized_passthrough["left"]).shape == ()
    assert normalized_passthrough["body"].shape == (7,)


def test_rejects_wrong_arena_action_shape() -> None:
    require_action_width(np.zeros((1, 23)), rank=2)
    with pytest.raises(ValueError, match="width 23"):
        require_action_width(np.zeros((1, 22)), rank=2)
    with pytest.raises(ValueError, match="rank-1"):
        require_action_width(np.zeros((1, 23)), rank=1)


@pytest.mark.parametrize(
    ("quaternion_xyzw", "expected_matrix"),
    [
        ([0.0, 0.0, 0.0, 1.0], np.diag([1.0, 1.0, 1.0])),
        ([1.0, 0.0, 0.0, 0.0], np.diag([1.0, -1.0, -1.0])),
        ([0.0, 1.0, 0.0, 0.0], np.diag([-1.0, 1.0, -1.0])),
        ([0.0, 0.0, 1.0, 0.0], np.diag([-1.0, -1.0, 1.0])),
    ],
)
def test_pinned_mimic_boundary_is_xyzw(quaternion_xyzw, expected_matrix) -> None:
    torch = pytest.importorskip("torch")
    from isaaclab.utils.math import matrix_from_quat

    actual = matrix_from_quat(torch.tensor([quaternion_xyzw], dtype=torch.float32))[0]

    np.testing.assert_allclose(actual.numpy(), expected_matrix, atol=1e-6)

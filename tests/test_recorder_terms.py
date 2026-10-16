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

"""Behavior of the recorder terms this app adds to a recorder manager another tool installed.

Unlike the other host tests, these build real Isaac Lab cfg objects: the question they answer --
which term names a recorder manager cfg already carries -- is one only ``@configclass`` semantics
can settle. No simulation is started, so they stay CPU-only, but they do need Isaac Lab importable
and the host CI image has none -- hence the skip below. Every case here needs it, the dict form
included, because ``add_workflow_recorder_terms`` reads the term names off real cfg objects.

The guard is on the top-level package alone, not on the submodule paths used below: an Isaac Lab
that no longer exposes those is a compatibility break these tests exist to report, not skip.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("isaaclab", reason="Isaac Lab is not installed on the CPU-only host image")

import torch  # noqa: E402
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402

from e2e_robot_workflows.data.recorders import PostResetH2FeedbackRecorder, add_workflow_recorder_terms  # noqa: E402

WORKFLOW_TERMS = {
    "record_post_step_joint_pos_target",
    "record_post_step_navigate_cmd",
    "record_post_step_base_height_cmd",
    "record_post_reset_h2_feedback",
}
"""The terms this app records on top of Isaac Lab's own action/state recorder."""


@pytest.mark.parametrize("active_terms", [[], ["lower_body_joint_pos"]])
def test_non_h2_recorders_do_not_gain_controller_snapshot_data(active_terms):
    manager = SimpleNamespace(active_terms=active_terms, get_term=lambda name: SimpleNamespace())
    recorder = SimpleNamespace(_env=SimpleNamespace(action_manager=manager))
    assert PostResetH2FeedbackRecorder.record_post_reset(recorder, None) == (None, None)


def test_h2_initial_snapshot_copies_motor_targets_with_feedback():
    position, velocity = torch.ones(1, 75), torch.zeros(1, 75)
    term = SimpleNamespace(get_feedback_state=lambda: {"last_actions": torch.ones(1, 14)})
    manager = SimpleNamespace(active_terms=["lower_body_joint_pos"], get_term=lambda name: term)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos_target=SimpleNamespace(torch=position), joint_vel_target=SimpleNamespace(torch=velocity)
        )
    )
    recorder = SimpleNamespace(_env=SimpleNamespace(action_manager=manager, scene={"robot": robot}))
    key, snapshot = PostResetH2FeedbackRecorder.record_post_reset(recorder, [0])
    assert key == "initial_state"
    assert snapshot["h2_agile_feedback"]["last_actions"].shape == (1, 14)
    position.zero_()
    assert snapshot["h2_actuator_targets"]["position"].sum() == 75


def test_a_recorder_manager_without_our_terms_gains_them() -> None:
    """AutoData assigns Isaac Lab's own recorder, which names none of this app's terms."""

    recorders = ActionStateRecorderManagerCfg()
    kept = recorders.record_post_step_states

    added = add_workflow_recorder_terms(recorders)

    assert set(added) == WORKFLOW_TERMS
    assert all(getattr(recorders, name) is not None for name in WORKFLOW_TERMS)
    # Terms are added, not the manager replaced, so what the caller installed survives.
    assert recorders.record_post_step_states is kept


def test_a_term_the_caller_switched_off_stays_off() -> None:
    """Isaac Lab's recorder manager skips a term whose cfg is ``None``, so that is how a
    caller disables one; re-adding ours over it would overrule a deliberate choice."""

    @configclass
    class WithoutNavigation(ActionStateRecorderManagerCfg):
        record_post_step_navigate_cmd = None

    recorders = WithoutNavigation()

    added = add_workflow_recorder_terms(recorders)

    assert set(added) == WORKFLOW_TERMS - {"record_post_step_navigate_cmd"}
    assert recorders.record_post_step_navigate_cmd is None


def test_a_command_with_no_height_channel_refuses_to_record_an_empty_column() -> None:
    """A 3-wide command must stop the recorder rather than yield a zero-width column.

    ``cmd[:, 3:4]`` on the H2 export's ``[vx, vy, wz]`` is legal torch and returns
    shape ``[N, 0]``, so without this the dataset would carry a ``base_height_cmd``
    field of no width on every frame and nothing would report it until something
    downstream tried to read it. The G1's 4-wide command is unaffected.
    """

    from e2e_robot_workflows.data.recorders import PostStepBaseHeightCommandRecorder

    recorder = PostStepBaseHeightCommandRecorder.__new__(PostStepBaseHeightCommandRecorder)
    recorder._env = _EnvWithLocomotionCommand(torch.zeros((1, 3)))

    with pytest.raises(RuntimeError, match="carries no base-height channel"):
        recorder.record_post_step()

    recorder._env = _EnvWithLocomotionCommand(torch.tensor([[0.1, 0.2, 0.3, 0.72]]))
    name, value = recorder.record_post_step()

    assert name == "base_height_cmd"
    assert value.shape == (1, 1) and float(value[0, 0]) == pytest.approx(0.72)


class _EnvWithLocomotionCommand:
    """The two attributes ``_locomotion_command`` reaches through, and nothing else."""

    def __init__(self, command) -> None:
        term = type("Term", (), {"locomotion_command": command})()
        manager = type(
            "ActionManager",
            (),
            {"active_terms": ["lower_body_joint_pos"], "get_term": staticmethod(lambda _name: term)},
        )()
        self.action_manager = manager


def test_the_dict_form_of_a_recorder_manager_cfg_is_read_the_same_way() -> None:
    """Isaac Lab reads a plain dict of terms as well as a cfg object, and skips a ``None``
    entry there too, so the two forms cannot disagree about what the caller asked for."""

    recorders: dict[str, object | None] = {"record_post_step_navigate_cmd": None}

    added = add_workflow_recorder_terms(recorders)

    assert set(added) == WORKFLOW_TERMS - {"record_post_step_navigate_cmd"}
    assert recorders["record_post_step_navigate_cmd"] is None
    assert all(recorders[name] is not None for name in added)


def test_arena_recorder_retains_raw_release_alongside_latched_subtasks():
    from e2e_robot_workflows.data.recorders import PreStepArenaDatagenInfoRecorder

    signals = {f"{hand}_hand_holds_object": torch.tensor([False]) for hand in ("left", "right")}
    recorder = PreStepArenaDatagenInfoRecorder.__new__(PreStepArenaDatagenInfoRecorder)
    recorder._latched_signals = {name: value.clone() for name, value in signals.items()}
    recorder._env = SimpleNamespace(
        obs_buf={"subtask_terms": signals},
        action_manager=SimpleNamespace(action=None),
        action_to_target_eef_pose=lambda action: {"left": None, "right": None},
        get_object_poses=lambda: {},
        get_robot_eef_pose=lambda name: None,
    )
    signals["left_hand_holds_object"][:] = True
    _, pickup = recorder.record_pre_step()
    signals["left_hand_holds_object"][:] = False
    _, release = recorder.record_pre_step()
    assert pickup["grasp_signals"]["left_hand_holds_object"].item()
    assert not release["grasp_signals"]["left_hand_holds_object"].item()
    assert release["subtask_term_signals"]["left_hand_holds_object"].item()

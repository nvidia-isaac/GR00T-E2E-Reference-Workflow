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

"""Recorder manager for Isaac Lab workflow tasks.

Isaac Lab's ``ActionStateRecorderManagerCfg`` already records everything the
generic demonstration record needs per step: ``actions``, ``processed_actions``,
``states``, and the full policy observation group as ``obs`` -- which includes
the head camera (``obs/robot_head_cam_rgb``) because it is a policy observation
term.

This subclass adds generic robot control signals the base recorder omits:

- ``joint_pos_target`` -- the joint position command actually sent to the
  articulation. For this app that is the *combined* Pink IK (upper body) + AGILE
  (lower body) command, in articulation joint order (so it shares the exact joint
  layout as ``states``/``robot_joint_pos``). Being a plain actuator-command
  signal it stays controller-agnostic -- any future controller populates the
  same buffer.
- ``navigate_cmd`` / ``base_height_cmd`` -- the AGILE locomotion command
  ``[vx, vy, wz]`` / ``[hip_height]`` fed to the lower body. Recorded every step
  regardless of whether the task moves (constant for a static task, varying for
  locomotion), so a dataset can drive GR00T's navigate/base-height action head.
  Absent in fixed-base mode, where there is no lower-body command.

Not every locomotion export publishes both. The H2 AGILE export takes
``[vx, vy, wz]`` with no height channel, so ``navigate_cmd`` covers its whole
command and ``base_height_cmd`` has nothing to record; a robot in that position
disables the base-height term rather than recording an empty column.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg

# Taken from the submodule, as Isaac Lab's own recorder cfgs do: ``isaaclab.utils`` is a
# lazy-loader stub, and its ``configclass`` attribute is the submodule rather than the
# decorator once anything has imported it -- which the recorder cfg import above does.
from isaaclab.utils.configclass import configclass

_ROBOT_ASSET = "robot"
_LOWER_BODY_ACTION_TERM = "lower_body_joint_pos"

# Where the hip-height channel sits in the G1 export's ``[vx, vy, wz, hip_height]``
# command. The H2 export has no such channel, so its command is 3 wide and this
# index is out of range -- which is checked rather than sliced.
_BASE_HEIGHT_INDEX = 3
_AUTODATA_EEFS = ("left", "right")
_AUTODATA_TERM_SIGNALS = ("left_hand_holds_object", "right_hand_holds_object")


class PostStepJointPositionTargetRecorder(RecorderTerm):
    """Record the commanded joint position target at the end of each step.

    Read post-step so it reflects the target the action terms applied this step
    (Pink IK upper body + AGILE lower body, combined by the articulation).
    """

    def record_post_step(self):
        robot = self._env.scene[_ROBOT_ASSET]
        return "joint_pos_target", robot.data.joint_pos_target.torch.clone()


class PostResetH2FeedbackRecorder(RecorderTerm):
    """Keep settled H2 controller history alongside the initial physical state."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        del env_ids  # H2's pinned policy supports exactly one environment.
        manager = self._env.action_manager
        if _LOWER_BODY_ACTION_TERM in manager.active_terms:
            term = manager.get_term(_LOWER_BODY_ACTION_TERM)
            if hasattr(term, "get_feedback_state"):
                robot = self._env.scene[_ROBOT_ASSET]
                return "initial_state", {
                    "h2_agile_feedback": term.get_feedback_state(),
                    "h2_actuator_targets": {
                        "position": robot.data.joint_pos_target.torch.clone(),
                        "velocity": robot.data.joint_vel_target.torch.clone(),
                    },
                }
        return None, None


def _locomotion_command(env):
    """The AGILE ``[vx, vy, wz, hip_height]`` command this step, or ``None`` when
    there is no lower-body action term (fixed-base mode)."""

    manager = env.action_manager
    if _LOWER_BODY_ACTION_TERM not in manager.active_terms:
        return None
    return getattr(manager.get_term(_LOWER_BODY_ACTION_TERM), "locomotion_command", None)


class PostStepNavigateCommandRecorder(RecorderTerm):
    """Record the AGILE navigate command ``[vx, vy, wz]`` (skipped in fixed mode)."""

    def record_post_step(self):
        cmd = _locomotion_command(self._env)
        if cmd is None:
            return None, None
        return "navigate_cmd", cmd[:, :3].clone()


class PostStepBaseHeightCommandRecorder(RecorderTerm):
    """Record the AGILE base-height command ``[hip_height]`` (skipped in fixed mode)."""

    def record_post_step(self):
        cmd = _locomotion_command(self._env)
        if cmd is None:
            return None, None
        if cmd.shape[-1] <= _BASE_HEIGHT_INDEX:
            raise RuntimeError(
                f"{_LOWER_BODY_ACTION_TERM!r} publishes a {cmd.shape[-1]}-wide locomotion command, which "
                "carries no base-height channel. Slicing it here would write a zero-width base_height_cmd "
                "column to every frame rather than raising. Set "
                "'recorders.record_post_step_base_height_cmd = None' for this robot, as the Arena "
                "recorder does."
            )
        return "base_height_cmd", cmd[:, _BASE_HEIGHT_INDEX : _BASE_HEIGHT_INDEX + 1].clone()


class PreStepArenaDatagenInfoRecorder(RecorderTerm):
    """Record the direct Arena pose/action contract consumed by AutoData."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._latched_signals = {
            name: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device) for name in _AUTODATA_TERM_SIGNALS
        }

    def record_post_reset(self, env_ids: Sequence[int] | None):
        index: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        for signal in self._latched_signals.values():
            signal[index] = False
        return None, None

    def record_pre_step(self):
        signal_observations = self._env.obs_buf["subtask_terms"]
        for name, latched in self._latched_signals.items():
            latched.logical_or_(signal_observations[name].reshape(-1).bool())

        target_poses = self._env.action_to_target_eef_pose(self._env.action_manager.action)
        return "obs/datagen_info", {
            "object_pose": self._env.get_object_poses(),
            "eef_pose": {name: self._env.get_robot_eef_pose(name) for name in _AUTODATA_EEFS},
            "target_eef_pose": {name: target_poses[name] for name in _AUTODATA_EEFS},
            "subtask_term_signals": {name: signal.clone() for name, signal in self._latched_signals.items()},
            "grasp_signals": {
                name: signal_observations[name].reshape(-1).bool().clone() for name in _AUTODATA_TERM_SIGNALS
            },
        }


@configclass
class PostStepJointPositionTargetRecorderCfg(RecorderTermCfg):
    """Config for :class:`PostStepJointPositionTargetRecorder`."""

    class_type: type[RecorderTerm] = PostStepJointPositionTargetRecorder


@configclass
class PostStepNavigateCommandRecorderCfg(RecorderTermCfg):
    """Config for :class:`PostStepNavigateCommandRecorder`."""

    class_type: type[RecorderTerm] = PostStepNavigateCommandRecorder


@configclass
class PostStepBaseHeightCommandRecorderCfg(RecorderTermCfg):
    """Config for :class:`PostStepBaseHeightCommandRecorder`."""

    class_type: type[RecorderTerm] = PostStepBaseHeightCommandRecorder


@configclass
class PreStepArenaDatagenInfoRecorderCfg(RecorderTermCfg):
    """Config for :class:`PreStepArenaDatagenInfoRecorder`."""

    class_type: type[RecorderTerm] = PreStepArenaDatagenInfoRecorder


@configclass
class WorkflowRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """App recorder: Isaac Lab action/state recorder plus the combined joint
    position command and the AGILE locomotion command (camera rides in ``obs``)."""

    record_post_step_joint_pos_target = PostStepJointPositionTargetRecorderCfg()
    record_post_step_navigate_cmd = PostStepNavigateCommandRecorderCfg()
    record_post_step_base_height_cmd = PostStepBaseHeightCommandRecorderCfg()
    record_post_reset_h2_feedback = RecorderTermCfg(class_type=PostResetH2FeedbackRecorder)


def add_workflow_recorder_terms(recorders: object) -> list[str]:
    """Add this app's recorder terms to a recorder manager cfg some other tool installed.

    Isaac AutoData assigns ``env_cfg.recorders`` itself and offers no way to ask for ours
    (``setup_env_config`` takes a ``recorder_cfg``, but its generation entry point never passes
    one), so a generated dataset would otherwise carry only Isaac Lab's fields -- no
    ``joint_pos_target``, and an ``action`` that covers just the joints the controller drives.

    Terms are added rather than the manager replaced, so the caller's own terms and its export
    settings survive; annotation, for one, installs a recorder carrying its subtask signals.
    Existing terms of the same name win, since the caller asked for those by name -- including
    a name it set to ``None``, which is how Isaac Lab's recorder manager is told to skip a term.

    Args:
        recorders: A recorder manager cfg, or the plain dict Isaac Lab also accepts.

    Returns:
        The names of the terms added, in declaration order.
    """

    inherited = set(ActionStateRecorderManagerCfg().__dict__)
    added = []
    for name, term_cfg in WorkflowRecorderManagerCfg().__dict__.items():
        if name in inherited or term_cfg is None:
            continue
        if isinstance(recorders, dict):
            if name not in recorders:
                recorders[name] = term_cfg
                added.append(name)
        elif not hasattr(recorders, name):
            setattr(recorders, name, term_cfg)
            added.append(name)
    return added


def add_arena_datagen_info_recorder_term(recorders: object) -> None:
    """Add the direct AutoData source contract to an Arena mimic recorder."""

    name = "record_pre_step_arena_datagen_info"
    term_cfg = PreStepArenaDatagenInfoRecorderCfg()
    if isinstance(recorders, dict):
        recorders.setdefault(name, term_cfg)
    elif not hasattr(recorders, name):
        setattr(recorders, name, term_cfg)

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

"""AGILE velocity locomotion for the H2 Plus, as an Isaac Lab action term.

The H2 export is an ONNX graph with a three-wide velocity command and recurrent
history owned by its caller. Its action output is already a joint-position
target, its joint observations are absolute and positional over all 31 body
joints, and it consumes scalar-first quaternions.

The graph declares a batch dimension of exactly one, so this term intentionally
drives one environment. Supporting more requires a dynamic-batch re-export.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_OBSERVATION_WIDTHS = {
    "root_link_quat_w": 4,
    "root_ang_vel_b": 3,
    "velocity_commands": 3,
    "joint_pos": 31,
    "joint_vel": 31,
}
_FEEDBACK_PORTS = (
    "last_actions",
    "base_ang_vel_history",
    "projected_gravity_history",
    "velocity_commands_history",
    "controlled_joint_pos_history",
    "controlled_joint_vel_history",
    "actions_history",
)
_ACTION_PORT = "action_joint_pos"
_CONSTANT_GAIN_PORTS = ("action_joint_pos_kp_gains", "action_joint_pos_kd_gains")
_SCALAR_FIRST_FROM_XYZW = (3, 0, 1, 2)


class H2AgileVelocityAction(ActionTerm):
    """Drive the H2 Plus lower body from the pinned AGILE ONNX export."""

    cfg: H2AgileVelocityActionCfg
    _asset: Articulation
    locomotion_command: torch.Tensor | None = None

    def __init__(self, cfg: H2AgileVelocityActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if self.num_envs != 1:
            raise ValueError(
                "The H2 AGILE export declares a batch dimension of 1, so it drives one environment; "
                f"got num_envs={self.num_envs}. Supporting more requires re-exporting the graph with a "
                "dynamic batch dimension."
            )

        import onnxruntime  # imported here so host-only imports do not require it

        self._session = onnxruntime.InferenceSession(cfg.policy_path, providers=["CPUExecutionProvider"])
        self._output_names = [output.name for output in self._session.get_outputs()]
        self._inference_output_names = [_ACTION_PORT, *(f"{port}_out" for port in _FEEDBACK_PORTS)]
        self._verify_graph_ports()
        self._verify_joint_order()

        self._joint_ids, driven_names = self._asset.find_joints(list(cfg.driven_joint_names), preserve_order=True)
        self._proprio_joint_ids, proprio_names = self._asset.find_joints(
            list(cfg.proprio_joint_names), preserve_order=True
        )
        if tuple(driven_names) != tuple(cfg.driven_joint_names):
            raise ValueError(f"H2 AGILE driven-joint projection changed: {driven_names}")
        if tuple(proprio_names) != tuple(cfg.proprio_joint_names):
            raise ValueError(f"H2 AGILE proprioception projection changed: {proprio_names}")

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        observation_width = sum(_OBSERVATION_WIDTHS.values())
        self._device_observations = torch.empty(
            (self.num_envs, observation_width), dtype=torch.float32, device=self.device
        )
        self._host_observations = torch.empty(
            (self.num_envs, observation_width),
            dtype=torch.float32,
            device="cpu",
            pin_memory=self._device_observations.device.type == "cuda",
        )
        self._observation_slices: dict[str, slice] = {}
        self._host_observation_arrays: dict[str, np.ndarray] = {}
        host_array = self._host_observations.numpy()
        start = 0
        for name, width in _OBSERVATION_WIDTHS.items():
            value_slice = slice(start, start + width)
            self._observation_slices[name] = value_slice
            self._host_observation_arrays[name] = host_array[:, value_slice]
            start += width

        self._feedback: dict[str, np.ndarray] = {}
        self._reset_feedback()

    @property
    def action_dim(self) -> int:
        return self.cfg.command_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        del env_ids
        self._reset_feedback()
        self.locomotion_command = None

    def get_feedback_state(self) -> dict[str, torch.Tensor]:
        """Snapshot the recurrent state needed to resume a settled recording."""

        return {name: torch.from_numpy(value.copy()) for name, value in self._feedback.items()}

    def restore_feedback_state(self, state: dict[str, torch.Tensor]) -> None:
        """Validate the complete snapshot before replacing any live history."""

        if set(state) != set(self._feedback):
            raise ValueError("H2 AGILE feedback snapshot has missing or unexpected ports")
        restored = {}
        for name, reference in self._feedback.items():
            value = state[name].detach().cpu().numpy()
            if value.shape != reference.shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid H2 AGILE feedback snapshot for {name}: {value.shape}")
            restored[name] = np.array(value, dtype=np.float32, copy=True)
        self._feedback = restored

    @torch.no_grad()
    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions)
        if self.locomotion_command is None:
            self.locomotion_command = torch.empty_like(self._raw_actions)
        self.locomotion_command.copy_(self._raw_actions)

        data = self._asset.data
        observations = self._device_observations
        slices = self._observation_slices
        observations[:, slices["root_link_quat_w"]] = data.root_link_quat_w.torch[:, _SCALAR_FIRST_FROM_XYZW]
        observations[:, slices["root_ang_vel_b"]] = data.root_ang_vel_b.torch
        observations[:, slices["velocity_commands"]] = self._raw_actions
        observations[:, slices["joint_pos"]] = data.joint_pos.torch[:, self._proprio_joint_ids]
        observations[:, slices["joint_vel"]] = data.joint_vel.torch[:, self._proprio_joint_ids]

        self._host_observations.copy_(observations, non_blocking=observations.device.type == "cuda")
        if observations.device.type == "cuda":
            torch.cuda.current_stream(observations.device).synchronize()

        inputs = dict(self._host_observation_arrays)
        inputs.update({f"{port}_in": self._feedback[port] for port in _FEEDBACK_PORTS})
        values = self._session.run(self._inference_output_names, inputs)
        outputs = dict(zip(self._inference_output_names, values, strict=True))
        for port in _FEEDBACK_PORTS:
            self._feedback[port] = outputs[f"{port}_out"]

        # The graph already applies its action scale and reference pose.
        self._processed_actions.copy_(torch.from_numpy(outputs[_ACTION_PORT]))

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(target=self._processed_actions, joint_ids=self._joint_ids)

    def _reset_feedback(self) -> None:
        driven = len(self.cfg.driven_joint_names)
        depth = self.cfg.history_length
        from e2e_robot_workflows.controllers.h2_feedback import feedback_shapes

        widths = feedback_shapes(num_envs=self.num_envs, driven=driven, depth=depth, command_dim=self.cfg.command_dim)
        self._feedback = {port: np.zeros(shape, dtype=np.float32) for port, shape in widths.items()}

    def _verify_graph_ports(self) -> None:
        expected_inputs = set(_OBSERVATION_WIDTHS) | {f"{port}_in" for port in _FEEDBACK_PORTS}
        expected_outputs = {_ACTION_PORT, *_CONSTANT_GAIN_PORTS} | {f"{port}_out" for port in _FEEDBACK_PORTS}
        actual_inputs = {model_input.name for model_input in self._session.get_inputs()}
        actual_outputs = set(self._output_names)
        if actual_inputs != expected_inputs or actual_outputs != expected_outputs:
            raise ValueError(
                "The AGILE export does not present the expected interface.\n"
                f"  inputs missing:    {sorted(expected_inputs - actual_inputs)}\n"
                f"  inputs unexpected: {sorted(actual_inputs - expected_inputs)}\n"
                f"  outputs missing:    {sorted(expected_outputs - actual_outputs)}\n"
                f"  outputs unexpected: {sorted(actual_outputs - expected_outputs)}"
            )

    def _verify_joint_order(self) -> None:
        live = tuple(self._asset.joint_names)
        declared = tuple(self.cfg.expected_joint_order)
        if live == declared:
            return
        detail = f"  live:     {len(live)} joints\n  declared: {len(declared)} joints"
        if len(live) == len(declared):
            first = next(index for index, pair in enumerate(zip(live, declared, strict=True)) if pair[0] != pair[1])
            detail = f"  first difference at index {first}: live {live[first]!r}, declared {declared[first]!r}"
        raise ValueError(
            "The H2 articulation's joint order does not match the order the AGILE export was "
            f"characterized against.\n{detail}\n"
            "The export's joint_pos and joint_vel ports are positional, so continuing would feed it a "
            "scrambled observation."
        )


@configclass
class H2AgileVelocityActionCfg(ActionTermCfg):
    """Configuration for :class:`H2AgileVelocityAction`."""

    class_type: type[ActionTerm] = H2AgileVelocityAction
    policy_path: str = ""
    driven_joint_names: tuple[str, ...] = ()
    proprio_joint_names: tuple[str, ...] = ()
    expected_joint_order: tuple[str, ...] = ()
    history_length: int = 5
    command_dim: int = 3


__all__ = ["H2AgileVelocityAction", "H2AgileVelocityActionCfg"]

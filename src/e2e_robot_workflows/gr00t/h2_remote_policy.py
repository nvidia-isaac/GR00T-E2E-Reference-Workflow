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

"""E2E-owned GR00T remote adapter for Arena's 61-D H2 joint embodiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
import yaml
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.action_scheduling import ActionChunkScheduler
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena_gr00t.policy.gr00t_core import (
    build_gr00t_policy_observations,
    extract_obs_numpy_from_torch,
)
from isaaclab_arena_gr00t.utils.io_utils import (
    load_gr00t_modality_config_from_file,
    load_robot_joints_config_from_yaml,
)

from e2e_robot_workflows.gr00t.h2_mapping import (
    H2_ARENA_JOINT_ACTION_DIM,
    build_h2_gr00t_action_np,
    validate_h2_action_schedule,
    validate_h2_joint_configs,
    validate_h2_live_joint_limits,
)
from e2e_robot_workflows.robots.h2 import load_h2_pink_contract, load_h2_robot_config


@dataclass(frozen=True)
class H2ClosedloopConfig:
    language_instruction: str
    robot_type: str
    action_horizon: int
    embodiment_tag: str
    modality_config_path: Path
    policy_joints_config_path: Path
    action_joints_config_path: Path
    state_joints_config_path: Path
    action_chunk_length: int
    pov_cam_name_sim: tuple[str, ...]
    navigate_command_limits: tuple[tuple[float, float], ...]
    target_image_size: tuple[int, int, int] = (480, 640, 3)

    @classmethod
    def load(cls, path: str | Path) -> H2ClosedloopConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data["pov_cam_name_sim"] = tuple(data.get("pov_cam_name_sim", ()))
        data["navigate_command_limits"] = load_h2_robot_config().navigate_command_limits
        data["target_image_size"] = tuple(data.get("target_image_size", (480, 640, 3)))
        for key in (
            "modality_config_path",
            "policy_joints_config_path",
            "action_joints_config_path",
            "state_joints_config_path",
        ):
            data[key] = Path(data[key])
        config = cls(**data)
        if config.robot_type != "unitree_h2plus":
            raise ValueError(f"unsupported H2 evaluation robot type {config.robot_type!r}")
        if config.embodiment_tag != "NEW_EMBODIMENT":
            raise ValueError("H2 evaluation requires GR00T's NEW_EMBODIMENT tag")
        validate_h2_action_schedule(config.action_horizon, config.action_chunk_length)
        if config.pov_cam_name_sim != ("robot_head_cam_rgb",):
            raise ValueError("H2 evaluation requires the Arena head camera")
        for config_path in (
            config.modality_config_path,
            config.policy_joints_config_path,
            config.action_joints_config_path,
            config.state_joints_config_path,
        ):
            if not config_path.is_file():
                raise FileNotFoundError(config_path)
        return config


@dataclass
class H2Gr00tRemotePolicyCfg(PolicyCfg):
    policy_config_yaml_path: str = "configs/stages/gr00t/h2_closedloop_eval.yaml"
    policy_device: str = "cuda"
    num_envs: int = 1
    remote_host: str = "localhost"
    remote_port: int = 5555
    remote_api_token: str | None = None
    remote_kill_on_exit: bool = False


@register_policy
class H2Gr00tRemotePolicy(PolicyBase[H2Gr00tRemotePolicyCfg]):
    """Translate H2 observations and learned joint groups without G1 task modes."""

    name = "e2e_h2_gr00t_remote"

    def __init__(self, config: H2Gr00tRemotePolicyCfg) -> None:
        super().__init__(config)
        self.policy_config = H2ClosedloopConfig.load(config.policy_config_yaml_path)
        self.num_envs = config.num_envs
        self.device = config.policy_device
        self.policy_joints = load_robot_joints_config_from_yaml(self.policy_config.policy_joints_config_path)
        self.action_joints = load_robot_joints_config_from_yaml(self.policy_config.action_joints_config_path)
        self.state_joints = load_robot_joints_config_from_yaml(self.policy_config.state_joints_config_path)
        validate_h2_joint_configs(self.policy_joints, self.action_joints, self.state_joints)
        self.modalities = load_gr00t_modality_config_from_file(
            self.policy_config.modality_config_path,
            self.policy_config.embodiment_tag,
        )
        self._scheduler = ActionChunkScheduler(
            num_envs=self.num_envs,
            action_chunk_length=self.policy_config.action_chunk_length,
            action_horizon=self.policy_config.action_horizon,
            action_dim=H2_ARENA_JOINT_ACTION_DIM,
            device=self.device,
            dtype=torch.float,
        )

        from gr00t.policy.server_client import PolicyClient

        self._client: Any | None = PolicyClient(
            host=config.remote_host,
            port=config.remote_port,
            api_token=config.remote_api_token,
            strict=False,
        )
        if not self._client.ping():
            raise ConnectionError(f"Cannot reach GR00T policy server at {config.remote_host}:{config.remote_port}")
        self.task_description: str | None = None

    def set_task_description(self, task_description: str | None) -> str:
        resolved = task_description or self.policy_config.language_instruction
        if not resolved:
            raise ValueError("H2 GR00T evaluation requires a language instruction")
        self.task_description = resolved
        return resolved

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        return self._scheduler.get_action(
            lambda: self._get_action_chunk(env, observation),
            hold_action=self._hold_action(observation),
        )

    def _hold_action(self, observation: dict[str, Any]) -> torch.Tensor:
        joint_pos = observation["policy"]["robot_joint_pos"].to(device=self.device, dtype=torch.float)
        hold = torch.zeros((self.num_envs, H2_ARENA_JOINT_ACTION_DIM), device=self.device)
        for joint, action_index in self.action_joints.items():
            hold[:, action_index] = joint_pos[:, self.state_joints[joint]]
        return hold

    def _get_action_chunk(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        if self.task_description is None or self._client is None:
            raise RuntimeError("H2 GR00T policy is not ready")
        images, joint_pos = extract_obs_numpy_from_torch(
            nested_obs=observation,
            camera_names=list(self.policy_config.pov_cam_name_sim),
        )
        request = build_gr00t_policy_observations(
            rgb_list_np=images,
            joint_pos_sim_np=joint_pos,
            task_description=self.task_description,
            policy_config=self.policy_config,
            robot_state_joints_config=self.state_joints,
            policy_joints_config=self.policy_joints,
            modality_configs=self.modalities,
        )
        policy_action, _ = self._client.get_action(request)
        base_env = getattr(env, "unwrapped", env)
        if base_env is None or not hasattr(base_env, "scene"):
            raise ValueError("H2 GR00T evaluation requires the live Arena environment for action-limit validation")
        robot = base_env.scene["robot"]
        raw_limits = robot.data.joint_pos_limits
        limit_tensor = getattr(raw_limits, "torch", raw_limits)
        limit_values = limit_tensor[0].detach().cpu().numpy()
        if limit_values.shape != (len(robot.joint_names), 2):
            raise ValueError(
                f"H2 live joint limits have shape {limit_values.shape}, expected {(len(robot.joint_names), 2)}"
            )
        live_limits = {
            name: (float(limit_values[index, 0]), float(limit_values[index, 1]))
            for index, name in enumerate(robot.joint_names)
        }
        pinned_limits = load_h2_pink_contract().learned_joint_limits
        validate_h2_live_joint_limits(live_limits, pinned_limits)
        action = build_h2_gr00t_action_np(
            policy_action,
            self.policy_joints,
            self.action_joints,
            joint_position_limits=pinned_limits,
            navigation_limits=self.policy_config.navigate_command_limits,
        )
        result = torch.from_numpy(action).to(device=self.device, dtype=torch.float)
        if result.shape[:2] != (self.num_envs, self.policy_config.action_horizon):
            raise ValueError(f"H2 GR00T server returned action shape {tuple(result.shape)}")
        return result

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._client is None:
            raise RuntimeError("H2 GR00T policy is closed")
        self._client.reset()
        self._scheduler.reset(slice(None) if env_ids is None else env_ids)

    @property
    def is_remote(self) -> bool:
        return True

    def shutdown_remote(self, *, kill_server: bool = False) -> None:
        """Close the client; the separately launched server is never owned here."""

        if kill_server:
            raise ValueError("The H2 evaluator does not own the external GR00T server process")
        self.close()

    def close(self) -> None:
        client = self._client
        try:
            if client is not None:
                socket = getattr(client, "socket", None)
                context = getattr(client, "context", None)
                if socket is not None:
                    socket.close(linger=0)
                if context is not None:
                    context.term()
        finally:
            self._client = None


__all__ = ["H2ClosedloopConfig", "H2Gr00tRemotePolicy", "H2Gr00tRemotePolicyCfg"]

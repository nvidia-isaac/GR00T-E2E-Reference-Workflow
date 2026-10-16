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

"""Check Arena's GR00T client contract without starting the simulator."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from numbers import Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

MODALITY_NAMES = ("video", "state", "action", "language")
REQUIRED_STATISTICS = ("mean", "std", "min", "max", "q01", "q99")


def _serialize_modality(config: Any) -> dict[str, Any]:
    """Convert a GR00T modality dataclass to its checkpoint JSON form."""
    return json.loads(json.dumps(asdict(config), default=lambda value: value.name))


def _require_checkpoint_contract(
    expected: dict[str, dict[str, Any]],
    checkpoint: dict[str, dict[str, Any]],
    expected_dimensions: dict[str, dict[str, int]],
    checkpoint_statistics: dict[str, Any],
) -> None:
    """Reject processor or statistics metadata that differs from Arena."""
    for modality_name in MODALITY_NAMES:
        if checkpoint.get(modality_name) != expected[modality_name]:
            raise SystemExit(f"Checkpoint modality mismatch: {modality_name}")
    for modality_name, dimensions in expected_dimensions.items():
        groups = checkpoint_statistics.get(modality_name)
        if not isinstance(groups, dict) or set(groups) != set(dimensions):
            raise SystemExit(
                f"Checkpoint statistics mismatch: {modality_name} groups expected "
                f"{sorted(dimensions)!r}, got {sorted(groups) if isinstance(groups, dict) else groups!r}"
            )
        for group_name, expected_width in dimensions.items():
            group_statistics = groups[group_name]
            if not isinstance(group_statistics, dict) or set(group_statistics) != set(REQUIRED_STATISTICS):
                raise SystemExit(
                    f"Checkpoint statistics mismatch: {modality_name}.{group_name} must contain exactly "
                    f"{list(REQUIRED_STATISTICS)!r}"
                )
            for statistic_name in REQUIRED_STATISTICS:
                values = group_statistics[statistic_name]
                if (
                    not isinstance(values, list)
                    or len(values) != expected_width
                    or any(isinstance(value, (list, dict)) for value in values)
                ):
                    raise SystemExit(
                        f"Checkpoint statistics mismatch: {modality_name}.{group_name}.{statistic_name} "
                        f"must be a rank-one vector of width {expected_width}"
                    )
                if any(
                    isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                    for value in values
                ):
                    raise SystemExit(
                        f"Checkpoint statistics mismatch: {modality_name}.{group_name}.{statistic_name} "
                        "must contain only finite numbers"
                    )


def _checkpoint_summary(checkpoint_root_arg: str | None, state_width: int, action_width: int) -> str:
    """Describe exactly which part of the checkpoint contract was inspected."""
    if checkpoint_root_arg is None:
        return (
            "validation_scope=config_only, checkpoint_metadata=not_checked, "
            f"expected_checkpoint_state={state_width}, expected_checkpoint_action={action_width}"
        )
    return (
        "validation_scope=checkpoint, checkpoint_metadata=validated, "
        f"expected_checkpoint_state={state_width}, expected_checkpoint_action={action_width}"
    )


def _checkpoint_contract(
    config: Any,
    modalities: dict[str, Any],
    group_dimensions: dict[str, int],
    checkpoint_root_arg: str | None,
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    expected_modalities = {name: _serialize_modality(modalities[name]) for name in MODALITY_NAMES}
    expected_dimensions = {
        name: {key: group_dimensions[key] for key in expected_modalities[name]["modality_keys"]}
        for name in ("state", "action")
    }
    if checkpoint_root_arg is not None:
        checkpoint_root = Path(checkpoint_root_arg)
        processor = json.loads((checkpoint_root / "processor_config.json").read_text(encoding="utf-8"))
        statistics = json.loads((checkpoint_root / "statistics.json").read_text(encoding="utf-8"))
        embodiment_key = config.embodiment_tag.lower()
        checkpoint_modalities = processor["processor_kwargs"]["modality_configs"].get(embodiment_key)
        if not isinstance(checkpoint_modalities, dict):
            raise SystemExit(f"Checkpoint processor is missing embodiment {embodiment_key}")
        _require_checkpoint_contract(
            expected_modalities,
            checkpoint_modalities,
            expected_dimensions,
            statistics.get(embodiment_key, {}),
        )
    return expected_modalities, expected_dimensions


def _ordered_live_joint_limits(
    joint_names: list[str], live_joint_limits: dict[str, tuple[float, float]]
) -> list[tuple[float, float]]:
    """Order an exact live-plant limit map like the mocked articulation."""

    if set(live_joint_limits) != set(joint_names):
        raise SystemExit("H2 mocked remote policy limits do not cover the exact 75-joint plant")
    return [live_joint_limits[joint] for joint in joint_names]


def _exercise_h2_remote_policy(
    policy_class: type[Any],
    policy_config: Any,
    policy_joints: dict[str, Any],
    live_joint_limits: dict[str, tuple[float, float]],
) -> None:
    """Exercise the real Arena request builder and scheduler without a policy server."""

    import numpy as np
    import torch
    from gr00t.policy import server_client

    from e2e_robot_workflows.gr00t.h2_remote_policy import H2Gr00tRemotePolicyCfg

    action_groups = ("left_arm", "right_arm", "left_hand", "right_hand")

    class FakePolicyClient:
        def __init__(self, **_: Any) -> None:
            self.calls = 0
            self.last_request: dict[str, Any] | None = None
            self.reset_count = 0
            self.socket = None
            self.context = None

        def ping(self) -> bool:
            return True

        def get_action(self, request: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
            self.calls += 1
            self.last_request = request
            action = {
                name: np.full(
                    (1, policy_config.action_horizon, len(policy_joints[name])),
                    0.01 * self.calls,
                    np.float32,
                )
                for name in action_groups
            }
            action["navigate_command"] = np.full((1, policy_config.action_horizon, 3), 0.25 * self.calls, np.float32)
            return action, {}

        def reset(self) -> None:
            self.reset_count += 1

    original_client = server_client.PolicyClient
    policy = None
    try:
        server_client.PolicyClient = FakePolicyClient
        policy = policy_class(H2Gr00tRemotePolicyCfg(policy_device="cpu", num_envs=1))
        policy.set_task_description(None)
        observation = {
            "policy": {"robot_joint_pos": torch.zeros((1, 75), dtype=torch.float32)},
            "camera_obs": {"robot_head_cam_rgb": torch.zeros((1, 480, 640, 3), dtype=torch.uint8)},
        }
        joint_names = [joint for joints in policy_joints.values() for joint in joints]
        joint_limits = torch.tensor(
            [_ordered_live_joint_limits(joint_names, live_joint_limits)],
            dtype=torch.float32,
        )

        environment = SimpleNamespace(
            scene={
                "robot": SimpleNamespace(
                    joint_names=joint_names,
                    data=SimpleNamespace(joint_pos_limits=joint_limits),
                )
            }
        )
        actions = [policy.get_action(environment, observation) for _ in range(policy_config.action_horizon + 1)]
        if any(action.shape != (1, 61) or not bool(torch.isfinite(action).all()) for action in actions):
            raise SystemExit("H2 mocked remote policy did not return finite 61-D actions")
        if policy._client.calls != 2:
            raise SystemExit(f"H2 action scheduler requested {policy._client.calls} chunks, expected 2")
        if not torch.all(actions[0][:, 58:] == 0.25) or not torch.all(actions[-1][:, 58:] == 0.5):
            raise SystemExit("H2 mocked remote policy did not preserve navigation commands")

        request = policy._client.last_request
        if request is None:
            raise SystemExit("H2 mocked remote policy sent no observation request")
        expected_state_shapes = {name: (1, 1, len(joints)) for name, joints in policy_joints.items()}
        actual_state_shapes = {name: tuple(np.asarray(value).shape) for name, value in request["state"].items()}
        if actual_state_shapes != expected_state_shapes:
            raise SystemExit(
                f"H2 remote request state shapes differ: expected {expected_state_shapes}, got {actual_state_shapes}"
            )
        if tuple(np.asarray(request["video"]["ego_view"]).shape) != (1, 1, 480, 640, 3):
            raise SystemExit("H2 remote request camera shape differs from the evaluation contract")
        if tuple(np.asarray(request["language"]["annotation.human.task_description"]).shape) != (1, 1):
            raise SystemExit("H2 remote request language shape differs from the evaluation contract")
        policy.reset()
        if policy._client.reset_count != 1:
            raise SystemExit("H2 remote reset did not reach the policy client")
    finally:
        if policy is not None:
            policy.close()
        server_client.PolicyClient = original_client


def _check_h2(config_path: Path, checkpoint_root_arg: str | None) -> None:
    from isaaclab_arena_gr00t.utils.io_utils import (
        load_gr00t_modality_config_from_file,
        load_robot_joints_config_from_yaml,
    )

    from e2e_robot_workflows.gr00t.h2_mapping import (
        H2_ARENA_JOINT_ACTION_DIM,
        validate_h2_joint_configs,
    )
    from e2e_robot_workflows.gr00t.h2_remote_policy import H2ClosedloopConfig, H2Gr00tRemotePolicy
    from e2e_robot_workflows.robots.h2 import load_h2_pink_contract, load_h2_robot_config

    config = H2ClosedloopConfig.load(config_path)
    policy_joints = load_robot_joints_config_from_yaml(config.policy_joints_config_path)
    action_joints = load_robot_joints_config_from_yaml(config.action_joints_config_path)
    state_joints = load_robot_joints_config_from_yaml(config.state_joints_config_path)
    validate_h2_joint_configs(policy_joints, action_joints, state_joints)
    modalities = load_gr00t_modality_config_from_file(config.modality_config_path, config.embodiment_tag)
    dimensions = {name: len(joints) for name, joints in policy_joints.items()}
    dimensions["navigate_command"] = 3
    _, checkpoint_dimensions = _checkpoint_contract(config, modalities, dimensions, checkpoint_root_arg)

    if sum(checkpoint_dimensions["state"].values()) != 75:
        raise SystemExit("H2 checkpoint state width must be 75")
    if sum(checkpoint_dimensions["action"].values()) != H2_ARENA_JOINT_ACTION_DIM:
        raise SystemExit("H2 checkpoint action width must be 61")

    robot = load_h2_robot_config()
    plant = load_h2_pink_contract()
    robot.resolve_policy()
    urdf_info = plant.validate_urdf(plant.resolve_urdf())
    _exercise_h2_remote_policy(
        H2Gr00tRemotePolicy,
        config,
        policy_joints,
        dict(urdf_info.joint_position_limits),
    )
    print(
        f"H2 GR00T evaluation contract passed: policy={H2Gr00tRemotePolicy.__name__}, "
        f"{_checkpoint_summary(checkpoint_root_arg, 75, 61)}, sim_state=75, sim_action=61, "
        f"asset_source={plant.source_contract_sha256}, arena_usd={plant.arena_usd_sha256}, "
        f"agile_revision={robot.policy_revision}, agile_sha256={robot.policy_sha256}, "
        "horizon=40, camera=robot_head_cam_rgb, mocked_remote_path=ok"
    )


def main() -> None:
    """Validate Arena alone or Arena together with a checkpoint directory."""
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: check_gr00t_eval.py <policy_config_yaml> [checkpoint_root]")

    config_path = Path(sys.argv[1])
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if raw_config.get("robot_type") == "unitree_h2plus":
        _check_h2(config_path, sys.argv[2] if len(sys.argv) == 3 else None)
        return

    from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import (
        Gr00tClosedloopPolicyCfg,
        TaskMode,
    )
    from isaaclab_arena_gr00t.policy.gr00t_core import compute_action_dim, load_gr00t_joint_configs
    from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import Gr00tRemoteClosedloopPolicy
    from isaaclab_arena_gr00t.utils.io_utils import create_config_from_yaml, load_gr00t_modality_config_from_file

    config = create_config_from_yaml(config_path, Gr00tClosedloopPolicyCfg)
    policy_joints, action_joints, state_joints = load_gr00t_joint_configs(config)
    modalities = load_gr00t_modality_config_from_file(config.modality_config_path, config.embodiment_tag)
    group_dimensions = {name: len(joints) for name, joints in policy_joints.items()}
    group_dimensions.update({"base_height_command": 1, "navigate_command": 3})
    _, checkpoint_dimensions = _checkpoint_contract(
        config,
        modalities,
        group_dimensions,
        sys.argv[2] if len(sys.argv) == 3 else None,
    )

    assert config.task_mode_name == TaskMode.G1_LOCOMANIPULATION.value
    assert config.pov_cam_name_sim == ["robot_head_cam_rgb"]
    assert len(action_joints) == len(state_joints) == 43
    assert compute_action_dim(TaskMode(config.task_mode_name), action_joints) == 50
    assert config.action_horizon == config.action_chunk_length == 40
    assert modalities["action"].delta_indices == list(range(40))
    assert sum(checkpoint_dimensions["state"].values()) == 31
    assert sum(checkpoint_dimensions["action"].values()) == 35

    print(
        f"GR00T evaluation contract passed: policy={Gr00tRemoteClosedloopPolicy.__name__}, "
        f"{_checkpoint_summary(sys.argv[2] if len(sys.argv) == 3 else None, 31, 35)}, "
        "sim_state=43, sim_action=50, "
        "horizon=40, camera=robot_head_cam_rgb"
    )


if __name__ == "__main__":
    main()

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

"""E2E task semantics for Arena Pink mimic environments."""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
import torch
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mimic_env_cfg import (
    MimicEnvCfg,
    SubTaskConfig,
    SubTaskConstraintConfig,
    SubTaskConstraintCoordinationScheme,
    SubTaskConstraintType,
)
from isaaclab.utils.configclass import configclass
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv

from e2e_robot_workflows.arena.h2_state import H2ResetStateMixin
from e2e_robot_workflows.arena.mimic_actions import (
    h2_body_pose_in_pelvis_frame,
    normalize_h2_mimic_action_inputs,
    normalize_mimic_action_inputs,
    require_h2_action_width,
)
from e2e_robot_workflows.robots.h2 import H2_PINK_ACTION_DIM, load_h2_pink_contract

ACTION_NOISE = 0.0


def make_arena_mimic_env(cfg, render_mode: str | None = None, **kwargs):
    """Construct Arena's G1 mimic env with only AutoData batch-shape normalization."""

    from isaaclab_arena.embodiments.g1.g1 import G1MimicEnv

    class E2EG1MimicEnv(G1MimicEnv):
        def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
            poses = super().get_object_poses(env_ids=env_ids)
            # The support is recorded scene state, not a manipulable Mimic anchor.
            support_name = getattr(self.cfg, "e2e_support_asset_name", None)
            return {name: pose for name, pose in poses.items() if name != support_name}

        def target_eef_pose_to_action(
            self,
            target_eef_pose_dict,
            gripper_action_dict,
            action_noise_dict=None,
            env_id: int = 0,
        ):
            target_poses, passthrough = normalize_mimic_action_inputs(
                target_eef_pose_dict,
                gripper_action_dict,
            )
            return super().target_eef_pose_to_action(
                target_poses,
                passthrough,
                action_noise_dict=action_noise_dict,
                env_id=env_id,
            )

    return E2EG1MimicEnv(cfg=cfg, render_mode=render_mode, **kwargs)


class H2ReplayEnv(H2ResetStateMixin, IsaacLabArenaManagerBasedRLEnv):
    """Joint-layout replay with the recorded H2 controller and actuator state."""


class H2MimicEnv(H2ResetStateMixin, ManagerBasedRLMimicEnv):
    """Expose the H2 58-D Pink semantic and 3-D AGILE passthrough to AutoData."""

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        if getattr(self.cfg, "e2e_h2_prepare_on_reset", False):
            from e2e_robot_workflows.devices.h2_ready import prepare_h2_recording

            self.h2_reset_preparation = prepare_h2_recording(self, embodiment="h2_agile_pink")
            return self.obs_buf, result[1]
        return result

    def _pelvis_relative_body_pose(self, body_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        indices: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        robot = self.scene["robot"]
        origins = self.scene.env_origins[indices]
        pelvis = robot.data.body_link_pose_w.torch[indices, robot.body_names.index("pelvis")].clone()
        body = robot.data.body_link_pose_w.torch[indices, robot.body_names.index(body_name)].clone()
        pelvis[:, :3] -= origins
        body[:, :3] -= origins
        pelvis_matrix = PoseUtils.make_pose(pelvis[:, :3], PoseUtils.matrix_from_quat(pelvis[:, 3:7]))
        body_matrix = PoseUtils.make_pose(body[:, :3], PoseUtils.matrix_from_quat(body[:, 3:7]))
        return torch.matmul(PoseUtils.pose_inv(pelvis_matrix), body_matrix)

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        contract = load_h2_pink_contract()
        frame_by_eef = {
            "left": contract.flange_frames[0],
            "right": contract.flange_frames[1],
            "body": contract.base_frame,
        }
        try:
            frame = frame_by_eef[eef_name]
        except KeyError as error:
            raise ValueError(f"unsupported H2 end effector {eef_name!r}") from error
        return self._pelvis_relative_body_pose(frame, env_ids)

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        del env_id
        poses, passthrough = normalize_h2_mimic_action_inputs(target_eef_pose_dict, gripper_action_dict)
        pose_actions: dict[str, torch.Tensor] = {}
        for hand in ("left", "right"):
            position, rotation = PoseUtils.unmake_pose(poses[hand].clone())
            quaternion = PoseUtils.quat_from_matrix(rotation)
            if action_noise_dict is not None:
                scale = action_noise_dict[hand]
                position += scale * torch.randn_like(position)
                quaternion += scale * torch.randn_like(quaternion)
                quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
            pose_actions[hand] = torch.cat((position, quaternion), dim=-1)
        contract = load_h2_pink_contract()
        upper_body = contract.pack_action(
            pose_actions["left"],
            pose_actions["right"],
            passthrough["left"],
            passthrough["right"],
        )
        return torch.cat((upper_body, passthrough["body"]), dim=-1)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        require_h2_action_width(action, rank=2)
        left_pose, right_pose, _, _ = load_h2_pink_contract().unpack_action(action[:, :H2_PINK_ACTION_DIM])
        return {
            "left": PoseUtils.make_pose(left_pose[:, :3], PoseUtils.matrix_from_quat(left_pose[:, 3:7])),
            "right": PoseUtils.make_pose(right_pose[:, :3], PoseUtils.matrix_from_quat(right_pose[:, 3:7])),
            "body": h2_body_pose_in_pelvis_frame(action),
        }

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        require_h2_action_width(actions)
        _, _, left_hand, right_hand = load_h2_pink_contract().unpack_action(actions[..., :H2_PINK_ACTION_DIM])
        return {"left": left_hand, "right": right_hand, "body": actions[..., -3:]}

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        indices: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        robot = self.scene["robot"]
        origins = self.scene.env_origins[indices]
        pelvis = robot.data.body_link_pose_w.torch[indices, robot.body_names.index("pelvis")].clone()
        pelvis[:, :3] -= origins
        pelvis_inverse = PoseUtils.pose_inv(
            PoseUtils.make_pose(pelvis[:, :3], PoseUtils.matrix_from_quat(pelvis[:, 3:7]))
        )
        object_poses = {}
        support_name = getattr(self.cfg, "e2e_support_asset_name", None)
        for object_name, object_state in self.scene.get_state(is_relative=True)["rigid_object"].items():
            if object_name == support_name:
                continue
            root_pose = object_state["root_pose"][indices]
            world_pose = PoseUtils.make_pose(root_pose[:, :3], PoseUtils.matrix_from_quat(root_pose[:, 3:7]))
            object_poses[object_name] = torch.matmul(pelvis_inverse, world_pose)
        return object_poses


def compose_arena_mimic_environment(
    *,
    device: str,
    num_envs: int,
    enable_cameras: bool,
    manipulation_hand: str,
    seed: int,
    embodiment: str | None = None,
    teleop_device: str | None = None,
    table_height_offset_range_m: Sequence[float] | None = None,
    movable_support: bool = False,
):
    """Compose and register the canonical single-environment Arena mimic task."""

    if num_envs != 1:
        raise ValueError("Arena Pink generation is qualified sequentially with one environment")

    from e2e_robot_workflows.arena.composition import compose
    from e2e_robot_workflows.arena.embodiment_capabilities import G1_ARENA_TELEOP_EMBODIMENT

    resolved_embodiment = embodiment or G1_ARENA_TELEOP_EMBODIMENT

    composed = compose(
        embodiment=resolved_embodiment,
        device=device,
        num_envs=num_envs,
        enable_cameras=enable_cameras,
        disable_task_terminations=False,
        mimic=True,
        manipulation_hand=manipulation_hand,
        seed=seed,
        teleop_device=teleop_device,
        table_height_offset_range_m=table_height_offset_range_m,
        movable_support=movable_support,
    )

    import gymnasium as gym

    if resolved_embodiment == G1_ARENA_TELEOP_EMBODIMENT:
        spec = gym.spec(composed.task_id)
        kwargs = dict(spec.kwargs)
        gym.registry.pop(composed.task_id)
        gym.register(
            id=composed.task_id,
            entry_point="e2e_robot_workflows.arena.mimic:make_arena_mimic_env",
            kwargs=kwargs,
            disable_env_checker=True,
        )
    return composed


def _subtask(object_ref: str, signal: str | None, *, interpolation_steps: int) -> SubTaskConfig:
    return SubTaskConfig(
        object_ref=object_ref,
        subtask_term_signal=signal,
        first_subtask_start_offset_range=(0, 0),
        subtask_term_offset_range=(0, 0),
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs={"nn_k": 3},
        action_noise=ACTION_NOISE,
        num_interpolation_steps=interpolation_steps,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
    )


@configclass
class E2EStaticAppleMimicEnvCfg(MimicEnvCfg):
    """Mimic segmentation for the static workcell instead of table-to-bin navigation."""

    pick_object_name: str = "pick_object"
    placement_target_name: str = "placement_target"
    manipulation_hand: str = "right"
    embodiment: str = "g1_wbc_agile_pink"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.manipulation_hand not in {"left", "right"}:
            raise ValueError(f"manipulation_hand must be left or right, got {self.manipulation_hand!r}")

        self.datagen_config.name = "e2e_static_apple"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 100
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.use_navigation_controller = False
        if self.embodiment == "g1_wbc_agile_pink":
            from isaaclab_arena_g1.g1_env.mdp.recorders.g1_locomanip_recorder_cfg import (
                G1LocomanipRecorderManagerCfg,
            )

            self.mimic_recorder_config = G1LocomanipRecorderManagerCfg()
        elif self.embodiment == "h2_agile_pink":
            from isaaclab_arena.utils.isaaclab_utils.recorders import ArenaEnvRecorderManagerCfg

            self.mimic_recorder_config = ArenaEnvRecorderManagerCfg()
        else:
            raise ValueError(f"unsupported Arena Pink mimic embodiment {self.embodiment!r}")

        active = self.manipulation_hand
        support = "left" if active == "right" else "right"
        if self.embodiment == "h2_agile_pink":
            self.subtask_configs = {
                active: [
                    _subtask(self.pick_object_name, f"{active}_hand_holds_object", interpolation_steps=0),
                    _subtask(self.pick_object_name, f"{support}_hand_holds_object", interpolation_steps=3),
                    _subtask(self.placement_target_name, None, interpolation_steps=3),
                ],
                support: [
                    _subtask(self.pick_object_name, f"{support}_hand_holds_object", interpolation_steps=0),
                    _subtask(self.placement_target_name, None, interpolation_steps=3),
                ],
                # Isaac Lab Mimic models the AGILE navigation passthrough as a body EEF.
                "body": [_subtask(self.pick_object_name, None, interpolation_steps=0)],
            }
            self.task_constraint_configs = [
                SubTaskConstraintConfig(
                    eef_subtask_constraint_tuple=[(active, 1), (support, 0)],
                    constraint_type=SubTaskConstraintType.COORDINATION,
                    coordination_scheme=SubTaskConstraintCoordinationScheme.TRANSFORM,
                    coordination_scheme_pos_noise_scale=0.0,
                    coordination_scheme_rot_noise_scale=0.0,
                    coordination_synchronize_start=False,
                ),
                SubTaskConstraintConfig(
                    eef_subtask_constraint_tuple=[(active, 2), (support, 1)],
                    constraint_type=SubTaskConstraintType.COORDINATION,
                    coordination_scheme=SubTaskConstraintCoordinationScheme.TRANSFORM,
                    coordination_scheme_pos_noise_scale=0.0,
                    coordination_scheme_rot_noise_scale=0.0,
                    coordination_synchronize_start=True,
                ),
            ]
        else:
            self.subtask_configs = {
                active: [
                    _subtask(self.pick_object_name, f"{active}_hand_holds_object", interpolation_steps=0),
                    _subtask(self.pick_object_name, None, interpolation_steps=3),
                ],
                support: [_subtask(self.pick_object_name, None, interpolation_steps=0)],
                "body": [_subtask(self.pick_object_name, None, interpolation_steps=0)],
            }
            self.task_constraint_configs = []


def make_static_apple_mimic_cfg(
    *,
    pick_object_name: str,
    placement_target_name: str,
    manipulation_hand: str,
    embodiment: str,
):
    """Return the callback consumed by Arena's pick-and-place task."""

    def factory(arm_mode):
        if getattr(arm_mode, "value", None) != "dual_arm":
            raise ValueError(f"E2E Arena Pink mimic requires dual_arm, got {arm_mode!r}")
        return E2EStaticAppleMimicEnvCfg(
            pick_object_name=pick_object_name,
            placement_target_name=placement_target_name,
            manipulation_hand=manipulation_hand,
            embodiment=embodiment,
        )

    return factory


__all__ = [
    "E2EStaticAppleMimicEnvCfg",
    "H2MimicEnv",
    "compose_arena_mimic_environment",
    "make_arena_mimic_env",
    "make_static_apple_mimic_cfg",
]

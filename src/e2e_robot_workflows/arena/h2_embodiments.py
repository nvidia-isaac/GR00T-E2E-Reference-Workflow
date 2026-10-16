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

"""E2E-owned H2plus embodiments registered through Arena's public seam."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
import isaaclab_arena.terms.transforms as transforms_terms
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.pink_ik import (
    DampingTaskCfg,
    LocalFrameTaskCfg,
    NullSpacePostureTaskCfg,
    PinkIKControllerCfg,
)
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
)
from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.terms.events import reset_all_articulation_joints
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.pose import Pose
from isaaclab_teleop import XrCfg
from isaaclab_teleop.xr_cfg import XrAnchorRotationMode

from e2e_robot_workflows.arena.embodiment_capabilities import (
    H2_ARENA_EMBODIMENT as H2_ARENA_JOINT_EMBODIMENT,
    H2_ARENA_TELEOP_EMBODIMENT as H2_ARENA_PINK_EMBODIMENT,
)
from e2e_robot_workflows.arena.h2_events import apply_high_friction_to_h2_fingers, follow_h2_head_camera
from e2e_robot_workflows.controllers.h2_agile_action import H2AgileVelocityActionCfg
from e2e_robot_workflows.controllers.h2_pink_action import H2PinkInverseKinematicsActionCfg
from e2e_robot_workflows.robots.h2 import load_h2_pink_contract, load_h2_robot_config
from e2e_robot_workflows.robots.h2.articulation import make_h2_articulation_cfg


def _asset_cache_root() -> str:
    configured = os.environ.get("E2E_H2_ASSET_CACHE")
    root = Path(configured).expanduser() if configured else Path.home() / ".cache/e2e-reference-app/h2plus"
    return str(root / "usd")


def make_h2_xr_cfg() -> XrCfg:
    """Return the workcell-aligned, pelvis-following H2 XR anchor."""

    return XrCfg(
        anchor_pos=(0.0, 0.0, -1.0),
        anchor_rot=(0.0, 0.0, 0.0, 1.0),
        anchor_prim_path="/World/envs/env_0/Robot/pelvis",
        anchor_rotation_mode=XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED,
        fixed_anchor_height=True,
    )


@configclass
class H2SceneCfg:
    """The public H2plus articulation, filled only when an H2 embodiment is selected."""

    robot: ArticulationCfg = MISSING


@configclass
class H2CameraCfg(ArenaCameraCfg):
    """Head-mounted RGB camera used by recording and evaluation."""

    robot_head_cam: CameraCfg = CameraCfg(
        # The published H2plus USD retains its articulated link hierarchy.
        # Point at the actual head rigid body. CameraCfg's public rigid-body
        # attachment path creates its managed ``camera`` child.
        prim_path=(
            "{ENV_REGEX_NS}/Robot/Geometry/pelvis/waist_yaw_link/waist_roll_link/"
            "torso_link/head_pitch_link/head_yaw_link"
        ),
        # Isaac Lab holds the latest image between updates, so recording still
        # writes one aligned image (occasionally repeated) per action sample.
        update_period=1.0 / load_h2_robot_config().camera_hz,
        # An interval event writes the live head-relative pose before each
        # camera observation; avoid a second FrameView pose refresh here.
        update_latest_camera_pose=False,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.1, 5.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(0.10, 0.0, 0.08),
            # The apple and plate are about 75 degrees below the H2 head at
            # reset. A 65-degree downward pitch and wider 12 mm field of view
            # retain both complete task objects as the free base settles.
            rot=(0.0, 0.53730, 0.0, 0.84339),
            convention="world",
        ),
    )


@configclass
class H2ObservationsCfg:
    """Named H2 state with pelvis-relative flange poses."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=base_mdp.last_action)
        robot_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_joint_vel = ObsTerm(func=base_mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        left_flange_pose_pelvis_frame = ObsTerm(
            func=transforms_terms.transform_pose_from_world_to_target_frame,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_link_name": "left_hand_flange",
                "target_frame_name": "pelvis",
            },
        )
        right_flange_pose_pelvis_frame = ObsTerm(
            func=transforms_terms.transform_pose_from_world_to_target_frame,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_link_name": "right_hand_flange",
                "target_frame_name": "pelvis",
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class H2JointActionCfg:
    """58 named upper-body targets followed by the three AGILE commands."""

    upper_body_joint_pos: ActionTermCfg = MISSING
    lower_body_joint_pos: ActionTermCfg = MISSING


@configclass
class H2PinkActionCfg:
    """58-D pelvis-relative Pink/Sharpa action plus three AGILE commands."""

    upper_body_ik: ActionTermCfg = MISSING
    lower_body_joint_pos: ActionTermCfg = MISSING


@configclass
class H2EventCfg:
    reset_all = EventTerm(func=reset_all_articulation_joints, mode="reset")
    apply_high_friction_to_h2_fingers: EventTerm | None = None
    follow_head_camera: EventTerm | None = None


class H2EmbodimentBase(EmbodimentBase):
    """Shared public-plant binding for the two H2 Arena controllers."""

    default_arm_mode = ArmMode.DUAL_ARM

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None) -> None:
        super().__init__(enable_cameras, initial_pose)
        robot = load_h2_robot_config()
        contract = load_h2_pink_contract()
        self.scene_config = H2SceneCfg()
        self.scene_config.robot = make_h2_articulation_cfg(
            robot,
            contract,
            cache_root=_asset_cache_root(),
        )
        self.camera_config = H2CameraCfg()
        self.observation_config = H2ObservationsCfg()
        self.event_config = H2EventCfg()
        if enable_cameras:
            follow_period = 1.0 / robot.control_hz
            self.event_config.follow_head_camera = EventTerm(
                func=follow_h2_head_camera,
                mode="interval",
                interval_range_s=(follow_period, follow_period),
            )
        from e2e_robot_workflows.arena.mimic import H2MimicEnv

        self.mimic_env = H2MimicEnv
        # FOLLOW_PRIM_SMOOTHED applies subsequent pelvis-yaw changes. Starting
        # at identity keeps the operator facing the workcell instead of 90° left.
        self.xr = make_h2_xr_cfg()

    def get_teleop_target_frame_prim_path(self) -> str:
        return "/World/envs/env_0/Robot/pelvis"

    def set_finger_contact_friction(
        self,
        *,
        material_path: str,
        static_friction: float,
        dynamic_friction: float,
        prim_name_markers: Sequence[str],
    ) -> None:
        self.event_config.apply_high_friction_to_h2_fingers = EventTerm(
            func=apply_high_friction_to_h2_fingers,
            mode="prestartup",
            params={
                "material_path": material_path,
                "static_friction": static_friction,
                "dynamic_friction": dynamic_friction,
                "prim_name_markers": tuple(prim_name_markers),
            },
        )


@register_asset
class H2AgileJointEmbodiment(H2EmbodimentBase):
    """H2 Arena embodiment for learned upper-body targets plus AGILE locomotion."""

    name = H2_ARENA_JOINT_EMBODIMENT

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None, lock_waist: bool = True) -> None:
        del lock_waist
        super().__init__(enable_cameras, initial_pose)
        robot = load_h2_robot_config()
        contract = load_h2_pink_contract()
        self.action_config = H2JointActionCfg()
        self.action_config.upper_body_joint_pos = JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(contract.arm_joint_names + contract.hand_joint_order),
            scale=1.0,
            use_default_offset=False,
            preserve_order=True,
        )
        self.action_config.lower_body_joint_pos = H2AgileVelocityActionCfg(
            asset_name="robot",
            policy_path=robot.resolve_policy(),
            driven_joint_names=robot.driven_joint_names,
            proprio_joint_names=contract.body_joint_order,
            expected_joint_order=contract.expected_joint_order,
            history_length=robot.history_length,
            command_dim=robot.command_dim,
        )


@register_asset
class H2AgilePinkEmbodiment(H2EmbodimentBase):
    """H2 Arena embodiment for PICO, replay, and Mimic generation."""

    name = H2_ARENA_PINK_EMBODIMENT

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None, lock_waist: bool = True) -> None:
        del lock_waist
        super().__init__(enable_cameras, initial_pose)
        robot = load_h2_robot_config()
        contract = load_h2_pink_contract()
        self.action_config = H2PinkActionCfg()
        self.action_config.upper_body_ik = H2PinkInverseKinematicsActionCfg(
            asset_name="robot",
            pink_controlled_joint_names=list(contract.arm_joint_names),
            hand_joint_names=list(contract.hand_joint_order),
            target_eef_link_names={
                "left_flange": contract.flange_frames[0],
                "right_flange": contract.flange_frames[1],
            },
            enable_gravity_compensation=True,
            controller=PinkIKControllerCfg(
                urdf_path=str(contract.resolve_urdf()),
                articulation_name="robot",
                base_link_name=contract.base_frame,
                num_hand_joints=len(contract.hand_joint_order),
                show_ik_warnings=True,
                fail_on_joint_limit_violation=True,
                variable_input_tasks=[
                    LocalFrameTaskCfg(
                        frame=contract.flange_frames[0],
                        base_link_frame_name=contract.base_frame,
                        position_cost=8.0,
                        orientation_cost=4.0,
                        lm_damping=5.0,
                        gain=0.25,
                    ),
                    LocalFrameTaskCfg(
                        frame=contract.flange_frames[1],
                        base_link_frame_name=contract.base_frame,
                        position_cost=8.0,
                        orientation_cost=4.0,
                        lm_damping=5.0,
                        gain=0.25,
                    ),
                    # Prefer reset shoulders/elbows while leaving wrists free to
                    # preserve both flange targets through the full arm null space.
                    NullSpacePostureTaskCfg(
                        controlled_frames=list(contract.flange_frames),
                        controlled_joints=[
                            name for name in contract.arm_joint_names if "_shoulder_" in name or "_elbow_" in name
                        ],
                        cost=0.1,
                        gain=0.1,
                        lm_damping=0.0,
                    ),
                ],
                fixed_input_tasks=[DampingTaskCfg(cost=1e-3)],
            ),
        )
        self.action_config.lower_body_joint_pos = H2AgileVelocityActionCfg(
            asset_name="robot",
            policy_path=robot.resolve_policy(),
            driven_joint_names=robot.driven_joint_names,
            proprio_joint_names=contract.body_joint_order,
            expected_joint_order=contract.expected_joint_order,
            history_length=robot.history_length,
            command_dim=robot.command_dim,
        )


__all__ = [
    "H2_ARENA_JOINT_EMBODIMENT",
    "H2_ARENA_PINK_EMBODIMENT",
    "H2AgileJointEmbodiment",
    "H2AgilePinkEmbodiment",
    "make_h2_xr_cfg",
]

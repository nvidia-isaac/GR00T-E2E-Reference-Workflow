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

"""Isaac Lab-native G1 static apple environment config.

Built on Isaac Lab's G1 locomanipulation stack
(``LocomanipulationG1EnvCfg``): the 29-DOF G1 with the three-finger Unitree
hand (``g1.usd`` / rev_1_0), Pink IK upper-body action (2 wrist poses + 14 hand
joints = 28-D), the AGILE lower-body locomotion action (4-D command
``[vx, vy, wz, hip_height]``), and the IsaacTeleop VR-controller pipeline
(Quest 3 / Pico 4 Ultra via the browser client). This module
layers the static apple task on top: warehouse backdrop, scaled packing table,
apple/plate assets, reset events, timing, and the fixed/agile lower-body mode
switch.

Lower-body mode (``G1StaticAppleEnvCfg.wbc_mode``):
- ``fixed``: root pinned; the AGILE action and its observation group are
  removed, only the Pink IK upper body acts. Stable without a balancer.
- ``agile``: free base balanced/steered by Isaac Lab's AGILE policy; the
  lower-body command is part of the action vector, so recordings capture the
  navigate/base-height commands the dataset contract expects.
"""

from __future__ import annotations

import os
from typing import Literal, cast

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm, ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg
from isaaclab_tasks.manager_based.locomanipulation.pick_place.fixed_base_upper_body_ik_g1_env_cfg import (
    _build_g1_upper_body_pipeline,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (
    LocomanipulationG1EnvCfg,
    LocomanipulationG1SceneCfg,
)
from isaaclab_teleop import IsaacTeleopCfg, XrCfg
from loguru import logger

from e2e_robot_workflows.environments.scene_prims import deactivate_scene_prims
from e2e_robot_workflows.robots.g1.config import load_g1_robot_config
from e2e_robot_workflows.tasks.static_apple.agile_action import InferenceAgileBasedLowerBodyAction
from e2e_robot_workflows.tasks.static_apple.spec import (
    BACKGROUND_NAME,
    BACKGROUND_POS,
    BACKGROUND_PRIMS_TO_DEACTIVATE,
    CONTROL_HZ,
    DEFAULT_WBC_MODE as _TASK_DEFAULT_WBC_MODE,
    DESTINATION_MASS_KG,
    DESTINATION_NAME,
    DESTINATION_SPAWN_XY,
    DESTINATION_SPAWN_XY_RANGE_M,
    ENABLE_WAIST as _TASK_ENABLE_WAIST,
    ENVIRONMENT,
    EPISODE_LENGTH_S,
    PACKING_TABLE_PRIMS_TO_DEACTIVATE,
    PHYSICS_DT_HZ,
    PICK_OBJECT_MASS_KG,
    PICK_OBJECT_NAME,
    PICK_SPAWN_XY,
    PICK_SPAWN_XY_RANGE_M,
    PICK_SPAWN_YAW_RANGE_RAD,
    SHELF_AIRGAP,
    SHELF_SURFACE_Z,
    TUNED_ASSET_SCALES,
    XR_ANCHOR_POS,
    XR_ANCHOR_ROT_XYZW,
    shelf_spawn_z,
)

BACKGROUND_USD = ENVIRONMENT.asset("background").usd_path
SUPPORT_USD = ENVIRONMENT.asset("support").usd_path
PICK_OBJECT_USD = ENVIRONMENT.asset("pick_object").usd_path
DESTINATION_USD = ENVIRONMENT.asset("destination").usd_path

_G1_ROBOT = load_g1_robot_config()
# Spawn XY from the robot config, Z from the standing pelvis height (feet on floor).
G1_LAB_INITIAL_BASE_POSITION_XYZ = (
    _G1_ROBOT.initial_position_xy[0],
    _G1_ROBOT.initial_position_xy[1],
    _G1_ROBOT.pelvis_height_m,
)
type WbcMode = Literal["fixed", "agile"]
WBC_MODES: tuple[WbcMode, ...] = ("fixed", "agile")
WAIST_JOINT_PATTERN = "waist_.*_joint"


def _validated_wbc_mode(mode: str) -> WbcMode:
    """Validate and normalize a lower-body control mode."""

    normalized_mode = mode.lower()
    if normalized_mode not in WBC_MODES:
        supported = ", ".join(WBC_MODES)
        raise ValueError(f"Unsupported WBC mode {mode!r}; expected one of: {supported}")
    return cast(WbcMode, normalized_mode)


DEFAULT_WBC_MODE: WbcMode = _validated_wbc_mode(_TASK_DEFAULT_WBC_MODE)


def _enable_waist_ik() -> bool:
    """Whether the Pink IK drives the waist: E2E_G1_ENABLE_WAIST env var overrides
    the G1 recording default."""

    override = os.environ.get("E2E_G1_ENABLE_WAIST")
    if override is not None:
        return override.lower() in ("1", "true", "yes")
    return _TASK_ENABLE_WAIST


def _shallowest_rigid_body_suffix(usd_path: str) -> str:
    """Path of the shallowest rigid body relative to the asset's default prim.

    The spawn prim (``pick_object`` / ``placement_target``) is the asset root, but some
    USDs nest the rigid body below it. The contact sensor must target the body, not the
    root, so it initializes on nested-body assets too. Empty when the default prim itself
    is the rigid body.
    """

    from isaaclab.utils.assets import retrieve_file_path  # local: pxr/AR resolver needs SimulationApp
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(retrieve_file_path(usd_path))
    default_path = stage.GetDefaultPrim().GetPath()
    bodies = [prim.GetPath() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not bodies:
        raise RuntimeError(f"No rigid body in {usd_path!r}; the apple/plate contact sensor needs one.")
    shallowest = min(bodies, key=lambda path: path.pathElementCount)
    relative = shallowest.MakeRelativePath(default_path).pathString
    return "" if relative in (".", "") else f"/{relative}"


def make_pick_object_spawn_cfg():
    """Build the declared apple spawn config."""

    return UsdFileCfg(
        usd_path=PICK_OBJECT_USD,
        scale=TUNED_ASSET_SCALES[PICK_OBJECT_NAME],
        rigid_props=PhysxRigidBodyPropertiesCfg(),
        mass_props=MassPropertiesCfg(mass=PICK_OBJECT_MASS_KG),
        activate_contact_sensors=True,  # required by object_contact_sensor
    )


def make_destination_spawn_cfg():
    """Build the declared dynamic plate spawn config (falls and rests like a real plate)."""

    return UsdFileCfg(
        usd_path=DESTINATION_USD,
        scale=TUNED_ASSET_SCALES[DESTINATION_NAME],
        rigid_props=PhysxRigidBodyPropertiesCfg(),
        mass_props=MassPropertiesCfg(mass=DESTINATION_MASS_KG),
        activate_contact_sensors=True,  # required by object_contact_sensor
    )


@configclass
class StaticAppleSceneCfg(LocomanipulationG1SceneCfg):
    """Static apple scene on Isaac Lab's G1 locomanipulation scene.

    Inherits the trihand G1 robot and PackingTable and adds the warehouse
    backdrop and apple/plate manipulation objects. The backdrop supplies the
    floor, so the inherited ground plane is removed.
    """

    # Warehouse backdrop (floor + walls + shelves). Referenced with its baked
    # colliders/lighting; no rigid body wrapper needed for a static environment.
    background = AssetBaseCfg(
        prim_path=ENVIRONMENT.asset("background").prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=BACKGROUND_POS,
            rot=ENVIRONMENT.asset("background").rotation_xyzw,
        ),
        spawn=UsdFileCfg(
            usd_path=BACKGROUND_USD,
            scale=ENVIRONMENT.asset("background").scale,
        ),
    )

    ground = None

    object = RigidObjectCfg(
        prim_path=ENVIRONMENT.asset("pick_object").prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(PICK_SPAWN_XY[0], PICK_SPAWN_XY[1], shelf_spawn_z(PICK_OBJECT_NAME)),
            rot=ENVIRONMENT.asset("pick_object").rotation_xyzw,
        ),
        spawn=make_pick_object_spawn_cfg(),
    )

    # Dynamic placement target: a real, movable plate. Spawned just above the table
    # it falls and rests under gravity (like the apple), so it needs no hand-tuned
    # resting height. It is *non-kinematic* -- a kinematic RigidObject would abort
    # in PhysX when `reset_all` writes zero root velocity to it
    # ("PxRigidDynamic::setLinearVelocity: Body must be non-kinematic!"); a dynamic
    # body accepts the velocity reset, so reset_scene_to_default returns it cleanly.
    destination = RigidObjectCfg(
        prim_path=ENVIRONMENT.asset("destination").prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(DESTINATION_SPAWN_XY[0], DESTINATION_SPAWN_XY[1], shelf_spawn_z(DESTINATION_NAME)),
            rot=ENVIRONMENT.asset("destination").rotation_xyzw,
        ),
        spawn=make_destination_spawn_cfg(),
    )

    # Apple<->plate contact sensor; success detection reads its force and combines it
    # with the placement footprint and settle speed to match the evaluation criteria.
    # See tasks/static_apple/success.py.
    object_contact_sensor = ContactSensorCfg(
        prim_path=ENVIRONMENT.asset("pick_object").prim_path,
        filter_prim_paths_expr=[ENVIRONMENT.asset("destination").prim_path],
    )


@configclass
class StaticAppleEventCfg:
    """Static apple reset/startup events (the Isaac Lab base defines none)."""

    # Writes every rigid object's declared init_state (pose + zero velocity) back to
    # the sim each episode. That covers the plate: it is a RigidObject, so it always
    # re-drops from its spawn pose instead of starting wherever the previous episode
    # left it. The Arena path gets the same guarantee from its own per-object reset
    # event (arena/external_environment.py).
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

    # Randomize the apple's spawn XY (and yaw) each episode so demonstrations are not
    # all from the identical pose. Runs after reset_all, so it overrides the default.
    reset_object = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-PICK_SPAWN_XY_RANGE_M, PICK_SPAWN_XY_RANGE_M),
                "y": (-PICK_SPAWN_XY_RANGE_M, PICK_SPAWN_XY_RANGE_M),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-PICK_SPAWN_YAW_RANGE_RAD, PICK_SPAWN_YAW_RANGE_RAD),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    # Randomize the plate's spawn XY each episode. Kept small; success detection reads
    # the live plate pose, so a shifted plate stays a valid target.
    reset_destination = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-DESTINATION_SPAWN_XY_RANGE_M, DESTINATION_SPAWN_XY_RANGE_M),
                "y": (-DESTINATION_SPAWN_XY_RANGE_M, DESTINATION_SPAWN_XY_RANGE_M),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("destination"),
        },
    )

    deactivate_static_apple_background_prims = (
        EventTerm(
            func=deactivate_scene_prims,
            mode="prestartup",
            params={
                "asset_root_name": ENVIRONMENT.asset("background").name,
                "prim_relative_paths": BACKGROUND_PRIMS_TO_DEACTIVATE,
            },
        )
        if BACKGROUND_PRIMS_TO_DEACTIVATE
        else None
    )

    # Remove the empty basket/tray baked into the PackingTable USD so the apple
    # sits directly on the table surface.
    deactivate_packing_table_props = EventTerm(
        func=deactivate_scene_prims,
        mode="prestartup",
        params={
            "asset_root_name": ENVIRONMENT.asset("support").name,
            "prim_relative_paths": PACKING_TABLE_PRIMS_TO_DEACTIVATE,
        },
    )

    # Finger friction (from g1.yaml) so the apple stays in the grasp. One alternation
    # regex matches each finger body once; num_buckets=1 pins a single material.
    set_finger_friction = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*(" + "|".join(_G1_ROBOT.finger_friction["prim_name_markers"]) + ").*"],
            ),
            "static_friction_range": (
                float(_G1_ROBOT.finger_friction["static_friction"]),
                float(_G1_ROBOT.finger_friction["static_friction"]),
            ),
            "dynamic_friction_range": (
                float(_G1_ROBOT.finger_friction["dynamic_friction"]),
                float(_G1_ROBOT.finger_friction["dynamic_friction"]),
            ),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )


@configclass
class G1StaticAppleEnvCfg(LocomanipulationG1EnvCfg):
    """Lab-native G1 static apple task config (trihand G1, Pink IK + AGILE)."""

    wbc_mode: WbcMode = DEFAULT_WBC_MODE
    enable_waist: bool = _enable_waist_ik()
    scene: StaticAppleSceneCfg = StaticAppleSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    events: StaticAppleEventCfg = StaticAppleEventCfg()
    e2e_support_surface_z_m: float = SHELF_SURFACE_Z
    e2e_release_clearance_m: float = ENVIRONMENT.release_clearance_m
    e2e_task_link_names: tuple[str, str] = _G1_ROBOT.task_link_names

    def __post_init__(self):
        self.wbc_mode = _validated_wbc_mode(self.wbc_mode)

        # Base wires the Pink IK URDF (pre-baked kinematics asset fetched from
        # Nucleus -- no USD->URDF conversion), tri-hand XR retargeters, and the
        # pelvis-following XR anchor.
        super().__post_init__()

        # Shared task timing.
        self.episode_length_s = EPISODE_LENGTH_S
        self.sim.dt = 1.0 / PHYSICS_DT_HZ
        self.decimation = max(1, PHYSICS_DT_HZ // CONTROL_HZ)
        self.sim.render_interval = self.decimation
        self.num_rerenders_on_reset = 1

        # The base sinks the packing table 0.3 m into the ground to lower its
        # top; realize the E2E-owned support definition instead.
        self.scene.packing_table.prim_path = ENVIRONMENT.asset("support").prim_path
        self.scene.packing_table.init_state.pos = ENVIRONMENT.support_position
        self.scene.packing_table.init_state.rot = ENVIRONMENT.asset("support").rotation_xyzw
        self.scene.packing_table.spawn.usd_path = SUPPORT_USD
        self.scene.packing_table.spawn.scale = ENVIRONMENT.asset("support").scale

        # Target the resolved rigid body (some USDs nest it below the spawn prim).
        pick_body = _shallowest_rigid_body_suffix(PICK_OBJECT_USD)
        plate_body = _shallowest_rigid_body_suffix(DESTINATION_USD)
        self.scene.object_contact_sensor.prim_path = ENVIRONMENT.asset("pick_object").prim_path + pick_body
        self.scene.object_contact_sensor.filter_prim_paths_expr = [
            ENVIRONMENT.asset("destination").prim_path + plate_body
        ]

        # Robot-USD override from the robot config; an unreachable USD fails at spawn
        # rather than silently substituting a different robot.
        if _G1_ROBOT.usd:
            self.scene.robot.spawn.usd_path = _G1_ROBOT.usd.format(
                ISAAC_NUCLEUS_DIR=ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR=ISAACLAB_NUCLEUS_DIR
            )

        # Stand the G1 on the floor at the configured spot, facing the table.
        # The base robot cfg already carries the natural bent-knee standing pose.
        self.scene.robot.init_state.pos = G1_LAB_INITIAL_BASE_POSITION_XYZ
        self.scene.robot.init_state.rot = _G1_ROBOT.initial_rotation_xyzw
        for pattern in _G1_ROBOT.base_leg_joint_patterns:
            self.scene.robot.init_state.joint_pos.pop(pattern, None)
        self.scene.robot.init_state.joint_pos.update(_G1_ROBOT.standing_leg_joint_pos)
        # Open-arm ready pose at spawn (both modes; AGILE only drives the legs).
        self.scene.robot.init_state.joint_pos.update(_G1_ROBOT.open_arm_joint_pos)

        # Waist: Isaac Lab's Pink IK controls the waist by default. Off by
        # default here (torso held) -- steadier for static manipulation; enable
        # via the robot config / E2E_G1_ENABLE_WAIST for extended reach.
        if not self.enable_waist:
            self._remove_waist_from_ik()

        if self.wbc_mode == "agile":
            self._enable_agile_wbc()
        else:
            self._enable_fixed_base()

        self._maybe_add_head_camera()

        # Recording is operator-accepted; no automatic success/distance termination.
        self.terminations.success = None
        self.terminations.object_too_far = None

    def _maybe_add_head_camera(self) -> None:
        """Attach the head RGB camera + its observation when cameras are enabled.

        Gated on the ``/isaaclab/cameras_enabled`` carb setting (set by
        ``--enable_cameras``) so camera-less runs (headless validation runs, plain
        viewers) build without the sensor. The observation term name matches
        the dataset schema's video field (``robot_head_cam_rgb``).
        """

        import carb

        cam = _G1_ROBOT.head_camera
        if not cam:
            logger.warning("no head_camera section in the robot config; recordings will have no video observation.")
            return
        if not carb.settings.get_settings().get("/isaaclab/cameras_enabled"):
            logger.info(
                "[env_cfg] cameras disabled: head camera not added (run with --enable_cameras to record video)."
            )
            return

        self.scene.robot_head_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + str(cam["prim"]),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(cam["offset_pos"]),
                rot=tuple(cam["offset_rot_xyzw"]),
                convention="ros",
            ),
            width=int(cam["width"]),
            height=int(cam["height"]),
            data_types=["rgb"],
            update_period=0.0,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=float(cam["focal_length"]),
                clipping_range=tuple(cam["clipping_range"]),
            ),
        )
        # Raw uint8 frames (normalize=False): the recorder stores what NVIDIA
        # Isaac GR00T trains on.
        self.observations.policy.robot_head_cam_rgb = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("robot_head_cam"), "data_type": "rgb", "normalize": False},
        )

    def _remove_waist_from_ik(self) -> None:
        """Hold the waist: drop it from the IK-controlled joints and posture task."""

        controlled = self.actions.upper_body_ik.pink_controlled_joint_names
        if WAIST_JOINT_PATTERN in controlled:
            controlled.remove(WAIST_JOINT_PATTERN)
        for task in self.actions.upper_body_ik.controller.variable_input_tasks:
            joints = getattr(task, "controlled_joints", None)
            if joints:
                task.controlled_joints = [name for name in joints if not name.startswith("waist_")]

    def _enable_fixed_base(self) -> None:
        """Pin the root: manipulation-only, no balancer needed.

        Removes the AGILE lower-body action and its observation group (the
        managers skip ``None`` entries) and drops the lower-body command
        retargeter from IsaacTeleop so the teleop action matches the 28-D Pink
        IK action.
        """

        self.scene.robot.spawn.articulation_props.fix_root_link = True
        self.actions.lower_body_joint_pos = None
        self.observations.lower_body_policy = None

        # Operator viewpoint from the task config (teleop.xr_anchor_*).
        self.xr = XrCfg(
            anchor_pos=XR_ANCHOR_POS,
            anchor_rot=XR_ANCHOR_ROT_XYZW,
        )
        # The upper-body builder returns (pipeline, retargeters-to-tune); the
        # session lifecycle wants just the pipeline (same wrapping as the
        # Isaac Lab fixed-base env).
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: _build_g1_upper_body_pipeline()[0],
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )

    def _enable_agile_wbc(self) -> None:
        """Free base balanced by Isaac Lab's AGILE locomotion policy.

        The base config already provides everything: the free-base robot, the
        AGILE action term (4-D ``[vx, vy, wz, hip_height]`` command), its
        observation group, and teleop retargeters that emit the lower-body
        command -- so recordings capture the navigate/base-height commands.
        """

        self.scene.robot.spawn.articulation_props.fix_root_link = False
        # Beta 2's Isaac Lab action runs TorchScript with gradients enabled;
        # keep its configuration and behavior while fixing the inference
        # boundary before the output is handed to Warp.
        self.actions.lower_body_joint_pos.class_type = InferenceAgileBasedLowerBodyAction


def make_env_cfg(
    *,
    wbc_mode: WbcMode = DEFAULT_WBC_MODE,
    device: str = "cuda:0",
    num_envs: int = 1,
) -> G1StaticAppleEnvCfg:
    """Build an explicitly configured Isaac Lab environment."""

    if num_envs < 1:
        raise ValueError(f"num_envs must be at least 1, got {num_envs}")

    cfg = G1StaticAppleEnvCfg(wbc_mode=wbc_mode)
    cfg.sim.device = device
    cfg.scene.num_envs = num_envs
    if getattr(cfg, "isaac_teleop", None) is not None:
        cfg.isaac_teleop.sim_device = device
    return cfg


__all__ = [
    "BACKGROUND_NAME",
    "BACKGROUND_USD",
    "DEFAULT_WBC_MODE",
    "DESTINATION_NAME",
    "DESTINATION_USD",
    "G1StaticAppleEnvCfg",
    "PICK_OBJECT_NAME",
    "PICK_OBJECT_USD",
    "SHELF_AIRGAP",
    "SHELF_SURFACE_Z",
    "SUPPORT_USD",
    "StaticAppleSceneCfg",
    "WBC_MODES",
    "WbcMode",
    "make_destination_spawn_cfg",
    "make_env_cfg",
    "make_pick_object_spawn_cfg",
]

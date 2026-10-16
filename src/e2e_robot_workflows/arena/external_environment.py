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

"""Thin Arena realization of the E2E-owned static apple environment.

Each supported robot has separate joint and Pink controller embodiments. Evaluation
uses the joint layout; OpenXR recording and AutoData use the Pink layout.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_teleop import XrAnchorRotationMode

from e2e_robot_workflows.arena.embodiment_capabilities import (
    G1_ARENA_EMBODIMENT,
    G1_ARENA_TELEOP_EMBODIMENT,
    H2_ARENA_EMBODIMENT,
    H2_ARENA_TELEOP_EMBODIMENT,
    capability_for_embodiment,
)
from e2e_robot_workflows.tasks.static_apple.config import load_static_apple_environment
from e2e_robot_workflows.tasks.static_apple.table_height import resolve_randomization


@dataclass
class E2EArenaExternalEnvironmentCfg(ArenaEnvironmentCfg):
    """Arena-only choices for the static apple realization."""

    enable_cameras: bool = False
    embodiment: str = G1_ARENA_EMBODIMENT
    lock_waist: bool = True
    teleop_device: str | None = None
    disable_task_terminations: bool = False
    """Run fixed-length episodes that end only on time-out, without task metrics.

    Runs that need a full rollout regardless of task outcome set this. It also suppresses
    ``object_dropped``, which would otherwise end an episode the moment the apple left the
    hand. Arena's success metric is dropped with it: it reads the termination being
    removed.
    """
    mimic: bool = False
    manipulation_hand: str = "right"
    episode_length_s: float | None = None
    """Optional episode time budget; task success and failure predicates are unchanged."""
    table_height_offset_range_m: list[float] | None = None
    """Optional metre bounds; a list because Arena derives CLI flags from field types."""
    movable_support: bool = False
    """Recordable kinematic table, independent of sampling so replay never redraws height."""


def _resolve_episode_length_s(override: float | None, default: float) -> float:
    value = default if override is None else override
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise ValueError("episode_length_s must be positive and finite")
    return float(value)


def _deactivate_scene_prims(
    env: Any,
    env_ids: Any,
    asset_root_name: str,
    prim_relative_paths: tuple[str, ...],
) -> None:
    """Deactivate selected asset-relative prims in every cloned scene."""

    del env_ids
    stage = env.sim.stage
    for env_prim_path in env.scene.env_prim_paths:
        for relative_path in prim_relative_paths:
            prim_path = f"{env_prim_path}/{asset_root_name}/{relative_path}"
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                stage.OverridePrim(prim_path).SetActive(False)
            else:
                warnings.warn(f"Scene prim to deactivate was not found: {prim_path}", stacklevel=1)


class E2EArenaExternalEnvironment(ArenaEnvironmentFactory[E2EArenaExternalEnvironmentCfg]):
    """Construct Arena types directly from ``STATIC_APPLE_ENVIRONMENT``."""

    name = "e2e_static_apple"
    _legacy_argparse_cfg_type = E2EArenaExternalEnvironmentCfg

    def build(self, cfg: E2EArenaExternalEnvironmentCfg):
        """Build the static apple scene, robot binding, and pick-and-place task."""

        from isaaclab import sim as sim_utils
        from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
        from isaaclab_arena.assets.background import Background
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose, PoseRange

        if cfg.embodiment in (H2_ARENA_EMBODIMENT, H2_ARENA_TELEOP_EMBODIMENT):
            # Policy runners enter this factory without compose(), so register the
            # consumer-owned H2 assets here, after Kit has started.
            import e2e_robot_workflows.arena.h2_embodiments  # noqa: F401

        capability = capability_for_embodiment(cfg.embodiment)
        definition = load_static_apple_environment(capability.environment_variant)
        episode_length_s = _resolve_episode_length_s(cfg.episode_length_s, definition.episode_length_s)
        capability.validate(embodiment=cfg.embodiment, teleop_device=cfg.teleop_device, mimic=cfg.mimic)
        robot = capability.load_binding()
        if cfg.manipulation_hand not in {"left", "right"}:
            raise ValueError(f"manipulation_hand must be left or right, got {cfg.manipulation_hand!r}")
        background_def = definition.asset("background")
        support_def = definition.asset("support")
        pick_def = definition.asset("pick_object")
        destination_def = definition.asset("destination")
        support_surface_z = definition.support_surface_z
        table_height_randomization = resolve_randomization(
            definition=definition, offset_range_m=cfg.table_height_offset_range_m
        )
        movable_support = cfg.movable_support or table_height_randomization is not None

        background = Background(
            name=background_def.name,
            prim_path=background_def.prim_path,
            usd_path=background_def.usd_path,
            scale=background_def.scale,
            object_min_z=-0.25,
            initial_pose=Pose(
                position_xyz=definition.background_position,
                rotation_xyzw=background_def.rotation_xyzw,
            ),
            tags=["e2e_environment", "background"],
        )
        support = Object(
            name=support_def.name,
            prim_path=support_def.prim_path,
            object_type=ObjectType.RIGID if movable_support else ObjectType.BASE,
            initial_pose=Pose(
                position_xyz=definition.support_position,
                rotation_xyzw=support_def.rotation_xyzw,
            ),
            spawner_cfg=sim_utils.UsdFileCfg(
                usd_path=support_def.usd_path,
                scale=support_def.scale,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=support_def.kinematic),
            ),
            tags=["e2e_environment", "support"],
        )
        if pick_def.mass_kg is None:
            raise ValueError("The static apple environment must declare a pick-object mass")
        pick_center = definition.spawn_position("pick_object")
        pick_half_range = definition.pick_spawn_xy_range_m
        pick_yaw_range = definition.pick_spawn_yaw_range_rad
        pick_object = Object(
            name=pick_def.name,
            prim_path=pick_def.prim_path,
            object_type=ObjectType.RIGID,
            usd_path=pick_def.usd_path,
            scale=pick_def.scale,
            initial_pose=PoseRange(
                position_xyz_min=(
                    pick_center[0] - pick_half_range,
                    pick_center[1] - pick_half_range,
                    pick_center[2],
                ),
                position_xyz_max=(
                    pick_center[0] + pick_half_range,
                    pick_center[1] + pick_half_range,
                    pick_center[2],
                ),
                rpy_min=(0.0, 0.0, -pick_yaw_range),
                rpy_max=(0.0, 0.0, pick_yaw_range),
            ),
            spawn_cfg_addon={
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
                "mass_props": sim_utils.MassPropertiesCfg(mass=pick_def.mass_kg),
            },
            tags=["e2e_environment", "pick_object"],
        )
        if destination_def.mass_kg is None:
            raise ValueError("The static apple environment must declare a plate (destination) mass")
        destination_center = definition.spawn_position("destination")
        destination_half_range = definition.destination_spawn_xy_range_m
        destination = Object(
            name=destination_def.name,
            prim_path=destination_def.prim_path,
            object_type=ObjectType.RIGID,
            usd_path=destination_def.usd_path,
            scale=destination_def.scale,
            initial_pose=PoseRange(
                position_xyz_min=(
                    destination_center[0] - destination_half_range,
                    destination_center[1] - destination_half_range,
                    destination_center[2],
                ),
                position_xyz_max=(
                    destination_center[0] + destination_half_range,
                    destination_center[1] + destination_half_range,
                    destination_center[2],
                ),
                rpy_min=(0.0, 0.0, 0.0),
                rpy_max=(0.0, 0.0, 0.0),
            ),
            spawn_cfg_addon={
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=destination_def.kinematic),
                "mass_props": sim_utils.MassPropertiesCfg(mass=destination_def.mass_kg),
            },
            tags=["e2e_environment", "destination"],
        )
        # The plate is a dynamic body the robot can shove, so it is reset per episode
        # rather than starting where the previous one left it. Its per-episode XY
        # scatter (destination_spawn_xy_range_m) mirrors the `reset_destination` event
        # term on the Isaac Lab path (tasks/static_apple/env_cfg.py); the two runtimes
        # must agree on the plate's per-episode starting distribution.

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
            lock_waist=cfg.lock_waist,
        )
        # Spawn the E2E-configured robot instead of Arena's default; an unreachable USD
        # fails at spawn rather than silently substituting a different robot.
        if robot.usd_override:
            embodiment.scene_config.robot.spawn.usd_path = robot.usd_override.format(
                ISAAC_NUCLEUS_DIR=ISAAC_NUCLEUS_DIR
            )
        if robot.fixed_xr_anchor_rotation:
            # G1 requires the static recording viewpoint; H2 keeps its pelvis-following frame.
            embodiment.xr.anchor_rot = (0.0, 0.0, 0.0, 1.0)
            embodiment.xr.anchor_rotation_mode = XrAnchorRotationMode.FIXED
        teleop_device = None
        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(*robot.initial_position_xy, robot.pelvis_height_m),
                rotation_xyzw=robot.initial_rotation_xyzw,
            )
        )
        embodiment.set_joint_initial_pos(robot.joint_initial_pos)
        embodiment.set_finger_contact_friction(
            material_path=str(robot.finger_friction["material_path"]),
            static_friction=float(robot.finger_friction["static_friction"]),
            dynamic_friction=float(robot.finger_friction["dynamic_friction"]),
            prim_name_markers=tuple(robot.finger_friction["prim_name_markers"]),
        )

        def env_cfg_callback(env_cfg):
            env_cfg.e2e_support_surface_z_m = support_surface_z
            env_cfg.e2e_release_clearance_m = definition.release_clearance_m
            env_cfg.e2e_task_link_names = robot.task_link_names
            if movable_support:
                env_cfg.e2e_support_asset_name = support_def.name
                env_cfg.e2e_support_root_z_m = definition.support_position[2]
                # Only height resets and recorded-pose restores move the kinematic table.
                # Arena's ordinary rigid reset also writes velocity, which PhysX forbids here.
                setattr(env_cfg.events, support_def.name, None)
            if table_height_randomization is not None:
                from isaaclab_arena.utils.configclass import combine_configclass_instances, make_configclass

                from e2e_robot_workflows.tasks.static_apple.table_height import randomize_support_height

                env_cfg.e2e_table_height_randomization = table_height_randomization
                # After the ordinary resets, before observations and the recorder's initial state.
                height_reset_cfg = make_configclass(
                    "TableHeightResetCfg",
                    [
                        (
                            "randomize_support_height",
                            EventTermCfg,
                            EventTermCfg(
                                func=randomize_support_height,
                                mode="reset",
                                params={"randomization": table_height_randomization},
                            ),
                        )
                    ],
                )()
                env_cfg.events = combine_configclass_instances("EventsCfg", env_cfg.events, height_reset_cfg)
            if cfg.mimic:
                from e2e_robot_workflows.tasks.static_apple.autodata_env_cfg import SubtaskTermsCfg

                env_cfg.observations.subtask_terms = SubtaskTermsCfg()
                for term, task_link_name in zip(
                    (
                        env_cfg.observations.subtask_terms.left_hand_holds_object,
                        env_cfg.observations.subtask_terms.right_hand_holds_object,
                    ),
                    robot.task_link_names,
                    strict=True,
                ):
                    term.params["object_cfg"] = SceneEntityCfg(pick_def.name)
                    term.params["task_link_name"] = task_link_name
                    term.params["grasp_distance_m"] = definition.release_clearance_m
                    term.params["min_object_height_m"] = pick_center[2] + definition.pickup_lift_clearance_m
                if cfg.embodiment == H2_ARENA_TELEOP_EMBODIMENT:
                    from e2e_robot_workflows.tasks.static_apple.h2_grasp import (
                        H2_GRASP_CRITERIA,
                        h2_hand_contact_filter_paths,
                        object_held_by_h2_hand,
                        reset_h2_grasp_history,
                    )

                    filters = h2_hand_contact_filter_paths(
                        env_cfg.scene.robot.spawn.usd_path, env_cfg.scene.robot.prim_path
                    )
                    # Keep the apple-to-plate sensor untouched: placement scoring must
                    # never mistake a grasp contact for support by the destination.
                    plate_contact_cfg = getattr(env_cfg.scene, f"contact_sensor_{pick_def.name}")
                    for hand in ("left", "right"):
                        sensor_name = f"contact_sensor_{pick_def.name}_{hand}_hand"
                        sensor_cfg = plate_contact_cfg.replace(
                            filter_prim_paths_expr=list(filters[hand]),
                            history_length=H2_GRASP_CRITERIA.contact_history_samples,
                            update_period=1.0 / robot.control_hz,
                        )
                        setattr(env_cfg.scene, sensor_name, sensor_cfg)
                        term = getattr(env_cfg.observations.subtask_terms, f"{hand}_hand_holds_object")
                        term.func = object_held_by_h2_hand
                        term.params["hand"] = hand
                        term.params["contact_sensor_cfg"] = SceneEntityCfg(sensor_name)
                    from isaaclab_arena.utils.configclass import combine_configclass_instances, make_configclass

                    grasp_reset_cfg = make_configclass(
                        "H2GraspResetCfg",
                        [
                            (
                                "clear_hand_contact_history",
                                EventTermCfg,
                                EventTermCfg(
                                    func=reset_h2_grasp_history,
                                    mode="reset",
                                    params={
                                        "sensor_names": tuple(
                                            f"contact_sensor_{pick_def.name}_{hand}_hand" for hand in ("left", "right")
                                        )
                                    },
                                ),
                            )
                        ],
                    )()
                    env_cfg.events = combine_configclass_instances("EventsCfg", env_cfg.events, grasp_reset_cfg)
            if definition.deactivate_background_prims:
                env_cfg.events.deactivate_e2e_background_prims = EventTermCfg(
                    func=_deactivate_scene_prims,
                    mode="prestartup",
                    params={
                        "asset_root_name": background_def.name,
                        "prim_relative_paths": definition.deactivate_background_prims,
                    },
                )
            if definition.deactivate_support_prims:
                env_cfg.events.deactivate_e2e_support_prims = EventTermCfg(
                    func=_deactivate_scene_prims,
                    mode="prestartup",
                    params={
                        "asset_root_name": support_def.name,
                        "prim_relative_paths": definition.deactivate_support_prims,
                    },
                )
            if cfg.disable_task_terminations:
                env_cfg.terminations.success = None
                env_cfg.terminations.object_dropped = None
                # Arena's success recorder asserts the ``success`` termination exists, and
                # its metrics are computed from the recorded terms, so the outcome
                # machinery has to be dropped as a whole rather than piecemeal.
                env_cfg.recorders.success_rate = None
                env_cfg.metrics = None
            else:
                from e2e_robot_workflows.tasks.static_apple.terminations import (
                    object_placed_on_target,
                )

                env_cfg.terminations.success = TerminationTermCfg(
                    func=object_placed_on_target,
                    params={
                        "placement_radius_m": definition.success_placement_xy_m,
                        "placement_z_m": definition.success_placement_z_m,
                        "min_object_height_above_target_m": definition.min_object_height_above_target_m,
                        "max_object_height_m": support_surface_z + definition.max_object_height_above_support_m,
                        "min_object_height_m": support_surface_z - definition.min_supported_height_below_support_m,
                        "min_target_height_m": support_surface_z - definition.min_supported_height_below_support_m,
                        "min_contact_force": definition.success_force_threshold,
                        "release_clearance_m": definition.release_clearance_m,
                        "max_object_speed": definition.success_velocity_threshold,
                        "task_link_names": robot.task_link_names,
                        "object_cfg": SceneEntityCfg(pick_def.name),
                        "target_cfg": SceneEntityCfg(destination_def.name),
                        "contact_sensor_cfg": SceneEntityCfg(f"contact_sensor_{pick_def.name}"),
                    },
                )
            if definition.physics_hz % robot.control_hz or robot.control_hz % robot.render_hz:
                raise ValueError(
                    "Arena physics, control, and render rates must divide exactly; "
                    f"got {definition.physics_hz}, {robot.control_hz}, and {robot.render_hz} Hz"
                )
            env_cfg.sim.dt = 1.0 / definition.physics_hz
            env_cfg.decimation = definition.physics_hz // robot.control_hz
            # Camera sensors own their lower capture cadence through
            # ``update_period``; rendering may run below controller cadence.
            env_cfg.sim.render_interval = env_cfg.decimation * (robot.control_hz // robot.render_hz)
            env_cfg.episode_length_s = episode_length_s
            env_cfg.num_rerenders_on_reset = 1
            if cfg.enable_cameras and cfg.embodiment in {H2_ARENA_EMBODIMENT, H2_ARENA_TELEOP_EMBODIMENT}:
                from isaaclab_arena.utils.configclass import combine_configclass_instances, make_configclass

                from e2e_robot_workflows.arena.h2_events import reset_h2_head_camera

                # The callback runs after Arena has composed the embodiment and
                # task reset terms. Appending here makes this the final reset
                # pose write, before ManagerBasedEnv renders reset observations.
                camera_reset_cfg = make_configclass(
                    "H2CameraResetCfg",
                    [
                        (
                            "follow_head_camera_after_reset",
                            EventTermCfg,
                            EventTermCfg(func=reset_h2_head_camera, mode="reset"),
                        )
                    ],
                )()
                env_cfg.events = combine_configclass_instances("EventsCfg", env_cfg.events, camera_reset_cfg)
            return env_cfg

        mimic_env_cfg_factory = None
        if cfg.mimic:
            from e2e_robot_workflows.arena.mimic import make_static_apple_mimic_cfg

            mimic_env_cfg_factory = make_static_apple_mimic_cfg(
                pick_object_name=pick_def.name,
                placement_target_name=destination_def.name,
                manipulation_hand=cfg.manipulation_hand,
                embodiment=cfg.embodiment,
            )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[background, support, pick_object, destination]),
            task=PickAndPlaceTask(
                pick_up_object=pick_object,
                destination_location=destination,
                background_scene=background,
                episode_length_s=episode_length_s,
                task_description=definition.language_instruction,
                force_threshold=definition.success_force_threshold,
                velocity_threshold=definition.success_velocity_threshold,
                # Per-axis predicate, so pass the inscribed box half-width: a full-radius
                # box reaches r*sqrt(2) at its corners and re-admits diagonal side touches.
                # Inscribing instead clips the circular gate along the axes, which is
                # accepted here: the apple settles toward the plate's centre, so a placement
                # does not come to rest in that outer band.
                max_separation=(
                    definition.success_placement_box_half_width_m,
                    definition.success_placement_box_half_width_m,
                    definition.success_placement_z_m,
                ),
                mimic_env_cfg_factory=mimic_env_cfg_factory,
            ),
            teleop_device=teleop_device,
            env_cfg_callback=env_cfg_callback,
        )


__all__ = [
    "G1_ARENA_EMBODIMENT",
    "G1_ARENA_TELEOP_EMBODIMENT",
    "H2_ARENA_EMBODIMENT",
    "H2_ARENA_TELEOP_EMBODIMENT",
    "E2EArenaExternalEnvironment",
    "E2EArenaExternalEnvironmentCfg",
]

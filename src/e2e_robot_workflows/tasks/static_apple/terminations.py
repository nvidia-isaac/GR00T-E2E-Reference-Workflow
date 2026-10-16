"""Success condition for the static apple transfer task.

Isaac Lab's ``task_done_pick_place`` cannot express this scene: it decides both the placement
box and the "arm pulled back" test along +x, but this G1 spawns rotated a quarter turn and
reaches along +y. The conditions below are distances instead of axis thresholds, so they hold
whichever way the robot faces.
"""

from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from e2e_robot_workflows.tasks.static_apple.table_height import support_height_offset

DEFAULT_OBJECT_CFG = SceneEntityCfg("object")
DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
DEFAULT_TARGET_CFG = SceneEntityCfg("destination")
DEFAULT_CONTACT_SENSOR_CFG = SceneEntityCfg("object_contact_sensor")


def object_placed_on_target(
    env: ManagerBasedRLEnv,
    placement_radius_m: float,
    placement_z_m: float,
    min_object_height_above_target_m: float,
    max_object_height_m: float,
    min_object_height_m: float,
    min_target_height_m: float,
    min_contact_force: float,
    release_clearance_m: float,
    max_object_speed: float,
    task_link_names: tuple[str, str],
    object_cfg: SceneEntityCfg = DEFAULT_OBJECT_CFG,
    robot_cfg: SceneEntityCfg = DEFAULT_ROBOT_CFG,
    target_cfg: SceneEntityCfg = DEFAULT_TARGET_CFG,
    contact_sensor_cfg: SceneEntityCfg = DEFAULT_CONTACT_SENSOR_CFG,
) -> torch.Tensor:
    """Whether the object rests on the placement target with the hand clear of it.

    The target centre is read from the scene each step rather than from its spawn constant. The
    target is a dynamic rigid body, so the arm can nudge it while reaching past its rim, and a
    fixed centre would then score a correct placement as a failure.

    Success additionally requires apple-to-plate contact and that both bodies remain above the
    support. AutoData disables the recording failure terms while it scores generated attempts, so
    without these checks a settled apple and plate together on the floor could be mislabeled as a
    successful placement.

    Args:
        env: The environment being stepped.
        placement_radius_m: How far from the target's centre the object may sit and still count.
        placement_z_m: Maximum absolute object-to-target root-height difference.
        min_object_height_above_target_m: Lowest object-to-target root-height delta allowed.
        max_object_height_m: Height above the environment origin below which the object is
            resting rather than held up. Given for the variant's nominal tabletop; a scene
            with a randomized table shifts it by that episode's live height offset.
        min_object_height_m: Minimum apple root height that still counts as supported
            placement, nominal-table-relative like ``max_object_height_m``.
        min_target_height_m: Minimum plate root height that still counts as supported
            placement, nominal-table-relative like ``max_object_height_m``.
        min_contact_force: Minimum apple-to-plate contact-force magnitude for a placement.
        release_clearance_m: How far every task link must be from the object to count as
            released, which keeps a still-grasped object from reading as placed.
        max_object_speed: Linear speed below which the object counts as settled. Angular velocity
            is left out: an apple resting in a plate can still roll slightly.
        task_link_names: Robot links that can place the object, one near each gripper.
        object_cfg: The object being placed.
        robot_cfg: The robot doing the placing.
        target_cfg: The target being placed onto.
        contact_sensor_cfg: Contact sensor filtering the object against the target.

    Returns:
        Boolean tensor, one entry per environment.
    """

    obj: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    robot = env.scene[robot_cfg.name]
    contact_sensor = env.scene[contact_sensor_cfg.name]

    object_pos = obj.data.root_pos_w.torch - env.scene.env_origins
    target_pos = target.data.root_pos_w.torch - env.scene.env_origins
    on_target = torch.linalg.vector_norm(object_pos[:, :2] - target_pos[:, :2], dim=1) <= placement_radius_m
    target_height_delta = object_pos[:, 2] - target_pos[:, 2]
    within_target_height = target_height_delta.abs() <= placement_z_m
    above_target = target_height_delta > min_object_height_above_target_m
    # Support and resting heights follow the table; the absolute failure plane does not.
    support_offset = support_height_offset(env)
    supported = (object_pos[:, 2] > min_object_height_m + support_offset) & (
        target_pos[:, 2] > min_target_height_m + support_offset
    )
    resting = object_pos[:, 2] < max_object_height_m + support_offset

    settled = torch.linalg.vector_norm(obj.data.root_vel_w.torch[:, :3], dim=1) < max_object_speed
    contact_force = torch.linalg.vector_norm(contact_sensor.data.force_matrix_w.torch, dim=-1).amax(dim=(1, 2))
    on_plate = contact_force > min_contact_force

    if not isinstance(task_link_names, list | tuple) or (
        len(task_link_names) != 2
        or any(not isinstance(name, str) or not name for name in task_link_names)
        or len(set(task_link_names)) != 2
    ):
        raise ValueError("task_link_names must contain two distinct non-empty robot links")
    link_indices = [robot.data.body_names.index(name) for name in task_link_names]
    link_pos = robot.data.body_pos_w.torch[:, link_indices, :] - env.scene.env_origins[:, None, :]
    released = (torch.linalg.vector_norm(link_pos - object_pos[:, None, :], dim=2) > release_clearance_m).all(dim=1)

    return on_target & within_target_height & above_target & supported & resting & settled & on_plate & released

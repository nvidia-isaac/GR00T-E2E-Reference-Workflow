"""H2 grasp evidence from named finger targets, measured closure, and apple contacts."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from e2e_robot_workflows.robots.h2 import H2_ARENA_ACTION_DIM, load_h2_pink_contract
from e2e_robot_workflows.tasks.static_apple.grasp_contract import H2_GRASP_CRITERIA
from e2e_robot_workflows.tasks.static_apple.table_height import support_height_offset


@cache
def h2_hand_body_names(hand: str) -> tuple[str, ...]:
    """The 24 rigid bodies in each pinned Sharpa hand, including its palm and flange."""

    if hand not in {"left", "right"}:
        raise ValueError(f"unsupported H2 hand {hand!r}")
    suffixes = (
        "hand_flange",
        "hand_C_MC",
        "thumb_CMC_VL",
        "thumb_MC",
        "thumb_MCP_VL",
        "thumb_PP",
        "thumb_DP",
        *(f"{finger}_{segment}" for finger in ("index", "middle", "ring") for segment in ("MCP_VL", "PP", "MP", "DP")),
        "pinky_MC",
        "pinky_MCP_VL",
        "pinky_PP",
        "pinky_MP",
        "pinky_DP",
    )
    names = tuple(f"{hand}_{suffix}" for suffix in suffixes)
    if len(names) != H2_GRASP_CRITERIA.contact_bodies_per_hand:
        raise ValueError("H2 hand body names disagree with the grasp-label contract")
    return names


def h2_hand_contact_filter_paths(usd_path: str | Path, robot_prim_path: str) -> dict[str, tuple[str, ...]]:
    """Resolve exact per-body filters from the already validated materialized robot USD."""

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim()
    if not root.IsValid():
        raise ValueError("H2 grasp contact filters require a robot USD with a default prim")
    root_path = str(root.GetPath())
    result: dict[str, tuple[str, ...]] = {}
    for hand in ("left", "right"):
        expected = h2_hand_body_names(hand)
        matches = [prim for prim in stage.Traverse() if prim.GetName() == f"{hand}_hand_flange"]
        if len(matches) != 1:
            raise ValueError(f"H2 grasp contact filters require one {hand} hand flange")
        bodies = [prim for prim in Usd.PrimRange(matches[0]) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        by_name = {prim.GetName(): str(prim.GetPath()) for prim in bodies}
        if len(bodies) != len(expected) or set(by_name) != set(expected):
            raise ValueError(
                f"H2 {hand} grasp contact body contract changed: "
                f"missing={sorted(set(expected) - set(by_name))}, unexpected={sorted(set(by_name) - set(expected))}"
            )
        result[hand] = tuple(f"{robot_prim_path.rstrip('/')}{by_name[name][len(root_path) :]}" for name in expected)
    return result


@cache
def _primary_flexion_targets(hand: str) -> tuple:
    """Resolve immutable named targets once, without per-observation device synchronization."""

    contract = load_h2_pink_contract()
    names = tuple(f"{hand}_{finger}_MCP_FE" for finger in H2_GRASP_CRITERIA.primary_flexion_fingers)
    closed = contract.hand_closed_targets
    opened = contract.hand_open_targets
    open_positions = tuple(opened[name] for name in names)
    travel = tuple(closed[name] - opened[name] for name in names)
    if any(value <= 0.0 for value in travel):
        raise ValueError("H2 primary flexion targets must close positively from their open positions")
    return names, tuple(14 + contract.hand_joint_order.index(name) for name in names), open_positions, travel


def reset_h2_grasp_history(env: Any, env_ids: Any, sensor_names: tuple[str, ...]) -> None:
    """Clear filtered history, which the pinned PhysX contact-sensor reset leaves intact."""

    indices = slice(None) if env_ids is None else env_ids
    for name in sensor_names:
        # Indexed assignment also updates the source for tensor/list environment IDs.
        env.scene[name].data.force_matrix_w_history.torch[indices] = 0.0


def object_held_by_h2_hand(
    env: Any,
    hand: str,
    task_link_name: str,
    grasp_distance_m: float,
    min_object_height_m: float,
    contact_sensor_cfg: Any,
    object_cfg: Any = None,
    robot_cfg: Any = None,
) -> Any:
    """Require close intent, partial measured closure, lift, proximity and sustained hand contact.

    Both hands may qualify during transfer. An open receiving palm does not qualify,
    even when it supports the apple. Contact history is sampled once per control period;
    reading this observation multiple times cannot advance its confirmation window.
    """

    import torch

    expected_bodies = h2_hand_body_names(hand)
    primary_joints, command_indices, open_targets, travel_targets = _primary_flexion_targets(hand)
    try:
        obj = env.scene[object_cfg.name if object_cfg is not None else "pick_object"]
        robot = env.scene[robot_cfg.name if robot_cfg is not None else "robot"]
        sensor = env.scene[contact_sensor_cfg.name]
        measured_all = robot.data.joint_pos.torch
        action = env.action_manager.action
        object_pos = obj.data.root_pos_w.torch - env.scene.env_origins
        link_index = robot.data.body_names.index(task_link_name)
        link_pos = robot.data.body_pos_w.torch[:, link_index] - env.scene.env_origins
        contacts = sensor.data.force_matrix_w_history.torch
        joint_indices = [robot.data.joint_names.index(name) for name in primary_joints]
    except (AttributeError, KeyError, ValueError) as error:
        raise ValueError(f"H2 {hand} grasp evidence is missing or inconsistent") from error
    num_envs = object_pos.shape[0]
    expected_contact_shape = (num_envs, H2_GRASP_CRITERIA.contact_history_samples, 1, len(expected_bodies), 3)
    if (
        tuple(action.shape) != (num_envs, H2_ARENA_ACTION_DIM)
        or tuple(measured_all.shape) != (num_envs, len(robot.data.joint_names))
        or tuple(object_pos.shape) != (num_envs, 3)
        or tuple(link_pos.shape) != (num_envs, 3)
        or tuple(contacts.shape) != expected_contact_shape
    ):
        raise ValueError(f"H2 {hand} grasp evidence has malformed tensor shapes")
    commanded = action[:, list(command_indices)]
    measured = measured_all[:, joint_indices]
    finite = torch.stack(
        [torch.isfinite(value).all() for value in (commanded, measured, object_pos, link_pos, contacts)]
    )
    if not finite.all().item():
        raise ValueError(f"H2 {hand} grasp evidence contains non-finite values")
    open_positions = commanded.new_tensor(open_targets)
    travel = commanded.new_tensor(travel_targets)
    command_closure = ((commanded - open_positions) / travel).clamp(0.0, 1.0).mean(dim=1)
    measured_closure = ((measured - open_positions) / travel).clamp(0.0, 1.0).mean(dim=1)
    # Preserve individual body forces until after taking norms: opposing fingers'
    # normal forces can cancel if their vectors are first summed across the hand.
    contact_magnitudes = torch.linalg.vector_norm(contacts, dim=-1).amax(dim=(-1, -2))
    sustained_contact = (contact_magnitudes > H2_GRASP_CRITERIA.contact_force_threshold_n).all(dim=1)
    near = torch.linalg.vector_norm(link_pos - object_pos, dim=1) < grasp_distance_m
    lifted = object_pos[:, 2] > min_object_height_m + support_height_offset(env)
    return (
        near
        & lifted
        & (command_closure >= H2_GRASP_CRITERIA.minimum_command_closure_fraction)
        & (measured_closure >= H2_GRASP_CRITERIA.minimum_measured_closure_fraction)
        & sustained_contact
    )

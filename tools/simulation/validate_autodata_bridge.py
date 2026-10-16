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

"""Check that Isaac AutoData can drive the static-apple environment.

Run under AutoData's interpreter, so this exercises the pieces that only break there: our env
cfg importing against AutoData's Isaac Lab, AutoData's own loaders parsing the descriptors this
repository owns, those descriptors agreeing with the live env's action and observation shape, and
the env id restoring this repository's recorder terms after AutoData installs its own.

It also exercises the success and subtask-boundary conditions in both directions, because a
threshold that never fires would not fail loudly: generation would simply export nothing, and
automatic annotation would reject every episode.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from validation_runner import run_validation
from validation_support import assert_finite as _assert_finite, collect_runtime_metadata

DEFAULT_RESULT_PATH = "/tmp/e2e_autodata_bridge_validation.json"
TASK_IDS = {
    "fixed": "Isaac-E2E-G1-StaticApple-PickPlace-AutoData-v0",
    "agile": "Isaac-E2E-G1-StaticApple-PickPlace-AutoData-Agile-v0",
}
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMBODIMENT_PATHS = {
    "fixed": os.path.join(REPO_ROOT, "configs/stages/autodata/static_apple_embodiment.yaml"),
    "agile": os.path.join(REPO_ROOT, "configs/stages/autodata/static_apple_agile_embodiment.yaml"),
}
TASK_DESCRIPTOR_PATHS = {
    "left": os.path.join(REPO_ROOT, "configs/stages/autodata/static_apple_left_task.yaml"),
    "right": os.path.join(REPO_ROOT, "configs/stages/autodata/static_apple_task.yaml"),
}
SETTLE_STEPS = 40
DROP_HEIGHT_M = 0.02
AGILE_HIP_HEIGHT_M = 0.72
AGILE_ACTION_WIDTH = 32
AGILE_HIP_HEIGHT_ACTION_INDEX = 31


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", default=os.environ.get("RESULT_PATH", DEFAULT_RESULT_PATH))
    parser.add_argument("--task-id", default=os.environ.get("TASK_ID"))
    parser.add_argument("--wbc-mode", choices=("fixed", "agile"), default=os.environ.get("WBC_MODE", "fixed"))
    parser.add_argument(
        "--manipulation-hand",
        choices=("left", "right"),
        default=os.environ.get("MANIPULATION_HAND", "right"),
    )
    return parser.parse_args()


def _assert_descriptors_match_env(
    env: Any,
    embodiment_path: str,
    task_descriptor_path: str,
    manipulation_hand: str,
    result: dict[str, Any],
) -> tuple[Any, Any]:
    """Fail unless AutoData's view of the action and observations matches the live env.

    Returns:
        The parsed embodiment adapter and task descriptor, for the checks that follow.
    """

    from isaac_autodata_interfaces.embodiments import embodiment_adapter_from_yaml
    from isaac_autodata_interfaces.tasks.task_descriptor import TaskDescriptor

    adapter = embodiment_adapter_from_yaml(embodiment_path)
    descriptor = TaskDescriptor.from_yaml(task_descriptor_path)

    env_action_dim = env.action_manager.total_action_dim
    if adapter.action_dim != env_action_dim:
        raise AssertionError(
            f"embodiment declares action_dim={adapter.action_dim} but the env exposes {env_action_dim}"
        )

    observations = env.observation_manager.compute()["policy"]
    for eef in (adapter.left, adapter.right):
        for key in (eef.pose_obs_keys.pos, eef.pose_obs_keys.quat):
            if key not in observations:
                raise AssertionError(f"embodiment reads observation {key!r}, which the env does not publish")

    scene_entities = set(env.scene.keys())
    for eef_name in descriptor.get_eef_names():
        for object_ref in descriptor.get_object_refs(eef_name):
            if object_ref not in scene_entities:
                raise AssertionError(f"task descriptor references object {object_ref!r}, absent from the scene")

    result["autodata_contract"] = {
        "wbc_mode": env.cfg.wbc_mode,
        "manipulation_hand": manipulation_hand,
        "action_dim": adapter.action_dim,
        "eef_names": list(descriptor.get_eef_names()),
        "term_signals": {
            eef_name: descriptor.get_term_signal_names(eef_name) for eef_name in descriptor.get_eef_names()
        },
    }
    return adapter, descriptor


def _assert_annotation_signals_readable(env: Any, adapter: Any, descriptor: Any, result: dict[str, Any]) -> None:
    """Fail unless AutoData can read every subtask signal the descriptor declares.

    Builds the same Datastream automatic annotation does and performs its startup read, so a
    missing observation group, a concatenated one, or a signal name that does not match the
    descriptor fails here instead of after a generation run has been queued.
    """

    from isaac_autodata_core.pool import DataGenInfoPool
    from isaac_autodata_interfaces.datastream import Datastream

    datastream = Datastream(
        env=env,
        task_descriptor=descriptor,
        embodiment_adapter=adapter,
        source_pool=DataGenInfoPool(
            task_descriptor=descriptor,
            embodiment_adapter=adapter,
            device=env.device,
            uses_start_signals=False,
        ),
        uses_start_signals=False,
    )
    signals = datastream.get_subtask_term_signals(env_ids=[0])
    if not signals:
        raise AssertionError("no subtask termination signals to annotate; automatic annotation would have nothing")
    result["autodata_contract"]["readable_term_signals"] = sorted(signals)


def _hold_action(env: Any, torch: Any) -> Any:
    """Build an action that holds both wrists, hands, and (when present) the AGILE base."""

    observations = env.observation_manager.compute()["policy"]
    action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    action[:, 0:3] = observations["left_eef_pos"]
    action[:, 3:7] = observations["left_eef_quat"]
    action[:, 7:10] = observations["right_eef_pos"]
    action[:, 10:14] = observations["right_eef_quat"]
    action[:, 14:28] = observations["hand_joint_state"]
    if action.shape[1] == AGILE_ACTION_WIDTH:
        # The AGILE tail is [vx, vy, wz, hip_height] at [28:32]. The zero-initialized
        # velocity commands hold position while only the final height command is set.
        action[:, AGILE_HIP_HEIGHT_ACTION_INDEX] = AGILE_HIP_HEIGHT_M
    return action


def _place_object(
    env: Any,
    object_pos: Any,
    torch: Any,
    settle_steps: int,
    hold_action: Any,
    entity: str = "object",
) -> None:
    """Teleport a scene body to a position relative to the env origin and step the sim.

    Defaults to the apple; the plate is teleported the same way to check the success target
    follows it.
    """

    body = env.scene[entity]
    pose = body.data.root_pose_w.torch.clone()
    pose[:, :3] = env.scene.env_origins + object_pos
    body.write_root_pose_to_sim_index(root_pose=pose)
    body.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(body.data.root_vel_w.torch))

    for _ in range(settle_steps):
        env.step(hold_action)


def _read_terms(
    env: Any,
    torch: Any,
    success_term: Any,
    manipulation_hand: str,
    task_link_names: dict[str, str],
) -> tuple[bool, bool, dict[str, float]]:
    """Read the success and hold terms plus the quantities they threshold.

    Reporting the measurements alongside the verdicts means a wrong threshold names the condition
    that disagreed rather than only the outcome.
    """

    hold_term = getattr(env.cfg.observations.subtask_terms, f"{manipulation_hand}_hand_holds_object")
    params = success_term.params

    pick_object = env.scene["object"]
    robot = env.scene["robot"]
    object_pos = (pick_object.data.root_pos_w.torch - env.scene.env_origins)[0]
    link_index = robot.data.body_names.index(task_link_names[manipulation_hand])
    link_pos = (robot.data.body_link_pose_w.torch[:, link_index, :3] - env.scene.env_origins)[0]
    # The term reads the target's live pose, so read it the same way here rather than from a
    # constant; otherwise these metrics would describe a placement the term did not judge.
    target_cfg = params.get("target_cfg")
    target_name = target_cfg.name if target_cfg is not None else "destination"
    target = (env.scene[target_name].data.root_pos_w.torch - env.scene.env_origins)[0, :2]
    contact_sensor = env.scene["object_contact_sensor"]
    contact_force = torch.linalg.vector_norm(contact_sensor.data.force_matrix_w.torch, dim=-1).amax(dim=(1, 2))[0]

    metrics = {
        "distance_to_target_m": float(torch.linalg.vector_norm(object_pos[:2] - target).item()),
        "target_x_m": float(target[0].item()),
        "target_y_m": float(target[1].item()),
        "target_height_m": float((env.scene[target_name].data.root_pos_w.torch - env.scene.env_origins)[0, 2].item()),
        "height_m": float(object_pos[2].item()),
        "max_linear_speed": float(torch.abs(pick_object.data.root_vel_w.torch[:, :3]).max().item()),
        "max_angular_speed": float(torch.abs(pick_object.data.root_vel_w.torch[:, 3:]).max().item()),
        "apple_plate_contact_force": float(contact_force.item()),
        "gripper_clearance_m": float(torch.linalg.vector_norm(link_pos - object_pos).item()),
    }
    placed = bool(success_term.func(env, **params)[0])
    held = bool(hold_term.func(env, **hold_term.params)[0])
    return placed, held, metrics


def _gripper_position(env: Any, manipulation_hand: str, task_link_names: dict[str, str]) -> Any:
    """Where the selected gripper reference link is, relative to the environment origin."""

    robot = env.scene["robot"]
    link_name = task_link_names[manipulation_hand]
    link_index = robot.data.body_names.index(link_name)
    return robot.data.body_link_pose_w.torch[:, link_index, :3] - env.scene.env_origins


def _execute(
    result: dict[str, Any],
    task_id: str,
    wbc_mode: str,
    manipulation_hand: str,
    embodiment_path: str,
    task_descriptor_path: str,
) -> Any:
    """Build the env under AutoData's runtime and assert the bridge's contracts."""

    import gymnasium as gym
    import torch

    import e2e_robot_workflows.tasks.static_apple  # noqa: F401  (registers the gym ids)
    from e2e_robot_workflows.tasks.static_apple.autodata_env_cfg import (
        TASK_LINK_NAMES,
        G1StaticAppleAutoDataAgileEnvCfg,
        G1StaticAppleAutoDataEnvCfg,
    )
    from e2e_robot_workflows.tasks.static_apple.spec import (
        DESTINATION_SPAWN_XY,
        ENVIRONMENT,
        PICK_OBJECT_NAME,
        PICK_SPAWN_XY,
        shelf_spawn_z,
    )

    # The placement gate is contract-owned (see ENVIRONMENT.success_placement_xy_m).
    placement_radius_m = ENVIRONMENT.success_placement_xy_m

    result["runtime"] = collect_runtime_metadata(torch)

    result["stage"] = "build_env_cfg"
    cfg_type = G1StaticAppleAutoDataEnvCfg if wbc_mode == "fixed" else G1StaticAppleAutoDataAgileEnvCfg
    cfg = cfg_type()
    if cfg.wbc_mode != wbc_mode:
        raise AssertionError(f"requested {wbc_mode!r} but the environment config selected {cfg.wbc_mode!r}")
    cfg.sim.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.scene.num_envs = 1
    cfg.seed = 42
    # AutoData reads observations per term rather than as one concatenated tensor.
    cfg.observations.policy.concatenate_terms = False
    if cfg.terminations.success is None:
        raise AssertionError("the AutoData env variant must define a success term")
    success_term = cfg.terminations.success
    # Match AutoData's setup: score with the extracted term, but do not reset while validation
    # deliberately moves the apple through success and failure poses.
    cfg.terminations = None
    spawn_range = cfg.events.reset_object.params["pose_range"]
    if spawn_range["x"] == (0.0, 0.0) or spawn_range["y"] == (0.0, 0.0):
        raise AssertionError(f"generation needs a randomized apple pose, got {spawn_range}")

    # AutoData's setup_env_config assigns this recorder over whatever the env cfg declares, so
    # install it here too: the env id's factory has to put our terms back, or a generated dataset
    # loses the articulation-wide joint command the converter reads.
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

    cfg.recorders = ActionStateRecorderManagerCfg()

    result["stage"] = "make_environment"
    env = gym.make(task_id, cfg=cfg).unwrapped
    try:
        result["stage"] = "initial_reset"
        observations, _ = env.reset()
        _assert_finite("reset_observations", observations, torch)
        hold_action = _hold_action(env, torch)

        result["stage"] = "recorder_contract"
        recorded_terms = sorted(env.recorder_manager.active_terms)
        result["robot_body_names"] = list(env.scene["robot"].body_names)
        if "record_post_step_joint_pos_target" not in recorded_terms:
            raise AssertionError(
                "generation would not record joint_pos_target, which the converter reads as the "
                f"action; recorder terms are {recorded_terms}"
            )
        result["recorder_terms"] = recorded_terms

        result["stage"] = "descriptor_contract"
        adapter, descriptor = _assert_descriptors_match_env(
            env,
            embodiment_path,
            task_descriptor_path,
            manipulation_hand,
            result,
        )

        result["stage"] = "annotation_signal_contract"
        _assert_annotation_signals_readable(env, adapter, descriptor, result)

        result["stage"] = "term_behavior"
        spawn_height = shelf_spawn_z(PICK_OBJECT_NAME)
        # Each case is (apple position relative to the env origin, placed?, held?, settle steps).
        # A held apple must not settle: released from the gripper it would only fall back down.
        cases = {
            "at_spawn": ((*PICK_SPAWN_XY, spawn_height + DROP_HEIGHT_M), False, False, SETTLE_STEPS),
            "on_plate": ((*DESTINATION_SPAWN_XY, spawn_height + DROP_HEIGHT_M), True, False, SETTLE_STEPS),
            # Past the placement radius, so the tolerance is not effectively unbounded.
            "past_plate": (
                (DESTINATION_SPAWN_XY[0] + 3.0 * placement_radius_m, DESTINATION_SPAWN_XY[1], spawn_height),
                False,
                False,
                SETTLE_STEPS,
            ),
            # In the hand, mid-transfer: the annotation signal's rising edge, and not a placement.
            "in_hand": (None, False, True, 1),
        }
        checks = {}
        total_steps = 0
        for name, (object_xy_z, expect_placed, expect_held, settle_steps) in cases.items():
            position = (
                _gripper_position(env, manipulation_hand, TASK_LINK_NAMES)
                if object_xy_z is None
                else torch.tensor(object_xy_z, device=env.device, dtype=torch.float32)
            )
            _place_object(env, position, torch, settle_steps, hold_action)
            placed, held, metrics = _read_terms(
                env,
                torch,
                success_term,
                manipulation_hand,
                TASK_LINK_NAMES,
            )
            checks[name] = {
                "expected": {"placed": expect_placed, "held": expect_held},
                "fired": {"placed": placed, "held": held},
                **metrics,
            }
            total_steps += settle_steps
        result["term_checks"] = checks

        mismatched = [name for name, check in checks.items() if check["fired"] != check["expected"]]
        if mismatched:
            raise AssertionError(f"terms disagreed for {mismatched}: {checks}")

        result["stage"] = "target_tracks_plate"
        # The plate is a dynamic body the arm can nudge, so the success term reads its live pose.
        # Move the plate and the target must move with it: against a hard-coded centre both cases
        # below would report the opposite verdict.
        plate_height = float((env.scene["destination"].data.root_pos_w.torch - env.scene.env_origins)[0, 2].item())
        moved_plate_xy = (DESTINATION_SPAWN_XY[0] - 2.0 * placement_radius_m, DESTINATION_SPAWN_XY[1])
        _place_object(
            env,
            torch.tensor((*moved_plate_xy, plate_height), device=env.device, dtype=torch.float32),
            torch,
            SETTLE_STEPS,
            hold_action,
            entity="destination",
        )
        total_steps += SETTLE_STEPS

        tracking = {}
        for name, apple_xy, expect_placed in (
            ("apple_at_old_plate_xy", DESTINATION_SPAWN_XY, False),
            ("apple_at_new_plate_xy", moved_plate_xy, True),
        ):
            _place_object(
                env,
                torch.tensor((*apple_xy, spawn_height + DROP_HEIGHT_M), device=env.device, dtype=torch.float32),
                torch,
                SETTLE_STEPS,
                hold_action,
            )
            placed, _, metrics = _read_terms(
                env,
                torch,
                success_term,
                manipulation_hand,
                TASK_LINK_NAMES,
            )
            tracking[name] = {
                "expected": {"placed": expect_placed},
                "fired": {"placed": placed},
                **metrics,
            }
            total_steps += SETTLE_STEPS
        result["target_tracking"] = tracking

        untracked = [name for name, check in tracking.items() if check["fired"] != check["expected"]]
        if untracked:
            raise AssertionError(f"the success target did not follow the plate for {untracked}: {tracking}")

        result["stage"] = "reject_dropped_target"
        # The generation env disables recorder failure terms. The success term itself must
        # therefore reject a plate and apple that land together below the table support.
        below_support_target_z = float(success_term.params["min_target_height_m"]) - 0.10
        below_support_object_z = float(success_term.params["min_object_height_m"]) - 0.10
        _place_object(
            env,
            torch.tensor((*moved_plate_xy, below_support_target_z), device=env.device, dtype=torch.float32),
            torch,
            SETTLE_STEPS,
            hold_action,
            entity="destination",
        )
        _place_object(
            env,
            torch.tensor((*moved_plate_xy, below_support_object_z), device=env.device, dtype=torch.float32),
            torch,
            SETTLE_STEPS,
            hold_action,
        )
        total_steps += 2 * SETTLE_STEPS
        dropped_target_placed, _, dropped_target_metrics = _read_terms(
            env,
            torch,
            success_term,
            manipulation_hand,
            TASK_LINK_NAMES,
        )
        result["dropped_target"] = {
            "expected": {"placed": False},
            "fired": {"placed": dropped_target_placed},
            **dropped_target_metrics,
        }
        if dropped_target_placed:
            raise AssertionError(f"a target below its support counted as placement: {result['dropped_target']}")
        result["completed_steps"] = total_steps
    except BaseException:
        env.close()
        raise
    return env


def main() -> int:
    args = _parse_args()
    task_id = args.task_id or TASK_IDS[args.wbc_mode]
    embodiment_path = EMBODIMENT_PATHS[args.wbc_mode]
    task_descriptor_path = TASK_DESCRIPTOR_PATHS[args.manipulation_hand]
    return run_validation(
        result_path=args.result_path,
        task_id=task_id,
        requested_steps=8 * SETTLE_STEPS + 1,
        execute=lambda result: _execute(
            result,
            task_id,
            args.wbc_mode,
            args.manipulation_hand,
            embodiment_path,
            task_descriptor_path,
        ),
        log_label="autodata-bridge",
        include_traceback=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())

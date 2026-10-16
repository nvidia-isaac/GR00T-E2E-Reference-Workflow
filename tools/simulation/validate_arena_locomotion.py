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

"""Exercise locomotion on the Arena-composed environment through headless GPU simulation.

The drive is the one ``validate_agile_locomotion.py`` runs against the built-in task --
the same phases, commands, and behavior thresholds, from :mod:`agile_validation` -- so the
two runtimes are held to one standard while composing the scene and the control stack
differently. Arena's whole-body controller takes the command in the seven-number
locomotion tail its layouts end with, and runs an ONNX export of WBC-AGILE rather than the
checkpoint the built-in task loads.

Two assertions from the built-in validation have no equivalent here and are not restated
in a weaker form. The built-in one reads a 12-number policy output out of a dedicated
``lower_body_policy`` observation group, which Arena has no counterpart for; instead the
policy's identity is pinned (``arena_wbc_agile_policy_revision``, asserted below and by
``tools/check_arena_pin.py``) and its effect is measured by the behavior phases, which a
policy that produced nothing could not pass.
"""

from __future__ import annotations

import argparse
import math
import os
import tomllib
from pathlib import Path
from typing import Any

from agile_validation import (
    BACKWARD_PHASE,
    EXPECTED_CONTROL_PERIOD_S,
    HIP_HEIGHT_M,
    OBJECT_SUPPORT_TOLERANCE_M,
    RESET_SETTLE_STEPS,
    SETTLE_PHASE,
    STOP_PHASE,
    TOTAL_CONTROL_STEPS,
    TURN_PHASE,
    LocomotionRuntime,
    assert_reset_to_initial_pose,
    assert_settled,
    assert_turned_in_place,
    assert_walked_backward,
    heading_xy,
    run_phase,
)
from validation_runner import run_validation
from validation_support import (
    assert_action_manager_reset,
    assert_finite,
    collect_runtime_metadata,
    quaternion_distance_rad,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_PATH = "/tmp/e2e_arena_locomotion_validation.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", default=os.environ.get("RESULT_PATH", DEFAULT_RESULT_PATH))
    parser.add_argument(
        "--embodiment",
        default=os.environ.get("ARENA_EMBODIMENT"),
        help="Arena G1 embodiment to drive. Defaults to the joint layout.",
    )
    return parser.parse_args()


def _portable_error_message(error: BaseException) -> str:
    """Keep structured artifacts useful without persisting host/container paths."""

    message = str(error)
    for path in (str(REPO_ROOT), os.environ.get("E2E_ARENA_ROOT", "")):
        if path and Path(path).is_absolute():
            message = message.replace(path, "<repository>" if path == str(REPO_ROOT) else "<arena>")
    return message


def _assert_pinned_locomotion_policy(result: dict[str, Any]) -> None:
    """Check the checkout's whole-body controller runs the policy ``runtime.toml`` pins.

    ``tools/check_arena_pin.py`` reads the same constant out of the checkout's source
    before anything starts; this reads it off the imported config, so the policy that
    actually loads is the one recorded as tested.
    """

    from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.config.configs import AgileConfig

    pinned = tomllib.loads((REPO_ROOT / "runtime.toml").read_text(encoding="utf-8"))["tested"]
    expected = pinned["arena_wbc_agile_policy_revision"]
    actual = AgileConfig.COMMIT_HASH_FOR_WBC_AGILE_POLICY
    result["policy"] = {
        "filename": AgileConfig.wbc_model_path.rsplit("/", 1)[-1],
        "wbc_agile_revision": actual,
        "wbc_agile_revision_expected": expected,
        "wbc_agile_revision_verified": actual == expected,
        "wbc_version": AgileConfig.wbc_version,
    }
    if actual != expected:
        raise AssertionError(f"Arena's WBC-AGILE policy commit {actual} does not match the pinned {expected}")


def _run_simulation(embodiment: str | None, result: dict[str, Any]) -> Any:
    import gymnasium as gym
    import torch
    from isaaclab.utils.math import quat_apply

    from e2e_robot_workflows.arena.composition import compose
    from e2e_robot_workflows.arena.external_environment import G1_ARENA_EMBODIMENT
    from e2e_robot_workflows.arena.g1_actions import hold_action, set_navigate_command
    from e2e_robot_workflows.environments import STATIC_APPLE_ENVIRONMENT
    from e2e_robot_workflows.environments.scene_keys import resolve_scene_entity_name

    if not torch.cuda.is_available():
        raise AssertionError("CUDA is unavailable to the Arena locomotion validation")

    result["runtime"] = collect_runtime_metadata(torch)
    result["stage"] = "check_policy_pin"
    _assert_pinned_locomotion_policy(result)

    result["stage"] = "compose_environment"
    embodiment = embodiment or G1_ARENA_EMBODIMENT
    result["embodiment"] = embodiment
    composed = compose(embodiment=embodiment, device="cuda:0")
    env_cfg = composed.env_cfg
    env_cfg.seed = 42
    control_period_s = env_cfg.sim.dt * env_cfg.decimation
    render_period_s = env_cfg.sim.dt * env_cfg.sim.render_interval
    if not math.isclose(control_period_s, EXPECTED_CONTROL_PERIOD_S, rel_tol=0.0, abs_tol=1.0e-9):
        raise AssertionError(
            f"the Arena locomotion validation requires a {EXPECTED_CONTROL_PERIOD_S:.6g} s control period, "
            f"got {control_period_s:.6g} s"
        )
    if not math.isclose(render_period_s, 0.02, rel_tol=0.0, abs_tol=1.0e-9):
        raise AssertionError(f"the G1 Arena renderer must remain at 50 Hz, got period {render_period_s:.6g} s")
    result["rates_hz"] = {
        "physics": round(1.0 / env_cfg.sim.dt),
        "control": round(1.0 / control_period_s),
        "render": round(1.0 / render_period_s),
    }

    result["stage"] = "make_environment"
    env = composed.make(gym)
    try:
        if env.num_envs != 1 or not str(env.device).startswith("cuda"):
            raise AssertionError(f"expected one CUDA environment, got num_envs={env.num_envs}, device={env.device}")

        result["stage"] = "initial_reset"
        observations, _ = env.reset()
        assert_finite("reset_observations", observations, torch)

        robot = env.scene["robot"]
        if robot.is_fixed_base:
            raise AssertionError("locomotion validation received a fixed-base robot")
        joint_count = len(robot.joint_names)
        action_term = env.action_manager.get_term(env.action_manager.active_terms[0])
        initial_root_state = robot.data.root_state_w.torch.clone()
        support_z = STATIC_APPLE_ENVIRONMENT.support_surface_z - OBJECT_SUPPORT_TOLERANCE_M

        def assert_policy_outputs() -> float:
            """Check the whole-body controller produced usable joint targets this step."""

            processed = action_term.processed_actions
            assert_finite("wbc_action.raw_actions", action_term.raw_actions, torch)
            assert_finite("wbc_action.processed_actions", processed, torch)
            if tuple(processed.shape) != (1, joint_count):
                raise AssertionError(
                    f"the whole-body controller wrote {tuple(processed.shape)} joint targets, "
                    f"expected {(1, joint_count)}"
                )
            if processed.requires_grad:
                raise AssertionError("whole-body controller outputs must be detached before Warp actuation")
            return float(processed.abs().max().item())

        runtime = LocomotionRuntime(
            # hold_action rejects a layout that is not the composed embodiment's, so the
            # action width is asserted by building the first command.
            make_action=lambda command: set_navigate_command(
                hold_action(env, embodiment, base_height_m=command[3]), *command[:3]
            ),
            pick_object_name=resolve_scene_entity_name(env.scene, "pick_object"),
            assert_policy_outputs=assert_policy_outputs,
        )
        phase_context = {
            "env": env,
            "runtime": runtime,
            "support_z": support_z,
            "result": result,
            "torch": torch,
            "quat_apply": quat_apply,
        }

        settle_outcome = run_phase(*SETTLE_PHASE, stage="phase_settle", **phase_context)
        backward_start_position = robot.data.root_state_w.torch[:, :3].clone()
        backward_start_heading = heading_xy(robot.data.root_state_w.torch[:, 3:7], torch, quat_apply)
        backward_outcome = run_phase(*BACKWARD_PHASE, stage="phase_backward", **phase_context)
        backward_root_position = robot.data.root_state_w.torch[:, :3].clone()
        stop_outcome = run_phase(*STOP_PHASE, stage="phase_stop", **phase_context)
        turn_initial_heading = heading_xy(robot.data.root_state_w.torch[:, 3:7], torch, quat_apply)
        turn_initial_position = robot.data.root_state_w.torch[:, :2].clone()
        turn_outcome = run_phase(*TURN_PHASE, stage="phase_turn", **phase_context)

        phase_outcomes = (settle_outcome, backward_outcome, stop_outcome, turn_outcome)
        backward_progress, backward_lateral_drift = assert_walked_backward(
            backward_start_position, backward_root_position, backward_start_heading, torch
        )
        stop_tail_mean_planar_speed, stop_tail_mean_abs_yaw_rate = assert_settled(
            stop_outcome, "the robot did not settle after the stop command"
        )
        turn_yaw_delta, turn_planar_displacement = assert_turned_in_place(
            turn_initial_heading,
            heading_xy(robot.data.root_state_w.torch[:, 3:7], torch, quat_apply),
            turn_initial_position,
            robot.data.root_state_w.torch[:, :2],
            torch,
        )

        result["stage"] = "second_reset"
        reset_observations, _ = env.reset()
        assert_action_manager_reset(env, reset_observations, torch)
        reset_position_error, reset_rotation_error = assert_reset_to_initial_pose(
            robot.data.root_state_w.torch.clone(), initial_root_state, torch, quaternion_distance_rad
        )

        reset_outcome = run_phase(
            "post_reset_settle",
            RESET_SETTLE_STEPS,
            (0.0, 0.0, 0.0, HIP_HEIGHT_M),
            stage="post_reset_settle",
            **phase_context,
        )
        if not bool((env.episode_length_buf == RESET_SETTLE_STEPS).all().item()):
            raise AssertionError(f"episode length did not advance after reset: {env.episode_length_buf.tolist()}")
        reset_tail_mean_planar_speed, reset_tail_mean_abs_yaw_rate = assert_settled(
            reset_outcome, "the robot did not recover its balance after reset"
        )

        result["simulation"] = {
            "action_dim": env.action_manager.total_action_dim,
            "action_terms": list(env.action_manager.active_terms),
            "backward_lateral_drift_m": backward_lateral_drift,
            "backward_progress_m": backward_progress,
            "control_period_s": control_period_s,
            "decimation": env_cfg.decimation,
            "device": str(env.device),
            "embodiment": embodiment,
            "env_type": type(env).__name__,
            "joint_count": joint_count,
            "max_pelvis_height_m": max(outcome.summary["max_pelvis_height_m"] for outcome in phase_outcomes),
            "max_pelvis_tilt_rad": max(outcome.summary["max_pelvis_tilt_rad"] for outcome in phase_outcomes),
            "max_policy_output_magnitude": max(
                outcome.summary["max_policy_output_magnitude"] for outcome in phase_outcomes
            ),
            "min_object_local_z_m": min(outcome.min_object_z for outcome in phase_outcomes),
            "min_pelvis_height_m": min(outcome.summary["min_pelvis_height_m"] for outcome in phase_outcomes),
            "num_envs": env.num_envs,
            "phases": {outcome.name: outcome.summary for outcome in phase_outcomes},
            "physics_dt_s": env_cfg.sim.dt,
            "pick_object_entity": runtime.pick_object_name,
            "post_reset_settle": {
                "episode_length": int(env.episode_length_buf[0].item()),
                "final_state": reset_outcome.final_state,
                "steps": RESET_SETTLE_STEPS,
                "tail_mean_abs_yaw_rate_rad_s": reset_tail_mean_abs_yaw_rate,
                "tail_mean_planar_speed_m_s": reset_tail_mean_planar_speed,
            },
            "reset": {
                "root_position_error_m": reset_position_error,
                "root_rotation_error_rad": reset_rotation_error,
            },
            "scene_entities": sorted(env.scene.keys()),
            "stop_tail_mean_abs_yaw_rate_rad_s": stop_tail_mean_abs_yaw_rate,
            "stop_tail_mean_planar_speed_m_s": stop_tail_mean_planar_speed,
            "support_limit_local_z_m": support_z,
            "turn_planar_displacement_m": turn_planar_displacement,
            "turn_yaw_delta_rad": turn_yaw_delta,
        }
        return env
    except BaseException:
        try:
            env.close()
        except BaseException as cleanup_error:
            result.setdefault("cleanup_errors", []).append(
                f"environment: {type(cleanup_error).__name__}: {_portable_error_message(cleanup_error)}"
            )
        raise


def main() -> int:
    args = _parse_args()

    def execute(result: dict[str, Any]) -> Any:
        result["stage"] = "simulation"
        return _run_simulation(args.embodiment, result)

    return run_validation(
        result_path=args.result_path,
        task_id="e2e_static_apple",
        requested_steps=TOTAL_CONTROL_STEPS,
        execute=execute,
        log_label="arena-locomotion-validation",
        error_message=_portable_error_message,
    )


if __name__ == "__main__":
    raise SystemExit(main())

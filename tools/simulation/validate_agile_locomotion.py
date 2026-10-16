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

"""Exercise AGILE locomotion through real headless GPU simulation.

The validation holds the G1 wrists and hands at their current poses while the
Isaac Lab AGILE policy settles, walks backward, stops, and turns.  It checks
that the free-base robot remains upright, follows the commanded motion, and
resets cleanly without depending on a rendered window.
"""

from __future__ import annotations

import argparse
import hashlib
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
    assert_builtin_policy_outputs,
    assert_reset_to_initial_pose,
    assert_settled,
    assert_turned_in_place,
    assert_walked_backward,
    derive_action_layout,
    heading_xy,
    make_builtin_action,
    run_phase,
)
from pink_validation import resolve_g1_pink_handles
from validation_runner import run_validation
from validation_support import (
    assert_action_manager_reset,
    assert_finite as _assert_finite,
    collect_runtime_metadata,
    quaternion_distance_rad,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_PATH = "/tmp/e2e_agile_locomotion_validation.json"
DEFAULT_TASK_ID = "Isaac-E2E-G1-StaticApple-PickPlace-Abs-v0"

MIN_POLICY_OUTPUT_MAGNITUDE = 1.0e-3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", default=os.environ.get("RESULT_PATH", DEFAULT_RESULT_PATH))
    parser.add_argument("--task-id", default=os.environ.get("TASK_ID", DEFAULT_TASK_ID))
    return parser.parse_args()


def _portable_error_message(error: BaseException) -> str:
    """Keep structured artifacts useful without persisting host/container paths."""

    message = str(error)
    replacements = (
        (str(REPO_ROOT), "<repository>"),
        (os.environ.get("E2E_AGILE_POLICY_PATH", ""), "<policy-override>"),
    )
    for path, placeholder in replacements:
        if path and Path(path).is_absolute():
            message = message.replace(path, placeholder)
    return message


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as policy_file:
        for chunk in iter(lambda: policy_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_policy_sha256() -> str:
    compatibility_path = REPO_ROOT / "runtime.toml"
    compatibility = tomllib.loads(compatibility_path.read_text(encoding="utf-8"))
    expected = os.environ.get("E2E_AGILE_POLICY_SHA256") or compatibility["tested"]["agile_policy_sha256"]
    expected = str(expected).lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("the expected AGILE policy SHA-256 must contain 64 hexadecimal characters")
    return expected


def _resolve_policy(cfg: Any, result: dict[str, Any]) -> None:
    """Resolve the policy before environment creation and record portable provenance."""

    from isaaclab.utils.assets import retrieve_file_path

    policy_override = os.environ.get("E2E_AGILE_POLICY_PATH")
    policy_source = "override" if policy_override else "default"
    configured_path = policy_override or cfg.actions.lower_body_joint_pos.policy_path
    resolved_path = Path(
        retrieve_file_path(
            configured_path,
            download_dir="/tmp/e2e-agile-policy",
            force_download=False,
        )
    )
    if not resolved_path.is_file():
        raise AssertionError("the resolved AGILE policy is not a local file")

    # The action term must consume the already-resolved local file.  Only
    # portable metadata is persisted; checkout/cache paths stay out of results.
    cfg.actions.lower_body_joint_pos.policy_path = str(resolved_path)
    actual_sha256 = _sha256(resolved_path)
    expected_sha256 = _expected_policy_sha256()
    result["policy"] = {
        "filename": resolved_path.name,
        "sha256": actual_sha256,
        "sha256_expected": expected_sha256,
        "sha256_verified": actual_sha256 == expected_sha256,
        "size_bytes": resolved_path.stat().st_size,
        "source": policy_source,
    }
    if actual_sha256 != expected_sha256:
        raise AssertionError(
            f"AGILE policy SHA-256 {actual_sha256} does not match the expected digest {expected_sha256}"
        )


def _run_simulation(task_id: str, result: dict[str, Any]) -> Any:
    import gymnasium as gym
    import torch
    from isaaclab.utils.math import quat_apply

    import e2e_robot_workflows.tasks.static_apple  # noqa: F401
    from e2e_robot_workflows.tasks.static_apple.env_cfg import SHELF_SURFACE_Z, make_env_cfg

    if not torch.cuda.is_available():
        raise AssertionError("CUDA is unavailable to the AGILE locomotion validation")

    result["runtime"] = collect_runtime_metadata(torch)

    result["stage"] = "build_env_cfg"
    cfg = make_env_cfg(wbc_mode="agile", device="cuda:0", num_envs=1)
    cfg.seed = 42
    if cfg.wbc_mode != "agile":
        raise AssertionError(f"AGILE locomotion validation requires agile WBC mode, got {cfg.wbc_mode!r}")
    control_period_s = cfg.sim.dt * cfg.decimation
    if not math.isclose(control_period_s, EXPECTED_CONTROL_PERIOD_S, rel_tol=0.0, abs_tol=1.0e-9):
        raise AssertionError(
            f"AGILE locomotion validation requires a {EXPECTED_CONTROL_PERIOD_S:.6g} s control period, "
            f"got {control_period_s:.6g} s"
        )
    if cfg.scene.robot.spawn.articulation_props.fix_root_link:
        raise AssertionError("AGILE locomotion validation received a fixed-base robot configuration")
    if cfg.actions.lower_body_joint_pos is None or cfg.observations.lower_body_policy is None:
        raise AssertionError("AGILE action or policy observation configuration is missing")

    result["stage"] = "resolve_policy"
    _resolve_policy(cfg, result)

    result["stage"] = "make_environment"
    env = gym.make(task_id, cfg=cfg).unwrapped
    try:
        if env.num_envs != 1 or not str(env.device).startswith("cuda"):
            raise AssertionError(f"expected one CUDA environment, got num_envs={env.num_envs}, device={env.device}")

        result["stage"] = "initial_reset"
        observations, _ = env.reset()
        _assert_finite("reset_observations", observations, torch)

        action_slices, upper_term, lower_term = derive_action_layout(env)
        action_context = resolve_g1_pink_handles(env.scene["robot"], upper_term)
        robot = env.scene["robot"]
        initial_root_state = robot.data.root_state_w.torch.clone()
        support_z = SHELF_SURFACE_Z - OBJECT_SUPPORT_TOLERANCE_M

        runtime = LocomotionRuntime(
            make_action=lambda command: make_builtin_action(
                env, action_slices, upper_term, action_context, command, torch
            ),
            pick_object_name="object",
            assert_policy_outputs=lambda: assert_builtin_policy_outputs(
                lower_term, env.obs_buf["lower_body_policy"], torch
            ),
            assert_upper_body=lambda: _assert_finite(
                "upper_body_ik.processed_actions", upper_term.processed_actions, torch
            ),
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
        phase_metrics = {outcome.name: outcome.summary for outcome in phase_outcomes}
        max_policy_output = max(outcome.summary["max_policy_output_magnitude"] for outcome in phase_outcomes)
        min_object_z = min(outcome.min_object_z for outcome in phase_outcomes)
        min_pelvis_height = min(outcome.summary["min_pelvis_height_m"] for outcome in phase_outcomes)
        max_pelvis_height = max(outcome.summary["max_pelvis_height_m"] for outcome in phase_outcomes)
        max_pelvis_tilt = max(outcome.summary["max_pelvis_tilt_rad"] for outcome in phase_outcomes)

        backward_progress, backward_lateral_drift = assert_walked_backward(
            backward_start_position, backward_root_position, backward_start_heading, torch
        )
        stop_tail_mean_planar_speed, stop_tail_mean_abs_yaw_rate = assert_settled(
            stop_outcome, "AGILE did not settle after the stop command"
        )
        final_heading = heading_xy(robot.data.root_state_w.torch[:, 3:7], torch, quat_apply)
        turn_yaw_delta, turn_planar_displacement = assert_turned_in_place(
            turn_initial_heading,
            final_heading,
            turn_initial_position,
            robot.data.root_state_w.torch[:, :2],
            torch,
        )
        if max_policy_output <= MIN_POLICY_OUTPUT_MAGNITUDE:
            raise AssertionError(f"AGILE policy output remained trivial: maximum magnitude {max_policy_output:.6g}")

        result["stage"] = "second_reset"
        reset_observations, _ = env.reset()
        assert_action_manager_reset(env, reset_observations, torch)

        reset_root_state = robot.data.root_state_w.torch.clone()
        reset_position_error, reset_rotation_error = assert_reset_to_initial_pose(
            reset_root_state, initial_root_state, torch, quaternion_distance_rad
        )

        # Upstream's AGILE action term retains its private policy outputs across
        # reset, so reset correctness is intentionally asserted at the manager
        # boundary above rather than requiring raw/processed policy tensors to
        # be zero here.
        lower_raw_after_reset = float(lower_term.raw_actions.abs().max().item())
        lower_processed_after_reset = float(lower_term.processed_actions.abs().max().item())
        reset_outcome = run_phase(
            "post_reset_settle",
            RESET_SETTLE_STEPS,
            (0.0, 0.0, 0.0, HIP_HEIGHT_M),
            stage="post_reset_settle",
            check_upper_action=False,
            **phase_context,
        )
        if not bool((env.episode_length_buf == RESET_SETTLE_STEPS).all().item()):
            raise AssertionError(f"episode length did not advance after reset: {env.episode_length_buf.tolist()}")
        reset_tail_mean_planar_speed, reset_tail_mean_abs_yaw_rate = assert_settled(
            reset_outcome, "AGILE did not recover its balance after reset"
        )

        result["simulation"] = {
            "action_dim": env.action_manager.total_action_dim,
            "action_slices": {
                name: [action_slice.start, action_slice.stop] for name, action_slice in action_slices.items()
            },
            "backward_lateral_drift_m": backward_lateral_drift,
            "backward_progress_m": backward_progress,
            "control_period_s": control_period_s,
            "decimation": cfg.decimation,
            "device": str(env.device),
            "env_type": type(env).__name__,
            "hand_joint_count": len(action_context["hand_joint_names"]),
            "max_pelvis_height_m": max_pelvis_height,
            "max_pelvis_tilt_rad": max_pelvis_tilt,
            "max_policy_output_magnitude": max_policy_output,
            "min_object_local_z_m": min_object_z,
            "min_pelvis_height_m": min_pelvis_height,
            "num_envs": env.num_envs,
            "phases": phase_metrics,
            "physics_dt_s": cfg.sim.dt,
            "post_reset_settle": {
                "final_state": reset_outcome.final_state,
                "episode_length": int(env.episode_length_buf[0].item()),
                "steps": RESET_SETTLE_STEPS,
                "tail_mean_abs_yaw_rate_rad_s": reset_tail_mean_abs_yaw_rate,
                "tail_mean_planar_speed_m_s": reset_tail_mean_planar_speed,
            },
            "reset": {
                "lower_processed_action_magnitude_retained": lower_processed_after_reset,
                "lower_raw_action_magnitude_retained": lower_raw_after_reset,
                "root_position_error_m": reset_position_error,
                "root_rotation_error_rad": reset_rotation_error,
            },
            "scene_entities": sorted(env.scene.keys()),
            "stop_tail_mean_abs_yaw_rate_rad_s": stop_tail_mean_abs_yaw_rate,
            "stop_tail_mean_planar_speed_m_s": stop_tail_mean_planar_speed,
            "support_limit_local_z_m": support_z,
            "turn_planar_displacement_m": turn_planar_displacement,
            "turn_yaw_delta_rad": turn_yaw_delta,
            "wbc_mode": cfg.wbc_mode,
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
        return _run_simulation(args.task_id, result)

    return run_validation(
        result_path=args.result_path,
        task_id=args.task_id,
        requested_steps=TOTAL_CONTROL_STEPS,
        execute=execute,
        log_label="agile-locomotion-validation",
        error_message=_portable_error_message,
    )


if __name__ == "__main__":
    raise SystemExit(main())

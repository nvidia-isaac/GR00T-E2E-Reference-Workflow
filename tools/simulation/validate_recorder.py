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

"""Verify the recorder captures a full demonstration through headless GPU sim.

Records a scripted "hold pose" take (no headset) through the same recorder path
as ``record_demos.py``, exports it to HDF5, then reopens the file and checks the
episode actually contains per-step actions and the head-camera observation. This
catches the class of bug where the recorder silently drops fields, and produces
a small current-schema dataset for the conversion stage to develop against.

Writes a JSON result to ``$RESULT_PATH`` with ``"status": "ok"`` only when the
recorded episode has non-empty actions and camera frames.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

DEFAULT_RESULT_PATH = "/tmp/e2e_recorder_smoke.json"
DEFAULT_TASK_ID = "Isaac-E2E-G1-StaticApple-PickPlace-Abs-v0"
DEFAULT_STEPS = 20
CAMERA_OBS_KEY = "robot_head_cam_rgb"
STATE_OBS_KEY = "robot_joint_pos"


def _verify_state_replay(env: Any) -> dict[str, Any]:
    """Check restored articulation/object transforms reach Fabric without physics steps."""

    import numpy as np
    import torch
    import usdrt
    from isaaclab.utils.datasets import EpisodeData
    from pxr import Gf, UsdUtils

    from e2e_robot_workflows.data.replay import make_state_renderer, replay_recorded_states

    initial_state = env.scene.get_state(is_relative=True)
    displaced_state = copy.deepcopy(initial_state)
    joint_id = env.scene["robot"].joint_names.index("left_shoulder_pitch_joint")
    displaced_state["articulation"]["robot"]["joint_position"][0, joint_id] += 0.35
    displaced_state["rigid_object"]["object"]["root_pose"][0, 0] += 0.15
    displaced_state["rigid_object"]["object"]["root_pose"][0, 2] += 0.10
    stage_id = UsdUtils.StageCache.Get().GetId(env.sim.stage).ToLongInt()
    fabric_stage = usdrt.Usd.Stage.Attach(stage_id)
    physics_steps_before = env.sim.get_physics_step_count()
    paths = [path for asset in env.scene.articulations.values() for path in asset.root_view.link_paths[0]]
    paths.extend(path for asset in env.scene.rigid_objects.values() for path in asset.root_view.prim_paths)
    scales_before = {
        path: np.linalg.norm(
            np.asarray(usdrt.Rt.Xformable(fabric_stage.GetPrimAtPath(path)).GetFabricHierarchyWorldMatrixAttr().Get())[
                :3, :3
            ],
            axis=1,
        )
        for path in paths
    }
    render_frame = make_state_renderer(env)
    max_position_error = max_rotation_error = max_scale_change = 0.0
    for frame, state in enumerate((initial_state, displaced_state, initial_state, None)):
        if state is None:
            env.reset()
            render_frame()
        else:
            episode = EpisodeData()
            episode.data = {
                "initial_state": initial_state,
                "states": {kind: assets for kind, assets in state.items() if assets},
                "actions": torch.zeros((1, env.action_manager.total_action_dim), device=env.device),
            }
            completed, expected = replay_recorded_states(env, episode, lambda: True, render_frame)
            if (completed, expected) != (1, 1):
                raise AssertionError(f"state replay did not consume its frame: {completed}/{expected}")
        assets = [
            (asset.root_view.link_paths[0], asset.root_view.get_link_transforms().numpy().reshape(-1, 7))
            for asset in env.scene.articulations.values()
        ]
        assets.extend(
            (asset.root_view.prim_paths, asset.root_view.get_transforms().numpy())
            for asset in env.scene.rigid_objects.values()
        )
        for paths, poses in assets:
            for path, pose in zip(paths, poses, strict=True):
                matrix = np.asarray(
                    usdrt.Rt.Xformable(fabric_stage.GetPrimAtPath(path)).GetFabricHierarchyWorldMatrixAttr().Get()
                )
                scales = np.linalg.norm(matrix[:3, :3], axis=1)
                position_error = float(np.max(np.abs(matrix[3, :3] - pose[:3])))
                expected_rotation = np.asarray(
                    Gf.Matrix3d(Gf.Rotation(Gf.Quatd(float(pose[6]), Gf.Vec3d(*map(float, pose[3:6]))).GetNormalized()))
                )
                rotation_error = float(np.max(np.abs(matrix[:3, :3] / scales[:, None] - expected_rotation)))
                scale_change = float(np.max(np.abs(scales - scales_before[path])))
                max_position_error = max(max_position_error, position_error)
                max_rotation_error = max(max_rotation_error, rotation_error)
                max_scale_change = max(max_scale_change, scale_change)
                if position_error > 1e-4 or rotation_error > 1e-4 or scale_change > 1e-5:
                    raise AssertionError(
                        f"rendered replay frame {frame} differs from PhysX at {path}: "
                        f"position_error={position_error:.6g} m, rotation_error={rotation_error:.6g}, "
                        f"scale_change={scale_change:.6g}"
                    )
    physics_steps = env.sim.get_physics_step_count() - physics_steps_before
    if physics_steps:
        raise AssertionError(f"recorded-state replay advanced physics by {physics_steps} steps")
    return {
        "frames": 3,
        "episodes": 3,
        "final_reset_rendered": True,
        "physics_steps": physics_steps,
        "max_position_error_m": max_position_error,
        "max_rotation_matrix_error": max_rotation_error,
        "max_scale_change": max_scale_change,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", default=os.environ.get("RESULT_PATH", DEFAULT_RESULT_PATH))
    parser.add_argument("--task-id", default=os.environ.get("TASK_ID", DEFAULT_TASK_ID))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--dataset-file", default=os.environ.get("DATASET_FILE"))
    return parser.parse_args()


def _write_result(path: str, result: dict[str, Any]) -> None:
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = result_path.with_name(f".{result_path.name}.tmp")
    tmp.write_text(f"{json.dumps(result, indent=2, sort_keys=True)}\n", encoding="utf-8")
    tmp.replace(result_path)


def _record_and_verify(task_id: str, steps: int, dataset_path: Path, result: dict[str, Any]) -> None:
    """Record a scripted take, export it, and populate ``result`` from the HDF5."""

    import gymnasium as gym
    import h5py
    import torch
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    import e2e_robot_workflows.tasks.static_apple  # noqa: F401 (registers the task)
    from e2e_robot_workflows.data.recorders import WorkflowRecorderManagerCfg

    cfg = parse_env_cfg(task_id, device="cuda:0", num_envs=1)
    cfg.env_name = task_id.split(":")[-1]
    cfg.terminations.time_out = None
    if hasattr(cfg.terminations, "success"):
        cfg.terminations.success = None
    cfg.observations.policy.concatenate_terms = False
    cfg.recorders = WorkflowRecorderManagerCfg()
    cfg.recorders.dataset_export_dir_path = str(dataset_path.parent)
    cfg.recorders.dataset_filename = dataset_path.name
    cfg.recorders.export_in_close = False
    cfg.recorders.export_in_record_pre_reset = False

    env = gym.make(task_id, cfg=cfg).unwrapped
    try:
        env.reset()
        upper_term = env.action_manager.get_term("upper_body_ik")
        upper_dim = upper_term.action_dim
        hand_dim = upper_term.hand_joint_dim
        hand_joint_ids = upper_term._hand_joint_ids
        total_dim = env.action_manager.total_action_dim
        robot = env.scene["robot"]
        names = robot.body_names
        left_wrist = names.index("left_wrist_yaw_link")
        right_wrist = names.index("right_wrist_yaw_link")
        # Hand joints are held at their default (reset) pose. The Pink IK action's
        # last ``hand_dim`` entries are absolute hand-joint targets, so leaving them
        # zero would drive every finger to 0 rad instead of holding.
        default_hand_pos = robot.data.default_joint_pos.torch[0, hand_joint_ids].clone()

        def hold_action() -> torch.Tensor:
            # Hold both wrists at their current world poses and the fingers at their
            # default pose; the lower-body slice (agile mode only) holds zero
            # velocity at the standing hip height.
            action = torch.zeros((env.num_envs, total_dim), device=env.device)
            body_state = robot.data.body_link_state_w.torch
            action[0, 0:7] = body_state[0, left_wrist, :7]
            action[0, 7:14] = body_state[0, right_wrist, :7]
            action[0, upper_dim - hand_dim : upper_dim] = default_hand_pos
            if total_dim > upper_dim:
                # AGILE lower-body command is [vx, vy, wz, hip_height]; fail loudly
                # if the action layout ever changes rather than mis-slicing.
                assert total_dim - upper_dim == 4, f"expected 4-D AGILE command, got {total_dim - upper_dim}"
                action[0, upper_dim:] = torch.tensor([0.0, 0.0, 0.0, 0.72], device=env.device)
            return action

        logger.info(f"recording a {steps}-step scripted take -> {dataset_path}")
        with torch.inference_mode():
            for _ in range(steps):
                env.step(hold_action())

        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        env.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], dtype=torch.bool, device=env.device))
        env.recorder_manager.export_episodes([0])
        result["action_dim"] = int(total_dim)
        result["state_replay"] = _verify_state_replay(env)
    finally:
        env.close()

    problems: list[str] = []
    with h5py.File(dataset_path, "r") as f:
        demos = list(f["data"].keys()) if "data" in f else []
        result["episodes"] = len(demos)
        if not demos:
            problems.append("no episodes")
        else:
            demo = f["data"][demos[0]]
            result["episode_groups"] = list(demo.keys())
            if "actions" in demo:
                result["recorded_steps"] = int(demo["actions"].shape[0])
                if demo["actions"].shape[0] == 0:
                    problems.append("actions has 0 steps")
            else:
                problems.append("no actions group")
            # Combined Pink IK + AGILE joint command (the joint-space action target).
            if "joint_pos_target" in demo:
                jpt = demo["joint_pos_target"]
                result["joint_pos_target_shape"] = list(jpt.shape)
                if "actions" in demo and jpt.shape[0] != demo["actions"].shape[0]:
                    problems.append("joint_pos_target step count != actions")
                # Width contract: the joint-space action must share the recorded
                # state's 43-DoF articulation joint layout (same order by
                # construction -- both are read from the articulation).
                state = demo["obs"].get(STATE_OBS_KEY) if "obs" in demo else None
                if state is not None and jpt.ndim == 2 and jpt.shape[1] != state.shape[1]:
                    problems.append(
                        f"joint_pos_target width {jpt.shape[1]} != state '{STATE_OBS_KEY}' width {state.shape[1]}"
                    )
            else:
                problems.append("no joint_pos_target group")
            if "obs" not in demo:
                problems.append("obs group missing from episode")
            obs_keys = list(demo["obs"].keys()) if "obs" in demo else []
            if CAMERA_OBS_KEY not in obs_keys:
                problems.append(f"camera obs '{CAMERA_OBS_KEY}' missing (have {obs_keys})")
            else:
                result["camera_shape"] = list(demo["obs"][CAMERA_OBS_KEY].shape)
                if demo["obs"][CAMERA_OBS_KEY].shape[0] == 0:
                    problems.append(f"camera obs '{CAMERA_OBS_KEY}' has 0 frames")

    result["problems"] = problems
    result["status"] = "ok" if not problems else "error"


def main() -> int:
    args = _parse_args()
    result: dict[str, Any] = {"status": "error", "task_id": args.task_id, "steps": args.steps}

    simulation_app = None
    dataset_path = (
        Path(args.dataset_file) if args.dataset_file else Path(tempfile.gettempdir()) / "e2e_recorder_smoke.hdf5"
    )
    dataset_path.unlink(missing_ok=True)
    try:
        from e2e_robot_workflows.compat import warmup_pinocchio_bindings

        warmup_pinocchio_bindings()

        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(headless=True, enable_cameras=True).app
        _record_and_verify(args.task_id, args.steps, dataset_path, result)
    except Exception as exc:  # noqa: BLE001 - report any failure through the result file
        result["status"] = "error"
        result["failure"] = repr(exc)
        logger.exception("recorder smoke failed")
    finally:
        _write_result(args.result_path, result)
        if simulation_app is not None:
            simulation_app.close()

    if result["status"] == "ok":
        logger.info(f"RECORDER SMOKE PASS: {result.get('recorded_steps')} steps, camera {result.get('camera_shape')}")
        return 0
    logger.error(f"RECORDER SMOKE FAIL: {result.get('problems') or result.get('failure')}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

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

"""Consumer-owned AutoData adapters for the canonical Arena environment.

AutoData stays unmodified. These factories delay its imports until the Arena/AutoData
interpreter is active and delegate pose/action semantics to Arena without quaternion reordering.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from e2e_robot_workflows.controllers.errors import H2MeasuredJointLimitViolation
from e2e_robot_workflows.data.completion import qualifies_success_hold


def run_generation_with_recovery(*, env, event_loop, env_loop, worker_factory, policy, stats, result, write_result):
    """Retry measured H2 faults only; incomplete recorder streams are never exported."""

    while True:
        reset_queue = asyncio.Queue()
        action_queue = asyncio.Queue()
        gathered = asyncio.gather(event_loop.create_task(worker_factory(reset_queue, action_queue)))
        fault = None
        try:
            env_loop(
                env,
                reset_queue,
                action_queue,
                event_loop,
                generation_policy_params=policy,
                stats=stats,
                data_gen_tasks=gathered,
            )
        except BaseException as error:
            fault = error
        finally:
            gathered.cancel()
            try:
                event_loop.run_until_complete(gathered)
            except asyncio.CancelledError:
                pass
            except BaseException as error:
                if fault is None:
                    raise
                if error is not fault:
                    message = f"generator_task: {type(error).__name__}: {error}"
                    result.setdefault("cleanup_errors", []).append(message)
                    fault.add_note(message)
        if fault is not None and (not isinstance(fault, H2MeasuredJointLimitViolation) or result.get("cleanup_errors")):
            raise fault
        if fault is None:
            return
        if stats["num_attempts_started"] != stats["num_attempts_completed"] + 1:
            raise RuntimeError("H2 fault recovery requires exactly one active attempt") from fault
        # process_action can fail after pre-step recording but before post-step states.
        # Reset only the in-memory recorder; the next worker requests a full env reset.
        env.recorder_manager.reset([0])
        stats["num_aborted"] += 1
        stats["num_attempts_completed"] += 1
        stats["num_attempts"] += 1
        result.setdefault("aborted_attempts", []).append(
            {"attempt": stats["num_attempts_started"], "reason": str(fault)}
        )
        result.update(stats)
        write_result(result)
        print(f"Rejected H2 attempt {stats['num_attempts_started']}; resetting: {fault}", flush=True)


async def qualify_and_export_final_state(
    *,
    env: Any,
    env_id: int,
    success_term: Any,
    action_queue: asyncio.Queue,
    hold_frames: int,
    max_settle_frames: int,
) -> bool:
    """Hold the final action until success is stable or the settle budget expires."""

    import torch

    if hold_frames < 0:
        raise ValueError("hold_frames must be non-negative")
    if max_settle_frames < hold_frames:
        raise ValueError("max_settle_frames must cover the complete success hold")

    env_id_tensor = torch.tensor([env_id], dtype=torch.int64, device=env.device)
    hold_action = env.action_manager.action[env_id].detach().clone()
    success_observations: list[bool] = []
    stable_success = False
    window_size = hold_frames + 1
    for settle_frame in range(max_settle_frames + 1):
        success_observations.append(bool(success_term.func(env, **success_term.params)[env_id]))
        if len(success_observations) >= window_size and qualifies_success_hold(
            success_observations[-window_size:], hold_frames=hold_frames
        ):
            stable_success = True
            break
        if settle_frame == max_settle_frames:
            break
        await action_queue.put((env_id, hold_action))
        await action_queue.join()

    success_tensor = torch.tensor([[stable_success]], dtype=torch.bool, device=env.device)
    env.recorder_manager.set_success_to_episodes(env_id_tensor, success_tensor)
    env.recorder_manager.export_episodes(env_id_tensor)
    return stable_success


def validate_generated_episode_counts(
    output_path: str | Path,
    *,
    successful_expected: int,
    failed_expected: int,
) -> None:
    """Require recorder exports to agree with the accepted/rejected attempt counts."""

    import h5py
    import numpy as np

    path = Path(output_path)
    failed_path = path.with_name(f"{path.stem}_failed{path.suffix}")
    for label, artifact, expected, success in (
        ("successful", path, successful_expected, True),
        ("failed", failed_path, failed_expected, False),
    ):
        if expected == 0 and not artifact.is_file():
            continue
        if not artifact.is_file():
            raise RuntimeError(f"missing {label} generation artifact: {artifact}")
        with h5py.File(artifact, "r") as dataset:
            episodes = dataset.get("data")
            if not isinstance(episodes, h5py.Group):
                raise RuntimeError(f"{label} artifact does not contain a valid /data group: {artifact}")
            actual = len(episodes)
            if actual != expected:
                raise RuntimeError(f"{label} artifact contains {actual} episodes, expected {expected}")
            for episode_name, episode in episodes.items():
                episode_success = episode.attrs.get("success")
                if not isinstance(episode_success, bool | np.bool_) or bool(episode_success) is not success:
                    raise RuntimeError(
                        f"{label} artifact episode {episode_name!r} has success={bool(episode_success)!r}"
                    )


def finalize_arena_generation(
    *,
    event_loop: asyncio.AbstractEventLoop | None,
    environment: Any | None,
    result: dict[str, Any],
    stats: dict[str, int],
    exit_code: int,
    record_artifacts: Callable[[dict[str, Any]], None],
    write_result: Callable[[dict[str, Any]], None],
) -> int:
    """Finish every generation resource even when a worker has already failed."""

    cleanup_errors = list(result.get("cleanup_errors", []))
    if environment is not None:
        try:
            environment.close()
        except Exception as error:  # noqa: BLE001 - continue closing the remaining resources
            cleanup_errors.append(f"environment: {type(error).__name__}: {error}")

    if event_loop is not None:
        try:
            event_loop.close()
        except Exception as error:  # noqa: BLE001 - the result still needs to be written
            cleanup_errors.append(f"event_loop: {type(error).__name__}: {error}")
        finally:
            asyncio.set_event_loop(None)

    result.update(stats)
    try:
        record_artifacts(result)
    except Exception as error:  # noqa: BLE001 - preserve a structured result without artifact hashes
        cleanup_errors.append(f"artifacts: {type(error).__name__}: {error}")

    if cleanup_errors:
        result["cleanup_errors"] = cleanup_errors
        result["status"] = "error"
        result.setdefault("failure", cleanup_errors[0])
        exit_code = 2
    write_result(result)
    return exit_code


def make_arena_embodiment_adapter() -> Any:
    """Build an AutoData embodiment adapter backed by Arena's public mimic API."""

    import torch
    from isaac_autodata_interfaces.embodiments.embodiment_adapter import EmbodimentAdapter

    class ArenaEmbodimentAdapter(EmbodimentAdapter):
        robot_asset_name = "robot"

        def get_eef_names(self) -> tuple[str, ...]:
            return ("left", "right")

        def get_eef_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
            assert self.env is not None, "Call bind_env(env) before reading state."
            return {name: self.env.get_robot_eef_pose(name, env_ids=env_ids) for name in self.get_eef_names()}

        def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
            assert self.env is not None, "Call bind_env(env) before converting actions."
            poses = self.env.action_to_target_eef_pose(action)
            return {name: poses[name] for name in self.get_eef_names()}

        def target_eef_pose_to_action(
            self,
            target_eef_pose_dict: dict[str, torch.Tensor],
            passthrough_action_dict: dict[str, torch.Tensor],
            action_noise_dict: dict[str, float] | None = None,
            env_id: int = 0,
        ) -> torch.Tensor:
            assert self.env is not None, "Call bind_env(env) before converting poses."
            return self.env.target_eef_pose_to_action(
                target_eef_pose_dict,
                passthrough_action_dict,
                action_noise_dict=action_noise_dict,
                env_id=env_id,
            )

        def actions_to_passthrough_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
            assert self.env is not None, "Call bind_env(env) before converting actions."
            return self.env.actions_to_gripper_actions(actions)

    return ArenaEmbodimentAdapter()


def make_arena_datastream(
    *,
    env: Any,
    task_descriptor: Any,
    source_dataset_path: str | None = None,
    source_pool: Any | None = None,
    uses_start_signals: bool = False,
) -> Any:
    """Compose AutoData's datastream while retaining Arena's pelvis-frame object poses."""

    import torch
    from isaac_autodata_interfaces.datastream import Datastream

    class ArenaDatastream(Datastream):
        def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
            return self.env.get_object_poses(env_ids=env_ids)

    adapter = make_arena_embodiment_adapter()
    if source_dataset_path is None and source_pool is None:
        from isaac_autodata_core import DataGenInfoPool

        source_pool = DataGenInfoPool(
            task_descriptor=task_descriptor,
            embodiment_adapter=adapter,
            device=env.device,
            uses_start_signals=uses_start_signals,
        )
    return ArenaDatastream(
        env=env,
        task_descriptor=task_descriptor,
        embodiment_adapter=adapter,
        source_dataset_path=source_dataset_path,
        source_pool=source_pool,
        uses_start_signals=uses_start_signals,
    )


__all__ = [
    "finalize_arena_generation",
    "make_arena_datastream",
    "make_arena_embodiment_adapter",
    "qualify_and_export_final_state",
    "validate_generated_episode_counts",
]

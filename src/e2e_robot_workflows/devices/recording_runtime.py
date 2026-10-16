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

"""Runtime requirements for physical teleoperation recording."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from e2e_robot_workflows.controllers.errors import H2MeasuredJointLimitViolation
from e2e_robot_workflows.data.artifact_paths import atomic_write_text
from e2e_robot_workflows.data.completion import EpisodeOutcome

_XR_TIMING_PHASES = (
    "teleop_advance_active",
    "teleop_advance_idle",
    "action_validity",
    "env_step",
    "completion_policy",
    "active_cycle",
)
_XR_COUNTERS = (
    "loop_iterations",
    "action_none_active",
    "action_none_idle",
    "valid_actions",
    "nonfinite_actions",
    "active_env_steps",
    "completion_calls",
    "active_render_calls",
    "idle_renders",
    "exports",
    "episode_resets",
    "tracking_loss_resets",
    "joint_limit_resets",
)


class RecordingControls:
    """Recorder state shared between teleop callbacks and the main loop."""

    def __init__(self, running: bool) -> None:
        self.running = running
        self.completed = False
        self.should_export = False
        self.should_reset = False
        self.pending_success = True
        self.auto_reset = False
        self.waiting_for_tracking = False
        self.tracking_reset_acknowledged = False
        self.reset_observed_this_advance = False
        self.success_hold_remaining: int | None = None
        self.pending_outcome: EpisodeOutcome | None = None
        self.auto_outcome_counts: Counter[str] = Counter()
        self.auto_success_count = 0
        self.admission_rejections = 0
        self.admitted_manipulation_hand: str | None = None

    def begin_advance(self) -> None:
        """Start one device sample and clear its transient RESET observation."""

        self.reset_observed_this_advance = False

    def finish(self, outcome: EpisodeOutcome, success: bool) -> None:
        """Queue an auto-detected episode outcome for export and reset."""

        if self.completed:
            return
        self.pending_outcome = outcome
        self.should_export = True
        self.pending_success = success
        self.auto_reset = True

    def request_export(self) -> None:
        if self.waiting_for_tracking:
            self.waiting_for_tracking = False
            self.tracking_reset_acknowledged = False
            logger.info("Tracking recovery stopped. Press START when ready.")
            return
        if not self.running:
            return
        self.running = False
        self.should_export = True
        logger.info("Finish requested: current episode will be exported.")

    def request_reset(self) -> None:
        if not self.running:
            return
        self.running = False
        self.should_reset = True
        logger.info("Reset requested: current episode will be skipped.")

    def discard_episode(self) -> None:
        """Reset without exporting, including when success follow-through was pending."""

        self.should_export = False
        self.pending_success = False
        self.should_reset = True
        self.auto_reset = True
        self.success_hold_remaining = None
        self.pending_outcome = None

    def recover_tracking(self) -> bool:
        """Discard once and wait for a reset-acknowledged valid device sample."""

        if not self.running:
            return False
        self.running = False
        self.should_reset = True
        self.waiting_for_tracking = True
        # IsaacTeleop dispatches RESET before returning the action. Preserve a
        # RESET observed in the same sample that first reports tracking loss.
        self.tracking_reset_acknowledged = self.reset_observed_this_advance
        logger.warning("Teleoperation tracking was lost; discarding the current episode.")
        return True

    def observe_device_reset(self) -> None:
        """Observe device RESET without treating it as an episode boundary."""

        if self.completed:
            return
        self.reset_observed_this_advance = True
        if self.waiting_for_tracking:
            self.tracking_reset_acknowledged = True

    def finish_tracking_recovery(self, *, action_is_finite: bool) -> bool:
        """Resume after device reset and one finite sample, discarding that sample."""

        if not (action_is_finite and self.waiting_for_tracking and self.tracking_reset_acknowledged):
            return False
        self.waiting_for_tracking = False
        self.tracking_reset_acknowledged = False
        return True

    def pause(self) -> None:
        self.running = False
        logger.info("Recording paused.")

    def resume(self) -> None:
        if self.completed:
            return
        self.waiting_for_tracking = False
        self.tracking_reset_acknowledged = False
        self.running = True
        logger.info("Recording resumed.")

    def complete(self) -> None:
        """End recording permanently while final reset and XR rendering finish."""

        self.completed = True
        self.running = False
        self.should_export = False
        self.should_reset = False
        self.auto_reset = False
        self.waiting_for_tracking = False
        self.tracking_reset_acknowledged = False
        self.reset_observed_this_advance = False
        self.success_hold_remaining = None
        self.pending_outcome = None


@contextmanager
def xr_recording_completion_notice(recorded_count: int) -> Iterator[None]:
    """Keep a completion card in the headset's rendered scene until context exit."""

    import omni.kit.app
    from omni.kit.xr.core import XRCore, XRTransformType

    app = omni.kit.app.get_app()
    xr_core = XRCore.get_singleton()
    layer = None
    container = None
    original_error = None
    try:
        if not app.get_extension_manager().set_extension_enabled_immediate("omni.kit.scene_view.xr_utils", True):
            raise RuntimeError("Recording saved, but the XR completion display could not be enabled.")

        import omni.ui as ui
        from omni.kit.scene_view.xr import XRSceneView
        from omni.kit.scene_view.xr_utils import UiContainer, WidgetComponent
        from omni.kit.scene_view.xr_utils.manipulator_components.widget_component import UpdatePolicy
        from omni.kit.scene_view.xr_utils.spatial_source import SpatialSource
        from pxr import Gf

        display = xr_core.get_input_device("displayDevice")
        if display is None:
            raise RuntimeError("Recording saved, but no XR display is available for the completion notice.")

        coordinate_system = xr_core.get_coordinate_system()
        meters_per_unit = coordinate_system.meters_per_unit
        layer = xr_core.create_xr_usd_layer("/_xr/recording_completion")
        # XR's UI layer handles the stage's up-axis conversion, as it does for
        # the headset menu. Place the card in front of the current display pose.
        forward = Gf.Vec3d(*layer.get_coordinate_system().get_forward_vector()) * (1.4 / meters_per_unit)
        pose = xr_core.reorient_transform_matrix_up_right(
            display.get_virtual_world_pose(), coordinate_system.up_axis == "y"
        )
        path = layer.get_top_level_prim_path() + "/notice"
        layer.add_transform(
            path=path,
            group="completion",
            transform=Gf.Matrix4d().SetTranslate(forward) * pose,
            transform_type=XRTransformType.stage,
        )

        def build_card() -> None:
            with ui.ZStack():
                ui.Rectangle(style={"background_color": 0xFF202020, "border_radius": 16})
                with ui.VStack(spacing=10):
                    ui.Spacer(height=28)
                    ui.Label(
                        "Recording complete",
                        alignment=ui.Alignment.CENTER,
                        style={"font_size": 42, "color": 0xFF00B876},
                    )
                    ui.Label(
                        f"{recorded_count} demonstration(s) saved",
                        alignment=ui.Alignment.CENTER,
                        style={"font_size": 30},
                    )
                    ui.Label("Session ending", alignment=ui.Alignment.CENTER, style={"font_size": 26})
                    ui.Spacer(height=28)

        container = UiContainer(
            XRSceneView,
            WidgetComponent(
                ui.Frame,
                width=1.2 / meters_per_unit,
                height=0.45 / meters_per_unit,
                resolution_scale=2,
                unit_to_pixel_scale=600 * meters_per_unit,
                update_policy=UpdatePolicy.ALWAYS,
                widget_kwargs={"build_fn": build_card},
            ),
            space_stack=SpatialSource.new_prim_path_source(path),
        )
        yield
    except BaseException as error:
        original_error = error
        raise
    finally:
        failures: list[BaseException] = []
        try:
            xr_core.request_disable_profile()
            deadline = time.monotonic() + 3.0
            while xr_core.is_xr_enabled() and app.is_running() and time.monotonic() < deadline:
                app.update()
            if xr_core.is_xr_enabled():
                raise RuntimeError("Recording saved, but the XR session did not stop after completion.")
        except BaseException as error:
            failures.append(error)
        try:
            if container is not None:
                container.root.clear()
        except BaseException as error:
            failures.append(error)
        try:
            if layer is not None and layer.is_valid():
                layer.invalidate()
        except BaseException as error:
            failures.append(error)
        if failures:
            primary = original_error if original_error is not None else failures.pop(0)
            for error in failures:
                note = f"XR completion cleanup also failed: {type(error).__name__}: {error}"
                primary.add_note(note)
                logger.error(note)
            if original_error is None:
                raise primary


def step_recording_environment(
    step: Callable[[], object], controls: RecordingControls, *, recover_h2_joint_limits: bool
) -> bool:
    """Discard a live H2 episode on a measured limit fault, never an invalid IK target."""

    try:
        step()
    except H2MeasuredJointLimitViolation as error:
        if not recover_h2_joint_limits:
            raise
        # Nothing from this episode is exported, even if success follow-through
        # was pending. Reset owns recovery; the controller's limits stay strict.
        controls.discard_episode()
        logger.warning("{}; discarding this episode and resetting. Saved episodes are retained.", error)
        return False
    return True


def align_h2_recording_action(action: Any, env: Any, alignment: Any) -> Any:
    """Capture live flange poses once, then record the aligned and bounded command."""

    capturing = not alignment.is_aligned
    reference = None
    if capturing:
        from e2e_robot_workflows.arena.h2_actions import h2_pink_hold_action

        reference = h2_pink_hold_action(env)[0]
    aligned = alignment.apply(action, reference_action=reference, dt=float(env.step_dt))
    if capturing:
        logger.info("H2 controller poses aligned to the current hands; move from this starting pose.")
    return aligned


def admit_h2_recording_episode(
    episode_data: Mapping[str, Any], controls: RecordingControls, *, manipulation_hand: str
) -> bool:
    """Validate only the small boundary arrays before a retained live take is exported."""

    import numpy as np

    from e2e_robot_workflows.data.dataset_manifest import (
        ArenaEpisodeRejection,
        DatasetContractError,
        validate_arena_episode_boundaries,
    )

    try:
        steps = len(episode_data["actions"])
        datagen = episode_data["obs"]["datagen_info"]
        arrays = {}
        for field in ("subtask_term_signals", "grasp_signals"):
            arrays[field] = {}
            for hand in ("left", "right"):
                values = datagen[field][f"{hand}_hand_holds_object"]
                if isinstance(values, list | tuple) and values and hasattr(values[0], "detach"):
                    import torch

                    values = torch.stack(values).detach().cpu().numpy()
                elif hasattr(values, "detach"):
                    values = values.detach().cpu().numpy()
                arrays[field][hand] = np.asarray(values)
    except (KeyError, TypeError) as error:
        raise DatasetContractError("Live H2 episode is missing its action/boundary recording fields") from error
    selected_hand = controls.admitted_manipulation_hand if manipulation_hand == "auto" else manipulation_hand
    try:
        detected_hand = validate_arena_episode_boundaries(
            arrays["subtask_term_signals"],
            steps=steps,
            manipulation_hand=selected_hand,
            require_handover=True,
            grasp_signals=arrays["grasp_signals"],
        )
    except ArenaEpisodeRejection as error:
        controls.discard_episode()
        controls.auto_reset = controls.running
        controls.admission_rejections += 1
        logger.warning("Demo rejected: {}; resetting without export.", error)
        return False
    controls.admitted_manipulation_hand = detected_hand
    return True


def close_recording_dataset(
    exported_count: Callable[[], int],
    close: Callable[[], None],
    finalize: Callable[[], None],
    *,
    original_error: BaseException | None = None,
) -> int:
    """Close the writer before stamping saved episodes, preserving an earlier failure."""

    count = 0
    failures: list[BaseException] = []
    try:
        count = exported_count()
    except BaseException as error:
        failures.append(error)
    try:
        close()
    except BaseException as error:
        failures.append(error)
    else:
        if count:
            try:
                finalize()
            except BaseException as error:
                failures.append(error)
    if failures:
        primary = original_error if original_error is not None else failures.pop(0)
        for error in failures:
            note = f"Recording cleanup/finalization also failed: {type(error).__name__}: {error}"
            primary.add_note(note)
            logger.error(note)
        if original_error is None:
            raise primary
    return count


@dataclass(slots=True)
class FixedTimingStats:
    """Constant-space host-wall timing totals for one recorder phase."""

    count: int = 0
    total_ns: int = 0
    max_ns: int = 0
    over_10ms_count: int = 0
    over_20ms_count: int = 0
    over_50ms_count: int = 0

    def observe(self, elapsed_ns: int) -> None:
        if elapsed_ns < 0:
            raise ValueError("elapsed_ns must be non-negative")
        self.count += 1
        self.total_ns += elapsed_ns
        self.max_ns = max(self.max_ns, elapsed_ns)
        self.over_10ms_count += elapsed_ns > 10_000_000
        self.over_20ms_count += elapsed_ns > 20_000_000
        self.over_50ms_count += elapsed_ns > 50_000_000

    def as_dict(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "mean_ms": self.total_ns / self.count / 1_000_000 if self.count else 0.0,
            "max_ms": self.max_ns / 1_000_000,
            "over_10ms_count": self.over_10ms_count,
            "over_20ms_count": self.over_20ms_count,
            "over_50ms_count": self.over_50ms_count,
        }


@dataclass(slots=True)
class XrRuntimeTelemetry:
    """Low-overhead, restart-safe timing report for an interactive XR session."""

    path: Path
    runtime: Mapping[str, Any]
    started_monotonic_ns: int = field(default_factory=time.perf_counter_ns)
    started_unix_s: float = field(default_factory=time.time)
    phases: dict[str, FixedTimingStats] = field(
        default_factory=lambda: {name: FixedTimingStats() for name in _XR_TIMING_PHASES}
    )
    counters: dict[str, int] = field(default_factory=lambda: dict.fromkeys(_XR_COUNTERS, 0))

    def observe(self, phase: str, elapsed_ns: int) -> None:
        self.phases[phase].observe(elapsed_ns)

    def increment(self, counter: str, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("counter increments must be non-negative")
        self.counters[counter] += amount

    def write_running(self) -> None:
        self._write(
            {
                "schema_version": 1,
                "scope": "interactive_recording_loop",
                "status": "running",
                "dataset_finalization_included": False,
                "runtime": dict(self.runtime),
            }
        )

    def final_report(
        self,
        *,
        status: str,
        exit_reason: str,
        recorded_count: int,
        auto_outcome_counts: Mapping[str, int],
        admission_rejections: int = 0,
        error_type: str | None = None,
        ended_monotonic_ns: int | None = None,
    ) -> dict[str, Any]:
        ended_ns = time.perf_counter_ns() if ended_monotonic_ns is None else ended_monotonic_ns
        active_steps = self.counters["active_env_steps"]
        active_wall_s = self.phases["active_cycle"].total_ns / 1_000_000_000
        control_period_s = float(self.runtime["control_period_s"])
        report: dict[str, Any] = {
            "schema_version": 1,
            "scope": "interactive_recording_loop",
            "status": status,
            "exit_reason": exit_reason,
            "dataset_finalization_included": False,
            "started_unix_s": self.started_unix_s,
            "session_wall_s": max(0, ended_ns - self.started_monotonic_ns) / 1_000_000_000,
            "runtime": dict(self.runtime),
            "counts": dict(self.counters),
            "timing_host_wall": {name: stats.as_dict() for name, stats in self.phases.items()},
            "active_simulated_s": active_steps * control_period_s,
            "active_effective_hz": active_steps / active_wall_s if active_wall_s else 0.0,
            "active_realtime_factor": active_steps * control_period_s / active_wall_s if active_wall_s else 0.0,
            "active_render_hz": self.counters["active_render_calls"] / active_wall_s if active_wall_s else 0.0,
            "recorded_count": recorded_count,
            "auto_outcome_counts": dict(auto_outcome_counts),
            "admission_rejections": admission_rejections,
        }
        if error_type is not None:
            report["error_type"] = error_type
        return report

    def write_final(self, **kwargs: Any) -> None:
        self._write(self.final_report(**kwargs))

    def _write(self, report: Mapping[str, Any]) -> None:
        atomic_write_text(self.path, json.dumps(report, indent=2, sort_keys=True) + "\n")


def require_xr_runtime_metrics_mode(*, metrics_enabled: bool, xr_enabled: bool, sample_steps: int) -> None:
    """Keep live XR telemetry out of non-XR and finite sample modes."""

    if not metrics_enabled:
        return
    if sample_steps:
        raise ValueError("--xr_runtime_metrics is for interactive XR recording, not --sample_steps")
    if not xr_enabled:
        raise ValueError("--xr_runtime_metrics requires an XR teleop device")


def require_h2_recording_cameras(*, h2_arena_recording: bool, cameras_enabled: bool) -> None:
    """Reject an H2 recording that cannot satisfy its dataset contract."""

    if h2_arena_recording and not cameras_enabled:
        raise ValueError("H2 Arena recording requires --enable_cameras")


def refresh_reset_view(env: Any) -> None:
    """Refresh a reset viewport without advancing physics or invalidating reset observations."""

    env.sim.render_context.reset_transform_cadence()
    env.sim.render()


def require_arena_teleop_mode(
    *,
    arena_recording: bool,
    h2_arena_recording: bool,
    sample_steps: int,
    teleop_device: str,
) -> None:
    """Reject selectors that cannot match the Arena retargeting graph."""

    if not arena_recording or sample_steps:
        return
    if teleop_device not in {"handtracking", "motion_controllers", "manusvive"}:
        raise ValueError(f"Arena has no retargeter for teleop device {teleop_device!r}")
    if h2_arena_recording and teleop_device != "motion_controllers":
        raise ValueError(
            "H2 Arena recording supports only --teleop_device motion_controllers; "
            "hand-tracking and Manus pipelines are not implemented"
        )


__all__ = [
    "FixedTimingStats",
    "RecordingControls",
    "XrRuntimeTelemetry",
    "admit_h2_recording_episode",
    "align_h2_recording_action",
    "close_recording_dataset",
    "refresh_reset_view",
    "require_arena_teleop_mode",
    "require_h2_recording_cameras",
    "require_xr_runtime_metrics_mode",
    "step_recording_environment",
    "xr_recording_completion_notice",
]

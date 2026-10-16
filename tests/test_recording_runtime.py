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

"""Behavioral checks for physical recorder runtime requirements."""

from __future__ import annotations

import argparse
import ast
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from loguru import logger

from e2e_robot_workflows.controllers.errors import H2MeasuredJointLimitViolation
from e2e_robot_workflows.data.completion import CompletionTransition, advance_success_hold
from e2e_robot_workflows.data.replay import make_state_renderer
from e2e_robot_workflows.devices.recording_runtime import (
    FixedTimingStats,
    RecordingControls,
    XrRuntimeTelemetry,
    admit_h2_recording_episode,
    align_h2_recording_action,
    close_recording_dataset,
    refresh_reset_view,
    require_arena_teleop_mode,
    require_h2_recording_cameras,
    require_xr_runtime_metrics_mode,
    step_recording_environment,
    xr_recording_completion_notice,
)


def test_h2_live_alignment_captures_only_once_and_recaptures_after_reset(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from e2e_robot_workflows.devices.h2_reset_alignment import H2ResetAlignment

    pose_reads = []
    reference = torch.zeros((1, 61))
    reference[:, 6] = reference[:, 13] = 1.0
    reference[:, 2] = reference[:, 9] = 0.183
    module = ModuleType("e2e_robot_workflows.arena.h2_actions")

    def hold_action(env):
        pose_reads.append(env)
        return reference.clone()

    module.h2_pink_hold_action = hold_action
    monkeypatch.setitem(sys.modules, module.__name__, module)
    env = SimpleNamespace(step_dt=0.01)
    alignment = H2ResetAlignment()
    raw = reference[0].clone()
    raw[9] -= 0.245
    raw[14:] = torch.linspace(0.0, 0.2, 47)
    first = align_h2_recording_action(raw, env, alignment)
    torch.testing.assert_close(first[:14], reference[0, :14])
    assert torch.equal(first[14:], raw[14:])
    raw[0] += 0.001
    second = align_h2_recording_action(raw, env, alignment)
    assert len(pose_reads) == 1
    assert second[0].item() == pytest.approx(0.001)
    alignment.reset()
    reference[:, 2] += 0.02
    third = align_h2_recording_action(raw, env, alignment)
    assert len(pose_reads) == 2
    torch.testing.assert_close(third[:14], reference[0, :14])


def _live_episode(left, right):
    return {
        "actions": [None] * len(left),
        "obs": {
            "robot_head_cam_rgb": object(),  # Admission must not touch or stack camera frames.
            "datagen_info": {
                "subtask_term_signals": {
                    "left_hand_holds_object": left,
                    "right_hand_holds_object": right,
                },
                "grasp_signals": {
                    "left_hand_holds_object": left,
                    "right_hand_holds_object": right,
                },
            },
        },
    }


def test_h2_admission_rejects_ambiguous_take_then_accepts_next_handover() -> None:
    controls = RecordingControls(running=True)
    bad = _live_episode([0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1])
    assert not admit_h2_recording_episode(bad, controls, manipulation_hand="left")
    assert controls.should_reset and controls.auto_reset and controls.running
    assert not controls.should_export
    assert controls.admission_rejections == 1
    assert not controls.auto_outcome_counts
    assert controls.auto_success_count == 0
    assert controls.admitted_manipulation_hand is None

    controls.should_reset = False
    controls.auto_reset = False
    good = _live_episode([0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1])
    assert admit_h2_recording_episode(good, controls, manipulation_hand="left")
    controls.finish(SimpleNamespace(value="success"), success=True)
    assert controls.should_export and controls.pending_success
    assert controls.admitted_manipulation_hand == "left"
    assert controls.auto_success_count == 0  # Count only after the queued export succeeds.


def test_h2_admission_latches_auto_hand_across_takes_and_handles_live_tensor_lists() -> None:
    torch = pytest.importorskip("torch")
    controls = RecordingControls(running=True)
    left = [torch.tensor(value) for value in (0, 1, 1, 1, 1, 1)]
    right = [torch.tensor(value) for value in (0, 0, 0, 1, 1, 1)]
    assert admit_h2_recording_episode(_live_episode(left, right), controls, manipulation_hand="auto")
    assert not admit_h2_recording_episode(_live_episode(right, left), controls, manipulation_hand="auto")
    assert controls.admitted_manipulation_hand == "left"


def test_h2_admission_does_not_hide_malformed_recorder_fields() -> None:
    from e2e_robot_workflows.data.dataset_manifest import DatasetContractError

    controls = RecordingControls(running=True)
    with pytest.raises(DatasetContractError):
        admit_h2_recording_episode({}, controls, manipulation_hand="left")
    with pytest.raises(DatasetContractError):
        admit_h2_recording_episode(_live_episode([0, float("nan")], [0, 1]), controls, manipulation_hand="left")
    assert not controls.should_reset


@pytest.fixture
def recorder_export():
    """Exercise the script's export and completion functions without starting Isaac Sim."""

    source = Path(__file__).resolve().parents[1] / "scripts/record_demos.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"finish_recording_episode", "maybe_auto_finish"}
    ]
    exports = []
    namespace = {
        "RecordingControls": RecordingControls,
        "H2_ARENA_RECORDING": True,
        "logger": logger,
        "args_cli": SimpleNamespace(dataset_export_mode="all", manipulation_hand="auto"),
        "admit_h2_recording_episode": admit_h2_recording_episode,
        "CompletionTransition": CompletionTransition,
        "advance_success_hold": advance_success_hold,
        "evaluate_outcome_or_none": lambda env: namespace["outcome"],
        "COMPLETION_POLICY": SimpleNamespace(hold_frames=20),
        "exported_episode_count": lambda env: len(exports),
    }

    def export(env, *, success):
        if success or namespace["args_cli"].dataset_export_mode == "all":
            exports.append(success)

    namespace["export_current_episode"] = export
    future = ast.parse("from __future__ import annotations").body[0]
    exec(compile(ast.Module(body=[future, *functions], type_ignores=[]), str(source), "exec"), namespace)
    return SimpleNamespace(
        finish=namespace["finish_recording_episode"],
        auto_finish=namespace["maybe_auto_finish"],
        namespace=namespace,
        exports=exports,
    )


@pytest.mark.parametrize("mode", ["all", "succeeded"])
@pytest.mark.parametrize("trigger", ["manual", "failure", "success"])
@pytest.mark.parametrize("boundaries", ["valid", "no_pickup", "ambiguous", "no_handover"])
def test_h2_common_export_admits_retained_takes(recorder_export, mode, trigger, boundaries) -> None:
    left, right = {
        "valid": ([0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1]),
        "no_pickup": ([0] * 6, [0] * 6),
        "ambiguous": ([0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]),
        "no_handover": ([0, 1, 1, 1, 1, 1], [0] * 6),
    }[boundaries]
    env = SimpleNamespace(
        episode_length_buf=[6],
        recorder_manager=SimpleNamespace(get_episode=lambda _: SimpleNamespace(data=_live_episode(left, right))),
    )
    recorder_export.namespace["args_cli"].dataset_export_mode = mode
    controls = RecordingControls(running=True)
    if trigger == "manual":
        controls.request_export()
    else:
        recorder_export.namespace["outcome"] = SimpleNamespace(
            value=trigger, is_terminal=True, is_success=trigger == "success"
        )
        recorder_export.auto_finish(env, controls)
        if trigger == "success":
            for _ in range(20):
                assert not controls.should_export
                recorder_export.auto_finish(env, controls)
        assert controls.should_export
    assert not controls.auto_outcome_counts and controls.auto_success_count == 0

    count = recorder_export.finish(env, controls, auto_success_enabled=True)

    retained = trigger == "success" or mode == "all"
    rejected = retained and boundaries != "valid"
    expected_count = int(retained and not rejected)
    assert count == (None if rejected else expected_count)
    assert recorder_export.exports == ([trigger == "success"] if expected_count else [])
    assert controls.should_reset and not controls.should_export
    assert controls.running == (trigger != "manual")
    assert controls.auto_reset == (trigger != "manual")
    assert controls.admitted_manipulation_hand == ("left" if expected_count else None)
    assert controls.auto_success_count == int(trigger == "success" and not rejected)
    expected_outcomes = {trigger: 1} if trigger != "manual" and not rejected else {}
    assert controls.admission_rejections == int(rejected)
    assert controls.auto_outcome_counts == expected_outcomes


def test_h2_zero_frame_manual_stop_resets_without_export(recorder_export) -> None:
    controls = RecordingControls(running=True)
    controls.request_export()
    # No actions or boundary arrays exist before the first environment step.
    env = SimpleNamespace(episode_length_buf=[0])

    assert recorder_export.finish(env, controls, auto_success_enabled=True) is None
    assert not recorder_export.exports
    assert controls.should_reset and not controls.should_export
    assert not controls.running and not controls.auto_reset
    assert not controls.auto_outcome_counts
    assert controls.admitted_manipulation_hand is None


def test_target_completion_resets_the_scene_and_blocks_panel_restart():
    """Pin the terminal lifecycle; XR pixel visibility needs simulator validation."""

    from contextlib import nullcontext

    source = Path(__file__).resolve().parents[1] / "scripts/record_demos.py"
    tree = ast.parse(source.read_text())
    loop = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_recording_loop")
    target = next(
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "args_cli.num_demos > 0 and recorded_count >= args_cli.num_demos"
    )
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"complete_recording_session", "reset_episode"}
    ]
    controls = RecordingControls(running=True)
    controls.should_reset = True
    controls.auto_reset = True
    controls.success_hold_remaining = 0
    controls.pending_outcome = object()
    state = {"object_pose": "plate"}
    frames = ["final_frame"]

    def render():
        # Kit can deliver a queued panel event during reset or final rendering.
        controls.resume()
        controls.request_export()
        controls.observe_device_reset()

    def reset_scene():
        state["object_pose"] = "spawn"
        frames.append("reset_frame")

    env = SimpleNamespace(
        reset=reset_scene,
        sim=SimpleNamespace(
            reset=lambda: None,
            render=render,
            is_rendering=False,
            is_stopped=lambda: False,
            render_context=SimpleNamespace(reset_transform_cadence=lambda: None),
        ),
        recorder_manager=SimpleNamespace(reset=frames.clear),
    )
    clock = iter([0.0, 0.0, 6.0])
    namespace = {
        "env": env,
        "controls": controls,
        "recorded_count": 3,
        "args_cli": SimpleNamespace(num_demos=3, xr=True),
        "H2_ARENA_RECORDING": False,
        "logger": logger,
        "refresh_reset_view": refresh_reset_view,
        "make_state_renderer": make_state_renderer,
        "xr_recording_completion_notice": lambda count: nullcontext(),
        "simulation_app": SimpleNamespace(is_running=lambda: True),
        "time": SimpleNamespace(monotonic=lambda: next(clock)),
    }
    future = ast.parse("from __future__ import annotations").body[0]
    body = ast.While(test=ast.Constant(True), body=[target, ast.Break()], orelse=[])
    code = ast.fix_missing_locations(ast.Module(body=[future, *helpers, body], type_ignores=[]))
    exec(compile(code, str(source), "exec"), namespace)

    assert state["object_pose"] == "spawn"
    assert not frames
    assert controls.completed and not controls.running
    assert not controls.should_export and not controls.should_reset and not controls.auto_reset
    assert controls.pending_outcome is None and controls.success_hold_remaining is None
    assert not controls.reset_observed_this_advance
    assert namespace["exit_reason"] == "target_reached"


@pytest.mark.parametrize("failure", [None, "extension", "display", "body", "shutdown", "cleanup"])
def test_xr_completion_cleans_up_failed_notices_without_masking_the_error(monkeypatch, failure):
    """Model the Kit hardware boundary; rendered headset visibility is checked in simulation."""

    from unittest.mock import MagicMock

    omni = MagicMock()
    monkeypatch.setitem(sys.modules, "omni", omni)
    for path in (
        "kit.app",
        "kit.xr.core",
        "ui",
        "kit.scene_view.xr",
        "kit.scene_view.xr_utils.manipulator_components.widget_component",
        "kit.scene_view.xr_utils.spatial_source",
    ):
        module, name = omni, "omni"
        for part in path.split("."):
            module, name = getattr(module, part), f"{name}.{part}"
            monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "pxr", MagicMock())
    app = omni.kit.app.get_app.return_value
    core = omni.kit.xr.core.XRCore.get_singleton.return_value
    layer = core.create_xr_usd_layer.return_value
    container = omni.kit.scene_view.xr_utils.UiContainer.return_value
    resources = SimpleNamespace(profile_enabled=True, layer_live=False, container_live=False)
    original_error = ValueError("render failed")
    shutdown_error = RuntimeError("profile shutdown failed")

    def create_layer(*_args):
        resources.layer_live = True
        return layer

    def create_container(*_args, **_kwargs):
        resources.container_live = True
        return container

    def disable_profile():
        if failure in {"shutdown", "cleanup"}:
            raise shutdown_error
        resources.profile_enabled = False

    def clear_container():
        if failure == "cleanup":
            raise RuntimeError("container cleanup failed")
        resources.container_live = False

    app.get_extension_manager.return_value.set_extension_enabled_immediate.return_value = failure != "extension"
    if failure == "display":
        core.get_input_device.return_value = None
    core.get_coordinate_system.return_value.meters_per_unit = 1.0
    core.create_xr_usd_layer.side_effect = create_layer
    omni.kit.scene_view.xr_utils.UiContainer.side_effect = create_container
    core.request_disable_profile.side_effect = disable_profile
    core.is_xr_enabled.side_effect = lambda: resources.profile_enabled
    container.root.clear.side_effect = clear_container
    layer.is_valid.side_effect = lambda: resources.layer_live
    layer.invalidate.side_effect = lambda: setattr(resources, "layer_live", False)

    caught = None
    try:
        with xr_recording_completion_notice(3):
            assert resources.profile_enabled and resources.layer_live and resources.container_live
            if failure in {"body", "cleanup"}:
                raise original_error
    except Exception as error:
        caught = error

    assert resources.profile_enabled is (failure in {"shutdown", "cleanup"})
    assert not resources.layer_live
    assert resources.container_live is (failure == "cleanup")
    if failure in {"body", "cleanup"}:
        assert caught is original_error
        if failure == "cleanup":
            assert any("profile shutdown failed" in note for note in caught.__notes__)
            assert any("container cleanup failed" in note for note in caught.__notes__)
    elif failure == "shutdown":
        assert caught is shutdown_error
    elif failure is not None:
        assert isinstance(caught, RuntimeError)
    else:
        assert caught is None


def test_h2_no_auto_success_is_rejected_before_simulator_start(monkeypatch, capsys) -> None:
    isaaclab = ModuleType("isaaclab")
    isaaclab.__path__ = []
    app = ModuleType("isaaclab.app")

    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser: argparse.ArgumentParser) -> None:
            parser.add_argument("--enable_cameras", action="store_true")
            parser.add_argument("--xr", action="store_true")

        def __init__(self, _args: object) -> None:
            raise AssertionError("H2 manual-success override must fail before simulator startup")

    app.AppLauncher = AppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", app)
    monkeypatch.setattr(
        sys,
        "argv",
        ["record_demos.py", "--arena", "--arena_embodiment", "h2_agile_pink", "--enable_cameras", "--no-auto_success"],
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/record_demos.py"), run_name="__main__")
    assert error.value.code == 2
    assert "H2 qualification requires measured auto-success" in capsys.readouterr().err


@pytest.mark.parametrize("has_original_error", (False, True))
def test_saved_recordings_are_finalized_after_writer_close_even_when_loop_failed(has_original_error: bool) -> None:
    calls = []
    original = RuntimeError("recording fault") if has_original_error else None
    count = close_recording_dataset(
        lambda: 1, lambda: calls.append("close"), lambda: calls.append("finalize"), original_error=original
    )
    assert count == 1
    assert calls == ["close", "finalize"]


def test_empty_recording_is_closed_but_not_stamped() -> None:
    calls = []
    assert close_recording_dataset(lambda: 0, lambda: calls.append("close"), lambda: calls.append("finalize")) == 0
    assert calls == ["close"]


def test_finalization_failure_remains_fatal_on_clean_loop_exit() -> None:
    def fail():
        raise ValueError("invalid saved boundary")

    with pytest.raises(ValueError, match="invalid saved boundary"):
        close_recording_dataset(lambda: 1, lambda: None, fail)


def test_secondary_finalization_failure_does_not_mask_original_loop_fault() -> None:
    original = RuntimeError("original IK fault")

    def fail():
        raise ValueError("invalid saved boundary")

    assert close_recording_dataset(lambda: 1, lambda: None, fail, original_error=original) == 1
    assert any("invalid saved boundary" in note for note in original.__notes__)


def test_failed_writer_close_prevents_reopening_dataset_for_stamping() -> None:
    calls = []

    def fail_close():
        raise RuntimeError("writer did not close")

    with pytest.raises(RuntimeError, match="writer did not close"):
        close_recording_dataset(lambda: 1, fail_close, lambda: calls.append("finalize"))
    assert not calls


def test_live_h2_limit_recovery_discards_pending_success_without_export() -> None:
    controls = RecordingControls(running=True)
    controls.should_export = True
    controls.pending_success = True
    controls.success_hold_remaining = 3
    controls.pending_outcome = object()

    def bad_step() -> None:
        raise H2MeasuredJointLimitViolation("left_wrist_yaw_joint excursion=6.86e-05 rad")

    assert not step_recording_environment(bad_step, controls, recover_h2_joint_limits=True)
    assert controls.running
    assert controls.should_reset and controls.auto_reset
    assert not controls.should_export and not controls.pending_success
    assert controls.success_hold_remaining is None
    assert controls.pending_outcome is None
    assert controls.auto_success_count == 0


def test_successful_recording_step_does_not_request_recovery() -> None:
    controls = RecordingControls(running=True)
    calls = []
    assert step_recording_environment(lambda: calls.append("step"), controls, recover_h2_joint_limits=True)
    assert calls == ["step"]
    assert not controls.should_reset and not controls.should_export


@pytest.mark.parametrize("error", [RuntimeError("invalid IK target"), ValueError("non-finite input")])
def test_live_h2_recovery_never_swallows_other_errors(error: Exception) -> None:
    controls = RecordingControls(running=True)

    def bad_step() -> None:
        raise error

    with pytest.raises(type(error), match=str(error)):
        step_recording_environment(bad_step, controls, recover_h2_joint_limits=True)
    assert not controls.should_reset and not controls.should_export


def test_joint_limits_remain_fatal_outside_live_h2_recording() -> None:
    controls = RecordingControls(running=True)

    def bad_step() -> None:
        raise H2MeasuredJointLimitViolation("measured state")

    with pytest.raises(H2MeasuredJointLimitViolation, match="measured state"):
        step_recording_environment(bad_step, controls, recover_h2_joint_limits=False)
    assert not controls.should_reset


def test_xr_reset_alone_never_finishes_an_episode() -> None:
    controls = RecordingControls(running=True)

    controls.begin_advance()
    controls.observe_device_reset()

    assert controls.running
    assert not controls.should_export
    assert not controls.should_reset


def test_tracking_recovery_preserves_reset_from_same_device_sample() -> None:
    controls = RecordingControls(running=True)

    controls.begin_advance()
    controls.observe_device_reset()
    assert controls.recover_tracking()

    assert controls.should_reset
    assert controls.finish_tracking_recovery(action_is_finite=True)
    assert not controls.waiting_for_tracking


def test_tracking_recovery_waits_for_later_reset_and_finite_sample() -> None:
    controls = RecordingControls(running=True)

    controls.begin_advance()
    assert controls.recover_tracking()
    assert not controls.finish_tracking_recovery(action_is_finite=True)

    controls.begin_advance()
    controls.observe_device_reset()
    assert not controls.finish_tracking_recovery(action_is_finite=False)
    controls.begin_advance()
    assert controls.finish_tracking_recovery(action_is_finite=True)


def test_tracking_recovery_ignores_an_idle_session() -> None:
    controls = RecordingControls(running=False)

    assert not controls.recover_tracking()
    assert not controls.should_reset
    assert not controls.waiting_for_tracking


def test_stop_during_tracking_recovery_pauses_without_exporting() -> None:
    controls = RecordingControls(running=True)
    controls.begin_advance()
    assert controls.recover_tracking()

    controls.request_export()

    assert not controls.running
    assert not controls.waiting_for_tracking
    assert not controls.should_export


def test_fixed_timing_stats_tracks_thresholds_without_samples() -> None:
    stats = FixedTimingStats()
    for elapsed_ms in (1, 11, 21, 51):
        stats.observe(elapsed_ms * 1_000_000)

    assert stats.as_dict() == {
        "count": 4,
        "mean_ms": 21.0,
        "max_ms": 51.0,
        "over_10ms_count": 3,
        "over_20ms_count": 2,
        "over_50ms_count": 1,
    }

    with pytest.raises(ValueError, match="non-negative"):
        stats.observe(-1)


def test_xr_runtime_telemetry_atomically_replaces_running_report(tmp_path) -> None:
    result = tmp_path / "xr-runtime.json"
    telemetry = XrRuntimeTelemetry(
        result,
        {"device": "cuda:0", "control_period_s": 0.01},
        started_monotonic_ns=1_000_000_000,
        started_unix_s=123.0,
    )
    telemetry.write_running()
    assert json.loads(result.read_text()) == {
        "dataset_finalization_included": False,
        "runtime": {"control_period_s": 0.01, "device": "cuda:0"},
        "schema_version": 1,
        "scope": "interactive_recording_loop",
        "status": "running",
    }

    telemetry.observe("active_cycle", 20_000_000)
    telemetry.observe("active_cycle", 20_000_000)
    telemetry.observe("env_step", 15_000_000)
    telemetry.increment("active_env_steps", 2)
    telemetry.increment("active_render_calls")
    telemetry.write_final(
        status="ok",
        exit_reason="keyboard_interrupt",
        recorded_count=1,
        auto_outcome_counts={"success": 1},
        ended_monotonic_ns=1_100_000_000,
    )

    report = json.loads(result.read_text())
    assert report["status"] == "ok"
    assert report["exit_reason"] == "keyboard_interrupt"
    assert report["scope"] == "interactive_recording_loop"
    assert report["dataset_finalization_included"] is False
    assert report["session_wall_s"] == pytest.approx(0.1)
    assert report["active_simulated_s"] == pytest.approx(0.02)
    assert report["active_effective_hz"] == pytest.approx(50.0)
    assert report["active_realtime_factor"] == pytest.approx(0.5)
    assert report["active_render_hz"] == pytest.approx(25.0)
    assert report["timing_host_wall"]["active_cycle"]["count"] == 2
    assert report["recorded_count"] == 1
    assert report["auto_outcome_counts"] == {"success": 1}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    ("xr_enabled", "sample_steps", "message"),
    [
        (False, 0, "requires an XR teleop device"),
        (True, 10, "not --sample_steps"),
    ],
)
def test_xr_runtime_metrics_rejects_unsupported_modes(xr_enabled: bool, sample_steps: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        require_xr_runtime_metrics_mode(
            metrics_enabled=True,
            xr_enabled=xr_enabled,
            sample_steps=sample_steps,
        )


def test_xr_runtime_metrics_allows_live_xr_or_disabled_metrics() -> None:
    require_xr_runtime_metrics_mode(metrics_enabled=True, xr_enabled=True, sample_steps=0)
    require_xr_runtime_metrics_mode(metrics_enabled=False, xr_enabled=False, sample_steps=10)


def test_h2_recording_requires_the_manifested_camera_stream() -> None:
    with pytest.raises(ValueError, match="requires --enable_cameras"):
        require_h2_recording_cameras(h2_arena_recording=True, cameras_enabled=False)

    require_h2_recording_cameras(h2_arena_recording=True, cameras_enabled=True)
    require_h2_recording_cameras(h2_arena_recording=False, cameras_enabled=False)


def test_reset_view_refresh_renders_without_advancing_physics_or_replacing_observations() -> None:
    calls: list[str] = []
    reset_observations = {"camera_obs": object()}

    class Simulation:
        render_context = type(
            "RenderContext",
            (),
            {"reset_transform_cadence": lambda self: calls.append("reset_transform_cadence")},
        )()

        def render(self) -> None:
            calls.append("render")

        def step(self, *, render: bool) -> None:
            raise AssertionError(f"reset refresh advanced physics with render={render}")

    env = type("Env", (), {"sim": Simulation(), "obs_buf": reset_observations})()

    refresh_reset_view(env)

    assert calls == ["reset_transform_cadence", "render"]
    assert env.obs_buf is reset_observations


@pytest.mark.parametrize("device", ["handtracking", "manusvive", "keyboard", "spacemouse"])
def test_h2_recording_rejects_a_selector_without_a_matching_controller_graph(device: str) -> None:
    message = "motion_controllers" if device in {"handtracking", "manusvive"} else "no retargeter"
    with pytest.raises(ValueError, match=message):
        require_arena_teleop_mode(
            arena_recording=True,
            h2_arena_recording=True,
            sample_steps=0,
            teleop_device=device,
        )


def test_h2_recording_allows_the_controller_graph_and_device_free_sample_mode() -> None:
    require_arena_teleop_mode(
        arena_recording=True,
        h2_arena_recording=True,
        sample_steps=0,
        teleop_device="motion_controllers",
    )
    require_arena_teleop_mode(
        arena_recording=True,
        h2_arena_recording=True,
        sample_steps=10,
        teleop_device="keyboard",
    )


@pytest.mark.parametrize("sample_steps,xr,expected", [(10, True, "prime"), (10, False, "prime"), (0, True, "ready")])
@pytest.mark.parametrize("embodiment", ["h2_agile_pink", "h2_agile_joint"])
def test_h2_sample_reset_primes_without_interactive_readiness(monkeypatch, sample_steps, xr, expected, embodiment):
    source = Path(__file__).resolve().parents[1] / "scripts/record_demos.py"
    node = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "reset_episode"
    )
    events = []
    monkeypatch.setattr(
        "e2e_robot_workflows.devices.h2_ready.prime_h2_controller",
        lambda env, selected: events.append(("prime", selected)),
    )
    monkeypatch.setattr(
        "e2e_robot_workflows.devices.h2_ready.prepare_h2_recording",
        lambda env, **kw: events.append(("ready", kw["embodiment"])),
    )
    namespace = {
        "gym": SimpleNamespace(Env=object),
        "H2_ARENA_RECORDING": True,
        "args_cli": SimpleNamespace(sample_steps=sample_steps, xr=xr),
        "arena_embodiment": embodiment,
        "simulation_app": SimpleNamespace(is_running=lambda: True),
        "refresh_reset_view": lambda env: events.append("refresh"),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    env = SimpleNamespace(
        sim=SimpleNamespace(reset=lambda: events.append("sim_reset")),
        reset=lambda: events.append("env_reset"),
        recorder_manager=SimpleNamespace(
            reset=lambda: events.append("recorder_reset"), record_post_reset=lambda ids: events.append("snapshot")
        ),
    )
    namespace["reset_episode"](env)
    assert events[:3] == ["sim_reset", "recorder_reset", "env_reset"]
    assert events[3] == (expected, embodiment)
    assert events[4:] == (["recorder_reset", "snapshot", "refresh"] if sample_steps else ["refresh"])


def test_environment_snapshot_settles_reset_scene_without_retaining_preparation_frames():
    from contextlib import nullcontext

    from e2e_robot_workflows.environments.runtime_snapshot import ENVIRONMENT_SNAPSHOT_SETTLE_STEPS

    source = Path(__file__).resolve().parents[1] / "scripts/record_demos.py"
    main = next(
        node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    node = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "args_cli.environment_snapshot is not None"
    )
    state = {"object_pose": "plate"}
    events = []
    frames = []
    hold_action = object()

    def reset(env):
        state.update(object_pose="unsettled_spawn")
        events.append("reset")

    def step(action):
        assert action is hold_action
        frames.append(dict(state))
        state.update(object_pose="settled_spawn")
        events.append("step")

    def clear_recorder():
        frames.clear()
        events.append("recorder_reset")

    env = SimpleNamespace(
        scene={},
        step=step,
        recorder_manager=SimpleNamespace(reset=clear_recorder),
    )
    namespace = {
        "args_cli": SimpleNamespace(environment_snapshot="snapshot.json"),
        "environment": object(),
        "env": env,
        "env_cfg": object(),
        "snapshot_mode": lambda cfg: "recording",
        "logger": logger,
        "resolve_role_to_scene_entity": lambda scene: {},
        "collect_environment_snapshot": lambda **kw: dict(state),
        "reset_episode": reset,
        "make_hold_action": lambda env: hold_action,
        "torch": SimpleNamespace(inference_mode=nullcontext),
        "ENVIRONMENT_SNAPSHOT_SETTLE_STEPS": ENVIRONMENT_SNAPSHOT_SETTLE_STEPS,
        "write_environment_snapshot": lambda path, snapshot: events.append(snapshot),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    assert events == [
        "reset",
        *(["step"] * ENVIRONMENT_SNAPSHOT_SETTLE_STEPS),
        "recorder_reset",
        {"object_pose": "settled_spawn"},
    ]
    assert not frames

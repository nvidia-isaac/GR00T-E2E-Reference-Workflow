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

"""Host-side tests for the GPU simulation validation lifecycle."""

from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SIMULATION_TOOLS = REPO_ROOT / "tools" / "simulation"


class FakeEnvironment:
    def __init__(self, events: list[str], close_error: BaseException | None = None) -> None:
        self.events = events
        self.close_error = close_error

    def close(self) -> None:
        self.events.append("environment.close")
        if self.close_error is not None:
            raise self.close_error


class FakeSimulationApp:
    def __init__(self, events: list[str], close_error: BaseException | None = None) -> None:
        self.events = events
        self.close_error = close_error

    def close(self) -> None:
        self.events.append("simulation_app.close")
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture
def validation_runner(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(SIMULATION_TOOLS))
    sys.modules.pop("validation_runner", None)
    return importlib.import_module("validation_runner")


def _install_runtime_modules(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    app: FakeSimulationApp | BaseException,
) -> None:
    compat_module = ModuleType("e2e_robot_workflows.compat")

    def warmup_pinocchio_bindings() -> None:
        events.append("pinocchio.warmup")

    compat_module.warmup_pinocchio_bindings = warmup_pinocchio_bindings  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "e2e_robot_workflows.compat", compat_module)

    isaaclab_module = ModuleType("isaaclab")
    isaaclab_module.__path__ = []  # type: ignore[attr-defined]
    app_module = ModuleType("isaaclab.app")

    class AppLauncher:
        def __init__(self, *, headless: bool, enable_cameras: bool) -> None:
            assert events == ["pinocchio.warmup"]
            assert headless is True
            assert enable_cameras is False
            events.append("app_launcher")
            if isinstance(app, BaseException):
                raise app
            self.app = app

    app_module.AppLauncher = AppLauncher  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)


def _capture_results(
    monkeypatch: pytest.MonkeyPatch,
    validation_runner: Any,
    events: list[str],
) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []

    def capture_result(_path: str, result: dict[str, Any]) -> None:
        events.append("write_result")
        snapshots.append(copy.deepcopy(result))

    monkeypatch.setattr(validation_runner, "write_result", capture_result)
    return snapshots


def test_success_preserves_order_and_scenario_results(
    monkeypatch: pytest.MonkeyPatch,
    validation_runner: Any,
) -> None:
    events: list[str] = []
    app = FakeSimulationApp(events)
    env = FakeEnvironment(events)
    _install_runtime_modules(monkeypatch, events, app)
    snapshots = _capture_results(monkeypatch, validation_runner, events)

    def execute(result: dict[str, Any]) -> FakeEnvironment:
        events.append("execute")
        result["runtime"] = {"version": "test"}
        result["simulation"] = {"steps": 3}
        result["completed_steps"] = 3
        return env

    assert (
        validation_runner.run_validation(
            result_path="/tmp/result.json",
            task_id="Test-Task-v0",
            requested_steps=3,
            execute=execute,
            log_label="test-validation",
        )
        == 0
    )

    assert events == [
        "pinocchio.warmup",
        "app_launcher",
        "execute",
        "environment.close",
        "write_result",
        "simulation_app.close",
        "write_result",
    ]
    assert len(snapshots) == 2
    assert snapshots[0]["status"] == "ok"
    assert snapshots[0]["stage"] == "complete"
    assert snapshots[0]["requested_steps"] == snapshots[0]["completed_steps"] == 3
    assert snapshots[0]["simulation_app_close_requested"] is True
    assert "simulation_app_close_completed" not in snapshots[0]
    assert snapshots[1]["simulation_app_close_completed"] is True
    assert snapshots[1]["runtime"] == {"version": "test"}
    assert snapshots[1]["simulation"] == {"steps": 3}


def test_primary_failure_is_portable_and_keeps_original_exception(
    monkeypatch: pytest.MonkeyPatch,
    validation_runner: Any,
) -> None:
    events: list[str] = []
    app = FakeSimulationApp(events)
    _install_runtime_modules(monkeypatch, events, app)
    snapshots = _capture_results(monkeypatch, validation_runner, events)
    failure = RuntimeError("/host/repository/policy.pt failed")

    def execute(_result: dict[str, Any]) -> None:
        events.append("execute")
        raise failure

    with pytest.raises(RuntimeError) as exc_info:
        validation_runner.run_validation(
            result_path="/tmp/result.json",
            task_id="Test-Task-v0",
            requested_steps=3,
            execute=execute,
            log_label="test-validation",
            error_message=lambda error: str(error).replace("/host/repository", "<repository>"),
            include_traceback=True,
        )

    assert exc_info.value is failure
    assert events == ["pinocchio.warmup", "app_launcher", "execute", "write_result"]
    assert snapshots[0]["error"] == "<repository>/policy.pt failed"
    assert snapshots[0]["error_type"] == "RuntimeError"
    assert snapshots[0]["status"] == "failed"
    assert "policy.pt failed" in snapshots[0]["traceback"]
    assert snapshots[0]["simulation_app_close_requested"] is False
    assert snapshots[0]["simulation_app_close_skipped_due_to_failure"] is True


def test_environment_close_failure_is_promoted_and_skips_kit_close(
    monkeypatch: pytest.MonkeyPatch,
    validation_runner: Any,
) -> None:
    events: list[str] = []
    app = FakeSimulationApp(events)
    env = FakeEnvironment(events, RuntimeError("/host/repository/env close failed"))
    _install_runtime_modules(monkeypatch, events, app)
    snapshots = _capture_results(monkeypatch, validation_runner, events)

    with pytest.raises(RuntimeError, match="environment: RuntimeError: <repository>/env close failed"):
        validation_runner.run_validation(
            result_path="/tmp/result.json",
            task_id="Test-Task-v0",
            requested_steps=1,
            execute=lambda _result: env,
            log_label="test-validation",
            error_message=lambda error: str(error).replace("/host/repository", "<repository>"),
        )

    assert events == ["pinocchio.warmup", "app_launcher", "environment.close", "write_result"]
    assert snapshots[0]["status"] == "failed"
    assert snapshots[0]["cleanup_errors"] == ["environment: RuntimeError: <repository>/env close failed"]
    assert snapshots[0]["simulation_app_close_skipped_due_to_failure"] is True
    assert "traceback" not in snapshots[0]


def test_simulation_app_close_failure_updates_second_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    validation_runner: Any,
) -> None:
    events: list[str] = []
    app = FakeSimulationApp(events, RuntimeError("Kit close failed"))
    env = FakeEnvironment(events)
    _install_runtime_modules(monkeypatch, events, app)
    snapshots = _capture_results(monkeypatch, validation_runner, events)

    with pytest.raises(RuntimeError, match="simulation_app: RuntimeError: Kit close failed"):
        validation_runner.run_validation(
            result_path="/tmp/result.json",
            task_id="Test-Task-v0",
            requested_steps=1,
            execute=lambda _result: env,
            log_label="test-validation",
        )

    assert events == [
        "pinocchio.warmup",
        "app_launcher",
        "environment.close",
        "write_result",
        "simulation_app.close",
        "write_result",
    ]
    assert snapshots[0]["status"] == "ok"
    assert snapshots[1]["status"] == "failed"
    assert snapshots[1]["cleanup_errors"] == ["simulation_app: RuntimeError: Kit close failed"]
    assert snapshots[1]["simulation_app_close_requested"] is True
    assert "simulation_app_close_completed" not in snapshots[1]


def test_app_launcher_failure_records_prelaunch_state(
    monkeypatch: pytest.MonkeyPatch,
    validation_runner: Any,
) -> None:
    events: list[str] = []
    failure = RuntimeError("launcher failed")
    _install_runtime_modules(monkeypatch, events, failure)
    snapshots = _capture_results(monkeypatch, validation_runner, events)

    with pytest.raises(RuntimeError) as exc_info:
        validation_runner.run_validation(
            result_path="/tmp/result.json",
            task_id="Test-Task-v0",
            requested_steps=1,
            execute=lambda _result: None,
            log_label="test-validation",
        )

    assert exc_info.value is failure
    assert events == ["pinocchio.warmup", "app_launcher", "write_result"]
    assert snapshots[0]["stage"] == "app_launcher"
    assert snapshots[0]["simulation_app_close_requested"] is False
    assert snapshots[0]["simulation_app_close_completed"] is True


def test_successful_result_write_failure_preserves_existing_shutdown_behavior(
    monkeypatch: pytest.MonkeyPatch,
    validation_runner: Any,
) -> None:
    events: list[str] = []
    app = FakeSimulationApp(events)
    env = FakeEnvironment(events)
    _install_runtime_modules(monkeypatch, events, app)
    write_failure = OSError("result path is unavailable")

    def fail_write(_path: str, _result: dict[str, Any]) -> None:
        events.append("write_result")
        raise write_failure

    monkeypatch.setattr(validation_runner, "write_result", fail_write)
    with pytest.raises(OSError) as exc_info:
        validation_runner.run_validation(
            result_path="/tmp/result.json",
            task_id="Test-Task-v0",
            requested_steps=1,
            execute=lambda _result: env,
            log_label="test-validation",
        )

    assert exc_info.value is write_failure
    assert events == ["pinocchio.warmup", "app_launcher", "environment.close", "write_result"]

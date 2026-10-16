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

"""Lifecycle runner for headless GPU simulation validations."""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from loguru import logger
from validation_support import write_result

ErrorMessage = Callable[[BaseException], str]
ExecuteValidation = Callable[[dict[str, Any]], Any | None]


def run_validation(
    *,
    result_path: str,
    task_id: str,
    requested_steps: int,
    execute: ExecuteValidation,
    log_label: str,
    error_message: ErrorMessage = str,
    include_traceback: bool = False,
    enable_cameras: bool = False,
) -> int:
    """Run one validation while preserving diagnostics across Kit teardown.

    Raw tracebacks may contain host paths, so leave ``include_traceback`` false
    when ``error_message`` sanitizes paths for portable artifacts.
    """

    started_at = time.monotonic()
    result: dict[str, Any] = {
        "completed_steps": 0,
        "requested_steps": requested_steps,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "task_id": task_id,
    }
    simulation_app = None
    env = None
    failure: BaseException | None = None
    failure_traceback = None

    try:
        result["stage"] = "pinocchio_warmup"
        from e2e_robot_workflows.compat import warmup_pinocchio_bindings

        warmup_pinocchio_bindings()

        result["stage"] = "app_launcher"
        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(headless=True, enable_cameras=enable_cameras).app
        env = execute(result)
        result["stage"] = "complete"
    except BaseException as exc:
        failure = exc
        failure_traceback = exc.__traceback__
        result.update(
            {
                "error": error_message(exc),
                "error_type": type(exc).__name__,
                "status": "failed",
            }
        )
        if include_traceback:
            result["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    finally:
        cleanup_errors: list[str] = list(result.get("cleanup_errors", []))
        if env is not None:
            try:
                env.close()
            except BaseException as exc:
                cleanup_errors.append(f"environment: {type(exc).__name__}: {error_message(exc)}")
        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
            if failure is None:
                failure = RuntimeError("; ".join(cleanup_errors))
                result.update(
                    {
                        "error": str(failure),
                        "error_type": type(failure).__name__,
                        "status": "failed",
                    }
                )
        elif failure is None:
            result["status"] = "ok"

        result["duration_s"] = round(time.monotonic() - started_at, 3)
        result["finished_at_utc"] = datetime.now(UTC).isoformat()
        result["simulation_app_close_requested"] = simulation_app is not None and failure is None
        if simulation_app is not None and failure is not None:
            # Kit teardown can terminate this interpreter with status zero on
            # startup failures, masking the validation failure. Let process
            # teardown release Kit after the recorded exception is re-raised.
            result["simulation_app_close_skipped_due_to_failure"] = True
        if simulation_app is None:
            result["simulation_app_close_completed"] = True
        try:
            write_result(result_path, result)
        except BaseException as write_error:
            if failure is None:
                raise
            logger.info(f"{log_label}: failed to write result JSON: {write_error}")

        # Persist the outcome before closing Kit. Some fatal startup paths can
        # terminate the process from inside close(), so diagnostics must not
        # depend on control returning from this call.
        if simulation_app is not None and failure is None:
            try:
                simulation_app.close()
                result["simulation_app_close_completed"] = True
            except BaseException as exc:
                cleanup_error = f"simulation_app: {type(exc).__name__}: {error_message(exc)}"
                result.setdefault("cleanup_errors", []).append(cleanup_error)
                failure = RuntimeError(cleanup_error)
                result.update(
                    {
                        "error": str(failure),
                        "error_type": type(failure).__name__,
                        "status": "failed",
                    }
                )
            finally:
                result["duration_s"] = round(time.monotonic() - started_at, 3)
                result["finished_at_utc"] = datetime.now(UTC).isoformat()
                try:
                    write_result(result_path, result)
                except BaseException as write_error:
                    if failure is None:
                        raise
                    logger.info(f"{log_label}: failed to update result JSON after cleanup: {write_error}")

    if failure is not None:
        raise failure.with_traceback(failure_traceback)
    return 0

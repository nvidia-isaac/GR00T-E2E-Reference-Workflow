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

"""Generic episode-completion interface for the recorder.

Keeps ``scripts/record_demos.py`` task-agnostic: a task registers a
``completion_policy_entry_point`` (``module:function``) alongside its gym env, and
the recorder resolves it into a :class:`CompletionPolicy` it consumes without
importing any task-specific detection logic. The static-apple policy lives in
``tasks/static_apple/completion.py``.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EpisodeOutcome(Protocol):
    """A terminal-or-running episode outcome the recorder can log and act on."""

    value: str

    @property
    def is_terminal(self) -> bool: ...

    @property
    def is_success(self) -> bool: ...


@runtime_checkable
class CompletionPolicy(Protocol):
    """A task's episode-completion policy, consumed by the generic recorder."""

    default_enabled: bool  # auto success/failure detection on by default for the task
    hold_frames: int  # frames to keep recording after success (follow-through)

    def evaluate(self, env: Any) -> EpisodeOutcome:
        """Return the current episode's outcome for the live environment."""
        ...

    def check_readable(self, env: Any) -> None:
        """Raise unless the live environment exposes everything :meth:`evaluate` reads.

        The recorder calls this once before a session, because :meth:`evaluate` is
        polled inside a live episode where a read error can only be logged.
        """
        ...


class CompletionTransition(StrEnum):
    """Recorder action selected from one completion-policy observation."""

    CONTINUE = "continue"
    START_HOLD = "start_hold"
    INTERRUPT_HOLD = "interrupt_hold"
    COMPLETE_SUCCESS = "complete_success"
    COMPLETE_FAILURE = "complete_failure"


def advance_success_hold(
    outcome: EpisodeOutcome | None,
    *,
    hold_remaining: int | None,
    hold_frames: int,
) -> tuple[CompletionTransition, int | None]:
    """Advance the recorder's consecutive-success follow-through state."""

    if hold_frames < 0 or (hold_remaining is not None and hold_remaining <= 0):
        raise ValueError("success hold frame counts must be non-negative")
    if hold_remaining is not None:
        if outcome is None or not outcome.is_terminal:
            return CompletionTransition.INTERRUPT_HOLD, None
        if not outcome.is_success:
            return CompletionTransition.COMPLETE_FAILURE, None
        remaining = hold_remaining - 1
        if remaining == 0:
            return CompletionTransition.COMPLETE_SUCCESS, None
        return CompletionTransition.CONTINUE, remaining
    if outcome is None or not outcome.is_terminal:
        return CompletionTransition.CONTINUE, None
    if not outcome.is_success:
        return CompletionTransition.COMPLETE_FAILURE, None
    if hold_frames == 0:
        return CompletionTransition.COMPLETE_SUCCESS, None
    return CompletionTransition.START_HOLD, hold_frames


def qualifies_success_hold(observations: Sequence[bool], *, hold_frames: int) -> bool:
    """Require initial success plus exactly ``hold_frames`` successful follow-through frames."""

    if hold_frames < 0 or len(observations) != hold_frames + 1:
        raise ValueError("success observations must contain the initial frame and every hold frame")
    return all(observations)


def resolve_completion_policy(entry_point: str, overrides: dict[str, float]) -> CompletionPolicy:
    """Build a task's completion policy from its ``module:function`` entry point.

    ``overrides`` are task-namespaced tuning values (from ``--completion_override``);
    the task's factory interprets and validates them.
    """

    module_path, _, func_name = entry_point.partition(":")
    if not module_path or not func_name:
        raise ValueError(f"completion policy entry point must be 'module:function', got: {entry_point!r}")
    factory = getattr(importlib.import_module(module_path), func_name)
    return factory(overrides)

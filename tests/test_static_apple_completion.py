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

"""Host-side tests for the static-apple completion policy (recorder integration)."""

from __future__ import annotations

import pytest

from e2e_robot_workflows.data.completion import (
    CompletionPolicy,
    CompletionTransition,
    advance_success_hold,
    qualifies_success_hold,
    resolve_completion_policy,
)
from e2e_robot_workflows.environments import STATIC_APPLE_ENVIRONMENT
from e2e_robot_workflows.tasks.static_apple.completion import build_policy
from e2e_robot_workflows.tasks.static_apple.success import Outcome

ENTRY_POINT = "e2e_robot_workflows.tasks.static_apple.completion:build_policy"


def test_default_policy_reads_contract_and_is_a_completion_policy() -> None:
    policy = build_policy()
    assert isinstance(policy, CompletionPolicy)  # satisfies the generic recorder interface
    assert policy.default_enabled is True
    assert policy.hold_frames == STATIC_APPLE_ENVIRONMENT.success_hold_frames
    # Success thresholds are eval-shared and contract-only (not session-tunable).
    assert policy.criteria.contact_force_threshold == STATIC_APPLE_ENVIRONMENT.success_force_threshold


def test_the_policy_checks_whether_it_can_read_a_scene() -> None:
    """A scene the detector cannot read is a wiring bug, and the recorder asks up front."""

    class _Scene:
        def keys(self) -> tuple[str, ...]:
            return ("robot",)

    class _Env:
        scene = _Scene()
        cfg = type("Cfg", (), {"e2e_task_link_names": ("left", "right")})()

    with pytest.raises(KeyError, match="pick_object"):
        build_policy().check_readable(_Env())


def test_overrides_apply_onto_contract_defaults() -> None:
    policy = build_policy({"drop_height_m": 0.3, "success_hold_frames": 5})
    assert policy.criteria.drop_height_m == 0.3
    assert policy.hold_frames == 5
    # Unspecified fields keep their contract defaults.
    assert policy.criteria.far_distance_m == STATIC_APPLE_ENVIRONMENT.failure_far_distance_m


def test_unknown_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown completion override"):
        build_policy({"contact_force_threshold": 0.5})  # success thresholds are contract-only


def test_non_positive_distance_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        build_policy({"drop_height_m": 0.0})
    with pytest.raises(ValueError, match="positive and finite"):
        build_policy({"far_distance_m": float("nan")})


def test_non_integer_hold_frames_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        build_policy({"success_hold_frames": 2.5})
    with pytest.raises(ValueError, match="non-negative integer"):
        build_policy({"success_hold_frames": -1})


def test_resolver_builds_the_registered_entry_point() -> None:
    policy = resolve_completion_policy(ENTRY_POINT, {"far_distance_m": 0.4})
    assert policy.criteria.far_distance_m == 0.4


def test_resolver_rejects_a_malformed_entry_point() -> None:
    with pytest.raises(ValueError, match="module:function"):
        resolve_completion_policy("not_an_entry_point", {})


def test_success_hold_requires_every_follow_through_frame() -> None:
    transition, remaining = advance_success_hold(Outcome.SUCCESS, hold_remaining=None, hold_frames=2)
    assert (transition, remaining) == (CompletionTransition.START_HOLD, 2)
    transition, remaining = advance_success_hold(Outcome.SUCCESS, hold_remaining=remaining, hold_frames=2)
    assert (transition, remaining) == (CompletionTransition.CONTINUE, 1)
    transition, remaining = advance_success_hold(Outcome.SUCCESS, hold_remaining=remaining, hold_frames=2)
    assert (transition, remaining) == (CompletionTransition.COMPLETE_SUCCESS, None)
    assert qualifies_success_hold([True, True, True], hold_frames=2)
    assert not qualifies_success_hold([True, False, True], hold_frames=2)
    with pytest.raises(ValueError, match="initial frame and every hold frame"):
        qualifies_success_hold([True, True], hold_frames=2)


def test_running_outcome_interrupts_and_restarts_success_hold() -> None:
    _, remaining = advance_success_hold(Outcome.SUCCESS, hold_remaining=None, hold_frames=2)
    transition, remaining = advance_success_hold(Outcome.RUNNING, hold_remaining=remaining, hold_frames=2)
    assert (transition, remaining) == (CompletionTransition.INTERRUPT_HOLD, None)
    transition, remaining = advance_success_hold(Outcome.SUCCESS, hold_remaining=remaining, hold_frames=2)
    assert (transition, remaining) == (CompletionTransition.START_HOLD, 2)


def test_failure_during_success_hold_finishes_as_failure() -> None:
    _, remaining = advance_success_hold(Outcome.SUCCESS, hold_remaining=None, hold_frames=2)
    transition, remaining = advance_success_hold(
        Outcome.FAILURE_DROPPED,
        hold_remaining=remaining,
        hold_frames=2,
    )
    assert (transition, remaining) == (CompletionTransition.COMPLETE_FAILURE, None)

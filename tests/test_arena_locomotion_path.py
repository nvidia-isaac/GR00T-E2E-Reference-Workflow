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

"""Contracts for driving locomotion on the Arena-composed environment.

The GPU behavior itself is checked by ``sim-arena-locomotion``. What is checked here is
what a host can check: command layout, robot-to-runtime routing, embodiment capabilities,
and the validation command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATE_REPOSITORY = (REPO_ROOT / "tools/validate_repository.sh").read_text(encoding="utf-8")


@pytest.fixture
def actions() -> Any:
    """The action helpers, which need torch and so are absent from the CPU-only image."""

    pytest.importorskip("torch", reason="torch is not installed on the CPU-only host image")
    from e2e_robot_workflows.arena import g1_actions

    return g1_actions


@pytest.mark.parametrize("action_dim", [23, 50], ids=["pink", "joint"])
def test_the_navigate_command_lands_in_the_locomotion_tail(actions: Any, action_dim: int) -> None:
    """Both layouts end with the same locomotion tail, and nothing ahead of it moves.

    Addressing the command from the end is what lets one writer serve two action widths;
    an absolute index would silently write into the arms on the other layout.
    """

    action = actions.torch.ones((1, action_dim))

    actions.set_navigate_command(action, 0.5, -0.25, 0.75)

    navigate_start = action_dim - actions.LOCOMOTION_COMMAND_DIM
    navigate_end = navigate_start + actions.NAVIGATE_COMMAND_DIM
    assert action[0, navigate_start:navigate_end].tolist() == [0.5, -0.25, 0.75]
    # The joint or wrist targets ahead of the tail, and the height and torso commands
    # after the navigate command, are the caller's to set.
    assert action[0, :navigate_start].tolist() == [1.0] * navigate_start
    assert action[0, navigate_end:].tolist() == [1.0] * (action_dim - navigate_end)


def test_the_navigate_command_writer_returns_the_action_it_wrote(actions: Any) -> None:
    """Callers chain it onto a hold command, so it must hand the same tensor back."""

    action = actions.torch.zeros((1, 50))

    assert actions.set_navigate_command(action, 0.1, 0.2, 0.3) is action


def test_locomotion_embodiments_accept_only_their_supported_controller_modes() -> None:
    """Each robot accepts joint driving and rejects OpenXR on that action layout."""

    from e2e_robot_workflows.arena.embodiment_capabilities import (
        G1_ARENA_EMBODIMENT,
        H2_ARENA_EMBODIMENT,
        capability_for_embodiment,
    )

    for embodiment in (G1_ARENA_EMBODIMENT, H2_ARENA_EMBODIMENT):
        capability = capability_for_embodiment(embodiment)
        capability.validate(embodiment=embodiment, teleop_device=None, mimic=False)
        with pytest.raises(ValueError, match="has no retargeter"):
            capability.validate(embodiment=embodiment, teleop_device="openxr", mimic=False)


def test_h2_robot_selection_always_resolves_to_an_h2_arena_embodiment() -> None:
    from e2e_robot_workflows.arena.embodiment_capabilities import (
        H2_ARENA_EMBODIMENT,
        H2_ARENA_TELEOP_EMBODIMENT,
        resolve_locomotion_embodiment,
    )

    assert (
        resolve_locomotion_embodiment(robot="h2", arena_requested=False, requested_embodiment=None)
        == H2_ARENA_EMBODIMENT
    )
    assert (
        resolve_locomotion_embodiment(
            robot="h2",
            arena_requested=True,
            requested_embodiment=H2_ARENA_TELEOP_EMBODIMENT,
        )
        == H2_ARENA_TELEOP_EMBODIMENT
    )


def test_g1_locomotion_preserves_built_in_and_arena_routes() -> None:
    from e2e_robot_workflows.arena.embodiment_capabilities import (
        G1_ARENA_EMBODIMENT,
        resolve_locomotion_embodiment,
    )

    assert resolve_locomotion_embodiment(robot="g1", arena_requested=False, requested_embodiment=None) is None
    assert (
        resolve_locomotion_embodiment(robot="g1", arena_requested=True, requested_embodiment=None)
        == G1_ARENA_EMBODIMENT
    )


def test_locomotion_route_rejects_cross_robot_embodiments() -> None:
    from e2e_robot_workflows.arena.embodiment_capabilities import (
        G1_ARENA_EMBODIMENT,
        H2_ARENA_EMBODIMENT,
        resolve_locomotion_embodiment,
    )

    with pytest.raises(ValueError, match="requires an H2 Arena embodiment"):
        resolve_locomotion_embodiment(
            robot="h2",
            arena_requested=True,
            requested_embodiment=G1_ARENA_EMBODIMENT,
        )
    with pytest.raises(ValueError, match="requires a G1 Arena embodiment"):
        resolve_locomotion_embodiment(
            robot="g1",
            arena_requested=True,
            requested_embodiment=H2_ARENA_EMBODIMENT,
        )


def test_the_locomotion_validation_mode_is_available() -> None:
    """The validation mode must dispatch to its runner and appear in usage text."""

    assert "sim-arena-locomotion) ./tools/validate_arena_locomotion.sh ;;" in VALIDATE_REPOSITORY
    usage = next(line for line in VALIDATE_REPOSITORY.splitlines() if line.startswith("Usage:"))
    assert "|sim-arena-locomotion" in usage, "the mode must appear in the usage text"

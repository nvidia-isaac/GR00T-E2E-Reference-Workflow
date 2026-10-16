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

"""Shared array-only Arena source-boundary admission, including live recorder inputs."""

from __future__ import annotations

import numpy as np
import pytest

from e2e_robot_workflows.data.dataset_manifest import (
    ArenaEpisodeRejection,
    DatasetContractError,
    validate_arena_episode_boundaries,
)


def _signals(first: str = "left") -> dict[str, np.ndarray]:
    receiver = "right" if first == "left" else "left"
    return {
        first: np.asarray((False, True, True, True)),
        receiver: np.asarray((False, False, True, True)),
    }


@pytest.mark.parametrize("hand", ("left", "right"))
@pytest.mark.parametrize("dtype", (np.bool_, np.int32, np.float32))
def test_shared_boundaries_admit_either_hand_with_binary_arrays(hand: str, dtype) -> None:
    signals = {side: value.astype(dtype) for side, value in _signals(hand).items()}
    assert validate_arena_episode_boundaries(signals, steps=4, manipulation_hand=hand) == hand


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        ((False,) * 4, (False,) * 4, "found none"),
        ((True,) * 4, (True,) * 4, "found none"),
        ((False, True, True, True), (False, True, True, True), "unambiguous"),
        ((False, True, True, True), (False,) * 4, "right-hand handover boundary"),
        ((False, True, True, True), (False, False, False, True), "leaves no placement samples"),
    ],
)
def test_shared_boundaries_reject_semantically_unusable_episodes(left, right, message: str) -> None:
    with pytest.raises(ArenaEpisodeRejection, match=message):
        validate_arena_episode_boundaries({"left": left, "right": right}, steps=4)


def test_shared_boundaries_preserve_explicit_and_session_latched_first_hand() -> None:
    admitted_hand = validate_arena_episode_boundaries(_signals("left"), steps=4)

    with pytest.raises(ArenaEpisodeRejection, match="first pickup boundary belongs to right"):
        validate_arena_episode_boundaries(_signals("right"), steps=4, manipulation_hand=admitted_hand)


def test_shared_boundary_detection_can_require_pickup_without_handover() -> None:
    signals = _signals()
    signals["right"][:] = False

    assert validate_arena_episode_boundaries(signals, steps=4, require_handover=False) == "left"


@pytest.mark.parametrize(
    "bad_signal",
    [
        None,
        [False],
        [[False], [True], [True], [True]],
        [0.0, np.nan, 1.0, 1.0],
        [0.0, 0.5, 1.0, 1.0],
        ["0", "1", "1", "1"],
    ],
)
def test_shared_boundaries_malformed_fields_are_not_recoverable_episode_rejections(bad_signal) -> None:
    signals = _signals()
    signals["right"] = bad_signal

    with pytest.raises(DatasetContractError) as error:
        validate_arena_episode_boundaries(signals, steps=4)
    assert not isinstance(error.value, ArenaEpisodeRejection)


def test_shared_boundaries_missing_fields_are_not_recoverable_episode_rejections() -> None:
    with pytest.raises(DatasetContractError, match="is missing") as error:
        validate_arena_episode_boundaries({"left": _signals()["left"]}, steps=4)
    assert not isinstance(error.value, ArenaEpisodeRejection)


@pytest.mark.parametrize("steps", (-1, True, 4.0))
def test_shared_boundaries_reject_invalid_sample_count(steps) -> None:
    with pytest.raises(DatasetContractError, match="sample count") as error:
        validate_arena_episode_boundaries(_signals(), steps=steps)
    assert not isinstance(error.value, ArenaEpisodeRejection)


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("giver", [[0, 1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 1, 0, 0, 0]])
def test_continuous_handover_requires_giver_grasp_until_receiver_acquires(hand, giver):
    receiver = "right" if hand == "left" else "left"
    raw = {hand: np.asarray(giver), receiver: np.asarray([0, 0, 0, 0, 1, 1, 0, 0])}
    latched = {name: np.logical_or.accumulate(value) for name, value in raw.items()}
    if all(giver[1:5]):
        assert validate_arena_episode_boundaries(latched, steps=8, grasp_signals=raw) == hand
    else:
        with pytest.raises(ArenaEpisodeRejection, match="continuous handover"):
            validate_arena_episode_boundaries(latched, steps=8, grasp_signals=raw)


def test_raw_grasp_evidence_must_agree_with_subtask_boundaries():
    with pytest.raises(DatasetContractError, match="disagrees"):
        validate_arena_episode_boundaries(
            _signals(), steps=4, grasp_signals={"left": [0, 0, 1, 1], "right": [0, 0, 1, 1]}
        )

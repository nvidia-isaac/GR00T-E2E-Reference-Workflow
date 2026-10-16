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

"""Simulator-backed behavior checks for the H2 live-head camera follower."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="torch is not installed on the CPU-only host image")
pytest.importorskip("isaaclab", reason="Isaac Lab is not installed on the CPU-only host image")

from e2e_robot_workflows.arena.h2_events import follow_h2_head_camera, reset_h2_head_camera  # noqa: E402


class _Camera:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(
            offset=SimpleNamespace(
                pos=(1.0, 0.0, 2.0),
                rot=(0.0, 0.0, 0.0, 1.0),
                convention="world",
            )
        )
        self.last_write = None

    def set_world_poses(self, positions, orientations, *, env_ids, convention) -> None:
        self.last_write = (positions.clone(), orientations.clone(), env_ids.clone(), convention)


class _RenderContext:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset_transform_cadence(self) -> None:
        self.reset_count += 1


def test_camera_follower_composes_selected_live_head_pose_and_offset() -> None:
    camera = _Camera()
    half_sqrt_two = 2.0**-0.5
    body_poses = torch.tensor(
        [
            [[[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]],
            [[[10.0, 20.0, 30.0, 0.0, 0.0, half_sqrt_two, half_sqrt_two]]],
        ]
    ).squeeze(1)
    robot = SimpleNamespace(
        body_names=["head_yaw_link"],
        data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=body_poses)),
    )
    env_ids = torch.tensor([1])
    render_context = _RenderContext()
    env = SimpleNamespace(
        scene={"robot_head_cam": camera, "robot": robot},
        sim=SimpleNamespace(render_context=render_context),
    )

    follow_h2_head_camera(env, env_ids)

    assert camera.last_write is not None
    position, orientation, written_env_ids, convention = camera.last_write
    torch.testing.assert_close(position, torch.tensor([[10.0, 21.0, 32.0]]), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        orientation,
        torch.tensor([[0.0, 0.0, half_sqrt_two, half_sqrt_two]]),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(written_env_ids, env_ids)
    assert convention == "world"
    assert render_context.reset_count == 1


def test_reset_camera_follower_refreshes_kinematics_before_reading_head_pose() -> None:
    camera = _Camera()
    stale_pose = torch.tensor([[[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]])
    reset_pose = torch.tensor([[[10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 1.0]]])
    body_link_pose = SimpleNamespace(torch=stale_pose)
    robot = SimpleNamespace(
        body_names=["head_yaw_link"],
        data=SimpleNamespace(body_link_pose_w=body_link_pose),
    )
    calls: list[str] = []

    def forward() -> None:
        calls.append("forward")
        body_link_pose.torch = reset_pose

    original_set_world_poses = camera.set_world_poses

    def set_world_poses(*args, **kwargs) -> None:
        calls.append("camera")
        original_set_world_poses(*args, **kwargs)

    camera.set_world_poses = set_world_poses
    env_ids = torch.tensor([0])
    render_context = _RenderContext()
    env = SimpleNamespace(
        scene={"robot_head_cam": camera, "robot": robot},
        sim=SimpleNamespace(forward=forward, render_context=render_context),
    )

    reset_h2_head_camera(env, env_ids)

    assert calls == ["forward", "camera"]
    assert camera.last_write is not None
    position, _, _, _ = camera.last_write
    torch.testing.assert_close(position, torch.tensor([[11.0, 20.0, 32.0]]))
    assert render_context.reset_count == 1

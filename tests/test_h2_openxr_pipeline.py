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

"""Behavioral contract for the registered 61-D H2 OpenXR graph."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("isaacteleop", reason="Isaac Teleop is not installed in the CPU-only test environment")

from isaacteleop.retargeting_engine.deviceio_source_nodes.deviceio_tensor_types import (
    DeviceIOControllerSnapshotTracked,
)
from isaacteleop.retargeting_engine.interface.tensor_group import TensorGroup
from isaacteleop.retargeting_engine.tensor_types import TransformMatrix
from isaacteleop.schema import (
    ControllerInputState,
    ControllerPose,
    ControllerSnapshot,
    ControllerSnapshotTrackedT,
    Point,
    Pose,
    Quaternion,
)
from scipy.spatial.transform import Rotation

from e2e_robot_workflows.arena.h2_openxr import build_h2_pink_openxr_pipeline
from e2e_robot_workflows.robots.h2 import load_h2_pink_contract, load_h2_robot_config


def _tracked_controller(
    *,
    x: float,
    stick_x: float = 0.0,
    stick_y: float = 0.0,
    trigger: float = 0.0,
    squeeze: float = 0.0,
    valid: bool = True,
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
):
    pose = ControllerPose(Pose(Point(x, 0.0, 1.0), Quaternion(*orientation_xyzw)), valid)
    inputs = ControllerInputState(False, False, False, False, stick_x, stick_y, squeeze, trigger)
    group = TensorGroup(DeviceIOControllerSnapshotTracked())
    group[0] = ControllerSnapshotTrackedT(ControllerSnapshot(pose, pose, inputs))
    return group


def _world_t_anchor(matrix: np.ndarray | None = None) -> TensorGroup:
    transform = TensorGroup(TransformMatrix())
    transform[0] = np.asarray(np.eye(4) if matrix is None else matrix, dtype=np.float32)
    return transform


def test_h2_pipeline_packs_poses_hands_and_velocity_without_padding() -> None:
    pipeline = build_h2_pink_openxr_pipeline()
    contract = load_h2_pink_contract()
    action = np.asarray(
        pipeline(
            {
                "controllers": {
                    "deviceio_controller_left": _tracked_controller(x=0.25, stick_x=0.4, stick_y=-0.8),
                    "deviceio_controller_right": _tracked_controller(
                        x=-0.25,
                        stick_x=0.6,
                        trigger=1.0,
                        squeeze=1.0,
                    ),
                },
                "world_T_anchor": {"value": _world_t_anchor()},
            }
        )["action"][0]
    )

    assert action.shape == (61,)
    assert np.allclose(action[:3], (0.25, 0.0, 1.0))
    assert np.allclose(action[7:10], (-0.25, 0.0, 1.0))
    assert np.allclose(action[3:7], (1.0, 0.0, 0.0, 0.0), atol=1e-6)
    assert np.allclose(action[10:14], (0.0, 1.0, 0.0, 0.0), atol=1e-6)
    expected_hands = [
        contract.hand_open_targets[name] if name.startswith("left_") else contract.hand_closed_targets[name]
        for name in contract.hand_joint_order
    ]
    assert np.allclose(action[14:58], expected_hands, atol=1e-6)
    assert np.allclose(action[-3:], (-7.0 / 18.0, -1.0 / 6.0, -7.0 / 36.0))


def test_h2_pipeline_transforms_controller_poses_into_the_anchor_frame() -> None:
    pipeline = build_h2_pink_openxr_pipeline()
    world_t_anchor = np.asarray(
        (
            (0.0, -1.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 2.0),
            (0.0, 0.0, 1.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    action = np.asarray(
        pipeline(
            {
                "controllers": {
                    "deviceio_controller_left": _tracked_controller(x=0.25, stick_y=0.8),
                    "deviceio_controller_right": _tracked_controller(x=-0.25, squeeze=1.0),
                },
                "world_T_anchor": {"value": _world_t_anchor(world_t_anchor)},
            }
        )["action"][0]
    )

    assert np.allclose(action[:3], (1.0, 2.25, 4.0))
    assert np.allclose(action[7:10], (1.0, 1.75, 4.0))
    assert np.allclose(action[3:7], (np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0), atol=1e-6)
    assert np.allclose(action[10:14], (-np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0), atol=1e-6)
    assert np.allclose(action[-3:], (7.0 / 18.0, 0.0, 0.0))


def test_h2_neutral_controller_orientation_matches_mirrored_flange_mounts() -> None:
    # IsaacTeleop's XR anchor converts Y-up OpenXR into Z-up USD. The H2
    # anchor starts at world identity, while its pelvis starts facing +Y.
    oxr_to_usd = np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
    pelvis_to_world = Rotation.from_quat(load_h2_robot_config().initial_rotation_xyzw).as_matrix()
    pelvis_t_oxr = np.eye(4)
    pelvis_t_oxr[:3, :3] = pelvis_to_world.T @ oxr_to_usd
    action = np.asarray(
        build_h2_pink_openxr_pipeline()(
            {
                "controllers": {
                    "deviceio_controller_left": _tracked_controller(x=-0.25),
                    "deviceio_controller_right": _tracked_controller(x=0.25),
                },
                "world_T_anchor": {"value": _world_t_anchor(pelvis_t_oxr)},
            }
        )["action"][0]
    )

    # These are the pinned Pink URDF's fixed wrist-to-flange joint origins,
    # expressed as extrinsic RPY, not the retargeter's configured offsets.
    for start, mount_rpy in ((3, (-90.0, 0.0, -90.0)), (10, (90.0, 0.0, 90.0))):
        flange = Rotation.from_quat(action[start : start + 4])
        mounting = Rotation.from_euler("xyz", mount_rpy, degrees=True)
        assert np.allclose(flange.as_matrix(), mounting.as_matrix(), atol=1e-6)
        assert np.allclose(flange.apply((0.0, 0.0, 1.0)), (1.0, 0.0, 0.0), atol=1e-6)


@pytest.mark.parametrize("grip_rpy", ((0.0, 0.0, 0.0), (23.0, -41.0, 67.0), (-85.0, 12.0, -137.0)))
def test_h2_flange_axes_follow_controller_rotation_with_mirrored_palms(grip_rpy) -> None:
    grip_rotation = Rotation.from_euler("xyz", grip_rpy, degrees=True)
    anchor_rotation = Rotation.from_euler("xyz", (31.0, -17.0, 53.0), degrees=True)
    transform = np.eye(4)
    transform[:3, :3] = anchor_rotation.as_matrix()
    transform[:3, 3] = (0.1, -0.2, 0.3)
    action = np.asarray(
        build_h2_pink_openxr_pipeline()(
            {
                "controllers": {
                    "deviceio_controller_left": _tracked_controller(
                        x=-0.25, orientation_xyzw=tuple(grip_rotation.as_quat())
                    ),
                    "deviceio_controller_right": _tracked_controller(
                        x=0.25, orientation_xyzw=tuple(grip_rotation.as_quat())
                    ),
                },
                "world_T_anchor": {"value": _world_t_anchor(transform)},
            }
        )["action"][0]
    )
    controller_rotation = anchor_rotation * grip_rotation
    for start, x_sign, y_sign, position_x in ((0, 1.0, -1.0, -0.25), (7, -1.0, 1.0, 0.25)):
        flange = Rotation.from_quat(action[start + 3 : start + 7])
        assert np.allclose(flange.apply((0.0, 0.0, 1.0)), controller_rotation.apply((0.0, 0.0, -1.0)), atol=1e-6)
        assert np.allclose(flange.apply((1.0, 0.0, 0.0)), controller_rotation.apply((x_sign, 0.0, 0.0)), atol=1e-6)
        assert np.allclose(flange.apply((0.0, 1.0, 0.0)), controller_rotation.apply((0.0, y_sign, 0.0)), atol=1e-6)
        assert np.allclose(
            action[start : start + 3], anchor_rotation.apply((position_x, 0.0, 1.0)) + transform[:3, 3], atol=1e-6
        )
    assert action.shape == (61,)


def test_h2_pipeline_marks_lost_pose_tracking_non_finite() -> None:
    pipeline = build_h2_pink_openxr_pipeline()
    action = np.asarray(
        pipeline(
            {
                "controllers": {
                    "deviceio_controller_left": _tracked_controller(x=0.0, valid=False),
                    "deviceio_controller_right": _tracked_controller(x=0.0, valid=False),
                },
                "world_T_anchor": {"value": _world_t_anchor()},
            }
        )["action"][0]
    )
    assert np.isnan(action[:14]).all()
    assert np.isfinite(action[14:]).all()
    assert np.array_equal(action[-3:], np.zeros(3, dtype=np.float32))

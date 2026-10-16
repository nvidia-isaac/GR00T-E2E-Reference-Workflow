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

"""Host-testable H2 PICO/OpenXR retargeting graph."""

from __future__ import annotations

from e2e_robot_workflows.robots.h2 import H2_ARENA_ACTION_DIM

H2_LEFT_FLANGE_ELEMENTS = (
    "left_pos_x",
    "left_pos_y",
    "left_pos_z",
    "left_quat_x",
    "left_quat_y",
    "left_quat_z",
    "left_quat_w",
)
H2_RIGHT_FLANGE_ELEMENTS = tuple(element.replace("left_", "right_") for element in H2_LEFT_FLANGE_ELEMENTS)
H2_VELOCITY_ELEMENTS = ("velocity_x", "velocity_y", "angular_velocity_z")
# Sharpa fingers extend along flange +Z; map that onto controller -Z. The
# mirrored X/Y axes follow the pinned URDF's left/right wrist-to-flange mounts
# (extrinsic RPY [-90, 0, -90] / [90, 0, 90]) after the XR-to-pelvis transform.
# Se3RetargeterConfig applies these offsets on the right as intrinsic XYZ.
H2_LEFT_CONTROLLER_ROTATION_OFFSET_RPY = (180.0, 0.0, 0.0)
H2_RIGHT_CONTROLLER_ROTATION_OFFSET_RPY = (0.0, 180.0, 0.0)


def build_h2_pink_openxr_pipeline():
    """Build the 61-D pelvis-anchor OpenXR pipeline used by H2 recording."""

    from isaacteleop.retargeters import Se3RetargeterConfig, TensorReorderer
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    from e2e_robot_workflows.devices.h2_motion_controller import (
        H2SharpaControllerHandsConfig,
        H2SharpaControllerHandsRetargeter,
        H2TrackingSafeSe3AbsRetargeter,
        H2VelocityCommandConfig,
        H2VelocityCommandRetargeter,
    )
    from e2e_robot_workflows.robots.h2 import load_h2_pink_contract

    contract = load_h2_pink_contract()
    controllers = ControllersSource(name="controllers")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    def connect_pose(side: str, offsets: tuple[float, float, float]):
        retargeter = H2TrackingSafeSe3AbsRetargeter(
            Se3RetargeterConfig(
                input_device=side,
                zero_out_xy_rotation=False,
                use_wrist_rotation=False,
                use_wrist_position=False,
                target_offset_roll=offsets[0],
                target_offset_pitch=offsets[1],
                target_offset_yaw=offsets[2],
            ),
            name=f"h2_{side}_flange_pose",
        )
        return retargeter.connect({side: transformed_controllers.output(side)})

    left_pose = connect_pose(ControllersSource.LEFT, H2_LEFT_CONTROLLER_ROTATION_OFFSET_RPY)
    right_pose = connect_pose(ControllersSource.RIGHT, H2_RIGHT_CONTROLLER_ROTATION_OFFSET_RPY)
    hands = H2SharpaControllerHandsRetargeter(
        H2SharpaControllerHandsConfig(
            joint_names=contract.hand_joint_order,
            open_joint_pos_by_name=contract.hand_open_targets,
            closed_joint_pos_by_name=contract.hand_closed_targets,
            tracking_loss_hold_samples=5,
        ),
        name="h2_sharpa_controller_hands",
    ).connect(
        {
            "controller_left": controllers.output(ControllersSource.LEFT),
            "controller_right": controllers.output(ControllersSource.RIGHT),
        }
    )
    velocity = H2VelocityCommandRetargeter(H2VelocityCommandConfig(), name="h2_velocity").connect(
        {
            "controller_left": controllers.output(ControllersSource.LEFT),
            "controller_right": controllers.output(ControllersSource.RIGHT),
        }
    )

    input_config = {
        "left_flange": list(H2_LEFT_FLANGE_ELEMENTS),
        "right_flange": list(H2_RIGHT_FLANGE_ELEMENTS),
        "hands": list(contract.hand_joint_order),
        "velocity": list(H2_VELOCITY_ELEMENTS),
    }
    output_order = [element for elements in input_config.values() for element in elements]
    if len(output_order) != H2_ARENA_ACTION_DIM or len(set(output_order)) != H2_ARENA_ACTION_DIM:
        raise ValueError("H2 OpenXR output must contain 61 uniquely named values")
    reorderer = TensorReorderer(
        input_config=input_config,
        output_order=output_order,
        name="h2_action_reorderer",
        input_types={
            "left_flange": "array",
            "right_flange": "array",
            "hands": "scalar",
            "velocity": "array",
        },
    ).connect(
        {
            "left_flange": left_pose.output("ee_pose"),
            "right_flange": right_pose.output("ee_pose"),
            "hands": hands.output("hand_joints"),
            "velocity": velocity.output("velocity"),
        }
    )
    return OutputCombiner({"action": reorderer.output("output")})


__all__ = ["H2_ARENA_ACTION_DIM", "build_h2_pink_openxr_pipeline"]

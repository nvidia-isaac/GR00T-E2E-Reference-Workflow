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

"""Static apple environment variant used for Isaac AutoData trajectory generation.

AutoData turns a handful of recorded demonstrations into a dataset by replaying each source
trajectory against a freshly randomized scene and keeping the attempts that work. That needs
three things the recording environment leaves out:

- A success condition. Recording accepts episodes on the operator's judgement, so
  ``G1StaticAppleEnvCfg`` clears ``terminations.success``. AutoData reads that term to score
  every generated attempt, and because it only checks that the attribute *exists*, leaving it
  ``None`` makes annotation silently pass every episode instead of failing loudly.
- A randomized object pose. Recording spawns the apple at a fixed spot, which would make each
  generated trial a copy of its source. Generation transforms the source trajectory onto a new
  apple pose, so the pose has to vary.
- A boolean signal per subtask boundary, in a ``subtask_terms`` observation group. Annotation
  samples these to find where one subtask ends, which is the alternative to a human marking the
  boundary in the viewer.
"""

from __future__ import annotations

from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils import configclass

from e2e_robot_workflows.robots.g1.config import load_g1_robot_config
from e2e_robot_workflows.tasks.static_apple.env_cfg import G1StaticAppleEnvCfg, WbcMode
from e2e_robot_workflows.tasks.static_apple.observations import object_held_by_link
from e2e_robot_workflows.tasks.static_apple.spec import (
    ENVIRONMENT,
    PICK_OBJECT_NAME,
    SHELF_SURFACE_Z,
    shelf_spawn_z,
)
from e2e_robot_workflows.tasks.static_apple.terminations import (
    object_placed_on_target,
)

MAX_OBJECT_HEIGHT_M = SHELF_SURFACE_Z + ENVIRONMENT.max_object_height_above_support_m
"""Above this the apple is still held rather than resting; the plate's own thickness is well
inside the margin."""

MIN_SUPPORTED_HEIGHT_M = SHELF_SURFACE_Z - ENVIRONMENT.min_supported_height_below_support_m
"""Lowest root height at which the apple and plate can still be supported by the table.

The margin permits normal contact-settling variation but rejects a plate or apple that fell to
the floor, where their XY alignment is not a placement.
"""

GENERATION_SPAWN_RANGE_M = 0.05
"""Half-width of the apple's spawn square during generation. Wide enough that DexMimicGen has
to retarget each source trajectory, small enough to stay within the G1's seated reach."""

_G1_TASK_LINK_NAMES = load_g1_robot_config().task_link_names
TASK_LINK_NAMES = {"left": _G1_TASK_LINK_NAMES[0], "right": _G1_TASK_LINK_NAMES[1]}
"""Links near each gripper's grasp centre, used for held and released tests."""

LIFT_CLEARANCE_M = ENVIRONMENT.pickup_lift_clearance_m
"""How far the apple must rise above its resting centre to count as picked up rather than
brushed. The transfer lifts it much further to clear the plate's rim."""

MIN_HELD_HEIGHT_M = shelf_spawn_z(PICK_OBJECT_NAME) + LIFT_CLEARANCE_M
"""Height above the environment origin that separates a held apple from one on the table."""


@configclass
class SubtaskTermsCfg(ObsGroup):
    """Subtask boundary signals read by AutoData's automatic annotation.

    Each term matches one hand-specific task descriptor. The launcher selects the hand that
    carries the apple, whose two subtasks split at pickup; the other hand has one support subtask.
    """

    left_hand_holds_object = ObsTerm(
        func=object_held_by_link,
        params={
            "task_link_name": TASK_LINK_NAMES["left"],
            # Held is the exact complement of the success term's release test, so an apple cannot
            # read as both held and placed.
            "grasp_distance_m": ENVIRONMENT.release_clearance_m,
            "min_object_height_m": MIN_HELD_HEIGHT_M,
        },
    )
    right_hand_holds_object = ObsTerm(
        func=object_held_by_link,
        params={
            "task_link_name": TASK_LINK_NAMES["right"],
            # Held is the exact complement of the success term's release test, so an apple cannot
            # read as both held and placed.
            "grasp_distance_m": ENVIRONMENT.release_clearance_m,
            "min_object_height_m": MIN_HELD_HEIGHT_M,
        },
    )

    def __post_init__(self):
        # AutoData looks the terms up by name, so the group must stay a dict of terms.
        self.concatenate_terms = False
        # Noise would make the boundary jitter between replays of the same demonstration.
        self.enable_corruption = False


@configclass
class G1StaticAppleAutoDataEnvCfg(G1StaticAppleEnvCfg):
    """Fixed-base static apple task configured for AutoData."""

    wbc_mode: WbcMode = "fixed"

    def __post_init__(self):
        super().__post_init__()

        self.observations.subtask_terms = SubtaskTermsCfg()

        self.terminations.success = DoneTerm(
            func=object_placed_on_target,
            params={
                # Contract-owned, shared with the recorder and with evaluation.
                "placement_radius_m": ENVIRONMENT.success_placement_xy_m,
                "placement_z_m": ENVIRONMENT.success_placement_z_m,
                "min_object_height_above_target_m": ENVIRONMENT.min_object_height_above_target_m,
                "max_object_height_m": MAX_OBJECT_HEIGHT_M,
                "min_object_height_m": MIN_SUPPORTED_HEIGHT_M,
                "min_target_height_m": MIN_SUPPORTED_HEIGHT_M,
                "min_contact_force": ENVIRONMENT.success_force_threshold,
                "release_clearance_m": ENVIRONMENT.release_clearance_m,
                "max_object_speed": ENVIRONMENT.success_velocity_threshold,
                "task_link_names": tuple(TASK_LINK_NAMES.values()),
            },
        )

        pose_range = self.events.reset_object.params["pose_range"]
        pose_range["x"] = (-GENERATION_SPAWN_RANGE_M, GENERATION_SPAWN_RANGE_M)
        pose_range["y"] = (-GENERATION_SPAWN_RANGE_M, GENERATION_SPAWN_RANGE_M)


@configclass
class G1StaticAppleAutoDataAgileEnvCfg(G1StaticAppleAutoDataEnvCfg):
    """Free-base static apple task configured for AutoData and AGILE replay."""

    wbc_mode: WbcMode = "agile"

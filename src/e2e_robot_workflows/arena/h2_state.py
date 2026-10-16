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

"""Restore the same H2 control state for Pink and joint-layout action replay."""

from __future__ import annotations

from e2e_robot_workflows.controllers.h2_feedback import validate_h2_initial_state


class H2ResetStateMixin:
    def reset_to(self, state, env_ids=None, seed=None, is_relative=False):
        # Validate before the parent mutates the physical scene or controller.
        validate_h2_initial_state(state)
        result = super().reset_to(state, env_ids=env_ids, seed=seed, is_relative=is_relative)
        self.action_manager.get_term("lower_body_joint_pos").restore_feedback_state(state["h2_agile_feedback"])
        targets = state["h2_actuator_targets"]
        robot = self.scene["robot"]
        robot.set_joint_position_target_index(target=targets["position"].to(self.device), env_ids=env_ids)
        robot.set_joint_velocity_target_index(target=targets["velocity"].to(self.device), env_ids=env_ids)
        self.scene.write_data_to_sim()
        self.observation_manager.reset(env_ids)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        self.recorder_manager.reset(env_ids)
        self.recorder_manager.record_post_reset([0] if env_ids is None else env_ids)
        return self.obs_buf, result[1]

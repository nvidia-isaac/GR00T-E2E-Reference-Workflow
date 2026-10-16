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

"""Build the demonstration recorder for an Arena-composed environment.

The task's ``mimic_recorder_config`` is copied when present. G1's monolithic WBC
action supplies locomotion inside its action recorder, so the separate navigation
and height terms are disabled. H2 keeps its separate navigation term and disables
only the height term its three-value AGILE command does not provide.
"""

from __future__ import annotations

import copy
from typing import Any

from isaaclab_arena.utils.isaaclab_utils.recorders import ArenaEnvRecorderManagerCfg

from e2e_robot_workflows.data.recorders import add_arena_datagen_info_recorder_term, add_workflow_recorder_terms


def make_recorder_manager_cfg(env_cfg: Any) -> ArenaEnvRecorderManagerCfg:
    """Return the recorder manager for recording on an Arena-composed ``env_cfg``.

    This app's terms arrive through the same
    :func:`~e2e_robot_workflows.data.recorders.add_workflow_recorder_terms` seam
    AutoData generation uses, which leaves a name the caller already set alone, so the
    terms switched off below stay off.

    Returns:
        Arena's demonstration recorder, extended with this app's applicable terms.
    """

    mimic_recorders = getattr(env_cfg, "mimic_recorder_config", None)
    recorders = copy.deepcopy(mimic_recorders) if mimic_recorders is not None else ArenaEnvRecorderManagerCfg()
    lower_body_cfg = getattr(env_cfg.actions, "lower_body_joint_pos", None)
    if lower_body_cfg is None:
        recorders.record_post_step_navigate_cmd = None
        recorders.record_post_step_base_height_cmd = None
    elif getattr(lower_body_cfg, "command_dim", 0) <= 3:
        # H2's export owns only [vx, vy, wz]; it has no height channel.
        recorders.record_post_step_base_height_cmd = None
    if not hasattr(env_cfg.observations, "camera_obs"):
        recorders.record_pre_step_flat_camera_observations = None
    add_workflow_recorder_terms(recorders)
    if mimic_recorders is not None:
        add_arena_datagen_info_recorder_term(recorders)
    return recorders


__all__ = ["make_recorder_manager_cfg"]

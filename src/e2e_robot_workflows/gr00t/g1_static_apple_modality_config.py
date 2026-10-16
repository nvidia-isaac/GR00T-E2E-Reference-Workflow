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

"""GR00T modality config for the G1 static-apple embodiment (GR00T N1.7).

Registers the modality/action layout GR00T reads when fine-tuning against the
LeRobot dataset produced by ``scripts/convert_hdf5_to_lerobot.py``. The group
names match the dataset's ``meta/modality.json``:

- state  = arm/hand/waist joints (sliced from ``observation.state``)
- action = those joints plus the AGILE ``base_height_command`` /
  ``navigate_command`` (from the ``teleop.*`` columns the converter emits)
- video  = ``ego_view``; language = ``annotation.human.task_description``

Action horizon is ``range(40)`` to match GR00T-N1.7-3B's ``max_action_horizon``
(40). The built-in ``unitree_g1_full_body_with_waist_height_nav_cmd`` uses
``range(50)`` (tuned for a 50-horizon model) and crashes N1.7 with a
``negative dimension`` tensor in the data processor.

Actions are ABSOLUTE (the recorder captures absolute joint targets); this mirrors
Arena's validated N1.7 recipe and avoids the relative-action transform.

Usage (from the GR00T repo, in the GR00T env), passing the absolute path to this
file for ``--modality-config-path``::

    --embodiment-tag new_embodiment \\
    --modality-config-path <e2e-repo>/src/e2e_robot_workflows/gr00t/g1_static_apple_modality_config.py

This module imports ``gr00t.*`` and is only importable inside the GR00T training
environment -- it is loaded by GR00T's ``--modality-config-path`` (via importlib),
never by this package.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

_ACTION_HORIZON = 40  # GR00T-N1.7-3B max_action_horizon


def _absolute() -> ActionConfig:
    """An absolute, non-EEF action group (joint targets / scalar commands)."""

    return ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
    )


# Leg joints are deliberately not action groups: AGILE drives the lower body from
# the base_height_command / navigate_command the policy predicts, so the action
# set is upper-body joints + those two locomotion commands, not per-leg targets.
_ACTION_GROUPS = [
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "waist",
    "base_height_command",
    "navigate_command",
]

g1_static_apple_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand", "waist"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(_ACTION_HORIZON)),
        modality_keys=_ACTION_GROUPS,
        action_configs=[_absolute() for _ in _ACTION_GROUPS],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(g1_static_apple_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

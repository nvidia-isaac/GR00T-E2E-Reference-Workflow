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

"""Environment factory that keeps this repository's recorded fields in AutoData runs.

Isaac AutoData configures the environment itself: ``setup_env_config`` assigns
``env_cfg.recorders`` unconditionally, using Isaac Lab's ``ActionStateRecorderManagerCfg``
unless its caller supplies a recorder, and ``scripts/generate_dataset.py`` never supplies one.
Whatever recorder the env cfg declares is therefore discarded before the environment is built.

That matters because the two recorders do not describe the same thing. Isaac Lab's records
``processed_actions``, the concatenation of each *active action term's* processed actions --
in fixed mode the Pink IK term alone, so the 14 arm and 14 hand joints it drives. Ours records
``joint_pos_target``, the command the articulation received across all 43 joints, in articulation
order. The converter's joint maps and the GR00T modality template are written against the latter,
so a dataset carrying only the former cannot be converted.

The gym id is the seam this repository still owns, so the terms are added here, on the config
AutoData hands to the environment.
"""

from __future__ import annotations

from typing import Any

from e2e_robot_workflows.data.recorders import add_workflow_recorder_terms


def make_autodata_env(cfg: Any, render_mode: str | None = None, **kwargs: Any) -> Any:
    """Build the static-apple environment with this repository's recorder terms in place.

    Mirrors ``ManagerBasedRLEnv``'s own signature, which this replaces as the id's entry point,
    so callers that pass ``render_mode`` keep working.
    """

    from isaaclab.envs import ManagerBasedRLEnv

    recorders = getattr(cfg, "recorders", None)
    if recorders is not None:
        add_workflow_recorder_terms(recorders)
    return ManagerBasedRLEnv(cfg=cfg, render_mode=render_mode, **kwargs)

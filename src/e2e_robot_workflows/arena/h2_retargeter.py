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

"""Arena registration for the H2 PICO/OpenXR retargeting graph."""

from __future__ import annotations

from isaaclab_arena.assets.register import register_retargeter
from isaaclab_arena.assets.retargeter_library import RetargetterBase

from e2e_robot_workflows.arena.embodiment_capabilities import H2_ARENA_TELEOP_EMBODIMENT
from e2e_robot_workflows.arena.h2_openxr import build_h2_pink_openxr_pipeline


@register_retargeter
class H2PinkOpenXRRetargeter(RetargetterBase):
    """Bind OpenXR exclusively to the Arena H2 Pink embodiment."""

    device = "openxr"
    embodiment = H2_ARENA_TELEOP_EMBODIMENT

    def get_pipeline_builder(self, embodiment: object):
        del embodiment
        return build_h2_pink_openxr_pipeline


__all__ = ["H2PinkOpenXRRetargeter"]

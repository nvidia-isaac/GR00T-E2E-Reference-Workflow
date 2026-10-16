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

"""Compatibility action for AGILE policy inference on Isaac Lab 3.0."""

from __future__ import annotations

import torch
from isaaclab_tasks.manager_based.locomanipulation.pick_place.mdp.actions import AgileBasedLowerBodyAction


class InferenceAgileBasedLowerBodyAction(AgileBasedLowerBodyAction):
    """Run Isaac Lab's AGILE policy without constructing an autograd graph.

    Isaac Lab 3.0 Beta 2 calls the TorchScript policy with gradients enabled.
    Its output is passed directly to Warp, which rejects CUDA tensors that
    require gradients. The policy is used only for runtime inference here.

    Also exposes the raw locomotion command ``[vx, vy, wz, hip_height]`` fed to
    the AGILE policy so the recorder can capture it as a first-class field (the
    navigate + base-height command GR00T's action head consumes). It is the
    command *input*, distinct from the joint targets the policy produces.
    """

    # Populated on the first ``process_actions``; ``None`` before then so the
    # recorder's ``getattr`` guard sees a defined attribute (skips recording
    # until the first action step) instead of risking an ``AttributeError``.
    locomotion_command: torch.Tensor | None = None

    @torch.no_grad()
    def process_actions(self, actions: torch.Tensor) -> None:
        self.locomotion_command = actions.clone()
        super().process_actions(actions)


__all__ = ["InferenceAgileBasedLowerBodyAction"]

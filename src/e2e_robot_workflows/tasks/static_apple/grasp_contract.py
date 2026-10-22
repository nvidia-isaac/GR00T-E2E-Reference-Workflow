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

"""Simulator-independent provenance for H2 held-object labels."""

from __future__ import annotations

from dataclasses import dataclass

H2_GRASP_MODE = "h2_contact_and_closure_v1"


@dataclass(frozen=True)
class H2GraspCriteria:
    """Shared grasp-label contract; geometry remains owned by the task definition."""

    contact_force_threshold_n: float = 0.01
    minimum_command_closure_fraction: float = 0.5
    minimum_measured_closure_fraction: float = 0.1
    contact_history_samples: int = 3
    contact_sample_period: str = "control_period"
    contact_aggregation: str = "maximum_per_body_normal_force_norm"
    contact_bodies_per_hand: int = 24
    primary_flexion_fingers: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")


H2_GRASP_CRITERIA = H2GraspCriteria()

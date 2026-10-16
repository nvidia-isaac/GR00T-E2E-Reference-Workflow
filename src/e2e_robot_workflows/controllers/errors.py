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

"""Controller faults that a live recorder can recover from by discarding an episode."""


class H2MeasuredJointLimitViolation(RuntimeError):
    """Measured H2 state is outside the Pink model's accepted joint limits.

    This does not permit stepping with the invalid state. Live XR recording may
    discard and reset the episode; generation may reject the attempt and reset.
    Replay and behavior validation must still fail.
    """

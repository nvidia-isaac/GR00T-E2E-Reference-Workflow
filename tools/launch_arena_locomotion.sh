#!/usr/bin/env bash
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

set -euo pipefail

# Drive the G1 under Arena's whole-body controller on the Arena-composed environment.
# Every argument is forwarded to scripts/locomotion_with_agile.py, which is launched with
# --arena already set.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--check" ]]; then
  exec "${REPO_ROOT}/tools/arena_launch.sh" --check
fi

exec "${REPO_ROOT}/tools/arena_launch.sh" \
  python "${REPO_ROOT}/scripts/locomotion_with_agile.py" --arena "$@"

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

# Drive the scripted locomotion phases on the Arena-composed environment, once per
# embodiment. The drive runs through tools/arena_launch.sh, which overlays the pinned
# Arena checkout onto this repository's environment. Both layouts are driven: they take
# the command in the same locomotion tail but differ in everything ahead of it, so a
# break in either is a break in this path.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_BASE=${ARENA_LOCOMOTION_RESULT_DIR:-"${REPO_ROOT}/test-results/arena-locomotion"}
if [[ "${RESULT_BASE}" != /* ]]; then
  RESULT_BASE="${REPO_ROOT}/${RESULT_BASE}"
fi
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="${RESULT_BASE}/${RUN_ID}"
EMBODIMENTS=(g1_wbc_agile_joint g1_wbc_agile_pink)

mkdir -p "${RESULT_DIR}"
export ACCEPT_EULA=${ACCEPT_EULA:-Y}
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}

cd "${REPO_ROOT}"

for embodiment in "${EMBODIMENTS[@]}"; do
  ./tools/arena_launch.sh \
    python tools/simulation/validate_arena_locomotion.py \
    --embodiment "${embodiment}" \
    --result-path "${RESULT_DIR}/${embodiment}-metrics.json" \
    2>&1 | tee "${RESULT_DIR}/${embodiment}.log"
done

printf 'Arena locomotion validation passed. Artifacts: %s\n' "${RESULT_DIR}"

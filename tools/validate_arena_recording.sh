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

# Record a short demonstration on the Arena-composed environment, once per embodiment,
# and check what came out. The recording runs through tools/launch_recorder.sh,
# which overlays the pinned Arena checkout onto this repository's environment; the
# checks afterwards run in that same environment. The take is finite and scripted, so
# it needs no headset.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
RESULT_BASE=${ARENA_RECORDING_RESULT_DIR:-"${REPO_ROOT}/test-results/arena-recording"}
if [[ "${RESULT_BASE}" != /* ]]; then
  RESULT_BASE="${REPO_ROOT}/${RESULT_BASE}"
fi
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="${RESULT_BASE}/${RUN_ID}"
# Both layouts are recorded: the action space differs between them, and so does the
# command that holds the scene still, so a break in either is a break in this path.
EMBODIMENTS=(g1_wbc_agile_pink g1_wbc_agile_joint)
# Only the Pink recording carries the end-effector and teleoperation columns GR00T's
# modality config asks for, so it is the one held to the conversion contract.
GR00T_EMBODIMENT=g1_wbc_agile_pink

mkdir -p "${RESULT_DIR}"
export ACCEPT_EULA=${ACCEPT_EULA:-Y}
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}

cd "${REPO_ROOT}"

datasets=()
for embodiment in "${EMBODIMENTS[@]}"; do
  dataset="${RESULT_DIR}/${embodiment}.hdf5"
  ./tools/launch_recorder.sh \
    --arena \
    --headless \
    --enable_cameras \
    --arena_embodiment "${embodiment}" \
    --sample_steps 300 \
    --sample_export \
    --dataset_file "${dataset}" \
    2>&1 | tee "${RESULT_DIR}/${embodiment}.log"
  datasets+=("--dataset" "${embodiment}=${dataset}")
done

"${PYTHON_BIN}" tools/simulation/check_arena_recording.py \
  --result-path "${RESULT_DIR}/arena-recording-metrics.json" \
  "${datasets[@]}"

# The dataset contract the fine-tuning stage actually consumes: --strict fails when a
# configured state, action, or camera field does not resolve from the recorded schema.
"${PYTHON_BIN}" scripts/convert_hdf5_to_lerobot.py \
  --inspect \
  --strict \
  --input "${RESULT_DIR}/${GR00T_EMBODIMENT}.hdf5" \
  --report "${RESULT_DIR}/lerobot-inspection.json" \
  >"${RESULT_DIR}/lerobot-inspection.log"

printf 'Arena recording validation passed. Artifacts: %s\n' "${RESULT_DIR}"

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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
RESULT_BASE=${COMMON_ENV_RESULT_DIR:-"${REPO_ROOT}/test-results/common-environment"}
if [[ "${RESULT_BASE}" != /* ]]; then
  RESULT_BASE="${REPO_ROOT}/${RESULT_BASE}"
fi
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="${RESULT_BASE}/${RUN_ID}"

mkdir -p "${RESULT_DIR}"
export ACCEPT_EULA=${ACCEPT_EULA:-Y}
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}

cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/record_demos.py \
  --teleop_device keyboard \
  --sample_steps 300 \
  --sample_action wrist_sine \
  --sample_export \
  --dataset_file "${RESULT_DIR}/recorder-sample.hdf5" \
  --environment_snapshot "${RESULT_DIR}/recording-environment.json" \
  2>&1 | tee "${RESULT_DIR}/recording.log"

./tools/launch_arena_eval.sh \
  --num_episodes 1 \
  --policy_type e2e_robot_workflows.arena.g1_smoke_policy.G1StandingSineSmokePolicy \
  --output_base_dir "${RESULT_DIR}/arena" \
  --environment_snapshot "${RESULT_DIR}/arena-environment.json" \
  --no_solve_relations \
  -- \
  --disable_task_terminations \
  2>&1 | tee "${RESULT_DIR}/arena.log"

arena_results="$(find "${RESULT_DIR}/arena" -name episode_results_rank0.jsonl -type f -print -quit)"
if [[ -z "${arena_results}" || ! -s "${arena_results}" ]]; then
  echo "Arena policy runner did not write non-empty episode results." >&2
  exit 1
fi

"${PYTHON_BIN}" tools/compare_environment_snapshots.py \
  "${RESULT_DIR}/recording-environment.json" \
  "${RESULT_DIR}/arena-environment.json" \
  | tee "${RESULT_DIR}/environment-comparison.json"

printf 'Common environment validation passed. Artifacts: %s\n' "${RESULT_DIR}"

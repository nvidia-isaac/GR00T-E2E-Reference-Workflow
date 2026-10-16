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
RESULT_BASE=${ARENA_H2_RESULT_DIR:-"${REPO_ROOT}/test-results/arena-h2"}
GR00T_ROOT=${ISAAC_GR00T_ROOT:-}
VALIDATION_PHASE=${E2E_H2_VALIDATION_PHASE:-all}
case "${VALIDATION_PHASE}" in
  all|startup|joint|pink-mimic|pipeline) ;;
  *) echo "E2E_H2_VALIDATION_PHASE must be all, startup, joint, pink-mimic, or pipeline." >&2; exit 2 ;;
esac
if [[ "${RESULT_BASE}" != /* ]]; then
  RESULT_BASE="${REPO_ROOT}/${RESULT_BASE}"
fi
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="${RESULT_BASE}/${RUN_ID}"

export ACCEPT_EULA=${ACCEPT_EULA:-Y}
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
[[ -n "${GR00T_ROOT}" ]] || { echo "Set ISAAC_GR00T_ROOT to the pinned, synced GR00T checkout." >&2; exit 2; }
[[ -d "${GR00T_ROOT}" ]] || { echo "ISAAC_GR00T_ROOT is not a directory: ${GR00T_ROOT}" >&2; exit 2; }
[[ -f "${GR00T_ROOT}/gr00t/data/stats.py" ]] || {
  echo "ISAAC_GR00T_ROOT is not a GR00T checkout (missing gr00t/data/stats.py): ${GR00T_ROOT}" >&2
  exit 2
}
mkdir -p "${RESULT_DIR}"
cd "${REPO_ROOT}"

run_case() {
  local label=$1
  shift
  if [[ "${VALIDATION_PHASE}" != all && "${VALIDATION_PHASE}" != "${label}" ]]; then
    return
  fi
  local result_path="${RESULT_DIR}/${label}-metrics.json"
  local log_path="${RESULT_DIR}/${label}.log"
  rm -f "${result_path}" "${log_path}"
  timeout --signal=TERM --kill-after=30s 300s ./tools/arena_launch.sh \
    python tools/simulation/validate_arena_h2.py \
    "$@" \
    --result-path "${result_path}" \
    2>&1 | tee "${log_path}"
  python -c \
    'import json, sys; result=json.load(open(sys.argv[1], encoding="utf-8")); valid=(result.get("status") == "ok" and result.get("stage") == "complete" and result.get("completed_steps") == result.get("requested_steps") and result.get("simulation_app_close_requested") is True and not result.get("cleanup_errors")); raise SystemExit(not valid)' \
    "${result_path}"
}

run_case startup --embodiment h2_agile_pink --enable-cameras --mimic --startup-only
run_case joint --embodiment h2_agile_joint --steps 200
run_case pink-mimic --embodiment h2_agile_pink --steps 60 --enable-cameras --mimic --station-keeping
if [[ "${VALIDATION_PHASE}" != all && "${VALIDATION_PHASE}" != pipeline ]]; then
  printf 'H2 %s validation passed (other phases not run). Artifacts: %s\n' "${VALIDATION_PHASE}" "${RESULT_DIR}"
  exit 0
fi

dataset_path="${RESULT_DIR}/h2_agile_pink.hdf5"
action_replay_result="${RESULT_DIR}/action-replay-metrics.json"
state_replay_result="${RESULT_DIR}/state-replay-metrics.json"
performance_result="${RESULT_DIR}/recording-performance.json"
recording_steps=300
rm -f "${dataset_path}" "${action_replay_result}" "${state_replay_result}" "${performance_result}"
timeout --signal=TERM --kill-after=30s 300s ./tools/launch_recorder.sh \
  --arena \
  --viz none \
  --enable_cameras \
  --device cpu \
  --arena_embodiment h2_agile_pink \
  --sample_steps "${recording_steps}" \
  --sample_export \
  --sample_performance_result "${performance_result}" \
  --dataset_file "${dataset_path}" \
  2>&1 | tee "${RESULT_DIR}/recording.log"

python -c \
  'import json,math,sys; r=json.load(open(sys.argv[1], encoding="utf-8")); valid=(r.get("status")=="ok" and r.get("steps")==int(sys.argv[2]) and r.get("warmup_steps")==30 and r.get("device")=="cpu" and math.isclose(r.get("physics_dt_s",0),0.005,abs_tol=1e-12) and r.get("decimation")==2 and r.get("render_interval")==4 and math.isclose(r.get("control_period_s",0),0.01,abs_tol=1e-12) and r.get("realtime_factor",0)>=0.35 and r.get("completion_policy_evaluated") is True and r.get("teleop_advance_included") is False); raise SystemExit(not valid)' \
  "${performance_result}" "${recording_steps}"

python -c \
  'import sys; from e2e_robot_workflows.data.dataset_manifest import validate_dataset; manifest=validate_dataset(sys.argv[1], require_lane="arena", require_stage="recording"); raise SystemExit(manifest["action"]["controller"] != "h2_agile_pink")' \
  "${dataset_path}"

timeout --signal=TERM --kill-after=30s 300s ./tools/launch_replay.sh \
  --lane arena \
  --viz none \
  --strict \
  --expected_episodes 1 \
  --dataset_file "${dataset_path}" \
  --result_file "${action_replay_result}" \
  2>&1 | tee "${RESULT_DIR}/action-replay.log"
python -c \
  'import json, sys; result=json.load(open(sys.argv[1], encoding="utf-8")); valid=(result.get("status") == "ok" and result.get("replay_mode") == "action" and result.get("completed_replays") == result.get("expected_replays") == 1 and result.get("total_frames") == int(sys.argv[2]) and not result.get("cleanup_errors")); raise SystemExit(not valid)' \
  "${action_replay_result}" "${recording_steps}"

timeout --signal=TERM --kill-after=30s 300s ./tools/launch_replay.sh \
  --lane arena \
  --replay-mode state \
  --viz none \
  --strict \
  --expected_episodes 1 \
  --dataset_file "${dataset_path}" \
  --result_file "${state_replay_result}" \
  2>&1 | tee "${RESULT_DIR}/state-replay.log"
python -c \
  'import json, sys; result=json.load(open(sys.argv[1], encoding="utf-8")); valid=(result.get("status") == "ok" and result.get("replay_mode") == "state" and result.get("completed_replays") == result.get("expected_replays") == 1 and result.get("total_frames") == int(sys.argv[2]) and not result.get("cleanup_errors")); raise SystemExit(not valid)' \
  "${state_replay_result}" "${recording_steps}"

lerobot_root="${RESULT_DIR}/lerobot"
uv run --frozen --no-sync --extra converter python scripts/convert_hdf5_to_lerobot.py \
  --strict \
  --input "${dataset_path}" \
  --output "${lerobot_root}" \
  --report "${RESULT_DIR}/lerobot-write.json" \
  >"${RESULT_DIR}/lerobot-write.log"
python -c \
  'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); w=r.get("write_result",{}); valid=(r.get("status")=="ok" and r["mapping_report"]["ready_for_writer"] is True and w.get("status")=="ok" and w.get("episodes")==1 and w.get("frames")==int(sys.argv[2]) and not w.get("warnings") and not w.get("dropped_modality_groups")); raise SystemExit(not valid)' \
  "${RESULT_DIR}/lerobot-write.json" "${recording_steps}"

modality_config="${REPO_ROOT}/src/e2e_robot_workflows/gr00t/h2_static_apple_modality_config.py"
./tools/gr00t_runtime.sh --gr00t-root "${GR00T_ROOT}" -- \
  python "${GR00T_ROOT}/gr00t/data/stats.py" \
  --dataset-path "${lerobot_root}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "${modality_config}" \
  2>&1 | tee "${RESULT_DIR}/gr00t-stats.log"
./tools/gr00t_runtime.sh --gr00t-root "${GR00T_ROOT}" -- \
  python tools/check_h2_gr00t_dataset.py \
  --dataset "${lerobot_root}" \
  --modality-config "${modality_config}" \
  --expected-episodes 1 \
  --expected-frames "${recording_steps}" \
  --result "${RESULT_DIR}/gr00t-loader.json" \
  2>&1 | tee "${RESULT_DIR}/gr00t-loader.log"

arena_root=${E2E_ARENA_ROOT:-$(./tools/fetch_arena_checkout.sh)}
./tools/launch_gr00t_eval.sh \
  --arena-root "${arena_root}" \
  --gr00t-root "${GR00T_ROOT}" \
  --robot h2 \
  --check \
  2>&1 | tee "${RESULT_DIR}/gr00t-eval-preflight.log"

printf 'Arena H2 %s validation passed. Artifacts: %s\n' "${VALIDATION_PHASE}" "${RESULT_DIR}"

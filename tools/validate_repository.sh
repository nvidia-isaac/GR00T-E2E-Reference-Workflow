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
export PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-/tmp/e2e-reference-app-pycache}

usage() {
  cat <<'EOF_USAGE'
Usage: ./tools/validate_repository.sh {static|test|host|docs|runtime-preflight|sim-static-apple|sim-agile-locomotion|sim-recorder|sim-common-environment|sim-arena-recording|sim-arena-locomotion|sim-arena-table-height-dr|sim-arena-h2}

sim-arena-h2 needs E2E_H2_ASSET_CACHE, E2E_H2_PINK_URDF, E2E_H2_AGILE_POLICY, and ISAAC_GR00T_ROOT.

Run this command through uv, for example: uv run ./tools/validate_repository.sh host.
EOF_USAGE
}

run_static_checks() {
  "${PYTHON_BIN}" -m pre_commit run --all-files --show-diff-on-failure
}

run_tests() {
  local -a pytest_args=(-q -p no:cacheprovider tests)
  if [[ -n "${JUNIT_XML:-}" ]]; then
    mkdir -p "$(dirname "${JUNIT_XML}")"
    pytest_args+=("--junitxml=${JUNIT_XML}")
  fi
  "${PYTHON_BIN}" -m pytest "${pytest_args[@]}"
}

run_docs() {
  if ! "${PYTHON_BIN}" -c "import sphinx" >/dev/null 2>&1; then
    printf '%s\n' \
      "Documentation dependencies are missing." \
      "Run 'uv sync --frozen' for the standard environment or" \
      "'uv sync --frozen --no-default-groups --only-group docs' for a docs-only environment." >&2
    return 1
  fi

  "${PYTHON_BIN}" -m sphinx -a -E -n -W --keep-going -b html docs docs/_build/html
  "${PYTHON_BIN}" -m sphinx -a -E -W --keep-going -b linkcheck docs docs/_build/linkcheck
}

write_sim_validation_junit() {
  local junit_path=$1 status=$2 testcase_name=$3
  mkdir -p "$(dirname "${junit_path}")"
  if [[ "${status}" -eq 0 ]]; then
    printf '%s\n' '<?xml version="1.0" encoding="UTF-8"?>' '<testsuite name="gpu-simulation-validation" tests="1" failures="0">' "  <testcase classname=\"simulation\" name=\"${testcase_name}\" />" '</testsuite>' >"${junit_path}"
  else
    printf '%s\n' '<?xml version="1.0" encoding="UTF-8"?>' '<testsuite name="gpu-simulation-validation" tests="1" failures="1">' "  <testcase classname=\"simulation\" name=\"${testcase_name}\">" '    <failure message="simulation validation command failed" />' '  </testcase>' '</testsuite>' >"${junit_path}"
  fi
}

run_simulation_validation() {
  local script=$1 result_path=$2 log_path=$3 junit_path=$4 testcase_name=$5
  mkdir -p "$(dirname "${result_path}")" "$(dirname "${log_path}")"
  rm -f "${result_path}" "${log_path}" "${junit_path}"
  local -a pipeline_status
  if RESULT_PATH="${result_path}" "${PYTHON_BIN}" "${script}" 2>&1 | tee "${log_path}"; then
    pipeline_status=("${PIPESTATUS[@]}")
  else
    pipeline_status=("${PIPESTATUS[@]}")
  fi
  local status=${pipeline_status[0]}
  if [[ "${pipeline_status[1]}" -ne 0 ]]; then
    status=${pipeline_status[1]}
  fi
  if [[ ! -s "${result_path}" ]] || ! "${PYTHON_BIN}" -c 'import json, sys; raise SystemExit(json.load(open(sys.argv[1], encoding="utf-8")).get("status") != "ok")' "${result_path}"; then
    status=1
  fi
  write_sim_validation_junit "${junit_path}" "${status}" "${testcase_name}"
  return "${status}"
}

require_h2_assets() {
  local -a missing=()
  [[ -n "${E2E_H2_ASSET_CACHE:-}" ]] || missing+=("E2E_H2_ASSET_CACHE")
  [[ -n "${E2E_H2_PINK_URDF:-}" ]] || missing+=("E2E_H2_PINK_URDF")
  [[ -n "${E2E_H2_AGILE_POLICY:-}" ]] || missing+=("E2E_H2_AGILE_POLICY")
  [[ -n "${ISAAC_GR00T_ROOT:-}" ]] || missing+=("ISAAC_GR00T_ROOT")
  if [[ "${#missing[@]}" -gt 0 ]]; then
    printf '%s\n' \
      "The Arena H2 check needs its materialized plant, Pink sidecar, AGILE checkpoint, and pinned GR00T checkout." \
      "Set: ${missing[*]}" >&2
    return 2
  fi
}

cd "${REPO_ROOT}"
case "${1:-}" in
  static) run_static_checks ;;
  test) run_tests ;;
  host) run_static_checks; run_tests ;;
  docs) run_docs ;;
  runtime-preflight) "${PYTHON_BIN}" tools/verify_runtime.py ;;
  sim-static-apple) run_simulation_validation tools/simulation/validate_static_apple.py "${STATIC_APPLE_VALIDATION_RESULT:-test-results/simulation/static-apple-metrics.json}" "${STATIC_APPLE_VALIDATION_LOG:-test-results/simulation/static-apple-validation.log}" "${STATIC_APPLE_VALIDATION_JUNIT:-${JUNIT_XML:-test-results/simulation/static-apple-junit.xml}}" static_apple_validation ;;
  sim-agile-locomotion) run_simulation_validation tools/simulation/validate_agile_locomotion.py "${AGILE_LOCOMOTION_VALIDATION_RESULT:-test-results/simulation/agile-metrics.json}" "${AGILE_LOCOMOTION_VALIDATION_LOG:-test-results/simulation/agile-validation.log}" "${AGILE_LOCOMOTION_VALIDATION_JUNIT:-${JUNIT_XML:-test-results/simulation/agile-junit.xml}}" agile_locomotion_validation ;;
  sim-recorder) run_simulation_validation tools/simulation/validate_recorder.py "${RECORDER_VALIDATION_RESULT:-test-results/simulation/recorder-metrics.json}" "${RECORDER_VALIDATION_LOG:-test-results/simulation/recorder-validation.log}" "${RECORDER_VALIDATION_JUNIT:-${JUNIT_XML:-test-results/simulation/recorder-junit.xml}}" recorder_validation ;;
  sim-common-environment) ./tools/validate_common_environment.sh ;;
  sim-arena-recording) ./tools/validate_arena_recording.sh ;;
  sim-arena-locomotion) ./tools/validate_arena_locomotion.sh ;;
  sim-arena-table-height-dr) ./tools/validate_arena_table_height.sh ;;
  sim-arena-h2) require_h2_assets; ./tools/validate_arena_h2.sh ;;
  -h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac

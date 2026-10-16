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
ARENA_ROOT=${E2E_ARENA_ROOT:-}
GR00T_ROOT=${ISAAC_GR00T_ROOT:-}
MODEL_PATH=${E2E_GR00T_MODEL_PATH:-}
CHECK_ONLY=false
ROBOT=g1
EVAL_ARGS=()
USER_ENVIRONMENT_ARGS=()
AFTER_SEPARATOR=false

while [[ "$#" -gt 0 ]]; do
  if [[ "${AFTER_SEPARATOR}" == true ]]; then
    USER_ENVIRONMENT_ARGS+=("$1")
    shift
    continue
  fi
  case "$1" in
    --arena-root)
      [[ "$#" -ge 2 ]] || { echo "--arena-root requires a path." >&2; exit 2; }
      ARENA_ROOT=$2
      shift 2
      ;;
    --arena-root=*)
      ARENA_ROOT=${1#*=}
      shift
      ;;
    --model-path)
      [[ "$#" -ge 2 ]] || { echo "--model-path requires a path." >&2; exit 2; }
      MODEL_PATH=$2
      shift 2
      ;;
    --model-path=*)
      MODEL_PATH=${1#*=}
      shift
      ;;
    --gr00t-root)
      [[ "$#" -ge 2 ]] || { echo "--gr00t-root requires a path." >&2; exit 2; }
      GR00T_ROOT=$2
      shift 2
      ;;
    --gr00t-root=*)
      GR00T_ROOT=${1#*=}
      shift
      ;;
    --check)
      CHECK_ONLY=true
      shift
      ;;
    --robot)
      [[ "$#" -ge 2 ]] || { echo "--robot requires g1 or h2." >&2; exit 2; }
      ROBOT=$2
      shift 2
      ;;
    --robot=*)
      ROBOT=${1#*=}
      shift
      ;;
    --)
      AFTER_SEPARATOR=true
      shift
      ;;
    *)
      EVAL_ARGS+=("$1")
      shift
      ;;
  esac
done

required_environment_args=()
case "${ROBOT}" in
  g1)
    policy_config="${REPO_ROOT}/configs/stages/gr00t/closedloop_eval.yaml"
    policy_type=isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy
    ;;
  h2)
    policy_config="${REPO_ROOT}/configs/stages/gr00t/h2_closedloop_eval.yaml"
    policy_type=e2e_robot_workflows.gr00t.h2_remote_policy.H2Gr00tRemotePolicy
    required_environment_args=(--embodiment h2_agile_joint)
    ;;
  *)
    echo "--robot must be g1 or h2, got: ${ROBOT}" >&2
    exit 2
    ;;
esac

reject_launcher_owned_args() {
  local argument option
  for argument in "$@"; do
    option=${argument%%=*}
    case "${option}" in
      --policy_type|--policy_config_yaml_path|--enable_cameras)
        echo "${option} is owned by launch_gr00t_eval.sh and cannot be overridden." >&2
        exit 2
        ;;
      --embodiment)
        if [[ "${ROBOT}" == h2 ]]; then
          echo "${option} is owned by launch_gr00t_eval.sh for H2 and cannot be overridden." >&2
          exit 2
        fi
        ;;
    esac
  done
}

reject_launcher_owned_args "${EVAL_ARGS[@]}"
reject_launcher_owned_args "${USER_ENVIRONMENT_ARGS[@]}"

if [[ "${CHECK_ONLY}" == true ]]; then
  if [[ "${#EVAL_ARGS[@]}" -ne 0 ]]; then
    echo "--check does not accept evaluation arguments." >&2
    exit 2
  fi
  if [[ "${#USER_ENVIRONMENT_ARGS[@]}" -ne 0 ]]; then
    echo "--check does not accept environment arguments." >&2
    exit 2
  fi
fi

# User options go first so the launcher-owned H2 embodiment remains authoritative even
# when argparse accepts an abbreviated spelling of the option.
environment_args=("${USER_ENVIRONMENT_ARGS[@]}" "${required_environment_args[@]}")

if [[ -z "${ARENA_ROOT}" ]]; then
  echo "Pass --arena-root with the pinned Isaac Lab Arena checkout." >&2
  exit 2
fi

gr00t_source=${GR00T_ROOT:-${ARENA_ROOT}/submodules/Isaac-GR00T}
if [[ ! -f "${gr00t_source}/gr00t/policy/server_client.py" ]]; then
  echo "Pass --gr00t-root with the pinned standalone Isaac-GR00T checkout, or use an" >&2
  echo "Arena checkout whose Isaac-GR00T submodule is populated." >&2
  exit 2
fi
expected_gr00t_revision="$(${UV_BIN:-uv} run --project "${REPO_ROOT}" --frozen --no-sync python -c \
  'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["tested"]["isaac_gr00t_revision"])' \
  "${REPO_ROOT}/runtime.toml")"
actual_gr00t_revision="$(git -C "${gr00t_source}" rev-parse HEAD)"
if [[ "${actual_gr00t_revision}" != "${expected_gr00t_revision}" ]]; then
  echo "Isaac-GR00T revision mismatch: expected ${expected_gr00t_revision}, got ${actual_gr00t_revision}" >&2
  exit 2
fi
if [[ -n "$(git -C "${gr00t_source}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Isaac-GR00T checkout must be clean: ${gr00t_source}" >&2
  exit 2
fi

export PYTHONPATH="${gr00t_source}"
export E2E_UV_EXTRA=gr00t
export E2E_ARENA_ROOT="${ARENA_ROOT}"
# Arena resolves the policy config's modality and joint-space paths against the working
# directory, and this repository owns all of them. Running here also keeps evaluation
# output out of the pinned Arena checkout, which launch_arena_eval.sh requires to be clean.
cd "${REPO_ROOT}"

if [[ "${CHECK_ONLY}" == true ]]; then
  checker_args=("${policy_config}")
  if [[ -n "${MODEL_PATH}" ]]; then
    checker_args+=("${MODEL_PATH}")
  fi
  exec "${REPO_ROOT}/tools/arena_launch.sh" \
    python "${REPO_ROOT}/tools/check_gr00t_eval.py" "${checker_args[@]}"
fi

exec "${REPO_ROOT}/tools/launch_arena_eval.sh" \
  "${EVAL_ARGS[@]}" \
  --policy_type "${policy_type}" \
  --policy_config_yaml_path "${policy_config}" \
  --enable_cameras \
  -- \
  "${environment_args[@]}"

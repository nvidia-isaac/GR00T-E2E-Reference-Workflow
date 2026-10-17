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

# User-site packages can shadow the pinned environment.
export PYTHONNOUSERSITE=1

# Runs Isaac AutoData against a pinned external checkout.
#
# Commands outside Arena use AutoData's provisioned interpreter. Arena generation overlays
# the pinned AutoData source into this repository's Arena interpreter because the two Isaac
# Lab runtimes cannot coexist in one process. The checkout stays clean.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INVOCATION_DIR="${PWD}"
AUTODATA_ROOT=${ISAAC_AUTODATA_ROOT:-"${REPO_ROOT}/../Isaac-AutoData-e2e"}
AUTODATA_PYTHON=${AUTODATA_PYTHON:-}
AUTODATA_CONDA_ENV=${AUTODATA_CONDA_ENV:-isaac_autodata}
OUTPUT_DIR=${E2E_AUTODATA_OUTPUT_DIR:-"${REPO_ROOT}/outputs/autodata"}
REQUIREMENTS="${REPO_ROOT}/tools/autodata_interpreter_requirements.txt"
UV_BIN=${UV_BIN:-uv}
CHECK_ONLY=false
ENABLE_CAMERAS=true
AUTO_ANNOTATE=true
LANE=native
WBC_MODE=agile
WBC_MODE_EXPLICIT=false
MANIPULATION_HAND=auto
INPUT_FILE=""
OUTPUT_FILE=""
RESULT_FILE=""
GENERATION_RUN_TIMEOUT_SECONDS=${E2E_AUTODATA_GENERATION_TIMEOUT_SECONDS:-420}
COMMAND=""
EXTRA_ARGS=()

usage() {
  cat >&2 <<'EOF'
Usage: tools/launch_autodata.sh <command> [--check] [options] [-- autodata args...]

Commands:
  provision         Clone AutoData beside this repo (if absent), check out the pinned
                    revision, create its conda interpreter, then run setup.
  setup             Install what the AutoData interpreter is missing, then exit.
  example           Generate from AutoData's shipped G1 pick-and-place demos (its own data).
  validate          Check that AutoData can drive our static-apple env, without needing demos.
  validate-dataset  Run AutoData's dataset validator over the datasets generated here.
  annotate          Annotate this repository's static-apple recordings for generation.
  generate          Amplify this repository's annotated static-apple demos.

Options:
  --autodata-root PATH    AutoData checkout (env: ISAAC_AUTODATA_ROOT).
  --autodata-python PATH  Interpreter to run it with (env: AUTODATA_PYTHON).
                          Defaults to the '${AUTODATA_CONDA_ENV}' conda env.
  --check                 Run the preflight checks and exit.
  --lane LANE             Select native (default) or arena. The input manifest must agree.
  --wbc-mode MODE         Native only: select fixed, agile (default), or auto. Auto reads the
                          explicit native action profile from the manifest before Isaac Sim starts.
  --manipulation-hand HAND
                          Select left, right, or auto (default). Auto reads the recorded manifest.
  --generation-run-timeout-seconds SECONDS
                          Hard wall-clock deadline for either generation lane (default: 420).
  --no-cameras            Native only: skip the head camera during generation. Faster, but
                          the dataset cannot be converted without video.
  --manual                Annotate by marking boundaries in the viewer instead of reading the
                          env's subtask signals.

Arena generation reads direct datagen_info from a manifested Pink recording. Pass
--requested_successes and --max_attempts after --; annotation is native-only.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    provision|setup|example|annotate|generate|validate|validate-dataset)
      [[ -z "${COMMAND}" ]] || { echo "Pass a single command, got '${COMMAND}' and '$1'." >&2; exit 2; }
      COMMAND=$1
      shift
      ;;
    --autodata-root)
      [[ "$#" -ge 2 ]] || { echo "--autodata-root requires a path." >&2; exit 2; }
      AUTODATA_ROOT=$2
      shift 2
      ;;
    --autodata-root=*)
      AUTODATA_ROOT=${1#*=}
      shift
      ;;
    --autodata-python)
      [[ "$#" -ge 2 ]] || { echo "--autodata-python requires a path." >&2; exit 2; }
      AUTODATA_PYTHON=$2
      shift 2
      ;;
    --autodata-python=*)
      AUTODATA_PYTHON=${1#*=}
      shift
      ;;
    --check)
      CHECK_ONLY=true
      shift
      ;;
    --lane)
      [[ "$#" -ge 2 ]] || { echo "--lane requires native or arena." >&2; exit 2; }
      LANE=$2
      shift 2
      ;;
    --lane=*)
      LANE=${1#*=}
      shift
      ;;
    --wbc-mode)
      [[ "$#" -ge 2 ]] || { echo "--wbc-mode requires fixed, agile, or auto." >&2; exit 2; }
      WBC_MODE=$2
      WBC_MODE_EXPLICIT=true
      shift 2
      ;;
    --wbc-mode=*)
      WBC_MODE=${1#*=}
      WBC_MODE_EXPLICIT=true
      shift
      ;;
    --manipulation-hand)
      [[ "$#" -ge 2 ]] || { echo "--manipulation-hand requires left, right, or auto." >&2; exit 2; }
      MANIPULATION_HAND=$2
      shift 2
      ;;
    --manipulation-hand=*)
      MANIPULATION_HAND=${1#*=}
      shift
      ;;
    --generation-run-timeout-seconds)
      [[ "$#" -ge 2 ]] || { echo "--generation-run-timeout-seconds requires a positive integer." >&2; exit 2; }
      GENERATION_RUN_TIMEOUT_SECONDS=$2
      shift 2
      ;;
    --generation-run-timeout-seconds=*)
      GENERATION_RUN_TIMEOUT_SECONDS=${1#*=}
      shift
      ;;
    --no-cameras)
      ENABLE_CAMERAS=false
      shift
      ;;
    --manual)
      AUTO_ANNOTATE=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

case "${WBC_MODE}" in
  auto|fixed|agile) ;;
  *) echo "Unsupported --wbc-mode '${WBC_MODE}'; expected fixed, agile, or auto." >&2; exit 2 ;;
esac
case "${LANE}" in
  native|arena) ;;
  *) echo "Unsupported --lane '${LANE}'; expected native or arena." >&2; exit 2 ;;
esac
if [[ "${LANE}" == arena && "${ENABLE_CAMERAS}" == false ]]; then
  echo "--no-cameras is only supported for native generation; Arena generation requires cameras." >&2
  exit 2
fi
if [[ "${LANE}" == arena && "${WBC_MODE_EXPLICIT}" == true ]]; then
  echo "--wbc-mode is only valid for native datasets." >&2
  exit 2
fi
case "${MANIPULATION_HAND}" in
  auto|left|right) ;;
  *) echo "Unsupported --manipulation-hand '${MANIPULATION_HAND}'; expected left, right, or auto." >&2; exit 2 ;;
esac
if [[ ! "${GENERATION_RUN_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Unsupported --generation-run-timeout-seconds '${GENERATION_RUN_TIMEOUT_SECONDS}'; expected a positive integer." >&2
  exit 2
fi

if [[ -z "${COMMAND}" && "${CHECK_ONLY}" != true ]]; then
  usage
  exit 2
fi

absolute_path() {
  case "$1" in
    /*) printf '%s' "$1" ;;
    *) printf '%s' "${INVOCATION_DIR}/$1" ;;
  esac
}

# AutoData's scripts run from its own checkout (the cd below), so a relative path a caller passes
# would resolve there rather than where they typed it. Rewrite it before the directory changes.
expect_path=""
for index in "${!EXTRA_ARGS[@]}"; do
  argument=${EXTRA_ARGS[index]}
  if [[ -n "${expect_path}" ]]; then
    resolved_path=$(absolute_path "${argument}")
    if [[ "${expect_path}" == input ]]; then
      INPUT_FILE=${resolved_path}
    elif [[ "${expect_path}" == output ]]; then
      OUTPUT_FILE=${resolved_path}
    else
      RESULT_FILE=${resolved_path}
    fi
    EXTRA_ARGS[index]="${resolved_path}"
    expect_path=""
    continue
  fi
  case "${argument}" in
    --input_file) expect_path=input ;;
    --input_file=*)
      INPUT_FILE=$(absolute_path "${argument#*=}")
      EXTRA_ARGS[index]="--input_file=${INPUT_FILE}"
      ;;
    --output_file) expect_path=output ;;
    --output_file=*)
      OUTPUT_FILE=$(absolute_path "${argument#*=}")
      EXTRA_ARGS[index]="--output_file=${OUTPUT_FILE}"
      ;;
    --result_file) expect_path=result ;;
    --result_file=*)
      RESULT_FILE=$(absolute_path "${argument#*=}")
      EXTRA_ARGS[index]="--result_file=${RESULT_FILE}"
      ;;
  esac
done
[[ -z "${expect_path}" ]] || { echo "--${expect_path}_file requires a path." >&2; exit 2; }

# The pinned AutoData revision from runtime.toml, read once and reused below: provision
# checks it out, and the guard verifies the checkout matches it.
autodata_revision="$("${UV_BIN}" run --project "${REPO_ROOT}" --frozen --no-sync python -c \
  'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["tested"]["isaac_autodata_revision"])' \
  "${REPO_ROOT}/runtime.toml")"

# Provision creates what the guards below require, then continues as `setup`: clone (if
# absent) and pin the checkout, init submodules, and build the conda interpreter.
if [[ "${COMMAND}" == "provision" ]]; then
  if [[ ! -e "${AUTODATA_ROOT}/.git" ]]; then
    echo "Cloning Isaac AutoData into ${AUTODATA_ROOT}"
    git clone https://github.com/isaac-sim/Isaac-AutoData.git "${AUTODATA_ROOT}"
  fi
  echo "Checking out the pinned AutoData revision ${autodata_revision:0:12}"
  git -C "${AUTODATA_ROOT}" fetch origin
  git -C "${AUTODATA_ROOT}" checkout "${autodata_revision}"
  # AutoData vendors Isaac Lab / Arena as submodules; conda_installer.sh -i needs them.
  git -C "${AUTODATA_ROOT}" submodule update --init --recursive
  provision_python="${AUTODATA_PYTHON}"
  if [[ -z "${provision_python}" ]]; then
    conda_base="$(conda info --base 2>/dev/null || true)"
    provision_python="${conda_base:-/nonexistent}/envs/${AUTODATA_CONDA_ENV}/bin/python"
  fi
  # Create the env if missing; install if it lacks Isaac Lab (finishes a half-built env).
  if [[ ! -x "${provision_python}" ]]; then
    echo "Creating the ${AUTODATA_CONDA_ENV} conda env (conda_installer.sh -c)"
    (cd "${AUTODATA_ROOT}" && ./conda_installer.sh -c)
  fi
  if ! "${provision_python}" -m pip show isaaclab >/dev/null 2>&1; then
    echo "Installing Isaac Sim / Isaac Lab / Arena / AutoData into the env (conda_installer.sh -i)"
    (cd "${AUTODATA_ROOT}" && ./conda_installer.sh -i)
  fi
  COMMAND=setup
fi

if [[ ! -f "${AUTODATA_ROOT}/scripts/generate_dataset.py" ]]; then
  echo "No Isaac AutoData checkout at ${AUTODATA_ROOT}; pass --autodata-root." >&2
  exit 2
fi
AUTODATA_ROOT="$(cd "${AUTODATA_ROOT}" && pwd)"

actual_revision="$(git -C "${AUTODATA_ROOT}" rev-parse HEAD)"
if [[ "${actual_revision}" != "${autodata_revision}" ]]; then
  echo "AutoData revision mismatch: expected ${autodata_revision}, got ${actual_revision}" >&2
  exit 2
fi
if [[ -n "$(git -C "${AUTODATA_ROOT}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "AutoData checkout is dirty; use a clean isolated checkout at the pinned revision." >&2
  exit 2
fi

# Isaac Sim's EULA prompt would block a non-interactive run.
export OMNI_KIT_ACCEPT_EULA=YES
# Preserve progress and preflight messages when stdout is redirected through a runner.
export PYTHONUNBUFFERED=1
# AutoData's packages live in its checkout rather than site-packages, and running a script by
# path puts the script's directory on sys.path instead of the root. Our src carries the task
# package that registers the env id AutoData resolves with gymnasium.
export PYTHONPATH="${AUTODATA_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

ARENA_GENERATION=false
[[ "${COMMAND}" == generate && "${LANE}" == arena ]] && ARENA_GENERATION=true

if [[ "${ARENA_GENERATION}" != true ]]; then
  if [[ -z "${AUTODATA_PYTHON}" ]]; then
    conda_base="$(conda info --base 2>/dev/null || true)"
    AUTODATA_PYTHON="${conda_base:-/nonexistent}/envs/${AUTODATA_CONDA_ENV}/bin/python"
  fi
  if [[ ! -x "${AUTODATA_PYTHON}" ]]; then
    echo "No AutoData interpreter at ${AUTODATA_PYTHON}; pass --autodata-python." >&2
    echo "Create one with ${AUTODATA_ROOT}/conda_installer.sh." >&2
    exit 2
  fi

  find_missing_modules() {
    # AutoData's own packages resolve from its checkout rather than from site-packages.
    (cd "${AUTODATA_ROOT}" && "${AUTODATA_PYTHON}" -c '
import importlib.util, sys
print(" ".join(name for name in sys.argv[1:] if importlib.util.find_spec(name) is None))
' "$@")
  }

  if ! missing="$(find_missing_modules isaac_autodata_core isaaclab)"; then
    echo "Could not query the AutoData interpreter at ${AUTODATA_PYTHON}." >&2
    exit 2
  fi
  if [[ -n "${missing}" ]]; then
    echo "The AutoData interpreter is missing: ${missing}" >&2
    echo "Install AutoData into it with ${AUTODATA_ROOT}/conda_installer.sh." >&2
    exit 2
  fi

  # Requirements are version-checked, not just import-checked: the daqp incompatibility this pins
  # past is a package that imports fine and then raises mid-episode.
  unsatisfied="$("${AUTODATA_PYTHON}" - "${REQUIREMENTS}" <<'PYTHON'
import sys
from importlib.metadata import PackageNotFoundError, version

from packaging.requirements import Requirement

unsatisfied = []
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.split("#", 1)[0].strip()
    if not line:
        continue
    requirement = Requirement(line)
    try:
        installed = version(requirement.name)
    except PackageNotFoundError:
        unsatisfied.append(f"{requirement} (absent)")
        continue
    if not requirement.specifier.contains(installed, prereleases=True):
        unsatisfied.append(f"{requirement} (found {installed})")
print(" ".join(unsatisfied))
PYTHON
  )"
  if [[ -n "${unsatisfied}" ]]; then
    if [[ "${COMMAND}" != "setup" ]]; then
      echo "The AutoData interpreter does not satisfy: ${unsatisfied}" >&2
      echo "Install with tools/launch_autodata.sh setup" >&2
      exit 2
    fi
    echo "Installing into ${AUTODATA_PYTHON}: ${unsatisfied}"
    "${AUTODATA_PYTHON}" -m pip install --requirement "${REQUIREMENTS}"
  fi

  if [[ "${COMMAND}" == "setup" ]]; then
    echo "AutoData integration extras are installed ($(basename "${REQUIREMENTS}"))."
    exit 0
  fi
fi

if [[ "${CHECK_ONLY}" == true ]]; then
  echo "AutoData ${autodata_revision:0:12} at ${AUTODATA_ROOT}"
  if [[ "${ARENA_GENERATION}" == true ]]; then
    "${REPO_ROOT}/tools/arena_launch.sh" --check
    echo "Interpreter: pinned Arena runtime"
  else
    echo "Interpreter ${AUTODATA_PYTHON}"
  fi
  exit 0
fi

SELECTED_WBC_MODE=""
SELECTED_MANIPULATION_HAND=""
SELECTED_LANE=""
case "${COMMAND}" in
  annotate|generate)
    [[ -n "${INPUT_FILE}" ]] || {
      echo "${COMMAND} requires --input_file so the fixed or AGILE action layout can be selected." >&2
      exit 2
    }
    REQUIRED_STAGE=recording
    if [[ "${COMMAND}" == generate && "${LANE}" == native ]]; then
      REQUIRED_STAGE=annotation
    fi
    REQUESTED_WBC_MODE=${WBC_MODE}
    [[ "${LANE}" != arena ]] || REQUESTED_WBC_MODE=auto
    PROFILE_ARGS=(
      -m e2e_robot_workflows.autodata.dataset_mode
      --input-file "${INPUT_FILE}"
      --requested-lane "${LANE}"
      --requested-mode "${REQUESTED_WBC_MODE}"
      --requested-hand "${MANIPULATION_HAND}"
      --required-stage "${REQUIRED_STAGE}"
    )
    if [[ "${ARENA_GENERATION}" == true ]]; then
      DATASET_PROFILE="$(${UV_BIN} run --project "${REPO_ROOT}" --frozen --no-sync python "${PROFILE_ARGS[@]}")"
    else
      DATASET_PROFILE="$("${AUTODATA_PYTHON}" "${PROFILE_ARGS[@]}")"
    fi
    read -r SELECTED_LANE SELECTED_WBC_MODE SELECTED_MANIPULATION_HAND <<< "${DATASET_PROFILE}"
    ;;
  validate)
    SELECTED_LANE=${LANE}
    SELECTED_WBC_MODE=${WBC_MODE}
    [[ "${SELECTED_WBC_MODE}" != auto ]] || SELECTED_WBC_MODE=fixed
    SELECTED_MANIPULATION_HAND=${MANIPULATION_HAND}
    [[ "${SELECTED_MANIPULATION_HAND}" != auto ]] || SELECTED_MANIPULATION_HAND=right
    ;;
esac

if [[ "${SELECTED_LANE}" == arena && "${COMMAND}" != generate ]]; then
  echo "Arena recordings carry direct datagen_info; use --lane arena generate without an annotation pass." >&2
  exit 2
fi

if [[ "${SELECTED_LANE}" == arena ]]; then
  AUTODATA_LOCK_ROOT="${XDG_RUNTIME_DIR:-/tmp}/e2e-reference-app-autodata-${UID}"
  mkdir -m 700 "${AUTODATA_LOCK_ROOT}" 2>/dev/null || true
  if [[ ! -d "${AUTODATA_LOCK_ROOT}" || -L "${AUTODATA_LOCK_ROOT}" || ! -O "${AUTODATA_LOCK_ROOT}" ]]; then
    echo "Arena AutoData lock directory is not a user-owned directory: ${AUTODATA_LOCK_ROOT}" >&2
    exit 2
  fi
  chmod 700 "${AUTODATA_LOCK_ROOT}"
  AUTODATA_LOCK_FILE="${AUTODATA_LOCK_ROOT}/generation.lock"
  if [[ -e "${AUTODATA_LOCK_FILE}" && ( ! -f "${AUTODATA_LOCK_FILE}" || -L "${AUTODATA_LOCK_FILE}" || ! -O "${AUTODATA_LOCK_FILE}" ) ]]; then
    echo "Arena AutoData lock is not a user-owned regular file: ${AUTODATA_LOCK_FILE}" >&2
    exit 2
  fi
  OLD_UMASK=$(umask)
  umask 077
  exec {AUTODATA_LOCK_FD}>>"${AUTODATA_LOCK_FILE}"
  umask "${OLD_UMASK}"
  chmod 600 "${AUTODATA_LOCK_FILE}"
  if ! flock -n "${AUTODATA_LOCK_FD}"; then
    echo "Another Arena AutoData generation is already running for this user." >&2
    exit 2
  fi
  if [[ "${SELECTED_WBC_MODE}" == h2_agile_pink ]]; then
    ARENA_TASK_DESCRIPTOR="${REPO_ROOT}/configs/stages/autodata/static_apple_arena_h2_task.yaml"
    if [[ "${SELECTED_MANIPULATION_HAND}" == left ]]; then
      ARENA_TASK_DESCRIPTOR="${REPO_ROOT}/configs/stages/autodata/static_apple_arena_h2_left_task.yaml"
    fi
  else
    ARENA_TASK_DESCRIPTOR="${REPO_ROOT}/configs/stages/autodata/static_apple_arena_task.yaml"
    if [[ "${SELECTED_MANIPULATION_HAND}" == left ]]; then
      ARENA_TASK_DESCRIPTOR="${REPO_ROOT}/configs/stages/autodata/static_apple_arena_left_task.yaml"
    fi
  fi
  "${REPO_ROOT}/tools/arena_launch.sh" --check
  mkdir -p "${OUTPUT_DIR}"
  [[ -n "${OUTPUT_FILE}" ]] || OUTPUT_FILE="${OUTPUT_DIR}/static_apple_arena_generated.hdf5"
  [[ -n "${RESULT_FILE}" ]] || RESULT_FILE="${OUTPUT_DIR}/static_apple_arena_generation_result.json"
  OUTPUT_BASENAME="$(basename -- "${OUTPUT_FILE}")"
  OUTPUT_STEM="${OUTPUT_BASENAME%.hdf5}"
  [[ "${OUTPUT_FILE}" == *.hdf5 && "${OUTPUT_STEM}" =~ [^.] && "${OUTPUT_STEM}" != *.hdf5 ]] || {
    echo "Arena --output_file must have exactly one .hdf5 suffix and a basename before it." >&2
    exit 2
  }
  [[ "${RESULT_FILE}" == *.json ]] || { echo "Arena --result_file must end in .json." >&2; exit 2; }
  FAILED_OUTPUT_FILE="${OUTPUT_FILE%.*}_failed.${OUTPUT_FILE##*.}"
  if ! "${UV_BIN}" run --project "${REPO_ROOT}" --frozen --no-sync python -c \
      'import pathlib,sys; paths=[pathlib.Path(p).resolve() for p in sys.argv[1:]]; raise SystemExit(len(paths)!=len(set(paths)))' \
      "${INPUT_FILE}" "${OUTPUT_FILE}" "${FAILED_OUTPUT_FILE}" "${RESULT_FILE}"; then
    echo "Arena input, output, failed-output, and result paths must be distinct." >&2
    exit 2
  fi
  for target in "${OUTPUT_FILE}" "${FAILED_OUTPUT_FILE}" "${RESULT_FILE}"; do
    [[ ! -e "${target}" && ! -L "${target}" ]] || {
      echo "Refusing to replace existing Arena generation artifact: ${target}" >&2
      exit 2
    }
  done
  mkdir -p "$(dirname "${OUTPUT_FILE}")" "$(dirname "${RESULT_FILE}")"
  RUN_DIRECTORY="$(mktemp -d "$(dirname "${OUTPUT_FILE}")/.arena-autodata.XXXXXX")"
  RUN_PUBLISHED=false
  # shellcheck disable=SC2329  # Invoked indirectly by the EXIT trap below.
  cleanup_arena_run() {
    if [[ "${RUN_PUBLISHED}" == true ]]; then
      rm -rf -- "${RUN_DIRECTORY}"
    else
      echo "Retained staged Arena generation artifacts after failure: ${RUN_DIRECTORY}" >&2
    fi
  }
  trap cleanup_arena_run EXIT
  RUN_OUTPUT_FILE="${RUN_DIRECTORY}/$(basename "${OUTPUT_FILE}")"
  RUN_FAILED_OUTPUT_FILE="${RUN_OUTPUT_FILE%.*}_failed.${RUN_OUTPUT_FILE##*.}"
  RUN_RESULT_FILE="${RUN_DIRECTORY}/result.json"
  RUN_ID="$(${UV_BIN} run --project "${REPO_ROOT}" --frozen --no-sync python -c 'import uuid; print(uuid.uuid4().hex)')"
  export ACCEPT_EULA=${ACCEPT_EULA:-Y}
  export PYTHONPATH="${AUTODATA_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  promote_arena_run() {
    "${UV_BIN}" run --project "${REPO_ROOT}" --frozen --no-sync python \
      -m e2e_robot_workflows.autodata.promotion \
      --staging-result "${RUN_RESULT_FILE}" \
      --staging-output "${RUN_OUTPUT_FILE}" \
      --staging-failed-output "${RUN_FAILED_OUTPUT_FILE}" \
      --output "${OUTPUT_FILE}" \
      --failed-output "${FAILED_OUTPUT_FILE}" \
      --result "${RESULT_FILE}"
    RUN_PUBLISHED=true
  }
  set +e
  timeout --foreground --signal=TERM --kill-after=15s "${GENERATION_RUN_TIMEOUT_SECONDS}s" \
    "${REPO_ROOT}/tools/arena_launch.sh" python "${REPO_ROOT}/scripts/generate_arena_dataset.py" \
    --task_descriptor "${ARENA_TASK_DESCRIPTOR}" \
    --manipulation_hand "${SELECTED_MANIPULATION_HAND}" \
    "${EXTRA_ARGS[@]}" \
    --input_file "${INPUT_FILE}" \
    --output_file "${RUN_OUTPUT_FILE}" \
    --result_file "${RUN_RESULT_FILE}" \
    --run_id "${RUN_ID}"
  RUNNER_STATUS=$?
  set -e
  if [[ "${RUNNER_STATUS}" == 124 || "${RUNNER_STATUS}" == 137 ]]; then
    echo "Arena AutoData timed out or was killed before completion (deadline: ${GENERATION_RUN_TIMEOUT_SECONDS}s)." >&2
    echo "Retained artifacts may be incomplete and are not ready for conversion." >&2
    echo "Retry in a new output directory with a larger --generation-run-timeout-seconds budget." >&2
  fi
  [[ -f "${RUN_RESULT_FILE}" ]] || {
    "${UV_BIN}" run --project "${REPO_ROOT}" --frozen --no-sync python -c \
      'import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]); s=pathlib.Path(sys.argv[2]); source_sha256=None
if s.is_file():
 with s.open("rb") as stream: source_sha256=hashlib.file_digest(stream,"sha256").hexdigest()
d={"status":"error","lane":"arena","run_id":sys.argv[4],"failure":f"runner exited {sys.argv[3]} before writing a result","source_sha256":source_sha256}; r.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")' \
      "${RUN_RESULT_FILE}" "${INPUT_FILE}" "${RUNNER_STATUS}" "${RUN_ID}"
    promote_arena_run
    echo "Arena AutoData exited ${RUNNER_STATUS} without a structured result: ${RESULT_FILE}" >&2
    exit 3
  }
  if [[ "${RUNNER_STATUS}" != 0 ]]; then
    "${UV_BIN}" run --project "${REPO_ROOT}" --frozen --no-sync python -c \
      'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()); d["status"]="error"; d.setdefault("failure",f"runner exited {sys.argv[2]} before completion"); pathlib.Path(sys.argv[3]).write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")' \
      "${RUN_RESULT_FILE}" "${RUNNER_STATUS}" "${RUN_RESULT_FILE}.updated"
    mv "${RUN_RESULT_FILE}.updated" "${RUN_RESULT_FILE}"
    promote_arena_run
    echo "Arena AutoData exited ${RUNNER_STATUS}; retained result: ${RESULT_FILE}" >&2
    exit 3
  fi
  if ! "${UV_BIN}" run --project "${REPO_ROOT}" --frozen --no-sync python -c \
      'import hashlib,json,pathlib,sys
from e2e_robot_workflows.data.dataset_manifest import validate_dataset
def sha(path):
 d=hashlib.sha256()
 with pathlib.Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""): d.update(block)
 return d.hexdigest()
r=json.load(open(sys.argv[1],encoding="utf-8"))
m=validate_dataset(sys.argv[4],require_lane="arena",require_producer="autodata",require_stage="generation")
ok=(r.get("status")=="ok" and r.get("run_id")==sys.argv[5] and r.get("source_sha256")==sha(sys.argv[2]) and r.get("task_descriptor_sha256")==sha(sys.argv[3]) and m["source"]["sha256"]==r["source_sha256"] and m["source"]["name"]==pathlib.Path(sys.argv[2]).name and r.get("artifacts",{}).get("generated",{}).get("sha256")==sha(sys.argv[4]))
raise SystemExit(not ok)' \
      "${RUN_RESULT_FILE}" "${INPUT_FILE}" "${ARENA_TASK_DESCRIPTOR}" "${RUN_OUTPUT_FILE}" "${RUN_ID}"; then
    "${UV_BIN}" run --project "${REPO_ROOT}" --frozen --no-sync python -c \
      'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()); failure="fresh Arena result, inputs, output, or immutable lineage check failed"; d["status"]="error"; d.setdefault("failure",failure); d["evidence_validation_failure"]=failure; pathlib.Path(sys.argv[2]).write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")' \
      "${RUN_RESULT_FILE}" "${RUN_RESULT_FILE}.updated"
    mv "${RUN_RESULT_FILE}.updated" "${RUN_RESULT_FILE}"
    promote_arena_run
    echo "Arena generation evidence validation failed: ${RESULT_FILE}" >&2
    exit 3
  fi
  promote_arena_run
  exit 0
fi

if [[ -n "${SELECTED_WBC_MODE}" ]]; then
  case "${SELECTED_WBC_MODE}" in
    fixed)
      AUTODATA_ENV_NAME=Isaac-E2E-G1-StaticApple-PickPlace-AutoData-v0
      AUTODATA_EMBODIMENT="${REPO_ROOT}/configs/stages/autodata/static_apple_embodiment.yaml"
      ACTION_WIDTH=28
      ;;
    agile)
      AUTODATA_ENV_NAME=Isaac-E2E-G1-StaticApple-PickPlace-AutoData-Agile-v0
      AUTODATA_EMBODIMENT="${REPO_ROOT}/configs/stages/autodata/static_apple_agile_embodiment.yaml"
      ACTION_WIDTH=32
      ;;
  esac
  case "${SELECTED_MANIPULATION_HAND}" in
    left) AUTODATA_TASK_DESCRIPTOR="${REPO_ROOT}/configs/stages/autodata/static_apple_left_task.yaml" ;;
    right) AUTODATA_TASK_DESCRIPTOR="${REPO_ROOT}/configs/stages/autodata/static_apple_task.yaml" ;;
  esac
  echo "AutoData ${COMMAND}: ${SELECTED_WBC_MODE} G1 mode (${ACTION_WIDTH}-D actions), ${SELECTED_MANIPULATION_HAND}-hand manipulation."
fi

mkdir -p "${OUTPUT_DIR}"
# AutoData's scripts resolve their own example paths against the checkout root, and its
# sitecustomize.py is picked up from there.
cd "${AUTODATA_ROOT}"

# The head camera is added only when Isaac Sim reports cameras enabled, and the recorder writes
# whatever the policy observation group holds -- so this flag is what puts video in the dataset.
CAMERA_ARGS=()
[[ "${ENABLE_CAMERAS}" == true ]] && CAMERA_ARGS+=(--enable_cameras)

# Automatic annotation reads the env's subtask_terms group, so it needs no viewer; the manual path
# needs one to mark boundaries in.
ANNOTATE_ARGS=()
[[ "${AUTO_ANNOTATE}" == true ]] && ANNOTATE_ARGS+=(--auto --headless)

case "${COMMAND}" in
  example)
    exec "${AUTODATA_PYTHON}" scripts/generate_dataset.py \
      --env_name Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
      --alg dexmimicgen \
      --task_descriptor isaac_autodata_examples/tasks/g1_pick_place.yaml \
      --embodiment isaac_autodata_examples/embodiments/g1_ik_abs.yaml \
      --input_file "${AUTODATA_ROOT}/datasets/annotated_datasets/dataset_g1_annotated.hdf5" \
      --output_file "${OUTPUT_DIR}/g1_example_generated.hdf5" \
      "${EXTRA_ARGS[@]}"
    ;;
  validate)
    # Run by path so the script's siblings in tools/simulation import, and under AutoData's
    # interpreter so this covers our env cfg building against AutoData's Isaac Lab.
    exec "${AUTODATA_PYTHON}" "${REPO_ROOT}/tools/simulation/validate_autodata_bridge.py" \
      --wbc-mode "${SELECTED_WBC_MODE}" \
      --manipulation-hand "${SELECTED_MANIPULATION_HAND}" \
      "${EXTRA_ARGS[@]}"
    ;;
  validate-dataset)
    # Defaults to everything generated here.
    datasets=()
    for argument in "${EXTRA_ARGS[@]}"; do
      datasets+=("$(absolute_path "${argument}")")
    done
    if [[ "${#datasets[@]}" -eq 0 ]]; then
      shopt -s nullglob
      datasets=("${OUTPUT_DIR}"/*.hdf5)
      shopt -u nullglob
      [[ "${#datasets[@]}" -gt 0 ]] || { echo "No datasets to validate in ${OUTPUT_DIR}." >&2; exit 2; }
    fi
    exec "${AUTODATA_PYTHON}" scripts/validate_dataset.py "${datasets[@]}"
    ;;
  annotate)
    [[ -n "${OUTPUT_FILE}" ]] || OUTPUT_FILE="${OUTPUT_DIR}/static_apple_annotated.hdf5"
    "${AUTODATA_PYTHON}" -m e2e_robot_workflows.autodata scripts/annotate_demos.py \
      --env_name "${AUTODATA_ENV_NAME}" \
      --task_descriptor "${AUTODATA_TASK_DESCRIPTOR}" \
      --embodiment "${AUTODATA_EMBODIMENT}" \
      --output_file "${OUTPUT_FILE}" \
      "${ANNOTATE_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
    exec "${AUTODATA_PYTHON}" -m e2e_robot_workflows.data.dataset_manifest stamp-autodata \
      --input-file "${OUTPUT_FILE}" \
      --source-file "${INPUT_FILE}" \
      --stage annotation
    ;;
  generate)
    [[ -n "${OUTPUT_FILE}" ]] || OUTPUT_FILE="${OUTPUT_DIR}/static_apple_generated.hdf5"
    [[ -n "${RESULT_FILE}" ]] || RESULT_FILE="${OUTPUT_DIR}/static_apple_generation_result.json"
    set +e
    timeout --foreground --signal=TERM --kill-after=15s "${GENERATION_RUN_TIMEOUT_SECONDS}s" \
      "${AUTODATA_PYTHON}" -m e2e_robot_workflows.autodata scripts/generate_dataset.py \
      --env_name "${AUTODATA_ENV_NAME}" \
      --alg dexmimicgen \
      --task_descriptor "${AUTODATA_TASK_DESCRIPTOR}" \
      --embodiment "${AUTODATA_EMBODIMENT}" \
      --output_file "${OUTPUT_FILE}" \
      --result_file "${RESULT_FILE}" \
      "${CAMERA_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
    RUNNER_STATUS=$?
    set -e
    [[ -f "${RESULT_FILE}" ]] || {
      "${AUTODATA_PYTHON}" -c \
        'import hashlib,json,pathlib,sys; r=pathlib.Path(sys.argv[1]); s=pathlib.Path(sys.argv[2]); d={"status":"error","lane":"native","failure":f"runner exited {sys.argv[3]} before writing a result","source_sha256":hashlib.sha256(s.read_bytes()).hexdigest() if s.is_file() else None}; r.parent.mkdir(parents=True,exist_ok=True); r.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")' \
        "${RESULT_FILE}" "${INPUT_FILE}" "${RUNNER_STATUS}"
      echo "Native AutoData exited ${RUNNER_STATUS} without a structured result: ${RESULT_FILE}" >&2
      exit 3
    }
    if [[ "${RUNNER_STATUS}" != 0 ]]; then
      "${AUTODATA_PYTHON}" -c \
        'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()); d["status"]="error"; d.setdefault("failure",f"runner exited {sys.argv[2]} before completion"); t=p.with_suffix(p.suffix+".tmp"); t.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n"); t.replace(p)' \
        "${RESULT_FILE}" "${RUNNER_STATUS}"
      echo "Native AutoData exited ${RUNNER_STATUS}; retained result: ${RESULT_FILE}" >&2
      exit 3
    fi
    "${AUTODATA_PYTHON}" -c \
      'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); assert d["num_success"]==d["requested_trials"] and d["num_success"]+d["num_failures"]==d["num_attempts"]' \
      "${RESULT_FILE}" || {
      echo "Native AutoData result did not satisfy its requested success count: ${RESULT_FILE}" >&2
      exit 3
    }
    if ! "${AUTODATA_PYTHON}" -m e2e_robot_workflows.data.dataset_manifest stamp-autodata \
        --input-file "${OUTPUT_FILE}" \
        --source-file "${INPUT_FILE}" \
        --stage generation; then
      "${AUTODATA_PYTHON}" -c \
        'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()); d.update(status="error",failure="final manifest stamping failed"); t=p.with_suffix(p.suffix+".tmp"); t.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n"); t.replace(p)' \
        "${RESULT_FILE}"
      exit 3
    fi
    "${AUTODATA_PYTHON}" -c \
      'import hashlib,json,pathlib,sys; r,o,s=map(pathlib.Path,sys.argv[1:]); d=json.loads(r.read_text()); d.update(status="ok",lane="native",source_sha256=hashlib.sha256(s.read_bytes()).hexdigest(),generated_sha256=hashlib.sha256(o.read_bytes()).hexdigest()); t=r.with_suffix(r.suffix+".tmp"); t.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n"); t.replace(r)' \
      "${RESULT_FILE}" "${OUTPUT_FILE}" "${INPUT_FILE}"
    echo "Generated manifested native dataset: ${OUTPUT_FILE}"
    ;;
esac

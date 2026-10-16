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

# Run a command in the pinned standalone GR00T environment. Some Ubuntu hosts
# provide BLAS/LAPACK binaries with the required .so.3 SONAME but omit the
# matching loader symlinks. Add temporary, ABI-checked links without changing
# the host or the GR00T environment.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_ROOT=${ISAAC_GR00T_ROOT:-}
UV_BIN=${UV_BIN:-uv}
CHECK_ONLY=false
COMMAND=()

usage() {
  cat >&2 <<'EOF'
Usage: tools/gr00t_runtime.sh --gr00t-root PATH [--check | -- COMMAND...]

  --gr00t-root PATH  Clean Isaac-GR00T checkout at the revision in runtime.toml.
  --check            Import TorchCodec and its video decoder, then exit.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
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
    --)
      shift
      COMMAND=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "${GR00T_ROOT}" ]] || { echo "Pass --gr00t-root." >&2; exit 2; }
[[ -f "${GR00T_ROOT}/pyproject.toml" && -x "${GR00T_ROOT}/.venv/bin/python" ]] || {
  echo "GR00T root must contain pyproject.toml and a synced .venv." >&2
  exit 2
}
if [[ "${CHECK_ONLY}" == true && "${#COMMAND[@]}" -ne 0 ]]; then
  echo "--check does not accept a command." >&2
  exit 2
fi
if [[ "${CHECK_ONLY}" != true && "${#COMMAND[@]}" -eq 0 ]]; then
  echo "Pass a command after --." >&2
  exit 2
fi

GR00T_ROOT="$(cd "${GR00T_ROOT}" && pwd)"

expected_revision="$("${UV_BIN}" run --project "${REPO_ROOT}" --frozen --no-sync python -c \
  'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["tested"]["isaac_gr00t_revision"])' \
  "${REPO_ROOT}/runtime.toml")"
actual_revision="$(git -C "${GR00T_ROOT}" rev-parse HEAD)"
if [[ "${actual_revision}" != "${expected_revision}" ]]; then
  echo "Isaac-GR00T revision mismatch: expected ${expected_revision}, got ${actual_revision}" >&2
  exit 2
fi
if [[ -n "$(git -C "${GR00T_ROOT}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Isaac-GR00T checkout must be clean: ${GR00T_ROOT}" >&2
  exit 2
fi

compat_dir="$(mktemp -d "${TMPDIR:-/tmp}/e2e-gr00t-runtime.XXXXXX")"
cleanup() {
  rm -f "${compat_dir}/libblas.so.3" "${compat_dir}/liblapack.so.3"
  rmdir "${compat_dir}"
}
trap cleanup EXIT
export LD_LIBRARY_PATH="${compat_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

ensure_library() {
  local soname=$1
  local candidate=$2
  if "${GR00T_ROOT}/.venv/bin/python" -c 'import ctypes, sys; ctypes.CDLL(sys.argv[1])' "${soname}" \
      >/dev/null 2>&1; then
    return
  fi
  [[ -f "${candidate}" ]] || { echo "Missing ${soname}; no compatible ${candidate}." >&2; exit 2; }
  local candidate_soname
  candidate_soname="$(objdump -p "${candidate}" | awk '$1 == "SONAME" {print $2; exit}')"
  if [[ "${candidate_soname}" != "${soname}" ]]; then
    echo "Refusing ${candidate}: expected SONAME ${soname}, got ${candidate_soname:-missing}." >&2
    exit 2
  fi
  ln -s "${candidate}" "${compat_dir}/${soname}"
}

ensure_library libblas.so.3 /lib/x86_64-linux-gnu/libblas.so
ensure_library liblapack.so.3 /lib/x86_64-linux-gnu/liblapack.so

if [[ "${CHECK_ONLY}" == true ]]; then
  UV_PROJECT_ENVIRONMENT="${GR00T_ROOT}/.venv" "${UV_BIN}" run --project "${GR00T_ROOT}" --frozen --no-sync python -c \
    'import torchcodec; print(f"TorchCodec {torchcodec.__version__}: {torchcodec.decoders.VideoDecoder.__name__}")'
  exit 0
fi

# An E2E worktree may override its environment. GR00T has a separate Python
# contract, so never let that override redirect its sync into the E2E runtime.
UV_PROJECT_ENVIRONMENT="${GR00T_ROOT}/.venv" "${UV_BIN}" run --project "${GR00T_ROOT}" --frozen "${COMMAND[@]}"

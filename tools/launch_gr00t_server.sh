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
CHECKPOINT_SOURCE=${E2E_GR00T_CHECKPOINT_SOURCE:-trained}
UV_BIN=${UV_BIN:-uv}
CHECK_ONLY=false
ROBOT=g1
SERVER_ARGS=()

while [[ "$#" -gt 0 ]]; do
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
    --gr00t-root)
      [[ "$#" -ge 2 ]] || { echo "--gr00t-root requires a path." >&2; exit 2; }
      GR00T_ROOT=$2
      shift 2
      ;;
    --gr00t-root=*)
      GR00T_ROOT=${1#*=}
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
    --checkpoint-source)
      [[ "$#" -ge 2 ]] || { echo "--checkpoint-source requires a value." >&2; exit 2; }
      CHECKPOINT_SOURCE=$2
      shift 2
      ;;
    --checkpoint-source=*)
      CHECKPOINT_SOURCE=${1#*=}
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
    --modality-config-path | --modality-config-path=* | \
      --embodiment-tag | --embodiment-tag=* | --dataset-path | --dataset-path=* | \
      --execution-horizon | --execution-horizon=* | --no-strict | --use-sim-policy-wrapper)
      echo "Server argument is owned by this launcher: $1" >&2
      exit 2
      ;;
    *)
      SERVER_ARGS+=("$1")
      shift
      ;;
  esac
done

case "${ROBOT}" in
  g1 | h2) ;;
  *) echo "--robot must be g1 or h2, got: ${ROBOT}" >&2; exit 2 ;;
esac

case "${CHECKPOINT_SOURCE}" in
  released | published | trained) ;;
  *)
    echo "E2E_GR00T_CHECKPOINT_SOURCE must be 'released', 'published', or 'trained'." >&2
    exit 2
    ;;
esac
if [[ -z "${ARENA_ROOT}" || -z "${GR00T_ROOT}" || -z "${MODEL_PATH}" ]]; then
  echo "Pass --arena-root, --gr00t-root, and --model-path." >&2
  exit 2
fi

server_script="${GR00T_ROOT}/gr00t/eval/run_gr00t_server.py"
if [[ ! -f "${GR00T_ROOT}/pyproject.toml" || ! -f "${server_script}" ]]; then
  echo "ISAAC_GR00T_ROOT does not contain the standalone Isaac-GR00T checkout." >&2
  exit 2
fi
hf_cli="${GR00T_ROOT}/.venv/bin/hf"
if [[ ! -x "${hf_cli}" ]]; then
  echo "Sync the standalone Isaac-GR00T environment before launching the server." >&2
  exit 2
fi

runtime_pins_output="$("${UV_BIN}" run --project "${REPO_ROOT}" --frozen --no-sync python -c \
  'import sys, tomllib; c = tomllib.load(open(sys.argv[1], "rb"))["tested"]; [print(c[k]) for k in ("isaac_gr00t_revision", "static_apple_checkpoint_repository", "static_apple_checkpoint_revision")]' \
  "${REPO_ROOT}/runtime.toml")"
readarray -t runtime_pins <<<"${runtime_pins_output}"
expected_revision="${runtime_pins[0]}"
checkpoint_repository="${runtime_pins[1]}"
expected_checkpoint_revision="${runtime_pins[2]}"
if [[ "${CHECKPOINT_SOURCE}" == "published" ]]; then
  published_pins_output="$("${UV_BIN}" run --project "${REPO_ROOT}" --frozen --no-sync python -c \
    'import sys, tomllib; c = tomllib.load(open(sys.argv[1], "rb"))["artifacts"][sys.argv[2]]; print(c["model_repository"]); print(c["model_revision"])' \
    "${REPO_ROOT}/runtime.toml" "${ROBOT}")"
  readarray -t published_pins <<<"${published_pins_output}"
  checkpoint_repository="${published_pins[0]}"
  expected_checkpoint_revision="${published_pins[1]}"
fi
actual_revision="$(git -C "${GR00T_ROOT}" rev-parse HEAD)"
if [[ "${actual_revision}" != "${expected_revision}" ]]; then
  echo "Isaac-GR00T revision mismatch: expected ${expected_revision}, got ${actual_revision}" >&2
  exit 2
fi
if [[ -n "$(git -C "${GR00T_ROOT}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Isaac-GR00T checkout must be clean: ${GR00T_ROOT}" >&2
  exit 2
fi

if ! "${GR00T_ROOT}/.venv/bin/python" -c \
  'from huggingface_hub import get_hf_file_metadata, hf_hub_url; get_hf_file_metadata(hf_hub_url("nvidia/Cosmos-Reason2-2B", "config.json"), token=True)' \
  >/dev/null 2>&1; then
  echo "Log in with '${hf_cli} auth login' after obtaining access to nvidia/Cosmos-Reason2-2B." >&2
  exit 2
fi

index_name="model.safetensors.index.json"
if [[ ! -f "${MODEL_PATH}/${index_name}" ]]; then
  echo "Checkpoint is missing required file: ${index_name}" >&2
  exit 2
fi
# The shard count follows the checkpoint's saved dtype and max_shard_size, so a trained
# checkpoint need not match the released layout. Read the weight files the index names.
# The index is checkpoint-supplied data, so it is validated before its names are joined
# with MODEL_PATH below: readable JSON, a non-empty weight_map, and bare filenames only
# (a name with a separator could point the existence check outside the checkpoint).
shard_output="$("${UV_BIN}" run --project "${REPO_ROOT}" --frozen --no-sync python -c '
import json, sys
try:
    names = sorted(set(json.load(open(sys.argv[1]))["weight_map"].values()))
except Exception as exc:
    sys.exit(f"Checkpoint index is unreadable or has no weight_map: {exc}")
if not names:
    sys.exit("Checkpoint index lists no weight files.")
for name in names:
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        sys.exit(f"Checkpoint index contains an unsafe shard name: {name!r}")
print("\n".join(names))
' "${MODEL_PATH}/${index_name}")" || exit 2
readarray -t shard_files <<<"${shard_output}"
required_files=(
  config.json processor_config.json statistics.json embodiment_id.json
  "${index_name}" "${shard_files[@]}"
)
for name in "${required_files[@]}"; do
  if [[ ! -f "${MODEL_PATH}/${name}" ]]; then
    echo "Checkpoint is missing required file: ${name}" >&2
    exit 2
  fi
done
if [[ "${CHECKPOINT_SOURCE}" == "released" || "${CHECKPOINT_SOURCE}" == "published" ]]; then
  for name in "${required_files[@]}"; do
    metadata="${MODEL_PATH}/.cache/huggingface/download/${name}.metadata"
    if [[ ! -f "${metadata}" ]] || [[ "$(head -n 1 "${metadata}")" != "${expected_checkpoint_revision}" ]]; then
      echo "Download ${checkpoint_repository}@${expected_checkpoint_revision}; invalid file: ${name}" >&2
      exit 2
    fi
  done
elif [[ ! -f "${MODEL_PATH}/trainer_state.json" || ! -f "${MODEL_PATH}/experiment_cfg/config.yaml" ]]; then
  echo "Trained checkpoint must contain trainer_state.json and experiment_cfg/config.yaml." >&2
  exit 2
fi
"${UV_BIN}" run --project "${REPO_ROOT}" --frozen --no-sync python -c \
  'import json, pathlib, sys; root = pathlib.Path(sys.argv[1]); config = json.loads((root / "config.json").read_text()); processor = json.loads((root / "processor_config.json").read_text()); modalities = processor["processor_kwargs"]["modality_configs"]["new_embodiment"]; config.get("model_type") == "Gr00tN1d7" or sys.exit("Expected Gr00tN1d7"); config.get("action_horizon") == 40 or sys.exit("Expected action_horizon 40"); modalities["action"]["delta_indices"] == list(range(40)) or sys.exit("Expected NEW_EMBODIMENT action horizon 40")' \
  "${MODEL_PATH}"
"${REPO_ROOT}/tools/launch_gr00t_eval.sh" \
  --arena-root "${ARENA_ROOT}" \
  --gr00t-root "${GR00T_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --robot "${ROBOT}" \
  --check
if [[ "${CHECK_ONLY}" == true ]]; then
  [[ "${#SERVER_ARGS[@]}" -eq 0 ]] || { echo "--check does not accept server arguments." >&2; exit 2; }
  exit 0
fi

exec "${REPO_ROOT}/tools/gr00t_runtime.sh" --gr00t-root "${GR00T_ROOT}" -- \
  python "${server_script}" \
  --model-path "${MODEL_PATH}" \
  --embodiment-tag NEW_EMBODIMENT \
  "${SERVER_ARGS[@]}"

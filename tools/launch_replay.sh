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

# Replay raw datasets only in the environment that owns their manifested action schema.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--check" ]]; then
  exec "${REPO_ROOT}/tools/arena_launch.sh" --check
fi

arena_lane=false
expect_lane=false
for argument in "$@"; do
  if [[ "${expect_lane}" == true ]]; then
    [[ "${argument}" == "native" || "${argument}" == "arena" ]] || {
      echo "--lane requires native or arena." >&2
      exit 2
    }
    [[ "${argument}" == "arena" ]] && arena_lane=true
    expect_lane=false
    continue
  fi
  case "${argument}" in
    --arena|--lane=arena) arena_lane=true ;;
    --lane=native) ;;
    --lane) expect_lane=true ;;
  esac
done
[[ "${expect_lane}" != true ]] || { echo "--lane requires native or arena." >&2; exit 2; }

if [[ "${arena_lane}" == true ]]; then
  exec "${REPO_ROOT}/tools/arena_launch.sh" python "${REPO_ROOT}/scripts/replay_demos.py" "$@"
fi

cd "${REPO_ROOT}"
exec uv run scripts/replay_demos.py "$@"

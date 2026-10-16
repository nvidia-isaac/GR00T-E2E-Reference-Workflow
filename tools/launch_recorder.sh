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

# Record demonstrations. The explicit lanes are `--recording-lane native|arena`.
# Native remains the default, and `--arena` remains a compatible alias. The Arena
# lane overlays the pinned checkout, so it must run through this launcher.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --check verifies the Arena pin; only the --arena path uses the Arena checkout.
if [[ "${1:-}" == "--check" ]]; then
  exec "${REPO_ROOT}/tools/arena_launch.sh" --check
fi

arena_lane=false
expect_lane=false
for argument in "$@"; do
  if [[ "${expect_lane}" == true ]]; then
    [[ "${argument}" == "native" || "${argument}" == "arena" ]] || {
      echo "--recording-lane requires native or arena." >&2
      exit 2
    }
    [[ "${argument}" == "arena" ]] && arena_lane=true
    expect_lane=false
    continue
  fi
  case "${argument}" in
    --arena|--recording-lane=arena) arena_lane=true ;;
    --recording-lane=native) ;;
    --recording-lane) expect_lane=true ;;
  esac
done
[[ "${expect_lane}" != true ]] || { echo "--recording-lane requires native or arena." >&2; exit 2; }

if [[ "${arena_lane}" == true ]]; then
  exec "${REPO_ROOT}/tools/arena_launch.sh" python "${REPO_ROOT}/scripts/record_demos.py" "$@"
fi

# Native lane: built-in Isaac Lab task, no Arena overlay.
cd "${REPO_ROOT}"
exec uv run scripts/record_demos.py "$@"

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

# Run a command against Isaac Lab Arena, in this repository's own environment. Arena's
# code is overlaid as source on the interpreter path -- its wheel ships no data files,
# so an installed copy cannot load the G1 joint order or the whole-body controller
# config -- while uv.lock owns its dependencies (the `arena` group). Every entry point
# that composes an Arena environment comes through here, which is also where the
# runtime.toml pin is enforced.
#
#   tools/arena_launch.sh python scripts/record_demos.py --arena ...
#   tools/arena_launch.sh --check     # verify the checkout against the pin, run nothing
#
# E2E_ARENA_ROOT names a checkout to use; without it a pinned one is fetched. Defaulting
# here rather than at each caller is what lets CI run these paths with no Arena-specific
# setup, exactly as a developer machine runs them.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN=${UV_BIN:-uv}
E2E_UV_EXTRA=${E2E_UV_EXTRA:-}
ARENA_ROOT=${E2E_ARENA_ROOT:-$("${REPO_ROOT}/tools/fetch_arena_checkout.sh")}
export E2E_ARENA_ROOT="${ARENA_ROOT}"

"${UV_BIN}" run \
  --project "${REPO_ROOT}" \
  --frozen \
  --no-sync \
  python "${REPO_ROOT}/tools/check_arena_pin.py" --arena-root "${ARENA_ROOT}" >/dev/null
if [[ "${1:-}" == "--check" ]]; then
  exit 0
fi

# One caller needs an optional extra of this project (the GR00T client), which is
# resolved rather than assumed present, so that call is the one that syncs.
run_args=(--project "${REPO_ROOT}" --frozen)
if [[ -n "${E2E_UV_EXTRA}" ]]; then
  run_args+=(--extra "${E2E_UV_EXTRA}")
else
  run_args+=(--no-sync)
fi

export PYTHONPATH="${ARENA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${UV_BIN}" run "${run_args[@]}" "$@"

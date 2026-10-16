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

# Materialize the Isaac Lab Arena source this app composes against, at the revision
# runtime.toml pins, and print where it is. Only the packages are fetched -- blobless,
# shallow, and sparse, without docs, submodules, or GR00T test data -- which is around
# 12 MB rather than the multi-gigabyte full history.
#
# A checkout already at the pinned revision is left exactly as it is, so pointing
# E2E_ARENA_ROOT at a full development clone keeps working. tools/check_arena_pin.py
# still has the last word on whether the result may be used.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST=${1:-${E2E_ARENA_ROOT:-${REPO_ROOT}/.arena}}
ARENA_REMOTE=${E2E_ARENA_REMOTE:-https://github.com/isaac-sim/IsaacLab-Arena.git}
UV_BIN=${UV_BIN:-uv}
SPARSE_PATHS=('/*' '!/docs/' '!/submodules/' '!isaaclab_arena_gr00t/tests/')

revision="$(
  "${UV_BIN}" run --project "${REPO_ROOT}" --frozen --no-sync \
    python "${REPO_ROOT}/tools/check_arena_pin.py" --print-pinned-revision
)"

if [[ -d "${DEST}/.git" ]] && [[ "$(git -C "${DEST}" rev-parse HEAD 2>/dev/null || true)" == "${revision}" ]]; then
  printf '%s\n' "${DEST}"
  exit 0
fi

if [[ -e "${DEST}" ]] && [[ "${DEST}" != "${REPO_ROOT}/.arena" ]]; then
  echo "${DEST} exists but is not at the pinned revision ${revision}." >&2
  echo "Check it out there yourself, or unset E2E_ARENA_ROOT to fetch a managed copy." >&2
  exit 2
fi

# Fetched by revision rather than by branch, so the pin decides what arrives. The
# destination sits inside this repository, so an existing checkout is detected by its
# .git directory: asking git would answer for the parent and operate on it.
#
# Callers read the path off stdout, so everything git has to say goes to stderr; an
# advisory line from some future git version would otherwise be read as part of the path.
{
  mkdir -p "${DEST}"
  [[ -d "${DEST}/.git" ]] || git init -q "${DEST}"
  git -C "${DEST}" remote add origin "${ARENA_REMOTE}" 2>/dev/null ||
    git -C "${DEST}" remote set-url origin "${ARENA_REMOTE}"
  GIT_LFS_SKIP_SMUDGE=1 git -C "${DEST}" fetch -q --depth 1 --filter=blob:none origin "${revision}"
  git -C "${DEST}" sparse-checkout set --no-cone "${SPARSE_PATHS[@]}"
  GIT_LFS_SKIP_SMUDGE=1 git -C "${DEST}" checkout -q --detach "${revision}"
} >&2

printf '%s\n' "${DEST}"

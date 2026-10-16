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

# Evaluate a policy on the Arena-composed environment through Arena's own policy runner.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--check" ]]; then
  exec "${REPO_ROOT}/tools/arena_launch.sh" --check
fi

# Arena's policy runner takes runner and policy options before the environment name and
# environment options after it. Arguments following a bare `--` are environment options.
runner_args=()
environment_args=()
after_separator=false
for argument in "$@"; do
  if [[ "${argument}" == "--" ]]; then
    after_separator=true
  elif [[ "${after_separator}" == true ]]; then
    environment_args+=("${argument}")
  else
    runner_args+=("${argument}")
  fi
done

exec "${REPO_ROOT}/tools/arena_launch.sh" \
  python -c \
  'from e2e_robot_workflows.compat import warmup_pinocchio_bindings; warmup_pinocchio_bindings(); from isaaclab_arena.evaluation.policy_runner import main; main()' \
  --external_environment_class_path \
  e2e_robot_workflows.arena.external_environment:E2EArenaExternalEnvironment \
  ${runner_args[@]+"${runner_args[@]}"} \
  e2e_static_apple \
  ${environment_args[@]+"${environment_args[@]}"}

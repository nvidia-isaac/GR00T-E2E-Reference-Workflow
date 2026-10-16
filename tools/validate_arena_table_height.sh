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

# Exercise height sampling, physical support, recording, and restore-only replay.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
RESULT_DIR=${ARENA_TABLE_HEIGHT_RESULT_DIR:-test-results/arena-table-height/$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "${RESULT_DIR}"
export ACCEPT_EULA=${ACCEPT_EULA:-Y}
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}

RESULT_PATH="${RESULT_DIR}/table-height-metrics.json" timeout --kill-after=30s 240s ./tools/arena_launch.sh \
  python tools/simulation/validate_arena_table_height.py 2>&1 | tee "${RESULT_DIR}/table-height.log"
python - "${RESULT_DIR}/table-height-metrics.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
assert result["status"] == "ok" and not result.get("cleanup_errors")
assert 0 < result["completed_steps"] <= result["requested_steps"]
assert [probe["offset_m"] for probe in result["probes"]] == [0.0, 0.03, -0.03, 0.03]
assert result["placement"]["placed"] and result["placement"]["outcome"] == "success"
PY

DATASET="${RESULT_DIR}/recording.hdf5"
timeout --kill-after=30s 240s ./tools/launch_recorder.sh --arena --headless --enable_cameras \
  --arena_embodiment g1_wbc_agile_pink --table_height_offset_range_m -0.03 0.03 \
  --seed 42 --num_demos 2 --sample_steps 120 --sample_export --dataset_file "${DATASET}" \
  2>&1 | tee "${RESULT_DIR}/recording.log"
python - "${DATASET}" <<'PY'
import sys
import h5py
from e2e_robot_workflows.data.dataset_manifest import SCENE_RANDOMIZATION_FIELD, validate_arena_recording_contract
manifest = validate_arena_recording_contract(sys.argv[1])
block = manifest["environment"][SCENE_RANDOMIZATION_FIELD]
assert block["table_height_offset_range_m"] == [-0.03, 0.03]
assert block["seed"] == manifest["seed"] == 42
with h5py.File(sys.argv[1]) as dataset:
    assert len(dataset["data"]) == 2
    heights = []
    for episode in dataset["data"].values():
        heights.append(float(episode[f"initial_state/rigid_object/{block['support_asset']}/root_pose"][0, 2]))
        assert block["support_asset"] not in episode["obs/datagen_info/object_pose"]
    assert heights[0] != heights[1], heights
    print(f"Recorded table root heights: {heights}")
PY

for mode in action state; do
  for order in 0,1 1,0; do
    result="${RESULT_DIR}/${mode}-${order}-replay"
    timeout --kill-after=30s 240s ./tools/launch_replay.sh --lane arena --viz none --strict \
      --seed 99 --select_episodes "${order}" --replay-mode "${mode}" --expected_episodes 2 \
      --dataset_file "${DATASET}" --result_file "${result}.json" 2>&1 | tee "${result}.log"
    python - "${result}.json" "${order}" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
assert result["status"] == "ok" and not result.get("cleanup_errors")
assert result["seed"] == 99 and result["completed_replays"] == 2
assert result["total_frames"] == 240
assert [episode["name"] for episode in result["episodes"]] == [f"demo_{i}" for i in sys.argv[2].split(",")]
PY
  done
done
if grep -H -E 'PhysX error: PxRigidDynamic::set(Linear|Angular)Velocity: Body must be non-kinematic' "${RESULT_DIR}"/*.log; then
  printf 'Invalid kinematic velocity commands were reported.\n' >&2
  exit 1
fi
printf 'Arena table-height randomization passed. Artifacts: %s\n' "${RESULT_DIR}"

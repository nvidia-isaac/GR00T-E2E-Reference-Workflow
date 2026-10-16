<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

(h2-evaluate)=
# H2 Policy Evaluation

Run a trained Unitree H2 GR00T policy in Isaac Lab-Arena and measure placement
success. Review the recorded motion separately to assess the full bimanual
pick, handover, and place sequence.

Learned-policy evaluation is implemented but not yet validated. The commands
below support integration checks and checkpoint assessment.

## Use a Published Checkpoint

```{include} ../_includes/artifacts/h2-checkpoint.md.inc
```

## Prerequisites

Use the published checkpoint above or a checkpoint from
[H2 Policy Training](fine-tune-h2.md), plus the assets from
[H2 Asset Preparation](prepare-h2.md). Run commands from the repository root on
the GPU workstation. Use one terminal for the policy server and another for
simulation evaluation.

## Preflight and Start the Policy Server

1. In the server terminal, fetch the managed Arena checkout and restore the
   GR00T and H2 asset paths. Use the `pink_urdf` value from the successful asset
   preparation JSON for `E2E_H2_PINK_URDF`. It ends in `unitree_h2.urdf`:

   ```bash
   export E2E_ROOT="$PWD"
   ```

   ```bash
   export ARENA_ROOT="$(./tools/fetch_arena_checkout.sh)"
   ```

   ```bash
   export ISAAC_GR00T_ROOT=/absolute/path/to/Isaac-GR00T
   ```

   ```bash
   export E2E_H2_ASSET_CACHE="$HOME/.cache/e2e-reference-app/h2plus"
   ```

   ```bash
   export E2E_H2_PINK_URDF="/absolute/path/to/unitree_h2.urdf"
   ```

   ```bash
   export E2E_H2_AGILE_POLICY=/absolute/path/to/agile_velocity.onnx
   ```

   See {ref}`prepare-arena-checkout` for checkout details. The standalone
   GR00T root is passed explicitly, so no Arena submodule is needed.

2. Choose **one** model. Run only the matching block in the same terminal:

   **Provided model** — downloaded above:

   ```bash
   export H2_CHECKPOINT="$E2E_ROOT/outputs/h2/published-checkpoint"
   export H2_CHECKPOINT_SOURCE=published
   ```

   **Your own model** — produced by policy training for H2:

   ```bash
   export H2_CHECKPOINT=/absolute/path/to/checkpoint-10000
   export H2_CHECKPOINT_SOURCE=trained
   ```

   Replace the path in the second block with your checkpoint directory. For
   the training smoke test, use `training-smoke/checkpoint-10` instead.
   Both commands below use your selected model and source.

3. Preflight the checkpoint:

   ```bash
   ./tools/launch_gr00t_server.sh \
     --robot h2 \
     --arena-root "$ARENA_ROOT" \
     --gr00t-root "$ISAAC_GR00T_ROOT" \
     --model-path "$H2_CHECKPOINT" \
     --checkpoint-source "$H2_CHECKPOINT_SOURCE" \
     --check
   ```

   The check must report the H2 policy with 75 state values, 61 action values,
   and a 40-step horizon.

4. Start the policy server in the same terminal:

   ```bash
   ./tools/launch_gr00t_server.sh \
     --robot h2 \
     --arena-root "$ARENA_ROOT" \
     --gr00t-root "$ISAAC_GR00T_ROOT" \
     --model-path "$H2_CHECKPOINT" \
     --checkpoint-source "$H2_CHECKPOINT_SOURCE" \
     --device cuda \
     --host 127.0.0.1 \
     --port 5555
   ```

5. Wait for `Server is ready and listening on tcp://127.0.0.1:5555` before
   launching evaluation. If checking only that `checkpoint-10` loads, follow
   [Shut Down](#shut-down) and return to training.

## Run a Visual Smoke Test

1. In the evaluation terminal, restore the runtime and output paths. Use the
   same `pink_urdf` and AGILE policy paths as the server terminal:

   ```bash
   export ARENA_ROOT="$(./tools/fetch_arena_checkout.sh)"
   ```

   ```bash
   export ISAAC_GR00T_ROOT=/absolute/path/to/Isaac-GR00T
   ```

   ```bash
   export H2_RUN_DIR=/absolute/path/to/outputs/h2/YYYYMMDD_HHMMSS
   ```

   ```bash
   export E2E_H2_ASSET_CACHE="$HOME/.cache/e2e-reference-app/h2plus"
   ```

   ```bash
   export E2E_H2_PINK_URDF="/absolute/path/to/unitree_h2.urdf"
   ```

   ```bash
   export E2E_H2_AGILE_POLICY=/absolute/path/to/agile_velocity.onnx
   ```

2. Run one episode:

   ```bash
   OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
   ./tools/launch_gr00t_eval.sh \
     --robot h2 \
     --arena-root "$ARENA_ROOT" \
     --gr00t-root "$ISAAC_GR00T_ROOT" \
     --remote_host 127.0.0.1 \
     --remote_port 5555 \
     --num_episodes 1 \
     --device cpu \
     --policy_device cpu \
     --output_base_dir "$H2_RUN_DIR/evaluation-smoke" \
     --viz kit \
     -- --episode_length_s 30
   ```

   Confirm the episode completes and writes the Arena metrics and report. A
   one-episode run checks integration; it does not qualify learned-policy quality.

## Evaluate a Batch

1. After the smoke test completes, run 100 episodes in the same evaluation
   terminal:

   ```bash
   OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
   ./tools/launch_gr00t_eval.sh \
     --robot h2 \
     --arena-root "$ARENA_ROOT" \
     --gr00t-root "$ISAAC_GR00T_ROOT" \
     --remote_host 127.0.0.1 \
     --remote_port 5555 \
     --num_episodes 100 \
     --device cpu \
     --policy_device cpu \
     --record_camera_video \
     --output_base_dir "$H2_RUN_DIR/evaluation" \
     --viz none \
     -- --episode_length_s 30
   ```

   Both examples allow 30 seconds of simulation time per episode. This changes
   only the time budget. Inference runs on the server GPU; simulation and client
   tensors use the CPU. Use distinct output directories and seeds for independent
   evaluation batches.

2. Open `index.html` in the timestamped directory under
   `$H2_RUN_DIR/evaluation/`. Check the completed episode count and review the
   per-episode results and camera videos:

   | Output | Contents |
   | --- | --- |
   | `episode_results_rank0.jsonl` | One record per completed episode |
   | `index.html` | Browsable evaluation report |
   | `robot-cam-env0-*-episode-*.mp4` | Per-episode policy-camera videos |

   The final terminal metrics line identifies the completed episode count and
   rates. For example, a ten-episode run with a locally trained checkpoint at
   step 10000, seed 43, and a 30-second episode budget printed:

   ```text
   [Rank 0/1] Metrics: {'num_episodes': 10, 'success_rate': 0.8, 'object_moved_rate': 1.0}
   ```

   That run reported eight placement successes and movement in all ten
   episodes. Use this example to recognize the output format; measure your
   selected checkpoint on the current task separately.
   Compare `num_episodes` with your requested count and the records in
   `episode_results_rank0.jsonl`. An incomplete run must not be reported as a
   completed 100-episode evaluation.

:::{important}
`success_rate` measures **placement success**: supported, settled contact between
apple and plate with both hands clear. It does not check pickup or handover
history. Review the recorded sequence separately before claiming complete
bimanual task success, and report that assessment apart from placement success.
:::

Compare rates only for a checkpoint trained on the matching H2 scene,
state and action formats, and camera stream. See
[H2 success criteria](../explanation/h2-architecture.md#success-and-handover)
for how the criteria for accepting a recording differ from evaluation.

## Shut Down

1. After the evaluator exits, return to the server terminal and press
   **Ctrl+C**. Wait for `Shutting down server...`.

2. Confirm port 5555 has no listener before starting another server:

   ```bash
   ss -ltnp 'sport = :5555'
   ```

   The output should contain no `LISTEN` entry. The H2 evaluator closes its
   client connection but leaves the separately launched policy server running.

## Expected Result

The timestamped evaluation directory contains the episode records, report, and
camera videos. Report placement success with the completed episode count and
checkpoint used. Full handover success remains a separate assessment.

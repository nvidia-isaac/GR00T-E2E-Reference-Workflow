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

(h2-train)=
# H2 Policy Training

Fine-tune GR00T N1.7 on the converted dataset of Unitree H2 demonstrations to produce a checkpoint
for simulation evaluation.

## Prerequisites

Use the published dataset below or complete
[H2 Dataset Conversion](convert-h2-dataset.md). Training uses the
standalone GR00T environment prepared in [H2 Asset Preparation](prepare-h2.md).
Open a fresh workstation terminal outside any distributed training job, then
run the following commands from the workflow repository root.

The 1,000-demo H2 recipe was validated on one 48 GB NVIDIA RTX 5880 Ada. It freezes
the visual backbone and LLM while training the projector and diffusion model.
Other GPU capacities may require a smaller batch.

## Use a Published Dataset

```{include} ../_includes/artifacts/h2-lerobot.md.inc
```

## Set the Training Inputs

1. Set the workspace and output paths in this terminal. For the provided
   dataset, choose a new `H2_RUN_DIR` for training outputs. For your own dataset,
   use the run directory containing your converted data:

   ```bash
   export E2E_ROOT="$PWD"
   ```

   ```bash
   export ISAAC_GR00T_ROOT=/absolute/path/to/Isaac-GR00T
   ```

   ```bash
   export H2_RUN_DIR=/absolute/path/to/outputs/h2/YYYYMMDD_HHMMSS
   ```

   ```bash
   export H2_MODALITY="$E2E_ROOT/src/e2e_robot_workflows/gr00t/h2_static_apple_modality_config.py"
   ```

   ```bash
   export TRAIN_OUTPUT="$H2_RUN_DIR/training"
   ```

   ```bash
   export H2_SMOKE_OUTPUT="$H2_RUN_DIR/training-smoke"
   ```

2. Choose **one** dataset. Run only the matching block:

   **Provided dataset** — downloaded above:

   ```bash
   export H2_LEROBOT="$E2E_ROOT/outputs/h2/downloads/generated/lerobot"
   ```

   **Your own dataset** — converted in your run directory:

   ```bash
   export H2_LEROBOT="$H2_RUN_DIR/lerobot"
   ```

   Both training commands below use your selected dataset.

3. Obtain access to both
   [GR00T N1.7](https://huggingface.co/nvidia/GR00T-N1.7-3B) and
   [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B), then
   authenticate and download the base model:

   ```bash
   "$ISAAC_GR00T_ROOT/.venv/bin/hf" auth login
   ```

   Verify the active account:

   ```bash
   "$ISAAC_GR00T_ROOT/.venv/bin/hf" auth whoami
   ```

   Download the base model:

   ```bash
   "$ISAAC_GR00T_ROOT/.venv/bin/hf" download nvidia/GR00T-N1.7-3B
   ```

   Check access to the processor configuration:

   ```bash
   "$ISAAC_GR00T_ROOT/.venv/bin/hf" download nvidia/Cosmos-Reason2-2B config.json
   ```

   The Cosmos download checks access before training loads its processor.

## Run a Training Smoke Test

1. In the same terminal, train for ten steps into a separate directory:

   ```bash
   (cd "$ISAAC_GR00T_ROOT" && \
   "$E2E_ROOT/tools/gr00t_runtime.sh" --gr00t-root "$ISAAC_GR00T_ROOT" -- \
     python gr00t/experiment/launch_finetune.py \
     --base-model-path nvidia/GR00T-N1.7-3B \
     --dataset-path "$H2_LEROBOT" \
     --output-dir "$H2_SMOKE_OUTPUT" \
     --embodiment-tag new_embodiment \
     --modality-config-path "$H2_MODALITY" \
     --global-batch-size 32 \
     --max-steps 10 \
     --num-gpus 1 \
     --save-steps 10 \
     --save-total-limit 5 \
     --no-tune-llm \
     --no-tune-visual \
     --tune-projector \
     --tune-diffusion-model \
     --dataloader-num-workers 4 \
     --color-jitter-params \
       brightness 0.3 \
       contrast 0.4 \
       saturation 0.5 \
       hue 0.08)
   ```

2. Require finite loss and `$H2_SMOKE_OUTPUT/checkpoint-10`. Use the
   [evaluation preflight and server startup](evaluate-h2-checkpoint.md#preflight-and-start-the-policy-server)
   with that checkpoint to verify it loads. Shut down the server before
   returning to this training terminal.

## Train the Full Checkpoint

1. Start a new run from the base model using the same validated dataset and
   modality:

   ```bash
   (cd "$ISAAC_GR00T_ROOT" && \
   "$E2E_ROOT/tools/gr00t_runtime.sh" --gr00t-root "$ISAAC_GR00T_ROOT" -- \
     python gr00t/experiment/launch_finetune.py \
     --base-model-path nvidia/GR00T-N1.7-3B \
     --dataset-path "$H2_LEROBOT" \
     --output-dir "$TRAIN_OUTPUT" \
     --embodiment-tag new_embodiment \
     --modality-config-path "$H2_MODALITY" \
     --global-batch-size 32 \
     --max-steps 10000 \
     --num-gpus 1 \
     --save-steps 1000 \
     --save-total-limit 5 \
     --no-tune-llm \
     --no-tune-visual \
     --tune-projector \
     --tune-diffusion-model \
     --dataloader-num-workers 4 \
     --color-jitter-params \
       brightness 0.3 \
       contrast 0.4 \
       saturation 0.5 \
       hue 0.08)
   ```

2. After training finishes, select the final checkpoint:

   ```bash
   export H2_CHECKPOINT="$TRAIN_OUTPUT/checkpoint-10000"
   ```

## Expected Result

`$H2_RUN_DIR/training/checkpoint-10000` contains the model, processor, statistics,
and training configuration needed by the H2 evaluation preflight. Checkpoints
are saved every 1,000 steps, retaining the most recent five.

## Next Stage

Continue to [H2 Policy Evaluation](evaluate-h2-checkpoint.md).

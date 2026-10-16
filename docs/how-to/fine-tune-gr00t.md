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

(fine-tune-nvidia-isaac-gr00t)=
(g1-policy-training)=
(g1-dataset-conversion-and-policy-training)=
# G1 Policy Training

Train a GR00T N1.7 policy on a converted Unitree G1 apple-to-plate dataset in
NVIDIA Isaac GR00T's separate environment. The output is a checkpoint for
[G1 policy evaluation](evaluate-checkpoint.md).

Use your own dataset converted in the previous module, or download the
training-ready dataset below. Both paths continue with training setup on this
page; the provided dataset does not require the earlier data-preparation modules.

## Use a Published Dataset

```{include} ../_includes/artifacts/g1-lerobot.md.inc
```

(convert-recording)=
(convert-the-recorded-hdf5-into-a-gr00t-lerobot-dataset)=
## Prerequisites

- A downloaded LeRobot dataset above, or a validated dataset from
  [G1 Dataset Conversion](convert-g1-dataset.md).
- Data recorded with Pink IK and AGILE in either Isaac Lab-Arena or native
  Isaac Lab, including their AutoData outputs.
  Both supply the navigation and base-height commands required by the G1
  modality. Native fixed-base data omits these fields and needs custom training
  and evaluation integration outside this recipe.
- A training GPU with at least 48 GB VRAM. The documented single-GPU
  configuration uses an NVIDIA RTX 6000 Ada; 128 GB system RAM is recommended.
  The recorded estimate is about 2–3 hours for 20,000 steps on that GPU.
- Authorized Hugging Face access to both
  [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) and
  [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B).
  The training processor loads Cosmos as a separate dependency.

Run the preparation commands from this repository's root. Training uses
a standalone Isaac-GR00T checkout, pinned by `runtime.toml`. For Unitree H2,
use [H2 policy training](fine-tune-h2.md).

(prepare-gr00t-checkout)=
## Prepare NVIDIA Isaac GR00T in Your Workspace

1. From this repository's root, create the standalone checkout and its locked
   environment. Run these blocks in order in the same terminal:

   ```bash
   export E2E_ROOT="$PWD"
   export ISAAC_GR00T_ROOT=/absolute/path/outside/this/repo/Isaac-GR00T
   ```

   ```bash
   git clone https://github.com/NVIDIA/Isaac-GR00T.git "$ISAAC_GR00T_ROOT"
   git -C "$ISAAC_GR00T_ROOT" checkout 4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e
   (cd "$ISAAC_GR00T_ROOT" && uv sync --frozen)
   "$E2E_ROOT/tools/gr00t_runtime.sh" \
     --gr00t-root "$ISAAC_GR00T_ROOT" --check
   ```

   Keep this checkout clean. For an existing checkout at that revision, sync
   its environment and run the same runtime check.

(download-the-gr00t-base-model)=
## Model Access

1. Login to HF and confirm authentication

   ```bash
   "$ISAAC_GR00T_ROOT/.venv/bin/hf" auth login
   ```

   ```bash
   "$ISAAC_GR00T_ROOT/.venv/bin/hf" auth whoami
   ```

2. Download the base model and verify Cosmos access before training:

   ```bash
   "$ISAAC_GR00T_ROOT/.venv/bin/hf" download nvidia/GR00T-N1.7-3B
   ```

   ```bash
   "$ISAAC_GR00T_ROOT/.venv/bin/hf" download nvidia/Cosmos-Reason2-2B config.json
   ```

   Both commands must succeed with the credentials used for training. The
   processor downloads its remaining Cosmos files when training starts. A
   locally cached GR00T base model does not establish Cosmos access.

(fine-tune-on-the-dataset)=
## Training Inputs

1. Open a fresh terminal outside any distributed training job **before** setting
   the paths. Set them in that terminal using absolute values:

   ```bash
   export E2E_ROOT=/absolute/path/to/e2e-reference-app
   export ISAAC_GR00T_ROOT=/absolute/path/to/Isaac-GR00T
   export TRAIN_OUTPUT="$E2E_ROOT/outputs/g1/training"
   export MODALITY_CONFIG="$E2E_ROOT/src/e2e_robot_workflows/gr00t/g1_static_apple_modality_config.py"
   ```

   Use a new training output directory for an independent run.

2. Choose **one** dataset using the tabs below. Run only the command for your chosen input:

   ::::{tab-set}

   :::{tab-item} Provided dataset

   Use the training-ready dataset downloaded above:

   ```bash
   export TRAIN_DATASET="$E2E_ROOT/outputs/g1/downloads/e2e_ra_g1_apple_to_plate_2500_generated_demos_lerobot"
   ```

   :::

   :::{tab-item} Your own dataset

   Use the dataset converted from your recordings:

   ```bash
   export TRAIN_DATASET="$E2E_ROOT/outputs/g1/lerobot"
   ```

   :::

   :::{tab-item} AutoData dataset

   Use the converted AutoData output:

   ```bash
   export TRAIN_DATASET="$E2E_ROOT/outputs/g1/lerobot-generated"
   ```

   :::
   ::::

   The fine-tuning command below uses your selected dataset.

(run-gr00t-finetune)=
(run-the-shared-fine-tuning-command)=
## Run Fine-Tuning

1. In the **same terminal** where the training variables are set, launch one
   GPU directly through Python:

   ```bash
   (cd "$ISAAC_GR00T_ROOT" && \
   "$E2E_ROOT/tools/gr00t_runtime.sh" --gr00t-root "$ISAAC_GR00T_ROOT" -- \
     python gr00t/experiment/launch_finetune.py \
     --base-model-path nvidia/GR00T-N1.7-3B \
     --dataset-path "$TRAIN_DATASET" \
     --output-dir "$TRAIN_OUTPUT" \
     --embodiment-tag new_embodiment \
     --modality-config-path "$MODALITY_CONFIG" \
     --global-batch-size 12 \
     --max-steps 20000 \
     --num-gpus 1 \
     --save-steps 5000 \
     --save-total-limit 5 \
     --no-tune-llm \
     --tune-visual \
     --tune-projector \
     --tune-diffusion-model \
     --dataloader-num-workers 8 \
     --color-jitter-params \
       brightness 0.3 \
       contrast 0.4 \
       saturation 0.5 \
       hue 0.08)
   ```

   :::{important}
   Use the G1 modality with `--embodiment-tag new_embodiment`. It specifies the
   40-step horizon required by GR00T N1.7. The built-in
   `unitree_g1_full_body_with_waist_height_nav_cmd` embodiment option will fail.
   :::

   This configuration tunes the visual backbone, projector, and diffusion
   model while freezing the LLM. Reduce the batch size if GPU memory is
   insufficient.

2. Confirm that training completes with finite loss and writes
   `"$TRAIN_OUTPUT/checkpoint-20000"`. Intermediate checkpoints are saved every
   5,000 steps. Keep the full checkpoint directory, including its processor,
   statistics, trainer state, and experiment configuration.

## Next Stage

Continue to [G1 policy evaluation](evaluate-checkpoint.md) with
`outputs/g1/training/checkpoint-20000`. Its preflight validates the checkpoint's
required files and modality before starting the server.

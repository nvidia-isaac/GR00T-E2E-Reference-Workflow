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

(fine-tuning)=
(unitree-g1-fine-tuning)=
# Unitree G1 Policy Training

Fine-tuning adapts GR00T N1.7 to a LeRobot dataset of physical G1 demonstrations and saves
checkpoints for LEAPP export.

<a id="hardware-requirement"></a>

## Prerequisites

- Run on an x86_64 workstation with an L40-class GPU or better (40+ GB VRAM).
- Copy the complete dataset from [Data Export](real-data-export.md) to this
  workstation. Use its absolute local path as `<dataset-dir>` below.
- Choose an absolute `<output-dir>` for the training run.
- Install `uv` using the [host tools instructions](../installation.md#host-tools).

<a id="fine-tune-gr00t-1-7-on-real-data"></a>

## Prepare GR00T N1.7

Use the [`nvidia-isaac/gr00t-leapp-export`](https://github.com/nvidia-isaac/gr00t-leapp-export/tree/gr00t_workflow_0.1) repository and the `gr00t_workflow_0.1` tag for fine-tuning a policy for Unitree G1 and downstream export compatibility.

:::{tip}
If you want to skip fine-tuning, use the provided [`nvidia/GR00T-N1.7-ApplePnP-V1`](https://huggingface.co/nvidia/GR00T-N1.7-ApplePnP-V1) real-robot model.
Continue to [LEAPP Export](real-leapp-export.md) with a local copy of the model
and its corresponding LeRobot dataset.
:::

1. Clone `gr00t-leapp-export`:

   ```bash
   git clone --branch gr00t_workflow_0.1 --recurse-submodules https://github.com/nvidia-isaac/gr00t-leapp-export.git
   ```

   Enter the repository and install its environment in the same terminal:

   ```bash
   cd gr00t-leapp-export
   ```

   ```bash
   uv sync --python 3.10
   ```

2. Accept model access terms for [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) and [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B), then authenticate:

   ```bash
   uv run hf auth login
   ```

<a id="configuration"></a>

## Training Configuration

| Parameter | Description | Value |
|-----------|------------|-------|
| `--base-model-path` | Pretrained GR00T weights | `nvidia/GR00T-N1.7-3B` |
| `--dataset-path` | Path to real-world LeRobot dataset | `<dataset-dir>` |
| `--embodiment-tag` | Robot embodiment identifier | `NEW_EMBODIMENT` |
| `--max-steps` | Total training steps | `10000` |
| `--save-steps` | Checkpoint frequency | `2000` |
| `--global-batch-size` | Effective batch size | `32` |

## Validate the Dataset

1. Before training, validate `<dataset-dir>/meta/info.json`:

   - `fps` matches the recorder `sync_rate` (30 Hz by default).
   - `total_episodes` and `total_frames` reflect the expected recording volume.

   Use at least 200 successful episodes for the baseline apple-to-plate recipe,
   following the [recording quality guidance](real-record.md#recording-quality).

## Modality Config

For datasets of Unitree G1 demonstrations, the converter writes a default modality config file:

```text
<dataset-dir>/new_embodiment_config_defaults.py
```

Use this file directly in the fine-tuning command.

It includes:

- Seven Unitree G1 `state` keys from `left_leg` through `right_hand`.
- `RELATIVE` action representation for arms.
- `ABSOLUTE` action representation for hands, waist, and optional locomotion or effort keys.
- The model-default 16-step action prediction horizon.
- Per-body-part `effort_<group>` action keys only when the recorder observed non-zero feed-forward torques.

:::{note}
For the default apple-to-plate dataset, no additional modality edits are required.
:::

## Launch Training

1. Run fine-tuning from the root of `gr00t-leapp-export`:

   ```bash
   CUDA_HOME=/usr/local/cuda \
   CUDA_VISIBLE_DEVICES=0 \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   uv run python gr00t/experiment/launch_finetune.py \
       --base-model-path nvidia/GR00T-N1.7-3B \
       --dataset-path <dataset-dir> \
       --embodiment-tag NEW_EMBODIMENT \
       --modality-config-path <dataset-dir>/new_embodiment_config_defaults.py \
       --num-gpus 1 \
       --output-dir <output-dir> \
       --max-steps 10000 \
       --save-steps 2000 \
       --global-batch-size 32 \
       --dataloader-num-workers 4
   ```

The training script saves checkpoints to `<output-dir>/checkpoint-<step>/`.

The `--max-steps`, `--save-steps`, and `--global-batch-size` values are starting points. Tune them for dataset size and available hardware.

:::{seealso}
Refer to the following resources for additional detail:

- [Fine-tune on a New Embodiment](https://github.com/nvidia-isaac/gr00t-leapp-export/blob/gr00t_workflow_0.1/getting_started/finetune_new_embodiment.md)
- [Modality Configuration](https://github.com/nvidia-isaac/gr00t-leapp-export/blob/gr00t_workflow_0.1/getting_started/data_config.md)
- [Hardware Recommendation](https://github.com/nvidia-isaac/gr00t-leapp-export/blob/gr00t_workflow_0.1/getting_started/hardware_recommendation.md)
:::

<a id="key-takeaways"></a>

## Next Stage

Confirm that `<output-dir>/checkpoint-<step>/` exists after training completes.
Use the chosen checkpoint directory as `<checkpoint-dir>` in
[Unitree G1 LEAPP Export](real-leapp-export.md), together with the same
`<dataset-dir>` used for training.

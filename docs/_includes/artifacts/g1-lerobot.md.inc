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

We provide [a training-ready LeRobot dataset of 2,500 generated G1 demonstrations](https://huggingface.co/datasets/nvidia/GR00T-E2E-Reference-Workflow-SIM-G1-N1.7-AppletoPlate-Dataset).
It is already converted: skip recording, data generation, and conversion and
continue with training setup after downloading it.

The same Hugging Face repository also contains source HDF5 files. Those are
separate from the LeRobot dataset used here. The command below downloads only
the converted `e2e_ra_g1_apple_to_plate_2500_generated_demos_lerobot` directory.
Run it from the workflow repository root.

If you converted your own dataset in the previous module, skip this download
and continue to {ref}`G1 policy training setup <prepare-gr00t-checkout>` below.

Before downloading, complete [Hugging Face access setup](../reference/models-and-datasets.md#access-and-storage).

```bash
uv run --frozen --no-sync hf download nvidia/GR00T-E2E-Reference-Workflow-SIM-G1-N1.7-AppletoPlate-Dataset \
  --revision 36234f4b48dad27f218d8e29ec7710ad5a90b48c \
  --repo-type dataset \
  --include 'e2e_ra_g1_apple_to_plate_2500_generated_demos_lerobot/**' \
  --local-dir outputs/g1/downloads
```

The dataset is saved in
`outputs/g1/downloads/e2e_ra_g1_apple_to_plate_2500_generated_demos_lerobot`,
which contains `data/`, `videos/`, and `meta/`. Select **Provided dataset** in
{ref}`Training Inputs <fine-tune-on-the-dataset>`.
Continue with {ref}`G1 policy training setup <prepare-gr00t-checkout>`,
starting with GR00T preparation and model access. The demonstrations use G1 in Isaac Lab-Arena and include navigation and
base-height commands; demonstrations recorded with a fixed-base G1 omit these commands.

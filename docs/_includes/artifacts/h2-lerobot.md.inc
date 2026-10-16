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

We provide [a dataset of H2 demonstrations](https://huggingface.co/datasets/nvidia/GR00T-E2E-Reference-Workflow-SIM-H2-N1.7-AppleHandoverPlate-Dataset) so you can skip recording, data generation, and
conversion, or check your work against our reference. Run the command below
from the workflow repository root to download it.

To use your own converted dataset, skip this download and continue with
[policy training inputs](../how-to/fine-tune-h2.md#set-the-training-inputs).

Before downloading, complete [Hugging Face access setup](../reference/models-and-datasets.md#access-and-storage).

```bash
uv run --frozen --no-sync hf download nvidia/GR00T-E2E-Reference-Workflow-SIM-H2-N1.7-AppleHandoverPlate-Dataset \
  --revision 72ad851a5d0b845d368f031c744876613fb0b3c7 \
  --repo-type dataset \
  --include 'generated/lerobot/**' \
  --local-dir outputs/h2/downloads
```

The dataset is saved in `outputs/h2/downloads/generated/lerobot`. Continue to
[Set the Training Inputs](../how-to/fine-tune-h2.md#set-the-training-inputs)
and select **Provided dataset**. Use a new `H2_RUN_DIR` for training outputs.
This dataset contains 1,000 generated episodes; the 10 human seeds are not extra training episodes. It is
self-contained and does not need the raw HDF5 files. To run the
[training-input validation](../how-to/convert-h2-dataset.md#validate-the-training-input),
set `H2_TRAINING_DEMOS=1000` and keep the downloaded `H2_LEROBOT` path.

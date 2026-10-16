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

We provide [a trained policy for G1](https://huggingface.co/nvidia/GR00T-E2E-Reference-Workflow-SIM-G1-N1.7-AppletoPlate-Model) so you can skip recording, data generation,
conversion, and training, or check your work against our reference. Run the
command below from the workflow repository root to download it, then follow [G1 policy evaluation](../how-to/evaluate-checkpoint.md) to run it
in simulation.

If you want to train your own model, return to
[G1 policy training](../how-to/fine-tune-gr00t.md).

Before downloading, complete [Hugging Face access setup](../reference/models-and-datasets.md#access-and-storage).

```bash
uv run --frozen --no-sync hf download nvidia/GR00T-E2E-Reference-Workflow-SIM-G1-N1.7-AppletoPlate-Model \
  --revision d3075997eb71095238324746f0507f5d707cd895 \
  --exclude optimizer.pt rng_state.pth scheduler.pt trainer_state.json training_args.bin \
  --local-dir outputs/g1/published-checkpoint
```

The model is saved in `outputs/g1/published-checkpoint`. Keep the
`.cache/huggingface/` directory created inside the download directory; the
launcher uses it to check that the model matches the pinned download.

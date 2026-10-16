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

We provide [a trained policy for H2](https://huggingface.co/nvidia/GR00T-E2E-Reference-Workflow-SIM-H2-N1.7-AppleHandoverPlate-Model) so you can skip recording, data generation,
conversion, and training, or check your work against our reference. Run the
command below from the workflow repository root to download it, then follow [H2 policy evaluation](../how-to/evaluate-h2-checkpoint.md) to run it
in simulation.

If you want to train your own model, proceed to
[H2 policy training](../how-to/fine-tune-h2.md).

Before downloading, complete [Hugging Face access setup](../reference/models-and-datasets.md#access-and-storage).

```bash
uv run --frozen --no-sync hf download nvidia/GR00T-E2E-Reference-Workflow-SIM-H2-N1.7-AppleHandoverPlate-Model \
  --revision 13397daebd2c34986fc18820374a4a2d33861482 \
  --exclude optimizer.pt rng_state.pth scheduler.pt trainer_state.json training_args.bin \
  --local-dir outputs/h2/published-checkpoint
```

The model is saved in `outputs/h2/published-checkpoint`. Select **Provided model**
when [setting up policy evaluation for H2](../how-to/evaluate-h2-checkpoint.md#preflight-and-start-the-policy-server);
that choice sets the checkpoint path and source for both preflight and server startup.
Keep the `.cache/huggingface/` directory created inside the download directory;
the launcher uses it to check that the model matches the pinned download.

The inference download does not need optimizer or trainer-resume state.
Model files, weight shards, processor, normalization statistics, embodiment,
and action-horizon checks still apply. Complete the simulator and GR00T setup
on the evaluation page; no headset is needed.

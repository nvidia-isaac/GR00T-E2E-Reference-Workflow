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

# Models and Datasets

We provide recorded HDF5, generated HDF5, LeRobot datasets, and trained models
for G1 and H2. Choose the input for the stage you want to run, then follow that
stage's setup instructions. The table lists earlier stages you can skip when
starting with each published input. Complete the simulation and model setup
required by the stage you select.

| Published Input | Earlier Stages to Skip | Start Here |
| --- | --- | --- |
| Recorded HDF5 (teleoperated recording; conversion-only) | Recording and generation | Dataset conversion |
| Generated HDF5 (AutoData-amplified; conversion-only) | Recording and generation | Dataset conversion |
| LeRobot dataset | Recording, generation and conversion | Policy training |
| Model | Recording through training | Policy evaluation |

AutoData generation requires a recording whose saved software versions and
simulation settings match this checkout. The published HDF5 files use different
settings and are provided for conversion.

Complete [Installation](../installation.md) first. For H2 conversion,
[prepare the GR00T runtime](../how-to/prepare-h2.md#prepare-the-gr00t-runtime)
and keep `ISAAC_GR00T_ROOT` set in this terminal. H2 simulation and replay
also need [Asset Preparation](../how-to/prepare-h2.md).

## Access and Storage

Use the Hugging Face links below to find the workflow models and datasets.
If a download requires authentication, sign in from the installed workflow
environment:

```bash
uv run --frozen --no-sync hf auth login
```

```bash
uv run --frozen --no-sync hf auth whoami
```

For training and inference, follow the GR00T N1.7 and Cosmos-Reason2-2B setup
instructions on the stage pages. If a download returns 401/403, check the active
account and follow the instructions on the repository page.

These commands use immutable revisions recorded in `runtime.toml` under
`artifacts`.

HDF5 and LeRobot downloads use the `dataset_revision` pin. Models use
`model_revision`. Keep the metadata embedded in each HDF5 file unchanged. It
records software versions, simulation settings, and the source recording for
generated data. Conversion preserves this information.

Download only the dataset or model you need. Choose a new output directory for
training so downloaded files and previous runs remain intact.

(g1-downloads)=
## Published G1 Recordings

The published 25-demonstration HDF5 recording (779 MB) is not compatible with the
current converter's environment contract. For training, download the
[already-converted LeRobot dataset](../how-to/fine-tune-gr00t.md#use-a-published-dataset)
instead. To convert your own recordings, follow
[G1 dataset conversion](../how-to/convert-g1-dataset.md#convert-recorded-demonstrations).

## G1 Generated Data for Conversion

Download [2,500 generated G1 demonstrations](../how-to/convert-g1-dataset.md#use-published-generated-data)
(79.2 GB).

## Demonstration Dataset for G1 Policy Training

```{include} ../_includes/artifacts/g1-lerobot.md.inc
```

## Trained Policy for G1

```{include} ../_includes/artifacts/g1-checkpoint.md.inc
```

Continue with [G1 policy training](../how-to/fine-tune-gr00t.md) or [evaluation](../how-to/evaluate-checkpoint.md).

(h2-downloads)=
## Recorded H2 Demonstrations for Conversion

Download [10 recorded H2 demonstrations](../how-to/convert-h2-dataset.md#use-a-published-recording)
(1.18 GB).

## H2 Generated Data for Conversion

Download [17 generated H2 demonstrations](../how-to/convert-h2-dataset.md#use-published-generated-demonstrations)
(1.89 GB).

## Demonstration Dataset for H2 Policy Training

```{include} ../_includes/artifacts/h2-lerobot.md.inc
```

## Trained Policy for H2

```{include} ../_includes/artifacts/h2-checkpoint.md.inc
```

Continue with [H2 policy training](../how-to/fine-tune-h2.md) or [evaluation](../how-to/evaluate-h2-checkpoint.md).

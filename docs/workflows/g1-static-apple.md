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

# Unitree G1 Static Apple Pick & Place

The simulated Static Apple Pick & Place workflow path uses a Unitree G1 to
move an apple to a plate in NVIDIA Isaac Lab. It carries those demonstrations
through GR00T N1.7 training and closed-loop evaluation.

<span class="workflow-status">AVAILABLE</span>

## Understand the Workflow

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Conceptual Overview
:link: ../explanation/concepts-overview
:link-type: doc
:class-card: workflow-card

Understand the role of each stage and the difference between simulation and
physical data collection.
:::

:::{grid-item-card} Architecture and Data Requirements
:link: ../explanation/architecture
:link-type: doc
:class-card: workflow-card

Review the pipeline, robot-control software, and data requirements before
running the stages.
:::

::::

## Before You Start

Complete [Installation](../installation.md) to prepare the workstation and
simulation environment. Recording demonstrations requires an XR headset;
check the [XR headset, handheld controller, and network requirements](../installation.md#xr-recording-requirements)
before starting the recording stage.

Use the [Environment Variable Reference](../reference/environment-variables.md)
when restoring runtime and run settings in a new terminal.

## Follow the Workflow Path

Follow the five stages in order, or use the
{ref}`published stage inputs <g1-downloads>`
to start at conversion, training, or evaluation. After recording, optional
[Isaac AutoData generation](../how-to/generate-data-with-autodata.md) can expand
your demonstrations before conversion. The recording guide uses Isaac Lab-Arena
with Pink IK for upper-body control and AGILE for lower-body control
(`g1_wbc_agile_pink`). Native Isaac Lab with AGILE is a supported alternative;
both provide the command fields required by the supplied training configuration.

:::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Simulation Practice
:link: ../tutorials/getting-started
:link-type: doc
:class-card: workflow-step-card

<span class="workflow-step-number">1</span>

Launch the scene and practice driving the Unitree G1 with AGILE.
:::

:::{grid-item-card} Demonstration Recording
:link: ../how-to/record-demonstrations
:link-type: doc
:class-card: workflow-step-card

<span class="workflow-step-number">2</span>

Record HDF5 demonstrations in Isaac Lab-Arena with Pink IK and AGILE using a
supported XR device. The task scores each episode automatically.
:::

:::{grid-item-card} Dataset Conversion
:link: ../how-to/convert-g1-dataset
:link-type: doc
:class-card: workflow-step-card

<span class="workflow-step-number">3</span>

Convert the HDF5 data to LeRobot and validate the fields required by the
policy training modality for Unitree G1.
:::

:::{grid-item-card} Policy Training
:link: ../how-to/fine-tune-gr00t
:link-type: doc
:class-card: workflow-step-card

<span class="workflow-step-number">4</span>

Train GR00T N1.7 on the validated dataset of Unitree G1 demonstrations.
:::

:::{grid-item-card} Policy Evaluation
:link: ../how-to/evaluate-checkpoint
:link-type: doc
:class-card: workflow-step-card

<span class="workflow-step-number">5</span>

Run the trained policy in the workflow's scene and collect episode metrics.
:::

:::::

## Workflow Artifacts

Commands use `outputs/g1/` under the repository root for one run. Choose new
recording and output paths when starting another run.

| Stage | Output | Next Input |
| --- | --- | --- |
| Recording in Isaac Lab-Arena with Pink IK and AGILE | `recording-arena.hdf5` | Direct conversion or Arena AutoData generation |
| Optional Arena generation | `generated-arena.hdf5` | Arena conversion |
| Conversion | `lerobot/` or `lerobot-generated/` | GR00T training dataset |
| Training | `training/checkpoint-20000/` | Checkpoint preflight and policy server |
| Evaluation | `evaluation/<timestamp>/` | Episode metrics and HTML report |

Keep the original manifested recordings. Merged HDF5 files cannot be used for
replay or AutoData with the current merger.

For the native AGILE alternative, the guides use `recording-native.hdf5` and
`generated-native.hdf5`. Native fixed-base data requires a custom training and
evaluation configuration because it lacks navigation and base-height commands.

:::{tip}
Start conversion with a [generated HDF5 download](../how-to/convert-g1-dataset.md#use-published-generated-data). LeRobot dataset and model downloads are included in the
[G1 policy training](../how-to/fine-tune-gr00t.md#use-a-published-dataset) and
[G1 policy evaluation](../how-to/evaluate-checkpoint.md#use-a-published-checkpoint) steps.
:::

## Hugging Face Dataset and Checkpoint

These simulation datasets and checkpoints are in private Hugging Face
repositories. Viewing or downloading them requires repository access and
acceptance of the NVIDIA access terms. The checkpoint below is separate from
the released checkpoint pinned in `runtime.toml`.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Dataset
:link: https://huggingface.co/datasets/nvidia/GR00T-E2E-Reference-Workflow-SIM-G1-N1.7-AppletoPlate-Dataset
:class-card: workflow-card hf-asset-card
:shadow: sm

<span class="hf-badge">DATASET ON HUGGING FACE</span>

`nvidia/GR00T-E2E-Reference-Workflow-SIM-G1-N1.7-AppletoPlate-Dataset`

Recorded HDF5, generated HDF5, and LeRobot demonstrations for the Unitree G1
static-apple simulation workflow.
:::

:::{grid-item-card} Checkpoint
:link: https://huggingface.co/nvidia/GR00T-E2E-Reference-Workflow-SIM-G1-N1.7-AppletoPlate-Model
:class-card: workflow-card hf-asset-card
:shadow: sm

<span class="hf-badge">MODEL ON HUGGING FACE</span>

`nvidia/GR00T-E2E-Reference-Workflow-SIM-G1-N1.7-AppletoPlate-Model`

Fine-tuned GR00T N1.7 checkpoint for the same Unitree G1 simulation task.
:::

::::

## Supporting Guidance

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Data Generation (Optional)
:link: ../how-to/generate-data-with-autodata
:link-type: doc
:class-card: workflow-card

Optional: turn a few AGILE recordings into many with Isaac AutoData. The
launcher selects the recorded action layout and manipulating hand automatically.
:::

::::

## Testing and Limitations

This workflow covers one Unitree G1 task on Linux x86_64. By default, recording
and checkpoint evaluation use the same success criteria.
Manually ending a recording does not change its success label. Pass
`--no-auto_success` to let the operator decide which episodes to accept. The
released checkpoint can test whether the components work together; its results
do not establish expected performance in this workflow's scene.

See [Reference](../reference/index.md) for supported configurations and limitations.

```{toctree}
:hidden:

Architecture and Data Requirements <../explanation/architecture>
Simulation Practice <../tutorials/getting-started>
Demonstration Recording <../how-to/record-demonstrations>
Data Generation (Optional) <../how-to/generate-data-with-autodata>
Dataset Conversion <../how-to/convert-g1-dataset>
Policy Training <../how-to/fine-tune-gr00t>
Policy Evaluation <../how-to/evaluate-checkpoint>
```

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

# Unitree H2 Plus Bimanual Pick, Handover & Place

Use a PICO 4 Ultra headset and its handheld XR controllers to record demonstrations
of a simulated Unitree H2 Plus picking up an apple, transferring it between
hands, and placing it on a plate.
It covers recording, conversion, training, and evaluation in Isaac Lab-Arena,
with optional Isaac AutoData generation.

<span class="workflow-status">AVAILABLE</span>

**Unitree H2 Plus · NVIDIA Isaac Lab-Arena · AutoData · GR00T N1.7**

H2 policy training is validated. Learned-policy evaluation is implemented but
not yet validated.

## Understand the Workflow

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Conceptual Overview
:link: ../explanation/concepts-overview
:link-type: doc
:class-card: workflow-card

Understand how recording, optional data generation, conversion, training, and
evaluation connect.
:::

:::{grid-item-card} Architecture and Data Requirements
:link: ../explanation/h2-architecture
:link-type: doc
:class-card: workflow-card

Review the Isaac Lab-Arena setup, robot-control software, dataset requirements,
and external asset ownership before running the stages.
:::

::::

## Before You Start

Complete [Installation](../installation.md) to prepare the workstation.
Recording requires a PICO 4 Ultra on the workstation's LAN and its paired
handheld XR controllers. Check the
[XR recording requirements](../installation.md#xr-recording-requirements)
and follow the [H2 CloudXR setup](../how-to/record-h2-demonstrations.md#start-cloudxr)
before recording.

Choose a {ref}`published stage input <h2-downloads>` to begin with conversion,
training, or evaluation. Published HDF5 conversion needs the GR00T runtime
but no headset, CloudXR session, or simulation assets. Each stage page lists
the setup required for its selected input.

## Follow the Workflow Path

To collect your own data, complete the six stages in order. After recording,
continue directly to conversion or use optional
[data generation](../how-to/generate-h2-demonstrations.md) first. Each card opens
a separate H2 stage page.

Recording uses Isaac Lab-Arena with Pink IK for the arms and AGILE for the
lower body (`h2_agile_pink`), operated with handheld XR controllers. The recording
guide selects this setup explicitly; the launcher's default native G1 mode does not
apply to H2.

Use the [Environment Variable Reference](../reference/environment-variables.md)
to restore asset paths, runtime settings, and the existing run in a new terminal.

:::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Asset Preparation
:link: ../how-to/prepare-h2
:link-type: doc
:class-card: workflow-step-card

<span class="workflow-step-number">1</span>

Prepare the specified Unitree H2 Plus simulation model, generate the URDF used
by Pink IK, and download the Unitree H2 AGILE policy.
:::

:::{grid-item-card} Simulation Validation
:link: ../how-to/validate-h2
:link-type: doc
:class-card: workflow-step-card

<span class="workflow-step-number">2</span>

Run the Unitree H2 simulation checks and practice driving the robot.
:::

:::{grid-item-card} Demonstration Recording
:link: ../how-to/record-h2-demonstrations
:link-type: doc
:class-card: workflow-step-card

<span class="workflow-step-number">3</span>

Record HDF5 demonstrations using the PICO 4 Ultra headset and handheld XR controllers.
The task scores each episode automatically; replay the accepted episodes.
:::

:::{grid-item-card} Dataset Conversion
:link: ../how-to/convert-h2-dataset
:link-type: doc
:class-card: workflow-step-card

<span class="workflow-step-number">4</span>

Convert the selected episodes to LeRobot and check that their state, action,
and video fields match the policy training requirements for Unitree H2.
:::

:::{grid-item-card} Policy Training
:link: ../how-to/fine-tune-h2
:link-type: doc
:class-card: workflow-step-card

<span class="workflow-step-number">5</span>

Train GR00T N1.7 on the validated dataset of Unitree H2 demonstrations.
:::

:::{grid-item-card} Policy Evaluation
:link: ../how-to/evaluate-h2-checkpoint
:link-type: doc
:class-card: workflow-step-card

<span class="workflow-step-number">6</span>

Run the learned 61-D joint policy, measure placement success, and review the
handover sequence separately.
:::

:::::

:::{tip}
Use a published artifact to start at a later stage:

- [Dataset conversion](../how-to/convert-h2-dataset.md#choose-the-training-hdf5)
  accepts downloaded recorded or generated HDF5 demonstrations.
- [Policy training](../how-to/fine-tune-h2.md#use-a-published-dataset)
  accepts the provided LeRobot dataset.
- [Policy evaluation](../how-to/evaluate-h2-checkpoint.md#use-a-published-checkpoint)
  accepts the provided trained model.
:::

## Workflow Artifacts

The H2 guides keep each run in `outputs/h2/<timestamp>/` under the repository
root. `H2_RUN_DIR` identifies that directory. Keep its paths when continuing
the run, and choose a new directory for another run.

| Stage | Output in the Run Directory | Next Input |
| --- | --- | --- |
| Recording and state replay | `source.hdf5` | Direct conversion or optional AutoData generation |
| Optional generation | `generated.hdf5` and `generation.json` | Replay and conversion of the generated HDF5 |
| Conversion and loader validation | `lerobot/` and `gr00t-loader.json` | Validated LeRobot dataset for GR00T training |
| Training | `training/checkpoint-<step>/` | Checkpoint preflight and policy server |
| Evaluation | `evaluation/<timestamp>/` | Episode metrics, HTML report, and camera videos |

For conversion, `H2_TRAINING_HDF5` selects the recorded or generated file and
`H2_TRAINING_DEMOS` holds its accepted episode count. The
[download instructions](../how-to/convert-h2-dataset.md#choose-the-training-hdf5)
set these values for published inputs.

## Hugging Face Dataset and Checkpoint

The checkpoint and dataset revisions are pinned in `runtime.toml`.
The evaluation guide accepts your own training output or the published inference
checkpoint.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Dataset
:link: https://huggingface.co/datasets/nvidia/GR00T-E2E-Reference-Workflow-SIM-H2-N1.7-AppleHandoverPlate-Dataset
:class-card: workflow-card hf-asset-card
:shadow: sm

<span class="hf-badge">DATASET ON HUGGING FACE</span>

`nvidia/GR00T-E2E-Reference-Workflow-SIM-H2-N1.7-AppleHandoverPlate-Dataset`

Recorded HDF5, generated HDF5, and LeRobot demonstrations for the Unitree H2 Plus
bimanual pick, handover, and place simulation workflow.
:::

:::{grid-item-card} Checkpoint
:link: https://huggingface.co/nvidia/GR00T-E2E-Reference-Workflow-SIM-H2-N1.7-AppleHandoverPlate-Model
:class-card: workflow-card hf-asset-card
:shadow: sm

<span class="hf-badge">MODEL ON HUGGING FACE</span>

`nvidia/GR00T-E2E-Reference-Workflow-SIM-H2-N1.7-AppleHandoverPlate-Model`

Fine-tuned GR00T N1.7 checkpoint for the same Unitree H2 simulation task.
:::

::::

## Supporting Guidance

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Data Generation (Optional)
:link: ../how-to/generate-h2-demonstrations
:link-type: doc
:class-card: workflow-card

Expand qualifying teleoperation demonstrations with Isaac AutoData when more data is
needed. Retain only attempts that pass the shared task predicate.
:::

::::

```{toctree}
:hidden:

Architecture and Data Requirements <../explanation/h2-architecture>
Asset Preparation <../how-to/prepare-h2>
Simulation Validation <../how-to/validate-h2>
Demonstration Recording <../how-to/record-h2-demonstrations>
Data Generation (Optional) <../how-to/generate-h2-demonstrations>
Dataset Conversion <../how-to/convert-h2-dataset>
Policy Training <../how-to/fine-tune-h2>
Policy Evaluation <../how-to/evaluate-h2-checkpoint>
```

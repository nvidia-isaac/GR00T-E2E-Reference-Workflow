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

# Unitree G1 Real-Robot Static Apple Pick-and-Place

This workflow path covers a Unitree G1 humanoid with Dex3-1 hands, a head-mounted
Intel RealSense camera, and a Jetson AGX Thor connected externally or through
the Unitree Thor backpack. Thor runs Isaac ROS; AGILE whole-body control
stabilizes the lower body while an operator or GR00T policy commands the upper
body for an apple-to-plate manipulation task.

<span class="workflow-status">AVAILABLE</span>

**Unitree G1 · Isaac ROS · Jetson AGX Thor · GR00T N1.7**

:::{warning}
Read [Unitree G1 Introduction and Safety](../real-robot-workflow/g1-introduction-and-safety.md)
in full before powering on the robot. Follow Unitree's operating documentation
and your organization's robot-safety procedures. Use at least two people: one
to operate and one to monitor safety.
:::

## Understand the Workflow

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Conceptual Overview
:link: ../explanation/concepts-overview
:link-type: doc
:class-card: workflow-card

Understand each stage of the reference workflow and how the physical
data path differs from the simulation path.
:::

:::{grid-item-card} Architecture and Data Requirements
:link: ../explanation/architecture
:link-type: doc
:class-card: workflow-card

Review where physical data collection fits in the pipeline and which data
formats and task settings each dataset and checkpoint requires before running
the stages.
:::

::::

## Follow the Workflow Path

Complete the stages in order. Each page states where it runs and what it
produces. Recording uses the Isaac ROS MCAP recorder on Thor. The native and
Arena HDF5 recording modes apply to simulation paths.

Use the [Environment Variable Reference](../reference/environment-variables.md#physical-g1)
when restoring the workspace and service environment in a new terminal.

:::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Introduction and Safety
:link: ../real-robot-workflow/g1-introduction-and-safety
:link-type: doc
:class-card: workflow-step-card

<span class="workflow-step-number">1</span>

Review the safety checkpoints for teleoperation, recording, and deployment.
Prepare the workspace and operating roles before connecting the robot.
:::

:::{grid-item-card} Isaac ROS Setup on Thor
:link: ../real-robot-workflow/isaac-ros-setup
:link-type: doc
:class-card: workflow-step-card

<span class="workflow-step-number">2</span>

Install the `isaac-ros` CLI, ROS 2 workspace, source repositories, container
environment, and RealSense support on the computer connected to the Unitree G1.
:::

:::{grid-item-card} Teleoperation
:link: ../real-robot-workflow/real-teleop
:link-type: doc
:class-card: workflow-step-card

<span class="workflow-step-number">3</span>

Verify the XR headset and CloudXR session in simulation, then run whole-body XR
teleoperation on real hardware through the ROS safety controller.
:::

:::{grid-item-card} Demonstration Recording
:link: ../real-robot-workflow/real-record
:link-type: doc
:class-card: workflow-step-card

<span class="workflow-step-number">4</span>

Capture teleoperation demonstrations in MCAP format, keeping only clean runs
that follow the recording guidance for GR00T training.
:::

:::{grid-item-card} Dataset Conversion
:link: ../real-robot-workflow/real-data-export
:link-type: doc
:class-card: workflow-step-card

<span class="workflow-step-number">5</span>

Install the converter, merge one or more recording sessions into a single
LeRobot dataset, and verify its output paths and metadata.
:::

:::{grid-item-card} Policy Training
:link: ../real-robot-workflow/real-fine-tuning-and-leapp
:link-type: doc
:class-card: workflow-step-card

<span class="workflow-step-number">6</span>

Validate the dataset and modality configuration, fine-tune GR00T N1.7 on the
real-world dataset, and locate the checkpoint for export.
:::

:::{grid-item-card} LEAPP Export
:link: ../real-robot-workflow/real-leapp-export
:link-type: doc
:class-card: workflow-step-card

<span class="workflow-step-number">7</span>

Package the fine-tuned policy into a deployable LEAPP bundle that runs from
C++ with its sensor inputs and command outputs already connected.
:::

:::{grid-item-card} Policy Deployment and Evaluation
:link: ../real-robot-workflow/real-deployment
:link-type: doc
:class-card: workflow-step-card

<span class="workflow-step-number">8</span>

Deploy the bundled apple-to-plate policy to Thor, replace it with your own
export, and measure real-world success rate.
:::

:::::

## Hugging Face Dataset and Checkpoint

Skip collecting or fine-tuning your own demonstrations with the published
physical-robot artifacts already linked from the conversion and fine-tuning
pages. These Hub pages may require accepting NVIDIA access terms.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Dataset
:link: https://huggingface.co/datasets/nvidia/GR00T-N1.7-AppleToPlate
:class-card: workflow-card hf-asset-card
:shadow: sm

<span class="hf-badge">DATASET ON HUGGING FACE</span>

`nvidia/GR00T-N1.7-AppleToPlate`

LeRobot demonstrations for the Unitree G1 real-robot static-apple workflow.
:::

:::{grid-item-card} Checkpoint
:link: https://huggingface.co/nvidia/GR00T-N1.7-ApplePnP-V1
:class-card: workflow-card hf-asset-card
:shadow: sm

<span class="hf-badge">MODEL ON HUGGING FACE</span>

`nvidia/GR00T-N1.7-ApplePnP-V1`

Fine-tuned GR00T N1.7 checkpoint for the same physical Unitree G1 task.
:::

::::

## Supporting Guidance

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Troubleshooting
:link: ../reference/troubleshooting
:link-type: doc
:class-card: workflow-card

Diagnose RealSense, CloudXR, ROS domain, and Thor throttling issues that
block teleoperation, recording, or deployment.
:::

::::

## Testing and Limitations

The repository's automated tests do not run on the physical robot. The
[setup requirements](../real-robot-workflow/isaac-ros-setup.md#prerequisites-and-version-boundary)
specify JetPack 7.1 / Isaac ROS 4.5 and explain why the exact v0.1 installation
cannot be fully reproduced. The historical 68% result also lacks
the original [scoring criteria](../real-robot-workflow/real-deployment.md#structured-evaluation-approach)
needed for direct comparison.

```{toctree}
:hidden:

Introduction and Safety <../real-robot-workflow/g1-introduction-and-safety>
Isaac ROS Setup on Thor <../real-robot-workflow/isaac-ros-setup>
Teleoperation <../real-robot-workflow/real-teleop>
Demonstration Recording <../real-robot-workflow/real-record>
Dataset Conversion <../real-robot-workflow/real-data-export>
Policy Training <../real-robot-workflow/real-fine-tuning-and-leapp>
LEAPP Export <../real-robot-workflow/real-leapp-export>
Policy Deployment and Evaluation <../real-robot-workflow/real-deployment>
```

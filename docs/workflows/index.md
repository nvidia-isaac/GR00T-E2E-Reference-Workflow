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

# Workflow Paths

Workflow paths implement the reference workflow from a prepared
system to a measurable robotics result. Choose a path below and follow its
setup instructions. Each workflow includes its ordered steps, architecture,
testing status, and limitations.

## Recording Mode by Workflow Path

Use the full command in the recording guide for your path. The simulation
launcher defaults to native G1 with a fixed base, so its defaults do not select
the recording mode needed by every path.

| Workflow Path | Recording Mode | Recording Guide |
| --- | --- | --- |
| Simulated G1 | Arena with `g1_wbc_agile_pink`; native AGILE is an alternative | [recording demonstrations with G1](../how-to/record-demonstrations.md) |
| Simulated H2 | Arena with `h2_agile_pink` and handheld XR controllers | [recording demonstrations with H2](../how-to/record-h2-demonstrations.md) |
| Physical G1 | Isaac ROS MCAP recorder on Thor | [recording demonstrations on the physical G1](../real-robot-workflow/real-record.md) |

## Choose a Workflow Path

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Unitree G1 Static Apple Pick & Place
:link: g1-static-apple
:link-type: doc
:class-card: workflow-card workflow-featured-card
:shadow: sm

<span class="workflow-status">AVAILABLE</span>

**Unitree G1 · NVIDIA Isaac Lab · GR00T N1.7**

Move an apple to a plate through simulation, demonstration collection,
dataset conversion, policy fine-tuning, and closed-loop evaluation.
:::

:::{grid-item-card} Unitree G1 Real-Robot Static Apple Pick-and-Place
:link: g1-real-static-apple
:link-type: doc
:class-card: workflow-card workflow-featured-card
:shadow: sm

<span class="workflow-status">AVAILABLE</span>

**Unitree G1 · Isaac ROS · Jetson AGX Thor · GR00T N1.7**

Collect physical demonstrations, fine-tune and export a policy, then deploy and
evaluate it on the Unitree G1.
:::

:::{grid-item-card} Unitree H2 Plus Bimanual Pick, Handover & Place
:link: h2-pick-handover-place
:link-type: doc
:class-card: workflow-card workflow-featured-card
:shadow: sm

<span class="workflow-status">AVAILABLE</span>

**Unitree H2 Plus · NVIDIA Isaac Lab-Arena · AutoData · GR00T N1.7**

Follow the Unitree H2 workflow in Isaac Lab-Arena. Teleoperation recording with
a PICO 4 Ultra headset and handheld XR controllers, limited AutoData
amplification, and policy training have been validated. Learned-policy evaluation
is implemented but not yet validated.
:::

::::

```{toctree}
:hidden:

g1-static-apple
g1-real-static-apple
h2-pick-handover-place
```

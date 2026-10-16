---
html_theme.sidebar_secondary.remove: true
---
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


# GR00T E2E Reference Workflow

:::::{div} workflow-hero
::::{grid} 1 1 12 12
:gutter: 4
:class-container: workflow-hero-grid

:::{grid-item}
:columns: 12 12 7 7
<p class="workflow-eyebrow">REFERENCE WORKFLOW</p>

<p class="workflow-hero-title">Build robot-learning systems end to end</p>

See how NVIDIA robotics and AI products work together in an end-to-end system.
Follow complete workflow paths that connect data collection to trained policies
in simulation and on physical robots.

```{button-ref} workflows/index
:color: primary
:class: workflow-primary-button

Explore workflow paths
```
:::

:::{grid-item}
:columns: 12 12 5 5
```{div} workflow-pipeline-mark
<span>SIM</span><b>→</b><span>DATA</span><b>→</b><span>POLICY</span><b>→</b><span>EVAL</span>
```
:::

::::
:::::

## Featured Workflow Paths

::::{grid} 1 1 12 12
:gutter: 3

:::{grid-item-card} Unitree G1 Static Apple Pick & Place
:columns: 12 12 6 6
:link: workflows/g1-static-apple
:link-type: doc
:class-card: workflow-card workflow-featured-card
:shadow: md

<span class="workflow-status">AVAILABLE</span>

**Unitree G1 · NVIDIA Isaac Lab · GR00T N1.7**

Record demonstrations, convert the data, fine-tune a policy, and evaluate it
in the same scene.

Open workflow path →
:::

:::{grid-item-card} Unitree G1 Real-Robot Static Apple Pick-and-Place
:columns: 12 12 6 6
:link: workflows/g1-real-static-apple
:link-type: doc
:class-card: workflow-card workflow-featured-card
:shadow: md

<span class="workflow-status">AVAILABLE</span>

**Unitree G1 · Isaac ROS · Jetson AGX Thor · GR00T N1.7**

Collect physical demonstrations, fine-tune and export a policy, then deploy and
evaluate it on the Unitree G1.

Open workflow path →
:::

:::{grid-item-card} Unitree H2 Plus Bimanual Pick, Handover & Place
:columns: 12 12 6 6
:link: workflows/h2-pick-handover-place
:link-type: doc
:class-card: workflow-card workflow-featured-card
:shadow: md

<span class="workflow-status">AVAILABLE</span>

**Unitree H2 Plus · NVIDIA Isaac Lab-Arena · AutoData · GR00T N1.7**

Record bimanual demonstrations, optionally expand them with AutoData, then
convert the data, train a policy, and evaluate it in simulation.
Policy training is validated. Learned-policy evaluation is implemented but not
yet validated.

Open workflow path →
:::

:::{grid-item-card} Add the Next Workflow Path
:columns: 12
:link: developer/build-workflow
:link-type: doc
:class-card: workflow-card workflow-expansion-card

<span class="workflow-status workflow-status-muted">WORKFLOW AUTHORS</span>

Use the existing workflow requirements to add another environment, robot configuration, or
policy path and register its documentation in the site navigation.

Build a Workflow Path →
:::

::::

```{toctree}
:hidden:

Installation <installation>
Conceptual Overview <explanation/concepts-overview>
Workflow Paths <workflows/index>
Developer Guide <reference/development>
Reference <reference/index>
```

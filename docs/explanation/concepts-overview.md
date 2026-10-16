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

# Conceptual Overview

GR00T End-to-End Reference Workflow connects the stages needed to teach a humanoid
robot a manipulation task. Each path defines its scope and validation
requirements so teams can inspect and validate individual stages.

The reference workflow provides simulation and physical-robot paths. Teams can
follow a complete path or reuse individual stages for another task, sensor
layout, or robot embodiment.

(what-end-to-end-means)=
## What E2E Means

An end-to-end robot-learning workflow includes:

1. **Environment setup** — define the robot, task objects, sensors, workspace,
   observations, and actions.
2. **Teleoperation and data collection** — record successful demonstrations in
   simulation or on the physical robot.
3. **Optional amplification** — in simulation, AutoData can expand a few
   accepted demonstrations into many before conversion.
4. **Data conversion** — map the recording into a LeRobot dataset with the
   state, image, action, and language fields expected by training.
5. **Post-training** — adapt a pretrained GR00T vision-language-action policy
   to the task and embodiment.
6. **Evaluation** — run the frozen policy in closed loop and measure task
   completion.
7. **Deployment** — export the physical-robot policy pipeline and connect it to
   live sensors and robot-control software.

Connecting the tools is not sufficient. A weak dataset, incorrect action
mapping, unstable robot-control software, or unrepeatable workspace can invalidate the
result even when every process starts successfully.

## Why Use Apple Pick and Place

The task is simple enough to debug stage by stage but still exercises:

- visual identification of the apple and plate;
- coordinated reaching, grasping, transfer, and release;
- humanoid balance while the upper body manipulates an object;
- observation and action mapping across collection and training;
- closed-loop policy execution and measurable success criteria.

```{figure} ../images/unitree-top-down.jpg
:alt: Unitree G1 standing at the apple-to-plate workspace
:width: 80%

The physical apple-to-plate workspace from behind the Unitree G1.
```

The simulation and physical workflow paths implement the same task intent, but they
do not imply identical scenes, observations, dynamics, or policy performance.
Each workflow must state its configuration, what has been tested, and its limitations.

## Two Data-Acquisition Paths

Demonstrations can originate in simulation or on the physical G1.

### Simulation

NVIDIA Isaac Lab-Arena composes the task, embodiment, and scene for recording demonstrations with G1,
all H2 simulation, and policy evaluation. Demonstrations with G1 can also be
recorded in native Isaac Lab. NVIDIA Isaac Capture supplies operator commands, and
the workflow records demonstrations in HDF5. Isaac AutoData can optionally
amplify accepted demonstrations before conversion.

```{figure} ../images/arena-stack-ari.png
:alt: Isaac Lab-Arena stack used to compose simulated tasks and embodiments

The upstream Isaac Lab-Arena stack. This workflow uses its task, scene, and
embodiment composition; the illustration also shows broader Arena capabilities.
```

The repository owns the simulation launchers, task integration, HDF5 mapping,
and validation requirements. Follow the [G1](../workflows/g1-static-apple.md) or
[H2](../workflows/h2-pick-handover-place.md) simulation path for executable steps.

### Physical Robot

The physical path uses NVIDIA Isaac Capture and ROS 2 to capture live camera,
robot-state, and command streams. ROS bags use MCAP as the recording container.
The physical workspace, camera placement, network, robot-control software state, and safety
procedure must stay consistent with the conditions used to collect the data.

Simulation and physical recording can be completed independently. They converge
on LeRobot as the training format, but a dataset remains specific to the
embodiment, sensor layout, action representation, and task conditions that
produced it. AutoData is a simulation-only optional branch; physical MCAP
recordings convert to LeRobot directly.

## Teleoperation and Whole-Body Control

Handheld XR controllers are input devices. Their tracked poses, trigger values,
and thumbstick inputs provide commands for teleoperation. Pink IK and the AGILE
whole-body controller are robot-control software that turns motion targets
into joint commands.

A humanoid must manipulate objects while remaining stable. The workflow
separates those responsibilities:

- the operator or learned policy supplies upper-body manipulation commands;
- inverse kinematics or joint-space control realizes the requested arm and hand
  motion;
- AGILE supplies standing, balance, or locomotion behavior where the selected
  robot-control configuration uses it;
- an action adapter maps those outputs into the environment or robot command
  space.

This separation lets the operator focus on reaching and grasping without
manually commanding every lower-body joint. The exact composition is
runtime-specific: the built-in Isaac Lab task, Arena embodiment, and physical
Isaac ROS stack use different robot-control implementations and model files.

Human and robot limbs also differ in proportions, limits, and reachable poses.
Teleoperation therefore requires retargeting: human device motion is mapped to
commands the robot can execute within its kinematic constraints.

```{figure} ../images/zYqxhAGHV26raIAh9p1BS.png
:alt: Operator collecting a robot demonstration through teleoperation

Teleoperation records operator demonstrations for imitation learning.
```

```{figure} ../images/isaac-teleop-hero.jpg
:alt: NVIDIA Isaac Capture in use

NVIDIA Isaac Capture provides device interfaces and retargeting for demonstration
collection.
```

```{figure} ../images/real-robot-teleop-example.gif
:alt: XR teleoperation of the physical Unitree G1

The physical workflow path records live robot and sensor streams during
teleoperation.
```

## Amplify Demonstrations With AutoData

Recording demonstrations by hand is the slowest part of the workflow. [Isaac
AutoData](../how-to/generate-data-with-autodata.md) is the optional branch: it
turns a few recorded episodes into many. For each attempt it randomizes the
scene, transforms a source episode's end-effector trajectory onto the new object
pose, replays it, and keeps the attempts that satisfy the task's success
condition.

Generation needs three things the recording environment does not expose as
Isaac Lab termination and observation terms: a success condition to score each
attempt, a randomized apple pose to transform trajectories onto, and a boolean
signal marking where one subtask ends. The AutoData task variant adds that
success termination and subtask boundary so generated attempts can be scored and
annotated; those terms score generation, not the original recordings.

The data-generation steps depend on how the simulation demonstrations were recorded:

- demonstrations recorded with G1 in native Isaac Lab (28/32-D actions) require an annotation stage that finds the
  pickup boundary;
- AutoData-compatible Isaac Lab-Arena recordings made with Pink IK already carry
  `datagen_info` and must not be annotated.

Joint-control recordings are not AutoData inputs. Both generation methods convert
articulation targets to LeRobot. Demonstrations recorded with G1 using Pink IK and AGILE in
Isaac Lab-Arena, or with AGILE in native Isaac Lab, provide navigation and
base-height commands for the supplied G1 modality. Demonstrations recorded with H2
in Isaac Lab-Arena provide navigation for its H2 modality. Demonstrations recorded
with a fixed-base G1 omit those
commands and need a custom modality. Raw actions never cross between
environments. Generated HDF5 must still be converted before training.

Failed perturbations remain discarded. A generated file with zero accepted
episodes means the source demonstrations, the boundary signal, or the success
condition did not admit a usable attempt.

## Why Convert to LeRobot

Collection and training have different storage requirements:

| Source | Capture format | Training format |
| --- | --- | --- |
| Isaac Lab simulation | HDF5 | LeRobot |
| AutoData generation | generated HDF5 | LeRobot |
| Physical G1 through ROS 2 | MCAP | LeRobot |

HDF5 and MCAP prioritize reliable capture and replay of runtime data. LeRobot
organizes episodes for machine-learning workflows around fields such as:

- `observation.images`;
- `observation.state`;
- `action`;
- task language.

Conversion is a semantic operation, not just a file-format change. Joint order,
camera identity, units, action representation, sampling rate, and episode
boundaries must match the modality configuration used for post-training.

## Post-Training a GR00T Policy

GR00T is a vision-language-action model. At inference time it consumes:

- RGB observations from the configured cameras;
- a language instruction, such as `move the apple to the plate`;
- the current robot state.

It predicts a chunk of future robot actions rather than one isolated action.
The action horizon, observation groups, action groups, and embodiment tag must
remain consistent across conversion, post-training, and inference.

Adapting GR00T to a new robot configuration requires specifying:

- controlled joints and valid action ranges;
- robot-state ordering and representation;
- camera count, names, and image shape;
- absolute or relative action encoding;
- action horizon and execution frequency.

Training produces a checkpoint. It does not establish that the policy can
complete the task.

```{figure} ../images/gr00t.png
:alt: GR00T vision-language-action model architecture

The upstream GR00T model architecture maps visual observations, language, and
robot state to action chunks. Each workflow path supplies its own modality
configuration and runtime integration.
```

## Inference, Evaluation, and Deployment

**Inference** runs a frozen checkpoint to produce actions from live
observations. **Evaluation** combines inference with repeated trials and
objective task measurements.

Simulation provides repeatable resets, parallel trials, full scene state, and
inexpensive failure. When a physical policy has a compatible simulation
embodiment and task, simulation should be its first controlled evaluation
environment. A low simulation success rate is evidence that the policy,
dataset, mapping, or task configuration needs more work; successful
server-simulator communication alone does not make that policy ready for
hardware. The physical-data workflow path can also be completed independently when
no simulation has a matching robot configuration and task.

Physical deployment adds a real-time pipeline:

1. preprocess camera and robot-state observations;
2. run GR00T inference;
3. decode and validate the predicted action;
4. pass commands through the robot-control software and ROS safety controllers.

LEAPP packages the preprocessing, inference, and postprocessing graph into
deployment artifacts for the physical runtime. This export is specific to the
physical path; simulation evaluation can run a compatible training checkpoint
directly through the Arena client/server integration.

Real-robot evaluation is the hardware validation point. Success requires both
task completion and stable operation within the tested workspace and safety
conditions.

## Diagnose the Stage, Not the Symptom

Failures should be traced to the stage that owns them:

- scene or sensor mismatch belongs to environment setup;
- poor or inconsistent motion belongs to data collection;
- empty or unusable amplified datasets belong to AutoData (wrong recording method,
  missing annotation, missing success termination, or a source that never
  satisfied the task);
- missing or reordered fields belong to conversion;
- unstable loss or poor action prediction belongs to post-training;
- action-shape, horizon, or server-client mismatches belong to inference
  integration;
- unsafe or unstable physical behavior requires an immediate stop and review of
  the hardware, robot-control software, and policy interfaces.

The [architecture](architecture.md) identifies the concrete runtime boundaries.
The workflow path pages provide the corresponding executable instructions.

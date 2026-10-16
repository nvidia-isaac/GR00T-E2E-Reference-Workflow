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

# Installation

Choose a [workflow path](workflows/index.md) before installing its environment.
This page prepares the workstation for simulated Unitree G1 and Unitree H2.

For physical Unitree G1, follow the [physical workflow path](workflows/g1-real-static-apple.md).
It provides separate Isaac ROS setup on Jetson AGX Thor and workstation setup
for training and export. The simulation environment below is not required for
that path.

## System Requirements

The simulation baseline follows the
[Isaac Sim 6.0 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html).
The repository's checked runtime supports Linux x86_64.

| Component | Minimum or required baseline | Recommended for this workflow |
| --- | --- | --- |
| OS and architecture | Ubuntu 22.04 or 24.04, x86_64 | Ubuntu 24.04, x86_64 |
| CPU | 4 cores | 8 or more cores |
| System RAM | 32 GB for simulation | 64 GB for simulation; 128 GB or more for fine-tuning |
| Simulation and recording GPU | NVIDIA RTX GPU with RT cores and at least 16 GB VRAM; upstream lists NVIDIA RTX 4080 | NVIDIA RTX 3090 is validated for the Unitree G1 stages except training; NVIDIA RTX 6000 Ada provides more memory for combined simulation and policy workloads |
| Fine-tuning GPU | 48 GB VRAM for this repository's tested training recipes | NVIDIA RTX 6000 Ada (48 GB); the Unitree H2 recipe is documented on NVIDIA RTX 5880 Ada (48 GB) |
| Linux NVIDIA driver | Use 580.95.05 as the Isaac Sim 6.0 tested baseline | A production driver qualified for your GPU and Isaac Sim version |
| CUDA runtime | CUDA 12.8 for the repository's pinned PyTorch build | Keep the versions installed by `uv.lock` |
| Storage | 50 GB SSD for the upstream simulator baseline | At least 500 GB SSD for environments, assets, datasets, and checkpoints |
| Python | 3.12, installed by `uv` | Use the repository-local environment |

The driver number is an upstream-tested baseline, not a measured lower support
limit for this application. A100 and H100 GPUs lack RT cores and cannot run the
simulator, even when suitable for a separate training server. The 16 GB simulator
baseline does not guarantee enough memory to run a policy server concurrently.
See the [policy training configuration for Unitree G1](how-to/fine-tune-gr00t.md#training-inputs)
and [policy training configuration for Unitree H2](how-to/fine-tune-h2.md) for recipe-specific settings.

The locked environment supplies CUDA user-space packages; a separate system CUDA
toolkit installation is not required for this repository's wheel-based runtime.
The host NVIDIA driver is still required. `nvidia-smi` reports the driver's CUDA
capability, which can be newer than the runtime used by PyTorch.

Inspect the machine before downloading. Check the OS and architecture:

```bash
cat /etc/os-release
```

```bash
uname -m
```

Check the CPU, memory, and available disk space:

```bash
lscpu
```

```bash
free -h
```

```bash
df -h .
```

Check the GPU and driver:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

Network access is required to the source project, Python package indexes, GitHub,
simulator assets, and Hugging Face. Training requires access to the gated model
repositories described in [Model Access](how-to/fine-tune-gr00t.md#model-access).
For XR recording, also prepare the [headset and network](#xr-recording-requirements)
before starting the recording stage.

## XR Recording Requirements

XR recording with NVIDIA Isaac Capture requires a Linux x86_64 workstation and a supported
headset on the workstation's LAN. Meta Quest 3 and PICO 4 Ultra can use the
browser client through NVIDIA CloudXR. If a firewall is enabled, allow TCP
ports 49100 and 48322 and UDP port 47998.

For Unitree G1, use XR hand tracking, handheld XR controllers, or Manus/Vive. Keyboard
and SpaceMouse control are not supported for recording the static-apple task.
Unitree H2 uses PICO 4 Ultra handheld XR controllers and requires Isaac Lab-Arena;
follow the [Unitree H2 Plus workflow](workflows/h2-pick-handover-place.md).

After installing the simulation environment below, follow
[Unitree G1 XR setup](how-to/record-demonstrations.md#xr-setup) to start CloudXR
and connect the headset for a recording session.

## Host Tools

Install [Git](https://git-scm.com/downloads) and
[uv](https://docs.astral.sh/uv/getting-started/installation/)
using their installation instructions before cloning the repository.


## Clone the Repository

These instructions select @@DOCS_REF@@. The documentation source commit is @@DOCS_COMMIT@@.


Choose a workspace with enough free space; the example uses
`~/robot-workflows`.
Run each command below in order in the same terminal.

@@DOCS_CHECKOUT@@

Keep the printed commit ID with your run results. To match this documentation
exactly, check out its displayed source commit with
`git checkout --detach <full-commit-id>`.


This repository has no Git submodules to initialize. Model weights and datasets
are downloaded separately by their workflow stages.

For simulation-policy training, place the external Isaac-GR00T checkout alongside this repository,
for example `~/robot-workflows/Isaac-GR00T`. It has its own environment. Follow
{ref}`prepare-gr00t-checkout` to clone and select the exact revision recorded in
`runtime.toml`; do not install GR00T into the simulator environment.

## Create and Check the Simulation Environment

From the `e2e-reference-app` directory:

```bash
uv sync --frozen
```

This installs the simulator, host tools, and documentation dependencies into
`.venv`. `uv.lock` specifies package versions; `runtime.toml` records the tested
software versions and model revisions. No external Isaac Lab checkout or container is needed.
Isaac Lab-Arena launchers fetch their pinned source checkout when needed.
Optional AutoData provisioning is covered by its workflow stage.

After accepting the NVIDIA Isaac Sim license terms, run the canonical runtime
preflight on the GPU workstation:

```bash
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  uv run --no-sync ./tools/validate_repository.sh runtime-preflight
```

Require a JSON result with `"status": "ok"`. The check verifies runtime versions,
CUDA availability, and the Pink IK solver dependencies. An `"arena": "skipped"`
result means no external Arena checkout was supplied; the stage launcher checks
that checkout when it is fetched. This preflight does not run a simulation or
qualify a physical robot.

## Choose a Workflow Path

- [Unitree G1 Static Apple Pick & Place](workflows/g1-static-apple.md): simulate, record,
  train, and evaluate a Unitree G1 policy.
- [Unitree G1 Real-Robot Static Apple Pick-and-Place](workflows/g1-real-static-apple.md):
  set up the physical robot and its Isaac ROS environment.
- [Unitree H2 Plus Bimanual Pick, Handover & Place](workflows/h2-pick-handover-place.md):
  prepare Unitree H2 assets, record demonstrations, and follow the Unitree H2 stages.

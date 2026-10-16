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

A reference workflow for developing robot-learning systems from demonstration
collection through datasets, policy training, and evaluation. Simulation uses
NVIDIA Isaac Lab; physical Unitree G1 operation uses NVIDIA Isaac ROS.

The workflow paths use Unitree G1 and Unitree H2 Plus robots to move an apple to
a plate. NVIDIA Isaac Capture supplies XR input. Simulation recording exports
HDF5; recording demonstrations on the physical G1 exports MCAP. Each path converts
its recordings to LeRobot with the robot-specific configuration required for training.

Read the [documentation](https://nvidia-isaac.github.io/GR00T-E2E-Reference-Workflow/).


## Choose a Workflow Path

| Workflow path | Start here |
| --- | --- |
| Simulated Unitree G1: apple pick-and-place | [G1 simulation](docs/workflows/g1-static-apple.md) |
| Physical Unitree G1: apple pick-and-place | [G1 physical robot](docs/workflows/g1-real-static-apple.md) |
| Simulated Unitree H2 Plus: bimanual pick, handover, and place | [H2 simulation](docs/workflows/h2-pick-handover-place.md) |

Physical G1 uses the Thor and workstation setup in its own guide. It does not
require the simulation environment or quick start below.

## Requirements

| Workflow stage | Hardware and access |
| --- | --- |
| Simulation, keyboard locomotion, XR recording, and evaluation | Ubuntu 22.04 or 24.04 on x86_64; 4 CPU cores, 32 GB RAM, 50 GB SSD storage, and an NVIDIA RTX GPU with at least 16 GB VRAM |
| Simulation XR teleoperation | The simulation workstation above plus a supported headset such as Meta Quest 2/3/3S or PICO 4 Ultra; Unitree H2 currently requires handheld XR controllers |
| Simulation-policy fine-tuning | A standalone Isaac-GR00T environment, a converted LeRobot dataset, access to GR00T and Cosmos, and a training GPU; the reference 20,000-step G1 run is validated on one NVIDIA RTX 6000 Ada |
| Physical G1 collection and deployment | Unitree G1, Jetson AGX Thor, camera and XR equipment; a separate workstation for training and export. Follow the [physical G1 requirements](docs/real-robot-workflow/g1-introduction-and-safety.md). |

The simulated Unitree G1 stages, except fine-tuning, have been validated on an NVIDIA RTX 3090.
Unitree H2 simulation, teleoperation recording with a PICO 4 Ultra headset and
handheld XR controllers, limited AutoData amplification, and policy training have been validated.
H2 learned-policy evaluation is implemented but not yet validated. Start with the
[Unitree H2 Plus Bimanual Pick, Handover & Place workflow path](docs/workflows/h2-pick-handover-place.md).
The simulation-policy fine-tuning recipes need 48 GB of VRAM. Physical G1 uses
the separate `gr00t-leapp-export` environment and hardware requirements in its
[fine-tuning guide](docs/real-robot-workflow/real-fine-tuning-and-leapp.md).

The Arena and Isaac AutoData paths additionally use their own source checkouts,
pinned by revision in `runtime.toml` rather than locked in `uv.lock`. The Arena
launchers fetch a managed checkout automatically; the AutoData launcher can
provision its checkout and isolated interpreter.

Install [uv](https://docs.astral.sh/uv/) and use a compatible NVIDIA driver.
Internet access is required for Python packages, simulator assets, AGILE, and
Hugging Face model downloads. See the NVIDIA
[Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html)
and the repository's [GR00T fine-tuning guide](docs/how-to/fine-tune-gr00t.md)
for the complete upstream requirements.

## Simulation Quick Start

Complete [Installation](docs/installation.md), then run from the repository root:

```bash
uv sync --frozen
```

Launch the simulator after accepting the NVIDIA Isaac Sim license terms:

```bash
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  uv run scripts/locomotion_with_agile.py --device cuda:0 --viz kit
```

Click the simulator viewport to give it keyboard focus. Use **Up/Down** to walk,
**Left/Right** to turn, and **A/D** to strafe. Releasing the keys stops the Unitree G1 and
returns it to balance. See [Getting Started](docs/tutorials/getting-started.md) for
additional launch options.

## E2E Workflow

The G1 simulation guide records in Isaac Lab-Arena with Pink IK and AGILE,
then follows these stages.
Native AGILE recording is also compatible with the supplied training configuration.

1. [Record demonstrations](docs/how-to/record-demonstrations.md) with a
   supported XR device.
2. Optionally [amplify those demonstrations](docs/how-to/generate-data-with-autodata.md)
   with Isaac AutoData, which turns a few recorded episodes into many.
3. [Convert the dataset](docs/how-to/convert-g1-dataset.md) from HDF5 to the
   GR00T LeRobot format.
4. [Train GR00T N1.7](docs/how-to/fine-tune-gr00t.md) with the matching modality
   configuration.
5. [Evaluate the fine-tuned checkpoint](docs/how-to/evaluate-checkpoint.md)
   through Arena in the same scene used for recording, then inspect the metrics.

The [physical G1 path](docs/workflows/g1-real-static-apple.md) records MCAP,
converts it to LeRobot, fine-tunes GR00T, and exports a LEAPP bundle for Isaac ROS
deployment and hardware evaluation.

Use the workflow chooser above for step-by-step instructions. See [Architecture](docs/explanation/architecture.md)
for the workflow and robot-control software. Repository checks, simulator
smoke tests, and integration setup are in the
[Developer Guide](docs/reference/development.md). Exact runtime ownership and
supported boundaries are in [Reference](docs/reference/index.md).

## License

Repository-owned code and documentation text are licensed under [Apache 2.0](LICENSE).
The adapted video converters and recorder rate limiter also retain Isaac Lab's
BSD-3-Clause terms in [NOTICE](NOTICE). Adjacent `.license` files provide notices for files whose
contents must remain unchanged.

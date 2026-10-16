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

(getting-started)=
# G1 Simulation

Launch the simulated Unitree G1 apple-to-plate scene and practice AGILE
locomotion before recording demonstrations.

## Requirements

Use the workstation, repository checkout, and locked environment prepared by
[Installation](../installation.md#system-requirements).
Run the commands below from the repository root.

(create-the-environment)=
Complete [Create and Check the Simulation Environment](../installation.md#create-and-check-the-simulation-environment)
before launching the viewer. Both G1 compositions use that environment.
On first use, Isaac Sim may request NVIDIA EULA acceptance and download robot,
scene, and policy assets.

## Drive the Unitree G1

1. Launch the native AGILE viewer:

   ```bash
   uv run scripts/locomotion_with_agile.py --device cuda:0 --viz kit
   ```

2. Click the viewport to give it keyboard focus. Use the controls below to move
   the robot, then release the movement keys to stop.

   | Input | Command |
   | --- | --- |
   | **Up** / **Down** | Walk forward / backward |
   | **Left** / **Right** | Turn |
   | **A** / **D** | Strafe |
   | Release **Up**, **Down**, **Left**, **Right**, **A**, and **D** | Stop and balance |

3. Confirm that the robot stands upright, responds to the controls, and balances
   when you release them. Stop the viewer with **Ctrl+C** in its terminal
   before starting another simulation.

Use `--help` to list options such as `--speed`, `--turn`, `--hip_height`, and
constant velocity commands such as `--vx 0.3`.

:::{note}
This viewer enables AGILE for its own process. Select the recording mode
separately using the commands in [G1 demonstration recording](../how-to/record-demonstrations.md).
:::

(drive-the-simulated-robot-through-isaac-lab-arena)=
## Optional Isaac Lab-Arena Simulation

Use this alternative to preview the composition used for checkpoint evaluation.
It has the same keyboard controls and uses Arena's own locomotion policy.

1. Launch it from the repository root:

   ```bash
   ./tools/launch_arena_locomotion.sh --viz kit
   ```

   The launcher fetches and validates the pinned Arena checkout, then uses the
   repository's locked environment. See
   [How Arena Is Obtained](../reference/development.md#how-arena-is-obtained)
   for checkout details.

2. Repeat the movement and balance check above, then stop the viewer with
   **Ctrl+C** in its terminal.

(next-step)=
## Next Stage

Continue to [G1 demonstration recording](../how-to/record-demonstrations.md)
to capture apple-to-plate demonstrations with a headset. This viewer stage
produces no training dataset.

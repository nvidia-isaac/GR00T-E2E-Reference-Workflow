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

(h2-validate)=
# H2 Simulation Validation

Check that the Unitree H2 simulation, recording, replay, and dataset conversion
work together, then try driving the robot before recording demonstrations.

## Prerequisites

Complete [H2 Asset Preparation](prepare-h2.md). On the GPU workstation, restore
its environment variables and run commands from the repository root.

## Run the Simulation Checks

1. Validate both H2 embodiments, the camera, Pink kinematics, Mimic round trips,
   and the recording, strict replay, and conversion pipeline:

   ```bash
   uv run --frozen --no-sync ./tools/validate_repository.sh sim-arena-h2
   ```

2. Confirm the command exits successfully. Inspect the reported validation logs
   if a stage fails before continuing to recording.

:::{important}
These checks convert and load a test recording of an unsuccessful episode to
verify the 75 state values, 61 action values, and video format. To validate
AutoData, you also need a successful teleoperation recording made in simulation
using a PICO 4 Ultra headset and handheld XR controllers.
:::

The timing check requires at least 0.35 real-time factor for a synchronized
300-step recorder workload. It excludes OpenXR and CloudXR. Check headset
latency and visual smoothness during recording. See
[H2 runtime timing](../explanation/h2-architecture.md#runtime-timing) for the
control rates and what this minimum performance requirement covers.

## Check Interactive Locomotion

1. Launch H2 in Isaac Lab-Arena:

   ```bash
   ./tools/launch_arena_locomotion.sh --robot h2 --viz kit
   ```

2. Use the **Arrow Keys** for forward motion and yaw. Press **A** and **D** to
   strafe. H2 accepts no hip-height command.

## Expected Result

The simulation checks complete and write their logs, metrics, and dataset checks
under `test-results/arena-h2/<UTC-timestamp>/` by default. The simulated H2
responds to locomotion input. You can then try recording the bimanual task.

## Next Stage

Continue to [H2 Demonstration Recording](record-h2-demonstrations.md).

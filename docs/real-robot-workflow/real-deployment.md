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

(deployment)=
(unitree-g1-deployment-and-evaluation)=
# Unitree G1 Policy Deployment and Evaluation

Deployment runs a GR00T N1.7 policy on the physical Unitree G1 for the
apple-to-plate task. Start with the bundled policy, then optionally evaluate
your own LEAPP export using the same launch procedure.

## Prerequisites

- Complete [Isaac ROS Setup on Thor](isaac-ros-setup.md) and the
  [G1 safety checks](g1-introduction-and-safety.md).
- Run on Thor, connected externally or as a G1 backpack, with an operator and
  a safety monitor.
- [Stop teleoperation](real-teleop.md#stop-teleoperation) and its recorder before
  starting the deployment stack.
- For a custom policy, copy the complete [LEAPP bundle](real-leapp-export.md)
  into `${ISAAC_ROS_WS}/policies/<export-name>/` on Thor.

## Prepare for Deployment

1. Open a Thor host terminal and enter the Isaac ROS environment:

   ```bash
   isaac-ros activate
   ```

2. Download and install the pre-trained GR00T Unitree G1 policy assets. The bundled model is [`nvidia/GR00T-N1.7-ApplePnP-V1`](https://huggingface.co/nvidia/GR00T-N1.7-ApplePnP-V1):

   ```bash
   ros2 run isaac_ros_gr00t_unitree_g1_install install_gr00t_unitree_g1.sh --eula
   ```

   This stores the bundled model files and LEAPP YAML under `${ISAAC_ROS_WS}/isaac_ros_assets/models/gr00t_unitree_g1/`. Run this command in the same container and workspace used for deployment so asset paths and file permissions match the runtime environment.

## Run the Policy on Real Hardware

:::{danger}
Before running the policy autonomously:

1. Clear all people from the robot's workspace.
2. Maintain visual contact with the robot at all times.

Review [Unitree G1 Introduction and Safety](g1-introduction-and-safety.md) before enabling autonomous policy control. This documentation is a workflow guide, not a robot operating guide.
:::

1. Before launching, recreate the [workspace setup](g1-introduction-and-safety.md#workspace-setup):

   - Black tablecloth
   - Approximately 75 cm table height
   - White 19 cm plate
   - Red apple

2. Use a gantry to position the robot at the table. Roll it close so the torso is almost touching the table, then rest the robot's hands flat on the table before enabling robot control. Lower the robot from the gantry before step 7.

   The expected execution sequence enables lower-body joints first for balancing, then enables upper-body joints to pick up the apple and place it on the plate.

3. Set up the network in a separate **Thor host terminal**, outside the container. Skip this step if you already ran network setup during [teleoperation](real-teleop.md) and have not rebooted Thor. Re-run it after a reboot or if you reconnect Ethernet.

   ```bash
   ${ISAAC_ROS_WS}/src/isaac_ros_robots/isaac_ros_robots_tools/scripts/setup_network.py
   ```

   Select the network interface connected to the Unitree G1. On Thor this is normally `enP2p1s0`.

4. In that host terminal, enter the Isaac ROS Docker environment:

   ```bash
   isaac-ros activate
   ```

5. Set up CUDA MPS (**inside the container**). This allows the GPU to be partitioned so the whole-body control policy runs in real time without contention from the GR00T policy.

   Without MPS, AGILE balancing inference and GR00T policy inference can compete
   for GPU resources, which can lead the robot to lose balance.

   ```bash
   ${ISAAC_ROS_WS}/src/isaac_ros_physical_ai/isaac_ros_unitree_g1_gr00t/scripts/setup_mps.sh
   ```

6. Launch the deploy stack with the pre-trained apple-to-plate policy (**inside the container**). Use the network interface you selected during network setup in step 3, for example `enP2p1s0` on Thor:

   ```bash
   ros2 launch isaac_ros_unitree_g1_gr00t unitree_g1_gr00t_agile.launch.py \
       hardware_type:=real \
       network_interface:=enP2p1s0 \
       use_foxglove:=false
   ```

   This uses the default pretrained policy installed by `install_gr00t_unitree_g1.sh --eula`.

   The command loads the policy onto the GPU, which takes up to 30 seconds.

   You can enable Foxglove by changing the last parameter. It is disabled here to keep the terminal output simpler.

   Before enabling control, check the camera view. Open a separate host terminal,
   run `isaac-ros activate`, then `rviz2`. Add
   **/realsense_d435_rgb/color/image_raw/Image** and confirm the apple, plate,
   and table are visible and within reach. Adjust the scene or camera while
   both ROS safety controllers remain at `blend_ratio=0.0`. Close RViz and keep
   this Isaac ROS terminal open for the ROS safety-controller commands below.

7. Enable the lower body:

   Lower the robot from the gantry so its feet are on the ground before you enable lower-body control. The AGILE policy balances the robot while standing, so **do not enable lower-body control while the robot is still suspended in air.**

   Lower-body control enables first, then upper-body manipulation follows.

   Similar to [teleoperation](real-teleop.md), the GR00T deploy stack uses a
   `blend_ratio` parameter to smoothly enable control.

   Unlike teleoperation, the GR00T deployment stack uses separate `blend_ratio`
   values (and separate ROS safety controllers) for the lower and upper body.

   Enable balancing by setting the lower-body `blend_ratio` to `1.0`:

   ```bash
   ros2 param set /safety_controller_lower_body blend_ratio 1.0
   ```

   The robot should now balance and stand on its own.

   :::{tip}
   Keep `ros2 param set /safety_controller_lower_body blend_ratio 0.0` in your
   shell history so you can disable control quickly if needed.
   :::

8. Enable the upper body:

   :::{important}
   If you are testing your own fine-tuned policy for the first time, gradually increase `blend_ratio` rather than jumping directly to `1.0`.
   :::

   When you set the upper-body blend ratio to `1.0`, the GR00T policy immediately starts picking up the apple and placing it on the plate.

   ```bash
   ros2 param set /safety_controller_upper_body blend_ratio 1.0
   ```

   To repeat the trial you can disable the upper body, reset the apple and plate
   on the table, and then enable the upper body again:

   ```bash
   ros2 param set /safety_controller_upper_body blend_ratio 0.0
   ```

   Now, reset the apple and plate, then run again:

   ```bash
   ros2 param set /safety_controller_upper_body blend_ratio 1.0
   ```

## Use Your Own Policy

To replace the bundled policy with your own export, first complete
[Stop Deployment](#stop-deployment). Then repeat the hardware launch sequence
with the custom bundle:

1. Place the exported LEAPP bundle in a directory that is mounted into the
   container, such as `${ISAAC_ROS_WS}/policies/<export-name>/`.

2. Make the LEAPP bundle world-readable inside the Docker container:

   ```bash
   chmod -R a+rX ${ISAAC_ROS_WS}/policies/<export-name>
   ```

3. Repeat the same instructions as before, but now when launching the app add
   the launch argument `gr00t_leapp_yaml_path` and point it to the YAML file in
   your exported LEAPP bundle.

   ```bash
   ros2 launch isaac_ros_unitree_g1_gr00t unitree_g1_gr00t_agile.launch.py \
       hardware_type:=real \
       network_interface:=enP2p1s0 \
       use_foxglove:=false \
       gr00t_leapp_yaml_path:=${ISAAC_ROS_WS}/policies/<export-name>/<export-name>.yaml
   ```

   You can enable Foxglove by changing the parameter to `true`. It is disabled here to keep the terminal output simpler.

## Structured Evaluation Approach

Evaluate the success rate of the policy.

The success rate depends on how closely your environment matches the environment used during data collection.

Run the evaluation with two people: one person stays at the computer to enable and disable the policy, while the other person resets the scene.

:::{important}
The v0.1 record specifies the placement grid and rollout count below, but does
not define the completion condition, timeout, or intervention scoring. Its
reported 68% success rate cannot establish a directly comparable benchmark
without those original criteria. Define and record these criteria before a new
evaluation; keep new results separate from the historical result.
:::

### Setup

Set up the evaluation grid:

1. With a dark marker or another subtle marking system, mark two `19 cm x 19 cm` squares on the table in front of the robot. One is for the apple positions, one is for the plate.
2. Divide each square into four equal positions and mark an additional position at the center, producing five apple positions and five plate positions.

These regions support a structured evaluation of the policy. Center the objects at their marked positions for each evaluation trial.

### Process

1. Begin evaluations. For the first, keep the apple in the first grid position and rotate the plate through all five plate positions.
2. Move the apple to the next grid position and rotate the plate through all five plate positions again.
3. Repeat this pattern until all apple and plate combinations have been evaluated.

This produces **25 unique apple-to-plate placement combinations**. Run the full set four times for a structured **100-rollout evaluation**.

The v0.1 reference evaluation repeated the 25-position set four times for 100 rollouts and measured a **68% success rate**. This is the v0.1 reference result, not a general baseline.

Record each rollout's apple and plate positions, result, elapsed time, and any
intervention. Include the checkpoint or bundle identity and software versions
with the results so another operator can interpret them.

## Stop Deployment

Follow the [ordered stop procedure](g1-introduction-and-safety.md#stop-and-abort-procedure)
before replacing the policy or ending a session:

1. In the safety-command terminal, disable robot control through both active ROS safety controllers:

   ```bash
   ros2 param set /safety_controller_upper_body blend_ratio 0.0
   ```

   ```bash
   ros2 param set /safety_controller_lower_body blend_ratio 0.0
   ```

2. Press **Ctrl+C** in the ROS launch terminal and wait for a clean exit.
3. Follow the approved manufacturer and lab procedures for securing and
   powering down the robot.

## Troubleshooting

- If the robot does not stand after enabling lower-body control, verify
   real-time prerequisites (`rtprio` and CUDA MPS) and network interface settings.
- If a custom upper-body policy does not start, verify that
   `gr00t_leapp_yaml_path` points to a container-visible LEAPP YAML file. For
   bundled-policy deployment, leave `gr00t_leapp_yaml_path` empty.
- If Triton or ONNX Runtime cannot load a custom LEAPP policy, make the bundle world-readable:

  ```bash
  chmod -R a+rX /path/to/your/leapp_export_dir
  ```

- If behavior regresses between runs, reset upper-body `blend_ratio` to `0.0`, re-place objects, and re-enable.

(conclusion)=
## Expected Result

Retain the rollout results, scoring rules, policy identity, and software versions.
These hardware observations establish the outcome of this evaluation. A
successful process launch or repository CI run alone does not qualify the policy.

:::{seealso}
Refer to the [Isaac ROS Unitree G1 GR00T package reference](https://nvidia-isaac-ros.github.io/v/release-4.5/repositories_and_packages/isaac_ros_physical_ai/isaac_ros_unitree_g1_gr00t/index.html) for additional package details.
:::

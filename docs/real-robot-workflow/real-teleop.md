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

(teleoperation)=
# Unitree G1 Teleoperation

XR teleoperation combines AGILE whole-body control, bimanual inverse kinematics,
and headset-driven finger control on the Unitree G1. Verify the headset and
CloudXR session in MuJoCo, then operate the physical robot.

## Prerequisites

- Complete [Isaac ROS Setup on Thor](isaac-ros-setup.md) and the
  [G1 safety checks](g1-introduction-and-safety.md).
- Use a Jetson AGX Thor connected to the G1, externally or in a Thor backpack.
- Prepare an XR headset and a display connected to Thor for the MuJoCo test.
- Verify the [CloudXR network requirements](https://docs.nvidia.com/cloudxr-sdk/latest/requirement/network_setup.html#network-requirements)
  and the [firewall ports](isaac-ros-setup.md#configure-cloudxr-firewall-ports).

:::{tip}
Before collecting data, review the [recording quality guidance](real-record.md#recording-quality).
:::

:::{warning}
Use at least two people for physical teleoperation:

- One person for teleoperation
- One person to monitor the robot for safety purposes
:::

## Activate Isaac ROS

The `gr00t_workflow` Docker layer from [Isaac ROS Setup on Thor](isaac-ros-setup.md) includes the teleop bringup package.

Start a new terminal and activate the Isaac ROS environment:

   ```bash
   isaac-ros activate
   ```

## Start CloudXR

:::{tip}
Update the headset OS and native browser before connecting.
:::

1. Start the CloudXR runtime and accept the EULA:

   ```bash
   python3 -m isaacteleop.cloudxr --accept-eula
   ```

   Keep this application running until you finish teleoperation.

2. On the XR headset, open the native browser and navigate to `https://nvidia.github.io/IsaacCapture/client/release-1.3.x/#/real/isaacros`.

3. Enter the Thor's host IP address in the **Server IP** field.

4. Select the **Click https://<ip>:48322/ to accept cert** link that appears on the page after entering the IP address.

5. Accept the certificate in the new browser page, then return to the CloudXR.js client page.

6. On the PICO headset, set **Video Codec** to **H.264** in the NVIDIA Isaac Capture
   client before connecting. Other codecs can prevent connection.

7. Select **Connect** to connect the headset to the teleop server.

## Initial Test in MuJoCo

Before connecting to a physical Unitree G1, use MuJoCo to verify that the headset, CloudXR session, and teleoperation commands are working.

This test runs on Thor without a real Unitree G1 connected. A display must be connected to Thor for this step.

Two headset options are available:

- Wear the headset normally with **AR Immersive** passthrough enabled on the client page. The simulation is not streamed to the headset.
- Remove the cushion and pull it down to the neck, then watch the monitor instead. When wearing the headset on the neck, ensure clothing does not block the bottom cameras of the headset for tracking.

1. Open a new terminal and activate Isaac ROS:

   ```bash
   isaac-ros activate
   ```

2. Activate the CloudXR environment variables:

   ```bash
   source ~/.cloudxr/run/cloudxr.env
   ```

3. Launch the teleop application in MuJoCo:

   ```bash
   ros2 launch isaac_ros_unitree_g1_teleop_bringup unitree_g1_teleop.launch.py \
       hardware_type:=mujoco \
       input_mode:=teleop
   ```

   The MuJoCo viewer opens with the Unitree G1 robot on Thor desktop. A virtual gantry holds the robot upright during startup.
   - Press **g** to toggle the gantry on or off.
   - Press **[** and **]** to shorten or lengthen the gantry rope.

4. With the XR controllers in your hands, move them and confirm that the
   simulated robot's arms track your motion. You can also use the joysticks on the
   XR controllers to make the robot walk.

   :::{admonition} Handheld XR Controller Inputs
   :class: note

   The pose of the XR controllers is used to command the pose of the robot's hands.

   The left joystick rotates the robot. The right joystick walks the robot forwards, backwards, or sideways.
   :::

5. After verifying that the handheld XR controllers are tracked correctly, stop the app with **Ctrl+C**.

   Stop MuJoCo before launching real-hardware teleoperation. If a leftover
   MuJoCo process still opens, the CloudXR client UI disappears after
   certificate acceptance, or ROS 2 topics from another robot appear, see
   [Troubleshooting](../reference/troubleshooting.md).

## Set Up the Robot Network

The `isaac_ros_robots` repository is cloned during [Isaac ROS Setup on Thor](isaac-ros-setup.md).

1. Confirm the Unitree G1 Ethernet cable is plugged directly into Thor and the robot is
   powered on.

2. Open a new terminal on the Thor and run the network setup script. Run this on the host, not inside the Docker container.

   ```bash
   ${ISAAC_ROS_WS}/src/isaac_ros_robots/isaac_ros_robots_tools/scripts/setup_network.py
   ```

3. When prompted, select the network interface that is physically connected to
   the Unitree G1. On Thor, this is normally `enP2p1s0`.

## Launch Real-Hardware Teleoperation

Before operating real hardware, complete this checklist:

- [ ] Clear the work area of people and hazards.
- [ ] Ensure the waist yaw joint is close to zero before launch.

1. Open a new terminal and start the Isaac ROS environment:

   ```bash
   isaac-ros activate
   ```

2. Activate the CloudXR environment variables:

   ```bash
   source ~/.cloudxr/run/cloudxr.env
   ```

3. Launch the teleop application. Use the network interface you selected during [network setup](#set-up-the-robot-network), for example `enP2p1s0` on Thor:

   ```bash
   ros2 launch isaac_ros_unitree_g1_teleop_bringup unitree_g1_teleop.launch.py \
       hardware_type:=real \
       input_mode:=teleop \
       network_interface:=enP2p1s0
   ```

4. Verify that teleoperation commands are arriving from the XR headset. Open a new terminal, enter the Isaac ROS environment, and echo the ROS topic. New commands should appear multiple times per second.

   ```bash
   isaac-ros activate
   ```

   ```bash
   ros2 topic echo /xr_teleop/ee_poses
   ```

   After confirming incoming commands, press **Ctrl+C** to stop the topic echo.
   Keep this terminal open for the ROS safety-controller commands below.

5. Reset the headset's world frame by pressing the **Home** button (the button
   with the circle) on the right XR controller for 2 seconds.

## Disable and Enable Control

By default, control is disabled. The stack uses a `blend_ratio` parameter to
smoothly enable control. At launch, `blend_ratio` is `0.0`, so only default
motor commands are sent. When you set `blend_ratio` to `1.0`, the ROS safety
controller fully enables the robot's balancing and teleoperation commands.
Changes ramp smoothly between values.

:::{note}
**Simulation vs. real hardware:** In the simulation test, `blend_ratio` defaults to 1.0, so control is active from the start.
:::

Before enabling control, wait for `Core InferenceController activated` in the
launch terminal. This can take about 40 seconds. While it waits for input,
the ROS inference controller logs
`Waiting for topic messages before activating core controller`.
Run the following commands in the separate Isaac ROS terminal used for the topic
check, with the launch stack still running.

1. To disable the robot, set the blend ratio to zero:

   ```bash
   ros2 param set /safety_controller blend_ratio 0.0
   ```

   :::{tip}
   Keep this command in your shell history so you can execute it quickly if something goes wrong.
   :::

2. To enable the robot, set the blend ratio to one. On real hardware, do this only after launch is healthy and the robot is standing on the ground:

   ```bash
   ros2 param set /safety_controller blend_ratio 1.0
   ```

The robot starts balancing and tracking your hand movements.

## Stop Teleoperation

If continuing directly to recording, keep CloudXR and the teleoperation stack
running and proceed to [Data Recording](real-record.md). To end a session:

Follow the [ordered stop procedure](g1-introduction-and-safety.md#stop-and-abort-procedure):

1. In the safety-command terminal, disable control:

   ```bash
   ros2 param set /safety_controller blend_ratio 0.0
   ```

2. Press **Ctrl+C** in the ROS launch terminal and wait for a clean exit.
3. Stop CloudXR with **Ctrl+C** in its terminal when the session is finished.

If robot behavior is unexpected, use the approved stop or abort procedure
before continuing.

<a id="key-takeaways"></a>

## Next Stage

Before recording, confirm that the camera view is clear, XR commands arrive,
and the robot tracks the operator while enabled. Continue to
[Unitree G1 Demonstration Recording](real-record.md).

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

(data-recording)=
(unitree-g1-data-recording)=
# Unitree G1 Demonstration Recording

The Isaac ROS recorder captures physical Unitree G1 teleoperation demonstrations
as MCAP sessions for conversion to LeRobot.

## Prerequisites

- Complete [real-hardware teleoperation](real-teleop.md#launch-real-hardware-teleoperation) on Thor, connected
  externally or as a backpack.
- Keep CloudXR and the teleoperation launch stack running in their terminals.
- Use a separate Thor terminal for the camera checks and recorder.
- Review [Recording Quality](#recording-quality) before the first take.

:::{warning}
Use at least two people for recording:

- One person for teleoperation
- One person to monitor the robot for safety purposes
:::

## Workspace Prep

1. Before recording, recreate the physical [workspace setup](g1-introduction-and-safety.md#workspace-setup).

2. Keep the table, object appearance, robot pelvis position, and head pose consistent across the dataset.

3. On the Thor host, outside the container, confirm the kernel sees the RealSense camera over USB:

   ```bash
   lsusb | grep -i intel
   ```

4. In the same terminal, enter Isaac ROS:

   ```bash
   isaac-ros activate
   ```

   Inside the container, confirm the camera is visible:

   ```bash
   rs-enumerate-devices
   ```

   The output should include a connected RealSense with its serial number and firmware version.

5. Confirm the ego-view camera frame in RViz:

   ```bash
   rviz2
   ```

   Select **Add > By topic > /realsense_d435_rgb/color/image_raw/Image > OK** and verify the workspace is visible before starting the recorder.

   Close RViz to return to the shell before starting the recorder.

## Start the Recorder

1. In the Isaac ROS terminal used for the camera checks, run the recorder:

   ```bash
   ros2 run isaac_ros_unitree_g1_recorder record -- \
       task_description:="move the apple to the plate"
   ```

2. Verify that the camera is capturing data. Two methods to check:

   - Use RViz to view camera topics
   - Use a second ROS-sourced terminal to list topics with `ros2 topic list`

   :::{dropdown} If camera data is not being recorded
   Follow the [ordered stop procedure](real-teleop.md#stop-teleoperation) before
   inspecting cables. Once the robot is stopped:

   1. Stop the recorder.
   2. Disconnect and reconnect the RealSense USB cable.
   3. Repeat the camera checks, teleoperation startup, and recorder launch.
   :::

3. Use the terminal UI:

   - Press **Space** to start an episode.
   - Press **Space** again to save.
   - Press **c** to discard.

Recordings are stored in `${ISAAC_ROS_WS}/recordings/`.

## Recording Quality

- For high-quality datasets, teleoperate slow movements and keep the robot body fixed.

- After completing the task, move the left hand out of view and hold it still for 2 to 3 seconds before saving the recording.

- For the apple-to-plate task, only the left hand should move and perform the task. Favor side grasps with a bottle-holding-style hand pose so the ego-view camera can see the hand and apple clearly.

- Vary the apple and plate position and orientation between episodes, but keep both reachable while the robot stands still. Keep apple type, plate type, table height, lighting, and camera setup fixed for the baseline dataset.

- Keep the right arm mostly outside the camera field of view. It is acceptable for right-hand fingers to appear occasionally because the right hand may enter view during deployment.

- Keep the baseline dataset focused on clean successful episodes. Add failed-grasp recovery examples only when you intentionally want a broader dataset, and keep them limited so they do not dominate the training signal. Collect at least 200 successful episodes for the baseline training recipe.

## Next Stage

Save or discard the active take, then exit the recorder. Follow
[Stop Teleoperation](real-teleop.md#stop-teleoperation) when finished with the
robot. Confirm that the saved session directories exist under
`${ISAAC_ROS_WS}/recordings/` and retain them as the source data.

Continue to [Unitree G1 Dataset Conversion](real-data-export.md). If conversion runs on
a workstation, copy the complete session directories there first.

:::{seealso}
Refer to the [isaac_ros_unitree_g1_recorder](https://nvidia-isaac-ros.github.io/v/release-4.5/repositories_and_packages/isaac_ros_physical_ai/isaac_ros_unitree_g1_recorder/index.html) documentation for full keyboard reference, launch arguments, recorded topics, services, build-from-source instructions, and troubleshooting.
:::

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

# Unitree G1 Introduction and Safety

Before working with the physical Unitree G1, review the workflow-specific
safety checkpoints for teleoperation, data recording, and policy deployment.

Support is bounded to the hardware and software releases documented for v0.1.
Physical hardware behavior is not tested by repository CI.

:::{important}
This documentation is a workflow guide, not an operating guide for the
Unitree G1. Use the manufacturer's documentation and your organization's robot-safety
procedures for operation, stop and abort behavior, power-on, power-down,
recovery, maintenance, and lab approval.
:::

For the complete procedure order and supported hardware, see the
[Unitree G1 real static apple workflow path](../workflows/g1-real-static-apple.md).

## Safety Protocols

:::{danger}
The Unitree G1 is a powerful humanoid robot with substantial mass and moving limbs.
Improper operation can injure people or damage the robot and workspace. Follow
all approved safety protocols.
:::

### Workflow Safety Checks

Complete these checks before every teleoperation, recording, or deployment
session:

- [ ] Clear unnecessary objects and obstacles from the workspace.
- [ ] Verify the table and task objects are positioned correctly.
- [ ] Route every cable so it cannot interfere with robot motion.
- [ ] Ensure at least one person can activate the approved external stop or
      abort mechanism at all times.
- [ ] Use at least two people: one operator and one safety monitor.
- [ ] Keep bystanders outside the robot's reach envelope.

### Stop and Abort Procedure

:::{important}
The Unitree G1 does not include a built-in e-stop button. The workflow's ROS
safety controller can stop robot motion, but it does not fully replace an e-stop.
:::

Stop the robot during teleoperation or deployment in this order:

1. **Disable control first.** Set `blend_ratio` to `0.0` on the active ROS safety
   controllers. Use the exact commands in the
   [teleoperation](real-teleop.md#disable-and-enable-control) and
   [deployment](real-deployment.md) procedures.
2. **Stop the ROS launch stack.** Press **Ctrl+C** in its terminal and wait for
   the process to exit cleanly.
3. **Disconnect Ethernet only as a last resort.** If the robot still behaves
   unexpectedly, disconnect the Ethernet cable between the Unitree G1 and Thor to cut
   motor commands.

Avoid disconnecting Ethernet while robot control remains enabled. Stopping the
stack first allows the ROS safety controller to ramp down cleanly.

### Manufacturer Operating Procedures

Use Unitree's procedures for power-on, power-down, recovery, and emergency
behavior. This workflow is not the authority for those procedures.

### Operational Safety Guidelines

1. Never place hands or body parts within the active workspace during
   autonomous operation.
2. Maintain visual contact with the robot throughout policy deployment.
3. Start new policy tests at slow execution speeds.
4. If behavior becomes unexpected, activate the approved stop or abort
   mechanism immediately. Do not physically restrain the robot.

## Robot Setup

Connect Thor to the Unitree G1 with:

- USB from the head-mounted RealSense camera to Thor. Plug it directly into a
  Thor backpack, or use a USB extension cable for an external Thor.
- Direct Ethernet from the Unitree G1 to Thor for network communication. Do not insert
  a network switch or USB-to-Ethernet dongle.

The intended topology is one USB camera connection and one direct Ethernet
control connection between the Unitree G1 and Thor.

## Workspace Setup

The task is a left-handed tabletop pick-and-place: the robot picks up a red
apple and places it on a white plate.

Place the Unitree G1 at a table approximately 75 cm high and cover the table with a
black tablecloth. Put a red apple on the robot's left and a white circular
plate, approximately 19 cm in diameter, on its right. Use a white wall behind
the table. Keep both objects in the head-mounted RealSense camera's field of
view and within comfortable reach of the left arm. Establish a safety
perimeter, an operator station outside the reach envelope, and lighting that
illuminates the task surface uniformly.

:::{important}
The apple must start to the left of the plate. The pretrained policy was
trained only on that arrangement.
:::

### Lighting

Keep lighting stable between data collection and deployment. Eliminate harsh
table shadows, avoid direct glare on the camera, and illuminate the full
workspace uniformly.

### Robot Start Pose

Place the Unitree G1 pelvis very close to the table edge, touching or lightly pushing
against it. Face the robot toward the table. Keep the head neutral so the neck
plane remains parallel to the top of the torso.

<a id="key-takeaways"></a>

## Next Stage

The workspace, cable routing, stop access, and two-person operating roles must
be ready before control is enabled. Continue to
[Isaac ROS Setup on Thor](isaac-ros-setup.md) to prepare the software environment
and verify the RealSense camera.

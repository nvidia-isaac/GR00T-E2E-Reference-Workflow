---
tocdepth: 2
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


# Troubleshooting

Use this page when hardware setup, camera access, image streaming, or compute-unit performance issues block the Unitree G1 workflow.

:::{important}
Stop the robot before troubleshooting hardware, cables, network connections, or camera access. Follow manufacturer safety procedures first.
:::

## Hardware Setup Reference

Use the Isaac hardware setup troubleshooting guide for the source troubleshooting reference:

```{button-link} https://nvidia-isaac-ros.github.io/v/release-4.5/troubleshooting/hardware_setup.html
:color: primary
:outline:

Open Isaac Hardware Setup Troubleshooting
```

## RealSense Camera Not Found

### Symptom

Use this section when no RealSense camera is detected and one of these conditions occurs:

- `rs-enumerate-devices` fails without `sudo`, but succeeds with `sudo rs-enumerate-devices`.
- `ros2 launch realsense2_camera rs_launch.py` reports `failed to set power state` or `The requested device ... is NOT found`.
- `realsense-viewer` doesn't detect any RealSense camera.

### Solution 1: Check RealSense Versions

Follow the RealSense setup instructions and confirm the installed versions for `librealsense`, `realsense2_camera`, and camera firmware.

Run the RealSense ROS node inside the Isaac ROS environment:

```bash
ros2 launch realsense2_camera rs_launch.py
```

Verify that the output reports the expected RealSense ROS and LibRealSense versions.

Check the camera firmware version:

```bash
rs-enumerate-devices
```

To update camera firmware:

1. Download the recommended D400 firmware binary from the [Intel RealSense firmware releases](https://dev.realsenseai.com/docs/firmware-releases-d400).
2. Extract the binary from the downloaded `.zip` file.
3. Run the firmware update command, replacing `<binary_filename>` with the downloaded binary name:

   ```bash
   rs-fw-update -f <binary_filename>
   ```

### Solution 2: Install RealSense udev Rules

Verify that `99-realsense-libusb.rules` exists in `/etc/udev/rules.d/`.

If it doesn't exist, run these commands on the host machine before activating
the Isaac ROS environment. Continue only after each command succeeds:

```bash
wget https://raw.githubusercontent.com/IntelRealSense/librealsense/v2.56.3/config/99-realsense-libusb.rules
```

```bash
sudo mv 99-realsense-libusb.rules /etc/udev/rules.d/
```

```bash
sudo udevadm control --reload-rules
```

```bash
sudo udevadm trigger
```

Confirm that each command exits without an error.

(isaac-teleop-ui-partially-disappears-after-certificate-acceptance)=
(nvidia-isaac-datacapture-ui-partially-disappears-after-certificate-acceptance)=
## NVIDIA Isaac Capture UI Partially Disappears After Certificate Acceptance

### Symptom

After accepting the CloudXR certificate in the headset browser, part of the NVIDIA Isaac Capture web client UI disappears or fails to render.

### Solution

Update the headset OS and native browser, then reopen the NVIDIA Isaac Capture web client and repeat the certificate acceptance step.

For PICO headsets: software version 5.15.5 was tested with this workflow.

## MuJoCo Teleop Interferes With Real Robot Teleop

### Symptom

When starting real robot teleop, a new MuJoCo window opens instead and the simulated robot falls because there is no virtual gantry.

### Cause

The MuJoCo teleop terminal or a background MuJoCo process is still running.

### Solution

Close the MuJoCo teleop window and stop the MuJoCo teleop terminal before starting real robot teleop. If the window is closed but the issue continues, stop any background MuJoCo processes and launch real robot teleop again.

## Input Images Must Have an Even Height and Width

### Symptom

Isaac ROS nodes terminate when given images with odd width or height. Terminal logs can include:

```text
[NitrosImage]: [convert_to_custom] Image width/height must be even for creation of gxf::VideoBuffer
terminate called after throwing an instance of 'std::runtime_error'
what():  [convert_to_custom] Odd Image width or height.
```

### Solution

Replace the input image source with one that produces images with even width and height.

## RealSense Does Not Stream IR Stereo Images

### Symptom

Inside the Isaac ROS environment, `realsense-viewer` doesn't show IR images, but depth images stream. Outside the Isaac ROS environment, no metadata containing projector status is attached to the IR frames.

### Check the Supported Camera Stack

Jetson Linux R38.4 uses Linux 6.8, as specified in its
[release notes](https://docs.nvidia.com/jetson/archives/r38.4/ReleaseNotes/Jetson_Linux_Release_Notes_r38.4.pdf).
The Kernel 5.15 DKMS repair does not support this Thor kernel. No replacement
kernel-module repair is qualified by this workflow.

1. Record the host kernel version with `uname -r` and camera firmware with
   `rs-enumerate-devices`.
2. Check the camera libraries and container layers against the
   [Isaac ROS 4.5 RealSense setup](https://nvidia-isaac-ros.github.io/v/release-4.5/getting_started/sensors/realsense_setup.html)
   and the [workflow setup](../real-robot-workflow/isaac-ros-setup.md#configure-realsense-support).
3. If IR streaming still fails, provide those versions and the camera logs to
   the hardware support owner before changing kernel modules. Check RGB
   streaming separately; an IR failure alone does not establish an RGB
   recording failure.

## RealSense Error: Failed to Resolve the Request

### Symptom

Any RealSense tutorial launches, but no images stream. Terminal logs include `Failed to resolve the request`.

### Solution

Confirm that the camera uses the correct firmware. Use the RealSense version and firmware checks in [RealSense Camera Not Found](#realsense-camera-not-found).

## RealSense Incompatible QoS Policy Error

### Symptom

A RealSense output topic isn't subscribed when a RealSense tutorial launches. Terminal logs report an incompatible QoS policy, such as `RELIABILITY_QOS_POLICY`.

### Solution

Set the QoS policy in the RealSense configuration file used by the launch graph so the RealSense publisher is compatible with the subscribing Isaac ROS node.

For the relevant RealSense image QoS setting, such as `depth_qos` or `color_qos`, use `SYSTEM_DEFAULT`.

## Intel RealSense Camera Accidentally Enables Laser Emitter

### Symptom

The Intel RealSense laser emitter turns on even though `emitter_enabled` is set to `0`.

### Solution

Run this command at runtime:

```bash
ros2 param set /camera/camera depth_module.emitter_enabled 0
```

## ROS_DOMAIN_ID Crosstalk

### Symptom

Use this section when multiple robots, Thor systems, or ROS 2 groups are on the same network and you see unexpected ROS traffic:

- `ros2 topic list` shows topics from another robot or workstation.
- Nodes connect to the wrong robot, camera, recorder, or deployment stack.
- Teleop, recording, or deployment behavior is unexpected or changes when another ROS 2 system starts.

### Cause

ROS 2 uses DDS discovery. When multicast is supported on the network, ROS 2 systems on the same `ROS_DOMAIN_ID` can discover each other across that network. This can cause crosstalk when multiple robots or ROS 2 workflows run in the same lab.

For background, see the ROS 2 Humble documentation for [`ROS_DOMAIN_ID`](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Domain-ID.html).

### Solution

1. Check the active domain ID. The default domain ID is `0`.

   ```bash
   echo "${ROS_DOMAIN_ID:-0}"
   ```

2. Assign a unique domain ID before launching nodes:

   ```bash
   export ROS_DOMAIN_ID=12
   ```

3. Confirm that only the expected nodes and topics are visible:

   ```bash
   ros2 node list
   ```

   ```bash
   ros2 topic list
   ```

Restart any launch files that were started with the wrong domain ID.

## Intel RealSense D455 Infra Camera Capped at 15 FPS

### Symptom

The Intel RealSense D455 infra camera is capped at 15 fps even though it's specified to run at 90 fps.

### Solution

The initial frame rate can be capped around 15 fps due to a known RealSense ROS issue. Enable auto exposure:

```bash
ros2 param set /camera/camera depth_module.enable_auto_exposure true
```

## System Throttled Due to Over-Current

### Symptom

A display attached to the compute unit reports `System throttled due to over-current`.

### Solution

This warning indicates hardware throttling due to over-current. It can reduce performance and cause jitter or non-deterministic behavior in the robotics pipeline.

Reduce system load, such as lowering image resolutions or frame rates.

:::{warning}
Disabling system over-current limits can allow the system to overheat and affect
silicon health. Use this only when measuring appropriate limits for the workload.
:::

To disable over-current throttling for the current boot, run:

```bash
sudo su
```

In the root shell, load the power-monitoring function:

```bash
source <(awk '/^function config_hwmon *\(\)/ {flag=1} flag; /^}/ && flag {flag=0}' /etc/systemd/nvpower.sh)
```

Then apply the setting in that same root shell:

```bash
config_hwmon ina3221 curr4_crit 81900 VDD_GPU_SOC
```

This setting does not persist across reboots.

Check how many times the system has throttled due to over-current since the last boot:

```bash
cat "$(find /sys/devices -name oc3_event_cnt -print -quit)"
```

If the command returns a non-zero value, the system has throttled due to over-current. The value indicates the number of throttle events.

## G1 Workflow Checks

Before changing G1 workflow commands, verify these G1-specific basics:

- Confirm the G1 and NVIDIA Thor are powered on and connected through the required USB and direct Ethernet cables.
- Confirm cable routing doesn't interfere with the robot's motion envelope.
- Confirm the RealSense camera connection is visible to the system.
- Confirm the Thor and G1 network interfaces are on the expected network.
- Confirm the robot is in a safe state before restarting teleoperation, recording, or deployment services.

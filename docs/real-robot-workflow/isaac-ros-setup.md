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

# Isaac ROS Setup on Thor

Set up the Isaac ROS development environment on the computer connected to the
Unitree G1. This provides the `isaac-ros` CLI, ROS 2 workspace, Isaac ROS source
repositories, containerized environment, RealSense support, and packages used
by the [Unitree G1 real static apple workflow path](../workflows/g1-real-static-apple.md).

## Prerequisites and Version Boundary

Run these steps on the Thor host. Enter the Isaac ROS container only where the
procedure requests activation. Run each command in order and continue only after
it succeeds. Review [G1 safety](g1-introduction-and-safety.md) before powering on
or connecting the robot.

The setup below retains the JetPack 7.1 / Jetson Linux R38.4 and Isaac ROS 4.5
baseline. The [Isaac ROS 4.5 requirements](https://nvidia-isaac-ros.github.io/v/release-4.5/getting_started/index.html#system-requirements)
identify that platform combination. The source checkouts below use its
`v4.5-0` release snapshots.

:::{important}
This release-aligned setup does not reconstruct the exact v0.1 qualified image.
The repository does not lock all binary package versions or the container image
digest. Preserve an existing qualified installation until its replacement is
qualified. Repository CI does not test physical hardware behavior.
:::

:::{dropdown} Source documentation

These references are pinned to Isaac ROS 4.5. They cover setup and the
Unitree G1 teleoperation and deployment packages:

- [Isaac ROS Getting Started](https://nvidia-isaac-ros.github.io/v/release-4.5/getting_started/index.html)
- [Isaac ROS RealSense Setup](https://nvidia-isaac-ros.github.io/v/release-4.5/getting_started/sensors/realsense_setup.html)
- [Unitree G1 Teleop Bringup](https://nvidia-isaac-ros.github.io/v/release-4.5/repositories_and_packages/isaac_ros_physical_ai/isaac_ros_unitree_g1_teleop_bringup/index.html)
- [Unitree G1 GR00T Deploy](https://nvidia-isaac-ros.github.io/v/release-4.5/repositories_and_packages/isaac_ros_physical_ai/isaac_ros_unitree_g1_gr00t/index.html)
:::

## Initialize the Compute Platform

1. Complete the [Jetson AGX Thor Quick Start Guide](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/quick_start.html), selecting JetPack 7.1 / R38.4 for this baseline.
2. Install the matching JetPack components using the [Jetson AGX Thor JetPack setup guide](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_jetpack.html#install-the-whole-jetpack-components). The latest guide may describe a newer release; keep the selected release consistent.
3. Verify the JetPack release:

   ```bash
   cat /etc/nv_tegra_release
   ```

   Confirm the output includes `R38 (release), REVISION: 4.0`.

4. Set the Thor power mode to `MAXN`. See the [power mode controls guide](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonThor.html#power-mode-controls).

   ```bash
   sudo /usr/sbin/nvpmodel -m 0
   ```

## Create the Isaac ROS Workspace

1. Create the workspace:

   ```bash
   mkdir -p "$HOME/workspaces/isaac_ros-dev/src"
   ```

2. Add its environment variable:

   ```bash
   echo "export ISAAC_ROS_WS=\"\${ISAAC_ROS_WS:-$HOME/workspaces/isaac_ros-dev/}\"" >> ~/.bashrc
   ```

3. Reload the environment:

   ```bash
   source ~/.bashrc
   ```

4. Enter the source directory and clone the Isaac ROS 4.5 snapshots in the
   same terminal:

   ```bash
   cd "${ISAAC_ROS_WS}/src"
   ```

   ```bash
   git clone --branch v4.5-0 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_physical_ai.git
   ```

   ```bash
   git clone --branch v4.5-0 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_robots.git
   ```

   ```bash
   git clone --branch v4.5-0 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_data_tools.git
   ```

5. Verify the checked-out commits with `git -C <repository> rev-parse HEAD`:

   | Repository | `v4.5-0` Commit |
   | --- | --- |
   | `isaac_ros_physical_ai` | `4172a10b274116c461780bfe8aa04f5fe24eb4ae` |
   | `isaac_ros_robots` | `d677275d4dc7a0bc75591251da5ac5f0250bbd5c` |
   | `isaac_ros_data_tools` | `727510a0a7a6da73f722d0b65ca4709488d95d69` |

   Stop if a tag resolves differently. These checkouts provide scripts and
   converter source; they do not pin the binary packages in the Docker image.

## Configure CloudXR Firewall Ports

CloudXR requires network ports to be open. For Quest and PICO headsets using
the WebXR client, open the CloudXR runtime and WebSocket Secure proxy ports
when `ufw` is enabled:

1. Open the minimum ports:

   ```bash
   sudo ufw allow 47998/udp
   ```

   ```bash
   sudo ufw allow 49100,48322/tcp
   ```

2. If you run the WebXR client from source, also open its web-server ports:

   ```bash
   sudo ufw allow 8080,8443/tcp
   ```

## Install and Initialize the Isaac ROS CLI

These instructions use the Docker workflow. For bare-metal or virtual
environment workflows, see the [Isaac ROS initialization documentation](https://nvidia-isaac-ros.github.io/v/release-4.5/getting_started/index.html#initialize-isaac-ros-cli).

1. Check the host locale:

   ```bash
   locale
   ```

   Set it to UTF-8 in the same terminal:

   ```bash
   sudo apt update
   ```

   ```bash
   sudo apt install locales
   ```

   ```bash
   sudo locale-gen en_US en_US.UTF-8
   ```

   ```bash
   sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
   ```

   ```bash
   export LANG=en_US.UTF-8
   ```

   Verify the settings:

   ```bash
   locale
   ```

2. Install Apt repository dependencies:

   ```bash
   sudo apt update
   ```

   ```bash
   sudo apt install curl gnupg
   ```

   ```bash
   sudo apt install software-properties-common
   ```

   ```bash
   sudo add-apt-repository universe
   ```

3. Configure the public NVIDIA Isaac ROS Apt repository for Jetson AGX Thor.
   Run these commands in the same host terminal:

   ```bash
   k="/usr/share/keyrings/nvidia-isaac-ros.gpg"
   ```

   ```bash
   curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key | sudo gpg --dearmor \
       | sudo tee -a $k > /dev/null
   ```

   ```bash
   f="/etc/apt/sources.list.d/nvidia-isaac-ros.list"
   ```

   ```bash
   sudo touch $f
   ```

   ```bash
   s="deb [signed-by=$k] https://isaac.download.nvidia.com/isaac-ros/release-4.5 noble-jetpack main"
   ```

   ```bash
   grep -qxF "$s" $f || echo "$s" | sudo tee -a $f
   ```

   ```bash
   sudo apt-get update
   ```

4. Install the Isaac ROS CLI from the configured 4.5 repository:

   ```bash
   sudo apt-get install isaac-ros-cli
   ```

5. Configure Docker to use the NVIDIA runtime by default:

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
   ```

6. Review the generated configuration:

   ```bash
   cat /etc/docker/daemon.json
   ```

   Expected output:

   ```json
   {
     "runtimes": {
       "nvidia": {
         "args": [],
         "path": "nvidia-container-runtime"
       }
     },
     "default-runtime": "nvidia"
   }
   ```

7. Add your user to the Docker group:

   ```bash
   sudo usermod -aG docker $USER
   ```

   Start a shell with the new group membership:

   ```bash
   newgrp docker
   ```

8. Restart Docker:

   ```bash
   sudo systemctl daemon-reload
   ```

   ```bash
   sudo systemctl restart docker
   ```

9. Verify Docker and the NVIDIA runtime. Docker must list its runtimes,
   `hello-world` must succeed, and the GPU check must print
   `NVIDIA runtime OK`.

   ```bash
   docker info | grep -E "Runtimes|Default Runtime"
   ```

   ```bash
   docker run --rm hello-world
   ```

   ```bash
   docker run --rm --gpus all ubuntu:24.04 bash -lc 'echo "NVIDIA runtime OK"'
   ```

   If `nvidia` is not the default runtime, correct `/etc/docker/daemon.json`,
   restart Docker, and repeat these checks before continuing.

10. Verify that the container can access the GPU:

   ```bash
   docker run --rm --gpus all ubuntu:24.04 nvidia-smi
   ```

   Successful output identifies NVIDIA Thor and reports no runtime-access
   error. Complete the Jetson AGX Thor Docker setup before continuing if this
   check fails.

11. Select Docker mode:

   ```bash
   sudo isaac-ros init docker
   ```

12. Enable real-time scheduling support:

   ```bash
   echo '--ulimit rtprio=99' > ~/.isaac_ros_dev-dockerargs
   ```

   This allows the ROS 2 controller manager to run on a real-time thread.

## Configure RealSense Support

This workflow uses the RealSense camera mounted in the Unitree G1's head. Connect it
directly to Thor when using the Thor backpack; use a USB extension cable when
Thor is external. The Ethernet connection between Thor and the Unitree G1 remains
direct. See [Unitree G1 Introduction and Safety](g1-introduction-and-safety.md#robot-setup).

1. Start with the camera unplugged and install the RealSense udev rules:

   ```bash
   wget https://raw.githubusercontent.com/realsenseai/librealsense/v2.56.3/config/99-realsense-libusb.rules
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

   Confirm that each command exits without an error, then connect the camera.

2. Configure the Isaac ROS Docker image with Physical AI workflow and
   RealSense support:

   ```bash
   mkdir -p ~/.config/isaac-ros-cli
   ```

   ```bash
   tee ~/.config/isaac-ros-cli/config.yaml << 'EOF'
   docker:
     image:
       additional_image_keys:
         - gr00t_workflow
         - realsense
   EOF
   ```

   The `realsense` layer supplies camera drivers. The `gr00t_workflow` layer
   supplies the ROS packages required by this workflow.

3. Build and activate the environment. Image keys select features; they do not
   lock a container digest or all package versions:

   ```bash
   isaac-ros activate --build-local
   ```

4. Verify camera detection:

   ```bash
   rs-enumerate-devices
   ```

5. In a graphical environment, optionally inspect the camera:

   ```bash
   realsense-viewer
   ```

   If the camera is not detected, recheck its USB connection, host udev rules,
   container configuration, and device enumeration before continuing. For the
   `rs-enumerate-devices`, udev, firmware, and streaming failures this workflow
   hits, see [Troubleshooting](../reference/troubleshooting.md#realsense-camera-not-found).

(what-happens-next)=
## Next Stage

The camera enumeration must identify the connected RealSense. Record the
source commits, installed package versions, and resulting container image
digest with the hardware qualification record. The source repositories are in
`${ISAAC_ROS_WS}/src`. The
`gr00t_workflow` layer includes:

- `ros-$ROS_DISTRO-isaac-ros-unitree-g1-teleop-bringup`
- `ros-$ROS_DISTRO-isaac-ros-unitree-g1-recorder`
- `ros-$ROS_DISTRO-isaac-ros-unitree-g1-gr00t`

Continue to [Unitree G1 Teleoperation](real-teleop.md).

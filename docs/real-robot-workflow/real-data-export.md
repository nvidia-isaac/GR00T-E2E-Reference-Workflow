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

(data-export-mcap-to-lerobot)=
(unitree-g1-data-export)=
# Unitree G1 Dataset Conversion

Dataset conversion turns recorded MCAP sessions into one LeRobot dataset for
GR00T N1.7 fine-tuning.

## Prerequisites

- Retain the complete session directories from [Data Recording](real-record.md).
- Run on an x86_64 workstation or Thor. The converter requires neither a GPU
  nor ROS, and can run on the host or inside the Isaac ROS container.
- Set `ISAAC_ROS_WS` to the workspace on this machine. On a workstation, copy
  the recordings into its `recordings/` directory and clone the pinned
  `isaac_ros_data_tools` source from
  [Isaac ROS Setup](isaac-ros-setup.md#create-the-isaac-ros-workspace).

## Choose the Dataset Directory

Set the output path once in the terminal used below. Choose a directory that
does not exist; change `lerobot_output` if that name is already in use.

```bash
DATASET_DIR="${ISAAC_ROS_WS}/recordings/lerobot_output"
```

## Use the Published Dataset (Optional)

To skip recording and conversion, download the published
[`nvidia/GR00T-N1.7-AppleToPlate`](https://huggingface.co/datasets/nvidia/GR00T-N1.7-AppleToPlate)
LeRobot dataset. Install the Hugging Face CLI and authenticate with an account
that has access to the dataset before downloading:

```bash
hf download \
    nvidia/GR00T-N1.7-AppleToPlate \
    --repo-type dataset \
    --local-dir "${DATASET_DIR}"
```

Then continue to [Validate Output for Fine-Tuning](#validate-output-for-fine-tuning).

<a id="install-the-converter"></a>

## Install the Converter With uv

Install `uv` using the [host tools instructions](../installation.md#host-tools)
before continuing.

1. Enter the converter source directory, then create and activate its environment
   in the same terminal:

   ```bash
   cd ${ISAAC_ROS_WS}/src/isaac_ros_data_tools/isaac_ros_mcap_lerobot_converter
   ```

   ```bash
   uv venv --python 3.12 .venv
   ```

   ```bash
   source .venv/bin/activate
   ```

   Install the converter in that environment:

   ```bash
   uv pip install -e .
   ```

## Convert a Recording Session

1. Run the following script to convert the rosbags of your recording sessions.
   Replace `<session_name>` with the name of your recording session.

   To combine multiple recording sessions, pass multiple `--rosbags-dir`
   arguments to convert all sessions into a single LeRobot dataset. Episodes
   are numbered sequentially across sessions in the order listed.

   ```bash
   mcap-to-lerobot \
       --rosbags-dir ${ISAAC_ROS_WS}/recordings/<session_name> \
       --output-dir "${DATASET_DIR}" \
       --task "move the apple to the plate" \
       --fps 30 \
       --robot-type unitree_g1
   ```

   :::{seealso}
   Refer to the [isaac_ros_mcap_lerobot_converter reference](https://nvidia-isaac-ros.github.io/v/release-4.5/repositories_and_packages/isaac_ros_data_tools/isaac_ros_mcap_lerobot_converter/index.html) for full CLI details and `--fps`/`sync_rate` guidance.
   :::

## Validate Output for Fine-Tuning

1. Check the dataset directory and metadata before training:

   ```bash
   ls "${DATASET_DIR}"
   ```

2. Confirm the output is ready:

   - [ ] Output directory contains four folders: `data/`, `meta/`, `videos/`, and `images/`.
   - [ ] Output directory contains `new_embodiment_config_defaults.py` for GR00T fine-tuning.
   - [ ] `data/`, `meta/`, and `videos/` contain data.
   - [ ] `images/` exists but is empty.
   - [ ] Episode count aligns with source sessions.
   - [ ] `fps` in metadata matches the recorder `sync_rate` (30 Hz by default).

<a id="key-takeaways"></a>

## Next Stage

The verified LeRobot dataset and `new_embodiment_config_defaults.py` are the
inputs to [Unitree G1 Policy Training](real-fine-tuning-and-leapp.md). Use the path
stored in `DATASET_DIR` as `<dataset-dir>` in training and LEAPP export. If
training runs on another machine, transfer the complete dataset and use its
destination path instead.

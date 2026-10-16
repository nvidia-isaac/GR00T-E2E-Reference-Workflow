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

(leapp-export)=
# Unitree G1 LEAPP Export

LEAPP export packages the fine-tuned G1 policy for deployment from C++ and
connects its sensor inputs and command outputs through policy metadata.

Run this stage on an x86_64 workstation. LEAPP export for GR00T is not
supported on Jetson AGX Thor.

## Prerequisites

- Use `<checkpoint-dir>` from [fine-tuning](real-fine-tuning-and-leapp.md) and
  the same training dataset as `<dataset-dir>`. Both must be local to the
  export workstation.
- Use the `gr00t_workflow_0.1` export repository for this physical G1 path.

## Prepare the Export Environment

1. If not already completed in [fine-tuning](real-fine-tuning-and-leapp.md), clone and set up the export repository:

   ```bash
   git clone --branch gr00t_workflow_0.1 --recurse-submodules https://github.com/nvidia-isaac/gr00t-leapp-export.git
   ```

   Enter the repository and install its environment in the same terminal:

   ```bash
   cd gr00t-leapp-export
   ```

   ```bash
   uv sync --python 3.10
   ```

## Run Export

:::{important}
For Unitree G1 fine-tunes, always pass `--joint_config export/data/g1_joints.json`.
Without this argument, export uses placeholder joint names and deployment joint mapping can fail or produce incorrect actions.
:::

1. Run the export from the `gr00t-leapp-export` repository:

   ```bash
   uv run python export/export_with_leapp.py \
       --model_path <checkpoint-dir> \
       --embodiment_tag new_embodiment \
       --dataset_path <dataset-dir> \
       --joint_config export/data/g1_joints.json \
       --output_name <export-name>
   ```

   You can set the arguments as follows:

   - `--model_path`: Path to `checkpoint-NNNN/` from fine-tuning.
   - `--embodiment_tag`: Always `new_embodiment` since the Unitree G1 is a new
      embodiment that is not supported by GR00T out of the box.
   - `--dataset_path`: Path to the LeRobot dataset you used for fine-tuning.
   - `--joint_config`: Path to the joint configuration of the Unitree G1 embodiment.
   - `--output_name`: Directory name for exported LEAPP bundle.

Tracing typically takes 5 to 15 minutes on a workstation GPU.
Successful completion ends with `Export completed: <export-name>`.

## Exported LEAPP Outputs

After export, the LEAPP bundle contains the following files:

- `<export-name>.yaml` — YAML file containing all policy metadata
- `<export-name>.png` — Network graph visualization of the policy
- `log.txt` — Logs from the LEAPP export

The bundle also includes the following ONNX files containing the policy weights:

- `backbone.onnx` and `backbone.onnx.data`
- `action_head.onnx` and `action_head.onnx.data`
- `preprocess_state.onnx`
- `preprocess_video.onnx`
- `decode_action.onnx`

The [deployment stack](real-deployment.md) consumes all of these files to run the policy and connect it to the correct sensor inputs and hardware outputs.

## Next Stage

Confirm the export completion message and that the bundle contains its YAML
and all referenced ONNX files. Copy the complete bundle to Thor under
`${ISAAC_ROS_WS}/policies/<export-name>/` for
[Unitree G1 Policy Deployment and Evaluation](real-deployment.md#use-your-own-policy).
The deployment YAML must remain beside the exported weights.

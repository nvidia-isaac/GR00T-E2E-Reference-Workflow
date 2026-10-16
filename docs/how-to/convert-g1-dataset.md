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

(g1-convert)=
# G1 Dataset Conversion

Convert recorded or generated Unitree G1 apple-to-plate demonstrations into
LeRobot format for GR00T N1.7 training.

:::{tip}
To use the provided training-ready dataset, skip this module and continue to
[G1 policy training](fine-tune-gr00t.md#use-a-published-dataset).
The next module includes the download command and training instructions.
:::

## Prerequisites

- Complete [Installation](../installation.md).
- Prepare replayed and inspected [demonstrations recorded with G1](record-demonstrations.md) or
  validated output from [G1 data generation](generate-data-with-autodata.md).
- Use data recorded with Pink IK and AGILE in either Isaac Lab-Arena or native
  Isaac Lab, including camera video, navigation commands, and base-height commands. The supplied policy training recipe requires
  these fields. Native fixed-base data omits the navigation and base-height
  commands and needs custom training and evaluation integration.

Run commands from this repository's root on the workstation. Choose the
conversion section matching your input and an output directory that does not
already exist. If the dataset is already converted, continue to
[Validate the Training Input](#validate-the-training-input).

## Convert Recorded Demonstrations

Use this section to convert your own recording from the earlier recording module.

:::{tip}
To train with provided
data, use the [published LeRobot dataset](fine-tune-gr00t.md#use-a-published-dataset)
instead.
:::

1. Set `G1_RECORDING` in the terminal used for conversion. This example selects
   your recording from Isaac Lab-Arena with Pink IK and AGILE. For native Isaac
   Lab with AGILE, use
   `outputs/g1/recording-native.hdf5` instead.

   ```bash
   G1_RECORDING="$PWD/outputs/g1/recording-arena.hdf5"
   ```

### Convert the Selected Recording

1. Convert your recording selected above:

   ```bash
   uv run --extra converter python scripts/convert_hdf5_to_lerobot.py \
     --input "$G1_RECORDING" \
     --output outputs/g1/lerobot
   ```

## Convert Generated Demonstrations

Choose either your own generated dataset below or [published generated data](#use-published-generated-data).
For native output, continue directly to [Convert the Selected File](#convert-the-selected-file).

1. For your own Arena output, set `G1_GENERATED` in the terminal used for conversion,
   then continue to [Convert the Selected File](#convert-the-selected-file):

   ```bash
   G1_GENERATED="$PWD/outputs/g1/generated-arena.hdf5"
   ```

### Use Published Generated Data

Alternatively, download the published dataset of 2,500 generated Arena episodes.
The HDF5 file is 79.2 GB. Allow additional space for the converted dataset.

1. Complete [Hugging Face access setup](../reference/models-and-datasets.md#access-and-storage),
   then download the generated dataset:

   ```bash
   uv run --frozen --no-sync hf download nvidia/GR00T-E2E-Reference-Workflow-SIM-G1-N1.7-AppletoPlate-Dataset \
     e2e_ra_g1_apple_to_plate_2500_generated_demos.hdf5 \
     --repo-type dataset --revision 36234f4b48dad27f218d8e29ec7710ad5a90b48c \
     --local-dir outputs/g1/downloads
   ```

2. Select the downloaded dataset for conversion:

   ```bash
   G1_GENERATED="$PWD/outputs/g1/downloads/e2e_ra_g1_apple_to_plate_2500_generated_demos.hdf5"
   ```

The HDF5 file includes camera frames and metadata describing its generation
settings and source recording. Conversion preserves this metadata. No separate
recording is needed for conversion.

### Convert the Selected File

The `--lane` option checks that the input is marked as generated data from the
selected workflow. The converter saves the original metadata and the HDF5 file's checksum
in `meta/e2e_workflow_lineage.json`.

1. Run one of the following commands, matching how the data was generated.

   For Isaac Lab-Arena output:

   ```bash
   uv run --extra converter python scripts/convert_hdf5_to_lerobot.py \
     --lane arena --input "$G1_GENERATED" \
     --output outputs/g1/lerobot-generated
   ```

   For native output:

   ```bash
   uv run --extra converter python scripts/convert_hdf5_to_lerobot.py \
     --lane native --input outputs/g1/generated-native.hdf5 \
     --output outputs/g1/lerobot-generated
   ```

Native output generated with `--no-cameras` cannot be converted for training.
Arena generation requires cameras.

## Validate the Training Input

1. Check the output directory for `data/`, `videos/`, and `meta/`, including
   `meta/info.json` and `meta/modality.json`. The converter writes the episode
   data and camera video; GR00T generates `meta/stats.json` on its first
   training run.

2. Confirm that `navigate_command` and `base_height_command` appear under
   `action` in `meta/modality.json`. Conversion can succeed without those
   groups, but the supplied policy training recipe requires them.

To inspect another input before writing, use its conversion command with
`--inspect --strict` and omit `--output`. It prints the resolved fields and
fails when a required field or mapping is missing.

## Expected Result

`outputs/g1/lerobot/` contains the converted recording, or
`outputs/g1/lerobot-generated/` contains the generated dataset. Keep the complete
directory for training.

## Next Stage

Continue to [G1 Policy Training](fine-tune-gr00t.md). Set `TRAIN_DATASET` to the
absolute path of the converted directory when preparing the training terminal.

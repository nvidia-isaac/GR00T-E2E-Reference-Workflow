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

(h2-convert)=
# H2 Dataset Conversion

Convert recorded or generated Unitree H2 demonstrations into LeRobot format,
then validate every episode through GR00T's training loader and video decoder.

:::{tip}
To skip conversion, continue to [H2 policy training](fine-tune-h2.md#use-a-published-dataset).
That step includes instructions for downloading our reference dataset.
:::

## Prerequisites

Complete [Installation](../installation.md) and
[Prepare the GR00T Runtime](prepare-h2.md#prepare-the-gr00t-runtime). In each new
terminal, restore `ISAAC_GR00T_ROOT` to that checkout. Run commands from the
repository root. Published downloads can proceed directly to conversion without
simulation assets, a headset, or a CloudXR session.

## Choose the Training HDF5

Choose one source below. Keep its file path and episode count together for
conversion and loader validation.

| Input | Episodes | Use It When |
| --- | --- | --- |
| [Published recording](#use-a-published-recording) | 10 | You want the smaller teleoperated input for conversion. |
| [Published generated demonstrations](#use-published-generated-demonstrations) | 17 | You want the larger AutoData-expanded input for conversion. |
| [Your own HDF5](#use-your-own-hdf5) | Your run's accepted count | You are continuing your recording or generation run. |

Both published HDF5 files are conversion-only. To generate new demonstrations,
use a compatible recording from the current workflow, as described in
[H2 Data Generation](generate-h2-demonstrations.md).

### Use a Published Recording

Download ten recorded H2 demonstrations (about 1.18 GB) for conversion to
LeRobot. Complete [Hugging Face access setup](../reference/models-and-datasets.md#access-and-storage)
before downloading. Run these commands from the repository root.

1. Create a new run directory for the downloaded input and subsequent outputs:

   ```bash
   H2_RUN_DIR="$PWD/outputs/h2/$(date +%Y%m%d_%H%M%S)"
   ```

   ```bash
   mkdir -p "$H2_RUN_DIR"
   ```

2. Download the recording:

   ```bash
   uv run --frozen --no-sync hf download nvidia/GR00T-E2E-Reference-Workflow-SIM-H2-N1.7-AppleHandoverPlate-Dataset \
     seed/raw/source-10.hdf5 \
     --revision 72ad851a5d0b845d368f031c744876613fb0b3c7 \
     --repo-type dataset \
     --local-dir "$H2_RUN_DIR/downloads"
   ```

3. After the download succeeds, select the recording and its episode count:

   ```bash
   H2_RECORDING="$H2_RUN_DIR/downloads/seed/raw/source-10.hdf5"
   H2_SOURCE_DEMOS=10
   H2_TRAINING_HDF5="$H2_RECORDING"
   H2_TRAINING_DEMOS="$H2_SOURCE_DEMOS"
   ```

4. Continue directly to [Convert the Selected HDF5](#convert-the-selected-hdf5).
   This file was recorded with earlier simulator and task settings and is
   provided for conversion only. Preserve its embedded metadata, which identifies
   those settings. To generate new demonstrations, use your own current recording
   as described in [H2 Data Generation](generate-h2-demonstrations.md).

### Use Published Generated Demonstrations

Download 17 generated H2 demonstrations (about 1.89 GB) to convert to LeRobot. Complete
[Hugging Face access setup](../reference/models-and-datasets.md#access-and-storage)
before downloading. Run these commands from the repository root.

1. Create a new run directory for the downloaded input and subsequent outputs:

   ```bash
   H2_RUN_DIR="$PWD/outputs/h2/$(date +%Y%m%d_%H%M%S)"
   ```

   ```bash
   mkdir -p "$H2_RUN_DIR"
   ```

2. Download the generated demonstrations:

   ```bash
   uv run --frozen --no-sync hf download nvidia/GR00T-E2E-Reference-Workflow-SIM-H2-N1.7-AppleHandoverPlate-Dataset \
     generated/raw/generated-017.hdf5 \
     --revision 72ad851a5d0b845d368f031c744876613fb0b3c7 \
     --repo-type dataset \
     --local-dir "$H2_RUN_DIR/downloads"
   ```

3. After the download succeeds, select the generated file and its episode count:

   ```bash
   H2_GENERATED="$H2_RUN_DIR/downloads/generated/raw/generated-017.hdf5"
   H2_TRAINING_HDF5="$H2_GENERATED"
   H2_TRAINING_DEMOS=17
   ```

4. Continue directly to [Convert the Selected HDF5](#convert-the-selected-hdf5).
   This file was generated with earlier simulator and task settings and is
   provided for conversion only. Preserve its embedded metadata, which identifies
   those settings and the source recording used for generation.

### Use Your Own HDF5

1. Select the training HDF5 and episode count in
   [H2 Demonstration Recording](record-h2-demonstrations.md) or optional
   [H2 Data Generation](generate-h2-demonstrations.md). Keep that run's
   `H2_TRAINING_HDF5`, `H2_TRAINING_DEMOS`, and `H2_RUN_DIR` values.
   If you open a new terminal, restore those values to the existing run before
   continuing.

2. Complete [H2 Asset Preparation](prepare-h2.md) and restore the simulation and
   H2 asset exports listed under [For New Terminals](prepare-h2.md#for-new-terminals).

3. Replay the saved robot actions in the simulator:

   ```bash
   ./tools/launch_replay.sh \
     --lane arena --viz none --strict --device cuda:0 --replay-mode action \
     --expected_episodes "$H2_TRAINING_DEMOS" \
     --dataset_file "$H2_TRAINING_HDF5"
   ```

   Action replay starts each episode from its recorded initial state and runs
   the saved actions through H2's robot-control software and simulated physics.
   It does not use headset input. The command fails on incomplete replay,
   missing required recording data, or incompatible robot settings.
   See [replay semantics](../explanation/h2-architecture.md#recording-and-replay)
   for the distinction from the recording stage's state replay.

4. After replay passes, continue to conversion below.

## Convert the Selected HDF5

Use these steps for either published download or your replayed current data.

1. Set a new output directory for the converted dataset:

   ```bash
   H2_LEROBOT="$H2_RUN_DIR/lerobot"
   ```

2. Convert the selected HDF5:

   ```bash
   uv run --frozen --extra converter python scripts/convert_hdf5_to_lerobot.py \
     --config "$PWD/configs/stages/gr00t_h2_lerobot_dataset.yaml" \
     --strict \
     --input "$H2_TRAINING_HDF5" \
     --output "$H2_LEROBOT"
   ```

   The H2 configuration rejects data from an incompatible robot or recording
   method. Omitting `--lane` permits either recorded or generated Arena episodes.

## Validate the Training Input

1. In the same terminal, generate GR00T statistics with the H2 modality:

   ```bash
   H2_MODALITY="$PWD/src/e2e_robot_workflows/gr00t/h2_static_apple_modality_config.py"
   ```

   ```bash
   ./tools/gr00t_runtime.sh --gr00t-root "$ISAAC_GR00T_ROOT" -- \
     python "$ISAAC_GR00T_ROOT/gr00t/data/stats.py" \
     --dataset-path "$H2_LEROBOT" \
     --embodiment-tag NEW_EMBODIMENT \
     --modality-config-path "$H2_MODALITY"
   ```

2. Validate the data consumed by training:

   ```bash
   ./tools/gr00t_runtime.sh --gr00t-root "$ISAAC_GR00T_ROOT" -- \
     python "$PWD/tools/check_h2_gr00t_dataset.py" \
     --dataset "$H2_LEROBOT" \
     --modality-config "$H2_MODALITY" \
     --expected-episodes "$H2_TRAINING_DEMOS" \
     --result "$H2_RUN_DIR/gr00t-loader.json"
   ```

   The check enforces the episode total, derives the frame total from LeRobot
   metadata, and decodes every video frame. It verifies the 75-D state and
   61-D learned action, then exits non-zero on any mismatch.

:::{note}
LeRobot stores the full 75-D joint target. The H2 GR00T modality selects 58
upper-body joint targets plus three navigation values. The G1 modality is
incompatible with this dataset.
:::

## Expected Result

`$H2_RUN_DIR/lerobot` contains the dataset and statistics.
`$H2_RUN_DIR/gr00t-loader.json` records successful loading and video validation.
Keep `H2_LEROBOT`, `H2_MODALITY`, and `H2_RUN_DIR` for training.

## Next Stage

Continue to [H2 Policy Training](fine-tune-h2.md).

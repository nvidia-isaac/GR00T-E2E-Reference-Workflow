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

(amplify-demonstrations-with-isaac-autodata)=
# G1 Data Generation With Isaac AutoData

Isaac AutoData expands recorded Unitree G1 apple-to-plate demonstrations by
replaying adapted trajectories in randomized scenes. This stage is optional.
Recordings made with Pink IK and AGILE in either Isaac Lab-Arena or native
Isaac Lab can also
[go directly to dataset conversion](convert-g1-dataset.md).
For Unitree H2 Plus, use {ref}`H2 data generation <h2-generate>`.

## Requirements

- Complete [Installation](../installation.md) on the GPU workstation.
- Use replayed and inspected [demonstrations recorded with G1](record-demonstrations.md)
  compatible with this checkout. Keep the original HDF5 file unchanged.
- Use a Linux GPU workstation and run commands from this repository's root.
- Prepare a clean AutoData checkout at the revision in `runtime.toml`.
  Native commands and `validate-dataset` also need AutoData's interpreter.
  Follow [First-Time Provisioning](#first-time-provisioning) if either is missing.

## Select an Isaac Lab-Arena Recording

1. Select your accepted recording in the terminal used for generation:

   ```bash
   G1_RECORDING="$PWD/outputs/g1/recording-arena.hdf5"
   ```

For native Isaac Lab recordings, use the native recording paths shown below instead.

Published HDF5 files were recorded with different simulator and task settings
and are provided for conversion. To skip recording and generation, use the
[generated HDF5 download](convert-g1-dataset.md#use-published-generated-data).

## Choose the Steps for Your Recording Method

Follow the steps for the method you used to record your demonstrations:

| Recording | Source HDF5 | Sequence |
| --- | --- | --- |
| Isaac Lab-Arena with Pink IK and AGILE | `outputs/g1/recording-arena.hdf5` | Generate → validate |
| Native Isaac Lab with a fixed base or AGILE | `outputs/g1/recording-native.hdf5` | Annotate → validate → generate → validate |

The commands call this recording method a **lane**: `--lane arena` selects
Isaac Lab-Arena, and `--lane native` selects native Isaac Lab.

The metadata embedded in the recording identifies its action layout and
manipulating hand. The command fails before simulation starts if the selected
recording method or robot-control configuration conflicts with that metadata.
Isaac Lab-Arena recordings made with Pink IK already contain subtask boundaries
and must not be annotated. Arena joint-control
recordings are for recording and replay, not AutoData.

Both recording methods support the supplied policy training and evaluation
configuration when they use AGILE. Native fixed-base data omits its required navigation and
base-height fields and needs custom training and evaluation integration.

### First-Time Provisioning

1. With Git and Conda on `PATH`, provision the pinned source and interpreter:

   ```bash
   ./tools/launch_autodata.sh provision
   ```

   The launcher sets `PYTHONNOUSERSITE=1` for all commands to prevent user-site
   packages from shadowing the environment.

   The default checkout is the sibling `Isaac-AutoData-e2e` and the Conda
   environment is `isaac_autodata`. Use `--autodata-root` or
   `ISAAC_AUTODATA_ROOT` for another checkout; use `--autodata-python` or
   `AUTODATA_PYTHON` for another interpreter. Provisioning includes the pinned
   extras required by native operations.

2. Check the prepared runtime:

   ```bash
   ./tools/launch_autodata.sh --check
   ```

   For Arena generation, also check its runtime:

   ```bash
   ./tools/launch_autodata.sh generate --lane arena --check
   ```

   The checks print the pinned checkout and interpreter. Arena generation uses
   this repository's locked Arena runtime. It does not use the AutoData
   interpreter for simulation.

(update-an-existing-interpreter)=
:::{tip}
For an existing pinned checkout and interpreter, run
`./tools/launch_autodata.sh setup` to refresh the extras. Use the same checkout
and interpreter overrides as the existing installation.
:::

## Isaac Lab-Arena Recording Generation

1. Generate directly from the selected Isaac Lab-Arena recording made with
   Pink IK. This requests one success within three attempts and a seven-minute
   process deadline:

   ```bash
   ./tools/launch_autodata.sh generate --lane arena \
     --generation-run-timeout-seconds 420 \
     --input_file "$G1_RECORDING" \
     --output_file outputs/g1/generated-arena.hdf5 \
     --result_file outputs/g1/generation-arena.json \
     -- --device cuda:0 --requested_successes 1 --max_attempts 3
   ```

   The input must contain successful episodes, subtask boundaries, and
   saved software versions and task settings that match this checkout. The
   recording metadata selects the left- or right-hand task description and
   retains the source commit.

2. Validate the generated file:

   ```bash
   ./tools/launch_autodata.sh validate-dataset outputs/g1/generated-arena.hdf5
   ```

   A completed run writes `outputs/g1/generated-arena.hdf5` and reports
   `status: ok` in `outputs/g1/generation-arena.json`, with attempt counts and
   artifact hashes. An exhausted attempt budget or deadline is a failed run,
   even if partial artifacts remain.

(amplify-your-own-recordings)=
## Native Recording Generation

1. Annotate your native recording to find where pickup ends and transport
   begins. `--wbc-mode auto` reads fixed or AGILE mode from the manifest:

   ```bash
   ./tools/launch_autodata.sh annotate --lane native --wbc-mode auto \
     --device cpu \
     --input_file outputs/g1/recording-native.hdf5 \
     --output_file outputs/g1/annotated-native.hdf5
   ```

   Only episodes satisfying the task's success condition are exported.
   An empty annotated file means no source episode passed that condition.
   For manual boundary annotation, add `--manual`: **N** plays, **B** pauses,
   **S** marks the boundary, and **Q** skips an episode.

2. Validate the annotated pool:

   ```bash
   ./tools/launch_autodata.sh validate-dataset outputs/g1/annotated-native.hdf5
   ```

3. Generate ten successful episodes, with a seven-minute process deadline:

   ```bash
   ./tools/launch_autodata.sh generate --lane native --wbc-mode auto \
     --generation-run-timeout-seconds 420 \
     --viz none --device cuda:0 --num_envs 5 --generation_num_trials 10 \
     --input_file outputs/g1/annotated-native.hdf5 \
     --output_file outputs/g1/generated-native.hdf5 \
     --result_file outputs/g1/generation-native.json
   ```

   This task retries failed attempts until ten successes or the deadline.
   Failed attempts are not exported. CUDA is recommended for AGILE generation;
   CPU supports small checks but runs substantially slower.

4. Validate the generated file:

   ```bash
   ./tools/launch_autodata.sh validate-dataset outputs/g1/generated-native.hdf5
   ```

   A completed run writes `outputs/g1/generated-native.hdf5` and reports
   `status: ok` in `outputs/g1/generation-native.json`. A timeout or failed run
   retains an error result. Investigate it before using the output for training.
   Continue to [G1 Dataset Conversion](convert-g1-dataset.md#convert-generated-demonstrations).

:::{tip}
For native trajectory inspection only, `generate --no-cameras` skips video.
That output cannot be converted for training. Arena generation requires cameras
and rejects `--no-cameras`.
:::

(convert-generated-data)=
## Next Stage

For data generated from either AGILE recording method above, continue to
[G1 Dataset Conversion](convert-g1-dataset.md#convert-generated-demonstrations)
with the validated generated HDF5 file. Select the command for the method
you used to generate the data, so its source details are retained in the
training dataset.

## Supporting Guidance

(check-the-bridge-to-this-repositorys-task)=
(generate-from-autodatas-own-demonstrations)=
Native Isaac Lab integration requirements, simulator validation commands, and
the upstream example are in [Native AutoData Integration](../reference/development.md#native-autodata-integration).

(migrate-a-legacy-recording-only-when-needed)=
Only older recordings without `e2e_workflow_manifest` need
[Legacy Recording Migration](../reference/development.md#legacy-recording-migration).
Do not migrate current recordings or newly merged files to work around a
missing or incompatible manifest.

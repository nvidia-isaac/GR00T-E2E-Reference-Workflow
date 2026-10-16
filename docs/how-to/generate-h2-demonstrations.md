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

(h2-generate)=
# H2 Data Generation With Isaac AutoData (Optional)

Use Isaac AutoData to generate perturbed demonstrations from accepted episodes
recorded with Unitree H2 in simulation. Skip this stage to [convert your recorded episodes](convert-h2-dataset.md#use-your-own-hdf5)
or [download generated demonstrations for conversion](convert-h2-dataset.md#use-published-generated-demonstrations).

## Prerequisites

Complete [H2 Asset Preparation](prepare-h2.md) on the GPU workstation. In each
new terminal, restore the simulation and H2 asset exports listed under
[For New Terminals](prepare-h2.md#for-new-terminals). Run commands from the
repository root.

`conda` is needed only if the `provision` command must create the isolated
AutoData runtime. Generation uses one environment because the pinned H2 AGILE
policy supports batch size one.

## Prepare a Recording

1. Complete [H2 Demonstration Recording](record-h2-demonstrations.md), including
   strict state replay, using the current H2 workflow.
2. Keep that run's `H2_RUN_DIR` and `H2_RECORDING` values. If you open a new
   terminal, restore those values to the existing run before continuing.

The published H2 HDF5 downloads were recorded with different task settings and
are provided for conversion. Generation requires a recording whose saved
software versions and simulation settings match this checkout.

## Prepare Isaac AutoData

1. On a fresh setup, provision the pinned AutoData checkout. Skip this command
   if it is already provisioned:

   ```bash
   ./tools/launch_autodata.sh provision
   ```

2. Verify the Isaac Lab-Arena generation route:

   ```bash
   ./tools/launch_autodata.sh generate --lane arena --check
   ```

   The check must print the pinned AutoData revision and report the pinned
   Isaac Lab-Arena runtime as its interpreter.

## Generate Demonstrations

1. Generate from the recorded HDF5 file:

   ```bash
   H2_GENERATED="$H2_RUN_DIR/generated.hdf5"
   ```

   ```bash
   H2_TARGET_DEMOS=50
   ```

   Choose a wall-clock budget before starting. It includes simulator startup,
   all successful and failed attempts, and dataset finalization. Measure a
   small run on the same workstation and size the budget from its time per
   attempt multiplied by `--max_attempts`, with extra time for startup,
   finalization, and variation. One hour is not a completion guarantee for
   50 successes. The four-hour example below is a starting budget; increase it
   if your measured attempt rate requires more time.

   ```bash
   H2_GENERATION_TIMEOUT_SECONDS=14400
   ```

   ```bash
   ./tools/launch_autodata.sh generate --lane arena \
     --generation-run-timeout-seconds "$H2_GENERATION_TIMEOUT_SECONDS" \
     --input_file "$H2_RECORDING" \
     --output_file "$H2_GENERATED" \
     --result_file "$H2_RUN_DIR/generation.json" \
     -- --device cuda:0 \
        --requested_successes "$H2_TARGET_DEMOS" \
        --max_attempts 100
   ```

   Arena recordings already contain the boundary signals, so no annotation
   step is needed. Recordings from another checkout are accepted when their
   saved robot and task settings, software versions, action layout, and
   pickup/placement markers match. The source commit and uncommitted-change
   status are retained in the metadata.

2. Confirm the launcher exits successfully and prints `Generation complete:
   50/50 successful demonstrations`, followed by the attempt count and the
   dataset and result paths. This summary appears after the launcher validates
   attempt accounting, manifests, lineage, and matching artifact hashes, and
   publishes the finalized dataset. `generation.json` must report `status: ok`
   and `num_success: 50`.

   Reaching the attempt limit or timeout before 50 successes is a failed run.
   Files retained after a failed run may lack a finalized manifest and are not
   ready for conversion. Inspect `generation.json`, then rerun into a new output
   directory with a budget sized for the observed attempt rate.

3. After successful generation, select its output for conversion:

   ```bash
   H2_TRAINING_HDF5="$H2_GENERATED"
   ```

   ```bash
   H2_TRAINING_DEMOS="$H2_TARGET_DEMOS"
   ```

Measured joint-limit faults consume an attempt and discard its incomplete
recording before retrying. Other faults stop generation. For a separate run,
choose a new output directory and pass `--generation_seed` after `--` to select
a seed.

## Expected Result

`$H2_RUN_DIR/generated.hdf5` contains the requested accepted perturbations.
`generation.json` records the result and hashes. Completed failed perturbations
remain in the sibling `generated_failed.hdf5` artifact for inspection.

## Next Stage

Continue to [H2 Dataset Conversion](convert-h2-dataset.md) to replay and validate
the generated episodes before training.

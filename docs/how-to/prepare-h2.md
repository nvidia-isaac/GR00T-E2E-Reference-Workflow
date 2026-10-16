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

(h2-prepare)=
# H2 Asset Preparation

Prepare the Unitree H2 Plus simulation model, Pink kinematics, and AGILE policy
for recording and evaluation in Isaac Lab-Arena.

## Prerequisites

Complete [Installation](../installation.md). Run commands from the repository
root on the GPU workstation. Recording later requires a PICO 4 Ultra headset
on the workstation's LAN and its paired handheld XR controllers.

Obtain access to the restricted
[`nvidia/agile_wbc_unitree_h2`](https://huggingface.co/nvidia/agile_wbc_unitree_h2)
repository before downloading its policy. If the repository is unavailable to
your account, request access from its owner. Accepting public model terms alone
may not grant access.

## Prepare the Simulation Assets

1. Sync the repository with the converter used by H2 validation:

   ```bash
   uv sync --frozen --extra converter
   ```

2. Generate the content-pinned Pink sidecar in the Isaac runtime:

   :::{warning}
   Read the [NVIDIA Omniverse license terms](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/common/NVIDIA_Omniverse_License_Agreement.html)
   before running the following setup block. The EULA exports record your
   acceptance and suppress the runtime license prompt. Run them only if you
   accept the terms.
   :::

   ```bash
   export ACCEPT_EULA=Y
   export OMNI_KIT_ACCEPT_EULA=YES
   export E2E_H2_ASSET_CACHE="$HOME/.cache/e2e-reference-app/h2plus"
   ```

   ```bash
   ./tools/arena_launch.sh python tools/materialize_h2_pink_sidecar.py \
     --cache-root "$E2E_H2_ASSET_CACHE" \
     --regenerate
   ```

   Successful completion verifies the model and generated sidecars against their
   pinned digests.

3. In the command's final JSON output, check that `status` is `ok`. Copy the
   value of `pink_urdf`, which ends in `unitree_h2.urdf`, into the export below:

   ```bash
   export E2E_H2_PINK_URDF="/absolute/path/to/unitree_h2.urdf"
   ```

## Download the AGILE Policy

1. Authenticate with the Hugging Face account that has policy access:

   ```bash
   uv run --frozen --no-sync hf auth login
   ```

   Verify the active account:

   ```bash
   uv run --frozen --no-sync hf auth whoami
   ```

2. Download the repository, revision, and filename specified in
   `configs/robots/h2.yaml`, then verify the configured digest. Continue only
   after each command succeeds:

   ```bash
   read -r H2_AGILE_REPOSITORY H2_AGILE_REVISION H2_AGILE_FILENAME < <(
     uv run --frozen --no-sync python -c \
       'from e2e_robot_workflows.robots.h2 import load_h2_robot_config; c = load_h2_robot_config(); print(c.policy_repository, c.policy_revision, c.policy_filename)'
   )
   ```

   ```bash
   E2E_H2_AGILE_POLICY="$(
     uv run --frozen --no-sync hf download --quiet \
       "$H2_AGILE_REPOSITORY" "$H2_AGILE_FILENAME" \
       --revision "$H2_AGILE_REVISION"
   )"
   ```

   ```bash
   export E2E_H2_AGILE_POLICY
   ```

   ```bash
   uv run --frozen --no-sync python -c \
     'from e2e_robot_workflows.robots.h2 import load_h2_robot_config; print(load_h2_robot_config().resolve_policy())'
   ```

   Confirm the command prints the downloaded `agile_velocity.onnx` path. A wrong
   file fails its SHA-256 check.

## Prepare the GR00T Runtime

1. Follow {ref}`prepare-gr00t-checkout` to create a clean, standalone Isaac-GR00T
   checkout at the revision pinned in `runtime.toml`.

2. Set its absolute path and verify the runtime:

   ```bash
   export ISAAC_GR00T_ROOT=/absolute/path/to/Isaac-GR00T
   ```

   ```bash
   (cd "$ISAAC_GR00T_ROOT" && uv sync --frozen)
   ```

   After synchronization succeeds, check the runtime:

   ```bash
   ./tools/gr00t_runtime.sh --gr00t-root "$ISAAC_GR00T_ROOT" --check
   ```

   Confirm the command prints `TorchCodec 0.4.0: VideoDecoder`. H2 validation
   uses GR00T's training loader and TorchCodec to check converted video data.

## For New Terminals

Use the shared [Environment Variable Reference](../reference/environment-variables.md)
to restore this setup's asset paths, runtime settings, and run variables in each
new terminal. It covers simulation, CloudXR, AutoData, training, and evaluation.

## Expected Result

The model and sidecars are cached outside the repository. The AGILE policy
passes its digest check, and the pinned GR00T runtime passes its loader check.

## Next Stage

Continue to [H2 Simulation Validation](validate-h2.md).

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

# Environment Variable Reference

Environment variables select external runtimes, assets, and services for the
simulated G1, simulated H2, and physical G1 workflow paths. The tables also list
the shell variables that carry a run's files between stages.

## Starting a New Terminal

Restore the settings for the stage you are resuming on the machine where it
runs. Use the same asset, dataset, and checkpoint paths as the preceding stage.
For an existing H2 run, retain `H2_RUN_DIR`; creating a new timestamped directory
would disconnect the later commands from your recording.

Export variables read by a launched process with `export NAME=value`.
[Shell variables](#run-and-artifact-shell-variables) are expanded by the command
examples and need an assignment in each terminal that uses them, even when the
guide originally exported them. Shell state does not transfer to a new terminal,
another machine, or an already running container.

| Stage | Restore |
| --- | --- |
| G1 simulation and recording | [Simulation runtime](#simulation-runtime); CloudXR setup from [Installation](../installation.md#xr-recording-requirements) |
| H2 simulation, recording, replay, and generation | Simulation runtime and [H2 assets](#h2-assets); [CloudXR](#h2-cloudxr-server) in the separate server terminal |
| G1 or H2 optional data generation | [AutoData overrides](#isaac-autodata), if provisioning used non-default paths; the selected run's input/output variables |
| G1 or H2 dataset loading, training, and evaluation | [GR00T checkout](#simulation-runtime), matching [run variables](#run-and-artifact-shell-variables); H2 assets for H2 simulation/evaluation |
| Physical G1 teleoperation, recording, and deployment | [Isaac ROS workspace and activation](#physical-g1); source the CloudXR environment in each teleoperation terminal |

## Simulation Runtime

| Variable | Requirement and Value | Setup |
| --- | --- | --- |
| `ACCEPT_EULA`, `OMNI_KIT_ACCEPT_EULA` | For non-interactive Isaac Sim launch, `Y` and `YES` respectively, only after accepting the license terms. | [Installation](../installation.md#create-and-check-the-simulation-environment) |
| `E2E_ARENA_ROOT` | Optional override for the pinned Isaac Lab-Arena checkout. Unset uses the managed `.arena/` checkout under this repository. | [How Arena Is Obtained](development.md#how-arena-is-obtained) |
| `E2E_ARENA_ROOT_PATH` | Optional GitLab project variable for an Isaac Lab-Arena checkout provisioned on the runner. The Arena CI jobs map it to `E2E_ARENA_ROOT`; local launchers read `E2E_ARENA_ROOT` directly. | [How Arena Is Obtained](development.md#how-arena-is-obtained) |
| `ISAAC_GR00T_ROOT` | Absolute path to the clean, synced Isaac-GR00T checkout pinned in `runtime.toml`. Required by the documented training, loader-validation, and evaluation commands. Launchers also accept `--gr00t-root`. | {ref}`Prepare the GR00T Checkout <prepare-gr00t-checkout>` |

The EULA settings apply to simulation processes, including replay and
generation. They do not accept the separate CloudXR license.

## H2 Assets

Restore these in each terminal that starts H2 simulation. The source contracts
are `configs/robots/h2.yaml` and `configs/robots/h2_pink.yaml`.

| Variable | Requirement and Value | Setup |
| --- | --- | --- |
| `E2E_H2_ASSET_CACHE` | Asset cache root used during preparation. The runtime default is `$HOME/.cache/e2e-reference-app/h2plus`. Export it when using another location, and always for `sim-arena-h2`, even at the default path. | [Prepare the Simulation Assets](../how-to/prepare-h2.md#prepare-the-simulation-assets) |
| `E2E_H2_PINK_URDF` | Required for the Pink recording, replay, and generation embodiment. Use the `pink_urdf` path printed by the sidecar materializer. | [Prepare the Simulation Assets](../how-to/prepare-h2.md#prepare-the-simulation-assets) |
| `E2E_H2_AGILE_POLICY` | Required H2 AGILE ONNX file. Its digest must match the robot configuration; a filename alone is insufficient. | [Download the AGILE Policy](../how-to/prepare-h2.md#download-the-agile-policy) |

The H2 validation command exercises both Pink and joint embodiments and needs
all three settings plus `ISAAC_GR00T_ROOT`.

## H2 CloudXR Server

Set these in the CloudXR terminal, separate from the recording terminal.
[Start CloudXR](../how-to/record-h2-demonstrations.md#start-cloudxr) provides the
client download and launch commands.

| Variable | Requirement and Value |
| --- | --- |
| `TELEOP_WEB_CLIENT_STATIC_DIR` | Directory containing the downloaded pinned `index.html` and `bundle.js` for `--host-client`. Restore the same directory when restarting the server. |

## Isaac AutoData

These overrides are optional. Restore any that you used during
[provisioning](../how-to/generate-data-with-autodata.md#first-time-provisioning).
`tools/launch_autodata.sh` owns their defaults; explicit command-line options
take precedence where provided.

| Variable | Default and Use |
| --- | --- |
| `ISAAC_AUTODATA_ROOT` | The sibling `Isaac-AutoData-e2e` directory. Selects the pinned clean checkout; also settable with `--autodata-root`. |
| `AUTODATA_PYTHON` | Interpreter in the `isaac_autodata` Conda environment. Override with an absolute Python path or `--autodata-python`. Native operations and `validate-dataset` use this interpreter; Arena generation uses the repository's Arena runtime. |
| `AUTODATA_CONDA_ENV` | `isaac_autodata`. Selects the Conda environment when `AUTODATA_PYTHON` is unset. |
| `E2E_AUTODATA_OUTPUT_DIR` | `outputs/autodata/` under this repository. Used when output paths are not supplied explicitly. |
| `E2E_AUTODATA_GENERATION_TIMEOUT_SECONDS` | `420` seconds. Whole-process generation deadline, overridden by `--generation-run-timeout-seconds`; use the [H2 timeout guidance](../how-to/generate-h2-demonstrations.md) for larger runs. |

## Run and Artifact Shell Variables

These names belong to the guide's shell commands. The applications receive
their values through file paths and command-line arguments. Recreate them for
the existing run before copying commands from a later stage.

| Variable | Value to Restore | Guide |
| --- | --- | --- |
| `E2E_ROOT` | Absolute path to this repository, captured before changing into Isaac-GR00T. | [G1 training](../how-to/fine-tune-gr00t.md), [H2 training](../how-to/fine-tune-h2.md) |
| `ARENA_ROOT` | Output of `./tools/fetch_arena_checkout.sh`, passed as `--arena-root`. This shell name does not replace the `E2E_ARENA_ROOT` launcher override. | [G1 evaluation](../how-to/evaluate-checkpoint.md), [H2 evaluation](../how-to/evaluate-h2-checkpoint.md) |
| `TRAIN_DATASET`, `MODALITY_CONFIG` | G1 LeRobot dataset and G1 modality configuration from the training guide. | [G1 training](../how-to/fine-tune-gr00t.md) |
| `TRAIN_OUTPUT` | Training output directory for the selected G1 or H2 run. | [G1 training](../how-to/fine-tune-gr00t.md), [H2 training](../how-to/fine-tune-h2.md) |
| `G1_CHECKPOINT`, `H2_CHECKPOINT` | Selected published model directory or completed checkpoint directory from the matching robot's training run. G1 also supports the released checkpoint described in its evaluation guide. | [G1 evaluation](../how-to/evaluate-checkpoint.md), [H2 evaluation](../how-to/evaluate-h2-checkpoint.md) |
| `G1_CHECKPOINT_SOURCE`, `H2_CHECKPOINT_SOURCE` | `published` for the downloaded workflow model or `trained` for your own checkpoint. For the separate G1 released checkpoint, use `released`. Keep the source paired with the selected checkpoint path. | [G1 evaluation](../how-to/evaluate-checkpoint.md), [H2 evaluation](../how-to/evaluate-h2-checkpoint.md) |
| `H2_RUN_DIR` | Absolute path to the existing `outputs/h2/<timestamp>/` directory. Choose a new directory when starting training with the published dataset. | [H2 recording](../how-to/record-h2-demonstrations.md), [H2 training](../how-to/fine-tune-h2.md) |
| `H2_RECORDING`, `H2_SOURCE_DEMOS` | Recorded source HDF5 and its expected accepted episode count. | [H2 recording](../how-to/record-h2-demonstrations.md) |
| `H2_GENERATED`, `H2_TARGET_DEMOS` | Generated HDF5 and requested successful generation count. | [H2 generation](../how-to/generate-h2-demonstrations.md) |
| `H2_GENERATION_TIMEOUT_SECONDS` | Measured wall-clock budget passed as `--generation-run-timeout-seconds`. This shell value overrides the launcher's environment default through that argument. | [H2 generation](../how-to/generate-h2-demonstrations.md) |
| `H2_TRAINING_HDF5`, `H2_TRAINING_DEMOS` | Selected source or generated HDF5 and its expected episode count. Keep the path and count paired. For published-dataset loader validation, set `H2_TRAINING_DEMOS=1000`; no HDF5 input is needed. | [H2 conversion](../how-to/convert-h2-dataset.md) |
| `H2_LEROBOT`, `H2_MODALITY` | Converted or downloaded dataset directory and H2 modality configuration. | [H2 conversion](../how-to/convert-h2-dataset.md), [H2 training](../how-to/fine-tune-h2.md) |
| `H2_SMOKE_OUTPUT` | Separate output directory for the short training check. | [H2 training](../how-to/fine-tune-h2.md) |
| `H2_AGILE_REPOSITORY`, `H2_AGILE_REVISION`, `H2_AGILE_FILENAME` | Temporary download values read from `configs/robots/h2.yaml`. Re-read them for a download; later stages use `E2E_H2_AGILE_POLICY`. | [H2 policy download](../how-to/prepare-h2.md#download-the-agile-policy) |
| `DATASET_DIR` | Physical G1 LeRobot output directory on the machine doing conversion. Transfer the complete directory when training elsewhere. | [Physical G1 conversion](../real-robot-workflow/real-data-export.md) |

## Physical G1

The physical path uses the Isaac ROS environment on Thor. Its x86_64 conversion,
training, and export stages use the environments stated in their own guides.

| Setting | Requirement and Value | Setup |
| --- | --- | --- |
| `ISAAC_ROS_WS` | Absolute workspace path on the current machine. Setup persists `$HOME/workspaces/isaac_ros-dev/` in the host's shell configuration. Verify that the same workspace is visible inside the container. | [Create the Isaac ROS Workspace](../real-robot-workflow/isaac-ros-setup.md#create-the-isaac-ros-workspace) |
| `LANG`, `LC_ALL` | UTF-8 locale; setup persists `en_US.UTF-8` for both on the Thor host and exports `LANG` in the setup terminal. | [Initialize the Isaac ROS CLI](../real-robot-workflow/isaac-ros-setup.md#install-and-initialize-the-isaac-ros-cli) |
| Isaac ROS and `ROS_DISTRO` | Run `isaac-ros activate` in each new terminal used for ROS commands. Use its configured ROS environment. | [Physical G1 teleoperation](../real-robot-workflow/real-teleop.md#activate-isaac-ros) |
| CloudXR environment | With CloudXR running, source `~/.cloudxr/run/cloudxr.env` inside each Isaac ROS terminal used to launch teleoperation. Use the generated file rather than copying individual service variables. | [Initial Test in MuJoCo](../real-robot-workflow/real-teleop.md#initial-test-in-mujoco) |
| `CUDA_HOME`, `CUDA_VISIBLE_DEVICES`, `PYTORCH_CUDA_ALLOC_CONF` | The training example supplies `/usr/local/cuda`, GPU `0`, and `expandable_segments:True` for that command only. Match the toolkit path and GPU selection to the training workstation. | [Launch Training](../real-robot-workflow/real-fine-tuning-and-leapp.md#launch-training) |

The physical training and LEAPP export guides pass dataset, checkpoint, and
bundle paths explicitly. They do not use the simulation launchers' GR00T or H2
environment variables.

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

# Reference

Reference pages collect the exact sources of runtime, configuration, and
validation behavior.

Use the [Environment Variable Reference](environment-variables.md) to restore
runtime, asset, and run settings when opening a new terminal in any workflow.

## Configuration and Version References

| Concern | Authoritative source |
| --- | --- |
| Python dependencies | `uv.lock` |
| Tested external revisions and released checkpoint | `runtime.toml` |
| Robot model and robot-control defaults | `configs/robots/` |
| Task discovery | `configs/tasks/` |
| Dataset and policy stages | `configs/stages/` |
| Validation entry points | `tools/validate_repository.sh` |

## Supported Configurations and Limitations

- Simulator, Isaac Lab-Arena, and XR behavior has been validated on Linux x86_64.
- The application has Unitree G1 and Unitree H2 Plus adapters for the static-apple task.
  Unitree G1 retains built-in and Isaac Lab-Arena paths; Unitree H2 is Isaac Lab-Arena-only.
- Recording and checkpoint evaluation share the settled-placement predicate.
  Recording demonstrations with H2 additionally requires pickup and receiving-hand boundaries and
  follow-through; its evaluation placement metric does not establish those
  handover events. With automatic completion enabled, manual finish/reset ends
  an episode but does not override the policy label. On Unitree G1, passing
  `--no-auto_success` switches to operator-owned acceptance. The Isaac
  AutoData variant of the task adds a success *termination* and a subtask boundary
  signal, which score and annotate generated attempts rather than recordings.
- Isaac Lab-Arena's code is an external source checkout rather than a locked
  dependency, because its wheel ships no data files. It is pinned by revision in
  `runtime.toml`, enforced by `tools/check_arena_pin.py`, fetched by
  `tools/fetch_arena_checkout.sh` when `E2E_ARENA_ROOT` is unset, and overlaid on
  this environment by `tools/arena_launch.sh`. Its dependencies are locked here, in
  the `arena` dependency group, so Isaac Lab-Arena runs in one interpreter with the simulator
  versions this repository pins.
- Isaac AutoData uses its own checkout, pinned by revision in `runtime.toml`, and
  is reached only through `tools/launch_autodata.sh`. Native operations use
  AutoData's isolated interpreter; Isaac Lab-Arena generation overlays its source on the
  repository's Isaac Lab-Arena runtime.
- The HDF5-to-LeRobot command writes a GR00T LeRobot v2.1 dataset by default,
  including each workflow's recorded command fields: demonstrations recorded with Unitree G1
  using Pink IK and AGILE in Isaac Lab-Arena, or AGILE in native Isaac Lab,
  include navigation and base height, while demonstrations with Unitree H2 include
  navigation only. `--inspect` opts
  out and only reports whether the configured fields resolve. Two omissions are
  deliberate: `meta/stats.json` is generated downstream in the GR00T environment,
  and the writer drops modality groups a run does not produce. The supplied modality for policy training and evaluation with Unitree G1
  expects its locomotion commands. Both recording methods described above,
  including generated data, provide them.
  Demonstrations recorded with a fixed-base G1 omit them and need custom training and inference
  integration.
- The `static_apple_v0` environment preserves the workflow's recording scene. It does not
  claim equivalence with Isaac Lab-Arena's first-party Galileo scene or released-checkpoint
  performance. The Arena path uses this same scene as `e2e_static_apple`
  rather than reusing Isaac Lab-Arena's scene.
- AGILE inputs are pinned at their ownership seams. The built-in Unitree G1 task uses
  the TorchScript digest and Isaac Lab-Arena Unitree G1 uses the Isaac Lab-Arena policy revision recorded in
  `runtime.toml`. Unitree H2 uses the repository, revision, filename, and digest in
  `configs/robots/h2.yaml`.
- Simulator assets, model checkpoints, and AGILE policies remain external.
  Initial setup requires network access and sufficient local storage.

For setup, local builds, validation, and integration commands, use the
[Developer Guide](development.md).

For hardware, camera, streaming, and compute-unit issues on the physical Unitree G1
path, use [Troubleshooting](troubleshooting.md).

```{toctree}
:hidden:

environment-variables
models-and-datasets
troubleshooting
```

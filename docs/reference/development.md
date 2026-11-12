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

# Developer Guide

The developer guide separates the everyday edit-and-build loop from the strict
validation commands used before handoff.

::::{grid} 1
:gutter: 3

:::{grid-item-card} Build a Workflow Path
:link: ../developer/build-workflow
:link-type: doc
:class-card: workflow-card workflow-expansion-card

Add a workflow path while preserving the repository's environment, adapter, data,
and validation contracts.
:::

::::

```{toctree}
:hidden:

Build a Workflow Path <../developer/build-workflow>
```

## Development Environment

The locked runtime uses Python 3.12. Install the complete simulator,
development, and documentation environment from the repository root:

```bash
uv sync --frozen
```

For CPU-only static checks and dataset work:

```bash
uv sync --frozen --no-default-groups --extra dev --extra converter
```

`uv.lock` specifies the dependency set, `runtime.toml` records the tested external
software versions and model revisions, and `tools/validate_repository.sh` owns the validation
commands.

## Simulation Assets

The static-apple scenes load versioned NVIDIA Isaac Sim assets over public HTTPS.
Scene loading requires network access but no asset-server login. Unitree G1 and
Unitree H2 share the warehouse, table, apple, and plate assets. The G1 robot uses
its own public USD configured in `configs/robots/g1.yaml`.

Set `E2E_ASSET_ROOT` to use a collected library mirror instead of the public
scene assets. The mirror must retain these directories and entry points,
including their referenced resources:

- `background_library/warehouse/`: `warehouse.usd`
- `object_library/table/`: `table.usd`
- `object_library/apple/`: `apple.usd`
- `object_library/clay_plate/`: `clay_plate.usd`

These library-relative names are independent of the public filenames.
An unset or empty variable selects the public defaults.

The robot is configured separately: `E2E_ASSET_ROOT` does not change the G1 USD.
To select a different robot USD, set `E2E_G1_ROBOT_CONFIG` to a complete G1 YAML
configuration with that USD in its `usd` field.

The public G1 USD uses its default white materials. Public transport URLs and
`E2E_ASSET_ROOT` are excluded from the environment contract digest. Library-relative
asset paths (`usd_relpath`) remain included. Matching contracts therefore do not
prove matching visual appearance. Keep the recording manifest's
`repository.revision` and the selected robot and scene URLs with each dataset to
distinguish asset variants when preparing policy data.

## Documentation Development Workflow

The standard `uv sync --frozen` command installs the documentation tools. For
a smaller docs-only environment, run:

```bash
uv sync --frozen --no-default-groups --only-group docs
```

During normal editing, build the NVIDIA-themed HTML site directly:

```bash
uv run --no-sync sphinx-build -b html docs docs/_build/html
```

This command performs an incremental build. The generated site is written to
`docs/_build/html/`. Open `docs/_build/html/index.html` directly or serve it
locally:

```bash
uv run --no-sync python -m http.server \
  --directory docs/_build/html \
  8000
```

Open `http://127.0.0.1:8000/` in a browser.

### Documentation Style

Use the [guide to recording demonstrations on the physical G1](../real-robot-workflow/real-record.md)
as the structural example for workflow stages:

1. State the task and expected result in a short opening paragraph.
2. Identify prerequisites, execution host, repository, environment, and terminal.
3. Divide the procedure into named sections with numbered actions. Use bullets
   for alternatives, and make optional stages explicit.
4. Show the expected output and carry its path into the next stage.

Use Title Case headings and parallel stage names. Write direct instructions,
keep UI labels and keyboard keys **bold**, and format commands, paths, variables,
and literal output as `code`. Use E2E in titles and end-to-end in ordinary prose.
Keep implementation explanations in reference or architecture pages.

Put each independent command in its own copyable code block. Group related
environment exports or variable assignments that prepare one step. Keep
multiline commands, pipelines, and heredocs intact. State when to switch
terminals or environments; keep alternative commands and example output in
separate blocks.

| Callout | Purpose |
| --- | --- |
| Note | Brief context or behavior distinction |
| Tip | Optional improvement or shortcut |
| Important | An easily missed requirement or limitation beside the affected step |
| Warning | A concrete risk, including data loss or invalid output, before the action |
| Danger | A serious physical hazard, preserving established safety requirements |
| Dropdown | Conditional troubleshooting or extended background |
| See also | Supporting references, separate from next-stage navigation |

Keep required actions in the main procedure. Keep callouts short and place them
before the action they affect. Use MyST colon-fenced directives such as
`:::{note}`; for a custom title, use `:::{admonition} Title` with `:class: note`
or the appropriate severity. Verify headings, callouts, cards, images, and
navigation in the rendered page at desktop and mobile widths.


## Documentation Validation Workflow

Before handing off a documentation change, run the documentation checks:

```bash
uv run --no-sync ./tools/validate_repository.sh docs
```

The command rebuilds every page in a fresh Sphinx environment. Warnings fail the
build, and Sphinx checks internal and external links.


Complete the [repository validation](#repository-validation) checks before
handing off changes to documentation dependencies, tooling, or CI.

## Repository Validation

Run all static checks and CPU tests:

```bash
uv run --no-sync ./tools/validate_repository.sh host
```

The host validation command runs static checks and CPU tests without starting Isaac Sim.

On a compatible Linux x86_64 GPU host, validate the runtime and simulator
behaviors:

```bash
uv run ./tools/validate_repository.sh runtime-preflight
```

```bash
uv run ./tools/validate_repository.sh sim-static-apple
```

```bash
uv run ./tools/validate_repository.sh sim-agile-locomotion
```

```bash
uv run ./tools/validate_repository.sh sim-recorder
```

The Unitree G1 Isaac Lab-Arena modes need a GPU and fetch the pinned Isaac Lab-Arena checkout themselves, as
described under [how Arena is obtained](#how-arena-is-obtained):

```bash
uv run --no-sync ./tools/validate_repository.sh sim-common-environment
```

```bash
uv run --no-sync ./tools/validate_repository.sh sim-arena-recording
```

```bash
uv run --no-sync ./tools/validate_repository.sh sim-arena-locomotion
```

The Unitree H2 Isaac Lab-Arena mode needs the same GPU runtime and additionally runs GR00T
statistics and its actual sharded training loader. It refuses to start until
the Unitree H2 robot model inputs and a clean, synced GR00T checkout at the `runtime.toml`
revision are supplied:

```bash
export E2E_H2_ASSET_CACHE=<path>/h2plus
```

```bash
export E2E_H2_PINK_URDF=<path>/unitree_h2.urdf
```

```bash
export E2E_H2_AGILE_POLICY=<path>/agile_velocity.onnx
```

```bash
export ISAAC_GR00T_ROOT=<path>/Isaac-GR00T
```

Then run the H2 check in the same terminal:

```bash
uv run --no-sync ./tools/validate_repository.sh sim-arena-h2
```

See [Run the Unitree H2 Plus Arena workflow path](../workflows/h2-pick-handover-place.md).


### Manual Simulator Checks

The three built-in-task modes remain supported manual
checks and provide the comparison path described in
[environment ownership](#environment-ownership).

Run `sim-arena-h2` manually with the AGILE checkpoint and materialized Pink sidecar.

## Shared-Environment Integration

The managed Unitree G1 Isaac Lab-Arena checks run in this repository's environment, with the
pinned Isaac Lab-Arena checkout overlaid on the interpreter path. The launcher obtains
that checkout automatically:

```bash
uv run --no-sync ./tools/validate_repository.sh sim-common-environment
```

### How Arena Is Obtained

Isaac Lab-Arena's dependencies are locked here, in the `arena` dependency group, so `uv sync`
installs them like everything else. Only Isaac Lab-Arena's own code comes from a checkout, which
`tools/arena_launch.sh` puts on `PYTHONPATH`. Its wheel is why: the packaging ships no
data files, so an installed Isaac Lab-Arena cannot load the Unitree G1 joint order or the whole-body
controller configuration, and the embodiment fails at composition time.

Without `E2E_ARENA_ROOT`, `tools/fetch_arena_checkout.sh` materializes the pinned
revision into `.arena/` as a blobless, shallow, sparse checkout without
documentation, submodules, or GR00T test data. Set
`E2E_ARENA_ROOT` to a checkout of your own to use that instead; it is never modified,
and one that is not at the pinned revision is refused rather than moved to it.

The sparse copy is also sufficient for closed-loop GR00T evaluation when the
launcher receives a standalone checkout through `--gr00t-root`. A full Isaac Lab-Arena
clone remains supported as the Isaac Lab-Arena root, but it still needs that separate,
explicitly pinned GR00T checkout.

The check records a finite wrist motion to HDF5, runs an Isaac Lab-Arena episode with a
deterministic shoulder action, and compares the realized environment snapshots.
Artifacts are retained under `test-results/common-environment/<timestamp>/`.

Both rollouts must leave the scene stable. The comparison checks each randomized
apple and plate position independently against the shared spawn window, then
compares their non-randomized heights within two centimeters. It compares the
robot by pelvis height within five centimeters. The Isaac Lab-Arena rollout therefore runs
with `--disable_task_terminations`, which keeps the episode going to its time-out
and suppresses `object_dropped`.

Isaac Lab-Arena's policy runner accepts runner and policy options before the environment
name and environment options after it. `tools/launch_arena_eval.sh` forwards any
argument following a bare `--` as an environment option.

### Isaac Lab-Arena Pin Enforcement

Isaac Lab-Arena is an external source checkout rather than a locked dependency, so
`uv.lock` cannot hold it to a revision. All launchers use `tools/check_arena_pin.py`
to check the checkout. It rejects a checkout that is not Isaac Lab-Arena, is at
the wrong version or revision, has uncommitted or untracked changes, or runs a locomotion policy other
than the pinned one. `tools/arena_launch.sh`, which every Isaac Lab-Arena entry point runs
through, and the runtime preflight both call it, so the shell and Python paths
enforce the same rules.

The locomotion policy pin needs a word, because the revision pin above already
implies it. Isaac Lab-Arena's whole-body controller downloads an ONNX export of WBC-AGILE
from a commit Isaac Lab-Arena pins in its own source, so a clean checkout at the pinned
revision can only have one. Recording it as
`arena_wbc_agile_policy_revision` states which locomotion policy was qualified,
and turns an Isaac Lab-Arena bump that changes it into a failed pin rather than a silently
swapped policy.

Verify a checkout directly:

```bash
uv run --no-sync ./tools/check_arena_pin.py --arena-root "${E2E_ARENA_ROOT}"
```

`runtime-preflight` checks the pin only when `E2E_ARENA_ROOT` is set, and reports
`"arena": "skipped"` otherwise: it reports on the environment as it finds it, and
fetching a checkout to have something to verify would be reporting on its own work.
The Isaac Lab-Arena modes do their own fetching, and check the pin on what they get.


### Choosing an Embodiment With Isaac Lab-Arena

The adapter accepts two Unitree G1 and two Unitree H2 embodiments whose action spaces differ, so they are
not interchangeable:

| Embodiment | Action layout | Use |
| --- | --- | --- |
| `g1_wbc_agile_joint` (default) | 50: 43 joint targets plus navigate, base-height, and torso-orientation commands | GR00T closed-loop evaluation, which emits exactly this layout |
| `g1_wbc_agile_pink` | 23: hand states and end-effector poses solved by Pink IK, plus the same locomotion commands | Unitree G1 human teleoperation and AutoData |
| `h2_agile_joint` | 61: 58 upper-body joint targets plus three AGILE velocity commands | Unitree H2 GR00T closed-loop evaluation |
| `h2_agile_pink` | 61: two 7-D pelvis-relative flange poses, 44 Sharpa targets, and three AGILE velocity commands | Unitree H2 teleoperation recording with PICO 4 Ultra handheld XR controllers, and AutoData |

Raw recordings do depend on this choice. Their manifest specifies the robot-control configuration and
action layout so replay returns to the same embodiment. Conversion maps validated
recordings made with Pink IK into the shared joint-space training format: 43 joints for Unitree G1
or 75 joints for Unitree H2.

### Recording on the Arena-Composed Environment

Recording defaults to the built-in Isaac Lab task. Pass `--arena` as the first argument
to record on the Isaac Lab-Arena-composed scene instead, which matches the GR00T evaluation path.
`tools/launch_recorder.sh --arena` runs `scripts/record_demos.py --arena`, checking the
pin and overlaying the checkout before running the recorder; a plain `uv run` cannot
import `isaaclab_arena`. The Isaac Lab-Arena embodiment defaults to `g1_wbc_agile_pink`, because
recording means teleoperation:

```bash
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
./tools/launch_recorder.sh --arena \
  --teleop_device handtracking \
  --enable_cameras \
  --dataset_file ./datasets/arena_demos.hdf5
```

Isaac Lab-Arena registers OpenXR retargeters for the Unitree G1 and Unitree H2 Pink embodiments, but no
keyboard or spacemouse retargeter, so interactive recording requires XR.
The finite `--sample_steps` check needs no device, because it steps the environment
itself with a command that holds the scene still -- the spawned joint pose under the
joint embodiment, and each wrist's current pose under Pink:

```bash
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
./tools/launch_recorder.sh --arena \
  --headless --enable_cameras \
  --sample_steps 300 --sample_export \
  --dataset_file /tmp/e2e-arena-recorder-smoke.hdf5
```

Three things differ from the built-in task's path:

- **Recorders.** Recording copies the task's `mimic_recorder_config` when it is
  available, then extends it with this app's `joint_pos_target` term through the same
  `add_workflow_recorder_terms` seam AutoData generation uses
  (`src/e2e_robot_workflows/arena/recorders.py`).
  Unitree G1's monolithic WBC action already records locomotion in its action group, so its
  separate `navigate_cmd` and `base_height_cmd` terms are disabled. Unitree H2 exposes a
  separate three-value lower-body action: its `navigate_cmd` term is retained, while
  `base_height_cmd` is disabled because the Unitree H2 AGILE export has no height channel.
- **Episode boundaries.** Every termination term is dropped, not just `time_out` and
  `success`. Isaac Lab-Arena's pick-and-place task also supplies `object_dropped`, which would
  end an episode when the apple fell below the background's minimum object height.
  The task's completion policy stays authoritative for labels and applies the same
  shared task predicate that evaluation scores with. Isaac Lab-Arena's success metric is
  dropped with the termination it reads.
- **Scene entity names.** The native scene inherits Isaac Lab's locomanipulation
  names (`object`, `destination`, `object_contact_sensor`); Isaac Lab-Arena spawns each asset
  under the configured name (`pick_object`, `placement_target`) and names
  the contact sensor after the object it watches. Success detection and environment
  snapshots resolve roles through
  `src/e2e_robot_workflows/environments/scene_keys.py`, so both work unchanged.
  The resolution is checked once, right after the environment is built, through the
  completion policy's `check_readable`: the per-step poll can only log a read error and
  keep recording, so an unresolvable name or a contact sensor watching nothing would
  otherwise reach the operator as "no episode ever succeeds" rather than as an error.
  `sim-arena-recording` runs that check on the Isaac Lab-Arena-composed scene.

Datasets of Unitree G1 demonstrations from either recording method use the Unitree G1 stage config;
datasets of Unitree H2 demonstrations use the pinned Unitree H2 stage config and 75-joint mapping. The writer
resolves a mapped field at the top level, under `obs/`, or by unique suffix match, so
the head camera resolves whether it was recorded as `obs/robot_head_cam_rgb` (built-in
task) or `camera_obs/robot_head_cam_rgb` (Isaac Lab-Arena), and the locomotion commands resolve
under `action/`.

The Pink embodiment is the supported end-to-end recording route because it can
teleoperate and supplies the task-space fields used by AutoData. A joint-embodiment
recording remains a valid recorder and conversion check: the converter writes its
required state, action, and video fields, then reports and drops unavailable optional
modality groups. Training such output requires a matching modality configuration;
the supplied Unitree G1 configuration expects the AGILE navigation and base-height columns.

### Validating the Isaac Lab-Arena Recording Path

`sim-arena-recording` is the repeatable version of the two commands above:

```bash
uv run --no-sync ./tools/validate_repository.sh sim-arena-recording
```

It records a finite scripted episode under each embodiment, then checks each exported
HDF5 for an episode, per-step `actions`, a `joint_pos_target` column as wide as the
recorded joint state, and non-empty head-camera frames. Finally it runs the converter's
`--inspect --strict` pass over the Pink recording, which fails when a required field
or mapping does not resolve; optional omissions are reported. Artifacts are retained under
`test-results/arena-recording/<timestamp>/`.

Both embodiments are recorded because their action spaces differ and so does the
command that holds the scene still, so one passing says little about the other. Only
the recording made with Pink IK is checked against the conversion requirements,
for the reason above.

The checks match the camera and action fields by leaf name rather than by path, because
Isaac Lab-Arena publishes the camera under `camera_obs/` and the built-in task under `obs/`, and
the converter resolves either. A check keyed to one layout would fail recordings that
convert perfectly well.

### Validating the Arena Locomotion Path

`sim-arena-locomotion` drives the same phases as `sim-agile-locomotion` -- settle,
walk backward, stop, turn in place, then reset and settle again -- on the
Isaac Lab-Arena-composed environment:

```bash
uv run --no-sync ./tools/validate_repository.sh sim-arena-locomotion
```

The phases, commands, and behavior limits come from `tools/simulation/agile_validation.py`
and are shared by both modes, so the two runtimes are held to one standard: the pelvis
stays inside its height band and tilt limit, the apple stays on its support, reverse
makes ground without drifting sideways, the stop and post-reset tails come to rest, and
the turn changes yaw without walking away. Both embodiments are driven, because they
share only the seven-number locomotion tail the command is written into. Artifacts are
retained under `test-results/arena-locomotion/<timestamp>/`.

One assertion does not carry across and is not restated in a weaker form. The built-in
mode reads a 12-number policy output out of a dedicated `lower_body_policy` observation
group, which Isaac Lab-Arena's whole-body controller has no counterpart for. In its place the
policy's identity is pinned and asserted from the imported config, and its effect is
measured by the phases above, which a policy producing nothing could not pass.

Interactive driving uses the same composition:

```bash
./tools/launch_arena_locomotion.sh --viz kit
```

## Environment Ownership

Isaac Lab-Arena handles composition and evaluation. Recording and raw replay
support both native Isaac Lab and Isaac Lab-Arena. The native implementation
remains supported for existing users, and Arena is an explicit alternative.
Their data uses a shared joint-space training format only after conversion.

`G1StaticAppleEnvCfg` therefore remains a supported data-production environment, not disposable
scaffolding. `sim-common-environment` checks that it and Isaac Lab-Arena still use the same task definition
without implying that their raw action schemas are interchangeable.

| Entry point | Current owner/status |
| --- | --- |
| `scripts/record_demos.py` | native by default; `--recording-lane arena` is explicit |
| `scripts/locomotion_with_agile.py` | native by default; `--arena` is explicit |
| `scripts/replay_demos.py` | complete: the manifest routes native Unitree G1 28/32-D, Isaac Lab-Arena Unitree G1 23/50-D, and Isaac Lab-Arena Unitree H2 61-D raw actions to their owning environment |
| `sim-static-apple` | native fixed-base behavior |
| `sim-agile-locomotion` | native cross-check alongside `sim-arena-locomotion` |
| `sim-recorder` | remains the native recorder/replay regression; Isaac Lab-Arena has its own recording checks |
| Isaac Lab-Arena AutoData | workflow-owned demonstration generation for Unitree G1 and Unitree H2 is implemented; see [G1 data generation](../how-to/generate-data-with-autodata.md#isaac-lab-arena-recording-generation) and [H2 data generation](../how-to/generate-h2-demonstrations.md) for the run procedures |

Isaac Lab-Arena is reachable through its pinned launcher.

### GR00T Modality Divergence

The modality layout this repository registers is not Isaac Lab-Arena's
`isaaclab_arena_gr00t/embodiments/g1/modality.json`. Two differences are intentional
and are the only ones:

- **Action horizon.** 40 here, against 50 in Isaac Lab-Arena's built-in Unitree G1 configuration. It is
  one parameter, appearing as `_ACTION_HORIZON` in
  `src/e2e_robot_workflows/gr00t/g1_static_apple_modality_config.py` and as
  `action_horizon` in `configs/stages/gr00t/closedloop_eval.yaml`; training and
  evaluation must state the same value, and 40 is what GR00T-N1.7-3B's
  `max_action_horizon` accepts. A 50-horizon config crashes N1.7 in its data
  processor with a negative tensor dimension.
- **Omitted action groups.** Isaac Lab-Arena's action layout includes `left_wrist_pose`,
  `right_wrist_pose`, and `torso_orientation_rpy_command`; this one does not. The wrist
  poses are kept as *state*, where they report the achieved pose, but not as actions: a
  commanded wrist pose is the input Pink IK solves, and `joint_pos_target` is what it
  solved to, so recording both would train two representations of one command. The
  torso-orientation command has no source at all on this task, whose AGILE action term
  publishes only the navigate and base-height commands.

Both are recorded in-repo rather than raised upstream: Isaac Lab-Arena's layout is correct for
the model and task it was tuned for, so this is a per-application choice rather than a
defect.

## Optional UI Smoke Tests

Launch the finite recorder check:

```bash
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
uv run --no-sync python scripts/record_demos.py \
  --viz kit \
  --teleop_device keyboard \
  --sample_steps 300 \
  --sample_action wrist_sine \
  --sample_export \
  --dataset_file /tmp/e2e-recorder-ui-smoke.hdf5
```

Launch the same scene through Isaac Lab-Arena:

```bash
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
./tools/launch_arena_eval.sh \
  --viz kit \
  --rendering_mode quality \
  --num_steps 600 \
  --policy_type e2e_robot_workflows.arena.g1_smoke_policy.G1StandingSineSmokePolicy \
  --output_base_dir /tmp/e2e-arena-ui \
  --no_solve_relations
```

Both checks exit automatically. A zero smoke-test success rate is valid because
these commands qualify loading, action flow, recording, metrics, and lifecycle,
not checkpoint performance.

Exact runtime ownership and supported boundaries are collected in
[Reference](index.md).

## Native AutoData Integration

Native AutoData annotation and generation use AutoData's isolated interpreter.
Arena generation uses this repository's Arena runtime with the pinned AutoData
source overlaid. Both require a clean AutoData checkout at the revision in
`runtime.toml`.

The native bridge registers this repository's environment ids before running
AutoData's unmodified scripts through
`python -m e2e_robot_workflows.autodata`. The fixed and AGILE environments expose
separate action spaces. Their descriptors live in `configs/stages/autodata/`:

| File | Defines |
| --- | --- |
| `static_apple_embodiment.yaml` | Fixed native 28-D actions |
| `static_apple_agile_embodiment.yaml` | Native 32-D actions with AGILE locomotion |
| `static_apple_task.yaml` | Right-hand transfer subtasks and boundary signal |
| `static_apple_left_task.yaml` | Left-hand transfer subtasks and boundary signal |

The generation environment supplies a success termination, randomized apple
pose, and subtask boundary signal. The recording environment instead delegates
episode admission to the task completion policy. AutoData's environment factory
also restores this repository's recorder terms, including `joint_pos_target`.
AGILE generation records `navigate_cmd` and `base_height_cmd`; fixed generation
omits them.

:::{important}
A `success` termination must be callable. AutoData's attribute-existence check
alone accepts `None`, which cannot score annotation or generation correctly.
:::

After provisioning, validate both native modes and the left-hand descriptor
against live environments:

```bash
./tools/launch_autodata.sh validate --wbc-mode fixed
```

```bash
./tools/launch_autodata.sh validate --wbc-mode agile
```

```bash
./tools/launch_autodata.sh validate --wbc-mode agile --manipulation-hand left
```

These checks compare the action width and observations with the descriptors.
They exercise placement success and the hand-held-object boundary in both
positive and negative cases. They require the AutoData simulator environment.

To check the installation with AutoData's own pre-annotated demonstrations:

```bash
./tools/launch_autodata.sh example \
  --viz none --device cpu --num_envs 5 --generation_num_trials 10
```

Validate the generated file after the example finishes:

```bash
./tools/launch_autodata.sh validate-dataset \
  outputs/autodata/g1_example_generated.hdf5
```

This example uses AutoData's task and data. It does not qualify this
repository's recorded demonstrations. Use `--viz kit` to view the attempts.

## Legacy Recording Migration

Migration applies only to older workflow recordings without the root HDF5
attribute `e2e_workflow_manifest`. Current recordings already contain it.
A newly merged file missing its manifest is not evidence of a legacy recording;
retain and use its original manifested inputs for replay and AutoData.

1. Check for the attribute without modifying the file:

   ```bash
   uv run --extra converter python - /absolute/path/to/recording.hdf5 <<'PY'
   import sys

   import h5py

   from e2e_robot_workflows.data.dataset_manifest import MANIFEST_ATTRIBUTE

   with h5py.File(sys.argv[1], "r") as dataset:
       print("Manifest present:", MANIFEST_ATTRIBUTE in dataset.attrs)
   PY
   ```

   If present, use the original recording. Investigate a malformed or
   incompatible manifest instead of removing it to force migration.

2. If absent, identify the original recording method, action layout, and manipulating hand
   from the collection configuration. A filename or action width alone cannot
   identify those settings. For a confirmed native G1 AGILE recording using the
   right hand and 32-D layout:

   ```bash
   uv run --extra converter python -m e2e_robot_workflows.data.dataset_manifest migrate-recording \
     --input-file datasets/legacy.hdf5 \
     --output-file datasets/legacy-migrated.hdf5 \
     --lane native \
     --layout native_g1_pink_agile_32d_v1 \
     --manipulation-hand right \
     --confirm-legacy-migration
   ```

   Use values matching the original recording. List supported layouts with
   `uv run --extra converter python -m e2e_robot_workflows.data.dataset_manifest migrate-recording --help`.

The output must be a new file. The original remains untouched. The copy is
marked `legacy_unverified`: annotation and generation accept it, but it remains
excluded from qualified workflow evidence. Return to
[G1 data generation](../how-to/generate-data-with-autodata.md#native-recording-generation)
with the migrated copy as the source.

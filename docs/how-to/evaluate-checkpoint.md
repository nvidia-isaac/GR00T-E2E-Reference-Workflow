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

(evaluate-a-fine-tuned-gr00t-checkpoint)=
# G1 Policy Evaluation

Evaluate a trained Unitree G1 policy on the apple-to-plate task in simulation.
A GR00T server loads the checkpoint while Isaac Lab-Arena runs episodes and
writes metrics. The two processes use separate locked environments. For
Unitree H2 Plus, use [H2 policy evaluation](evaluate-h2-checkpoint.md).

Use the published checkpoint below or the complete
`outputs/g1/training/checkpoint-20000` directory from
[G1 policy training](fine-tune-gr00t.md). Run commands from this repository's
root on the evaluation workstation.

## Use a Published Checkpoint

```{include} ../_includes/artifacts/g1-checkpoint.md.inc
```

## Requirements

- This repository is synced with `uv sync --frozen`.
- Isaac Lab-Arena is checked out cleanly at the revision in `runtime.toml`.
- Isaac-GR00T is checked out separately and cleanly at the revision in
  `runtime.toml`, with `uv sync --frozen` already completed.
- The Hugging Face login used by the GR00T environment can access
  `nvidia/Cosmos-Reason2-2B`.
- The checkpoint was trained for `NEW_EMBODIMENT` using GR00T N1.7, a
  40-step action horizon, `ego_view`, five Unitree G1 state groups (31 values), and
  seven action groups (35 values). Isaac Lab-Arena maps those groups to its 43-joint
  state and 50-dimensional simulator action and obtains `ego_view` from
  `robot_head_cam_rgb`.

The supplied training recipe accepts data recorded with Pink IK and AGILE in
either Isaac Lab-Arena or native Isaac Lab. Both provide the navigation
and base-height action groups this evaluator requires.
Checkpoints trained with either group removed need custom evaluation integration.

:::{important}
For a locally trained checkpoint, the path must name its `checkpoint-<step>` directory, not its parent
training-output directory. The launcher checks the required model, processor,
statistics, trainer-state, and experiment-configuration files for this default route.
`--model-path` is treated as a locally trained checkpoint by default. The
released Hugging Face checkpoint requires an explicit
`--checkpoint-source released`. For the workflow model downloaded above, use
`--checkpoint-source published`.
:::

(prepare-arena-checkout)=
## Prepare the Isaac Lab-Arena Checkout

1. In the first terminal, set the paths and fetch the managed sparse checkout
   at the revision in `runtime.toml`:

   ```bash
   export E2E_ROOT="$PWD"
   export ISAAC_GR00T_ROOT=/absolute/path/to/Isaac-GR00T
   export ARENA_ROOT="$(./tools/fetch_arena_checkout.sh)"
   ```

2. Verify the checkout in the same terminal:

   ```bash
   uv run --frozen --no-sync ./tools/check_arena_pin.py \
     --arena-root "$ARENA_ROOT"
   ```

   The sparse checkout includes Isaac Lab-Arena's source-only GR00T client.
   Passing the standalone checkout through `--gr00t-root` supplies the model
   implementation, so no submodule is required. A clean full Isaac Lab-Arena
   clone at the same revision also remains supported as `E2E_ARENA_ROOT`, with
   the same explicit standalone GR00T root.

3. Choose **one** model using the tabs below. Run only the matching block in the same terminal:

   ::::{tab-set}

   :::{tab-item} Provided model

   Use the model downloaded above:

   ```bash
   export G1_CHECKPOINT="$E2E_ROOT/outputs/g1/published-checkpoint"
   export G1_CHECKPOINT_SOURCE=published
   ```

   :::

   :::{tab-item} Your own model

   Use the model produced by policy training for G1:

   ```bash
   export G1_CHECKPOINT="$E2E_ROOT/outputs/g1/training/checkpoint-20000"
   export G1_CHECKPOINT_SOURCE=trained
   ```

   If you used a different training output directory, adjust the checkpoint path.

   :::
   ::::

   Both commands below use your selected model and source.

## Run the Preflight

The preflight checks both source revisions, clean checkouts, gated-model
access, the trained-checkpoint layout, GR00T N1.7, and the checkpoint's complete
observation and action formats and normalization statistics against the
Isaac Lab-Arena client's requirements.

1. In that first terminal, run the preflight:

   ```bash
   ./tools/launch_gr00t_server.sh \
     --arena-root "${ARENA_ROOT}" \
     --gr00t-root "$ISAAC_GR00T_ROOT" \
     --model-path "$G1_CHECKPOINT" \
     --checkpoint-source "$G1_CHECKPOINT_SOURCE" \
     --check
   ```

   This command does not load the checkpoint or start Isaac Sim. A passing run
   prints:

   ```text
   GR00T evaluation contract passed: policy=Gr00tRemoteClosedloopPolicy, validation_scope=checkpoint, checkpoint_metadata=validated, expected_checkpoint_state=31, expected_checkpoint_action=35, sim_state=43, sim_action=50, horizon=40, camera=robot_head_cam_rgb
   ```

   Fix any preflight failure before starting evaluation. Do not bypass a
   revision, model-type, joint, camera, or action-horizon mismatch.

## Start the Policy Server

1. In the same first terminal, start the server after preflight passes:

   ```bash
   ./tools/launch_gr00t_server.sh \
     --arena-root "${ARENA_ROOT}" \
     --gr00t-root "$ISAAC_GR00T_ROOT" \
     --model-path "$G1_CHECKPOINT" \
     --checkpoint-source "$G1_CHECKPOINT_SOURCE" \
     --device cuda \
     --host 127.0.0.1 \
     --port 5555
   ```

2. Wait for:

   ```text
   Server is ready and listening on tcp://127.0.0.1:5555
   ```

   Keep this process running while Isaac Lab-Arena evaluates the policy.

## Run One Visual Episode

1. In a second terminal at this repository's root, set the runtime paths and run:

   ```bash
   export ARENA_ROOT="$(./tools/fetch_arena_checkout.sh)"
   export ISAAC_GR00T_ROOT=/absolute/path/to/Isaac-GR00T
   ```

   ```bash
   OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
   ./tools/launch_gr00t_eval.sh \
     --arena-root "${ARENA_ROOT}" \
     --gr00t-root "${ISAAC_GR00T_ROOT}" \
     --viz kit \
     --remote_host 127.0.0.1 \
     --remote_port 5555 \
     --num_episodes 1 \
     --output_base_dir outputs/g1/evaluation
   ```

   This is a smoke evaluation. Inspect the complete workflow scene: the
   Unitree G1 should be upright, with the apple, plate, table, and head-camera
   view configured correctly. The run
   should complete one episode with a 300-step time limit, print metrics, write
   results under `outputs/g1/evaluation/`, and close Isaac Sim automatically.
   Task terminations may end it earlier.

2. Use `--viz none` instead of `--viz kit` for unattended evaluation.

(measure-policy)=
## Measure the Policy

1. After the quick visual test passes, run enough complete episodes for the desired
   statistical confidence. For example:

   ```bash
   OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
   ./tools/launch_gr00t_eval.sh \
     --arena-root "${ARENA_ROOT}" \
     --gr00t-root "${ISAAC_GR00T_ROOT}" \
     --viz none \
     --remote_host 127.0.0.1 \
     --remote_port 5555 \
     --num_episodes 100 \
     --output_base_dir outputs/g1/evaluation
   ```

   Isaac Lab-Arena reports:

   - `success_rate`: episodes with settled, supported placement and both hands clear;
   - `object_moved_rate`: episodes in which the apple moved; and
   - progress details for intermediate task objectives.

   Each run creates a timestamped directory below `--output_base_dir` containing
   `episode_results_rank0.jsonl` and `index.html`. The final terminal metrics line
   gives the completed episode count and rates. For example, a one-episode
   smoke evaluation with the published checkpoint and seed 42 printed:

   ```text
   [Rank 0/1] Metrics: {'num_episodes': 1, 'success_rate': 0.0, 'object_moved_rate': 1.0}
   ```

   The apple moved, but placement success was not reached within the 300-step
   limit. Treat this as an integration check, not a performance estimate.
   Compare `num_episodes` with your requested count and the records in
   `episode_results_rank0.jsonl`. An incomplete run must not be reported as a
   completed 100-episode evaluation.

   The success rate is comparable to the training workflow only when the
   checkpoint used data from the same workflow environment and the same robot,
   camera, joint ordering, modality configuration, and action horizon.

## Shut Down

Isaac Lab-Arena exits when the requested episode count is reached.

1. In the server terminal, press **Ctrl+C** and wait for:

   ```text
   Shutting down server...
   ```

2. Confirm that port 5555 is free before starting another server:

   ```bash
   ss -ltn 'sport = :5555'
   ```

   The output should have no listening socket row.

## Next Stage

Review the timestamped `index.html` and `episode_results_rank0.jsonl` under
`outputs/g1/evaluation/`. Use failed episodes to decide whether to collect
more demonstrations, generate more data, or revise training. Keep the dataset,
checkpoint, and evaluation report together as the evidence for this run.

## Optional: Use the Pinned Released Checkpoint

1. To exercise the integration without training, download the released
   static-apple checkpoint pinned in `runtime.toml`:

   ```bash
   checkpoint_pin="$(
     uv run --no-sync python -c \
       'import tomllib; p=tomllib.load(open("runtime.toml", "rb"))["tested"]; print(p["static_apple_checkpoint_repository"] + "@" + p["static_apple_checkpoint_revision"])'
   )"
   ```

   ```bash
   "$ISAAC_GR00T_ROOT/.venv/bin/hf" download \
     "${checkpoint_pin%@*}" \
     --revision "${checkpoint_pin#*@}" \
     --exclude optimizer.pt rng_state.pth scheduler.pt trainer_state.json training_args.bin \
     --local-dir outputs/g1/released-checkpoint
   ```

2. Select this checkpoint in the server terminal, then rerun the preflight
   and server commands above:

   ```bash
   export G1_CHECKPOINT="$E2E_ROOT/outputs/g1/released-checkpoint"
   export G1_CHECKPOINT_SOURCE=released
   ```

   The released checkpoint was trained on a different scene, so its metrics
   prove integration only and are not a performance baseline for this workflow's
   scene.

## Troubleshooting

- `Expected Gr00tN1d7`, `Expected action_horizon 40`, or
  `Checkpoint modality mismatch`: select the N1.7 checkpoint produced by the
  matching training and modality configuration.
- `Trained checkpoint must contain ...`: point
  `--model-path` at `checkpoint-<step>`, not the parent output directory.
- A Cosmos access error: log in with
  `/absolute/path/to/Isaac-GR00T/.venv/bin/hf auth login` after obtaining
  access.
- `Cannot reach GR00T policy server`: wait for the server-ready message and
  verify the host and port passed to both processes.
- A zero success rate with valid JSONL/report output means integration worked
  but the policy did not complete the task. Inspect the scene and dataset,
  then evaluate more episodes before drawing a performance conclusion.

The corresponding NVIDIA reference flow is documented in
[Sim Evaluation](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/simulation-workflow/sim-evaluation.html).

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

(h2-contracts)=
# H2 Architecture and Data Requirements

Unitree H2 uses Isaac Lab-Arena to connect simulation recording, AutoData
generation, and GR00T evaluation. Each stage checks the data formats and
robot-control settings it requires.

## Simulation and Robot-Control Setup

The `h2_agile_pink` robot configuration combines Pink IK for arm control with
AGILE for lower-body control. It handles teleoperation recording in simulation,
replay, and AutoData. Operators record demonstrations using a PICO 4 Ultra
headset and handheld XR controllers. The `h2_agile_joint` configuration handles
GR00T evaluation, replacing Pink IK with direct joint targets.
Both use the same Unitree H2 Plus simulation model, AGILE checkpoint, camera,
workcell, and placement predicate.

The model is materialized from the exact Unitree H2 Plus asset with
`Physics=physx`, `hands=sharpa`, and `collision_profile=full`. A content-pinned,
kinematics-only URDF is generated for Pink. Generated assets remain outside
the repository. The restricted AGILE policy's revision and digest are owned by
`configs/robots/h2.yaml`.

## Action and State Formats

| Stage | Number of Values | Meaning |
| --- | ---: | --- |
| Teleoperation, replay, and AutoData (`h2_agile_pink`) | 61 | Two flange poses, 44 Sharpa joints, and `[vx, vy, wz]` |
| GR00T evaluation | 61 | 58 named upper-body joint targets and `[vx, vy, wz]` |
| Recorded state and target | 75 | Complete live articulation order |

During teleoperation, pose, trigger, and thumbstick data from the handheld XR
controllers is mapped into the 61-value action format before the robot-control
software receives it. AutoData produces the same action format.

The 61-D formats are not interchangeable. Dataset metadata and the selected
robot configuration (`h2_agile_pink` or `h2_agile_joint`) determine which format
is used. LeRobot retains the complete joint target, while the
H2 GR00T modality selects the 58-D upper body and 3-D navigation projection.

## Runtime Timing

The composed runtime uses 200 Hz physics, 100 Hz AGILE/Pink control, 50 Hz
rendering, and 25 Hz camera capture. The fixed-batch AGILE checkpoint runs in
ONNX Runtime on the CPU. Recording keeps simulation on the CPU to avoid a
GPU-to-CPU synchronization at every control step.

The simulation checks require at least 0.35 real-time factor for a synchronized
300-step recorder workload. This minimum performance requirement excludes OpenXR
and CloudXR and does not measure interactive headset performance. The operator
assesses latency and visual smoothness while recording demonstrations with the headset.

H2 generation is sequential. The AGILE ONNX export has batch size one, and the
Arena Mimic composition enforces `num_envs=1`.

## XR Input Alignment and Robot Recovery

After each reset, H2 holds its raised hands while its stance settles for three
seconds of simulation time. Startup blocks if forearm or hand clearance falls
below 4 cm, downward motion exceeds 0.5 m/s, or the robot does not settle.
Clearance uses collision-geometry bounds at each control step. These startup
checks do not provide collision avoidance during teleoperation.

Settling frames are excluded from the dataset. The ready simulation state,
AGILE history, and motor targets become the episode's initial state. There is
no further robot reset before alignment.

After **Play** or an automatic scene reset, the first valid pose sample from
each handheld XR controller aligns its tracked pose with the corresponding
robot hand. The robot holds its hands in place instead of moving them to the
XR controllers' absolute positions. Subsequent hand targets are limited to
0.75 m/s and 3 rad/s in simulation time. Effective targets
are recorded, so replay does not recalibrate them.

Pink IK favors the configured bent-elbow posture in the remaining degrees of
freedom. This is a posture preference, not a hard elbow constraint. Wrist
limits and unreachable poses can still constrain motion.

Grip mapping is binary and hysteretic. Either trigger or squeeze closes the
hand. Transient malformed samples retain the previous command; sustained
invalid input opens the hand. Pose-tracking or XR-session loss discards the
active episode, and recording resumes when tracking returns.

The sticks have a 10% neutral deadzone. With neutral sticks, station keeping
adds bounded planar and yaw corrections. Moving either stick overrides the
assistance; stopping latches a new position. Effective navigation commands are
recorded for replay and training. `--no-h2-station-keep` disables assistance.
It does not pin the base or eliminate vertical settling.

## Success and Handover

The placement predicate requires the apple to be supported and settled on the
plate, with contact between them and both hands clear. Evaluation's
`success_rate` measures this final placement predicate. It contains no pickup
or handover history.

Accepting a recording additionally requires the bimanual sequence: the selected
hand picks up the apple, the other hand receives it, and the receiving hand
places it. Pickup requires a lifted apple near the hand, a close command,
measured partial finger closure, and apple-to-hand contact over three control
samples. Contact and closure establish the pickup signal but do not guarantee
a mechanically stable grasp.

The giving hand must retain measured grasp through the receiving hand's first
grasp sample. Releasing onto the table before the second pickup is rejected.
Acceptance also requires 20 successful follow-through frames and valid pickup
and receiving-hand boundaries. Incomplete or ambiguous handovers are discarded
before counting or export, including with `--dataset_export_mode all`.
Manual success labels are disabled. With automatic hand selection, later takes
must match the first accepted take's initial pickup hand.

AutoData requires both pickup edges and re-anchors the final paired motion to
the plate at the receiving-hand edge. Older recordings without the required
grasp signals are not automatically relabeled.

## Recording and Replay

A measured joint-limit fault discards and resets the active recording episode,
including with `--dataset_export_mode all`. Previously saved episodes remain
intact. Invalid IK targets still abort. Replay and software validation do not
recover from joint-limit faults.

Exported takes are finalized after the file closes, including when a later
episode raises an error. Finalization failures still fail the run. A loop
performance summary alone does not establish a usable dataset.

State replay restores saved robot and object states at every frame for visual
review. Action replay re-runs Pink, AGILE, and contact physics, so it can expose
drift as the simulated robot follows those commands. In either mode, `--strict` requires complete consumption of
the selected frame streams and successful cleanup. It does not assert identical
physics trajectories.

H2 action replay requires the AGILE feedback history and both position and velocity
actuator targets for Pink and joint layouts. Older files lacking both kinds of
snapshot can be inspected in state mode but cannot be used to validate a workflow run.
Partial snapshots are rejected.

AutoData prepares the robot-control software and settles H2 after each attempt reset.
Measured joint-limit faults discard the attempt's incomplete recorder streams
before retrying with fresh queues. They consume the attempt budget. Saved
demonstrations remain intact, and unrelated faults stop generation.

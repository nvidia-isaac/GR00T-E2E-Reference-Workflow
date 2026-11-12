# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-side contracts for the Isaac Lab 3.0 migration."""

from __future__ import annotations

import math
import re
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

from e2e_robot_workflows.arena.embodiment_capabilities import (
    G1_ARENA_EMBODIMENT,
    G1_ARENA_TELEOP_EMBODIMENT,
    H2_ARENA_EMBODIMENT,
    H2_ARENA_TELEOP_EMBODIMENT,
    capability_for_embodiment,
)
from e2e_robot_workflows.robots.g1 import load_g1_robot_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_g1_initial_rotation_is_xyzw_and_faces_positive_y() -> None:
    """The configured +90 degree yaw must use Isaac Lab 3's XYZW order."""

    x, y, z, w = load_g1_robot_config().initial_rotation_xyzw

    assert math.isclose(x * x + y * y + z * z + w * w, 1.0, rel_tol=0.0, abs_tol=2e-5)
    rotated_x_axis = (
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + z * w),
        2.0 * (x * z - y * w),
    )
    assert math.isclose(rotated_x_axis[0], 0.0, abs_tol=2e-5)
    assert math.isclose(rotated_x_axis[1], 1.0, abs_tol=2e-5)
    assert math.isclose(rotated_x_axis[2], 0.0, abs_tol=2e-5)


def test_runtime_scripts_use_explicit_proxy_array_torch_views() -> None:
    """Isaac Lab 3 asset data must be converted explicitly before Torch use."""

    properties = "body_link_state_w|joint_pos|joint_vel|root_state_w|root_lin_vel_b|root_ang_vel_b"
    implicit_access = re.compile(rf"\.data\.({properties})(?!\.torch)")
    for relative_path in (
        "scripts/locomotion_with_agile.py",
        "scripts/record_demos.py",
        "tools/simulation/agile_validation.py",
        "tools/simulation/pink_validation.py",
        "tools/simulation/validate_static_apple.py",
        "tools/simulation/validate_agile_locomotion.py",
        "tools/simulation/validate_autodata_bridge.py",
        "src/e2e_robot_workflows/tasks/static_apple/terminations.py",
        "src/e2e_robot_workflows/tasks/static_apple/observations.py",
    ):
        contents = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert implicit_access.search(contents) is None, relative_path


def test_fixed_mode_replaces_the_inherited_locomotion_teleop_pipeline() -> None:
    """Fixed mode needs a 28-D upper-body pipeline, not inherited 32-D teleop."""

    env_cfg = (REPO_ROOT / "src/e2e_robot_workflows/tasks/static_apple/env_cfg.py").read_text(encoding="utf-8")
    recorder = (REPO_ROOT / "scripts/record_demos.py").read_text(encoding="utf-8")

    assert "_build_g1_upper_body_pipeline" in env_cfg
    assert "self.isaac_teleop = IsaacTeleopCfg(" in env_cfg
    assert "OpenXRDeviceCfg" not in env_cfg
    assert "from isaaclab_teleop import remove_camera_configs" in recorder
    assert 'getattr(env_cfg, "isaac_teleop", None) is not None' in recorder
    assert 'platform.machine() not in {"x86_64", "AMD64"}' in recorder


def test_agile_policy_output_is_inference_only_before_warp_actuation() -> None:
    """Beta 2's gradient-tracked policy output is incompatible with Warp."""

    action_adapter = (REPO_ROOT / "src/e2e_robot_workflows/tasks/static_apple/agile_action.py").read_text(
        encoding="utf-8"
    )
    env_cfg = (REPO_ROOT / "src/e2e_robot_workflows/tasks/static_apple/env_cfg.py").read_text(encoding="utf-8")
    agile_validation = (REPO_ROOT / "tools/simulation/agile_validation.py").read_text(encoding="utf-8")
    agile_entrypoint = (REPO_ROOT / "tools/simulation/validate_agile_locomotion.py").read_text(encoding="utf-8")

    assert "class InferenceAgileBasedLowerBodyAction(AgileBasedLowerBodyAction):" in action_adapter
    assert "@torch.no_grad()" in action_adapter
    assert "class_type = InferenceAgileBasedLowerBodyAction" in env_cfg
    assert "lower_term.processed_actions.requires_grad" in agile_validation
    assert 'REPO_ROOT / "runtime.toml"' in agile_entrypoint


def test_static_apple_wbc_mode_is_explicit_configuration() -> None:
    """First-party entry points must not select WBC behavior through process state."""

    env_cfg = (REPO_ROOT / "src/e2e_robot_workflows/tasks/static_apple/env_cfg.py").read_text(encoding="utf-8")
    viewer = (REPO_ROOT / "scripts/locomotion_with_agile.py").read_text(encoding="utf-8")
    recorder = (REPO_ROOT / "scripts/record_demos.py").read_text(encoding="utf-8")

    assert "wbc_mode: WbcMode = DEFAULT_WBC_MODE" in env_cfg
    assert 'if self.wbc_mode == "agile":' in env_cfg
    assert "G1StaticAppleEnvCfg(wbc_mode=wbc_mode)" in env_cfg
    assert 'make_env_cfg(wbc_mode="agile"' in viewer
    assert '"--wbc-mode"' in recorder


def test_shared_environment_entrypoints_fail_closed_before_startup() -> None:
    recorder = (REPO_ROOT / "scripts/record_demos.py").read_text(encoding="utf-8")
    common_launcher = (REPO_ROOT / "tools/arena_launch.sh").read_text(encoding="utf-8")
    pin_checker = (REPO_ROOT / "tools/check_arena_pin.py").read_text(encoding="utf-8")
    rejection = 'parser.error("--environment_snapshot is supported only for the built-in static-apple task.")'

    assert rejection in recorder
    assert recorder.index(rejection) < recorder.index("app_launcher = AppLauncher(args_cli)")
    with pytest.raises(ValueError, match="unsupported Arena embodiment"):
        capability_for_embodiment("not_registered")
    # Every Arena entry point runs through the common launcher, which checks the pin
    # before it runs anything, so the fail-closed rules live in one place.
    assert "tools/check_arena_pin.py" in common_launcher
    for launcher_name in (
        "launch_recorder.sh",
        "launch_arena_eval.sh",
        "launch_arena_locomotion.sh",
        "launch_gr00t_eval.sh",
    ):
        launcher = (REPO_ROOT / "tools" / launcher_name).read_text(encoding="utf-8")
        assert "tools/arena_launch.sh" in launcher, f"{launcher_name} must launch through the common launcher"
    assert "runtime.toml" in pin_checker
    assert '"status", "--porcelain=v1", "--untracked-files=all"' in pin_checker


def test_arena_embodiment_choice_preserves_the_evaluation_action_layout() -> None:
    """Joint evaluation and Pink teleoperation remain distinct for both robots."""

    for joint, pink in (
        (G1_ARENA_EMBODIMENT, G1_ARENA_TELEOP_EMBODIMENT),
        (H2_ARENA_EMBODIMENT, H2_ARENA_TELEOP_EMBODIMENT),
    ):
        capability = capability_for_embodiment(joint)
        assert capability is capability_for_embodiment(pink)
        capability.validate(embodiment=joint, teleop_device=None, mimic=False)
        capability.validate(embodiment=pink, teleop_device="openxr", mimic=True)
        with pytest.raises(ValueError, match="Mimic generation requires"):
            capability.validate(embodiment=joint, teleop_device=None, mimic=True)


def test_environment_parity_run_holds_the_scene_still() -> None:
    """The parity check compares realized object poses, so nothing may disturb them.

    Two things would: Arena's contact-based ``success`` termination cuts the episode short
    when the apple merely settles against the plate, and a smoke policy commanding zero
    joint targets sweeps the arms down across the table. Both are asserted as source text
    because ``isaaclab_arena`` is not installable on the host.
    """

    adapter = (REPO_ROOT / "src/e2e_robot_workflows/arena/external_environment.py").read_text(encoding="utf-8")
    policy = (REPO_ROOT / "src/e2e_robot_workflows/arena/g1_smoke_policy.py").read_text(encoding="utf-8")
    actions = (REPO_ROOT / "src/e2e_robot_workflows/arena/g1_actions.py").read_text(encoding="utf-8")
    validation = (REPO_ROOT / "tools/validate_common_environment.sh").read_text(encoding="utf-8")

    assert "disable_task_terminations: bool = False" in adapter
    assert "env_cfg.terminations.success = None" in adapter
    assert "env_cfg.terminations.object_dropped = None" in adapter
    assert "env_cfg.recorders.success_rate = None" in adapter
    assert "env_cfg.metrics = None" in adapter
    assert "--disable_task_terminations" in validation
    assert "action[:, :joint_count] = robot.data.default_joint_pos.torch" in actions
    assert "joint_hold_action(env.unwrapped, base_height_m=self.config.base_height_m)" in policy


def test_arena_launcher_passes_environment_options_after_the_environment_name() -> None:
    """Arena's runner takes environment options only after the environment name."""

    launcher = (REPO_ROOT / "tools/launch_arena_eval.sh").read_text(encoding="utf-8")

    assert 'if [[ "${argument}" == "--" ]]; then' in launcher
    assert launcher.index('${runner_args[@]+"${runner_args[@]}"}') < launcher.index("e2e_static_apple")
    assert launcher.index("e2e_static_apple") < launcher.index('${environment_args[@]+"${environment_args[@]}"}')


def test_both_runtimes_spawn_the_declared_dynamic_plate() -> None:
    """Recording and Arena evaluation must agree on the plate's mass and per-episode pose.

    ``isaaclab_arena`` is not installable on the host, so the adapter is asserted as
    source text -- the same approach the other Arena contracts here use.
    """

    adapter = (REPO_ROOT / "src/e2e_robot_workflows/arena/external_environment.py").read_text(encoding="utf-8")
    env_cfg = (REPO_ROOT / "src/e2e_robot_workflows/tasks/static_apple/env_cfg.py").read_text(encoding="utf-8")

    # A dynamic plate takes its mass from the contract, not from USD-derived density.
    assert "mass_props=MassPropertiesCfg(mass=DESTINATION_MASS_KG)" in env_cfg
    assert '"mass_props": sim_utils.MassPropertiesCfg(mass=destination_def.mass_kg)' in adapter
    # ...and returns to its declared spawn pose each episode instead of drifting
    # across takes: on the recording side because it is a RigidObject and therefore
    # covered by reset_scene_to_default, on the Arena side because its per-object
    # pose reset event is left enabled.
    assert "destination = RigidObjectCfg(" in env_cfg
    assert 'reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")' in env_cfg
    assert "destination.disable_reset_pose()" not in adapter


def test_gr00t_evaluation_reuses_the_pinned_arena_client_contract() -> None:
    launcher = (REPO_ROOT / "tools/launch_gr00t_eval.sh").read_text(encoding="utf-8")
    server_launcher = (REPO_ROOT / "tools/launch_gr00t_server.sh").read_text(encoding="utf-8")
    checker = (REPO_ROOT / "tools/check_gr00t_eval.py").read_text(encoding="utf-8")
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = tomllib.loads((REPO_ROOT / "runtime.toml").read_text(encoding="utf-8"))["tested"]

    assert "submodules/Isaac-GR00T" in launcher
    assert '"${REPO_ROOT}/configs/stages/gr00t/closedloop_eval.yaml"' in launcher
    assert 'cd "${REPO_ROOT}"' in launcher
    assert "Gr00tRemoteClosedloopPolicy" in launcher
    # --check runs the checker itself through the common launcher, which enforces the pin.
    assert '"${REPO_ROOT}/tools/arena_launch.sh" \\\n    python "${REPO_ROOT}/tools/check_gr00t_eval.py"' in launcher
    assert "--enable_cameras" in launcher
    assert "_require_checkpoint_contract" in checker
    assert "run_gr00t_server.py" in server_launcher
    assert '"modality_configs"]["new_embodiment"]' in server_launcher
    assert 'if [[ "${CHECK_ONLY}" == true ]]' in server_launcher
    assert 'hf_hub_url("nvidia/Cosmos-Reason2-2B", "config.json")' in server_launcher
    assert '"${REPO_ROOT}/tools/launch_gr00t_eval.sh" \\' in server_launcher
    assert '--arena-root "${ARENA_ROOT}"' in server_launcher
    assert "expected_checkpoint_revision" in server_launcher
    assert "runtime_pins_output=" in server_launcher
    assert "E2E_GR00T_CHECKPOINT_SOURCE" in server_launcher
    assert "E2E_GR00T_CHECKPOINT_SOURCE:-trained" in server_launcher
    assert "trainer_state.json" in server_launcher
    assert '--modality-config-path "${modality_config}"' not in server_launcher
    assert "--model-path=*" in server_launcher
    assert runtime["isaac_gr00t_revision"] == "4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e"
    assert project["project"]["optional-dependencies"]["gr00t"] == [
        "msgpack==1.2.3",
        "pyzmq==27.0.1",
    ]


@pytest.mark.parametrize("after_separator", [False, True], ids=["evaluation", "environment"])
@pytest.mark.parametrize(
    ("override", "owned_option"),
    [
        (["--policy_type", "example.BadPolicy"], "--policy_type"),
        (["--policy_type=example.BadPolicy"], "--policy_type"),
        (["--policy_config_yaml_path", "bad.yaml"], "--policy_config_yaml_path"),
        (["--policy_config_yaml_path=bad.yaml"], "--policy_config_yaml_path"),
        (["--enable_cameras"], "--enable_cameras"),
        (["--embodiment", "g1_wbc_agile_joint"], "--embodiment"),
        (["--embodiment=g1_wbc_agile_joint"], "--embodiment"),
    ],
)
def test_gr00t_h2_launcher_rejects_owned_argument_overrides(
    after_separator: bool, override: list[str], owned_option: str
) -> None:
    arguments = [str(REPO_ROOT / "tools/launch_gr00t_eval.sh"), "--robot", "h2"]
    if after_separator:
        arguments.append("--")
    arguments.extend(override)

    result = subprocess.run(arguments, cwd=REPO_ROOT, check=False, capture_output=True, text=True)

    qualifier = " for H2" if owned_option == "--embodiment" else ""
    assert result.returncode == 2
    assert result.stderr.strip() == (
        f"{owned_option} is owned by launch_gr00t_eval.sh{qualifier} and cannot be overridden."
    )


@pytest.mark.parametrize(
    ("passthrough", "expected_error"),
    [
        (["--num_episodes", "1"], "--check does not accept evaluation arguments."),
        (["--", "--num_envs", "1"], "--check does not accept environment arguments."),
    ],
    ids=["evaluation", "environment"],
)
def test_gr00t_check_rejects_both_passthrough_channels(passthrough: list[str], expected_error: str) -> None:
    result = subprocess.run(
        [str(REPO_ROOT / "tools/launch_gr00t_eval.sh"), "--robot", "h2", "--check", *passthrough],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stderr.strip() == expected_error


def test_gr00t_policy_config_matches_the_finetuned_checkpoint_shape() -> None:
    """This repository owns the evaluation config, so its pinned shape must not drift."""

    config = yaml.safe_load((REPO_ROOT / "configs/stages/gr00t/closedloop_eval.yaml").read_text(encoding="utf-8"))
    dataset = yaml.safe_load((REPO_ROOT / "configs/stages/gr00t_lerobot_dataset.yaml").read_text(encoding="utf-8"))

    assert config["action_horizon"] == config["action_chunk_length"] == 40
    assert config["embodiment_tag"] == "NEW_EMBODIMENT"
    assert config["task_mode_name"] == "g1_locomanipulation"
    assert config["pov_cam_name_sim"] == "robot_head_cam_rgb"
    for key in (
        "modality_config_path",
        "policy_joints_config_path",
        "action_joints_config_path",
        "state_joints_config_path",
    ):
        assert (REPO_ROOT / config[key]).is_file(), key
    # Evaluation must read back the joint ordering the dataset converter wrote.
    for key in ("policy_joints_config_path", "action_joints_config_path", "state_joints_config_path"):
        assert config[key] == dataset["robot"][key]

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

"""Host-side contracts for the Isaac AutoData integration."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
import re
import runpy
import subprocess
import sys
import tomllib
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

from e2e_robot_workflows.tasks.static_apple.spec import AUTODATA_AGILE_GYM_ID, AUTODATA_GYM_ID, TASK_NAME

REPO_ROOT = Path(__file__).resolve().parents[1]
AUTODATA_CONFIG_DIR = REPO_ROOT / "configs/stages/autodata"
HAND_JOINT_COUNT = 14
POSE_WIDTH = 7
TELEOP_FIELDS = {"navigate_command", "base_height_command"}
"""Commands only the AGILE lower body produces, which fixed-mode generation cannot record."""


def _load(relative_path: str) -> dict[str, Any]:
    return yaml.safe_load((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_autodata_is_pinned_as_an_external_tool() -> None:
    """AutoData runs from its own checkout, so only a revision can pin it."""

    runtime = tomllib.loads((REPO_ROOT / "runtime.toml").read_text(encoding="utf-8"))["tested"]

    assert runtime["isaac_autodata"] == "0.1.0"
    assert re.fullmatch(r"[0-9a-f]{40}", runtime["isaac_autodata_revision"])


def test_launcher_enforces_the_pin_and_passes_repo_owned_configs() -> None:
    """The launcher is the only entry point, so the guards have to live in it."""

    launcher = (REPO_ROOT / "tools/launch_autodata.sh").read_text(encoding="utf-8")

    assert "isaac_autodata_revision" in launcher
    assert "AutoData revision mismatch" in launcher
    # Isaac Sim would otherwise block on its EULA prompt with no terminal to answer from.
    assert "OMNI_KIT_ACCEPT_EULA=YES" in launcher
    assert AUTODATA_GYM_ID in launcher
    assert AUTODATA_AGILE_GYM_ID in launcher
    # AGILE is the supported default; fixed recordings require an explicit fixed or auto request.
    assert "WBC_MODE=agile" in launcher
    assert "agile (default), or auto" in launcher
    for config in (
        "static_apple_task.yaml",
        "static_apple_left_task.yaml",
        "static_apple_embodiment.yaml",
        "static_apple_agile_embodiment.yaml",
    ):
        assert config in launcher, config
    # AutoData's packages resolve from its checkout rather than site-packages.
    assert 'PYTHONPATH="${AUTODATA_ROOT}' in launcher
    # Generation records video only when Isaac Sim reports cameras enabled, and without video the
    # dataset cannot be converted -- so this is on by default.
    assert "CAMERA_ARGS+=(--enable_cameras)" in launcher
    # Annotation reads the env's own subtask signals rather than waiting for a keyboard.
    assert "ANNOTATE_ARGS+=(--auto --headless)" in launcher
    # The source is checked before Isaac Sim creates an output file or sees the first action.
    assert "e2e_robot_workflows.autodata.dataset_mode" in launcher
    assert "--requested-mode" in launcher
    assert "--requested-hand" in launcher
    assert "PYTHONUNBUFFERED=1" in launcher
    # The how-to reaches AutoData's dataset validator through the launcher, like everything else.
    assert "validate-dataset)" in launcher


def test_arena_generator_preserves_failure_status_through_sim_shutdown() -> None:
    entrypoint = ast.parse(_read("scripts/generate_arena_dataset.py")).body[-1]
    assert isinstance(entrypoint, ast.If)
    events: list[tuple] = []

    class Shutil:
        @staticmethod
        def rmtree(path, *, ignore_errors: bool) -> None:
            events.append(("rmtree", path, ignore_errors))

    class SimulationApp:
        @staticmethod
        def close(*, exit_code: int) -> None:
            events.append(("close", exit_code))

    namespace = {
        "__name__": "__main__",
        "main": lambda: 7,
        "shutil": Shutil,
        "simulation_app": SimulationApp,
        "SNAPSHOT_ROOT": "snapshot",
    }
    try:
        exec(compile(ast.Module(body=[entrypoint], type_ignores=[]), "generate_arena_dataset.py", "exec"), namespace)
    except SystemExit as error:
        assert error.code == 7
    else:
        raise AssertionError("Arena generator entry point did not exit")

    assert events == [("rmtree", "snapshot", True), ("close", 7)]


@pytest.mark.parametrize(
    "collision",
    [
        "result_input",
        "result_output",
        "result_failed",
        "result_descriptor",
        "output_input",
        "output_descriptor",
        "failed_input",
        "failed_descriptor",
        "extension_only_output",
        "all_dot_stem_output",
        "repeated_extension_output",
        "failed_output_hardlink",
    ],
)
def test_arena_generator_rejects_unsafe_artifact_paths_before_app_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision: str,
) -> None:
    source = tmp_path / "source_failed.hdf5"
    descriptor = tmp_path / "task_failed.hdf5"
    output = tmp_path / "generated.hdf5"
    source.write_bytes(b"source")
    descriptor.write_bytes(b"descriptor")

    if collision == "output_input":
        output = source
    elif collision == "output_descriptor":
        output = descriptor
    elif collision == "failed_input":
        output = tmp_path / "source.hdf5"
    elif collision == "failed_descriptor":
        output = tmp_path / "task.hdf5"
    elif collision == "extension_only_output":
        output = tmp_path / ".hdf5"
    elif collision == "all_dot_stem_output":
        output = tmp_path / "..hdf5"
    elif collision == "repeated_extension_output":
        output = tmp_path / "generated.hdf5.hdf5"
    elif collision == "failed_output_hardlink":
        output.write_bytes(b"generated")
        os.link(output, tmp_path / "generated_failed.hdf5")

    class ForbiddenAppLauncher:
        @staticmethod
        def add_app_launcher_args(_parser) -> None:
            return None

        def __init__(self, _args) -> None:
            raise AssertionError("AppLauncher must not start before artifact-path admission")

    fake_isaaclab = types.ModuleType("isaaclab")
    fake_app = types.ModuleType("isaaclab.app")
    fake_app.AppLauncher = ForbiddenAppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_app)
    result = {
        "result_input": source,
        "result_output": output,
        "result_failed": tmp_path / "generated_failed.hdf5",
        "result_descriptor": descriptor,
    }.get(collision, tmp_path / "result.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_arena_dataset.py",
            "--input_file",
            str(source),
            "--output_file",
            str(output),
            "--result_file",
            str(result),
            "--run_id",
            "test-run",
            "--task_descriptor",
            str(descriptor),
            "--manipulation_hand",
            "right",
        ],
    )

    if collision in {"extension_only_output", "all_dot_stem_output", "repeated_extension_output"}:
        with pytest.raises(SystemExit) as error:
            runpy.run_path(str(REPO_ROOT / "scripts/generate_arena_dataset.py"), run_name="collision_test")
        assert error.value.code == 2
    else:
        with pytest.raises(ValueError, match="must differ"):
            runpy.run_path(str(REPO_ROOT / "scripts/generate_arena_dataset.py"), run_name="collision_test")

    assert source.read_bytes() == b"source"
    assert descriptor.read_bytes() == b"descriptor"
    if collision == "failed_output_hardlink":
        failed_output = tmp_path / "generated_failed.hdf5"
        assert output.read_bytes() == b"generated"
        assert output.samefile(failed_output)
    elif output not in {source, descriptor}:
        assert not output.exists()


def test_arena_task_descriptors_select_one_hand_and_retain_failed_attempts() -> None:
    descriptors = {
        "right": _load("configs/stages/autodata/static_apple_arena_task.yaml"),
        "left": _load("configs/stages/autodata/static_apple_arena_left_task.yaml"),
    }

    for hand, descriptor in descriptors.items():
        assert descriptor["generation_policy"]["guarantee_success"] is False
        assert descriptor["generation_policy"]["keep_failed"] is True
        assert set(descriptor["subtasks"]) == {"left", "right"}
        assert descriptor["subtasks"][hand][0]["subtask_term_signal"] == f"{hand}_hand_holds_object"
        assert {subtask["object_ref"] for subtasks in descriptor["subtasks"].values() for subtask in subtasks} == {
            "pick_object"
        }
        assert all(
            subtask["action_noise"] == 0.0 for subtasks in descriptor["subtasks"].values() for subtask in subtasks
        )


def test_h2_task_descriptors_reanchor_after_the_handover() -> None:
    descriptors = {
        "right": _load("configs/stages/autodata/static_apple_arena_h2_task.yaml"),
        "left": _load("configs/stages/autodata/static_apple_arena_h2_left_task.yaml"),
    }

    for hand, descriptor in descriptors.items():
        support_hand = "left" if hand == "right" else "right"
        active_subtasks = descriptor["subtasks"][hand]
        support_subtasks = descriptor["subtasks"][support_hand]
        assert [subtask["object_ref"] for subtask in active_subtasks] == [
            "pick_object",
            "pick_object",
            "placement_target",
        ]
        assert [subtask.get("subtask_term_signal") for subtask in active_subtasks] == [
            f"{hand}_hand_holds_object",
            f"{support_hand}_hand_holds_object",
            None,
        ]
        assert [subtask["object_ref"] for subtask in support_subtasks] == [
            "pick_object",
            "placement_target",
        ]
        assert [subtask.get("subtask_term_signal") for subtask in support_subtasks] == [
            f"{support_hand}_hand_holds_object",
            None,
        ]
        expected_constraints = [
            {
                "constraint_type": "coordination",
                "eef_subtask_constraint_tuple": [[hand, 1], [support_hand, 0]],
                "coordination_scheme": "transform",
                "coordination_scheme_pos_noise_scale": 0.0,
                "coordination_scheme_rot_noise_scale": 0.0,
                "coordination_synchronize_start": False,
            },
            {
                "constraint_type": "coordination",
                "eef_subtask_constraint_tuple": [[hand, 2], [support_hand, 1]],
                "coordination_scheme": "transform",
                "coordination_scheme_pos_noise_scale": 0.0,
                "coordination_scheme_rot_noise_scale": 0.0,
                "coordination_synchronize_start": True,
            },
        ]
        assert descriptor["constraints"] == expected_constraints


def test_launcher_rejects_a_native_mode_for_the_arena_lane() -> None:
    result = subprocess.run(
        [
            str(REPO_ROOT / "tools/launch_autodata.sh"),
            "generate",
            "--lane",
            "arena",
            "--wbc-mode",
            "fixed",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stderr.strip() == "--wbc-mode is only valid for native datasets."


def test_launcher_rejects_camera_free_arena_before_external_preflight(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("UV_BIN", str(tmp_path / "uv-must-not-run"))

    result = subprocess.run(
        [
            str(REPO_ROOT / "tools/launch_autodata.sh"),
            "generate",
            "--lane",
            "arena",
            "--no-cameras",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stderr.strip() == (
        "--no-cameras is only supported for native generation; Arena generation requires cameras."
    )


def test_launcher_keeps_camera_free_native_generation_supported(tmp_path: Path, monkeypatch) -> None:
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/usr/bin/env bash\nprintf 'test-revision\\n'\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    missing_autodata = tmp_path / "missing-autodata"
    monkeypatch.setenv("UV_BIN", str(fake_uv))

    result = subprocess.run(
        [
            str(REPO_ROOT / "tools/launch_autodata.sh"),
            "generate",
            "--lane",
            "native",
            "--no-cameras",
            "--autodata-root",
            str(missing_autodata),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stderr.strip() == f"No Isaac AutoData checkout at {missing_autodata}; pass --autodata-root."


def test_provision_accepts_an_existing_linked_worktree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "autodata-source"
    source.mkdir()
    (source / "scripts").mkdir()
    (source / "scripts/generate_dataset.py").touch()

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", "-C", str(source), *args], check=True, capture_output=True, text=True)

    git("init", "--quiet")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    git("config", "commit.gpgsign", "false")
    git("add", "--all")
    git("commit", "--quiet", "--message", "AutoData")
    revision = git("rev-parse", "HEAD").stdout.strip()
    git("remote", "add", "origin", str(source))
    worktree = tmp_path / "autodata-worktree"
    git("worktree", "add", "--quiet", "--detach", str(worktree), "HEAD")
    assert (worktree / ".git").is_file()

    fake_uv = tmp_path / "uv"
    fake_uv.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' '{revision}'\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    monkeypatch.setenv("UV_BIN", str(fake_uv))

    result = subprocess.run(
        [
            str(REPO_ROOT / "tools/launch_autodata.sh"),
            "provision",
            "--autodata-root",
            str(worktree),
            "--autodata-python",
            str(fake_python),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Cloning Isaac AutoData" not in result.stdout
    assert "Checking out the pinned AutoData revision" in result.stdout


def test_launcher_absolutizes_caller_supplied_paths() -> None:
    """The launcher cds into the AutoData checkout, so a relative path would resolve there.

    ``--input_file`` is the one path a caller passes, and the how-to spells it relative to this
    repository. Rewriting it has to happen before the directory changes.
    """

    launcher = _read("tools/launch_autodata.sh")

    assert 'INVOCATION_DIR="${PWD}"' in launcher
    assert launcher.index('INVOCATION_DIR="${PWD}"') < launcher.index('cd "${AUTODATA_ROOT}"')
    # Both spellings argparse accepts, or the rewrite silently misses one of them.
    assert "--input_file) expect_path=input" in launcher
    assert "--input_file=*)" in launcher
    assert 'EXTRA_ARGS[index]="--input_file=${INPUT_FILE}"' in launcher
    assert "--output_file) expect_path=output" in launcher
    assert launcher.index("expect_path=input") < launcher.index('cd "${AUTODATA_ROOT}"')


def test_interpreter_requirements_are_version_checked() -> None:
    """daqp imports fine at the wrong version and then fails mid-episode, so pin and check it."""

    requirements = _read("tools/autodata_interpreter_requirements.txt")
    launcher = _read("tools/launch_autodata.sh")

    pins = dict(line.split("==", 1) for line in requirements.splitlines() if line.strip() and not line.startswith("#"))
    # Isaac Lab's Pink IK passes warm-start arguments that older daqp rejects outright.
    assert pins["daqp"] == "0.8.5"
    # Imported by this repository's env cfg, which AutoData builds when driving our task.
    assert "loguru" in pins

    assert "autodata_interpreter_requirements.txt" in launcher
    # A presence check would accept the daqp version that fails, so the preflight compares versions.
    assert "specifier.contains" in launcher
    assert "launch_autodata.sh setup" in launcher


def test_generated_datasets_convert_through_the_recording_stage_config() -> None:
    """Generation records what recording records, so one stage config converts both."""

    assert not (AUTODATA_CONFIG_DIR / "lerobot_dataset.yaml").exists(), (
        "a second converter stage config means generated and recorded datasets have drifted apart"
    )

    recorded = _load("configs/stages/gr00t_lerobot_dataset.yaml")
    # The field the converter reads is the articulation-wide command, which only this
    # repository's recorder terms write.
    assert recorded["hdf5_fields"]["action"] == "joint_pos_target"
    # Fixed-mode generation has no AGILE term and so no locomotion commands; the converter
    # tolerates that because these are optional, and the writer prunes their modality groups.
    assert TELEOP_FIELDS <= set(recorded["hdf5_fields"])
    assert TELEOP_FIELDS <= set(recorded["optional_fields"])


def test_generation_records_this_repositorys_action_field() -> None:
    """AutoData replaces the env cfg's recorder, so the gym id has to put ours back.

    Isaac Lab's recorder writes ``processed_actions`` -- each active action term's own output,
    which in fixed mode is the Pink IK term's arm and hand joints alone. The converter's joint
    maps are written against the 43-joint articulation command, so without this the generated
    dataset cannot be converted.
    """

    registrations = _read("src/e2e_robot_workflows/tasks/static_apple/__init__.py")
    factory = _read("src/e2e_robot_workflows/tasks/static_apple/autodata_env.py")
    recorders = _read("src/e2e_robot_workflows/data/recorders.py")

    # The env id is the seam AutoData leaves us: it builds the env through whatever it resolves.
    assert 'entry_point="e2e_robot_workflows.tasks.static_apple.autodata_env:make_autodata_env"' in registrations
    assert "G1StaticAppleAutoDataAgileEnvCfg" in registrations
    assert "add_workflow_recorder_terms(recorders)" in factory

    # Adding terms rather than replacing the manager keeps annotation's own recorder intact.
    assert "def add_workflow_recorder_terms(" in recorders
    assert "joint_pos_target" in recorders

    # Only the GPU bridge check can prove it on a built env, so it installs AutoData's recorder
    # first and then looks for ours.
    bridge = _read("tools/simulation/validate_autodata_bridge.py")
    assert "cfg.recorders = ActionStateRecorderManagerCfg()" in bridge
    assert "record_post_step_joint_pos_target" in bridge


def test_generated_modality_drops_only_the_teleop_columns() -> None:
    """One template serves both runs because the writer emits it minus what a run cannot produce.

    A modality.json naming a column the parquet lacks makes GR00T ``KeyError`` on load, so the
    groups have to go -- but only those, and only for the runs that omit their column.
    """

    from e2e_robot_workflows.gr00t.lerobot_writer import _prune_modality

    template = json.loads(_read("configs/stages/gr00t/modality.json"))
    intact = copy.deepcopy(template)
    # What fixed-mode generation writes: joints, the assembled wrist poses, no teleop commands.
    produced = {"observation.state", "action", "observation.eef_pose"}

    dropped = _prune_modality(template, produced)

    assert set(dropped) == TELEOP_FIELDS
    for group, column in dropped.items():
        assert column.startswith("teleop.")
        assert group not in template["action"]
    assert template["action"] == {k: v for k, v in intact["action"].items() if k not in TELEOP_FIELDS}
    assert {k: v for k, v in template.items() if k != "action"} == {k: v for k, v in intact.items() if k != "action"}

    # An AGILE recording produces those columns, so the same template is emitted whole.
    agile = json.loads(_read("configs/stages/gr00t/modality.json"))
    assert _prune_modality(agile, produced | {f"teleop.{field}" for field in TELEOP_FIELDS}) == {}
    assert agile == intact


def test_embodiment_describes_the_fixed_mode_action_layout() -> None:
    """The descriptor is what AutoData believes the action vector is, so it must match ours."""

    embodiment = _load("configs/stages/autodata/static_apple_embodiment.yaml")
    layout = embodiment["action_layout"]

    assert embodiment["type"] == "absolute_pose_whole_body_bimanual"
    left, right, hands = layout["left_pose_slice"], layout["right_pose_slice"], layout["hand_joints_slice"]
    assert left == [0, POSE_WIDTH]
    assert right == [POSE_WIDTH, 2 * POSE_WIDTH]
    assert hands == [2 * POSE_WIDTH, 2 * POSE_WIDTH + HAND_JOINT_COUNT]
    # Fixed mode pins the root and drops the AGILE term, so there is no base command to pass through.
    assert "passthrough_channels" not in layout

    hand_indices = {side: embodiment["eefs"][side]["gripper_action_indices"] for side in ("left", "right")}
    assert not set(hand_indices["left"]) & set(hand_indices["right"])
    assert sorted(hand_indices["left"] + hand_indices["right"]) == list(range(HAND_JOINT_COUNT))
    for side in ("left", "right"):
        assert embodiment["eefs"][side]["pose_obs_keys"] == {
            "pos": f"{side}_eef_pos",
            "quat": f"{side}_eef_quat",
        }


def test_embodiment_describes_the_agile_action_layout() -> None:
    """AGILE recordings append one four-value locomotion passthrough channel."""

    embodiment = _load("configs/stages/autodata/static_apple_agile_embodiment.yaml")
    layout = embodiment["action_layout"]

    assert embodiment["type"] == "absolute_pose_whole_body_bimanual"
    assert layout["left_pose_slice"] == [0, POSE_WIDTH]
    assert layout["right_pose_slice"] == [POSE_WIDTH, 2 * POSE_WIDTH]
    assert layout["hand_joints_slice"] == [2 * POSE_WIDTH, 2 * POSE_WIDTH + HAND_JOINT_COUNT]
    assert layout["passthrough_channels"] == {"body": [28, 32]}
    assert sorted(
        embodiment["eefs"]["left"]["gripper_action_indices"] + embodiment["eefs"]["right"]["gripper_action_indices"]
    ) == list(range(HAND_JOINT_COUNT))


def test_task_descriptor_matches_the_scene_and_the_annotated_signal() -> None:
    """DexMimicGen indexes subtasks by scene object, so the reference must exist in the scene."""

    task = _load("configs/stages/autodata/static_apple_task.yaml")
    env_cfg = _read("src/e2e_robot_workflows/tasks/static_apple/env_cfg.py")
    variant = _read("src/e2e_robot_workflows/tasks/static_apple/autodata_env_cfg.py")

    assert task["name"] == TASK_NAME
    assert task["algo"] == "dexmimicgen"
    assert task["generation_policy"]["guarantee_success"] is True

    for subtasks in task["subtasks"].values():
        for subtask in subtasks:
            # The apple is registered under this name by StaticAppleSceneCfg.
            assert subtask["object_ref"] == "object"
            assert f"    {subtask['object_ref']} = RigidObjectCfg(" in env_cfg

    # The right hand does the transfer in two phases; only the boundary between them is signalled.
    right, left = task["subtasks"]["right"], task["subtasks"]["left"]
    assert len(right) == 2
    assert "subtask_term_signal" not in right[1]
    assert len(left) == 1 and "subtask_term_signal" not in left[0]

    # Annotation looks the signal up by name in the env's observation group, so a rename on either
    # side has to be a rename on both.
    assert f"    {right[0]['subtask_term_signal']} = ObsTerm(" in variant


def test_left_hand_task_descriptor_mirrors_the_transfer() -> None:
    """Either XR hand can perform the transfer without changing task semantics."""

    task = _load("configs/stages/autodata/static_apple_left_task.yaml")
    variant = _read("src/e2e_robot_workflows/tasks/static_apple/autodata_env_cfg.py")
    left, right = task["subtasks"]["left"], task["subtasks"]["right"]

    assert task["name"] == TASK_NAME
    assert len(left) == 2
    assert left[0]["subtask_term_signal"] == "left_hand_holds_object"
    assert "subtask_term_signal" not in left[1]
    assert len(right) == 1 and "subtask_term_signal" not in right[0]
    assert "    left_hand_holds_object = ObsTerm(" in variant


def test_autodata_env_variant_supplies_what_generation_needs() -> None:
    """AutoData accepts a None success term silently, so the variant must set a real one."""

    variant = _read("src/e2e_robot_workflows/tasks/static_apple/autodata_env_cfg.py")
    base = _read("src/e2e_robot_workflows/tasks/static_apple/env_cfg.py")

    # Recording clears the term; generation cannot score attempts without it.
    assert "self.terminations.success = None" in base
    assert "self.terminations.success = DoneTerm(" in variant
    assert "object_placed_on_target" in variant
    # A fixed apple pose would make every generated trial a copy of its source.
    assert 'pose_range["x"] = (-GENERATION_SPAWN_RANGE_M, GENERATION_SPAWN_RANGE_M)' in variant
    assert 'pose_range["y"] = (-GENERATION_SPAWN_RANGE_M, GENERATION_SPAWN_RANGE_M)' in variant
    # Automatic annotation reads the signals by name from this group, so it cannot be concatenated.
    assert "self.observations.subtask_terms = SubtaskTermsCfg()" in variant
    assert "self.concatenate_terms = False" in variant
    # Separate registered configs make the action space unambiguous before managers are built.
    assert 'wbc_mode: WbcMode = "fixed"' in variant
    assert "class G1StaticAppleAutoDataAgileEnvCfg" in variant
    assert 'wbc_mode: WbcMode = "agile"' in variant


def test_hold_signals_and_release_test_use_configured_gripper_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect the instantiated Isaac Lab terms rather than matching their source text."""

    class Term:
        def __init__(self, *, func: object, params: dict[str, object]) -> None:
            self.func = func
            self.params = params

    class BaseEnvCfg:
        def __init__(self) -> None:
            self.observations = types.SimpleNamespace()
            self.terminations = types.SimpleNamespace()
            self.events = types.SimpleNamespace(
                reset_object=types.SimpleNamespace(params={"pose_range": {}}),
            )
            self.__post_init__()

        def __post_init__(self) -> None:
            pass

    object_held_by_link = object()
    object_placed_on_target = object()
    modules = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.managers": types.ModuleType("isaaclab.managers"),
        "isaaclab.utils": types.ModuleType("isaaclab.utils"),
        "e2e_robot_workflows.tasks.static_apple.env_cfg": types.ModuleType("env_cfg"),
        "e2e_robot_workflows.tasks.static_apple.observations": types.ModuleType("observations"),
        "e2e_robot_workflows.tasks.static_apple.terminations": types.ModuleType("terminations"),
    }
    modules["isaaclab.managers"].ObservationGroupCfg = object
    modules["isaaclab.managers"].ObservationTermCfg = Term
    modules["isaaclab.managers"].TerminationTermCfg = Term
    modules["isaaclab.utils"].configclass = lambda cls: cls
    modules["e2e_robot_workflows.tasks.static_apple.env_cfg"].G1StaticAppleEnvCfg = BaseEnvCfg
    modules["e2e_robot_workflows.tasks.static_apple.env_cfg"].WbcMode = str
    modules["e2e_robot_workflows.tasks.static_apple.observations"].object_held_by_link = object_held_by_link
    modules["e2e_robot_workflows.tasks.static_apple.terminations"].object_placed_on_target = object_placed_on_target
    for name, module in modules.items():
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    source = REPO_ROOT / "src/e2e_robot_workflows/tasks/static_apple/autodata_env_cfg.py"
    spec = importlib.util.spec_from_file_location("_test_autodata_env_cfg", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from e2e_robot_workflows.robots.g1.config import load_g1_robot_config
    from e2e_robot_workflows.tasks.static_apple.spec import ENVIRONMENT

    links = load_g1_robot_config().task_link_names
    terms = module.SubtaskTermsCfg()
    terms.__post_init__()
    success = module.G1StaticAppleAutoDataEnvCfg().terminations.success

    for hand, link in zip(("left", "right"), links, strict=True):
        term = getattr(terms, f"{hand}_hand_holds_object")
        assert term.func is object_held_by_link
        params = term.params
        assert params["task_link_name"] == link
        assert params["grasp_distance_m"] == ENVIRONMENT.release_clearance_m
    assert success.func is object_placed_on_target
    assert success.params["task_link_names"] == links
    assert success.params["release_clearance_m"] == ENVIRONMENT.release_clearance_m


def test_success_term_reads_the_targets_live_pose() -> None:
    """The plate is a dynamic body the arm can nudge while reaching past its rim, so measuring
    against its spawn constant would score a correct placement as a failure."""

    term = _read("src/e2e_robot_workflows/tasks/static_apple/terminations.py")
    base = _read("src/e2e_robot_workflows/tasks/static_apple/env_cfg.py")
    variant = _read("src/e2e_robot_workflows/tasks/static_apple/autodata_env_cfg.py")

    assert 'DEFAULT_TARGET_CFG = SceneEntityCfg("destination")' in term
    assert "target_pos = target.data.root_pos_w.torch - env.scene.env_origins" in term
    # The centre is no longer something the caller can hard-code.
    assert "target_xy" not in variant
    # env.scene resolves that name against the scene cfg's own attribute.
    assert "    destination = RigidObjectCfg(" in base


def test_success_term_requires_apple_plate_contact_and_table_support() -> None:
    """AutoData only runs the success term, so it must reject apple/plate pairs on the floor."""
    term = _read("src/e2e_robot_workflows/tasks/static_apple/terminations.py")
    variant = _read("src/e2e_robot_workflows/tasks/static_apple/autodata_env_cfg.py")
    bridge = _read("tools/simulation/validate_autodata_bridge.py")

    assert 'DEFAULT_CONTACT_SENSOR_CFG = SceneEntityCfg("object_contact_sensor")' in term
    assert "contact_sensor.data.force_matrix_w.torch" in term
    assert "target_pos[:, 2] > min_target_height_m" in term
    assert "object_pos[:, 2] > min_object_height_m" in term
    assert '"min_contact_force": ENVIRONMENT.success_force_threshold' in variant
    assert '"min_target_height_m": MIN_SUPPORTED_HEIGHT_M' in variant
    assert '"min_object_height_m": MIN_SUPPORTED_HEIGHT_M' in variant
    assert 'result["stage"] = "reject_dropped_target"' in bridge
    assert "a target below its support counted as placement" in bridge


def test_success_term_params_match_its_signature() -> None:
    """The termination manager calls the term as ``func(env, **params)``, so a name that drifts on
    one side would surface as a TypeError on the first step of a generation run."""

    module = ast.parse(_read("src/e2e_robot_workflows/tasks/static_apple/terminations.py"))
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "object_placed_on_target"
    )
    argument_names = [argument.arg for argument in function.args.args]
    accepted = set(argument_names) - {"env"}
    required = set(argument_names[: len(argument_names) - len(function.args.defaults)]) - {"env"}

    variant = ast.parse(_read("src/e2e_robot_workflows/tasks/static_apple/autodata_env_cfg.py"))
    call = next(
        node for node in ast.walk(variant) if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "DoneTerm"
    )
    passed = {key.value for keyword in call.keywords if keyword.arg == "params" for key in keyword.value.keys}

    assert passed, "the success term must be configured with explicit params"
    assert not passed - accepted, f"params the term does not accept: {sorted(passed - accepted)}"
    assert not required - passed, f"required params not supplied: {sorted(required - passed)}"

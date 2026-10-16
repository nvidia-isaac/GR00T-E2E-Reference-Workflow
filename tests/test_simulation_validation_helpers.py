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

"""Host-side tests for extracted simulation validation helpers."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SIMULATION_TOOLS = REPO_ROOT / "tools" / "simulation"


@pytest.fixture
def simulation_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, Any]:
    monkeypatch.syspath_prepend(str(SIMULATION_TOOLS))
    for module_name in ("agile_validation", "pink_validation", "validate_agile_locomotion"):
        sys.modules.pop(module_name, None)
    agile_validation = importlib.import_module("agile_validation")
    pink_validation = importlib.import_module("pink_validation")
    agile_entrypoint = importlib.import_module("validate_agile_locomotion")
    return agile_validation, pink_validation, agile_entrypoint


@pytest.fixture
def h2_validation_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(SIMULATION_TOOLS))
    sys.modules.pop("validate_arena_h2", None)
    return importlib.import_module("validate_arena_h2")


def test_h2_station_keeping_adds_all_requested_steps(h2_validation_module: Any) -> None:
    assert h2_validation_module._expected_completed_steps("h2_agile_joint", 200) == 200
    assert h2_validation_module._expected_completed_steps("h2_agile_pink", 60) == 2580
    assert h2_validation_module._expected_completed_steps("h2_agile_pink", 60, True) == 3580
    assert h2_validation_module._expected_completed_steps("h2_agile_pink", 60, False, True) == 2964
    assert h2_validation_module._expected_completed_steps("h2_agile_pink", 60, True, True) == 3964


def test_h2_joint_walk_holds_one_upper_body_reference_instead_of_tracking_sag() -> None:
    tree = ast.parse((SIMULATION_TOOLS / "validate_arena_h2.py").read_text())
    walk = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_qualify_agile_walk")
    captures = [
        node
        for node in ast.walk(walk)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "h2_joint_hold_action"
    ]
    assert len(captures) == 1
    for loop in (node for node in ast.walk(walk) if isinstance(node, ast.For)):
        assert captures[0] not in list(ast.walk(loop))


def test_h2_station_keeping_rejects_non_pink_cli(h2_validation_module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["validate_arena_h2.py", "--station-keeping"])
    with pytest.raises(SystemExit) as error:
        h2_validation_module._parse_args()
    assert error.value.code == 2


def test_h2_startup_rejects_station_keeping(
    h2_validation_module: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_arena_h2.py", "--embodiment", "h2_agile_pink", "--mimic", "--startup-only", "--station-keeping"],
    )
    with pytest.raises(SystemExit) as error:
        h2_validation_module._parse_args()
    assert error.value.code == 2
    assert "--startup-only cannot be combined with --station-keeping" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ("--startup-only", "--station-keeping"))
def test_h2_validation_accepts_each_pink_probe(
    h2_validation_module: Any, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["validate_arena_h2.py", "--embodiment", "h2_agile_pink", "--mimic", flag])
    args = h2_validation_module._parse_args()
    assert args.startup_only == (flag == "--startup-only")
    assert args.station_keeping == (flag == "--station-keeping")


def test_h2_render_only_refresh_proves_render_without_state_or_physics_change(h2_validation_module: Any) -> None:
    torch = pytest.importorskip("torch", reason="torch is not installed on the CPU-only host image")
    calls: list[str] = []

    class Simulation:
        render_generation = 11
        physics_steps = 23

        def get_physics_step_count(self) -> int:
            return self.physics_steps

        def render(self) -> None:
            calls.append("render")
            self.render_generation += 1

    scene_state = {"rigid_object": {"apple": {"root_pose": torch.tensor([[1.0, 2.0, 3.0]])}}}
    env = SimpleNamespace(
        _sim_step_counter=19,
        sim=Simulation(),
        scene=SimpleNamespace(get_state=lambda: scene_state),
        obs_buf={"policy": torch.tensor([[4.0, 5.0]])},
    )

    metrics = h2_validation_module._qualify_render_only_refresh(env, torch, lambda value: value.sim.render())

    assert calls == ["render"]
    assert metrics == {
        "environment_step_counter_after": 19,
        "environment_step_counter_before": 19,
        "observation_tensor_count": 1,
        "observations_unchanged": True,
        "physics_step_count_after": 23,
        "physics_step_count_before": 23,
        "render_generation_after": 12,
        "render_generation_before": 11,
        "render_generation_delta": 1,
        "scene_state_unchanged": True,
        "scene_tensor_count": 1,
    }


def test_h2_render_only_refresh_rejects_an_observation_mutation(h2_validation_module: Any) -> None:
    torch = pytest.importorskip("torch", reason="torch is not installed on the CPU-only host image")

    class Simulation:
        render_generation = 3

        def get_physics_step_count(self) -> int:
            return 7

    env = SimpleNamespace(
        _sim_step_counter=5,
        sim=Simulation(),
        scene=SimpleNamespace(get_state=lambda: {"robot": {"joint_position": torch.tensor([[0.0]])}}),
        obs_buf={"policy": torch.tensor([[1.0]])},
    )

    def mutate_observation(value: Any) -> None:
        value.obs_buf["policy"].add_(1.0)
        value.sim.render_generation += 1

    with pytest.raises(AssertionError, match="observation buffer changed"):
        h2_validation_module._qualify_render_only_refresh(env, torch, mutate_observation)


def test_h2_camera_temporal_score_accepts_a_repeatable_current_pose_match(h2_validation_module: Any) -> None:
    metrics = h2_validation_module._score_camera_temporal_alignment(
        reference_separation=20.0,
        repeat_noise=1.0,
        candidate_to_previous=18.0,
        candidate_to_current=2.0,
    )

    assert metrics == {
        "status": "aligned_to_current_pose",
        "current_alignment_score": pytest.approx(0.8),
        "current_distance_limit": pytest.approx(10.0),
        "noise_limit": pytest.approx(3.0),
    }


def test_h2_camera_temporal_score_rejects_a_previous_pose_match(h2_validation_module: Any) -> None:
    with pytest.raises(AssertionError, match="aligned to the previous head pose"):
        h2_validation_module._score_camera_temporal_alignment(
            reference_separation=20.0,
            repeat_noise=1.0,
            candidate_to_previous=2.0,
            candidate_to_current=18.0,
        )


@pytest.mark.parametrize(
    ("candidate_to_previous", "candidate_to_current"),
    [(10.0, 10.0), (30.0, 18.0)],
    ids=("equidistant", "too-far-from-current"),
)
def test_h2_camera_temporal_score_rejects_ambiguous_frames(
    h2_validation_module: Any,
    candidate_to_previous: float,
    candidate_to_current: float,
) -> None:
    with pytest.raises(AssertionError, match="temporal alignment is ambiguous"):
        h2_validation_module._score_camera_temporal_alignment(
            reference_separation=20.0,
            repeat_noise=1.0,
            candidate_to_previous=candidate_to_previous,
            candidate_to_current=candidate_to_current,
        )


@pytest.mark.parametrize(
    ("reference_separation", "repeat_noise", "message"),
    [(7.99, 0.0, "not visually distinct"), (20.0, 3.01, "not repeatable")],
    ids=("weak-pose-signal", "unstable-reference-renders"),
)
def test_h2_camera_temporal_score_requires_informative_references(
    h2_validation_module: Any,
    reference_separation: float,
    repeat_noise: float,
    message: str,
) -> None:
    with pytest.raises(AssertionError, match=message):
        h2_validation_module._score_camera_temporal_alignment(
            reference_separation=reference_separation,
            repeat_noise=repeat_noise,
            candidate_to_previous=18.0,
            candidate_to_current=2.0,
        )


def test_agile_phase_sequence_and_total_steps_are_explicit(simulation_modules: tuple[Any, Any, Any]) -> None:
    agile_validation, _pink_validation, _agile_entrypoint = simulation_modules

    assert agile_validation.PHASES == (
        ("settle", 32, (0.0, 0.0, 0.0, 0.72)),
        ("backward", 64, (-0.2, 0.0, 0.0, 0.72)),
        ("stop", 32, (0.0, 0.0, 0.0, 0.72)),
        ("turn", 64, (0.0, 0.0, 0.4, 0.72)),
    )
    assert agile_validation.RESET_SETTLE_STEPS == 48
    assert agile_validation.TOTAL_CONTROL_STEPS == 240


def test_both_runtimes_are_held_to_one_locomotion_standard() -> None:
    """The drive and its limits must not fork per runtime.

    Two drivers now run this schedule, on the built-in task and on the Arena-composed
    environment. A behavior threshold redefined in either driver would let one runtime
    pass a drive the other could not, which is exactly the comparison these validations
    exist to make.
    """

    shared = ("HIP_HEIGHT_M", "MIN_BACKWARD_PROGRESS_M", "MAX_STOP_TAIL_PLANAR_SPEED_M_S", "MIN_TURN_YAW_DELTA_RAD")
    for driver in ("validate_agile_locomotion.py", "validate_arena_locomotion.py"):
        source = (SIMULATION_TOOLS / driver).read_text(encoding="utf-8")
        for constant in shared:
            assert f"\n{constant} = " not in source, f"{driver} redefines the shared limit {constant}"
        assert "from agile_validation import (" in source


def test_gr00t_checker_rejects_a_missing_config_argument() -> None:
    checker = REPO_ROOT / "tools/check_gr00t_eval.py"

    completed = subprocess.run([sys.executable, str(checker)], check=False, capture_output=True, text=True)

    assert completed.returncode != 0
    assert completed.stderr.strip() == "usage: check_gr00t_eval.py <policy_config_yaml> [checkpoint_root]"


@pytest.fixture
def gr00t_checker() -> Any:
    checker_path = REPO_ROOT / "tools/check_gr00t_eval.py"
    spec = importlib.util.spec_from_file_location("check_gr00t_eval", checker_path)
    assert spec is not None and spec.loader is not None
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    return checker


def _valid_gr00t_checkpoint_contract(checker: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_modality = {
        "delta_indices": [0],
        "modality_keys": ["left_arm"],
        "sin_cos_embedding_keys": None,
        "mean_std_embedding_keys": None,
        "action_configs": None,
    }
    expected = {name: dict(expected_modality) for name in checker.MODALITY_NAMES}
    checkpoint = {name: dict(expected_modality) for name in checker.MODALITY_NAMES}
    statistics = {
        modality: {"left_arm": {statistic: [0.0, 0.0] for statistic in checker.REQUIRED_STATISTICS}}
        for modality in ("state", "action")
    }
    return expected, checkpoint, statistics


def test_gr00t_checker_accepts_complete_checkpoint_statistics(gr00t_checker: Any) -> None:
    expected, checkpoint, statistics = _valid_gr00t_checkpoint_contract(gr00t_checker)
    dimensions = {"state": {"left_arm": 2}, "action": {"left_arm": 2}}

    gr00t_checker._require_checkpoint_contract(expected, checkpoint, dimensions, statistics)


def test_gr00t_checker_rejects_modality_drift(gr00t_checker: Any) -> None:
    expected, checkpoint, statistics = _valid_gr00t_checkpoint_contract(gr00t_checker)
    dimensions = {"state": {"left_arm": 2}, "action": {"left_arm": 2}}
    checkpoint["video"]["modality_keys"] = ["different"]

    with pytest.raises(SystemExit, match="Checkpoint modality mismatch: video"):
        gr00t_checker._require_checkpoint_contract(expected, checkpoint, dimensions, statistics)


@pytest.mark.parametrize("statistic", ("mean", "std", "min", "max", "q01", "q99"))
def test_gr00t_checker_requires_every_checkpoint_statistic(gr00t_checker: Any, statistic: str) -> None:
    expected, checkpoint, statistics = _valid_gr00t_checkpoint_contract(gr00t_checker)
    dimensions = {"state": {"left_arm": 2}, "action": {"left_arm": 2}}
    del statistics["state"]["left_arm"][statistic]

    with pytest.raises(SystemExit, match=r"Checkpoint statistics mismatch: state\.left_arm must contain exactly"):
        gr00t_checker._require_checkpoint_contract(expected, checkpoint, dimensions, statistics)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([[0.0, 0.0]], "must be a rank-one vector of width 2"),
        ([0.0], "must be a rank-one vector of width 2"),
        ([0.0, float("nan")], "must contain only finite numbers"),
        ([0.0, float("inf")], "must contain only finite numbers"),
        ([0.0, "0.0"], "must contain only finite numbers"),
    ],
    ids=("wrong-rank", "wrong-width", "nan", "infinity", "non-numeric"),
)
def test_gr00t_checker_rejects_malformed_checkpoint_statistics(
    gr00t_checker: Any, values: list[Any], message: str
) -> None:
    expected, checkpoint, statistics = _valid_gr00t_checkpoint_contract(gr00t_checker)
    dimensions = {"state": {"left_arm": 2}, "action": {"left_arm": 2}}
    statistics["action"]["left_arm"]["q99"] = values

    with pytest.raises(SystemExit, match=message):
        gr00t_checker._require_checkpoint_contract(expected, checkpoint, dimensions, statistics)


def test_gr00t_checker_labels_config_only_and_checkpoint_validation(gr00t_checker: Any) -> None:
    assert gr00t_checker._checkpoint_summary(None, 75, 61) == (
        "validation_scope=config_only, checkpoint_metadata=not_checked, "
        "expected_checkpoint_state=75, expected_checkpoint_action=61"
    )
    assert gr00t_checker._checkpoint_summary("/checkpoint", 75, 61) == (
        "validation_scope=checkpoint, checkpoint_metadata=validated, "
        "expected_checkpoint_state=75, expected_checkpoint_action=61"
    )


def test_gr00t_checker_orders_only_an_exact_live_joint_limit_map(gr00t_checker: Any) -> None:
    names = ["second", "first"]
    limits = {"first": (-1.0, 1.0), "second": (-2.0, 2.0)}

    assert gr00t_checker._ordered_live_joint_limits(names, limits) == [(-2.0, 2.0), (-1.0, 1.0)]

    with pytest.raises(SystemExit, match="exact 75-joint plant"):
        gr00t_checker._ordered_live_joint_limits(names, {"first": (-1.0, 1.0)})


def test_gr00t_server_rejects_an_unknown_checkpoint_source() -> None:
    launcher = REPO_ROOT / "tools/launch_gr00t_server.sh"

    completed = subprocess.run(
        [
            str(launcher),
            "--arena-root",
            "/unused/arena",
            "--gr00t-root",
            "/unused/gr00t",
            "--model-path",
            "/unused/checkpoint",
            "--checkpoint-source",
            "unknown",
            "--check",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stderr.strip() == "E2E_GR00T_CHECKPOINT_SOURCE must be 'released', 'published', or 'trained'."


def test_run_phase_aggregates_steps_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
    simulation_modules: tuple[Any, Any, Any],
) -> None:
    agile_validation, _pink_validation, _agile_entrypoint = simulation_modules
    commands: list[tuple[float, float, float, float]] = []
    upper_checks: list[str] = []
    step_index = 0
    metrics = (
        {
            "body_planar_speed_m_s": 0.3,
            "body_yaw_rate_rad_s": -0.2,
            "object_local_z_m": 0.9,
            "pelvis_height_m": 0.7,
            "pelvis_tilt_rad": 0.1,
            "policy_output_magnitude": 0.01,
        },
        {
            "body_planar_speed_m_s": 0.2,
            "body_yaw_rate_rad_s": 0.1,
            "object_local_z_m": 0.8,
            "pelvis_height_m": 0.65,
            "pelvis_tilt_rad": 0.2,
            "policy_output_magnitude": 0.03,
        },
        {
            "body_planar_speed_m_s": 0.1,
            "body_yaw_rate_rad_s": -0.05,
            "object_local_z_m": 0.85,
            "pelvis_height_m": 0.75,
            "pelvis_tilt_rad": 0.15,
            "policy_output_magnitude": 0.02,
        },
    )

    def make_action(command: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        commands.append(command)
        return command

    def assert_runtime_state(**_kwargs: Any) -> dict[str, float]:
        nonlocal step_index
        value = metrics[step_index]
        step_index += 1
        return value

    monkeypatch.setattr(agile_validation, "_assert_runtime_state", assert_runtime_state)

    class FakeEnvironment:
        def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
            return action, "reward", "terminated", "truncated", {}

    result = {"completed_steps": 4}
    command = (-0.2, 0.0, 0.0, 0.72)
    outcome = agile_validation.run_phase(
        "backward",
        3,
        command,
        stage="phase_backward",
        env=FakeEnvironment(),
        runtime=agile_validation.LocomotionRuntime(
            make_action=make_action,
            pick_object_name="object",
            assert_policy_outputs=lambda: 0.0,
            assert_upper_body=lambda: upper_checks.append("upper body"),
        ),
        support_z=0.5,
        result=result,
        torch=object(),
        quat_apply=object(),
    )

    assert commands == [command, command, command]
    assert result == {"completed_steps": 7, "stage": "phase_backward"}
    assert upper_checks == ["upper body"] * 3
    assert outcome.name == "backward"
    assert outcome.final_state == metrics[-1]
    assert outcome.min_object_z == 0.8
    assert outcome.planar_speeds == [0.3, 0.2, 0.1]
    assert outcome.absolute_yaw_rates == [0.2, 0.1, 0.05]
    assert outcome.summary == {
        "command": list(command),
        "max_pelvis_height_m": 0.75,
        "max_pelvis_tilt_rad": 0.2,
        "max_policy_output_magnitude": 0.03,
        "min_pelvis_height_m": 0.65,
        "steps": 3,
    }


def test_tail_motion_uses_only_the_requested_tail(simulation_modules: tuple[Any, Any, Any]) -> None:
    agile_validation, _pink_validation, _agile_entrypoint = simulation_modules
    outcome = agile_validation.PhaseOutcome(
        name="stop",
        summary={},
        final_state={},
        min_object_z=0.0,
        planar_speeds=[1.0, 2.0, 3.0],
        absolute_yaw_rates=[0.1, 0.2, 0.3],
    )

    assert agile_validation.tail_motion(outcome, 2) == (2.5, 0.25)
    with pytest.raises(AssertionError, match="at least one"):
        agile_validation.tail_motion(outcome, 0)
    with pytest.raises(AssertionError, match="too short"):
        agile_validation.tail_motion(outcome, 4)


def test_live_pink_action_refreshes_wrist_and_hand_state(simulation_modules: tuple[Any, Any, Any]) -> None:
    _agile_validation, pink_validation, _agile_entrypoint = simulation_modules
    body_state = np.arange(14, dtype=np.float64).reshape(1, 2, 7)
    joint_positions = np.array([[0.25, 0.5, 0.75]], dtype=np.float64)
    robot = SimpleNamespace(
        body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
        data=SimpleNamespace(
            body_link_state_w=SimpleNamespace(torch=body_state),
            joint_pos=SimpleNamespace(torch=joint_positions),
        ),
        find_joints=lambda _names: ([0, 2], ["left_hand", "right_hand"]),
    )
    action_term = SimpleNamespace(
        action_dim=16,
        hand_joint_dim=2,
        cfg=SimpleNamespace(hand_joint_names=[".*_hand"]),
    )
    env = SimpleNamespace(num_envs=1, device="cpu", scene={"robot": robot})
    fake_torch = SimpleNamespace(zeros=lambda shape, device: np.zeros(shape, dtype=np.float64))

    handles = pink_validation.resolve_g1_pink_handles(robot, action_term)
    first_action = pink_validation.make_live_pink_action(env, action_term, handles, fake_torch)
    body_state[0, 0, 0] = 42.0
    joint_positions[0, 2] = 1.5
    second_action = pink_validation.make_live_pink_action(env, action_term, handles, fake_torch)

    np.testing.assert_array_equal(first_action[0, :14], np.arange(14, dtype=np.float64))
    np.testing.assert_array_equal(first_action[0, 14:], [0.25, 0.75])
    assert second_action[0, 0] == 42.0
    assert second_action[0, 15] == 1.5
    assert first_action[0, 0] == 0.0
    assert first_action[0, 15] == 0.75

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

"""Drive a free-base humanoid under an AGILE locomotion policy (arrow keys to walk).

Holds the upper body where it is while the locomotion policy balances and walks the legs:

    uv run scripts/locomotion_with_agile.py --viz kit
    ./tools/launch_arena_locomotion.sh --robot h2 --viz kit
    ./tools/launch_arena_locomotion.sh --viz kit   # the G1 drive, Arena-composed

The built-in and Arena G1 paths remain distinct. H2 has only the Arena path and uses
its pinned three-number ``[vx, vy, wz]`` AGILE export.

Controls (window focused):  Up/Down = walk forward/back, Left/Right = turn,
A/D = strafe. Release keys to stop and balance in place.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from loguru import logger

# Register Pinocchio bindings before Isaac Sim loads (Isaac Lab #4090).
from e2e_robot_workflows.arena.embodiment_capabilities import resolve_locomotion_embodiment
from e2e_robot_workflows.compat import warmup_pinocchio_bindings
from e2e_robot_workflows.tasks.static_apple.spec import GYM_ID

warmup_pinocchio_bindings()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drive a humanoid (balance/walk) under an AGILE policy.")
parser.add_argument(
    "--robot",
    choices=("g1", "h2"),
    default="g1",
    help="Which robot to drive. H2 always uses the Arena embodiment.",
)
parser.add_argument("--steps", type=int, default=0, help="Steps to run (0 = until the window closes).")
parser.add_argument("--speed", type=float, default=0.4, help="Walk speed for arrow keys (m/s).")
parser.add_argument("--turn", type=float, default=0.6, help="Turn rate for arrow keys (rad/s).")
parser.add_argument(
    "--hip_height",
    type=float,
    default=None,
    help="Base hip-height command (m), G1 only; the H2 export has no height channel. Defaults to 0.72.",
)
parser.add_argument("--vx", type=float, default=0.0, help="Constant base forward speed (m/s), added to keys.")
parser.add_argument("--vy", type=float, default=0.0, help="Constant base lateral speed (m/s), added to keys.")
parser.add_argument("--wz", type=float, default=0.0, help="Constant base yaw rate (rad/s), added to keys.")
parser.add_argument(
    "--arena",
    action="store_true",
    help="Compose the environment through Isaac Lab Arena instead of the built-in task.",
)
parser.add_argument(
    "--arena_embodiment",
    type=str,
    default=None,
    help="Arena embodiment to compose with --arena. Defaults to the joint layout.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.robot == "h2" and args_cli.hip_height is not None:
    parser.error("--hip_height is G1 only: the H2 AGILE export takes [vx, vy, wz] with no height channel.")
try:
    args_cli.arena_embodiment = resolve_locomotion_embodiment(
        robot=args_cli.robot,
        arena_requested=args_cli.arena,
        requested_embodiment=args_cli.arena_embodiment,
    )
except ValueError as error:
    parser.error(f"{error}.")
args_cli.arena = args_cli.arena_embodiment is not None
G1_HIP_HEIGHT_M = 0.72 if args_cli.hip_height is None else args_cli.hip_height

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import e2e_robot_workflows.tasks.static_apple  # noqa: F401 (registers the task)
from e2e_robot_workflows.tasks.static_apple.env_cfg import make_env_cfg


def disable_time_out(env_cfg: object) -> None:
    """Drop the episode timeout so the session runs until the operator stops it."""

    terminations = env_cfg.terminations
    if getattr(terminations, "time_out", None) is not None:
        terminations.time_out = None


class ArrowKeyDriver:
    """Read held arrow/A/D keys and turn them into an AGILE velocity command.

    Best-effort: if no app window/keyboard is available (headless), it is a no-op.
    """

    def __init__(self) -> None:
        self._held: set = set()
        self._sub = None
        try:
            import carb.input
            import omni.appwindow

            self._ci = carb.input
            keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._sub = carb.input.acquire_input_interface().subscribe_to_keyboard_events(keyboard, self._on_event)
        except Exception as exc:  # noqa: BLE001 - keyboard is optional (headless)
            logger.info(f"[locomotion_with_agile] keyboard driving unavailable: {exc}")

    def _on_event(self, event, *args) -> bool:
        if event.type == self._ci.KeyboardEventType.KEY_PRESS:
            self._held.add(event.input)
        elif event.type == self._ci.KeyboardEventType.KEY_RELEASE:
            self._held.discard(event.input)
        return True

    def command(self, speed: float, turn: float) -> tuple[float, float, float]:
        if self._sub is None:
            return 0.0, 0.0, 0.0
        k = self._ci.KeyboardInput
        held = self._held
        vx = speed * ((k.UP in held) - (k.DOWN in held))
        vy = speed * ((k.A in held) - (k.D in held))
        wz = turn * ((k.LEFT in held) - (k.RIGHT in held))
        return vx, vy, wz


def build_builtin_env() -> tuple[gym.Env, Callable[[float, float, float], torch.Tensor]]:
    """Compose the built-in AGILE task and return it with its command builder."""

    cfg = make_env_cfg(wbc_mode="agile", device=args_cli.device, num_envs=1)
    disable_time_out(cfg)
    env = gym.make(GYM_ID, cfg=cfg).unwrapped
    env.reset()

    # Action layout: [upper_body_ik(2x7 wrist poses + 14 hand joints = 28),
    # lower_body_joint_pos(4: vx, vy, wz, hip_height)]. Derive the slice
    # boundaries from the action manager instead of hardcoding.
    upper_dim = env.action_manager.get_term("upper_body_ik").action_dim
    total_dim = env.action_manager.total_action_dim

    robot = env.scene["robot"]
    names = robot.body_names
    left_wrist = names.index("left_wrist_yaw_link")
    right_wrist = names.index("right_wrist_yaw_link")

    def make_action(vx: float, vy: float, wz: float) -> torch.Tensor:
        # Pink IK target = the wrists' current world poses (arms stay put, never an
        # infeasible target), hands open; trailing 4 dims = the AGILE command.
        action = torch.zeros((env.num_envs, total_dim), device=env.device)
        action[0, 0:7] = robot.data.body_link_state_w.torch[0, left_wrist, :7]
        action[0, 7:14] = robot.data.body_link_state_w.torch[0, right_wrist, :7]
        action[0, upper_dim:] = torch.tensor([vx, vy, wz, G1_HIP_HEIGHT_M], device=env.device)
        return action

    return env, make_action


def build_arena_env() -> tuple[gym.Env, Callable[[float, float, float], torch.Tensor]]:
    """Compose the selected Arena robot and return it with its command builder."""

    from e2e_robot_workflows.arena.composition import compose
    from e2e_robot_workflows.arena.g1_actions import hold_action, set_navigate_command

    embodiment = args_cli.arena_embodiment
    if embodiment is None:
        raise RuntimeError("Arena locomotion route has no embodiment")
    logger.info(f"Arena embodiment: {embodiment}")
    composed = compose(embodiment=embodiment, device=args_cli.device)
    disable_time_out(composed.env_cfg)
    env = composed.make(gym)
    env.reset()

    def make_action(vx: float, vy: float, wz: float) -> torch.Tensor:
        # Rebuilt every step so the arms keep holding where they are as the base moves.
        action = hold_action(env, embodiment, base_height_m=G1_HIP_HEIGHT_M)
        if embodiment.startswith("h2_"):
            from e2e_robot_workflows.arena.h2_actions import set_h2_navigate_command

            return set_h2_navigate_command(action, vx, vy, wz)
        return set_navigate_command(action, vx, vy, wz)

    return env, make_action


def build_env() -> tuple[gym.Env, Callable[[float, float, float], torch.Tensor]]:
    """Compose whichever environment the arguments selected."""

    return build_arena_env() if args_cli.arena else build_builtin_env()


def main() -> None:
    env, make_action = build_env()
    driver = ArrowKeyDriver()
    logger.info("Driving AGILE: Up/Down = walk, Left/Right = turn, A/D = strafe. Ctrl+C or close window to stop.")

    step = 0
    # inference_mode: the AGILE policy output must be grad-free for the
    # warp-based joint-target path (same pattern as record_demos).
    with torch.inference_mode():
        while simulation_app.is_running() and (args_cli.steps == 0 or step < args_cli.steps):
            kvx, kvy, kwz = driver.command(args_cli.speed, args_cli.turn)
            env.step(make_action(args_cli.vx + kvx, args_cli.vy + kvy, args_cli.wz + kwz))
            step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

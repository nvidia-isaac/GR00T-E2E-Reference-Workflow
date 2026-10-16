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

"""Compose Arena's realization of this app's environment, for any entry point.

Recording, locomotion, and validation all need the same three things back from Arena --
the gym id it registered, the environment config it compiled, and the constructor
arguments it wants forwarded -- so the call lives here rather than in whichever script
needed it first. Each caller still chooses its robot-specific joint or Pink embodiment,
because evaluation and OpenXR action layouts are not interchangeable.

Importing this module does not import Arena. Arena is an external source checkout, so
its imports stay inside :func:`compose`, which runs only under a launcher that puts
Arena on the path (``tools/launch_arena_*.sh``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ComposedEnvironment:
    """What ``gym.make`` needs to instantiate an Arena-composed environment."""

    task_id: str
    env_cfg: Any
    env_kwargs: dict[str, Any]
    embodiment: str
    """Echoed back because callers report it and pick their hold command from it."""

    def make(self, gym: Any) -> Any:
        """Instantiate and unwrap the environment.

        ``env_kwargs`` carries what the builder asks the constructor for; Arena passes
        its ``variation_recorder`` there, so forwarding it keeps variation sampling
        recorded.
        """

        return gym.make(self.task_id, cfg=self.env_cfg, **self.env_kwargs).unwrapped


def compose(
    *,
    embodiment: str,
    device: str,
    num_envs: int = 1,
    enable_cameras: bool = False,
    teleop_device: str | None = None,
    disable_task_terminations: bool = True,
    mimic: bool = False,
    manipulation_hand: str = "right",
    seed: int = 42,
    table_height_offset_range_m: Sequence[float] | None = None,
    movable_support: bool = False,
) -> ComposedEnvironment:
    """Build and register Arena's realization of the static-apple environment.

    Args:
        embodiment: Registered Arena robot/controller embodiment to compose.
        device: Simulation device, e.g. ``"cuda:0"``.
        num_envs: Environments to clone.
        enable_cameras: Whether the embodiment publishes its camera observations.
        teleop_device: Arena device to register, or ``None`` for a run that steps the
            environment itself.
        disable_task_terminations: Run to a fixed length regardless of task outcome.
            Defaults on, because every caller here owns its own episode boundaries.
        mimic: Build the embodiment's Mimic environment for AutoData delegation.
        manipulation_hand: Hand whose trajectory is split into pickup and placement.
        seed: Arena placement and Isaac Lab environment seed.
        table_height_offset_range_m: Opt-in per-episode tabletop-height offsets in metres,
            or ``None`` for the nominal fixed height.
        movable_support: Compose the table as a kinematic rigid body whose pose is recorded
            and restored. Randomizing the height implies it; replay needs it on its own.

    Returns:
        The registered gym id, the compiled config, and Arena's constructor arguments.
    """

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    # E2E owns these registrations; importing the module is the public Arena
    # extension seam and leaves the managed checkout unchanged.
    import e2e_robot_workflows.arena.h2_embodiments  # noqa: F401
    import e2e_robot_workflows.arena.h2_retargeter  # noqa: F401

    # The adapter subclasses Arena's own base classes, so importing it is importing Arena.
    from e2e_robot_workflows.arena.external_environment import (
        E2EArenaExternalEnvironment,
        E2EArenaExternalEnvironmentCfg,
    )

    environment = E2EArenaExternalEnvironment().build(
        E2EArenaExternalEnvironmentCfg(
            enable_cameras=enable_cameras,
            embodiment=embodiment,
            teleop_device=teleop_device,
            disable_task_terminations=disable_task_terminations,
            mimic=mimic,
            manipulation_hand=manipulation_hand,
            # Arena builds its own CLI from the config's field types, which it restricts
            # to lists rather than pairs.
            table_height_offset_range_m=(
                None if table_height_offset_range_m is None else list(table_height_offset_range_m)
            ),
            movable_support=movable_support,
        )
    )
    builder = ArenaEnvBuilder(
        environment,
        ArenaEnvBuilderCfg(num_envs=num_envs, device=device, mimic=mimic, seed=seed),
    )
    task_id, env_cfg, env_kwargs = builder.build_registered()
    return ComposedEnvironment(task_id=task_id, env_cfg=env_cfg, env_kwargs=env_kwargs, embodiment=embodiment)


__all__ = ["ComposedEnvironment", "compose"]

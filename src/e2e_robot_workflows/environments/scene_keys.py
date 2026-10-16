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

"""Resolve environment roles to the scene entity names a live runtime uses.

The same declared environment lands under different scene keys depending on who
built the scene. The native Isaac Lab task inherits the locomanipulation scene's
names (``object``, ``destination``, ``background``), while the Arena adapter spawns
each asset under the name the contract declares (``pick_object``,
``placement_target``, ``scene_background``), and each task names the pick object's
contact sensor its own way. Anything reading the live scene by key -- success
detection, environment snapshots -- has to work under both, so it asks here instead
of hardcoding one runtime's names.
"""

from __future__ import annotations

from typing import Any

from e2e_robot_workflows.environments.static_apple import STATIC_APPLE_ENVIRONMENT

ROBOT_ROLE = "robot"
PICK_OBJECT_ROLE = "pick_object"
# Names the native Isaac Lab static-apple scene uses, keyed by contract role.
NATIVE_SCENE_ENTITY_NAMES: dict[str, str] = {
    "background": "background",
    "support": "packing_table",
    PICK_OBJECT_ROLE: "object",
    "destination": "destination",
    ROBOT_ROLE: "robot",
}
NATIVE_CONTACT_SENSOR_NAME = "object_contact_sensor"


def candidate_scene_entity_names(role: str) -> tuple[str, ...]:
    """Return the scene entity names ``role`` may appear under, native name first."""

    if role not in NATIVE_SCENE_ENTITY_NAMES:
        raise KeyError(f"Unknown environment role: {role!r}")
    names = [NATIVE_SCENE_ENTITY_NAMES[role]]
    if role != ROBOT_ROLE:
        names.append(STATIC_APPLE_ENVIRONMENT.asset(role).name)
    # dict.fromkeys keeps declaration order while dropping the duplicate for roles
    # whose native and declared names already agree.
    return tuple(dict.fromkeys(names))


def candidate_contact_sensor_names() -> tuple[str, ...]:
    """Return the names the pick object's contact sensor may appear under.

    The built-in task registers the sensor under a fixed name, while Arena's
    pick-and-place task names it after the object it watches.
    """

    declared = STATIC_APPLE_ENVIRONMENT.asset(PICK_OBJECT_ROLE).name
    return (NATIVE_CONTACT_SENSOR_NAME, f"contact_sensor_{declared}")


def _resolve(scene: Any, candidates: tuple[str, ...], described_as: str) -> str:
    """Return the first of ``candidates`` that ``scene`` exposes.

    Raises:
        KeyError: The scene exposes none of them.
    """

    # Bound to a name first: an InteractiveScene is not a mapping, so `in scene`
    # would not be the equivalent one-liner ruff's SIM118 assumes.
    spawned = scene.keys()
    for name in candidates:
        if name in spawned:
            return name
    raise KeyError(f"No scene entity for {described_as}; tried {candidates} in {sorted(scene.keys())}")


def resolve_scene_entity_name(scene: Any, role: str) -> str:
    """Return the name ``role`` is spawned under in ``scene``.

    Raises:
        KeyError: The role is unknown, or the scene exposes none of its names.
    """

    return _resolve(scene, candidate_scene_entity_names(role), f"role {role!r}")


def resolve_contact_sensor_name(scene: Any) -> str:
    """Return the name the pick object's contact sensor carries in ``scene``.

    Raises:
        KeyError: The scene has no contact sensor for the pick object.
    """

    return _resolve(scene, candidate_contact_sensor_names(), "the pick object's contact sensor")


def resolve_role_to_scene_entity(scene: Any) -> dict[str, str]:
    """Map every contract role to its scene entity name in ``scene``."""

    return {role: resolve_scene_entity_name(scene, role) for role in NATIVE_SCENE_ENTITY_NAMES}


__all__ = [
    "NATIVE_CONTACT_SENSOR_NAME",
    "NATIVE_SCENE_ENTITY_NAMES",
    "candidate_contact_sensor_names",
    "candidate_scene_entity_names",
    "resolve_contact_sensor_name",
    "resolve_role_to_scene_entity",
    "resolve_scene_entity_name",
]

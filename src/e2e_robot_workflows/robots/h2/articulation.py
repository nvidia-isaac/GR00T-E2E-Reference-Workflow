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

"""Arena articulation for the pinned public H2plus plant.

Two constraints on this asset came out of loading it in Isaac Sim, and both are
honored here rather than rediscovered:

- No ``articulation_props`` override. Setting one produced a second articulation
  root below the robot prim; preserving the asset-authored root loads cleanly.
  Nothing needs the override anyway, since the H2 has no joint to the world and
  so is already free-based.
- Explicit gains are required. The USD authors zero drive stiffness, so an
  articulation that inherits the asset's gains goes limp. The driven joints take
  the gains the AGILE export itself emits; the rest take the hold values from
  ``configs/robots/h2.yaml``.

Armature is deliberately left unset so the asset's authored value stands. It is a
uniform ``0.01`` across all 31 joints, which looks like a default rather than a
measurement, but the export walked without it being touched, so it remains
unswept and is the first thing to vary if the envelope disappoints.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from e2e_robot_workflows.arena.h2_spawn import materialize_h2plus_arena_usd
from e2e_robot_workflows.robots.h2.config import H2RobotConfig
from e2e_robot_workflows.robots.h2.pink_contract import H2PinkContract, materialize_published_usd

H2_PRIM_PATH = "{ENV_REGEX_NS}/Robot"


def make_h2_articulation_cfg(
    config: H2RobotConfig,
    contract: H2PinkContract,
    *,
    cache_root: str,
    prim_path: str = H2_PRIM_PATH,
) -> ArticulationCfg:
    """Build the 75-joint H2plus articulation used by every Arena H2 mode."""

    actuators: dict[str, ImplicitActuatorCfg] = {
        "agile_driven": ImplicitActuatorCfg(
            joint_names_expr=list(config.driven_joint_names),
            stiffness=config.driven_joint_stiffness,
            damping=config.driven_joint_damping,
        )
    }
    actuators["pink_arms"] = ImplicitActuatorCfg(
        joint_names_expr=list(contract.arm_joint_names),
        stiffness={name: 20.0 if "_wrist_" in name else 200.0 for name in contract.arm_joint_names},
        damping={name: 1.0 if "_wrist_" in name else 5.0 for name in contract.arm_joint_names},
    )
    actuators["held_head_and_waist"] = ImplicitActuatorCfg(
        joint_names_expr=["waist_yaw_joint", "head_pitch_joint", "head_yaw_joint"],
        stiffness={"waist_yaw_joint": 300.0, "head_.*": 50.0},
        damping={"waist_yaw_joint": 5.0, "head_.*": 2.0},
    )
    actuators["sharpa_hands"] = ImplicitActuatorCfg(
        joint_names_expr=list(contract.hand_joint_order),
        stiffness=20.0,
        damping=1.0,
    )

    source_usd = materialize_published_usd(contract.simulation_usd, cache_root=cache_root)
    arena_usd = materialize_h2plus_arena_usd(
        source_usd,
        cache_base=Path(cache_root).parent,
        source_contract_sha256=contract.source_contract_sha256,
        variants=contract.simulation_usd.variant_selection,
        source_pelvis_translation_xyz=contract.source_pelvis_translation_xyz,
        recipe_version=contract.arena_recipe_version,
        expected_sha256=contract.arena_usd_sha256,
    )

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(arena_usd),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=config.initial_base_position,
            rot=config.initial_rotation_xyzw,
            # Explicit reset joints override the export's reference offsets.
            # The reference alone is not a supported equilibrium stance.
            joint_pos={
                **dict(config.reference_joint_pos),
                **dict.fromkeys(contract.arm_joint_names, 0.0),
                **config.initial_joint_pos,
                **dict.fromkeys(contract.hand_joint_order, 0.0),
            },
        ),
        actuators=actuators,
    )


__all__ = ["H2_PRIM_PATH", "make_h2_articulation_cfg"]

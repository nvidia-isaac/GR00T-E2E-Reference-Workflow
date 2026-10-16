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

"""Pink-specific helpers shared by GPU simulation validations."""

from __future__ import annotations

from typing import Any

from validation_support import package_version


def run_daqp_sanity() -> dict[str, Any]:
    """Solve a constrained QP through the same qpsolvers/DAQP path as Pink."""

    import numpy as np
    from qpsolvers import available_solvers, solve_qp

    if "daqp" not in available_solvers:
        raise AssertionError(f"DAQP is unavailable; installed qpsolvers backends: {available_solvers}")

    # min 0.5*x'Px + q'x, with 0 <= x <= [2, 3]. The exact optimum is [1, 2].
    identity = np.eye(2)
    solution = solve_qp(
        P=identity,
        q=np.array([-1.0, -2.0]),
        G=np.vstack((identity, -identity)),
        h=np.array([2.0, 3.0, 0.0, 0.0]),
        # Keep this warm start: it is the compatibility canary for the
        # qpsolvers/DAQP versions pinned by runtime.toml.
        initvals=np.zeros(2),
        solver="daqp",
    )
    if solution is None or not np.allclose(solution, np.array([1.0, 2.0]), atol=1.0e-5):
        raise AssertionError(f"DAQP sanity solve returned an unexpected solution: {solution}")

    return {
        "backend": "daqp",
        "daqp_version": package_version("daqp"),
        "qpsolvers_version": package_version("qpsolvers"),
        "solution": solution.tolist(),
    }


def resolve_g1_pink_handles(robot: Any, action_term: Any) -> dict[str, Any]:
    """Resolve the G1 wrists and hand joints required by a Pink action."""

    try:
        left_wrist_id = robot.body_names.index("left_wrist_yaw_link")
        right_wrist_id = robot.body_names.index("right_wrist_yaw_link")
    except ValueError as exc:
        raise AssertionError("the G1 wrist links required by Pink are missing") from exc

    hand_joint_ids, hand_joint_names = robot.find_joints(action_term.cfg.hand_joint_names)
    if len(hand_joint_ids) != action_term.hand_joint_dim:
        raise AssertionError(
            f"Pink expected {action_term.hand_joint_dim} hand joints but resolved {len(hand_joint_ids)}"
        )
    return {
        "hand_joint_ids": hand_joint_ids,
        "hand_joint_names": hand_joint_names,
        "left_wrist_id": left_wrist_id,
        "right_wrist_id": right_wrist_id,
    }


def make_live_pink_action(env: Any, action_term: Any, handles: dict[str, Any], torch: Any) -> Any:
    """Hold the G1 wrists and hands at their current poses."""

    robot = env.scene["robot"]
    action = torch.zeros((env.num_envs, action_term.action_dim), device=env.device)
    action[:, 0:7] = robot.data.body_link_state_w.torch[:, handles["left_wrist_id"], :7]
    action[:, 7:14] = robot.data.body_link_state_w.torch[:, handles["right_wrist_id"], :7]
    action[:, 14:] = robot.data.joint_pos.torch[:, handles["hand_joint_ids"]]
    return action

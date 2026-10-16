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

"""Fail-closed H2plus asset and 58-D Pink/Sharpa contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlsplit

from e2e_robot_workflows.common import load_yaml_mapping
from e2e_robot_workflows.data.artifact_paths import file_sha256

if TYPE_CHECKING:
    import torch

_DEFAULT_CONTRACT_PATH = Path(__file__).resolve().parents[4] / "configs" / "robots" / "h2_pink.yaml"

H2_PINK_ACTION_DIM = 58
H2_PINK_POSE_DIM = 7
H2_PINK_HAND_DIM = 44
H2_PINK_HAND_DIM_PER_SIDE = 22
H2_PINK_LEFT_POSE_SLICE = slice(0, 7)
H2_PINK_RIGHT_POSE_SLICE = slice(7, 14)
H2_PINK_HAND_SLICE = slice(14, 58)
H2_LIVE_JOINT_COUNT = 75
H2_LIVE_BODY_COUNT = 80
H2_NAVIGATE_COMMAND_DIM = 3
H2_ARENA_ACTION_DIM = H2_PINK_ACTION_DIM + H2_NAVIGATE_COMMAND_DIM


def _validate_sha256(label: str, digest: str) -> None:
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must have a lowercase hexadecimal SHA-256 digest")


@dataclass(frozen=True)
class PublishedUsdDependency:
    """One relative file in the published H2plus USD package."""

    path: str
    sha256: str


@dataclass(frozen=True)
class PublishedUsdContract:
    """Content identity and variant selection for the public H2plus package."""

    url: str
    sha256: str
    variants: tuple[tuple[str, str], ...]
    dependencies: tuple[PublishedUsdDependency, ...]

    def __post_init__(self) -> None:
        parsed_url = urlsplit(self.url)
        if parsed_url.scheme != "https" or not parsed_url.netloc or not Path(parsed_url.path).name:
            raise ValueError(f"simulation USD must be an HTTPS file URL, got {self.url!r}")
        _validate_sha256("simulation USD", self.sha256)
        if not self.variants:
            raise ValueError("simulation USD must declare variant selections")
        dependency_paths: set[str] = set()
        for dependency in self.dependencies:
            parsed_dependency = urlsplit(dependency.path)
            path = PurePosixPath(dependency.path)
            if (
                not dependency.path
                or path.is_absolute()
                or ".." in path.parts
                or parsed_dependency.scheme
                or parsed_dependency.netloc
                or parsed_dependency.query
                or parsed_dependency.fragment
            ):
                raise ValueError(f"simulation USD dependency must be a relative URL path: {dependency.path!r}")
            if dependency.path in dependency_paths:
                raise ValueError(f"simulation USD dependency is repeated: {dependency.path!r}")
            dependency_paths.add(dependency.path)
            _validate_sha256(f"simulation USD dependency {dependency.path!r}", dependency.sha256)

    @property
    def variant_selection(self) -> dict[str, str]:
        return dict(self.variants)

    @property
    def contract_sha256(self) -> str:
        payload = {
            "url": self.url,
            "sha256": self.sha256,
            "variants": tuple(sorted(self.variants)),
            "dependencies": tuple(sorted((entry.path, entry.sha256) for entry in self.dependencies)),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @property
    def declared_relative_paths(self) -> tuple[str, ...]:
        root_name = Path(urlsplit(self.url).path).name
        return (root_name, *(entry.path for entry in self.dependencies))


@dataclass(frozen=True)
class KinematicsUrdfInfo:
    """Structural facts measured from a generated kinematics sidecar."""

    root_link: str
    link_names: tuple[str, ...]
    actuated_joint_names: tuple[str, ...]
    joint_position_limits: tuple[tuple[str, tuple[float, float]], ...]


@dataclass(frozen=True)
class H2PinkContract:
    """The exact H2plus/Pink compatibility boundary required before PICO."""

    simulation_usd: PublishedUsdContract
    urdf_env_var: str
    urdf_sha256: str
    source_contract_sha256: str
    arena_usd_sha256: str
    arena_recipe_version: int
    generation_recipe_version: int
    isaacsim_version: str
    expected_body_count: int
    base_frame: str
    source_pelvis_translation_xyz: tuple[float, float, float]
    flange_frames: tuple[str, str]
    action_dim: int
    arm_joint_names: tuple[str, ...]
    body_joint_order: tuple[str, ...]
    hand_joint_order: tuple[str, ...]
    learned_joint_position_limits: tuple[tuple[str, tuple[float, float]], ...]
    hand_open_position: float
    hand_closed_position_by_pattern: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        _validate_sha256("Pink URDF", self.urdf_sha256)
        _validate_sha256("Pink source contract", self.source_contract_sha256)
        _validate_sha256("H2 Arena runtime USD", self.arena_usd_sha256)
        if self.source_contract_sha256 != self.simulation_usd.contract_sha256:
            raise ValueError("Pink source contract does not identify the configured public H2plus package")
        if self.arena_recipe_version <= 0 or self.generation_recipe_version <= 0 or not self.isaacsim_version:
            raise ValueError("Pink generation recipe and Isaac Sim version must be pinned")
        if self.expected_body_count != H2_LIVE_BODY_COUNT:
            raise ValueError(f"H2plus must declare {H2_LIVE_BODY_COUNT} live bodies")
        if len(self.source_pelvis_translation_xyz) != 3 or not all(
            math.isfinite(value) for value in self.source_pelvis_translation_xyz
        ):
            raise ValueError("published H2plus pelvis translation must contain three finite values")
        if self.action_dim != H2_PINK_ACTION_DIM:
            raise ValueError(f"H2 Pink action must be {H2_PINK_ACTION_DIM}-D, got {self.action_dim}")
        if len(self.arm_joint_names) != 14 or len(set(self.arm_joint_names)) != 14:
            raise ValueError("H2 Pink must declare 14 distinct arm joints")
        if len(self.body_joint_order) != 31 or len(set(self.body_joint_order)) != 31:
            raise ValueError("H2plus must declare 31 distinct body joints")
        if len(self.hand_joint_order) != H2_PINK_HAND_DIM or len(set(self.hand_joint_order)) != H2_PINK_HAND_DIM:
            raise ValueError(f"H2 Sharpa must declare {H2_PINK_HAND_DIM} distinct hand joints")
        learned_names = self.arm_joint_names + self.hand_joint_order
        if tuple(name for name, _ in self.learned_joint_position_limits) != learned_names:
            raise ValueError("H2 Pink learned-joint limits must follow the 14-arm then 44-hand action order")
        for joint_name, limits in self.learned_joint_position_limits:
            if len(limits) != 2 or not all(math.isfinite(value) for value in limits) or limits[0] > limits[1]:
                raise ValueError(f"H2 Pink learned-joint limits for {joint_name!r} are invalid")
        if (
            len(self.expected_joint_order) != H2_LIVE_JOINT_COUNT
            or len(set(self.expected_joint_order)) != H2_LIVE_JOINT_COUNT
        ):
            raise ValueError("H2plus Pink contract must cover 75 distinct live joints")
        if not set(self.arm_joint_names) <= set(self.body_joint_order):
            raise ValueError("H2 Pink arm joints must be part of the 31-joint body order")
        if len(self.left_hand_indices) != H2_PINK_HAND_DIM_PER_SIDE:
            raise ValueError(f"H2 Sharpa must expose {H2_PINK_HAND_DIM_PER_SIDE} left-hand joints")
        if len(self.right_hand_indices) != H2_PINK_HAND_DIM_PER_SIDE:
            raise ValueError(f"H2 Sharpa must expose {H2_PINK_HAND_DIM_PER_SIDE} right-hand joints")
        if not math.isfinite(self.hand_open_position):
            raise ValueError("H2 Sharpa open target must be finite")
        if not self.hand_closed_position_by_pattern:
            raise ValueError("H2 Sharpa controller synergy must declare closed targets")
        for joint_name in self.hand_joint_order:
            matches = [
                value for pattern, value in self.hand_closed_position_by_pattern if re.fullmatch(pattern, joint_name)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"H2 Sharpa joint {joint_name!r} must match exactly one controller synergy rule, got {len(matches)}"
                )
            if not math.isfinite(matches[0]):
                raise ValueError(f"H2 Sharpa closed target for {joint_name!r} must be finite")
            lower, upper = self.learned_joint_limits[joint_name]
            if not lower <= self.hand_open_position <= upper or not lower <= matches[0] <= upper:
                raise ValueError(f"H2 Sharpa synergy target for {joint_name!r} is outside its pinned position limits")

    @property
    def expected_joint_order(self) -> tuple[str, ...]:
        return self.body_joint_order + self.hand_joint_order

    @property
    def left_hand_indices(self) -> tuple[int, ...]:
        return tuple(index for index, name in enumerate(self.hand_joint_order) if name.startswith("left_"))

    @property
    def right_hand_indices(self) -> tuple[int, ...]:
        return tuple(index for index, name in enumerate(self.hand_joint_order) if name.startswith("right_"))

    @property
    def learned_joint_limits(self) -> dict[str, tuple[float, float]]:
        """Pinned sidecar limits for the 58 GR00T action joints."""

        return dict(self.learned_joint_position_limits)

    @property
    def hand_open_targets(self) -> dict[str, float]:
        """Open targets in the asserted live Sharpa order."""

        return dict.fromkeys(self.hand_joint_order, self.hand_open_position)

    @property
    def hand_closed_targets(self) -> dict[str, float]:
        """Controller pinch/power targets in the asserted live Sharpa order."""

        targets: dict[str, float] = {}
        for joint_name in self.hand_joint_order:
            targets[joint_name] = next(
                value for pattern, value in self.hand_closed_position_by_pattern if re.fullmatch(pattern, joint_name)
            )
        return targets

    def validate_urdf(self, path: str | Path) -> KinematicsUrdfInfo:
        """Verify identity and the joint/frame contract of a generated sidecar."""

        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Pink URDF is missing: {resolved}")
        actual_sha256 = file_sha256(resolved)
        if actual_sha256 != self.urdf_sha256:
            raise ValueError(f"unexpected Pink URDF: expected sha256:{self.urdf_sha256}, found sha256:{actual_sha256}")
        info = inspect_kinematics_urdf(resolved)
        if len(info.link_names) != self.expected_body_count:
            raise ValueError(f"Pink URDF must contain {self.expected_body_count} links, got {len(info.link_names)}")
        if info.root_link != self.base_frame:
            raise ValueError(f"Pink URDF root must be {self.base_frame!r}, got {info.root_link!r}")
        if set(info.actuated_joint_names) != set(self.expected_joint_order):
            missing = sorted(set(self.expected_joint_order) - set(info.actuated_joint_names))
            unexpected = sorted(set(info.actuated_joint_names) - set(self.expected_joint_order))
            raise ValueError(f"Pink/live joint mismatch: missing={missing}, unexpected={unexpected}")
        urdf_limits = dict(info.joint_position_limits)
        for joint_name, expected_limits in self.learned_joint_position_limits:
            actual_limits = urdf_limits.get(joint_name)
            if actual_limits is None or any(
                not math.isclose(actual, expected, abs_tol=1e-7, rel_tol=0.0)
                for actual, expected in zip(actual_limits, expected_limits, strict=True)
            ):
                raise ValueError(
                    f"Pink URDF position limits for {joint_name!r} do not match the pinned learned-joint contract: "
                    f"expected {expected_limits}, got {actual_limits}"
                )
        missing_frames = sorted(set(self.flange_frames) - set(info.link_names))
        if missing_frames:
            raise ValueError(f"Pink URDF is missing flange frames: {missing_frames}")
        return info

    def resolve_urdf(self) -> Path:
        value = os.environ.get(self.urdf_env_var)
        if not value:
            raise RuntimeError(f"Set {self.urdf_env_var} to the generated, pinned H2 Pink URDF.")
        path = Path(value).expanduser().resolve()
        self.validate_urdf(path)
        return path

    def pack_action(
        self,
        left_pose: torch.Tensor,
        right_pose: torch.Tensor,
        left_hand: torch.Tensor,
        right_hand: torch.Tensor,
    ) -> torch.Tensor:
        """Pack two XYZW poses and two hands into the 58-D semantic layout."""

        import torch

        _assert_last_dimension("left_pose", left_pose, H2_PINK_POSE_DIM)
        _assert_last_dimension("right_pose", right_pose, H2_PINK_POSE_DIM)
        _assert_last_dimension("left_hand", left_hand, H2_PINK_HAND_DIM_PER_SIDE)
        _assert_last_dimension("right_hand", right_hand, H2_PINK_HAND_DIM_PER_SIDE)
        if not (left_pose.shape[:-1] == right_pose.shape[:-1] == left_hand.shape[:-1] == right_hand.shape[:-1]):
            raise ValueError("H2 Pink action components must have the same leading shape")
        hands = torch.empty((*left_hand.shape[:-1], H2_PINK_HAND_DIM), dtype=left_hand.dtype, device=left_hand.device)
        hands[..., self.left_hand_indices] = left_hand
        hands[..., self.right_hand_indices] = right_hand
        return torch.cat((left_pose, right_pose, hands), dim=-1)

    def unpack_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the 58-D semantic layout into two poses and two hands."""

        _assert_last_dimension("action", action, H2_PINK_ACTION_DIM)
        hands = action[..., H2_PINK_HAND_SLICE]
        return (
            action[..., H2_PINK_LEFT_POSE_SLICE],
            action[..., H2_PINK_RIGHT_POSE_SLICE],
            hands[..., self.left_hand_indices],
            hands[..., self.right_hand_indices],
        )


def _assert_last_dimension(name: str, value: torch.Tensor, expected: int) -> None:
    if value.ndim == 0 or value.shape[-1] != expected:
        raise ValueError(f"{name} must end in dimension {expected}, got {tuple(value.shape)}")


def materialize_published_usd(
    contract: PublishedUsdContract,
    *,
    cache_root: str | Path,
    open_url: Callable[..., Any] | None = None,
) -> Path:
    """Download and atomically materialize the exact USD package closure."""

    bundle_root = Path(cache_root) / contract.contract_sha256
    root_name = Path(urlsplit(contract.url).path).name
    files = ((root_name, contract.url, contract.sha256),) + tuple(
        (entry.path, urljoin(contract.url, entry.path), entry.sha256) for entry in contract.dependencies
    )
    opener = open_url or urllib.request.urlopen
    for relative_path, url, expected_sha256 in files:
        target = bundle_root / PurePosixPath(relative_path)
        if target.is_file() and file_sha256(target) == expected_sha256:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=bundle_root.parent,
                prefix=f".{contract.contract_sha256}.{target.name}.",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)
                digest = hashlib.sha256()
                with opener(url, timeout=30) as response:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        file.write(block)
                        digest.update(block)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"published H2plus file {url!r} changed: "
                    f"expected sha256:{expected_sha256}, found sha256:{actual_sha256}"
                )
            os.replace(temporary_path, target)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    actual_paths = {path.relative_to(bundle_root).as_posix() for path in bundle_root.rglob("*") if path.is_file()}
    expected_paths = set(contract.declared_relative_paths)
    if actual_paths != expected_paths:
        raise ValueError(
            f"materialized H2plus package closure differs from the contract: "
            f"missing={sorted(expected_paths - actual_paths)}, unexpected={sorted(actual_paths - expected_paths)}"
        )
    return bundle_root / root_name


def _sort_xml_attributes(element: ET.Element) -> None:
    if element.attrib:
        attributes = sorted(element.attrib.items())
        element.attrib.clear()
        element.attrib.update(attributes)
    for child in element:
        _sort_xml_attributes(child)


def postprocess_kinematics_urdf(
    raw_path: str | Path,
    output_path: str | Path,
    *,
    root_translation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Path:
    """Reduce converter output to deterministic pelvis-rooted kinematics."""

    tree = ET.parse(raw_path)
    robot = tree.getroot()
    if robot.tag != "robot":
        raise ValueError("USD converter output is not a URDF robot")

    for material in tuple(robot.findall("material")):
        robot.remove(material)
    for link in robot.findall("link"):
        for tag in ("inertial", "visual", "collision"):
            for child in tuple(link.findall(tag)):
                link.remove(child)

    world = robot.find("link[@name='world']")
    floating = robot.find("joint[@name='floating_base']")
    if world is None or floating is None or floating.get("type") != "floating":
        raise ValueError("expected the converter's synthetic world/floating_base pair")
    parent = floating.find("parent")
    child = floating.find("child")
    if parent is None or child is None or parent.get("link") != "world" or child.get("link") != "pelvis":
        raise ValueError("synthetic floating base does not connect world to pelvis")
    robot.remove(floating)
    robot.remove(world)

    # UsdToUrdfConverter writes joints directly below the articulation root in
    # asset coordinates, even though the generated floating_base has no origin.
    # Rebase those origins by the measured source-USD pelvis translation before
    # making pelvis the URDF root.
    for joint in robot.findall("joint"):
        parent = joint.find("parent")
        if parent is None or parent.get("link") != "pelvis":
            continue
        origin = joint.find("origin")
        if origin is None:
            if any(root_translation_xyz):
                raise ValueError(f"pelvis child joint {joint.get('name')!r} has no origin to rebase")
            continue
        xyz = tuple(float(value) for value in origin.get("xyz", "0 0 0").split())
        if len(xyz) != 3:
            raise ValueError(f"pelvis child joint {joint.get('name')!r} has malformed xyz")
        origin.set(
            "xyz",
            " ".join(f"{value - offset:.12g}" for value, offset in zip(xyz, root_translation_xyz, strict=True)),
        )

    used_joint_names: set[str] = set()
    for joint in robot.findall("joint"):
        name = joint.get("name")
        if not name:
            raise ValueError("URDF joint has no name")
        if name in used_joint_names:
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                raise ValueError(f"duplicate joint {name!r} has no parent/child")
            name = f"{parent.get('link')}_to_{child.get('link')}_fixed"
            if name in used_joint_names:
                raise ValueError(f"cannot deterministically disambiguate duplicate joint {name!r}")
            joint.set("name", name)
        used_joint_names.add(name)
        joint_type = joint.get("type")
        if joint_type == "floating":
            raise ValueError(f"unexpected floating joint after pelvis-root conversion: {name}")
        for tag in ("dynamics", "calibration", "safety_controller"):
            for element in tuple(joint.findall(tag)):
                joint.remove(element)
        if joint_type not in {"fixed"}:
            limit = joint.find("limit")
            if limit is None:
                raise ValueError(f"actuated joint {name!r} has no limits")
            if joint_type in {"revolute", "prismatic"} and not {"lower", "upper"} <= set(limit.attrib):
                raise ValueError(f"bounded joint {name!r} has incomplete position limits")
            limit.attrib.setdefault("effort", "1")
            limit.attrib.setdefault("velocity", "1")

    _sort_xml_attributes(robot)
    ET.indent(tree, space="  ")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as file:
            temporary_path = Path(file.name)
        tree.write(temporary_path, encoding="utf-8", xml_declaration=True, short_empty_elements=True)
        with temporary_path.open("ab") as file:
            file.write(b"\n")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def inspect_kinematics_urdf(path: str | Path) -> KinematicsUrdfInfo:
    """Inspect root, frames, and actuated joints without loading Pinocchio."""

    robot = ET.parse(path).getroot()
    links = tuple(link.get("name", "") for link in robot.findall("link"))
    if len(links) != len(set(links)) or "" in links:
        raise ValueError("URDF link names must be present and unique")
    child_links = {
        child.get("link", "") for joint in robot.findall("joint") if (child := joint.find("child")) is not None
    }
    roots = set(links) - child_links
    if len(roots) != 1:
        raise ValueError(f"URDF must have exactly one root link, got {sorted(roots)}")
    joint_names = tuple(joint.get("name", "") for joint in robot.findall("joint"))
    if len(joint_names) != len(set(joint_names)) or "" in joint_names:
        raise ValueError("URDF joint names must be present and unique")
    actuated_joints = tuple(joint for joint in robot.findall("joint") if joint.get("type") not in {"fixed", "floating"})
    actuated = tuple(joint.get("name", "") for joint in actuated_joints)
    position_limits = []
    for joint in actuated_joints:
        limit = joint.find("limit")
        if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            position_limits.append(
                (joint.get("name", ""), (float(limit.attrib["lower"]), float(limit.attrib["upper"])))
            )
    return KinematicsUrdfInfo(
        root_link=roots.pop(),
        link_names=links,
        actuated_joint_names=actuated,
        joint_position_limits=tuple(position_limits),
    )


@cache
def load_h2_pink_contract(path: str | Path | None = None) -> H2PinkContract:
    """Load the checked H2plus/Pink compatibility contract."""

    data = load_yaml_mapping(path or _DEFAULT_CONTRACT_PATH)
    if data.get("schema_version") != 1:
        raise ValueError(f"unsupported H2 Pink contract schema: {data.get('schema_version')!r}")
    simulation = data["simulation_usd"]
    kinematics = data["pink_kinematics"]
    action = data["action"]
    return H2PinkContract(
        simulation_usd=PublishedUsdContract(
            url=str(simulation["url"]),
            sha256=str(simulation["sha256"]),
            variants=tuple((str(name), str(value)) for name, value in simulation["variants"].items()),
            dependencies=tuple(
                PublishedUsdDependency(path=str(dependency_path), sha256=str(dependency_sha256))
                for dependency_path, dependency_sha256 in simulation["dependencies"].items()
            ),
        ),
        urdf_env_var=str(kinematics["urdf_env_var"]),
        urdf_sha256=str(kinematics["urdf_sha256"]),
        source_contract_sha256=str(kinematics["source_contract_sha256"]),
        arena_usd_sha256=str(kinematics["arena_usd_sha256"]),
        arena_recipe_version=int(kinematics["arena_recipe_version"]),
        generation_recipe_version=int(kinematics["generation_recipe_version"]),
        isaacsim_version=str(kinematics["isaacsim_version"]),
        expected_body_count=int(kinematics["expected_body_count"]),
        base_frame=str(kinematics["base_frame"]),
        source_pelvis_translation_xyz=tuple(float(value) for value in kinematics["source_pelvis_translation_xyz"]),
        flange_frames=tuple(str(name) for name in kinematics["flange_frames"]),
        action_dim=int(action["dimension"]),
        arm_joint_names=tuple(str(name) for name in action["arm_joint_names"]),
        body_joint_order=tuple(str(name) for name in action["body_joint_order"]),
        hand_joint_order=tuple(str(name) for name in action["hand_joint_order"]),
        learned_joint_position_limits=tuple(
            (str(name), tuple(float(value) for value in limits))
            for name, limits in action["learned_joint_position_limits"].items()
        ),
        hand_open_position=float(action["controller_synergy"]["open_position"]),
        hand_closed_position_by_pattern=tuple(
            (str(entry["pattern"]), float(entry["closed_position"]))
            for entry in action["controller_synergy"]["closed_targets"]
        ),
    )


__all__ = [
    "H2_ARENA_ACTION_DIM",
    "H2_LIVE_BODY_COUNT",
    "H2_LIVE_JOINT_COUNT",
    "H2_NAVIGATE_COMMAND_DIM",
    "H2_PINK_ACTION_DIM",
    "H2_PINK_HAND_DIM",
    "H2_PINK_HAND_DIM_PER_SIDE",
    "H2_PINK_HAND_SLICE",
    "H2_PINK_LEFT_POSE_SLICE",
    "H2_PINK_POSE_DIM",
    "H2_PINK_RIGHT_POSE_SLICE",
    "H2PinkContract",
    "KinematicsUrdfInfo",
    "PublishedUsdContract",
    "PublishedUsdDependency",
    "file_sha256",
    "inspect_kinematics_urdf",
    "load_h2_pink_contract",
    "materialize_published_usd",
    "postprocess_kinematics_urdf",
]

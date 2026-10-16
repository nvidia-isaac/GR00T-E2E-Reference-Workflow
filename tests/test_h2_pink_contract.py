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

from __future__ import annotations

import hashlib
import io
import os
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest

import e2e_robot_workflows.robots.h2.pink_contract as pink_contract_module
import tools.materialize_h2_pink_sidecar as materializer_module
from e2e_robot_workflows.robots.h2.pink_contract import (
    H2_LIVE_JOINT_COUNT,
    H2_PINK_ACTION_DIM,
    PublishedUsdContract,
    PublishedUsdDependency,
    file_sha256,
    inspect_kinematics_urdf,
    load_h2_pink_contract,
    materialize_published_usd,
    postprocess_kinematics_urdf,
)

torch = pytest.importorskip("torch")


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_h2_contract_is_parsed_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    original_load = pink_contract_module.load_yaml_mapping
    loaded_paths: list[object] = []

    def counted_load(path):
        loaded_paths.append(path)
        return original_load(path)

    load_h2_pink_contract.cache_clear()
    monkeypatch.setattr(pink_contract_module, "load_yaml_mapping", counted_load)
    try:
        first = load_h2_pink_contract()
        second = load_h2_pink_contract()
    finally:
        load_h2_pink_contract.cache_clear()

    assert first is second
    assert loaded_paths == [pink_contract_module._DEFAULT_CONTRACT_PATH]


def test_public_h2_contract_is_complete_and_pose_hand_round_trip_is_lossless() -> None:
    contract = load_h2_pink_contract()
    left_pose = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    right_pose = left_pose + 20
    left_hand = torch.arange(44, dtype=torch.float32).reshape(2, 22) + 40
    right_hand = left_hand + 80

    action = contract.pack_action(left_pose, right_pose, left_hand, right_hand)
    unpacked = contract.unpack_action(action)

    assert action.shape == (2, H2_PINK_ACTION_DIM)
    assert len(contract.expected_joint_order) == H2_LIVE_JOINT_COUNT
    assert tuple(contract.learned_joint_limits) == contract.arm_joint_names + contract.hand_joint_order
    assert contract.source_contract_sha256 == contract.simulation_usd.contract_sha256
    for expected, actual in zip((left_pose, right_pose, left_hand, right_hand), unpacked, strict=True):
        torch.testing.assert_close(actual, expected)


def test_public_usd_contract_hash_is_independent_of_mapping_order() -> None:
    dependencies = (
        PublishedUsdDependency("payloads/b.usda", _digest(b"b")),
        PublishedUsdDependency("payloads/a.usda", _digest(b"a")),
    )
    contract = PublishedUsdContract(
        url="https://assets.example/robot.usda",
        sha256=_digest(b"root"),
        variants=(("hands", "sharpa"), ("Physics", "physx")),
        dependencies=dependencies,
    )
    reordered = replace(
        contract,
        variants=tuple(reversed(contract.variants)),
        dependencies=tuple(reversed(contract.dependencies)),
    )

    assert reordered.contract_sha256 == contract.contract_sha256


def test_public_usd_materialization_verifies_the_exact_closure(tmp_path: Path) -> None:
    root = b"root"
    payload = b"payload"
    contract = PublishedUsdContract(
        url="https://assets.example/robot.usda",
        sha256=_digest(root),
        variants=(("Physics", "physx"),),
        dependencies=(PublishedUsdDependency("payloads/robot.usda", _digest(payload)),),
    )
    downloads = {
        "https://assets.example/robot.usda": root,
        "https://assets.example/payloads/robot.usda": payload,
    }
    bundle_root = tmp_path / contract.contract_sha256
    expected_files = {"robot.usda": _digest(root), "payloads/robot.usda": _digest(payload)}

    def open_url(url: str, *, timeout: int) -> io.BytesIO:
        assert timeout == 30
        actual_files = {
            path.relative_to(bundle_root).as_posix(): file_sha256(path)
            for path in bundle_root.rglob("*")
            if path.is_file()
        }
        assert actual_files.items() <= expected_files.items()
        return io.BytesIO(downloads[url])

    materialized = materialize_published_usd(contract, cache_root=tmp_path, open_url=open_url)
    assert materialized.read_bytes() == root
    assert (materialized.parent / "payloads" / "robot.usda").read_bytes() == payload

    (materialized.parent / "unexpected.usda").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="unexpected=.*unexpected.usda"):
        materialize_published_usd(contract, cache_root=tmp_path, open_url=open_url)


def test_interrupted_h2_urdf_publication_never_exposes_a_partial_final_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = b"<robot name='H2'/>\n"
    source_usd = tmp_path / "source.usda"
    contract = replace(load_h2_pink_contract(), urdf_sha256=_digest(generated))
    cache_root = tmp_path / "cache"

    monkeypatch.setattr(type(contract), "validate_urdf", lambda _contract, _path: None)
    monkeypatch.setattr(materializer_module, "version", lambda _name: contract.isaacsim_version)
    monkeypatch.setattr(materializer_module, "load_h2_pink_contract", lambda _path: contract)
    monkeypatch.setattr(materializer_module, "materialize_published_usd", lambda *_args, **_kwargs: source_usd)
    for function_name in ("_verify_source_pelvis_translation", "_export_raw_urdf", "_verify_pinocchio"):
        monkeypatch.setattr(materializer_module, function_name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        materializer_module,
        "postprocess_kinematics_urdf",
        lambda _raw, output, **_kwargs: output.write_bytes(generated),
    )
    monkeypatch.setattr(
        materializer_module,
        "materialize_h2plus_arena_usd",
        lambda *_args, **_kwargs: pytest.fail("URDF publication bypassed atomic replacement"),
    )

    class PublicationInterrupted(Exception):
        pass

    def interrupt_atomic_replace(_source: str | Path, _destination: str | Path) -> None:
        raise PublicationInterrupted

    monkeypatch.setattr(os, "replace", interrupt_atomic_replace)

    with pytest.raises(PublicationInterrupted):
        materializer_module._generate(
            tmp_path / "contract.yaml",
            cache_root,
            regenerate=False,
            simulation_app=object(),
        )

    assert not list(cache_root.rglob("unitree_h2.urdf"))


def test_kinematics_postprocessor_is_deterministic_and_pelvis_rooted(tmp_path: Path) -> None:
    raw = tmp_path / "raw.urdf"
    raw.write_text(
        """<?xml version="1.0"?>
<robot name="H2">
  <link name="world"/>
  <link name="pelvis"><visual/><collision/><inertial/></link>
  <link name="left_hand_flange"/>
  <link name="left_hand"/>
  <link name="right_hand_flange"/>
  <link name="right_hand"/>
  <joint name="floating_base" type="floating"><parent link="world"/><child link="pelvis"/></joint>
  <joint name="shoulder" type="revolute">
    <parent link="pelvis"/><child link="left_hand_flange"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1"/><dynamics damping="1"/><calibration reference_position="0"/>
  </joint>
  <joint name="PhysicsFixedJoint" type="fixed"><parent link="left_hand_flange"/><child link="left_hand"/></joint>
  <joint name="right_shoulder" type="revolute">
    <parent link="pelvis"/><child link="right_hand_flange"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" velocity="2"/>
  </joint>
  <joint name="PhysicsFixedJoint" type="fixed"><parent link="right_hand_flange"/><child link="right_hand"/></joint>
</robot>
""",
        encoding="utf-8",
    )
    first = postprocess_kinematics_urdf(raw, tmp_path / "first.urdf")
    second = postprocess_kinematics_urdf(raw, tmp_path / "second.urdf")
    first_root = ET.parse(first).getroot()
    info = inspect_kinematics_urdf(first)

    assert first.read_bytes() == second.read_bytes()
    assert info.root_link == "pelvis"
    assert set(info.actuated_joint_names) == {"shoulder", "right_shoulder"}
    assert first_root.find("link[@name='world']") is None
    assert first_root.find(".//visual") is None
    assert first_root.find(".//collision") is None
    assert first_root.find(".//inertial") is None
    assert first_root.find(".//dynamics") is None
    assert first_root.find(".//calibration") is None
    joint_names = [joint.get("name") for joint in first_root.findall("joint")]
    assert len(joint_names) == len(set(joint_names))
    shoulder_limit = first_root.find("joint[@name='shoulder']/limit")
    assert shoulder_limit is not None
    assert shoulder_limit.attrib == {"effort": "1", "lower": "-1", "upper": "1", "velocity": "1"}


def test_runtime_urdf_validation_checks_digest_joint_bijection_and_frames(tmp_path: Path) -> None:
    contract = load_h2_pink_contract()
    root = ET.Element("robot", {"name": "H2"})
    ET.SubElement(root, "link", {"name": "pelvis"})
    parent = "pelvis"
    for index, joint_name in enumerate(contract.expected_joint_order):
        child = f"joint_link_{index}"
        ET.SubElement(root, "link", {"name": child})
        joint = ET.SubElement(root, "joint", {"name": joint_name, "type": "revolute"})
        ET.SubElement(joint, "parent", {"link": parent})
        ET.SubElement(joint, "child", {"link": child})
        ET.SubElement(joint, "axis", {"xyz": "0 0 1"})
        ET.SubElement(joint, "limit", {"effort": "1", "lower": "-2", "upper": "2", "velocity": "1"})
        parent = child
    for frame in contract.flange_frames:
        ET.SubElement(root, "link", {"name": frame})
        joint = ET.SubElement(root, "joint", {"name": f"{frame}_fixed", "type": "fixed"})
        ET.SubElement(joint, "parent", {"link": "pelvis"})
        ET.SubElement(joint, "child", {"link": frame})
    existing_link_count = len(root.findall("link"))
    for index in range(contract.expected_body_count - existing_link_count):
        link_name = f"fixed_aux_{index}"
        ET.SubElement(root, "link", {"name": link_name})
        joint = ET.SubElement(root, "joint", {"name": f"{link_name}_joint", "type": "fixed"})
        ET.SubElement(joint, "parent", {"link": "pelvis"})
        ET.SubElement(joint, "child", {"link": link_name})
    path = tmp_path / "h2.urdf"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    matching = replace(
        contract,
        urdf_sha256=file_sha256(path),
        learned_joint_position_limits=tuple((name, (-2.0, 2.0)) for name in contract.learned_joint_limits),
    )

    info = matching.validate_urdf(path)

    assert info.root_link == "pelvis"
    assert set(info.actuated_joint_names) == set(contract.expected_joint_order)
    changed_limits = list(matching.learned_joint_position_limits)
    changed_limits[0] = (changed_limits[0][0], (-1.9, 2.0))
    drifted = replace(matching, learned_joint_position_limits=tuple(changed_limits))
    with pytest.raises(ValueError, match="position limits.*pinned"):
        drifted.validate_urdf(path)
    with pytest.raises(ValueError, match="unexpected Pink URDF"):
        contract.validate_urdf(path)

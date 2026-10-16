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

"""Behavioral checks for the deterministic H2plus Arena sidecar."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pxr", reason="OpenUSD is not installed on the CPU-only host image")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt  # noqa: E402

from e2e_robot_workflows.arena.h2_spawn import (  # noqa: E402
    _repair_published_sharpa_wrists,
    _wrist_body_path,
    _zero_pelvis_spawn_translation,
)

_INERTIAL_ATTRIBUTES = (
    "physics:centerOfMass",
    "physics:diagonalInertia",
    "physics:mass",
    "physics:principalAxes",
)
_MESH_ATTRIBUTES = (
    "extent",
    "faceVertexCounts",
    "faceVertexIndices",
    "points",
    "primvars:normals",
    "primvars:normals:indices",
)


def _author_mesh_data(prim: Usd.Prim, *, point_count: int, face_count: int, corrected: bool) -> None:
    value = 1.0 if corrected else 2.0
    prim.CreateAttribute("extent", Sdf.ValueTypeNames.Float3Array).Set(
        Vt.Vec3fArray([Gf.Vec3f(-value), Gf.Vec3f(value)])
    )
    prim.CreateAttribute("points", Sdf.ValueTypeNames.Point3fArray).Set(Vt.Vec3fArray(point_count, Gf.Vec3f(value)))
    prim.CreateAttribute("faceVertexCounts", Sdf.ValueTypeNames.IntArray).Set(Vt.IntArray(face_count, 3))
    prim.CreateAttribute("faceVertexIndices", Sdf.ValueTypeNames.IntArray).Set(Vt.IntArray(face_count * 3, 0))
    prim.CreateAttribute("primvars:normals", Sdf.ValueTypeNames.Normal3fArray).Set(
        Vt.Vec3fArray(face_count, Gf.Vec3f(0.0, 0.0, value))
    )
    normal_indices = prim.CreateAttribute("primvars:normals:indices", Sdf.ValueTypeNames.IntArray)
    normal_indices.Set(Sdf.ValueBlock() if corrected else Vt.IntArray(face_count, 0))


def _author_inertial(prim: Usd.Prim, *, wrist_only: bool) -> None:
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(0.19722487032413483 if wrist_only else 0.6160820126533508)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.027, 0.001, 0.002) if wrist_only else Gf.Vec3f(0.08, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(
        Gf.Vec3f(0.0002, 0.00021, 0.00016) if wrist_only else Gf.Vec3f(0.0019, 0.0017, 0.0005)
    )
    principal_axes = Gf.Quatf(0.9238795, Gf.Vec3f(0.3826834, 0.0, 0.0)) if wrist_only else Gf.Quatf(1.0)
    mass.CreatePrincipalAxesAttr(principal_axes)


def _make_pelvis_fixture(translation: tuple[float, float, float]) -> tuple[Usd.Stage, Usd.Prim]:
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/H2").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.Scope.Define(stage, "/H2/Geometry")
    pelvis = UsdGeom.Xform.Define(stage, "/H2/Geometry/pelvis")
    pelvis.AddTranslateOp().Set(Gf.Vec3d(*translation))
    pelvis.AddOrientOp().Set(Gf.Quatf(1.0))
    pelvis.AddScaleOp().Set(Gf.Vec3f(1.0))
    UsdPhysics.RigidBodyAPI.Apply(pelvis.GetPrim())
    child = UsdGeom.Xform.Define(stage, "/H2/Geometry/pelvis/child")
    child.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))
    return stage, child.GetPrim()


def test_pelvis_spawn_normalization_removes_only_the_pinned_root_offset() -> None:
    stage, child = _make_pelvis_fixture((0.0, 0.0, 1.2))
    pelvis = stage.GetPrimAtPath("/H2/Geometry/pelvis")
    paths_before = tuple(str(prim.GetPath()) for prim in stage.Traverse())
    assert tuple(
        UsdGeom.Xformable(child).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    ) == (
        0.0,
        0.0,
        1.7,
    )

    _zero_pelvis_spawn_translation(stage, (0.0, 0.0, 1.2))

    assert tuple(str(prim.GetPath()) for prim in stage.Traverse()) == paths_before
    assert pelvis.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdGeom.Xformable(pelvis).GetLocalTransformation() == Gf.Matrix4d(1.0)
    assert tuple(
        UsdGeom.Xformable(child).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    ) == (
        0.0,
        0.0,
        0.5,
    )


def test_pelvis_spawn_normalization_rejects_source_transform_drift() -> None:
    stage, _ = _make_pelvis_fixture((0.0, 0.0, 1.1))

    try:
        _zero_pelvis_spawn_translation(stage, (0.0, 0.0, 1.2))
    except RuntimeError as error:
        assert "pelvis spawn transform changed" in str(error)
    else:
        raise AssertionError("pelvis transform drift was accepted")


def test_pelvis_spawn_normalization_rejects_reset_xform_stack() -> None:
    stage, _ = _make_pelvis_fixture((0.0, 0.0, 1.2))
    UsdGeom.Xformable(stage.GetPrimAtPath("/H2/Geometry/pelvis")).SetResetXformStack(True)

    try:
        _zero_pelvis_spawn_translation(stage, (0.0, 0.0, 1.2))
    except RuntimeError as error:
        assert "pelvis xform operations changed" in str(error)
    else:
        raise AssertionError("pelvis reset xform stack was accepted")


def _make_wrist_repair_fixture(tmp_path: Path) -> tuple[Usd.Stage, Path]:
    source_usd = tmp_path / "source" / "unitree_h2.usda"
    hands_path = source_usd.parent / "payloads" / "hands" / "sharpa.usda"
    hands_path.parent.mkdir(parents=True)
    hands_stage = Usd.Stage.CreateNew(str(hands_path))
    stage = Usd.Stage.CreateInMemory()

    for side, (corrected_points, corrected_faces) in {
        "left": (1479, 2958),
        "right": (1472, 2944),
    }.items():
        body_path = _wrist_body_path(side)
        source_body = UsdGeom.Xform.Define(hands_stage, body_path).GetPrim()
        body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
        _author_inertial(source_body, wrist_only=True)
        _author_inertial(body, wrist_only=False)

        for suffix, collision in ((f"{side}_wrist_yaw_link", False), (f"{side}_wrist_yaw_link_1", True)):
            wrapper_path = f"{body_path}/{suffix}"
            source_wrapper = UsdGeom.Xform.Define(hands_stage, wrapper_path).GetPrim()
            wrapper = UsdGeom.Xform.Define(stage, wrapper_path).GetPrim()
            _author_mesh_data(
                source_wrapper,
                point_count=corrected_points,
                face_count=corrected_faces,
                corrected=True,
            )
            _author_mesh_data(
                wrapper,
                point_count=corrected_points,
                face_count=corrected_faces,
                corrected=True,
            )
            target = UsdGeom.Mesh.Define(stage, f"{wrapper_path}/{side}_wrist_yaw_link").GetPrim()
            _author_mesh_data(target, point_count=9643, face_count=19282, corrected=False)
            if collision:
                collision_api = UsdPhysics.CollisionAPI.Apply(target)
                collision_api.CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(target).CreateApproximationAttr("convexHull")

    hands_stage.SetDefaultPrim(hands_stage.GetPrimAtPath("/H2"))
    hands_stage.GetRootLayer().Save()
    stage.SetDefaultPrim(stage.GetPrimAtPath("/H2"))
    return stage, source_usd


def test_sharpa_wrist_repair_moves_the_pinned_geometry_without_changing_topology(tmp_path: Path) -> None:
    stage, source_usd = _make_wrist_repair_fixture(tmp_path)
    hands_stage = Usd.Stage.Open(str(source_usd.parent / "payloads" / "hands" / "sharpa.usda"))
    assert hands_stage is not None
    paths_before = [str(prim.GetPath()) for prim in stage.Traverse()]

    _repair_published_sharpa_wrists(stage, source_usd)

    assert [str(prim.GetPath()) for prim in stage.Traverse()] == paths_before
    for side, (point_count, face_count) in {"left": (1479, 2958), "right": (1472, 2944)}.items():
        body_path = _wrist_body_path(side)
        body = stage.GetPrimAtPath(body_path)
        source_body = hands_stage.GetPrimAtPath(body_path)
        assert body.GetAttribute("physics:mass").Get() == 0.19722487032413483
        for name in _INERTIAL_ATTRIBUTES:
            assert body.GetAttribute(name).Get() == source_body.GetAttribute(name).Get()
        for suffix, purpose in (
            (f"{side}_wrist_yaw_link", UsdGeom.Tokens.default_),
            (f"{side}_wrist_yaw_link_1", UsdGeom.Tokens.guide),
        ):
            wrapper = stage.GetPrimAtPath(f"{body_path}/{suffix}")
            target = stage.GetPrimAtPath(f"{body_path}/{suffix}/{side}_wrist_yaw_link")
            source_wrapper = hands_stage.GetPrimAtPath(f"{body_path}/{suffix}")
            for name in _MESH_ATTRIBUTES:
                assert not wrapper.GetAttribute(name)
                assert target.GetAttribute(name).Get() == source_wrapper.GetAttribute(name).Get()
            assert len(target.GetAttribute("points").Get()) == point_count
            assert len(target.GetAttribute("faceVertexCounts").Get()) == face_count
            assert target.GetAttribute("primvars:normals:indices").Get() is None
            assert UsdGeom.Mesh.ValidateTopology(
                target.GetAttribute("faceVertexIndices").Get(),
                target.GetAttribute("faceVertexCounts").Get(),
                len(target.GetAttribute("points").Get()),
            ) == (True, "")
            assert UsdGeom.PrimvarsAPI(target).GetPrimvar("normals").GetInterpolation() == UsdGeom.Tokens.uniform
            assert UsdGeom.Imageable(target).GetPurposeAttr().Get() == purpose
            collision = suffix.endswith("_1")
            assert target.HasAPI(UsdPhysics.CollisionAPI) is collision
            assert target.HasAPI(UsdPhysics.MeshCollisionAPI) is collision
            if collision:
                assert UsdPhysics.CollisionAPI(target).GetCollisionEnabledAttr().Get() is True
                assert UsdPhysics.MeshCollisionAPI(target).GetApproximationAttr().Get() == "convexHull"

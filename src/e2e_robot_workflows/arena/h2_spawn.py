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

"""Deterministic Arena runtime USD for the published H2plus Sharpa variant."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from e2e_robot_workflows.data.artifact_paths import file_sha256

_ROOT_BODY = "pelvis"
_ROOT_BODY_PATH = "/H2/Geometry/pelvis"
_NESTED_SHARPA_ROOTS = {"left_hand_flange", "right_hand_flange"}
_NESTED_ARTICULATION_APIS = {"NewtonArticulationRootAPI", "PhysicsArticulationRootAPI"}
_REQUIRED_GENERATION_SCHEMAS = ("MjcJointAPI", "NewtonArticulationRootAPI", "PhysxRigidBodyAPI")
_EXPECTED_VARIANTS = {"Physics": "physx", "hands": "sharpa", "collision_profile": "full"}
_WRIST_INERTIAL_ATTRIBUTES = (
    "physics:centerOfMass",
    "physics:diagonalInertia",
    "physics:mass",
    "physics:principalAxes",
)
_WRIST_MESH_ATTRIBUTES = (
    "extent",
    "faceVertexCounts",
    "faceVertexIndices",
    "points",
    "primvars:normals",
    "primvars:normals:indices",
)
_WRIST_MESH_COUNTS = {
    "left": (1479, 2958),
    "right": (1472, 2944),
}
_MERGED_WRIST_MESH_COUNTS = (9643, 19282)
_MERGED_WRIST_MASS = 0.6160820126533508
_WRIST_ONLY_MASS = 0.19722487032413483


def _wrist_body_path(side: str) -> str:
    return (
        "/H2/Geometry/pelvis/waist_yaw_link/waist_roll_link/torso_link/"
        f"{side}_shoulder_pitch_link/{side}_shoulder_roll_link/{side}_shoulder_yaw_link/"
        f"{side}_elbow_link/{side}_wrist_roll_link/{side}_wrist_pitch_link/{side}_wrist_yaw_link"
    )


def _zero_pelvis_spawn_translation(stage: object, expected_translation_xyz: tuple[float, float, float]) -> None:
    """Remove the source asset's authored world spawn offset from its root body.

    ``ArticulationCfg.init_state`` owns the live root pose. Keeping the published
    pelvis translation in the flattened sidecar applies that offset a second
    time to nested USD children such as the head camera.
    """

    from pxr import Gf, UsdGeom

    pelvis = stage.GetPrimAtPath(_ROOT_BODY_PATH)
    if not pelvis or pelvis.GetTypeName() != "Xform":
        raise RuntimeError(f"published H2plus pelvis layout changed: {_ROOT_BODY_PATH}")
    xformable = UsdGeom.Xformable(pelvis)
    operations = xformable.GetOrderedXformOps()
    operation_types = [operation.GetOpType() for operation in operations]
    expected_operation_types = [
        UsdGeom.XformOp.TypeTranslate,
        UsdGeom.XformOp.TypeOrient,
        UsdGeom.XformOp.TypeScale,
    ]
    if xformable.GetResetXformStack() or operation_types != expected_operation_types:
        raise RuntimeError(
            "published H2plus pelvis xform operations changed: "
            f"reset_stack={xformable.GetResetXformStack()}, types={operation_types}"
        )
    expected = Gf.Matrix4d(1.0)
    expected.SetTranslateOnly(Gf.Vec3d(*expected_translation_xyz))
    actual = xformable.GetLocalTransformation()
    if actual != expected:
        raise RuntimeError(
            "published H2plus pelvis spawn transform changed: "
            f"expected translation-only {expected_translation_xyz}, found {actual}"
        )

    value_type = type(operations[0].Get())
    if not operations[0].Set(value_type(0.0, 0.0, 0.0)):
        raise RuntimeError("failed to zero the H2plus pelvis spawn translation")
    if xformable.GetLocalTransformation() != Gf.Matrix4d(1.0):
        raise RuntimeError("failed to normalize the H2plus pelvis spawn transform")


def _repair_published_sharpa_wrists(stage: object, source_usd: Path) -> None:
    """Move the package's wrist-only opinions onto its expanded mesh prims.

    The published H2plus ``hands=sharpa`` layer contains the correct wrist-only
    inertials and meshes, but its mesh attributes land on instance-root Xforms
    instead of the mesh prims below them.  The Physics variant also wins over
    its inertial opinions.  Copy those already-pinned values to their intended
    prims after flattening, without introducing another asset dependency.
    """

    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    hands_path = source_usd.parent / "payloads" / "hands" / "sharpa.usda"
    hands_layer = Sdf.Layer.FindOrOpen(str(hands_path))
    if hands_layer is None:
        raise RuntimeError(f"published H2plus Sharpa layer is missing: {hands_path}")

    root = stage.GetDefaultPrim()
    before_paths = tuple(str(prim.GetPath()) for prim in Usd.PrimRange(root))
    destination = stage.GetRootLayer()
    for side in ("left", "right"):
        body_path = Sdf.Path(_wrist_body_path(side))
        body = stage.GetPrimAtPath(body_path)
        if not body or body.GetTypeName() != "Xform" or not body.HasAPI(UsdPhysics.MassAPI):
            raise RuntimeError(f"published H2plus {side} wrist body layout changed: {body_path}")
        if body.GetAttribute("physics:mass").Get() != _MERGED_WRIST_MASS:
            raise RuntimeError(
                f"published H2plus {side} merged wrist mass changed: {body.GetAttribute('physics:mass').Get()}"
            )

        source_body = hands_layer.GetPrimAtPath(body_path)
        if source_body is None:
            raise RuntimeError(f"published H2plus Sharpa layer lacks {body_path}")
        source_mass = hands_layer.GetPropertyAtPath(body_path.AppendProperty("physics:mass"))
        if source_mass is None or source_mass.default != _WRIST_ONLY_MASS:
            raise RuntimeError(f"published H2plus {side} wrist-only mass changed")
        for name in _WRIST_INERTIAL_ATTRIBUTES:
            source_path = body_path.AppendProperty(name)
            if hands_layer.GetPropertyAtPath(source_path) is None:
                raise RuntimeError(f"published H2plus Sharpa layer lacks {source_path}")
            if not Sdf.CopySpec(hands_layer, source_path, destination, source_path):
                raise RuntimeError(f"failed to restore H2plus Sharpa inertial {source_path}")

        source_mesh_defaults: dict[str, object] | None = None
        merged_mesh_defaults: dict[str, object] | None = None
        corrected_point_count, corrected_face_count = _WRIST_MESH_COUNTS[side]
        for suffix, collision in ((f"{side}_wrist_yaw_link", False), (f"{side}_wrist_yaw_link_1", True)):
            wrapper_path = body_path.AppendChild(suffix)
            target_path = wrapper_path.AppendChild(f"{side}_wrist_yaw_link")
            wrapper = stage.GetPrimAtPath(wrapper_path)
            target = stage.GetPrimAtPath(target_path)
            if not wrapper or wrapper.GetTypeName() != "Xform" or not target or target.GetTypeName() != "Mesh":
                raise RuntimeError(f"published H2plus {side} wrist mesh layout changed: {target_path}")
            if target.HasAPI(UsdPhysics.CollisionAPI) is not collision:
                raise RuntimeError(f"published H2plus {side} wrist collision layout changed: {target_path}")
            actual_counts = (
                len(target.GetAttribute("points").Get() or ()),
                len(target.GetAttribute("faceVertexCounts").Get() or ()),
            )
            if actual_counts != _MERGED_WRIST_MESH_COUNTS:
                raise RuntimeError(f"published H2plus {side} merged wrist mesh changed: {actual_counts}")
            actual_defaults = {name: target.GetAttribute(name).Get() for name in _WRIST_MESH_ATTRIBUTES}
            if merged_mesh_defaults is not None and actual_defaults != merged_mesh_defaults:
                raise RuntimeError(f"published H2plus {side} merged visual and collision wrist meshes differ")
            merged_mesh_defaults = actual_defaults

            defaults: dict[str, object] = {}
            for name in _WRIST_MESH_ATTRIBUTES:
                source_path = wrapper_path.AppendProperty(name)
                source_property = hands_layer.GetPropertyAtPath(source_path)
                if source_property is None:
                    raise RuntimeError(f"published H2plus Sharpa layer lacks {source_path}")
                defaults[name] = source_property.default
            corrected_counts = (len(defaults["points"]), len(defaults["faceVertexCounts"]))
            if corrected_counts != (corrected_point_count, corrected_face_count):
                raise RuntimeError(f"published H2plus {side} wrist-only mesh changed: {corrected_counts}")
            if len(defaults["primvars:normals"]) != corrected_face_count:
                raise RuntimeError(f"published H2plus {side} wrist-only normals changed")
            if source_mesh_defaults is not None and defaults != source_mesh_defaults:
                raise RuntimeError(f"published H2plus {side} visual and collision wrist meshes differ")
            source_mesh_defaults = defaults

            for name in _WRIST_MESH_ATTRIBUTES:
                source_path = wrapper_path.AppendProperty(name)
                target_property_path = target_path.AppendProperty(name)
                if not Sdf.CopySpec(hands_layer, source_path, destination, target_property_path):
                    raise RuntimeError(f"failed to restore H2plus Sharpa mesh {target_property_path}")
                wrapper_property = destination.GetPropertyAtPath(source_path)
                if wrapper_property is None or wrapper_property.default != source_mesh_defaults[name]:
                    raise RuntimeError(f"flattened H2plus Sharpa opinion is missing: {source_path}")
                destination.GetPrimAtPath(wrapper_path).RemoveProperty(wrapper_property)
            normals = UsdGeom.PrimvarsAPI(target).GetPrimvar("normals")
            if not normals.SetInterpolation(UsdGeom.Tokens.uniform):
                raise RuntimeError(f"failed to set uniform H2plus Sharpa normals on {target_path}")
            if collision:
                UsdGeom.Imageable(target).CreatePurposeAttr(UsdGeom.Tokens.guide)

        if body.GetAttribute("physics:mass").Get() != _WRIST_ONLY_MASS:
            raise RuntimeError(f"failed to restore H2plus {side} wrist-only inertial")
        for name in _WRIST_MESH_ATTRIBUTES:
            expected = source_mesh_defaults[name]
            for suffix in (f"{side}_wrist_yaw_link", f"{side}_wrist_yaw_link_1"):
                target_path = body_path.AppendChild(suffix).AppendChild(f"{side}_wrist_yaw_link")
                actual = destination.GetPropertyAtPath(target_path.AppendProperty(name))
                if actual is None or actual.default != expected:
                    raise RuntimeError(f"failed to restore H2plus Sharpa mesh property {target_path}.{name}")
                if name == "primvars:normals" and actual.GetInfo("interpolation") != UsdGeom.Tokens.uniform:
                    raise RuntimeError(f"failed to restore uniform H2plus Sharpa normals on {target_path}")

        collision_path = body_path.AppendChild(f"{side}_wrist_yaw_link_1").AppendChild(f"{side}_wrist_yaw_link")
        if UsdGeom.Imageable(stage.GetPrimAtPath(collision_path)).GetPurposeAttr().Get() != UsdGeom.Tokens.guide:
            raise RuntimeError(f"failed to hide H2plus {side} wrist collision mesh")

    after_paths = tuple(str(prim.GetPath()) for prim in Usd.PrimRange(root))
    if after_paths != before_paths:
        raise RuntimeError("H2plus Sharpa wrist repair changed articulation topology")


def materialize_h2plus_arena_usd(
    source_usd: Path,
    *,
    cache_base: Path,
    source_contract_sha256: str,
    variants: dict[str, str],
    source_pelvis_translation_xyz: tuple[float, float, float],
    recipe_version: int,
    expected_sha256: str,
    regenerate: bool = False,
) -> Path:
    """Flatten H2plus into one pinned articulation for Arena.

    The published Sharpa variant authors an articulation root on each hand.
    Isaac Lab rejects those roots nested below the pelvis articulation.  A
    reference overlay cannot mask them reliably after Kit composes the asset,
    so this sidecar removes the two APIs and flattens the selected variants.
    Instance roots are expanded first because USD's generated prototype names
    otherwise make the flattened bytes nondeterministic.
    """

    from pxr import Sdf, Usd, UsdPhysics

    if variants != _EXPECTED_VARIANTS:
        raise RuntimeError(f"H2plus Arena sidecar requires variants {_EXPECTED_VARIANTS}, found {variants}")

    output_dir = cache_base / "arena" / source_contract_sha256 / f"recipe-{recipe_version}" / expected_sha256
    output_path = output_dir / "unitree_h2_arena.usdc"
    if output_path.is_file() and not regenerate:
        digest = file_sha256(output_path)
        if digest != expected_sha256:
            raise ValueError(f"unexpected H2 Arena overlay: expected sha256:{expected_sha256}, found sha256:{digest}")
        return output_path

    registry = Usd.SchemaRegistry()
    missing_schemas = [name for name in _REQUIRED_GENERATION_SCHEMAS if not registry.GetTypeFromSchemaTypeName(name)]
    if missing_schemas:
        raise RuntimeError(f"H2plus Arena USD generation requires registered Isaac Sim schemas: {missing_schemas}")

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output_dir,
            prefix=".unitree_h2_arena.",
            suffix=".usdc",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        temporary_path.unlink()
        stage = Usd.Stage.CreateInMemory(load=Usd.Stage.LoadAll)
        root = stage.DefinePrim("/H2")
        root.GetReferences().AddReference(str(source_usd))
        stage.SetDefaultPrim(root)
        for name, value in variants.items():
            variant_set = root.GetVariantSet(name)
            if value not in variant_set.GetVariantNames():
                raise RuntimeError(f"published H2plus variant {name!r} no longer provides {value!r}")
            if not variant_set.SetVariantSelection(value):
                raise RuntimeError(f"failed to select published H2plus variant {name}={value}")

        instance_paths = [prim.GetPath() for prim in Usd.PrimRange(root) if prim.IsInstance()]
        for path in instance_paths:
            if not stage.GetPrimAtPath(path).SetInstanceable(False):
                raise RuntimeError(f"failed to expand H2plus instance {path}")
        remaining_instances = [str(prim.GetPath()) for prim in Usd.PrimRange(root) if prim.IsInstance()]
        if remaining_instances:
            raise RuntimeError(f"failed to expand H2plus instances: {remaining_instances}")

        flattened = stage.Flatten()
        stage = Usd.Stage.Open(flattened, load=Usd.Stage.LoadAll)
        root = stage.GetDefaultPrim()
        _zero_pelvis_spawn_translation(stage, source_pelvis_translation_xyz)
        _repair_published_sharpa_wrists(stage, source_usd)
        articulation_roots = [prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
        actual_names = {prim.GetName() for prim in articulation_roots}
        expected_names = {_ROOT_BODY, *_NESTED_SHARPA_ROOTS}
        if actual_names != expected_names:
            paths = sorted(str(prim.GetPath()) for prim in articulation_roots)
            raise RuntimeError(f"published H2plus articulation-root layout changed: {paths}")
        for prim in articulation_roots:
            if prim.GetName() in _NESTED_SHARPA_ROOTS:
                schemas = [schema for schema in prim.GetAppliedSchemas() if schema not in _NESTED_ARTICULATION_APIS]
                if not prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(schemas)):
                    raise RuntimeError(f"failed to join Sharpa root {prim.GetPath()} to the H2 articulation")
        remaining_roots = [
            str(prim.GetPath()) for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        if remaining_roots != ["/H2/Geometry/pelvis"]:
            raise RuntimeError(f"failed to normalize H2plus articulation roots: {remaining_roots}")

        if not stage.GetRootLayer().Export(str(temporary_path)):
            raise RuntimeError(f"failed to export flattened H2plus Arena USD: {temporary_path}")
        digest = file_sha256(temporary_path)
        if digest != expected_sha256:
            raise RuntimeError(
                f"generated H2 Arena USD changed: expected sha256:{expected_sha256}, found sha256:{digest}"
            )
        if output_path.exists():
            if output_path.read_bytes() != temporary_path.read_bytes():
                raise RuntimeError(f"content-addressed H2 Arena USD differs from regeneration: {output_path}")
            temporary_path.unlink()
            temporary_path = None
        else:
            os.replace(temporary_path, output_path)
            temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


__all__ = ["materialize_h2plus_arena_usd"]

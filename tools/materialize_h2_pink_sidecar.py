#!/usr/bin/env python3
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

"""Materialize the pinned H2plus package and deterministic Pink URDF."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import traceback
from importlib.metadata import version
from pathlib import Path

from e2e_robot_workflows.arena.h2_spawn import materialize_h2plus_arena_usd
from e2e_robot_workflows.robots.h2.pink_contract import (
    file_sha256,
    load_h2_pink_contract,
    materialize_published_usd,
    postprocess_kinematics_urdf,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = REPO_ROOT / "configs" / "robots" / "h2_pink.yaml"


def _default_cache_root() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME")
    return (Path(cache_home) if cache_home else Path.home() / ".cache") / "e2e-reference-app" / "h2plus"


def _verify_pinocchio(path: Path, expected_joint_names: tuple[str, ...], flange_frames: tuple[str, str]) -> None:
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(path))
    model_joint_names = set(model.names[1:])
    if model.nq != len(expected_joint_names) or model.nv != len(expected_joint_names):
        raise RuntimeError(f"Pinocchio expected nq=nv={len(expected_joint_names)}, got nq={model.nq}, nv={model.nv}")
    if model_joint_names != set(expected_joint_names):
        missing = sorted(set(expected_joint_names) - model_joint_names)
        unexpected = sorted(model_joint_names - set(expected_joint_names))
        raise RuntimeError(f"Pinocchio/live joint mismatch: missing={missing}, unexpected={unexpected}")
    missing_frames = [frame for frame in flange_frames if not model.existFrame(frame)]
    if missing_frames:
        raise RuntimeError(f"Pinocchio model is missing flange frames: {missing_frames}")


def _export_raw_urdf(
    source_usd: Path,
    raw_urdf: Path,
    variants: dict[str, str],
    simulation_app: object,
) -> None:
    import omni.kit.app

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_manager.set_extension_enabled_immediate("isaacsim.asset.exporter.urdf", True)
    simulation_app.update()
    from isaacsim.asset.exporter.urdf.converter.usd_to_urdf import UsdToUrdfConverter

    converter = UsdToUrdfConverter(
        str(source_usd),
        root_prim_path="/H2",
        mesh_dir_name="meshes",
        variant_selections=variants,
    )
    converter.convert(str(raw_urdf))


def _verify_source_pelvis_translation(
    source_usd: Path,
    variants: dict[str, str],
    expected_xyz: tuple[float, float, float],
) -> None:
    import numpy as np
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(source_usd), load=Usd.Stage.LoadAll)
    root = stage.GetDefaultPrim()
    for name, value in variants.items():
        root.GetVariantSet(name).SetVariantSelection(value)
    pelvis = stage.GetPrimAtPath("/H2/Geometry/pelvis")
    if not pelvis:
        raise RuntimeError("published H2plus pelvis prim is missing")
    transform = UsdGeom.Xformable(pelvis).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = tuple(float(value) for value in transform.ExtractTranslation())
    rotation = np.asarray(transform.ExtractRotationMatrix())
    if not np.allclose(translation, expected_xyz, atol=1e-9, rtol=0.0) or not np.allclose(
        rotation, np.eye(3), atol=1e-9, rtol=0.0
    ):
        raise RuntimeError(
            f"published H2plus pelvis transform changed: translation={translation}, rotation={rotation.tolist()}"
        )


def _generate(
    contract_path: Path,
    cache_root: Path,
    *,
    regenerate: bool,
    simulation_app: object,
) -> dict[str, object]:
    contract = load_h2_pink_contract(contract_path)
    actual_isaacsim = version("isaacsim")
    if actual_isaacsim != contract.isaacsim_version:
        raise RuntimeError(f"H2 Pink sidecar requires Isaac Sim {contract.isaacsim_version}, found {actual_isaacsim}")

    source_usd = materialize_published_usd(contract.simulation_usd, cache_root=cache_root / "usd")
    _verify_source_pelvis_translation(
        source_usd,
        contract.simulation_usd.variant_selection,
        contract.source_pelvis_translation_xyz,
    )
    recipe_root = cache_root / "pink" / contract.source_contract_sha256 / f"recipe-{contract.generation_recipe_version}"
    output_path = recipe_root / contract.urdf_sha256 / "unitree_h2.urdf"
    if output_path.is_file() and not regenerate:
        info = contract.validate_urdf(output_path)
        _verify_pinocchio(output_path, contract.expected_joint_order, contract.flange_frames)
    else:
        recipe_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=recipe_root, prefix="generation-") as temporary_directory:
            temporary_root = Path(temporary_directory)
            raw_urdf = temporary_root / "unitree_h2_raw.urdf"
            generated_urdf = temporary_root / "unitree_h2.urdf"
            _export_raw_urdf(source_usd, raw_urdf, contract.simulation_usd.variant_selection, simulation_app)
            postprocess_kinematics_urdf(
                raw_urdf,
                generated_urdf,
                root_translation_xyz=contract.source_pelvis_translation_xyz,
            )
            actual_digest = file_sha256(generated_urdf)
            if actual_digest != contract.urdf_sha256:
                raise RuntimeError(
                    f"generated H2 Pink URDF changed: expected sha256:{contract.urdf_sha256}, "
                    f"found sha256:{actual_digest}"
                )
            info = contract.validate_urdf(generated_urdf)
            _verify_pinocchio(generated_urdf, contract.expected_joint_order, contract.flange_frames)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                if output_path.read_bytes() != generated_urdf.read_bytes():
                    raise RuntimeError(f"content-addressed H2 Pink output differs from regeneration: {output_path}")
            else:
                os.replace(generated_urdf, output_path)

    arena_usd = materialize_h2plus_arena_usd(
        source_usd,
        cache_base=cache_root,
        source_contract_sha256=contract.source_contract_sha256,
        variants=contract.simulation_usd.variant_selection,
        source_pelvis_translation_xyz=contract.source_pelvis_translation_xyz,
        recipe_version=contract.arena_recipe_version,
        expected_sha256=contract.arena_usd_sha256,
        regenerate=regenerate,
    )

    return {
        "status": "ok",
        "simulation_usd": str(source_usd),
        "simulation_usd_sha256": contract.simulation_usd.sha256,
        "source_contract_sha256": contract.source_contract_sha256,
        "variants": contract.simulation_usd.variant_selection,
        "generation_recipe_version": contract.generation_recipe_version,
        "isaacsim": actual_isaacsim,
        "pink_urdf": str(output_path),
        "pink_urdf_sha256": contract.urdf_sha256,
        "arena_usd": str(arena_usd),
        "arena_usd_sha256": contract.arena_usd_sha256,
        "root_link": info.root_link,
        "link_count": len(info.link_names),
        "actuated_joint_count": len(info.actuated_joint_names),
        "flange_frames": list(contract.flange_frames),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--cache-root", type=Path, default=_default_cache_root())
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Run the exporter again and require byte-identical content instead of reusing the verified sidecar.",
    )
    args = parser.parse_args()
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        result = _generate(
            args.contract,
            args.cache_root,
            regenerate=args.regenerate,
            simulation_app=simulation_app,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
    except BaseException:
        traceback.print_exc()
        # Preserve the failure status before Kit's fast-shutdown path can
        # replace it with zero. This call does not return in fast-shutdown mode.
        simulation_app.close(skip_cleanup=True, exit_code=1)
        raise
    else:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

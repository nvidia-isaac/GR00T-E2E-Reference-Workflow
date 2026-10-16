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

"""Convert Isaac Lab HDF5 demos to a GR00T LeRobot dataset (or inspect the fields).

By default the script writes the dataset named by the stage config's ``writer``
entry point (GR00T LeRobot v2.1: parquet + mp4 + meta); ``meta/stats.json`` is
generated downstream by GR00T's ``gr00t/data/stats.py`` (the writer reports the
exact command). With ``--inspect`` it writes nothing and only reports whether the
configured state/action/camera fields resolve from the recorded episode schema.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Isaac Lab HDF5 demos to GR00T LeRobot (default); --inspect to only check fields."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Dataset config. Defaults by manifested embodiment; unmanifested inputs use the G1 config.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="HDF5 file. Defaults to data_root/hdf5_name in config.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Future LeRobot dataset root override.")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--lane",
        choices=("native", "arena"),
        default=None,
        help="Require a manifested lane. Dual-lane AutoData conversion also requires generation stage.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when configured fields are missing.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--inspect",
        action="store_true",
        help="Only report whether configured fields resolve; write nothing.",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Write the LeRobot dataset (default). Requires resolvable required fields.",
    )
    return parser.parse_args()


DEFAULT_WRITER = "e2e_robot_workflows.gr00t.lerobot_writer:write_lerobot_dataset"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_G1_CONFIG = REPO_ROOT / "configs/stages/gr00t_lerobot_dataset.yaml"
DEFAULT_H2_CONFIG = REPO_ROOT / "configs/stages/gr00t_h2_lerobot_dataset.yaml"
_MANIFEST_PROFILE_FIELDS = ("lane", "controller", "layout")


def read_optional_manifest(input_path: Path | None) -> dict[str, Any] | None:
    """Return a valid manifest, preserving only the explicit legacy no-manifest fallback."""

    if input_path is None or not input_path.is_file():
        return None
    import h5py

    from e2e_robot_workflows.data.dataset_manifest import MANIFEST_ATTRIBUTE, read_manifest

    try:
        with h5py.File(input_path, "r") as dataset:
            if MANIFEST_ATTRIBUTE not in dataset.attrs:
                return None
        return read_manifest(input_path, conversion=True)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Dataset manifest admission failed: {error}") from error


def resolve_config_path(requested: Path | None, input_path: Path | None) -> Path:
    """Select H2 only from its manifest, never from the coincidental 61-D width."""

    if requested is not None:
        return requested
    manifest = read_optional_manifest(input_path)
    if manifest is not None:
        controller = manifest.get("action", {}).get("controller")
        if controller == "h2_agile_pink":
            return DEFAULT_H2_CONFIG
        if isinstance(controller, str) and controller.startswith("h2_"):
            raise SystemExit(f"Unsupported H2 conversion source controller: {controller}")
    return DEFAULT_G1_CONFIG


def resolve_writer(config: dict[str, Any]):
    """Import the ``module:function`` writer named by the config (GR00T by default).

    The callable's contract is ``write(hdf5_path, config, output_root) -> dict``;
    a different VLA/policy target supplies its own config + writer.
    """
    import importlib

    spec = config.get("writer", DEFAULT_WRITER)
    module_path, _, func_name = spec.partition(":")
    if not module_path or not func_name:
        raise SystemExit(f"Config 'writer' must be 'module:function', got: {spec!r}")
    try:
        module = importlib.import_module(module_path)
        return getattr(module, func_name)
    except (ImportError, AttributeError) as exc:
        raise SystemExit(f"Could not import writer '{spec}': {exc}") from exc


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read dataset configs in this environment.") from exc

    try:
        with path.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file) or {}
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {path}: {error}") from error
    if not isinstance(data, dict):
        raise SystemExit(f"Expected mapping in config: {path}")
    return data


def resolve_hdf5_path(config: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override.expanduser()
    data_root = Path(config.get("data_root", ".")).expanduser()
    hdf5_name = config.get("hdf5_name")
    if not hdf5_name:
        raise SystemExit("Config must define hdf5_name or --input must be provided.")
    return data_root / hdf5_name


def flatten_hdf5_group(group: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    import h5py

    fields: dict[str, dict[str, Any]] = {}
    for key, value in group.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, h5py.Group):
            fields.update(flatten_hdf5_group(value, path))
        else:
            fields[path] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return fields


def inspect_hdf5(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("h5py is required to inspect Isaac Lab HDF5 datasets in this environment.") from exc

    with h5py.File(path, "r") as hdf5_file:
        if "data" not in hdf5_file:
            raise SystemExit(f"No /data group found in {path}")
        data_group = hdf5_file["data"]
        episode_names = sorted(data_group.keys())
        episodes = {}
        for episode_name in episode_names:
            episode_group = data_group[episode_name]
            episodes[episode_name] = {
                "attrs": {key: json_safe(value) for key, value in episode_group.attrs.items()},
                "fields": flatten_hdf5_group(episode_group),
            }
        return {
            "path": str(path),
            "attrs": {key: json_safe(value) for key, value in data_group.attrs.items()},
            "episode_count": len(episode_names),
            "episodes": episodes,
        }


def validate_manifest_admission(
    config: dict[str, Any],
    workflow_manifest: dict[str, Any] | None,
    inspection: dict[str, Any] | None,
) -> None:
    """Require the input provenance to match the conversion config's declared profiles."""

    admission = config.get("manifest_admission")
    if not isinstance(admission, dict):
        raise ValueError("config manifest_admission must be a mapping")
    required = admission.get("required")
    if not isinstance(required, bool):
        raise ValueError("config manifest_admission.required must be a boolean")

    raw_profiles = admission.get("accepted_profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("config manifest_admission.accepted_profiles must be a non-empty list")
    accepted_profiles: set[tuple[str, str, str]] = set()
    for index, profile in enumerate(raw_profiles):
        if not isinstance(profile, dict) or set(profile) != set(_MANIFEST_PROFILE_FIELDS):
            raise ValueError(
                f"config manifest_admission.accepted_profiles[{index}] must contain exactly "
                f"{list(_MANIFEST_PROFILE_FIELDS)}"
            )
        values = tuple(profile[field] for field in _MANIFEST_PROFILE_FIELDS)
        if not all(isinstance(value, str) and value for value in values):
            raise ValueError(f"config manifest_admission.accepted_profiles[{index}] values must be non-empty strings")
        accepted_profiles.add(values)
    if len(accepted_profiles) != len(raw_profiles):
        raise ValueError("config manifest_admission.accepted_profiles contains duplicates")

    raw_widths = admission.get("unmanifested_action_widths")
    if not isinstance(raw_widths, list) or any(
        isinstance(width, bool) or not isinstance(width, int) or width <= 0 for width in raw_widths
    ):
        raise ValueError("config manifest_admission.unmanifested_action_widths must be a list of positive integers")
    unmanifested_widths = set(raw_widths)
    if required and unmanifested_widths:
        raise ValueError("a manifest-required config cannot accept unmanifested action widths")
    if not required and not unmanifested_widths:
        raise ValueError("a manifest-optional config must declare accepted unmanifested action widths")

    if workflow_manifest is not None:
        action = workflow_manifest["action"]
        actual = (workflow_manifest["lane"], action["controller"], action["layout"])
        if actual not in accepted_profiles:
            raise ValueError(
                "dataset manifest profile is not accepted by this config: "
                f"lane={actual[0]!r}, controller={actual[1]!r}, layout={actual[2]!r}"
            )
        return
    if required:
        raise ValueError("this conversion config requires a workflow manifest")
    if inspection is None or not inspection["episodes"]:
        return

    actual_widths: set[int] = set()
    for episode_name, episode in inspection["episodes"].items():
        action_field = episode["fields"].get("actions")
        shape = action_field.get("shape") if isinstance(action_field, dict) else None
        if not isinstance(shape, list) or len(shape) != 2 or not isinstance(shape[1], int):
            raise ValueError(f"unmanifested data/{episode_name}/actions must have shape (T, W)")
        actual_widths.add(shape[1])
    unsupported = sorted(actual_widths - unmanifested_widths)
    if unsupported:
        raise ValueError(
            f"unmanifested action widths {unsupported} are not accepted by this config; "
            "a matching workflow manifest is required"
        )


def bind_manifest_video_fps(config: dict[str, Any], workflow_manifest: dict[str, Any] | None) -> None:
    """Bind H2 video cadence to the recorded control cadence."""

    if workflow_manifest is None or workflow_manifest["action"]["controller"] != "h2_agile_pink":
        return
    from e2e_robot_workflows.data.dataset_manifest import manifest_control_hz

    video = config.get("video")
    if not isinstance(video, dict):
        raise ValueError("H2 conversion config video must be a mapping")
    video["fps"] = manifest_control_hz(workflow_manifest)


def json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def resolve_field(expected_name: str, available_paths: set[str]) -> str | None:
    # Match the writer exactly: top-level and obs/ are explicit locations;
    # anything else is admitted only when its suffix is unique.
    for candidate in (expected_name, f"obs/{expected_name}"):
        if candidate in available_paths:
            return candidate
    suffix_matches = sorted(path for path in available_paths if path.endswith(f"/{expected_name}"))
    if len(suffix_matches) > 1:
        raise ValueError(f"field {expected_name!r} is ambiguous: {suffix_matches}")
    return suffix_matches[0] if len(suffix_matches) == 1 else None


def build_mapping_report(config: dict[str, Any], inspection: dict[str, Any] | None) -> dict[str, Any]:
    hdf5_fields = config.get("hdf5_fields", {}) or {}
    lerobot_fields = config.get("lerobot_fields", {}) or {}
    if not isinstance(hdf5_fields, dict) or not isinstance(lerobot_fields, dict):
        raise SystemExit("Config hdf5_fields and lerobot_fields must be mappings.")

    first_episode = None
    available_paths: set[str] = set()
    if inspection and inspection["episodes"]:
        first_episode = sorted(inspection["episodes"])[0]
        available_paths = set(inspection["episodes"][first_episode]["fields"])

    required_fields = set(config.get("required_fields", ("state", "action", "video")))
    missing_required_source_mappings = sorted(
        key for key in required_fields if not isinstance(hdf5_fields.get(key), str) or not hdf5_fields[key].strip()
    )
    missing_required_target_mappings = sorted(
        key
        for key in required_fields
        if not isinstance(lerobot_fields.get(key), str) or not lerobot_fields[key].strip()
    )
    mappings = []
    missing = []
    missing_required = []
    missing_optional = []
    for semantic_key, hdf5_name in hdf5_fields.items():
        resolved_path = resolve_field(str(hdf5_name), available_paths) if available_paths else None
        lerobot_key = lerobot_fields.get(semantic_key)
        item = {
            "semantic_key": semantic_key,
            "hdf5_name": hdf5_name,
            "resolved_hdf5_path": resolved_path,
            "lerobot_key": lerobot_key,
        }
        mappings.append(item)
        if first_episode is not None and resolved_path is None:
            missing.append(item)
            if semantic_key in required_fields:
                missing_required.append(item)
            else:
                missing_optional.append(item)

    # Modality groups the writer will leave out of the dataset's own modality.json, because
    # this recording does not produce the column they index (a fixed-base recording omits the
    # AGILE navigate/base-height commands the template references). Reported, not disqualifying.
    dropped_modality_columns: list[str] = []
    if first_episode is not None:
        from e2e_robot_workflows.gr00t.lerobot_writer import (
            _assembled_columns,
            _load_template,
            _modality_referenced_columns,
            _optional_columns,
        )

        modality = _load_template((config.get("paths", {}) or {}).get("modality_template_path"))
        if modality is not None:
            produced = {lerobot_fields.get("state"), lerobot_fields.get("action")}
            for opt_hdf5_name, lr_name in _optional_columns(config):
                if resolve_field(str(opt_hdf5_name), available_paths):
                    produced.add(lr_name)
            for lr_name, comp_names in _assembled_columns(config).items():
                if all(resolve_field(str(comp), available_paths) for comp in comp_names):
                    produced.add(lr_name)
            dropped_modality_columns = sorted(_modality_referenced_columns(modality) - produced)

    return {
        "language_instruction": config.get("language_instruction"),
        "task_index": config.get("task_index"),
        "first_episode_checked": first_episode,
        "mappings": mappings,
        "missing": missing,
        "missing_required": missing_required,
        "missing_required_source_mappings": missing_required_source_mappings,
        "missing_required_target_mappings": missing_required_target_mappings,
        "missing_optional": missing_optional,
        "dropped_modality_columns": dropped_modality_columns,
        "required_fields": sorted(required_fields),
        "ready_for_writer": bool(
            inspection is not None
            and first_episode is not None
            and not missing_required
            and not missing_required_source_mappings
            and not missing_required_target_mappings
        ),
    }


def main() -> None:
    args = parse_args()
    input_override = args.input.expanduser().resolve() if args.input is not None else None
    config_path = resolve_config_path(args.config, input_override).expanduser().resolve()
    try:
        from e2e_robot_workflows.gr00t.lerobot_writer import resolve_lerobot_config_paths

        config = resolve_lerobot_config_paths(load_yaml(config_path), config_path=config_path)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Dataset config admission failed: {error}") from error
    hdf5_path = resolve_hdf5_path(config, input_override).resolve()
    output_root = args.output or Path(config.get("data_root", ".")).expanduser()

    workflow_manifest = read_optional_manifest(hdf5_path)
    input_sha256 = None
    if args.lane is not None and workflow_manifest is None:
        raise SystemExit("Dataset manifest admission failed: a lane-qualified conversion requires a workflow manifest")
    if workflow_manifest is not None:
        from e2e_robot_workflows.data.dataset_manifest import file_sha256, validate_dataset

        try:
            workflow_manifest = validate_dataset(
                hdf5_path,
                require_lane=args.lane if args.lane is not None else None,
                require_stage="generation" if args.lane is not None else None,
                conversion=True,
            )
        except (OSError, ValueError) as error:
            raise SystemExit(f"Dataset manifest admission failed: {error}") from error
        input_sha256 = file_sha256(hdf5_path)

    inspection = None
    if hdf5_path.exists():
        inspection = inspect_hdf5(hdf5_path)
    elif args.strict:
        raise SystemExit(f"HDF5 file does not exist: {hdf5_path}")

    try:
        validate_manifest_admission(config, workflow_manifest, inspection)
        bind_manifest_video_fps(config, workflow_manifest)
    except ValueError as error:
        raise SystemExit(f"Dataset config admission failed: {error}") from error

    try:
        mapping_report = build_mapping_report(config, inspection)
    except ValueError as error:
        raise SystemExit(f"Dataset field admission failed: {error}") from error
    do_write = not args.inspect  # write is the default; --inspect opts out
    write_root = (args.output or (Path(config.get("data_root", ".")).expanduser() / "lerobot")).expanduser().resolve()
    report_path = args.report.expanduser().resolve() if args.report is not None else None
    from e2e_robot_workflows.gr00t.lerobot_writer import lerobot_config_input_paths

    protected_inputs = {hdf5_path, config_path, *lerobot_config_input_paths(config)}
    if report_path in protected_inputs:
        raise SystemExit(f"Report path must differ from every conversion input: {report_path}")
    if do_write and report_path is not None and (report_path == write_root or report_path.is_relative_to(write_root)):
        raise SystemExit(f"Report path must remain outside LeRobot output directory: {report_path}")
    if do_write and report_path is not None and write_root.is_relative_to(report_path):
        raise SystemExit(f"LeRobot output directory must remain outside report path: {write_root}")
    report = {
        "status": (
            "missing_input" if inspection is None else "ok" if mapping_report["ready_for_writer"] else "not_ready"
        ),
        "config": str(config_path),
        "input_hdf5": str(hdf5_path),
        "output_root": str(write_root if do_write else output_root),
        "inspection": inspection,
        "mapping_report": mapping_report,
        "workflow_manifest": workflow_manifest,
        "input_sha256": input_sha256,
    }

    if do_write:
        if not mapping_report["ready_for_writer"]:
            report["write_result"] = {
                "status": "skipped",
                "reason": "required fields or mappings not resolvable",
            }
            report["status"] = "not_ready"
        else:
            write = resolve_writer(config)
            try:
                from e2e_robot_workflows.data.artifact_paths import (
                    atomic_write_text,
                    publish_directory_without_replace,
                )

                if hdf5_path == write_root or hdf5_path.is_relative_to(write_root):
                    raise ValueError(f"LeRobot output directory must not contain its input dataset: {write_root}")
                if os.path.lexists(write_root):
                    raise FileExistsError(f"refusing to replace existing LeRobot output: {write_root}")
                write_root.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix=f".{write_root.name}.conversion-", dir=write_root.parent
                ) as staging_parent:
                    staging_root = Path(staging_parent) / "dataset"
                    write_result = write(hdf5_path, config, staging_root)
                    if not isinstance(write_result, dict) or write_result.get("status") != "ok":
                        raise ValueError(f"dataset writer did not report success: {write_result!r}")
                    if not staging_root.is_dir():
                        raise ValueError(f"dataset writer did not create its output directory: {staging_root}")
                    if workflow_manifest is not None:
                        staged_lineage_path = staging_root / "meta" / "e2e_workflow_lineage.json"
                        atomic_write_text(
                            staged_lineage_path,
                            json.dumps(
                                {
                                    "lane": workflow_manifest["lane"],
                                    "source_hdf5_sha256": input_sha256,
                                    "source_manifest": workflow_manifest,
                                },
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n",
                        )
                    publish_directory_without_replace(staging_root, write_root)

                for key, value in tuple(write_result.items()):
                    if isinstance(value, str):
                        write_result[key] = value.replace(str(staging_root), str(write_root))
                if workflow_manifest is not None:
                    write_result["lineage_path"] = str(write_root / "meta" / "e2e_workflow_lineage.json")
                report["write_result"] = write_result
            except Exception as exc:  # noqa: BLE001 - surface writer failures in the report
                report["write_result"] = {"status": "error", "error": repr(exc)}
                report["status"] = "write_failed"

    # Derive next_step from the actual outcome, not just the requested mode.
    write_status = report.get("write_result", {}).get("status")
    if not do_write:
        report["next_step"] = (
            "Fields resolve; run without --inspect to produce the LeRobot dataset."
            if mapping_report["ready_for_writer"]
            else "Required fields or mappings are unresolved; fix the config or recording before writing."
        )
    elif write_status == "ok":
        report["next_step"] = "Run gr00t/data/stats.py to generate meta/stats.json, then load in GR00T."
    elif write_status == "skipped":
        report["next_step"] = (
            "Nothing written: required fields or mappings not resolvable; fix the config or recording."
        )
    else:
        report["next_step"] = "Write failed; inspect write_result for details and retry."

    report_text = json.dumps(report, indent=2, sort_keys=True)
    if report_path is not None:
        from e2e_robot_workflows.data.artifact_paths import atomic_write_text

        atomic_write_text(report_path, report_text + "\n")
    print(report_text)

    if args.strict and not mapping_report["ready_for_writer"]:
        raise SystemExit(1)
    if do_write and report.get("write_result", {}).get("status") not in ("ok", None):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

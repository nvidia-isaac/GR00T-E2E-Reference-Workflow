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

"""Write Isaac Lab HDF5 demonstrations into the GR00T LeRobot v2.1 dataset format.

Consumed by ``scripts/convert_hdf5_to_lerobot.py`` (the convert stage). The
output layout, field mapping, fps, and meta templates all come from
``configs/stages/gr00t_lerobot_dataset.yaml`` -- this module holds no task-specific
constants.

Output (GR00T LeRobot ``codebase_version`` v2.1)::

    <root>/meta/{info.json, tasks.jsonl, episodes.jsonl, modality.json}
    <root>/data/chunk-000/episode_000000.parquet ...
    <root>/videos/chunk-000/<video_key>/episode_000000.mp4 ...

Per-frame video is stored in the mp4 (not the parquet); the parquet holds the
low-dimensional features plus indices, synced by ``timestamp``.

Recorded ``state``/``action`` joint arrays are in Isaac Lab's articulation order;
they are reordered into GR00T's grouped policy order (from the ``robot`` joint
configs) so ``modality.json``'s per-group slices line up.

``meta/stats.json`` (per-feature mean/std/min/max/q01/q99) is intentionally NOT
written here: GR00T's loader generates it from the parquet columns via
``gr00t/data/stats.py``. Surface the exact downstream command.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from e2e_robot_workflows.data.artifact_paths import publish_directory_without_replace

_VIDEO_CODEC = "h264"
_VIDEO_PIX_FMT = "yuv420p"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_FILE_FIELDS = (
    ("robot", "policy_joints_config_path"),
    ("robot", "action_joints_config_path"),
    ("robot", "state_joints_config_path"),
    ("paths", "modality_template_path"),
    ("paths", "info_template_path"),
)


def lerobot_config_input_paths(config: dict[str, Any]) -> set[Path]:
    """Return the resolved files a conversion config reads."""

    paths: set[Path] = set()
    for section_name, field_name in _CONFIG_FILE_FIELDS:
        value = (config.get(section_name) or {}).get(field_name)
        if isinstance(value, str) and value:
            paths.add(Path(value).expanduser().resolve())
    return paths


def resolve_lerobot_config_paths(
    config: dict[str, Any],
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve repository config references independently of the process CWD.

    Paths in repository-owned stage configs are relative to the repository root.
    Paths in an external ``--config`` are relative to that config's directory.
    The returned copy contains absolute paths and rejects every configured file
    that is absent, so conversion cannot silently skip remapping or templates.
    """

    resolved = deepcopy(config)
    if config_path is None:
        reference_root = _REPO_ROOT
    else:
        resolved_config_path = config_path.expanduser().resolve()
        try:
            resolved_config_path.relative_to(_REPO_ROOT)
        except ValueError:
            reference_root = resolved_config_path.parent
        else:
            reference_root = _REPO_ROOT

    raw_data_root = resolved.get("data_root", ".")
    if not isinstance(raw_data_root, str) or not raw_data_root.strip():
        raise ValueError("config data_root must be a non-empty path")
    data_root = Path(raw_data_root).expanduser()
    if not data_root.is_absolute():
        data_root = reference_root / data_root
    resolved["data_root"] = str(data_root.resolve())

    for section_name, field_name in _CONFIG_FILE_FIELDS:
        section = resolved.get(section_name)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ValueError(f"config {section_name!r} must be a mapping")
        if field_name not in section:
            continue
        raw_path = section[field_name]
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"config {section_name}.{field_name} must be a non-empty path")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = reference_root / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"configured file {section_name}.{field_name} does not exist: {path}")
        section[field_name] = str(path)
        resolved[section_name] = section
    return resolved


def _staged_output_path(
    work: Path,
    template: str,
    *,
    label: str,
    required_root: str,
    **fields: Any,
) -> Path:
    """Format one config-owned output path and keep it inside ``work``."""

    if not isinstance(template, str) or not template.strip():
        raise ValueError(f"{label} template must be a non-empty relative path")
    try:
        relative = Path(template.format(**fields))
    except (IndexError, KeyError, ValueError) as error:
        raise ValueError(f"invalid {label} template {template!r}: {error}") from error
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts or "\\" in str(relative):
        raise ValueError(f"{label} template escapes the dataset staging directory: {relative}")
    if not relative.parts or relative.parts[0] != required_root:
        raise ValueError(f"{label} template must remain below {required_root}/: {relative}")
    destination = (work / relative).resolve()
    if not destination.is_relative_to(work.resolve()):
        raise ValueError(f"{label} template escapes the dataset staging directory: {relative}")
    return destination


def _resolve_field(demo: h5py.Group, hdf5_name: str) -> h5py.Dataset | None:
    """Find a recorded field by name: top-level, under obs/, or by suffix."""

    for candidate in (hdf5_name, f"obs/{hdf5_name}"):
        if candidate in demo:
            return demo[candidate]
    # Fall back to a unique suffix match anywhere in the episode tree.
    matches: list[tuple[str, h5py.Dataset]] = []

    def _visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and name.endswith(f"/{hdf5_name}"):
            matches.append((name, obj))

    demo.visititems(_visit)
    if len(matches) > 1:
        raise ValueError(f"field {hdf5_name!r} is ambiguous in episode {demo.name!r}: {[name for name, _ in matches]}")
    return matches[0][1] if matches else None


def _encode_video(frames: np.ndarray, path: Path, fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", format="ffmpeg", pixelformat=_VIDEO_PIX_FMT, macro_block_size=1
    )
    try:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame))
    finally:
        writer.close()


def _feature_names(config: dict[str, Any], key: str, dim: int) -> list[str]:
    """Joint names for the 43-DoF layout when available, else generic names."""

    order_path = (config.get("robot", {}) or {}).get(f"{key}_joints_config_path")
    if order_path and Path(order_path).exists():
        import yaml

        joints = (yaml.safe_load(Path(order_path).read_text()) or {}).get("joints", {})
        if (
            isinstance(joints, dict)
            and joints
            and all(isinstance(v, int) for v in joints.values())
            and len(joints) == dim
        ):
            return [name for name, _ in sorted(joints.items(), key=lambda kv: kv[1])]
    return [f"{key}_{i}" for i in range(dim)]


def _joint_remap(config: dict[str, Any], key: str) -> tuple[list[int] | None, list[str] | None]:
    """Permutation reordering a sim-order joint array (``key``) into GR00T's grouped
    policy order, plus the grouped joint names.

    Recorded joint fields (``robot_joint_pos``, ``joint_pos_target``) are in Isaac
    Lab's articulation order; ``modality.json`` slices assume the grouped policy
    order. For each
    grouped joint, pick its column from the sim-order array. Returns ``(None, None)``
    when the two joint configs are not both provided (write stays faithful).
    """

    robot = config.get("robot", {}) or {}
    sim_path = robot.get(f"{key}_joints_config_path")
    policy_path = robot.get("policy_joints_config_path")
    if not (sim_path and policy_path and Path(sim_path).exists() and Path(policy_path).exists()):
        return None, None

    import yaml

    sim = (yaml.safe_load(Path(sim_path).read_text()) or {}).get("joints", {})  # {name: sim_index}
    groups = (yaml.safe_load(Path(policy_path).read_text()) or {}).get("joints", {})  # {group: [names]}
    policy_names = [name for joint_list in groups.values() for name in joint_list]
    missing = [name for name in policy_names if name not in sim]
    if missing:
        raise ValueError(f"'{key}' policy joints absent from sim joint config {sim_path}: {missing}")
    return [sim[name] for name in policy_names], policy_names


def _apply_joint_remap(arr: np.ndarray, perm: list[int] | None, episode: str, key: str) -> np.ndarray:
    """Reorder columns of ``arr`` by ``perm`` (sim -> policy order); no-op if ``perm`` is None."""

    if perm is None:
        return arr
    if arr.ndim != 2 or arr.shape[1] != len(perm):
        raise ValueError(
            f"episode '{episode}' {key} width {arr.shape[1] if arr.ndim == 2 else arr.shape} "
            f"!= joint config length {len(perm)}; cannot remap sim->policy order"
        )
    return arr[:, perm]


def _optional_columns(config: dict[str, Any]) -> list[tuple[str, str]]:
    """``(hdf5_name, lerobot_column)`` for each optional field that maps to a
    LeRobot column -- extra low-dim signals (e.g. teleop commands) written verbatim
    beside state/action. Fields without a ``lerobot_fields`` entry are skipped."""

    hdf5_fields = config.get("hdf5_fields", {}) or {}
    lerobot_fields = config.get("lerobot_fields", {}) or {}
    specs = []
    semantics = dict.fromkeys([*(config.get("required_fields", ()) or ()), *(config.get("optional_fields", ()) or ())])
    for semantic in semantics:
        if semantic in {"state", "action", "video"}:
            continue
        hdf5_name, lr_name = hdf5_fields.get(semantic), lerobot_fields.get(semantic)
        if isinstance(hdf5_name, str) and hdf5_name and isinstance(lr_name, str) and lr_name:
            specs.append((hdf5_name, lr_name))
    return specs


def _read_optional(
    demo: h5py.Group,
    specs: list[tuple[str, str]],
    n: int,
    episode: str,
    required_columns: set[str],
) -> dict[str, np.ndarray]:
    """Resolve configured auxiliary columns, rejecting any required absence."""

    out: dict[str, np.ndarray] = {}
    for hdf5_name, lr_name in specs:
        ds = _resolve_field(demo, hdf5_name)
        if ds is None:
            if lr_name in required_columns:
                raise ValueError(f"episode '{episode}' missing required field '{lr_name}'")
            continue
        arr = np.asarray(ds[:], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2 or arr.shape[0] != n:
            raise ValueError(f"episode '{episode}' optional '{lr_name}' has shape {arr.shape}, expected ({n}, D)")
        out[lr_name] = arr
    return out


def _assembled_columns(config: dict[str, Any]) -> dict[str, list[str]]:
    """``{lerobot_column: [hdf5_name, ...]}`` from ``config['assembled_columns']`` --
    columns built by concatenating several recorded fields in order (e.g. eef_pose
    = left pos+quat, right pos+quat). Component keys resolve through ``hdf5_fields``."""

    hdf5_fields = config.get("hdf5_fields", {}) or {}
    out: dict[str, list[str]] = {}
    for lr_name, components in (config.get("assembled_columns", {}) or {}).items():
        out[lr_name] = [hdf5_fields.get(c, c) for c in components]
    return out


def _read_assembled(
    demo: h5py.Group,
    assembled: dict[str, list[str]],
    n: int,
    episode: str,
    required_columns: set[str],
) -> dict[str, np.ndarray]:
    """Concatenate assembled columns, rejecting missing components when required."""

    out: dict[str, np.ndarray] = {}
    for lr_name, comp_names in assembled.items():
        parts = []
        for hdf5_name in comp_names:
            ds = _resolve_field(demo, hdf5_name)
            if ds is None:
                if lr_name in required_columns:
                    raise ValueError(
                        f"episode '{episode}' required assembled field '{lr_name}' is missing component '{hdf5_name}'"
                    )
                parts = None
                break
            arr = np.asarray(ds[:], dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.ndim != 2 or arr.shape[0] != n:
                raise ValueError(
                    f"episode '{episode}' assembled '{lr_name}' component '{hdf5_name}' "
                    f"has shape {arr.shape}, expected ({n}, D)"
                )
            parts.append(arr)
        if parts is None:
            continue
        arr = np.concatenate(parts, axis=1)
        out[lr_name] = arr
    return out


def write_lerobot_dataset(hdf5_path: Path, config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """Convert every episode in ``hdf5_path`` into a LeRobot v2.1 dataset.

    Returns a summary dict (episodes, frames, warnings, output paths). Faithfully
    serializes the fields mapped in ``config['hdf5_fields']``; warns when a
    mapped field's width disagrees with ``modality.json`` rather than reshaping.
    """

    config = resolve_lerobot_config_paths(config)
    hdf5_fields = config["hdf5_fields"]
    lerobot_fields = config["lerobot_fields"]
    fps = int(config.get("video", {}).get("fps", 50))
    chunks_size = 1000
    paths = config.get("paths", {})
    data_tmpl = paths.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video_tmpl = paths.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    state_key, action_key, video_key = hdf5_fields["state"], hdf5_fields["action"], hdf5_fields["video"]
    lr_state, lr_action, lr_video = lerobot_fields["state"], lerobot_fields["action"], lerobot_fields["video"]

    hdf5_path = hdf5_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if hdf5_path == output_root or hdf5_path.is_relative_to(output_root):
        raise ValueError(f"LeRobot output directory must not contain its input dataset: {output_root}")
    if os.path.lexists(output_root):
        raise FileExistsError(f"refusing to replace existing LeRobot output: {output_root}")
    if not isinstance(lr_video, str) or not lr_video or "/" in lr_video or "\\" in lr_video:
        raise ValueError(f"LeRobot video field must be one path-safe component, got: {lr_video!r}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    # Build into a staging dir and move it into place only after every episode
    # converts, so a failure partway through never destroys an existing dataset.
    work = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))

    warnings: list[str] = []
    episodes_meta: list[dict[str, Any]] = []
    task = config.get("language_instruction", "")
    state_dim = action_dim = None
    video_hw: tuple[int, int] | None = None
    global_index = 0

    # Reorder recorded sim/articulation joint order into GR00T's grouped policy
    # order so modality.json's per-group slices line up (no-op unless configured).
    state_perm, state_names = _joint_remap(config, "state")
    action_perm, action_names = _joint_remap(config, "action")
    if state_perm is not None or action_perm is not None:
        logger.info("remapping state/action joints from sim order to grouped policy order")

    # Optional low-dim columns (teleop commands written verbatim, eef_pose assembled
    # from its parts) emitted beside state/action when the recorded fields are present.
    opt_specs = _optional_columns(config)
    assembled_specs = _assembled_columns(config)
    required_semantics = set(config.get("required_fields", ("state", "action", "video")))
    required_columns = {
        lerobot_fields[semantic]
        for semantic in required_semantics
        if semantic not in {"state", "action", "video"}
        and isinstance(lerobot_fields.get(semantic), str)
        and lerobot_fields[semantic]
    }
    opt_active: dict[str, int] | None = None

    try:
        (work / "meta").mkdir(parents=True, exist_ok=True)
        with h5py.File(hdf5_path, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0)
            for ep_index, name in enumerate(demos):
                demo = f["data"][name]
                state_ds = _resolve_field(demo, state_key)
                action_ds = _resolve_field(demo, action_key)
                video_ds = _resolve_field(demo, video_key)
                if state_ds is None or action_ds is None or video_ds is None:
                    raise ValueError(f"episode '{name}' missing a required field (state/action/video)")

                state = np.asarray(state_ds[:], dtype=np.float32)
                action = np.asarray(action_ds[:], dtype=np.float32)
                frames = np.asarray(video_ds[:])
                state = _apply_joint_remap(state, state_perm, name, "state")
                action = _apply_joint_remap(action, action_perm, name, "action")
                n = state.shape[0]
                if not (action.shape[0] == n == frames.shape[0]):
                    raise ValueError(
                        f"episode '{name}' has misaligned lengths: state {state.shape[0]}, "
                        f"action {action.shape[0]}, video {frames.shape[0]}"
                    )
                # Every episode must share the first episode's widths, else the
                # per-episode parquet would disagree with the single info.json.
                ep_state_dim, ep_action_dim = int(state.shape[1]), int(action.shape[1])
                ep_video_hw = (int(frames.shape[1]), int(frames.shape[2]))
                if state_dim is None:
                    state_dim, action_dim, video_hw = ep_state_dim, ep_action_dim, ep_video_hw
                elif (ep_state_dim, ep_action_dim, ep_video_hw) != (state_dim, action_dim, video_hw):
                    raise ValueError(
                        f"episode '{name}' dims (state {ep_state_dim}, action {ep_action_dim}, "
                        f"video {ep_video_hw}) differ from the first episode "
                        f"(state {state_dim}, action {action_dim}, video {video_hw}); "
                        "all episodes must share one layout"
                    )

                # video -> mp4
                video_path = _staged_output_path(
                    work,
                    video_tmpl,
                    label="video_path",
                    required_root="videos",
                    episode_chunk=ep_index // chunks_size,
                    video_key=lr_video,
                    episode_index=ep_index,
                )
                data_path = _staged_output_path(
                    work,
                    data_tmpl,
                    label="data_path",
                    required_root="data",
                    episode_chunk=ep_index // chunks_size,
                    episode_index=ep_index,
                )
                if video_path == data_path or video_path.exists() or data_path.exists():
                    raise ValueError(f"episode '{name}' maps to an already-written dataset path")
                _encode_video(frames, video_path, fps)

                # parquet rows (video lives in the mp4; parquet syncs by timestamp)
                idx = np.arange(n, dtype=np.int64)
                columns = {
                    lr_state: pa.array(list(state), type=pa.list_(pa.float32(), state_dim)),
                    lr_action: pa.array(list(action), type=pa.list_(pa.float32(), action_dim)),
                    "timestamp": pa.array((idx / fps).astype(np.float32)),
                    "frame_index": pa.array(idx),
                    "episode_index": pa.array(np.full(n, ep_index, dtype=np.int64)),
                    "index": pa.array(np.arange(global_index, global_index + n, dtype=np.int64)),
                    "task_index": pa.array(np.zeros(n, dtype=np.int64)),
                }
                # optional low-dim columns (verbatim + assembled, no joint remap) --
                # present only when recorded, and required to be uniform across episodes.
                ep_optional = _read_optional(demo, opt_specs, n, name, required_columns)
                ep_optional.update(_read_assembled(demo, assembled_specs, n, name, required_columns))
                if opt_active is None:
                    opt_active = {lr: arr.shape[1] for lr, arr in ep_optional.items()}
                elif set(ep_optional) != set(opt_active):
                    raise ValueError(
                        f"episode '{name}' optional columns {sorted(ep_optional)} != first episode {sorted(opt_active)}"
                    )
                for lr, arr in ep_optional.items():
                    if arr.shape[1] != opt_active[lr]:
                        raise ValueError(f"episode '{name}' optional '{lr}' width {arr.shape[1]} != {opt_active[lr]}")
                    columns[lr] = pa.array(list(arr), type=pa.list_(pa.float32(), arr.shape[1]))
                table = pa.table(columns)
                data_path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, data_path)

                episodes_meta.append({"episode_index": ep_index, "tasks": [task], "length": n})
                global_index += n

        total_frames = global_index
        total_episodes = len(episodes_meta)
        if total_episodes == 0:
            raise ValueError(f"no episodes found in {hdf5_path}: its '/data' group is empty")

        # modality-mismatch warning (faithful write, no silent reshape)
        modality = _load_template(paths.get("modality_template_path"))
        # The modality.json written below tells GR00T how to slice every low-dim column, so it
        # may only name columns this dataset holds -- GR00T KeyErrors on the rest. Runs that omit
        # an optional column ship the template minus those groups, which is what lets one
        # template serve fixed-base and AGILE runs of the same task alike.
        produced_lowdim = {lr_state, lr_action, *(opt_active or {})}
        dropped_groups = _prune_modality(modality, produced_lowdim)
        if dropped_groups:
            msg = (
                f"dropped modality groups whose columns this dataset lacks: {sorted(dropped_groups)}; "
                "fixed-base runs omit the AGILE navigate/base-height commands. Drop the matching "
                "action groups from the GR00T modality config used to train on this dataset.\n"
            )
            warnings.append(msg)
            logger.warning(msg)
        for feature_key, dim, mod_section in ((lr_state, state_dim, "state"), (lr_action, action_dim, "action")):
            expected = _modality_width(modality, mod_section, feature_key)
            if expected is not None and dim != expected:
                msg = (
                    f"{feature_key} recorded width {dim} != modality.json "
                    f"{mod_section} width {expected}; reconcile before GR00T training.\n"
                )
                warnings.append(msg)
                logger.warning(msg)

        # meta files
        features = _build_features(
            config, lr_state, lr_action, lr_video, state_dim, action_dim, video_hw, fps, state_names, action_names
        )
        for lr_name, dim in (opt_active or {}).items():
            features[lr_name] = {"dtype": "float32", "shape": [dim], "names": [f"{lr_name}_{i}" for i in range(dim)]}
        _write_info(work, config, paths, features, total_episodes, total_frames, chunks_size, fps)
        _write_jsonl(work / "meta" / "tasks.jsonl", [{"task_index": 0, "task": task}])
        _write_jsonl(work / "meta" / "episodes.jsonl", episodes_meta)
        if modality is not None:
            (work / "meta" / "modality.json").write_text(json.dumps(modality, indent=2) + "\n", encoding="utf-8")

        # Everything converted: publish the complete staging directory without
        # replacing any prior dataset, including one created by a racing writer.
        publish_directory_without_replace(work, output_root)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise

    # GR00T's loader requires meta/stats.json but generates it itself from the
    # parquet columns. Surface the exact downstream command.
    stats_cmd = f"python -c \"from gr00t.data.stats import generate_stats; generate_stats('{output_root}')\""
    logger.info(f"wrote {total_episodes} episodes / {total_frames} frames -> {output_root}")
    logger.info(f"next: generate meta/stats.json in the GR00T env -> {stats_cmd}")
    return {
        "status": "ok",
        "output_root": str(output_root),
        "episodes": total_episodes,
        "frames": total_frames,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "optional_columns": opt_active or {},
        "dropped_modality_groups": sorted(dropped_groups),
        "warnings": warnings,
        "stats_json": "generate in the GR00T env (loader requires meta/stats.json)",
        "generate_stats_cmd": stats_cmd,
    }


def _load_template(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    template_path = Path(path)
    if not template_path.is_file():
        raise ValueError(f"configured template does not exist: {template_path}")
    return json.loads(template_path.read_text())


def _modality_width(modality: dict[str, Any] | None, section: str, original_key: str) -> int | None:
    """Max slice end for entries in a modality section that target ``original_key``."""

    if not modality or section not in modality:
        return None
    end = None
    for entry in modality[section].values():
        if entry.get("original_key", original_key) == original_key:
            end = entry["end"] if end is None else max(end, entry["end"])
    return end


def _prune_modality(modality: dict[str, Any] | None, produced_columns: set[str]) -> dict[str, str]:
    """Drop the modality groups whose ``original_key`` column this dataset does not hold.

    Mutates the loaded template in place, so the caller writes a ``modality.json`` describing
    only what the parquet contains. Pruning rather than refusing is what lets one template serve
    a task's fixed-base and AGILE runs, which differ by exactly these optional columns.

    Returns:
        Each dropped group name mapped to the column it referenced.
    """

    dropped: dict[str, str] = {}
    for section in ("state", "action"):
        entries = (modality or {}).get(section) or {}
        for name, entry in list(entries.items()):
            key = entry.get("original_key") if isinstance(entry, dict) else None
            if key and key not in produced_columns:
                dropped[name] = key
                del entries[name]
    return dropped


def _modality_referenced_columns(modality: dict[str, Any] | None) -> set[str]:
    """LeRobot columns named by ``original_key`` in modality.json's state/action groups.

    GR00T's action/state heads index these low-dim columns directly, so a group naming one the
    dataset lacks is dropped at write time by :func:`_prune_modality`. Groups without an explicit
    ``original_key`` slice the section's base feature (``observation.state`` / ``action``), which
    is always written, so they are ignored here.
    """

    refs: set[str] = set()
    for section in ("state", "action"):
        for entry in ((modality or {}).get(section, {}) or {}).values():
            key = entry.get("original_key") if isinstance(entry, dict) else None
            if key:
                refs.add(key)
    return refs


def _build_features(
    config, lr_state, lr_action, lr_video, state_dim, action_dim, video_hw, fps, state_names=None, action_names=None
) -> dict[str, Any]:
    h, w = video_hw
    scalar = lambda dt: {"dtype": dt, "shape": [1], "names": None}  # noqa: E731
    return {
        lr_state: {
            "dtype": "float32",
            "shape": [state_dim],
            "names": state_names or _feature_names(config, "state", state_dim),
        },
        lr_action: {
            "dtype": "float32",
            "shape": [action_dim],
            "names": action_names or _feature_names(config, "action", action_dim),
        },
        lr_video: {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": float(fps),
                "video.height": h,
                "video.width": w,
                "video.channels": 3,
                "video.codec": _VIDEO_CODEC,
                "video.pix_fmt": _VIDEO_PIX_FMT,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        "timestamp": scalar("float32"),
        "frame_index": scalar("int64"),
        "episode_index": scalar("int64"),
        "index": scalar("int64"),
        "task_index": scalar("int64"),
    }


def _write_info(output_root, config, paths, features, total_episodes, total_frames, chunks_size, fps) -> None:
    info = _load_template(paths.get("info_template_path")) or {"codebase_version": "v2.1", "chunks_size": chunks_size}
    total_chunks = (total_episodes + chunks_size - 1) // chunks_size
    info.update(
        {
            "robot_type": (config.get("robot", {}) or {}).get("type"),
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_videos": total_episodes,
            "total_chunks": max(total_chunks, 1),
            "chunks_size": chunks_size,
            "fps": fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": paths.get("data_path"),
            "video_path": paths.get("video_path"),
            "features": features,
        }
    )
    (output_root / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

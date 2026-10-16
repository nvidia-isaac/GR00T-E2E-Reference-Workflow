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

"""Publish one staged Arena AutoData run without overwriting prior evidence."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from e2e_robot_workflows.data.artifact_paths import file_sha256


def _link_without_replace(source: Path, destination: Path) -> tuple[int, int]:
    """Create one public link and return the linked file identity."""

    source_status = source.stat(follow_symlinks=False)
    identity = source_status.st_dev, source_status.st_ino
    pinned_source = destination.with_name(f".{destination.name}.source-{uuid.uuid4().hex}")
    pin_created = False
    destination_created = False
    try:
        os.link(source, pinned_source)
        pin_created = True
        pinned_status = pinned_source.stat(follow_symlinks=False)
        if (pinned_status.st_dev, pinned_status.st_ino) != identity:
            raise RuntimeError(f"source artifact changed while linking: {source}")

        os.link(pinned_source, destination)
        destination_created = True
        status = destination.stat(follow_symlinks=False)
        if (status.st_dev, status.st_ino) != identity:
            raise RuntimeError(f"published artifact changed while linking: {destination}")

        pinned_source.unlink()
        pin_created = False
        return identity
    except BaseException as error:
        cleanup_errors: list[str] = []
        if destination_created:
            try:
                retained = _remove_created_link(destination, identity)
                if retained is not None:
                    cleanup_errors.append(f"preserved a concurrent artifact at quarantine path {retained}")
            except OSError as cleanup_error:
                cleanup_errors.append(str(cleanup_error))
        if pin_created:
            try:
                pinned_source.unlink(missing_ok=True)
            except OSError as cleanup_error:
                cleanup_errors.append(f"could not remove pinned source {pinned_source}: {cleanup_error}")
        if cleanup_errors:
            error.add_note("failed to roll back link attempt: " + "; ".join(cleanup_errors))
        raise


def _stage_json(payload: dict[str, Any], destination: Path) -> Path:
    """Write the result beside its destination before publishing any artifacts."""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = Path(stream.name)
        return temporary_path
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _remove_created_link(destination: Path, identity: tuple[int, int]) -> Path | None:
    """Atomically quarantine destination, removing it only when it is our link.

    A concurrent writer may replace ``destination`` at any point. Moving the
    current directory entry to an exclusive sibling name closes the previous
    stat-then-unlink race: subsequent inspection and removal operate on a name
    that is private to this rollback. A foreign entry is restored with hard-link
    no-replace semantics, or retained at the returned quarantine path if another
    writer has already occupied the public name.
    """

    descriptor, quarantine_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.rollback-",
        suffix=".quarantine",
    )
    os.close(descriptor)
    quarantine = Path(quarantine_name)
    try:
        os.replace(destination, quarantine)
    except FileNotFoundError:
        quarantine.unlink(missing_ok=True)
        return None
    except BaseException:
        quarantine.unlink(missing_ok=True)
        raise

    try:
        status = quarantine.stat(follow_symlinks=False)
    except OSError as error:
        raise OSError(
            error.errno,
            f"could not inspect quarantined artifact; retained at {quarantine}",
            str(destination),
        ) from error

    if (status.st_dev, status.st_ino) == identity:
        try:
            quarantine.unlink()
        except OSError as error:
            raise OSError(
                error.errno,
                f"could not remove this run's link; retained at {quarantine}",
                str(destination),
            ) from error
        return None

    try:
        os.link(quarantine, destination, follow_symlinks=False)
    except FileExistsError:
        return quarantine
    except OSError as error:
        raise OSError(
            error.errno,
            f"could not restore a foreign artifact; retained at {quarantine}",
            str(destination),
        ) from error
    # Keep the private link even after restoring the public name. Another writer
    # can replace that name immediately; the quarantine link guarantees its data
    # survives and gives the operator an exact recovery path.
    return quarantine


def promote_run(
    *,
    staging_result: Path,
    staging_output: Path,
    staging_failed_output: Path,
    output: Path,
    failed_output: Path,
    result: Path,
) -> dict[str, Any]:
    """Publish present HDF5 artifacts and rewrite their result entries to stable paths."""

    payload = json.loads(staging_result.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Arena AutoData result must be a JSON object")
    for target in (output, failed_output, result):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to replace existing Arena generation artifact: {target}")
    output.parent.mkdir(parents=True, exist_ok=True)
    failed_output.parent.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)

    artifacts = payload.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("Arena AutoData result artifacts must be a JSON object")
    staged_artifacts: list[tuple[Path, Path]] = []
    for key, source, destination in (
        ("generated", staging_output, output),
        ("failed", staging_failed_output, failed_output),
    ):
        if source.is_file():
            digest = file_sha256(source)
            staged_artifacts.append((source, destination))
            artifacts[key] = {"path": str(destination), "sha256": digest}
            if key == "generated":
                payload["generated_sha256"] = digest
        else:
            artifacts.pop(key, None)

    staged_result_output = _stage_json(payload, result)
    created_links: list[tuple[Path, tuple[int, int]]] = []
    try:
        for source, destination in (*staged_artifacts, (staged_result_output, result)):
            created_links.append((destination, _link_without_replace(source, destination)))
        staged_result_output.unlink()
    except BaseException as error:
        cleanup_errors: list[str] = []
        for destination, identity in reversed(created_links):
            try:
                retained = _remove_created_link(destination, identity)
                if retained is not None:
                    cleanup_errors.append(
                        f"{destination}: preserved a concurrent artifact at quarantine path {retained}"
                    )
            except OSError as cleanup_error:
                cleanup_errors.append(f"{destination}: {cleanup_error}")
        try:
            staged_result_output.unlink(missing_ok=True)
        except OSError as cleanup_error:
            cleanup_errors.append(f"{staged_result_output}: {cleanup_error}")
        if cleanup_errors:
            error.add_note("failed to roll back published artifacts: " + "; ".join(cleanup_errors))
        raise

    for source, _destination in staged_artifacts:
        source.unlink(missing_ok=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-result", required=True, type=Path)
    parser.add_argument("--staging-output", required=True, type=Path)
    parser.add_argument("--staging-failed-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--failed-output", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    payload = promote_run(
        staging_result=args.staging_result,
        staging_output=args.staging_output,
        staging_failed_output=args.staging_failed_output,
        output=args.output,
        failed_output=args.failed_output,
        result=args.result,
    )
    if payload.get("status") == "ok":
        if {"num_success", "requested_successes", "num_attempts"} <= payload.keys():
            print(
                f"Generation complete: {payload['num_success']}/{payload['requested_successes']} "
                f"successful demonstrations in {payload['num_attempts']} attempts."
            )
        print(f"Dataset: {args.output}")
        print(f"Result: {args.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

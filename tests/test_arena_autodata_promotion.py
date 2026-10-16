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
import json
import subprocess
import sys
from pathlib import Path

import pytest

from e2e_robot_workflows.autodata import promotion
from e2e_robot_workflows.autodata.promotion import promote_run


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize(
    ("status", "existing_output", "include_counts"),
    [("ok", False, True), ("error", False, True), ("ok", True, True), ("ok", False, False)],
)
def test_promotion_reports_completion_only_after_a_successful_publish(
    tmp_path: Path, status: str, existing_output: bool, include_counts: bool
) -> None:
    staging_output = tmp_path / "staged.hdf5"
    staging_result = tmp_path / "staged.json"
    output = tmp_path / "generated.hdf5"
    result = tmp_path / "generation.json"
    staging_output.write_bytes(b"generated")
    payload = {"status": status}
    if include_counts:
        payload.update(num_success=2, requested_successes=2, num_attempts=3)
    staging_result.write_text(json.dumps(payload), encoding="utf-8")
    if existing_output:
        output.write_bytes(b"previous run")

    command = subprocess.run(
        [
            sys.executable,
            "-m",
            "e2e_robot_workflows.autodata.promotion",
            "--staging-result",
            str(staging_result),
            "--staging-output",
            str(staging_output),
            "--staging-failed-output",
            str(tmp_path / "staged_failed.hdf5"),
            "--output",
            str(output),
            "--failed-output",
            str(tmp_path / "generated_failed.hdf5"),
            "--result",
            str(result),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if existing_output:
        assert command.returncode != 0
        assert output.read_bytes() == b"previous run"
        assert not result.exists()
    else:
        assert command.returncode == 0, command.stderr
        assert output.read_bytes() == b"generated"
        assert json.loads(result.read_text(encoding="utf-8"))["status"] == status
    if status == "ok" and not existing_output and include_counts:
        assert "Generation complete: 2/2 successful demonstrations in 3 attempts." in command.stdout
    else:
        assert "Generation complete:" not in command.stdout
    if status == "ok" and not existing_output:
        assert f"Dataset: {output}" in command.stdout
        assert f"Result: {result}" in command.stdout


def test_promotion_retains_staged_success_and_failure_artifacts(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    staging_output = staging / "generated.hdf5"
    staging_failed = staging / "generated_failed.hdf5"
    staging_result = staging / "result.json"
    staging_output.write_bytes(b"generated")
    staging_failed.write_bytes(b"failed")
    staging_result.write_text(
        json.dumps(
            {
                "status": "error",
                "artifacts": {
                    "generated": {"path": str(staging_output), "sha256": _digest(b"generated")},
                    "failed": {"path": str(staging_failed), "sha256": _digest(b"failed")},
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "published" / "generated.hdf5"
    failed_output = tmp_path / "published" / "generated_failed.hdf5"
    result = tmp_path / "results" / "run.json"

    payload = promote_run(
        staging_result=staging_result,
        staging_output=staging_output,
        staging_failed_output=staging_failed,
        output=output,
        failed_output=failed_output,
        result=result,
    )

    assert output.read_bytes() == b"generated"
    assert failed_output.read_bytes() == b"failed"
    assert not staging_output.exists()
    assert not staging_failed.exists()
    assert staging_result.is_file()
    assert payload["artifacts"] == {
        "generated": {"path": str(output), "sha256": _digest(b"generated")},
        "failed": {"path": str(failed_output), "sha256": _digest(b"failed")},
    }
    assert json.loads(result.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize("target_kind", ["regular_file", "dangling_symlink"])
def test_promotion_never_replaces_a_published_target(tmp_path: Path, target_kind: str) -> None:
    staging_result = tmp_path / "staging-result.json"
    staging_output = tmp_path / "staging-output.hdf5"
    staging_result.write_text('{"status": "ok"}', encoding="utf-8")
    staging_output.write_bytes(b"new")
    output = tmp_path / "output.hdf5"
    if target_kind == "regular_file":
        output.write_bytes(b"existing")
    else:
        output.symlink_to(tmp_path / "missing.hdf5")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        promote_run(
            staging_result=staging_result,
            staging_output=staging_output,
            staging_failed_output=tmp_path / "staging-failed.hdf5",
            output=output,
            failed_output=tmp_path / "failed.hdf5",
            result=tmp_path / "result.json",
        )

    if target_kind == "regular_file":
        assert output.read_bytes() == b"existing"
    else:
        assert output.is_symlink()
    assert staging_output.read_bytes() == b"new"
    assert not (tmp_path / "result.json").exists()


def test_promotion_rolls_back_public_artifacts_when_result_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging_output = tmp_path / "staging-output.hdf5"
    staging_failed = tmp_path / "staging-failed.hdf5"
    staging_result = tmp_path / "staging-result.json"
    staging_output.write_bytes(b"generated")
    staging_failed.write_bytes(b"failed")
    staging_result.write_text('{"status": "error"}', encoding="utf-8")
    output = tmp_path / "published" / "output.hdf5"
    failed_output = tmp_path / "published" / "failed.hdf5"
    result = tmp_path / "results" / "result.json"
    real_link = promotion.os.link

    def fail_result_link(source: Path, destination: Path) -> None:
        if Path(destination) == result:
            raise OSError("injected result publish failure")
        real_link(source, destination)

    monkeypatch.setattr(promotion.os, "link", fail_result_link)

    with pytest.raises(OSError, match="injected result publish failure"):
        promote_run(
            staging_result=staging_result,
            staging_output=staging_output,
            staging_failed_output=staging_failed,
            output=output,
            failed_output=failed_output,
            result=result,
        )

    assert staging_output.read_bytes() == b"generated"
    assert staging_failed.read_bytes() == b"failed"
    assert staging_result.is_file()
    assert not output.exists()
    assert not failed_output.exists()
    assert not result.exists()
    assert not list(result.parent.glob(f".{result.name}.*.tmp"))


def test_promotion_rollback_preserves_a_destination_created_by_another_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging_output = tmp_path / "staging-output.hdf5"
    staging_failed = tmp_path / "staging-failed.hdf5"
    staging_result = tmp_path / "staging-result.json"
    staging_output.write_bytes(b"generated")
    staging_failed.write_bytes(b"failed")
    staging_result.write_text('{"status": "error"}', encoding="utf-8")
    output = tmp_path / "published" / "output.hdf5"
    failed_output = tmp_path / "published" / "failed.hdf5"
    result = tmp_path / "results" / "result.json"
    real_link = promotion.os.link

    def race_failed_output(source: Path, destination: Path) -> None:
        if Path(destination) == failed_output:
            failed_output.write_bytes(b"other writer")
        real_link(source, destination)

    monkeypatch.setattr(promotion.os, "link", race_failed_output)

    with pytest.raises(FileExistsError):
        promote_run(
            staging_result=staging_result,
            staging_output=staging_output,
            staging_failed_output=staging_failed,
            output=output,
            failed_output=failed_output,
            result=result,
        )

    assert staging_output.read_bytes() == b"generated"
    assert staging_failed.read_bytes() == b"failed"
    assert not output.exists()
    assert failed_output.read_bytes() == b"other writer"
    assert not result.exists()


def test_promotion_detects_a_destination_replaced_immediately_after_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging_output = tmp_path / "staging-output.hdf5"
    staging_result = tmp_path / "staging-result.json"
    staging_output.write_bytes(b"generated")
    staging_result.write_text('{"status": "ok"}', encoding="utf-8")
    output = tmp_path / "published" / "output.hdf5"
    result = tmp_path / "results" / "result.json"
    real_link = promotion.os.link

    def replace_after_link(source: Path, destination: Path, **kwargs) -> None:
        real_link(source, destination, **kwargs)
        if Path(destination) == output:
            output.unlink()
            output.write_bytes(b"other writer")

    monkeypatch.setattr(promotion.os, "link", replace_after_link)

    with pytest.raises(RuntimeError, match="changed while linking") as captured:
        promote_run(
            staging_result=staging_result,
            staging_output=staging_output,
            staging_failed_output=tmp_path / "staging-failed.hdf5",
            output=output,
            failed_output=tmp_path / "published" / "failed.hdf5",
            result=result,
        )

    assert staging_output.read_bytes() == b"generated"
    assert output.read_bytes() == b"other writer"
    assert not result.exists()
    assert any("preserved a concurrent artifact" in note for note in captured.value.__notes__)
    assert not list(output.parent.glob(f".{output.name}.source-*"))


def test_promotion_does_not_publish_a_source_replaced_during_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging_output = tmp_path / "staging-output.hdf5"
    replacement = tmp_path / "replacement-output.hdf5"
    staging_result = tmp_path / "staging-result.json"
    staging_output.write_bytes(b"generated")
    replacement.write_bytes(b"replacement")
    staging_result.write_text('{"status": "ok"}', encoding="utf-8")
    output = tmp_path / "published" / "output.hdf5"
    result = tmp_path / "results" / "result.json"
    real_link = promotion.os.link

    def replace_source_before_pin(source: Path, destination: Path, **kwargs) -> None:
        if Path(source) == staging_output and Path(destination).name.startswith(f".{output.name}.source-"):
            replacement.replace(staging_output)
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(promotion.os, "link", replace_source_before_pin)

    with pytest.raises(RuntimeError, match="source artifact changed while linking"):
        promote_run(
            staging_result=staging_result,
            staging_output=staging_output,
            staging_failed_output=tmp_path / "staging-failed.hdf5",
            output=output,
            failed_output=tmp_path / "published" / "failed.hdf5",
            result=result,
        )

    assert staging_output.read_bytes() == b"replacement"
    assert not output.exists()
    assert not result.exists()
    assert not list(output.parent.glob(f".{output.name}.source-*"))


def test_promotion_rollback_preserves_a_replacement_created_after_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging_output = tmp_path / "staging-output.hdf5"
    staging_result = tmp_path / "staging-result.json"
    staging_output.write_bytes(b"generated")
    staging_result.write_text('{"status": "ok"}', encoding="utf-8")
    output = tmp_path / "published" / "output.hdf5"
    result = tmp_path / "results" / "result.json"
    real_link = promotion.os.link
    real_replace = promotion.os.replace

    def fail_result_link(source: Path, destination: Path, **kwargs) -> None:
        if Path(destination) == result:
            raise OSError("injected result publish failure")
        real_link(source, destination, **kwargs)

    def replace_then_race(source: Path, destination: Path) -> None:
        real_replace(source, destination)
        if Path(source) == output:
            output.write_bytes(b"other writer")

    monkeypatch.setattr(promotion.os, "link", fail_result_link)
    monkeypatch.setattr(promotion.os, "replace", replace_then_race)

    with pytest.raises(OSError, match="injected result publish failure"):
        promote_run(
            staging_result=staging_result,
            staging_output=staging_output,
            staging_failed_output=tmp_path / "staging-failed.hdf5",
            output=output,
            failed_output=tmp_path / "published" / "failed.hdf5",
            result=result,
        )

    assert staging_output.read_bytes() == b"generated"
    assert output.read_bytes() == b"other writer"
    assert not result.exists()
    assert not list(output.parent.glob(f".{output.name}.rollback-*.quarantine"))


def test_promotion_restores_and_quarantines_a_foreign_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging_output = tmp_path / "staging-output.hdf5"
    staging_result = tmp_path / "staging-result.json"
    staging_output.write_bytes(b"generated")
    staging_result.write_text('{"status": "ok"}', encoding="utf-8")
    output = tmp_path / "published" / "output.hdf5"
    result = tmp_path / "results" / "result.json"
    real_link = promotion.os.link

    def replace_then_fail(source: Path, destination: Path, **kwargs) -> None:
        if Path(destination) == result:
            output.unlink()
            output.write_bytes(b"other writer")
            raise OSError("injected result publish failure")
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(promotion.os, "link", replace_then_fail)

    with pytest.raises(OSError, match="injected result publish failure") as captured:
        promote_run(
            staging_result=staging_result,
            staging_output=staging_output,
            staging_failed_output=tmp_path / "staging-failed.hdf5",
            output=output,
            failed_output=tmp_path / "published" / "failed.hdf5",
            result=result,
        )

    quarantines = list(output.parent.glob(f".{output.name}.rollback-*.quarantine"))
    assert output.read_bytes() == b"other writer"
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes() == b"other writer"
    assert any(str(quarantines[0]) in note for note in captured.value.__notes__)
    assert staging_output.read_bytes() == b"generated"
    assert not result.exists()


def test_promotion_reports_quarantine_when_two_writers_replace_during_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging_output = tmp_path / "staging-output.hdf5"
    staging_result = tmp_path / "staging-result.json"
    staging_output.write_bytes(b"generated")
    staging_result.write_text('{"status": "ok"}', encoding="utf-8")
    output = tmp_path / "published" / "output.hdf5"
    result = tmp_path / "results" / "result.json"
    real_link = promotion.os.link

    def race_links(source: Path, destination: Path, **kwargs) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == result:
            output.unlink()
            output.write_bytes(b"first other writer")
            raise OSError("injected result publish failure")
        if source_path.name.endswith(".quarantine") and destination_path == output:
            output.write_bytes(b"second other writer")
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(promotion.os, "link", race_links)

    with pytest.raises(OSError, match="injected result publish failure") as captured:
        promote_run(
            staging_result=staging_result,
            staging_output=staging_output,
            staging_failed_output=tmp_path / "staging-failed.hdf5",
            output=output,
            failed_output=tmp_path / "published" / "failed.hdf5",
            result=result,
        )

    quarantines = list(output.parent.glob(f".{output.name}.rollback-*.quarantine"))
    assert output.read_bytes() == b"second other writer"
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes() == b"first other writer"
    assert any(str(quarantines[0]) in note for note in captured.value.__notes__)
    assert staging_output.read_bytes() == b"generated"
    assert not result.exists()

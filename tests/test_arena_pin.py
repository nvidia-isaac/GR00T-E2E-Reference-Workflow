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

"""Contracts for the external Isaac Lab Arena pin check.

Arena is discovered by path rather than locked by ``uv.lock``, so this check is the
only thing that keeps a composition or evaluation run honest about which Arena
revision it exercised. It is exercised here against throwaway git checkouts.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from check_arena_pin import ARENA_WBC_CONFIG_PATH, ArenaPinError, read_contract, verify_arena_pin  # noqa: E402

PINNED = read_contract()


def _write_runtime_contract(
    directory: Path,
    revision: str,
    version: str = PINNED["version"],
    policy_revision: str = PINNED["wbc_agile_policy_revision"],
) -> Path:
    """Write a minimal ``runtime.toml`` pinning the Arena revision, version, and policy."""

    runtime_path = directory / "runtime.toml"
    runtime_path.write_text(
        f'[tested]\nisaaclab_arena = "{version}"\nisaaclab_arena_revision = "{revision}"\n'
        f'arena_wbc_agile_policy_revision = "{policy_revision}"\n',
        encoding="utf-8",
    )
    return runtime_path


def _make_arena_checkout(
    directory: Path,
    version: str = PINNED["version"],
    name: str = "isaaclab_arena",
    policy_revision: str | None = PINNED["wbc_agile_policy_revision"],
    policy_config: str | None = None,
) -> str:
    """Create a one-commit git checkout that looks like Arena and return its revision.

    ``policy_revision`` of ``None`` leaves out the whole-body controller config, standing
    in for an Arena layout this check cannot read the locomotion policy out of.
    ``policy_config`` replaces that file's contents outright. Both are committed, because
    a checkout with uncommitted files fails the cleanliness rule before anything else.
    """

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "{version}"\n', encoding="utf-8")
    if policy_revision is not None or policy_config is not None:
        config_path = directory / ARENA_WBC_CONFIG_PATH
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            policy_config or f'class AgileConfig:\n    COMMIT_HASH_FOR_WBC_AGILE_POLICY = "{policy_revision}"\n',
            encoding="utf-8",
        )
    run = lambda *arguments: subprocess.run(  # noqa: E731 - terse local git helper
        ["git", "-C", str(directory), *arguments], check=True, capture_output=True, text=True
    )
    run("init", "--quiet")
    run("config", "user.email", "test@example.invalid")
    run("config", "user.name", "Test")
    run("config", "commit.gpgsign", "false")
    run("add", "--all")
    run("commit", "--quiet", "--message", "Arena")
    return run("rev-parse", "HEAD").stdout.strip()


def test_matching_checkout_reports_its_revision(tmp_path: Path) -> None:
    arena_root = tmp_path / "arena"
    revision = _make_arena_checkout(arena_root)

    pinned = verify_arena_pin(arena_root, _write_runtime_contract(tmp_path, revision))

    assert pinned == {
        "root": str(arena_root),
        "version": PINNED["version"],
        "revision": revision,
        "wbc_agile_policy_revision": PINNED["wbc_agile_policy_revision"],
    }


def test_wrong_revision_is_rejected(tmp_path: Path) -> None:
    arena_root = tmp_path / "arena"
    _make_arena_checkout(arena_root)
    runtime_path = _write_runtime_contract(tmp_path, "0" * 40)

    with pytest.raises(ArenaPinError, match="revision mismatch"):
        verify_arena_pin(arena_root, runtime_path)


def test_wrong_version_is_rejected(tmp_path: Path) -> None:
    arena_root = tmp_path / "arena"
    revision = _make_arena_checkout(arena_root, version="0.0.1")
    runtime_path = _write_runtime_contract(tmp_path, revision)

    with pytest.raises(ArenaPinError, match="version mismatch"):
        verify_arena_pin(arena_root, runtime_path)


def test_foreign_project_is_rejected(tmp_path: Path) -> None:
    arena_root = tmp_path / "arena"
    revision = _make_arena_checkout(arena_root, name="something_else")
    runtime_path = _write_runtime_contract(tmp_path, revision)

    with pytest.raises(ArenaPinError, match="not Isaac Lab Arena"):
        verify_arena_pin(arena_root, runtime_path)


@pytest.mark.parametrize(
    ("pyproject", "expected"),
    [
        ("this is not toml", "unreadable pyproject.toml"),
        ('[build-system]\nrequires = ["setuptools"]\n', "declares no project name and version"),
        ('[project]\nname = "isaaclab_arena"\n', "declares no project name and version"),
    ],
    ids=["malformed", "no-project-table", "no-version"],
)
def test_unreadable_project_metadata_is_reported_as_a_failed_pin(tmp_path: Path, pyproject: str, expected: str) -> None:
    """A checkout this repository does not own must not surface a parse traceback.

    ``E2E_ARENA_ROOT`` can name any directory, and the launchers and the preflight only
    handle ``ArenaPinError``, so a raw ``TOMLDecodeError`` or ``KeyError`` would escape
    as a crash instead of the actionable message the rest of the checks produce.
    """

    arena_root = tmp_path / "arena"
    revision = _make_arena_checkout(arena_root)
    runtime_path = _write_runtime_contract(tmp_path, revision)
    (arena_root / "pyproject.toml").write_text(pyproject, encoding="utf-8")

    with pytest.raises(ArenaPinError, match=expected):
        verify_arena_pin(arena_root, runtime_path)


def test_missing_checkout_is_rejected(tmp_path: Path) -> None:
    runtime_path = _write_runtime_contract(tmp_path, "0" * 40)

    with pytest.raises(ArenaPinError, match="does not contain an Isaac Lab Arena checkout"):
        verify_arena_pin(tmp_path / "absent", runtime_path)


@pytest.mark.parametrize("dirt", ["tracked", "untracked"])
def test_dirty_checkout_is_rejected(tmp_path: Path, dirt: str) -> None:
    """An unpinned edit changes what Arena imports without moving HEAD."""

    arena_root = tmp_path / "arena"
    revision = _make_arena_checkout(arena_root)
    runtime_path = _write_runtime_contract(tmp_path, revision)
    if dirt == "tracked":
        pyproject_path = arena_root / "pyproject.toml"
        pyproject_path.write_text(
            f"{pyproject_path.read_text(encoding='utf-8')}\n# uncommitted local edit\n", encoding="utf-8"
        )
    else:
        (arena_root / "stray_module.py").write_text("# not committed\n", encoding="utf-8")

    with pytest.raises(ArenaPinError, match="must be clean"):
        verify_arena_pin(arena_root, runtime_path)


def test_a_checkout_running_another_locomotion_policy_is_rejected(tmp_path: Path) -> None:
    """An Arena bump may change which locomotion policy loads; that must not be silent."""

    arena_root = tmp_path / "arena"
    revision = _make_arena_checkout(arena_root, policy_revision="a" * 40)
    runtime_path = _write_runtime_contract(tmp_path, revision)

    with pytest.raises(ArenaPinError, match="locomotion policy commit mismatch"):
        verify_arena_pin(arena_root, runtime_path)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, "no readable whole-body controller config"),
        ("class AgileConfig:\n    pass\n", "declares no WBC-AGILE policy commit"),
    ],
    ids=["missing", "no-commit"],
)
def test_a_checkout_whose_locomotion_policy_cannot_be_read_is_rejected(
    tmp_path: Path, config: str | None, expected: str
) -> None:
    """Reading the policy commit out of source must fail the pin, not raise."""

    arena_root = tmp_path / "arena"
    revision = _make_arena_checkout(arena_root, policy_revision=None, policy_config=config)
    runtime_path = _write_runtime_contract(tmp_path, revision)

    with pytest.raises(ArenaPinError, match=expected):
        verify_arena_pin(arena_root, runtime_path)


@pytest.mark.parametrize("pin", ["isaaclab_arena_revision", "arena_wbc_agile_policy_revision"])
def test_the_runtime_contract_pins_full_revisions(pin: str) -> None:
    """A short, moving, or symbolic revision would let a pin drift silently."""

    runtime = tomllib.loads((REPO_ROOT / "runtime.toml").read_text(encoding="utf-8"))["tested"]

    assert re.fullmatch(r"[0-9a-f]{40}", runtime[pin]), (
        f"{pin} must be a full 40-character hexadecimal commit SHA, got {runtime[pin]!r}"
    )

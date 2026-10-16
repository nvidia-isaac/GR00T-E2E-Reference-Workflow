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

"""Verify an external Isaac Lab Arena checkout against the pinned runtime contract.

Arena is an external source checkout discovered through ``E2E_ARENA_ROOT`` rather than
a resolved dependency, so nothing in ``uv.lock`` holds it to a revision. This check is
what stands between a stale or dirty Arena working tree and a run that silently
exercises different code than ``runtime.toml`` claims was tested. The launchers and the
runtime preflight both call it, so the shell and Python paths enforce the same rules.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARENA_PROJECT_NAME = "isaaclab_arena"
ARENA_WBC_CONFIG_PATH = Path("isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/config/configs.py")
_WBC_AGILE_REVISION_PATTERN = re.compile(r"""COMMIT_HASH_FOR_WBC_AGILE_POLICY\s*=\s*["']([0-9a-f]{40})["']""")
EXIT_MISMATCH = 2


class ArenaPinError(Exception):
    """An Arena checkout does not satisfy the pinned runtime contract."""


def _git(arena_root: Path, *arguments: str) -> str:
    """Return trimmed ``git`` output from the Arena checkout."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(arena_root), *arguments],
            capture_output=True,
            check=True,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - git is a hard requirement
        raise ArenaPinError("git is required to verify the Arena revision.") from exc
    except subprocess.CalledProcessError as exc:
        raise ArenaPinError(f"Arena checkout is not a git repository: {arena_root}") from exc
    return completed.stdout.strip()


def read_contract(runtime_path: Path = REPO_ROOT / "runtime.toml") -> dict[str, str]:
    """Return the pinned Arena version, revision, and locomotion policy commit."""

    contract = tomllib.loads(runtime_path.read_text(encoding="utf-8"))["tested"]
    return {
        "version": contract["isaaclab_arena"],
        "revision": contract["isaaclab_arena_revision"],
        "wbc_agile_policy_revision": contract["arena_wbc_agile_policy_revision"],
    }


def _read_arena_project_metadata(pyproject_path: Path) -> dict[str, str]:
    """Return the name and version a checkout's ``pyproject.toml`` declares.

    Raises:
        ArenaPinError: The file is unreadable, is not valid TOML, or declares no
            project name and version. ``E2E_ARENA_ROOT`` can name any directory, and
            callers handle a failed pin rather than a parse error.
    """

    try:
        project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]
        return {"name": project["name"], "version": project["version"]}
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ArenaPinError(f"Arena checkout has an unreadable pyproject.toml: {pyproject_path} ({exc})") from exc
    except (KeyError, TypeError) as exc:
        raise ArenaPinError(
            f"Arena checkout declares no project name and version: {pyproject_path} (missing {exc})"
        ) from exc


def _read_wbc_agile_policy_revision(config_path: Path) -> str:
    """Return the WBC-AGILE commit Arena's whole-body controller downloads its policy from.

    Read as source text because this check runs in this repository's interpreter, which
    has no Arena on its path.

    Raises:
        ArenaPinError: The config is unreadable or declares no policy commit.
    """

    try:
        source = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ArenaPinError(
            f"Arena checkout has no readable whole-body controller config: {config_path} ({exc})"
        ) from exc
    match = _WBC_AGILE_REVISION_PATTERN.search(source)
    if match is None:
        raise ArenaPinError(f"Arena declares no WBC-AGILE policy commit: {config_path}")
    return match.group(1)


def verify_arena_pin(arena_root: Path, runtime_path: Path = REPO_ROOT / "runtime.toml") -> dict[str, str]:
    """Check the Arena checkout's identity, version, revision, cleanliness, and policy.

    Raises:
        ArenaPinError: The checkout is missing, is not Arena, is at the wrong
            revision or version, has uncommitted or untracked changes, or downloads a
            locomotion policy other than the pinned one.
    """

    pinned = read_contract(runtime_path)
    pyproject_path = arena_root / "pyproject.toml"
    if not pyproject_path.is_file():
        raise ArenaPinError(f"E2E_ARENA_ROOT does not contain an Isaac Lab Arena checkout: {arena_root}")

    project = _read_arena_project_metadata(pyproject_path)
    if project["name"] != ARENA_PROJECT_NAME:
        raise ArenaPinError(f"E2E_ARENA_ROOT is not Isaac Lab Arena: found project {project['name']!r}")
    if project["version"] != pinned["version"]:
        raise ArenaPinError(f"Arena version mismatch: expected {pinned['version']}, got {project['version']}")

    revision = _git(arena_root, "rev-parse", "HEAD")
    if revision != pinned["revision"]:
        raise ArenaPinError(f"Arena revision mismatch: expected {pinned['revision']}, got {revision}")
    # Untracked files count as dirty: an unpinned module dropped into the checkout
    # changes what Arena imports without changing HEAD.
    if _git(arena_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ArenaPinError(f"Arena checkout must be clean: {arena_root}")

    # A clean checkout at the pinned revision already fixes this commit, so this catches
    # an Arena bump that changes which locomotion policy runs, rather than a dirty tree.
    policy_revision = _read_wbc_agile_policy_revision(arena_root / ARENA_WBC_CONFIG_PATH)
    if policy_revision != pinned["wbc_agile_policy_revision"]:
        raise ArenaPinError(
            f"Arena's WBC-AGILE locomotion policy commit mismatch: expected "
            f"{pinned['wbc_agile_policy_revision']}, got {policy_revision}. This Arena revision runs a "
            "different locomotion policy than the one recorded as tested; re-qualify locomotion behavior "
            "and update arena_wbc_agile_policy_revision in runtime.toml."
        )

    return {
        "root": str(arena_root),
        "version": project["version"],
        "revision": revision,
        "wbc_agile_policy_revision": policy_revision,
    }


def main() -> int:
    """Verify the Arena checkout named on the command line or in the environment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arena-root",
        type=Path,
        default=os.environ.get("E2E_ARENA_ROOT") or None,
        help="Arena source checkout to verify. Defaults to E2E_ARENA_ROOT.",
    )
    parser.add_argument(
        "--print-pinned-revision",
        action="store_true",
        help="Print the pinned revision and exit, for callers that create the checkout.",
    )
    arguments = parser.parse_args()
    if arguments.print_pinned_revision:
        print(read_contract()["revision"])
        return 0
    if arguments.arena_root is None:
        print(
            f"Set E2E_ARENA_ROOT to a pinned Isaac Lab Arena {read_contract()['version']} checkout.",
            file=sys.stderr,
        )
        return EXIT_MISMATCH
    try:
        pinned = verify_arena_pin(arguments.arena_root.expanduser().resolve())
    except ArenaPinError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_MISMATCH
    print(json.dumps({"status": "ok", **pinned}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

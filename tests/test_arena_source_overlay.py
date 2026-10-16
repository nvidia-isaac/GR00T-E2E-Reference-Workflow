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

"""Contracts for consuming Isaac Lab Arena as a source overlay.

Arena's code is not a locked dependency and its dependencies are not a checkout: the
wheel ships no data files, so the G1 embodiment cannot load its joint order from an
installed copy, while everything Arena imports resolves through ``uv.lock``. That split
is the design, and each half drifts in its own way -- an unlocked requirement surfaces
as an ``ImportError`` mid-run, and a dependency from anywhere but the lock is a version
nobody pinned. The behavior this enables is checked by ``sim-arena-recording``,
``sim-arena-locomotion``, and ``sim-common-environment``.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
RUNTIME = tomllib.loads((REPO_ROOT / "runtime.toml").read_text(encoding="utf-8"))["tested"]
LAUNCHER = (REPO_ROOT / "tools/arena_launch.sh").read_text(encoding="utf-8")
FETCH = (REPO_ROOT / "tools/fetch_arena_checkout.sh").read_text(encoding="utf-8")


def test_arenas_dependencies_are_locked_and_leave_the_simulator_pins_alone() -> None:
    """Arena's requirements belong in the lock, where every other version lives.

    Arena declares the simulator in dependency groups, which a consumer never receives,
    so this group disturbs nothing -- that is what makes locking it affordable. What it
    must not do is go missing: the overlay installs nothing, so an unlocked requirement
    is discovered only when a GPU run reaches the import.
    """

    names = {
        requirement.split(">")[0].split("=")[0].split("[")[0] for requirement in PYPROJECT["dependency-groups"]["arena"]
    }
    locked = {
        package["name"]: package.get("version")
        for package in tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]
    }

    assert {"onnxruntime", "pandas", "scipy", "pydantic", "sbi"} <= names
    assert "isaaclab-arena" not in names, "Arena's code comes from the checkout, not from a wheel"
    assert "arena" in PYPROJECT["tool"]["uv"]["default-groups"], "a plain `uv run` must be able to compose Arena"
    assert locked["isaaclab"] == RUNTIME["isaaclab"]
    assert locked["isaacsim"] == RUNTIME["isaacsim"]
    assert locked["torch"] == RUNTIME["torch"]


def test_the_launcher_overlays_the_checkout_instead_of_switching_environments() -> None:
    """One interpreter, holding this repository's lock, with Arena's source on its path.

    Running Arena from its own uv project meant a second multi-gigabyte simulator install
    and two interpreters per validation, one of which could not see this repository's own
    tools, so the launcher must not quietly grow a ``--project`` pointing back at Arena.
    Appending to ``PYTHONPATH`` rather than assigning is what lets ``launch_gr00t_eval.sh``
    add Arena's GR00T client before calling in.
    """

    assert 'export PYTHONPATH="${ARENA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"' in LAUNCHER
    assert '--project "${REPO_ROOT}"' in LAUNCHER
    assert '--project "${ARENA_ROOT}"' not in LAUNCHER
    assert "isaaclab-from-wheel" not in LAUNCHER
    # The pin is still enforced here, and still before anything runs.
    assert LAUNCHER.index("check_arena_pin.py") < LAUNCHER.index("exec ")


def test_the_fetch_script_takes_its_revision_from_the_pin() -> None:
    """One parser for ``runtime.toml``; a second in shell is how the two disagree."""

    printed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools/check_arena_pin.py"), "--print-pinned-revision"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()

    assert printed == RUNTIME["isaaclab_arena_revision"]
    assert "--print-pinned-revision" in FETCH
    assert "--depth 1 --filter=blob:none origin" in FETCH, "fetch by revision, not by branch"
    assert "'!/submodules/'" in FETCH, "the submodules are SSH-only and this path does not need them"


def test_the_fetch_script_puts_nothing_but_the_path_on_stdout() -> None:
    """The launcher reads the checkout path off stdout, so git must not write there.

    A line of advisory git output would be read as part of the path and surface as a
    missing directory, pointing at the pin check instead of at whatever git said.
    """

    body = FETCH[FETCH.index("SPARSE_PATHS=") :]
    printing = [line.strip() for line in body.splitlines() if line.lstrip().startswith(("printf", "echo"))]

    assert "} >&2" in body, "the git commands must be grouped into stderr"
    assert {line for line in printing if ">&2" not in line} == {"printf '%s\\n' \"${DEST}\""}


def test_the_fetch_script_will_not_rewrite_a_checkout_it_does_not_own(tmp_path: Path) -> None:
    """A developer's Arena clone is theirs; the managed one under .arena/ is ours.

    The sparse checkout would narrow a working tree somebody is developing in, and moving
    its HEAD would discard what they had checked out, so a provided path that does not
    already match the pin is refused rather than corrected.
    """

    (tmp_path / "not-arena").mkdir()

    completed = subprocess.run(
        [str(REPO_ROOT / "tools/fetch_arena_checkout.sh"), str(tmp_path / "not-arena")],
        capture_output=True,
        check=False,
        cwd=REPO_ROOT,
        text=True,
    )

    assert completed.returncode == 2
    assert "not at the pinned revision" in completed.stderr
    assert not list((tmp_path / "not-arena").iterdir()), "the directory must be left untouched"

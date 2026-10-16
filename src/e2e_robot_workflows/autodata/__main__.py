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

"""Run an Isaac AutoData script with this repository's environments registered.

AutoData resolves ``--env_name`` through plain ``gymnasium.make`` and offers no hook for
importing extra task packages, so the ids registered here are invisible to it. This runs the
requested script unmodified, as ``__main__``, once our registrations have happened -- which
keeps the pinned AutoData checkout free of local edits.
"""

from __future__ import annotations

import runpy
import sys

USAGE = "usage: python -m e2e_robot_workflows.autodata <autodata-script.py> [args...]"


def main(argv: list[str] | None = None) -> None:
    """Import this repository's task registrations, then run the given script as __main__."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit(USAGE)
    script, *script_args = args

    # Imported for its gym.register side effect, before the script resolves --env_name.
    import e2e_robot_workflows.tasks.static_apple  # noqa: F401

    sys.argv = [script, *script_args]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()

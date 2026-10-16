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

"""YAML config-file helpers (PyYAML imported lazily so imports stay light)."""

from __future__ import annotations

from typing import Any


def load_yaml(path) -> Any:
    """Load a YAML file (utf-8, ``safe_load``). Raises RuntimeError if PyYAML is absent."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YAML config files.") from exc

    with open(path, encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def load_yaml_mapping(path) -> dict[str, Any]:
    """Load a YAML file that must be a top-level mapping (dict)."""

    data = load_yaml(path) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML config: {path}")
    return data

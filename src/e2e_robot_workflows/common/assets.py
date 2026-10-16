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

"""Resolve/download asset files (model checkpoints, configs, etc.).

Dependency-light (stdlib only) so it can be imported anywhere. The cache
directory defaults to ``E2E_ASSET_CACHE_DIR`` or a temp path.
"""

from __future__ import annotations

import os
import urllib.request

DEFAULT_CACHE_DIR = os.environ.get("E2E_ASSET_CACHE_DIR", "/tmp/e2e_robot_workflows/cache")


def download_cached(url: str, filename: str, cache_dir: str | None = None) -> str:
    """Download ``url`` to ``<cache_dir>/<filename>`` once; return the local path.

    Subsequent calls reuse the cached file.
    """

    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, filename)
    if not os.path.isfile(dest):
        urllib.request.urlretrieve(url, dest)  # noqa: S310 - caller supplies a trusted URL
    return dest


def resolve_local_or_download(local: str | None, url: str, cache_filename: str, cache_dir: str | None = None) -> str:
    """Return ``local`` if it is an existing file, else download+cache ``url``."""

    if local and os.path.isfile(local):
        return local
    return download_cached(url, cache_filename, cache_dir)

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

"""Sphinx configuration for the GR00T End-to-End Reference Workflow documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

docs_repository_provider = "github"
docs_project_url = "https://github.com/nvidia-isaac/GR00T-E2E-Reference-Workflow"
docs_clone_url = "https://github.com/nvidia-isaac/GR00T-E2E-Reference-Workflow.git"

project = "GR00T E2E Reference Workflow"
author = "NVIDIA Corporation"
copyright = "2026, NVIDIA Corporation"

extensions = [
    "versioning",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

source_suffix = {".md": "markdown"}
root_doc = "index"
language = "en"
exclude_patterns = ["_build", "_ext", "_templates", "Thumbs.db", ".DS_Store"]
nitpicky = True

myst_enable_extensions = ["colon_fence"]
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 3

# Custom CSS supplies the accessible light and dark Mermaid palette.
mermaid_light_theme = "base"
mermaid_dark_theme = "base"
mermaid_init_config = {
    "startOnLoad": False,
    "flowchart": {
        "nodeSpacing": 32,
        "rankSpacing": 44,
    },
}

html_theme = "nvidia_sphinx_theme"
html_title = "GR00T E2E Reference Workflow"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_copy_source = False
html_theme_options = {
    # Site navigation lives in the primary sidebar; versioning adds its selector here.
    "navbar_center": [],
    "icon_links": [
        {
            "name": "GitHub",
            "url": docs_project_url,
            "icon": "fa-brands fa-github",
        }
    ],
    "navigation_with_keys": True,
    "show_nav_level": 2,
    "show_toc_level": 2,
}

copybutton_prompt_text = r"\$ |>>> |\.\.\. "
copybutton_prompt_is_regexp = True

linkcheck_anchors = True
linkcheck_retries = 2
linkcheck_timeout = 20
# Private workflow assets are verified with authenticated Hub metadata.
# Remove each exact exception when that repository becomes publicly accessible.
linkcheck_ignore = [
    r"https://huggingface\.co/nvidia/agile_wbc_unitree_h2$",
    r"https://huggingface\.co/datasets/nvidia/GR00T-E2E-Reference-Workflow-SIM-G1-N1\.7-AppletoPlate-Dataset$",
    r"https://huggingface\.co/nvidia/GR00T-E2E-Reference-Workflow-SIM-G1-N1\.7-AppletoPlate-Model$",
    r"https://huggingface\.co/datasets/nvidia/GR00T-E2E-Reference-Workflow-SIM-H2-N1\.7-AppleHandoverPlate-Dataset$",
    r"https://huggingface\.co/nvidia/GR00T-E2E-Reference-Workflow-SIM-H2-N1\.7-AppleHandoverPlate-Model$",
]

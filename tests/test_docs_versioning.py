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

"""Exercise rendered documentation identity, copied commands, and invalid contexts."""

import html
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
SHA = "0123456789abcdef0123456789abcdef01234567"


def ci_context(kind, ref, prefix, **extra):
    """Provide explicit CI inputs independent of the ref running the test suite."""
    return {
        "CI": "true",
        "DOCS_KIND": kind,
        "DOCS_REF": ref,
        "PAGES_PREFIX": prefix,
        "CI_COMMIT_SHA": SHA,
        "CI_DEFAULT_BRANCH": "main",
        "CI_COMMIT_BRANCH": ref,
        "CI_COMMIT_REF_NAME": ref,
        "CI_PIPELINE_SOURCE": "push",
        "CI_PAGES_URL": "http://example.org/project/docs/",
        **extra,
    }


def github_context(kind, ref, prefix, **extra):
    """Supply public documentation identity without relying on GitLab variables."""
    return {
        "CI": "true",
        "DOCS_KIND": kind,
        "DOCS_REF": ref,
        "PAGES_PREFIX": prefix,
        "DOCS_SITE_URL": "https://example.org/project/docs/",
        "GITHUB_ACTIONS": "true",
        "GITHUB_SHA": SHA,
        "GITHUB_REF_NAME": ref,
        "GITHUB_REF_TYPE": "branch",
        "GITHUB_EVENT_NAME": "push",
        **extra,
    }


@pytest.mark.parametrize(
    ("provider", "environment", "label", "command", "prefix"),
    [
        ("gitlab", ci_context("development", "main", ""), "main (development)", "git clone --branch main", ""),
        (
            "gitlab",
            ci_context("release", "release/v0.2", "v0-2"),
            "v0.2 (release branch)",
            "git clone --branch release/v0.2",
            "v0-2/",
        ),
        (
            "gitlab",
            ci_context("candidate", "rc3", "rc3", CI_COMMIT_TAG="rc3", CI_COMMIT_BRANCH=""),
            "rc3 (snapshot)",
            "git clone --branch rc3",
            "rc3/",
        ),
        (
            "gitlab",
            ci_context(
                "preview",
                "feature/docs",
                "mr-123",
                CI_PIPELINE_SOURCE="merge_request_event",
                CI_MERGE_REQUEST_IID="123",
                CI_MERGE_REQUEST_SOURCE_BRANCH_NAME="feature/docs",
            ),
            "MR !123 (preview)",
            f"git checkout --detach {SHA}",
            "mr-123/",
        ),
        (
            "gitlab",
            ci_context("unpublished", "feature/docs", "", CI_PIPELINE_SOURCE="web"),
            "feature/docs (unpublished)",
            "git clone --branch feature/docs",
            None,
        ),
        ("gitlab", {}, "Local build", "git clone --branch main", None),
        (
            "github",
            github_context("development", "main", ""),
            "main (latest published)",
            "git clone --branch main",
            "",
        ),
        (
            "github",
            github_context("release", "release/v0.2", "v0-2"),
            "v0.2 (release branch)",
            "git clone --branch release/v0.2",
            "v0-2/",
        ),
        (
            "github",
            ci_context("release", "release/v0.2", "v0-2"),
            "v0.2 (release branch)",
            "git clone --branch release/v0.2",
            "v0-2/",
        ),
        (
            "github",
            github_context(
                "preview",
                "feature/docs",
                "pr-123",
                GITHUB_EVENT_NAME="pull_request",
                GITHUB_REF_NAME="123/merge",
                GITHUB_HEAD_REF="feature/docs",
            ),
            "PR #123 (preview)",
            "git clone --branch main",
            "pr-123/",
        ),
        (
            "github",
            ci_context(
                "preview",
                "feature/docs",
                "mr-123",
                CI_PIPELINE_SOURCE="merge_request_event",
                CI_MERGE_REQUEST_IID="123",
                CI_MERGE_REQUEST_SOURCE_BRANCH_NAME="feature/docs",
            ),
            "MR !123 (preview)",
            "git clone --branch main",
            "mr-123/",
        ),
        (
            "github",
            ci_context("unpublished", "feature/docs", "", CI_PIPELINE_SOURCE="web"),
            "feature/docs (unpublished)",
            "git clone --branch main",
            None,
        ),
        (
            "github",
            ci_context("candidate", "rc3", "rc3", CI_COMMIT_TAG="rc3", CI_COMMIT_BRANCH=""),
            "rc3 (snapshot)",
            "git clone --branch main",
            "rc3/",
        ),
        (
            "github",
            github_context("candidate", "rc5", "rc5", GITHUB_REF_TYPE="tag"),
            "rc5 (snapshot)",
            "git clone --branch rc5",
            "rc5/",
        ),
        ("github", {}, "Local build", "git clone --branch main", None),
    ],
)
def test_rendered_documentation_matches_source(tmp_path, provider, environment, label, command, prefix):
    """Check the built pages and copied commands through the real Sphinx pipeline."""
    pytest.importorskip("sphinx")
    pytest.importorskip("nvidia_sphinx_theme")
    source = tmp_path / "docs"
    shutil.copytree(REPOSITORY / "docs", source, ignore=shutil.ignore_patterns("_build", "__pycache__"))
    with (source / "conf.py").open("a") as config:
        config.write(f'\ndocs_repository_provider = "{provider}"\n')
    expected_sha = SHA if environment else ""
    if provider == "github" and not environment:
        subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Documentation test",
                "-c",
                "user.email=docs@example.org",
                "-c",
                "core.hooksPath=/dev/null",
                "commit",
                "--allow-empty",
                "--quiet",
                "-m",
                "Public export",
            ],
            cwd=tmp_path,
            check=True,
        )
        expected_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    output = tmp_path / "html"
    output.mkdir()
    (output / "versions.json").write_text("stale catalog")
    clean_env = {
        key: value for key, value in os.environ.items() if not key.startswith(("CI", "DOCS_", "PAGES_", "GITHUB_"))
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-a",
            "-E",
            "-n",
            "-W",
            "--keep-going",
            "-b",
            "html",
            str(source),
            str(output),
        ],
        env={**clean_env, **environment},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rendered = (output / "installation.html").read_text()
    text = html.unescape(re.sub(r"<[^>]+>", "", rendered))
    assert 'class="bd-header-announcement"' not in rendered
    assert command in text
    copied_commands = [
        html.unescape(re.sub(r"<[^>]+>", "", block)).strip()
        for block in re.findall(r"<pre\b[^>]*>(.*?)</pre>", rendered, re.S)
    ]
    for setup_command in (
        "mkdir -p ~/robot-workflows",
        "cd ~/robot-workflows",
        "cd GR00T-E2E-Reference-Workflow",
        "git rev-parse HEAD",
    ):
        assert setup_command in copied_commands
    for lfs_command in ("git lfs install --local", "git lfs pull"):
        assert (lfs_command in copied_commands) == (provider == "gitlab")
    selected_command = next(block for block in copied_commands if command in block)
    assert len(selected_command.replace("\\\n", "").splitlines()) == 1
    assert "@@DOCS_" not in text
    info = json.loads((output / "build-info.json").read_text())
    assert info["label"] == label
    assert info["sha"] == expected_sha
    if expected_sha:
        assert info["ref"] == environment.get("DOCS_REF", "main")
    assert "documentation-source-commit" not in text
    assert "source commit is" not in text
    assert "https://github.com/nvidia-isaac/GR00T-E2E-Reference-Workflow.git" in text
    assert 'class="fa-brands fa-github' in rendered
    assert "From the GR00T-E2E-Reference-Workflow directory:" in text
    if provider == "github":
        if expected_sha:
            assert expected_sha not in text
        assert "git fetch origin" not in text
        if environment.get("DOCS_KIND") in {"preview", "unpublished"}:
            assert "These instructions select main." in text
    if prefix is not None:
        assert f"https://example.org/project/docs/{prefix}installation.html" in rendered
        assert info["site_url"] == f"https://example.org/project/docs/{prefix}"
        assert "https://example.org/project/docs/versions.json" in rendered
        for page in ("installation.html", "reference/index.html"):
            page_html = (output / page).read_text()
            assert 'class="bd-header-announcement"' not in page_html
            links = re.findall(r'<a\b[^>]*href="([^"]+)"[^>]*data-version="([^"]+)"', page_html)
            assert set(links) == {
                (f"https://example.org/project/docs/{page}", "main"),
                (f"https://example.org/project/docs/v0-2/{page}", "v0.2"),
            }
            buttons = re.findall(r'<button\b[^>]*class="version-switcher__button[^>]*>(.*?)</button>', page_html, re.S)
            assert buttons and all(label in html.unescape(button) for button in buttons)
    else:
        assert 'rel="canonical"' not in rendered
        assert 'class="version-switcher' not in rendered
    if environment.get("DOCS_KIND") == "development":
        catalog = json.loads((output / "versions.json").read_text())
        assert [entry["version"] for entry in catalog] == ["main", "v0.2"]
        assert catalog[0]["name"] == label
        assert next(entry["url"] for entry in catalog if entry["version"] == "v0.2") == (
            "https://example.org/project/docs/v0-2/"
        )
    else:
        assert not (output / "versions.json").exists()


@pytest.mark.parametrize(
    ("provider", "changes"),
    [
        ("gitlab", {"DOCS_KIND": ""}),
        ("gitlab", {"CI_COMMIT_SHA": "short"}),
        ("gitlab", {"DOCS_REF": "main"}),
        ("gitlab", {"PAGES_PREFIX": ""}),
        ("gitlab", {"PAGES_PREFIX": "../main"}),
        ("gitlab", {"PAGES_PREFIX": "rc2"}),
        ("github", {"GITHUB_SHA": "short"}),
        ("github", {"GITHUB_REF_NAME": "main"}),
        ("github", {"GITHUB_REF_TYPE": "tag"}),
    ],
)
def test_inconsistent_release_context_is_rejected(monkeypatch, provider, changes):
    """Reject metadata that could label or publish release documentation incorrectly."""
    monkeypatch.syspath_prepend(str(REPOSITORY / "docs" / "_ext"))
    versioning = importlib.import_module("versioning")
    catalog = versioning.read_catalog(REPOSITORY / "docs" / "versions.json")
    context = github_context if provider == "github" else ci_context
    with pytest.raises(ValueError):
        versioning.resolve_context(
            {**context("release", "release/v0.2", "v0-2"), **changes}, REPOSITORY, catalog, provider
        )


def test_checkout_commands_are_numbered(monkeypatch):
    """Present checkout commands as an ordered procedure."""
    monkeypatch.syspath_prepend(str(REPOSITORY / "docs" / "_ext"))
    versioning = importlib.import_module("versioning")
    catalog = versioning.read_catalog(REPOSITORY / "docs" / "versions.json")
    context = versioning.resolve_context(ci_context("development", "main", ""), REPOSITORY, catalog)

    block = versioning.checkout_block(context, "ssh://example.org/repo.git")
    numbered_steps = [line for line in block.splitlines() if re.fullmatch(r"[1-9][0-9]*\. .+:", line)]

    assert len(numbered_steps) == 7
    assert numbered_steps[0] == "1. Create a workspace directory:"
    assert numbered_steps[-1] == "7. Print the checked-out source commit:"


def test_future_release_can_build_before_catalog_registration(monkeypatch):
    """Allow a release deployment to precede its addition to the shared catalog."""
    monkeypatch.syspath_prepend(str(REPOSITORY / "docs" / "_ext"))
    versioning = importlib.import_module("versioning")
    catalog = versioning.read_catalog(REPOSITORY / "docs" / "versions.json")
    context = versioning.resolve_context(ci_context("release", "release/v0.3", "v0-3"), REPOSITORY, catalog)
    assert context.site_url == "https://example.org/project/docs/v0-3/"
    assert "git clone --branch release/v0.3" in versioning.checkout_block(context, "ssh://example.org/repo.git")
    catalog.append({"version": "v0.3", "ref": "release/v0.3", "name": "v0.3", "path": "v0-2"})
    with pytest.raises(ValueError):
        versioning.resolve_context(ci_context("release", "release/v0.3", "v0-3"), REPOSITORY, catalog)

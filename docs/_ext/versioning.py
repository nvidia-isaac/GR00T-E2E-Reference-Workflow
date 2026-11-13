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

"""Render documentation identity and checkout instructions from the CI ref."""

import json
import os
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class BuildContext:
    kind: str
    ref: str
    sha: str
    path_prefix: str
    site_url: str
    pipeline_url: str
    label: str
    version: str
    catalog_url: str


def read_catalog(path):
    """Validate published versions before generating selector URLs."""
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("Documentation version catalog must be a nonempty list")
    for key in ("version", "ref", "path"):
        values = [entry[key] for entry in entries]
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate documentation catalog {key}")
    for entry in entries:
        if not all(isinstance(entry[key], str) and entry[key] for key in ("version", "ref", "name")):
            raise ValueError("Documentation catalog entries need version, ref, and name")
        if entry["path"] and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", entry["path"]):
            raise ValueError("Invalid documentation catalog path")
    if [(entry["version"], entry["ref"]) for entry in entries if not entry["path"]] != [("main", "main")]:
        raise ValueError("Only main may own the root documentation path")
    return entries


def resolve_context(env, repository, catalog, provider="gitlab"):
    """Validate CI routing, or identify a local build without a publication URL."""
    if provider not in {"gitlab", "github"}:
        raise ValueError(f"Unknown documentation repository provider: {provider!r}")
    if not env.get("CI"):
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=False
        )
        return BuildContext(
            "local",
            "main",
            result.stdout.strip() if result.returncode == 0 else "",
            "",
            "",
            "",
            "Local build",
            "local",
            "",
        )

    github = env.get("GITHUB_ACTIONS") == "true"
    kind, ref, prefix = (env.get(key, "") for key in ("DOCS_KIND", "DOCS_REF", "PAGES_PREFIX"))
    sha = env.get("GITHUB_SHA" if github else "CI_COMMIT_SHA", "")
    if github:
        source_ref = env.get("GITHUB_REF_NAME", "")
        branch = source_ref if env.get("GITHUB_REF_TYPE") == "branch" else ""
        tag = source_ref if env.get("GITHUB_REF_TYPE") == "tag" else ""
        default_branch = "main"
    else:
        branch, tag, source_ref = (
            env.get(key, "") for key in ("CI_COMMIT_BRANCH", "CI_COMMIT_TAG", "CI_COMMIT_REF_NAME")
        )
        default_branch = env.get("CI_DEFAULT_BRANCH", "main")
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or not ref:
        raise ValueError("CI documentation needs a source ref and full commit SHA")
    if kind == "development":
        valid = ref == default_branch == branch and not prefix
        version = ref
        label = f"{ref} (latest published)" if provider == "github" else f"{ref} (development)"
    elif kind == "release":
        valid = ref == branch and ref.startswith("release/") and bool(prefix)
        version = ref.removeprefix("release/")
        label = f"{version} (release branch)"
    elif kind == "candidate":
        valid = ref == tag == prefix and bool(re.fullmatch(r"rc[1-9][0-9]*", ref))
        version, label = ref, f"{ref} (snapshot)"
    elif kind == "preview":
        if github:
            match = re.fullmatch(r"([1-9][0-9]*)/merge", source_ref)
            iid = match[1] if match else ""
            valid = (
                env.get("GITHUB_EVENT_NAME") == "pull_request"
                and bool(iid)
                and ref == env.get("GITHUB_HEAD_REF")
                and prefix == f"pr-{iid}"
            )
            version, label = prefix, f"PR #{iid} (preview)"
        else:
            iid = env.get("CI_MERGE_REQUEST_IID", "")
            valid = (
                env.get("CI_PIPELINE_SOURCE") == "merge_request_event"
                and iid.isdecimal()
                and ref == env.get("CI_MERGE_REQUEST_SOURCE_BRANCH_NAME")
                and prefix == f"mr-{iid}"
            )
            version, label = prefix, f"MR !{iid} (preview)"
    elif kind == "unpublished":
        valid = ref == source_ref and not prefix
        version, label = "unpublished", f"{ref} (unpublished)"
    else:
        raise ValueError(f"Unknown CI documentation kind: {kind!r}")
    if prefix and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", prefix):
        valid = False
    for entry in catalog:
        if kind not in {"preview", "unpublished"} and (entry["ref"] == ref or entry["path"] == prefix):
            valid = valid and entry["ref"] == ref and entry["path"] == prefix
    if not valid:
        raise ValueError("Documentation context does not match its CI source ref and destination")

    site_url = catalog_url = ""
    if kind != "unpublished":
        base = urlsplit(env.get("DOCS_SITE_URL") or env.get("CI_PAGES_URL", ""))
        if base.scheme not in {"http", "https"} or not base.hostname or base.query or base.fragment or base.username:
            raise ValueError("Published documentation needs a valid Pages base URL")
        # GitLab 17.7 supplies the project root, without the parallel deployment prefix.
        scheme = base.scheme if env.get("DOCS_SITE_URL") else "https"
        root = urlunsplit((scheme, base.netloc, base.path.rstrip("/") + "/", "", ""))
        site_url = root + (prefix + "/" if prefix else "")
        catalog_url = root + "versions.json"
    pipeline_url = env.get("CI_PIPELINE_URL", "")
    if github:
        pipeline_url = ""
        if env.get("GITHUB_REPOSITORY") and env.get("GITHUB_RUN_ID"):
            pipeline_url = (
                f"{env.get('GITHUB_SERVER_URL', 'https://github.com')}/{env['GITHUB_REPOSITORY']}"
                f"/actions/runs/{env['GITHUB_RUN_ID']}"
            )
    return BuildContext(kind, ref, sha, prefix, site_url, pipeline_url, label, version, catalog_url)


def checkout_block(context, clone_url, provider="gitlab"):
    """Generate checkout commands for the selected ref or exact preview revision."""
    directory = shlex.quote(Path(urlsplit(clone_url).path).name.removesuffix(".git"))
    commands = ["mkdir -p ~/robot-workflows", "cd ~/robot-workflows"]
    if context.kind == "preview":
        commands.extend(
            [
                f"git clone --no-checkout {shlex.quote(clone_url)}",
                f"cd {directory}",
                f"git fetch origin {context.sha}",
                f"git checkout --detach {context.sha}",
            ]
        )
    else:
        commands.extend(
            [
                f"git clone --branch {shlex.quote(context.ref)} \\\n  {shlex.quote(clone_url)}",
                f"cd {directory}",
            ]
        )
    if provider == "gitlab":
        commands.extend(["git lfs install --local", "git lfs pull"])
    commands.append("git rev-parse HEAD")
    return "\n\n".join(f"```bash\n{command}\n```" for command in commands)


def configure(app, config):
    """Apply source identity and native version navigation before Sphinx builds."""
    catalog = read_catalog(Path(app.confdir) / "versions.json")
    if config.docs_repository_provider == "github":
        for entry in catalog:
            if entry["version"] == "main":
                entry["name"] = "main (latest published)"
    context = resolve_context(os.environ, Path(app.confdir).parent, catalog, config.docs_repository_provider)
    if os.environ.get("DOCS_PREPUBLICATION") == "1":
        if config.docs_repository_provider != "github" or not context.sha:
            raise ValueError("Prepublication link checking requires a GitHub build with a commit SHA")
        config.linkcheck_ignore = [
            *config.linkcheck_ignore,
            re.escape(f"{config.docs_project_url}/commit/{context.sha}") + "$",
        ]
    app.docs_context = context
    config.version = context.version
    config.release = context.sha
    config.html_baseurl = context.site_url
    options = dict(config.html_theme_options)
    if context.catalog_url:
        root = context.catalog_url.removesuffix("versions.json")
        app.docs_versions = [
            {
                "version": entry["version"],
                "name": entry["name"],
                "url": root + (entry["path"] + "/" if entry["path"] else ""),
            }
            for entry in catalog
        ]
        options["navbar_center"] = [*options.get("navbar_center", []), "components/workflow-version-switcher"]
        config.html_context = {
            **config.html_context,
            "docs_version_label": context.label,
            "docs_version": context.version,
            "docs_catalog_url": context.catalog_url,
            "docs_versions": app.docs_versions,
        }
        app.add_js_file("version-switcher.js", defer="defer")
    config.html_theme_options = options


def render_checkout(app, docname, source):
    """Expand installation commands before MyST parses the fenced code block."""
    if docname != "installation":
        return
    context = app.docs_context
    if app.config.docs_repository_provider == "github" and context.kind in {"preview", "unpublished", "candidate"}:
        context = replace(context, kind="development", ref="main")
    for marker, value in (
        ("@@DOCS_REF@@", f"`{context.ref}`"),
        ("@@DOCS_CHECKOUT@@", checkout_block(context, app.config.docs_clone_url, app.config.docs_repository_provider)),
    ):
        if source[0].count(marker) != 1:
            raise ValueError(f"installation.md must contain exactly one {marker}")
        source[0] = source[0].replace(marker, value)


def write_metadata(app, exception):
    """Write publication identity and let only main supply the shared catalog."""
    if exception is not None or app.statuscode or app.builder.format != "html":
        return
    context = app.docs_context
    output = Path(app.outdir)
    (output / "build-info.json").write_text(json.dumps(asdict(context), indent=2) + "\n")
    if context.kind == "development":
        (output / "versions.json").write_text(json.dumps(app.docs_versions, indent=2) + "\n")
    else:
        (output / "versions.json").unlink(missing_ok=True)


def setup(app):
    """Register the configuration and build hooks for documentation versioning."""
    app.add_config_value("docs_project_url", "", "html")
    app.add_config_value("docs_clone_url", "", "html")
    app.add_config_value("docs_repository_provider", "gitlab", "html")
    app.connect("config-inited", configure)
    app.connect("source-read", render_checkout)
    app.connect("build-finished", write_metadata)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Contributing to GR00T E2E Reference Workflow

Contributions are reviewed in the
[source project](https://github.com/nvidia-isaac/GR00T-E2E-Reference-Workflow).
Submit changes for review against `main`.

## Ways to Contribute

Contributions may include:

- Bug reports
- Feature requests
- Code changes
- Documentation improvements

## Style Guide

Follow the [Isaac Lab coding style](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html#coding-style)
and the existing conventions in the files you change. This repository's formatter
and linter configuration defines the checks applied to contributions.

Write documentation in MyST Markdown. Follow the physical G1 pages and the
[documentation style guidance](docs/reference/development.md#documentation-style)
for section structure, commands, notes, and warnings.

Documentation should explain what the project does, why its design matters, and
how readers can use it. Write for readers who are new to the project, use active
voice and plain language, and prefer concrete examples over ambiguous wording.
Update the affected documentation whenever behavior changes.

## Issue Tracking

Use issues to track scoped, executable work with a clear deliverable. Use the
project's designated discussion channel for broader proposals, questions, and
ideas that are not ready to become defined work.

Start enhancement, bugfix, or change requests in the
[project issue tracker](https://github.com/nvidia-isaac/GR00T-E2E-Reference-Workflow/issues).
Have the request reviewed by the maintainers before code review, and link the
issue in the review description.


## Coding Guidelines

- Follow the conventions in the relevant module when adding or changing code.

- Avoid introducing unnecessary complexity into existing code so that maintainability and readability are preserved.

- Branch from current `main` and target `main` when submitting changes for review.

- Keep review requests small and focused on one concern:
  - Avoid committing commented-out code.
  - Submit unrelated changes separately and identify dependencies in their descriptions.
  - Keep implementation, tests, and documentation synchronized within each change.

- Write commit titles using imperative mood and [these rules](https://chris.beams.io/posts/git-commit/), and reference the Issue number that is being addressed. Following is the recommended format for commit texts:
  ```text
  <Commit Title>

  <Commit Body>
  ```

- Ensure that the Sphinx build log is clean, meaning no warnings or errors should be present.

- Ensure that all code blocks execute correctly prior to submitting your code.

- Make sure that you can contribute your work to open source (no license and/or patent conflict is introduced by your code). You will need to [`sign-off`](#signing-your-work) your commit.

## Submitting a Change

1. Clone the source project using [Installation](docs/installation.md), or fork
   it and clone the fork if you do not have branch access.
   In a fork, add the source project as `upstream` and fetch its `main` branch.

2. Create a branch from current `main`:

   ```bash
   git fetch origin main
   ```

   ```bash
   git switch -c <branch-name> origin/main
   ```

   For a fork, use `upstream/main` as the starting point.

3. Make the change and run the repository's validation commands:

   ```bash
   uv run --no-sync ./tools/validate_repository.sh host
   ```

   ```bash
   uv run --no-sync ./tools/validate_repository.sh docs
   ```


4. Commit with a sign-off and push the branch:

   ```bash
   git add <changed-files>
   ```

   ```bash
   git commit -s -m "<Commit title>"
   ```

   ```bash
   git push -u origin <branch-name>
   ```

5. Submit the branch for review in the source project targeting `main`. Describe the
   resulting behavior, link the issue, and report the validation performed.
   Use **Draft** status while work is incomplete. Address review
   feedback and validation failures before requesting merge.


## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.
  * Note that "sign-off" is not the same as a "signed commit". "Signed-off-by" is a tag that is included in a commit message, and can be done through `-s` or inserting the text at the bottom of a commit message manually (see below). The flag `-S` is for a "signed commit", which is cryptographically identifiable. For more information on the differences, see [About commit signature verification](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification) and [About commit signoffs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/managing-the-commit-signoff-policy-for-your-repository#about-commit-signoffs)

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```text
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```text
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.

    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```

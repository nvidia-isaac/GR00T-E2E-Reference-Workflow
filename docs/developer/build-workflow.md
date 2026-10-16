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

# Build a Workflow Path

A workflow path connects one environment to the runtime adapters, data
formats, policy execution, validation, and documentation needed to reproduce an
end-to-end result.

The repository does not provide a workflow path generator. Use the current
Static Apple workflows as working examples and define the requirements for each stage.

## Define the Workflow Components

1. Add or reuse the robot-independent environment definition under
   `src/e2e_robot_workflows/environments/`.
2. Add the required runtime adapters: direct Isaac Lab task code belongs under
   `src/e2e_robot_workflows/tasks/<task>/`, while Isaac Lab-Arena composition belongs
   under `src/e2e_robot_workflows/arena/`.
3. Add small discovery and robot pointers under `configs/tasks/` and
   `configs/robots/`.
4. Add data or policy configuration under `configs/stages/` and the matching
   package module when code is required.
5. Add host contract tests and the smallest GPU simulation check that proves
   the runtime behavior.

Keep environment semantics in the shared environment definition. Runtime
adapters should realize that definition instead of owning a second copy of the
scene definition.

## Add the User Path

Create one workflow path overview under `docs/workflows/`. Link its stages in the
order that a user performs them. Keep task instructions, architectural
explanation, and exact reference data on separate pages, but make them
discoverable from the workflow path overview.

1. Add a card for the overview to [Workflow Paths](../workflows/index.md) and
   register the overview in that page's hidden `toctree`.
2. Register new stage pages in the overview's hidden `toctree`, in reading
   order. Existing shared pages already registered elsewhere need only a link.
3. Follow the [documentation style](../reference/development.md#documentation-style)
   and describe only capabilities and test results supported by evidence.

Cards provide entry links; `toctree` entries provide sidebar and previous/next
navigation. Check both in the rendered site before submitting the change.

## Validate the Contribution

Run host validation while developing:

```bash
uv run --no-sync ./tools/validate_repository.sh host
```

Build and check the documentation separately:

```bash
uv run --no-sync ./tools/validate_repository.sh docs
```

Run the relevant simulator validation on a compatible NVIDIA GPU host. The
available entry points and their scope are listed in the
[Developer Guide](../reference/development.md).

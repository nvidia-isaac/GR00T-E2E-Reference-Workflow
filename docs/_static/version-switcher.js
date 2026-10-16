/*
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
*/

document.addEventListener("DOMContentLoaded", async () => {
  const switchers = [...document.querySelectorAll(".workflow-version-switcher")];
  if (!switchers.length) return;

  switchers.forEach((switcher) => {
    switcher.addEventListener("click", async (event) => {
      const link = event.target.closest("a[data-home]");
      if (!link || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      let destination = link.href;
      try {
        const response = await fetch(destination, {method: "HEAD", signal: AbortSignal.timeout(3000)});
        if (response.status === 404) destination = link.dataset.home;
      } catch {
        // Top-level navigation can complete authentication that fetch cannot.
      }
      window.location.assign(destination);
    });
  });

  try {
    const response = await fetch(switchers[0].dataset.catalogUrl);
    if (!response.ok) return;
    const versions = await response.json();
    if (!Array.isArray(versions) || !versions.length || versions.some((entry) =>
      !entry || typeof entry.version !== "string" || typeof entry.name !== "string" ||
      typeof entry.url !== "string" || !/^https?:\/\//.test(entry.url)
    )) return;
    const published = versions.filter((entry) => entry.version === "main" || /^v\d+\.\d+(?:\.\d+)?$/.test(entry.version));
    if (!published.length) return;
    switchers.forEach((switcher) => {
      const links = published.map((entry) => {
        const link = document.createElement("a");
        link.className = "dropdown-item list-group-item list-group-item-action py-1";
        link.classList.toggle("active", entry.version === switcher.dataset.version);
        link.href = entry.url + switcher.dataset.page;
        link.dataset.version = entry.version;
        link.dataset.home = entry.url;
        link.textContent = entry.name;
        return link;
      });
      switcher.querySelector(".version-switcher__menu").replaceChildren(...links);
    });
  } catch {
    // Keep the links rendered at build time when the shared catalog is unavailable.
  }
});

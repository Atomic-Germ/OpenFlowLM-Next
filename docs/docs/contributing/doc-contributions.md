---
title: "Documentation Contributions"
description: "Author, preview, and verify Hugo Lotus Docs pages and benchmark tables for openflowlm.org."
icon: "menu_book"
weight: 60
---

OpenFlowLM's website (`openflowlm.org`) builds from `docs/` using Hugo Extended (`v0.140.0+`) and the Lotus Docs module.

## Directory Layout

```text
docs/
├── hugo.toml                        # Mounts, menus, and Lotus Docs config
├── serve-docs.sh                    # Live-reload server & static build wrapper
├── assets/                          # SCSS (oflm-lotus.scss), JS charts, images/
├── content/                         # Top-level pages (/, /models/, /engines/, /benchmarks/, /team/)
├── layouts/                         # Custom layouts, badge/quants/wip shortcodes, alert hooks
└── docs/                            # Docs mounted at /docs/
    ├── _index.md                    # Overview (/docs/)
    ├── install/                     # Getting Started (/docs/install/)
    │   ├── requirements.md          # Hardware & Quickstart (/docs/install/requirements/)
    │   ├── windows.md               # Windows (/docs/install/windows/)
    │   └── linux/                   # Linux (/docs/install/linux/{packages,docker,nix}/)
    ├── instructions/                # User Guide (/docs/instructions/)
    │   ├── cli.md                   # CLI Usage (/docs/instructions/cli/)
    │   ├── config.md                # Configuration (/docs/instructions/config/)
    │   └── server/                  # Server Mode (/docs/instructions/server/{basics,openapi,images,tool_calling,webui}/)
    ├── models/                      # Model family reference cards (/docs/models/*)
    └── contributing/                # Contributor & developer guides (/docs/contributing/*)
```

---

## Authoring Rules

1. **Audience boundary & OS wording:** Everything outside `/docs/contributing/` targets OpenFlowLM users, not NPU kernel developers. Omit compiler or internal hardware jargon (`MLIR-AIE`, `IRON`, `AIE2P`, `memtiles`, `WideDeltaNet`, `BFP16ebs8`, `Full-ELF`, or `XCLBIN`) in user-facing prose. Summarize supported operating systems as **Windows & Linux** (keeping exact OS versions such as Windows 11 for installer and driver requirements), with Packages, Docker, and Nix & NixOS nested under Linux (`/docs/install/linux/`).
2. **YAML front matter:** Include `title`, `description` (under 125 chars, no backticks, em-dashes, or emojis), `icon` (Material Icon name), and integer `weight`:

    ```yaml
    ---
    title: "Packages"
    description: "Install DEB, RPM, or portable tar.gz packages and configure AMD XDNA drivers on Linux."
    icon: "inventory_2"
    weight: 10
    ---
    ```

3. **Headings, TOC, code blocks & numbered lists:** Start body headings at `##` (Lotus Docs renders `<h1>` from `title`), avoid heading emojis, use `## Limits` for constraint sections (without appending `& WIP`), and **never place Markdown links inside `##` or `###` headings** (Hugo's `.TableOfContents` emits broken unprefixed links on subpath builds). Surround language-tagged code fences (`bash`, `powershell`, `nix`, `json`, `yaml`, `python`, `javascript`, `text`) with blank lines. Inside numbered lists (`1.`, `2.`, `3.`), indent child code fences, GitHub alerts, tables, sub-lists, and continuation paragraphs by **4 spaces** so Goldmark emits one contiguous `<ol>` without restarting numbering at `1`.
4. **Badges (`{{</* b */>}}` & `{{</* quants */>}}`):** Reserve badges for model pages and structured comparison tables; avoid badge clutter on install, CLI, and server prose guides. Separate badge categories into titled columns or precede each group with `<span class="oflm-group-label">Category:</span>`, and enforce canonical badge order:
    - **Capabilities:** `Chat` (`llm`) -> `Vision` (`vlm`) -> `Embeddings` (`embed`) -> `Speech` (`asr`) -> `Image` (`image`) -> `Reasoning` (`think`) -> `Tools` (`tools`).
    - **Runtime & Format:** `NPU` (`open`) -> `NPU + CPU` (`hybrid`) -> `Legacy NPU` (`legacy`), followed by `GGUF` (`gguf`) -> `Safetensors` (`safetensors`) -> `Pre-packed` (`bundle`) -> `Legacy Q4NX` (`q4nx`).
    - **Hybrid tooltips, unsupported variants & quantizations:** Pass `tip="..."` on every `{{</* b k="hybrid" tip="..." */>}}` (`NPU + CPU`) badge to explain the NPU vs. CPU split in the `ⓘ` tooltip instead of repeating breakdown prose next to the badge. Render unsupported model sizes or variants with `{{</* b k="unsupported" t="..." mono="true" */>}}`. List supported quantizations explicitly using `qbest` (`★` preferred) and `quant` (never `qdirect` or `Q*..Q*` ranges).
5. **Model pages (`docs/docs/models/*.md`) & scannable layout:** Restrict `.oflm-spec-grid` infoboxes **exclusively to model pages (`docs/docs/models/*.md`)**, keeping each grid to 4 `.oflm-spec-item` boxes without redundant summary prose directly underneath. In every model page's `## Variants & Built-in Shortcuts` table, place `Custom GGUF` (`oflm add`) rows **first**, before any built-in `oflm pull` shortcut rows. Keep paragraphs to 1-2 short sentences using Markdown tables, `.oflm-two-col`, GitHub alerts (`> [!NOTE]`, `> [!TIP]`, `> [!WARNING]`, `> [!CAUTION]`), and `{{</* wip issue="..." title="..." */>}}` callouts. Use root-relative URLs with trailing slashes (`/docs/install/linux/packages/`, `/docs/instructions/cli/`, `/models/`, `/benchmarks/`), and link repository files outside `docs/` via `https://github.com/Atomic-Germ/OpenFlowLM-Next/blob/main/...`.
6. **Unified benchmarks & anchors:** All benchmark results live on `/benchmarks/` (`docs/content/benchmarks/_index.md`; do not create a separate benchmarks subfolder under `docs/docs/`). Preserve `## Before the First Build` in `docs/docs/install/linux/packages.md` (linked from `README.md`), document `oflm add` GGUF installation rather than `q4nx-build`, and keep `bench-embed` numbers identical to captured CLI output.

---

## Local Preview & Verification

```bash
# Lint front matter, links, headings, and user-page prose quality
python3 utilities/check_docs_quality.py

# Verify bench-embed numbers and settings registry
python3 utilities/check_embed_bench_docs.py
utilities/check-settings-ci.sh

# Live preview at http://127.0.0.1:1313 (auto-downloads Hugo Extended v0.147.9 if needed)
./docs/serve-docs.sh

# Production static build into docs/public/
./docs/serve-docs.sh --build
```

Stage only your edited files (`git add docs/docs/...`, never `git add -A`) before opening a PR.

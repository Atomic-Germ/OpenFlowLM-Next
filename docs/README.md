# OpenFlowLM Website and Documentation Guide

This directory (`docs/`) contains the Hugo source for **https://openflowlm.org**, built on [Lotus Docs](https://lotusdocs.dev/) and [Apache ECharts](https://echarts.apache.org/) inside [`Atomic-Germ/OpenFlowLM-Next`](https://github.com/Atomic-Germ/OpenFlowLM-Next).

> [!NOTE]
> Looking to compile the `oflm` C++ runtime and open XDNA2 kernels from source? Read [`docs/contributing/build.md`](docs/contributing/build.md) (published at `/docs/contributing/build/`).

## Directory Layout

Hugo mounts both `docs/content/` (landing pages, catalog, leaderboard, and submission UI) and `docs/docs/` (versioned reference documentation at `/docs/*`) via `[module.mounts]` in [`hugo.toml`](hugo.toml):

```text
docs/
├── hugo.toml                   # Site config, Lotus Docs params, top menu, and module mounts
├── go.mod / go.sum             # Hugo Go modules (Lotus Docs + Bootstrap 5 SCSS)
├── serve-docs.sh               # Zero-setup Linux/macOS preview & build helper (Hugo Extended v0.147.9)
├── DEPLOY-CLOUDFLARE.md        # Cloudflare Pages + Community Benchmarks Worker deployment guide
├── CNAME                       # Custom domain marker
├── content/                    # Top-level pages mounted at /
│   ├── _index.md               # Home page (/)
│   ├── models.md               # Interactive GGUF model catalog (/models/)
│   ├── engines.md              # NPU runtimes matrix (/engines/)
│   ├── benchmarks/             # Benchmarks overview, community leaderboard, detail, and submit pages
│   └── team.md                 # Project story and contributors (/team/)
├── docs/                       # Reference documentation mounted at /docs/
│   ├── _index.md               # Documentation overview (/docs/)
│   ├── install/                # Windows 11 MSI and Linux (linux/packages.md, linux/docker.md, linux/nix.md)
│   ├── instructions/           # CLI, config, and server/ guides (server/webui.md, server/images.md, server/tool_calling.md, server/openapi.md)
│   ├── models/                 # Model catalog overview and 15 per-family reference cards
│   └── contributing/           # Build system, C++, open AIE kernels, testing, tools, and docs guides
├── data/                       # Structured YAML/JSON data for shortcodes and ECharts
│   └── benchmarks/             # Pre-verified benchmark datasets rendered by chart shortcodes
├── layouts/                    # Custom landing templates, Lotus Docs overrides, and shortcodes
│   └── shortcodes/             # bench-chart, embed-chart, models-explorer, engines-matrix, b, quants, wip
├── assets/                     # SCSS overrides (scss/oflm-lotus.scss), brand logos, and SVG illustrations
├── static/                     # Static JS (echarts.min.js, oflm-charts.js, community-bench.js) and CNAME
└── public/                     # Generated static site output (git-ignored; created by hugo --minify)
```

## Local Preview and Static Build

Run [`serve-docs.sh`](serve-docs.sh) from the repository root on Linux or macOS:

```bash
./docs/serve-docs.sh
```

`serve-docs.sh` checks `PATH` for Hugo Extended (`>= 0.140.0`). When Hugo Extended is missing or non-extended, the script downloads Hugo Extended `v0.147.9` into `${XDG_CACHE_HOME:-$HOME/.cache}/oflm-hugo/v0.147.9/hugo` without root access, then starts the live-reload server at `http://127.0.0.1:1313` (override the port with `PORT=8080 ./docs/serve-docs.sh`).

To build the production static bundle into `docs/public/`:

```bash
./docs/serve-docs.sh --build
```

## Authoring Documentation Pages

Every Markdown file under `docs/docs/` and `docs/content/` starts with YAML front matter:

```yaml
---
title: "IBM Granite"
description: "Run IBM Granite 4.2 3B on open XDNA2 kernels via GGUF-direct installation."
icon: "psychology"
weight: 30
katex: false
---
```

Follow these authoring conventions across all pages:

1. **Front matter:** Provide `title`, `description` (one sentence under 125 characters), `icon` (Material Symbols name), and `weight` for sidebar ordering. Never place backticks, emojis, or em-dashes inside `title` or `description`. Set `katex: true` on pages that render `$...$` or `$$...$$` math.
2. **Heading hierarchy:** Lotus Docs renders `<h1 class="content-title">{{ .Title }}</h1>` from front matter automatically. Start body sections at `##` and omit decorative emojis and links from headings.
3. **Code fences:** Surround every fenced code block with blank lines and attach an explicit language identifier (`bash`, `powershell`, `python`, `json`, `yaml`, `toml`, or `text`).
4. **Callouts and images:** Use GitHub-style alerts (`> [!NOTE]`, `> [!TIP]`, `> [!IMPORTANT]`, `> [!WARNING]`) for callout boxes. Reference bundled assets under `docs/assets/` with relative paths such as `images/hero-terminal.svg`.
5. **Charts and data:** Interactive throughput and latency charts load Apache ECharts (`static/js/echarts.min.js` and `static/js/oflm-charts.js`) via the `bench-chart` and `embed-chart` shortcodes backed by `docs/data/`.

## Documentation Quality Checks

Run both verification scripts from the repository root before opening a pull request:

```bash
python3 utilities/check_embed_bench_docs.py
python3 utilities/check_docs_quality.py
```

`check_embed_bench_docs.py` verifies published embedding benchmark tables against captured `oflm bench-embed` output. `check_docs_quality.py` checks YAML front matter, code-fence tags and spacing, heading cleanliness, and prose style across `docs/content/` and `docs/docs/`.

## Deployment

- **GitHub Pages preview and custom domain CI:** `.github/workflows/docs-pages.yml` runs both quality linters, builds `docs/public/` with Hugo Extended `0.147.9`, strips `public/CNAME` automatically on `*.github.io` fork previews, and deploys to GitHub Pages.
- **Cloudflare Pages and Community Benchmarks API:** Read [`DEPLOY-CLOUDFLARE.md`](DEPLOY-CLOUDFLARE.md) for Cloudflare Pages (`hugo --minify` with root directory `docs`) and `benchmarks-worker/` (`/api/*` + D1) setup.

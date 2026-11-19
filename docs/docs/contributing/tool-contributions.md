---
title: "Tool Contributions"
description: "Maintain and test Python utilities for GGUF installation, API testing, and registry generation."
icon: "handyman"
weight: 50
---

Python toolchain utilities live under `utilities/`. Place reusable `.xclbin` or library inspection scripts inside `utilities/` rather than temporary directories.

## Utilities Map

| Tool | Location | Status & Role |
|---|---|---|
| `oflm-add` | `utilities/oflm-add/` | **Active** (`oflm add`): stdlib-only GGUF/HF/ModelScope installer, `spec_hash` kernel linker, and `settings.json` writer |
| `oflm-test` | `utilities/oflm-test/` | **Active**: live server conformance suite (`--llm`, `--vision`, `--audio`, `--embedding`, `--tools`, `--api`) |
| `env.sh` / `env.fish` | `utilities/` | **Active**: discovers XRT, ROCm/HIP, `mlir-aie`, Peano (`PEANO_INSTALL_DIR`), and `OFLM_*` paths (`env.sh --check`) |
| Kernel scripts | `utilities/` | **Active**: `export-kernels.py`, `gen-kernel-spec.py`, `build-new-spec.py`, `catalogue-add-point.py`, `register-spec.py` |
| Settings & CI | `utilities/` | **Active**: `generate-registry.py`, `scan-settings.py`, `check-settings-ci.sh`, `check_embed_bench_docs.py` |
| `bench-configs` / `binary-inventory` | `utilities/` | **Active**: sweep configs/logs and `precompiled_artifacts.json` auditor |
| `q4nx-build` | `utilities/q4nx-build/` | **Deprecated**: legacy `.q4nx` maintenance only (never add new builders or formats) |

---

## Working on `oflm-add` and `oflm-test`

<div class="oflm-two-col">
  <div class="oflm-card">

### `oflm-add` ([README](https://github.com/Atomic-Germ/OpenFlowLM-Next/blob/main/utilities/oflm-add/README.md))

Keep `oflm-add` strictly on the Python 3 standard library. It verifies GGUF quant types (`Q4_1`, `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `IQ4`), walks `base_model` metadata for templates, links `<model>/open_kernels`, and writes `settings.json` with zero manual steps:

```bash
source utilities/env.sh
python3 utilities/oflm-add/oflm-add.py unsloth/Qwen3.5-4B-GGUF --dry-run
uv tool install --editable utilities/oflm-add
```

  </div>
  <div class="oflm-card">

### `oflm-test` ([README](https://github.com/Atomic-Germ/OpenFlowLM-Next/blob/main/utilities/oflm-test/README.md))

Runs against a live `oflm serve` instance and exits non-zero on any `FAIL` or `ERROR`:

```bash
uv tool install --editable utilities/oflm-test
oflm-test --all
oflm-test --vision --model gemma3:4b qwen3vl-it:4b
oflm-test --embedding   # requires oflm serve -e 1
```

  </div>
</div>

## Submission Rules

1. Run `utilities/check-settings-ci.sh` whenever modifying CLI flags or `OFLM_*` environment variables.
2. Verify `oflm add`, `oflm list`, and `oflm serve` end to end without manual file edits.
3. Stage explicit paths only (`git add utilities/<path>`, never `git add -A` or `git add .`).

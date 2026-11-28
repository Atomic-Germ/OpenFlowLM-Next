---
title: "Code Contributions"
description: "Code style, settings registry checks, and workflow for OpenFlowLM engine and CLI pull requests."
icon: "code"
weight: 20
---

Contribute C++ and Rust changes across OpenFlowLM's inference engines and `oflm` CLI. Open a [GitHub Issue](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues) first before starting large refactors.

## Style & Performance Rules

- **Conventions:** Match surrounding code: `snake_case` functions/variables, `PascalCase` types, `UPPER_SNAKE_CASE` constants, and `_member` private fields where already used.
- **Explicit errors:** Propagate XRT, file, and dispatch errors with actionable context; never swallow failures or fall back silently.
- **Measured speedups:** Record `oflm bench` or `oflm bench-embed` throughput across short and long contexts (up to 256k tokens where supported) before and after performance changes.

## Engine Subsystems

| Path | Subsystem | Role |
|---|---|---|
| `src/open_qwen36/` | Open LLM, MoE & VLM Engine | Executes `manifest.json` layer programs and streams GGUF pools (`Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `IQ4`, `BF16`) |
| `src/open_npue/` | Open NPUE | Runs batched BERT/BGE/Nomic/MiniLM/GTE encoders on `BERT-h*` kernel sets |
| `src/open_embedding/` | Open EmbeddingGemma | Runs `embed-gemma:300m` GGUF linear projections on `npu_matmul_f32` + CPU attention |
| `src/open_diffusion/` | Open Diffusion | Runs FLUX.2 [klein] 4B generation and editing in one context via full-ELF reconfiguration |
| `src/open_whisper/` | Open Whisper | Open encoder GEMM + CPU decoder runtime (not yet wired into default `export-kernels.py`, [#240](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/240)) |
| `src/src/`, `src/include/utils/vm_args.hpp` | `oflm` CLI & Server | Implements subcommands and HTTP routes (no `--log-level` flag; use `/verbose` in `oflm run`) |

---

## Development & CI Checklist

1. **Branch & stage explicitly:**

    > [!CAUTION]
    > Stage individual files (`git add src/...`), never `git add -A`, `git add .`, or `git add --all`. Check `git diff --cached --stat` before committing so local weights in `src/models/` or `build/` are never staged.

2. **Build & run tests:**

    ```bash
    cmake --build --preset linux-default
    ctest --preset linux-default --output-on-failure
    oflm validate && oflm run llama3.2:1b
    ```

3. **Verify the generated settings registry:**

    `src/config/settings.json` generates `src/config/settings_registry_generated.hpp` via `utilities/generate-registry.py`. Every CLI flag and `OFLM_*` env var scanned in C++ must appear in `settings.json` or `NON_SETTING_FLAGS` / `NON_SETTING_ENV_VARS`.

    Because CI checks your branch **merged with `origin/main`**, run the merge-aware check script before pushing (see [`src/create_new_model.md`](https://github.com/Atomic-Germ/OpenFlowLM-Next/blob/main/src/create_new_model.md)):

    ```bash
    utilities/check-settings-ci.sh origin/main
    ```

4. **Open your PR:** Update `src/model_list.json` when adding or changing official models, attach test and benchmark results, and submit on [GitHub](https://github.com/Atomic-Germ/OpenFlowLM-Next/pulls).

---
title: "CLI Usage"
description: "Interactive chat, GGUF model installation, NPU validation, and throughput benchmarking from the CLI."
icon: "terminal"
weight: 10
---

- [System Commands](#system-commands)
- [Commands Inside CLI Mode](#commands-inside-cli-mode)
- [Benchmarking Tool](#benchmarking-tool)
- [Configuration and Model Registration](#configuration-and-model-registration)

---

## System Commands

| Command | Purpose |
|---|---|
| `oflm validate [--json]` | Verify NPU driver, firmware, `memlock`, and XRT runtime |
| `oflm add <repo-or-file>` | Download a GGUF repository or file and register it in `settings.json` |
| `oflm run <tag>` | Start interactive terminal chat (downloads tag if missing) |
| `oflm serve [tag]` | Start the OpenAI- and Ollama-compatible HTTP server ([Server Mode](/docs/instructions/server/)) |
| `oflm image <tag> "<prompt>"` | Generate or edit an image on the NPU |
| `oflm pull <tag> [--force]` | Download a built-in shortcut tag (`--force` overwrites corrupted files) |
| `oflm list [--filter all\|installed\|not-installed] [--json] [--quiet]` | List available and installed models |
| `oflm check <tag>` | Verify SHA-256 hashes of downloaded files |
| `oflm remove <tag>` | Delete a downloaded model from local storage |
| `oflm config [show\|set\|unset\|browse\|edit\|review]` | Manage persistent settings in `settings.json` |
| `oflm bench <tag>` / `oflm bench-embed <tag>` | Measure LLM or embedding NPU throughput |
| `oflm port` | Print the resolved default server port |
| `oflm version [--json]` | Print the installed release version |
| `oflm help` | Display CLI usage summary |

{{< wip issue="240" title="Built-in Registry Migration" >}}`oflm add <hf-repo-or-file>` already loads compatible GGUF models directly. Built-in `oflm pull <tag>` shortcuts in `model_list.json` are migrating from legacy `.q4nx` archives to upstream GGUFs under #240, and speculative decoding (`--speculative` / `OFLM_SPECULATIVE`) is in progress under #189 and #240.{{< /wip >}}

> [!NOTE]
> On Linux, `oflm validate` checks your `/dev/accel/accel*` NPU device and XRT runtime. If `--json` reports `"runtime_ok": false`, run `xrt-smi examine` and verify `xrt-plugin-amdxdna` is installed (see [Packages](/docs/install/linux/packages/)).

### Storage Paths & ModelScope

Models default to `C:\Users\<USER>\.oflm\models` on Windows and `~/.config/oflm/models` on Linux. Set `OFLM_MODEL_PATH` to a custom parent folder (weights resolve in `$OFLM_MODEL_PATH/models`) and verify with `oflm config show`:

```bash
export OFLM_MODEL_PATH="/your/custom/oflm-root"
```

Pass `--modelscope` to `pull`, `add`, `check`, `run`, or `bench-embed` to fetch weights from ModelScope instead of Hugging Face:

```bash
oflm pull llama3.2:3b --modelscope
```

### Runtime & Server Launch Flags

| Flag | Applies To | Default | Effect |
|---|---|---|---|
| `--pmode <mode>` | `run`, `serve` | `performance` | NPU power mode: `default`, `powersaver`, `balanced`, `performance`, `turbo` |
| `--ctx-len`, `-c <N>` | `run`, `serve`, `bench` | model default | Context window (min 512; exact token count without power-of-two rounding) |
| `--prefill-chunk-len <N>` | `run`, `serve` | `4096` | Tokens per prompt chunk (min 512) |
| `--prefill-mode <mode>` | `run`, `serve` | `fast` | Prompt memory/speed preset: `fast`, `lean` (~1.1 GiB lower RAM on Qwen3.6 35B), or `bfp16` (Granite 4.2 3B) |
| `--prefill-precision <p>` | `run`, `serve` | `paired` | Prompt math precision preset: `paired` (default), `three_term`, `weight_high`, or `single` (fastest) |
| `--speculative` | `run`, `serve` | `false` | Speculative draft speedup where supported ({{< b "wip" "#189" >}} {{< b "wip" "#240" >}}) |
| `--img-pre-resize`, `-r <0-8>` | `run`, `serve` | `0` | Resize vision input height (`0` original, `1` 480p, `2` 720p, `3` 1080p, up to `8` 4320p) |
| `--asr`, `-a <0\|1>` | `run`, `serve` | `0` | Co-load Whisper speech-to-text (`--asrmodel`, default `whisper-v3:turbo`) |
| `--embed`, `-e <0\|1>` | `serve` | `0` | Co-load embedding encoder (`--embeddingmodel`, default `embed-gemma:300m`) |
| `--imagegen <0\|1>` | `serve` | `0` | Keep image generation model loaded (`--imagemodel`, default `flux2-klein:4b`) |
| `--port`, `-p <N>` | `serve`, `port` | `52625` | HTTP port for current process (`OFLM_SERVE_PORT` sets persistent default) |
| `--host <addr>` | `serve` | `127.0.0.1` | Bind address (`serve` only) |
| `--cors <0\|1>` | `serve` | `1` | Toggle CORS headers (set `--cors 0` or add an auth proxy when binding off `127.0.0.1`) |
| `--preemption <0\|1>` | `run`, `serve` | `0` | Experimental NPU job preemption (requires specialized driver support) |

```bash
# Examples
oflm run gemma3:4b --pmode balanced --ctx-len 8192
oflm serve qwen3.6-moe:35b-a3b --prefill-mode lean --port 8000
oflm serve llama3.2:1b --asr 1 --embed 1 --embeddingmodel bge-base:en-v1.5
```

### Generate or Edit an Image (`oflm image`)

`oflm image` runs FLUX.2 [klein] 4B end-to-end on the NPU (~3.7 s at 512x512 and ~12.0 s at 1024x1024 warm in `turbo` mode). See [Image Generation and Editing](/docs/instructions/server/images/) for CLI and `/v1/images/*` details.

```bash
oflm image flux2-klein:4b "a red fox in fresh snow" -o fox.png
oflm image flux2-klein:4b "Make it a watercolor painting" --image coffee.jpg --seed 7 -o watercolor.png
```

| Option | Default | Description |
|---|---|---|
| `-o`, `--out` | `oflm-<seed>.png` | Output `.png`, `.jpg`, or `.jpeg` |
| `--size` | `1024` (auto on edits) | `512` or `1024` square pixels |
| `--seed` | random | Non-negative 64-bit integer for deterministic output |
| `--steps` | `0` (uses `4`) | Denoising steps (1 to 50) |
| `--image` | *(none)* | Reference `.png` or `.jpg` to edit (centre-cropped, EXIF applied) |

---

## Commands Inside CLI Mode

Inside `oflm run`, slash commands control the active session:

| Slash Command | Action |
|---|---|
| `/?`, `/help` | List interactive slash commands |
| `/show` | Print model architecture, size, and active context length |
| `/load <tag>` | Unload current model, clear KV cache, and load `<tag>` |
| `/pull <tag>` | Download `<tag>` without leaving the session |
| `/list` | Show available and installed models |
| `/clear` | Reset KV cache for a fresh conversation |
| `/history` / `/save` | Print or save conversation history |
| `/status` / `/verbose` | Show session totals or toggle per-turn tok/s stats |
| `/think` | Toggle reasoning output (Qwen3, Qwen3.5, and other thinking models) |
| `/bye` | Exit interactive session |

### File and Vision Inputs (`/input`)

Quote **only** the UTF-8 text file or `.jpg`/`.png` image path, followed by your prompt:

```text
/input "/path/to/alice_in_wonderland.txt" Summarize it in 200 words
/input "/path/to/cat.jpg" describe this image
```

Vision inputs work automatically on open-engine GGUF VLMs when vision projector (`mmproj`) weights are installed (Qwen3-VL, Qwen3.5, and Qwen3.6-MoE; see [Models](/docs/models/)).

> [!CAUTION]
> KV-cache memory scales with `ctx-len` and system RAM (for example, Llama 3.1 8B fits up to 32k context on a 32 GB machine). If background apps exhaust free RAM when allocating NPU memory, XRT fails with `0xc01e0200` (`could not page-in all of the required allocations`).

### Session Sampling Parameters (`/set`)

Run `/set <key> <value>` using these exact key names (invalid `ctx-len` values abort the session):

| Key | Parameter | Key | Parameter |
|---|---|---|---|
| `topk` / `topp` / `minp` | Top-k, top-p, min-p | `ctx-len` | Max context length |
| `temp` | Temperature | `gen-lim` | Max generated tokens |
| `rep-pen` / `freq-pen` / `pres-pen` | Penalties | `r-eff` | Reasoning effort (`low`, `medium`, `high`; GPT-OSS) |
| `sys-msg` | System prompt | `prefill-chunk-len` | Compatibility no-op inside `/set` |

---

## Benchmarking Tool

### LLM Sweep (`oflm bench`)

Sweep context lengths from 1k to 32k (default 2 timed iterations per stage; keep JSON `max_length` within the model's context limit) and write a CSV to the working directory:

```bash
oflm bench llama3.2:1b
oflm bench llama3.2:1b --bench-iterations 4
oflm bench llama3.2:3b -i utilities/bench-configs/bench-1k.json
```

```text
[OFLM]  === Benchmark Results ===

 Context Length |              TTFT (s) |      Prefill Speed (tok/s) |     Decoding Speed (tok/s)
----------------------------------------------------------------------------------------------------
             1k |       0.815 ±   0.012 |        1233.57 ±     17.91 |       61.39 ±      0.46
             2k |       1.144 ±   0.035 |        1728.16 ±     55.68 |       59.04 ±      0.36
             4k |       1.955 ±   0.009 |        2001.55 ±      9.22 |       54.32 ±      0.47
             8k |       3.821 ±   0.010 |        2037.62 ±      5.40 |       46.89 ±      0.19
            16k |       9.144 ±   0.011 |        1698.20 ±      2.08 |       37.20 ±      0.14
            32k |      26.346 ±   0.037 |        1177.16 ±      1.53 |       26.38 ±      0.07
----------------------------------------------------------------------------------------------------
```

### Embedding Sweep (`oflm bench-embed`)

`oflm bench-embed` sweeps power-of-two batch sizes (`1` to `128` by default) and compares batched execution against a single-item loop:

```bash
oflm bench-embed bge-base:en-v1.5
oflm bench-embed bge-base --max-batch 32 --bench-iterations 5
oflm bench-embed nomic-embed-text:v1.5 --prompt-name query
oflm bench-embed bge-base -i utilities/bench-configs/bench-embed-32.json
```

`--prompt-name` is required for models declaring named task prompts (Nomic), rejected for models without them (BGE, MiniLM, GTE), and optional on EmbeddingGemma.

```text
[OFLM]  === Embedding Benchmark Results ===

  Batch |         Batched (s) | Looped (s) |  Speedup |             Texts/s |          Tokens/s
--------------------------------------------------------------------------------------------------
      1 |      0.0253 +- 0.0008 |     0.0250 |    0.99x |        39.5 +-  1.2 |        1186 +-  36
      8 |      0.0314 +- 0.0008 |     0.2018 |    6.43x |       255.0 +-  6.9 |        5451 +- 148
     64 |      0.2056 +- 0.0020 |     1.5513 |    7.55x |       311.3 +-  3.0 |        6868 +-  65
    128 |      0.5059 +- 0.0018 |     3.1064 |    6.14x |       253.0 +-  0.9 |        5582 +-  20
--------------------------------------------------------------------------------------------------
```

Runs write `bench_embed_<tag>_<date>[_<cpu>].csv`. See [Benchmarks](/benchmarks/) and [`utilities/bench-configs/README.md`](https://github.com/Atomic-Germ/OpenFlowLM-Next/blob/main/utilities/bench-configs/README.md).

### Community Submission (`--submit`)

Upload benchmark CSVs to the [Benchmarks](/benchmarks/) page:

```bash
oflm bench llama3.2:1b --submit
oflm bench-embed bge-base:en-v1.5 --submit
oflm bench qwen3.5:4b --submit --submit-consent yes
```

| Flag | Env Var | Purpose |
|---|---|---|
| `--submit` | *(none)* | Upload benchmark CSV on completion |
| `--submit-consent yes\|no` | `OFLM_BENCH_SUBMIT_CONSENT` | Include or omit anonymous hardware survey (CPU, RAM, OS, NPU) |
| `--submit-url URL` | `OFLM_BENCH_SUBMIT_URL` | Override API URL (default `https://oflm-bench.openflowlm.org`) |

On first `--submit`, interactive terminals prompt once (`[y/N]`) and save your choice plus a hashed random `install_id` in `bench_consent.json` (non-interactive runs default survey consent to `no`).

---

## Configuration and Model Registration

<a id="change-default-context-length-max"></a>

### Persistent Settings (`oflm config`)

Never edit the built-in `model_list.json` directly. Store overrides in your user `settings.json` via [`oflm config`](/docs/instructions/config/):

```bash
oflm config show
oflm config set --config-model llama3.1:8b --config-key ctx_len --config-value 32768
oflm config browse
```

| Environment Variable | Scope | Purpose |
|---|---|---|
| `OFLM_MODEL_PATH` | `global` | Parent directory containing `models/` |
| `OFLM_SETTINGS_PATH` | `global` | User `settings.json` path |
| `OFLM_SERVE_PORT` | `serve` | Default `oflm serve` port (`52625`) |
| `OFLM_HF_OWNER` | `pull` | Override default repository owner for `oflm pull` |
| `OFLM_CONFIG_PATH` / `OFLM_MODELINFO_PATH` | `global` | Custom paths to `model_list.json` / `model_info.json` |
| `OFLM_XCLBIN_PATH` | `global` | Parent directory containing bundled NPU acceleration files |
| `OFLM_OPEN_PREFILL_MODE` / `OFLM_OPEN_PREFILL_PRECISION` / `OFLM_SPECULATIVE` | `runtime` | Per-run performance and precision overrides |

### Add a Model (`oflm add`)

Install compatible GGUF models from Hugging Face, ModelScope, a URL, a local directory, or a `.gguf` file with no manual conversion:

```bash
oflm add unsloth/Qwen3.5-4B-GGUF
oflm add unsloth/gemma-3-4b-it-GGUF:Q4_K_M
oflm add ggml-org/embeddinggemma-300M-GGUF --name embed-gemma:300m
oflm add ./local-model.gguf
oflm add unsloth/Qwen3.5-4B-GGUF --dry-run
```

`oflm add` reads the GGUF header, matches it to a supported model family, and registers your new tag in `settings.json`. BERT and Nomic embedding GGUFs prepare a local `.npue` cache on first load, while EmbeddingGemma runs directly from GGUF.

| Pipeline | Supported GGUF Quantizations |
|---|---|
| {{< b "llm" "LLMs & VLMs" >}} | {{< quants "llm" >}} |
| {{< b "embed" "Embeddings" >}} | {{< quants "embed" >}} |

| Flag | Meaning |
|---|---|
| `--tag`, `--name <NAME>` | Custom local tag |
| `--family <NAME>` | Explicit model family override |
| `--dry-run` | Preview installation plan without writing files |
| `--text-only` / `--mmproj <REPO>` | Skip vision projector or download it from another repo |
| `--open-kernels <DIR>` / `--xclbin-from <REF>` / `--xclbin-dir <DIR>` | Explicit NPU acceleration directory override |
| `--no-xclbin` | Download weights without linking NPU acceleration files |
| `--config <FILE>` / `--models-root <DIR>` | Alternate `settings.json` or models folder |
| `--modelscope` | Download from ModelScope |
| `--no-verify` / `--force` | Skip SHA-256 verification or overwrite an existing tag |
| `--quiet` / `--verbose` | Quiet output or list skipped GGUF files |

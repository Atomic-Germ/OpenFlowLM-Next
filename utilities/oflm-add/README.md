# oflm-add

Install a pre-converted OFLM (Q4NX) model and register it with OpenFlowLM.

## Installation

```bash
uv tool install oflm-add    # recommended, uses uv + PEP 723 inline script
pip install oflm-add        # also works via pip
# or from source:
git clone https://github.com/atomic-germ/Q4NX_Converter.git cd Q4NX_Converter && \
uv tool install . --force || pip install -e .
```

## Usage

### From a Hugging Face repo (recommended)

```bash
oflm-add Atomic-Germ/Qwen3.5-9B-Claude-4.8-Opus-NPU2 --tag qwen3.5-claude:9b
```

The `--tag` argument must match the entry in OpenFlowLM's model registry (e.g., `qwen3.5-claude:9b`, `gptoss-distill:20b`). Run `oflm-add --help` to see all options.

### From a ModelScope repo

Hugging Face is the default hub; pass `--modelscope` for a bare repo id, or just paste a ModelScope URL and it is detected automatically:

```bash
oflm-add --modelscope Atomic-Germ/Ornith-1.0-9B-NPU2 --family qwen3.5
oflm-add https://www.modelscope.ai/models/Atomic-Germ/Ornith-1.0-9B-NPU2 --family qwen3.5
```

Both the international hub (`modelscope.ai`) and the original one (`modelscope.cn`) are queried, and downloads are size- and sha256-verified like the Hugging Face path. A local ModelScope SDK cache (`~/.cache/modelscope`) is used when present.

### From a local checkout

If you've already cloned or downloaded a repo containing the required files (`config.json`, `model.q4nx`, `tokenizer.json`, `tokenizer_config.json`), and optionally `chat_template.jinja`:

```bash
cd ~/repos/Atomic-Germ-Qwen3_5-9B-Claude-4_8-Opus-NPU2
oflm-add . --tag qwen3.5-claude:9b
```

### From a directory (no repo URL)

If you just want to install from an existing folder with the model files:

```bash
cd ~/models/Qwen3.5-9B-Claude-4.8-Opus-NPU2
oflm-add . --tag qwen3.5-claude:9b
```

## How It Works

`oflm-add`:

1. Reads `config.json` from the target model directory to extract metadata (family, engine, size, context length).
2. Validates that all required files exist (`model.q4nx`, `tokenizer.json`, etc.).
3. Writes a user-level registry at `$OFLM_CONFIG_PATH/model_list.json` (default: `~/.config/oflm/model_list.json`).
4. Adds a symlink into `$OFLM_XCLBIN_PATH/xclbins/` pointing to the model's kernel folder (`model.q4nx.xbin`). The xclbin directory name is taken from the matching official OpenFlowLM entry, keyed by family and size (e.g., `Darwin-36B-Opus-NPU2 -> Qwen3.6-35B-A3B-NPU2`). Custom OFLM models never ship xclbins because they are closed source binaries; the kernel symlink always comes from the official model it matches.
5. Links the **open kernel set** that matches the model, if one is installed (step 4's rule does not apply to it — see below).

### Open kernels are matched by spec, not by model name

Closed kernels are built for one official model, so they are linked by name. Open
kernels are built for a *spec*: the model's shape (layers, widths, heads, RoPE, vocab)
plus the weight format each projection is stored at. Every shape-identical model — a
fine-tune, a distill, a re-upload — can drive the same set, and no rebuild is needed to
add one.

So `oflm-add` derives the model's spec from what it just installed (`config.json`, the
real vocabulary from `tokenizer.json`, and the per-tensor format read out of the
`model.q4nx` header — no weight byte is touched), hashes it, and looks for an installed
set whose `open_kernels/manifest.json` carries the same `spec_hash`. It searches the
user xclbins directory first, then the system one, and prefers a set filed under the
model's own name. Deriving the spec needs an `open_kernels/` checkout; `oflm-add` finds
one beside itself in the repo, or at `$OPEN_KERNELS_DIR`.

On a match it links the set at `<model dir>/open_kernels`, which is the second place
the engine looks (`OFLM_OPEN_KERNELS_DIR`, then `<model dir>/open_kernels`, then
`<xclbins root>/<model name>/open_kernels`) and the one that does not depend on how
`OFLM_XCLBIN_PATH` is set. On Windows a plain symlink needs developer mode, so a
directory junction is used as the fallback; if neither works, `oflm-add` prints the
`OFLM_OPEN_KERNELS_DIR=...` line to use instead.

With no match, nothing changes — the model runs on the closed kernels — and `oflm-add`
prints the exact command that would build a set for it:

```bash
python open_kernels/export_qwen36_kernels.py --model-dir ~/.oflm/models/<Model>
```

`--open-kernels DIR` uses a set you name outright and skips the search. `--no-xclbin`
still skips both links.

## What This Project Is (and Isn't)

This repo contains **only** `oflm-add`, a minimal, dependency-free Python tool for installing pre-converted Q4NX models into OpenFlowLM. It does **not** include:
- A converter (`convert.py`) — that lives upstream in the AMD project.
- The `q4nx/` conversion library or its model-specific implementations.
- Configuration files under a `configs/` directory (those belong to the converter, not `oflm-add`).

## Project Structure

```text
oflm_add/          # Installable Python package (stdlib-only)
├── __init__.py    # Core logic: registry writing, symlink creation
└── __main__.py    # CLI entry point (argparse → oflm_add.main())

oflm-add.py        # Standalone shim; delegates to oflm_add.main()

dist/              # Wheel + sdist after `uv build`
  └── .gitignore   # Ignore built artifacts in the repo
```

## License

Apache-2.0 — see [`LICENSE`](./LICENSE).
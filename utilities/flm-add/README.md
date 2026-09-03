# flm-add

Install a pre-converted FLM (Q4NX) model and register it with FastFlowLM.

## Installation

```bash
uv tool install flm-add    # recommended, uses uv + PEP 723 inline script
pip install flm-add        # also works via pip
# or from source:
git clone https://github.com/atomic-germ/Q4NX_Converter.git cd Q4NX_Converter && \
uv tool install . --force || pip install -e .
```

## Usage

### From a Hugging Face repo (recommended)

```bash
flm-add Atomic-Germ/Qwen3.5-9B-Claude-4.8-Opus-NPU2 --tag qwen3.5-claude:9b
```

The `--tag` argument must match the entry in FastFlowLM's model registry (e.g., `qwen3.5-claude:9b`, `gptoss-distill:20b`). Run `flm-add --help` to see all options.

### From a ModelScope repo

Hugging Face is the default hub; pass `--modelscope` for a bare repo id, or just paste a ModelScope URL and it is detected automatically:

```bash
flm-add --modelscope Atomic-Germ/Ornith-1.0-9B-NPU2 --family qwen3.5
flm-add https://www.modelscope.ai/models/Atomic-Germ/Ornith-1.0-9B-NPU2 --family qwen3.5
```

Both the international hub (`modelscope.ai`) and the original one (`modelscope.cn`) are queried, and downloads are size- and sha256-verified like the Hugging Face path. A local ModelScope SDK cache (`~/.cache/modelscope`) is used when present.

### From a local checkout

If you've already cloned or downloaded a repo containing the required files (`config.json`, `model.q4nx`, `tokenizer.json`, `tokenizer_config.json`), and optionally `chat_template.jinja`:

```bash
cd ~/repos/Atomic-Germ-Qwen3_5-9B-Claude-4_8-Opus-NPU2
flm-add . --tag qwen3.5-claude:9b
```

### From a directory (no repo URL)

If you just want to install from an existing folder with the model files:

```bash
cd ~/models/Qwen3.5-9B-Claude-4.8-Opus-NPU2
flm-add . --tag qwen3.5-claude:9b
```

## How It Works

`flm-add`:

1. Reads `config.json` from the target model directory to extract metadata (family, engine, size, context length).
2. Validates that all required files exist (`model.q4nx`, `tokenizer.json`, etc.).
3. Writes a user-level registry at `$FLM_CONFIG_PATH/model_list.json` (default: `~/.config/flm/model_list.json`).
4. Adds a symlink into `$FLM_XCLBIN_PATH/xclbins/` pointing to the model's kernel folder (`model.q4nx.xbin`). The xclbin directory name is taken from the matching official FastFlowLM entry, keyed by family and size (e.g., `Darwin-36B-Opus-NPU2 -> Qwen3.6-35B-A3B-NPU2`). Custom FLM models never ship xclbins because they are closed source binaries; the kernel symlink always comes from the official model it matches.

## What This Project Is (and Isn't)

This repo contains **only** `flm-add`, a minimal, dependency-free Python tool for installing pre-converted Q4NX models into FastFlowLM. It does **not** include:
- A converter (`convert.py`) — that lives upstream in the AMD project.
- The `q4nx/` conversion library or its model-specific implementations.
- Configuration files under a `configs/` directory (those belong to the converter, not `flm-add`).

## Project Structure

```text
flm_add/          # Installable Python package (stdlib-only)
├── __init__.py    # Core logic: registry writing, symlink creation
└── __main__.py    # CLI entry point (argparse → flm_add.main())

flm-add.py        # Standalone shim; delegates to flm_add.main()

dist/              # Wheel + sdist after `uv build`
  └── .gitignore   # Ignore built artifacts in the repo
```

## License

Apache-2.0 — see [`LICENSE`](./LICENSE).
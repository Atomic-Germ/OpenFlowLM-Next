# gguf-inspect

Phase 0 baseline tooling: metadata inspection for GGUF model sets.

## What it does

`gguf-inspect` reports everything an installer must know before deciding how to
load or pack a model — without opening the NPU or decoding any tensor payload:

- architecture (`general.architecture`)
- block structure: `block_count`, `nextn_predict_layers`, trunk layers
  (`block_count - nextn_predict_layers`), and MTP/speculative classification
- projector components (`clip` arch + `general.base_model`)
- tensor-carried RoPE factors (`rope_freqs.weight`, etc.)
- tokenizer/template (`tokenizer.chat_template`, bos/eos/pad ids, pre/tokenizer model)
- a candidate install tag (`general.basename` + `size_label`)
- execution options: file_type, quantization_version, dtype histogram
- precise unsupported requirements that block execution
- provenance: HF snapshot hash (auto-derived from the path) or `--source`

## Usage

```sh
# human-readable report
python3 utilities/gguf-utils/gguf-inspect/gguf-inspect path/to/file.gguf
# or via the shim
./utilities/gguf-utils/gguf-inspect/gguf-inspect path/to/file.gguf

# machine-readable (also usable as an input to install planning)
python3 utilities/gguf-utils/gguf-inspect/gguf-inspect path/to/file.gguf --json

# focus on sections
python3 utilities/gguf-utils/gguf-inspect/gguf-inspect path/to/file.gguf --only blocks rope tokenizer exec tags
```

## How it works

It needs the GGUF python reader, and takes the first one it finds:

1. an importable `gguf` — install it with
   `pip install git+https://github.com/ggml-org/llama.cpp` (this is what CI
   does; note that this pulls in transformers/tokenizers/protobuf and will
   downgrade them, so prefer a dedicated venv over the project's `ironvenv`);
2. a copy vendored at `third_party/Guanaco/llama.cpp/gguf-py`, searched for by
   walking up from this file, so the tool works in either the nested or the flat
   layout and never depends on a `../llama.cpp` checkout being present.

Only the header and tensor inventory are read; payloads are never touched and the NPU
is never opened.

## Tests

```sh
python3 utilities/gguf-utils/gguf-inspect/tests/test_inspect.py
```

The baseline pins both pilot HF snapshots so the exit-gate facts cannot silently
drift. Those snapshots are not in the repository: each is resolved from the local
Hugging Face cache (`HF_HUB_CACHE` / `HF_HOME/hub` / `~/.cache/huggingface/hub`)
at its pinned revision, and can be pointed elsewhere with `GGUF_TEST_QWEN`,
`GGUF_TEST_MMPROJ` and `GGUF_TEST_LLAMA`. Tests whose model is unavailable skip,
so the suite is green on CI where no model is cached.

The suite also runs a `PureLogicTests` group that needs no model at all: it writes
a minimal synthetic GGUF and checks the dtype allowlist, the candidate-tag
derivation and the source/snapshot split. Those run everywhere, CI included, and
are what keeps the metadata logic covered without a multi-gigabyte download.

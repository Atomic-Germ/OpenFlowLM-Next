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
python3 utilities/gguf-inspect/gguf-inspect path/to/file.gguf
# or via the shim
./utilities/gguf-inspect/gguf-inspect path/to/file.gguf

# machine-readable (also usable as an input to install planning)
python3 utilities/gguf-inspect/gguf-inspect path/to/file.gguf --json

# focus on sections
python3 utilities/gguf-inspect/gguf-inspect path/to/file.gguf --only blocks rope tokenizer exec tags
```

## How it works

It reads the pinned GGUF python reader vendored under
`third_party/Guanaco/llama.cpp/gguf-py`. The reader is located relative to this
file, so the tool never depends on a `../llama.cpp` checkout being present. Only
the header and tensor inventory are read; payloads are never touched and the NPU
is never opened.

## Tests

```sh
python3 utilities/gguf-inspect/tests/test_inspect.py
```

The baseline pins both pilot HF snapshots so the exit-gate facts cannot silently
drift.

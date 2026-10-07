# oflm-support

Inspect a local q4nx model against the open kernel families available in an
OpenFlowLM-Next checkout. This command is **diagnostic**, not an allowlist and
not a test-suite verdict. The installer remains the project-owned `oflm-add`
implementation; the app's `oflm add` command is its wrapper, not a different
installer or policy layer.

```bash
python utilities/oflm-support/oflm_support.py \
  ~/.config/oflm/models/Some-Finetune-NPU2 \
  --xclbin-root ~/.config/oflm/xclbins \
  --xclbin-root src/xclbins
```

Machine-readable output:

```bash
python utilities/oflm-support/oflm_support.py MODEL_DIR --json
```

Set `OFLM_SUPPORT_ISSUE_URL` when you want `oflm-add`/`oflm-support` to print a
prefilled issue URL. The destination is configurable so an organization/repo
transfer does not require changing the installed tool.

## Requirements and curation are different answers

The report keeps these separate:

- **Requirements** come from the model's `config.json`, tokenizer, q4nx tensor
  header, and (when available) the family recipe. This is the content the
  runtime will actually consume.
- **Curated status** says whether that exact spec has a checked-in recipe spec
  or the model name has a model-list entry. It is evidence of maintainer
  attention, not permission to use kernels.
- **Kernel candidates** include exact hashes and same-family manifests. A hash
  mismatch never erases a same-family candidate; descriptor differences are
  printed so a person/runtime can diagnose them.
- **Family xclbin donors** report shipped model directories whose
  `details.family` matches the config-derived runtime family. The model itself
  need not be in `model_list.json`.

The helper deliberately does not link anything or declare a model supported.
It is a first, non-invasive slice for debugging routing and for being called by
`oflm-add`/`q4nx-build` later without changing their existing behavior. Exact
hashes remain useful provenance; the kernel-family compatibility path is not
restricted to hashes or model names.

## Limitations

- This first slice inspects a local converted model directory, not a remote HF
  repo directly. `oflm-add` downloads the files anyway; remote-only inspection
  can be added separately without touching weight files.
- A family/shape candidate is a hypothesis. The report includes runtime
  manifest differences instead of silently claiming that a candidate is
  validated. A real mismatch should become an actionable runtime error naming
  the config field, tensor, or geometry that disagreed.
- Without an open-kernels checkout (`OPEN_KERNELS_DIR`, adjacent checkout, or
  cwd), recipe validation is unavailable. Installed manifest and family donor
  discovery can still be added through a generated release index.

`model_list.json` is a model-discovery convenience. It does not define the set
of models the open toolchain may attempt to run.

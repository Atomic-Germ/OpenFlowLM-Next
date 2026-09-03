# Add `open_npue`: a second embedding backend, six new models

Adds the [NpuEmbeddings][upstream] engine as a second `AutoEmbeddingModel`, and
turns `--embed` into something you can point at a model.

```
flm serve llama3.2:1b --embed 1 --embeddingmodel bge-base:en-v1.5
```

**It is additive.** `open_embedding` stays exactly where it is and keeps serving
`embed-gemma:300m`. Nothing that works today changes: `flm help` is
byte-identical apart from the two lines describing the new flag.

---

## Why

Issue #690 asked for enough of the stack to be open that the community can add
models. The embedding side currently serves one. This adds six, all source-built
with no closed binary anywhere and with the IRON kernel source that produces
their xclbins — and it makes the embedding side a **registry** rather than a
funnel, so adding the seventh is one line.

The two backends are genuinely complementary and both are worth having:

| | `open_embedding` | `open_npue` |
|---|---|---|
| shape | CPU forward pass, 5 projections/layer offloaded | whole layer = **4 GEMM dispatches**, one resident xclbin, one `hw_context` |
| weights | safetensors, converted per call | pre-tiled in a `.npue`, staged on the device **once** |
| batching | one text | batch tiers 4/16/32/128 — right-sized, not padded |
| generality | **any** shape with a matching xclbin | needs a compiled design per **geometry** |
| models | EmbeddingGemma-300M | bge-{small,base,large}, all-MiniLM, nomic, gte-multilingual |

*Generality by default; a fast path where a design exists.*

Design sets are keyed by **GEMM geometry, not fine-tune name** — this tree's own
kernel policy falling out for free. One `BERT-768-NPUE` set serves bge-base,
gte-multilingual **and** nomic. Four sets cover seven models.

---

## Measured

**End-to-end HTTP latency**, same binary, same endpoint, median of three after a
warm-up. Wall clock — it includes tokenization, the host half of the encode,
JSON and the socket. **Not an NPU kernel claim.**

| request | `open_embedding` (EmbeddingGemma-300M) | `open_npue` (bge-base, 109M) |
|---:|---:|---:|
| 1 text | 1195.7 ms | **34.3 ms** |
| 4 texts | 1182.8 ms/text | **26.8 ms/text** |
| 16 texts | 1202.6 ms/text | **25.3 ms/text** |

**Read that as engine + model, not engine alone.** EmbeddingGemma-300M is 2.75×
the parameters of bge-base and carries a 262k-token vocabulary against 30.5k.
The honest claim is that the shipped pairing is ~35–47× faster per text; how
much is the engine and how much the model is not separable from these numbers.

**Correctness.** The vectors are **bit-identical** to the upstream engine's own
binary — 768 of 768 components exact, max abs diff `0.000e+00` — from a
container this tree downloaded and packed itself. Against sentence-transformers,
upstream's golden gate reads `1-cos 2.284e-04` for bge-base on this datapath.
(`open_embedding`'s own validation reports E8 cosine 0.999993; that is its
claim, not a measurement made here.)

**~5.8× is still on the table.** `AutoEmbeddingModel::embed()` takes one text,
so `handle_embeddings` loops. Sixteen texts cost 405 ms through the endpoint and
**70 ms** as one batched call to the same engine. `NpueEmbedding::embed_batch()`
exists and is unused — widening the base class deserves to be judged on its own,
in its own PR.

---

## What a user does

```
flm pull bge-base:en-v1.5      # BAAI's own files, 438 MB
flm serve llama3.2:1b --embed 1 --embeddingmodel bge-base:en-v1.5
curl -s localhost:52625/v1/embeddings -H 'content-type: application/json' \
  -d '{"model":"bge-base:en-v1.5","input":["A man is playing a guitar on stage."]}'
```

**Nothing is re-hosted.** The model entry points at BAAI's repository, so the
weights a user gets are the author's bytes with the author's hash. The `.npue`
container — the same weights pre-tiled for the array — is packed **locally on
first run**, which costs about a minute for a 109M model and is then mmapped
forever after.

The whole cold path was exercised for this PR: download → pack → serve →
`/v1/embeddings`, bit-exact at the end of it.

---

## Three downloader fixes

Each was found by *using* the downloader, not by reading it, and each is the
same shape: something goes wrong and the tool reports success.

1. **`build_download_list()` silently skipped files.** A file that
   `model_list.json` requires and `model_info.json` does not describe hit a bare
   `continue`; `pull_model()` then downloaded nothing and printed success. A
   model added to the first file without the second was un-downloadable, in
   silence. It names them and refuses now.

2. **`get_missing_files()` only checked that a file exists.** An interrupted
   download leaves a truncated file, the next `pull` skips it (only absent files
   are fetched), and `pull_model` prints *"All files verified successfully"*.
   Observed here for real: a 50 MB `model.safetensors` where the manifest says
   417 MB, reported as verified. It compares the manifest's size now.

3. **`check_model_compatibility()` reported every author-hosted model as
   `Outdated`, forever.** `LM_Config` defaults `flm_version` to `"0.0.0"` when
   `config.json` has no such key — and no upstream HuggingFace checkpoint has
   one, because it is a field this project writes. **`embed-gemma:300m` was in
   exactly that state on a freshly pulled tree**: a warning triangle in
   `flm list`, and `ensure_embed_model_loaded()` re-pulling a complete, correct
   download on every start. Since "absent" and `"0.0.0"` were already
   indistinguishable, treating the default as *"not an FLM artifact"* changes
   nothing for anything that really carries a version. Both embedding models
   read `✅` now; `embed-gemma` did not before.

These are separable from the backend and can be split out if you would rather
review them alone.

---

## Two repository fixes, which came first because nothing built

`git submodule update --init` — the first thing a new contributor runs —
**failed outright**:

```
fatal: no submodule mapping found in .gitmodules for path 'docs/ExampleNPU'
```

`docs/ExampleNPU` is a gitlink with no `.gitmodules` entry, and git refuses to
touch **any** submodule while one is present. It is a leftover: it was the tree
`open_embedding` was ported *from*, and that port has landed.

Removing it exposed the second problem. `.gitmodules` listed four submodules and
**none of them was one** — `utilities/flm-add`, `q4nx-build` and `flm-test` are
vendored (6, 51 and 15 tracked blobs), and `third_party/tokenizers-cpp`, which
`src/CMakeLists.txt` `add_subdirectory()`s and links into `flm`, **did not
exist**. The second was invisible because of the first: the command that would
have reported it was already refusing to run.

Both are separate commits at the base of this branch.

---

## Building

Nothing new is required.

```powershell
cmake -S src -B src/build -G Ninja `
      -DCMAKE_BUILD_TYPE=Release `
      -DFLM_VERSION=0.9.25 -DNPU_VERSION=0.9.25 `
      -DFLM_USE_HRX=OFF `
      -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build src/build --target flm
```

Three CMake choices are load-bearing rather than stylistic, and
`src/open_npue_adapter/README.md` explains each:

* **`/arch:AVX2` per-source** (`-mavx2 -mfma` off MSVC). Half an encode is
  host-side AVX2 intrinsics behind `#if defined(__AVX2__)` with correct scalar
  fallbacks, so **without the flag it compiles, runs, returns the right vectors
  and is 2.1–2.6× slower.**
* **The include path is per-source, not global.** `open_npue/tokenizer.hpp` and
  `src/include/tokenizer/tokenizer.hpp` share a basename.
* **`NOT FLM_USE_HRX`**, exactly like `npu_matmul.cpp`.

### The trap worth knowing about

**`npue_encoder.hpp` must be included by exactly one translation unit**, and it
is — the public header is a PIMPL. That is enforcement, not tidiness.

The first version of the adapter included the engine header directly. It
compiled, linked, ran and returned **slightly wrong numbers**: `1-cos 1.04e-04`
against the same engine in its own binary, on byte-identical container, design
set and text. Because `/arch:AVX2` is per-source, every inline engine function
instantiated in `rest_handler.cpp` compiled down the **scalar** path while the
same functions in the `open_npue` objects compiled down the **AVX2** one. Two
definitions of one inline function is an ODR violation; the linker keeps one
COMDAT copy per function, chosen by link order. Both paths are correct and
reduce in different orders, so the mixture is a plausible, unit-norm,
deterministic, **wrong** vector. No warning, no link error.

General form: *a header-only library whose code is guarded by ISA macros cannot
be included by a host TU compiled at a different ISA level.*

---

## Everything refuses rather than guessing

Each of these produces a **correctly shaped, correctly normed, deterministic
vector** if it is allowed to guess. That is the whole reason they are errors.

| situation | before | now |
|---|---|---|
| unknown embedding tag | rewritten to `embed-gemma:300m` and served | error naming the known tags |
| model has prompts, task matches none | — | error naming what it does offer |
| two `.npue` in one directory | — | error; they differ in datapath or seq |
| no design set | — | error naming both places it looked |
| second model in one process | — | the engine's `ShapeLease` refuses |
| `model_list.json` names a file the manifest lacks | skipped, "success" | error |
| truncated download | "verified successfully" | treated as missing |

---

## Provenance

`src/open_npue/` is a **synced copy** — see its `SYNCED.md` for the source
commit and per-file sha256. Edit it upstream, not here: the gates that make its
numbers mean anything (a HuggingFace golden gate, cross-lane bitwise agreement
at four lanes, an MTEB bridge, a p99 tail gate, an end-to-end gate against a
live reference, and four per-tokenizer byte-exactness harnesses) live there and
want an NPU. `src/open_npue_adapter/` is fork-owned and the sync tool does not
touch it.

The synced sources are **MIT**, relicensed on copy by their sole author;
upstream is Apache-2.0.

`src/xclbins/BERT-768-NPUE/` is 604 KB of xclbin and instruction streams built
from IRON source, with a `toolchain.json` recording the mlir-aie version, the
Peano version and the git HEAD that produced it.

---

## Not done, and not pretended

* **Only `bge-base:en-v1.5` is wired.** The other five need a `model_list.json`
  entry, a `model_info.json` manifest and one registry line each; two of them
  need no new design set at all.
* **`usage` in the response is still `{0, 0}`** — hardcoded in
  `handle_embeddings` for both backends, untouched here.
* **Two co-resident `hw_context` objects are unmeasured.** `flm` holds the LLM's
  and the engine holds its own. They were never actually exercised together
  here, because the LLM could not load for want of its own xclbins.
* **Linux is unverified.** The engine's platform-independent subset compiles
  there at C++17 and C++20 (upstream runs that check), but `flm` itself was only
  built and run on Windows for this PR.

[upstream]: https://github.com/vegardberget/NpuEmbeddings

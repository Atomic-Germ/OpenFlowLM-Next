# `open_npue` — a second embedding backend

> ## ⚠️ DRAFT — needs testing on other machines
>
> **This is a first cut, not a finished change.** Everything below was measured,
> but all of it on **one machine**: one Ryzen AI 9 HX 370 (Strix Point), one
> XRT, one MSVC, one vcpkg tree. Nothing here has been run by anyone else.
>
> What most needs a second pair of eyes and a second machine:
>
> * **Does it build for you at all?** The CMake changes are small but they
>   touch the shared source list, and the vcpkg/Boost path this was built
>   through may not be the one you use.
> * **Do you get the same vectors?** They are bit-identical to the upstream
>   binary *here*. They are **not** expected to be bit-identical across
>   different host ISA levels — that is measured and documented below — so a
>   mismatch is informative rather than automatically a bug. `1-cos` against
>   sentence-transformers is the check that should hold anywhere.
> * **Linux is unverified for `flm`.** The engine's platform-independent subset
>   compiles there at C++17 and C++20, but the binary was only built and run on
>   Windows.
> * **The two co-resident `hw_context` objects are unmeasured**, because the
>   LLM never loaded here for want of its own xclbins. A real
>   `serve <llm> --embed 1` on a machine with a full xclbin tree is the test
>   that has not happened.
> * **The downloader fixes touch shared code.** They are separable and should
>   probably be reviewed on their own.
>
> Treat the numbers as *what one machine did*, and the design as a proposal.

`src/open_npue/` is a synced copy of the [NpuEmbeddings][upstream] engine.
`src/open_npue_adapter/` is the fork-owned glue that makes it an
`AutoEmbeddingModel`.

It is **additive**. `open_embedding` stays exactly where it is and keeps serving
`embed-gemma:300m`; this adds six encoders the tree did not have.

```
flm serve llama3.2:1b --embed 1 --embeddingmodel bge-base:en-v1.5
curl -s localhost:52625/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"model":"bge-base:en-v1.5","input":["A man is playing a guitar on stage."]}'
```

---

## What it is, and how it differs from `open_embedding`

The two are genuinely complementary and both are worth having.

| | `open_embedding` | `open_npue` |
|---|---|---|
| shape | CPU forward pass, 5 projections/layer offloaded | whole layer = **4 GEMM dispatches**, one resident xclbin, one `hw_context` |
| weights | safetensors, converted per call | pre-tiled in a `.npue` container, staged on the device **once** |
| batching | one text | batch tiers (4/16/32/128) — a request is right-sized, not padded |
| generality | **any** shape with a matching xclbin | needs a compiled design per **geometry** |
| models | EmbeddingGemma-300M | bge-{small,base,large}, all-MiniLM, nomic, gte-multilingual |

Design sets are keyed by **GEMM geometry, not by fine-tune name** — which is
kernel policy falling out for free -- but "geometry" means more than width.
`design_fits()` checks hidden, intermediate, **whether the FFN is gated**, and
the **datapath**, so two models share a set only when all four agree:

| family | hidden / inter | gated | datapath | serves |
|---|---|---|---|---|
| `BERT-h384-bfp16` | 384 / 1536 | no | bfp16 | all-MiniLM-L6-v2 |
| `BERT-h384-bf16` | 384 / 1536 | no | **bf16** | bge-small-en-v1.5 |
| `BERT-h768-bfp16` | 768 / 3072 | no | bfp16 | bge-base-en-v1.5 |
| `BERT-h768-gated-bfp16` | 768 / 3072 | **yes** | bfp16 | nomic **and** gte-multilingual |
| `BERT-h1024-bfp16` | 1024 / 4096 | no | bfp16 | bge-large (`tile_n` 32) |

**Five sets covering six models**, and the sharing is real but
narrower than width alone suggests. Two of the distinctions are worth naming
because they are easy to get wrong: nomic and gte have a **gated** FFN and
bge-base does not, so they cannot share a 768 set; and `bge-small` runs on
**plain bf16** rather than bfp16, because it is the one model that failed
upstream's MTEB gate on the emulated datapath. The container and the design
each record their datapath and a mismatched pair is refused rather than read as
garbage.

*Generality by default; a fast path where a design exists.*

---

## Measured

**End-to-end HTTP latency**, same binary, same endpoint, median of three after
a warm-up. This is wall-clock request latency, **not** an NPU kernel claim: it
includes tokenization, the host-side half of the encode, JSON and the socket.

| request | `open_embedding` (EmbeddingGemma-300M) | `open_npue` (bge-base, 109M) |
|---:|---:|---:|
| 1 text | 1195.7 ms | **34.3 ms** |
| 4 texts | 1182.8 ms/text | **26.8 ms/text** |
| 16 texts | 1202.6 ms/text | **25.3 ms/text** |

**Read that as engine + model, not as engine alone**: EmbeddingGemma-300M is
2.75× the parameters of bge-base and carries a 262k-token vocabulary against
30.5k. The honest statement is that the shipped pairing is ~35–47× faster per
text; how much of that is the engine and how much the model is not separable
from these numbers.

**Correctness.** The vectors are **bit-identical** to the upstream engine's own
binary — 768 of 768 components exact, max abs diff `0.000e+00` — with a
container this tree packed itself from BAAI's own files. Against
sentence-transformers, upstream's golden gate reads `1-cos 2.284e-04` for
bge-base on this datapath. (`open_embedding`'s own validation reports E8 cosine
0.999993; that is its claim, not a measurement made here.)

**All six models verified end to end**, each `flm pull` → pack → `serve` →
`POST /v1/embeddings`, each compared against the same engine in its own binary:

| tag | dims | datapath the design records | vs `npuembed --embed` |
|---|---:|---|---|
| `all-minilm:l6-v2` | 384 | bfp16-emulated MMAC, C as bf16 | **384/384 exact** |
| `bge-small:en-v1.5` | 384 | **bf16 MMAC, C as fp32** | **384/384 exact** |
| `bge-base:en-v1.5` | 768 | bfp16-emulated MMAC, C as bf16 | **768/768 exact** |
| `bge-large:en-v1.5` | 1024 | bfp16-emulated MMAC, C as bf16 | **1024/1024 exact** |
| `nomic-embed-text:v1.5` | 768 | bfp16-emulated MMAC, C as bf16 | **768/768 exact** |
| `gte-multilingual:base` | 768 | bfp16-emulated MMAC, C as bf16 | **768/768 exact** |

`bge-small` is the row worth looking at twice: it is the one model that runs on
**plain bf16**, because it failed upstream's MTEB gate on the emulated datapath.
The container and the design each record their datapath and a mismatched pair is
refused — so the fact that it loaded, and loaded on the right one, is the guard
working through this tree.

And `gte-multilingual` does what it is for: English and French sentences with the
same meaning score **cos 0.9144**, unrelated Norwegian and Chinese 0.36 and 0.30.

`utilities/test_open_npue.ps1` is the harness; it takes `-Upstream <path>`.

### Same model, both engines

The table above pairs each engine with the model it ships with, which conflates
engine and model. This isolates them: **EmbeddingGemma-300M through both.**

`open_task_prefix(task_query)` returns `"task: search result | query: "`, and
that is *exactly* what the NpuEmbeddings container's own `query` prompt is — so
the same model, the same prompt and the same text go through both.

| | `open_embedding` | NpuEmbeddings' arch=1 path |
|---|---:|---:|
| 1 text | 1195.7 ms (HTTP round trip) | **105 ms** (encode only) |
| 16 texts | 1202.6 ms/text | **9.4 ms/text** |
| agreement | — | `1-cos` **1.430e-04** vs the other |

The two are not measured the same way — one is an HTTP round trip and one is an
encode — so the per-request overhead has to be bounded before the ratio means
anything. It is small: a 16-text bge-base request costs 405 ms through the
endpoint against 70 ms of encode, i.e. **≈21 ms per request** of HTTP, JSON and
loop. Against 1196 ms that is under 2%.

So, on the same model: **≈11× at one text and ≈128× at sixteen**, the gap
widening because one engine batches and the other cannot.

**And the accuracy result is the more interesting half.** Two engines written
independently — a CPU fp32 forward pass with five per-layer projections
offloaded, against four whole-layer GEMM dispatches over a resident xclbin with
bfp16-emulated MACs — agree on the same input to **`1-cos` 1.430e-04**. Neither
was written against the other. That is mutual validation of both, and it is
measured here rather than cited from either project's own claims.

> **This comparison is engine-to-engine, not what this PR wires up.**
> EmbeddingGemma is arch=1 in NpuEmbeddings and runs through a code path that
> lives in its CLI rather than in the library, so `NpueEmbedding` cannot serve
> it today. `embed-gemma:300m` stays with `open_embedding`, which is the right
> outcome regardless: it is live, validated code and this PR is additive.

**There is ~5.8× still on the table.** `AutoEmbeddingModel::embed()` takes one
text, so `handle_embeddings` loops. Sixteen texts cost 405 ms through the
endpoint and **70 ms** as one batched call to the same engine — 5.8×, which is
exactly what upstream measures for a single text through the smallest tier.
`NpueEmbedding::embed_batch()` exists and is unused; widening the base class is
a separate change that deserves to be judged on its own.

---

## Building

Nothing new is required. `open_npue` compiles with the rest of `flm`.

```powershell
cmake -S src -B src/build -G Ninja `
      -DCMAKE_BUILD_TYPE=Release `
      -DFLM_VERSION=0.9.25 -DNPU_VERSION=0.9.25 `
      -DFLM_USE_HRX=OFF `
      -DCMAKE_TOOLCHAIN_FILE=C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build src/build --target flm
```

Three things about the CMake are load-bearing rather than stylistic:

* **`/arch:AVX2` per-source** (`-mavx2 -mfma` off MSVC). Roughly half an encode
  is host-side AVX2 intrinsics behind `#if defined(__AVX2__)` with correct
  scalar fallbacks, so **without the flag it compiles, runs, returns the right
  vectors and is 2.1–2.6× slower** — measured upstream on three models with
  every accuracy gate green. MSVC x64 defaults to the SSE2 baseline.
* **The include path is per-source, not global.** `open_npue/tokenizer.hpp` and
  `src/include/tokenizer/tokenizer.hpp` are different files with the same
  basename; on the global path the wrong one wins for some translation unit
  depending on directory order.
* **`NOT FLM_USE_HRX`**, exactly like `npu_matmul.cpp`: `npu_device.cpp` talks
  to XRT directly.

### The one trap worth reading before you touch this

**`npue_encoder.hpp` must be included by exactly one translation unit**, and
that unit must carry `/arch:AVX2`. It is enforced by construction — the public
header is a PIMPL, so nothing else can see the engine — and the enforcement is
not decorative.

The first version of the adapter included the engine header directly. It
compiled, linked, ran and returned **slightly wrong numbers**: `1-cos 1.04e-04`
against the same engine in its own binary, on byte-identical container, design
set and text. Because `/arch:AVX2` is per-source, every inline engine function
instantiated in `rest_handler.cpp` compiled down the **scalar** path while the
same functions in the `open_npue` objects compiled down the **AVX2** one. Two
definitions of one inline function is an ODR violation; the linker keeps one
COMDAT copy per function, chosen by link order. Both paths are correct and
reduce in different orders, so the resulting mixture is a plausible, unit-norm,
deterministic, **wrong** vector. There is no warning and no link error.

General form: *a header-only library whose code is guarded by ISA macros cannot
be included by a host translation unit compiled at a different ISA level.*

---

## The design sets are BUILT, not checked in

`src/xclbins/BERT-*/` is `.gitignore`d. Build it from `npu_offload/gemm_rtp/`,
which has one exact command per design family and the three AIE kernel sources
the generator compiles; `check_design_sets.py` there verifies the flags still
describe the sets on disk.

All five families rebuild from an empty tree to **88 of 96 files
byte-identical**. The eight that differ are the five `final.xclbin`s, by 402
bytes of 631,126 (0.064%) — UUIDs and build timestamps, in 5-6 tight clusters
each, with the embedded AIE core ELFs identical.

Without them `flm` still builds; an `open_npue` model refuses to load, naming
the README and the command that fixes it.


## Adding a model

Two entries and, once, a design set.

1. **`src/model_list.json`** — the model's files, pointing at the **author's own
   upstream repository**. Nothing is re-hosted: the weights a user gets are the
   author's bytes with the author's hash.
   * `npue_design_family` names the compiled design set (a geometry).
   * `npue_tile_n` when the widths need one other than 48 (bge-large needs 32:
     the design asserts `N % (tile_n * n_cols) == 0` and its `N ∈ {1024, 3072,
     4096}`).
   * `flm_min_version` should be `"0.0.0"` — see below.
2. **`src/model_info.json`** — the per-file manifest (`path`, `size`, `oid`).
   `build_download_list()` needs it to build URLs; the HuggingFace API path
   that would make it unnecessary is commented out. Generate it from
   `https://huggingface.co/api/models/<repo>/tree/main`.
3. **`AutoEmbeddingModel/all_embedding_model.hpp`** — one line in the registry.

The `.npue` container is **packed on first run** from the checkpoint, by
`npue::prepare_model_auto()` in the engine. The packing decision (architecture,
tile width, pooling, source repository) deliberately lives upstream: a second
copy of it here would have to agree byte for byte, and upstream's byte-identity
gate would stop being evidence about these containers the moment they diverged.
A first run therefore costs a pack — about a minute for a 109M model — and
every run after it mmaps the result.

The design set is a directory of one xclbin plus its instruction streams. Ship
it as `<model_dir>/npue_designs/` (model-local, wins) or under the installed
xclbin tree as `xclbins/<family>/`, mirroring `open_embedding`'s own two-tier
lookup.

---

## Refusals, and why each one is there

Every one of these produces a **correctly shaped, correctly normed,
deterministic vector** if it is allowed to guess instead. That is why they
refuse.

| situation | what happens |
|---|---|
| unknown embedding tag | error naming the known tags — it used to be rewritten to `embed-gemma:300m` and served |
| model has prompts, task maps to none of them | error naming what the model does offer |
| model has no prompts, a prompt is given | error |
| two `.npue` containers in one directory | error — they differ in datapath or sequence length; name one with `npue_container` |
| no design set for the geometry | error naming both places it looked |
| a second model loaded in one process | the engine's `ShapeLease` refuses; geometry is process-wide |
| `model_list.json` names a file `model_info.json` does not | error — it used to be skipped silently and reported as a successful download |

[upstream]: https://github.com/vegardberget/NpuEmbeddings

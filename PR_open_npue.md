# Add `open_npue`: a second embedding backend (draft)

> This is a draft. The first version had the xclbins in as binary files. They
> are **gone now** — `.gitignore`d and built from `npu_offload/gemm_rtp/`,
> which is in the PR. Deleting them is what proved the generator was not
> self-sufficient: three AIE kernel sources had never been copied, so this
> repository could not build a single design family. Fixed, and all five now
> rebuild from an empty tree to **88 of 96 files byte-identical**, the rest
> being UUIDs and build timestamps inside the xclbins.
>
> **Linux is confirmed.** A second machine — Fedora Rawhide on a Framework 13
> AI 340, no IDE — has built and run it. That retires the largest caveat this
> PR opened with; the remaining ones are listed below and are unchanged.
>
> **The endpoint tests now live in `utilities/flm-test`**, not in a script of
> their own. See *Testing it* below.

> ### Depended on #3 — *"Fix the submodule metadata: a fresh clone cannot build"* — **now merged**
>
> Without it this tree did not configure, so a reviewer could not build this
> branch to look at it: `git submodule update --init` failed outright on a
> dangling `docs/ExampleNPU` gitlink, and `third_party/tokenizers-cpp` — which
> `src/CMakeLists.txt` `add_subdirectory()`s and links into `flm` — did not
> exist.
>
> Those two commits are the base of this branch, so they still appear in this
> diff until it is rebased on the merged `main`; after that this is ten commits
> of embedding backend and nothing else.

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
> * ~~**Linux is unverified for `flm`.**~~ **Verified** — built and run on
>   Fedora Rawhide (Framework 13 AI 340). The engine's platform-independent
>   subset also compiles there at C++17 and C++20, which is the check upstream
>   runs on every change.
> * **The two co-resident `hw_context` objects are unmeasured**, because the
>   LLM never loaded here for want of its own xclbins. A real
>   `serve <llm> --embed 1` on a machine with a full xclbin tree is the test
>   that has not happened.
> * **The downloader fixes touch shared code.** They are separable and should
>   probably be reviewed on their own.
>
> Treat the numbers as *what one machine did*, and the design as a proposal.

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

## Five fixes to the shared paths

Each was found by *using* the thing, not by reading it, and each is the same
shape: something goes wrong and the tool carries on as though it had not.

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

4. **One malformed byte wedged the whole server, permanently.** This is the
   one to look at, because the mechanism is not where anyone would look:

   1. A route handler calls `json::parse(req.body())`; invalid UTF-8 throws
      `parse_error.101`.
   2. The catch builds `{"error": "Handler exception: " + e.what()}` — and
      nlohmann puts **the offending bytes** into `e.what()` (`last read: ...`).
   3. `.dump()` on that throws `type_error.316`, *"invalid UTF-8 byte"*,
      because the string it is asked to serialise is the invalid one.
   4. That second exception escapes the catch, so `process_next_npu_request()`
      never runs, the NPU access lock taken **before** the handler is never
      released, and every later request hangs — valid ones included.

   **The error handler was broken by exactly the input it was reporting.**
   Reproduced with a single lone `0xE5` byte followed by a well-formed request
   that never returned. Error paths now dump with
   `json::error_handler_t::replace`, which substitutes U+FFFD instead of
   throwing: an error path must not be able to fail on the thing that made it
   run. After the fix the malformed request gets a clean JSON error and the
   next valid one answers in 0.5 s.

   It is not specific to this PR's backend — it is in the shared request path
   and affects `open_embedding` identically.

5. **`/v1/embeddings` ignored the `model` field**, and this is the worst of the
   five because the answer did not merely go wrong — it **asserted it was
   right**. One embedding model is loaded per server, and a request naming any
   other one was served by the loaded model with the reply labelled with the
   tag that had been *asked for*. So a client comparing `response.model`
   against its own request saw agreement. On a server holding bge-base:

   ```
   request model=bge-base:en-v1.5      -> response.model = bge-base:en-v1.5
   request model=not-a-model:v9        -> response.model = not-a-model:v9
   request model=gte-multilingual:base -> response.model = gte-multilingual:base
   ```

   All three vectors byte-identical. A RAG deployment embedding documents with
   one model and queries with another, against one `flm`, would retrieve
   nonsense with no signal anywhere. It refuses now, naming what *is* loaded, as
   `invalid_request_error` / `model_not_found`.

   Found by writing the client test below and expecting a refusal.

These are separable from the backend and can be split out if you would rather
review them alone.

---

## The two repository fixes this depended on (#3, merged)

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

Both are **split out into their own PR** — they have nothing to do with
embeddings and should go in on their own merit. They are at the base of this
branch, so they appear in this diff until that one lands.

---

## And a freshly built `flm.exe` now runs

It could not, and the way it failed sent you looking in the wrong place. Two
things were missing beside the executable:

* **The 22 engine DLLs** (`src/lib/<backend>/*.dll`). They are load-time
  imports, so without them the process dies at `0xC0000135` **before `main()`**,
  printing nothing at all. Windows then falls through to the next PATH entry —
  which on a machine with FastFlowLM installed is
  `C:\Program Files\flm\flm.exe`. So running
  `flm.exe serve ... --embeddingmodel x` from the build directory reported
  *"unrecognised option '--embeddingmodel'"*, **from a completely different
  binary than the one just built.** The symptom named a flag; the cause was a
  missing DLL two steps earlier with a silent fallback in between.
* **The xclbin tree.** `find_xclbin_path()` already looks for
  `<exe_dir>/xclbins` and calls it *"the portable development-tree location"* —
  but nothing ever put one there, so every model failed with *"no design set"*
  until `FLM_XCLBIN_PATH` was set by hand.

The DLLs are copied (into the build dir and `src/out`, both gitignored, so
nothing enters the repository); the xclbin tree is a **junction**, because it is
hundreds of megabytes and because a rebuilt kernel should be visible
immediately. `mklink /J` needs no privileges; if it fails the build warns and
names `FLM_XCLBIN_PATH` rather than erroring.

From a clean shell in `src/build`, with nothing set:

```
.\flm.exe serve llama3.2:1b --embed 1 --embeddingmodel bge-base:en-v1.5
```

Note the `.\`. Where the current directory is not searched for executables
(`NoDefaultCurrentDirectoryInExePath`), a bare `flm.exe` resolves to an
installed `flm` and never to your build. That is a Windows setting, not
something this repository can fix — but it is worth knowing, because it is how
the DLL problem above disguised itself.


## Testing it

Two harnesses, both in `utilities/`, and neither needs anything outside this
repository. The first is specific to this backend; the second is the
repository's existing suite, extended.

**`test_open_npue.ps1`** — the whole catalogue, end to end. It starts a server
per model (one embedding model per process: the geometry is process-wide and a
`ShapeLease` refuses a second), checks the OpenAI response shape, unit norms,
that a paraphrase is nearer than an unrelated sentence, that
`gte-multilingual` is actually multilingual, and that a malformed request does
not take the server down. **All six models pass in 1.7 minutes.**

Given `-Upstream <path-to-NpuEmbeddings>` it also compares every vector against
`npuembed --embed`. That mode is why the ODR bug was caught: every
self-contained check passed on the wrong vectors.

**`utilities/flm-test`** — the repository's own suite, which is where the
client-side endpoint checks for this backend now live. This started as a
separate script; it was folded in on review, because two suites testing the same
endpoint is how they drift apart.

```
flm serve llama3.2:1b --embed 1 --embeddingmodel bge-base:en-v1.5
flm-test --embedding --model bge-base:en-v1.5
```

Three changes went in, all of them in `flm_test/tasks.py`:

**E9 Model Identity, a new check.** A request naming a model the server cannot
have loaded must be **refused**, not answered; and an accepted request must
report the model that was asked for. This is the check that found fix 5 above,
and it is the one failure the other eight cannot see — a substituted model's
vector is correctly shaped, correctly normed, deterministic, batch-consistent
and semantically sensible, so E1–E7 all pass on it. E8 makes it *worse*: if the
model substituted in is the one the bundled reference was made from, E8 passes
too, and the whole suite reports success on an answer for a model nobody asked
for.

**E8 now SKIPs instead of failing** for a model its bundled vectors are not for.
The reference is `google/embeddinggemma-300m`; two models embed the same text
into different spaces by design, so a cosine between them carries no information
about either. Without this, every model in this PR failed E8 for a reason that
said nothing about them. `REFERENCE_MODELS` is the set to extend when reference
vectors for another model are added.

**The six tags are in `EMBED_MODELS`**, so the suite runs against them without
an explicit `--model` filter.

Also: **E2 now reports whether the draws were bit-identical** or merely inside
`STABILITY_THRESHOLD`. Same verdict either way — some backends are legitimately
non-exact — but the distinction belongs in the record. This backend is
bit-identical, and that matters: the encode is deterministic, so any difference
between draws is a per-row bug (lane aliasing, tier misindexing, shared
scratch). Upstream has hit two of those and both produced plausible vectors that
differed only between rows.

Unit tests for all of it are in `utilities/flm-test/tests/test_embedding_checks.py`
— 51 tests, no server required:

```
python tests/test_embedding_checks.py
```

The separation property the standalone script tested is E5, which the suite
already had.


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

### `clean_build.bat`

```
clean_build.bat [vcpkg-root]
```

At the repo root, for a plain `cmd` prompt. It loads the MSVC environment via
`vswhere`, **deletes `srcuild`**, configures, retries once, builds, and then
says where the binary is and whether the AIE design sets exist.

Deleting the build tree is the whole point rather than tidiness — see *A failed
configure poisons the tree* below.

### A fresh clone on Windows used to die before reaching any of this

Found by testing from an empty directory rather than an existing tree, which is
the only way this class of thing shows up:

```
CMake Error at third_party/tokenizers-cpp/sentencepiece/CMakeLists.txt:196:
  file failed to create symbolic link '.../sentencepiece/third_party/absl':
  A required privilege is not held by the client.
```

Windows grants `SeCreateSymbolicLinkPrivilege` only under Developer Mode or
elevation, and neither is a reasonable thing to require to build a program. The
message also points at a third-party `CMakeLists.txt`, so it reads as somebody
else's bug — and it is, but it is not fixable there, because that is an
upstream submodule.

It is avoidable here. sentencepiece guards the call with
`if(NOT EXISTS .../third_party/absl)`, and a **directory junction** satisfies
that guard while needing no privileges at all — `mklink /J`, the same trick
this PR already uses to put the xclbin tree beside the executable. `src/CMakeLists.txt`
creates it before `add_subdirectory(tokenizers-cpp)`.

**One honest limitation, stated in the code as well.** abseil-cpp is fetched by
`FetchContent` *inside* sentencepiece's own CMakeLists, so on a genuinely fresh
clone the junction target does not exist yet when this runs. That first
configure fetches abseil and then still fails; **the second one succeeds**, and
prints `sentencepiece: junction third_party/absl -> abseil-cpp/absl`. Making
the first pass work would mean cloning abseil here and copying sentencepiece's
`GIT_TAG` into this file — a duplicated version pin, which is a worse problem
than one repeated command. The configure says so when it is in that state
rather than leaving the reader to guess.

Verified by deleting the junction from a fresh clone and re-running: created,
exit 0.

### A failed configure poisons the tree, and that is the worse half

The junction fix above makes the *second* configure succeed. It is not enough
on its own, because of what the first failure leaves behind:

CMake writes `CMAKE_TOOLCHAIN_FILE` into the cache but **does not re-apply a
toolchain file to an existing cache**. So after a configure that died partway,
every later configure of that directory runs with `VCPKG_TOOLCHAIN` false — and
`src/CMakeLists.txt` then takes its *"bare self-hosted CI runner"* branch, which
hardcodes `C:/dev/boost_1_88_0` and links
`libboost_program_options-vc143-mt-x64-1_88` by raw name. That file does not
exist on a normal machine: vcpkg installs `boost_program_options-vc145-mt-x64-1_91.lib`,
shared rather than static.

The result is 332 files compiling for ten minutes and then

```
LINK : fatal error LNK1181: cannot open input file
       'libboost_program_options-vc143-mt-x64-1_88.lib'
```

— an error naming a Boost version nobody asked for, from a branch meant for a
CI runner, because of a symlink failure several minutes earlier. Nothing in the
message connects those.

**So a failed configure does not cost a retry, it silently changes which
dependencies the build uses.** `clean_build.bat` deletes the build tree
unconditionally, which is cheaper than explaining when to.

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
| a request naming a model that is not loaded | served, **labelled with the requested tag** | error naming what is loaded |
| a malformed request body | wedged the server permanently | clean JSON error, server survives |

---

## The kernels are source, and the binaries are gone

`src/xclbins/BERT-*/` is **no longer in the repository**. It is `.gitignore`d
and built from `npu_offload/gemm_rtp/`, which sits beside `npu_offload/matmul/`
and follows its convention: the design, the driver, and a README with **one
exact command per design family**.

| file | what it is |
|---|---|
| `gemm_pretiled.py` | the IRON design — whole-array pre-tiled GEMM, its ObjectFifo dataflow, the `mm.cc` invocation |
| `export_gemm_rtp.py` | the driver: every (shape, batch tier) stream against one xclbin |
| `npue.py` | the container format, for the B-tiling the packer must match |
| `toolchain_provenance.py` | writes `toolchain.json` beside the design |
| `families.json` | the five families and their flags — the only place those are written down |
| `build.ps1` | builds all five, skipping what is already built, then checks them |
| `check_design_sets.py` | checks the built sets against `families.json` |

plus three AIE kernel sources at `npu_offload/m5-eltwise/kernels/`, the path
`gemm_pretiled.py` computes for them: `narrow_f32_bf16.cc` (`--c-bf16`, so four
of the five families), `narrow_i32_bf16.cc` (`--int8`) and `gelu_poly.cc`
(`--epilogue gelu`).

`mm.cc` is deliberately **not** vendored: it is mlir-aie's own, taken from the
installed toolchain so it always matches the compiler that builds it.

### Deleting the binaries is what proved the source was not sufficient

The first version of this PR shipped the xclbins and claimed the generator
reproduced them — *"19 of 20 files byte-identical"*. That measurement was real
and the claim it supported was not, because **the reproduction was run from the
upstream tree**, where a file this repository did not have still existed. It
demonstrated that the generator is deterministic. It never tested whether this
repository could build anything, which is the claim that was made.

Deleting all five sets and building from an empty tree answered that in eleven
seconds:

```
FileNotFoundError: ExternalFunction 'narrow_3072_f32_bf16':
  source file not found: npu_offload/m5-eltwise/kernels/narrow_f32_bf16.cc
```

**No family could be built.** The three `.cc` sources above had never been
copied. What hid it is worth naming: `gemm_pretiled.py` computes that path
relative to itself, and the relative layout is identical in both trees — so the
code was correct, the file was absent, and nothing said so until someone
compiled.

### What the rebuild says now

All five families, built from an empty `src/xclbins/`, using only this
repository and the mlir-aie toolchain:

| family | files | byte-identical | `final.xclbin` delta |
|---|---:|---:|---|
| `BERT-h384-bfp16` | 20 | **19** | 82 / 127,454 |
| `BERT-h384-bf16` | 20 | **19** | 77 / 122,334 |
| `BERT-h768-bfp16` | 20 | **19** | 79 / 127,454 |
| `BERT-h768-gated-bfp16` | 20 | **19** | 82 / 127,454 |
| `BERT-h1024-bfp16` | 8 | **7** | 82 / 126,430 |

**88 of 96 files byte-identical.** The eight that differ are the five xclbins,
by **402 bytes of 631,126 — 0.064%** — and not scattered: 5 to 6 tight clusters
each, holding the binary UUID, the UUID again as hex in the metadata JSON, and
`"TimeStamp"`. The embedded AIE core ELFs are identical, which is the part that
matters and is what a scattered diff would have disproved.

About three minutes per family on a Ryzen AI 9 HX 370.

### And the rebuilt sets were then run

Reproduction is not correctness, so the catalogue harness was run against them
with `-Upstream`, which compares every component against `npuembed --embed`
from an independent NpuEmbeddings build:

```
all 6 models pass, and are bit-identical to the upstream binary
```

Per model: 384, 384, 768, 1024, 768, 768 components, **all exact**. The five
sets the fork's `flm` can reach are the five it just built -- there is no
upstream artifact directory anywhere in this tree -- so these are vectors from
design sets compiled here, matching a binary that was never told about them.

The line that matters most is `bge-small`:

```
[NPUE] loaded BAAI-bge-small-en-v1.5, hidden 384, seq 64, bf16 MMAC, C as fp32
       384/384 components exact -- BIT-IDENTICAL to the upstream binary
```

`C as fp32` is the corrected `BERT-h384-bf16`, built without `--c-bf16`. It is
a **new** artifact -- its predecessor in this PR was built with the wrong flag
-- and it lands byte for byte on the validated one.

### It also found two wrong commands, and both were the silent kind

Neither is detectable by building. Both produce a valid design that is not the
one that was validated.

1. **`BERT-h384-bf16` was documented with `--c-bf16`; the set that shipped has
   `c_dtype: f32`.** It is the one family without that flag. bge-small is also
   the one model held back from the bfp16 datapath — it failed the MTEB gate,
   bit-reproducibly — so following the README put precisely the conservative
   model on a narrower accumulator than it was validated for. The runtime reads
   `c_dtype` from `design.json` and adapts, so nothing crashes and nothing
   warns; the numbers just change. Fixed: no `--c-bf16` there, and the README
   now says the command differs from the other four in *two* places.
2. **`BERT-h1024-bfp16` was documented with four batch tiers and shipped with
   one.** This one turned out to be the README being right and the artifact
   being under-provisioned, which took a measurement to establish rather than a
   guess — see below.

### bge-large was shipping one batch tier, and it cost 18.3x

`use_tier()` picks the smallest tier that fits, so a design with only
`tiers: [128]` pads **every** request to batch 128 — 8192 rows through the
array where a four-tier design uses 256, with the host buffers sized from the
tier too, so bias, norm and attention run over the padding as well.

Both design sets built, identical in every other parameter, and swept:

| texts | `[128]` | `[4,16,32,128]` | ratio | |
|---:|---:|---:|---:|---|
| 1 | 1.650 s | 0.090 s | **18.3x** | 4-tier picks 4 |
| 4 | 1.630 s | 0.090 s | **18.1x** | picks 4 |
| 16 | 1.660 s | 0.250 s | **6.6x** | picks 16 |
| 32 | 1.620 s | 0.480 s | **3.4x** | picks 32 |
| 128 | 1.640 s | 1.630 s | **1.006x** | **control — both pick 128** |

Median of three runs after a discarded warm-up; encode wall clock, i.e.
end-to-end request latency and **not** an NPU kernel claim. The n=128 row is
the control that validates the method — both designs select tier 128 there, so
it must come out ~1.00 or nothing else in the table means anything. And the
one-tier design sitting flat at ~1.64 s whatever the request size is the
signature of padding.

**The vectors are bit-identical** at n = 1, 4, 32 and 128 — 0.000e+00 delta,
compared in float64 because a float32 cosine over 1024 terms drifts from 1 for
identical inputs. A tier is a padding choice, not an arithmetic one, so the
accuracy gates pass by construction: these are the bytes that already passed
them. That is what makes adopting it here safe rather than brave.

So this PR builds four tiers. **Upstream still ships one** — replacing a
validated artifact there means re-running the whole release sweep, which is a
release decision and not this PR's to take. It is filed as T66 in that
repository's register with this table in it.

Now that the sets are built rather than committed, **the README is the
artifact**, so `check_design_sets.py` compares `families.json` against the
`design.json` each produces — eleven fields per family, non-zero exit on any
disagreement. It catches both defects above.

```
$ python npu_offload/gemm_rtp/check_design_sets.py
ok       BERT-h1024-bfp16
ok       BERT-h384-bf16
ok       BERT-h384-bfp16
ok       BERT-h768-bfp16
ok       BERT-h768-gated-bfp16
```

### The cost, stated plainly

**A reviewer without mlir-aie and Peano can build `flm` but cannot run an
`open_npue` model.** That is a real regression in reviewability, taken
deliberately: the sets ship pre-built in the distributed package, and a binary
sitting in a repository beside the source that allegedly produces it is a claim
nobody checks — as this PR has now demonstrated about itself. `NpueEmbedding`'s
missing-design error names the README and the one command that fixes it.


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

`src/xclbins/BERT-*/` is **not in the repository**: it is `.gitignore`d and
built from `npu_offload/gemm_rtp/`, about three minutes per family. Each built
set carries a `toolchain.json` recording the mlir-aie version, the Peano
version and the git HEAD that produced it.

---

## Not done, and not pretended

* **The seventh model is not here.** `embed-gemma:300m` stays with
  `open_embedding`, deliberately: NpuEmbeddings' arch=1 path lives in its CLI
  rather than its library, so this backend cannot serve it.
* **`embed_batch()` exists and is unused.** `AutoEmbeddingModel::embed()` takes
  one text, so `handle_embeddings` loops — worth ~5.8× on a 16-text request,
  measured. Widening the base class is one line plus a default that loops, and
  it deserves its own PR.
* **`usage` in the response is still `{0, 0}`** — hardcoded in
  `handle_embeddings` for both backends, untouched here.
* **Two co-resident `hw_context` objects are a real hazard, and only half
  measured.** The test harness reproduced it by accident: it left one model's
  server running while starting the next model's reference process, and the
  reference **hung indefinitely** — no output, no progress, no error. Killing
  the server made it complete instantly. So two processes each holding a
  context on this NPU do not queue, they block. `flm serve <llm> --embed 1`
  holds the LLM's context and the engine's at once, which is the same shape,
  and it has **not** been exercised here because the LLM never loaded for want
  of its own xclbins.
* ~~**Linux is unverified.**~~ **Verified after this PR was opened**: a second
  machine built and ran `flm` on Fedora Rawhide (Framework 13 AI 340, no IDE).
  The engine's platform-independent subset also compiles there at C++17 and
  C++20, which is the check upstream runs on every change. The numbers in this
  PR are still one machine's; the *build* is now two.

[upstream]: https://github.com/vegardberget/NpuEmbeddings

<!-- NOTE before posting: check the [upstream] URL above. It was inferred from
     a git identity, not read off the remote, and provenance is the one section
     that has to be verifiable. -->

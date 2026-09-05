# `gemm_rtp` — the AIE designs behind `open_npue`

This is the source for `src/xclbins/BERT-*/`. Four IRON scripts and one command
per design family.

`open_npue` runs a whole encoder layer as **four GEMM dispatches over one
resident xclbin in one `hw_context`**. That xclbin carries several instruction
streams — one per (shape, batch tier) — and the runtime selects among them at
dispatch, so a request is right-sized rather than padded. These scripts build
it.

## The scripts

| file | what it is |
|---|---|
| `gemm_pretiled.py` | the IRON design: the whole-array pre-tiled GEMM, its ObjectFifo dataflow and the `mm.cc` kernel invocation |
| `export_gemm_rtp.py` | the driver — builds every (shape, tier) stream against one xclbin and writes the design set |
| `npue.py` | the container format, for `gemm_b_layout()` / `layout_hash()` and the B-tiling the packer must match |
| `toolchain_provenance.py` | writes `toolchain.json` beside the design |
| `families.json` | the five design families and their flags — the ONLY place those are written down |
| `build.ps1` | builds all five in order, skipping what is already built, then checks them |
| `check_design_sets.py` | checks the built sets against `families.json` |

And **three AIE kernel sources**, one directory over, at
`npu_offload/m5-eltwise/kernels/` — the path `gemm_pretiled.py` computes for
them:

| file | needed by |
|---|---|
| `narrow_f32_bf16.cc` | **`--c-bf16`, so all five families below** |
| `narrow_i32_bf16.cc` | `--int8` |
| `gelu_poly.cc` | `--epilogue gelu` |

`mm.cc`, the vectorised matmul kernel, is **not** among them: it is mlir-aie's
own, taken from `aie_kernels/aie2p/` in the installed toolchain so that it
always matches the compiler that builds it.

**Where the authoritative commands are.** These five build the sets THIS
repository serves. Upstream builds six (it has an EmbeddingGemma set this fork
does not use) and keeps them in `tools/export_shipped_designs.ps1`, checked by
`tools/check_design_sets.py` -- the same discipline, and the same file name,
as the one here.
The two command lists differ in destination and in that one extra set; the
flags per geometry are the same, and both repositories check their own list
against their own artifacts. If they ever disagree on a flag, upstream is the
source of truth.

They are a **synced copy**; upstream is
[NpuEmbeddings](https://github.com/vegardberget/NpuEmbeddings), MIT here and
Apache-2.0 there. Edit them upstream.

## Environment

They need mlir-aie and Peano — the same environment as `npu_offload/matmul/`.

```powershell
cd C:\dev\mlir-aie
. .\iron_env.ps1          # MUST be dot-sourced
```

Two traps that cost an hour each if you meet them cold:

* **`XILINX_XRT` must stay unset.** It poisons Windows builds
  (`iron_setup.py` says so). Use `XRT_ROOT`.
* **Set the device explicitly** — the design scripts do, but if you write your
  own: without `iron.set_current_device(from_name("npu2", n_cols=None))` the
  arch silently falls back to NPU1, bf16 `mac_dims` become `(4,8,4)` instead of
  `(4,8,8)`, and the bfp16 emulation becomes a **no-op**. No error.
  `iron.get_current_device()` still says NPU2.

## Building them

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1        # MUST be dot-sourced
cd <repo>; .\npu_offload\gemm_rtp\build.ps1
```

That is the whole thing. Five families in order, ~20 minutes, skipping any that
are already built. `-Force` rebuilds, `-Only <name>` does one, `-Dst <dir>`
builds elsewhere. It checks the IRON toolchain is on the path *before* spending
four minutes discovering it is not, and it ends by running
`check_design_sets.py`, so a green run means the sets exist **and** match the
flags they were supposed to be built with.

**There are deliberately no commands to copy in this file.** Every one of the
three ways to get them wrong has now cost somebody a session:

| what was pasted | what happened |
|---|---|
| `--out <dst>/BERT-...` | PowerShell rejects `<` as a reserved operator *during parsing* — `The '<' operator is reserved for future use`, naming neither the placeholder nor the substitution that was forgotten |
| `--out $dst\BERT-...` | PowerShell does **not** error on an undefined variable, it expands it to nothing — so this became `--out \BERT-...` and built to the **drive root**. Successfully. Four times. |
| two families at once | `purge()` deletes matching entries from the shared `~/.npu/cache` on content markers, and `qkv`/`attn_out` depend on neither `--gated-ffn` nor `--intermediate`, so the two hidden-768 families own identical markers for 8 of their 16 entries and each deletes the other's builds |

The third now refuses in under a second (`export_gemm_rtp.py` holds a lock);
the first two are gone because the commands are.

## The families, and what makes each one different

The flags live in **`families.json`**, which is the only place they are written
down: `build.ps1` builds from it and `check_design_sets.py` verifies the built
sets against it, so there is no second copy to drift. This table is prose about
that file, not a second source.

| family | serves | hidden / ffn | gated | datapath | `tile_n` |
|---|---|---|---|---|---|
| `BERT-h384-bfp16` | all-minilm:l6-v2 | 384 / 1536 | no | bfp16, C bf16 | 48 |
| `BERT-h384-bf16` | bge-small:en-v1.5 | 384 / 1536 | no | **plain bf16, C fp32** | 48 |
| `BERT-h768-bfp16` | bge-base:en-v1.5 | 768 / 3072 | no | bfp16, C bf16 | 48 |
| `BERT-h768-gated-bfp16` | nomic-embed-text:v1.5 **and** gte-multilingual:base | 768 / 3072 | **yes** | bfp16, C bf16 | 48 |
| `BERT-h1024-bfp16` | bge-large:en-v1.5 | 1024 / 4096 | no | bfp16, C bf16 | **32** |

**Every one of those columns is load-bearing.** A set is selected at load time
by `hidden`, `intermediate`, `gated_ffn` *and* the datapath, so a set built
with the wrong flags is not a slower design — it is one the wrong model loads,
or none does. Three of the rows are worth reading twice:

* **`BERT-h384-bf16` is the only family with neither `--emulate-bfp16` nor
  `--c-bf16`.** bge-small failed the bfp16 MTEB gate at −0.5010,
  bit-reproducibly, so it stays on the plain datapath — and its C stays fp32.
  This README once documented it *with* `--c-bf16`, which builds a design that
  loads happily and is not the one that passed the gates.
* **`BERT-h1024-bfp16` uses `tile_n` 32, not 48.** The design asserts
  `N % (tile_n * n_cols) == 0` and bge-large's N is in {1024, 3072, 4096}. 64
  divides them but needs 65,536 B of a 63 KB L1 budget.
* **`BERT-h768-bfp16` and `BERT-h768-gated-bfp16` share their `qkv` and
  `attn_out` shapes exactly** — neither depends on `--gated-ffn` or
  `--intermediate` — which is why they must never be built concurrently.

## Is the source really the source?

Checkable, and checked from an EMPTY `src/xclbins/` — which matters, because an
earlier version of this claim was measured in a tree that still had a file this
repository lacks. It showed the generator was deterministic; it never showed
that this repository could build anything.

All five families, rebuilt with the commands above:

| family | files | byte-identical | `final.xclbin` delta |
|---|---:|---:|---|
| `BERT-h384-bfp16` | 20 | 19 | 82 / 127,454 |
| `BERT-h384-bf16` | 20 | 19 | 77 / 122,334 |
| `BERT-h768-bfp16` | 20 | 19 | 79 / 127,454 |
| `BERT-h768-gated-bfp16` | 20 | 19 | 82 / 127,454 |
| `BERT-h1024-bfp16` | 8 | 7 | 82 / 126,430 |

**88 of 96 byte-identical.** Every instruction stream, every `design.json`,
every `toolchain.json`. The five xclbins differ by **402 bytes of 631,126 —
0.064%** — in 5 to 6 tight clusters each: the binary UUID, the same UUID as hex
in the metadata JSON, and `"TimeStamp"`. The embedded AIE core ELFs are
identical, which a scattered diff would have disproved.

The sets were then **run**, not just compared: `utilities/test_open_npue.ps1
-Upstream <NpuEmbeddings>` reports *"all 6 models pass, and are bit-identical
to the upstream binary"* -- every component of every model, against a binary
built from the other repository. Reproducing bytes and producing correct
vectors are different claims; this is the second one.

## bge-large and its batch tiers

One xclbin carries an instruction stream per (operation, batch tier), and
`use_tier()` picks the smallest tier that fits -- falling back to the largest
when nothing does. The upstream `bge-large` artifact ships `tiers: [128]`, so
**every** request is padded to batch 128: `rows = 128 x 64 = 8192` where a
four-tier design would use `4 x 64 = 256`. `chunk()` sizes the host buffers
from the tier too, so bias, norm and attention run over the padded rows as
well.

Measured here, both design sets identical in every other parameter:

| texts | `[128]` | `[4,16,32,128]` | ratio | |
|---:|---:|---:|---:|---|
| 1 | 1.650 s | 0.090 s | **18.3x** | 4-tier picks 4 |
| 4 | 1.630 s | 0.090 s | **18.1x** | picks 4 |
| 16 | 1.660 s | 0.250 s | **6.6x** | picks 16 |
| 32 | 1.620 s | 0.480 s | **3.4x** | picks 32 |
| 128 | 1.640 s | 1.630 s | **1.006x** | **control -- both pick 128** |

Median of three runs after a discarded warm-up. Encode wall clock: end-to-end
request latency, **not** an NPU kernel claim. The n=128 row is the control that
validates the method, and the one-tier design being flat at ~1.64 s whatever
the request size is the signature of the padding.

**The vectors are bit-identical** at n = 1, 4, 32 and 128 -- 0.000e+00 delta,
compared in float64. A tier is a padding choice, not an arithmetic one, so the
accuracy gates pass by construction: the bytes are the ones that already
passed them.

Hence four tiers here. Upstream still ships one, because replacing a shipped
artifact there means re-running the whole release sweep; it is filed as
[T66](https://github.com/vegardberget/NpuEmbeddings/blob/main/research/OPEN-THREADS.md).

## Checking the sets against their spec

```powershell
python check_design_sets.py
```

`build.ps1` runs this at the end, and it is worth running alone after editing
`families.json`. It reads that file, derives the `design.json` each family must
produce, and compares eleven fields per set; non-zero exit on any disagreement.

It used to parse this README, which was a *transcript* of the commands rather
than the commands themselves — the same prose-versus-artifact gap one level up,
and the gap both shipped defects came through. Verified in both directions
rather than assumed: give `BERT-h384-bf16` a `--c-bf16` in `families.json` and
it fails with *"c_dtype: families.json says 'bf16', design.json says 'f32'"*.

## What is not here yet

`mm.cc` — the vectorised AIE kernel the design invokes — comes from mlir-aie's
own `aie_kernels/aie2p/`, unmodified. It is not vendored because it is not
ours to vendor and because taking it from the toolchain guarantees it matches
the version that compiles it.

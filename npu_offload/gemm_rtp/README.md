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

## Rebuilding a design family

**Every flag below is load-bearing.** A design set is selected at load time by
`hidden`, `intermediate`, `gated_ffn` **and the datapath** — the runtime refuses
a mismatched pair rather than reading it as garbage — so a set built with the
wrong flags is not a slower design, it is one no model will load.

The flag that is easiest to forget is `--c-bf16`. Omitting it builds a design
that is correct, complete, and identical in every other respect; it simply
narrows C to fp32 instead of bf16, and no model in the catalogue will accept it.

```powershell
# hidden 384, plain FFN, bfp16  ->  all-minilm:l6-v2
python export_gemm_rtp.py --hidden 384 --intermediate 1536 --qkv-n 1152 `
    --emulate-bfp16 --c-bf16 -n 48 --batches 4,16,32,128 `
    --tg-depth 2 --tb-rows 4 --out <dst>/BERT-h384-bfp16

# hidden 384, plain FFN, PLAIN bf16  ->  bge-small:en-v1.5
#   bge-small is the one model that stayed on the unemulated datapath: it
#   failed upstream's MTEB gate on bfp16, bit-reproducibly. It is ALSO the one
#   family with no --c-bf16: C stays fp32. Note that BOTH flags are missing,
#   and that this command therefore differs from the other four in two places,
#   not one.
python export_gemm_rtp.py --hidden 384 --intermediate 1536 --qkv-n 1152 `
    -n 48 --batches 4,16,32,128 `
    --tg-depth 2 --tb-rows 4 --out <dst>/BERT-h384-bf16

# hidden 768, plain FFN, bfp16  ->  bge-base:en-v1.5
python export_gemm_rtp.py --hidden 768 --intermediate 3072 --qkv-n 2304 `
    --emulate-bfp16 --c-bf16 -n 48 --batches 4,16,32,128 `
    --tg-depth 2 --tb-rows 4 --out <dst>/BERT-h768-bfp16

# hidden 768, GATED FFN, bfp16  ->  nomic-embed-text:v1.5 AND gte-multilingual:base
python export_gemm_rtp.py --hidden 768 --intermediate 3072 --qkv-n 2304 `
    --gated-ffn --emulate-bfp16 --c-bf16 -n 48 --batches 4,16,32,128 `
    --tg-depth 2 --tb-rows 4 --out <dst>/BERT-h768-gated-bfp16

# hidden 1024, plain FFN, bfp16, TILE 32  ->  bge-large:en-v1.5
#   -n 32, not 48: the design asserts N % (tile_n * n_cols) == 0 and
#   bge-large's N is in {1024, 3072, 4096}, so 48 is illegal. 64 divides them
#   but needs 65,536 B of a 63 KB L1 budget, so 32 it is.
#   --batches 128, NOT 4,16,32,128: this is the one family that ships a
#   single batch tier, matching the validated upstream artifact. The four-tier
#   version builds cleanly and is very likely better -- use_tier() rounds up,
#   so today every short bge-large request is padded to batch 128 -- but it has
#   not been through the accuracy gates, and this README builds what was
#   measured, not what ought to work.
python export_gemm_rtp.py --hidden 1024 --intermediate 4096 --qkv-n 3072 `
    --emulate-bfp16 --c-bf16 -n 32 --batches 128 `
    --tg-depth 2 --tb-rows 4 --out <dst>/BERT-h1024-bfp16
```

About three minutes per family on a Ryzen AI 9 HX 370.

`--tg-depth 2` software-pipelines the runtime sequence's task groups and is
worth 1.034–1.141× of array time, bit-identical. **`--tg-depth 3` compiles
clean and then times out on hardware**, reproducibly — 2 is the maximum that
runs.

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

## Check the README against what it builds

    python check_readme.py

The design sets are not in git, so this README is not documentation about the
artifacts — it **is** the artifacts. A wrong flag here does not fail to build,
it builds something else. `check_readme.py` derives the `design.json` each
command must produce and compares, eleven fields per family, non-zero exit on
disagreement.

It exists because two commands here were wrong, and neither was visible from a
successful build: `BERT-h384-bf16` was documented with `--c-bf16` against a
shipped set with `c_dtype: f32` (putting the one model held back from the
aggressive datapath on a narrower accumulator than it was validated for), and
`BERT-h1024-bfp16` was documented with four batch tiers against a set with one.

## What is not here yet

`mm.cc` — the vectorised AIE kernel the design invokes — comes from mlir-aie's
own `aie_kernels/aie2p/`, unmodified. It is not vendored because it is not
ours to vendor and because taking it from the toolchain guarantees it matches
the version that compiles it.

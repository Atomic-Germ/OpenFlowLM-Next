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
#   failed upstream's MTEB gate on bfp16, bit-reproducibly. Note the missing
#   --emulate-bfp16.
python export_gemm_rtp.py --hidden 384 --intermediate 1536 --qkv-n 1152 `
    --c-bf16 -n 48 --batches 4,16,32,128 `
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
python export_gemm_rtp.py --hidden 1024 --intermediate 4096 --qkv-n 3072 `
    --emulate-bfp16 --c-bf16 -n 32 --batches 4,16,32,128 `
    --tg-depth 2 --tb-rows 4 --out <dst>/BERT-h1024-bfp16
```

About three minutes per family on a Ryzen AI 9 HX 370.

`--tg-depth 2` software-pipelines the runtime sequence's task groups and is
worth 1.034–1.141× of array time, bit-identical. **`--tg-depth 3` compiles
clean and then times out on hardware**, reproducibly — 2 is the maximum that
runs.

## Is the source really the source?

Yes, and it is checkable rather than asserted. Rebuilding
`BERT-h768-gated-bfp16` with the command above and comparing against the set
shipped in `src/xclbins/`:

**19 of 20 files byte-identical** — all sixteen instruction streams,
`design.json` and `toolchain.json`. Only `final.xclbin` differs, by **80 bytes
of 127,454 (0.06%) in 9 short runs**, all in the header and metadata regions:
the UUID and build stamps that every xclbin link writes fresh. The design
itself is reproduced exactly.

That is the check worth repeating after any change here, and it is why
`design.json` records the parameters: a set whose metadata does not match its
streams is the failure this comparison catches.

## What is not here yet

`mm.cc` — the vectorised AIE kernel the design invokes — comes from mlir-aie's
own `aie_kernels/aie2p/`, unmodified. It is not vendored because it is not
ours to vendor and because taking it from the toolchain guarantees it matches
the version that compiles it.

# dit_conv: 3×3 / 1×1 convolution as implicit GEMM, windowed in the memtile

This is FLUX.2 [klein]'s VAE decoder conv on XDNA2. There is one xclbin per tap count
(`DC_TAPS` = 9 or 1). Every conv of the model is an instruction stream over it.

## Why a new design, not `dit_gemm` streams

- **Output width.** `dit_gemm` spreads N over 8 core columns of 128 (N % 1024). The
  VAE's convs have 128-512 output channels, so at best half the array would work.
- **DDR traffic.** Windowing through shim access patterns reads every activation 9
  times from DDR. With 128 output channels that is ~117 GB/s at 15 TFLOPS, against
  ~55 GB/s available.

## Dataflow (per column, 8 independent columns)

The column's 4 cores take:
- 4 consecutive output rows;
- the same 128 pixels;
- the same 128 output channels.

Each core keeps its 128 px × 128 cout C tile in L1 for the whole K walk,
`(cin chunk of 64, ky, kx)`. The microkernel is `dit_gemm`'s (`../dit_gemm/mm_dit.cc`):
- A is bf16, converted to bfp16 in-core;
- B is bfp16ebs8;
- C is re-rounded to bf16 every 64 of K.

**A, the activations**, runs on explicit DMA: IRON `TileDma` + `Lock` + `Flow`. An
ObjectFifo link cannot carry it, because its MM2S transfer length is always its L2
object's size (`AIEObjectFifoStatefulTransform.cpp:1435-1510`), and the window re-reads
each band element up to 9 times.

- **Shim → memtile S2MM: a band.**
  - A band is 6 input rows × 130 px × one 64-channel chunk, 97.5 KB, ping-pong.
  - It is one 4-D shim BD per tile: `[cin chunks][6][130][64]`.
  - The S2MM stores it chunk-major, `[8 ch-block][6 rows][130 px][8 ch]`.
- **Memtile MM2S → core r: one channel per core.**
  - Each ky is its own BD, at row offset `r + ky`, reading
    `[kx 3][quarter 4][ch-block 8][32 px × 8 ch]`.
  - That is the 9 taps' 32 × 64 A quarters, in the column-block-major order the core's
    S2MM turns into 8×8 blocks (`dit_gemm`'s `a_l2l1` pattern).
  - The chunk-major band is what makes each read 4-D. Row-major would need 6 dims.
- **Locks.** A band's full lock is released with value 12 (4 readers × 3 ky BDs), and each
  BD acquires 1. The S2MM waits for all 12 to hand the band back.
- **Core S2MM**: 2 × 4 KB A buffers, with counting locks (init 2 / 0).

**B, the weights**: an ObjectFifo shim → memtile → the column's 4 cores (broadcast). Each
tile's stream:
- starts with one **bias object**, whose first 256 bytes are the 128 biases in raw bf16;
  `conv.cc` starts C as that row instead of zero;
- then holds the `pack_b` tiles of the tile's 128 output channels.

**C**: an ObjectFifo join of the 4 cores at the memtile, and one shim drain per tile into
the output's layout.

**Budgets per memtile:**
- **Channels**: 6 S2MM (A, B, 4 × C) and 6 MM2S (4 × A, B, C). That is the whole budget.
- **BDs**: aie2p memtiles give channels 0/2/4 BDs 0-23 and channels 1/3/5 BDs 24-47. The C
  join's output takes 8, so the four readers sit on MM2S 0/2/4/1 (`READER_CH`): even
  24/24, odd 22/24.
- **L2**: 2 × 97.5 KB band + 2 × 9 KB B + 2 × 128 KB C = 469 KB of 512.

## Layout

Activations are NHWC bf16 in a zero-bordered buffer:
- (H+2) rows of `pitch` pixels;
- the image at row 1, column 1;
- producers write the interior only.

Grids narrower than 128 (W = 64 or 32, i.e. the 512 px image's latent stages):
- A core still computes 128 px, of which the first W are real.
- The drain sends the other 128/W − 1 parts past the output row's right border, Wo + 1
  apart. Such an output needs pitch ≥ (128/W)(Wo + 2), which is the default.

The spec is in `dit_conv.py`'s docstring. Its main pieces:
- `x`/`y` views are `{off, pitch, border}`. `border` 0 marks a plain [H·W, C] input and is
  allowed for 1×1 only.
- `up` makes 4 output phases, written to (2y+py, 2x+px):
  - with 3×3: a nearest-2× upsample then a conv, the upsample folded into the weights
    (`conv_pack.phase_weights`);
  - with 1×1: any per-phase map (`pack_conv_phases`). The VAE's latent_in uses it:
    unpatchify + BN + post_quant_conv.

## Files

- `dit_conv.py`: the design, the spec (`resolve_spec`) and the tile order (`tiles`).
- `conv.cc`: the bias prologue. The matmul is `../dit_gemm/mm_dit.cc`.
- `conv_pack.py`: `pack_conv`, `pack_conv_phases` and `unpack_conv`. These are shared by
  the tests and `open_kernels/vae_decoder.py`.
- `make_test.py`: spec → build → run on the NPU (pyxrt) → check against fp64.
  - The first and last rows and 4 middle rows are checked exactly.
  - Every pixel is checked through a random output-channel projection.
  - The zero border must survive.

```
python make_test.py --H 32 --W 128 --Cin 64 --Cout 128 --out <t>          # writes spec.json
DC_TAPS=9 DC_SPEC=<spec.json> python ../../build_design.py dit_conv.py <b>
python make_test.py --H 32 --W 128 --Cin 64 --Cout 128 --out <t> --build <b>
```

(`C:\dev\fa-work\conv_test.sh` batches this.)

## Results (2026-09-26, HX 370, turbo)

Gate: rel_fro < 3e-2 against the kernel's own weights. All pass with the border intact:

| shape (H×W, Cin→Cout) | rel_fro | time | TFLOPS |
|---|---:|---:|---:|
| 128² 512→512 | 1.19e-2 | 6.58 ms | 11.7 |
| 256² 512→512 | 1.19e-2 | 24.7 ms | 12.5 |
| 512² 256→256 | 0.97e-2 | 17.9 ms | 17.3 |
| 1024² 128→128 | 0.83e-2 | 18.1 ms | 17.1 |
| 512² 512→256 | 1.19e-2 | 50.8 ms | 12.2 |
| upsample 256²→512², 512 | 0.93e-2 | 85.2 ms | 14.5 |
| 64² 512→512 (W = 64: half the pixels are garbage) | 1.20e-2 | 3.31 ms | 5.8 |
| 1×1 512² 512→256 | 0.75e-2 | 25.4 ms | **2.7** |
| conv_out 1024² 128→3 (padded to 128) | 0.83e-2 | 18.3 ms | (0.4 real) |
| latent_in, 1×1 4-phase, W = 32 and 64, plain input | 0.69e-2 | 0.4-0.5 ms | — |

The 1024² image's 3×3 convs add up to ~0.75 s.

Open items:
- **1×1 is slow.** With 8 K steps per tile, the per-tile A band (128 B runs every
  Cin·2 B) and the drain are not hidden.
- **conv_out** computes 125 zero channels.
- **Upsample convs** do the naive FLOPs. A 2×2-tap build would save 55%.

## Traps

1. **Memtile BD pools.** Even and odd channels have separate 24-BD pools; count before
   building. `open_kernels/emit_mlir.py --lower` prints the ObjectFifo-lowered use per
   channel, though its count for explicit TileDma chains is only per head BD.
2. **ObjectFifo channel auto-assignment** sees explicit `TileDma` channels and
   `shim_dma_allocation`s (from a `Flow`'s `shim_symbol`), so the mix is safe. Place
   every tile explicitly anyway.
3. **Raw shim tasks** (`shim_dma_single_bd_task` + `dma_start_task`) inside an IRON
   sequence must be freed. They are appended to the TaskGroup's free actions
   (`tg._actions`).
4. **A shim BD's outermost dimension is ≤ 64.** The A fill's outer dim is the cin-chunk
   count (≤ 8).

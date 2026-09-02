# NPU2 BF16 Matmul Bring-Up

This directory contains the standalone mlir-aie proof of concept for replacing
the open embedding engine's CPU dense projections with NPU2 BF16 matmul.

## Designs

- `matmul_single_core.py`: tiled 64x64x64 reference design on one AIE core.
- `matmul_whole_array.py`: 4-row by N-column design adapted from the
  commit-matched mlir-aie v1.3.4 whole-array example.
- `host.cpp`: variable-shape XRT runner and FP32 reference checker.

The vectorized `mm.cc` kernel consumes blocked A/B operands and produces blocked
C. The IRON ObjectFifo DMA transforms pack and unpack those layouts; host buffers
remain ordinary row-major matrices.

The output BO must be initialized and synchronized to the device before launch.
Without `bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE)`, unwritten cache lines produce
nondeterministic partial results.

## Build And Run

Compile the 16-core Q/output projection shape:

```bash
python matmul_whole_array.py \
  -M 512 -K 768 -N 768 --tile-n 64 --n-aie-cols 4 \
  --xclbin-path build/whole_array_bf16_512x768x768_4col.xclbin \
  --insts-path build/whole_array_bf16_512x768x768_4col_insts.bin \
  --dev npu2
```

Build and run the host:

```bash
bash build_host.sh
MM_M=512 MM_K=768 MM_N=768 ./matmul.exe \
  -x build/whole_array_bf16_512x768x768_4col.xclbin \
  -k MLIR_AIE \
  -i build/whole_array_bf16_512x768x768_4col_insts.bin
```

Set `XILINX_XRT=/opt/xilinx/xrt` if it is not already configured.

## Hardware Results

Measured on NPU2 at PCI `0000:c2:00.1`. Times cover XRT kernel submission and
completion, not FP32/BF16 conversion or CPU reference generation.

| Shape MxKxN | Topology | Tile N | Kernel time | Max abs error |
| --- | --- | ---: | ---: | ---: |
| 64x768x768 | 1 core | 64 | 2.00 ms | 0.00293 |
| 512x768x768 | 1 core | 64 | 7.74 ms | 0.00293 |
| 512x768x768 | 16 cores | 64 | 2.62 ms | 0.00293 |
| 512x768x768 | 32 cores | 32 | 3.57 ms | 0.00293 |
| 512x768x768 | 32 cores | 48 | 3.53 ms | 0.00293 |
| 512x768x1152 | 16 cores | 48 | 2.89 ms | 0.00195 |
| 512x768x1152 | 32 cores | 48 | 3.62 ms | 0.00195 |
| 512x1152x768 | 16 cores | 64 | 2.90 ms | 0.00366 |

The 16-core topology wins for these shapes because narrower 32-core output
tiles increase DMA and per-tile overhead.

## Engine Integration

The initial integration boundary is `open_embedding::Engine::matmul_t()`:

1. Convert each static FP32 projection weight from stored `[N,K]` form to BF16
   row-major `[K,N]` once at load time.
2. Convert and pad each FP32 activation from `[T,K]` to BF16 `[M,K]`, with M
   rounded to a compiled transfer size (initially 512).
3. Dispatch the shape-specific xclbin/instruction pair and convert the first T
   output rows back to FP32.
4. Keep norms, RoPE, softmax, residuals, pooling, and the contrastive head on CPU
   until projection correctness passes the E1-E8 oracle suite.

The first artifact set covers transformer Q/output (`768x768`), MLP gate/up
(`768x1152`), and MLP down (`1152x768`) projections at M=512.

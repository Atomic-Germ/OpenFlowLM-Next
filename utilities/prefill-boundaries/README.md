# Host prefill boundaries

Run the production CPU DeltaNet and attention stages against an independent
FP64 NumPy reference at Qwen3.8-27B dimensions. The C ABI bridge calls
`src/open_qwen36/block_host.cpp` directly, with FP32 convolution history and
paired-BF16 KV. Synthetic projection outputs and weights condition the check;
this does not load model weights or execute NPU projections.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE \
  ironvenv/bin/python utilities/prefill-boundaries/check.py \
  --out Models/qwen38-27b/prefill-host-boundaries/wide
```

Requires Linux x86_64 with AVX2, g++, OpenMP and NumPy. `--small` uses smaller
shapes with the same 3:1 value/key head ratio; `--seed` changes the fixture.
The checker always recompiles the library, using `-ffp-contract=off` to match
the production host arithmetic. `build.json` records the compiler, commands,
compiler-emitted local dependencies and library hash. The report pins source,
specification, build record and all input/output artifacts. A failed rerun
removes a previous successful report before building and writes a failed report.

Each prompt length (1, 2, 255, 256, 257) starts with nonzero conv/recurrent
history and a three-row KV prefix, then appends one continuation token.
Prompt chunks are at most 256 tokens. Padded calls allocate 256 rows with NaN
inputs beyond the real count; output padding and recurrent padding must be zero.
The original cache prefix and unwritten tail must remain exact after every call.

All outputs, conv history, recurrent state and raw cache must match one-token
native calls byte for byte. In addition, DeltaNet/attention outputs, final
recurrent state and reconstructed K/V must each pass an independent FP64 gate:
`max(abs(actual-reference)) / max(max(abs(reference)), 1e-12) < 1e-3`.
FP32 conv history must match the oracle exactly. Native state is never fed into
the oracle. Acceptance requires every gate, not merely native/native agreement.

On 2026-10-18, all five small and five wide cases passed. Wide worst errors:
DeltaNet **3.9750e-7**, attention **8.8449e-7**, recurrent state **2.5863e-7**,
reconstructed cache **6.1067e-6**. Native partition parity and conv history
were exact in every case. These are synthetic CPU-stage results, not full-model
prefill, hybrid/NPU route parity, long-context or performance acceptance.

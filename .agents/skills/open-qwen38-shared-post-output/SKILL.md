---
name: open-qwen38-shared-post-output
description: Build shared BF16/FP32 DeltaNet post-output specializations and compare complete lx binaries against audited causal controls before production ABI promotion.
---

# Shared post-output preparation

Use after open-qwen38-fp32-post-output. The shared post.cc specializes output
storage with DN_POST_FP32. Linear.POST_F32 selects its compile flag, output
argument, FIFO element, dense post DMA width and direct FP32 output preparation.
Production recipes keep POST_F32=false until the new scratch/manifest ABI is
promoted and validated. Do not change their geometry silently.

The private builder selects the common flag in both lx and xlayer. Only its
scratch placement and six input elements remain private: append 24576 bytes
at 393216, total 417792; state stays 3317760 bytes. No arithmetic source edits
are applied by the builder. The control keeps the original scratch geometry.

Run all host tests before building; source-generating tests and compilation
must not overlap. Use fresh output directories and retain failed attempts.

```bash
source utilities/env.sh
export OFLM_KEEP_FAILED=1
ironvenv/bin/python utilities/deltanet-post-fp32/build.py \
  --kernels Models/qwen38-27b/prefill-production/kernels-model-v1 \
  --mode control --out Models/qwen38-27b/deltanet-post-fp32/shared-control-v1
ironvenv/bin/python utilities/deltanet-post-fp32/build.py \
  --kernels Models/qwen38-27b/prefill-production/kernels-model-v1 \
  --mode fp32 --out Models/qwen38-27b/deltanet-post-fp32/shared-fp32-v1
```

The builder validates the current bundle semantics, compiler dependencies and
all 11 core memory budgets, with the existing 6144-byte stack. Require exact
insts.bin and whole-XCLBIN equality apart from the exporter's documented build
stamp fields against control-v2 / fp32-v1. Bind those references to their
passing control-run-v1 / fp32-run-v1 report and audit hashes; do not re-run
historical source validators against changed current source. Do not rewrite old
metadata. Binary equivalence is not a fresh numerical or performance run.

Production promotion still requires recipe-owned scratch/manifest semantics,
full-depth short and long-prefix checks, and the explicit performance gate.

## Binary identity and measured build checkpoint

Use `utilities/deltanet-post-fp32/identity.py` with --build shared-control-v1,
--reference control-v2, --reference-run control-run-v1, and a fresh --out JSON.
Repeat for shared-fp32-v1 / fp32-v1 / fp32-run-v1. Paths are relative to
Models/qwen38-27b/deltanet-post-fp32 here. See the utility README for full commands.
It checks current build/resource provenance and binds the reference build to
its passing historical report/audit and all 729 archived buffers. Historical
source hashes are not re-run against current code; artifacts have no exemption.

On 2026-10-26 both comparisons pass. Instructions are byte-exact; whole XCLBINs
differ only in 79 / 77 stamp bytes. Both 11-core builds fit: maximum text
15936 / 16000, maximum assigned L1 buffer end 62336, stacks 6144. Each records
849 compiler dependency entries. Toolchain: mlir-aie 1.4.2 and
llvm-aie 21.0.0.2026080301+c9c5ecb7. Full suite: 1878 passed, 47 skipped.

SHA256 (control / FP32):
- insts.bin: 061178d182c8511bb9d9e9cad0f7671b10108d60f6939912285d62fb8f0944e5 /
  498ae97db19f69cd86a7de62ac7baa33669a66a05fc617403f6838986253bc70
- final.xclbin: af8ebba8921f7c806e0cb1dd2e6785f9647dd4d98fbe65863be9a126a662f28e /
  6265a58a430f5e4bee7ab0d35040cafbea662c698f2a46358148423f64f6fe6b

This checkpoint does not load the full model or rerun numerical propagation.
Production post-output remains BF16; recipe-owned FP32 scratch/manifest export
and full-depth/long-prefix numerical validation are still required.

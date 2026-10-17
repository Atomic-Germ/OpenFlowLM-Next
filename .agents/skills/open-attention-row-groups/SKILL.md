---
name: open-attention-row-groups
description: Diagnose and validate attention GEMM builds with odd row-block counts, including Phi-4 mini ag_s256 and ag_pv256 tiler failures.
---

# Attention GEMM row groups

Use `npu_offload_pipeline` for the general toolchain setup. This note covers
`gemm_pretiled.py` C-drain grouping and the dense recipe's attention kernels.

## Failure and invariant

Phi-4 mini has 24 query heads and 8 KV heads. With T=256, attention GEMM
has M=768, or three row blocks of m*4=256. The default two-block C-drain
group fails `TensorTiler2D.step_tiler` with `allow_partial=False` in dimension 0.
This is a shape/grouping issue; a Python 3.14 traceback alone does not establish
an interpreter incompatibility.

Choose the largest divisor of the row-block count within the requested group
limit. Preserve the DMA-stride guard that forces one row block above 2**20.
Both C taps and the runtime fill/drain schedule must use the resulting
`tb_n_rows`. Do not merely enable partial C taps without checking the matching
A/B transfers and ping-pong tail. The runtime guards the final half with
`c_index >= len(C_tiles)`.

The dense recipe's `KERNEL_SOURCES` must include `attn_gemm.py`,
`gemm_pretiled.py` and `npue.py`, so changes invalidate exported build keys.

## Targeted compile checks

From the repository root, activate `ironvenv/bin/activate` and, if required,
`/opt/xilinx/xrt/setup.sh`. Use a separate output directory to avoid replacing
production exports with partial manifests:

```bash
AG_M=768 AG_K=128 AG_N=256 AG_COLS=8 python open_kernels/build_design.py \
  open_kernels/designs/attn_block/attn_gemm.py /tmp/oflm-phi4-ag-s256-check
AG_M=768 AG_K=256 AG_N=128 AG_COLS=4 python open_kernels/build_design.py \
  open_kernels/designs/attn_block/attn_gemm.py /tmp/oflm-phi4-ag-pv256-check
AG_M=512 AG_K=128 AG_N=256 AG_COLS=8 python open_kernels/build_design.py \
  open_kernels/designs/attn_block/attn_gemm.py /tmp/oflm-ag-even-check
```

All three produced `BUILD_OK`, xclbin, insts.bin and insts.elf on 2026-10-08
with the local Python 3.14 toolchain. The Phi-4 dimensions were checked against
`for_spec(spec).builds(spec)` for `phi4-mini-4b.json`; cache source inclusion
was also checked. These are compile checks, not numerical NPU validation or
a full model export/package build.

Docker packaging and container toolchain isolation are separate changes;
the targeted compile checks above do not depend on them.

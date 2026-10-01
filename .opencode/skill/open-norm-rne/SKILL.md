---
name: open-norm-rne
description: Trace real wide RMSNorm statistics and validate opt-in integer-lane FP32 products and compensated sums against unchanged model references.
---

# Real RMSNorm rounding

Read `specs/open-engine/plans/qwen38-norm-precision.md` for acceptance and
remaining limitations. The source frame is cold0/layer4 xn from
`open_kernels/designs/wide_deltanet/build_attention_boundary/carry/full`.
Channel786 has one local BF16 error; the source's conditional and model
references agree. Do not regenerate full references or weaken their gates.

`LN_NORM_RNE=1` requires streamed LN, `LN_STREAM_COMPENSATED=1` and
`LN_RESIDUAL_RNE=1`. It uses the integer-lane product in `fp32_mul_rne.h`,
exact FP32 add/sub for Kahan statistics and a compensated final reduction.
Scaling uses exact FP32 products; scalar mean and reciprocal square root
keep their existing implementation. It does not promise exact FP64 RMSNorm.

```bash
base=open_kernels/designs/wide_deltanet/build_norm_boundary
LN_N=5120 LN_STREAM_COMPENSATED=1 LN_RESIDUAL_RNE=1 LN_NORM_RNE=1 \
  PATH=/opt/xilinx/xrt/bin:$PATH ironvenv/bin/python open_kernels/build_design.py \
  open_kernels/designs/ln/ln.py "$base/rne"
ironvenv/bin/python utilities/test-wide-ln.py prepare --build-dir "$base/rne" --exact-residual
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel "$base/rne/ln.cfg"
ironvenv/bin/python utilities/test-wide-ln.py compare --build-dir "$base/rne"
```

Use fresh output directories for replays. `utilities/diagnose-wide-ln.py prepare
--source <source> --kernel <validated LN> --out <fresh>` defaults to layer4 xn.
Run `replay.cfg`, then `compare --out <fresh>`. The strict conditional BF16
gate must fail baseline `build_residual_precision/ln_rne` at index786 and pass
the new kernel. Model differences and repeated-run equality are separate.
`--tag` and `--field xn|xm` select another frame without changing its reference.

For internal statistics, build with `LN_TRACE_STATS=1 LN_TRACE_INDEX=786`
in addition to the desired arithmetic flags, then prepare with `--trace
--index 786`. Index must match the build. This diagnostic kernel replaces xn
with FP32 trace data:32 sum lanes,32 correction lanes, then total, mean,
inverse root, weighted input and pre-BF16 output. It cannot run model inference.
Trace comparisons always retain `passed: false` and exit1; inspect their
statistics and repeat/residual results rather than interpreting this as an
acceptance run. The ordinary LN primitive gate rejects this trace artifact.

Keep `ln_add_rne` and `ln_mul_rne` as shared noinline COMDAT functions.
Making them static duplicated both routines across accumulator/finish workers
and overflowed program memory. The passing kernel has11536 B text and57600 B
allocated data plus reserved stack; DMA instructions are unchanged. In a short
18-input sample it took median0.863 ms versus0.213 ms for the previous mode.

Rebuild without `LN_NORM_RNE` into a separate directory and compare all36
y/xn captures from the18 strict cases with `build_residual_precision/ln_rne`.
Run `ironvenv/bin/python -m pytest specs/open-engine/tests -q`. The host test
compiles the actual integer multiplier and checks300000 random pairs and289
boundary pairs, including subnormals, signed zero, overflow and nonfinite values.
This is test infrastructure, never a CPU inference fallback.

Full replay uses original `build_full_model` fixtures and:
- output projection `build_ffn_boundary/carry/projection_k6144`;
- FFN `build_down_precision/carry/ffn`;
- attention `build_model_precision/attention`;
- attention projection `build_attention_boundary/carry/projection_k5120`;
- LN `build_norm_boundary/rne`.

Pass these to `utilities/replay-wide-model.py`, run `decode.cfg`, then
`test-wide-full-model.py compare` and `diagnose-wide-model-rounding.py`.
Keep all thresholds, input tokens, weights and references unchanged.

Recorded full result:18891/18896 slice and4121/4122 decode checks pass; six
numerical failures remain. All3564 calls, tokens and reset pass. Local norm
differences fall26 ->9 and residual sums stay exact. The first ten norm
boundaries are exact; next layer5 xn has five propagated differences. A layer4
FFN diagnostic finds one BF16 h mismatch at14949 despite exact xm. Trace
up/gate and activation math there next; do not infer full support from this
local norm fix or the matching token sequence.

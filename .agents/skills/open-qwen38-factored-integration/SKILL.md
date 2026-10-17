---
name: open-qwen38-factored-integration
description: Build the production compensated factored DeltaNet export and validate all 64 layers with causal head gates, independent references and streamed weights.
---

# Production compensated DeltaNet recurrence

Use after the isolated [full-layer replay](../open-qwen38-factored-full-layer/SKILL.md).
The recipe selects `dnx_factored.h` for the validated dense H5120, FF17408,
16 key / 48 value heads, head dim 128, four-tap, split norm/down geometry,
with Q4_1 linear, linear-output, attention and FFN projections. Other geometries
and Q8/mixed-role configurations retain `dnx.h`. Activation/state layouts,
weight packing, FIFOs, scheduling and 6144-byte stacks are unchanged.

The manifest declares `linear_recurrence: factored_compensated`. Decode bundle
validation rejects an older manifest even though its buffer ABI still matches.
Rebuild/export through the normal recipe; no weight conversion or manual
`q4nx-build` packing step is required.

`dnx_precise.h` preserves all nine triple-BF16 products for state updates.
`dnx_factored.h` computes `decay*S_old^T*q + (k^T*q)*delta` using compensated
high/low sums. Helpers use `inline __attribute__((noinline))` so the linker
shares identical functions across entry TUs. Changing this back to `static`
exceeds full-layer program memory. Tests compare executable header text with
the accepted isolated candidate; comments and include names may differ.

Historical private control evidence uses checkout `3e1c2e8` and its matching
export. The private full-layer generator explicitly retains legacy DNX wrappers
so its own `dnx.h` injection cannot be bypassed by production header selection.
Do not relabel old fixtures or replay them against incompatible manifests.

## Build and export

Native Ryzen AI 9 365 / Strix XDNA2, Python 3.14, mlir-aie 1.4.2,
Peano 21.0.0.2026080301+c9c5ecb7, XRT `/opt/xilinx/xrt`.
Compiler subprocesses and NPU dispatch require execution-tool escalation on
this host. Do not run source-generating tests during compilation.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest -q specs/open-engine/tests
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
ironvenv/bin/python -u open_kernels/export_qwen36_kernels.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --out Models/qwen38-27b/production-factored-v2/kernels \
  --build-root Models/qwen38-27b/production-factored-v2/build --max-ctx 2048 -j 2
ironvenv/bin/python utilities/audit-kernel-export.py \
  --export Models/qwen38-27b/production-factored-v2/kernels \
  --build-root Models/qwen38-27b/production-factored-v2/build \
  --out Models/qwen38-27b/production-factored-v2/build-audit.json
```

Audit all 41 sets / 82 hashes. All five `build/lx/final.prj/dnx_*.o.d`
files must resolve both `designs/layer_x/dnx_factored.h` and `dnx_precise.h`,
and must not resolve legacy `dnx.h`. Check linked core symbols for one shared
`compensated_mac` and one `mac3` on each main core. IRON copies TUs before
compilation, so inspect actual dependency files, not just generated sources.
Use `utilities/attention-entry-replay/resources.py` for lx/ax with per-build
`build.json` containing the SHA256 of `final.xclbin` and `insts.bin`.

## Fresh full-depth validation

Verify model and all 65 pool hashes against retained `input-sha256.json`
before reuse. `--reuse-pools` itself checks sizes, not contents. Regenerate
all 195 independent reference outputs; CPU arithmetic is unchanged, so they
must match the retained reference hashes. Keep a separate new fixture.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -u open_kernels/model/make_decode.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-factored-v2/kernels \
  --layers 64 --tokens 3 --max-ctx 2048 --capture-boundaries \
  --out Models/qwen38-27b/production-factored-v2/decode \
  --pool-dir Models/qwen38-27b/production-full/pools --reuse-pools
ironvenv/bin/python utilities/stream-decode-weights.py \
  --cfg Models/qwen38-27b/production-factored-v2/decode/run_decode.cfg \
  --out Models/qwen38-27b/production-factored-v2/decode/run_streamed.cfg
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  /tmp/qwen38-fp32-harness/run_kernel \
  Models/qwen38-27b/production-factored-v2/decode/run_streamed.cfg
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python open_kernels/model/compare_decode.py \
  --out Models/qwen38-27b/production-factored-v2/decode --tokens 3
```

CPU preparation may run without `--kernel-dir` during compilation; bind the
audited export afterward using the same arguments plus `--cfg-only`.
Run `deltanet_inputs.py`, `deltanet_state.py`, `lm_head.py`,
`attention_heads.py`, `attention_projections.py`, `ffn_front.py` and
`ffn_partials.py` with the same `--model-dir`, `--kernel-dir` and `--out`.
Both FFN checkers need `--layers` listing all indices 0 through 63.

Retain every JSON and exit status. **Also assert
`deltanet_state.json["head_diagnostics_passed"]`**: that command's exit code
reflects only whole-tensor gates. Require all 6912 head states/outputs at the
unchanged maxrel <1e-4, cosine >0.9999999 bounds, plus exact history shifts
and zero padding. References carry their own causal state from zero.

Hash inputs, sources, export, harness, captures and reports. Streamed execution
does not establish resident runtime identity/reset or serving isolation.
Full resident validation needs approximately 35–40 GiB available RAM.
Block-prefill numerical parity and resource/performance acceptance remain
separate gates; three-position decode is not a long-context benchmark.

Limited resident reset regression (four layers, forced token IDs):

```bash
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OMP_NUM_THREADS=2 \
  /tmp/qwen38-paired-runtime/open_qwen36_cli \
  --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-factored-v2/kernels \
  --layers 4 --max-ctx 2048 --ids 248045,8678,198 \
  --det-step 3 --det-full --decode-route npu --pmode none
```

## Measured build — 2026-10-16

All 41 sets / 82 binary hashes pass the export audit. lx/ax compile in
141/133 seconds; maximum text is 15840/13760 bytes, maximum L1 buffer end
62336/59920, and stacks remain 6144. All 11 lx and 15 ax compute cores fit.
Each of eight lx main cores has one shared `compensated_mac` and one `mac3`.
All five DNX dependency files resolve both production correction headers.
The full host suite passes 1301 tests (47 skipped); native CTest passes 4/4.
Four-layer native full-buffer reset reports 0/3 differing replays.

## Full-depth results and open input gate

Build key `f5d922fd91d14b6cfa12f85d5c284a8f39efa56c305d2d572cd6a7009e4586b4`.
All 66 model/pool hashes and all 195 freshly regenerated independent reference
files match retained provenance. Streamed buffers: 2.963 GiB; 198 dispatches.
192/192 residuals pass, worst 0.0033520097452 (L63/T1) <0.005. Logit correlations
0.9999989786 / 0.9999982079 / 0.9999986382; argmax 8678 / 198 / 2 matches CPU.

All 6912 head-state/output cases pass: worst 9.34859257e-7 / 2.03650630e-6.
Conditioned recurrence 144/144 passes, with exact history and zero padding.
Attention heads/branches (1152/48), LM-head (3) and FFN front/down (192 each)
pass. Composed DeltaNet state/output also passes 144/144 (worst
0.0011783585 / 0.0028182071).

**The complete input gate still fails:** 143/144 cases pass; L34/T0 normalized
K maxrel is 0.00583593928545 >0.005. `deltanet_inputs.py` exits 1. Do not promote
this export to full numerical acceptance or relax the threshold. Six of seven
boundary checker processes pass; `checker-status.json` retains the failure.

The layer-34 controlled replay in `l34-legacy/` and `l34-factored/` uses the
same incoming residuals from the new full-depth fixture and each kernel's own
causal state from zero. Configs come from `full_layer_replay.config` with
`layers=[34]`, `tokens=3`, and the respective original/new `kernels/lx` paths.
The manifests' layouts (excluding the recurrence tag), packs and layer-type
contracts match; both binary pairs are checked against their original export
hashes. `input-audit.json` pins binaries, constants, pool, zero buffers, inputs
and configs before dispatch. Run each retained `run.cfg` serially with the
harness command above. `l34-entry-identity.json` records bit-exact xn/QKV/records/
history for all three positions; all candidate act/state/residual buffers also
match the full-depth capture bytewise. This is an entry-identity experiment,
not a claim that the old recurrence passes the new head gates.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/deltanet_projection.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-factored-v2/kernels \
  --out Models/qwen38-27b/production-factored-v2/decode --layer 34
```

This diagnostic intentionally exits 1 for the retained failure. All primary
record metrics match `deltanet_inputs.json` exactly. At T0, norm-only K effect
is 1.87577657e-5, QKV-only effect 0.00583192249; observed K versus projection
from captured norm is 0.00585469028. Each variant carries independent unrounded
conv history; no diagnostic replaces acceptance. Source, effective weights and
captures are hashed. Q8 projections are rejected rather than silently scored
with a Q4_1 oracle. Five tests cover causal history, non-finite/incomplete inputs,
quantization validation and retaining primary failure despite a diagnostic match.

Next: isolated QKV projection precision using these saved inputs, with byte-exact
control and resource audit before another integration. No full-model residency
is needed for that experiment. Full-depth resident identity/reset still needs
35–40 GiB available RAM; prefill, serving and final performance remain open.


The final rebuild also covers attention-only recipes: absent linear geometry
must retain legacy generated wrappers and omit the recurrence tag. Two RED →
GREEN regressions cover 9B and 27B attention-only specs. All ten wide generated
TUs remain byte-identical after this guard. The final export has freshly
regenerated references and a new hardware run; all 534 NPU captures match the
first factored export bytewise. `wide-generated-identity.json` and
`capture-identity.json` retain that comparison. Use `production-factored-v2/`
for final evidence; `production-factored/` retains the earlier export.

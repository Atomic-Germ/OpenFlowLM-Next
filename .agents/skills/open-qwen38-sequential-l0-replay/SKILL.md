---
name: open-qwen38-sequential-l0-replay
description: Replay production Qwen3.8 L0 through 243 causal positions with one weight pool, byte-exact retained native controls and complete activation/state captures.
---

# Production L0 long-prefix replay

Use `utilities/prefill-sequential-drift/native_l0.py` after the CPU predecessor
attribution. This is a diagnostic control, not a precision correction or model
acceptance. Preserve production lx binaries and actual native state from zero.
Never reset state between positions or seed it with the CPU oracle.

Preparation validates retained expanded boundary provenance, the current recipe
against the production bundle, and all original captures. It repacks only L0
using the shared Python packer and extracts original FP32 embeddings for P0–242.
One pool is resident. Every position dumps residual, complete act and state.
Do not interpret boundaries until all 243 residuals match the native CLI trace
byte-for-byte. Numerical tolerances are not an identity substitute.

## Build the standard harness

No kernel or library arithmetic changes are needed. Build current harness
sources, retain the CMake/compiler dependency files and hash the executable.
Use a fresh build directory under the ignored model artifact root. On this host:

```bash
cmake -S open_kernels/harness \
  -B Models/qwen38-27b/prefill-sequential-drift/harness-l0-v1 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DXRT_INCLUDE_DIR=/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/include \
  -DXRT_LIB_DIR=/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/lib
cmake --build Models/qwen38-27b/prefill-sequential-drift/harness-l0-v1 -j2
```

Compilation/NPU access may require execution-tool escalation. Do not run another
NPU job concurrently; the runner uses the shared resident-validation lock and
checks device availability. Full-model residency is unnecessary.

## Prepare, capture and independently re-audit

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/native_l0.py prepare \
  --baseline Models/qwen38-27b/prefill-expanded-cli/boundaries-native-v1 \
  --out Models/qwen38-27b/prefill-sequential-drift/native-l0-v2
XILINX_XRT=/opt/xilinx/xrt LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 \
  ironvenv/bin/python utilities/prefill-sequential-drift/native_l0.py run \
  --out Models/qwen38-27b/prefill-sequential-drift/native-l0-v2 \
  --harness Models/qwen38-27b/prefill-sequential-drift/harness-l0-v1/run_kernel
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/native_l0.py audit \
  --out Models/qwen38-27b/prefill-sequential-drift/native-l0-v2
```

Use a new output directory for each attempt. Retain failures; never overwrite
native evidence. The audit rereads all 729 captures and verifies pinned inputs,
sources, executable and logs. `model_acceptance` stays false even if identity
passes. Target subsequent independent sublayer diagnosis at P37/P241 and nearby
positions, comparing entry, conv, recurrence, output and FFN boundaries while
preserving complete causal histories. Production long-prefix acceptance remains
a separate all-layer requirement.

Link against the source-tree XRT development library, but execute with the
installed `/opt/xilinx/xrt` runtime and XDNA driver. The source-tree runtime
lacks that driver and returns `No such device with index 0`. Retain this failed
run separately (`native-l0-v1`); use a new preparation (`native-l0-v2`) to retry.

## Independent boundary scoring

After the exact control audit, run:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/l0_boundaries.py \
  --capture Models/qwen38-27b/prefill-sequential-drift/native-l0-v2 \
  --out Models/qwen38-27b/prefill-sequential-drift/l0-boundaries-v1
```

The scorer keeps the independent FP64 entry/conv/recurrent trajectory from
zero for every row and strict recurrence conditioned on captured records.
Selected Z/post/out and FFN comparisons use captured inputs. Production scratch
retains combined SiLU(gate)*up, not separate raw up/gate projections. Never infer
unobserved projection accuracy or whole-model acceptance from those gates.

## Validated checkpoint — 2026-10-24

`native-l0-v2`: 243 NPU dispatches, all 243 residuals byte-exact to the full-model
CLI, independently audited 729 captures (906,743,808 bytes). Logical NPU buffers
0.239906 GiB. `harness-l0-v1/build-evidence.json` pins GCC 15.2.0 and 326 compiler
dependencies; no kernel rebuild occurred. Initial runtime failure is preserved.

`l0-boundaries-v1`: 243/243 independent entry/conv/state gates, 11,664 strict
recurrence heads, and six selected Z/post/out/FFN front/down cases pass. Host
suite: 1792 passed / 47 skipped. Post norm/gate BF16 maxrel at P37/P241 is
0.0029327378 / 0.0029708429. Diagnose its signed contribution on a common final
residual scale before changing arithmetic; passing local gates do not close
sequential long-prefix acceptance.

## Signed post-output attribution (CPU only)

Use the exact audited capture without rebuilding or rerunning the NPU:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/post_attribution.py run \
  --capture Models/qwen38-27b/prefill-sequential-drift/native-l0-v2 \
  --out Models/qwen38-27b/prefill-sequential-drift/post-attribution-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/post_attribution.py audit \
  --out Models/qwen38-27b/prefill-sequential-drift/post-attribution-v1
```

Two independent histories start at zero: the existing mixed oracle must reproduce
stored references bytewise, and the FP64 reference supplies recurrence/Z inputs.
The signed chain propagates each boundary change through the nonlinear FFN on a
common final-reference scale. It separates oracle rounding/math, recurrence, Z,
BF16 post rounding, remaining post arithmetic, output projection, residual add
and conditioned FFN. Never add scalar maxima or treat this replacement-order
decomposition as a unique attribution. Approximate materialized weights: 4.282
GiB for one layer including mixed/FP64 copies.

Validated 2026-10-24: six exact reference reconstructions and six byte-exact
causal NPZ re-audits, exact signed closure. P37/P241 native maxrel
0.0011891532 / 0.0012928460 is dominated by post-rounding contributions
0.0012200074 / 0.0013051784. Frozen-downstream CPU predictions without rounding
are 0.0000308542 / 0.0000123325. P242 worsens (0.0000202869 -> 0.0000891240)
due to cancellation with output-projection error. Production/model acceptance
is unchanged. Host suite: 1801 passed / 47 skipped, nine RED-to-GREEN tests.

Next register/build a private FP32 post-output and direct-FP32 output-preparation
experiment with exact native controls over all 243 positions. Audit widened
scratch/streams and core resources; include the P242 cancellation case. CPU
counterfactual gains are not a substitute for fresh candidate NPU evidence.

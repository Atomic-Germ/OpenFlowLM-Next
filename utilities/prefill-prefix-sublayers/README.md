# Early prefill drift boundaries

Private weighted-candidate instrumentation at L34/P17, L37/P7 and L53/P17.
The first two rows sample early drift growth; the last has the largest
ideal-history gap in both routes of the retained CPU scan. These are diagnostic
choices, not proven faulty operations. No production arithmetic or XCLBIN edit.

The builder composes the existing weighted GEMM and residual/sublayer patchers,
then redirects the capture header. Only copied `core.cpp`/`cli.cpp` change.
Keep four-product FFN up/gate and original down arithmetic byte-identical.
Reject an existing build directory. Record all helper/staged/compiler dependency,
CMake cache, CLI and reused low-weight kernel hashes. Read this before building:

```sh
xrt_sdk=/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt
ironvenv/bin/python utilities/prefill-prefix-sublayers/build.py \
  --out Models/qwen38-27b/prefill-prefix-sublayers/build \
  --xrt-include "$xrt_sdk/include" --xrt-lib "$xrt_sdk/lib"
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 \
ironvenv/bin/python utilities/prefill-prefix-sublayers/run.py \
  --build Models/qwen38-27b/prefill-prefix-sublayers/build \
  --baseline Models/qwen38-27b/prefill-weighted-cli/v2/candidate/run \
  --out Models/qwen38-27b/prefill-prefix-sublayers/run
```

Run all three routes serially with exclusive NPU access and >=35 GiB available
RAM. Use the retained 20-token prompt plus independent continuation. Capture
only real rows. Prefix/state capture explicitly refuses a nonzero block start
or a real length other than 20, so another block cannot masquerade as the
complete causal prefix.

Require 66 exact baseline logits, 192 exact residual files, 81 complete
captures and 39 exact hybrid pairs before interpretation. Three sequential act
snapshots expose the established FFN boundaries; each hybrid target retains:

- selected input, pre-FFN residual, FFN norm, up/gate, H and down;
- all 20 real rows of entry norm, QKV, Z and gated DeltaNet output;
- raw state immediately before/after the full block, and selected out projection.

State-after is **P19**, not the target's state. Verify exact extent, zero initial
state and padding, FP32 history ABI and exact conv carry from captured QKV tail.
Reconstruct pre-FFN residual bytewise from captured input and output projection.
The host boundary reference uses unrounded FP64 arithmetic on the entire
captured projection prefix and its own zero state. It is a conditioned diagnostic,
not a primary reference or actual native recurrent state. Compare all 20 rows
and the final recurrent state; preserve any failed numerical gates.

Reconstruct three independent causal CPU outputs against retained refs before
using FFN metrics. Keep original-input versus captured-norm projection checks,
conditioned host state/DeltaNet output, output projection, local FFN and global
propagation separate. Full-model/long-prefix acceptance is always false.
An exact instrumentation control does not resolve the existing 3974/4032
residual failure or establish model support.

For CPU-only re-audit use `check.py` with `--build`, `--baseline`, `--run` and
`--out .../run/sublayer-audit.json`. Preserve native logs and any failed audit
before retrying; never repeat resident runs just to fix a CPU checker.

## Validated checkpoint (2026-10-21)

GCC 15.2.0; native CTest **4/4**; **587 compiler dependencies** pinned.
CLI SHA256: `6af988abba552db12da0cd2e41b9bf29c85d9ce51c4495a56f5c686d37d8a610`.
All three resident routes complete with **66/66 logits, 192/192 residuals**
byte-exact to the retained candidate. All **81 captures** validate, including
**39 exact hybrid pairs**. Three independent selected CPU outputs reconstruct
stored references bytewise; all six hybrid conv tails and pre-FFN residual
reconstructions are exact. State is captured after P19, independently of targets.

**54/54 conditioned FFN gates pass**, worst maxrel **0.0016358962**.
**732/732 projection/host boundary gates pass**, worst **0.0022380252**
(QKV from actual input, L34 over the captured prefix). These include six
per-position metrics across 20 positions, three layers, two hybrid schedules,
plus selected output projection and block-end state checks (720 + 12).

| Layer | Worst prefix QKV maxrel | Worst prefix Z maxrel | Worst conditioned og maxrel | Selected output projection maxrel | Block-end state maxrel |
|---|---:|---:|---:|---:|---:|
| 34 | 0.0022380252 | 0.0014517693 | 1.0677252e-6 | 0.0009515609 | 9.2116065e-7 |
| 37 | 0.0009451065 | 0.0013125434 | 8.8183857e-7 | 0.0003931275 | 7.2322780e-7 |
| 53 | 0.0010144976 | 0.0011105445 | 1.4542305e-6 | 0.0012094501 | 4.8807962e-7 |

QKV/Z columns use the actual-input reference; captured-norm results are close.
QKV is an aggregate vector gate, not separate per-Q/K/V or per-head acceptance.
Host state/og differences on captured inputs are much smaller than projection
errors. This motivates a CPU weight-rounding attribution for QKV/Z/out; it does
not yet prove the source of the remaining global drift or justify a kernel edit.

Global hybrid output maxrel is **0.0009041704 / 0.0011400684 / 0.0078214144**
at the three targets. L53/P17 already fails pre-FFN (**0.0075342533**), and
ideal FFN on its actual pre-FFN input still fails (**0.0079466232**).
Full retained residual accuracy remains **3974/4032 FAIL**. No production or
candidate arithmetic changed, and no throughput claim is made.

Ten new RED-to-GREEN tests; full host suite **1581 passed, 47 skipped**.
Evidence and logs are under `Models/qwen38-27b/prefill-prefix-sublayers/`.
Next use captured complete prefixes to separate Q/K/V, Z and out rounding,
keeping native history and primary references distinct. Require exact replay
controls before any new NPU precision variant or production promotion.

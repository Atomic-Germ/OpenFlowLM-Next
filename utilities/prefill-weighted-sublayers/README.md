# Weighted-candidate causal sublayer capture

Instrument only copied private weighted-FFN sources to capture L55/P7 and
L51/P17 (zero-based), the first and worst remaining short-prompt residual
failures. Reuse the unchanged weighted CLI builder and sublayer patcher with a
new target header; no production arithmetic, weights or kernel changes.
Record input, pre-FFN residual, norm, up/gate, activation and down for each
hybrid schedule, plus sequential act scratch. Real rows only, never padding.
Preserve absolute positions and actual native causal state.

The native build retains source/staged/binary/cache and compiler dependency
hashes, including this wrapper and capture header. Use fresh directories:

```bash
xrt_sdk=/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt
ironvenv/bin/python utilities/prefill-weighted-sublayers/build.py \
  --out Models/qwen38-27b/prefill-weighted-sublayers/build \
  --xrt-include "$xrt_sdk/include" --xrt-lib "$xrt_sdk/lib"
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 \
ironvenv/bin/python utilities/prefill-weighted-sublayers/run.py \
  --build Models/qwen38-27b/prefill-weighted-sublayers/build \
  --baseline Models/qwen38-27b/prefill-weighted-cli/v2/candidate/run \
  --out Models/qwen38-27b/prefill-weighted-sublayers/run
```

Native runs require exclusive NPU access and at least 35 GiB available RAM.
Run sequential, hybrid and block-major processes serially on the retained
20-token prompt plus one continuation through all 64 layers. Copy only retained
independent references; never create references from candidate outputs.
No new XCLBIN is built. GCC/CMake Release/OpenMP/XRT builds and CTest precede
native execution; the verified private low-weight kernel is reused unchanged.

Before interpretation require all 66 logits and 192 residual files exact to
the weighted candidate, exactly 26 sublayer captures and 12 exact hybrid pairs.
The checker revalidates baseline/build/input provenance, selected input against
the preceding layer and residual/down reconstruction. Independently reconstruct
each selected causal CPU output bytewise; stream one layer's weights at a time.

Conditioned FFN gates use actual captured input and unrounded FP64 arithmetic.
Keep them separate from global pre-FFN/output drift and ideal-FFN diagnostics.
Ideal state driven by native inputs is diagnostic, never actual native history
or a replacement primary reference. A successful diagnostic can coexist with
failed global accuracy. Sequential scratch cannot provide separate up/gate.
The CLI builds with GCC 15.2.0 and passes CTest **4/4**; all **587 compiler
dependencies** are pinned. CLI SHA256:
`7eb9a6b9d5a737fc3d78222f622704123d87a2c32ef9c4e2837cb3d6e4838d76`.
TDD: eight new tests fail before implementation and then pass; full host suite
**1558 passed, 47 skipped**.

## Measured diagnosis (2026-10-21)

All three full-resident native routes complete. **66/66 logits and 192/192
residual files are byte-identical** to the weighted candidate baseline.
All 26 sublayer captures are complete and the 12 hybrid pairs are exact.
Both selected independent causal CPU outputs reconstruct the stored FP32
references bytewise. All **36 conditioned FFN gates pass**, worst maxrel
0.0020749792. This does not close global model acceptance.

| Hybrid row | Input maxrel | Pre-FFN maxrel | Native output maxrel | Ideal FFN on actual pre-FFN input |
|---|---:|---:|---:|---:|
| L55/P7 | 0.0029030925 | 0.0061193443 | 0.0062761194 | 0.0060717536 |
| L51/P17 | 0.0057009925 | 0.0092505118 | 0.0138594086 | 0.0138246489 |

Native-vs-ideal-FFN output errors are only 0.0002031324 / 0.0003275429;
ideal FFN still fails the original global <0.005 bound. Pre-FFN differences
against ideal causal state driven by native input prefixes are
0.0006651352 / 0.0004594719. These diagnostic comparisons point to propagated
input/history drift rather than a remaining FFN-only correction. They do not
identify a single earlier faulty operation. Sequential selected outputs pass
at 0.0002325612 / 0.0008600592.

Next scan preceding layers with retained native/independent input prefixes to
localize where the drift grows. Ideal CPU state remains diagnostic, not actual
native state. Preserve full-model residual FAIL (3974/4032 passing) and defer
long-prefix acceptance/production promotion.

The first CPU audit mistook selected `_res.bin` captures for extra full-layer
traces. A regression now permits only the exact declared sublayer names while
still rejecting missing/extra layer traces. Preserve `inventory-failure.json`
and `native-audit.log`; the corrected CPU-only re-audit in `cpu-audit.log` uses
the same native captures. Re-audit without reloading the model:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-weighted-sublayers/check.py \
  --build Models/qwen38-27b/prefill-weighted-sublayers/build \
  --baseline Models/qwen38-27b/prefill-weighted-cli/v2/candidate/run \
  --run Models/qwen38-27b/prefill-weighted-sublayers/run \
  --out Models/qwen38-27b/prefill-weighted-sublayers/run/sublayer-audit.json
```

Artifacts remain in `Models/qwen38-27b/prefill-weighted-sublayers/`. Native
arithmetic and all production kernels/weights remain unchanged.

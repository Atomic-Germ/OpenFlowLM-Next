# Composed attention/FFN prefill replay

Private N14336/K5120/T256 attention entry control and low-weight GEMM builds.
Reuse verified N5120/K6144 output, N34816/K5120 up/gate and N5120/K17408 down
builds. Never modify production arithmetic or widen global acceptance bounds.
New CPU bridges compile production paired-KV attention and extract the exact
production RMSNorm/activation expressions, with source anchors and full compiler
dependencies. Fresh directories retain failed experiments.

The input is the complete retained 20-token L63 prefix. P1 is selected for
composition through output/residual, postnorm, FFN up/gate, activation and down.
First reproduce native entry norm, all Q/K/V/gate and paired KV/attention bytes,
selected output projection, FFN norm/up/gate/activation/down and final residual.
Original/private high and low input products must match individually. Corrected
weight arithmetic is `(WhXh+WhXl)+(WlXh+WlXl)` with FP32 additions. Use offsets
0/128 for prefixes and 1/129 for the selected FFN row. Every dispatch must have
zero inactive columns; corrected intermediates must agree across placements.
Block and one-token host calls must produce identical attention and cache.

Primary per-head entry references use original FP32 normalized inputs and
source-identical effective weights. Host gates use an independent FP64 causal
attention/KV reference, with maxrel <0.001. Projection gates keep maxrel <0.005,
rel_fro <=0.005; all entry heads must strictly improve. Final conditioned FFN
and independent layer output are separate gates. The latter retains the CPU's
own incoming prefix. Local acceptance never establishes full-model propagation.

Build and prepare (from the project root):

```bash
ironvenv/bin/python utilities/prefill-attention-replay/build.py --mode control \
  --out Models/qwen38-27b/prefill-attention-replay/builds/control
ironvenv/bin/python utilities/prefill-attention-replay/build.py --mode low \
  --out Models/qwen38-27b/prefill-attention-replay/builds/low
ironvenv/bin/python utilities/prefill-attention-replay/bridge.py \
  --out Models/qwen38-27b/prefill-attention-replay/bridge
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-attention-replay/check.py prepare \
  --out Models/qwen38-27b/prefill-attention-replay/prepared-v2 \
  --attribution Models/qwen38-27b/prefill-combined-attribution/v1 \
  --capture Models/qwen38-27b/prefill-combined-sublayers/native-v1 \
  --audit Models/qwen38-27b/prefill-combined-sublayers/native-v1/sublayer-audit-v2.json
```

Run serially with exclusive device access. The checker holds the shared resident
validation lock and checks device ownership before each dispatch. No full-model
residency; only individual projection weights are sent to the NPU.

```bash
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
OMP_WAIT_POLICY=PASSIVE ironvenv/bin/python utilities/prefill-attention-replay/check.py run \
  --out Models/qwen38-27b/prefill-attention-replay/native-v2 \
  --prepared Models/qwen38-27b/prefill-attention-replay/prepared-v2 \
  --build-root Models/qwen38-27b/prefill-attention-replay/builds \
  --out-build Models/qwen38-27b/prefill-out-replay/builds \
  --ffn-build Models/qwen38-27b/prefill-paired-weights/v2 \
  --down-build Models/qwen38-27b/prefill-down-replay/builds \
  --harness-build Models/qwen38-27b/prefill-out-replay/harness \
  --bridge Models/qwen38-27b/prefill-attention-replay/bridge
```

Repeat with `audit` instead of `run` to recompute all reference/host/numerical
checks from hashed raw captures without NPU dispatch. Exit 1 means numerical
failure; exceptions invalidate execution/provenance/control acceptance.

## Validated result — 2026-10-23

`prepared-v2/` and `native-v2/`: **64/64 dispatches** complete. All native
entry/host/KV/output/FFN/residual controls are exact, as are individual control
products and both slot placements. Zero inactive columns and native host
block/single-token identity pass. All eight packed/source weight identities pass.

| Gate | Candidate maxrel | Candidate rel_fro |
|---|---:|---:|
| Entry, worst of 2240 head cases | 4.8850442e-6 | 3.0968432e-6 |
| Output projection, worst of 40 rows | 2.6402108e-6 | — |
| Up/gate, worst of 4 projections | 1.4807326e-6 | — |
| Down projection, both placements | 1.1706927e-5 | 3.1340009e-6 |
| Conditioned FFN | 2.5746182e-5 | 4.7519969e-6 |
| Independent L63/P1 layer output | **0.0040805495** | **0.0017167913** |

All entry cases strictly improve both errors. All **1280 host head/cache gates**
pass, worst maxrel **1.1655479e-5 <0.001**. All projection, conditioned FFN and
independent output gates pass. The final residual improves from **0.0096485275**
to **0.0040805495**, crossing the unchanged 0.005 bound. Its pre-FFN maxrel is
0.0023744032. The independent CPU re-audit recomputes every product combination,
host boundary and metric from raw captures. `model_acceptance` remains false:
this fixes a retained L63/P1 suffix, with actual incoming error still present.
The previous full-depth **4000/4032** residual result remains FAIL.

Both new entry builds use 801 compiler dependencies, 32 cores, max text
4784/5568 bytes, L1 buffer end 61448 bytes and configured stack 4096 bytes.
The fresh GCC 15.2.0 host bridges pin 342 attention / 105 FFN dependencies.
The reused harness pins 326 dependencies. No production source or packing change.

SHA256 (complete dependency and resource inventories are in build JSON):

- Entry control XCLBIN: `fb10ef6badff7925714b1fcfc87f2cee00ccc43525e520fad7af2d27e4927890`
- Entry low XCLBIN: `00748378478c9cca21007c0dd6be7b02e9262884bc8e1b74fa1f29239d21f7cb`
- Both instruction streams: `032ec3405135f5d08e3406b7cd50e26292adc3a4947399b228932602ee111ea4`
- Both instruction ELFs: `583ba1ada734e5f23052953e73f5be40f2daa795fd1ac67e98ef4601f1c6173d`
- Attention bridge: `7378fd6f60ca75c6a4dc2f9c4289c4a0712e95b9f9fe616ba49f8416ec97c37b`
- FFN bridge: `a3c18dbd7ec7bc3cad75bd9339e922cae923f59fbdd5f6106b6ac06df54f3ca3`

The first run stopped after 28 dispatches because the retained native output
was stored in an FP64 container. Its values equalled the FP32 reconstruction,
but strict dtype/byte identity correctly rejected it. A lossless FP32 conversion
now rejects nonrepresentable or nonfinite values. The failed v1 run and checker
snapshot remain retained; v2 uses a fresh preparation and fresh NPU outputs.
Fourteen RED-to-GREEN tests; **1723 passed, 47 skipped** in the full host suite.

Next build a private CLI extending the combined policy to attention entry/out
and FFN down with exact role guards. Then require fresh full-depth propagation,
independent all-layer gates and repeated resident/reset controls. Local success
cannot replace those tests; long-prefix and production acceptance remain open.

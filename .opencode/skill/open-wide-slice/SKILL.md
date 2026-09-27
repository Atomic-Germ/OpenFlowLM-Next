---
name: open-wide-slice
description: Reproduce the synthetic eight-layer H5120 Qwen38 slice, with distinct production-packed weights, device activation chaining and isolated DeltaNet/KV state. Use for B7 slice regressions and preparation of multi-token decode validation.
---

# Wide eight-layer slice

Read `specs/open-engine/plans/wide-eight-layer-slice.md` for evidence and scope.
The synthetic slice passes its inherited acceptance gates. Strict head-local
recurrent-state diagnostics pass8/24 layer-token tensors, so do not claim
strict primitive equivalence or model support. Next is multi-token decode with
token selection; current token inputs are independent synthetic hidden vectors.

Prerequisites: the accepted artifacts and `layer-fixture.json` /
`layer-results.json` from `open-wide-deltanet-precision` and
`open-wide-attention-layer`. Build/validate those using their skills if missing.
This stage reuses their binaries, including the distinct N16384 DeltaNet and
N14336 attention projections. It does not build a new xclbin.

From repository root:

```bash
base=open_kernels/designs/wide_deltanet/build_slice
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-wide-slice.py prepare --out "$base"
HARNESS_TIMEOUT_MS=30000 open_kernels/harness/out/run_kernel "$base/slice.cfg"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-wide-slice.py compare --out "$base"
ironvenv/bin/python -m pytest specs/open-engine/tests -q
```

Preparation generates about2 GB of distinct packed weights and takes several
minutes for FP64 references. Use separate output directories for new experiments;
prepare replaces files and clears old outputs. Run hardware sequentially, with
normal tool escalation for `/dev/accel/accel0` when needed. Expected default:
370 NPU calls,3098 passing checks, worst final output error4.11248e-5 versus5e-3.

The prefix comes from explicit `layer_types`, not the interval field. It must
contain six linear and two full layers. Weight RNG streams use seed38429 plus
layer index. Four evaluations are two cold tokens at0/1 and two warm tokens at
255/256; `--tokens` accepts2..8 but only the default is recorded as validated.
All layer outputs and states repeat exactly after a complete reset.

`recipes/wide_slice.py` scopes each standalone recipe's BO operands while
sharing LN, output projection and FFN kernel contexts. Do not rename numeric
offsets or confuse a kernel alias with a BO of the same name. Scratch is separate
for DeltaNet and attention; persistent `stateN`/`cacheN` is separate for every
layer. Stream pool/constant files into the shared BOs before each layer, then
copy only device `y` into `x`. Only the first layer receives a host input.

Keep input/output boundary dumps, prior-state comparisons, canaries, every
state-head padding check, cache untouched-row checks and full reset comparisons.
Reference files must not appear as loads in the cfg. The oracle in
`utilities/wide_slice_reference.py` must evolve its own states and activations.
The packed pool and constant hashes must remain distinct across all eight layers.
Do not adjust the seed or tolerances to resolve a numerical regression.

The declared BO total is424205768 bytes plus XRT overhead; packed files total
1988808704 bytes. Ten contexts execute sequentially, not as one fused placement.
No runtime instruction patcher, embedding/LM-head/token-selection loop, packed
model,64-layer inference or performance claim is validated here. Existing fused
DMA and catalogue guards remain in place.

---
name: open-wide-decode
description: Reproduce synthetic eight-layer autoregressive Qwen38 decode with the full Q8 vocabulary, final RMSNorm, device-logit token selection and persistent DeltaNet/KV state.
---

# Wide autoregressive decode

Read `specs/open-engine/plans/wide-autoregressive-decode.md` for measured results
and remaining model-integration gates. This is synthetic eight-layer validation,
not a packed 64-layer model or a runtime performance claim.
Recorded default:5149 passing checks across684 NPU calls, logits correlation
at least0.9999999228, all selected tokens equal to the independent reference,
and a byte-identical full cold replay. Strict state-head diagnostics pass10/36.

Prerequisites are the accepted `open-wide-slice` and `open-wide-lm-head`
artifacts. The prepare command checks their accepted status and the hashes of
reused weights, constants and kernels. It links existing packed fixtures instead
of copying or repacking them. Keep their original directories available.

Build the open harness with matching XRT headers/libraries. This host's SDK is
in `/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt`:

```bash
cmake -S open_kernels/harness -B /tmp/oflm-wide-decode-harness \
  -DCMAKE_BUILD_TYPE=Release \
  -DXRT_INCLUDE_DIR=/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/include \
  -DXRT_LIB_DIR=/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/lib
cmake --build /tmp/oflm-wide-decode-harness -j4
base=open_kernels/designs/wide_deltanet/build_decode
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-wide-decode.py prepare --out "$base"
HARNESS_TIMEOUT_MS=30000 /tmp/oflm-wide-decode-harness/run_kernel "$base/decode.cfg"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-wide-decode.py compare --out "$base"
ironvenv/bin/python -m pytest specs/open-engine/tests -q
```

Set XRT runtime library paths as required by the host, and use normal device
escalation if sandbox-hidden. Run NPU harnesses sequentially. Preparation requires
a new output directory; use `--out` for a rerun rather than deleting accepted
evidence. It generates a 2.54 GB BF16 embedding matrix (seed38430) and references.
No new xclbins are needed; only the harness gains two control directives.

`greedy logits vocab token` rejects nonfinite logits, takes the lowest index on
a tie, and writes a uint32 token. `embed x embedding.bin token vocab hidden`
reads exactly that BF16 row and widens its bits to FP32. The host performs no
neural operator. Final norm and full248320-row Q8 LM head execute on the NPU.
Only sequence start/reset loads the seed token248045. Later tokens must come
from device logits. Do not load reference activations or reference-selected IDs.

Reference states and token selection evolve independently. Preserve all inherited
slice gates, per-layer state isolation, untouched KV rows, canaries, byte-exact
activation chaining, the complete three-token reset replay and finite full-length
logits correlation>.9999 with identical argmax. The conditional LM-head check
(maxrel<1e-4, cosine>.9999999) supplements the independent end-to-end comparison.
Keep strict head-local recurrent-state diagnostics visible even when they exceed
the stricter primitive threshold. Do not tune seeds or loosen acceptance bounds.

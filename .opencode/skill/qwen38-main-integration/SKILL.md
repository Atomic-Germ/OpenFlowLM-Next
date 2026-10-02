---
name: qwen38-main-integration
description: Rebuild and check the corrected Qwen38 FFN after main merges, compare upstream head-order fixes, and replay immutable full-model fixtures.
---

# Main integration with corrected wide kernels

Read `specs/open-engine/plans/qwen38-main-merge.md`. This workflow checks
integration of main through06b98bc with the precision work throughf4e5bd9.
It supersedes the old upstream-regression skill's baseline arithmetic for
this revision. Keep original fixture hashes, seeds and tolerances.

The general Qwen converter now restores arbitrary value/key head ratios.
The separate27B streaming converter already does the same. Run
`utilities/q4nx-build/tests/test_qwen35_27b.py` and `test_qwen35_vheads.py`
to guard equivalence and independent labelled-head semantics; don't apply
the permutation twice or remove the bounded-memory27B override.

The upstream block-prefill route includes host neural operations and changes
runtime defaults. It is not the all-NPU harness and cannot close its numerical
gate. Keep catalogue guards. Preserve both prefill and precision headers in
recipe cache dependencies, and the manifest's banked transpose and Q8 split.

Rebuild the corrected FFN in a fresh directory (layer_x builds must be serial):

```bash
base=open_kernels/designs/wide_deltanet/build_main_merge
ironvenv/bin/python utilities/probe-qwen35-wide.py --scope ffn --ffn 17408 \
  --ffn-correction --product-correction --block-carry --segment-carry \
  --activation-carry --out "$base"
ironvenv/bin/python utilities/test-segmented-dense.py prepare --build-dir "$base/ffn"
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel "$base/ffn/segmented.cfg"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-segmented-dense.py compare --build-dir "$base/ffn"
```

All26 checks must pass. Compare all13 `gotN.bin` activation arenas bytewise
with `build_activation_boundary/carry_add/ffn`; the recorded merge leaves them
unchanged. Text16176 B plus24 B coefficient data fits; use `llvm-size -A`.
Instructions retain SHA256e8cc08c29bbca5afbdfe59493baf6d4655e6853f0a4c369b25847e91337ce708.

Full replay uses the original immutable source and the current norm correction:

```bash
wide=open_kernels/designs/wide_deltanet
ironvenv/bin/python utilities/replay-wide-model.py \
  --source "$wide/build_full_model" --out "$wide/build_main_merge/full" \
  --output-projection "$wide/build_ffn_boundary/carry/projection_k6144" \
  --ffn "$wide/build_main_merge/ffn" --ln "$wide/build_norm_finish/carry" \
  --attention "$wide/build_model_precision/attention" \
  --attention-projection "$wide/build_attention_boundary/carry/projection_k5120"
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel "$wide/build_main_merge/full/decode.cfg"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-wide-full-model.py compare --out "$wide/build_main_merge/full"
```

Before this merge, full acceptance fails two checks: cold-1-layer63 residual
maxrel0.022905298362964267 and cold-1 final_norm0.01282051282051282. Preserve
those failures when assessing regressions. Compare captures/results to
`build_norm_finish/full`; matching tokens alone is insufficient.
The recorded post-merge run completes3564 calls and all8190 dumps are bytewise
identical to that baseline, including reset/state/logits. It retains the same
two accuracy failures; main does not close PR4's numerical gate.

Run `ironvenv/bin/python -m pytest specs/open-engine/tests -q` and the full
converter suite. Converter dependencies can live under `/tmp` via `pip
--target`, passed through `PYTHONPATH`; ironvenv itself need not change.
Recorded temporary directory: `/tmp/oflm-main-merge-deps`, with CPU torch,
gguf, safetensors, einops and huggingface-hub.

For the host runtime use the CMake/CTest and independent block fixture commands
from `qwen38-upstream-regression/SKILL.md`, with a fresh build directory and
the matching installed XRT SDK. Compile C++ pool tests with `-Isrc
-Isrc/include -Iopen_kernels/harness` and `pools.cpp manifest.cpp q4nx_file.cpp`.
Compile server `openai_compat_test.cpp` with `-mavx2`, `-Isrc -Isrc/include`
and the SDK include directory; run with `src/model_list.json` as its argument.
These are CPU helpers, not a live server acceptance test.

# Main integration and Qwen3.8 overlap review

2026-10-03. Integrate `origin/main` at `06b98bcc0e35fb93f4d051833fcdddd7c870beed`
into `implement-qwen38-27b-support` after `f4e5bd9`. At fetch time `upstream/main`
and `origin/main` point to the same commit. Six main commits were missing from
the integration branch. User changes to `.gitignore`, `Dockerfile` and
`LLM_Coding_Agent_Plan.md` are excluded from this integration.

## What changes for Qwen

- `06b98bc` (#143) restores tiled GGUF value heads to grouped HF order for
  arbitrary value/key head ratios. This directly addresses Qwen3.8-27B's
  48/16 heads and explicit4096-row Q/K prefix. Our bounded-memory27B converter
  already implements the same permutation. Eight added equivalence checks cover
  gate, alpha/beta, A/dt, output codes/scales, QKV and convolution. Both
  implementations agree; the existing converted model does not need repacking
  for this fix. The27B override and its BF16 alpha/beta companions are preserved.
- `c69dbd5` (#124) adds dense Qwen3.5-family batched prefill, exact two-part
  Q8-to-Q4_1 packing, manifest fields and runtime weight loading. It also fixes
  skipping prompt processing when a requested block route is unavailable.
  Block prefill now defaults on for eligible prompts/kernel sets; `=0` disables
  it. The route includes host neural stages and is separate from our fully-NPU
  64-layer harness. Its validated smaller-model results do not establish27B
  support or close our numerical gate. Wide catalogue guards remain intact.
- `8c83712` (#118) addresses program-memory overflow in all-Q8 MoE builds and
  improves export diagnostics. The changed common generator retains our dense
  Q4 FFN flags/schedule; an actual rebuild and device comparison verify this.
- `e4c022e` (#138) improves server errors, request handling and model-name
  handling. CPU compatibility tests cover these helpers; this review does not
  claim a live HTTP/model smoke test.
- README and device-free BERT export changes are unrelated to27B arithmetic.

The two textual merge conflicts are resolved by retaining both sets of recipe
cache dependencies and both `transpose_banked` documentation and the new Q8
`split` field in `PackOp`. No changes to numerical tolerances, model weights,
catalogue limits or the precision algorithms are needed for this merge.

## Checks

| Check | Result |
|---|---|
|Open-engine CPU tests|844 passed,47 skipped|
|Converter tests|97 passed,42 subtests passed|
|Standalone C++ runtime|Release build succeeds|
|CTest manifest/vision checks|3/3 pass|
|Block host independent fixture|PASS|
|C++ pool packing tests|PASS,0 failures|
|Server compatibility helpers|146 checks,0 failures|
|Rebuilt corrected FFN|26/26 checks over13 inputs pass|
|FFN before/after integration|All13 activation arenas byte-identical|
|Full-model hardware replay|3564 calls complete; all8190 captured tensors byte-identical|

Missing converter dependencies were installed in `/tmp/oflm-main-merge-deps`
without changing ironvenv. C++ tests use the matching XRT development SDK at
`/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt`.
The upstream `model_list.hpp` retains CRLF; `git -c core.whitespace=cr-at-eol
diff --cached --check` checks the merge without converting that unrelated file.

## Rebuilt artifact and reproduction

The corrected H5120/FF17408 FFN is rebuilt in
`open_kernels/designs/wide_deltanet/build_main_merge/ffn`, keeping product,
block, segment and activation carry. Program text is16176 B, coefficient
data24 B; DMA instructions are unchanged.

| Artifact | SHA256 |
|---|---|
|FFN xclbin|`ee08fe3b8286598ea14ab28aa71322745c0fbfef01d971b0713589413e235083`|
|Instructions|`e8cc08c29bbca5afbdfe59493baf6d4655e6853f0a4c369b25847e91337ce708`|

Use [qwen38-main-integration](../../../.opencode/skill/qwen38-main-integration/SKILL.md)
for exact build/test commands. Logs are `/tmp/main-merge-*.log`, with the full
hardware log under `build_main_merge/full/hardware.log`. The original
`build_full_model` fixtures and all unchanged validated kernels are reused.

## Full-depth result

All8190 dumps, including logits, residuals, recurrent/KV state, reset outputs
and canaries, match `build_norm_finish/full` byte for byte. Thus the main
integration neither improves nor worsens the measured NPU decode arithmetic.
Greedy tokens remain248045 ->8678 ->198 ->2. Residual additions remain exact;
norm diagnostics remain810497 model differences,11 local norm differences and
810495 propagated-input differences (overlapping counts).

The two pre-existing full-model failures remain: `cold-1-layer63/y` maxrel
0.022905298362964267 versus0.005, and `cold-1/final_norm` maxrel
0.01282051282051282 versus0.008. Matching captures and tokens do not turn these
into passes. PR4 still needs numerical closure and runtime integration; the
next isolated precision case remains layer8 FFN activation749.

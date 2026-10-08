# Host attention refuses the families it doesn't compute (fixes #171)

**Status (2026-10-08): done.** Implemented as OPEN-HOST-ATTN-GUARD; K2 checked on hardware. Original plan: Branch `fix/171-host-attn-guard` from
upstream/main 3f3ff9f2, worktree `C:\code\openflowlm-171`.

## The bug

`Core::host_attn_layer` (`src/open_qwen36/core.cpp:2200`) computes K2's attention: no q/k norm,
no bias, no gate, no sliding window, full half-split rotation, 1/sqrt(hd) scale. Its guard
(`core.cpp:2221`) checks only the geometry, so a Qwen3 model passes and runs without its q/k
norm. Measured on Qwen3-8B on 2026-10-06: the output is garbage, and the exit code is 0.

## How it is reached (checked on main)

- `OFLM_OPEN_HOST_ATTN=1`: block prefill (`step_gemm_block_layer`, `core.cpp:2460`).
- `OFLM_OPEN_HOST_ATTN_DECODE=1`: the initial decode route (`core.cpp:341`).
- `set_decode_route("host")`: the CLI's `--decode-route host` and `sched_route` (`cli.cpp:338`).
- `auto` reaches it only past `decode_auto_threshold_`. That defaults to SIZE_MAX
  (`core.cpp:342`), so it needs `OFLM_DECODE_AUTO_THRESHOLD` set.
- **So without one of these flags nobody reaches it.** `oflm serve` has no route option, but
  the two env vars still work under it.

## The fix

1. **One predicate, an allow-list by family:**
   `std::string host_attention_refusal(const Manifest&)` in `manifest.hpp/.cpp`.
   - It returns "" when the host attention computes this model exactly, and otherwise why not.
   - It accepts `family == "k2"`, with the existing geometry check moved in from
     `host_attn_layer`. It refuses every other family by name, saying what that family has
     that the host path lacks, for example "qwen3: a q/k RMSNorm before RoPE".
   - **Why by family, not by feature.** The variants live in the recipe as family properties
     (`QKNORM_POST_ROPE`, `QKV_BIAS_FAMILIES`, Granite's multiplier, Gemma 3's window). They
     aren't manifest fields, so the engine can't detect them one by one. This mirrors how
     `attn_block.prep` and FAST_ATTENTION admit families: one at a time, by measurement.
   - llama3 probably computes correctly here (no norm, no bias, its scaling folded into
     `rope_inv_freq`), but it hasn't been measured, so it stays out.
2. **Check it when the route is chosen,** not on the first dispatch:
   - `Core` constructor: either env set on a refused model throws at load, with the predicate's
     message.
   - `set_decode_route("host")` throws with the message.
   - `decode_route_at` under Auto returns "npu" for a refused model, so a set
     `OFLM_DECODE_AUTO_THRESHOLD` never switches mid-generation into a throw. It logs this once.
   - `host_attn_layer` keeps calling the predicate as its last line of defence.
3. No change to K2's behaviour or to the default routes.

## Spec impact

**New requirement OPEN-HOST-ATTN-GUARD** (no existing requirement covers the host attention;
#139 added it without one). Applies to: `src/open_qwen36`. **Verification: test.**
- The fixtures `manifest_qwen3_4b`, `manifest_hy_mt2_7b`, `manifest_gemma3_4b` and
  `manifest_phi4_mini_4b` are each refused, and the message names the family.
- The qwen3_4b fixture with `family` set to `k2` is accepted. Its geometry is K2's: full
  rotary, hd 128.
- That same fixture with `rotary_dim` halved is refused on geometry.
- Lives in `manifest_test.cpp`, which compiles `manifest.cpp` only and needs no NPU.

The route-selection behaviour (load-time refusal, `auto` staying on the NPU) needs a model on
the NPU, so it is a **manual** procedure in the spec, not a test.
- On Qwen3-8B, `OFLM_OPEN_HOST_ATTN=1` and `OFLM_OPEN_HOST_ATTN_DECODE=1` each fail at load with
  the message.
- `--decode-route host` fails with the message.
- `OFLM_DECODE_AUTO_THRESHOLD=8` with `--decode-route auto` gives the same 24 tokens as the
  default route.
- The default route's output is unchanged.

## Not verified

- No K2 model is installed here, so K2 still working is covered only by the unit test. The
  PR will say so.

# K2 prefill: the attention products at `prep` `rope`

**Status:** done 2026-10-09, on `feat/k2-horizon-7b` (#184).

## Why

#184's description said K2's prefill attention ran on the host. It did not.
- Below 64 tokens the app prefills token by token.
- From 64 up, the block route ran attention as one `dxB` dispatch a token.

K2's kernel set already carried the attention products (`ag_s` / `ag_pv`). The engine skipped them because the only host half it knew was `qknorm_rope`, and K2 has no q / k norm.

## Change

- **`prep` `rope`:** the rotation alone.
  - `recipes/dense.py` declares it for `BLOCK_ATTN_ROPE = ("k2",)`.
  - `manifest.cpp` accepts it.
  - `attention_prep` skips the norm when it has none.
- **Q split at `rope`:** the scores take Q as a bf16 hi / lo pair, two dispatches summed on the host, as the decode kernel splits q.
  - Measured necessary: one bf16 Q put the products at 0.9955 from the sequential route at 28 tokens, where `dxB` is 0.9986.
- **Startup log:** a dense set without the products now says "one dxB dispatch a token", not "attention on the host".

## Spec impact

| ID | | Verification |
|---|---|---|
| OPEN-PREFILL-ATTN | modified: the second `prep` value, the Q split, the K2 result | unit: `test_prefill_attn.py` (K2 emits `rope`), `manifest_test.cpp` (`rope` parses); manual: `attn_route_check.py`, `oflm-test --llm` |

## Results

See OPEN-PREFILL-ATTN, "Result 2026-10-09":
- 2.2x at 512-999 tokens and 3.6x at 2000 on the CLI.
- Through `oflm serve`, time to first token was 2x at 849 tokens and 3x at 2781.
- The 28-token route check stays under the 0.9998 bar (0.99958). At that length `dxB` is no closer to the sequential route, and the app never sends that length down the block route.

# oflm.exe applies OMP_WAIT_POLICY itself

2026-10-07. Status: **done** (OPEN-PREFILL-BATCH, Result 2026-10-07). Branch `fix/oflm-omp-wait-policy`. Issue #176.

## Why

The open engine sets `OMP_WAIT_POLICY=PASSIVE` from a static initializer, and `oflm.exe`
delay-loads vcomp so that the initializer runs before vcomp reads the variable. That works
in `open_qwen36_cli`. It doesn't work in `oflm.exe`, because eight of the closed model DLLs it
links at load time import vcomp themselves:

`qwen3_6_moe_npu`, `qwen3_5vl_npu`, `qwen3_5_omni_npu`, `qwen3vl_npu`, `gemma_npu`,
`gemma4e_npu`, `gemma4_12b_npu`, `gpt_oss_npu`

(found with `dumpbin /dependents` over every DLL beside `oflm.exe`). So vcomp is in the
process before any of our code runs, and the host workers spin through every NPU dispatch.
The spec records this as known and unfixed (OPEN-PREFILL-BATCH, 2026-09-21 result: 1.34-1.39x
on server prefill). `oflm bench` on the q8 35B today: 84-94 tok/s prefill as shipped, against
109-118 with the variable set before launch.

## Change

`src/CMakeLists.txt`: add `/DELAYLOAD` for those eight DLLs beside the existing
`/DELAYLOAD:VCOMP140.DLL`. Each one then loads on its first call, and vcomp with it, after
the initializer has set the policy. The link succeeds, so `oflm.exe` imports no data symbols
from them; a delay-loaded DLL can't supply data imports.

Consequences:
- The closed engines in those DLLs also run with `PASSIVE` now, since the variable is set
  process-wide before they load. Before, they ran with spinning workers. Checked on hardware
  below.
- If one of those DLLs is missing, `oflm.exe` now starts and fails when that model is first
  used (delay-load exception 0xC06D007E), instead of failing before `main()` with
  0xC0000135. The other DLLs are unchanged.
- No other DLL beside `oflm.exe` imports vcomp, so the list is complete today. A closed DLL
  added later that imports vcomp would bring the warning back, and the warning names it.

## Spec impact

**OPEN-PREFILL-BATCH, modified.** The criterion "The flag is necessary and NOT sufficient: in
`oflm.exe` eight implicitly linked closed model DLLs import vcomp themselves, so the warning
fires there and the variable has to come from the environment" becomes: `oflm.exe` also
delay-loads those eight, so the policy applies in-process and the warning doesn't fire. The
criterion stays that the warning is accurate. No new IDs, nothing removed.

## Checks (manual, hardware)

1. `oflm bench` on the 35B with no `OMP_WAIT_POLICY` in the environment: no WARNING, `host
   threads: 24 (OMP_WAIT_POLICY=PASSIVE)` logged, prefill matching a run with the variable set
   before launch.
2. A closed engine that is now delay-loaded still runs: `OFLM_QWEN36_ENGINE=closed oflm bench`
   on the 35B.
3. `oflm-test --llm` through `oflm serve`.

## Outcome (2026-10-07)

1. **Pass.** No `OMP_WAIT_POLICY` in the environment: no warning, `host threads: 24
   (OMP_WAIT_POLICY=PASSIVE)` logged. `oflm bench` on the 35B, q4_1 set, alternated:
   135.5 / 137.3 tok/s at 1k and 138.7 / 140.3 at 2k, against 137.1 / 139.5 and 140.8 / 141.0
   with the variable set before launch.
2. **Changed check.** `OFLM_QWEN36_ENGINE=closed` on the 35B segfaults right after "Loading
   model", and does so identically on a binary built without this change, so it's
   pre-existing and not caused by delay-loading. The closed check used Qwen3-VL 4B instead
   (`qwen3vl_npu`, one of the eight): it runs, and against the binary without the change it
   prefills at 473.7 / 505.2 vs 508.7 / 493.6 tok/s at 1k, 554.4 / 585.6 vs 593.8 / 591.8 at
   2k, decode 16.7-17.8 both ways. That's within run-to-run spread, or at worst a few percent
   on a 2-second closed prefill.
3. **Pass.** `oflm-test --llm` through `oflm serve` (35B, q4_1 set, nothing in the environment): PASS 5 of 5, no warning in the server log.

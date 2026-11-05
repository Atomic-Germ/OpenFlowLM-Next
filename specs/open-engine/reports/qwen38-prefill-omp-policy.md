# Qwen3.8-27B prefill: the OpenMP wait policy was never applied (PR #2)

**Verdict: 15% off prefill TTFT, bit-exact output.** One cause, found by measuring rather
than by guessing, and it is a bug that had been shipping quietly.

---

## 1. What PR #1 pointed at, and why it was wrong

PR #1's per-layer profile attributed **3 047 ms** of a 35.0 s prefill block to `gemm_tile_ms`,
i.e. `host::tile_x` — 49.5% of all host time. That made "rewrite the tiling loop" the obvious
PR #2.

Measuring the function itself, before rewriting it
([`tile-x-bench.cpp`](../../../utilities/prefill-block/tile-x-bench.cpp)), killed that plan:

| path | qkv\|z (K=5120) | down (K=17408) |
|---|---|---|
| shipped scalar loop → host memory | 0.055 ms | 0.21 ms |
| shipped scalar loop → mapped XRT BO | 0.10 ms | 0.35 ms |
| AVX2 rewrite → host memory | 0.055 ms | 0.21 ms |

A 256-token block makes 256 calls, so at those rates tiling costs **~30 ms**, not 3 047 ms —
two orders of magnitude apart. And the loop is already at memory bandwidth: the AVX2 rewrite
gives **nothing**, because the compiler vectorises `f32_to_bf16` (three integer ops) fine.

So the profile was not measuring the tiling. It was measuring the tiling *plus 100×*.

## 2. The actual cause

`gemm_tile_ms`'s window starts before `tile_x` and ends after it. What else lands inside it:
`tile_x` is a `#pragma omp parallel for`, and the box's own notes (in `core.cpp`, from the
2582-token work) record that **a spinning OpenMP worker throttles the NPU** through the shared
package power budget. When the workers spin, the host stage in front of every dispatch costs
several times what the work costs.

The engine already knew this and tries to fix it:

```cpp
struct OmpWaitPolicy { OmpWaitPolicy() { ... setenv("OMP_WAIT_POLICY", "PASSIVE", 0); } };
const OmpWaitPolicy g_omp_wait_policy;
```

On Windows that works, because `vcomp` is delay-loaded and reads the variable at its first
call. **On Linux it does nothing.** libgomp reads `OMP_WAIT_POLICY` in its ELF constructor,
which runs when the library is mapped — before any constructor of the program. The variable
then reads back correctly, so the log line has said `OMP_WAIT_POLICY=PASSIVE` for as long as
it has existed while the workers spun.

## 3. The fix

`open_qwen36::reexec_with_passive(argv)`, called at the top of both entry points
([`cli.cpp`](../../../src/open_qwen36/cli.cpp) and [`src/main.cpp`](../../../src/src/main.cpp)),
before anything touches the device or OpenMP:

- if `OFLM_OPEN_OMP_REEXEC` is set, return — one hop only;
- if `OFLM_OPEN_OMP_PASSIVE=0`, return — the documented opt-out still works;
- otherwise `setenv` the policy, set the guard, and `execv("/proc/self/exe", argv)`.

The second process is one libgomp always reads correctly. Windows keeps the constructor and
returns from this function immediately.

**The guard is `OFLM_OPEN_OMP_REEXEC` and must not be `getenv("OMP_WAIT_POLICY")`.** The static
initialiser above has already set that variable in this very process; a check on it returns
early and the fix silently does nothing. The first version of this patch had exactly that bug,
measured 34.5 s instead of 29.4 s, and is now pinned by
[`test_omp_reexec.py`](../tests/test_omp_reexec.py).

## 4. Result

Same binary, same prompt, alternating (`OFLM_OPEN_OMP_REEXEC=1` disarms the re-exec):

| | total | `gemm_tile` | host | GEMM dispatches |
|---|---|---|---|---|
| A: policy not applied | 34.83 s | 3 001 ms | 5 788 ms | 24 804 ms |
| B: re-exec (the fix) | **29.37 s** | **498 ms** | 1 940 ms | 24 306 ms |
| C: variable set by the caller | 29.47 s | — | — | — |

**−15.7% on one 256-token block.** B and C agree to 0.3%, which is what makes B the fix and
not a coincidence. The GEMM dispatch time barely moves (24.8 → 24.3 s, i.e. the array is no
longer being throttled by the host's spinning) — the whole saving is in the host stages.

Through the shipped `oflm`:

| TTFT | before | after | Δ |
|---|---|---|---|
| 256 tokens | 38.37 s | **32.87 s** | **−14.3%** |
| 512 tokens | 73.36 s | **62.68 s** | **−14.6%** |
| 1024 tokens | 141.93 s | **123.23 s** | **−13.2%** |

Decode is untouched, as expected — the decode chain has no host parallel regions to wake:
**2.083 tok/s** against 2.074 before.

## 5. Correctness

| check | result |
|---|---|
| greedy output, policy off vs on | **8 ids byte-identical** (64-token prompt, block route) |
| greedy output, fix vs pre-fix | **16 ids byte-identical** (256-token prompt) |
| block boundaries 255 / 256 / 257 | one block / one block / two blocks — the policy's own prediction |
| block boundaries 511 / 512 / 513 | two / two / three blocks |
| prefill shape checks | PASS, all (linear in blocks, flat in active rows) |
| `pytest specs/` | **2 512 passed, 172 skipped, 0 failed** |
| `ctest` (`src/open_qwen36`) | **4/4** |
| shipped `oflm` build (Docker path) | PASS |

The policy changes no arithmetic — it changes when a worker sleeps. A run where the workers
spin and a run where they sleep produce the same numbers; that is why every comparison above
is on tokens, not on timings.

## 6. Rejected experiments

| hypothesis | how it was tested | verdict |
|---|---|---|
| `tile_x`'s scalar inner loop is the cost | isolated benchmark vs an AVX2 rewrite on the real shapes | **rejected** — 0.055 ms/call either way, already at memory bandwidth |
| writing into a mapped XRT BO is the cost | same loop into host memory vs into a real BO | **rejected** — 1.6–1.9×, not 100× |
| fewer threads would help | `OFLM_OPEN_OMP_THREADS=1` with the policy applied | **rejected** — 31.5 s against 28.8 s at the full count; the note in `core.cpp` already says the ordering reverses once the workers sleep |
| setting the policy later, in `main()` | re-exec as the mechanism, measured through the engine | **confirmed** — but only with the `OFLM_OPEN_OMP_REEXEC` guard, not a `getenv("OMP_WAIT_POLICY")` one |

## 7. Limitations

- The 256-token A/B is **single-run per arm**. The 15% is far larger than the run-to-run
  spread seen on this box (under 2% between repeats of the same arm), and the shipped-`oflm`
  numbers reproduce it at three lengths, but this is not a five-repetition campaign.
- Thermal state was not recorded. The NPU was in `performance` mode throughout, and the CPU
  governor was `powersave` — unchanged between arms, so it does not bias the comparison.
- The fix applies the policy to the **whole process**, including any future host stage that
  would prefer spinning. That is the right default here (the engine's own notes measure
  PASSIVE as better for a prefill block), but it is a process-wide decision, not per-region.
- `OFLM_OPEN_OMP_PASSIVE=0` remains the escape hatch, and it now actually does something on
  Linux.

## 8. Next

The profile's remaining host time is now ~1.9 s of a 29.4 s block (6.5%), and the GEMM
dispatches are 24.3 s (83%). The next target is the array, not the host: PR #3's Q4 GEMM work.

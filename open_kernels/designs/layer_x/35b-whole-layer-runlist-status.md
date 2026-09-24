
## 2026-09-24 (later): the merged `lax` needs NO runtime queue map — a single compiled table works for BOTH layer types

The blocker recorded above ("one emitter core serving both layer types ... no single compiled table
can be correct for both kinds") is **wrong**. The merged `lax` design has ONE physical set of
`w{c}` fifos, shared by the `lxf` and `axf` control texts, so ONE channel map is correct for both.
`check_ondv_channels.py <lax build>` confirms it: `0:0x1d21c 1:0x1d214 2:0x1d21c 3:0x1d21c
4:0x1d21c 5:0x1d21c 6:0x1d214 7:0x1d214` (= `qmap_lax.bin`). Setting `ondv_ctrl.h`'s
`kOndvQueue` to exactly that map and building both `lax` kinds (1600 B emitter text, the same
size as the completing `lx` build) gives:

```
run lxf ... -> state 4 (4.526 ms)      # linear-attention whole layer, on-device experts
run axf ... -> state 4 (2.226 ms)      # full-attention whole layer, on-device experts
```

Both complete (state 4) with their routed experts retargeted and enqueued ON-DEVICE, in ONE
xclbin. The runtime `cfg[2+col]` queue read is unnecessary, which also retires the whole
`cfg[2..9]` delivery contradiction. (Confirmed independently: the guarded-read build fed `ax`'s
map still completes, i.e. the guard fell back and `cfg[2..9]` is genuinely not delivered. And a
branchless emitter variant that read the mask from `cfg[0]`'s low bits TDR'd with byte-identical
emitted words — the emitter core is compiler-sensitive, but the compiled table removes the need.)

### The runlist now mixes both layer types — with per-run buffers

`xrt::runlist` cannot mix an ELF (`kernel`) and a classic (`kernelx`) kernel: that fails
immediately with `ERT_CMD_STATE_NEW`. Use `kernel` (ELF) for both control texts. The two `insts.elf`
streams (built `LAX_KIND=0/1`) register against ONE `xclbin X` and submit in ONE runlist:

```
runlist_add rl lxf pool rx0 consts kvD act ptD st0 cfg
runlist_add rl axf pool rx1 consts kv1 act pt1 st1 cfg
...
```

Two failure modes had to be removed, both **buffer races**, not expert/emitter bugs:

* sharing `xres`/`act` across runs times out (`l a a`, `l l a a`, and the 3:1 token pattern);
  per-run `xres` (and a per-run `state`) fixes it;
* `a a a` (three full-attention in a row) also times out; per-run buffers fix the pattern cases.

With per-run `xres`+`state` (and per-ax per-run `kv`/`ptab`) the 35B's own 3:1 prefix submits in
ONE runlist:

```
# l l l a l l l a (the model's layer order prefix)
runlist_exec rl [8 runs, 0 reporting completed] -> ok (17.475 ms)
```

~2.2 ms per whole layer, ~10 tok/s class, on ONE submit, with on-device expert routing. The
HOST_PUSH builds (`build_laxhp_*`) fail the identical patterns, so the limit is the control-text
switch / shim state, not the emitters.

### The remaining blocker: the runlist deadlocks at 9+ whole-layer runs

`runlist_exec` completes at 8 runs and times out (`ERT_CMD_STATE_TIMEOUT`) at 9, for both the ELF
and the classic path, for all-`lxf` chains, and even when every buffer is per-run (so it is not
buffer sharing). `xrt::runlist`'s own chunk size is 24 (`xrt_kernel.cpp`:
`static constexpr size_t submit_size = 24`), so this is not an XRT length cap; and the dense model
(~28-32 layers) reaches its one-submit/token through the same API, so chains of >8 do run on this
box. The 9-layer deadlock is therefore **`lax`-specific**: something accumulates per whole layer in
the shim/ERT state (the most likely candidate is the per-layer control `Pipeline.configure` /
packet BDs that are configured once and never drained — exactly the "shim task-queue overflow"
lead in `LAX-MERGE-NOTES.md`). Next step: bisect what is left behind after one whole-layer run
(dump the shim TileControl/BD state, or add a per-layer drain/barrier) so layers can chain; then
the 40-layer token is ONE submit.

### 2026-09-24 (cont.): the 9-layer limit is ~8 ERT command slots PER hw_context

Decisive experiments (all on an idle device, per-run buffers, `lax` ELF kernels):

| test | result |
| --- | --- |
| 8 runs, one runlist | ok (17-20 ms) |
| 9 runs, one runlist | TDR |
| 8 runs (r0) then 1 run (r1), same process/context | r0 ok, **r1 TDR** |
| 4 + 4 + 4 runs, same context | first two ok, **third TDR** |
| 8 runs, then a **plain `run`** (not a runlist) | plain run TDR |
| 8 runs on context **X**, then 2 runs on a **second context Y** (same xclbin) | both ok |

So the device is not wedged and the design is not leaking shim state: each whole-layer
execution consumes **one ERT command slot** from its `hw_context`, and the context has ~8.
XRT's `get_ert_slots()` (`xdna-driver/xrt/.../common/device.cpp`) computes
`slots = max(min_slots, num_cus*2+1)` then `size = max(cq_size/slots, max_cu_size)`,
`slots = cq_size/size` — i.e. the **per-command size (`max_cu_size`, ~ the 74 KB control
text) caps the slot count at ~8**; the `Runtime.ert_slotsize` xrt.ini override (read, since
`verbosity` takes effect via `XRT_INI_PATH`) does not raise it for the xdna path.

Consequence for the objective: a 40-layer token cannot be 40 runs in one `xrt::runlist`
(the hw_context runs out of slots at 8). The runtime's per-ctx ELF route reached 40 runs
only because its xclbin's command size left more slots. Two ways to one submit/token:

1. **One whole-token control text** (all 40 layers, per-layer buffer offsets): one `run` =
   one command = one submit, no slot pressure. This is the robust route.
2. Raise the context's command capacity (smaller `max_cu_size` / larger CQ — driver or
   xclbin level), so a 40-run runlist fits.

## 2026-09-24 (lax decode): the whole 40-layer token, real weights, parity PASS, ONE submit

The merged `lax` kernels now run the real Qwen3.6-35B-A3B decode: all 40 whole layers (30
linear + 10 full attention, the model's 3:1 order) with the routed experts retargeted and
enqueued ON-DEVICE, in ONE xclbin, as **ONE `xrt::runlist` submit on ONE hw_context**, and
the token's logits match the fp64 reference (`model/compare_decode.py`):

```
runlist_exec r0 [40 runs] -> ok
token 0 (position 0): logits corr 0.999998  argmax ours 846 ref 846
  top5 ours [846, 8678, 1421, 2244, 1785] ref [846, 8678, 1421, 2244, 1785]      PASS
```

Three tokens (positions 0..2, KV cache + DeltaNet state carried, `attnpos` per token) all PASS
(corr 0.999998 / 0.999993 / 0.999998, argmax 846 / 198 / 3710 = the reference), and the same
run as five 8-layer submits (`--per 8`) gives identical results. Speed, steady state: 75.8 ms for
the 40 layers in ONE submit, 86.4 ms with the final norm + lm_head = **11.6 tok/s** (16.5x the
~0.7 tok/s starting point); five submits 85.7 / 96.3 ms = 10.4 tok/s. Evidence:
`1bit-MONSTER-goal/benchmarks/RESULTS-lax-decode-35b-2026-09-24.md`.

### Correction: nothing before this was a decode

The earlier "(cont. 12) COMPLETE 40-layer MoE decode (5 submits, ~11.6 tok/s)" ran ONE
`insts.bin` (N=8, one layer's text x8, no per-layer offsets) five times over UNFILLED buffers
(`/tmp/t_5ctx.cfg` never loads pool/consts/act). It measured throughput of layer-shaped work;
it computed nothing, and no `lax` layer had ever been checked against real data. Run on real
layer-0 weights, every `lax` build returned NaN. Three bugs, all fixed on branch `lax-decode`:

1. **The emitter read the router output as its pool base** (`lax.py` emitter body). It took
   the rout and cfg x elements as `xin.acquire(1)` twice. Object-fifo acquire counts are
   cumulative: the second `acquire(1)`, while one element is held, returns the SAME element,
   so `cfg` aliased `rout` and the base was the first two router probabilities
   (`0xa47b3abcc180`: exactly the IO_PAGE_FAULT addresses). Found with an echo of what the
   emitter actually read (`ONDV_DBG=1`, `designs/router/ondv_echo.cc`, drained to the unused
   kv argument). Fix: `rc = xin.acquire(2); r, cfg = rc[0], rc[1]`. This is also the whole
   story behind "cfg[2..9] is not delivered to the core" and "act[A_ROUT+2048] is not what the
   core reads" above: the core never looked at the cfg element.
2. **The shared expert's fills raced the last routed down** (`ONDV_EMIT_SHARED=1`). The host
   sequence enqueues the shared expert's up|gate as soon as it has awaited the last routed
   wave's hidden drain, i.e. possibly BEFORE the emitter pushes that wave's down; all are
   81920 B on the same MM2S queue, so nothing hangs and the core pairs the wrong weights. Now
   the emitter pushes the shared expert too (a ninth slot in `ondv_ctrl.h`, two more chunk BDs)
   and the host does not. Without it: corr 0.99889 (varies run to run, below the 0.9999 bar).
3. **The "8 whole layers per hw_context" cap was a lock overflow, not a CERT/TXN budget**
   (`ONDV_PKTDONE_ACQ=1`). Every chunk BD of the emitter's control stream releases `pktdone`
   and nothing acquired it, so the lock only counted up (one per chunk per layer) and AIE2
   locks are 6-bit. The emitter now takes the chunks' releases back once per layer. Measured on
   real layers, one context: 7 runs ok / 8 TDR before; 8, 9, 16, 24 and **40 runs ok** after.
   So cont. 14-17's "~16K per-context TXN op budget" reading is wrong: it matched the op count
   because both scale with layers.

Everything before the MoE block in `lax` was already right: on real layer 0, xn, qkv, z, the
DeltaNet in/out, og, out, the post-attention residual, xm and the router output (and its
top-8) all match the proven `lx` path at corr 1.000000, and the shared expert's hidden matched
wherever its column's stream was not corrupted by the race.

### Build and run (branch `lax-decode`)

```bash
source ~/ironenv142/bin/activate
export PATH=/home/bcloud/Xilinx/2026.1/Vitis/aietools/bin:$VIRTUAL_ENV/bin:/usr/bin:$PATH
unset PYTHONPATH
cd open_kernels
export OPEN_KERNELS_SPEC=recipes/specs/qwen36-35b-a3b.json MOE_ONDEVICE_ROUTE=1 \
       ONDV_EMIT_SHARED=1 ONDV_PKTDONE_ACQ=1
LAX_KIND=0 python build_design.py designs/layer_x/lax.py OUT/lax_l
LAX_KIND=1 python build_design.py designs/layer_x/lax.py OUT/lax_a
# weights + fp64 reference (all-q4_1, the lax builds' spec); pools are 40 x 512 MB
python3 model/make_decode.py --requant --layers 40 --tokens N --out DEC --pool-dir POOLS
python3 model/lax_decode_cfg.py --out DEC --lax-l OUT/lax_l --lax-a OUT/lax_a --per 40 --tokens N
harness/build/run_kernel DEC/run_lax_p40.cfg
python3 model/compare_decode.py --out DEC --tokens N
```

`insts.bin` md5: lax_l `31244de2088ccd8dac27df72afdd0c29`, lax_a `7025eebc7663feb97cdebf6cfef7b2fb`.

### (lax decode, cont.) Greedy generation on the NPU: "The capital of France is Paris."

`model/lax_decode_cfg.py --prompt-ids ... --gen N --embed EMB` writes a generation program: each
position feeds its token's embedding (`feed`, harness), runs the 40 layers as ONE submit, and
where a next token is wanted runs norm + lm_head and takes the argmax (`greedy`), which the next
position feeds back (`feed xres embed last`). `EMB` is the q4nx's bf16 `model.embed_tokens.weight`
as a raw file.

Prompt (chat template, thinking off, 24 tokens): `What is the capital of France? Answer in one
sentence.` Output: `The capital of France is Paris.<|im_end|>` (the harness has no stop
condition; what follows the end-of-turn token is the model continuing past it).

Wall time per position, steady state: prompt 76.4 ms (no head), generated **89.9 ms median
(89.2-90.2) = 11.1 tok/s end to end** (feed + one submit + norm + lm_head + argmax readback).
The first position of a process is cold (8.8 s: first touch of the ~21 GB of BOs). 0 faults.

### (lax decode, cont.) Chat on the NPU: `model/lax_chat.py`

`lax_chat.py` drives one persistent harness session (`run_kernel -` reads its program from
stdin), so the ~21 GB of weights load once (10 s) and the KV cache and DeltaNet state carry
across turns; each turn feeds only its new tokens. Greedy, thinking off, a turn ends at
`<|im_end|>`. A three-turn session:

```
Q: What is the capital of France? Answer in one sentence.
A: The capital of France is Paris.                              7 tok, 11.1 tok/s
Q: And what is its population, roughly?
A: The population of Paris is approximately 2.1 million people within the city proper,
   though the larger metropolitan area is home to around 12 million people.   32 tok, 11.1 tok/s
Q: Write a haiku about a neural processing unit.
A: Silicon mind wakes, / Parallel paths light the way, / Data flows like rain.   18 tok, 11.0 tok/s
```

Prompts run at ~12 tok/s (no head per prompt token). Harness additions: `feed`, `greedy`, `tick`,
`stopat`, and the stdin program mode.

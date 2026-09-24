
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

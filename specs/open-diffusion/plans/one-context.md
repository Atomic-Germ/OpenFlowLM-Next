# Plan: the whole pipeline in one hardware context

Status: **for review (2026-09-29).** Nothing below is implemented in the engine yet. The
measurements come from probes in `utilities/dit-chain/switch_probe.py` and
`utilities/reconfig-probe/`.

## Why

A 512² image changes kernel set 822 times, and each change costs 2.0-2.4 ms: ~1.8 s per
image at either size, **~34% of 512² and ~13% of 1024²**. The probes split that cost:

| | cost |
|---|---:|
| host round trip (start → the wait returns), same context | 0.005-0.16 ms |
| the same, with the Windows timer at 1 ms | unchanged |
| a context switch between two xclbins | 2.0-2.4 ms, whatever the op or the set |
| `load_pdi` inside one full-ELF context | ~2.0 ms (no better: it is PDI loading) |
| register-write reconfiguration (`aiecc --expand-load-pdis`) into gemm / ew / fa | **0.30 / 0.51 / 0.70 ms** |
| control-packet reconfiguration (`--load-pdi-to-ctrl-pkt`) | does not build: no free shim DMA channel |

So the time is spent writing the array's configuration, not on the host. Register writes
in the instruction stream are 3-7× cheaper than the firmware's PDI load.

**The prototype** (`utilities/reconfig-probe/fullelf_generate.py`) composes every
set's built streams into one full ELF, with one device per set.
- Each stream is its device's own kernel, `<set>:<stream>`.
- Each set gets a configure-only kernel, `main:cfg_<set>`.
- The replay issues `cfg_<set>` only where the set changes, and queues everything on the
  one context.

At 512² with the text encoder skipped (`--ctx-ref`), it ran 831 dispatches plus 632
configurations:
- **3.51-3.54 s against 4.53 s** for `generate.py` on the same inputs;
- **latents and PNG byte-identical**, 3 runs.

## Done when

- `oflm image` holds **one** hardware context for every resolution and step count.
- Its pixels are byte-identical to `generate.py`'s on the same ids and noise.
- It is faster on a quiet machine (OPEN-DIFFUSION-PERF). The estimate is ~5.2 → ~4.0 s at
  512² and ~13.6 → ~12.4 s at 1024². The remaining switch cost is ~0.3 s.

## Spec impact (`specs/open-diffusion/spec.md`)

No new requirement IDs.

Modified:
- **OPEN-DIFFUSION-NPU-ONLY.**
  - "patch `te_attn`'s `valid_len`" becomes "set `te_attn`'s `valid_len` parameter". The
    mechanism is decided in step 2.
  - "The host queues runs within one kernel set and blocks, without polling, on the last
    run before switching sets" becomes: "Every run goes to one hardware context, queued.
    The host blocks, without polling, only at phase boundaries, when its window of runs
    in flight is full, and on the image's last run."
- **OPEN-DIFFUSION-DETERMINISM.** It already says "the same pixels as the pyxrt runner",
  but no test checks that. It gains an acceptance criterion and a test:
  - the engine's PNG for study prompt 0 at 512² (fixed ids and noise) is byte-identical
    to `generate.py`'s;
  - `specs/open-diffusion/tests/test_engine_matches_pyxrt.py` checks it (needs the NPU
    and a built kernel directory, and says which is missing when skipped).

  This is the guard for this change. A wrong reconfiguration would be silent, and it can
  be checked deterministically.
- **OPEN-DIFFUSION-PACKAGE.**
  - The installed kernel set becomes one full ELF (`diffusion.elf`) plus its manifest,
    format `oflm-open-diffusion-kernels-v2`.
  - `export_dit_kernels.py` builds the ELF after the streams.
  - `--install` ships the ELF and not the six sets' xclbins.
  - The engine refuses a v1 set, naming the format.
  - **The model directory does not change.** The layout hash covers the stream specs and
    the weight packing, and neither moves, so `Cyronius/FLUX.2-klein-4B-NPU2` needs no
    re-upload.
- **OPEN-DIFFUSION-PERF.** New measured table, same rules.

Nothing is removed.

## Steps

### 0. Branch

`feat/one-context` from `origin/feat/images-api`, untied
(`git switch -c feat/one-context --no-track origin/feat/images-api`). It becomes a third
stacked PR on #137, because the engine to change is #137's (one `Engine` for every
resolution, 1-50 steps). The first commit adds the probes and a
`utilities/reconfig-probe/README.md` with the tables above.

### 1. Build the whole set as one ELF, assembled from separate builds

The prototype compiled one composed module: 67 streams took **671 s and ~10 GB of RAM**,
9.5 min of it in MLIR passes before any core compiled. Two aiecc behaviours scale with
the whole module's size (runtime sequences are almost all of it):
- **`AIEAssignBufferAddresses` walks every op of the device once per tile**
  (`AIEAssignBuffers.cpp:155`, `:572`, inside the per-tile loop at `:714`). On gemm-only
  modules it went 1.13 s → 7.39 s for 1 → 4 streams, 62% of the address pipeline.
- **The per-core compile clones the whole module once per core** (`Actions.h:159`): all
  devices and all sequences, ~160 times at 512².

A per-stream build has one device and one sequence, so neither bites there. The ELF is
therefore assembled, not compiled as one module:
- **One configuration build per set** (a device with one sequence plus `main:cfg_<set>`):
  the PDI and the cfg control code. ~6 s for gemm.
- **Each stream's full-ELF control code, built alone.** In parallel, a few seconds each.
  It could be a by-product of the exporter's existing per-stream builds.
- **`full_elf_config.json` written by us, then `aiebu-asm -t aie2_config`.** This is
  aiecc's own last step (`aiecc.cpp:1544` shells out to it).

Checked: a solo build's `gemm.pdi` and `t_emb1` control code are byte-identical to the
67-stream build's. The `cfg_gemm` code differs in one byte, the PDI id, which depends on
device order. So the assembler fixes one PDI numbering and builds each set's cfg against
it, or patches that byte.

The expected build is minutes for all 141 streams, with normal RAM, cached by input hash
as the exporter does per stream. One ELF for every resolution stays the default. One
ELF per resolution is now only a fallback if context creation or ELF size misbehaves.

### 2. `te_attn`'s `valid_len` (decision gate)

Today the host patches 32 instruction words per prompt. The instructions now live inside
the ELF, so that stops working. In order of preference:

1. **A scratchpad parameter.** mlir-aie's `ScratchpadParameter`; XRT 2.21's C++ has
   `run::get_ctrl_scratchpad_bo()`, pyxrt does not. The `apply_mask` RTP becomes a
   parameter that the core reads. First a C++ probe on a one-core design, then `dit_fa`'s
   text mode.
2. **A scalar kernel argument patched by XRT at `set_arg`,** if aiebu emits scalar patch
   entries for a runtime-sequence i32 argument. Checked by the same probe.
3. **Fallback:** the text encoder stays on its xclbin contexts. The engine drains once
   between the text phase and the conditioning. This keeps ~90% of the saving (the text
   phase is ~110 switches, ~0.24 s), but holds four contexts instead of one.

The step ends with a decision recorded here before step 3.

### 3. The engine (`src/open_diffusion/engine.cpp`)

- Load `xrt::elf(diffusion.elf)` into one `xrt::hw_context`. Create an `xrt::ext::kernel`
  per stream (`<set>:<stream>`) and one per set (`main:cfg_<set>`). Buffers become
  `xrt::ext::bo`.
- The op list gets a `cfg` entry before each op whose set differs from the previous
  op's. This includes step and resolution boundaries, because the previous image may have
  ended on another set.
- Runs are queued with a window. XRT's per-context queue limit decides its size; the
  probe used 64. The host blocks at phase boundaries (7 per image, ~0.1 ms each), so
  `Timing.phases` keeps its meaning. `--profile` still waits after every op and charges a
  `cfg` to the op after it.
- `valid_len` follows step 2's decision.
- The six-context path is deleted. `generate.py` (pyxrt, xclbins) stays as the reference
  implementation that the new test compares against. OFLM's other open engines already
  load ELFs (`xrt::elf` → `xrt::ext::kernel`), so the driver requirement is not new. If
  context creation fails, the error still names the driver.

### 4. Packaging

- `export_dit_kernels.py --install` copies `diffusion.elf` and writes the v2 manifest
  last: the ELF, its kernel names, the layout hash.
- `kernels_usable` / `find_kernels` read v2.
- The xclbins root in the installer gets the ELF instead of the six set directories.

### 5. Verify, measure, write back

- New test `test_engine_matches_pyxrt.py`. The existing ones must pass:
  - `tokens_test` and `schedule_test`;
  - `specs/open-diffusion/tests/`;
  - the 31 Images API integration tests (with `OFLM_MODEL_PATH=C:\Users\josha\.flm`).
- OPEN-DIFFUSION-PERF on a quiet machine in turbo, 4 runs each at 512² and 1024², plus
  host CPU per image.
- Merge the spec changes, update `.opencode/skill/open-diffusion`, and add a skill for
  composing and building the full ELF (AGENTS.md: document each new kernel build).
  Archive this plan.

## Risks

- **Assembling the ELF ourselves** (step 1) depends on aiecc's `full_elf_config.json`
  format and aiebu-asm staying stable across toolchain updates. `toolchain.json` already
  records the versions. A byte-identity check against a small aiecc-built ELF catches
  drift.
- **`valid_len`** (step 2). The fallback keeps most of the gain.
- **Queue depth.** ~1,650 runs per image instead of ~1,050. The window bounds what is in
  flight; the probe ran 64 without trouble.
- **Reconfiguration leaving state behind.** Register writes do not reset the array the
  way a PDI load does. Whole images came out byte-identical 3 times, and the new test
  keeps checking.
- **Unmeasured side effect:** `--image 1` holding one context instead of six should make
  residency beside `--asr` / `--embed` easier. It will be checked in step 5, not claimed
  now.

## Not in this plan

- **Shrinking each set's configuration** (`ew` writes ~340 KB, `gemm` ~180 KB). It is the
  lever on the remaining ~0.3 s, a follow-up once this lands.
- **Phase 7's fusions.** A switch now costs ~0.4 ms instead of ~2.1, so fusing the
  elementwise ops into gemm to avoid switches is worth ~0.2 s, not ~1 s. The `qk`
  epilogue and bfp16 activations still stand on the work they remove.

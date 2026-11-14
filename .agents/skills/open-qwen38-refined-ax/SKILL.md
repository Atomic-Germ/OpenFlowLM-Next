---
name: open-qwen38-refined-ax
description: Build private full-ax NPU projection refinement with quantization-residual tables and FP32 band sums, then compare 240 causal positions with exact native and staged controls.
---

# Integrated L3 projection refinement

Read `utilities/l3-refined-ax/README.md` for complete commands and acceptance
contracts. This builds a private full-ax control and candidate from the current
recipe; it does not change the production export or installed runtime.

## Build and run

1. Source `utilities/env.sh`; set `OPENBLAS_NUM_THREADS=1`. Use the tree's
   `ironvenv/bin/python`. Keep all artifacts in fresh directories under Models.
2. Run `utilities/l3-refined-ax/build.py --role control --spec <spec> --out <control>`.
3. Run the same builder with `--role candidate --out <candidate>`.
   Both roles derive private generated TUs. Every compiler dependency, copied
   TU, linked ELF and allocated MLIR must pass provenance/resource checks.
4. Prepare `replay.py prepare --baseline <audited l3-tail-sweep> --control-build
   <control> --candidate-build <candidate> --out <fresh run>`.
5. Run `replay.py run --out <run> --harness <audited run_kernel>` under the
   tool's existing exclusive device lock, then `replay.py audit --out <run>`.
   No full-model residency is required: one 232,783,872-byte L3 pool is loaded.

## Arithmetic and memory

- Keep the original six runtime arguments, scratch layout, DMA sequence and
  position-table patching. Never add a second ptab transfer or relax attnpos.
- On the main cores, prepare the high INT16 table from native FP32 norm with
  production consistent activation sums. Reconstruct its INT16 activation and
  subtract on the NPU with private integer IEEE binary32 RNE arithmetic;
  prepare the low table with the same production helper.
- Use a 23,040-byte table buffer: high table at zero, low at 11,520. The second
  table is dead after Q/gate/K/V; later production preparation may overwrite it.
  Keep one separate 64-float low-band buffer. Never alias it with live y.
- Both projection passes call the SAME `gemv_q4_gy` entry. `refine_low.cc`
  forwards to that symbol with a table offset, preserving accumulation order.
  Consume each streamed weight once, evaluate high and low, then release it.
  Sum the two completed bands on the NPU with the same exact integer RNE
  helper. Vector accumulator subtraction/addition is not byte-exact here.
- Keep output projection, FFN and attention workers unchanged. Do not supply
  refined projections or OG externally; this experiment tests integration.

## Causal and numerical controls

Each role runs 240 positions from zero with its own persistent NPU KV. Restore
original current layer inputs and preceding scratch, but never restore native
KV between candidate positions. This is a single-layer causal replay, not
cross-layer propagation or a full-model generation run.

Require 720 native byte-exact control captures. Then require 960 refined
projection vectors, 240 complete scratch buffers, 240 KV prefixes and 240
residuals to match staged evidence exactly. Build expected scratch from the
audited forced-tail capture, replacing only Q/gate/K/V/OG with independently
audited refined values. Preserve all other bytes.

Recompute residual/head measurements against retained independent references.
Output/FFN sublayer gates transfer through byte-exact complete scratch/residual
identity to the audited sweep; this does not re-dequantize their weights.
Keep the primary and full-model reference regression lists separate; retain
P46/P148 and P239. Execution completion does not override numerical FAIL.
No full-model acceptance, request isolation or performance claim follows from
this local experiment. Changing pinned source requires fresh build/preparation.


## Rounding probe and retained result

The first integrated build failed exact projection identity. The isolated
`probe.py` / `diagnose.py` workflow in the README separates entry-table
preparation from final projection addition. It uses the actual candidate
helpers, consumes no model weights, and packs high/low into one FIFO to stay
within the two input-DMA-channel limit. Include the production GEMV header
path. Retain unsuccessful builds; do not relabel them or weaken controls.

Original vector helpers matched 240/240 high tables, 162/240 low tables and
2/240 complete sums (1532 differing elements). `add_rne.h` fixes both residual
subtraction and final addition using guard/round/sticky bits, cancellation,
subnormals and signed zero. Host C++ tests compare against the host FPU.

Corrected probe `l3-refined-probe-v4` / `l3-refined-diagnosis-v2` matches all
240 high tables, low tables and sums. Separate audit binds 480 captures.
Probe resource audit: one core, .text 4400, L1 end 34,304, stack 6144;
813 compiler inputs. Earlier source snapshots are retained under
`l3-refined-initial-sources` for diagnosis history only.

Successful full-ax builds: `l3-refined-ax-control-v2` and
`l3-refined-ax-candidate-v2`; run/audit: `l3-refined-ax-v2`, all under
`Models/qwen38-27b/post-prefill-drift/`. Both 15-core builds have identical
instruction streams, maximum .text 13,760 and L1 end 59,920. Candidate main
cores use 12,528 B text and L1 end 54,016; stacks remain 6144. Control/candidate
pin 858/868 compiler inputs.

All 480 dispatches, 720 native controls and 960 candidate projections pass
exact identity; all 240 candidate scratch/KV-prefix/residual captures match.
Independent audit binds 1440 captures and recomputes residual/head metrics.
All 5760 primary and conditioned candidate heads and all 240 local residual
cases pass. The primary 178/61/1 and model-reference 155/83/2 improvement/
regression/tie counts are preserved. Fourteen new TDD tests; full host suite
2057 passed, 47 skipped. Next validate full-depth propagation in a separate
runtime build. Local integration does not establish model acceptance.

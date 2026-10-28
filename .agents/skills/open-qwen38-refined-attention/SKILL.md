---
name: open-qwen38-refined-attention
description: Build current production L3 attention replay and validate two-pass refined projections with byte-exact native controls and causal paired-BF16 KV over 240 positions.
---

# Refined projections through current native attention

Use `utilities/l3-attention-replay/` after the separately audited two-pass
`utilities/l3-projection-replay/refine.py` experiment. Read its README for
commands and numerical results. This is an isolated attention experiment;
it does not integrate the projection correction into production full-ax.

## Required workflow

1. Source `utilities/env.sh`; set `OPENBLAS_NUM_THREADS=1`.
2. Build `build.py --spec <standard export>/spec.json --out <fresh Models directory>`.
   The design reuses current `xlayer.attn_workers` and `attn_kernels` directly.
   Never reuse the old `attention-kv-replay` probe for this ABI: it requires
   the previous BF16 production format and explicitly rejects paired KV.
3. Run `replay.py prepare --source <audited refinement> --build <build> --out <fresh output>`.
   It validates the upstream refinement report/audit/captures and recomputes
   the actual host FP32 sum. Independent references retain their original
   source/export identities. Do not relabel retained bundles with new keys.
4. Run `replay.py run --out <output> --harness <audited run_kernel>` using the
   existing device lock. No full-model residency is needed.
5. Run `replay.py audit --out <output>` independently. Do not interpret the
   candidate unless 240/240 native OG outputs and 240/240 entire KV prefixes
   reproduce bytewise. Audit completion and numerical PASS are separate.

## Arithmetic and DMA

- NH24/KVH4/HD256/NHL4, six cores, VEXP1/RB1, unchanged production workers.
- One dispatch per trajectory, 240 positions each. Input records: 61,440 B
  (`meta | ptab | Q | K | V | gate`); raw projections stay FP32.
- Paired cache rows: 8192 B (`K_hi | K_lo | V_hi | V_lo`). Every position
  consumes only its earlier NPU-generated cache rows. Position zero reads
  one masked dummy row. Finish DMA groups before advancing the position.
- Production core 0 emits KV then its OG on one FIFO; the other five cores
  emit only OG. This avoids extra debug output or duplicated arithmetic.
- Reset the cache and poison output with NaNs before each trajectory.
- The refined projections include an explicit host FP32 addition of two NPU
  GEMV passes. Do not describe them as fully integrated NPU correction.

## Provenance and resource gates

IRON copies `.cc` translation units into `final.prj`. Verify each copy against
its production source; compiler `.o.d` files must bind actual `attn.h` and
`attn_pair.h`. Pin copied units, all compiler dependencies, source/build keys,
linked ELFs, allocated MLIR, XCLBIN and instruction stream. Do not turn a
failed provenance build into a successful record by editing its evidence;
use a fresh directory after fixing the checker.

Validated `Models/qwen38-27b/post-prefill-drift/l3-attention-build-v2`:
814 compiler inputs; maximum .text 13,760/16,384 B; L1 end 59,920/65,536 B;
stack 6144 B unchanged; logical data BOs 22,609,920 B. These resource results
are for isolated attention, not a full-ax candidate.

## Retained evidence and limitations

Run: `Models/qwen38-27b/post-prefill-drift/l3-attention-v1`.
Prerequisite: `.../l3-refinement-v1`; original native: `.../native-l3-v1`.
Two complete trajectories; 240 exact native OG controls, 240 exact KV prefixes.
Primary and conditioned candidate comparisons pass all 5760 head gates with
unchanged maxabs <0.02 / cosine >0.9999. Primary maxabs falls from
2.720021153e-4 to 5.185089467e-5. P125/P205 improve, but **P239 regresses**
slightly (2.560080227e-5 → 2.574159746e-5); keep it as a distinct cancellation
regression case. The numerical gate passing does not imply every case improves.

Next validate output projection, FFN and complete layer residual with an exact
native control before private full-ax integration. Full-model precision,
long-prefix request isolation and production performance remain open. Never
relax thresholds or claim model support from this replay. No installed model
configuration or running service is changed.

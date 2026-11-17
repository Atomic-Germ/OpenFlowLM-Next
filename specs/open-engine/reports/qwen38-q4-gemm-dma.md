# Qwen3.8 Q4 prefill GEMM: bounded DMA batching

## Scope and review base

This opt-in change groups up to four consecutive DMA visits before awaiting
output drains and reclaiming descriptors. Traversal stays row-group-major,
token-tile-minor; partial batches add no padded work. The compiler remains the
authority for the 16-BD shim limit. Batch values are 1, 2 and 4; activation-reuse
mode refuses batching. **The default remains 1; no production recipe opts in.**

No GEMM arithmetic, FP32 accumulation, paired-product reduction, FIFO depth,
weights, kernel ABI, manifest schema, routing, attention or DeltaNet arithmetic
changes. Ordinary exports clear ambient tuning; the five private replay builders
copy/hash the new planner. No new model support or precompiled assets are shipped.

Review the DMA optimization against the prefill-startup-policy and
manifest-compatibility base `836072db` (branch `perf/npu-prefill-scheduling`).
At the checked upstream main `9b1409d3`, that tip is not an ancestor. Land those
prerequisites first or review against their branch. The broad upstream diff
includes earlier OpenMP and manifest-mapping work and is not the GEMM-only
review range.

## Evidence and machine

The [structured evidence](qwen38-q4-gemm-dma.json) retains source, compiler,
binary, fixture and output hashes plus separate initial/confirmation timings.
Raw captures live under `Models/qwen38-27b/pr3-gemm/` and are not shipping assets.
Build, staging, comparison and server commands are in the
[validation workflow](../../../utilities/q4-gemm/README.md).

Measurements: 2026-11-15/16, Ryzen AI 9 365, approximately 91 GiB RAM, Strix XDNA2
8 columns / 6x8 topology, Linux 7.0.0-43-generic, glibc 2.43, GCC 15.2.0,
amdxdna 2.26.0_20260817, firmware 1.1.2.64. CPU governor `powersave`, NPU power
mode `default`, desktop/services active. Isolated-campaign sampled load was
2.10–3.30; confirmation start load was 0.92 / 0.86 / 1.44. ACPI temperature was
54 C, not NPU die temperature. Load and temperature were recorded, not controlled.
This is evidence for one host/session, not a platform-wide guarantee.

## Kernel and model equivalence

- Five projection shapes, each high and low: N/K 14336/5120, 16384/5120,
  34816/5120, 5120/17408 and 5120/6144. All ten cases have byte-identical outputs
  between batch 1 and 4, with unchanged `rel_fro <= 0.005` FP64-reference gate.
  High errors are 0.001662–0.001730; low errors are 1.00e-7–2.13e-7.
- Five alternating pairs per case; each process excludes one warm-up dispatch
  and measures ten. All 1,100 dispatches complete. Host submit-to-completion
  reductions are 0.84–3.81%; they exclude BO sync and context creation and are
  not device-active or end-to-end timing.
- Twenty builds have identical compute-core images across DMA variants, all
  32 cores checked. Maximum program size 5,568 bytes (<16,384); maximum compute
  buffer end 61,448 (<=65,536); stack reservation 4,096. Driver-owned allocation
  is not measured. N768/K512/T512 exercises six visits in a 4+2 tail, high/low.
- Default batch-1 N14336/K5120/T256 instructions match the pre-edit stream:
  `032ec3405135f5d08e3406b7cd50e26292adc3a4947399b228932602ee111ea4`.
- Full 64-layer comparison: 147 checks, 17 exact logit rows, 112 exact
  conv/recurrent/KV slices (48 linear layers and 16 attention layers), identical
  16-token greedy continuation. This is captured control/candidate equivalence,
  not an independent per-layer residual-oracle measurement.
- Four-layer 95/96/97, 255/256/257 and 511/512/513 boundaries all match.
  Direct production CLI checks pass on both arms. Private arms retain the same
  manifest/head mapping: 20 GEMM binaries differ and 72 other binaries match.

## End-to-end performance

All measurements exclude model loading. Paired relative change and the ratio
of arm medians are different statistics; neither is substituted for the other.

| Prompt | Control median ms | Candidate median ms | Result |
|---|---:|---:|---|
| 256 | 29,432 | 28,759 | -2.29%, five of five pairs faster |
| 512 | 59,365 | 58,186 | -1.99%, five of five pairs faster |
| 1024, initial | 118,743 | 118,029 | -0.60%, four of five faster: **FAIL** |
| 1024, fixed confirmation | 121,395 | 118,727 | -2.20% arm-median ratio; **-2.43% paired median**, 14/14 faster |
| 3008 engine tokens | 377,564.5 | 369,279.5 | -2.19%; two runs per arm, supporting only |

The confirmation uses one excluded warm-up pair and exactly 14 measured pairs,
AB/BA order balanced seven each. Its [predeclared criterion](qwen38-q4-gemm-1024-confirmation-criterion.md)
requires median paired change <=-1%, at least 12/14 faster, a negative 95% CI,
and decode/RSS changes <=+3%. All pass. Median paired delta is -2,978 ms, IQR
-3,683 to -1,984 ms, bootstrap CI -3,683 to -1,984 ms; exact one-sided sign
p=1/16384. Decode median 485 to 481.5 ms/token (-0.72%), RSS +0.001%.

Statistics use per-pair differences and relative ratios. Bootstrap: Python random
seed 20261115, 10,000 resamples of 14 paired deltas; sorted indices 249 and 9749.
The criterion was written on disk before collection, not committed separately
then. Original five-pair evidence remains FAIL. Pair 11's control took 152.9 s;
its cause is unproven and it remains in the primary result. Excluding it only as
a sensitivity check gives 13/13 faster, paired median -2.32%, CI -3663 to -1984 ms.

## Server and runtime provenance

The retained candidate-specific server run selects each private arm through the
engine's own kernel-directory log, passes A->B->A (exact A), matches fresh-process
B and produces matching answers between arms. Each arm has 11 graded checks;
reset is N/A (no endpoint). Main-process live mappings show matching distro
XRT 2.25.00 core/coreutil/xdna plugin hashes. The original generic server run
used production kernels and does not count as candidate evidence.

Historical timed benchmark processes **did not record loaded-library maps**.
Later runs of the same command shape under the tree environment map matching
`/opt/xilinx/xrt` 2.26.0 libraries on both arms. That supports current runtime
comparability but does not retroactively prove the historical processes' exact
runtime. The 14-pair result is retained with this provenance limitation; it is
not claimed as verified XRT 2.25 performance. New samplers follow timing-wrapper
children and hash libraries outside the timed inference window.

No errors were found in the retained server logs. A no-active-contexts report is
not a capacity measurement: the server campaign did not retain before/after
capacity brackets or a full kernel-journal window. No-degradation throughout
that campaign is **not established**. Earlier capacity fluctuated between 16
and 8 with no proven trigger or validated recovery; that separate investigation
is not part of the functional DMA implementation.

## Acceptance boundaries

Numerical captured-boundary equivalence and the fixed performance criterion pass.
Production promotion is not requested. Remaining NOT MEASURED / unverified:
independent per-layer residual oracle, HTTP streaming TTFT, driver-owned NPU
memory, Windows, rebuilt tiled exports, device-active/transfer attribution,
thermal-controlled repetition, verified historical runtime identity and the
server campaign's capacity brackets/journal. The upstream dependency must land
before a standalone upstream merge. Do not start another optimization campaign
to conceal an unmeasured gate or change precision/defaults to bypass one.

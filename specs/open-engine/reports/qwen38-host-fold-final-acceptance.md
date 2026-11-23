# PR4B final acceptance

**Decision: MERGE CANDIDATE. Defaults remain OFF. No push or merge.**

This decision applies to the scoped host-fold optimization on the machine below, not to general model support or a full-context claim.

## Machine and fixed protocol

Measured on prihlop-laptop2 / EVO14-AI9-STP, AMD Ryzen AI 9 365 (20 logical CPUs), Ubuntu 26.04.1 LTS, Linux 7.0.0-44-generic, glibc 2.43, NPU Strix aie2p 6x8.
XRT 2.26.0 (e9db9ab15f10173f8d2fc93ff92ab4c7eb09d2e6), amdxdna 2.26.0_20260817 (2b71d02b641561a40ab2db1c26a69620794f4e9f), firmware 1.1.2.64.
The [committed protocol](qwen38-host-fold-final-acceptance-plan.md) is fixed by commit 037f2fcb. Earlier incomplete observations are not pooled.
Threads: OMP_NUM_THREADS=8, OMP_WAIT_POLICY=PASSIVE, OPENBLAS_NUM_THREADS=1. CLI power mode is none: no policy write. NPU governor is unchanged/unverified during the campaign; the post-campaign platform report says default.
Observed CPU governor: powersave. Pre/post capture system thermal sensors: 50.0–71.0 °C; one-minute load: 0.69–3.96; available host RAM: 57.31–60.27 GiB. These are boundary observations, not continuous NPU thermals.
Each inference launch holds the model flock, requires at least 35 GiB available host RAM and no unrelated accelerator client. Every capture records actual executable and mapped core/coreutil/xdna library identities.

## Identity and pre-timing gates

- Capacity: the supplied archived probe created all 11 required manifest contexts. The capacity-result.json and initialization metadata under recovery-20261123-1246 are fingerprinted in the JSON report. Maximum capacity is NOT MEASURED. No operator recovery was needed or performed; runtime availability is not attributed to a reset.
- Parent initialization: unchanged control and candidate each completed 8 prompt tokens plus 16 generated NPU steps across 64 layers, with matching IDs and runtime. Metadata remains in recovery-20261123-1246/initialization-passed.json.
- Fresh numerical evidence: 22 captures. Archived control/candidate each pass fresh A and distinct A/B/A with all real-vocabulary logits for every prompt position and 17 generated IDs. The list schedule is layer-major; block options do not turn that path into block-major.
- Fresh block-major boundaries: 95/96/97, 255/256/257 and 511/512/513. Last-prompt and 17 continuation logit buffers, exact greedy IDs, and state buffers at layers 0/3/62/63 agree byte-for-byte. These sampled layers cover linear convolution/recurrent and attention/KV state. This is not an all-layer-state or per-request buffer-zeroing claim.
- Actual candidate application: rebuilt from c5d20c85 with the existing Docker image and writable copied tokenizer dependencies; kernel export disabled. 1,252 archived source files independently match the staged tree. Both OFF/ON application prompt runs return apple; both candidate-specific servers pass distinct A/B/A requests with 17 completion tokens, exact API choices and consumed-engine-token traces.
- Application control is the same rebuilt candidate binary OFF, allowed for this integration comparison. It is not the primary timing control. The API does not expose full logits/state or an independent final sampled ID; complete 17-ID continuation equality is supplied by the CLI numerical gates.
- Application metadata is isolated scratch metadata describing the validated local files. This is not an unmodified shipped-install acceptance claim. The shipped download metadata differs from this historical local container.

### Immutable timing arms

| Arm | Source | Executable SHA-256 |
|---|---|---|
| control | ad8671440c9c3ab3671a00aa1cf9bfe962a210dc | 76ecff0ffd796895dfe28e6a580de85a1877f60183cfa7c1457273bec9268c47 |
| candidate | c5d20c85defced034d8f2912953f10a2acf97969 | 3c7741cf08340818b2b9958e970c4d357fb352363b5f641a07b3f2eff6692e01 |

Executables are under Models/qwen38-27b/pr4b/{control,candidate}/build/open_qwen36_cli. Only candidate sets OFLM_OPEN_FOLD_DEDUP_INPUT_SPLIT=1; unchanged control has no candidate switch. Identical model/kernel content hashes are audited independently.

## Fixed timing result

Exactly 14 new measured 256-token pairs (seven per order), six each for 512/1024 prefill and short decode, plus one excluded warmup pair per scenario: 72 captures total. No retry, selective omission or extra confirmation pair. Both prefill arms include layer and fold profiling; decode includes the same layer profiling in both arms.
Primary prefill metric is the outer CLI prompt wall, without subtracting serialization or substituting inner timers. Decode is 32 real prompt tokens and 32 generated steps (max-tokens 33).

| Scenario | Pairs | Median paired reduction | Wins | Median difference | Result |
|---|---:|---:|---:|---:|---|
| prefill 256 | 14 | 2.562018% | 14/14 | 768.000 ms | PASS |
| prefill 512 | 6 | 2.226910% | 6/6 | 1344.000 ms | PASS |
| prefill 1024 | 6 | 2.102409% | 6/6 | 2577.500 ms | PASS |
| decode 32 | 6 | 0.109890% | 3/6 | 0.500 ms | PASS |

Primary two-sided 95% percentile bootstrap median-reduction interval: **[2.072525%, 3.003963%]**, 100,000 resamples, seed 26026. Median >=1%, wins >=11/14 and interval lower endpoint >0 all pass.
Primary paired-reduction range: 0.935990–3.399519%; MAD: 0.518717 percentage points.
The fixed paired-median 1% regression limits pass for 512/1024 prefill and short decode. Decode pair 5 individually regresses 1.330377%; it remains included. Its paired median does not regress.

### Every measured pair

| Scenario | Pair | Control ms | Candidate ms | Reduction |
|---|---:|---:|---:|---:|
| prefill 256 | 0 | 29700.000 | 29009.000 | 2.326599% |
| prefill 256 | 1 | 30021.000 | 29221.000 | 2.664801% |
| prefill 256 | 2 | 29808.000 | 29529.000 | 0.935990% |
| prefill 256 | 3 | 29928.000 | 29192.000 | 2.459235% |
| prefill 256 | 4 | 29733.000 | 29183.000 | 1.849797% |
| prefill 256 | 5 | 29999.000 | 29175.000 | 2.746758% |
| prefill 256 | 6 | 29933.000 | 28975.000 | 3.200481% |
| prefill 256 | 7 | 29645.000 | 28990.000 | 2.209479% |
| prefill 256 | 8 | 29916.000 | 28899.000 | 3.399519% |
| prefill 256 | 9 | 29903.000 | 28913.000 | 3.310705% |
| prefill 256 | 10 | 29697.000 | 29113.000 | 1.966529% |
| prefill 256 | 11 | 30027.000 | 29125.000 | 3.003963% |
| prefill 256 | 12 | 29862.000 | 29284.000 | 1.935570% |
| prefill 256 | 13 | 29991.000 | 29128.000 | 2.877530% |
| prefill 512 | 0 | 60094.000 | 58833.000 | 2.098379% |
| prefill 512 | 1 | 60294.000 | 59046.000 | 2.069858% |
| prefill 512 | 2 | 60264.000 | 58925.000 | 2.221890% |
| prefill 512 | 3 | 60403.000 | 58956.000 | 2.395576% |
| prefill 512 | 4 | 60441.000 | 59092.000 | 2.231929% |
| prefill 512 | 5 | 60462.000 | 58830.000 | 2.699216% |
| prefill 1024 | 0 | 122508.000 | 120363.000 | 1.750906% |
| prefill 1024 | 1 | 123089.000 | 120910.000 | 1.770264% |
| prefill 1024 | 2 | 123317.000 | 120164.000 | 2.556825% |
| prefill 1024 | 3 | 123025.000 | 120536.000 | 2.023166% |
| prefill 1024 | 4 | 123416.000 | 120098.000 | 2.688468% |
| prefill 1024 | 5 | 122201.000 | 119535.000 | 2.181652% |
| decode 32 | 0 | 455.000 | 454.000 | 0.219780% |
| decode 32 | 1 | 452.000 | 451.000 | 0.221239% |
| decode 32 | 2 | 450.000 | 451.000 | -0.222222% |
| decode 32 | 3 | 455.000 | 448.000 | 1.538462% |
| decode 32 | 4 | 453.000 | 453.000 | 0.000000% |
| decode 32 | 5 | 451.000 | 457.000 | -1.330377% |

## Direct fold and process memory

Each fold entry below is the median per-capture sum of direct fold windows, not an extrapolation. RSS/HWM are sampled whole-process observations and are not total NPU memory.

| Prefill tokens | Control split / total fold ms | Candidate split / total fold ms |
|---:|---:|---:|
| 256 | 1360.048 / 2614.775 | 661.551 / 1925.241 |
| 512 | 2727.986 / 5212.122 | 1312.706 / 3794.477 |
| 1024 | 5532.610 / 10524.066 | 2685.885 / 7644.737 |

| Scenario | Median control RSS / HWM MiB | Median candidate RSS / HWM MiB | Max paired RSS increase MiB |
|---|---:|---:|---:|
| prefill 256 | 33446.539 / 33466.584 | 33446.482 / 33466.740 | 46.637 |
| prefill 512 | 33446.566 / 33466.637 | 33446.441 / 33466.449 | 18.566 |
| prefill 1024 | 33446.566 / 33466.725 | 33446.480 / 33466.475 | 0.016 |
| decode 32 | 33459.736 / 33466.736 | 33446.484 / 33466.197 | -0.070 |

Every measured RSS/HWM comparison passes max(1% of paired control, 32 MiB). The JSON keeps each observed peak, limit and difference; it does not equate RSS with NPU allocation.
NPU memory bytes: **UNKNOWN**. Available XRT reports expose host/platform/partitions, with total_memory_usage=N/A in the post-campaign partition report. Basic accel sysfs exposes no byte counter. The post-campaign report shows no live hardware contexts; that is not a live per-capture memory measurement. Per-capture runtime mappings, pre/post device/system conditions and all RSS/HWM samples remain in raw captures.

## Evidence, environment repair and limits

Initial real app run used a directory symlink and shipped metadata triggered automatic download. Config changed, original weight bytes stayed intact. Process stopped; downloaded config, 321347584-byte partial weight and install record retained under download-incident. Configuration reconstructed from pre-normalization backup only after exact original SHA256 match; original inode changed. Later application attempts use isolated small-file copies, local file metadata and a weight-file symlink. Timing begins after restoration; no algorithm/default/kernel change.

Configuration restoration SHA-256 is 30cf5c47d75b13934df23683f6f8f660d948d3c8e4e3ca91ad90b2278c536d6b. The new inode is explicitly recorded; initial numerical captures retain the original inode metadata. Content hashes match across initialization, numerical and timing evidence. No old record is rewritten to conceal the change.
Accepted application logs/requests and ancillary files also have a separate parent-finalization content inventory. That inventory is explicitly an audit-time seal, not reconstructed runtime capture metadata.
The first writable Docker configure required the omitted candidate license and harness headers; both were copied from the same candidate commit. A loopback client inherited an invalid proxy, and an early server capture treated harness SIGINT as an error. Those attempts remain retained separately, never as timing or accepted captures. The final server capture explicitly records its harness-issued SIGINT exit.
Optional genuine 4000-token plus 32-step decode is **NOT MEASURED**: the existing validated decode-capture contract excludes block-prefill logs, while its sequential 4000-token prefill is not practical for this bounded campaign. No seek/fake context or extrapolation is used.
Full-context, independent API logits/state, complete all-layer/per-request state isolation and continuous NPU-memory/thermal measurements are outside these passing scoped gates. Existing linked closed libraries remain application dependencies; this work neither removes nor replaces them.

Raw evidence root: Models/qwen38-27b/pr4b/final-acceptance/. The [JSON report](qwen38-host-fold-final-acceptance.json) contains all individual pairs, all 72 capture paths/hashes/commands/environments, observed runtime identities, evidence/build hashes, initialization references and machine conditions. The original [PR4B Markdown](qwen38-host-fold-dedup.md), [JSON](qwen38-host-fold-dedup.json), old raw hashes and recovery directory remain unchanged.
Reproduction tools: [numerical driver](../../../utilities/prefill-block/fold_dedup_final.py), [application capture](../../../utilities/prefill-block/fold_dedup_app.py), [application audit](../../../utilities/prefill-block/fold_dedup_app_audit.py), [fixed timing](../../../utilities/prefill-block/fold_dedup_final_timing.py), [statistics/report](../../../utilities/prefill-block/fold_dedup_final_report.py), [final audit](../../../utilities/prefill-block/fold_dedup_final_audit.py). Fresh output paths are mandatory; do not rerun into existing evidence.

## Portable checks and repository scope

Focused tests: 7 passed. Full portable suite: 2,937 passed, 172 skipped. Skips are NOT MEASURED, with exact capability/server prerequisites retained in pytest-full-with-skips.log; they are not inference passes.
Merged-target settings check: **NOT RUN: check refuses dirty tree; parent must run after final commits**. The earlier dirty-tree refusal is retained separately, not counted as a pass. No new production setting, flag or environment variable is introduced.
Only acceptance tooling, portable tests and these separate reports change. Production optimization code, defaults, kernels and criteria are unchanged. Evidence/protocol commits do not authorize a default promotion, push, merge, rebase, reset, reload, reboot, firmware or power-policy change.

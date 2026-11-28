---
title: "Test Contributions"
description: "Run and extend CTest targets, live server conformance suites, and kernel verification checks."
icon: "science"
weight: 40
---

OpenFlowLM verifies changes across four layers: fast CTest unit/smoke binaries, live `oflm-test` server conformance, XDNA2 operator verification, and static CI verifiers.

## Test Suites

| Layer | Location | What It Checks |
|---|---|---|
| CTest targets | `CMakeLists.txt`, `src/test/` | `oflm_smoke`, `openai_compat`, `OPEN-VISION-IMAGE-READ`, `bench_embed`, and `add_npu_test()` fixtures |
| Live API suite | `utilities/oflm-test/` | Running `oflm serve` across `--llm`, `--vision`, `--audio`, `--embedding`, `--tools`, and `--api` |
| Operator checks | `utilities/build-new-spec.py`, `open_kernels/designs/` | NPU outputs vs. FP64/NumPy references before updating `catalogue.py` |
| Static verifiers | `utilities/` | `check-settings-ci.sh` (flags/env vars) and `check_embed_bench_docs.py` (published `bench-embed` numbers) |

---

## Running Tests

```bash
# 1. CTest smoke & unit suite
cmake --preset linux-default && cmake --build --preset linux-default
ctest --preset linux-default --output-on-failure
ctest --preset linux-default -R openai_compat -V

# 2. Live server conformance (start 'oflm serve qwen3.5:4b -e 1' in another terminal)
oflm-test --all
oflm-test --llm --tools --model qwen3.5:4b
oflm-test --embedding

# 3. Kernel export & hardware verification
source utilities/env.sh
python utilities/build-new-spec.py --repo Qwen/Qwen3-8B
```

## Authoring Guidelines & Benchmarks

- **Cover refusals:** Test malformed JSON, unknown tags, and unsupported flags alongside happy paths, asserting non-zero exits and clear messages.
- **No external network calls:** Use bundled sample assets or deterministic synthetic inputs.
- **Guard NPU-less CI:** GitHub Actions runs `linux-debug` (`OFLM_BUILD_KERNELS=OFF`) without `/dev/accel/accel0`; CTest smoke targets must skip cleanly when no NPU is present, and `utilities/check-settings-ci.sh` catches unclassified flags merged from `origin/main`.

Benchmark throughput and upload results to the [Benchmarks hub](/benchmarks/) (see [`utilities/oflm-test/README.md`](https://github.com/Atomic-Germ/OpenFlowLM-Next/blob/main/utilities/oflm-test/README.md)):

```bash
oflm bench qwen3.5:4b --ctx-len 131072 --submit
oflm bench-embed embed-gemma:300m
python3 utilities/check_embed_bench_docs.py
```

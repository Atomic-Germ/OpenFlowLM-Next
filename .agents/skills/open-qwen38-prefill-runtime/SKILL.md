---
name: open-qwen38-prefill-runtime
description: Build the shared manifest-v4 paired-input/weight prefill runtime with exact four-product host sums and fail-closed role validation.
---

# Integrated paired prefill runtime

The shared Core accepts manifest v4 `gemm_block.low_kernels`: high GEMM name
mapped to a declared BF16 dequantization-residual kernel. Each low descriptor
must say `weight_component: bf16_residual`, have a separate context/stream and
no instruction patch. Current role/geometry support is the validated wide
dense qwen35 recipe with FP32 convolution history and paired KV.

Keep the sum order (W_hi X_hi + W_hi X_lo) + (W_lo X_hi + W_lo X_lo).
Copy mapped device output before another dispatch and retain host-owned sums.
Stage-major raw dispatch bypasses the wrapper and must be disabled on paired
routes. Both layer-major and block-major schedules then use the wrapper.
Manifests without low_kernels retain the original single-dispatch behavior.

Run CPU regressions before builds:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest specs/open-engine/tests -q
cmake -S src/open_qwen36 -B Models/qwen38-27b/prefill-runtime/cli-v2 \
  -DCMAKE_BUILD_TYPE=Release -DXRT_INCLUDE_DIR=/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/include \
  -DXRT_LIB_DIR=/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/lib
cmake --build Models/qwen38-27b/prefill-runtime/cli-v2 -j4
ctest --test-dir Models/qwen38-27b/prefill-runtime/cli-v2 --output-on-failure
```

Use fresh build directories and installed XRT. A runtime build alone does not
create low-weight kernels or establish integrated model acceptance. Production
recipe/export integration and fresh resident comparison remain separate gates;
never mutate historical manifests to make them look like new exports. Retain
old private binaries/captures for exact control. Full resident tests require
at least 35 GiB available RAM and exclusive NPU access, checked at run time.

## Runtime checkpoint, 2026-10-25

1832 host tests pass, 47 skip; 18 new cases are RED-to-GREEN. CLI-v2 builds
and CTest passes 4/4. The standard application builds in the existing
`openflowlm-build:ubuntu26` Docker image using `cmake --build build --target
oflm -j4`. Its build directory belongs to the Docker builder user; an explicit
host UID cannot write the retained dependency files. No service is started.

For a startup smoke test, run the image with the tree mounted at `/code` and
`OFLM_CONFIG_PATH=/code/src/model_list.json`, then `/code/build/src/oflm version`.
The setting names a file, not a directory. It prints `OFLM v0.1.0` and exits 0.

Preserve cli-v1: installed runtime XRT lacks development headers on this host,
so that initial compilation failed. Use the existing XRT development tree for
compilation and installed XRT runtime for hardware execution. The initial
CTest expected manifest v4 to be unsupported; it now checks v5, retaining the
unknown-version guard. No model residency or NPU validation occurred here.

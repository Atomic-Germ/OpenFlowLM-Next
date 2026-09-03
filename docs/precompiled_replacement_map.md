# Precompiled Artifact Replacement Map

This document maps the shipping precompiled surface and defines replacement
units. The machine-readable evidence is in `docs/precompiled_artifacts.json`
and can be regenerated with:

```bash
python utilities/binary-inventory/inventory.py
```

## Scope And Evidence

The default inventory covers `src/lib` and `src/xclbins`:

| Category | Count | Replacement policy |
|---|---:|---|
| Closed XCLBIN NPU kernels | 222 | Recompile from open Iron/mlir-aie designs |
| Open XCLBIN NPU kernels | 24 | Already ours; shipped with the app, rebuilt at build time |
| Closed engine/primitive binaries | 133 | Replace host implementation and both XRT/HRX builds |
| Third-party runtime binaries | 23 | Build or package from upstream open source |
| Runtime support archives | 2 | Rebuild AIEBU from upstream source |
| **Total** | **404** | **288 unique payloads, 472,115,761 bytes** |

The 24 open kernels are the EmbeddingGemma `npu_matmul_f32` designs (6 shapes
× 2 M pads, plus instruction streams). They are counted separately by the
inventory tool so work we have already done is never listed as pending.

EmbeddingGemma is now fully closed-binary removed: the `gemma_embedding`
library and its XRT/HRX/Linux/Windows copies are gone, and installers and the
standalone test use the open engine. This section reflects that post-cleanup
state.

`docs/precompiled_artifacts_all.json` additionally inventories archived copies
and development instruction streams outside the shipping surface. Excluding
Git data, dependency environments, and build directories, the full workspace
contains 781 artifacts: 443 XCLBINs, 12 instruction streams, 94 ELF shared
libraries, 131 PE DLLs, 99 COFF libraries, and 2 static archives. These total
952,939,355 bytes but largely duplicate the shipping tree and ExampleNPU
snapshot; replacement tracking uses the 386-artifact shipping set to avoid
double-counting.

The 139 closed engine artifacts include Linux `.so`, Windows `.dll`, Windows
`.lib`, and one backup `.so`. There are 47 ELF libraries: 24 XRT and 23 HRX.
HRX has no Gemma4-12B engine, so the backend sets are not currently symmetric.

Every XCLBIN has the same section set: `MEM_TOPOLOGY`, `AIE_PARTITION`,
`EMBEDDED_METADATA`, `IP_LAYOUT`, `CONNECTIVITY`, `GROUP_CONNECTIVITY`, and
`GROUP_TOPOLOGY`. None has `BUILD_METADATA`. Versions are 2.19.0 (4), 2.20.0
(7), 2.21.0 (2), and 2.21.75 (209). Exact original source, compile flags, and
kernel shapes therefore cannot be recovered from metadata alone.

## Closed Host Libraries

The application links 22 closed libraries. Public headers and call sites show
that they comprise five foundational primitives and 17 family engines. The
table also records the now-unlinked `gemma_embedding` residual binaries.

### Foundational Primitives

| Library | Known contract | Replacement plan |
|---|---|---|
| `q4_npu_eXpress` | `Q4NX`, inherited `SafeTensors`, quantized tensor loading and dequantization | Prefer migration to an open safetensors/model abstraction rather than preserving the C++ binary ABI. Extend `utilities/q4nx-build` with open-output builders so architecture detection and tensor mappings are shared. Keep a temporary open Q4NX reader only where converted models require it. |
| `gemm` | NPU sequence generation, bias/output offsets, no/GeLU/SiLU activation | Generalize the validated Iron whole-array GEMM into a shared shape compiler and runtime dispatcher. Add integer/quantized variants after BF16-to-FP32 validation. |
| `dequant` | Q4_1 and packed-Q8-in-Q4NX sequence generation | Specify Q4 block layouts from `q4nx-build`, implement CPU oracle first, then Iron kernels that feed GEMM without unnecessary round trips. |
| `mha` | Attention sequence generation and chunk sizing | Highest reverse-interface risk: no public header exists. Publish an open attention API based on model semantics instead of recreating the hidden class ABI; implement causal, sliding, global, bidirectional, and attention-sink modes incrementally. |
| `lm_head` | Weight load, asynchronous execute/wait, exposed input buffer | Implement as shared tiled vocabulary projection. Determine whether current engines actually require the standalone library before preserving its asynchronous interface. |

`q4_npu_eXpress` has the highest fan-out: inspected model DSOs leave
`SafeTensors::load_weights` unresolved for final link resolution. `mha` is the
largest source gap because the repository has no declaration for its exported
`MHA`/`mha_type_t` interface.

### Model Engines

| Library | Models/components | Rough open replacement |
|---|---|---|
| `gemma_embedding` | EmbeddingGemma | Replaced by `src/open_embedding`; removed, including residual XRT/HRX/Windows binary copies and installer/test references. |
| `gemma_text_npu` | Gemma3 text | First causal engine. Reuse Gemma3 math from open embedding; add causal masks, KV cache, prefill/decode, and LM head. |
| `gemma_npu` | Gemma3 vision-language | Build on open Gemma3 text, then add SigLIP/vision encoder, projector, image payload, and multimodal RoPE/token placement. |
| `llama_npu` | Llama3 1B/3B/8B and DeepSeek-Llama 8B | Implement standard GQA decoder with RoPE and KV cache; compile per hidden/intermediate/head shape. |
| `nanbeige_npu` | Nanbeige4.1 3B | Reuse Llama-like engine with family config and weight-name adapter; validate architectural deviations before adding kernels. |
| `phi4_npu` | Phi4 mini | Reuse dense decoder primitives; add long/short RoPE behavior and Phi-specific config/weight mapping. |
| `qwen2_npu` | Qwen2.5 text | Dense GQA decoder with Qwen config/weight mapping. |
| `qwen3_npu` | Qwen3 sizes, instruct/thinking derivatives, DeepSeek-Qwen3 | Dense decoder plus Q/K norm. Fine-tunes link to size-specific family bundles. |
| `lfm2_npu` | LFM2/LFM2.5 text variants | Add short-convolution and hybrid layer scheduler around shared attention/GEMM. Keep CPU reference for convolution before NPU offload. |
| `gpt_oss_npu` | GPT-OSS 20B and Safeguard | Add MoE routing/expert dispatch, attention sinks, and short-sequence GEMM specialization. |
| `qwen2vl_npu` | Qwen2.5-VL | Build on Qwen2 text; add vision encoder, windowed vision attention, projector, payload ABI, and multimodal positions. |
| `qwen3vl_npu` | Qwen3-VL 4B | Build on Qwen3 text; add vision encoder/projector and payload handling. |
| `qwen3_5vl_npu` | Qwen3.5 0.8B/2B/4B/9B VLM | Implement hybrid attention, Gate Delta Net, convolution, vision stack, and recurrent state management. |
| `qwen3_5_omni_npu` | Qwen3.5 Omni | Extend Qwen3.5 with image/audio payloads and speech generation; preserve thinker hidden-state handoff to `say()`. |
| `qwen3_6_moe_npu` | Qwen3.6 35B-A3B VLM | Extend Qwen3.5 hybrid core with MoE routing and expert execution. |
| `gemma4e_npu` | Gemma4 E2B/E4B vision/audio | Implement multimodal encoders/projectors and SWA/global/skip layer scheduling after Gemma text and vision primitives exist. |
| `gemma4_12b_npu` | Gemma4 12B vision/audio | Separate family: global/sliding/image-global attention and combined audio/image projection differ from Gemma4-E. |
| `whisper_npu` | Whisper V3 Turbo | Separate encoder-decoder engine; preserve existing FFmpeg/FFT/mel host preprocessing, then replace encoder attention/MM, decoder MV, and output head. |

All normal language engines implement `causal_lm`; Qwen Omni and Whisper have
distinct contracts. New open engines should be source-linked rather than
recreated as ABI-compatible C++ DSOs unless an external consumer requires that
ABI.

## XCLBIN Replacement Bundles

The 222 XCLBINs contain only 106 unique payloads. Seven groups of entire model
directories are byte-identical, proving that fine-tunes should link to family
assets rather than duplicate them.

| Open bundle boundary | Current dirs | Files | Distinctive work |
|---|---:|---:|---|
| Gemma3-1B text | 1 | 5 | Dense causal attention, dequant, layer, LM head, MM |
| Gemma3-4B multimodal derivatives | 4 | 28 | One byte-identical 7-kernel bundle for Gemma, MedGemma, and TranslateGemma |
| Gemma4-E2B/E4B | 2 | 20 | Shared core; size-specific layer, LM head, vision attention |
| Gemma4-12B | 1 | 7 | Global/sliding/image-global attention, audio-image MM, fused dequant-MM |
| Qwen3 dense | 7 | 28 | Size bundles for 0.6B, 1.7B, 4B, 8B; derivatives link to 4B/8B |
| Qwen3-VL 4B | 1 | 6 | Vision attention and vision MM over Qwen3 core |
| Qwen3.5 VLM | 4 | 32 | Shared Gate Delta Net/attention/conv/MM; four size-specific layers/heads/vision |
| Qwen3.5 Omni | 1 | 10 | Adds high MM and window attention |
| Qwen3.6 MoE | 1 | 9 | Hybrid core plus MoE layer and fused dequant-MM |
| Llama3/DeepSeek-Llama | 4 | 16 | 1B/3B/8B shape variants; DeepSeek 8B is byte-identical to Llama 8B |
| LFM2/LFM2.5 | 5 | 25 | Shared attention/conv/dequant/MM; three layer variants |
| GPT-OSS 20B | 2 | 12 | One byte-identical bundle; expert and short-sequence MM kernels |
| Qwen2.5 text | 1 | 4 | Dense decoder primitives |
| Qwen2.5-VL | 1 | 7 | Vision MM plus full/window vision attention |
| Phi4 mini | 1 | 4 | Dense decoder primitives |
| Nanbeige4.1 | 1 | 4 | Dense decoder primitives |
| Whisper V3 | 1 | 5 | Encoder attention/dequant/MM, decoder MV, head |
| **Total** | **38** | **222** | **18 architecture/shape replacement units** |

The open EmbeddingGemma kernels are model-local `npu_matmul_f32` assets and no
longer appear under `src/xclbins`; they are the reference implementation for
the family-bundle compiler.

## Confirmed Identical Directory Groups

1. Gemma3-4B, MedGemma-4B, MedGemma-1.5-4B, TranslateGemma-4B.
2. Qwen3-4B base, Instruct-2507, Thinking-2507.
3. Qwen3-8B and DeepSeek-R1-0528-Qwen3-8B.
4. Llama3.1-8B and DeepSeek-R1-Distill-Llama-8B.
5. GPT-OSS-20B and GPT-OSS-Safeguard-20B.
6. LFM2-1.2B and LFM2.5-1.2B.
7. LFM2-2.6B and LFM2-2.6B-Transcript.

`Qwen3.5-OMNI-NPU2` has local XCLBINs but no `model_list.json` entry.
EmbeddingGemma has a model entry but no closed XCLBIN directory.

## NPU Kernel Distribution

Compiled kernels follow the same two locations as the closed-source stack, with
an escape hatch for new models.

**Established families ship kernels with the application.** They live in
`src/xclbins/<Model-Dir>/npu_matmul_f32/`, install to
`<xclbin_prefix>/xclbins/<Model-Dir>/npu_matmul_f32/`, are built at build time,
and are deliberately absent from a model's `files` list and `model_info.json`.
Model repos carry weights and configuration only. `src/CMakeLists.txt` installs
the whole `xclbins` tree, so promoting a family is just adding a directory.

**New or prototype models may ship their own kernels** in the model directory's
`npu_matmul_f32/` via `q4nx-build --open-embedding --npu-assets <dir>`, so a
brand-new model works end to end before promotion. Promote it later by moving
the kernels into `src/xclbins/`.

Lookup order is model-local first, then the app family directory, then a
`embed-gemma` family fallback. `utils::find_xclbin_path()` throws when no xclbin
tree exists; the engine catches that and degrades to CPU-only, so the open
engine never hard-requires the xclbin tree.

## Completed: EmbeddingGemma Binary Removal

The open EmbeddingGemma engine builds and the residual closed references are
now removed:

- `src/wix/flm.wxs`: removed the `gemma_embedding.dll` component.
- `src/inno/flm.iss`: removed the `gemma_embedding.dll` source entry.
- `src/test/gemma_embedding/CMakeLists.txt`: no longer links `gemma_embedding`
  or compiles the deleted `modeling_gemma_embedding.cpp`. It is now a
  standalone open-engine target that links only Boost, threads, tokenizers-cpp,
  and XRT.
- `src/test/gemma_embedding/Makefile`: replaced the legacy closed-engine build
  with a thin CMake wrapper.
- `src/lib/xrt` and `src/lib/hrx`: removed all six residual
  `gemma_embedding`/`libgemma_embedding` Linux and Windows binaries.
- `src/test/gemma_embedding/test.cpp`: uses `open_embedding::Engine` directly.

Verified after removal:

- Main XRT build succeeds: `cmake --build src/build -j4`.
- Standalone embedding test builds: `/tmp/opencode/open-embedding-test3`.
- `readelf -d` on the standalone test shows only
  `libxrt_coreutil`, Boost, and system libraries. No closed engine library
  appears in `NEEDED`.
- Shipping inventory dropped from 386 to 380 artifacts and from 139 to 133
  closed engine/primitive binaries.

The shared test helper, `src/test/CMakeLists.txt`, still unconditionally links
`q4_npu_eXpress`, `lm_head`, `dequant`, `gemm`, and `mha` for other model
tests. Any future open-engine test must not use that helper until it is
refactored to link those primitives conditionally.

## Third-Party Precompiled Files

These are not reverse-engineering targets. Replace checked-in Windows binaries
with reproducible upstream builds or package-manager dependencies:

| Upstream component | Files | Plan |
|---|---:|---|
| FFmpeg | 7 DLLs | Build pinned FFmpeg from source, matching the existing portable Linux approach. |
| FFTW | 3 DLLs + 3 import libraries | Build pinned FFTW or consume vcpkg targets. |
| Protobuf/Abseil | 4 DLLs | Build pinned Protobuf and Abseil through CMake/vcpkg. |
| curl/zlib | 2 DLLs | Build pinned upstream releases or consume vcpkg. |
| MSVC runtime | 3 DLLs | Do not rebuild; use Microsoft redistributable packaging rules rather than repository copies. |
| pkgconf | 1 DLL | Development tool; remove from runtime bundle unless dependency analysis proves it is required. |
| AIEBU | 1 Linux archive + 1 Windows archive | Build from the open upstream AIEBU source already represented by repository headers. |

## Execution Order

1. Establish shared open model/tensor interfaces in `utilities/q4nx-build` and
   the C++ runtime; keep safetensors as the canonical open weight source.
2. Productize the open GEMM compiler/runtime: shape inventory input, family
   bundle output, BF16-to-FP32 validation, and deterministic artifact metadata.
3. Implement open dequant and LM-head primitives with CPU oracles.
4. Define and implement the open attention API, starting with causal/GQA and
   then sliding/global/bidirectional modes.
5. Build Gemma3-1B text as the first causal engine and use it to validate KV
   cache, prefill, decode, sampling integration, and bundle installation.
6. Add dense Llama-like families: Llama3, Nanbeige, Phi4, Qwen2, Qwen3.
7. Add hybrid and MoE families: LFM2, GPT-OSS, Qwen3.5, Qwen3.6.
8. Add vision/audio families after their text cores: Gemma3-VL, Qwen VL,
   Gemma4, Qwen Omni, then Whisper.
9. For each completed family, remove its closed library triplets and duplicate
   XCLBIN directories only after CPU/NPU agreement, API tests, and performance
   benchmarks pass.

If a required closed behavior cannot yet be reproduced, expose an explicit
`not implemented` error for that model/operation. Do not silently route through
an unidentified closed binary or claim feature parity.

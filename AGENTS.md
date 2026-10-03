# Help a model out!

You can install or request toolchain components

Each time you build a new set of xclbins and libraries, document your process and create a new skill file for the next to build on.

Skills provide specialized instructions and workflows for specific tasks.
Use the skill tool to load a skill when a task matches its description.

The new format Q4NX models for this will be located on huggingface `Atomic-Germ/*-nB-OpenNPU2`, each will need a builder in q4nx-build as they gain xclbin support. Family xclbins will be distributed with the application, and a converted model ships with the rest. Each model links to a family xclbin, regardless of fine-tuning or derivitive models. The purpose of this is to allow quick additions of new but shape-identical models without updates needed, distributed through `oflm-add` to install and link properly.

As you encounter closed-source requirements, note them and replace with "not implemented" messaging before coding a replacement.

When investigating pre-compiled xclbins and libraries, save any useful tools you may create to the `utilities/` directory rather than leaving them in a temp folder.

The `oflm-test` tool at `utilities/oflm-test` is capable of running a full test suite for `--llm`, `--vision`, `--embed`, or `--tools`.

The `q4nx-build` tool at `utilities/q4nx-build` should always be in-sync with the expected formats of the models as each gains support, there should be no manual steps left. If unavoidable, the user should always recieve instruction for it.

For each successfully packed model, create or update a skill to ensure the next agent does not need to reproduce research for the next addition.

Note: Peano (llvm-aie) has not been added to PATH to avoid conflict with
      system clang/clang++. It can be found in:
      ./ironvenv/lib/python3.12/site-packages/llvm-aie/bin

Activate the ironvenv/bin/activate; use source utilities/mlir-aie/utils/env_setup.sh also *if needed*

<available_skills>
  <skill>
    <name>npu_offload_pipeline</name>
    <description>End-to-end workflow for offloading dense GEMM operations to AMD NPU2 via mlir-aie/iron. Use when: compiling NPU xclbins, integrating NPU backends into embedding/LLM engines, validating NPU vs CPU reference, debugging XRT dispatch issues, or extending to new model architectures.</description>
    <location>.opencode/skill/npu_offload_pipeline.md</location>
  </skill>
  <skill>
    <name>open-granite-kernels</name>
    <description>Build, verify and ship the open XDNA2 kernel sets (dx ln lm_head_q4) that run IBM Granite 4.2 3B on the dense recipe. Use when rebuilding those xclbins, adding another Granite size, debugging "no open kernels found" for granite:3b, or when a Granite container's attention_multiplier is refused at load.</description>
    <location>.opencode/skill/open-granite-kernels/SKILL.md</location>
  </skill>
  <skill>
    <name>open-wide-deltanet-ab</name>
    <description>Build and validate the separate open banked alpha/beta dispatch at H5120/H2560, including xn streaming and the fused glue DMA constraint.</description>
    <location>.opencode/skill/open-wide-deltanet-ab/SKILL.md</location>
  </skill>
  <skill>
    <name>open-wide-deltanet-chain</name>
    <description>Build and validate the synthetic 48-head AB, conv/record and recurrent-state chain for the WideDeltaNet A7 gate, including default32-head regression and known precision limits.</description>
    <location>.opencode/skill/open-wide-deltanet-chain/SKILL.md</location>
  </skill>
  <skill>
    <name>open-dense-activation-stream</name>
    <description>Build and validate layer_x Q4 projection probes at K5120/K6144 using depth-two streamed xn/xm inputs and actual main-core scratch.</description>
    <location>.opencode/skill/open-dense-activation-stream/SKILL.md</location>
  </skill>
  <skill>
    <name>open-segmented-dense</name>
    <description>Build and validate segmented dense Q4 down GEMV at K17408, segment-major DMA and local accumulation, including the remaining full-FFN bf16 rounding accuracy gate.</description>
    <location>.opencode/skill/open-segmented-dense/SKILL.md</location>
  </skill>
  <skill>
    <name>open-dense-ffn-precision</name>
    <description>Reproduce the strict synthetic FFN accuracy gate with up/gate traces, precise vector SiLU, and scale-invariant cosine for tiny tensors.</description>
    <location>.opencode/skill/open-dense-ffn-precision/SKILL.md</location>
  </skill>
  <skill>
    <name>open-wide-ln</name>
    <description>Build and validate streamed residual RMSNorm at width 5120, including actual L1 placement and legacy 2048/4096 hardware regression.</description>
    <location>.opencode/skill/open-wide-ln/SKILL.md</location>
  </skill>
  <skill>
    <name>open-wide-lm-head</name>
    <description>Build and validate the standalone Q8 LM head at K5120 with full vocabulary, production packing, independent FP64 references and K4096 regression.</description>
    <location>.opencode/skill/open-wide-lm-head/SKILL.md</location>
  </skill>
  <skill>
    <name>open-wide-attention</name>
    <description>Build and validate Q24/KV4/HD256/ROT64 attention using the production ax worker, per-head gates, device-carried KV state and Q16/KV4 regression.</description>
    <location>.opencode/skill/open-wide-attention/SKILL.md</location>
  </skill>
  <skill>
    <name>open-wide-deltanet-layer</name>
    <description>Build the baseline synthetic H5120/FF17408 DeltaNet layer, prerequisite kernels and padded state adapters.</description>
    <location>.opencode/skill/open-wide-deltanet-layer/SKILL.md</location>
  </skill>
  <skill>
    <name>qwen38-upstream-regression</name>
    <description>Reproduce upstream compatibility checks: rebuilt Q4 projections/FFN, Linux runtime, converter and isolated PR branches.</description>
    <location>.opencode/skill/qwen38-upstream-regression/SKILL.md</location>
  </skill>
  <skill>
    <name>open-wide-deltanet-precision</name>
    <description>Build and validate the corrected Q4, precise conv and compensated recurrence composition that closes the synthetic H5120 DeltaNet layer gate.</description>
    <location>.opencode/skill/open-wide-deltanet-precision/SKILL.md</location>
  </skill>
  <skill>
    <name>open-wide-attention-layer</name>
    <description>Build and validate the complete synthetic H5120/FF17408 attention layer with production packing, segmented FFN and persistent device KV cache.</description>
    <location>.opencode/skill/open-wide-attention-layer/SKILL.md</location>
  </skill>
  <skill>
    <name>open-wide-slice</name>
    <description>Validate the synthetic eight-layer H5120 Qwen38 slice with distinct packed weights, device activation chaining and isolated recurrent/KV state.</description>
    <location>.opencode/skill/open-wide-slice/SKILL.md</location>
  </skill>
  <skill>
    <name>open-wide-decode</name>
    <description>Validate synthetic eight-layer autoregressive decode with final RMSNorm, full-vocabulary Q8 logits, token feedback and persistent layer state.</description>
    <location>.opencode/skill/open-wide-decode/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-model</name>
    <description>Download and stream-convert real Qwen38-27B weights, restore 48-head GGUF order, and prepare or validate the standalone 64-layer model.</description>
    <location>.opencode/skill/open-qwen38-model/SKILL.md</location>
  </skill>
  <skill>
    <name>open-wide-model-precision</name>
    <description>Build compensated projection/FFN, precise attention and RMSNorm probes, and replay the real 64-layer model against unchanged reference fixtures.</description>
    <location>.opencode/skill/open-wide-model-precision/SKILL.md</location>
  </skill>
  <skill>
    <name>open-q4-product-precision</name>
    <description>Reproduce Q4 block cancellation and build compensated-product projection/FFN probes with unchanged full-model replay gates.</description>
    <location>.opencode/skill/open-q4-product-precision/SKILL.md</location>
  </skill>
  <skill>
    <name>open-q4-block-carry</name>
    <description>Diagnose real FFN activation boundaries with up/gate traces and validate Q4 block-residual carry against unchanged full-model references.</description>
    <location>.opencode/skill/open-q4-block-carry/SKILL.md</location>
  </skill>
  <skill>
    <name>open-down-segment-carry</name>
    <description>Build and validate compensated segmented Q4 down reduction, including real layer0 rounding and unchanged full-model replay.</description>
    <location>.opencode/skill/open-down-segment-carry/SKILL.md</location>
  </skill>
  <skill>
    <name>open-residual-rne</name>
    <description>Build and validate exact vector FP32 residual addition in wide RMSNorm, including signed zero and unchanged full-model replay.</description>
    <location>.opencode/skill/open-residual-rne/SKILL.md</location>
  </skill>
  <skill>
    <name>open-attention-boundary</name>
    <description>Isolate real attention rounding with captured Q/K/V/gate and cache, then validate block-carry projections and precise attention against unchanged full-model references.</description>
    <location>.opencode/skill/open-attention-boundary/SKILL.md</location>
  </skill>
  <skill>
    <name>open-norm-rne</name>
    <description>Trace real wide RMSNorm statistics and validate integer-lane FP32 products and compensated sums against unchanged full-model references.</description>
    <location>.opencode/skill/open-norm-rne/SKILL.md</location>
  </skill>
  <skill>
    <name>open-ffn-activation-carry</name>
    <description>Build compensated FFN activation math with exact additions and compact down loops, then validate real rounding boundaries and full-model replay.</description>
    <location>.opencode/skill/open-ffn-activation-carry/SKILL.md</location>
  </skill>
  <skill>
    <name>open-norm-scale-carry</name>
    <description>Build compensated RMSNorm scaling and validate real layer5 rounding, prior norm behavior and unchanged full-model references.</description>
    <location>.opencode/skill/open-norm-scale-carry/SKILL.md</location>
  </skill>
  <skill>
    <name>qwen38-main-integration</name>
    <description>Rebuild corrected FFN after main integration and check converter head-order equivalence, runtime compatibility and full-model regressions.</description>
    <location>.opencode/skill/qwen38-main-integration/SKILL.md</location>
  </skill>
  <skill>
    <name>open-ffn-sigmoid-series</name>
    <description>Trace real SiLU stages and build the small-argument sigmoid series with compact FFN loops, then validate rounding boundaries and full-model references.</description>
    <location>.opencode/skill/open-ffn-sigmoid-series/SKILL.md</location>
  </skill>
  <skill>
    <name>qwen38-wide-main-integration</name>
    <description>Merge shared WideDeltaNet and rolled-band upstream changes while preserving and checking later precision modes.</description>
    <location>.opencode/skill/qwen38-wide-main-integration/SKILL.md</location>
  </skill>
  <skill>
    <name>open-deltanet-projection-carry</name>
    <description>Diagnose real DeltaNet boundaries and validate compensated QKV/Z projections against immutable full-model fixtures.</description>
    <location>.opencode/skill/open-deltanet-projection-carry/SKILL.md</location>
  </skill>
  <skill>
    <name>open-wide-ab-carry</name>
    <description>Build compensated banked AB projections, isolate dot and nonlinear errors, and replay immutable full-model fixtures.</description>
    <location>.opencode/skill/open-wide-ab-carry/SKILL.md</location>
  </skill>
</available_skills>

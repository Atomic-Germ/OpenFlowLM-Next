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
</available_skills>

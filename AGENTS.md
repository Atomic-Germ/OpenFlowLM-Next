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
    <name>open-qwen38-production-preflight</name>
    <description>Normalize existing Qwen3.5-family metadata, compare Qwen3.8-27B source tensors, and build all 41 production kernel sets on Linux.</description>
    <location>.opencode/skill/open-qwen38-production-preflight/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen35-fp32-down-replay</name>
    <description>Build and validate direct FP32 input preparation for split dense FFN, replay isolated Qwen3.8 layers and run an all-layer precision sweep without full-model residency.</description>
    <location>.opencode/skill/open-qwen35-fp32-down-replay/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-fp32-production</name>
    <description>Build the complete corrected Qwen3.8 export and verify full-depth decode propagation, runtime identity and split-down partials.</description>
    <location>.opencode/skill/open-qwen38-fp32-production/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen35-fp32-front-replay</name>
    <description>Build and validate isolated FP32 norm-to-up/gate/SiLU with separate NPU projection captures and a BF16 control.</description>
    <location>.opencode/skill/open-qwen35-fp32-front-replay/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-fp32-norm-integration</name>
    <description>Build and validate full lx/ax FP32 norm-to-up/gate integration, widened scratch ABI and full-depth decode regression.</description>
    <location>.opencode/skill/open-qwen38-fp32-norm-integration/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-attention-replay</name>
    <description>Replay production gated attention with captured Q/KV/softmax boundaries and compare BF16/FP32 outputs without full-model residency.</description>
    <location>.opencode/skill/open-qwen38-attention-replay/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-paired-kv-replay</name>
    <description>Build and validate isolated paired-BF16 K/V attention with FP32 output, an exact BF16 control and local memory checks.</description>
    <location>.opencode/skill/open-qwen38-paired-kv-replay/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-paired-attention-integration</name>
    <description>Build full Qwen3.8 layers with paired KV and FP32 attention output-to-projection, then validate the new cache/scratch ABI and full-depth decode.</description>
    <location>.opencode/skill/open-qwen38-paired-attention-integration/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-fp32-attention-entry</name>
    <description>Build isolated full-ax FP32 entry norm and compare against a byte-exact paired-attention control with forced inputs and causal KV history.</description>
    <location>.opencode/skill/open-qwen38-fp32-attention-entry/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-fp32-entry-integration</name>
    <description>Build and validate production FP32 attention-entry norm, its scratch/stream ABI and full-depth Qwen3.8 decode with independent projection gates.</description>
    <location>.opencode/skill/open-qwen38-fp32-entry-integration/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-fp32-linear-entry</name>
    <description>Build isolated full-lx FP32 norm-to-QKV/Z/alpha/beta and compare against a byte-exact production control with causal conv/recurrent state.</description>
    <location>.opencode/skill/open-qwen38-fp32-linear-entry/SKILL.md</location>
  </skill>
</available_skills>

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
    <name>open-qwen38-prefill-weighted-cli</name>
    <description>Build private paired-weight FFN CLIs and validate causal propagation with exact controls and residual tracing.</description>
    <location>.agents/skills/open-qwen38-prefill-weighted-cli/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-prefill-ffn-replay</name>
    <description>Compose paired-weight up/gate, native host activation and original down with exact controls and independent layer-output gates.</description>
    <location>.agents/skills/open-qwen38-prefill-ffn-replay/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-prefill-paired-weights</name>
    <description>Build private paired-weight GEMM kernels and validate causal up/gate precision with byte-exact native controls.</description>
    <location>.agents/skills/open-qwen38-prefill-paired-weights/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-prefill-upgate-replay</name>
    <description>Replay captured hybrid up/gate with exact native controls and diagnose weight-rounding error without full-model residency.</description>
    <location>.agents/skills/open-qwen38-prefill-upgate-replay/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-prefill-sublayer-trace</name>
    <description>Capture selected causal prefill sublayers and separate incoming residual drift from conditioned FFN arithmetic with exact native controls.</description>
    <location>.agents/skills/open-qwen38-prefill-sublayer-trace/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-prefill-residual-trace</name>
    <description>Build a private synchronized native prefill residual tracer and audit per-layer drift after byte-exact baseline logit identity.</description>
    <location>.agents/skills/open-qwen38-prefill-residual-trace/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-prefill-host-boundaries</name>
    <description>Build the CPU prefill bridge and validate FP32 history, paired KV and 255/256/257-token boundaries against an independent reference.</description>
    <location>.agents/skills/open-qwen38-prefill-host-boundaries/SKILL.md</location>
  </skill>
  <skill>
    <name>npu_offload_pipeline</name>
    <description>End-to-end workflow for offloading dense GEMM operations to AMD NPU2 via mlir-aie/iron. Use when: compiling NPU xclbins, integrating NPU backends into embedding/LLM engines, validating NPU vs CPU reference, debugging XRT dispatch issues, or extending to new model architectures.</description>
    <location>.agents/skills/npu_offload_pipeline.md</location>
  </skill>
  <skill>
    <name>open-granite-kernels</name>
    <description>Build, verify and ship the open XDNA2 kernel sets (dx ln lm_head_q4) that run IBM Granite 4.2 3B on the dense recipe. Use when rebuilding those xclbins, adding another Granite size, debugging "no open kernels found" for granite:3b, or when a Granite container's attention_multiplier is refused at load.</description>
    <location>.agents/skills/open-granite-kernels/SKILL.md</location>
  </skill>
  <skill>
    <name>open-wide-deltanet-ab</name>
    <description>Build and validate the separate open banked alpha/beta dispatch at H5120/H2560, including xn streaming and the fused glue DMA constraint.</description>
    <location>.agents/skills/open-wide-deltanet-ab/SKILL.md</location>
  </skill>
  <skill>
    <name>open-wide-deltanet-chain</name>
    <description>Build and validate the synthetic 48-head AB, conv/record and recurrent-state chain for the WideDeltaNet A7 gate, including default32-head regression and known precision limits.</description>
    <location>.agents/skills/open-wide-deltanet-chain/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-production-preflight</name>
    <description>Normalize existing Qwen3.5-family metadata, compare Qwen3.8-27B source tensors, and build all 41 production kernel sets on Linux.</description>
    <location>.agents/skills/open-qwen38-production-preflight/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen35-fp32-down-replay</name>
    <description>Build and validate direct FP32 input preparation for split dense FFN, replay isolated Qwen3.8 layers and run an all-layer precision sweep without full-model residency.</description>
    <location>.agents/skills/open-qwen35-fp32-down-replay/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-fp32-production</name>
    <description>Build the complete corrected Qwen3.8 export and verify full-depth decode propagation, runtime identity and split-down partials.</description>
    <location>.agents/skills/open-qwen38-fp32-production/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen35-fp32-front-replay</name>
    <description>Build and validate isolated FP32 norm-to-up/gate/SiLU with separate NPU projection captures and a BF16 control.</description>
    <location>.agents/skills/open-qwen35-fp32-front-replay/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-fp32-norm-integration</name>
    <description>Build and validate full lx/ax FP32 norm-to-up/gate integration, widened scratch ABI and full-depth decode regression.</description>
    <location>.agents/skills/open-qwen38-fp32-norm-integration/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-attention-replay</name>
    <description>Replay production gated attention with captured Q/KV/softmax boundaries and compare BF16/FP32 outputs without full-model residency.</description>
    <location>.agents/skills/open-qwen38-attention-replay/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-paired-kv-replay</name>
    <description>Build and validate isolated paired-BF16 K/V attention with FP32 output, an exact BF16 control and local memory checks.</description>
    <location>.agents/skills/open-qwen38-paired-kv-replay/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-paired-attention-integration</name>
    <description>Build full Qwen3.8 layers with paired KV and FP32 attention output-to-projection, then validate the new cache/scratch ABI and full-depth decode.</description>
    <location>.agents/skills/open-qwen38-paired-attention-integration/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-fp32-attention-entry</name>
    <description>Build isolated full-ax FP32 entry norm and compare against a byte-exact paired-attention control with forced inputs and causal KV history.</description>
    <location>.agents/skills/open-qwen38-fp32-attention-entry/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-fp32-entry-integration</name>
    <description>Build and validate production FP32 attention-entry norm, its scratch/stream ABI and full-depth Qwen3.8 decode with independent projection gates.</description>
    <location>.agents/skills/open-qwen38-fp32-entry-integration/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-fp32-linear-entry</name>
    <description>Build isolated full-lx FP32 norm-to-QKV/Z/alpha/beta and compare against a byte-exact production control with causal conv/recurrent state.</description>
    <location>.agents/skills/open-qwen38-fp32-linear-entry/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-fp32-conv-history</name>
    <description>Build isolated FP32 conv history with half-tile streams and validate causal state against a byte-exact BF16-history control.</description>
    <location>.agents/skills/open-qwen38-fp32-conv-history/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-fp32-linear-integration</name>
    <description>Build and validate production FP32 DeltaNet entry/history, its state ABI and host carry, and full-depth decode with independent boundary gates.</description>
    <location>.agents/skills/open-qwen38-fp32-linear-integration/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-recurrence-replay</name>
    <description>Build private production-layout DeltaNet control and triple-BF16 recurrence kernels and validate causal head precision without model residency.</description>
    <location>.agents/skills/open-qwen38-recurrence-replay/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-factored-readout</name>
    <description>Build and validate isolated compensated factored DeltaNet readout with causal state and unchanged head precision gates.</description>
    <location>.agents/skills/open-qwen38-factored-readout/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-factored-full-layer</name>
    <description>Build private full-lx factored DeltaNet with shared helpers, audit compiler dependencies and resource budgets, and validate causal full-layer replay.</description>
    <location>.agents/skills/open-qwen38-factored-full-layer/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-factored-integration</name>
    <description>Build the production compensated factored DeltaNet export and validate all 64 layers with causal head gates and streamed weights.</description>
    <location>.agents/skills/open-qwen38-factored-integration/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-qkv-precision</name>
    <description>Build isolated Q4 QKV control and consistent-sum probes, capture activation tables, and validate causal record precision without full-model residency.</description>
    <location>.agents/skills/open-qwen38-qkv-precision/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-consistent-full-layer</name>
    <description>Build private full-lx consistent Q4 sums and validate causal DeltaNet, Z/out projections, FFN and residuals with streamed weights.</description>
    <location>.agents/skills/open-qwen38-consistent-full-layer/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-consistent-integration</name>
    <description>Build production consistent Q4 sums and validate full-depth lx/ax decode, independent boundaries and resident identity/reset.</description>
    <location>.agents/skills/open-qwen38-consistent-integration/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-ffn-projection-sweep</name>
    <description>Build a production-arithmetic FFN probe and validate separate up/gate projections across all layers with exact norm/activation identity.</description>
    <location>.agents/skills/open-qwen38-ffn-projection-sweep/SKILL.md</location>
  </skill>
  <skill>
    <name>open-qwen38-paired-prefill-input</name>
    <description>Build private paired-BF16-input GEMM CLIs and audit full-depth causal prefill against a byte-exact control.</description>
    <location>.agents/skills/open-qwen38-paired-prefill-input/SKILL.md</location>
  </skill>
</available_skills>

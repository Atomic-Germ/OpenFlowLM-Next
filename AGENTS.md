# Help a model out!

You can install or request toolchain components

Each time you build a new set of xclbins and libraries, document your process and create a new skill file for the next to build on.

Skills provide specialized instructions and workflows for specific tasks.
Use the skill tool to load a skill when a task matches its description.

The new format Q4NX models for this will be located on huggingface `Atomic-Germ/*-nB-OpenNPU2`, each will need a builder in q4nx-build as they gain xclbin support. Family xclbins will be distributed with the application, and a converted model ships with the rest. Each model links to a family xclbin, regardless of fine-tuning or derivitive models. The purpose of this is to allow quick additions of new but shape-identical models without updates needed, distributed through `flm-add` to install and link properly.

As you encounter closed-source requirements, note them and replace with "not implemented" messaging before coding a replacement.

When investigating pre-compiled xclbins and libraries, save any useful tools you may create to the `utilities/` directory rather than leaving them in a temp folder.

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
</available_skills>

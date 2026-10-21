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

The whole toolchain is assembled by one script, found at run time (no path is
hardcoded in it):

```bash
source utilities/env.sh      # XRT, ROCm/HIP, MLIR-AIE + Peano, the tree's config
source utilities/env.fish    # same, for fish
```

It sets XRT (`xrt-smi`, `xclbinutil`, `aiebu-asm`), ROCm/HIP (`hipcc`),
MLIR-AIE + Peano (`aiecc`, `aie-opt`, the `aie` python package) and the
project's own paths (`OFLM_CONFIG_PATH`, `OFLM_XCLBIN_PATH`, `OFLM_MODEL_PATH`,
`OFLM_SETTINGS_PATH`), so `oflm` run from the tree reads the tree. Run
`utilities/env.sh --check` to see what it found. Peano stays out of PATH: its
clang would shadow the system one, and `aiecc` is handed it through
`PEANO_INSTALL_DIR`. `source third_party/mlir-aie/utils/env_setup.sh` is still
there for a from-source mlir-aie build.

`OFLM_MODEL_PATH` holds the directory that CONTAINS the models, not the
models: the registry names the subdirectory (`model_path`: `models`), so the
models are in `$OFLM_MODEL_PATH/models`, and `oflm-add` resolves it the same
way. `oflm config show` prints where each of these paths points right now.

A model in the models directory is only visible when the registry knows it:
the shipped `model_list.json` for the official ones, and the `models` section
of `settings.json` for the ones `oflm add` installed. A directory that is
there but not in either is invisible to `oflm list`. Register it again with
`oflm add` on the local directory — it takes one, and it does not copy the
weights that are already there:

```bash
source utilities/env.sh
oflm add src/models/Ornith-1.0-35B-A3B-NPU2 --family qwen3.6-moe \
             --xclbin-from Ornith-1.0-35B-A3B-NPU2
```

`--family` and `--xclbin-from` are not optional in practice: without them the
tool matches the closest official entry by name, and a 35B-A3B MoE ends up
with the kernels of an 8B model. A model whose directory is a symlink to
another one is the same weights under a second name — registering it would
replace the official entry of that tag in the merged registry, so leave it
alone.

`env.fish` is only a front end: it runs `env.sh` under bash and imports the
variables it exported, so the discovery stays in one place. The two stay in
step through `OFLM_ENV_DUMP=1`, which makes `env.sh` print its variables as
`KEY=VALUE` lines instead of its report.

## Git rules for agents

**Never use `git add -A`, `git add .`, or `git add --all`.** Stage the
specific paths you changed, by name:

```bash
git add src/config/config_store.cpp src/test/config_store_test.cpp
```

`src/models/` holds tens of gigabytes of model weights and `build/` holds a
build tree; a blanket add walks both, takes minutes, and can time out
mid-index-write. It also stages files that belong to other work in the tree,
so a commit picks up changes you never made and never looked at.

If you are unsure what is yours, `git status --short` first, then name the
paths explicitly. Never `git add` a path you did not edit.

Before committing, `git diff --cached --stat` must show only your files.

## The settings registry is generated, and CI checks it

`src/config/settings.json` is the source of truth for the user-facing
settings. `utilities/generate-registry.py` turns it into
`src/config/settings_registry_generated.hpp` (committed), and
`utilities/scan-settings.py` reads the source tree for the CLI flags and the
`OFLM_*` variables the code actually uses.

CI regenerates the registry and fails if the result differs from the
committed file. It also fails, before that, when the scan finds a flag or a
variable that `settings.json` does not classify: **every** CLI flag and
**every** `OFLM_*` variable the code reads must be

- a setting in `src/config/settings.json` (with type, scope, delivery,
  category), or
- listed in `NON_SETTING_ENV_VARS` / `NON_SETTING_FLAGS` in
  `utilities/generate-registry.py`, with a reason.

**The check runs against your branch MERGED WITH THE TARGET BRANCH.** Two
things break it, and only the first is yours:

1. your branch adds a CLI flag or an `OFLM_*` variable and does not
   classify it;
2. the target branch adds one while your branch is open — you did nothing,
   and the merge brings it in. This happened with `OFLM_SPECULATIVE`, added
   to `main` by #184, which broke the check on a branch that had never
   touched it.

So before you push, run the check the way CI runs it — it merges the target
branch in a scratch worktree, so your tree is untouched:

```bash
utilities/check-settings-ci.sh            # defaults to origin/main
utilities/check-settings-ci.sh origin/main
```

It tells you what to do when it fails: the flag or variable to add, and
whether the merged tree produces a different registry (in which case copy the
regenerated files over and commit them).

Adding a setting is documented in `src/create_new_model.md`.

<available_skills>
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
</available_skills>

---
name: open-qwen36-kernels
description: Build, verify and ship the open XDNA2 kernel sets (lx0 lx1 ax0 ax1 ln lm_head_q8) that the open Qwen3.6-MoE engine (src/open_qwen36) loads. Use when rebuilding those xclbins after a design change, checking a rebuild against a previous one, packaging them for a release, or debugging "no open kernels found" at model load.
---

# Open Qwen3.6-MoE kernels

The kernels are **built, not checked in** (`.gitignore`:
`src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels/`). Source:
`open_kernels/designs/{layer_x,ln,lm_head_q8}`. Reference docs:

1. `../../../src/open_qwen36/README.md` — the engine, the build table, the
   reproducibility result.
2. `../../../open_kernels/harness/README.md` — testing one design on the NPU
   (`run_kernel`, `make_test.py` / `compare.py`, fixtures).
3. `../../../open_kernels/PROVENANCE.md` — where the designs come from, licences,
   what is generated vs. tracked.

## Build (WSL, mlir-aie 1.4.2)

```
source ~/ironenv142/bin/activate                 # pip install -r ironvenv-requirements.txt
export PATH=~/xrt-tools/bin:$PATH                # xclbinutil + aiebu-asm from an XRT build
python open_kernels/export_qwen36_kernels.py     # ~4 min; -> src/xclbins/<model>/open_kernels/
```

Knobs per set are in the script's `SETS` table (`LX_PART`, `AX_PART`,
`LMHEAD_N`, `LMHEAD_CORES`); it clears them from the caller's shell first.
`--only`, `--out`, `--no-build`, `--check DIR`. `toolchain.json` beside the
output records versions, this tree's commit and sha256s.

## Verify

- `--check <previous export>`: `insts.bin` must be byte-identical; `final.xclbin`
  may differ only in build stamps (axlf unique id / timestamp / UUID, PDI UUID,
  boot-image header unique id + checksum, the mirror-JSON tail). Anything else
  is a real change — find out why before shipping.
- One design on the NPU: `(cd open_kernels/designs/ln && python make_test.py)`
  then on Windows `..\..\harness\out\run_kernel.exe run.cfg && python compare.py`.
- Whole decode step vs the fp64 oracle: `open_kernels/model/README.md`
  (`make_decode.py` / `compare_decode.py`; needs the model's `.q4nx`).
- Through the app: `flm serve qwen3.6-moe:35b-a3b` logs *"Qwen3.6-MoE on the
  open kernels"*; `FLM_QWEN36_ENGINE=open` fails loudly if the kernels are
  missing. Then `flm-test --llm --model qwen3.6-moe:35b-a3b`.

## Rules

- Never commit the xclbins; the distributed package is the only place they
  ship pre-built. Update `SETS` and the README table when a set is added.
- Fixtures that need captured FLM buffers read `OPEN_KERNELS_CAPS`
  (`open_kernels/fixture_paths.py`); do not hard-code capture or checkout paths.
- Record a rebuild's `--check` outcome and toolchain versions in the README
  when the toolchain pin changes.

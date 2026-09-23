# Qwen3.8-27B text-only bring-up

Status: in progress. No 27B hardware geometry promoted to the catalogue.

## Baseline (2026-09-23)

- Repository: `c23b1a57f23b6342457b8099ef1f14e3d73f4d47`.
- Initial worktree: only the user's untracked implementation-instructions Markdown.
- Python 3.14.4, NumPy 2.3.5, pytest 9.0.2.
- `ironvenv` absent; `mlir-aie` and `llvm-aie` not installed in host Python.
- XRT 2.26.0 (`e9db9ab15f10173f8d2fc93ff92ab4c7eb09d2e6`),
  amdxdna 2.26.0_20260817, firmware 1.1.2.64.
- Outside the sandbox, `xrt-smi examine` detects Strix aie2p (6x8),
  PCI `0000:67:00.1`, Ryzen AI 9 365. `/dev/accel/accel0` is hidden inside
  the sandbox. Docker is accessible outside the sandbox.
- `python3 -m pytest specs/open-engine/tests/test_qwen35.py -q`:
  **37 passed**.
- `python3 -m pytest specs/open-engine/tests -q`:
  **557 passed, 47 skipped, 2 failed**. Both failures predate changes:
  `test_vision_deepstack.py::test_the_merger_is_the_exact_gelu_not_the_tanh_one`
  and `test_vision_vit_windowed.py::test_the_activations_are_silu_and_the_exact_gelu`
  require missing `scipy.special`. Do not weaken or delete these tests.

## Phase 1: official geometry

Fixture is the unchanged official `config.json` from
`https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json`.
SHA256: `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`.
The fixture retains vision metadata for provenance; only its text tower is used.

The existing parser already derives `qwen35`, hidden 5120, FFN 17408,
64 layers, 24Q/4KV, head dim 256, rotary dim 64, 16 key / 48 value
DeltaNet heads of dimension 128, QKV width 10240, value width 6144,
vocabulary 248320, and `(linear, linear, linear, full) * 16`.
No architecture or parser change is needed.

TDD: added the geometry tests first (2 fixture-missing errors), then installed
the pinned fixture. Geometry plus existing Qwen3.5 tests: **39 passed**.
Hardware: **NOT VALIDATED**. Next: capture resource blockers as tests.

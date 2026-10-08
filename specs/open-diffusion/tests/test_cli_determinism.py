# Traces: OPEN-DIFFUSION-DETERMINISM (canonical spec: specs/open-diffusion/spec.md)
"""Two `oflm image` runs with the same prompt, size and seed write the same bytes.

Needs the NPU, an oflm.exe built with the open diffusion engine (OFLM_EXE, default
src/out/oflm.exe) and the installed flux2-klein:4b; skipped, saying which is missing,
without them. ~20 s at 512^2.
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
OFLM = Path(os.environ.get("OFLM_EXE", REPO / "src" / "out" / "oflm.exe"))
# the runtime's defaults: %USERPROFILE%\.oflm on Windows, ~/.config/oflm elsewhere
DEFAULT_ROOT = Path.home() / (".oflm" if os.name == "nt" else ".config/oflm")
MODEL_ROOT = Path(os.environ.get("OFLM_MODEL_PATH", DEFAULT_ROOT)) / "models"


def run_image(out: Path) -> bytes:
    r = subprocess.run([str(OFLM), "image", "flux2-klein:4b", "a red fox in fresh snow",
                        "--size", "512", "--seed", "1", "-o", str(out)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return out.read_bytes()


@pytest.mark.skipif(not OFLM.is_file(), reason=f"no oflm.exe at {OFLM} (set OFLM_EXE)")
@pytest.mark.skipif(not (MODEL_ROOT / "FLUX.2-klein-4B-NPU2" / "bundle.json").is_file(),
                    reason=f"flux2-klein:4b is not installed under {MODEL_ROOT}")
def test_same_seed_same_bytes(tmp_path):
    a = run_image(tmp_path / "a.png")
    b = run_image(tmp_path / "b.png")
    assert a[:8] == b"\x89PNG\r\n\x1a\n"
    assert a == b

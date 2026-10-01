# Traces: OPEN-DIFFUSION-DETERMINISM (canonical spec: specs/open-diffusion/spec.md)
"""Two `oflm image` runs with the same prompt, size and seed write the same bytes -- and two
edits of the same reference (`--image`).

Needs the NPU, an oflm.exe built with the open diffusion engine (OFLM_EXE, default
src/out/oflm.exe) and the installed flux2-klein:4b; skipped, saying which is missing,
without them. ~20 s at 512^2.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
OFLM = Path(os.environ.get("OFLM_EXE", REPO / "src" / "out" / "oflm.exe"))
MODEL_ROOT = Path(os.environ.get("OFLM_MODEL_PATH", Path.home() / ".oflm")) / "models"


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


def _model_edits() -> bool:
    try:
        return "512" in json.loads((MODEL_ROOT / "FLUX.2-klein-4B-NPU2" / "bundle.json").read_text()).get("edits", {})
    except (OSError, ValueError):
        return False


@pytest.mark.skipif(not OFLM.is_file(), reason=f"no oflm.exe at {OFLM} (set OFLM_EXE)")
@pytest.mark.skipif(not _model_edits(), reason=f"the flux2-klein:4b under {MODEL_ROOT} has no 512 edit configuration")
def test_the_same_edit_twice_gives_the_same_bytes(tmp_path):
    first = run_image(tmp_path / "ref.png")              # a 512x512 reference, made by the engine
    outs = []
    for name in ("a.png", "b.png"):
        r = subprocess.run([str(OFLM), "image", "flux2-klein:4b", "make it night", "--image",
                            str(tmp_path / "ref.png"), "--seed", "1", "-o", str(tmp_path / name)],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        outs.append((tmp_path / name).read_bytes())
    assert outs[0][:8] == b"\x89PNG\r\n\x1a\n"
    assert outs[0] == outs[1]
    assert outs[0] != first

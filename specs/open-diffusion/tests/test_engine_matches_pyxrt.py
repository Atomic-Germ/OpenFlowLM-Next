# Traces: OPEN-DIFFUSION-DETERMINISM (canonical spec: specs/open-diffusion/spec.md)
"""The native engine's pixels equal the pyxrt runner's on the same token ids and noise (and,
for an edit, the same prepared reference).

The engine runs every set in one hardware context per resolution, configured by
register writes (open_kernels/compose_elf.py); utilities/dit-chain/generate.py runs the
six sets' own xclbin contexts. A wrong reconfiguration is silent -- the image just
drifts -- so this compares them on study prompts 0 and 1 at 512^2, text encoder included.

Needs the NPU and:
  - the standalone engine CLI (OFLM_DIFFUSION_CLI, default src/open_diffusion/out/open_diffusion_cli.exe);
  - the installed model (OFLM_MODEL_PATH/models/FLUX.2-klein-4B-NPU2) and kernel set
    (OFLM_DIFFUSION_KERNELS_DIR, default src/xclbins/FLUX.2-klein-4B-NPU2/open_kernels);
  - a built kernel directory for generate.py (OFLM_DIFFUSION_BUILD_KERNELS, default C:\\dev\\klein-kernels);
  - the study inputs (OFLM_DIFFUSION_GOLDENS, default C:\\dev\\ditref-out\\goldens_pipe_512:
    utilities/dit-ref/capture_pipeline_inputs.py);
  - the IRON environment's Python with pyxrt (OFLM_IRON_PYTHON, default
    C:\\dev\\mlir-aie\\ironenv\\Scripts\\python.exe; XRT's python bindings from XRT_ROOT).
Skipped, naming what is missing, without them. ~60 s.
"""

import json
import os
import struct
import subprocess
import zlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CLI = Path(os.environ.get("OFLM_DIFFUSION_CLI", REPO / "src" / "open_diffusion" / "out" / "open_diffusion_cli.exe"))
MODEL = Path(os.environ.get("OFLM_MODEL_PATH", Path.home() / ".oflm")) / "models" / "FLUX.2-klein-4B-NPU2"
KERNELS = Path(os.environ.get("OFLM_DIFFUSION_KERNELS_DIR", REPO / "src" / "xclbins" / "FLUX.2-klein-4B-NPU2" / "open_kernels"))
BUILD = Path(os.environ.get("OFLM_DIFFUSION_BUILD_KERNELS", r"C:\dev\klein-kernels"))
GOLDENS = Path(os.environ.get("OFLM_DIFFUSION_GOLDENS", r"C:\dev\ditref-out\goldens_pipe_512"))
IRON_PY = Path(os.environ.get("OFLM_IRON_PYTHON", r"C:\dev\mlir-aie\ironenv\Scripts\python.exe"))
XRT_ROOT = Path(os.environ.get("XRT_ROOT", r"C:\Xilinx\XRT"))

NEEDS = {
    f"no engine CLI at {CLI} (src/open_diffusion/build.cmd)": CLI.is_file(),
    f"no model at {MODEL}": (MODEL / "bundle.json").is_file(),
    f"no installed kernel set at {KERNELS}": (KERNELS / "diffusion_kernels.json").is_file(),
    f"no built kernel directory at {BUILD}": (BUILD / "dit_kernels.json").is_file(),
    f"no study inputs at {GOLDENS}": (GOLDENS / "ids_0.npy").is_file(),
    f"no IRON Python at {IRON_PY}": IRON_PY.is_file(),
}


def png_rgb(path: Path) -> tuple[int, int, bytes]:
    """(width, height, RGB8 rows) of an 8-bit RGB, non-interlaced PNG: both writers' form."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path
    pos, idat, w, h = 8, b"", 0, 0
    while pos < len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, h, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", body)
            assert (depth, color, interlace) == (8, 2, 0), (path, depth, color, interlace)
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    raw, stride, out, prev = zlib.decompress(idat), 3 * w, bytearray(), bytearray(3 * w)
    for y in range(h):
        f, line = raw[y * (stride + 1)], bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            a = line[i - 3] if i >= 3 else 0
            b, c = prev[i], (prev[i - 3] if i >= 3 else 0)
            if f == 1:
                line[i] = (line[i] + a) & 0xFF
            elif f == 2:
                line[i] = (line[i] + b) & 0xFF
            elif f == 3:
                line[i] = (line[i] + (a + b) // 2) & 0xFF
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 0xFF
        out += line
        prev = line
    return w, h, bytes(out)


PROMPTS = 2        # two prompt lengths: te_attn runs a different valid_len head for each


@pytest.mark.skipif(not all(NEEDS.values()), reason="; ".join(k for k, ok in NEEDS.items() if not ok))
def test_engine_pixels_equal_pyxrt(tmp_path):
    import numpy as np

    ref_dir = tmp_path / "pyxrt"
    env = dict(os.environ, PYTHONPATH=str(XRT_ROOT / "python"),
               PATH=f"{XRT_ROOT};{XRT_ROOT / 'lib'};{os.environ.get('PATH', '')}")
    r = subprocess.run([str(IRON_PY), str(REPO / "utilities" / "dit-chain" / "generate.py"),
                        "--kernels", str(BUILD), "--size", "512", "--study", str(GOLDENS),
                        "--prompts", str(PROMPTS), "--out", str(ref_dir)],
                       capture_output=True, text=True, timeout=600, env=env)
    assert (ref_dir / "report.json").is_file(), r.stdout[-2000:] + r.stderr[-2000:]
    report = json.loads((ref_dir / "report.json").read_text())
    for i in range(PROMPTS):
        # the study's ids are padded to 512; the engine takes the prompt's own tokens
        n_real = report["images"][i]["tokens"]
        ids = np.load(GOLDENS / f"ids_{i}.npy")[:n_real]
        engine_png = tmp_path / f"engine_{i}.png"
        r = subprocess.run([str(CLI), "--model", str(MODEL), "--kernels", str(KERNELS), "--size", "512",
                            "--ids", ",".join(str(int(t)) for t in ids),
                            "--noise", str(GOLDENS / f"noise_{i}.npy"), "--out", str(engine_png)],
                           capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        want, got = png_rgb(ref_dir / f"{i:02d}.png"), png_rgb(engine_png)
        assert got[:2] == want[:2] == (512, 512)
        diff = sum(1 for x, y in zip(got[2], want[2]) if x != y)
        assert diff == 0, f"prompt {i} ({n_real} tokens): {diff} of {len(want[2])} channel values differ"


EDIT_GOLDENS = Path(os.environ.get("OFLM_DIFFUSION_EDIT_GOLDENS", r"C:\dev\ditref-out\klein_edit_512_s4"))


def _has_edit_512() -> bool:
    try:
        bundle = json.loads((MODEL / "bundle.json").read_text())
        manifest = json.loads((KERNELS / "diffusion_kernels.json").read_text())
        build = json.loads((BUILD / "dit_kernels.json").read_text())
    except (OSError, ValueError):
        return False
    return ("512" in bundle.get("edits", {}) and "512e512" in manifest.get("elf", {})
            and 512 in build.get("edits", []))


EDIT_NEEDS = dict(NEEDS) | {
    f"no 512 edit configuration in the model, the installed kernels and {BUILD}": _has_edit_512(),
    f"no edit study inputs at {EDIT_GOLDENS} (utilities/dit-ref/capture_edit_goldens.py)":
        (EDIT_GOLDENS / "ref_0.npy").is_file(),
}


@pytest.mark.skipif(not all(EDIT_NEEDS.values()), reason="; ".join(k for k, ok in EDIT_NEEDS.items() if not ok))
def test_engine_edit_pixels_equal_pyxrt(tmp_path):
    """An edit at 512^2 (its VAE encoder included): the engine's pixels equal generate.py --edit's."""
    import numpy as np

    ref_dir = tmp_path / "pyxrt"
    env = dict(os.environ, PYTHONPATH=str(XRT_ROOT / "python"),
               PATH=f"{XRT_ROOT};{XRT_ROOT / 'lib'};{os.environ.get('PATH', '')}")
    r = subprocess.run([str(IRON_PY), str(REPO / "utilities" / "dit-chain" / "generate.py"),
                        "--kernels", str(BUILD), "--size", "512", "--edit", "--study", str(EDIT_GOLDENS),
                        "--prompts", "1", "--out", str(ref_dir)],
                       capture_output=True, text=True, timeout=600, env=env)
    assert (ref_dir / "report.json").is_file(), r.stdout[-2000:] + r.stderr[-2000:]
    n_real = json.loads((ref_dir / "report.json").read_text())["images"][0]["tokens"]
    ids = np.load(EDIT_GOLDENS / "ids_0.npy")[:n_real]
    engine_png = tmp_path / "engine_edit.png"
    r = subprocess.run([str(CLI), "--model", str(MODEL), "--kernels", str(KERNELS), "--size", "512",
                        "--ids", ",".join(str(int(t)) for t in ids),
                        "--noise", str(EDIT_GOLDENS / "noise_0.npy"), "--ref", str(EDIT_GOLDENS / "ref_0.npy"),
                        "--out", str(engine_png)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    want, got = png_rgb(ref_dir / "00.png"), png_rgb(engine_png)
    assert got[:2] == want[:2] == (512, 512)
    diff = sum(1 for x, y in zip(got[2], want[2]) if x != y)
    assert diff == 0, f"edit 0 ({n_real} tokens): {diff} of {len(want[2])} channel values differ"

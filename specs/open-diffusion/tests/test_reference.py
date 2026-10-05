# Traces: OPEN-DIFFUSION-REFERENCE (canonical spec: specs/open-diffusion/spec.md)
"""An edit's reference image, prepared on the host as diffusers' PIL preprocessing prepares it.

src/open_diffusion/reference.cpp (through open_diffusion_reference_tool, which
src/open_diffusion/build.cmd builds; OFLM_REFERENCE_TOOL overrides the path) against PIL:
EXIF orientation, RGB conversion, the centre crop to a square and the LANCZOS resize.
Inputs are made here with PIL. Host code only: no NPU.
"""

from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageOps  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
TOOL = Path(os.environ.get("OFLM_REFERENCE_TOOL",
                           REPO / "src" / "open_diffusion" / "out" / "open_diffusion_reference_tool.exe"))
pytestmark = pytest.mark.skipif(not TOOL.is_file(),
                                reason=f"no reference tool at {TOOL} (src/open_diffusion/build.cmd)")


def scene(w: int, h: int, seed: int = 0) -> np.ndarray:
    """Smooth gradients plus hard-edged discs: what a resize's ringing shows on."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([128 + 100 * np.sin(x / 37 + y / 53), 128 + 100 * np.cos(x / 91 - y / 29),
                    (x * 255 // w + y * 255 // h) // 2], -1)
    for _ in range(30):
        cx, cy, r = rng.integers(0, w), rng.integers(0, h), rng.integers(20, 200)
        img[(x - cx) ** 2 + (y - cy) ** 2 < r * r] = rng.integers(0, 256, 3)
    return np.clip(img, 0, 255).astype(np.uint8)


def save(tmp: Path, img: Image.Image, name: str, **kw) -> Path:
    p = tmp / name
    img.save(p, **kw)
    return p


def prepare(path: Path, R: int, tmp: Path) -> tuple[int, np.ndarray | None, str]:
    """(exit code, [R, R, 3] uint8 or None, stdout + stderr)."""
    out = tmp / "out.raw"
    r = subprocess.run([str(TOOL), str(path), str(R), str(out)], capture_output=True, text=True,
                       timeout=120)
    img = np.fromfile(out, np.uint8).reshape(R, R, 3) if r.returncode == 0 else None
    return r.returncode, img, r.stdout + r.stderr


def pil_prepare(img: Image.Image, R: int) -> np.ndarray:
    """diffusers' path for a reference: load_image (EXIF orientation, RGB), then this
    pipeline's square crop and PIL LANCZOS to R (utilities/dit-ref/capture_edit_goldens.py)."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return np.asarray(img if s == R else img.resize((R, R), Image.Resampling.LANCZOS))


def test_a_square_png_at_the_output_size_passes_through(tmp_path):
    a = scene(512, 512)
    code, got, _ = prepare(save(tmp_path, Image.fromarray(a), "a.png"), 512, tmp_path)
    assert code == 0
    assert np.array_equal(got, a)


@pytest.mark.parametrize("w, h, R", [(4000, 3000, 512), (3000, 4000, 1024), (640, 480, 512),
                                     (300, 300, 512), (1000, 750, 1024)])
def test_crop_and_resize_match_pil(tmp_path, w, h, R):
    """The centred square, resized as PIL resizes it (PNG: no decoder in the comparison)."""
    a = scene(w, h, seed=w + h)
    code, got, text = prepare(save(tmp_path, Image.fromarray(a), "a.png"), R, tmp_path)
    assert code == 0, text
    want = pil_prepare(Image.fromarray(a), R)
    d = np.abs(got.astype(int) - want.astype(int))
    assert d.max() <= 1, f"max {d.max()} levels from PIL"


def test_a_4000x3000_jpeg_gives_the_centred_crop(tmp_path):
    """Bars outside the centred 3000x3000 square must not reach the output; the rest is
    PIL's within JPEG decoding's own differences (stb_image vs libjpeg)."""
    a = scene(4000, 3000)
    a[..., 1] = a.mean(-1)                         # the inside holds no magenta: G >= min(R, B) / 3
    a[:, :500] = (255, 0, 255)
    a[:, -500:] = (255, 0, 255)
    p = save(tmp_path, Image.fromarray(a), "a.jpg", quality=95)
    code, got, text = prepare(p, 512, tmp_path)
    assert code == 0, text
    assert "centre-cropped to 3000x3000" in text
    want = pil_prepare(Image.open(p), 512)
    d = np.abs(got.astype(int) - want.astype(int))
    assert d.mean() < 0.5 and np.percentile(d, 99.9) <= 3, (d.mean(), np.percentile(d, 99.9))
    g = got.astype(int)
    magenta = (g[..., 0] > 150) & (g[..., 2] > 150) & (g[..., 1] < 60)
    assert not magenta.any(), "a bar outside the centred square reached the output"


@pytest.mark.parametrize("mode", ["RGBA", "LA", "L", "P"])
def test_other_modes_become_rgb_as_pil_converts_them(tmp_path, mode):
    a = scene(256, 256)
    img = Image.fromarray(a)
    if mode in ("RGBA", "LA"):
        img = img.convert(mode)
        alpha = Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (256, 1)))
        img.putalpha(alpha)                       # varying alpha: dropped, not composited
    else:
        img = img.convert(mode)
    code, got, text = prepare(save(tmp_path, img, f"a_{mode}.png"), 256, tmp_path)
    assert code == 0, text
    assert np.array_equal(got, np.asarray(img.convert("RGB")))


def with_orientation(jpeg: bytes, orientation: int) -> bytes:
    """The same JPEG bytes with an EXIF APP1 segment (orientation tag) inserted after SOI:
    the pixels decode identically, with and without it."""
    exif = Image.Exif()
    exif[0x0112] = orientation
    payload = exif.tobytes()                                  # b"Exif\x00\x00" + TIFF
    assert payload.startswith(b"Exif\x00\x00")
    seg = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
    assert jpeg[:2] == b"\xff\xd8"
    return jpeg[:2] + seg + jpeg[2:]


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation_is_applied_as_pil_applies_it(tmp_path, orientation):
    """A 600x400 JPEG tagged with orientation o gives exactly the untagged file's output
    transposed as ImageOps.exif_transpose transposes for o (the centre crop commutes with
    it: the sides differ by an even count)."""
    buf = io.BytesIO()
    Image.fromarray(scene(600, 400)).save(buf, format="JPEG", quality=95)
    plain = tmp_path / "plain.jpg"
    plain.write_bytes(buf.getvalue())
    tagged = tmp_path / f"o{orientation}.jpg"
    tagged.write_bytes(with_orientation(buf.getvalue(), orientation))
    assert ImageOps.exif_transpose(Image.open(tagged)).size == ((400, 600) if orientation >= 5 else (600, 400))

    # the untagged frame as the tool decodes it: rows 0..399, its centre 400 columns
    code, stored, text = prepare(plain, 400, tmp_path)
    assert code == 0, text
    code, got, text = prepare(tagged, 400, tmp_path)
    assert code == 0, text
    if orientation != 1:
        assert f"EXIF orientation {orientation}" in text
    im = Image.fromarray(stored)
    im.getexif()[0x0112] = orientation
    assert np.array_equal(got, np.asarray(ImageOps.exif_transpose(im)))


def test_orientation_6_matches_pil_end_to_end(tmp_path):
    """The phone case: a landscape-stored JPEG shown portrait, through PIL's own
    load_image path (within JPEG decoding's differences)."""
    buf = io.BytesIO()
    Image.fromarray(scene(800, 600)).save(buf, format="JPEG", quality=95)
    p = tmp_path / "phone.jpg"
    p.write_bytes(with_orientation(buf.getvalue(), 6))
    code, got, text = prepare(p, 512, tmp_path)
    assert code == 0, text
    assert "reference 600x800" in text
    d = np.abs(got.astype(int) - pil_prepare(Image.open(p), 512).astype(int))
    assert d.mean() < 0.5 and np.percentile(d, 99.9) <= 3, (d.mean(), np.percentile(d, 99.9))


@pytest.mark.parametrize("w, h, why", [
    (63, 100, "at least 64 px"),
    (100, 63, "at least 64 px"),
    (900, 100, "more elongated than 8:1"),
])
def test_bad_shapes_are_refused_by_name(tmp_path, w, h, why):
    p = save(tmp_path, Image.fromarray(scene(w, h)), "a.png")
    code, _, text = prepare(p, 512, tmp_path)
    assert code == 3 and why in text, text


def test_over_64_megapixels_is_refused_before_decoding(tmp_path):
    p = save(tmp_path, Image.new("RGB", (9000, 8000), (10, 20, 30)), "big.png")
    code, _, text = prepare(p, 512, tmp_path)
    assert code == 3 and "megapixel limit" in text, text


@pytest.mark.parametrize("fmt", ["BMP", "WEBP", "GIF"])
def test_other_formats_are_refused_as_not_implemented(tmp_path, fmt):
    try:
        p = save(tmp_path, Image.fromarray(scene(128, 128)), f"a.{fmt.lower()}", format=fmt)
    except (KeyError, OSError):
        pytest.skip(f"this PIL cannot write {fmt}")
    code, _, text = prepare(p, 512, tmp_path)
    assert code == 3 and "not PNG or JPEG" in text and "not implemented" in text, text


def test_a_truncated_png_is_refused(tmp_path):
    p = save(tmp_path, Image.fromarray(scene(256, 256)), "a.png")
    p.write_bytes(p.read_bytes()[:200])
    code, _, text = prepare(p, 512, tmp_path)
    assert code == 3, text

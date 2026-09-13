#!/usr/bin/env python3
"""Write the PNG fixtures OPEN-VISION-IMAGE-READ checks against.

Each case is a .png plus a .rgb holding the tightly packed RGB24 the reader is
supposed to produce. The expectation is computed here from the source pixels,
independently of the C++ decoder.

The PNGs are hand-built rather than saved through Pillow, because Pillow will
not write what these cases need: it ignores `bits=` for L images (so the 1/2/4
bit depths came out as 8), it has no interlaced-PNG writer, and it never chose
the Average predictor. Writing the chunks directly lets each file CYCLE all
five row filters down its rows, so a decoder missing any one of them fails.

    python specs/open-engine/tests/make_png_fixtures.py
"""
from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "fixtures" / "png"
# The two images oflm-test bundles, copied in so the test does not reach across
# the tree into a tool's package data.
BUNDLED = HERE.parents[2] / "utilities" / "oflm-test" / "oflm_test" / "test_files" / "image"

W, H = 37, 23          # not a multiple of 8, so the sub-byte depths pad their rows
CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}

# Adam7: (x start, y start, x step, y step) per pass.
ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
         (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]


def chunk(tag: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body))


def pack_row(samples: np.ndarray, depth: int) -> bytes:
    """One scanline of samples (already reduced to `depth` bits) into bytes."""
    flat = samples.reshape(-1)
    if depth == 8:
        return flat.astype(np.uint8).tobytes()
    if depth == 16:
        return flat.astype(">u2").tobytes()
    per = 8 // depth
    out = bytearray((len(flat) + per - 1) // per)
    for i, v in enumerate(flat):
        shift = 8 - depth * (i % per + 1)
        out[i // per] |= int(v) << shift
    return bytes(out)


def paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def filter_row(cur: bytes, prev: bytes, ft: int, bpp: int) -> bytes:
    if ft == 0:
        return cur
    out = bytearray(len(cur))
    for i, x in enumerate(cur):
        a = cur[i - bpp] if i >= bpp else 0
        b = prev[i] if prev else 0
        c = prev[i - bpp] if (prev and i >= bpp) else 0
        if ft == 1:
            out[i] = (x - a) & 0xFF
        elif ft == 2:
            out[i] = (x - b) & 0xFF
        elif ft == 3:
            out[i] = (x - ((a + b) >> 1)) & 0xFF
        else:
            out[i] = (x - paeth(a, b, c)) & 0xFF
    return bytes(out)


def scanlines(samples: np.ndarray, depth: int, nch: int, first_filter: int = 0) -> bytes:
    """Filtered scanlines, cycling the five predictors down the rows."""
    h = samples.shape[0]
    bpp = max(1, (nch * depth + 7) // 8)
    raw = bytearray()
    prev = b""
    for y in range(h):
        cur = pack_row(samples[y], depth)
        ft = (y + first_filter) % 5
        raw.append(ft)
        raw += filter_row(cur, prev, ft, bpp)
        prev = cur
    return bytes(raw)


def encode(path: Path, samples: np.ndarray, depth: int, color_type: int,
           palette: bytes | None = None, interlace: int = 0, level: int = 9) -> None:
    """samples is (h, w, nch) of ints already in range for `depth`."""
    h, w, nch = samples.shape
    assert nch == CHANNELS[color_type], (nch, color_type)
    if interlace == 0:
        idat = scanlines(samples, depth, nch)
    else:
        idat = bytearray()
        for pi, (x0, y0, dx, dy) in enumerate(ADAM7):
            sub = samples[y0::dy, x0::dx, :]
            if sub.shape[0] == 0 or sub.shape[1] == 0:
                continue
            idat += scanlines(sub, depth, nch, first_filter=pi)
        idat = bytes(idat)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, color_type, 0, 0, interlace))
    if palette is not None:
        png += chunk(b"PLTE", palette)
    png += chunk(b"IDAT", zlib.compress(bytes(idat), level))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def to_rgb24(samples: np.ndarray, depth: int, color_type: int, palette: bytes | None) -> bytes:
    """What the reader must produce: RGB24, alpha dropped rather than composited,
    16-bit samples truncated to their high byte.

    Pillow agrees with this on every case here except 16-bit grayscale, where it
    opens the file as I;16 and its RGB conversion saturates at 255 instead of
    scaling. Taking the high byte is what sws_scale does, so that is the
    expectation; Pillow is not the oracle for that one case.
    """
    a = samples
    if depth == 16:
        a = a >> 8
    elif depth < 8:
        a = a * (255 // ((1 << depth) - 1))     # 1/2/4-bit greys scale to 0..255
    a = a.astype(np.uint8)
    if color_type == 0:
        rgb = np.repeat(a, 3, axis=2)
    elif color_type == 4:
        rgb = np.repeat(a[:, :, :1], 3, axis=2)
    elif color_type in (2, 6):
        rgb = a[:, :, :3]
    else:
        pal = np.frombuffer(palette, dtype=np.uint8).reshape(-1, 3)
        rgb = pal[samples[:, :, 0]]
    return np.ascontiguousarray(rgb).tobytes()


def case(name: str, samples: np.ndarray, depth: int, color_type: int,
         palette: bytes | None = None, interlace: int = 0, want_rgb: bool = True,
         level: int = 9) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    encode(OUT / f"{name}.png", samples, depth, color_type, palette, interlace, level)
    if want_rgb:
        (OUT / f"{name}.rgb").write_bytes(to_rgb24(samples, depth, color_type, palette))
    print(f"  {name}.png  {samples.shape[1]}x{samples.shape[0]} depth={depth} "
          f"type={color_type} interlace={interlace}")


def noise(nch: int, seed: int, hi: int) -> np.ndarray:
    """A gradient plus noise, so every predictor is a plausible choice somewhere."""
    rng = np.random.default_rng(seed)
    g = np.linspace(0, hi, W, dtype=np.float64)[None, :, None]
    v = np.linspace(0, hi, H, dtype=np.float64)[:, None, None]
    a = (g * 0.5 + v * 0.5 + rng.integers(-hi // 6 - 1, hi // 6 + 1, (H, W, nch))).clip(0, hi)
    return a.astype(np.int64)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    case("rgb8", noise(3, 1, 255), 8, 2)
    case("rgba8", noise(4, 2, 255), 8, 6)
    case("gray8", noise(1, 3, 255), 8, 0)
    case("graya8", noise(2, 4, 255), 8, 4)
    case("rgb16", noise(3, 5, 65535), 16, 2)
    case("gray16", noise(1, 6, 65535), 16, 0)
    case("gray4", noise(1, 7, 15), 4, 0)
    case("gray2", noise(1, 8, 3), 2, 0)
    case("gray1", noise(1, 9, 1), 1, 0)

    rng = np.random.default_rng(10)
    pal256 = rng.integers(0, 256, (256, 3), dtype=np.uint8).tobytes()
    case("palette8", noise(1, 11, 255), 8, 3, palette=pal256)
    pal16 = rng.integers(0, 256, (16, 3), dtype=np.uint8).tobytes()
    case("palette4", noise(1, 12, 15), 4, 3, palette=pal16)

    # Refused, not decoded: the .rgb would only mislead, so it is not written.
    case("interlaced", noise(3, 13, 255), 8, 2, interlace=1, want_rgb=False)

    # The other two deflate block types. zlib level 9 emits dynamic-Huffman
    # blocks for everything above, so without these two the decoder's stored and
    # fixed paths are never executed.
    case("rgb8_stored", noise(3, 14, 255), 8, 2, level=0)
    case("rgb8_fixed", noise(3, 15, 255)[:4, :5], 8, 2, level=1)

    whole = (OUT / "rgb8.png").read_bytes()
    (OUT / "truncated.png").write_bytes(whole[: len(whole) - 40])
    print("  truncated.png")

    # The two real images oflm-test sends - the ones that were being dropped.
    # They are NOT copied in: the test reads them where they already live and
    # checks the sha256 of the RGB24 Pillow produces, so the fixture directory
    # does not carry 2.4 MB of a second copy.
    from PIL import Image
    expect = {}
    for name in ("paris", "spectrogram"):
        src = BUNDLED / f"{name}.png"
        if not src.is_file():
            print(f"  SKIP {name}: {src} not found")
            continue
        rgb = np.ascontiguousarray(np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8))
        expect[name] = {"width": rgb.shape[1], "height": rgb.shape[0],
                        "sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}
        print(f"  {name}: {rgb.shape[1]}x{rgb.shape[0]} sha256 {expect[name]['sha256'][:16]} (Pillow)")
    (OUT / "bundled.json").write_text(json.dumps(expect, indent=2) + chr(10))

    # Pillow is an independent decoder, so use it as a cross-check on every
    # generated case it can read. It disagrees on 16-bit grayscale only, for the
    # reason to_rgb24 records.
    bad = []
    for f in sorted(OUT.glob("*.rgb")):
        png = f.with_suffix(".png")
        if not png.is_file():
            continue
        got = np.ascontiguousarray(np.asarray(Image.open(png).convert("RGB"), dtype=np.uint8)).tobytes()
        if got != f.read_bytes():
            bad.append(f.stem)
    unexpected = [b for b in bad if b != "gray16"]
    print(f"  Pillow cross-check: disagrees on {bad or 'nothing'}")
    if unexpected:
        print(f"  ERROR: unexpected Pillow disagreement on {unexpected}")
        return 1

    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

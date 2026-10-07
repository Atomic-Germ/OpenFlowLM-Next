"""Score a complete decode fixture against make_decode.py's pinned CPU reference.

    python open_kernels/model/compare_decode.py [--tokens N] [--out DIR]

All declared tokens and layers are required. Logits require correlation > 0.9999
and equal argmax; every residual requires normalized max error < 0.005. Top-5
and residual correlation are diagnostics. All captures must be finite and have
exact manifest-derived dimensions. Reference hashes must match the fixture.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from decode_reference import load_reference, read_capture, sfx

HERE = Path(__file__).resolve().parent
LOGIT_CORRELATION_MIN = 0.9999
RESIDUAL_MAXREL = 0.005


def correlation(got: np.ndarray, ref: np.ndarray) -> float:
    # A constant vector has no Pearson correlation; avoid runtime warnings.
    if np.ptp(got) == 0 or np.ptp(ref) == 0:
        return float('nan')
    return float(np.corrcoef(got, ref)[0, 1])


def compare(out: Path, tokens: int | None = None) -> bool:
    meta = load_reference(out)
    if tokens is not None and tokens != meta['tokens']:
        raise ValueError(f'--tokens {tokens} differs from declared tokens {meta["tokens"]}; compare the complete run')
    print(f'reference: {meta["layers"]} layers, {meta["tokens"]} tokens, '
          f'routing {meta["routing"]}, spec {meta["spec_hash"]}')
    allok = True
    for t in range(meta['tokens']):
        suffix = sfx(t)
        name = f'ref_logits{suffix}.bin'
        ref = read_capture(out / name, meta['vocab'], meta['references'][name])
        ours = read_capture(out / f'y_logits{suffix}.bin', meta['vocab'])
        corr = correlation(ours, ref)
        print(f'token {t} (position {t}): logits corr {corr:.6f}  argmax ours {int(ours.argmax())} '
              f'ref {int(ref.argmax())}  top5 ours {np.argsort(-ours)[:5].tolist()} '
              f'ref {np.argsort(-ref)[:5].tolist()} (top5 diagnostic)')
        ok = corr > LOGIT_CORRELATION_MIN and int(ours.argmax()) == int(ref.argmax())
        for layer in range(meta['layers']):
            name = f'ref_res{layer}{suffix}.bin'
            r = read_capture(out / name, meta['hidden'], meta['references'][name])
            g = read_capture(out / f'y_res{layer}{suffix}.bin', meta['hidden'])
            scale = float(np.abs(r).max())
            error = float(np.abs(g - r).max())
            maxrel = error / scale if scale else (0.0 if error == 0 else float('inf'))
            layer_ok = maxrel < RESIDUAL_MAXREL
            print(f'  layer {layer}: residual corr {correlation(g, r):.6f} '
                  f'maxrel {maxrel:.6e} (bound < {RESIDUAL_MAXREL}) '
                  f'{"PASS" if layer_ok else "FAIL"}')
            ok &= layer_ok
        print(f'token {t}: {"PASS" if ok else "FAIL"}')
        allok &= ok
    return bool(allok)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tokens', type=int, default=None,
                    help='must equal fixture token count; default checks all declared tokens')
    ap.add_argument('--out', default=str(HERE / 'out'))
    a = ap.parse_args()
    if a.tokens is not None and a.tokens <= 0:
        ap.error('--tokens must be positive')
    try:
        ok = compare(Path(a.out), a.tokens)
    except (OSError, ValueError) as e:
        print(f'FAIL: {e}')
        return 1
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

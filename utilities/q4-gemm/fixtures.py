"""Deterministic pool-byte fixtures and FP64 high/low projection references."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'open_kernels'))
sys.path.insert(0, str(ROOT / 'npu_offload/gemm_rtp'))
from q4_1_pack import dequant_pool, pack_q4_1_pool, random_q4_1_blocks
from npue import tile_b


def references(weights: np.ndarray, activations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use exact dequantized weights for high, and their BF16 residual for low.

    The high reference retains the existing BF16 weight-rounding error budget.
    The low reference independently checks the companion's mathematical role.
    """
    high = weights.astype(bfloat16).astype(np.float32)
    low = (weights - high).astype(bfloat16).astype(np.float32)
    x64 = activations.astype(np.float64).T
    return ((weights.astype(np.float64) @ x64).astype('<f4'),
            (low.astype(np.float64) @ x64).astype('<f4'))


def create(n: int, k: int, tokens: int, seed: int, out: Path) -> None:
    if n <= 0 or n % 256 or k <= 0 or k % 256 or tokens <= 0 or tokens % 256:
        raise ValueError('positive 256-aligned dimensions required')
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    pool = pack_q4_1_pool(random_q4_1_blocks(n, k, rng), rs=2)
    activations = rng.standard_normal((tokens, k)).astype(np.float32).astype(bfloat16)
    # Decode the packed bytes, not the pre-pack tensor, just like pool_reference.
    weights = dequant_pool(pool, n, k, rs=2)
    high, low = references(weights, activations)
    tiled = tile_b(np.ascontiguousarray(activations.T).view(np.uint16), 64, 32, 8, 8, order='k,n')
    for name, values in [('weights.bin', pool), ('activations.bin', tiled),
                         ('reference-high.bin', high), ('reference-low.bin', low)]:
        values.tofile(out / name)
    paths = [Path(__file__).resolve(), ROOT / 'open_kernels/q4_1_pack.py',
             ROOT / 'npu_offload/gemm_rtp/npue.py', *out.glob('*.bin')]
    hashes = {}
    for path in paths:
        with path.open('rb') as stream:
            hashes[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (out / 'fixture.json').write_text(json.dumps(dict(n=n, k=k, tokens=tokens, seed=seed,
                                                    files=hashes), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--seed', type=int, default=26026)
    parser.add_argument('--out', type=lambda v: Path(v).resolve(), required=True)
    args = parser.parse_args()
    create(args.n, args.k, args.tokens, args.seed, args.out)

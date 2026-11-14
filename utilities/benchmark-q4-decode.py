#!/usr/bin/env python3
"""Check exact CPU Q4 dequantization equivalence and time the legacy/new layouts.

Run with OPENBLAS_NUM_THREADS=1 and --model-dir pointing to a Q4NX container.
This measures reference preparation, not NPU inference performance.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'open_kernels' / 'model'))
from q4nx import Q4NX, bf16_to_f32, dq_chunks_q4_1, q4_1_chunks_of  # noqa: E402


def legacy_decode(chunks):
    """Frozen pre-optimization oracle, including its strided advanced indexing."""
    nch = chunks.shape[0]
    meta = bf16_to_f32(np.ascontiguousarray(chunks[:, :1024]).view(np.uint16))
    d, mn = meta[:, :256], meta[:, 256:]
    q = chunks[:, 1024:]
    n = np.empty((nch, 8192), dtype=np.float32)
    n[:, 0::2] = q & 0xF
    n[:, 1::2] = q >> 4
    r = np.arange(32)[:, None, None]
    bc = np.arange(8)[None, :, None]
    i = np.arange(32)[None, None, :]
    p = (r // 16) * 4096 + bc * 512 + i * 16 + (r % 16)
    j = bc * 32 + r + 0 * i
    vals = n[:, p.reshape(-1)].reshape(nch, 32, 8, 32)
    return vals * d[:, j.reshape(-1)].reshape(nch, 32, 8, 32) + mn[:, j.reshape(-1)].reshape(nch, 32, 8, 32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--tensor', default='model.layers.0.mlp.up_proj.weight')
    a = ap.parse_args()
    q = Q4NX(a.model_dir / 'model.q4nx')
    chunks = q4_1_chunks_of(q, a.tensor)
    start = time.perf_counter()
    old = legacy_decode(chunks)
    old_seconds = time.perf_counter() - start
    start = time.perf_counter()
    new = dq_chunks_q4_1(chunks)
    new_seconds = time.perf_counter() - start
    identical = old.tobytes() == new.tobytes()
    print(json.dumps(dict(tensor=a.tensor, chunks=len(chunks), byte_identical=identical,
                         legacy_seconds=old_seconds, contiguous_seconds=new_seconds,
                         speedup=old_seconds / new_seconds), indent=2))
    return 0 if identical else 1


if __name__ == '__main__':
    sys.exit(main())

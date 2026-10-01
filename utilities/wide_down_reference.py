"""FP64 partials from unchanged pool bytes; diagnostic only, never device input."""
import numpy as np
from ml_dtypes import bfloat16
from q4_1_pack import chunk_geometry, dequant_chunk
from recipes.segmented_dense import segments


def partial_reference(pool, x, n, k, limit=4096):
    parts = segments(k,limit)
    result = np.zeros((len(parts),n),np.float64)
    activation = x.astype(bfloat16).astype(np.float64)
    if activation.shape!=(k,): raise ValueError('activation shape')
    _,_,rows,cols = chunk_geometry(n,k,2)
    for c,(r,col) in enumerate(zip(rows,cols)):
        w = dequant_chunk(pool[c*5120:(c+1)*5120]).astype(np.float64)
        result[col//limit,r:r+32] += w@activation[col:col+256]
    return result


def reduction_diagnosis(parts):
    rounded = parts.astype(np.float32)
    acc = np.zeros(parts.shape[1],np.float32)
    for part in rounded: acc = (acc+part).astype(np.float32)
    return dict(exact=parts.sum(axis=0).astype(np.float32),
                rounded_partials=rounded.astype(np.float64).sum(axis=0).astype(np.float32),
                sequential=acc)

"""Independent FP64 oracle for gated decode attention; test tooling only."""
import numpy as np
from ml_dtypes import bfloat16


def norm_rope(x, weights, cs):
    x = x.astype(np.float64)
    y = x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+1e-6) * weights.astype(np.float64)
    half = len(cs)//2
    a, b = y[..., :half].copy(), y[..., half:2*half].copy()
    c, s = cs[:half].astype(np.float64), cs[half:].astype(np.float64)
    y[..., :half] = a*c-b*s
    y[..., half:2*half] = b*c+a*s
    return y


def decode(qg, kvn, norms, cs, cache, pos):
    nh, hd = qg.shape[1:]
    kvh = kvn.shape[1]
    if nh % kvh or not 0 <= pos <= len(cache):
        raise ValueError('invalid grouped-query geometry or cache position')
    q = norm_rope(qg[0],norms[0],cs)
    k = norm_rope(kvn[0],norms[1],cs).astype(np.float32).astype(bfloat16)
    v = kvn[1].astype(bfloat16)
    new = np.stack([k,v])
    # Only past rows participate. Poisoned future rows test device masking.
    rows = np.concatenate([cache[:pos],new[None]],axis=0).astype(np.float64)
    heads = np.arange(nh)//(nh//kvh)
    keys, values = rows[:,0,heads,:], rows[:,1,heads,:]
    scores = np.einsum('hd,thd->ht',q,keys)/np.sqrt(hd)
    p = np.exp(scores-scores.max(axis=1,keepdims=True))
    p /= p.sum(axis=1,keepdims=True)
    output = np.einsum('ht,thd->hd',p,values)
    gate = qg[1].astype(np.float64)
    output *= np.exp(-np.logaddexp(0,-gate))
    return new, output


def metric(got, ref, tolerance):
    if got.shape != ref.shape:
        return dict(passed=False, error='shape mismatch')
    g, r = got.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
    if not np.isfinite(g).all() or not np.isfinite(r).all():
        return dict(passed=False, error='nonfinite')
    gs, rs = np.max(np.abs(g)), np.max(np.abs(r))
    error = float(np.max(np.abs(g-r)))
    rel = float(error/rs) if rs else (0.0 if error == 0 else None)
    if np.array_equal(g,r):
        cos = 1.0
    elif not gs or not rs:
        cos = 0.0
    else:
        a,b = g/gs,r/rs
        cos = float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
    return dict(passed=rel is not None and rel<tolerance and cos>.9999,
                maxrel=rel, cosine=cos)

"""Where the per-token prefill path loses precision, stage by stage.

The block (layer-major) route is accurate; the per-token `step()` route is 17x
worse per token and fails every logit. Both run the same kernels, so the deficit
is inside the layer, and the useful question is WHICH operation first exceeds the
whole-layer accuracy the block route achieves.

This compares each captured scratch region against an independent FP64
reference computed from the same layer input. The regions are the manifest's own
offsets, so the comparison is of what the NPU actually wrote, not of a
re-derivation:
  xn  -- entry RMSNorm
  qkv -- fused QKV projection
  z   -- gate projection
  o   -- DeltaNet recurrence output, before the output projection

A stage that is already worse than the block route's ENTIRE layer output is
where a fix has to start; a stage below it is not the culprit even if it is not
bit-exact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'open_kernels'))
sys.path.insert(0, str(ROOT / 'open_kernels' / 'model'))

STAGES = ('xn', 'qkv', 'z', 'o')


def rel(got, ref):
    """max|got-ref| / max|ref|; None when the reference row has no scale."""
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    if got.shape != ref.shape:
        raise ValueError(f'shape mismatch: {got.shape} vs {ref.shape}')
    den = float(np.max(np.abs(ref)))
    return None if den == 0.0 else float(np.max(np.abs(got - ref)) / den)


def first_stage_over(stages, ceiling):
    """The first stage whose error exceeds `ceiling`, in pipeline order.

    This is the actionable output: it names the operation a fix must start at,
    rather than the largest number anywhere in the layer.
    """
    for name in STAGES:
        value = stages.get(name)
        if value is None:
            continue
        if value > ceiling:
            return name
    return None


def verdict(stages, ceiling):
    name = first_stage_over(stages, ceiling)
    return dict(stages=stages, ceiling=ceiling,
                first_stage_over_ceiling=name,
                within_block_route_accuracy=name is None,
                note=('every observed stage is at or below the block route\'s whole-layer '
                      'accuracy' if name is None else
                      f'{name} already exceeds the block route\'s whole-layer accuracy'))


def probe(model_dir, kernel_manifest, capture_dir, token, layer=0):
    """Compare the captured stage regions of one token against an FP64 reference."""
    from q4nx import Q4NX
    from recipes.load import spec_from_model_dir
    from recipes.qwen35 import layout
    import deltanet_inputs as di
    import deltanet_state as ds

    model_dir, capture_dir = Path(model_dir), Path(capture_dir)
    spec = spec_from_model_dir(model_dir)
    man = json.loads(Path(kernel_manifest).read_text())
    off = layout(spec, max_ctx=man['max_ctx_default'])
    model = Q4NX(model_dir / 'model.q4nx')
    model.native_q8 = lambda name: False
    try:
        x = np.asarray(model.embed(token, spec.hidden), dtype='<f4').astype(np.float64)
        weights, _ = di.load_weights(model, spec, layer)
        exact = x / np.sqrt(np.mean(x * x) + spec.norm_eps) * weights['norm']
        qkv = weights['qkv'] @ exact
        records, _ = di.records_reference(spec, exact, qkv,
                                          np.zeros((spec.conv_kernel - 1, spec.lin_qkv_dim)), weights)
        _, o = ds.step_reference(np.zeros((spec.lin_value_heads, 128, 128)), records)
        wz = model.matmul_w(f'model.layers.{layer}.self_attn.gate_proj.weight',
                            spec.lin_value_width, spec.hidden).astype(np.float64)
    finally:
        model.mm.close()
        model.f.close()

    def read(name, count):
        path = capture_dir / f'{name}.bin'
        if not path.is_file():
            raise FileNotFoundError(f'missing capture: {path}')
        data = np.fromfile(path, dtype='<f4')
        if data.size < count:
            raise ValueError(f'{path}: {data.size} floats, need {count}')
        return data[:count]

    observed = dict(xn=read('xn', spec.hidden),
                    qkv=read('qkv', spec.lin_qkv_dim),
                    z=read('z', spec.lin_value_width),
                    o=read('o', spec.lin_value_width).reshape(-1, 128))
    reference = dict(xn=exact, qkv=qkv, z=wz @ exact, o=o)
    return {name: rel(observed[name], reference[name]) for name in STAGES}


def run(a):
    stages = probe(a.model, a.manifest, a.capture, a.token, a.layer)
    result = dict(complete=True, model_acceptance=False, layer=a.layer, token=a.token,
                  route='per-token step() scratch regions',
                  scope='stage localisation inside one layer; a diagnosis, not acceptance')
    result.update(verdict(stages, a.ceiling))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for name in STAGES:
        value = stages[name]
        flag = '  <-- first over ceiling' if name == result['first_stage_over_ceiling'] else ''
        print(f'  {name:4s} maxrel {"n/a" if value is None else f"{value:.3e}"}{flag}')
    print(f'ceiling (block-route whole-layer accuracy): {a.ceiling:.3e}')
    print(f'verdict: {result["note"]}')
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--token', type=int, required=True)
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--ceiling', type=float, default=4.129e-06,
                   help="the block route's whole-layer maxrel at L0 (drift-profile-v1)")
    p.add_argument('--out', type=Path)
    raise SystemExit(run(p.parse_args()))

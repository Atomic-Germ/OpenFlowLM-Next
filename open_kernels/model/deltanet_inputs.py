"""FP64 DeltaNet record preparation and composed state from captured layer inputs.

Primary norm, projections and convolution history remain unrounded. Both conv
history and recurrent state carry independently from zero across all positions.
Captured-norm projection and captured-input glue checks localize errors; they
cannot replace the primary residual-to-state gate. One layer's weights are
loaded at a time, with no NPU execution or full-model residency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decode_bundle import load_bundle
from decode_reference import load_reference, sfx
from deltanet_state import linear_dtypes, conv_values, finite, read_boundaries, step_reference, compare_sequence, metric as recurrence_metric
from ffn_front import metric, MAXREL
from q4nx import Q4NX, bf16_to_f32, f32_to_bf16
from recipes.load import spec_from_model_dir
from recipes.manifest import manifest
from recipes.qwen35 import layout
from recipes.qwen36moe import norm_split
from recipes.spec import LINEAR

FIELDS = dict(k=slice(0,128), q=slice(128,256), v=slice(256,384), decay=384, beta=385)


def validate_weights(spec, weights):
    h, kh, vh, nch = spec.hidden, spec.lin_key_heads, spec.lin_value_heads, spec.lin_qkv_dim
    if (h <= 0 or kh <= 0 or vh <= 0 or vh % kh or spec.lin_key_dim != 128 or
            spec.lin_value_dim != 128 or nch != (2*kh+vh)*128 or spec.conv_kernel < 2 or
            not np.isfinite(spec.norm_eps) or spec.norm_eps <= 0):
        raise ValueError('unsupported DeltaNet preparation geometry/epsilon')
    shapes = dict(norm=(h,), qkv=(nch,h), conv=(spec.conv_kernel,nch),
                  alpha=(vh,h), beta=(vh,h), a=(vh,), dt=(vh,))
    if set(weights) != set(shapes):
        raise ValueError('incomplete DeltaNet weights')
    for name, shape in shapes.items():
        if finite(weights[name]).shape != shape:
            raise ValueError(f'invalid {name} weight shape')
    if np.any(weights['a'] > 0):
        raise ValueError('ssm_a must store nonpositive -exp(A_log)')


def sigmoid(x):
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1/(1+e), e/(1+e))


def records_reference(spec, xn, qkv, history, weights):
    """FP64 depthwise conv/SiLU/L2 and alpha/beta, with unrounded next history."""
    xn, qkv, history = map(finite, (xn, qkv, history))
    kh, vh, nch = spec.lin_key_heads, spec.lin_value_heads, spec.lin_qkv_dim
    if (xn.shape != (spec.hidden,) or qkv.shape != (nch,) or
            history.shape != (spec.conv_kernel-1,nch) or kh <= 0 or vh % kh):
        raise ValueError('invalid DeltaNet preparation inputs')
    seq = np.vstack([history, qkv])
    conv = finite(np.sum(seq*weights['conv'], axis=0))
    conv = conv*sigmoid(conv)
    kw = kh*128
    q, k = (conv[start:start+kw].reshape(kh,128) for start in (0,kw))
    q = q/np.sqrt(np.sum(q*q,axis=1,keepdims=True)+1e-6)
    k = k/np.sqrt(np.sum(k*k,axis=1,keepdims=True)+1e-6)
    alpha = finite(weights['alpha']@xn+weights['dt'])
    beta = sigmoid(finite(weights['beta']@xn))
    decay = np.exp(weights['a']*np.logaddexp(0,alpha))
    r = np.zeros((vh,512),np.float64)
    r[:,:128] = np.repeat(k,vh//kh,axis=0)
    r[:,128:256] = np.repeat(q,vh//kh,axis=0)
    r[:,256:384] = conv[2*kw:].reshape(vh,128)
    r[:,384] = decay; r[:,385] = beta
    return finite(r), seq[1:].copy()


def record_metrics(got, ref, strict=False):
    got, ref = finite(got), finite(ref)
    if got.shape != ref.shape or ref.ndim != 2 or ref.shape[1] != 512:
        raise ValueError('invalid record comparison shapes')
    result = {}
    for name, field in FIELDS.items():
        if strict:
            score = recurrence_metric(got[:,field],ref[:,field])
            score['passed'] = score['maxrel'] is not None and score['maxrel'] < 1e-4 and score['cosine'] > .99999
        else:
            score = metric(got[:,field],ref[:,field])
        result[name] = score
    return result


def score_sequence(spec, weights, inputs, captures, *, conv_dtype='bf16'):
    validate_weights(spec, weights)
    if len(inputs) < 2 or len(inputs) != len(captures):
        raise ValueError('incomplete multi-token position coverage')
    # This also validates full capture shapes, records, finite values, padding,
    # conv storage and the independent recurrence conditioned on captured records.
    recurrence = compare_sequence(captures, conv_dtype=conv_dtype)
    history = np.zeros((spec.conv_kernel-1,spec.lin_qkv_dim),np.float64)
    observed_history = history.copy()
    state = np.zeros((spec.lin_value_heads,128,128),np.float64)
    diagnostic_history = {name:history.copy() for name in ('entry','history','entry_and_history')}
    diagnostic_state = {name:state.copy() for name in diagnostic_history}
    rows = []
    for token, (x, c) in enumerate(zip(inputs,captures)):
        x, xn = finite(x), finite(c['xn'])
        if x.shape != (spec.hidden,) or xn.shape != x.shape:
            raise ValueError('invalid residual/entry norm shape')
        exact = x/np.sqrt(np.mean(x*x)+spec.norm_eps)*weights['norm']
        qkv = weights['qkv']@exact
        rec, history = records_reference(spec,exact,qkv,history,weights)
        state, output = step_reference(state,rec)
        local, _ = records_reference(spec,xn,c['qkv'],observed_history,weights)
        observed_history = conv_values(c['conv'], conv_dtype)
        fields = record_metrics(c['records'],rec)
        local_fields = record_metrics(c['records'],local,strict=True)
        primary = dict(norm=metric(xn,exact), qkv=metric(c['qkv'],qkv),
                       conv_history=metric(observed_history,history),
                       composed_state=metric(c['state'][:,:128],state),
                       composed_output=metric(c['output'],output))
        conditioned_qkv = metric(c['qkv'],weights['qkv']@xn)
        rounded = bf16_to_f32(f32_to_bf16(exact.astype(np.float32))).astype(np.float64)
        rounded_qkv = weights['qkv']@rounded
        diagnostics = {}
        for name in diagnostic_history:
            dx, dqkv = (exact,qkv) if name == 'history' else (rounded,rounded_qkv)
            dr, dh = records_reference(spec,dx,dqkv,diagnostic_history[name],weights)
            ds, do = step_reference(diagnostic_state[name],dr)
            diagnostic_state[name] = ds
            diagnostic_history[name] = (bf16_to_f32(f32_to_bf16(dh.astype(np.float32))).astype(np.float64)
                                        if name != 'entry' else dh)
            diagnostics[name] = dict(captured=dict(records=record_metrics(c['records'],dr),
                                                   state=metric(c['state'][:,:128],ds), output=metric(c['output'],do)),
                                     effect=dict(records=record_metrics(dr,rec),state=metric(ds,state),output=metric(do,output)))
        primary_ok = all(m['passed'] for m in (*primary.values(),*fields.values()))
        local_ok = all(m['passed'] for m in local_fields.values())
        rows.append(dict(token=token, passed=primary_ok and local_ok and conditioned_qkv['passed'] and recurrence[token]['passed'],
                         primary_passed=primary_ok, **primary, records=fields,
                         conditioned_glue_passed=local_ok, conditioned_glue=local_fields,
                         conditioned_qkv=conditioned_qkv, rounded_norm_qkv=metric(c['qkv'],rounded_qkv),
                         rounding_diagnostics=diagnostics,
                         recurrence=recurrence[token]))
    return rows


def read_entry(data, offsets, hidden, *, norm_dtype="bfloat16"):
    if norm_dtype not in ("bfloat16", "float32"):
        raise ValueError("unsupported linear entry dtype")
    size = 4 if norm_dtype == "float32" else 2
    off = offsets.A_XN
    if hidden <= 0 or len(data) != offsets.A_BYTES or off < 0 or off % size or off+hidden*size > len(data):
        raise ValueError('invalid linear entry buffer/region')
    a = np.frombuffer(data, '<f4' if size == 4 else '<u2', count=hidden, offset=off)
    return finite(a if size == 4 else bf16_to_f32(a))


def load_weights(model, spec, layer):
    prefix = f'model.layers.{layer}.'
    names = dict(norm='input_layernorm.weight', qkv='linear_attn.qkv_proj.weight',
                 conv='linear_attn.ssm_conv1d.weight', alpha='linear_attn.ssm_alpha_proj.bf16.weight',
                 beta='linear_attn.ssm_beta_proj.bf16.weight', a='linear_attn.ssm_a', dt='linear_attn.ssm_dt.bias')
    weights, hashes = {}, {}
    for role, suffix in names.items():
        name = prefix+suffix
        if role == 'qkv':
            # Match the effective quantization of the pool, as other model oracles do.
            value = model.matmul_w(name,spec.lin_qkv_dim,spec.hidden)
        else:
            value = model.f32(name) if role in ('a','dt') else model.bf16(name)
        value = np.ascontiguousarray(finite(value),dtype='<f8')
        weights[role] = value
        hashes[name] = hashlib.sha256(memoryview(value).cast('B')).hexdigest()
    validate_weights(spec,weights)
    return weights, hashes


def check(model_dir, kernel_dir, out):
    spec = spec_from_model_dir(model_dir)
    if spec.family != 'qwen35' or not norm_split(spec):
        raise ValueError('requires qwen35 split norm with retained linear entry')
    meta = load_reference(out)
    selected = load_bundle(kernel_dir,manifest(spec,max_ctx=meta['max_ctx']))
    if any(meta[k] != selected[k] for k in ('build_key','spec_hash')):
        raise ValueError('fixture/export identity mismatch')
    if meta['hidden'] != spec.hidden or not 0 < meta['layers'] <= spec.num_layers or meta['tokens'] < 2:
        raise ValueError('invalid fixture dimensions/multi-token coverage')
    layers = [l for l in range(meta['layers']) if spec.layer_types[l] == LINEAR]
    if not layers:
        raise ValueError('fixture has no linear layers')
    norm_dtype, conv_dtype = linear_dtypes(selected)
    offsets = layout(spec,max_ctx=meta['max_ctx'])
    model = Q4NX(model_dir/'model.q4nx')
    results, hashes, weight_hashes = [], {}, {}

    def read(name):
        data = (out/name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        return data

    initial = read(f'zstate_{LINEAR}.bin')
    if len(initial) != offsets.STATE_BYTES or any(initial):
        raise ValueError('fixture must start from complete zero state')
    for layer in layers:
        weights, digests = load_weights(model,spec,layer)
        captures, inputs = [], []
        for token in range(meta['tokens']):
            suffix = sfx(token)
            name = f'y_res{layer-1}{suffix}.bin' if layer else f'xres{token}.bin'
            data = read(name)
            if len(data) != spec.hidden*4:
                raise ValueError('wrong residual capture size')
            inputs.append(finite(np.frombuffer(data,'<f4')))
            act = read(f'y_act{layer}{suffix}.bin')
            c = read_boundaries(act,read(f'y_state{layer}{suffix}.bin'),spec,offsets,conv_dtype=conv_dtype)
            c['xn'] = read_entry(act,offsets,spec.hidden,norm_dtype=norm_dtype)
            captures.append(c)
        results.extend(dict(layer=layer,**r) for r in score_sequence(spec,weights,inputs,captures,conv_dtype=conv_dtype))
        weight_hashes.update(digests)
        del weights
    return dict(version=1, passed=all(r['passed'] for r in results), scope=__doc__.strip(),
                conditioning='captured incoming layer residuals; independent unrounded conv/state carry',
                layers=layers,tokens=meta['tokens'],heads=spec.lin_value_heads,
                maxrel_bound=MAXREL,conditioned_glue_maxrel_bound=1e-4,conditioned_glue_cosine_min=.99999,
                head_diagnostics_passed=all(r['recurrence']['head_diagnostics_passed'] for r in results),
                spec_hash=selected['spec_hash'],build_key=selected['build_key'],
                fixture_sha256=hashlib.sha256((out/'decode_reference.json').read_bytes()).hexdigest(),
                capture_sha256=hashes,weight_sha256=weight_hashes,
                weight_hash_encoding='little-endian float64 effective weights',results=results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','kernel-dir','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    args = ap.parse_args()
    try:
        result = check(args.model_dir,args.kernel_dir,args.out)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0 if result['passed'] else 1
    except (OSError,ValueError,KeyError,RuntimeError) as e:
        print(f'FAIL: {e}',file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

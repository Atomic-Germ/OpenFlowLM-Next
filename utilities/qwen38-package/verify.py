"""Verify a Qwen3.8-27B NPU2 package is complete, linked and self-consistent.

Phase 16 requires that every tensor `qwen35.pack_plan()` names is found exactly
once and that no unexpected tensor is silently ignored. Phase 23 requires a
reproducible package wired to the normal model-add/link mechanism with
`details.family = qwen3.5`.

The converter itself only *warns* about a missing name-map entry (so a family
with a synthesized lm_head still converts), which is the right behaviour for the
converter and the wrong guarantee for a released package. This tool turns that
warning into a gate: it walks the plan, resolves every tensor against the packed
container, and fails on a miss, a duplicate, or a stray tensor that no op reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'open_kernels'))
sys.path.insert(0, str(ROOT / 'open_kernels' / 'model'))

from recipes.load import spec_from_model_dir           # noqa: E402
from recipes.qwen35 import pack_plan, recipe           # noqa: E402
from q4nx import Q4NX                                  # noqa: E402

# Every role a pack op can read a tensor through. `apply_op` reads exactly one
# `op["tensor"]` per op, so the plan's op list IS the required-tensor list.
OP_KEYS = ('pool', 'consts', 'ops')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def eos_status(tokenizer_config, generation_config):
    """Does the runtime have an EOS array it can actually stop on?

    `automodel.cpp` refuses a model when `tokenizer_config["eos_token_id"]` is
    missing or is not an array, and the runtime stops generation only on an id it
    finds there. HF Qwen-family repos ship the field as null with the end-of-turn
    token in `eos_token`, and the stop set can also live in
    `generation_config.json`. The converter has helpers for both (resolve the
    token string, merge generation_config's ids) but a package that skipped them
    looks complete and then refuses to serve, which is what happened here.
    """
    cfg = tokenizer_config or {}
    gen = generation_config or {}
    raw = cfg.get('eos_token_id')
    ids = raw if isinstance(raw, list) else []
    gen_ids = gen.get('eos_token_id')
    gen_ids = gen_ids if isinstance(gen_ids, list) else ([gen_ids] if gen_ids is not None else [])
    present = isinstance(raw, list) and bool(raw)
    missing = [int(t) for t in gen_ids if int(t) not in ids]
    return dict(present=present, ids=[int(t) for t in ids],
                generation_ids=[int(t) for t in gen_ids],
                agrees=present and not missing,
                resolved=missing)


def ops_of(plan):
    """Flatten the plan into (layer_type, where, op) triples.

    A plan section is PER LAYER TYPE: `layer_types.linear_attention.pool` holds
    the DeltaNet tensors and `layer_types.full_attention.pool` the attention
    ones, and a given layer is only ever one of the two. Keeping the type with
    each op is what makes the per-layer expansion correct.
    """
    out = []
    for layer_type, section in plan['layer_types'].items():
        for key in OP_KEYS:
            for op in section.get(key, []):
                out.append((layer_type, f'layer_types.{layer_type}.{key}', op))
    for key in ('lm_head', 'embed', 'norm'):
        section = plan.get(key) or {}
        for kind in ('ops', 'tensors'):
            for op in section.get(kind, []):
                out.append((None, key + '.' + kind, op))
        if 'tensor' in section:
            out.append((None, key, section))
    return out


def required_tensors(plan):
    """Tensor name -> ops and layer types that read it.

    A tensor legitimately read by more than one op (the fused q_proj is read
    twice, once per half) is reported with its count rather than forbidden: the
    requirement is that each op resolves, not that no tensor is reused.
    """
    counts = {}
    for layer_type, where, op in ops_of(plan):
        name = op.get('tensor')
        if not isinstance(name, str):
            continue
        entry = counts.setdefault(name, {'ops': 0, 'where': [], 'layer_types': []})
        entry['ops'] += 1
        if where not in entry['where']:
            entry['where'].append(where)
        if layer_type and layer_type not in entry['layer_types']:
            entry['layer_types'].append(layer_type)
    return counts


def expand(name, layer):
    return name.replace('{l}', str(layer))


def check(model_dir, kernels, report=None):
    model_dir, kernels = Path(model_dir).resolve(), Path(kernels).resolve()
    spec = spec_from_model_dir(model_dir)
    if (spec.family != 'qwen35' or spec.hidden != 5120 or spec.intermediate != 17408
            or spec.num_layers != 64):
        raise ValueError('not the Qwen3.8-27B qwen35 geometry')
    plan = pack_plan(spec)
    # The recipe must accept this spec: an off-grid point is not a release.
    recipe(spec, max_ctx=spec.max_ctx if hasattr(spec, 'max_ctx') else 2048)
    model = Q4NX(model_dir / 'model.q4nx')
    result = dict(complete=False, model_acceptance=False,
                  model=str(model_dir), kernels=str(kernels),
                  model_sha256=digest(model_dir / 'model.q4nx'),
                  spec=dict(family=spec.family, hidden=spec.hidden, intermediate=spec.intermediate,
                            layers=spec.num_layers, vocab=spec.vocab, real_vocab=spec.real_vocab,
                            q_heads=spec.num_heads, kv_heads=spec.num_kv_heads, head_dim=spec.head_dim,
                            rotary_dim=spec.rotary_dim, lin_key_heads=spec.lin_key_heads,
                            lin_value_heads=spec.lin_value_heads, lin_key_dim=spec.lin_key_dim,
                            lin_value_dim=spec.lin_value_dim, lin_qkv_dim=spec.lin_qkv_dim,
                            lin_value_width=spec.lin_value_width,
                            layer_pattern=spec.layer_types[:8]))
    try:
        names = set(model.tensors)
        missing, resolved, checked = [], 0, 0
        for name, info in sorted(required_tensors(plan).items()):
            if '{l}' not in name:
                checked += 1
                if name in names:
                    resolved += 1
                else:
                    missing.append(dict(tensor=name, op=name, where=info['where']))
                continue
            # Per layer, but only for the layer types the op belongs to: a
            # linear_attention op is never required on an attention layer, and
            # expanding it over all 64 would report 48 false misses.
            for layer in range(spec.num_layers):
                layer_type = spec.layer_types[layer]
                if info['layer_types'] and layer_type not in info['layer_types']:
                    continue
                checked += 1
                concrete = expand(name, layer)
                if concrete in names:
                    resolved += 1
                else:
                    missing.append(dict(tensor=concrete, op=name, layer_type=layer_type,
                                        where=info['where']))
        result['required'] = checked
        result['resolved'] = resolved
        result['missing'] = missing

        # Nothing silently ignored: every packed tensor must be read by some op
        # that can actually run on that layer's type.
        expected = set()
        for name, info in required_tensors(plan).items():
            if '{l}' not in name:
                expected.add(name)
                continue
            for layer in range(spec.num_layers):
                if info['layer_types'] and spec.layer_types[layer] not in info['layer_types']:
                    continue
                expected.add(expand(name, layer))
        unused = sorted(names - expected)
        result['unused_tensors'] = unused
        result['packed_tensors'] = len(names)

        # The bundle the package links to must validate against the same spec.
        manifest = json.loads((kernels / 'manifest.json').read_text())
        toolchain = json.loads((kernels / 'toolchain.json').read_text())
        result['kernels_manifest_version'] = manifest['manifest_version']
        result['kernels_spec_hash'] = manifest['spec_hash']
        result['kernels_build_key'] = manifest['build_key']
        result['kernels_toolchain_key_agrees'] = toolchain.get('build_key') == manifest['build_key']
        result['kernels_real_vocab'] = manifest['layout'].get('real_vocab')
        result['kernels_builds'] = len(manifest['builds'])
        result['kernels_binaries'] = len(toolchain.get('sha256', {}))
        post = manifest['layer_types']['linear_attention'].get('post_output')
        result['kernels_post_output'] = post
        result['family_link'] = manifest['family']
        result['real_vocab_agrees'] = result['kernels_real_vocab'] == spec.real_vocab
        hashes_present = all(
            (kernels / name).is_file()
            for build in manifest['builds']
            for name in (f'{build}/final.xclbin', f'{build}/insts.bin'))
        result['kernels_artifacts_present'] = hashes_present

        result['model_files'] = {n: (model_dir / n).is_file() for n in
                                 ('config.json', 'model.q4nx', 'tokenizer.json',
                                  'tokenizer_config.json', 'chat_template.jinja')}
        # Phase 25 checks that belong to the package, not to kernel code: the
        # runtime must be able to stop, and the chat template must ship.
        tokenizer_config = json.loads((model_dir / 'tokenizer_config.json').read_text()) \
            if (model_dir / 'tokenizer_config.json').is_file() else {}
        generation_config = json.loads((model_dir / 'generation_config.json').read_text()) \
            if (model_dir / 'generation_config.json').is_file() else {}
        result['eos'] = eos_status(tokenizer_config, generation_config)
        template = model_dir / 'chat_template.jinja'
        result['chat_template'] = dict(
            present=template.is_file(),
            has_im_end='<|im_end|>' in template.read_text(errors='replace') if template.is_file() else False,
            in_tokenizer_config='chat_template' in tokenizer_config)
        passed = (not missing and not unused and hashes_present
                  and result['kernels_toolchain_key_agrees'] and result['real_vocab_agrees']
                  and manifest['manifest_version'] == 4
                  and manifest['family'] == 'qwen35'
                  and post == dict(dtype='float32', offset=393216, bytes=24576)
                  and all(result['model_files'].values())
                  and result['eos']['agrees']
                  and result['chat_template']['present'])
        result.update(complete=True, passed=bool(passed))
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        model.mm.close()
        model.f.close()
        if report:
            Path(report).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--kernels', type=Path, required=True)
    p.add_argument('--report', type=Path)
    a = p.parse_args()
    r = check(a.model_dir, a.kernels, a.report)
    print(json.dumps({k: r[k] for k in ('complete', 'passed', 'resolved', 'unused_tensors',
                                        'missing', 'kernels_builds', 'family_link')}, indent=2))
    return 0 if r['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

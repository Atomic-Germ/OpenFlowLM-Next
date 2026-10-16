"""Select the validated recurrence without changing buffers or weight packing."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from recipes.load import load_spec
from recipes.qwen35 import recipe, pack_plan
from recipes.manifest import manifest
from recipes import qwen36moe as geometry
from recipes.cache import source_files
from decode_bundle import load_bundle

ROOT = Path(__file__).resolve().parents[3]
LAYER = ROOT / 'open_kernels/designs/layer_x'


def spec():
    return load_spec(ROOT / 'open_kernels/recipes/specs/qwen35-27b.json')


def generator():
    s = importlib.util.spec_from_file_location('_factored_gen', LAYER / 'gen_kernels.py')
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_production_selects_factored_header_for_every_recurrence_entry():
    r = recipe(spec()); g = generator()
    assert r.linear.FACTORED_READOUT
    sources = g.files(r)
    for name, legacy in g.DNX.items():
        assert sources[name] == legacy.replace('"dnx.h"', '"dnx_factored.h"')


@pytest.mark.parametrize('change', [
    {'hidden': 4096}, {'lin_value_heads': 32}, {'conv_kernel': 3},
    {'family': 'qwen36moe'}, {'intermediate': 18944}, {'num_experts': 8}, {'quant': 'q8'},
    *({'quant': {role: 'q8'}} for role in ('linear', 'linear_out', 'attn', 'ffn')),
])
def test_unvalidated_geometry_or_quantization_keeps_legacy_recurrence(change):
    assert not geometry.factored_recurrence(replace(spec(), **change))


def test_legacy_wrappers_and_private_injection_are_preserved():
    g = generator()
    s = load_spec(ROOT / 'open_kernels/recipes/specs/qwen35-9b.json')
    r = recipe(s)
    assert not r.linear.FACTORED_READOUT
    for name, code in g.DNX.items():
        assert g.files(r)[name] == code
    import sys
    sys.path.insert(0, str(ROOT / 'utilities/deltanet-replay'))
    from full_layer_common import generated_sources
    for name, code in g.DNX.items():
        assert generated_sources(spec(), shared=True)[name] == code


@pytest.mark.parametrize('size', ['9b', '27b'])
def test_attention_only_recipe_generates_without_linear_geometry(size):
    s = load_spec(ROOT / f'open_kernels/recipes/specs/qwen35-{size}.json')
    s = replace(s, layer_types=('full_attention',) * s.num_layers)
    r = recipe(s); g = generator()
    assert r.linear is None
    assert not geometry.factored_recurrence(s)
    assert 'linear_recurrence' not in manifest(s)['layout']
    sources = g.files(r)
    for name, legacy in g.DNX.items():
        assert sources[name] == legacy


def test_production_arithmetic_matches_validated_shared_headers():
    for target, source in [('dnx_factored.h', 'factored.h'), ('dnx_precise.h', 'precise.h')]:
        expected = (ROOT / 'utilities/deltanet-replay' / source).read_text()
        expected = expected.replace('static __attribute__((noinline))', 'inline __attribute__((noinline))')
        expected = expected.replace('"precise.h"', '"dnx_precise.h"')
        # Comments may document production use; executable text must match.
        code = lambda text: '\n'.join(line.split('//')[0].rstrip() for line in text.splitlines() if line.split('//')[0].strip())
        assert code((LAYER / target).read_text()) == code(expected)
        assert LAYER / target in source_files(spec())


def test_manifest_refuses_pre_correction_bundle_and_preserves_packing(tmp_path):
    s = spec(); new = manifest(s)
    assert new['layout']['linear_recurrence'] == 'factored_compensated'
    with patch.object(geometry, 'factored_recurrence', return_value=False):
        old = manifest(s); old_pack = pack_plan(s); old_layout = recipe(s).layout
    assert old_pack == pack_plan(s) and old_layout == recipe(s).layout
    assert 'linear_recurrence' not in old['layout']
    (tmp_path / 'manifest.json').write_text(json.dumps(old))
    (tmp_path / 'toolchain.json').write_text('{}')
    with pytest.raises(ValueError, match='recipe'):
        load_bundle(tmp_path, new)

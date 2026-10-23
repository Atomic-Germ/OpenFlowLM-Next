"""Production activation sums must select only validated geometry and pin the export."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
from recipes import qwen36moe as geometry
from recipes.load import load_spec
from recipes.qwen35 import recipe, pack_plan
from recipes.manifest import manifest
from decode_bundle import load_bundle

ROOT = Path(__file__).resolve().parents[3]


def spec():
    return load_spec(ROOT/'open_kernels/recipes/specs/qwen35-27b.json')


def generated(s):
    loader = importlib.util.spec_from_file_location('_consistent_production_gen', ROOT/'open_kernels/designs/layer_x/gen_kernels.py')
    mod = importlib.util.module_from_spec(loader); loader.loader.exec_module(mod)
    return mod.files(recipe(s))


def test_validated_recipe_selects_consistent_sum_only_in_direct_fp32_prep():
    s = spec()
    assert geometry.consistent_q4_sum(s)
    assert recipe(s).ffn.CONSISTENT_SUM
    sources = generated(s)
    selected = [name for name, code in sources.items() if '#define GEMV_Q4_CONSISTENT_SUM 1' in code]
    assert selected == ['dense_prep_f32.cc']
    assert 'gemv_q4_prep_f32_direct_blocks(' in sources[selected[0]]
    assert '"dnx_factored.h"' in sources['dnx_row.cc']


@pytest.mark.parametrize('change', [
    {'hidden':4096}, {'intermediate':18944}, {'lin_value_heads':32},
    {'num_experts':8}, {'conv_kernel':3}, {'head_dim':128},
    {'layer_types':('full_attention',)*64}, {'family':'qwen36moe'}, {'quant':'q8'},
    *({'quant':{role:'q8'}} for role in ('attn','linear','linear_out','ffn')),
])
def test_unvalidated_geometry_or_quantization_keeps_original_sum(change):
    assert not geometry.consistent_q4_sum(replace(spec(), **change))


def test_small_recipe_keeps_unmodified_generated_sources():
    s=load_spec(ROOT/'open_kernels/recipes/specs/qwen35-9b.json')
    assert not recipe(s).ffn.CONSISTENT_SUM
    assert all('GEMV_Q4_CONSISTENT_SUM' not in code for code in generated(s).values())


def test_manifest_identifies_arithmetic_without_changing_abi_or_packing(tmp_path):
    s=spec();new=manifest(s)
    assert new['layout']['q4_activation_sum']=='quantized_fp32'
    with patch.object(geometry,'consistent_q4_sum',return_value=False):
        old=manifest(s);old_sources=generated(s);old_pack=pack_plan(s);old_layout=recipe(s).layout
    assert old_pack==pack_plan(s) and old_layout==recipe(s).layout
    assert 'q4_activation_sum' not in old['layout']
    for name,code in old_sources.items():
        assert generated(s)[name].replace('#define GEMV_Q4_CONSISTENT_SUM 1\n','')==code
    (tmp_path/'manifest.json').write_text(json.dumps(old));(tmp_path/'toolchain.json').write_text('{}')
    with pytest.raises(ValueError,match='recipe'):load_bundle(tmp_path,new)


def test_enabled_arithmetic_matches_the_validated_private_replacement():
    sys.path.insert(0,str(ROOT/'utilities/qkv-precision'))
    from qkv_build import consistent_header
    header=(ROOT/'open_kernels/designs/gemv_q4/gemv_tab.h').read_text()
    prefix,rest=header.split('#if defined(GEMV_Q4_CONSISTENT_SUM) && GEMV_Q4_CONSISTENT_SUM\n')
    active,rest=rest.split('#else\n',1);legacy,suffix=rest.split('#endif\n',1)
    assert prefix+active+suffix==consistent_header(prefix+legacy+suffix)

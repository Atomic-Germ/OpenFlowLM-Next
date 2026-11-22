"""The standard exporter must carry all five paired-prefill companions."""
from dataclasses import replace
from pathlib import Path
import pytest
from recipes.load import load_spec
from recipes.manifest import manifest
from recipes import qwen35

ROOT=Path(__file__).resolve().parents[3]


def spec(name='qwen35-27b'):
    return load_spec(ROOT/'open_kernels/recipes/specs'/(name+'.json'))


def test_wide_route_exports_all_companions_with_version_negotiation():
    m=manifest(spec())
    assert m['manifest_version']==4
    lows=set()
    for lt in m['layer_types'].values():
        g=lt['gemm_block']
        assert set(g['low_kernels'])=={s['kernel'] for s in g['program']+g['ffn_program']}
        for high,low in g['low_kernels'].items():
            assert low==high+'_low'
            d=m['kernels'][low];assert d['weight_component']=='bf16_residual'
            assert d['context']!=m['kernels'][high]['context']
            b=m['builds'][low]
            assert b['design']=='gemm_q4_prefill/gemm_q4_prefill.py'
            assert b['env']==dict(m['builds'][high]['env'],GQP_WEIGHT_LOW='1',GQP_BREUSE='0')
            assert b['build_dir']!=m['builds'][high]['build_dir']
            lows.add(low)
    assert len(lows)==5 and len(m['builds'])==46


def test_other_sizes_and_moe_do_not_enable_paired_projection():
    for name in ('qwen35-9b','qwen36-35b-a3b'):
        m=manifest(spec(name))
        assert m['manifest_version']<4
        assert all('low_kernels' not in lt.get('gemm_block',{}) for lt in m['layer_types'].values())
        assert all('GQP_WEIGHT_LOW' not in b['env'] for b in m['builds'].values())


@pytest.mark.parametrize('name', ['qwen35-27b', 'qwen35-9b', 'qwen36-35b-a3b'])
def test_dma_experiment_is_not_selected_by_production_recipes(name, monkeypatch):
    monkeypatch.setenv('GQP_DMA_BATCH', '4')
    m = manifest(spec(name))
    assert all('GQP_DMA_BATCH' not in b['env'] for b in m['builds'].values())


def test_export_build_environment_drops_ambient_experiments_but_keeps_recipe_choice(monkeypatch):
    import export_qwen36_kernels as exporter
    monkeypatch.setenv('GQP_DMA_BATCH', '4')
    monkeypatch.setenv('GQP_WEIGHT_LOW', '1')
    monkeypatch.setenv('OPEN_KERNELS_SPEC', 'ambient-spec')
    path = ROOT / 'open_kernels/recipes/specs/qwen35-27b.json'
    default = exporter.build_environment({'GQP_N': '14336'}, path)
    assert 'GQP_DMA_BATCH' not in default
    assert 'GQP_WEIGHT_LOW' not in default
    assert default['OPEN_KERNELS_SPEC'] == str(path)
    explicit = exporter.build_environment({'GQP_DMA_BATCH': '2', 'GQP_WEIGHT_LOW': '1'}, path)
    assert explicit['GQP_DMA_BATCH'] == '2'
    assert explicit['GQP_WEIGHT_LOW'] == '1'


@pytest.mark.parametrize('field,value',[('hidden',4096),('intermediate',12288),('lin_value_heads',32),
                                      ('num_heads',16),('norm_eps',1e-5),('quant','q8')])
def test_precision_gate_is_geometry_and_quantization_scoped(field,value):
    assert qwen35.paired_prefill(spec())
    assert not qwen35.paired_prefill(replace(spec(),**{field:value}))

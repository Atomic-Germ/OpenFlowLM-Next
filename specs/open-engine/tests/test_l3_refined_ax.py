"""Integrated refinement must retain native DMA/ABI and explicit FP32 composition."""
import ast
import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[3]


def load():
    s=importlib.util.spec_from_file_location('refined_ax_build',ROOT/'utilities/l3-refined-ax/build.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_candidate_keeps_six_buffers_and_native_attention_ffn():
    m=load();s=m.render('candidate');ast.parse(s)
    assert 'forced: In' not in s and 'a_forced' not in s
    assert 'XL.attn_workers(of_ain, of_aout, of_og, a, afns)' in s
    assert 'X.ffn_body(win, xin, yout, B, K)' in s
    assert 'px.fill(x_prod, a_act, bt(AA_BYTES, AA_OG, OG_ELEMS * ELEM))' in s
    assert 'pa_in.fill(ain_p, a_ptab, bt(PTAB_BYTES, PTAB_ROW, PTAB_ROW))' in s
    assert 'refined_bands(win, xin, yout, B, K, low_y, prep_pair, gy_low, sum_pair)' in s


def test_control_has_no_refinement_helpers_or_forced_input():
    m=load();s=m.render('control');ast.parse(s)
    assert 'refined_bands' not in s and 'low_y' not in s and 'a_forced' not in s
    assert 'X.prep_bands(win, xin, yout, B, K, HID, XN_ELEMS' in s


def test_native_dma_sequence_is_byte_identical_for_both_builds():
    m=load();native=(ROOT/'open_kernels/designs/layer_x/ax.py').read_text()
    def sequence(s):return s[s.index('    def dense_sequence('):s.index('\n\nDESIGN =')]
    assert sequence(m.render('candidate'))==sequence(native)
    assert sequence(m.render('control'))==sequence(native)


def test_refined_bands_reuse_each_weight_before_release():
    m=load();s=m.render('candidate')
    high=s.index('K["gy"](we, tab, ye, g, X.per_band(HID), 2)')
    low=s.index('gy_low(we, tab, low_y, g, X.per_band(HID), 2)')
    release=s.index('win.release(1)',high)
    assert high<low<release<s.index('sum_pair(ye, low_y)')


def test_only_candidate_generates_helpers_and_has_two_table_capacity():
    m=load();spec=m.load_spec(ROOT/'open_kernels/recipes/specs/qwen35-27b.json')
    control=m.generated(spec,'control');candidate=m.generated(spec,'candidate')
    assert set(candidate)-set(control)=={'refine_prep.cc','refine_low.cc','refine_sum.cc','add_rne.h'}
    assert all(candidate[k]==v for k,v in control.items())
    assert 'max(X.TAB_BYTES, 23040)' in m.render('candidate')
    assert 'gemv_q4_gy(t, tab + 11520' in candidate['refine_low.cc']


def test_prep_uses_consistent_tables_and_fp32_subtraction():
    m=load();s=(m.HERE/'refine_prep.cc').read_text()
    assert '#define GEMV_Q4_CONSISTENT_SUM 1' in s
    assert 'aie::to_float<float>' in s and 'add_rne' in s
    assert 'aie::sub' not in s
    assert 'gemv_q4_prep_f32_direct_blocks(low, tab + 11520' in s


def test_changed_anchor_and_unknown_role_are_rejected():
    m=load()
    with pytest.raises(ValueError,match='anchor'):m.render('candidate','changed')
    with pytest.raises(ValueError,match='role'):m.render('unknown')

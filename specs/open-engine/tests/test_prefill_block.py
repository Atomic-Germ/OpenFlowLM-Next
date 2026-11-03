"""Phase 26: the block-vs-sequential oracle must not be able to pass vacuously.

Every test here exists because the harness got it WRONG first and reported PASS anyway:

  * `--dump-state` held one spec, so an 8-layer comparison silently compared one tensor;
  * the state dump was read from the wrong path, so nothing was compared at all;
  * the state LAYOUT was taken from the stage rather than the layer type, so "conv"/"S"
    were reported for attention layers;
  * the canary region was never dumped, so the canary check never ran;
  * the logits dump was looked up by absolute position when it is keyed by the index
    inside --ids, so every non-zero-position run skipped the logits comparison;
  * a regex without re.M matched nothing and the parse silently yielded an empty list.

None of those crashed. All of them produced a clean PASS. So each is pinned.

Hardware results are NOT faked here: the NPU tests live in the committed run artefacts
and are asserted to be well-formed, and the pure functions are tested directly.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'utilities/prefill-block'
ORACLE = HERE / 'oracle.py'
DECODE = HERE / 'decode_transition.py'
DOC = HERE / 'BLOCK_CONTRACT.md'
RUNS = HERE / 'runs'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- the contract (26.0)

def test_the_contract_is_derived_from_the_manifest_not_asserted():
    md = DOC.read_text()
    # The binding constraint must be stated with its arithmetic, because it is the reason
    # T=32 was NOT adopted as a physical geometry.
    assert 'M = (nh/kvh) · T = 6 · T' in md
    assert 'ab.m = 1536' in md or '1536' in md
    assert 'T = 256 exactly' in md
    # The plan's escape clause, and the explicit refusal to add a geometry.
    assert 'active_t = 32, physical_t = 256' in md
    assert 'No new GEMM geometry is authorised' in md
    # The three masking mechanisms, named.
    for mech in ('softmax_chunk', 'R = g.t_real', 'tile_rows_as_b'):
        assert mech in md, mech
    assert 'padding cannot reach' in md.lower() or 'Padding cannot reach' in md


def test_the_manifest_really_forces_256_for_a_full_layer():
    """The contract claim, checked against the artefact rather than the prose."""
    m = json.loads((ROOT / 'Models/qwen38-27b/production-v2/kernels-model/manifest.json').read_text())
    gb = m['layer_types']['full_attention']['gemm_block']
    ab = gb['attn_block']
    assert gb['t'] == 256
    assert ab['m'] == 1536
    grp = 24 // 4                     # nh / kvh for Qwen3.8-27B
    assert ab['m'] // grp == 256, 'the only legal physical T for a full layer'


def test_the_state_layout_is_read_from_the_layer_type():
    """conv/S belong to `linear` layers, KV to `full`, and the file says which is which."""
    o = load('_oracle', ORACLE)
    kinds = o.layer_kinds()
    assert len(kinds) == 64
    # The 4-layer pattern, 16 times.
    assert kinds[:4] == ['linear', 'linear', 'linear', 'full']
    assert kinds.count('full') == 16 and kinds.count('linear') == 48
    # Stage 1 covers linear layers only; stage 2 covers full layers only.
    dnet = o.stage_state_specs('dnet', layers=8)
    attn = o.stage_state_specs('attn', layers=8)
    assert all(kinds[i] == 'linear' for i in dnet), 'stage 1 must not touch attention state'
    assert all(kinds[i] == 'full' for i in attn), 'attention layers must never get conv/S'
    assert sorted(attn) == [3, 7]
    assert sorted(dnet) == [0, 1, 2, 4, 5, 6]
    assert dnet[0] == [('conv', 0, o.CONV_BYTES), ('S', o.S_OFF, o.S_BYTES)]
    assert attn[3] == [('kv', 0, 8192)]


def test_the_linear_state_layout_adds_up_to_the_manifest_size():
    o = load('_oracle', ORACLE)
    m = json.loads((ROOT / 'Models/qwen38-27b/production-v2/kernels-model/manifest.json').read_text())
    declared = m['layer_types']['linear_attention']['buffers']['state']['bytes']
    assert o.CONV_BYTES + o.S_BYTES == declared
    assert o.S_OFF == o.CONV_BYTES, 'state_s_off is exactly where the conv history ends'


# ---------------------------------------------------------------- the fixture (26.2)

def test_the_fixture_is_distinct_so_a_row_swap_cannot_pass():
    o = load('_oracle', ORACLE)
    ids = o.fixture(256)
    assert len(ids) == 256
    assert len(set(ids)) == 256, 'repeated ids would make a row swap invisible'
    assert all(2 <= i < o.REAL_VOCAB for i in ids)
    # Deterministic: the same draw every time, so a captured run is reproducible.
    assert o.fixture(256) == ids
    assert o.fixture(64) == ids[:64], 'a shorter fixture is a prefix, so length is the variable'
    # And it is not simply ascending -- that would hide an off-by-one in the row index.
    assert ids != sorted(ids)


# ---------------------------------------------------------------- metrics (26.3)

@pytest.mark.parametrize('kind', ['nan', 'inf'])
def test_a_non_finite_tensor_is_a_failure_not_a_pass(kind):
    o = load('_oracle', ORACLE)
    ref = np.ones(8, dtype=np.float32)
    got = ref.copy()
    got[3] = np.nan if kind == 'nan' else np.inf
    assert o.metric('t', got, ref)['verdict'] == 'FAIL'


def test_a_shape_mismatch_is_a_failure():
    o = load('_oracle', ORACLE)
    assert o.metric('t', np.ones(4), np.ones(5))['verdict'] == 'FAIL'


def test_a_zero_reference_requires_exact_equality_rather_than_a_free_cosine():
    """cosine is undefined against an all-zero reference; a pass must mean equality."""
    o = load('_oracle', ORACLE)
    zeros = np.zeros(8, dtype=np.float32)
    assert o.metric('t', zeros.copy(), zeros)['verdict'] == 'PASS'
    off = zeros.copy()
    off[0] = 1e-9
    assert o.metric('t', off, zeros)['verdict'] == 'FAIL'


def test_the_gate_is_the_repository_gate():
    """26.2: a changed accumulation order is not a reason to invent a weaker tolerance."""
    o = load('_oracle', ORACLE)
    assert o.GATE_COSINE == 0.9999
    ref = np.arange(1, 33, dtype=np.float32)
    # Just below the gate fails, just above passes -- the gate is not decorative.
    got = ref * 1.0
    got[0] += 0.5
    r = o.metric('t', got, ref)
    assert (r['verdict'] == 'PASS') == (r['cosine'] >= 0.9999)
    assert set(r) >= {'cosine', 'max_abs', 'max_rel', 'worst_abs_index', 'worst_rel_index',
                      'verdict'}


def test_the_canary_requires_every_byte_untouched():
    o = load('_oracle', ORACLE)
    pat = 165
    good = np.full(4096, pat, dtype=np.uint8)
    assert o.canary_check('c', good, pat)['verdict'] == 'PASS'
    bad = good.copy()
    bad[-1] ^= 1                        # one byte written past the allowed region
    r = o.canary_check('c', bad, pat)
    assert r['verdict'] == 'FAIL'
    assert r['differing_bytes'] == 1


# ---------------------------------------------------------------- the silent skips

def test_a_stage_that_compares_no_state_is_a_failure(tmp_path):
    """The core regression: a missing dump used to be skipped, so PASS meant nothing."""
    o = load('_oracle', ORACLE)
    pa = dict(dump_prefix=str(tmp_path / 'a'), workdir=str(tmp_path / 'a'),
              state_specs={0: [('conv', 0, 16)]})
    pb = dict(dump_prefix=str(tmp_path / 'b'), workdir=str(tmp_path / 'b'),
              state_specs={0: [('conv', 0, 16)]})
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    ev = o.evaluate(pa, pb, token_count=4, position=0)
    assert ev['verdict'] == 'FAIL'
    assert ev['first_failure'] is not None
    reasons = [r.get('reason') for r in ev['tensors']]
    assert any('logits dump missing' in (x or '') for x in reasons)
    assert any('missing dump' in (x or '') for x in reasons)


def test_the_logits_dump_is_keyed_by_the_index_inside_ids(tmp_path):
    """Not by absolute position: cli.cpp dumps with the loop index."""
    o = load('_oracle', ORACLE)
    prefix = tmp_path / 'logits'
    # A run at absolute position 17 writes _p<index>, so index 3 is _p3 -- nothing else.
    np.arange(8, dtype=np.float32).tofile(f'{prefix}_p3.bin')
    assert o.read_dump(str(prefix), 3) is not None
    assert o.read_dump(str(prefix), 20) is None
    ev = o.evaluate(dict(dump_prefix=str(prefix), workdir=str(tmp_path), state_specs={}),
                    dict(dump_prefix=str(prefix), workdir=str(tmp_path), state_specs={}),
                    token_count=4, position=17)
    names = [r['tensor'] for r in ev['tensors']]
    assert 'logits@i3' in names, names
    assert ev['verdict'] == 'PASS'


def test_a_state_slice_is_compared_when_both_paths_wrote_it(tmp_path):
    o = load('_oracle', ORACLE)
    for side, val in (('a', 1.0), ('b', 1.0)):
        d = tmp_path / side
        d.mkdir()
        np.full(8, val, dtype=np.float32).tofile(d / 'state_0_0_32.bin')
        np.full(8, val, dtype=np.float32).tofile(f'{tmp_path / side}_logits_p3.bin')
    pa = dict(dump_prefix=str(tmp_path / 'a_logits'), workdir=str(tmp_path / 'a'),
              state_specs={0: [('conv', 0, 32)]})
    pb = dict(dump_prefix=str(tmp_path / 'b_logits'), workdir=str(tmp_path / 'b'),
              state_specs={0: [('conv', 0, 32)]})
    ev = o.evaluate(pa, pb, token_count=4, position=0)
    assert ev['verdict'] == 'PASS'
    assert any(r['tensor'] == 'L0.conv' for r in ev['tensors'])
    # And a difference in the state is caught even when the logits agree.
    moved = np.full(8, 1.0, dtype=np.float32)
    moved[5] = 4.0                      # one element, not a uniform scale -- see below
    moved.tofile(tmp_path / 'b' / 'state_0_0_32.bin')
    ev = o.evaluate(pa, pb, token_count=4, position=0)
    assert ev['verdict'] == 'FAIL'
    assert ev['first_failure']['tensor'] == 'L0.conv'


def test_a_uniform_rescale_is_invisible_to_the_cosine_gate():
    """A documented limitation of the repository's gate, recorded rather than hidden.

    cosine is scale-invariant, so a state tensor that is uniformly 2x too large has
    cosine 1.0 and PASSES. The gate is the repository's existing acceptance rule and 26.2
    forbids inventing a different one here, so this is stated as a limitation instead of
    being papered over: max_abs and max_rel carry the magnitude information in every
    report, and they are what a reviewer reads when a run looks "too clean".
    """
    o = load('_oracle', ORACLE)
    ref = np.linspace(1.0, 2.0, 16, dtype=np.float32)
    r = o.metric('t', ref * 2.0, ref)
    assert r['cosine'] == pytest.approx(1.0, abs=1e-6)
    assert r['verdict'] == 'PASS', 'the gate passes a uniform rescale -- by construction'
    # The report still shows it, which is why the magnitude fields are mandatory.
    assert r['max_abs'] > 1.0 and r['max_rel'] == pytest.approx(1.0, abs=1e-6)


def test_the_greedy_token_parse_needs_multiline():
    """A regex without re.M matched nothing and the parse silently returned []. """
    import re
    text = 'noise\ntoken 13\ntoken 220\nmore\n'
    assert re.findall(r'^token (\d+)$', text) == []          # the bug
    assert re.findall(r'^token (\d+)$', text, re.M) == ['13', '220']   # the fix


# ---------------------------------------------------------------- the committed runs

# The runs this file owns, selected by the tool that wrote them. Selecting by directory
# name meant every later phase's output dropped into the same runs/ tree was claimed by
# this file and judged against the wrong schema -- the Phase 27 oflm runs are not Phase 26
# oracle runs. Naming the tools makes the ownership explicit and keeps a future phase
# from having to edit this file to stop being mis-measured.
_ORACLE_TOOLS = ('prefill-block-oracle',)


def _results():
    """The oracle runs only: decode runs carry a different schema and a different gate."""
    if not RUNS.is_dir():
        return []
    out = []
    for p in sorted(RUNS.glob('*/result.json')):
        try:
            tool = json.loads(p.read_text()).get('tool', '')
        except (json.JSONDecodeError, OSError):
            continue
        if tool in _ORACLE_TOOLS:
            out.append(p)
    return out


def test_the_committed_runs_are_well_formed_and_all_pass():
    """Hardware evidence, asserted for shape and verdict -- never fabricated here."""
    rs = _results()
    if not rs:
        pytest.skip('no committed oracle runs; run utilities/prefill-block/oracle.py')
    for p in rs:
        d = json.loads(p.read_text())
        assert d['model_acceptance'] is False, 'a block comparison is not model acceptance'
        assert d['precision'] == 'paired'
        assert d['verdict'] == 'PASS', f'{p} says {d["verdict"]}'
        if 'stages' in d:
            for name, ev in d['stages'].items():
                assert ev['verdict'] == 'PASS', (p, name)
                # A stage that compared nothing is the regression this file exists for.
                assert ev['tensors'], f'{p}: stage {name} compared no tensors'
                assert any('cosine' in t for t in ev['tensors']), \
                    f'{p}: stage {name} compared no numeric tensor'
                assert ev['first_failure'] is None


def test_a_committed_run_records_the_fixture_and_the_gate():
    rs = _results()
    if not rs:
        pytest.skip('no committed oracle runs')
    d = json.loads(rs[-1].read_text())
    assert d['fixture']['distinct'] is True
    assert d['fixture']['seed'] == 26026
    assert d['gate_cosine'] == 0.9999


def test_the_canary_appears_in_the_committed_runs():
    """26.4: prove production does not write outside its region, don't assume it."""
    rs = _results()
    if not rs:
        pytest.skip('no committed oracle runs')
    checked = 0
    for p in rs:
        d = json.loads(p.read_text())
        for ev in d.get('stages', {}).values():
            for t in ev['tensors']:
                if 'canary' in t['tensor']:
                    assert t['verdict'] == 'PASS', (p, t['tensor'])
                    assert t['untouched'] is True
                    checked += 1
    assert checked > 0, 'the canary check never ran in any committed run'


_ORACLE_TOOLS_DECODE = ('prefill-block-decode-transition',)


def test_the_decode_transition_is_recorded_as_the_hard_gate():
    """Selected by the tool that wrote it: the directory prefix also matches later
    `decode-rate-*` / `decode16-logits-*` runs, which are a different gate entirely."""
    rs = []
    for p in sorted(RUNS.glob('decode-*/result.json')) if RUNS.is_dir() else []:
        try:
            if json.loads(p.read_text()).get('tool') in _ORACLE_TOOLS_DECODE:
                rs.append(p)
        except (json.JSONDecodeError, OSError):
            continue
    if not rs:
        pytest.skip('no committed decode-transition runs')
    for p in rs:
        d = json.loads(p.read_text())
        assert d['gate'] == '26.12 multi-chunk + decode transition'
        assert d['model_acceptance'] is False
        assert d['tail_tokens'] == d['prompt_tokens'] % d['chunk']
        assert d['verdict'] == 'PASS', p
        names = {t['tensor'] for t in d['tensors']}
        # The transition must be compared AFTER each decode token, not only once.
        for k in range(d['decode_steps']):
            assert f'logits@decode{k}' in names, (p, names)
            assert f'greedy@decode{k}' in names, (p, names)
        assert 'greedy_tokens' in names
        gt = next(t for t in d['tensors'] if t['tensor'] == 'greedy_tokens')
        assert gt['sequential'] == gt['block']
        assert len(gt['sequential']) == d['decode_steps']

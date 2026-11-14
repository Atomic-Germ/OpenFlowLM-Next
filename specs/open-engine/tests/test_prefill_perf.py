"""Phase 21 prefill measurement must parse the engine's own split and stay honest.

The gate asks for a wall-time budget that explains >=90% of prefill. A parser
that quietly turns a tautology into 100% coverage would pass that gate without
measuring anything, so the arithmetic identity is tested directly: the
unexplained part is the prologue outside the route PLUS the route's own residue,
never `preproc + total`.
"""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    s = importlib.util.spec_from_file_location('prefill_perf', ROOT/'utilities/prefill-perf/measure.py')
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


LOG = """resident after 7.6 s
  gemm-prompt [0,256) t_real=256: 30095.8 ms (GEMM 1.0, host 29999.0, per-token 0.0, lm_head 0.0)
      mid 29000.0 (dn conv 100.0, dn rule 50.0, attn 0.0), prenorm 10.0, gemm tile 20.0, gemm tr 30.0, tail 5.0, shared 0.0, state 0.0, sync 3.0, setup 1.0 | moe prep 0.0, patch 0.0, run 0.0, read 0.0 | qkv scatter 0.0 | residue 900.8
      lx                  512 calls   41000.0 ms total   80.078 mean   60.100 min
      ax                  256 calls   20000.0 ms total   78.125 mean   70.000 min
prefill 256 tokens: 31000 ms (121 ms/token)
"""


def test_parse_extracts_every_stage_the_engine_prints():
    m = load()
    p = m.parse(LOG)
    assert p['tokens'] == 256 and p['prefill_ms'] == 31000.0
    assert p['stages']['total'] == 30095.8
    assert p['stages']['part0'] == 1.0 and p['stages']['part1'] == 29999.0
    assert p['fine']['mid'] == 29000.0 and p['fine']['dn_conv'] == 100.0
    assert p['fine']['residue'] == 900.8
    assert p['dispatch']['lx']['calls'] == 512
    assert p['dispatch']['ax']['calls'] == 256
    assert p['resident_s'] == 7.6


def test_unexplained_is_prologue_plus_residue_not_the_whole_route():
    """The engine's own identity is part0+part1+route+lmhead+residue = total, so a
    budget built as `preproc + total` always reports 100% and measures nothing."""
    m = load()
    p = m.parse(LOG)
    assert p['residue_ms'] == pytest.approx(95.8, abs=1e-6)
    assert p['preproc_ms'] == pytest.approx(904.2, abs=1e-6)
    assert p['unexplained_ms'] == pytest.approx(1000.0, abs=1e-6)
    assert p['coverage'] == pytest.approx(30000.0/31000.0, abs=1e-9)
    assert p['coverage'] < 1.0


def test_budget_fails_a_row_that_hides_more_than_ten_percent():
    m = load()
    good = m.parse(LOG)
    row = dict(schedule='hybrid', length=256, median=good)
    assert m.budget([row])['complete'] is True
    hidden = dict(good, residue_ms=4000.0, unexplained_ms=4000.0 + good['preproc_ms'],
                  coverage=(31000.0 - 4000.0 - good['preproc_ms'])/31000.0)
    assert m.budget([dict(schedule='hybrid', length=256, median=hidden)])['complete'] is False


def test_parse_refuses_a_run_without_a_prefill_line():
    m = load()
    assert m.parse('resident after 1.0 s\n') is None
    assert m.parse('') is None


def test_sequential_route_has_ttft_but_no_invented_budget():
    """The sequential route loops step() per token and prints no stage line. Its
    TTFT is real; a coverage computed anyway would be fabricated."""
    m = load()
    p = m.parse('resident after 7.6 s\nprefill 255 tokens: 116043 ms (455 ms/token)\n')
    assert p is not None
    assert p['stages_available'] is False
    assert p['coverage'] is None and p['unexplained_ms'] is None
    assert p['tok_per_s'] == pytest.approx(255000.0/116043.0, abs=1e-6)
    assert m.budget([dict(schedule='sequential', length=255, median=p)])['complete'] is False


def test_median_keeps_spread_and_is_not_a_best_case():
    m = load()
    rows = [dict(prefill_ms=x, tok_per_s=256000.0/x) for x in (100.0, 300.0, 200.0)]
    med, spread = m.median_row(rows)
    assert med['prefill_ms'] == 200.0
    assert spread['prefill_ms']['min'] == 100.0 and spread['prefill_ms']['max'] == 300.0
    assert spread['prefill_ms']['n'] == 3


def test_prompt_ids_stay_inside_the_vocabulary():
    m = load()
    ids = m.prompt(2048)
    assert len(ids) == 2048
    assert all(isinstance(t, int) and 0 <= t < 248077 for t in ids)


def test_schedules_map_to_the_documented_flags():
    m = load()
    assert m.EXTRA['sequential'] == []
    assert '--gemm-block' in m.EXTRA['hybrid']
    assert '--block-major' in m.EXTRA['block_major']


def _row(schedule, length, wall, part0, part1, preproc, residue, fine=None):
    return dict(schedule=schedule, length=length,
                median=dict(prefill_ms=wall, tok_per_s=length/(wall/1000.0),
                            stages=dict(total=wall-preproc, part0=part0, part1=part1,
                                        route=0.0, lmhead=0.0),
                            fine=fine or {}, preproc_ms=preproc, residue_ms=residue,
                            stages_available=True))


def test_median_keeps_the_stage_availability_flag():
    """`median_row` flattens only numbers, so a boolean is silently dropped and
    every real row ends up reporting 'no budget'. Availability must survive."""
    m = load()
    rows = [m.parse(LOG), m.parse(LOG), m.parse(LOG)]
    med, _ = m.median_row([{k: v for k, v in s.items()
                            if k not in ('tag', 'schedule', 'length')} for s in rows])
    assert 'stages_available' in med
    assert med['stages_available'] == 1.0
    budget = m.budget([dict(schedule='hybrid', length=256, median=m.unflatten(med))])
    assert budget['rows'][0]['coverage'] is not None
    assert budget['rows'][0]['coverage'] > 0.9


def test_ab_pairs_the_same_length_and_quantifies_the_delta():
    """Phase 22 has to name where the time went, not just that a variant was
    faster: the delta is split into the GEMM-blocked and other-host halves."""
    m = load()
    off = m.budget([_row('hybrid', 256, 40000.0, 26000.0, 4000.0, 800.0, 200.0)])
    on = m.budget([_row('hybrid', 256, 21000.0, 9000.0, 3800.0, 800.0, 300.0)])
    ab = m.compare(off, on, labels=('correction-on', 'correction-off'))
    assert len(ab['pairs']) == 1
    pair = ab['pairs'][0]
    assert pair['length'] == 256
    assert pair['wall_saved_ms'] == pytest.approx(19000.0)
    assert pair['speedup'] == pytest.approx(40000.0/21000.0, abs=1e-9)
    # The saving is attributed to the GEMM-blocked half (17000) plus the rest.
    assert pair['gemm_saved_ms'] == pytest.approx(17000.0)
    assert pair['other_saved_ms'] == pytest.approx(200.0)
    assert pair['accounted_fraction'] > 0.9


def test_ab_refuses_unpaired_or_ambiguous_reports():
    m = load()
    a = m.budget([_row('hybrid', 256, 100.0, 50.0, 10.0, 1.0, 1.0)])
    b = m.budget([_row('hybrid', 512, 100.0, 50.0, 10.0, 1.0, 1.0)])
    with pytest.raises(ValueError):
        m.compare(a, b)
    dup = m.budget([_row('hybrid', 256, 100.0, 50.0, 10.0, 1.0, 1.0),
                    _row('hybrid', 256, 90.0, 40.0, 10.0, 1.0, 1.0)])
    with pytest.raises(ValueError):
        m.compare(a, dup)

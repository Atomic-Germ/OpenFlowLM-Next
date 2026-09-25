"""Segment-major dense down GEMV: geometry, unchanged pool slices and worker order."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from recipes import qwen36moe as Q
from recipes.spec import ModelSpec
from q4_1_pack import chunk_geometry

ROOT = Path(__file__).resolve().parents[3]


def spec(ff=17408):
    data = json.loads((Path(__file__).parent / 'fixtures/config_qwen38_27b.json').read_text())
    s = ModelSpec.from_hf_config(data)
    return ModelSpec.from_dict(dict(s.to_dict(), intermediate=ff))


def test_segmented_geometry_reuses_dead_deltanet_scratch():
    s = spec()
    c, f = Q.common(s, 'dense'), Q.ffn_geometry(s)
    assert f.DOWN_SEGMENTS == ((0, 8192), (8192, 8192), (16384, 1024))
    assert (c.KWIDE, c.PER_CALL, c.DS_FLOATS) == (8192, 2, 1280)
    assert c.ROWS_PC == 640 and c.DS_FLOATS >= c.ROWS_PC
    assert Q.core_l1(c.TAB_BYTES, c.MS_FLOATS, c.DS_FLOATS, c.PER_CALL) == 59392


@pytest.mark.parametrize('ff', [4096, 8192, 12288])
def test_fitting_legacy_geometry_keeps_one_full_table(ff):
    s = spec(ff)
    assert Q.ffn_geometry(s).DOWN_SEGMENTS == ()
    assert Q.kwide(s, 'dense') == max(6144, ff)


def test_segmented_q8_is_explicitly_not_implemented():
    s = ModelSpec.from_dict(dict(spec().to_dict(), quant='q8'))
    with pytest.raises(Q.OpRangeError, match='not implemented.*segmented.*Q4'):
        Q.common(s, 'dense')


def test_dma_slices_cover_original_pool_once_in_segment_major_order():
    from recipes.segmented_dense import segments, weight_slice
    n, k = 5120, 17408
    _, _, rows, cols = chunk_geometry(n, k, 2)
    visited = []
    for start, width in segments(k):
        for band in range(n // 64):
            off, size = weight_slice(k, band, start, width)
            indices = np.arange(off // 5120, (off + size) // 5120)
            assert set(rows[indices]) == {band * 64, band * 64 + 32}
            assert cols[indices].min() == start
            assert cols[indices].max() == start + width - 256
            visited.extend(indices)
    assert sorted(visited) == list(range(len(rows)))


@pytest.mark.parametrize('k', [512, 1024, 8192, 8704, 17408, 24576])
def test_generic_segment_alignment_and_tail(k):
    from recipes.segmented_dense import segments
    parts = segments(k)
    assert sum(w for _, w in parts) == k
    assert all(w <= 8192 and w % 256 == 0 and off % 1024 == 0 for off, w in parts)
    assert [off for off, _ in parts] == list(range(0, k, 8192))


@pytest.mark.parametrize('k', [0, -256, 257])
def test_invalid_segment_width_rejected(k):
    from recipes.segmented_dense import segments
    with pytest.raises(ValueError):
        segments(k)


def worker_functions(ns):
    path = ROOT / 'open_kernels/designs/layer_x/xcommon.py'
    names = {'segmented_down_body', 'segmented_down_sequence'}
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), ns)


def test_worker_retains_all_bands_until_last_segment_and_resets_next_token():
    from test_dense_activation_stream import Fifo
    f = Q.ffn_geometry(spec())
    nbands = f.DOWN_PC
    ds = np.full(nbands * 64, np.nan)
    ms = np.zeros(128)
    tab = []
    events, outputs = [], []
    class Output:
        def acquire(self, n):
            assert n == 1
            self.y = np.full(64, np.nan)
            return self.y
        def release(self, n):
            outputs.append(self.y.copy())
    def prep(e, table, width, i):
        if i == 0:
            table[:] = [e, width]
        assert e == table[0] + i
    def gms(w, table, scratch, group, pb, dst):
        assert dst == 0
        # Distinct segment/band values reveal swapped accumulation slots.
        if group == 0:
            scratch[:64] = 0
        scratch[:64] += w + 1
    def acc(scratch, sums, band, first):
        events.append((tab[0], band, first))
        sl = slice(band * 64, (band + 1) * 64)
        sums[sl] = scratch[:64] if first else sums[sl] + scratch[:64]
    def emit(sums, y, band):
        y[:] = sums[band * 64:(band + 1) * 64]
    ns = dict(FFN=f, range_=range, ELEM=4096, per_band=lambda k: k // 128,
              n_groups=lambda k: k // 256)
    worker_functions(ns)
    total_groups = nbands * sum(width // 256 for _, width in f.DOWN_SEGMENTS)
    for _ in range(2):
        ns['segmented_down_body'](Fifo(total_groups), Fifo(17), Output(),
                                  dict(tab=tab, ms=ms, ds=ds),
                                  dict(prepf=prep, gms=gms, down_acc=acc, down_out=emit))
    assert events[:30] == [(start // 1024, b, int(start == 0))
                           for start, _ in f.DOWN_SEGMENTS for b in range(nbands)]
    expected = np.zeros((nbands, 64))
    cursor = 0
    for _, width in f.DOWN_SEGMENTS:
        ng = width // 256
        for b in range(nbands):
            expected[b] += sum(range(cursor + 1, cursor + ng + 1))
            cursor += ng
    np.testing.assert_array_equal(outputs[:nbands], expected)
    np.testing.assert_array_equal(outputs[nbands:], expected)


def test_host_schedule_drains_before_weights_and_reads_each_h_segment_once():
    from recipes.segmented_dense import weight_slice
    f = Q.ffn_geometry(spec())
    events = []
    class Pipe:
        def fill(self, ep, tensor, tap):
            events.append(('fill', ep, tensor, tap))
        def drain(self, ep, tensor, tap):
            events.append(('drain', ep, tensor, tap))
        def finish(self, *eps):
            events.append(('finish',))
    ns = dict(FFN=f, N_CORES=8, BAND_ROWS=64, ELEM=4096, FF=17408,
              POOL_BYTES=100000000, bt=lambda total, off, n: (off, n),
              weight_slice=weight_slice)
    worker_functions(ns)
    ns['segmented_down_sequence'](Pipe(), Pipe(), Pipe(), 'pool', 'act',
                                  list(range(8)), 'x', list(range(8)),
                                  1000000, 4096, 80000, 123456)
    fills = [e for e in events if e[0] == 'fill']
    assert [e[3] for e in fills if e[1] == 'x'] == [(4096, 32768), (36864, 32768), (69632, 4096)]
    assert all(e[0] == 'drain' for e in events[:8])
    weights = [e for e in fills if e[2] == 'pool']
    assert len(weights) == 3 * 8 * 10
    expected = []
    for start, width in f.DOWN_SEGMENTS:
        for band in range(10):
            for c in range(8):
                off, size = weight_slice(17408, c * 10 + band, start, width)
                expected.append(('fill', c, 'pool', (123456 + off, size)))
    assert weights == expected


def test_recipe_validates_segment_widths_and_keeps_whole_layer_guard(monkeypatch):
    from recipes import qwen35
    monkeypatch.setenv('OPEN_KERNELS_UNVALIDATED', '1')
    monkeypatch.delenv('OPEN_KERNELS_WIDE_GLUE_PROBE', raising=False)
    widths = []
    original = qwen35.require_gemv
    def record(s, role, k, rows, pc):
        widths.append((role, k))
        return original(s, role, k, rows, pc)
    monkeypatch.setattr(qwen35, 'require_gemv', record)
    with pytest.raises(Q.OpRangeError, match='fused wide glue'):
        qwen35.recipe(spec())
    assert [k for role, k in widths if role == 'ffn'] == [5120, 8192, 8192, 1024]
    assert not any(k == 17408 for _, k in widths)


def test_attention_only_segmented_path_allocates_local_output_sums():
    from recipes.spec import FULL
    s = ModelSpec.from_dict(dict(spec().to_dict(), layer_types=[FULL] * 64))
    c = Q.common(s, 'dense')
    assert c.DN_DIM == 0 and c.DS_FLOATS == c.ROWS_PC == 640
    assert Q.core_l1(c.TAB_BYTES, c.MS_FLOATS, c.DS_FLOATS, c.PER_CALL) <= Q.L1_BUDGET


@pytest.mark.parametrize('args', [(17408, -1, 0, 8192), (17408, 0, 1, 8192),
                                 (17408, 0, 16384, 2048), (17408, 0, 0, 0)])
def test_invalid_dma_slice_is_rejected(args):
    from recipes.segmented_dense import weight_slice
    with pytest.raises(ValueError):
        weight_slice(*args)


def test_segmented_ffn_cannot_be_promoted_by_already_validated_gemv_widths(monkeypatch):
    from recipes import qwen35
    monkeypatch.delenv('OPEN_KERNELS_UNVALIDATED', raising=False)
    with pytest.raises(Q.OpRangeError, match='segmented FFN.*not yet validated'):
        qwen35.recipe(spec())

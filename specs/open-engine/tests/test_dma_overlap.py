"""Upstream DMA overlap must preserve region dependencies and every head's state."""
import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def extract(path, name, ns):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    return ns[name]


def test_oldest_drain_releases_qkv_without_waiting_for_z_or_other_channels():
    completed = []
    class Group:
        def finish(self): completed.append(self.region)
    class Endpoint:
        def drain(self, tensor, tap, wait, group):
            assert wait
            group.region = (self, tap)
    pipeline = extract(ROOT/'open_kernels/ironutil.py', 'Pipeline', dict(deque=deque, TaskGroup=Group))(3)
    a, b = Endpoint(), Endpoint()
    pipeline.drain(a, 'act', 'qkv')
    pipeline.drain(a, 'act', 'z')
    pipeline.drain(b, 'act', 'other')
    pipeline.finish_oldest(a)
    assert completed == [(a, 'qkv')]
    pipeline.finish(a)
    assert completed == [(a, 'qkv'), (a, 'z')]
    pipeline.finish_oldest(a)  # empty channel is harmless
    pipeline.finish(b)
    assert completed[-1] == (b, 'other')


@pytest.mark.parametrize('heads', [32, 48])
def test_head_major_schedule_preserves_padded_state_and_endpoint_order(heads):
    events = []
    class Pipe:
        def fill(self, ep, tensor, tap): events.append(('fill', ep, tensor, tap))
        def drain(self, ep, tensor, tap): events.append(('drain', ep, tensor, tap))
    per_core = heads//8
    rec, ohb, stride, state_off = 2048, 512, 140*128*4, 61440
    ns = dict(N_CORES=8, DN_HEADS_PC=per_core, CALL_BYTES=4096,
              R=SimpleNamespace(linear=SimpleNamespace(RECORD_BYTES=rec, O_HEAD_BYTES=ohb)),
              bt=lambda total, off, size: (total, off, size))
    sequence = extract(ROOT/'open_kernels/designs/layer_x/xcommon.py', 'dn_sequence', ns)
    sequence(Pipe(), Pipe(), 'state', 'act', list(range(8)), list(range(8)),
             1000000, 10000, 200000, state_off+heads*stride, state_off, stride)
    assert len(events) == heads*5
    # All eight cores receive the first head before a throttle can wait on the
    # same core's next head. Each endpoint retains record,S,write S,o,S order.
    assert [e[1] for e in events[::5]] == list(range(8))*per_core
    seen = []
    for c in range(8):
        own = [e for e in events if e[1] == c]
        for h in range(per_core):
            head = c*per_core+h
            record, read1, write, output, read2 = own[h*5:(h+1)*5]
            assert record == ('fill', c, 'act', (1000000,10000+head*rec,4096))
            address = (state_off+heads*stride,state_off+head*stride,stride)
            assert read1 == read2 == ('fill', c, 'state', address)
            assert write == ('drain', c, 'state', address)
            assert output == ('drain', c, 'act', (1000000,200000+head*ohb,ohb))
            seen.append(head)
    assert sorted(seen) == list(range(heads))

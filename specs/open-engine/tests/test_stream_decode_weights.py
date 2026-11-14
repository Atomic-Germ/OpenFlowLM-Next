"""Streaming changes weight residency only, preserving dispatch/capture ordering."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]

def transform(text):
    spec = importlib.util.spec_from_file_location('stream_decode_weights', ROOT/'utilities/stream-decode-weights.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.transform(text)


def test_reuse_pool_with_reload_before_every_dispatch():
    source = ('device\nbuf pool0 128 /weights/0.bin\nbuf state0 32\n'
              'buf pool1 256 /weights/1.bin\nbuf state1 32\n'
              'run lx pool0 xres consts0 state0 act0\ndump act0 /out/a\n'
              'run ax pool1 xres consts1 state1 act1 ptab\n'
              'run lx pool0 xres consts0 state0 act0\n')
    out = transform(source)
    assert 'buf streamed_pool 256\n' in out
    assert 'buf pool' not in out
    assert out.count('load streamed_pool /weights/0.bin\nrun lx streamed_pool') == 2
    assert 'load streamed_pool /weights/1.bin\nrun ax streamed_pool' in out
    assert 'buf state0 32\n' in out and 'buf state1 32\n' in out
    assert 'dump act0 /out/a\n' in out


@pytest.mark.parametrize('source', [
    'buf pool0 128\nrun lx pool0 xres\n',
    'buf pool0 128 /a\nrun lx pool1 xres\n',
    'buf pool0 128 /a\nbuf streamed_pool 128\nrun lx pool0 xres\n',
    'buf pool0 128 /a\nload pool0 /other\nrun lx pool0 xres\n',
    'buf pool0 128 /a\nrun other pool0 xres\n',
])
def test_reject_unsupported_pool_uses(source):
    with pytest.raises(ValueError):
        transform(source)

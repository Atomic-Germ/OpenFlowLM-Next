"""Wide RMSNorm must consume bounded input windows and normalize across both halves."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def module():
    path = ROOT / 'open_kernels/designs/ln/ln_stream.py'
    spec = importlib.util.spec_from_file_location('ln_stream_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_streamed_worker_uses_one_input_at_a_time_and_resets_statistics():
    mod = module()
    n = 5120
    saved, sums = np.full(n, np.nan), np.full(32, np.nan)
    class Input:
        def __init__(self, data):
            self.data, self.held = list(data), False
        def acquire(self, count):
            assert count == 1 and not self.held
            self.held = True
            return self.data[0]
        def release(self, count):
            assert count == 1 and self.held
            self.held = False
            self.data.pop(0)
    class Output:
        def __init__(self):
            self.data = []
        def acquire(self, count):
            assert count == 1
            self.current = np.full(n, np.nan)
            return self.current
        def release(self, count):
            self.data.append(self.current.copy())
    def copy(x, y, half):
        y[half*n//2:(half+1)*n//2] = x
    def acc(a, y, ss, out, half):
        sl = slice(half*n//2, (half+1)*n//2)
        y[sl] += a
        if half == 0:
            ss[:] = 0
        ss[:] += np.sum(y[sl]**2) / len(ss)
        out[:n//2] = y[sl]
    def finish(y, ss, w, out):
        out[:] = y * w / np.sqrt(ss.sum()/n + 1e-6)
    for sign in (1, -1):
        x = np.linspace(-2, 3, n) * sign
        a, w = np.linspace(.5, 1, n), np.linspace(.8, 1.2, n)
        inp = Input([x[:n//2], a[:n//2], x[n//2:], a[n//2:], w])
        out = Output()
        mod.body(inp, out, saved, sums, copy, acc, finish)
        assert not inp.data and not inp.held and len(out.data) == 3
        np.testing.assert_array_equal(np.concatenate([p[:n//2] for p in out.data[:2]]), x+a)
        np.testing.assert_allclose(out.data[2], (x+a)*w/np.sqrt(np.mean((x+a)**2)+1e-6), rtol=1e-14)


def test_streamed_schedule_interleaves_halves_and_drains_before_inputs():
    mod = module()
    events = []
    class Pipe:
        def fill(self, ep, tensor, tap): events.append(('fill', tensor, tap))
        def drain(self, ep, tensor, tap): events.append(('drain', tensor, tap))
        def finish(self): events.append(('finish',))
    mod.sequence(Pipe(), lambda total, off, size: (total, off, size),
                 5120, 'x', 'add', 'w', 'y', 'xn', 'in', 'out')
    assert events == [('drain', 'y', (5120, 0, 5120)), ('drain', 'xn', (5120, 0, 5120)),
                      ('fill', 'x', (5120, 0, 2560)), ('fill', 'add', (5120, 0, 2560)),
                      ('fill', 'x', (5120, 2560, 2560)), ('fill', 'add', (5120, 2560, 2560)),
                      ('fill', 'w', (5120, 0, 5120)), ('finish',)]


def test_streamed_scratch_budget():
    mod = module()
    assert mod.l1_bytes(5120) == 57472
    with pytest.raises(ValueError, match='budget'):
        mod.check_width(6144)
    for n in (0, 5121):
        with pytest.raises(ValueError): mod.check_width(n)


@pytest.mark.parametrize('family', ['qwen35', 'qwen36moe', 'dense'])
def test_recipe_cache_covers_streamed_worker(family):
    recipe = __import__('recipes.' + family, fromlist=['KERNEL_SOURCES'])
    files = {p.relative_to(ROOT / 'open_kernels').as_posix()
             for pattern in recipe.KERNEL_SOURCES
             for p in (ROOT / 'open_kernels').glob(pattern)}
    assert 'designs/ln/ln_stream.py' in files


def test_acceptance_handles_zero_and_rejects_nonfinite_or_wrong_results():
    spec = importlib.util.spec_from_file_location('ln_probe', ROOT / 'utilities/test-wide-ln.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    from ml_dtypes import bfloat16
    zero = np.zeros(5120, bfloat16)
    assert probe.metrics(zero, zero)['passed']
    for wrong in (np.ones(5120, bfloat16), np.full(5120, np.nan, bfloat16)):
        assert not probe.metrics(wrong, zero)['passed']
    # One large outlier and widespread one-ulp errors must not pass.
    ref = np.ones(5120, bfloat16)
    wrong = ref.copy()
    wrong[0] = 2
    assert not probe.metrics(wrong, ref)['passed']
    wrong = np.full(5120, 1.0078125, bfloat16)
    assert not probe.metrics(wrong, ref)['passed']

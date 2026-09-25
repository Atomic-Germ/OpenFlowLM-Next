"""Bounded-memory wide residual RMSNorm worker and matching DMA schedule."""


def l1_bytes(n):
    # Two input elements, one output, saved fp32 residual, 32 statistics, stack.
    return 10 * n + 32 * 4 + 0x1800


def check_width(n):
    if n <= 0 or n % 64:
        raise ValueError('streamed LN width must be positive and divisible by 64')
    if l1_bytes(n) > 60 * 1024:
        raise ValueError(f'streamed LN width {n} exceeds the 60 KiB L1 budget')


def body(inp, out, saved, sums, copy, acc, finish):
    for half in range(2):
        x = inp.acquire(1)
        copy(x, saved, half)
        inp.release(1)
        add = inp.acquire(1)
        y = out.acquire(1)
        acc(add, saved, sums, y, half)
        out.release(1)
        inp.release(1)
    w = inp.acquire(1)
    xn = out.acquire(1)
    finish(saved, sums, w, xn)
    out.release(1)
    inp.release(1)


def sequence(pipe, tap, n, x, add, w, y, xn, inp, out):
    pipe.drain(out, y, tap(n, 0, n))
    pipe.drain(out, xn, tap(n, 0, n))
    for half in range(2):
        pipe.fill(inp, x, tap(n, half * n // 2, n // 2))
        pipe.fill(inp, add, tap(n, half * n // 2, n // 2))
    pipe.fill(inp, w, tap(n, 0, n))
    pipe.finish()

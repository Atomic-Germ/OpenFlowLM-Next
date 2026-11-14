r"""The split norm helper: the same stages, for a width the fused one cannot hold.

The fused helper (designs/ln/ln_y.cc, ln_xn.cc) takes five input elements at once --
the residual's two halves, the norm weights and the added term's two halves -- and
holds them in the norm core's L1 while it works. At HID 5120 those five 10 KB
elements plus the output element and the stack do not fit the core's 64 KB, so the
helper STREAMS instead: the residual add goes half by half (ln_add2), the sum goes
to DDR, and the norm reads it back (ln_nr). Same arithmetic, same bits -- ln_add2
forms the sum exactly as ln_y does.

Two designs need it: the MoE / Qwen3.5 tail (designs/layer_x/xcommon.py) and the
dense one (designs/dense/dx.py). The bodies live here so the two cannot drift: a
helper that adds the residual one way in one design and another way in the other
is a bug waiting for the width that reaches both.
"""
from __future__ import annotations


def split_stages(ain, aout, f_nr, f_add2):
    """Stages 1 and 2 of the split norm helper (recipes/qwen36moe.py norm_split):

      1. layer-entry norm                 [x0 x1 lnw] -> [xn]
      2. res_i = x_i + a_i, i = 0, 1      [x_i a_i]   -> [res_i]     (res goes to DDR)
         xm = post_attention_norm(res)    [res0 res1 w] -> [xm]      (res read back)
    Three input elements at most, so the fifo is 3 deep and 4 elements are ever held.
    """
    e = ain.acquire(3)
    o = aout.acquire(1)
    f_nr(e[0], e[1], e[2], o)
    aout.release(1)
    ain.release(3)
    for _ in range(2):
        e = ain.acquire(2)
        o = aout.acquire(1)
        f_add2(e[0], e[1], o)
        aout.release(1)
        ain.release(2)
    e = ain.acquire(3)
    o = aout.acquire(1)
    f_nr(e[0], e[1], e[2], o)
    aout.release(1)
    ain.release(3)


def ln_split_body(ain, aout, f_nr, f_add2):
    """... 3. xres_i = res_i + out2_i         [res_i out2_i] -> [xres_i]"""
    split_stages(ain, aout, f_nr, f_add2)
    for _ in range(2):
        e = ain.acquire(2)
        o = aout.acquire(1)
        f_add2(e[0], e[1], o)
        aout.release(1)
        ain.release(2)


def ln_split_body3(ain, aout, f_nr, f_add2, f_add3):
    """... 3. xres_i = res_i + (out2_i + out2b_i), the two down pieces' outputs
    (recipes/qwen36moe.py down_split)   [res_i out2_i out2b_i] -> [xres_i]"""
    split_stages(ain, aout, f_nr, f_add2)
    for _ in range(2):
        e = ain.acquire(3)
        o = aout.acquire(1)
        f_add3(e[0], e[1], e[2], o)
        aout.release(1)
        ain.release(3)

"""Experimental norm worker: FP32 entry and unchanged split FFN stages."""

def entry_body(ain,aout,nr32,add2,add3):
    e=ain.acquire(3)
    for i in range(2):
        o=aout.acquire(1);nr32(e[0],e[1],e[2],o,i);aout.release(1)
    ain.release(3)
    for _ in range(2):
        e=ain.acquire(2);o=aout.acquire(1)
        add2(e[0],e[1],o);aout.release(1);ain.release(2)
    e=ain.acquire(3)
    for i in range(2):
        o=aout.acquire(1);nr32(e[0],e[1],e[2],o,i);aout.release(1)
    ain.release(3)
    for _ in range(2):
        e=ain.acquire(3);o=aout.acquire(1)
        add3(e[0],e[1],e[2],o);aout.release(1);ain.release(3)


def install_entry_worker(X):
    if not X.LN_SPLIT or not X.DOWN_SPLIT or not X.FFN.XM_F32:
        raise ValueError('experimental entry requires split FP32 FFN norm/down')
    def worker(ni,no,L):
        return entry_body,[ni.cons(),no.prod(),L['ln_nr32'],L['ln_add2'],L['ln_add3']]
    X.ln_dense_worker=worker

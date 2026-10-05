"""MADV_DONTNEED eviction of source-tensor pages after conversion.

GGUFReader already mmaps the source file, so tensor bytes fault in lazily,
but the pages remain in the page cache once read. Walking the file is a
one-pass, read-mostly scan: after a tensor is converted its source bytes are
never touched again, so the kernel can be told it can reclaim those pages
right away (the same eviction Guanaco's herdcache does with MADV_DONTNEED),
which keeps a multi-hundred-GB converted source and a multi-GB working set
from competing.
"""
from __future__ import annotations

import ctypes

_MADV_DONTNEED = 4
_libc = None


def _get_libc():
    global _libc
    if _libc is None:
        try:
            _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        except OSError:
            return None
    return _libc


def drop_pages(tensor) -> int:
    """Drop the given tensor's mapped bytes from the page cache.

    Returns the number of bytes advised away (0 if the platform refused).
    """
    libc = _get_libc()
    if libc is None:
        return 0
    try:
        addr = int(tensor.ctypes.data)
        n = int(tensor.nbytes)
        if n > 0:
            libc.madvise(
                ctypes.c_void_p(addr), ctypes.c_size_t(n), ctypes.c_int(_MADV_DONTNEED)
            )
        return n
    except Exception:
        return 0

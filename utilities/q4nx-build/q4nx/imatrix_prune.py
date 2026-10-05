"""Importance-matrix pruning of a dense FFN, for models whose FFN will not fit a core.

WHAT THIS IS. A dense FFN is three tensors sharing one axis: `up` and `gate`
are `[hidden, FF]`, `down` is `[FF, hidden]`. Narrowing that axis to K columns
makes the whole FFN fit a main core's L1 (see recipes/qwen36moe.py `per_call`),
which is the only thing standing between a wide model and an export at all.

The axis is a set of K of the FF's neurons. WHICH K is the entire question:

  * keeping the FIRST K (a tail chop) drops a contiguous slab, and the surviving
    up/gate columns were never trained knowing the dropped ones were gone;
  * keeping the K whose input activations carried the most mass keeps the neurons
    the model actually uses.

An importance matrix ("imatrix") records per-input activation energy per tensor
over a calibration set, which is exactly the second criterion. llama.cpp writes
one as a GGUF of `<tensor>.in_sum2` / `<tensor>.counts` pairs next to the model;
`--imatrix` points at that file.

MEASURED, on the Qwen3.8-27B imatrix (all 64 layers, K = 12288 of 17408):

    top-K by importance   88.5% mean activation mass retained, 83.2% worst
    first K (tail chop)   70.6% by construction

so the ranking is worth roughly 18 points over chopping the tail. That is a
large margin, but it is NOT the same kind of loss as a quantisation choice:
Q4_K keeps every weight at ~4.5 bits, while this deletes 29% of the FFN
outright. Do not treat a pruned container as interchangeable with a denser one
on the same model -- see the README text `pruned_note()` contributes.

THE AXIS MUST BE GATHERED CONSISTENTLY. `up`/`gate` carry the axis as
COLUMNS, `down` as ROWS, and the same index set selects both. Pruning one
tensor per layer independently, or taking the top-K of each tensor's own scores,
produces a container that loads, quantizes cleanly, and computes a different
network than the ranking describes -- the failure is silent. Hence
`ffn_index_set`, which derives ONE index set per layer from `ffn_down`'s scores
alone and every caller obeys it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch

# The GGUF names the three FFN tensors per layer like this; the pruner keys on
# them, not on the converted q4nx names, because it runs before conversion.
FFN_DOWN = "ffn_down.weight"
FFN_UP = "ffn_up.weight"
FFN_GATE = "ffn_gate.weight"


def layer_of(gguf_name: str) -> int | None:
    """The block index in a GGUF tensor name, or None if it is not a layer tensor."""
    parts = gguf_name.split(".")
    if len(parts) < 2 or not parts[0].startswith("blk"):
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def imatrix_path(explicit: str | None, model_dir: Path | None) -> Path | None:
    """Locate the imatrix: an explicit --imatrix wins, else a file beside the GGUF.

    An imatrix has no standard filename, so a sidecar is matched by CONTENT
    (a GGUF whose tensors are `*.in_sum2` / `*.counts`) rather than by name --
    llama.cpp's own convention varies (`imatrix.gguf`, `*-imatrix.gguf`, a
    quant-specific name). Returns None when there is nothing to read, which the
    caller turns into an explanation rather than a silent full-width pack.
    """
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"--imatrix: no such file: {p}")
        return p
    if model_dir and model_dir.is_dir():
        for p in sorted(model_dir.glob("*.gguf")):
            if p.name.endswith("-q4_0.gguf") or "Q4_0" in p.name:
                continue        # a weights file, not an imatrix
            if _looks_like_imatrix(p):
                return p
    return None


def _looks_like_imatrix(path: Path) -> bool:
    try:
        from gguf import GGUFReader
    except Exception:
        return False
    try:
        r = GGUFReader(str(path))
    except Exception:
        return False
    names = [t.name for t in r.tensors]
    return bool(names) and any(n.endswith(".in_sum2") for n in names)


class Imatrix:
    """Per-tensor, per-input-column activation energy read from an imatrix GGUF."""

    def __init__(self, path: Path):
        from gguf import GGUFReader
        self.path = Path(path)
        r = GGUFReader(str(self.path))
        self._raw = {t.name: t for t in r.tensors}
        self._cache: Dict[str, np.ndarray] = {}

    def has(self, gguf_name: str) -> bool:
        return f"{gguf_name}.in_sum2" in self._raw

    def scores(self, gguf_name: str) -> np.ndarray:
        """Activation energy per input column of `gguf_name`, ascending index order.

        llama.cpp stores these as F32 arrays; a `.counts` entry alongside is the
        token count the sums were accumulated over, which is uniform across
        tensors for one calibration run and so does not change a RANKING. It is
        deliberately not applied -- dividing by a per-tensor count would rescale
        magnitudes that are only ever compared within one tensor.
        """
        if gguf_name in self._cache:
            return self._cache[gguf_name]
        key = f"{gguf_name}.in_sum2"
        if key not in self._raw:
            raise KeyError(f"imatrix {self.path.name}: no {key}")
        v = np.asarray(self._raw[key].data, dtype=np.float32)
        self._cache[gguf_name] = v
        return v

    def layers(self) -> list[int]:
        out = set()
        for n in self._raw:
            L = layer_of(n.removesuffix(".in_sum2").removesuffix(".counts"))
            if L is not None and n.endswith(".in_sum2"):
                out.add(L)
        return sorted(out)

    def counts(self, gguf_name: str) -> np.ndarray:
        """The `.counts` companion of a tensor's in_sum2.

        For per-tensor imatrix this is one scalar (token count over the
        calibration run); for FUSED EXPERT tensors it is per-expert (shape
        [1, num_experts]) -- the dispatch frequency that drives Guanaco's
        imatrix prior, and the score used to rank experts for pruning.
        """
        key = f"{gguf_name}.counts"
        if key not in self._raw:
            raise KeyError(f"imatrix {self.path.name}: no {key}")
        return np.asarray(self._raw[key].data, dtype=np.float64)


def ffn_index_set(imx: Imatrix, layer: int, keep: int) -> np.ndarray:
    """The K surviving intermediate columns for ONE layer, as a sorted index array.

    Derived from `ffn_down` alone. `ffn_down` is `[FF, hidden]`, so its
    `in_sum2` scores the hidden axis -- the weights arriving from the residual
    stream -- which is the same quantity `ffn_up` / `ffn_gate` see on their
    input columns. Using one tensor's ranking for all three is what keeps the
    gathered FFN a coherent network; scoring each tensor separately does not.
    """
    scores = imx.scores(f"blk.{layer}.{FFN_DOWN}")
    n = scores.shape[0]
    if keep >= n:
        return np.arange(n, dtype=np.int64)
    # argpartition then sort: the full argsort of 17408 floats x 64 layers is
    # wasted work when only the top K is wanted.
    top = np.argpartition(scores, n - keep)[n - keep:]
    top.sort()
    return top.astype(np.int64)


def gather_ffn(w: torch.Tensor, idx: np.ndarray, *, ffn_width: int) -> torch.Tensor:
    """Select `idx` from an FFN tensor on its intermediate axis, by LENGTH.

    The axis is identified by WHICH DIMENSION IS THE FF WIDTH, never by a
    hardcoded row/column position. That matters because ggml stores these
    tensors column-major: `gguf.dequantize` returns the TRANSPOSE of the shape
    the GGUF header declares, so `ffn_down` arrives as [hidden, FF] and
    `ffn_up` / `ffn_gate` as [FF, hidden] -- the opposite of what the header
    says, and the opposite of each other. Keying on the declared orientation
    indexes the wrong axis, and because K < FF the result is not even an error:
    it gathers the wrong neurons and the container converts cleanly while
    computing a different network.
    """
    if w.shape[0] == ffn_width and w.shape[1] != ffn_width:
        axis = 0
    elif w.shape[1] == ffn_width and w.shape[0] != ffn_width:
        axis = 1
    else:
        raise ValueError(
            f"--prune-ffn: cannot tell which axis is the FF axis of a {tuple(w.shape)} "
            f"tensor; expected one dimension to be {ffn_width} and the other to differ")
    t = torch.as_tensor(idx, dtype=torch.long)
    return w.index_select(axis, t).contiguous()


def retention_report(imx: Imatrix, keep: int, layers: list[int]) -> list[tuple[int, float]]:
    """(layer, fraction of activation mass retained) for each layer, for the log."""
    out = []
    for L in layers:
        s = imx.scores(f"blk.{L}.{FFN_DOWN}")
        total = float(s.sum())
        if total <= 0.0:
            out.append((L, 0.0))
            continue
        idx = ffn_index_set(imx, L, keep)
        out.append((L, float(s[idx].sum()) / total))
    return out


def pruned_note(keep: int, ffn: int, retained: float | None, mtp: int = 0) -> str:
    """The README sentence that makes this artifact honestly labelled."""
    kept = f"{retained * 100:.1f}%" if retained is not None else "an unmeasured share"
    return (
        f"**This model's FFN was pruned from {ffn} to {keep} neurons by importance "
        f"matrix (imatrix) ranking**, keeping the {keep} whose input activations "
        f"carried the most mass ({kept} retained on average). This is a structural "
        f"reduction, not a quantisation choice: those neurons are gone, so this is "
        f"NOT equivalent in quality to the unpruned model at any bit width. It is "
        f"however not the same as truncating the FFN, which retains only "
        f"{keep / ffn * 100:.1f}% of the mass by construction."
        + (f" The model's {mtp} multi-token-prediction (speculative decoding) "
           f"block{'s are' if mtp != 1 else ' is'} also omitted: unused without "
           f"speculative decoding, and not counted in num_hidden_layers."
           if mtp else "")
    )


def env_imatrix() -> str | None:
    """OFLM_IMATRIX, so a pack invoked through `oflm pack` can name the imatrix
    without the caller re-typing the flag (the C++ dispatcher shells out)."""
    return os.environ.get("OFLM_IMATRIX") or None

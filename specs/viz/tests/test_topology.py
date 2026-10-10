# Traces: VIZ-TOPOLOGY
import collections
import struct

import pytest
from conftest import FIX, mlir_pair

from viz.topology import TopologyError, extract


def ddr_bytes(insts: bytes) -> dict[int, int]:
    """stream_patch::ddr_bytes: each BD blockwrite paired with the DDR patch after it, per buffer argument."""
    w = struct.unpack(f"<{len(insts) // 4}I", insts)
    op_len = {0: 6, 1: 12, 3: 7, 0x80: 4, 0x81: 12}
    per: dict[int, int] = collections.Counter()
    bd, i = None, 4
    while i < len(w):
        if w[i] == 1:
            bd = i
        elif w[i] == 0x81 and i + 11 < len(w) and bd is not None and w[bd + 2] + 4 == w[i + 6]:
            per[w[i + 8]] += w[bd + 4] * 4
        i += op_len.get(w[i], 1)
    return dict(per)


@pytest.fixture(scope="module")
def lx0():
    return extract(*mlir_pair("lx0"))


def test_lx0_tiles_cores_and_streams(lx0):
    cores = sorted(tuple(c["tile"]) for c in lx0["cores"])
    assert cores == sorted([(c, 2) for c in range(8)] + [(0, 3), (1, 3), (2, 3)])
    assert sorted(t[0] for t in lx0["tiles"] if t[2] == "shim") == list(range(8))
    assert not [t for t in lx0["tiles"] if t[2] == "mem"]
    assert lx0["fifos"]["w0"] == {"src": [0, 0], "dst": [[0, 2]], "depth": 2, "elem_bytes": 10240,
                                  "shim": {"dir": "MM2S", "ch": lx0["fifos"]["w0"]["shim"]["ch"]}}
    assert len(lx0["fifos"]["x"]["dst"]) == 8
    assert len(lx0["tasks"]) == 263
    assert {"gemv_q8_gy", "dnx_row", "ln_nr", "router_acc"} <= {f for c in lx0["cores"] for f in c["funcs"]}


def test_two_streams_of_one_image(lx0):
    lx1 = extract(*mlir_pair("lx1"))
    for k in ("tiles", "cores", "fifos", "links"):
        assert lx0[k] == lx1[k]
    assert (len(lx0["tasks"]), len(lx1["tasks"])) == (263, 330)


def test_task_bytes_match_the_compiled_stream(lx0):
    mine = collections.Counter()
    for t in lx0["tasks"]:
        for b in t["bds"]:
            mine[b["arg"]] += b["len"] * (t["repeat"] + 1)
    assert dict(mine) == ddr_bytes((FIX / "mlir" / "lx0" / "insts.bin").read_bytes())
    assert mine[0] == 31457280


def test_unplaced_cores_resolve_through_the_placed_file():
    lm = extract(*mlir_pair("lm_head_q8"))
    assert sorted(tuple(c["tile"]) for c in lm["cores"]) == [(c, r) for c in (0, 1) for r in (2, 3, 4, 5)]
    assert lm["fifos"]["w2"]["src"] == [1, 0] and lm["fifos"]["w2"]["dst"] == [[0, 4]]
    ln = extract(*mlir_pair("ln"))
    assert [c["tile"] for c in ln["cores"]] == [[0, 2]] and ln["fifos"]["in"]["src"][1] == 0


def test_mem_tiles_and_links():
    g = extract(*mlir_pair("gemm_n2048_k512"))
    mem = [t for t in g["tiles"] if t[2] == "mem"]
    assert mem and all(t[1] == 1 for t in mem)
    assert len(g["links"]) == 20 and len(g["cores"]) == 32
    for lk in g["links"]:
        assert lk["tile"][1] == 1
        assert all(g["fifos"][n]["dst"] == [lk["tile"]] for n in lk["in"])
        assert all(g["fifos"][n]["src"] == lk["tile"] for n in lk["out"])


def test_refuses_a_core_it_cannot_place():
    aie, placed = mlir_pair("lm_head_q8")
    lines = placed.splitlines()
    i = next(k for k, ln in enumerate(lines) if "link_files" in ln and "loc(#" in ln)
    lines[i] = lines[i][:lines[i].rindex(" loc(")]
    with pytest.raises(TopologyError, match=r"core %\w+ \(aie\.mlir:\d+\) has no placed tile"):
        extract(aie, "\n".join(lines))


def test_refuses_a_task_on_an_undeclared_fifo():
    aie, placed = mlir_pair("ln")
    aie = aie.replace("dma_configure_task_for @in ", "dma_configure_task_for @nosuch ", 1)
    with pytest.raises(TopologyError, match="@nosuch, which is not a declared fifo"):
        extract(aie, placed)

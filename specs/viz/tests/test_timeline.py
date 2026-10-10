# Traces: VIZ-TIMELINE-DECODE
import collections

import pytest

from viz.timeline import BW_GBPS, attnpos, core_ops, decode_timeline


@pytest.fixture(scope="module")
def two(two_context):
    m, topos = two_context
    return m, topos, decode_timeline(m, topos)


def test_every_dispatch_in_manifest_order(two):
    m, _, tl = two
    want = [s["kernel"] for lt in m["layers"] for s in m["layer_types"][lt]["program"] if s["op"] == "run"]
    want += [s["kernel"] for s in m["tail"] if s["op"] == "run"]
    got = [e["name"] for e in tl["events"] if e["kind"] == "dispatch"]
    assert got == want and len(got) == 82


def test_one_route_between_each_layers_two_dispatches(two):
    m, _, tl = two
    ev = tl["events"]
    for layer in range(len(m["layers"])):
        mine = [e for e in ev if e.get("layer") == layer]
        assert [(e["kind"], e["name"]) for e in mine if e["kind"] != "ctx"] == \
            [("dispatch", mine[0]["name"]), ("host", "route"), ("dispatch", mine[-1]["name"])]


def test_events_tile_the_step(two):
    _, _, tl = two
    ev = tl["events"]
    assert ev[0]["t0"] == 0
    for a, b in zip(ev, ev[1:]):
        assert b["t0"] == pytest.approx(a["t1"], abs=2e-3)
    assert ev[-1]["t1"] == pytest.approx(tl["total_us"], abs=2e-3)
    assert sum(e["kind"] == "ctx" for e in ev) == 22


def test_every_weight_task_names_its_tensor(two):
    _, _, tl = two
    S = tl["strings"]
    for key, tp in tl["templates"].items():
        for x in tp["tasks"]:
            if S[x[4]] in ("pool", "consts", "lmpool"):
                assert S[x[5]] not in ("layer weights", "layer constants", "lm_head weights"), (key, x)
    lx1 = tl["templates"]["lx1@linear_attention"]
    assert lx1["expert_slots"] == 8
    assert {S[x[5]] for x in lx1["tasks"]} >= {"experts down", "mlp.share_down_exps_proj"}


def test_cores_run_their_whole_program(two):
    m, topos, tl = two
    S = tl["strings"]
    for lt, spec in m["layer_types"].items():
        runs = [s["kernel"] for s in spec["program"] if s["op"] == "run"]
        rtp = {(e[1], e[2]): e[3] for k in runs for e in topos[k]["events"] if e[0] == "rtp"}
        want = collections.Counter(op[1] for c in topos[runs[0]]["cores"] for op in core_ops(c["program"], rtp) if op[0] == "call")
        got = collections.Counter()
        for k in runs:
            tp = tl["templates"][f"{k}@{lt}"]
            assert tp["anomalies"] == 0
            for sp in tp["spans"]:
                got[S[sp[1]]] += sp[4]
        assert got == want, lt


def test_one_context_layer_types_differ_only_by_runtime_words(one_context):
    m, topos = one_context
    tl = decode_timeline(m, topos)
    assert sum(e["kind"] == "ctx" for e in tl["events"]) == 3
    lx0, ax0 = tl["templates"]["lx0@linear_attention"], tl["templates"]["ax0@full_attention"]
    assert lx0["image"] == ax0["image"]
    S = tl["strings"]
    calls = lambda tp: collections.Counter(S[s[1]] for s in tp["spans"])
    assert calls(lx0)["dnx_row"] > 0 and calls(ax0)["dnx_row"] == 0
    assert calls(ax0)["attn_q"] > 0 and calls(lx0)["attn_q"] == 0


def test_attnpos_sizes_the_kv_window(two_context):
    m, topos = two_context
    kd, lay = m["kernels"]["ax0"], m["layout"]
    def window(pos):
        t = attnpos(topos["ax0"], kd, lay, pos)
        return [b for x in t["tasks"] for b in x["bds"] if b["arg"] == 3 and x["dir"] == "in"][0]
    assert kd["rb"] == 4
    assert window(1)["len"] == 3 * lay["kv_row"]
    assert window(9)["len"] == 11 * lay["kv_row"]
    row = [b for x in attnpos(topos["ax0"], kd, lay, 9)["tasks"] for b in x["bds"] if b["arg"] == 3 and x["dir"] == "out"]
    assert [b["off"] for b in row] == [9 * lay["kv_row"]]


def test_calibration_is_recorded(two):
    _, _, tl = two
    cal = tl["calibration"]
    assert cal["bw_gbps"] == BW_GBPS and "OPEN-DECODE-ONE-CONTEXT" in cal["source"]
    assert tl["position"] == 1

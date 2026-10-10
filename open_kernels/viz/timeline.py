"""A set's manifest + topologies -> one decode step of NPU and host events (VIZ-TIMELINE-DECODE)."""
from __future__ import annotations

import hashlib
import json
import random

# Fitted so the k35q41 step models 70.3 ms, OPEN-DECODE-ONE-CONTEXT's position-1 median (2026-09-22).
BW_GBPS = 40.1
CTX_SWITCH_US = 950.0       # OPEN-DECODE-ONE-CONTEXT: the 27B's two-context ax0 penalty (2026-09-23)
CALL_US = 0.25              # floor for a core call whose data is already in tile memory
HOST_US = {"embed": 20.0, "attnpos": 10.0, "route": 45.0, "readback": 150.0, "sample": 250.0}
EPS = 1e-9


class TimelineError(ValueError):
    pass


def _eval(e, rtp):
    if isinstance(e, int):
        return e
    if e is None:
        raise TimelineError("a loop bound the parser could not resolve")
    if e[0] == "rtp":
        return rtp.get((e[1], e[2]), 0)
    if e[0] == "sub":
        return _eval(e[1], rtp) - _eval(e[2], rtp)
    raise TimelineError(f"loop bound {e!r}")


def core_ops(program: list, rtp: dict, limit: int = 4_000_000):
    """One pass of a core program: ("call", fn, consumed ranges held), ("prod" | "relc", fifo, lo, hi)."""
    held: dict = {}
    count = [0]

    def run(prog):
        for op in prog:
            k = op[0]
            if k == "call":
                count[0] += 1
                if count[0] > limit:
                    raise TimelineError(f"a core program runs past {limit} calls in one pass")
                yield ("call", op[1], [(f, lo, hi) for (f, m), (lo, hi) in held.items() if m == "C" and hi > lo])
            elif k == "acq":
                lo, hi = held.get((op[1], op[2]), (0, 0))
                held[(op[1], op[2])] = (lo, max(hi, lo + op[3]))
            elif k == "rel":
                lo, hi = held.get((op[1], op[2]), (0, 0))
                held[(op[1], op[2])] = (lo + op[3], max(hi, lo + op[3]))
                yield ("prod" if op[2] == "P" else "relc", op[1], lo, lo + op[3])
            elif k == "forever":
                yield from run(op[1])
            elif k == "for":
                lb, ub, st = (_eval(x, rtp) for x in op[1:4])
                for _ in range(max(0, -(-(ub - lb) // st))):
                    yield from run(op[4])

    yield from run(program)


class Fifo:
    """One objectfifo's elements: issue, transfer, arrival, release and drain times."""

    def __init__(self, name: str, f: dict, cores_at: dict):
        self.name, self.eb, self.depth = name, f["elem_bytes"], f["depth"]
        self.shim_in = f["src"][1] == 0
        self.shim_out = any(d[1] == 0 for d in f["dst"])
        self.consumers = [cores_at[tuple(d)] for d in f["dst"] if tuple(d) in cores_at]
        self.room_from: list[tuple[Fifo, int]] = []
        self.inputs: list[Fifo] = []
        self.issued: list[list] = []
        self.n_issued = 0.0
        self.ip = 0
        self.arr: list[float] = []
        self.x0: list[float] = []
        self.drain: list[float] = []
        self.d0: list[float] = []
        self.rel: dict[int, list[float]] = {c: [] for c in self.consumers}

    def avail(self) -> int:
        return min(i.avail() for i in self.inputs) if self.inputs else len(self.arr)

    def avail_t(self, j: int) -> float:
        return max(i.avail_t(j) for i in self.inputs) if self.inputs else self.arr[j]

    def cover(self, j: int):
        while self.ip < len(self.issued) and self.issued[self.ip][1] <= j + EPS:
            self.ip += 1
        return self.issued[self.ip] if self.ip < len(self.issued) else None

    def room_t(self, j: int):
        """When every consumer has released element j - depth; None while one has not."""
        k = j - self.depth
        if k < 0:
            return 0.0
        t = 0.0
        for f, c in self.room_from or [(self, c) for c in self.consumers]:
            r = f.rel[c]
            if len(r) <= k:
                return None
            t = max(t, r[k])
        return t


class CoreRun:
    def __init__(self, idx: int, gen):
        self.idx = idx
        self.gen = gen
        self.t = 0.0
        self.spans: list[list] = []
        self.pending = next(gen, None)


def _task_bytes(t: dict) -> int:
    return sum(b["len"] for b in t["bds"]) * (t["repeat"] + 1)


def simulate_pass(topos: list[dict], gaps: list[float], bw_gbps: float = BW_GBPS) -> dict:
    """Discrete-event run of one layer program's dispatches over one image, cores fresh; times in us."""
    img = topos[0]
    bpu = bw_gbps * 1e3
    cores_at = {tuple(c["tile"]): i for i, c in enumerate(img["cores"])}
    fifos = {n: Fifo(n, f, cores_at) for n, f in img["fifos"].items()}
    for lk in img["links"]:
        outs = [fifos[o] for o in lk["out"]]
        for o in outs:
            o.inputs = [fifos[i] for i in lk["in"]]
        for i in lk["in"]:
            fifos[i].room_from = [(o, c) for o in outs for c in o.consumers]
    rtp: dict = {}
    for e in img["events"]:
        if e[0] == "rtp":
            rtp[(e[1], e[2])] = e[3]
    cores = [CoreRun(i, core_ops(c["program"], rtp)) for i, c in enumerate(img["cores"])]
    for f in fifos.values():
        for c in f.consumers:
            f.rel.setdefault(c, [])
    ddr = 0.0
    out = [{"t0": None, "t1": None, "tasks": {}, "anomalies": 0} for _ in topos]
    out[0]["t0"] = 0.0
    st = {"d": 0, "ei": 0, "t": 0.0}

    def seq_eager() -> bool:
        moved = False
        while st["d"] < len(topos):
            d = st["d"]
            ev = topos[d]["events"]
            if st["ei"] < len(ev):
                e = ev[st["ei"]]
                if e[0] == "start":
                    t = topos[d]["tasks"][e[1]]
                    f = fifos[t["fifo"]]
                    n = _task_bytes(t) / f.eb
                    tk = {"lo": f.n_issued, "hi": f.n_issued + n, "issue": st["t"], "t0": None, "done": None}
                    f.issued.append([tk["lo"], tk["hi"], st["t"], tk])
                    f.n_issued += n
                    out[d]["tasks"][e[1]] = tk
                elif e[0] == "await":
                    tk = out[d]["tasks"].get(e[1])
                    if tk is None or tk["done"] is None:
                        return moved
                    st["t"] = max(st["t"], tk["done"])
                elif e[0] == "rtp":
                    rtp[(e[1], e[2])] = e[3]
                st["ei"] += 1
                moved = True
                continue
            if any(tk["done"] is None for tk in out[d]["tasks"].values()):
                return moved
            end = max([st["t"]] + [tk["done"] for tk in out[d]["tasks"].values()])
            out[d]["t1"] = end
            st["d"], st["ei"] = d + 1, 0
            if d + 1 < len(topos):
                st["t"] = end + (gaps[d] if d < len(gaps) else 0.0)
                out[d + 1]["t0"] = st["t"]
            moved = True
        return moved

    def core_eager() -> bool:
        moved = False
        for c in cores:
            while c.pending is not None and c.pending[0] != "call":
                op = c.pending
                f = fifos[op[1]]
                n = int(round(op[3] - op[2]))
                if op[0] == "prod":
                    f.arr.extend([c.t] * n)
                else:
                    f.rel.setdefault(c.idx, []).extend([c.t] * n)
                c.pending = next(c.gen, None)
                moved = True
        return moved

    def finish(f: Fifo, j: int, t0: float, t1: float):
        cov = f.cover(j)
        if cov is None:
            return
        tk = cov[3]
        if tk["t0"] is None:
            tk["t0"] = t0
        if j + 1 >= cov[1] - EPS:
            tk["done"] = t1

    shim_in = [f for f in fifos.values() if f.shim_in]
    shim_out = [f for f in fifos.values() if f.shim_out]
    while True:
        while seq_eager() | core_eager():
            pass
        if st["d"] >= len(topos):
            break
        best, key = None, None
        for f in shim_in:
            j = len(f.arr)
            if j >= f.n_issued - EPS:
                continue
            cov, room = f.cover(j), f.room_t(j)
            if cov is None or room is None:
                continue
            r = max(cov[2], room)
            if key is None or r < key:
                best, key = (0, f, j), r
        for f in shim_out:
            j = len(f.drain)
            if j >= f.n_issued - EPS or len(f.arr) <= j:
                continue
            cov = f.cover(j)
            if cov is None:
                continue
            r = max(cov[2], f.arr[j])
            if key is None or r < key:
                best, key = (1, f, j), r
        for c in cores:
            op = c.pending
            if op is None:
                continue
            r = c.t
            for fn, lo, hi in op[2]:
                f = fifos[fn]
                need = int(hi + 0.5)
                if f.avail() < need:
                    r = None
                    break
                r = max(r, f.avail_t(need - 1))
            if r is not None and (key is None or r < key):
                best, key = (2, c, None), r
        if best is None:
            d = st["d"]
            stuck = [tk for tk in out[d]["tasks"].values() if tk["done"] is None]
            if not stuck:
                raise TimelineError("an instruction stream awaits a task it never started")
            tf = max([st["t"], ddr] + [c.t for c in cores])
            for f in fifos.values():
                for lo, hi, issue, tk in f.issued:
                    if tk["done"] is not None or tk not in stuck:
                        continue
                    lst, lst0 = (f.arr, f.x0) if f.shim_in else (f.drain, f.d0)
                    while len(lst) < hi - EPS:
                        lst0.append(tf)
                        lst.append(tf)
                        if f.shim_out and len(f.arr) < len(lst):
                            f.arr.append(tf)
                    tk["t0"] = tf if tk["t0"] is None else tk["t0"]
                    tk["done"] = tf
                    out[d]["anomalies"] += 1
            continue
        kind, obj, j = best
        if kind == 2:
            obj.spans.append([obj.idx, obj.pending[1], key, key + CALL_US])
            obj.t = key + CALL_US
            obj.pending = next(obj.gen, None)
        else:
            s0 = max(key, ddr)
            ddr = s0 + obj.eb / bpu
            if kind == 0:
                obj.x0.append(s0)
                obj.arr.append(ddr)
            else:
                obj.d0.append(s0)
                obj.drain.append(ddr)
            finish(obj, j, s0, ddr)
    res = []
    for o in out:
        times = {i: (tk["t0"], tk["done"]) for i, tk in o["tasks"].items()}
        res.append({"t0": o["t0"], "t1": o["t1"], "times": times, "anomalies": o["anomalies"], "spans": []})
    for c in cores:
        for s in c.spans:
            dd = next((x for x in reversed(res) if s[2] >= x["t0"] - EPS), res[0])
            dd["spans"].append(s)
    return {"dispatches": res, "stalled_cores": sum(1 for c in cores if c.pending is not None and c.spans)}


def merge_spans(spans: list[list], gap: float = 20.0) -> list[list]:
    """Join a core's back-to-back calls of the same function into one span: [core, fn, t0, t1, calls]."""
    spans = sorted(spans, key=lambda s: (s[0], s[2]))
    out: list[list] = []
    for s in spans:
        if out and out[-1][0] == s[0] and out[-1][1] == s[1] and s[2] - out[-1][3] <= gap:
            out[-1][3] = max(out[-1][3], s[3])
            out[-1][4] += 1
        else:
            out.append([s[0], s[1], s[2], s[3], 1])
    return out


def _op_bytes(op: dict, chunk: int) -> int | None:
    k = op["op"]
    if k in ("std_perm", "std_fuse", "q8_perm"):
        return op["nch"] * chunk
    if k == "bf16_gemm":
        return op["nch"] * 16384
    if k == "expert_stripes":
        return op["experts"] * 2 * op["stripes"] * op["stripe_bytes"]
    if k == "expert_down":
        return op["experts"] * op["expert_bytes"]
    if k == "put":
        return op.get("cap")
    if k == "transpose_banked":
        return -(-op["rows"] // 32) * op["cols"] * 32 * op["elem"]
    if k == "transpose":
        return op["cols"] * (op.get("dst_rows") or op["rows"]) * op.get("elem", 2)
    if k == "conv_transpose":
        return op["taps"] * op["groups"] * op["width"] * 2
    return None


def regions(ops: list[dict], chunk: int, buf_bytes: int) -> list[tuple[int, int, dict]]:
    """Each pack op's byte range in its buffer; an op of unknown size runs to the next op."""
    srt = sorted(ops, key=lambda o: o["dst"])
    out = []
    for i, op in enumerate(srt):
        nxt = srt[i + 1]["dst"] if i + 1 < len(srt) else buf_bytes
        n = _op_bytes(op, chunk)
        out.append((op["dst"], op["dst"] + n if n else nxt, op))
    return out


def tensor_name(t: str) -> str:
    for p in ("model.layer.{l}.", "model.layers.{l}.", "model.", "layers.{l}."):
        if t.startswith(p):
            t = t[len(p):]
    return t[:-len(".weight")] if t.endswith(".weight") else t


BUFFER_LABELS = {
    "xres": "residual stream", "act": "activation scratch", "consts": "layer constants",
    "ptab": "RoPE position table", "hn": "final-norm output", "logits": "logits", "normw": "final norm weight",
    "zero": "zeros", "xresf": "residual (after the last layer)", "pool": "layer weights",
    "lmpool": "lm_head weights",
}


def task_label(arg: str, off: int, regs: list | None, state_kind: str | None) -> tuple[str, int | None, str]:
    """(label, expert index or None, kind) for a task touching `arg` at byte `off`."""
    if arg == "state":
        return ("KV cache" if state_kind == "kv" else "DeltaNet state", None, "state")
    if regs:
        for lo, hi, op in regs:
            if lo <= off < hi:
                k = op["op"]
                if k == "expert_stripes":
                    per = 2 * op["stripes"] * op["stripe_bytes"]
                    rel = off - lo
                    return (f"experts up|gate stripe {(rel % per) // (2 * op['stripe_bytes'])}", rel // per, "weight")
                if k == "expert_down":
                    return ("experts down", (off - lo) // op["expert_bytes"], "weight")
                return (tensor_name(op.get("tensor") or op.get("up") or k), None, "weight")
    return (BUFFER_LABELS.get(arg, arg), None, "weight" if arg in ("pool", "lmpool") else "act")


class Strings:
    def __init__(self):
        self.items: list[str] = []
        self.idx: dict[str, int] = {}

    def __call__(self, s: str) -> int:
        if s not in self.idx:
            self.idx[s] = len(self.items)
            self.items.append(s)
        return self.idx[s]


def image_key(topo: dict) -> str:
    body = {k: topo[k] for k in ("tiles", "cores", "fifos", "links")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]


def _runs(prog: list) -> list:
    return [s for s in prog if s["op"] == "run"]


def attnpos(topo: dict, kd: dict, layout: dict, pos: int) -> dict:
    """stream_patch::attn_apply on the topology: the KV window fill, new-row drain and record fill at `pos`."""
    kv_row, ptab_row, rb, window = layout.get("kv_row", 2048), layout.get("ptab_row", 1024), kd.get("rb", 1), kd.get("window", 0)
    start = pos + 1 - window if window and pos + 1 > window else 0
    valid = pos - start
    nf = rb * (valid // rb + 1) - 1 if rb > 1 else max(valid, 1)
    out = dict(topo, tasks=[dict(t, bds=[dict(b) for b in t["bds"]]) for t in topo["tasks"]])
    seen = {"window": 0, "row": 0, "record": 0}
    for t in out["tasks"]:
        for b in t["bds"]:
            if b["arg"] == 3 and b["off"] == 0:
                b["len"] = b["ext"] = nf * kv_row
                b["off"] = start * kv_row
                seen["window"] += 1
            elif b["arg"] == 3 and b["off"] == kv_row:
                b["off"] = pos * kv_row
                seen["row"] += 1
            elif b["arg"] == 5:
                b["off"] = pos * ptab_row
                seen["record"] += 1
    if any(n != 1 for n in seen.values()):
        raise TimelineError(f"attnpos: expected one KV window fill, new-row drain and record fill, found {seen}")
    return out


def decode_timeline(m: dict, topos: dict[str, dict], bw_gbps: float = BW_GBPS, seed: int = 1, pos: int = 1) -> dict:
    """One decode step at position `pos`: host stages, every dispatch in manifest order, context switches; us."""
    S = Strings()
    kernels = m["kernels"]
    topos = {k: attnpos(t, kernels[k], m["layout"], pos) if kernels.get(k, {}).get("patch") == "attnpos" else t
             for k, t in topos.items()}
    chunk = m["layout"].get("chunk_bytes", 5120)
    images: dict[str, dict] = {}
    templates: dict[str, dict] = {}

    def template(k: str, lt: str | None, sd: dict, args: list[str], regs: dict, state_kind):
        topo = topos[k]
        tasks = []
        for i, t in enumerate(topo["tasks"]):
            t0, t1 = sd["times"][i]
            bd = t["bds"][0]
            arg = args[bd["arg"]] if bd["arg"] < len(args) else f"arg{bd['arg']}"
            label, expert, kind = task_label(arg, bd["off"], regs.get(arg), state_kind)
            tasks.append([t["fifo"], round(t0 - sd["t0"], 3), round(t1 - sd["t0"], 3), _task_bytes(t),
                          S(arg), S(label), -1 if expert is None else expert, S(kind), t["dir"]])
        experts = sorted({x[6] for x in tasks if x[6] >= 0})
        slot = {e: i for i, e in enumerate(experts)}
        for x in tasks:
            if x[6] >= 0:
                x[6] = slot[x[6]]
        key = image_key(topo)
        images.setdefault(key, topo)
        tkey = f"{k}@{lt}" if lt else k
        templates[tkey] = {
            "kernel": k, "image": key, "dur": round(sd["t1"] - sd["t0"], 3), "tasks": tasks,
            "spans": [[c, S(fn), round(a - sd["t0"], 3), round(b - sd["t0"], 3), n]
                      for c, fn, a, b, n in merge_spans(sd["spans"])],
            "anomalies": sd["anomalies"], "expert_slots": len(experts), "bytes": sum(x[3] for x in tasks)}
        return tkey

    pool_bytes = (m.get("pack") or {}).get("pool_bytes") or m["layout"].get("pool_bytes", 1 << 62)
    lt_tpl: dict[str, list[str]] = {}
    for lt, spec in m["layer_types"].items():
        prog = spec["program"]
        runs = _runs(prog)
        if not runs:
            continue
        gaps = []
        for a, b in zip(runs, runs[1:]):
            between = prog[prog.index(a) + 1:prog.index(b)]
            g = sum(HOST_US["route"] for s in between if s["op"] == "moeroute2")
            if kernels[a["kernel"]]["context"] != kernels[b["kernel"]]["context"]:
                g += CTX_SWITCH_US
            gaps.append(g)
        sim = simulate_pass([topos[s["kernel"]] for s in runs], gaps, bw_gbps)
        pack = spec.get("pack", {})
        regs = {"pool": regions(pack.get("pool", []), chunk, pool_bytes),
                "consts": regions(pack.get("consts", []), chunk, (spec.get("buffers") or {}).get("consts", 1 << 62))}
        state = (spec.get("buffers") or {}).get("state") or {}
        lt_tpl[lt] = [template(s["kernel"], lt, d, s.get("args") or [], regs, state.get("kind"))
                      for s, d in zip(runs, sim["dispatches"])]
    tail_tpl = []
    lm = (m.get("pack") or {}).get("lm_head", {})
    for s in _runs(m.get("tail", [])):
        sim = simulate_pass([topos[s["kernel"]]], [], bw_gbps)
        regs = {"lmpool": regions(lm.get("ops", []), chunk, lm.get("pool_bytes", 1 << 62))}
        tail_tpl.append(template(s["kernel"], None, sim["dispatches"][0], s.get("args") or [], regs, None))

    rng = random.Random(seed)
    n_exp = (m["layout"].get("moe") or {}).get("experts", 0)
    events: list[dict] = []
    clock = {"t": 0.0, "ctx": kernels[_runs(m["tail"])[-1]["kernel"]]["context"] if m.get("tail") else None}

    def host(name, us, layer=None):
        t = clock["t"]
        events.append({"lane": "cpu", "kind": "host", "name": name, "t0": round(t, 3), "t1": round(t + us, 3),
                       **({"layer": layer} if layer is not None else {})})
        clock["t"] = t + us

    def dispatch(k, key, layer=None):
        c = kernels[k]["context"]
        if clock["ctx"] is not None and c != clock["ctx"]:
            t = clock["t"]
            events.append({"lane": "npu", "kind": "ctx", "name": f"{clock['ctx']} -> {c}", "t0": round(t, 3),
                           "t1": round(t + CTX_SWITCH_US, 3)})
            clock["t"] = t + CTX_SWITCH_US
        clock["ctx"] = c
        tp = templates[key]
        t = clock["t"]
        ev = {"lane": "npu", "kind": "dispatch", "name": k, "tpl": key, "t0": round(t, 3), "t1": round(t + tp["dur"], 3)}
        if layer is not None:
            ev["layer"] = layer
        if tp["expert_slots"] and n_exp:
            ev["experts"] = sorted(rng.sample(range(n_exp), min(tp["expert_slots"], n_exp)))
        events.append(ev)
        clock["t"] = t + tp["dur"]

    host("embed", HOST_US["embed"])
    if any(kd.get("patch") == "attnpos" for kd in kernels.values()):
        host("attnpos", HOST_US["attnpos"])
    for layer, lt in enumerate(m["layers"]):
        keys = iter(lt_tpl[lt])
        for s in m["layer_types"][lt]["program"]:
            if s["op"] == "moeroute2":
                host("route", HOST_US["route"], layer)
            elif s["op"] == "run":
                dispatch(s["kernel"], next(keys), layer)
    for s, key in zip(_runs(m.get("tail", [])), tail_tpl):
        dispatch(s["kernel"], key)
    host("readback", HOST_US["readback"])
    host("sample", HOST_US["sample"])

    out_images, fifo_idx = {}, {}
    for key, topo in images.items():
        names = list(topo["fifos"])
        fifo_idx[key] = {n: i for i, n in enumerate(names)}
        out_images[key] = {"tiles": topo["tiles"], "links": topo["links"], "funcs": topo["funcs"],
                           "cores": [{"tile": c["tile"], "funcs": c["funcs"]} for c in topo["cores"]],
                           "fifos": [{"name": n, **topo["fifos"][n]} for n in names]}
    for tp in templates.values():
        fi = fifo_idx[tp["image"]]
        for x in tp["tasks"]:
            x[0] = fi[x[0]]
    return {
        "position": pos,
        "total_us": round(clock["t"], 3),
        "calibration": {"bw_gbps": bw_gbps, "ctx_switch_us": CTX_SWITCH_US, "call_us": CALL_US, "host_us": HOST_US,
                        "source": "DDR bandwidth fitted to OPEN-DECODE-ONE-CONTEXT's 35B step (70.3 ms median at "
                                  "position 1, 2026-09-22); a context switch is that requirement's 27B two-context "
                                  "ax0 penalty; host stage times are nominal except the route gap "
                                  "(OPEN-DECODE-PIPELINE: 0.02-0.07 ms)"},
        "events": events,
        "templates": templates,
        "images": out_images,
        "strings": S.items,
    }

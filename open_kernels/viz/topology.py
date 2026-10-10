"""One build's final.prj -> tiles, data paths, core programs and DMA tasks (VIZ-TOPOLOGY)."""
from __future__ import annotations

import json
import re
from pathlib import Path

VERSION = 1
FOREVER = 2**63 - 1
DTYPE_BYTES = {"i8": 1, "ui8": 1, "si8": 1, "i16": 2, "ui16": 2, "si16": 2, "bf16": 2, "f16": 2,
               "i32": 4, "ui32": 4, "si32": 4, "f32": 4, "i64": 8, "ui64": 8, "f64": 8}


class TopologyError(ValueError):
    pass


def memref_bytes(spec: str) -> tuple[int, int]:
    """'64x32xf32' -> (element count, bytes per element)."""
    *dims, dt = spec.strip().split("x")
    if dt not in DTYPE_BYTES:
        raise TopologyError(f"memref<{spec}>: unknown element type {dt!r}")
    n = 1
    for d in dims:
        n *= int(d)
    return n, DTYPE_BYTES[dt]


_LTILE = re.compile(r"^\s*%(\w+) = aie\.logical_tile<(\w+)>\((\?|\d+), (\?|\d+)\)")
_FIFO = re.compile(r"^\s*aie\.objectfifo @(\w+)\(%(\w+)(?: dimensionsToStream \[[^\]]*\])?, \{([^}]*)\}, "
                   r"\[?(\d+)[^)]*\) : !aie\.objectfifo<memref<([^>]*)>>")
_LINK = re.compile(r"^\s*aie\.objectfifo\.link \[([^\]]*)\] -> \[([^\]]*)\]")
_BUF = re.compile(r"^\s*%(\w+) = aie\.buffer\(%(\w+)\) \{[^}]*sym_name = \"(\w+)\"[^}]*\} : memref<([^>]*)>")
_FUNC = re.compile(r"^\s*func\.func private @(\w+)\(.*?(?:link_with = \"([^\"]+)\")?\}?$")
_CORE = re.compile(r"^\s*%(\w+) = aie\.core\(%(\w+)\)")
_CONST = re.compile(r"^\s*%(\w+) = arith\.constant (-?\d+) : \w+")
_LOAD = re.compile(r"^\s*%(\w+) = memref\.load %(\w+)\[%(\w+)\]")
_CAST = re.compile(r"^\s*%(\w+) = arith\.index_cast %(\w+)")
_SUBI = re.compile(r"^\s*%(\w+) = arith\.subi %(\w+), %(\w+)")
_FOR = re.compile(r"^\s*(?:%[\w:#]+ = )?scf\.for %\w+ = %(\w+) to %(\w+) step %(\w+)")
_FIFO_OP = re.compile(r"aie\.objectfifo\.(acquire|release) @(\w+)\((Consume|Produce), (\d+)\)")
_CALL = re.compile(r"func\.call @(\w+)\(")
_SEQ = re.compile(r"^\s*aie\.runtime_sequence(?: @\w+)?\((.*)\)\s*\{")
_SEQ_ARG = re.compile(r"%(\w+): memref<([^>]*)>")
_TASK = re.compile(r"^\s*%(\w+) = aiex\.dma_configure_task_for @(\w+) \{")
_BD = re.compile(r"aie\.dma_bd\(%(\w+) : memref<([^>]*)>(.*)\)")
_START = re.compile(r"^\s*aiex\.dma_start_task\(%(\w+)\)")
_AWAIT = re.compile(r"^\s*aiex\.dma_await_task\(%(\w+)\)")
_RTPW = re.compile(r"^\s*aiex\.npu\.rtp_write\(@(\w+), (\d+), (%\w+|-?\d+)\)")
_REPEAT = re.compile(r"repeat_count = (\d+)")

_PTILE = re.compile(r"^\s*%(\w+) = aie\.tile\((\d+), (\d+)\)")
_LOC = re.compile(r'^#(loc\d*) = loc\("([^"]*)":(\d+):\d+\)')
_FLOW = re.compile(r"^\s*aie\.flow\(%(\w+), (\w+) : (\d+), %(\w+), (\w+) : (\d+)\)(?: loc\(#(loc\d*)\))?")
_ALLOC = re.compile(r"^\s*aie\.shim_dma_allocation @(\w+?)(?:_shim_alloc)?\(%(\w+), (MM2S|S2MM), (\d+)\)")
_TRAIL_LOC = re.compile(r"loc\(#(loc\d*)\)\s*$")


def _depth(line: str) -> int:
    return line.count("{") - line.count("}")


def _region(lines: list[str], i: int) -> int:
    """Index of the line closing the region opened on line i."""
    d = _depth(lines[i])
    j = i
    while d > 0:
        j += 1
        if j >= len(lines):
            raise TopologyError(f"unclosed region at line {i + 1}")
        d += _depth(lines[j])
    return j


def _expr(env: dict, name: str):
    return env.get(name)


def _core_program(lines: list[str], buffers: dict) -> list:
    """The ops of one core body as a nested list: call / acq / rel / for / forever."""
    env: dict = {}
    root: list = []
    stack = [root]
    for line in lines:
        if m := _CONST.match(line):
            env[m[1]] = int(m[2])
        elif m := _LOAD.match(line):
            env[m[1]] = ["rtp", m[2], env.get(m[3])] if m[2] in buffers else None
        elif m := _CAST.match(line):
            env[m[1]] = env.get(m[2])
        elif m := _SUBI.match(line):
            a, b = env.get(m[2]), env.get(m[3])
            env[m[1]] = a - b if isinstance(a, int) and isinstance(b, int) else (["sub", a, b] if a is not None and b is not None else None)
        elif m := _FOR.match(line):
            lb, ub, st = env.get(m[1]), env.get(m[2]), env.get(m[3])
            body: list = []
            stack[-1].append(["forever", body] if ub == FOREVER else ["for", lb, ub, st, body])
            stack.append(body)
        elif line.strip().startswith("}") and len(stack) > 1:
            stack.pop()
        else:
            for m in _FIFO_OP.finditer(line):
                stack[-1].append(["acq" if m[1] == "acquire" else "rel", m[2], m[3][0], int(m[4])])
            for m in _CALL.finditer(line):
                stack[-1].append(["call", m[1]])
    return root


def parse_aie(text: str) -> dict:
    """The logical design: tiles, fifos, links, buffers, functions, core programs, runtime sequence."""
    lines = text.splitlines()
    out = {"device": None, "ltiles": {}, "fifos": {}, "links": [], "buffers": {}, "funcs": {},
           "cores": [], "args": [], "tasks": {}, "events": []}
    if m := re.search(r"aie\.device\((\w+)\)", text):
        out["device"] = m[1]
    seq_at = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if m := _LTILE.match(line):
            out["ltiles"][m[1]] = {"kind": m[2], "col": None if m[3] == "?" else int(m[3]),
                                   "row": None if m[4] == "?" else int(m[4])}
        elif m := _FIFO.match(line):
            n, b = memref_bytes(m[5])
            out["fifos"][m[1]] = {"line": i + 1, "producer": m[2],
                                  "consumers": re.findall(r"%(\w+)", m[3]), "depth": int(m[4]),
                                  "elem_bytes": n * b}
        elif line.lstrip().startswith("aie.objectfifo @"):
            raise TopologyError(f"aie.mlir:{i + 1}: objectfifo form not understood: {line.strip()[:120]}")
        elif m := _LINK.match(line):
            out["links"].append({"in": re.findall(r"@(\w+)", m[1]), "out": re.findall(r"@(\w+)", m[2])})
        elif m := _BUF.match(line):
            n, b = memref_bytes(m[4])
            out["buffers"][m[3]] = {"var": m[1], "tile": m[2], "bytes": n * b}
        elif m := _FUNC.match(line):
            out["funcs"][m[1]] = m[2]
        elif m := _CORE.match(line):
            j = _region(lines, i)
            out["cores"].append({"var": m[1], "ltile": m[2], "line": i + 1, "body": lines[i + 1:j]})
            i = j
        elif _SEQ.match(line):
            seq_at = i
            break
        i += 1
    rtp_bufs = {k for k in out["buffers"]}
    for c in out["cores"]:
        c["program"] = _core_program(c.pop("body"), rtp_bufs)
    if seq_at is None:
        raise TopologyError("aie.mlir has no aie.runtime_sequence")
    for name, spec in _SEQ_ARG.findall(_SEQ.match(lines[seq_at])[1]):
        n, b = memref_bytes(spec)
        out["args"].append({"name": name, "bytes": n * b, "elem_bytes": b})
    argi = {a["name"]: k for k, a in enumerate(out["args"])}
    env: dict = {}
    tid = {}
    cur = None
    for line in lines[seq_at + 1:]:
        if m := _CONST.match(line):
            env[m[1]] = int(m[2])
        elif m := _TASK.match(line):
            cur = {"fifo": m[2], "bds": [], "repeat": 0}
            tid[m[1]] = cur
        elif cur is not None and (m := _BD.search(line)):
            _, eb = memref_bytes(m[2])
            rest = m[3]
            off = re.search(r"offset = (\d+)", rest)
            ln = re.search(r"len = (\d+)", rest)
            sizes = re.search(r"sizes = \[([^\]]*)\]", rest)
            strides = re.search(r"strides = \[([^\]]*)\]", rest)
            if m[1] not in argi:
                raise TopologyError(f"dma_bd on %{m[1]}, which is not a runtime_sequence argument")
            n = int(ln[1]) if ln else memref_bytes(m[2])[0]
            ext = n
            if sizes and strides:
                sz = [int(x) for x in sizes[1].split(",")]
                stv = [int(x) for x in strides[1].split(",")]
                ext = sum((s - 1) * t for s, t in zip(sz, stv)) + 1
            cur["bds"].append({"arg": argi[m[1]], "off": (int(off[1]) if off else 0) * eb,
                               "len": n * eb, "ext": ext * eb})
        elif cur is not None and line.strip().startswith("}"):
            if r := _REPEAT.search(line):
                cur["repeat"] = int(r[1])
            cur = None
        elif m := _START.match(line):
            t = tid[m[1]]
            if "id" not in t:
                t["id"] = len(out["tasks"])
                out["tasks"][t["id"]] = t
            out["events"].append(["start", t["id"]])
        elif m := _AWAIT.match(line):
            out["events"].append(["await", tid[m[1]]["id"]])
        elif m := _RTPW.match(line):
            v = env.get(m[3][1:]) if m[3].startswith("%") else int(m[3])
            if v is None:
                raise TopologyError(f"rtp_write(@{m[1]}, {m[2]}) of {m[3]}, which is not a constant")
            out["events"].append(["rtp", m[1], int(m[2]), v])
    return out


def parse_placed(text: str) -> dict:
    """The placed design: physical tiles, flows and cores keyed by their aie.mlir lines, shim channels."""
    lines = text.splitlines()
    out = {"tiles": {}, "locs": {}, "flows": [], "cores": [], "allocs": {}}
    for line in lines:
        if m := _LOC.match(line):
            if m[2].endswith("aie.mlir"):
                out["locs"][m[1]] = int(m[3])
    i = 0
    while i < len(lines):
        line = lines[i]
        if m := _PTILE.match(line):
            out["tiles"][m[1]] = (int(m[2]), int(m[3]))
        elif m := _FLOW.match(line):
            out["flows"].append({"src": m[1], "dst": m[4], "line": out["locs"].get(m[7]) if m[7] else None})
        elif m := _CORE.match(line):
            j = _region(lines, i)
            lm = _TRAIL_LOC.search(lines[j])
            out["cores"].append({"tile": m[2], "line": out["locs"].get(lm[1]) if lm else None})
            i = j
        elif m := _ALLOC.match(line):
            out["allocs"][m[1]] = {"tile": m[2], "dir": m[3], "ch": int(m[4])}
        i += 1
    return out


def _kind(row: int) -> str:
    return "shim" if row == 0 else "mem" if row == 1 else "core"


def extract(aie_text: str, placed_text: str) -> dict:
    """Join the logical and placed designs; refuse anything that does not map, naming it."""
    A, P = parse_aie(aie_text), parse_placed(placed_text)
    tiles = P["tiles"]
    phys: dict[str, tuple[int, int]] = {}
    for v, t in A["ltiles"].items():
        if t["col"] is not None and t["row"] is not None:
            phys[v] = (t["col"], t["row"])
    core_at = {c["line"]: c for c in P["cores"] if c["line"] is not None}
    cores = []
    for c in A["cores"]:
        pc = core_at.get(c["line"])
        if pc is not None:
            phys[c["ltile"]] = tiles[pc["tile"]]
        if c["ltile"] not in phys:
            raise TopologyError(f"core %{c['var']} (aie.mlir:{c['line']}) has no placed tile")
        calls = []
        _collect_calls(c["program"], calls)
        cores.append({"tile": list(phys[c["ltile"]]), "funcs": calls, "program": c["program"]})
    by_line: dict[int, list] = {}
    for f in P["flows"]:
        if f["line"] is not None:
            by_line.setdefault(f["line"], []).append(f)
    fifos = {}
    for name, f in A["fifos"].items():
        fl = by_line.get(f["line"], [])
        alloc = P["allocs"].get(name)
        if fl:
            srcs = {tiles[x["src"]] for x in fl}
            if len(srcs) != 1:
                raise TopologyError(f"fifo @{name}: flows leave {len(srcs)} tiles")
            src = srcs.pop()
            dst = sorted({tiles[x["dst"]] for x in fl})
        else:
            src = phys.get(f["producer"])
            dst = [phys.get(c) for c in f["consumers"]]
            if alloc and src is None:
                src = tiles[alloc["tile"]]
            if alloc and None in dst and len(dst) == 1:
                dst = [tiles[alloc["tile"]]]
            if src is None or None in dst:
                raise TopologyError(f"fifo @{name} (aie.mlir:{f['line']}): an endpoint has no placed tile")
            dst = sorted(set(dst))
        fifos[name] = {"src": list(src), "dst": [list(d) for d in dst], "depth": f["depth"],
                       "elem_bytes": f["elem_bytes"]}
        if alloc:
            fifos[name]["shim"] = {"dir": alloc["dir"], "ch": alloc["ch"]}
    links = []
    for lk in A["links"]:
        for n in lk["in"] + lk["out"]:
            if n not in fifos:
                raise TopologyError(f"objectfifo.link names @{n}, which is not declared")
        links.append({"in": lk["in"], "out": lk["out"], "tile": fifos[lk["in"][0]]["dst"][0]})
    tasks = []
    for t in sorted(A["tasks"].values(), key=lambda t: t["id"]):
        f = fifos.get(t["fifo"])
        if f is None:
            raise TopologyError(f"DMA task on @{t['fifo']}, which is not a declared fifo")
        if f["src"][1] != 0 and all(d[1] != 0 for d in f["dst"]):
            raise TopologyError(f"DMA task on @{t['fifo']}, which has no shim end")
        if not t["bds"]:
            raise TopologyError(f"DMA task {t['id']} on @{t['fifo']} has no buffer descriptor")
        tasks.append({"fifo": t["fifo"], "bds": t["bds"], "repeat": t["repeat"],
                      "dir": "in" if f["src"][1] == 0 else "out"})
    if not tasks:
        raise TopologyError("the runtime sequence starts no DMA task")
    used = sorted({tuple(c["tile"]) for c in cores}
                  | {tuple(f["src"]) for f in fifos.values()}
                  | {tuple(d) for f in fifos.values() for d in f["dst"]})
    rtp = {k: {"tile": list(phys[b["tile"]]), "bytes": b["bytes"]}
           for k, b in A["buffers"].items() if b["tile"] in phys}
    return {
        "version": VERSION,
        "device": A["device"],
        "tiles": [[c, r, _kind(r)] for c, r in used],
        "cores": cores,
        "fifos": fifos,
        "links": links,
        "buffers": rtp,
        "funcs": {k: v for k, v in A["funcs"].items()},
        "args": [{"bytes": a["bytes"], "elem_bytes": a["elem_bytes"]} for a in A["args"]],
        "tasks": tasks,
        "events": A["events"],
    }


def _collect_calls(prog: list, out: list) -> None:
    for op in prog:
        if op[0] == "call" and op[1] not in out:
            out.append(op[1])
        elif op[0] == "for":
            _collect_calls(op[4], out)
        elif op[0] == "forever":
            _collect_calls(op[1], out)


def extract_prj(prj: Path) -> dict:
    prj = Path(prj)
    a, p = prj / "aie.mlir", prj / "input_with_addresses.mlir"
    for f in (a, p):
        if not f.is_file():
            raise TopologyError(f"{f} missing")
    return extract(a.read_text(encoding="utf-8"), p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import sys
    t = extract_prj(Path(sys.argv[1]))
    print(json.dumps({"tiles": len(t["tiles"]), "cores": len(t["cores"]), "fifos": len(t["fifos"]),
                      "links": len(t["links"]), "tasks": len(t["tasks"])}))

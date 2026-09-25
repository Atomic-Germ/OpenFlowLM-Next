"""Build a config that dumps the NPU's on-device router output for every layer and token.

The reference driver (model/make_decode.py) looks for `y_rout{layer}{sfx}.bin` in its output
directory and, when present, adopts the NPU's own top-8 instead of its own - printing every
case where the two differ.  So this dump is both the comparison and the way to make the
comparison fair.

The A_ROUT region holds the 256 router logits first (float index 0..255), then the chosen
top-8 as int32 at float index 256 (byte 1024) and their normalised weights after it, so a
4096-byte dump carries logits + choice + weights.
"""
import re
import sys
from pathlib import Path

A_ROUT, AA_ROUT = 176128, 83968            # lax layout: linear stream / attention stream
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/stage13/run_stage13_b.cfg")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/route16/run_route.cfg")
OUT = DST.parent
OUT.mkdir(parents=True, exist_ok=True)

lines = SRC.read_text().splitlines()
res, suffix, pending = [], "", []
tok = 0
for ln in lines:
    res.append(ln)
    m = re.match(r"run (?:lx|ax)f0 pool(\d+)", ln)
    if m:
        L = int(m.group(1))
        pending.append(L)
        continue
    if ln.startswith("dump logits"):
        f = Path(ln.split()[2])
        suffix = "" if f.name == "y_logits.bin" else f.name[:-4].replace("y_logits", "")
        # a token's act BOs are reused by the next token, so dump before moving on
        for L in pending:
            off = AA_ROUT if L % 4 == 3 else A_ROUT
            res.append(f"dump act{L} {(OUT / f'y_rout{L}{suffix}.bin').as_posix()} 4096 {off}")
        pending = []
        tok += 1
if pending:                                # no logits dump (e.g. a layer-limited cfg)
    for L in pending:
        off = AA_ROUT if L % 4 == 3 else A_ROUT
        res.append(f"dump act{L} {(OUT / f'y_rout{L}{suffix}.bin').as_posix()} 4096 {off}")
DST.write_text("\n".join(res) + "\n")
print(f"{DST}: {len(lines)} -> {len(res)} lines, {tok} token blocks, "
      f"{len(re.findall(r'^dump act', chr(10).join(res), re.M))} routing dumps")

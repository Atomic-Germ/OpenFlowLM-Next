"""Bundles per basic block of one function in a Peano .s file (loops show as the blocks between a 'Loop Header' and its
back edge): python bb_bundles.py <file.s> [function-prefix, default _Z14attn_rowb_impl]."""
import re, sys
from pathlib import Path

lines = Path(sys.argv[1]).read_text().splitlines()
pre = sys.argv[2] if len(sys.argv) > 2 else "_Z14attn_rowb_impl"
st = [i for i, ln in enumerate(lines) if ln.startswith(pre)][-1]
en = next(i for i in range(st, len(lines)) if lines[i].startswith(".Lfunc_end"))
cur, cnt, loop, out = "entry", 0, "", []
for ln in lines[st + 1:en]:
    m = re.match(r"^(\.LBB\S+):", ln)
    if m:
        out.append((cur, cnt, loop))
        cur, cnt, loop = m.group(1), 0, ""
    elif "Loop Header" in ln:
        loop = "LOOP " + ln.split("Depth=")[-1].strip()
    elif ln.strip() and not ln.lstrip().startswith(("//", ".")):
        cnt += 1
out.append((cur, cnt, loop))
for b, c, l in out:
    print(f"{b:12s} {c:5d} {l}")
print("total", sum(c for _, c, _ in out))

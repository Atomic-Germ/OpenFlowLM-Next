r"""Move the per-chunk outputs of a run (<run>/c00/p0_logits.f32 ...) into <run>/all with global prompt numbers.

    python merge_chunks.py <run dir> [--chunk 15]
"""
import os
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
size = int(sys.argv[sys.argv.index("--chunk") + 1]) if "--chunk" in sys.argv else 15
(run / "all").mkdir(exist_ok=True)
for c in sorted(run.glob("c[0-9][0-9]")):
    base = int(c.name[1:]) * size
    for f in c.glob("p*_*"):
        m = re.match(r"p(\d+)_(.*)", f.name)
        os.replace(f, run / "all" / f"p{base + int(m.group(1))}_{m.group(2)}")
print("merged", len(list((run / "all").glob("p*_logits.f32"))), "prompts in", run / "all")

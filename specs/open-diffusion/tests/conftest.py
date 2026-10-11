import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "open_kernels",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

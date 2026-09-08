import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OPEN_KERNELS = REPO / "open_kernels"
for p in (OPEN_KERNELS, OPEN_KERNELS / "model", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

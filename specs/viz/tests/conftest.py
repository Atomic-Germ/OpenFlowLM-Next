import gzip
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
FIX = Path(__file__).resolve().parent / "fixtures"
if str(REPO / "open_kernels") not in sys.path:
    sys.path.insert(0, str(REPO / "open_kernels"))


def gz_text(p: Path) -> str:
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return f.read()


def mlir_pair(name: str) -> tuple[str, str]:
    d = FIX / "mlir" / name
    return gz_text(d / "aie.mlir.gz"), gz_text(d / "input_with_addresses.mlir.gz")


def kernel_set(tag: str) -> tuple[dict, dict]:
    m = json.loads(gz_text(FIX / tag / "manifest.json.gz"))
    topos = {p.name[:-len(".json.gz")]: json.loads(gz_text(p)) for p in (FIX / tag / "topo").glob("*.json.gz")}
    return m, topos


@pytest.fixture(scope="session")
def two_context():
    return kernel_set("two")


@pytest.fixture(scope="session")
def one_context():
    return kernel_set("one")

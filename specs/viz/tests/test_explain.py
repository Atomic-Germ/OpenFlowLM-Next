# Traces: VIZ-EXPLAIN-COVERAGE
import copy

import pytest
from conftest import REPO

from viz import explain as X
from viz.export import assemble


@pytest.fixture(scope="module")
def entries():
    return X.load()


@pytest.mark.parametrize("which", ["two_context", "one_context"])
def test_every_key_the_page_can_ask_for_has_an_explainer(which, entries, request):
    m, topos = request.getfixturevalue(which)
    assert X.missing(assemble(m, topos), entries) == []


def test_a_function_without_an_explainer_is_reported(two_context, entries):
    m, topos = copy.deepcopy(two_context)
    topos["ln"]["cores"][0]["funcs"].append("ln_nosuch")
    viz = assemble(m, topos)
    assert X.missing(viz, entries) == ["core:ln_nosuch"]


def test_every_cited_requirement_exists(entries):
    assert X.unknown_specs(entries, X.spec_ids(REPO / "specs")) == []
    bad = X.parse("## core:x\nTitle: x\nSpec: OPEN-MANIFEST, OPEN-NO-SUCH-ID\n\nbody")
    assert X.unknown_specs(bad, X.spec_ids(REPO / "specs")) == [("core:x", "OPEN-NO-SUCH-ID")]


def test_glob_keys_and_section_fields():
    e = X.parse("## fifo:w*\nTitle: Weights\nSummary: one line\nSource: a/b.cc\n\nPara one.\n\n- item\n")
    assert X.find(e, "fifo:w3")["title"] == "Weights" and X.find(e, "fifo:x") is None
    assert X.find(e, "fifo:w3")["body"] == "Para one.\n\n- item"

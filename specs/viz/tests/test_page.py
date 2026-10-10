# Traces: VIZ-PAGE-FILL
import json
import re

from viz.page import SLOTS, render


def blob(html: str, sid: str) -> str:
    m = re.search(rf'<script id="{sid}" type="[^"]+">(.*?)</script>', html, re.S)
    return m[1].replace("<\\/", "</")


def test_each_slot_filled_once_and_no_payload_closes_its_script():
    data = json.dumps({"note": "</script><script>alert(1)</script> /*VIZ_EXPLAIN*/"})
    html = render(data, "## core:x\nTitle: </b>", json.dumps({"tag": "a:b"}))
    assert "</script><script>alert" not in html
    assert json.loads(blob(html, "viz-data"))["note"].endswith("/*VIZ_EXPLAIN*/")
    assert blob(html, "viz-explain") == "## core:x\nTitle: </b>"
    assert json.loads(blob(html, "viz-model")) == {"tag": "a:b"}
    assert not SLOTS.search(html.replace(data.replace("</", "<\\/"), ""))
    assert "@font-face" in html

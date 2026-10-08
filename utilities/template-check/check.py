"""Diff a model's chat template as the app's vendored minja renders it against transformers; exit 1 on any difference."""
import argparse
import difflib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXE = HERE / "render.exe"

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the weather.\nCity names in English.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "days": {"type": "integer"},
        "metric": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string"}}},
        "required": ["city"]}}}
REFS = {"type": "function", "function": {
    "name": "book", "description": "Book a room.",
    "parameters": {"type": "object", "$defs": {"Room": {"type": "object", "properties": {
        "kind": {"type": "string", "enum": ["single", "double"]}}}},
        "properties": {"room": {"$ref": "#/$defs/Room"}, "nights": {"type": "integer", "default": 1}},
        "required": ["room"]}}}

USER = {"role": "user", "content": "Explain what an NPU is in two sentences."}
CALL = {"role": "assistant", "content": "", "reasoning_content": "Need the weather.",
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 2, "tags": ["a", "b"]}}}]}
RESULT = {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 21}"}

DEFAULT_CASES = [
    {"messages": [USER]},
    {"messages": [{"role": "system", "content": "Be terse."}, USER]},
    {"messages": [USER, {"role": "assistant", "content": "An NPU is a chip.", "reasoning_content": ""},
                  {"role": "user", "content": "And a GPU?"}]},
    {"messages": [USER, {"role": "assistant", "content": "An NPU is a chip.", "reasoning_content": "Short answer."},
                  {"role": "user", "content": "And a GPU?"}]},
    {"messages": [USER], "tools": [WEATHER]},
    {"messages": [{"role": "system", "content": "Be terse."}, USER], "tools": [WEATHER]},
    {"messages": [{"role": "user", "content": "Weather in Paris?"}, CALL, RESULT], "tools": [WEATHER]},
    {"messages": [{"role": "user", "content": "Book a double."}], "tools": [REFS]},
    {"messages": [USER], "extra_context": {"reasoning_effort": "medium"}},
    {"messages": [USER], "add_generation_prompt": False},
]


def build(rebuild=False):
    if EXE.exists() and not rebuild:
        return
    subprocess.run(["cmd.exe", "/c", str(HERE / "build.cmd")], check=True)


def load_template(model_dir: Path):
    cfg = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    jinja = model_dir / "chat_template.jinja"
    tmpl = jinja.read_text(encoding="utf-8") if jinja.exists() else cfg["chat_template"]
    tok = lambda v: v if isinstance(v, str) or v is None else v.get("content")
    return tmpl, tok(cfg.get("bos_token")) or "", tok(cfg.get("eos_token")) or ""


def minja_render(template_path: Path, bos, eos, cases, polyfills):
    lines = "\n".join(json.dumps({**c, "polyfills": polyfills}) for c in cases) + "\n"
    p = subprocess.run([str(EXE), str(template_path), bos, eos], input=lines.encode("utf-8"),
                       capture_output=True, check=False)
    sys.stderr.write(p.stderr.decode("utf-8", "replace"))
    return [json.loads(l) for l in p.stdout.decode("utf-8").splitlines() if l.strip()]


def hf_render(tokenizer, case):
    try:
        return {"text": tokenizer.apply_chat_template(
            case["messages"], tools=case.get("tools"), tokenize=False,
            add_generation_prompt=case.get("add_generation_prompt", True), **case.get("extra_context", {}))}
    except Exception as e:  # noqa: BLE001 -- the template's own raise_exception lands here
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--cases", type=Path, help="extra JSONL cases, appended to the defaults")
    ap.add_argument("--polyfills", action="store_true", help="render as minja's default apply() does")
    ap.add_argument("--rebuild", action="store_true", help="rebuild render.exe (after editing minja)")
    a = ap.parse_args()
    build(a.rebuild)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(a.model_dir))
    tmpl, bos, eos = load_template(a.model_dir)
    tmpl_path = HERE / "_template.jinja"
    tmpl_path.write_text(tmpl, encoding="utf-8", newline="")
    cases = list(DEFAULT_CASES)
    if a.cases:
        cases += [json.loads(l) for l in a.cases.read_text(encoding="utf-8").splitlines() if l.strip()]
    got = minja_render(tmpl_path, bos, eos, cases, a.polyfills)
    if len(got) == 1 and "error" in got[0] and got[0]["error"].startswith("parse:"):
        print("minja cannot parse the template:", got[0]["error"][:2000])
        return 1
    bad = 0
    for i, (case, m) in enumerate(zip(cases, got)):
        h = hf_render(tokenizer, case)
        if "error" in h and "error" in m:
            print(f"case {i}: both raise ({h['error'][:80]})")
            continue
        if m.get("text") == h.get("text"):
            print(f"case {i}: identical ({len(h['text'])} chars)")
            continue
        bad += 1
        print(f"case {i}: DIFFERS")
        if "error" in m or "error" in h:
            print("  minja:", m.get("error", "ok")[:1500])
            print("  transformers:", h.get("error", "ok")[:300])
            continue
        diff = difflib.unified_diff(h["text"].splitlines(True), m["text"].splitlines(True),
                                    "transformers", "minja")
        sys.stdout.writelines(list(diff)[:60])
    print(f"{len(cases) - bad}/{len(cases)} cases agree")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Hold every published bench-embed number to the binary's own output.

SPDX-License-Identifier: MIT

WHY THIS EXISTS. The first version of docs/docs/benchmarks/embeddings_results.md
was transcribed by hand from the CSVs, and five of 161 cells were wrong in the
last digit -- 310.0 where the binary printed 309.9, 157.7 for 157.6, and three
more. Nobody reading the page could ever have caught that, and no test did,
because a benchmark page is prose as far as the build is concerned.

It also catches the more dangerous version of the same thing: a page that is
internally consistent but describes a DIFFERENT RUN. Re-measure after a code
change, update four numbers in the table and forget the fifth, and the page is
a mixture of two experiments with nothing to show it.

The authority is what the binary PRINTED, not the CSV. Those disagree at the
last digit -- the CSV stores 2 decimals and the table rounds to 1, and the two
round 157.65 differently -- so re-rounding the CSV produces false positives.
Whatever a user sees on their terminal is what the page must say.

USAGE
    python utilities/check_embed_bench_docs.py <sweep.log> [more.log ...]

where each log is the captured stdout of one or more `oflm bench-embed` runs.
Exits non-zero on the first disagreement, listing every one.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NL_JOIN = ""   # every log line, stripped and "|"-joined, for verbatim checks

# The printed table row:
#   batch | batched +- sd | looped | speedupx | texts +- sd | tokens[ +- sd]
ROW = re.compile(
    r"^\s*(\d+) \|\s+([\d.]+) \+- \s*([\d.]+) \|\s+([\d.]+) \|\s+([\d.]+)x \|"
    r"\s+([\d.]+) \+- \s*([\d.]+) \|\s+(not reported|[\d.]+)")

# "  <tag>, task <name> ..." -- the footer line that says which model the table
# above belongs to. Parsing the model from the footer rather than from a
# surrounding banner means a log with no section markers still works.
FOOTER = re.compile(r"^  ([A-Za-z0-9.:_-]+), task ")

# tag in the log -> the display name the docs use
DISPLAY = {
    "all-minilm:l6-v2": "all-MiniLM-L6-v2",
    "bge-small:en-v1.5": "bge-small-en-v1.5",
    "bge-base:en-v1.5": "bge-base-en-v1.5",
    "bge-large:en-v1.5": "bge-large-en-v1.5",
    "nomic-embed-text:v1.5": "nomic-embed-text-v1.5",
    "gte-multilingual:base": "gte-multilingual-base",
    "embed-gemma:300m": "EmbeddingGemma-300M",
}

# Which metric each documented table holds, and which batch sizes its columns
# are, in order. `skip` drops leading non-metric columns (the dims column).
# Which models each table MUST list. Without this the checker passed on a
# gutted table: doc_rows() returns only the rows it recognises and zip()
# truncates to the shorter sequence, so deleting a model row or a trailing
# column still printed "0 problems". A checker whose own coverage is unchecked
# is worse than none, because its answer still looks like a verification.
ALL_SEVEN = ["all-MiniLM-L6-v2", "bge-small-en-v1.5", "bge-base-en-v1.5",
      "nomic-embed-text-v1.5", "gte-multilingual-base", "bge-large-en-v1.5",
      "EmbeddingGemma-300M"]
ADAPTER_FOUR = ["all-MiniLM-L6-v2", "bge-base-en-v1.5", "bge-large-en-v1.5", "EmbeddingGemma-300M"]

TABLES = [
    # (file, heading substring, metric, batches, leading cols to skip, models)
    ("docs/docs/benchmarks/embeddings_results.md",
     "Throughput (texts per second", "texts", [1, 4, 8, 16, 32, 64, 128], 1, ALL_SEVEN),
    ("docs/docs/benchmarks/embeddings_results.md",
     "Latency (seconds", "lat", [1, 4, 16, 32, 64, 128], 0, ALL_SEVEN),
    ("docs/docs/benchmarks/embeddings_results.md",
     "Batch speedup", "sp", [1, 2, 4, 8, 16, 32, 64, 128], 0, ALL_SEVEN),
    ("docs/docs/benchmarks/embeddings_results.md",
     "Token throughput", "tok", [16, 32, 64, 128], 0, ALL_SEVEN),
    ("src/open_npue_adapter/README.md",
     "is now measured, on every model", "sp", [1, 4, 8, 16, 32, 64, 128], 0, ADAPTER_FOUR),
]

# The summary table restates three DERIVED values per model: the peak texts/s,
# the batch it occurs at, and the single-text latency in milliseconds. Derived
# numbers are where a page drifts first, because re-measuring updates the big
# tables and leaves the summary behind.
ONE_NUMBER = ("docs/docs/benchmarks/embeddings_results.md",
              "| **Model** | **embeddings/s (peak)** |")


def parse_truth(paths):
    """{(display_name, batch): {metric: printed_string}} from the logs."""
    truth = {}
    for path in paths:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        pending = []
        for line in text.splitlines():
            m = ROW.match(line)
            if m:
                pending.append(m)
                continue
            f = FOOTER.match(line)
            if f and pending:
                tag = f.group(1)
                name = DISPLAY.get(tag)
                if name is None:
                    print("note: log has a table for unknown tag %r" % tag)
                    pending = []
                    continue
                for r in pending:
                    truth[(name, int(r.group(1)))] = {
                        "lat": r.group(2),
                        "loop": r.group(4),
                        "sp": r.group(5),
                        "texts": r.group(6),
                        "tok": r.group(8),
                    }
                pending = []
    return truth


def doc_rows(doc_text, heading):
    """(rows, raw_names) for the first markdown table after `heading`.

    `raw_names` is EVERY data row's first cell, in order, including ones this
    checker does not recognise. The first version returned only recognised
    rows, which threw away exactly the evidence the caller's "unexpected row"
    check needs -- so that check could never fire -- and let two rows for one
    model overwrite each other silently.
    """
    try:
        seg = doc_text[doc_text.index(heading):]
    except ValueError:
        return None, None
    rows = {}
    raw_names = []
    # A markdown table is a header line, an alignment line, then data. Count
    # the lines rather than pattern-matching the header text: the first
    # version skipped a first cell of "Model", and the adapter README heads
    # its table with "batch" -- which the new duplicate/extra check then
    # reported as an unexpected model row. It was right to.
    seen = 0
    for line in seg.splitlines():
        if not line.startswith("|"):
            if seen:
                break
            continue
        seen += 1
        cells = [c.strip() for c in line.strip("|").split("|")]
        if seen == 1:
            continue                      # header
        if seen == 2:
            if not all(set(c) <= set("-: ") and c for c in cells):
                return None, None         # not a table after all
            continue                      # alignment row
        name = cells[0].replace("*", "").replace("(control)", "").strip()
        raw_names.append(name)
        if name in DISPLAY.values():
            rows[name] = cells[1:]
    return rows, raw_names


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    global NL_JOIN
    NL_JOIN = "".join(Path(a).read_text(encoding="utf-8", errors="replace")
                      for a in argv[1:])
    NL_JOIN = "".join(l.strip() + "|" for l in NL_JOIN.splitlines())
    truth = parse_truth(argv[1:])
    if not truth:
        print("FATAL no printed bench-embed tables found in %s" % ", ".join(argv[1:]))
        return 2
    print("parsed %d printed rows from %d log(s)" % (len(truth), len(argv) - 1))

    compared = 0
    bad = []
    missing_tables = []
    for rel, heading, metric, batches, skip, want_models in TABLES:
        path = REPO / rel
        if not path.exists():
            missing_tables.append("%s (file missing)" % rel)
            continue
        rows, raw_names = doc_rows(path.read_text(encoding="utf-8"), heading)
        if rows is None:
            missing_tables.append("%s :: %s (heading missing)" % (rel, heading))
            continue
        if not rows:
            missing_tables.append("%s :: %s (no model rows)" % (rel, heading))
            continue
        # every expected model present, nothing else, and nothing twice --
        # checked against raw_names, because the recognised-rows dict cannot
        # show an extra row or a duplicate
        for want in want_models:
            if want not in rows:
                bad.append("%-46s %-22s ROW MISSING from %s"
                           % (rel, want, heading))
        for got in raw_names:
            if got not in want_models:
                bad.append("%-46s %-22s UNEXPECTED row in %s"
                           % (rel, got, heading))
        for got in set(raw_names):
            if raw_names.count(got) > 1:
                bad.append("%-46s %-22s appears %d times in %s"
                           % (rel, got, raw_names.count(got), heading))
        for name, cells in rows.items():
            metric_cells = cells[skip:]
            # exact width, so a deleted trailing column cannot be truncated away
            if len(metric_cells) != len(batches):
                bad.append("%-46s %-22s has %d metric cells, want %d, in %s"
                           % (rel, name, len(metric_cells), len(batches), heading))
                continue
            for batch, cell in zip(batches, metric_cells):
                cell = cell.replace("*", "").strip()
                want = truth.get((name, batch), {}).get(metric)
                if cell in ("—", "-", "--", ""):
                    # A dash is legitimate ONLY where the sweep has no such
                    # stage -- EmbeddingGemma stops at 16. Accepting it
                    # unconditionally meant replacing any measured value with
                    # an em dash passed.
                    if want is not None:
                        bad.append("%-46s %-22s b=%-4d %-6s doc=%-9s printed=%s"
                                   % (rel, name, batch, metric, "(dash)", want))
                    continue
                if want is None:
                    # The docs quote a point the logs do not cover. That is not
                    # automatically wrong -- embed-gemma is swept to 16 only --
                    # but it must not be silently accepted either.
                    bad.append("%-46s %-22s b=%-4d %-6s doc=%-9s printed=(NOT IN LOG)"
                               % (rel, name, batch, metric, cell))
                    continue
                compared += 1
                if cell != want:
                    bad.append("%-46s %-22s b=%-4d %-6s doc=%-9s printed=%s"
                               % (rel, name, batch, metric, cell, want))

    # --- the derived summary table ------------------------------------------
    rel, heading = ONE_NUMBER
    path = REPO / rel
    if path.exists():
        rows, _raw = doc_rows(path.read_text(encoding="utf-8"), heading)
        if not rows:
            missing_tables.append("%s :: %s (summary table not found)" % (rel, heading))
        for want in ALL_SEVEN:
            if want not in (rows or {}):
                bad.append("%-46s %-22s ROW MISSING from the summary table"
                           % (rel, want))
        for name, cells in (rows or {}).items():
            # exact, like the metric tables. `< 3` let an EXTRA cell through,
            # which is the same drift the width check exists to stop.
            if len(cells) != 3:
                bad.append("%-46s %-22s summary row has %d cells, want exactly 3"
                           % (rel, name, len(cells)))
                continue
            peak_doc = cells[0].replace("*", "").strip()
            at_doc = cells[1].replace("*", "").strip()
            ms_doc = cells[2].replace("*", "").replace("ms", "").strip()
            # recompute the peak from the log rather than trusting the label
            best_b, best_v = None, -1.0
            for (n, b), v in truth.items():
                if n == name and v["texts"] != "not reported":
                    if float(v["texts"]) > best_v:
                        best_v, best_b = float(v["texts"]), b
            if best_b is None:
                bad.append("%-46s %-22s summary: no texts/s in the log" % (rel, name))
                continue
            compared += 3
            if peak_doc != truth[(name, best_b)]["texts"]:
                bad.append("%-46s %-22s summary peak doc=%-9s printed=%s"
                           % (rel, name, peak_doc, truth[(name, best_b)]["texts"]))
            tied = sorted(b for (n, b), v in truth.items()
                          if n == name and v["texts"] != "not reported"
                          and float(v["texts"]) == best_v)
            if at_doc not in [str(b) for b in tied]:
                bad.append("%-46s %-22s summary peak-batch doc=%-9s achieved at %s"
                           % (rel, name, at_doc,
                              ", ".join(str(b) for b in tied)))
            lat1 = truth.get((name, 1), {}).get("lat")
            if lat1 is None:
                bad.append("%-46s %-22s summary: no batch-1 latency in the log"
                           % (rel, name))
            else:
                want_ms = "%.1f" % (float(lat1) * 1000.0)
                if ms_doc != want_ms:
                    bad.append("%-46s %-22s summary one-text doc=%-9s from %ss = %s ms"
                               % (rel, name, ms_doc, lat1, want_ms))
    else:
        missing_tables.append("%s (file missing)" % rel)

    # --- verbatim sample blocks ---------------------------------------------
    # cli.md pastes real printed rows as an example. An example that no longer
    # matches any run is the same defect as a wrong table, with less excuse.
    cli = REPO / "docs/docs/instructions/cli.md"
    if cli.exists():
        logs = NL_JOIN
        for line in cli.read_text(encoding="utf-8").splitlines():
            st = line.strip()
            if re.match(r"^\d+ \|\s+[\d.]+ \+- ", st):
                compared += 1
                if st not in logs:
                    bad.append("docs/docs/instructions/cli.md  sample row not in any log: %s"
                               % st)

    for m in missing_tables:
        print("FAIL table not found: %s" % m)
    for b in bad:
        print("FAIL %s" % b)
    print("\ncompared %d documented cells against the binary output, %d problem(s)"
          % (compared, len(bad) + len(missing_tables)))
    return 1 if (bad or missing_tables) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

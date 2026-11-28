#!/usr/bin/env python3
"""Lint Hugo Markdown documentation for front matter, formatting, and prose quality.

SPDX-License-Identifier: MIT

Scans Markdown (.md) files under docs/content/ and docs/docs/ (or explicit
paths passed via --paths / positional arguments) and exits non-zero if any
violation is found:

  1. Missing YAML front matter (---) or missing `title:` / `description:`.
  2. Raw backticks (`) inside YAML `title:` or `description:`.
  3. Decorative emoji characters in Markdown headings (^#{1,6}\\s+.*).
  4. Unspaced or untagged opening fenced code blocks, or closing fenced code
     blocks not followed by a blank line or EOF (with stateful fence tracking).
  5. Em-dashes (U+2014, '—') in prose outside code fences and inline code spans.
  6. Banned repo-internal jargon and AI-slop phrases in prose outside code
     fences and inline code spans.
  7. Leftover 'smolvla' references or obsolete internal URLs (/docs/install_win,
     /docs/install_lin, /docs/install/nix/, /benchmarks/community/).
  8. Inline-code backtick clutter around capitalized prose model family names
     outside fenced code blocks.
  9. `<div class="oflm-spec-grid">` in `docs/docs/` outside `docs/docs/models/*.md`
     (ignoring fenced code blocks in `docs/docs/contributing/doc-contributions.md`).
 10. `"Practical Limits & WIP"` anywhere in `docs/`.
 11. `Custom` (`oflm add`) rows appearing after non-`Custom` shortcut rows in
     `## Variants & Built-in Shortcuts` tables in `docs/docs/models/*.md`.
 12. Unknown or fabricated GitHub issue/PR references (`[unknown-issue-ref]`) in
     `docs/` (`.md`, `.json`, `.html`, `.yaml`) via `issue="..."`, `"issue": "..."`,
     or `/issues/<N>` (must be `<= 245` and in the known valid issue/PR set of
     `Atomic-Germ/OpenFlowLM-Next`).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATHS = [
    REPO_ROOT / "docs" / "content",
    REPO_ROOT / "docs" / "docs",
]
DEFAULT_ISSUE_SCAN_PATHS = [
    REPO_ROOT / "docs" / "content",
    REPO_ROOT / "docs" / "docs",
    REPO_ROOT / "docs" / "data",
    REPO_ROOT / "docs" / "layouts",
]

DELETED_ISSUE_NUMBERS: frozenset[int] = frozenset({46, 175, 182, 222})
ISSUE_SCAN_SUFFIXES = (".md", ".json", ".html", ".yaml", ".yml")
OWN_GITHUB_REPOS = ("atomic-germ/openflowlm-next", "atomic-germ/openflowlm")
ISSUE_ATTR_RE = re.compile(r"""\bissue\s*=\s*["']([^"']*?\d[^"']*?)["']""")
ISSUE_JSON_YAML_RE = re.compile(
    r"""(?:["']issue["']|^\s*-?\s*issue)\s*:\s*(?:["']([^"']*?\d[^"']*?)["']|#?(\d+)\b)"""
)
ISSUE_PATH_RE = re.compile(
    r"""(?:https?://([^/\s"'<>()]+)((?:/[^/\s"'<>()]+)*?))?/issues/(\d+)\b""",
    re.IGNORECASE,
)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE_OPEN_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
INLINE_CODE_RE = re.compile(r"(`+)([^`]*?)\1")
HUGO_SHORTCODE_RE = re.compile(r"\{\{[<%].*?[>%]\}\}")
SPEC_GRID_DIV_RE = re.compile(r"""<div\s+class=["'][^"']*\boflm-spec-grid\b""", re.IGNORECASE)
PRACTICAL_LIMITS_WIP_RE = re.compile(r"Practical\s+Limits\s*(?:&|&amp;|and)\s*WIP", re.IGNORECASE)
VARIANTS_SHORTCUTS_HEADING_RE = re.compile(
    r"^Variants\s*(?:&|&amp;|and)\s*Built-in\s+Shortcuts\b", re.IGNORECASE
)
TABLE_SEP_RE = re.compile(r"^\|[\s:|=-]+\|?\s*$")
CUSTOM_WORD_RE = re.compile(r"\bcustom\b", re.IGNORECASE)
OFLM_ADD_RE = re.compile(r"\boflm\s+add\b", re.IGNORECASE)

# Decorative emoji ranges (excludes ASCII, standard arrows, and math symbols).
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # Misc Symbols & Pictographs, Emoticons, Transport, Supplemental
    "\U0001F1E6-\U0001F1FF"  # Regional indicator symbols
    "\U00002600-\U000027BF"  # Misc Symbols & Dingbats
    "\U0000231A-\U0000231B"  # Watch / Hourglass
    "\U000023E9-\U000023FA"  # Media control symbols
    "\U000025AA-\U000025AB"
    "\U000025B6\U000025C0"
    "\U000025FB-\U000025FE"
    "\U00002B50\U00002B55"
    "\U0000FE0F"             # Variation Selector-16
    "]"
)

BANNED_PHRASES: list[tuple[str, re.Pattern[str]]] = [
    ("plumbing", re.compile(r"\bplumbing\b", re.IGNORECASE)),
    ("escape hatch", re.compile(r"\bescape\s+hatch(?:es)?\b", re.IGNORECASE)),
    ("fail-closed", re.compile(r"\bfail[- ]closed\b", re.IGNORECASE)),
    ("CLI harness", re.compile(r"\bcli\s+harness(?:es)?\b", re.IGNORECASE)),
    ("delve", re.compile(r"\bdelve(?:s|d|ing)?\b", re.IGNORECASE)),
    ("tapestry", re.compile(r"\btapestry\b", re.IGNORECASE)),
    ("seamlessly", re.compile(r"\bseamlessly\b", re.IGNORECASE)),
    ("effortlessly", re.compile(r"\beffortlessly\b", re.IGNORECASE)),
    ("game-changing", re.compile(r"\bgame[- ]changing\b", re.IGNORECASE)),
    ("first-class citizen", re.compile(r"\bfirst[- ]class\s+citizen(?:s)?\b", re.IGNORECASE)),
]

# Legitimate neural-network / hardware terms containing "gate" or "gates".
ALLOWED_GATE_TERMS_RE = re.compile(
    r"(?:"
    r"\b(?:up|q)\s*/\s*gates?\b"
    r"|\bgates?\s*/\s*up\b"
    r"|\b(?:swiglu|geglu|glu|silu|gelu|sigmoid|mlp|ffn|expert|shared[- ]expert"
    r"|router|attention|output|input|forget|linear|logic|xor|and|or|nand|nor)\s+gates?\b"
    r"|\bgates?\s+(?:proj(?:ection|ections)?|weights?|tensors?|matrices|matrix"
    r"|layers?|activations?|outputs?|values?|vectors?|and\s+up)\b"
    r"|\bup\s+and\s+gates?\b"
    r")",
    re.IGNORECASE,
)
BANNED_GATE_RE = re.compile(r"\bgates?\b", re.IGNORECASE)

SMOLVLA_RE = re.compile(r"smolvla", re.IGNORECASE)

OBSOLETE_URLS: list[tuple[str, re.Pattern[str]]] = [
    ("/docs/install_win", re.compile(r"/docs/install_win\b")),
    ("/docs/install_lin", re.compile(r"/docs/install_lin\b")),
    ("/docs/install/nix/", re.compile(r"/docs/install/nix/")),
    ("/docs/benchmarks", re.compile(r"/docs/benchmarks\b")),
    ("/benchmarks/community/", re.compile(r"(?:^|/|\"|\(|\s)benchmarks/community/")),
    ("/docs/instructions/images/", re.compile(r"(?:^|/|\"|\(|\s)docs/instructions/images/")),
    ("/docs/instructions/server/obsidian", re.compile(r"docs/instructions/server/obsidian", re.IGNORECASE)),
    ("/docs/instructions/server/msft_AI_toolkit", re.compile(r"docs/instructions/server/msft_ai_toolkit", re.IGNORECASE)),
    ("/docs/instructions/server/rag_LangChain", re.compile(r"docs/instructions/server/rag_langchain", re.IGNORECASE)),
    ("/docs/instructions/server/websearch_LangChain", re.compile(r"docs/instructions/server/websearch_langchain", re.IGNORECASE)),
    ("images/demos/", re.compile(r"images/demos/")),
    ("images/architecture.png", re.compile(r"images/architecture\.png")),
]

# Developer/compiler NPU jargon and disallowed badge abbreviations on user-facing pages
# (everything outside docs/docs/contributing/).
USER_PAGE_JARGON: list[tuple[str, re.Pattern[str]]] = [
    ("AIE2P", re.compile(r"\bAIE2P\b", re.IGNORECASE)),
    ("MLIR-AIE", re.compile(r"\bMLIR-AIE\b", re.IGNORECASE)),
    ("memtile", re.compile(r"\bmemtiles?\b", re.IGNORECASE)),
    ("WideDeltaNet", re.compile(r"\bWideDeltaNet\b")),
    ("BFP16ebs8", re.compile(r"\bBFP16ebs8\b", re.IGNORECASE)),
    ("Dual-Lane DMA", re.compile(r"\bDual-Lane\s+DMA\b", re.IGNORECASE)),
    ("4-Row Verify", re.compile(r"\b4-Row\s+Verify\b", re.IGNORECASE)),
    ("Full-ELF", re.compile(r"\bFull-ELF\b", re.IGNORECASE)),
    ("XCLBIN", re.compile(r"\bXCLBINs?\b", re.IGNORECASE)),
    ("100% NPU", re.compile(r"100%\s*NPU", re.IGNORECASE)),
    ("quant-range (Q*..Q*)", re.compile(r"\bQ\d[A-Za-z0-9_]*\.\.[A-Za-z0-9_]+\b")),
]

# Capitalized prose model names that should not be wrapped in inline backticks
# (lowercase CLI tags like `qwen3.5:4b`, family IDs like `qwen35`, repo slugs
# containing `/`, and `-NPU2` xclbin bundle IDs are allowed).
PROSE_MODEL_BACKTICK_RE = re.compile(
    r"^(?:"
    r"Qwen\s*\d(?:\.\d+)?"
    r"|Gemma\s*[34]"
    r"|EmbeddingGemma"
    r"|MedGemma"
    r"|TranslateGemma"
    r"|Llama\s*3(?:\.\d+)?"
    r"|Granite\s*4(?:\.\d+)?"
    r"|Phi-[34]"
    r"|FLUX\.2"
    r"|Whisper"
    r"|LFM2(?:\.5)?"
    r"|DeepSeek"
    r"|BERT(?!-h(?:\d|\*))"
    r"|BGE"
    r"|MiniLM"
    r"|NomicBERT"
    r"|Nanbeige"
    r"|GPT-OSS"
    r")\b"
)

ALLOWED_INLINE_CODE_SUFFIXES = (
    "-NPU2",
    ".gguf",
    ".q4nx",
    ".npue",
    ".xclbin",
    ".safetensors",
    ".json",
    ".py",
    ".cpp",
    ".hpp",
    ".h",
    ".cc",
    ".md",
    ".sh",
)


def is_prose_model_backtick_clutter(code_span: str) -> bool:
    s = code_span.strip()
    if not s:
        return False
    if "/" in s or "\\" in s or "=" in s:
        return False
    if s.startswith(("oflm ", "--", "-")):
        return False
    if s.endswith(ALLOWED_INLINE_CODE_SUFFIXES):
        return False
    return PROSE_MODEL_BACKTICK_RE.search(s) is not None


def strip_quotes(val: str) -> str:
    val = val.strip()
    if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
        return val[1:-1]
    return val


def parse_front_matter(lines: list[str]) -> tuple[dict[str, tuple[int, str]] | None, int]:
    """Parse leading YAML front matter.

    Returns (fields, body_start_idx) where fields maps top-level key -> (1-based line, value),
    or (None, 0) if valid YAML front matter is missing.
    """
    if not lines or lines[0].strip() != "---":
        return None, 0

    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() in ("---", "..."):
            end_idx = idx
            break
    if end_idx is None:
        return None, 0

    fields: dict[str, tuple[int, str]] = {}
    idx = 1
    key_re = re.compile(r"^([A-Za-z0-9_-]+)\s*:\s*(.*)$")
    while idx < end_idx:
        line = lines[idx]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            idx += 1
            continue
        m = key_re.match(line)
        if m:
            key = m.group(1)
            raw_val = m.group(2).strip()
            line_no = idx + 1
            if raw_val in ("|", "|-", "|+", ">", ">-", ">+"):
                block_lines: list[str] = []
                idx += 1
                while idx < end_idx and (lines[idx].startswith((" ", "\t")) or not lines[idx].strip()):
                    block_lines.append(lines[idx].strip())
                    idx += 1
                fields[key] = (line_no, " ".join(part for part in block_lines if part))
                continue
            fields[key] = (line_no, strip_quotes(raw_val))
        idx += 1

    return fields, end_idx + 1


def is_effectively_blank(line: str, in_blockquote: bool = False) -> bool:
    s = line.strip()
    if not s:
        return True
    if in_blockquote and re.match(r"^(?:>\s*)+$", s):
        return True
    return False


def strip_inline_code_and_shortcodes(line: str) -> str:
    cleaned = INLINE_CODE_RE.sub(" ", line)
    cleaned = HUGO_SHORTCODE_RE.sub(" ", cleaned)
    return cleaned


def split_markdown_table_cells(row: str) -> list[str]:
    def _scrub_pipes(m: re.Match[str]) -> str:
        return m.group(0).replace("|", " ")

    safe = row.replace(r"\|", " ")
    safe = INLINE_CODE_RE.sub(_scrub_pipes, safe)
    safe = HUGO_SHORTCODE_RE.sub(_scrub_pipes, safe)
    s = safe.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def check_file(path: Path, max_issue: int | None = None) -> list[str]:
    errors: list[str] = []
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        rel = path

    rel_posix = rel.as_posix()
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    is_in_docs_tree = rel_posix.startswith("docs/")
    is_docs_subpage = rel_posix.startswith("docs/docs/")
    is_docs_model_page = rel.parent.as_posix() == "docs/docs/models" and rel.suffix.lower() == ".md"
    is_doc_contrib_guide = rel_posix == "docs/docs/contributing/doc-contributions.md"
    is_user_page = not rel_posix.startswith("docs/docs/contributing/")

    # 1. YAML front matter presence and required keys
    fields, body_start = parse_front_matter(lines)
    if fields is None:
        errors.append(f"{rel}:1: [front-matter] missing YAML front matter ('---')")
        body_start = 0
    else:
        for req_key in ("title", "description"):
            if req_key not in fields or not fields[req_key][1].strip():
                errors.append(f"{rel}:1: [front-matter] missing non-empty '{req_key}:' in YAML front matter")
            else:
                key_line, key_val = fields[req_key]
                # 2. Raw backticks inside title or description
                if "`" in key_val:
                    errors.append(
                        f"{rel}:{key_line}: [front-matter-backtick] raw backtick found in YAML '{req_key}:'"
                    )
                if "\u2014" in key_val:
                    errors.append(
                        f"{rel}:{key_line}: [em-dash] em-dash ('\u2014') found in YAML '{req_key}:'"
                    )
                if EMOJI_RE.search(key_val):
                    errors.append(
                        f"{rel}:{key_line}: [front-matter-emoji] decorative emoji found in YAML '{req_key}:'"
                    )
                if is_in_docs_tree and PRACTICAL_LIMITS_WIP_RE.search(key_val):
                    errors.append(
                        f"{rel}:{key_line}: [banned-phrase] 'Practical Limits & WIP' is disallowed in docs/"
                    )
                if is_user_page:
                    for j_label, j_re in USER_PAGE_JARGON:
                        if j_re.search(key_val):
                            errors.append(
                                f"{rel}:{key_line}: [dev-jargon] developer jargon {j_label!r} found on user-facing page"
                            )

        # Check front matter lines for smolvla and obsolete URLs (except Hugo redirect aliases:)
        in_aliases = False
        for fm_idx in range(1, max(body_start - 1, 1)):
            fm_line = lines[fm_idx]
            fm_line_no = fm_idx + 1
            if re.match(r"^aliases\s*:", fm_line):
                in_aliases = True
            elif re.match(r"^[A-Za-z0-9_-]+\s*:", fm_line):
                in_aliases = False

            if SMOLVLA_RE.search(fm_line):
                errors.append(f"{rel}:{fm_line_no}: [obsolete-ref] leftover 'smolvla' reference found")
            if not in_aliases:
                for url_label, url_re in OBSOLETE_URLS:
                    if url_re.search(fm_line):
                        errors.append(
                            f"{rel}:{fm_line_no}: [obsolete-url] obsolete URL {url_label!r} found"
                        )

    in_fence = False
    fence_char = ""
    fence_len = 0
    fence_open_line = 0
    fence_in_blockquote = False

    in_variants_section = False
    in_variants_table = False
    variants_table_row_idx = 0
    seen_non_custom_variant_row = False

    idx = body_start
    while idx < len(lines):
        line = lines[idx]
        line_no = idx + 1

        if SMOLVLA_RE.search(line):
            errors.append(f"{rel}:{line_no}: [obsolete-ref] leftover 'smolvla' reference found")
        for url_label, url_re in OBSOLETE_URLS:
            if url_re.search(line):
                errors.append(f"{rel}:{line_no}: [obsolete-url] obsolete URL {url_label!r} found")

        # Support optional blockquote prefix when detecting code fences
        bq_match = re.match(r"^(\s*(?:>\s*)+)(.*)$", line)
        in_bq = bq_match is not None
        unquoted_line = bq_match.group(2) if bq_match else line

        if not in_fence:
            fence_m = FENCE_OPEN_RE.match(unquoted_line)
            if fence_m:
                marker = fence_m.group(2)
                info = fence_m.group(3).strip()

                # Check blank line before opening fence (unless at start of body)
                if idx > body_start and not is_effectively_blank(lines[idx - 1], in_blockquote=in_bq):
                    errors.append(
                        f"{rel}:{line_no}: [code-fence-spacing] opening code fence must be preceded by a blank line"
                    )
                # Check explicit language identifier
                if not info:
                    errors.append(
                        f"{rel}:{line_no}: [code-fence-lang] opening code fence is missing a language tag (e.g. ```bash)"
                    )

                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
                fence_open_line = line_no
                fence_in_blockquote = in_bq
                in_variants_table = False
                idx += 1
                continue

            non_code_line = INLINE_CODE_RE.sub(" ", unquoted_line)

            if is_docs_subpage and not is_docs_model_page and SPEC_GRID_DIV_RE.search(non_code_line):
                errors.append(
                    f"{rel}:{line_no}: [spec-grid-scope] <div class=\"oflm-spec-grid\"> is only allowed in docs/docs/models/*.md"
                )

            if is_in_docs_tree and PRACTICAL_LIMITS_WIP_RE.search(
                non_code_line if is_doc_contrib_guide else unquoted_line
            ):
                errors.append(
                    f"{rel}:{line_no}: [banned-phrase] 'Practical Limits & WIP' is disallowed in docs/"
                )

            # Outside code fence: check headings, variant table order, inline-code model clutter, em-dashes, and banned phrases
            heading_m = HEADING_RE.match(unquoted_line.lstrip())
            if heading_m:
                heading_level = len(heading_m.group(1))
                heading_text = heading_m.group(2)
                if EMOJI_RE.search(heading_text):
                    errors.append(
                        f"{rel}:{line_no}: [heading-emoji] decorative emoji found in heading: {heading_text.strip()!r}"
                    )
                if is_docs_model_page:
                    in_variants_table = False
                    if heading_level == 2 and VARIANTS_SHORTCUTS_HEADING_RE.match(heading_text.strip()):
                        in_variants_section = True
                        seen_non_custom_variant_row = False
                    elif heading_level <= 2 and in_variants_section:
                        in_variants_section = False
            elif is_docs_model_page and in_variants_section:
                stripped_table_line = unquoted_line.strip()
                if stripped_table_line.startswith("|"):
                    if not in_variants_table:
                        in_variants_table = True
                        variants_table_row_idx = 0
                        seen_non_custom_variant_row = False
                    else:
                        variants_table_row_idx += 1
                        if not (variants_table_row_idx == 1 and TABLE_SEP_RE.match(stripped_table_line)):
                            cells = split_markdown_table_cells(stripped_table_line)
                            lead_cols = " | ".join(cells[:2]) if len(cells) >= 3 else (cells[0] if cells else "")
                            is_custom_lead = CUSTOM_WORD_RE.search(lead_cols) is not None
                            has_oflm_add = OFLM_ADD_RE.search(stripped_table_line) is not None
                            if is_custom_lead and has_oflm_add:
                                if seen_non_custom_variant_row:
                                    errors.append(
                                        f"{rel}:{line_no}: [variant-table-order] 'Custom' ('oflm add') row in "
                                        f"'## Variants & Built-in Shortcuts' must appear before non-Custom shortcut rows"
                                    )
                            elif not is_custom_lead:
                                seen_non_custom_variant_row = True
                else:
                    in_variants_table = False

            line_without_shortcodes = HUGO_SHORTCODE_RE.sub(" ", unquoted_line)
            for code_m in INLINE_CODE_RE.finditer(line_without_shortcodes):
                code_text = code_m.group(2)
                if is_prose_model_backtick_clutter(code_text):
                    errors.append(
                        f"{rel}:{line_no}: [inline-code-clutter] prose model name `{code_text}` should not be backticked"
                    )

            prose = strip_inline_code_and_shortcodes(unquoted_line)

            if "\u2014" in prose:
                errors.append(f"{rel}:{line_no}: [em-dash] em-dash ('\u2014') found in prose")

            for label, pattern in BANNED_PHRASES:
                if pattern.search(prose):
                    errors.append(
                        f"{rel}:{line_no}: [banned-phrase] banned phrase {label!r} found in prose"
                    )

            prose_without_allowed_gates = ALLOWED_GATE_TERMS_RE.sub(" ", prose)
            if BANNED_GATE_RE.search(prose_without_allowed_gates):
                errors.append(
                    f"{rel}:{line_no}: [banned-phrase] jargon 'gate'/'gates' found in prose outside neural/hardware gate terms"
                )

            if is_user_page:
                for j_label, j_re in USER_PAGE_JARGON:
                    if j_re.search(non_code_line):
                        errors.append(
                            f"{rel}:{line_no}: [dev-jargon] developer jargon {j_label!r} found on user-facing page"
                        )
        else:
            # Inside code fence: check spec-grid / Practical Limits & WIP outside doc-contributions.md, plus closing fence
            if is_docs_subpage and not is_docs_model_page and not is_doc_contrib_guide:
                if SPEC_GRID_DIV_RE.search(unquoted_line):
                    errors.append(
                        f"{rel}:{line_no}: [spec-grid-scope] <div class=\"oflm-spec-grid\"> is only allowed in docs/docs/models/*.md"
                    )
            if is_in_docs_tree and not is_doc_contrib_guide and PRACTICAL_LIMITS_WIP_RE.search(unquoted_line):
                errors.append(
                    f"{rel}:{line_no}: [banned-phrase] 'Practical Limits & WIP' is disallowed in docs/"
                )

            close_re = re.compile(rf"^\s*{re.escape(fence_char)}{{{fence_len},}}\s*$")
            if close_re.match(unquoted_line):
                if idx + 1 < len(lines) and not is_effectively_blank(
                    lines[idx + 1], in_blockquote=fence_in_blockquote
                ):
                    errors.append(
                        f"{rel}:{line_no}: [code-fence-spacing] closing code fence must be followed by a blank line"
                    )
                in_fence = False
                fence_char = ""
                fence_len = 0

        idx += 1

    if in_fence:
        errors.append(f"{rel}:{fence_open_line}: [code-fence] unclosed fenced code block")

    errors.extend(check_issue_refs_lines(rel, lines, max_issue=max_issue))

    return errors


def check_issue_refs_lines(
    rel: Path, lines: list[str], max_issue: int | None = None
) -> list[str]:
    errors: list[str] = []
    for idx, line in enumerate(lines):
        line_no = idx + 1
        found_nums: list[int] = []

        for m in ISSUE_ATTR_RE.finditer(line):
            raw_val = m.group(1)
            if "{{" in raw_val or "%" in raw_val:
                continue
            for num_s in re.findall(r"\d+", raw_val):
                found_nums.append(int(num_s))

        for m in ISSUE_JSON_YAML_RE.finditer(line):
            raw_val = m.group(1) or m.group(2) or ""
            if "{{" in raw_val or "%" in raw_val:
                continue
            for num_s in re.findall(r"\d+", raw_val):
                found_nums.append(int(num_s))

        for m in ISSUE_PATH_RE.finditer(line):
            host = m.group(1)
            repo_prefix = (m.group(2) or "").strip("/")
            num_s = m.group(3)
            if host is not None:
                if host.lower() != "github.com" or repo_prefix.lower() not in OWN_GITHUB_REPOS:
                    continue
            found_nums.append(int(num_s))

        reported_on_line: set[int] = set()
        for issue_num in found_nums:
            if issue_num in reported_on_line:
                continue
            invalid = (
                issue_num <= 0
                or issue_num in DELETED_ISSUE_NUMBERS
                or (max_issue is not None and issue_num > max_issue)
            )
            if invalid:
                reported_on_line.add(issue_num)
                cap_note = f" <= #{max_issue}" if max_issue is not None else ""
                errors.append(
                    f"{rel}:{line_no}: [unknown-issue-ref] unknown or deleted issue/PR "
                    f"#{issue_num} (expected valid Atomic-Germ/OpenFlowLM-Next issue{cap_note})"
                )
    return errors


def check_issue_refs_file(path: Path, max_issue: int | None = None) -> list[str]:
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        rel = path
    text = path.read_text(encoding="utf-8")
    return check_issue_refs_lines(rel, text.splitlines(), max_issue=max_issue)


EXCLUDED_DIRS = {
    (REPO_ROOT / "docs" / "public").resolve(),
    (REPO_ROOT / "docs" / "resources").resolve(),
}


def collect_markdown_files(targets: list[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for target in targets:
        resolved = target if target.is_absolute() else (Path.cwd() / target)
        if not resolved.exists():
            resolved = REPO_ROOT / target
        if resolved.is_file() and resolved.suffix.lower() == ".md":
            rp = resolved.resolve()
            if rp not in seen:
                seen.add(rp)
                files.append(resolved)
        elif resolved.is_dir():
            for md in sorted(resolved.rglob("*.md")):
                rp = md.resolve()
                if any(ex in rp.parents for ex in EXCLUDED_DIRS):
                    continue
                if rp not in seen:
                    seen.add(rp)
                    files.append(md)
    return files


def collect_issue_scan_files(targets: list[Path], exclude: set[Path] | None = None) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set(exclude or ())
    for target in targets:
        resolved = target if target.is_absolute() else (Path.cwd() / target)
        if not resolved.exists():
            resolved = REPO_ROOT / target
        if resolved.is_file() and resolved.suffix.lower() in ISSUE_SCAN_SUFFIXES:
            rp = resolved.resolve()
            if rp not in seen:
                seen.add(rp)
                files.append(resolved)
        elif resolved.is_dir():
            for cand in sorted(resolved.rglob("*")):
                if not cand.is_file() or cand.suffix.lower() not in ISSUE_SCAN_SUFFIXES:
                    continue
                rp = cand.resolve()
                if any(ex in rp.parents for ex in EXCLUDED_DIRS):
                    continue
                if rp not in seen:
                    seen.add(rp)
                    files.append(cand)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check Hugo Markdown documentation for front matter, formatting, and prose quality."
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=None,
        help="Optional files or directories to scan (default: docs/content and docs/docs).",
    )
    parser.add_argument(
        "--max-issue",
        type=int,
        default=None,
        help="Optional upper bound for valid GitHub issue/PR numbers during point-in-time audits.",
    )
    parser.add_argument(
        "positional_paths",
        nargs="*",
        help="Optional positional files or directories to scan.",
    )
    args = parser.parse_args(argv)

    raw_paths: list[str] = []
    if args.paths:
        raw_paths.extend(args.paths)
    if args.positional_paths:
        raw_paths.extend(args.positional_paths)

    target_paths = [Path(p) for p in raw_paths] if raw_paths else DEFAULT_PATHS
    md_files = collect_markdown_files(target_paths)
    checked_md_resolved = {f.resolve() for f in md_files}
    issue_scan_targets = [Path(p) for p in raw_paths] if raw_paths else DEFAULT_ISSUE_SCAN_PATHS
    extra_issue_files = collect_issue_scan_files(issue_scan_targets, exclude=checked_md_resolved)

    if not md_files and not extra_issue_files:
        print("ERROR: no documentation files found in specified paths.", file=sys.stderr)
        return 2

    all_errors: list[str] = []
    for md_file in md_files:
        all_errors.extend(check_file(md_file, max_issue=args.max_issue))
    for issue_file in extra_issue_files:
        all_errors.extend(check_issue_refs_file(issue_file, max_issue=args.max_issue))

    total_checked = len(md_files) + len(extra_issue_files)
    if all_errors:
        for err in all_errors:
            print(err, file=sys.stderr)
        print(
            f"\nFAILED: {len(all_errors)} quality violation(s) across {total_checked} file(s).",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: {len(md_files)} Markdown file(s) and {len(extra_issue_files)} auxiliary docs file(s) "
        f"passed documentation quality checks."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

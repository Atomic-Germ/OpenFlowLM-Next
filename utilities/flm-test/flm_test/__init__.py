#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from .tasks import LLMTask, EmbeddingTask, AudioTask, VisionTask, ToolCallingTask
from typing import Any

SUITE_NAMES = ("llm", "embedding", "audio", "vision", "tools")


def resolve_suites(args):
    """Decide which test suites must run.

    Rules:
      * ``--all`` enables every suite except embedding, which stays exclusive.
      * Embedding is mutually exclusive with the chat-based suites: requesting
        it alongside any of them (or with ``--all``) runs only the embedding
        suite, because those tests assume a server started with only an embed
        model loaded (``flm serve -e 1``) and must not require a full model.

    Returns ``(suites, note)`` where ``suites`` maps suite name -> bool and
    ``note`` is an (possibly empty) informational message for the user.
    """
    explicit = {name: getattr(args, name) for name in SUITE_NAMES}
    note = ""
    if args.all:
        suites = {name: name != "embedding" for name in SUITE_NAMES}
        note = ("Note: --all excludes the embedding suite; run `flm-test --embedding` "
                "separately so only an embed model needs to be loaded.")
    else:
        suites = dict(explicit)

    # An explicit --embedding request wins over --all or any chat-based suite:
    # the embedding tests must never share a run with suites needing a full model.
    if explicit["embedding"] and (args.all or any(explicit[name] for name in ("llm", "audio", "vision", "tools"))):
        suites = {name: name == "embedding" for name in SUITE_NAMES}
        note = ("--embedding is mutually exclusive with "
                "--llm/--audio/--vision/--tools; running only the embedding tests.")
    return suites, note


def main():
    parser = argparse.ArgumentParser(description="Test runner for FLM models.")
    parser.add_argument('--llm', action='store_true', help="Run LLM tests")
    parser.add_argument('--embedding', action='store_true', help="Run Embedding tests")
    parser.add_argument('--audio', action='store_true', help="Run Audio tests")
    parser.add_argument('--vision', action='store_true', help="Run vision tests")
    parser.add_argument('--tools', action='store_true',
                        help="Run tool-calling tests (five complexity levels)")
    parser.add_argument('--all', action='store_true',
                        help="Run all suites except embedding (which stays exclusive)")
    parser.add_argument('--gen-lim', type=int, default=-1, help="Maximum number of tokens to generate")
    parser.add_argument('--temp', '--temperature', type=float, default=0.3, metavar='TEMP',
                        help="Sampling temperature for chat-based tests (e.g. 0.7). "
                             "Defaults to 0.3, a common setting for reliable tool calling.")
    parser.add_argument('--reasoning', type=str, default=None, metavar='LEVEL',
                        choices=["none", "low", "medium", "high"],
                        help="Reasoning effort sent as `reasoning_effort` with every chat request. "
                             "low/medium/high progressively enable thinking; none disables it. "
                             "Omit the flag to leave each model's own default behaviour untouched.")
    parser.add_argument("--port", type=str, default="52625", help="Port your FLM instance is running on.")
    parser.add_argument('--backend-os', type=str, default="linux", choices=["linux", "windows"], help="OS of the FLM backend (default: linux)")
    parser.add_argument('--model', type=str, nargs='+', metavar='MODEL_ID',
                        help="Only test the specified model(s). Can be repeated or space-separated. "
                             "Example: --model gemma3:4b  or  --model gemma3:4b qwen3vl-it:4b")

    args = parser.parse_args()

    suites, note = resolve_suites(args)

    if note:
        print(note + "\n")

    if not any(suites.values()):
        parser.print_help()
        return

    try:
        print("Please ensure you have started the FLM server and have the correct URL and port. \n")

        host="http://127.0.0.1"
        port=str(args.port)
        endpoint="/v1"
        baseurl=f"{host}:{port}{endpoint}"

        model_filter = args.model  # list[str] | None

        if suites["llm"]:
            LLMTask(baseurl, args.backend_os, model_filter=model_filter).run(max_completion_tokens=args.gen_lim, temperature=args.temp, reasoning=args.reasoning)

        if suites["embedding"]:
            EmbeddingTask(baseurl, args.backend_os, model_filter=model_filter).run()

        if suites["audio"]:
            AudioTask(baseurl, args.backend_os, model_filter=model_filter).run(temperature=args.temp, reasoning=args.reasoning)

        if suites["vision"]:
            VisionTask(baseurl, args.backend_os, model_filter=model_filter).run(max_generation_tokens=args.gen_lim, temperature=args.temp, reasoning=args.reasoning)

        if suites["tools"]:
            ToolCallingTask(baseurl, args.backend_os, model_filter=model_filter).run(max_completion_tokens=args.gen_lim, temperature=args.temp, reasoning=args.reasoning)

    except Exception as e:
        print(f"Error during testing: {e}")


if __name__ == "__main__":
    main()

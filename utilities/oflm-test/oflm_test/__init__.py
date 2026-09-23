#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from .tasks import (LLMTask, EmbeddingTask, AudioTask, VisionTask, ToolCallingTask,
                    ApiConformanceTask, TranscriptionTask)

SUITE_NAMES = ("llm", "embedding", "audio", "vision", "tools", "api")
# Suites that need a chat model loaded; embedding is the odd one out.
CHAT_SUITES = ("llm", "audio", "vision", "tools", "api")


def resolve_suites(args):
    """Decide which test suites must run.

    Rules:
      * ``--all`` enables every suite except embedding, which stays exclusive.
      * Embedding is mutually exclusive with the chat-based suites: requesting
        it alongside any of them (or with ``--all``) runs only the embedding
        suite, because those tests assume a server started with only an embed
        model loaded (``oflm serve -e 1``) and must not require a full model.

    Returns ``(suites, note)`` where ``suites`` maps suite name -> bool and
    ``note`` is an (possibly empty) informational message for the user.
    """
    explicit = {name: getattr(args, name) for name in SUITE_NAMES}
    note = ""
    if args.all:
        suites = {name: name != "embedding" for name in SUITE_NAMES}
        note = ("Note: --all excludes the embedding suite; run `oflm-test --embedding` "
                "separately so only an embed model needs to be loaded.")
    else:
        suites = dict(explicit)

    # An explicit --embedding request wins over --all or any chat-based suite:
    # the embedding tests must never share a run with suites needing a full model.
    if explicit["embedding"] and (args.all or any(explicit[name] for name in CHAT_SUITES)):
        suites = {name: name == "embedding" for name in SUITE_NAMES}
        note = ("--embedding is mutually exclusive with "
                "--llm/--audio/--vision/--tools/--api; running only the embedding tests.")
    return suites, note


def print_summary(results) -> int:
    """Print what every suite decided and return the number of hard failures.

    The tool used to end here with a CSV and no verdict, so a suite could go red
    and the command still looked like it had succeeded.
    """
    print("\n=== Summary ===")
    if not results:
        print("No suites ran.")
        return 0
    hard_failures = 0
    for result in results:
        counts = ", ".join(f"{verdict} {count}"
                           for verdict, count in sorted(result.verdicts.items()))
        print(f"  {result.name:<10} {counts or 'no checks ran'}")
        hard_failures += result.hard_failures
        # A suite that checked nothing is not a suite that passed. Without this
        # a missing model makes the whole run green, which is the failure this
        # summary exists to stop.
        if not result.total:
            hard_failures += 1
    for result in results:
        for failure in result.failures:
            print(f"    {result.name}: {failure}")
        if not result.total:
            print(f"    {result.name}: no checks ran; was a model for this suite loaded?")
    if hard_failures:
        print(f"\n{hard_failures} hard failure(s). SOFT-FAIL is noted but does not "
              f"fail the run.")
    else:
        print("\nNo hard failures.")
    return hard_failures


def main():
    parser = argparse.ArgumentParser(description="Test runner for OFLM models.")
    parser.add_argument('--llm', action='store_true', help="Run LLM tests")
    parser.add_argument('--embedding', action='store_true', help="Run Embedding tests")
    parser.add_argument('--audio', action='store_true', help="Run Audio tests")
    parser.add_argument('--vision', action='store_true', help="Run vision tests")
    parser.add_argument('--tools', action='store_true',
                        help="Run tool-calling tests (seven complexity levels)")
    parser.add_argument('--api', action='store_true',
                        help="Run server conformance tests (error status, model identity, "
                             "finish_reason, request isolation, stream parity)")
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
    parser.add_argument("--port", type=str, default="52625", help="Port your OFLM instance is running on.")
    parser.add_argument('--backend-os', type=str, default="linux", choices=["linux", "windows"], help="OS of the OFLM backend (default: linux)")
    parser.add_argument('--model', type=str, nargs='+', metavar='MODEL_ID',
                        help="Only test the specified model(s). Can be repeated or space-separated. "
                             "Example: --model gemma3:4b  or  --model gemma3:4b qwen3vl-it:4b")
    parser.add_argument('--exit-zero', action='store_true',
                        help="Always exit 0, even when checks fail. Without it the run exits 1 "
                             "on any FAIL or ERROR so a script or CI job can gate on it.")

    args = parser.parse_args()

    suites, note = resolve_suites(args)

    if note:
        print(note + "\n")

    if not any(suites.values()):
        parser.print_help()
        return 0

    print("Please ensure you have started the OFLM server and have the correct URL and port. \n")

    host = "http://127.0.0.1"
    port = str(args.port)
    endpoint = "/v1"
    baseurl = f"{host}:{port}{endpoint}"

    model_filter = args.model  # list[str] | None
    results = []
    try:
        run_suites(args, baseurl, suites, model_filter, results)
    except Exception as e:
        # Still a failed run, not a printed line: this used to be swallowed and
        # the command exited 0 with whatever CSV it had managed to write.
        print(f"\nError during testing: {e}")
        print_summary([r for r in results if r is not None])
        return 0 if args.exit_zero else 1

    hard_failures = print_summary([r for r in results if r is not None])
    if args.exit_zero:
        return 0
    return 1 if hard_failures else 0


def run_suites(args, baseurl, suites, model_filter, results):
    if suites["llm"]:
        results.append(LLMTask(baseurl, args.backend_os, model_filter=model_filter)
                       .run(max_completion_tokens=args.gen_lim, temperature=args.temp,
                            reasoning=args.reasoning))

    if suites["embedding"]:
        results.append(EmbeddingTask(baseurl, args.backend_os, model_filter=model_filter).run())

    if suites["audio"]:
        results.append(AudioTask(baseurl, args.backend_os, model_filter=model_filter)
                       .run(temperature=args.temp, reasoning=args.reasoning))

        # The audio task above posts chat.completions, which Whisper refuses as a
        # non-chat model; this one is the only thing that exercises
        # /v1/audio/transcriptions.
        results.append(TranscriptionTask(baseurl, args.backend_os,
                                         model_filter=model_filter).run())

    if suites["vision"]:
        results.append(VisionTask(baseurl, args.backend_os, model_filter=model_filter)
                       .run(max_generation_tokens=args.gen_lim, temperature=args.temp,
                            reasoning=args.reasoning))

    if suites["tools"]:
        results.append(ToolCallingTask(baseurl, args.backend_os, model_filter=model_filter)
                       .run(max_completion_tokens=args.gen_lim, temperature=args.temp,
                            reasoning=args.reasoning))

    if suites["api"]:
        results.append(ApiConformanceTask(baseurl, args.backend_os, model_filter=model_filter).run())


if __name__ == "__main__":
    sys.exit(main())

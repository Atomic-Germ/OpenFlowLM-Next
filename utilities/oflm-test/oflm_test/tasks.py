import base64
import csv
import os
import re
import time
import json
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# Media assets ship inside the package; anchor paths here so tests work
# regardless of the current working directory (repo checkout or pip/uv install).
PACKAGE_DIR = Path(__file__).resolve().parent

# Specialised (non-general-chat) models excluded from chat-based suites.
NON_CHAT_MODELS = ("gpt-oss:20b", "gpt-oss-sg:20b", "qwen3.5:4b", "qwen3.5:9b",
                   "medgemma:4b", "medgemma1.5:4b", "translategemma:4b")

# Verdicts that make a run fail. SOFT-FAIL is noted and does not, which is what
# it already means everywhere else in the suite.
HARD_VERDICTS = ("FAIL", "ERROR")
# Verdicts that are not a judgement at all, so they are not counted.
IGNORED_VERDICTS = ("N/A", "SKIPPED", "SKIP", "")


@dataclass
class SuiteResult:
    """What one suite decided, so the runner can summarise and set an exit code.

    Without this the tool could only ever report: a FAIL landed in a CSV cell
    and waited for someone to open the file.
    """
    name: str
    verdicts: Counter = field(default_factory=Counter)
    failures: list[str] = field(default_factory=list)

    @property
    def hard_failures(self) -> int:
        return sum(self.verdicts[v] for v in HARD_VERDICTS)

    @property
    def total(self) -> int:
        return sum(self.verdicts.values())


class BaseTestTask(ABC):
    """
    Abstract base class for all testing tasks.
    Enforces a standard interface for running tests and saving results.
    """
    SUITE_NAME = "suite"
    MUSIC_PATTERN = re.compile(
        r"\b(music|melod\w*|song|tune|rhythm|beat|tempo|instrument\w*|drum\w*|bass|synth\w*|vocal\w*|chord\w*|harmo\w*)\b",
        re.IGNORECASE,
    )

    def __init__(self, base_url, backend_os="linux", model_filter: list[str] | None = None):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key="oflm")
        self.version = self._get_oflm_version()
        self.models = self._fetch_all_models()
        if model_filter:
            filtered = [m for m in self.models if m in model_filter]
            if not filtered:
                print(f"Warning: none of the requested model(s) {model_filter} were found in the server's model list. "
                      f"Available: {self.models}")
            self.models = filtered
        self.results_dir = os.path.join("results", self.timestamp, backend_os)
        os.makedirs(self.results_dir, exist_ok=True)
        self.result = SuiteResult(self.SUITE_NAME)

    # OFLM's OpenAI-compatible API accepts `reasoning_effort` with "low",
    # "medium" or "high" (thinking enabled) and "none" (thinking disabled).
    REASONING_LEVELS = ("none", "low", "medium", "high")

    @staticmethod
    def _reasoning_kwargs(reasoning: str | None) -> dict:
        """Extra kwargs for chat.completions.create setting a reasoning effort.
        Returns {} when no level is requested so nothing is sent to the server
        (never JSON null) and the model keeps its own default behaviour."""
        return {"reasoning_effort": reasoning} if reasoning else {}

    def get_csv_filename(self, task_name: str) -> str:
        return os.path.join(self.results_dir, f"{task_name}_results_v{self.version}.csv")

    def record(self, verdict, where: str):
        """Tally one check verdict and return it unchanged, so call sites can wrap.

        Accepts either the (verdict, detail) tuple the checks return or a bare
        string like "PASS" or "ERROR: connection refused".
        """
        text = verdict[0] if isinstance(verdict, tuple) else str(verdict)
        name, _, inline_detail = text.partition(":")
        name = name.strip().upper()
        if name in IGNORED_VERDICTS:
            return verdict
        self.result.verdicts[name] += 1
        if name in HARD_VERDICTS:
            detail = verdict[1] if isinstance(verdict, tuple) else inline_detail.strip()
            self.result.failures.append(f"{where}: {name}" + (f" ({detail})" if detail else ""))
        return verdict

    def _post_json(self, path: str, body, timeout: int = 600, raw_body: bytes | None = None):
        """One raw POST; returns (status, parsed_json_or_None, raw_text).

        Raw rather than through the OpenAI SDK because for the conformance
        checks the HTTP status *is* the assertion, and the SDK hides it: a 4xx
        becomes an exception and a 200 carrying an error body becomes a success.
        """
        data = raw_body if raw_body is not None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(f"{self.base_url}{path}", data=data,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status, text = response.status, response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            status, text = e.code, e.read().decode("utf-8", "replace")
        try:
            return status, json.loads(text), text
        except json.JSONDecodeError:
            return status, None, text

    @staticmethod
    def _error_fields(body) -> tuple[str | None, str | None, str | None]:
        """(code, type, message) out of an OpenAI-shaped error body.

        A body whose `error` is a bare string is the shape the server produced
        when its own error handling threw, so it is reported rather than
        smoothed over: the message comes back and code and type are None.
        """
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            return error.get("code"), error.get("type"), error.get("message")
        if isinstance(error, str):
            return None, None, error
        return None, None, None

    def _get_oflm_version(self) -> str:
        print("\nChecking oflm version...")
        try:
            response = urllib.request.urlopen(f"{self.base_url}/version", timeout=5)
            version_data = json.loads(response.read().decode('utf-8'))
            oflm_version = version_data.get("version", "unknown_version")
            print(f"Detected oflm version: {oflm_version}")
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as e:
            print(f"Error fetching oflm version: {e}")
            oflm_version = "unknown_version"
        return oflm_version

    def _fetch_all_models(self) -> list:
        print("\nFetching available models...")
        try:
            response = urllib.request.urlopen(f"{self.base_url}/models", timeout=5)
            models_json = json.loads(response.read().decode('utf-8'))
            model_list = models_json.get("data", [])
            model_id = [m["id"] for m in model_list]
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
            print(f"Error fetching models: {e}")
            model_id = []
        return model_id

    def _collect_stream_meta(self, response) -> tuple[str, str, str | None, object, str | None]:
        """Accumulates a stream into (reasoning, content, finish_reason, usage, model).

        A stream that never reports a finish_reason yields None for it, which is
        itself a finding rather than a missing value to paper over.
        """
        reasoning_content, output_content = "", ""
        finish_reason, usage, model = None, None, None
        for chunk in response:
            usage = getattr(chunk, "usage", None) or usage
            model = getattr(chunk, "model", None) or model
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            delta = choice.delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_content += delta.reasoning_content
            if delta.content:
                output_content += delta.content
        return reasoning_content, output_content, finish_reason, usage, model

    def _collect_stream(self, response) -> tuple[str, str]:
        """Accumulates streamed chunks into (reasoning_content, output_content)."""
        reasoning_content, output_content = self._collect_stream_meta(response)[:2]
        return reasoning_content, output_content

    @abstractmethod
    def run(self, *args, **kwargs):
        """Must be implemented by all subclasses"""
        pass


class LLMTask(BaseTestTask):
    """
    Tests language models over two-round conversations, streamed and not.

    The checks here are deliberately not about answer quality: they are the
    things the server owes every client whatever the weights say. An answer
    that is empty, that reports no finish_reason, that comes back under a
    different model's name or with usage counts that do not add up is a server
    defect, and this suite used to write all four to CSV without comment.
    """

    SUITE_NAME = "llm"

    PROMPT = "Teach me Maxwell's equations."
    FOLLOWUP_NON_STREAM = "Summarize your answer."
    FOLLOWUP_STREAM = "Explain why they are important."

    # A code the model is asked to keep, then asked for again. Nothing but the
    # earlier turn can supply it, so a server that drops history cannot guess it.
    MEMORY_CODE = "QX-7731-ZB"
    MEMORY_PROMPT = (f"Remember this reference code exactly: {MEMORY_CODE}. "
                     f"Reply with just the code and nothing else.")
    MEMORY_FOLLOWUP = "What was the reference code I gave you? Reply with just the code."

    def __init__(self, base_url, backend_os="linux", model_filter: list[str] | None = None):
        super().__init__(base_url, backend_os, model_filter=model_filter)
        self.models = [m for m in self.models if m not in NON_CHAT_MODELS]
        self.csv_filename = self.get_csv_filename("llm")

    @staticmethod
    def _check_round(requested_model, reported_model, content, finish_reason, usage):
        """What the server owes a chat client, whatever the model answered."""
        if not (content or "").strip():
            return ("FAIL", "empty response")
        if not finish_reason:
            return ("FAIL", "response carries no finish_reason, so a client "
                            "cannot tell a complete answer from a truncated one")
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            if None in (prompt_tokens, completion_tokens, total_tokens):
                return ("SOFT-FAIL", f"usage is incomplete: {usage}")
            if prompt_tokens <= 0 or completion_tokens <= 0:
                return ("FAIL", f"usage reports {prompt_tokens} prompt and "
                                f"{completion_tokens} completion tokens")
            if prompt_tokens + completion_tokens != total_tokens:
                return ("FAIL", f"usage does not add up: {prompt_tokens} + "
                                f"{completion_tokens} != {total_tokens}")
        if reported_model and reported_model != requested_model:
            return ("SOFT-FAIL", f"requested '{requested_model}', response reports "
                                 f"'{reported_model}'")
        if not reported_model:
            return ("SOFT-FAIL", "response names no model, so a client cannot "
                                 "confirm which one answered")
        return ("PASS", f"finish_reason '{finish_reason}', {len(content)} chars")

    def _one_round(self, model_id, messages, stream, max_completion_tokens,
                   temperature, reasoning_kwargs):
        """One chat call; returns (reasoning, content, finish_reason, usage, model)."""
        response = self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            stream=stream,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            **reasoning_kwargs,
        )
        if stream:
            return self._collect_stream_meta(response)
        choice = response.choices[0]
        reasoning_content = getattr(choice.message, "reasoning_content", "") or ""
        return (reasoning_content, choice.message.content or "",
                getattr(choice, "finish_reason", None), getattr(response, "usage", None),
                getattr(response, "model", None))

    def _run_two_rounds(self, writer, model_id, prompt, followup_prompt, stream, max_completion_tokens,
                        temperature=None, reasoning=None):
        reasoning_kwargs = self._reasoning_kwargs(reasoning)
        mode = "Stream" if stream else "Non-Stream"
        messages = [{"role": "user", "content": prompt}]

        rounds = (prompt, followup_prompt)
        for index, round_prompt in enumerate(rounds):
            try:
                print(f"Prompt: {round_prompt}")
                reasoning_content, content, finish_reason, usage, reported = self._one_round(
                    model_id, messages, stream, max_completion_tokens, temperature, reasoning_kwargs)
                verdict = self._check_round(model_id, reported, content, finish_reason, usage)
                self.record(verdict, f"{model_id} / {mode} / {round_prompt[:32]}")
                writer.writerow([model_id, mode, round_prompt, reasoning_content or "N/A",
                                 content, finish_reason or "N/A", f"{verdict[0]}: {verdict[1]}"])
                print(f"Check result: {verdict[0]} ({verdict[1]})")
                time.sleep(1)
                if index + 1 < len(rounds):
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": rounds[index + 1]})
            except Exception as e:
                print(f"Error occurred in round '{round_prompt[:32]}', model: {model_id}: {e}")
                self.record(f"ERROR: {e}", f"{model_id} / {mode} / {round_prompt[:32]}")
                writer.writerow([model_id, mode, round_prompt, f"ERROR: {e}", "N/A", "N/A",
                                 f"ERROR: {e}"])
                return

    def _run_memory_probe(self, writer, model_id, max_completion_tokens, temperature, reasoning):
        """Round two must be able to see round one, or the history never arrived."""
        reasoning_kwargs = self._reasoning_kwargs(reasoning)
        messages = [{"role": "user", "content": self.MEMORY_PROMPT}]
        try:
            _, first, _, _, _ = self._one_round(model_id, messages, False, max_completion_tokens,
                                                temperature, reasoning_kwargs)
            messages.append({"role": "assistant", "content": first})
            messages.append({"role": "user", "content": self.MEMORY_FOLLOWUP})
            reasoning_content, content, finish_reason, usage, reported = self._one_round(
                model_id, messages, False, max_completion_tokens, temperature, reasoning_kwargs)
            round_verdict = self._check_round(model_id, reported, content, finish_reason, usage)
            if round_verdict[0] == "FAIL":
                # An empty round two is an empty response, not a model that
                # forgot: say which one it is.
                verdict = round_verdict
            elif self.MEMORY_CODE in (content or ""):
                verdict = ("PASS", "the earlier turn reached the model")
            elif self.MEMORY_CODE not in (first or ""):
                # It never echoed the code in round one, so round two failing to
                # repeat it says nothing about whether the history was carried.
                verdict = ("SOFT-FAIL", f"the model never echoed {self.MEMORY_CODE} in round one, "
                                        f"so the retention check is inconclusive")
            else:
                verdict = ("FAIL", f"expected {self.MEMORY_CODE} from the earlier turn, "
                                   f"got '{(content or '')[:80]}'")
            self.record(verdict, f"{model_id} / Context Retention")
            writer.writerow([model_id, "Non-Stream", self.MEMORY_FOLLOWUP, reasoning_content or "N/A",
                             content, finish_reason or "N/A", f"{verdict[0]}: {verdict[1]}"])
            print(f"Context retention check: {verdict[0]} ({verdict[1]})")
            time.sleep(1)
        except Exception as e:
            print(f"Error occurred in context retention probe, model: {model_id}: {e}")
            self.record(f"ERROR: {e}", f"{model_id} / Context Retention")
            writer.writerow([model_id, "Non-Stream", self.MEMORY_FOLLOWUP, f"ERROR: {e}",
                             "N/A", "N/A", f"ERROR: {e}"])

    def run(self, max_completion_tokens=-1, temperature=0.3, reasoning=None):
        with open(self.csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Model", "Mode", "Input", "Reasoning Content", "Output Content",
                             "Finish Reason", "Check Result"])
            print("\n=== Starting LLM Tests ===")
            print(f"Models found: {len(self.models)}")
            for model_id in self.models:
                print(f"\n--- Testing LLM model: {model_id} ---")
                print("\nTesting non-stream mode...\n")
                self._run_two_rounds(writer, model_id, self.PROMPT, self.FOLLOWUP_NON_STREAM,
                                     stream=False, max_completion_tokens=max_completion_tokens,
                                     temperature=temperature, reasoning=reasoning)
                print("\nTesting stream mode...\n")
                self._run_two_rounds(writer, model_id, self.PROMPT, self.FOLLOWUP_STREAM,
                                     stream=True, max_completion_tokens=max_completion_tokens,
                                     temperature=temperature, reasoning=reasoning)
                self._run_memory_probe(writer, model_id, max_completion_tokens, temperature, reasoning)
                print(f"Finished testing model: {model_id}")
        print(f"\nLLM tests complete. Saved to {self.csv_filename}")
        return self.result


class EmbeddingTask(BaseTestTask):
    """
    Tests OpenAI-compatible text embeddings across eleven automated checks:

      E1 Response Structure      The response is a well-formed embeddings payload
                                 with a non-empty numeric vector.
      E2 Repeatability           The same input is drawn N times and the count of
                                 inconsistent draws is reported, so a single
                                 outlier draw cannot masquerade as a clean run
                                 or as a build defect.
      E3 Batch & Index Integrity A batched request returns one embedding per
                                 input, in order, with unique indexes.
      E4 Dimensionality          Every vector in a batch shares the same,
                                 positive dimension.
      E5 Semantic Ordering       Semantically-related pairs land closer together
                                 in the embedding space than unrelated pairs.
      E6 Cross-Path Consistency  The same weights reached via two delivery paths
                                 (single-input and batched requests) in the same
                                 run must agree, pinning any bad number on OFLM
                                 rather than on a single bad machine draw.
E7 Batch Reference
       Consistency                A larger set of draws of the same input is
                                  compared against the batch-path reference in
                                  the same run, exposing the outlier rate and
                                  magnitude that a small sample can miss.
       E8 Reference Agreement     Embeddings match the bundled reference vectors
                                  produced by the validated numpy pipeline for
                                  google/embeddinggemma-300m, pinning the API
                                  path to a known-good implementation. The
                                  bundled vectors are for ONE model, so this
                                  check SKIPs for any other -- see
                                  REFERENCE_MODELS.
      E9 Model Identity          The server serves the model it was asked for,
                                 or refuses. The one failure the checks above
                                 cannot see, because a substituted model's
                                 vectors pass every one of them.
      E10 Task Prompt Honoured   The request's task prompt reaches the model:
                                 the same text under a query prompt and under a
                                 document prompt must not embed identically.
      E11 Unknown Task Prompt    A task prompt the server cannot resolve is
                                 refused, never quietly replaced with the
                                 default.

    Like the tool-calling suite, each check produces a PASS / SOFT-FAIL / FAIL
    verdict with a detail line, all written to CSV. Unlike the chat-based suites
    there is no streaming mode, temperature or reasoning, and the server is
    assumed to be running with only an embed model loaded (`oflm serve -e 1`).
    """

    SUITE_NAME = "embedding"

    EMBED_MODELS = [
        "embed-gemma:300m", "embed-gemma",
        # open_npue backend -- BERT-family encoders on the NPU. Listed here so
        # the suite runs against them without an explicit --model filter; each
        # is served by `oflm serve <llm> --embed 1 --embeddingmodel <tag>`.
        "bge-base:en-v1.5", "bge-small:en-v1.5", "bge-large:en-v1.5",
        "all-minilm:l6-v2", "nomic-embed-text:v1.5", "gte-multilingual:base",
    ]
    DEFAULT_EMBED_MODEL = "embed-gemma:300m"

    SAMPLE_TEXT = "The embedding model should capture the meaning of this sentence."
    BATCH_INPUTS = [
        "Hello, world!",
        "OpenFlowLM is a local inference server.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    RELATED_PAIRS = [("cat", "kitten"), ("ocean", "sea")]
    UNRELATED_PAIRS = [("cat", "car"), ("ocean", "desert")]

    REPEAT_COUNT = 10
    STABILITY_THRESHOLD = 0.999
    AGREEMENT_THRESHOLD = 0.999
    INCONSISTENT_PAIR_RATIO_FAIL = 0.25
    REFERENCE_DRAW_COUNT = 30
    REFERENCE_AGREEMENT_THRESHOLD = 0.999
    REFERENCE_FILE = PACKAGE_DIR / "test_files" / "embedding_reference.json"
    # The bundled oracle vectors are for ONE model. Comparing another model's
    # embeddings against them is not a weaker test, it is a meaningless one:
    # two models embed the same text into different spaces by design, so a
    # cosine between them carries no information about either. E8 therefore
    # SKIPs rather than reporting a failure it cannot substantiate.
    REFERENCE_MODELS = {"embed-gemma:300m", "embed-gemma"}
    # A tag no server can have loaded, for E9.
    IMPOSSIBLE_MODEL = "oflm-test-no-such-embedding-model"
    # REST names for the two task prompts every prompted model has, and a name
    # no table contains. See task_names() in src/server/openai_compat.hpp.
    QUERY_PROMPT = "search_query"
    DOCUMENT_PROMPT = "search_document"
    IMPOSSIBLE_PROMPT = "not_a_task"
    CHECK_NAMES = [
        "E1 Response Structure",
        "E2 Repeatability",
        "E3 Batch & Index Integrity",
        "E4 Dimensionality",
        "E5 Semantic Ordering",
        "E6 Cross-Path Consistency",
        "E7 Batch Reference Consistency",
        "E8 Reference Agreement",
        "E9 Model Identity",
        "E10 Task Prompt Honoured",
        "E11 Unknown Task Prompt",
    ]

    def __init__(self, base_url, backend_os="linux", model_filter: list[str] | None = None):
        super().__init__(base_url, backend_os, model_filter=model_filter)
        # Keep only recognised embedding models; honour any user-supplied filter.
        self.models = [m for m in self.models if m in self.EMBED_MODELS]
        # OFLM does not always advertise the embed model on /v1/models even when
        # it is loaded (`oflm serve -e 1`). Without an explicit --model filter,
        # fall back to the standard embedding model so the suite still runs; if
        # it is not actually loaded, each check fails gracefully with an ERROR
        # row instead of silently skipping.
        if not model_filter and not self.models:
            self.models = [self.DEFAULT_EMBED_MODEL]
        self.csv_filename = self.get_csv_filename("embedding")
        # (draw, cosine-to-reference) records produced by E7 and dumped to CSV.
        self._reference_draws: list[tuple[list[float], float]] = []
        # Bundled reference vectors from the validated numpy oracle (E8).
        reference = json.loads(self.REFERENCE_FILE.read_text(encoding="utf-8"))
        self._reference_source = reference.get("reference", "an unnamed model")
        self._reference_entries = [
            (entry["text"], entry["embedding"]) for entry in reference["entries"].values()
        ]

    # -------------------------------------------------------------- helpers

    def _embed_response(self, model_id: str, input_text: str | list[str]):
        """One embeddings call; returns the raw response payload."""
        return self.client.embeddings.create(model=model_id, input=input_text)

    def _embed_raw(self, model_id: str, input_text: str, **extra):
        """One embeddings POST outside the SDK; returns (status, body, text).

        E10 and E11 turn on the status code and the error code, and the SDK
        shows neither: it raises on a 4xx and accepts a 200 that carries an
        error body.
        """
        return self._post_json("/embeddings", {"model": model_id, "input": input_text, **extra})

    @staticmethod
    def _raw_vector(body) -> list[float] | None:
        data = body.get("data") if isinstance(body, dict) else None
        if isinstance(data, list) and data and isinstance(data[0], dict):
            vector = data[0].get("embedding")
            if isinstance(vector, list) and vector:
                return vector
        return None

    def _embed(self, model_id: str, input_text: str) -> list[float]:
        """One embeddings call; returns the first vector."""
        response = self._embed_response(model_id, input_text)
        return response.data[0].embedding

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors; 0.0 for degenerate inputs."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _vector_preview(vector: list[float] | None, limit: int = 4) -> str:
        """Compact CSV-friendly preview: first few values plus total length."""
        if not vector:
            return ""
        head = ", ".join(f"{float(v):.6f}" for v in vector[:limit])
        return f"[{head}, ...] ({len(vector)} dims)"

    def _write_row(self, writer, model_id, check_name, input_text, verdict_detail, vector=None):
        verdict, detail = verdict_detail if isinstance(verdict_detail, tuple) else (verdict_detail, "")
        writer.writerow([model_id, check_name, input_text,
                         len(vector) if vector else "N/A",
                         self._vector_preview(vector),
                         f"{verdict}: {detail}"])

    def _run_check(self, writer, model_id, check_name, input_text, check):
        """Runs a (verdict, detail, vector) producing closure with error handling."""
        try:
            result, vector = check()
            verdict, detail = result if isinstance(result, tuple) else (result, "")
            self.record(result, f"{model_id} / {check_name}")
            self._write_row(writer, model_id, check_name, input_text, result, vector)
            print(f"  {check_name}: {verdict} ({detail})")
        except Exception as e:
            print(f"  {check_name}: ERROR ({e})")
            self.record(f"ERROR: {e}", f"{model_id} / {check_name}")
            self._write_row(writer, model_id, check_name, input_text, f"ERROR: {e}")
        time.sleep(1)

    # -------------------------------------------------------------- checks

    @staticmethod
    def _check_response_structure(response):
        """E1: object types, non-empty data and a numeric vector."""
        if getattr(response, "object", None) != "list":
            return ("FAIL", f"expected response object 'list', got "
                            f"'{getattr(response, 'object', None)!r}'"), None
        data = getattr(response, "data", None)
        if not data:
            return ("FAIL", "no embedding data returned"), None
        entry = data[0]
        if getattr(entry, "object", None) != "embedding":
            return ("FAIL", f"expected data object 'embedding', got "
                            f"'{getattr(entry, 'object', None)!r}'"), None
        vector = getattr(entry, "embedding", None)
        if not isinstance(vector, list) or not vector:
            return ("FAIL", "embedding vector is empty or missing"), None
        if not all(isinstance(v, (int, float)) for v in vector):
            return ("FAIL", "embedding vector contains non-numeric values"), None
        return ("PASS", f"valid response with {len(vector)}-dim embedding"), vector

    def _check_repeatability(self, model_id: str):
        """E2: draw the same input N times and count inconsistent pairs.

        A single draw can be an outlier on its own (a build can report either a
        defect or a clean run from one pass), so the verdict is based on the
        outlier *rate* across repeats rather than any single comparison.
        """
        draws = [self._embed(model_id, self.SAMPLE_TEXT) for _ in range(self.REPEAT_COUNT)]
        total_pairs = self.REPEAT_COUNT * (self.REPEAT_COUNT - 1) // 2
        bad_pairs = 0
        worst = 1.0
        for i in range(len(draws)):
            for j in range(i + 1, len(draws)):
                similarity = self._cosine_similarity(draws[i], draws[j])
                worst = min(worst, similarity)
                if similarity < self.STABILITY_THRESHOLD:
                    bad_pairs += 1
        ratio = bad_pairs / total_pairs if total_pairs else 0.0
        if bad_pairs == 0:
            # Report exactness as well as the threshold. A deterministic
            # backend returns the same bytes every time; one that merely stays
            # inside STABILITY_THRESHOLD is doing something non-deterministic
            # that this check would otherwise pass in silence. The verdict is
            # unchanged -- some backends are legitimately non-exact -- but the
            # distinction belongs in the record.
            exact = all(d == draws[0] for d in draws[1:])
            how = ("bit-identical" if exact
                   else f"not bit-identical, worst cosine {worst:.6f}")
            return ("PASS", f"all {self.REPEAT_COUNT} draws consistent "
                            f"({how})"), draws[0]
        if ratio >= self.INCONSISTENT_PAIR_RATIO_FAIL:
            return ("FAIL", f"{bad_pairs}/{total_pairs} draw pairs inconsistent "
                            f"(worst cosine {worst:.6f})"), draws[0]
        return ("SOFT-FAIL", f"{bad_pairs}/{total_pairs} draw pairs inconsistent, "
                             f"single-draw flicker (worst cosine {worst:.6f})"), draws[0]

    def _check_batch_integrity(self, model_id: str):
        """E3: N inputs yield N embeddings in order with unique indexes."""
        response = self._embed_response(model_id, self.BATCH_INPUTS)
        data = getattr(response, "data", None)
        if not data:
            return ("FAIL", "no embedding data returned for batch"), None
        if len(data) != len(self.BATCH_INPUTS):
            return ("FAIL", f"expected {len(self.BATCH_INPUTS)} embeddings, "
                            f"got {len(data)}"), None
        indexes = [getattr(d, "index", None) for d in data]
        if indexes != list(range(len(data))):
            return ("FAIL", f"unexpected data indexes {indexes}"), data[0].embedding
        return ("PASS", f"{len(data)} embeddings returned in order"), data[0].embedding

    def _check_dimensionality(self, model_id: str):
        """E4: every vector in a batch shares the same positive dimension."""
        response = self._embed_response(model_id, self.BATCH_INPUTS)
        data = getattr(response, "data", None)
        if not data:
            return ("FAIL", "no embedding data returned for batch"), None
        dims = {len(getattr(d, "embedding", []) or []) for d in data}
        dim = next(iter(dims))
        if len(dims) != 1 or dim <= 0:
            return ("FAIL", f"inconsistent or invalid dimensions across batch: "
                            f"{sorted(dims)}"), None
        return ("PASS", f"consistent {dim}-dim embeddings across batch"), data[0].embedding

    def _check_semantic_ordering(self, model_id: str):
        """E5: related text pairs should sit closer than unrelated pairs."""
        related = [
            self._cosine_similarity(self._embed(model_id, a), self._embed(model_id, b))
            for a, b in self.RELATED_PAIRS
        ]
        unrelated = [
            self._cosine_similarity(self._embed(model_id, a), self._embed(model_id, b))
            for a, b in self.UNRELATED_PAIRS
        ]
        mean_related = sum(related) / max(len(related), 1)
        mean_unrelated = sum(unrelated) / max(len(unrelated), 1)
        vector = self._embed(model_id, self.RELATED_PAIRS[0][0])
        if mean_related > mean_unrelated:
            return ("PASS", f"related avg {mean_related:.4f} > unrelated avg "
                            f"{mean_unrelated:.4f}"), vector
        return ("FAIL", f"related avg {mean_related:.4f} <= unrelated avg "
                        f"{mean_unrelated:.4f}"), vector

    def _check_cross_path_consistency(self, model_id: str):
        """E6: same weights, two delivery paths in the same run must agree.

        If a bad number can be reproduced through a different API path in the
        same run it is a statement about OFLM, not about a single bad hardware
        draw.
        """
        single = self._embed(model_id, self.SAMPLE_TEXT)
        batch = self._embed_response(model_id, [self.SAMPLE_TEXT])
        data = getattr(batch, "data", None)
        if not data:
            return ("FAIL", "batch delivery path returned no data"), single
        batch_vec = data[0].embedding
        similarity = self._cosine_similarity(single, batch_vec)
        if similarity >= self.AGREEMENT_THRESHOLD:
            return ("PASS", f"single and batch delivery paths agree "
                            f"(cosine {similarity:.6f})"), single
        return ("FAIL", f"single and batch delivery paths disagree "
                        f"(cosine {similarity:.6f})"), single

    def _check_batch_reference_consistency(self, model_id: str):
        """E7: per-draw cosine to the batch-path reference across a larger N.

        A 10-draw all-pairs sample can still hide the shape of an instability.
        E7 draws the input REFERENCE_DRAW_COUNT times and compares every draw
        to the batch-path embedding of the same text, so the outlier rate and
        magnitude surface per-draw. Each (draw, cosine) record is kept in
        self._reference_draws so `run()` can dump the raw vectors to the CSV
        for comparison across builds.
        """
        batch = self._embed_response(model_id, [self.SAMPLE_TEXT])
        data = getattr(batch, "data", None)
        if not data:
            return ("FAIL", "batch-path reference returned no data"), None
        reference = data[0].embedding
        draws = [self._embed(model_id, self.SAMPLE_TEXT)
                 for _ in range(self.REFERENCE_DRAW_COUNT)]
        self._reference_draws = [
            (draw, self._cosine_similarity(draw, reference)) for draw in draws
        ]
        outliers = [c for _, c in self._reference_draws if c < self.STABILITY_THRESHOLD]
        ratio = len(outliers) / len(self._reference_draws)
        worst = min((c for _, c in self._reference_draws), default=1.0)
        if not outliers:
            verdict = ("PASS", f"all {len(draws)} draws agree with the batch "
                               f"reference (worst cosine {worst:.6f})")
        elif ratio >= self.INCONSISTENT_PAIR_RATIO_FAIL:
            verdict = ("FAIL", f"{len(outliers)}/{len(draws)} draws deviate from "
                                f"the batch reference (worst cosine {worst:.6f})")
        else:
            verdict = ("SOFT-FAIL", f"{len(outliers)}/{len(draws)} draws deviate "
                                    f"from the batch reference, single-draw flicker "
                                    f"(worst cosine {worst:.6f})")
        return verdict, reference

    def _check_reference_agreement(self, model_id: str):
        """E8: compare live embeddings against the bundled oracle vectors.

        The reference is the output of the validated numpy implementation of
        the official google/embeddinggemma-300m pipeline, so agreement pins the
        whole API path (prefix, tokenizer, forward pass, pooling, projection,
        normalisation) to a known-good implementation rather than to another
        build of itself.
        """
        if model_id not in self.REFERENCE_MODELS:
            return ("SKIP", f"no bundled reference for '{model_id}' -- the "
                            f"oracle vectors in {self.REFERENCE_FILE.name} are "
                            f"for {self._reference_source}. A cosine between "
                            f"two different models' spaces would mean nothing, "
                            f"so this check reports nothing rather than a "
                            f"failure it cannot substantiate."), None
        worst = 1.0
        worst_text = ""
        for text, expected in self._reference_entries:
            similarity = self._cosine_similarity(self._embed(model_id, text), expected)
            if similarity < worst:
                worst, worst_text = similarity, text
        if worst >= self.REFERENCE_AGREEMENT_THRESHOLD:
            first = self._embed(model_id, self._reference_entries[0][0])
            return ("PASS", f"all {len(self._reference_entries)} texts agree with the "
                            f"oracle reference (worst cosine {worst:.6f})"), first
        return ("FAIL", f"'{worst_text}' deviates from the oracle reference "
                        f"(cosine {worst:.6f})"), self._embed(model_id, worst_text)

    def _check_model_identity(self, model_id: str):
        """E9: the server serves the model it was asked for, or refuses.

        THIS IS THE ONE FAILURE THE CHECKS ABOVE CANNOT SEE. A server that
        ignores the `model` field and answers every request from whichever
        model it happens to have loaded returns a vector that is correctly
        shaped, correctly normed, deterministic, batch-consistent and
        semantically sensible -- so E1 through E7 all pass on it. E8 makes it
        worse rather than better: if the substituted model is the one the
        bundled reference was made from, E8 passes too, and the whole suite
        goes green on an answer for the wrong model.

        It is not hypothetical. This server echoed the requested tag back in
        the response and otherwise ignored it, so a request naming any model
        returned the loaded model vectors under the name that had been asked
        for. Nothing downstream could tell.

        Two halves, both answerable against a running server:
          a) a request naming a model the server cannot have loaded must be
             refused, not answered;
          b) an accepted request must report a model, and it should be the one
             that was asked for.
        """
        try:
            response = self._embed_response(self.IMPOSSIBLE_MODEL, self.SAMPLE_TEXT)
        except Exception:
            pass                    # refused: the correct behaviour
        else:
            data = getattr(response, "data", None)
            vector = getattr(data[0], "embedding", None) if data else None
            if not vector:
                # It did not raise, but it did not embed anything either --
                # most likely an error reported inside a 2xx envelope. That is
                # its own contract problem, but it is not a substitution, so
                # do not accuse it of one.
                return ("SOFT-FAIL", f"a request for "
                                     f"'{self.IMPOSSIBLE_MODEL}' did not raise "
                                     f"but returned no embedding data; the "
                                     f"model was refused inside a success "
                                     f"envelope rather than by an error "
                                     f"status"), None
            return ("FAIL", f"server answered a request for "
                            f"'{self.IMPOSSIBLE_MODEL}', a model it cannot "
                            f"have loaded, with a {len(vector)}-dim vector, "
                            f"reporting it as "
                            f"'{getattr(response, 'model', None)}'. It is "
                            f"serving some other model's vectors under the "
                            f"requested name, and every other check in this "
                            f"suite would pass on them."), vector

        response = self._embed_response(model_id, self.SAMPLE_TEXT)
        data = getattr(response, "data", None)
        vector = getattr(data[0], "embedding", None) if data else None
        reported = getattr(response, "model", None)
        if not reported:
            return ("SOFT-FAIL", "unknown models are refused, but the response "
                                 "names no model, so a client cannot confirm "
                                 "which one answered"), vector
        if reported != model_id:
            return ("SOFT-FAIL", f"requested '{model_id}', response reports "
                                 f"'{reported}'"), vector
        return ("PASS", f"unknown model refused; '{model_id}' served and "
                        f"reported as itself"), vector

    def _check_task_prompt_honoured(self, model_id: str):
        """E10: the task prompt in the request reaches the model.

        This server passed task_query unconditionally, so every document was
        embedded as a query. The vector was correctly shaped, correctly normed
        and deterministic either way, and E1 through E9 all pass on it -- the
        only visible difference is against the same text under the other prompt.
        No oracle vectors needed: the assertion is that the field does
        something, not that it produces any particular number.

        Two refusals here are deliberate and are SKIPs rather than failures: a
        model with no task concept (the BERT family) refuses a prompt name
        outright, and a model whose prompts serve neither query nor document
        refuses to pick one. Both are the behaviour the fix kept on purpose.
        """
        status, body, text = self._embed_raw(model_id, self.SAMPLE_TEXT,
                                             prompt_name=self.QUERY_PROMPT)
        if status >= 400:
            code, _, message = self._error_fields(body)
            if "no task prompts" in (message or ""):
                return ("SKIP", f"'{model_id}' has no task prompts and refused one, "
                                f"which is what it should do"), None
            if "none of them matches" in (message or ""):
                return ("SKIP", f"'{model_id}' declares task prompts but none serves "
                                f"'{self.QUERY_PROMPT}', and it refused rather than picking "
                                f"one, which is what it should do"), None
            return ("FAIL", f"'{self.QUERY_PROMPT}' was refused with HTTP {status} "
                            f"({code}): {(message or text)[:160]}"), None
        query_vector = self._raw_vector(body)
        if not query_vector:
            return ("FAIL", f"HTTP {status} carried no embedding: {text[:160]}"), None

        status, body, text = self._embed_raw(model_id, self.SAMPLE_TEXT,
                                             prompt_name=self.DOCUMENT_PROMPT)
        document_vector = self._raw_vector(body)
        if status >= 400 or not document_vector:
            return ("FAIL", f"'{self.DOCUMENT_PROMPT}' returned HTTP {status} with no "
                            f"embedding: {text[:160]}"), query_vector

        across = self._cosine_similarity(query_vector, document_vector)
        if across >= self.STABILITY_THRESHOLD:
            return ("FAIL", f"the same text under '{self.QUERY_PROMPT}' and "
                            f"'{self.DOCUMENT_PROMPT}' embeds to cosine {across:.6f}: the "
                            f"request's task prompt is being dropped, and every vector "
                            f"comes back under whichever task the server picked"), query_vector

        repeat_status, repeat_body, repeat_text = self._embed_raw(
            model_id, self.SAMPLE_TEXT, prompt_name=self.QUERY_PROMPT)
        repeat_vector = self._raw_vector(repeat_body)
        if repeat_status >= 400 or not repeat_vector:
            # Say the draw failed rather than blaming noise for a request that
            # never came back.
            return ("SOFT-FAIL", f"the two prompts differ (cosine {across:.6f}), but the repeat "
                                 f"draw of '{self.QUERY_PROMPT}' returned HTTP {repeat_status} "
                                 f"with no embedding, so this run cannot rule out noise: "
                                 f"{repeat_text[:160]}"), query_vector
        within = self._cosine_similarity(query_vector, repeat_vector)
        if within < self.STABILITY_THRESHOLD:
            # The prompts differ, but so do two draws of the same prompt, so the
            # difference above is not evidence the field was read.
            return ("SOFT-FAIL", f"the two prompts differ (cosine {across:.6f}) but so do two "
                                 f"draws of '{self.QUERY_PROMPT}' (cosine {within:.6f}), so "
                                 f"this run cannot tell the prompt apart from noise"), query_vector
        return ("PASS", f"'{self.QUERY_PROMPT}' against '{self.DOCUMENT_PROMPT}' measures "
                        f"cosine {across:.6f}, repeat draws {within:.6f}"), query_vector

    def _check_unknown_task_prompt(self, model_id: str):
        """E11: a task prompt the server cannot resolve is refused.

        Falling back to the default is the same defect one step quieter: the
        caller asked for one task, got another, and nothing in the response
        says so.
        """
        status, body, text = self._embed_raw(model_id, self.SAMPLE_TEXT,
                                             prompt_name=self.IMPOSSIBLE_PROMPT)
        vector = self._raw_vector(body)
        if status < 400 and vector:
            return ("FAIL", f"'{self.IMPOSSIBLE_PROMPT}' was accepted with HTTP {status} and "
                            f"answered with a {len(vector)}-dim vector under some other task "
                            f"prompt; nothing downstream can tell"), vector
        if status < 400:
            return ("SOFT-FAIL", f"'{self.IMPOSSIBLE_PROMPT}' was refused inside an HTTP "
                                 f"{status} success envelope rather than by an error "
                                 f"status: {text[:160]}"), None
        code, error_type, message = self._error_fields(body)
        if not code or not error_type:
            return ("SOFT-FAIL", f"refused with HTTP {status}, but the body is not an "
                                 f"OpenAI-shaped error: {text[:160]}"), None
        return ("PASS", f"refused with HTTP {status} {code}"), None

    def _reference_draw_rows(self, model_id: str, check_name: str):
        """One CSV row per E7 draw, carrying the full raw vector for diffing."""
        rows = []
        total = len(self._reference_draws)
        for idx, (vector, cosine) in enumerate(self._reference_draws, 1):
            dims = len(vector) if vector else 0
            full = json.dumps(vector)
            rows.append([
                model_id,
                f"{check_name} (draw {idx}/{total})",
                self.SAMPLE_TEXT,
                dims,
                full,
                f"cosine vs batch reference: {cosine:.6f}",
            ])
        return rows

    def run(self):
        print("\n=== Starting Embedding Tests ===")
        print(f"Models found: {len(self.models)}")
        if not self.models:
            print("No embedding models found. Start the server with the embed model "
                  "loaded, e.g. `oflm serve -e 1`.")
            print(f"Embedding tests complete. Saved to {self.csv_filename}")
            return self.result

        with open(self.csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Model", "Check", "Input", "Embedding Dim", "Vector Preview", "Check Result"])
            for i, model_id in enumerate(self.models, 1):
                print(f"\n--- Testing embedding model ({i}/{len(self.models)}): {model_id} ---")
                self._run_check(writer, model_id, self.CHECK_NAMES[0], self.SAMPLE_TEXT,
                                lambda: self._check_response_structure(
                                    self._embed_response(model_id, self.SAMPLE_TEXT)))
                self._run_check(writer, model_id, self.CHECK_NAMES[1], self.SAMPLE_TEXT,
                                lambda: self._check_repeatability(model_id))
                self._run_check(writer, model_id, self.CHECK_NAMES[2], json.dumps(self.BATCH_INPUTS),
                                lambda: self._check_batch_integrity(model_id))
                self._run_check(writer, model_id, self.CHECK_NAMES[3], json.dumps(self.BATCH_INPUTS),
                                lambda: self._check_dimensionality(model_id))
                self._run_check(writer, model_id, self.CHECK_NAMES[4], json.dumps(self.RELATED_PAIRS),
                                lambda: self._check_semantic_ordering(model_id))
                self._run_check(writer, model_id, self.CHECK_NAMES[5], self.SAMPLE_TEXT,
                                lambda: self._check_cross_path_consistency(model_id))
                self._run_check(writer, model_id, self.CHECK_NAMES[6], self.SAMPLE_TEXT,
                                lambda: self._check_batch_reference_consistency(model_id))
                for row in self._reference_draw_rows(model_id, self.CHECK_NAMES[6]):
                    writer.writerow(row)
                self._run_check(writer, model_id, self.CHECK_NAMES[7],
                                "bundle: oracle reference texts",
                                lambda: self._check_reference_agreement(model_id))
                self._run_check(writer, model_id, self.CHECK_NAMES[8],
                                self.IMPOSSIBLE_MODEL,
                                lambda: self._check_model_identity(model_id))
                self._run_check(writer, model_id, self.CHECK_NAMES[9],
                                f"{self.QUERY_PROMPT} vs {self.DOCUMENT_PROMPT}",
                                lambda: self._check_task_prompt_honoured(model_id))
                self._run_check(writer, model_id, self.CHECK_NAMES[10],
                                self.IMPOSSIBLE_PROMPT,
                                lambda: self._check_unknown_task_prompt(model_id))
                print(f"Finished testing model: {model_id}")
        print(f"\nEmbedding tests complete. Saved to {self.csv_filename}")
        return self.result

class AudioTask(BaseTestTask):
    SUITE_NAME = "audio"

    AUDIO_MODELS = ["whisper-v3:turbo"]

    def __init__(self, base_url, backend_os="linux", model_filter: list[str] | None = None):
        super().__init__(base_url, backend_os, model_filter=model_filter)
        # Without an explicit --model filter, keep only recognised audio models;
        # an explicit filter always wins so any audio-capable model can be tested.
        if not model_filter:
            self.models = [m for m in self.models if m in self.AUDIO_MODELS]
        self.test_audio_path = PACKAGE_DIR / "test_files" / "audio" / "atomic-germ.mp3"
        self.csv_filename = self.get_csv_filename("audio")

    def _load_audio_base64(self, audio_path) -> str:
        with open(audio_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode('utf-8')

    def run(self, max_generation_tokens=-1, temperature=0.3, reasoning=None):
        prompt = "Describe what you hear in this audio clip."
        followup_prompt = "What kind of mood or genre would this clip fit into?"

        audio_b64 = self._load_audio_base64(self.test_audio_path)
        reasoning_kwargs = self._reasoning_kwargs(reasoning)

        with open(self.csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Model", "Input", "Reasoning Content", "Output Content", "Music Mention Check"])
            print("\n=== Starting Audio Tests ===")
            print(f"Models found: {len(self.models)}")
            for i, model_id in enumerate(self.models, 1):
                print(f"\n--- Testing audio models ({i}/{len(self.models)}): {model_id} ---")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
                        ],
                    }
                ]

                # first round
                try:
                    print(f"Prompt: {prompt}")
                    response = self.client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        stream=True,
                        max_completion_tokens=max_generation_tokens,
                        temperature=temperature,
                        **reasoning_kwargs,
                    )
                    reasoning_content, output_content = self._collect_stream(response)
                    music_check = "PASS" if self.MUSIC_PATTERN.search(output_content) else "SOFT-FAIL"
                    self.record(music_check, f"{model_id} / Music Mention")
                    writer.writerow([model_id, prompt, reasoning_content or "N/A", output_content, music_check])
                    if music_check == "PASS":
                        print("Music mention check: PASS")
                    else:
                        print("Music mention check: SOFT-FAIL (noted only, not a hard failure)")
                    print("Done.")
                    time.sleep(1)
                    messages.append({"role": "assistant", "content": output_content})
                    messages.append({"role": "user", "content": followup_prompt})
                except Exception as e:
                    print(f"Error occurred in first round, model: {model_id}: {e}")
                    self.record(f"ERROR: {e}", f"{model_id} / audio round 1")
                    writer.writerow([model_id, prompt, f"ERROR: {e}", "N/A", "ERROR"])

                # second round
                try:
                    print(f"Follow-up Prompt: {followup_prompt}")
                    response = self.client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        stream=True,
                        max_completion_tokens=max_generation_tokens,
                        temperature=temperature,
                        **reasoning_kwargs,
                    )
                    reasoning_content, output_content = self._collect_stream(response)
                    writer.writerow([model_id, followup_prompt, reasoning_content or "N/A", output_content, "N/A"])
                    print("Done.")
                    time.sleep(1)
                except Exception as e:
                    print(f"Error occurred in second round, model: {model_id}: {e}")
                    self.record(f"ERROR: {e}", f"{model_id} / audio round 2")
                    writer.writerow([model_id, followup_prompt, f"ERROR: {e}", "N/A", "N/A"])
                print(f"Finished testing model: {model_id}")
        print(f"Audio tests complete. Saved to {self.csv_filename}")
        return self.result

class VisionTask(BaseTestTask):
    SUITE_NAME = "vision"

    EXPECTED_TEXT = ("The capital of France is Paris. It is a major global city "
                     "and serves as the nation's center for finance, commerce, "
                     "culture, arts, fashion, and science.")
    SEAGULL_PATTERN = re.compile(r"\b(?:sea[\s-]?)?gulls?\b", re.IGNORECASE)

    def __init__(self, base_url, backend_os="linux", model_filter: list[str] | None = None):
        super().__init__(base_url, backend_os, model_filter=model_filter)
        self.test_image1_path = PACKAGE_DIR / "test_files" / "image" / "paris.png"
        self.test_image2_path = PACKAGE_DIR / "test_files" / "image" / "seagull.jpeg"
        self.test_image3_path = PACKAGE_DIR / "test_files" / "image" / "spectrogram.png"
        self.csv_filename = self.get_csv_filename("vision")
        # Tolerant regex: case-insensitive, flexible whitespace, optional apostrophe.
        escaped_words = [re.escape(word) for word in self.EXPECTED_TEXT.split()]
        escaped_words = [w.replace(r"\'", "'?").replace("'", "'?") for w in escaped_words]
        self.expected_text_pattern = re.compile(
            r"\s+".join(escaped_words),
            re.IGNORECASE,
        )
        for i, model in enumerate(self.models, 1):
            print(f"  {i}. {model}")

    def _load_image_base64(self, image_path) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def run(self, max_generation_tokens=-1, temperature=0.3, reasoning=None):
        prompt = "Extract text from the first image, describe the second one, and imagine what the spectrogram might sound like."
        followup_prompt = "Make a story that connects the images together."
        followup_prompt_music = "What kind of sound does the spectrogram represent?"

        image1_b64 = self._load_image_base64(self.test_image1_path)
        image2_b64 = self._load_image_base64(self.test_image2_path)
        image3_b64 = self._load_image_base64(self.test_image3_path)
        reasoning_kwargs = self._reasoning_kwargs(reasoning)

        with open(self.csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Model", "Input", "Reasoning Content", "Output Content",
                             "Text Extraction Check", "Seagull Mention Check", "Spectrogram Music Check"])
            print("\n=== Starting Vision Tests ===")
            print(f"Models found: {len(self.models)}")
            for model_id in self.models:
                print(f"\n--- Testing VLMs: {model_id} ---")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image1_b64}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{image2_b64}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{image3_b64}"}},
                        ],
                    }
                ]

                # first round
                seagull_in_description = "ERROR"
                music_in_description = "ERROR"
                try:
                    print(f"Prompt: {prompt}")
                    response = self.client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        stream=True,
                        max_completion_tokens=max_generation_tokens,
                        temperature=temperature,
                        **reasoning_kwargs,
                    )
                    reasoning_content, output_content = self._collect_stream(response)
                    text_check = "PASS" if self.expected_text_pattern.search(output_content) else "FAIL"
                    self.record(text_check, f"{model_id} / Text Extraction")
                    if text_check == "PASS":
                        print("Text extraction check: PASS")
                    else:
                        print("Text extraction check: FAIL (expected text not found in response)")
                    seagull_in_description = "PASS" if self.SEAGULL_PATTERN.search(output_content) else "FAIL"
                    music_in_description = "PASS" if self.MUSIC_PATTERN.search(output_content) else "FAIL"
                    writer.writerow([model_id, prompt, reasoning_content or "N/A", output_content,
                                     text_check, seagull_in_description, music_in_description])
                    print("Done.")
                    time.sleep(1)
                    messages.append({"role": "assistant", "content": output_content})
                    messages.append({"role": "user", "content": followup_prompt})
                except Exception as e:
                    print(f"Error occurred in first round, model: {model_id}: {e}")
                    self.record(f"ERROR: {e}", f"{model_id} / vision round 1")
                    writer.writerow([model_id, prompt, f"ERROR: {e}", "N/A", "ERROR", "ERROR", "ERROR"])

                # second round
                round2_ok = False
                try:
                    print(f"Follow-up Prompt: {followup_prompt}")
                    response = self.client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        stream=True,
                        max_completion_tokens=max_generation_tokens,
                        temperature=temperature,
                        **reasoning_kwargs,
                    )
                    reasoning_content, output_content = self._collect_stream(response)
                    seagull_in_story = "PASS" if self.SEAGULL_PATTERN.search(output_content) else "FAIL"
                    writer.writerow([model_id, followup_prompt, reasoning_content or "N/A", output_content,
                                     "N/A", seagull_in_story, "N/A"])
                    self.record("PASS" if "PASS" in (seagull_in_story, seagull_in_description)
                                else "FAIL", f"{model_id} / Seagull Mention")
                    if seagull_in_story == "PASS":
                        print("Seagull check (story): PASS")
                    elif seagull_in_description == "PASS":
                        print("Seagull check: PASS (recognized in description)")
                    else:
                        print("Seagull check: FAIL (no mention of a seagull in story or description)")
                    print("Done.")
                    time.sleep(1)
                    messages.append({"role": "assistant", "content": output_content})
                    messages.append({"role": "user", "content": followup_prompt_music})
                    round2_ok = True
                except Exception as e:
                    print(f"Error occurred in second round, model: {model_id}: {e}")
                    self.record(f"ERROR: {e}", f"{model_id} / vision round 2")
                    writer.writerow([model_id, followup_prompt, f"ERROR: {e}", "N/A", "N/A", "ERROR", "N/A"])

                # third round (spectrogram; informational, not a hard failure)
                music_check = "SKIPPED"
                if round2_ok:
                    try:
                        print(f"Follow-up Prompt: {followup_prompt_music}")
                        response = self.client.chat.completions.create(
                            model=model_id,
                            messages=messages,
                            stream=True,
                            max_completion_tokens=max_generation_tokens,
                            temperature=temperature,
                        )
                        reasoning_content, output_content = self._collect_stream(response)
                        music_check = "PASS" if self.MUSIC_PATTERN.search(output_content) else "SOFT-FAIL"
                        writer.writerow([model_id, followup_prompt_music, reasoning_content or "N/A",
                                         output_content, "N/A", "N/A", music_check])
                        self.record("PASS" if "PASS" in (music_check, music_in_description)
                                    else "SOFT-FAIL", f"{model_id} / Spectrogram Music")
                        if music_check == "PASS":
                            print("Spectrogram music check: PASS")
                        elif music_in_description == "PASS":
                            print("Spectrogram music check: PASS (recognized in description)")
                        else:
                            print("Spectrogram music check: SOFT-FAIL (noted only, not a hard failure)")
                        print("Done.")
                        time.sleep(1)
                    except Exception as e:
                        print(f"Error occurred in third round, model: {model_id}: {e}")
                        self.record(f"ERROR: {e}", f"{model_id} / vision round 3")
                        writer.writerow([model_id, followup_prompt_music, f"ERROR: {e}", "N/A",
                                         "N/A", "N/A", "ERROR"])
                else:
                    writer.writerow([model_id, followup_prompt_music,
                                     "SKIPPED: second round failed", "N/A", "N/A", "SKIPPED"])
                print(f"Finished testing model: {model_id}")
        print(f"Vision tests complete. Saved to {self.csv_filename}")
        return self.result


class ToolCallingTask(BaseTestTask):
    """
    Tests OpenAI-compatible function/tool calling at seven escalating
    complexity levels, each run in both streaming and non-streaming mode:

      L1 Basic Tool Call      One obvious call whose arguments appear verbatim
                              in the prompt.
      L2 Argument Extraction  The argument must be inferred from an indirect
                              reference, with a decoy forecast tool present.
      L3 Tool Restraint       A question answerable without tools; the model
                              should not call anything (negative control).
      L4 Parallel Tool Calls  Several independent calls belong in one turn.
      L5 Multi-Turn Tool Loop The model must call a tool, consume the locally
                              executed result, and ground its final answer in it.
      L6 Tool Result Fidelity A code in the last few tokens of a tool result must
                              come back verbatim, which prompt trimming that eats
                              the tail of the result would break.
      L7 Nullable Schema      A parameter typed ["string", "null"] must not fail
                              the request, and the tool must still be called.

    Automated checks validate tool names, JSON argument validity/values and,
    for L5, the arithmetic derived from the tool result ($54 total).
    """

    SUITE_NAME = "tools"

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name, e.g. 'Paris'."},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                                 "description": "Temperature unit; defaults to celsius."},
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather_forecast",
                "description": "Get the multi-day weather forecast for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name, e.g. 'Paris'."},
                        "days": {"type": "integer", "description": "Number of forecast days."},
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_item_price",
                "description": "Look up the unit price of an item in the store catalog.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string", "description": "Item name, e.g. 'widget'."},
                    },
                    "required": ["item"],
                },
            },
        },
    ]

    # Deterministic mock backing stores so the loop level needs no external services.
    WEATHER_DB = {
        "paris": {"temperature_celsius": 18, "conditions": "partly cloudy", "humidity_percent": 63},
        "tokyo": {"temperature_celsius": 24, "conditions": "sunny", "humidity_percent": 55},
    }
    PRICE_DB = {"widget": 20.0, "gadget": 49.99, "doohickey": 7.25}
    DEFAULT_TEMPERATURE_CELSIUS = 21
    DEFAULT_PRICE_USD = 9.99

    BASIC_PROMPT = "What's the current weather in Paris right now? Use the tools available to you."
    EXTRACTION_PROMPT = ("A friend of mine lives in the city where the Eiffel Tower stands. "
                         "Use your tools to tell me the current weather there.")
    RESTRAINT_PROMPT = ("Do not call any tools. Answer directly from your own knowledge: "
                        "what is the capital of France?")
    PARALLEL_PROMPT = "Using your tools, compare the current weather in Paris and Tokyo."
    LOOP_PROMPT = ("Use the price lookup tool to check the unit price of a 'widget', then tell me "
                   "what 3 widgets would cost after a 10% discount. Do the math yourself.")
    # L6: the value the model must repeat sits in the last few tokens of the tool result
    # (templates that sort keys put zz_code last), so any truncation of the result shows up.
    FIDELITY_PROMPT = ("Fetch ticket 42 and reply with ONLY the exact value of its `zz_code` field, "
                       "nothing else.")
    FIDELITY_TOOL = {
        "type": "function",
        "function": {
            "name": "get_ticket",
            "description": "Fetch a support ticket by id.",
            "parameters": {
                "type": "object",
                "properties": {"ticket_id": {"type": "string", "description": "Ticket id, e.g. '42'."}},
                "required": ["ticket_id"],
            },
        },
    }
    FIDELITY_RESULT = {"a_status": "open", "zz_code": "ZQX-7731"}
    # L7: a JSON-schema type array, the shape Pydantic optionals and most MCP servers emit
    NULLABLE_PROMPT = "Search my notes for 'budget' in any folder. Use the tool."
    NULLABLE_TOOL = {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Search the user's notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search text."},
                    "folder": {"type": ["string", "null"], "description": "Folder to search, or null for all."},
                },
                "required": ["query"],
            },
        },
    }

    LEVEL_NAMES = [
        "L1 Basic Tool Call",
        "L2 Argument Extraction",
        "L3 Tool Restraint",
        "L4 Parallel Tool Calls",
        "L5 Multi-Turn Tool Loop",
        "L6 Tool Result Fidelity",
        "L7 Nullable Schema",
    ]

    FINAL_ANSWER_PATTERN = re.compile(r"\b54(\.0{1,2})?\b")
    MAX_TOOL_ROUNDS = 3

    def __init__(self, base_url, backend_os="linux", model_filter: list[str] | None = None):
        super().__init__(base_url, backend_os, model_filter=model_filter)
        self.models = [m for m in self.models if m not in NON_CHAT_MODELS]
        self.csv_filename = self.get_csv_filename("tools")

    @staticmethod
    def _parse_arguments(raw) -> dict | None:
        """Normalises tool-call arguments into a dict; None if not valid JSON object."""
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    @staticmethod
    def _format_tool_calls(tool_calls: list[dict]) -> str:
        """Compact CSV-friendly summary of requested tool calls."""
        if not tool_calls:
            return "None"
        formatted = []
        for tc in tool_calls:
            args = ToolCallingTask._parse_arguments(tc.get("arguments"))
            formatted.append({"name": tc.get("name"),
                              "arguments": args if args is not None else tc.get("arguments")})
        return json.dumps(formatted, ensure_ascii=False)

    def _execute_tool(self, name: str, arguments: dict) -> str:
        """Local mock implementations backing every advertised tool."""
        if name == "get_current_weather":
            location = str(arguments.get("location", "unknown"))
            data = dict(self.WEATHER_DB.get(location.strip().lower(),
                                            {"temperature_celsius": self.DEFAULT_TEMPERATURE_CELSIUS,
                                             "conditions": "sunny",
                                             "humidity_percent": 40}))
            data["location"] = location
            if arguments.get("unit") == "fahrenheit":
                data["temperature_fahrenheit"] = round(data["temperature_celsius"] * 9 / 5 + 32, 1)
            return json.dumps(data)
        if name == "get_weather_forecast":
            location = str(arguments.get("location", "unknown"))
            return json.dumps({
                "location": location,
                "forecast": [
                    {"day": i + 1,
                     "high_celsius": 20 + (i % 3),
                     "low_celsius": 12 + (i % 2),
                     "conditions": "sunny" if i % 2 == 0 else "cloudy"}
                    for i in range(int(arguments.get("days") or 3))
                ],
            })
        if name == "lookup_item_price":
            item = str(arguments.get("item", "unknown"))
            price = self.PRICE_DB.get(item.strip().lower(), self.DEFAULT_PRICE_USD)
            return json.dumps({"item": item, "unit_price_usd": price})
        return json.dumps({"error": f"unknown tool '{name}'"})

    def _collect_stream_with_tools(self, response) -> tuple[str, str, list[dict]]:
        """Like _collect_stream but also assembles streamed tool-call deltas."""
        reasoning_content, output_content = "", ""
        accumulated: dict[int, dict] = {}
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "reasoning_content", None):
                reasoning_content += delta.reasoning_content
            if delta.content:
                output_content += delta.content
            for tc_delta in (getattr(delta, "tool_calls", None) or []):
                index = tc_delta.index if tc_delta.index is not None else len(accumulated)
                entry = accumulated.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if tc_delta.id:
                    entry["id"] += tc_delta.id
                fn = getattr(tc_delta, "function", None)
                if fn is not None:
                    if fn.name:
                        entry["name"] += fn.name
                    if fn.arguments:
                        entry["arguments"] += fn.arguments
        tool_calls = [{"id": entry["id"], "name": entry["name"], "arguments": entry["arguments"]}
                      for _, entry in sorted(accumulated.items())]
        return reasoning_content, output_content, tool_calls

    def _call_model(self, model_id, messages, stream, max_completion_tokens, temperature, reasoning=None,
                    tools=None):
        """One chat completion with tools bound; returns (reasoning, content, tool_calls)."""
        response = self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=tools if tools is not None else self.TOOLS,
            stream=stream,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            **self._reasoning_kwargs(reasoning),
        )
        if stream:
            return self._collect_stream_with_tools(response)
        message = response.choices[0].message
        tool_calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            for tc in (message.tool_calls or [])
        ]
        reasoning = getattr(message, "reasoning_content", None) or ""
        return reasoning, message.content or "", tool_calls

    # ------------------------------------------------------------------ checks

    def _verify_weather_call(self, tool_calls, expected_city_fragment):
        """Shared check for L1/L2: correct tool, valid JSON, expected location."""
        if not tool_calls:
            return ("FAIL", "no tool call issued")
        first = tool_calls[0]
        if first["name"] != "get_current_weather":
            return ("FAIL", f"expected 'get_current_weather', got '{first['name'] or '<empty>'}'")
        args = self._parse_arguments(first["arguments"])
        if args is None:
            return ("FAIL", "tool arguments are not a valid JSON object")
        location = str(args.get("location", ""))
        if expected_city_fragment.lower() not in location.lower():
            return ("FAIL", f"expected location to contain '{expected_city_fragment}', got '{location}'")
        return ("PASS", f"called get_current_weather with location '{location}'")

    def _check_restraint(self, content, tool_calls):
        """L3 negative control: no tool calls, and a direct Paris answer."""
        if tool_calls:
            names = [tc["name"] for tc in tool_calls]
            return ("FAIL", f"called {names} when no tool was needed")
        if not content.strip():
            return ("FAIL", "empty response")
        if "paris" in content.lower():
            return ("PASS", "answered directly without any tool call")
        return ("SOFT-FAIL", "no tool call, but answer did not mention Paris")

    def _check_parallel(self, content, tool_calls):
        """L4: at least two calls covering both requested cities."""
        if not tool_calls:
            return ("FAIL", "no tool call issued")
        locations = []
        for tc in tool_calls:
            if tc["name"] != "get_current_weather":
                continue
            args = self._parse_arguments(tc["arguments"])
            if args:
                locations.append(str(args.get("location", "")).lower())
        has_paris = any("paris" in loc for loc in locations)
        has_tokyo = any("tokyo" in loc for loc in locations)
        if len(tool_calls) >= 2 and has_paris and has_tokyo:
            return ("PASS", f"{len(tool_calls)} parallel calls covering both cities")
        if len(tool_calls) == 1:
            return ("SOFT-FAIL", "only one call issued; parallel calling not exercised")
        return ("FAIL", "calls did not cover both requested cities")

    def _check_lookup_call(self, tool_calls):
        """L5 first round: price lookup requested for a widget."""
        if not tool_calls:
            return ("FAIL", "no tool call issued for the price lookup")
        first = tool_calls[0]
        if first["name"] != "lookup_item_price":
            return ("FAIL", f"expected 'lookup_item_price', got '{first['name'] or '<empty>'}'")
        args = self._parse_arguments(first["arguments"])
        if args is None:
            return ("FAIL", "tool arguments are not a valid JSON object")
        item = str(args.get("item", ""))
        if "widget" not in item.lower():
            return ("FAIL", f"expected item to contain 'widget', got '{item}'")
        return ("PASS", f"requested price for '{item}'")

    def _check_final_answer(self, content):
        """L5 last round: final answer reflects 3 x $20 less 10% => $54."""
        if self.FINAL_ANSWER_PATTERN.search(content):
            return ("PASS", "final answer contains the computed total ($54)")
        return ("FAIL", "computed total $54 not found in final answer")

    # ----------------------------------------------------------------- runners

    def _write_row(self, writer, model_id, level_name, mode, prompt,
                   reasoning, content, tool_calls, verdict_detail):
        verdict, detail = verdict_detail if isinstance(verdict_detail, tuple) else (verdict_detail, "")
        self.record(verdict_detail, f"{model_id} / {level_name} / {mode}")
        writer.writerow([model_id, level_name, mode, prompt,
                         reasoning or "N/A", content or "N/A",
                         self._format_tool_calls(tool_calls),
                         verdict if not detail else f"{verdict}: {detail}"])

    def _run_single_round_level(self, writer, model_id, level_name, prompt, checker,
                                stream, max_completion_tokens, temperature, reasoning=None):
        mode = "Stream" if stream else "Non-Stream"
        messages = [{"role": "user", "content": prompt}]
        try:
            print(f"{level_name}: {prompt}")
            reasoning_content, content, tool_calls = self._call_model(
                model_id, messages, stream, max_completion_tokens, temperature, reasoning)
            verdict = checker(content, tool_calls)
            self._write_row(writer, model_id, level_name, mode, prompt,
                            reasoning_content, content, tool_calls, verdict)
            print(f"Check result: {verdict[0]} ({verdict[1]})")
        except Exception as e:
            print(f"Error occurred in {level_name}, model: {model_id}: {e}")
            self._write_row(writer, model_id, level_name, mode, prompt,
                            f"ERROR: {e}", "N/A", [], "ERROR")
        time.sleep(1)

    def _run_loop_level(self, writer, model_id, stream, max_completion_tokens, temperature, reasoning=None):
        level_name = self.LEVEL_NAMES[4]
        mode = "Stream" if stream else "Non-Stream"
        followup_label = f"{self.LOOP_PROMPT} [with tool result fed back]"
        messages = [{"role": "user", "content": self.LOOP_PROMPT}]
        wrote_first_row = False
        try:
            print(f"{level_name}: {self.LOOP_PROMPT}")
            finished = False
            for round_index in range(self.MAX_TOOL_ROUNDS):
                reasoning_content, content, tool_calls = self._call_model(
                    model_id, messages, stream, max_completion_tokens, temperature, reasoning)
                if round_index == 0:
                    verdict = self._check_lookup_call(tool_calls)
                    self._write_row(writer, model_id, level_name, mode, self.LOOP_PROMPT,
                                    reasoning_content, content, tool_calls, verdict)
                    wrote_first_row = True
                    print(f"Lookup check result: {verdict[0]} ({verdict[1]})")
                if not tool_calls:
                    verdict = self._check_final_answer(content)
                    self._write_row(writer, model_id, level_name, mode,
                                    self.LOOP_PROMPT if round_index == 0 else followup_label,
                                    reasoning_content, content, tool_calls, verdict)
                    print(f"Final answer check result: {verdict[0]} ({verdict[1]})")
                    finished = True
                    break
                messages.append({
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {"id": tc["id"], "type": "function",
                         "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                        for tc in tool_calls
                    ],
                })
                for tc in tool_calls:
                    args = self._parse_arguments(tc["arguments"]) or {}
                    result = self._execute_tool(tc["name"], args)
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                time.sleep(1)
            if not finished:
                verdict = ("FAIL", f"still requesting tools after {self.MAX_TOOL_ROUNDS} rounds")
                self._write_row(writer, model_id, level_name, mode, followup_label,
                                reasoning_content, content, tool_calls, verdict)
                print(f"Final answer check result: FAIL ({verdict[1]})")
        except Exception as e:
            print(f"Error occurred in {level_name}, model: {model_id}: {e}")
            label = followup_label if wrote_first_row else self.LOOP_PROMPT
            self._write_row(writer, model_id, level_name, mode, label,
                            f"ERROR: {e}", "N/A", [], "ERROR")
        time.sleep(1)

    def _check_fidelity(self, content, tool_calls):
        """L6: the tail value of the tool result is reproduced verbatim, with no further call."""
        if tool_calls:
            return ("FAIL", f"called {[tc['name'] for tc in tool_calls]} instead of answering from the result")
        if self.FIDELITY_RESULT["zz_code"] in (content or ""):
            return ("PASS", "tool result reached the model intact")
        return ("FAIL", f"expected '{self.FIDELITY_RESULT['zz_code']}' in the answer, got '{(content or '')[:80]}'")

    def _check_nullable(self, content, tool_calls):
        """L7: the request survives a type-array schema and the tool gets called."""
        if not tool_calls:
            return ("FAIL", "no tool call issued")
        first = tool_calls[0]
        if first["name"] != "search_notes":
            return ("FAIL", f"expected 'search_notes', got '{first['name'] or '<empty>'}'")
        args = self._parse_arguments(first["arguments"])
        if args is None:
            return ("FAIL", "tool arguments are not a valid JSON object")
        if "budget" not in str(args.get("query", "")).lower():
            return ("FAIL", f"expected query to mention 'budget', got '{args.get('query')}'")
        return ("PASS", f"called search_notes with query '{args.get('query')}'")

    def _run_fidelity_level(self, writer, model_id, stream, max_completion_tokens, temperature, reasoning=None):
        level_name = self.LEVEL_NAMES[5]
        mode = "Stream" if stream else "Non-Stream"
        messages = [
            {"role": "user", "content": self.FIDELITY_PROMPT},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_fidelity_1", "type": "function",
                 "function": {"name": "get_ticket", "arguments": json.dumps({"ticket_id": "42"})}}]},
            {"role": "tool", "tool_call_id": "call_fidelity_1", "content": json.dumps(self.FIDELITY_RESULT)},
        ]
        label = f"{self.FIDELITY_PROMPT} [with tool result fed back]"
        try:
            print(f"{level_name}: {self.FIDELITY_PROMPT}")
            reasoning_content, content, tool_calls = self._call_model(
                model_id, messages, stream, max_completion_tokens, temperature, reasoning,
                tools=self.TOOLS + [self.FIDELITY_TOOL])
            verdict = self._check_fidelity(content, tool_calls)
            self._write_row(writer, model_id, level_name, mode, label,
                            reasoning_content, content, tool_calls, verdict)
            print(f"Check result: {verdict[0]} ({verdict[1]})")
        except Exception as e:
            print(f"Error occurred in {level_name}, model: {model_id}: {e}")
            self._write_row(writer, model_id, level_name, mode, label, f"ERROR: {e}", "N/A", [], "ERROR")
        time.sleep(1)

    def _run_nullable_level(self, writer, model_id, stream, max_completion_tokens, temperature, reasoning=None):
        level_name = self.LEVEL_NAMES[6]
        mode = "Stream" if stream else "Non-Stream"
        messages = [{"role": "user", "content": self.NULLABLE_PROMPT}]
        try:
            print(f"{level_name}: {self.NULLABLE_PROMPT}")
            reasoning_content, content, tool_calls = self._call_model(
                model_id, messages, stream, max_completion_tokens, temperature, reasoning,
                tools=self.TOOLS + [self.NULLABLE_TOOL])
            verdict = self._check_nullable(content, tool_calls)
            self._write_row(writer, model_id, level_name, mode, self.NULLABLE_PROMPT,
                            reasoning_content, content, tool_calls, verdict)
            print(f"Check result: {verdict[0]} ({verdict[1]})")
        except Exception as e:
            # a template that cannot render the schema comes back as an HTTP error, which is the failure
            print(f"Error occurred in {level_name}, model: {model_id}: {e}")
            self._write_row(writer, model_id, level_name, mode, self.NULLABLE_PROMPT,
                            f"ERROR: {e}", "N/A", [], ("FAIL", f"request failed: {str(e)[:120]}"))
        time.sleep(1)

    def run(self, max_completion_tokens=-1, temperature=0.3, reasoning=None):
        single_round_levels = [
            (self.LEVEL_NAMES[0], self.BASIC_PROMPT,
             lambda content, tcs: self._verify_weather_call(tcs, "Paris")),
            (self.LEVEL_NAMES[1], self.EXTRACTION_PROMPT,
             lambda content, tcs: self._verify_weather_call(tcs, "Paris")),
            (self.LEVEL_NAMES[2], self.RESTRAINT_PROMPT, self._check_restraint),
            (self.LEVEL_NAMES[3], self.PARALLEL_PROMPT, self._check_parallel),
        ]

        with open(self.csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Model", "Complexity Level", "Mode", "Input",
                             "Reasoning Content", "Output Content", "Tool Calls", "Check Result"])
            print("\n=== Starting Tool Calling Tests ===")
            print(f"Models found: {len(self.models)}")
            if reasoning:
                print(f"Reasoning effort requested: {reasoning}")
            for model_id in self.models:
                print(f"\n--- Testing tool-calling model: {model_id} ---")
                for mode_label, stream in (("Non-Stream", False), ("Stream", True)):
                    print(f"\nTesting {mode_label.lower()} mode...")
                    for level_name, prompt, checker in single_round_levels:
                        self._run_single_round_level(writer, model_id, level_name, prompt, checker,
                                                     stream, max_completion_tokens, temperature,
                                                     reasoning=reasoning)
                    self._run_loop_level(writer, model_id, stream, max_completion_tokens, temperature,
                                         reasoning=reasoning)
                    self._run_fidelity_level(writer, model_id, stream, max_completion_tokens, temperature,
                                             reasoning=reasoning)
                    self._run_nullable_level(writer, model_id, stream, max_completion_tokens, temperature,
                                             reasoning=reasoning)
                print(f"Finished testing model: {model_id}")
        print(f"\nTool calling tests complete. Saved to {self.csv_filename}")
        return self.result


class ApiConformanceTask(BaseTestTask):
    """
    Server conformance: what the OpenAI-compatible API owes every client,
    whatever the weights say.

      A1 Error Status       A refused request comes back with a non-2xx status
                            and an OpenAI-shaped error body, so the error text
                            actually reaches the caller.
      A2 Model Identity     A tag the server cannot load is refused, not
                            answered by whichever model happens to be loaded.
      A3 Finish Reason      An answer cut at max_tokens says so, and one that
                            ended on its own says that.
      A4 Request Isolation  Settings a request does not carry come from the
                            model's defaults, not from the previous request.
      A5 Stream Parity      The two response modes report the same metadata for
                            the same request.

    Every defect these come from returned a well-formed 200 with a plausible
    body, which is why the model-quality suites never saw any of them. A1 and A2
    go through raw HTTP rather than the OpenAI SDK, because there the status code
    IS the assertion and the SDK hides it: a 4xx becomes an exception and a 200
    carrying an error body becomes a success.
    """

    SUITE_NAME = "api"

    # A tag no server can have loaded, for A2.
    IMPOSSIBLE_MODEL = "oflm-test-no-such-model:0b"
    # Long enough that eight tokens cannot finish it, so the cut is unambiguous.
    LONG_PROMPT = ("Write a detailed multi-paragraph history of the transistor, "
                   "covering the 1947 Bell Labs work and everything after it.")
    SHORT_PROMPT = "Reply with exactly one word: ok"
    ISOLATION_PROMPT = "In one short sentence, what is an NPU?"
    TRUNCATION_LIMIT = 8
    PARITY_LIMIT = 24
    ISOLATION_LIMIT = 40

    def __init__(self, base_url, backend_os="linux", model_filter: list[str] | None = None):
        super().__init__(base_url, backend_os, model_filter=model_filter)
        self.models = [m for m in self.models if m not in NON_CHAT_MODELS]
        # These checks are about the server, not the weights, so one model
        # answers them all. Without a --model filter, take the first rather than
        # making the server swap models on the NPU for no extra coverage.
        if not model_filter:
            self.models = self.models[:1]
        self.csv_filename = self.get_csv_filename("api")

    # -------------------------------------------------------------- helpers

    def _chat_raw(self, body=None, raw_body=None):
        return self._post_json("/chat/completions", body, raw_body=raw_body)

    def _chat(self, model_id, prompt, stream=False, max_completion_tokens=None,
              temperature=None, reasoning=None):
        """One chat call through the SDK; returns (reasoning, content, finish_reason, usage, model)."""
        kwargs = {}
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        kwargs.update(self._reasoning_kwargs(reasoning))
        response = self.client.chat.completions.create(
            model=model_id, messages=[{"role": "user", "content": prompt}], stream=stream, **kwargs)
        if stream:
            return self._collect_stream_meta(response)
        choice = response.choices[0]
        return (getattr(choice.message, "reasoning_content", "") or "",
                choice.message.content or "", getattr(choice, "finish_reason", None),
                getattr(response, "usage", None), getattr(response, "model", None))

    def _write_row(self, writer, model_id, check_name, probe, status, detail, verdict_detail):
        verdict, verdict_text = verdict_detail
        writer.writerow([model_id, check_name, probe, status if status is not None else "N/A",
                         detail, f"{verdict}: {verdict_text}"])

    def _run_probe(self, writer, model_id, check_name, probe, probe_fn):
        """Runs one (verdict, detail, status, note) producing closure, with error handling."""
        try:
            verdict, status, note = probe_fn()
            self.record(verdict, f"{model_id} / {check_name} / {probe}")
            self._write_row(writer, model_id, check_name, probe, status, note, verdict)
            print(f"  {check_name} [{probe}]: {verdict[0]} ({verdict[1]})")
        except Exception as e:
            print(f"  {check_name} [{probe}]: ERROR ({e})")
            self.record(f"ERROR: {e}", f"{model_id} / {check_name} / {probe}")
            self._write_row(writer, model_id, check_name, probe, None, str(e), ("ERROR", str(e)))
        time.sleep(1)

    # -------------------------------------------------------------- checks

    def _judge_refusal(self, status, body, text, expected_code=None):
        """A request that must be refused: was it, and could the client tell?"""
        code, error_type, message = self._error_fields(body)
        if status < 400:
            if message is not None:
                return ("FAIL", f"answered HTTP {status} with an error in the body, so the "
                                f"error never reaches a client as an error: "
                                f"{message[:160]}")
            return ("FAIL", f"answered HTTP {status} instead of refusing: {text[:160]}")
        if message is None:
            return ("SOFT-FAIL", f"refused with HTTP {status}, but the body carries no error "
                                 f"object a client can read: {text[:160]}")
        if code is None:
            return ("SOFT-FAIL", f"refused with HTTP {status}, but the error is a bare string "
                                 f"rather than an OpenAI error object, so a client cannot read "
                                 f"a code off it: {message[:160]}")
        # The type of error.code is deliberately not asserted. It is a string on
        # the errors the server builds for a client and a number on the outer
        # catch's 500, and status_for() accepts both; the defect was in the
        # reader, and what a client can see of it is the status above.
        if not isinstance(code, (str, int)):
            return ("SOFT-FAIL", f"refused with HTTP {status}, but error.code is "
                                 f"{type(code).__name__} {code!r}")
        if not error_type:
            return ("SOFT-FAIL", f"refused with HTTP {status} {code}, but the error names no type")
        if expected_code and code != expected_code:
            return ("SOFT-FAIL", f"refused with HTTP {status}, but the code is '{code}' rather "
                                 f"than '{expected_code}'")
        return ("PASS", f"refused with HTTP {status} {code}")

    def _probe_malformed_json(self, model_id):
        status, body, text = self._chat_raw(raw_body=b'{"model": "' + model_id.encode() + b'", "mess')
        return self._judge_refusal(status, body, text), status, text[:200]

    def _probe_no_messages(self, model_id):
        status, body, text = self._chat_raw({"model": model_id})
        return self._judge_refusal(status, body, text), status, text[:200]

    def _probe_messages_wrong_type(self, model_id):
        status, body, text = self._chat_raw({"model": model_id, "messages": "hello"})
        return self._judge_refusal(status, body, text), status, text[:200]

    def _probe_unknown_model(self, model_id):
        """A2: the chat half of the embedding suite's E9."""
        status, body, text = self._chat_raw({
            "model": self.IMPOSSIBLE_MODEL,
            "messages": [{"role": "user", "content": self.SHORT_PROMPT}],
            "max_completion_tokens": self.PARITY_LIMIT})
        choices = body.get("choices") if isinstance(body, dict) else None
        if status < 400 and choices:
            answer = ""
            try:
                answer = choices[0]["message"]["content"] or ""
            except (KeyError, TypeError, IndexError):
                pass
            return ("FAIL", f"answered a request for '{self.IMPOSSIBLE_MODEL}', a model it "
                            f"cannot have loaded, with HTTP {status} and a completion "
                            f"reported as '{body.get('model')}': '{answer[:80]}'. Some other "
                            f"model answered under the requested name."), status, text[:200]
        return self._judge_refusal(status, body, text, expected_code="model_not_found"), status, text[:200]

    def _probe_model_echo(self, model_id):
        """A2, second half: an accepted request names the model that answered."""
        status, body, text = self._chat_raw({
            "model": model_id,
            "messages": [{"role": "user", "content": self.SHORT_PROMPT}],
            "max_completion_tokens": self.PARITY_LIMIT})
        if status >= 400:
            return ("ERROR", f"HTTP {status}: {text[:160]}"), status, text[:200]
        reported = body.get("model") if isinstance(body, dict) else None
        if not reported:
            return ("SOFT-FAIL", "the response names no model, so a client cannot confirm "
                                 "which one answered"), status, text[:200]
        if reported != model_id:
            return ("SOFT-FAIL", f"requested '{model_id}', response reports "
                                 f"'{reported}'"), status, text[:200]
        return ("PASS", f"reported as '{reported}'"), status, ""

    def _probe_truncated(self, model_id, stream):
        """A3: an answer cut at max_tokens must not claim it finished."""
        _, content, finish_reason, _, _ = self._chat(
            model_id, self.LONG_PROMPT, stream=stream,
            max_completion_tokens=self.TRUNCATION_LIMIT, temperature=0.0)
        note = f"finish_reason={finish_reason!r}, {len(content)} chars"
        if finish_reason is None:
            return ("FAIL", "no finish_reason at all, so a client cannot tell a complete "
                            "answer from a truncated one"), None, note
        if finish_reason == "length":
            return ("PASS", f"reported 'length' at {self.TRUNCATION_LIMIT} tokens"), None, note
        if finish_reason == "stop":
            return ("FAIL", f"an answer cut at {self.TRUNCATION_LIMIT} tokens reported 'stop', "
                            f"which is what a complete answer reports; the cut is invisible "
                            f"to the caller"), None, note
        return ("SOFT-FAIL", f"reported '{finish_reason}' rather than 'length'"), None, note

    def _probe_complete(self, model_id, stream):
        """A3, the other half: a normal answer still reports 'stop'."""
        _, content, finish_reason, _, _ = self._chat(
            model_id, self.SHORT_PROMPT, stream=stream, temperature=0.0)
        note = f"finish_reason={finish_reason!r}, {len(content)} chars"
        if finish_reason is None:
            return ("FAIL", "no finish_reason on a complete answer"), None, note
        if finish_reason == "stop":
            return ("PASS", "reported 'stop'"), None, note
        if finish_reason == "length":
            return ("SOFT-FAIL", "reported 'length' on an unbounded request; the answer may "
                                 "genuinely have hit the context limit"), None, note
        return ("SOFT-FAIL", f"reported '{finish_reason}'"), None, note

    def _probe_isolation(self, model_id):
        """A4: what one request sets must not survive into the next.

        One client asking for reasoning_effort low turned thinking on for
        everybody else, and the answers stayed plausible throughout.
        """
        def probe():
            reasoning, content, _, _, _ = self._chat(
                model_id, self.ISOLATION_PROMPT,
                max_completion_tokens=self.ISOLATION_LIMIT, temperature=0.0)
            return bool(reasoning.strip()), content

        # Two probes back to back first. Decoding on the NPU is not always
        # reproducible, and without knowing that up front a difference after the
        # poisoning request cannot be told from ordinary run-to-run drift.
        before_thinking, before_text = probe()
        _, baseline_text = probe()
        deterministic = before_text == baseline_text

        self._chat(model_id, self.ISOLATION_PROMPT, max_completion_tokens=self.ISOLATION_LIMIT,
                   temperature=1.0, reasoning="high")
        after_thinking, after_text = probe()

        note = (f"thinking before={before_thinking} after={after_thinking}; "
                f"deterministic={deterministic}; identical text={before_text == after_text}")
        if before_thinking != after_thinking:
            return ("FAIL", f"a request carrying reasoning_effort 'high' changed what a later "
                            f"request with no reasoning_effort does: thinking was "
                            f"{before_thinking} before it and {after_thinking} after"), None, note
        if deterministic and before_text != after_text:
            return ("FAIL", "two back-to-back probes answered identically, and the same probe "
                            "after a request that set temperature 1.0 did not: a setting from "
                            "that request survived into the next one"), None, note
        if not deterministic:
            return ("PASS", "thinking state survived the intervening request unchanged; the "
                            "text comparison is not usable here, since two back-to-back probes "
                            "already differ"), None, note
        return ("PASS", "an intervening request changed nothing for the next one"), None, note

    def _probe_parity(self, model_id):
        """A5: both response modes describe the same request the same way."""
        _, non_stream_content, non_stream_reason, _, non_stream_model = self._chat(
            model_id, self.LONG_PROMPT, stream=False,
            max_completion_tokens=self.PARITY_LIMIT, temperature=0.0)
        _, stream_content, stream_reason, _, stream_model = self._chat(
            model_id, self.LONG_PROMPT, stream=True,
            max_completion_tokens=self.PARITY_LIMIT, temperature=0.0)
        note = (f"non-stream finish_reason={non_stream_reason!r} model={non_stream_model!r}; "
                f"stream finish_reason={stream_reason!r} model={stream_model!r}")
        if stream_reason is None:
            return ("FAIL", "the stream never sent a chunk carrying a finish_reason, so a "
                            "streaming client cannot tell why generation ended"), None, note
        if non_stream_reason != stream_reason:
            return ("FAIL", f"the same request reports finish_reason '{non_stream_reason}' "
                            f"non-streamed and '{stream_reason}' streamed"), None, note
        if non_stream_model != stream_model:
            return ("SOFT-FAIL", f"the two modes name different models: "
                                 f"'{non_stream_model}' and '{stream_model}'"), None, note
        if not stream_content.strip():
            return ("FAIL", "the stream produced no content"), None, note
        return ("PASS", f"both modes report '{stream_reason}' for '{stream_model}'"), None, note

    # -------------------------------------------------------------- runner

    def run(self):
        print("\n=== Starting API Conformance Tests ===")
        if not self.models:
            print("No chat model found. Start the server with a model loaded, e.g. `oflm serve`.")
            self.record(("ERROR", "no chat model available to test against"), "api suite")
            return self.result

        with open(self.csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Model", "Check", "Probe", "HTTP Status", "Detail", "Check Result"])
            for model_id in self.models:
                print(f"\n--- API conformance against: {model_id} ---")
                for probe_name, probe_fn in (
                        ("malformed JSON", self._probe_malformed_json),
                        ("no messages field", self._probe_no_messages),
                        ("messages is not a list", self._probe_messages_wrong_type)):
                    self._run_probe(writer, model_id, "A1 Error Status", probe_name,
                                    lambda fn=probe_fn: fn(model_id))
                self._run_probe(writer, model_id, "A2 Model Identity", self.IMPOSSIBLE_MODEL,
                                lambda: self._probe_unknown_model(model_id))
                self._run_probe(writer, model_id, "A2 Model Identity", "reported model",
                                lambda: self._probe_model_echo(model_id))
                for mode_label, stream in (("Non-Stream", False), ("Stream", True)):
                    self._run_probe(writer, model_id, "A3 Finish Reason",
                                    f"truncated / {mode_label}",
                                    lambda s=stream: self._probe_truncated(model_id, s))
                    self._run_probe(writer, model_id, "A3 Finish Reason",
                                    f"complete / {mode_label}",
                                    lambda s=stream: self._probe_complete(model_id, s))
                self._run_probe(writer, model_id, "A4 Request Isolation", "reasoning_effort",
                                lambda: self._probe_isolation(model_id))
                self._run_probe(writer, model_id, "A5 Stream Parity", "same request both ways",
                                lambda: self._probe_parity(model_id))
                print(f"Finished API conformance for: {model_id}")
        print(f"\nAPI conformance tests complete. Saved to {self.csv_filename}")
        return self.result

import base64
import csv
import os
import re
import time
import json
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# Media assets ship inside the package; anchor paths here so tests work
# regardless of the current working directory (repo checkout or pip/uv install).
PACKAGE_DIR = Path(__file__).resolve().parent

# Specialised (non-general-chat) models excluded from chat-based suites.
NON_CHAT_MODELS = ("gpt-oss:20b", "gpt-oss-sg:20b", "qwen3.5:4b", "qwen3.5:9b",
                   "medgemma:4b", "medgemma1.5:4b", "translategemma:4b")

class BaseTestTask(ABC):
    """
    Abstract base class for all testing tasks.
    Enforces a standard interface for running tests and saving results.
    """
    MUSIC_PATTERN = re.compile(
        r"\b(music|melod\w*|song|tune|rhythm|beat|tempo|instrument\w*|drum\w*|bass|synth\w*|vocal\w*|chord\w*|harmo\w*)\b",
        re.IGNORECASE,
    )

    def __init__(self, base_url, backend_os="linux", model_filter: list[str] | None = None):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key="flm")
        self.version = self._get_flm_version()
        self.models = self._fetch_all_models()
        if model_filter:
            filtered = [m for m in self.models if m in model_filter]
            if not filtered:
                print(f"Warning: none of the requested model(s) {model_filter} were found in the server's model list. "
                      f"Available: {self.models}")
            self.models = filtered
        self.results_dir = os.path.join("results", self.timestamp, backend_os)
        os.makedirs(self.results_dir, exist_ok=True)

    # FLM's OpenAI-compatible API accepts `reasoning_effort` with "low",
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

    def _get_flm_version(self) -> str:
        print("\nChecking flm version...")
        try:
            response = urllib.request.urlopen(f"{self.base_url}/version", timeout=5)
            version_data = json.loads(response.read().decode('utf-8'))
            flm_version = version_data.get("version", "unknown_version")
            print(f"Detected flm version: {flm_version}")
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as e:
            print(f"Error fetching flm version: {e}")
            flm_version = "unknown_version"
        return flm_version

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

    def _collect_stream(self, response) -> tuple[str, str]:
        """Accumulates streamed chunks into (reasoning_content, output_content)."""
        reasoning_content, output_content = "", ""
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_content += delta.reasoning_content
            if delta.content:
                output_content += delta.content
        return reasoning_content, output_content

    @abstractmethod
    def run(self, *args, **kwargs):
        """Must be implemented by all subclasses"""
        pass


class LLMTask(BaseTestTask):

    def __init__(self, base_url, backend_os="linux", model_filter: list[str] | None = None):
        super().__init__(base_url, backend_os, model_filter=model_filter)
        self.models = [m for m in self.models if m not in NON_CHAT_MODELS]
        self.csv_filename = self.get_csv_filename("llm")

    def _run_two_rounds(self, writer, model_id, prompt, followup_prompt, stream, max_completion_tokens,
                        temperature=None, reasoning=None):
        reasoning_kwargs = self._reasoning_kwargs(reasoning)
        mode = "Stream" if stream else "Non-Stream"
        messages = [{"role": "user", "content": prompt}]

        # first round
        try:
            print(f"Prompt: {prompt}")
            response = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                stream=stream,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                **reasoning_kwargs,
            )
            if stream:
                reasoning_content, output_content = self._collect_stream(response)
            else:
                reasoning_content = getattr(response.choices[0].message, "reasoning_content", "N/A") or "N/A"
                output_content = response.choices[0].message.content or ""
            writer.writerow([model_id, mode, prompt, reasoning_content or "N/A", output_content])
            print("Done.")
            time.sleep(1)
            messages.append({"role": "assistant", "content": output_content})
            messages.append({"role": "user", "content": followup_prompt})
        except Exception as e:
            print(f"Error occurred in first round, model: {model_id}: {e}")
            writer.writerow([model_id, mode, prompt, f"ERROR: {e}", "N/A"])

        # second round
        try:
            print(f"Follow-up Prompt: {followup_prompt}")
            response = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                stream=stream,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                **reasoning_kwargs,
            )
            if stream:
                reasoning_content, output_content = self._collect_stream(response)
            else:
                reasoning_content = getattr(response.choices[0].message, "reasoning_content", "N/A") or "N/A"
                output_content = response.choices[0].message.content or ""
            writer.writerow([model_id, mode, followup_prompt, reasoning_content or "N/A", output_content])
            print("Done.")
            time.sleep(1)
        except Exception as e:
            print(f"Error occurred in second round, model: {model_id}: {e}")
            writer.writerow([model_id, mode, followup_prompt, f"ERROR: {e}", "N/A"])

    def run(self, max_completion_tokens=-1, temperature=0.3, reasoning=None):
        prompt = "Teach me Maxwell's equations."
        followup_prompt = "Summarize your answer."

        stream_prompt = "Teach me Maxwell's equations."
        stream_followup_prompt = "Explain why they are important."

        with open(self.csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Model", "Mode", "Input", "Reasoning Content", "Output Content"])
            print("\n=== Starting LLM Tests ===")
            print(f"Models found: {len(self.models)}")
            for model_id in self.models:
            # for model_id in self.models[2:4]:  # Limit to first 2 models for testing purposes
                print(f"\n--- Testing LLM model: {model_id} ---")
                # print("Testing non-stream mode...\n")
                # self._run_two_rounds(writer, model_id, prompt, followup_prompt, stream=False, max_completion_tokens=max_completion_tokens)
                print("\nTesting stream mode...\n")
                self._run_two_rounds(writer, model_id, stream_prompt, stream_followup_prompt, stream=True,
                                     max_completion_tokens=max_completion_tokens, temperature=temperature,
                                     reasoning=reasoning)
                print(f"Finished testing model: {model_id}")
        print(f"\nLLM tests complete. Saved to {self.csv_filename}")


class EmbeddingTask(BaseTestTask):
    """
    Tests OpenAI-compatible text embeddings across seven automated checks:

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
                                 run must agree, pinning any bad number on FLM
                                 rather than on a single bad machine draw.
E7 Batch Reference
       Consistency                A larger set of draws of the same input is
                                  compared against the batch-path reference in
                                  the same run, exposing the outlier rate and
                                  magnitude that a small sample can miss.
       E8 Reference Agreement     Embeddings match the bundled reference vectors
                                  produced by the validated numpy pipeline for
                                  google/embeddinggemma-300m, pinning the API
                                  path to a known-good implementation.

    Like the tool-calling suite, each check produces a PASS / SOFT-FAIL / FAIL
    verdict with a detail line, all written to CSV. Unlike the chat-based suites
    there is no streaming mode, temperature or reasoning, and the server is
    assumed to be running with only an embed model loaded (`flm serve -e 1`).
    """

    EMBED_MODELS = ["embed-gemma:300m", "embed-gemma"]
    DEFAULT_EMBED_MODEL = "embed-gemma:300m"

    SAMPLE_TEXT = "The embedding model should capture the meaning of this sentence."
    BATCH_INPUTS = [
        "Hello, world!",
        "FastFlowLM is a local inference server.",
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
    CHECK_NAMES = [
        "E1 Response Structure",
        "E2 Repeatability",
        "E3 Batch & Index Integrity",
        "E4 Dimensionality",
        "E5 Semantic Ordering",
        "E6 Cross-Path Consistency",
        "E7 Batch Reference Consistency",
        "E8 Reference Agreement",
    ]

    def __init__(self, base_url, backend_os="linux", model_filter: list[str] | None = None):
        super().__init__(base_url, backend_os, model_filter=model_filter)
        # Keep only recognised embedding models; honour any user-supplied filter.
        self.models = [m for m in self.models if m in self.EMBED_MODELS]
        # FLM does not always advertise the embed model on /v1/models even when
        # it is loaded (`flm serve -e 1`). Without an explicit --model filter,
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
        self._reference_entries = [
            (entry["text"], entry["embedding"]) for entry in reference["entries"].values()
        ]

    # -------------------------------------------------------------- helpers

    def _embed_response(self, model_id: str, input_text: str | list[str]):
        """One embeddings call; returns the raw response payload."""
        return self.client.embeddings.create(model=model_id, input=input_text)

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
            self._write_row(writer, model_id, check_name, input_text, result, vector)
            print(f"  {check_name}: {verdict} ({detail})")
        except Exception as e:
            print(f"  {check_name}: ERROR ({e})")
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
            return ("PASS", f"all {self.REPEAT_COUNT} draws consistent "
                            f"(worst cosine {worst:.6f})"), draws[0]
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
        same run it is a statement about FLM, not about a single bad hardware
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
                  "loaded, e.g. `flm serve -e 1`.")
            print(f"Embedding tests complete. Saved to {self.csv_filename}")
            return

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
                print(f"Finished testing model: {model_id}")
        print(f"\nEmbedding tests complete. Saved to {self.csv_filename}")

class AudioTask(BaseTestTask):
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
                    writer.writerow([model_id, followup_prompt, f"ERROR: {e}", "N/A", "N/A"])
                print(f"Finished testing model: {model_id}")
        print(f"Audio tests complete. Saved to {self.csv_filename}")

class VisionTask(BaseTestTask):

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
                        writer.writerow([model_id, followup_prompt_music, f"ERROR: {e}", "N/A",
                                         "N/A", "N/A", "ERROR"])
                else:
                    writer.writerow([model_id, followup_prompt_music,
                                     "SKIPPED: second round failed", "N/A", "N/A", "SKIPPED"])
                print(f"Finished testing model: {model_id}")
        print(f"Vision tests complete. Saved to {self.csv_filename}")


class ToolCallingTask(BaseTestTask):
    """
    Tests OpenAI-compatible function/tool calling at five escalating
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

    Automated checks validate tool names, JSON argument validity/values and,
    for L5, the arithmetic derived from the tool result ($54 total).
    """

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

    LEVEL_NAMES = [
        "L1 Basic Tool Call",
        "L2 Argument Extraction",
        "L3 Tool Restraint",
        "L4 Parallel Tool Calls",
        "L5 Multi-Turn Tool Loop",
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

    def _call_model(self, model_id, messages, stream, max_completion_tokens, temperature, reasoning=None):
        """One chat completion with tools bound; returns (reasoning, content, tool_calls)."""
        response = self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=self.TOOLS,
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
                print(f"Finished testing model: {model_id}")
        print(f"\nTool calling tests complete. Saved to {self.csv_filename}")

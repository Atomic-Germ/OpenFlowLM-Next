/// \file benchmark_embed_util.hpp
/// \brief The pure part of `oflm bench-embed`: no device, no model, no clock.
///
/// SPDX-License-Identifier: MIT
///
/// Split out from benchmark_embed.hpp so it can be TESTED. Everything here is a
/// function of its arguments -- the sweep plan, the stage count, the task-name
/// lookup, the corpus cycling -- and every one of them was a place where a wrong
/// answer would have been silent: a shortened sweep, a prompt the endpoint
/// cannot be asked for, a run labelled as the built-in experiment when it was
/// not. benchmark_embed_test.cpp holds them to it without an NPU.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "AutoEmbeddingModel/auto_embedding_model.hpp"  // embedding_task_type_t
#include "nlohmann/json.hpp"
#include "openai_compat.hpp"                           // task_names, embedding_batch_dim

namespace benchmarking {

/// Where a run's texts came from. Printed, because a reader who assumes the
/// built-in corpus is reading a different experiment than the one that ran.
enum class CorpusSource { BuiltIn, Config };

inline const char* corpus_source_name(CorpusSource s) {
    return s == CorpusSource::BuiltIn ? "built-in" : "from the config file";
}

/// What kind of task prefix a backend actually applied.
///
/// Three values because "no prompt names" means two different things, and
/// collapsing them printed a falsehood: OpenGemma_Embedding declares NO names
/// and still prefixes every text via open_task_prefix(), while the BERT family
/// has no task concept at all. The first version of this benchmark reported
/// both as "no prefix applied".
enum class PrefixKind { None, ContainerPrompt, BackendHardcoded };

/// The whole sweep, decided before anything is timed.
struct EmbedBenchPlan {
    int max_batch = 128;
    int iterations = 2;
    std::string task_name;              ///< REST name; empty means "query"
    bool task_explicit = false;         ///< did the CLI or the config say one?
    std::vector<std::string> corpus;
    CorpusSource corpus_source = CorpusSource::BuiltIn;
};

/// The built-in corpus.
///
/// Sixteen distinct sentences, cycled to fill a batch. Distinctness does not
/// change the timing -- there is no data-dependent branching in an encoder --
/// but the cycling is printed rather than implied. They sit in the length range
/// a RAG chunk does; the design pads every row to its own compiled `seq`
/// regardless, which is why text length is not the interesting axis here.
inline const std::vector<std::string>& builtin_corpus() {
    static const std::vector<std::string> texts = {
        "The Ryzen AI NPU runs encoder-only models as a sequence of GEMM dispatches over one resident xclbin.",
        "Vector databases store dense embeddings and retrieve them by approximate nearest-neighbour search.",
        "A retrieval-augmented pipeline embeds the query, fetches the closest chunks, and passes them to a chat model.",
        "Batching matters more than raw arithmetic throughput when the per-dispatch overhead dominates.",
        "Sentence transformers pool token vectors into one fixed-width vector per input text.",
        "Cosine similarity compares direction and ignores magnitude, which is why embeddings are normalised.",
        "The tokenizer splits text into subword units before any matrix multiplication happens.",
        "Quantisation trades a little numerical accuracy for a large reduction in memory traffic.",
        "Attention cost grows with the square of the sequence length, so long documents are chunked.",
        "A design compiled for one geometry serves every model whose GEMM shapes match it exactly.",
        "Host-side layer normalisation was measured faster and more accurate than its device dispatch at these widths.",
        "Weights are pre-tiled offline so the runtime never rearranges them on the critical path.",
        "An embedding for the wrong task is correctly shaped and correctly normed, so nothing downstream can flag it.",
        "Energy per thousand sequences is a better figure of merit for a laptop than peak throughput.",
        "The shim DMA is the only path to DRAM, so it bounds what any kernel above it can achieve.",
        "Reproducibility means the same input produces the same bytes, on the same binary and the same host ISA level.",
    };
    return texts;
}

/// Fill `n` texts by cycling `corpus`.
inline std::vector<std::string> take_texts(const std::vector<std::string>& corpus, int n) {
    if (corpus.empty())
        throw std::runtime_error("take_texts: the corpus is empty");
    if (n < 0)
        throw std::runtime_error("take_texts: negative count");
    std::vector<std::string> out;
    out.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; i++)
        out.push_back(corpus[static_cast<size_t>(i) % corpus.size()]);
    return out;
}

/// The largest `--max-batch` this command will accept.
///
/// Not a hardware limit: the largest batch tier any shipped design declares is
/// 128, so everything above that measures how the host chunks a request. It is
/// here so an accidental extra zero fails immediately instead of looking like
/// a hang.
inline constexpr int kMaxBatchCeiling = 8192;

/// How many doubling stages a `--max-batch` implies.
///
/// REFUSES a limit the sweep cannot land on. `--max-batch 3` used to give two
/// stages -- 1 and 2, never 3 -- so the option documented as "the largest batch
/// swept" quietly swept something else, and the result depended on an
/// undocumented rounding rule. Computed by shifting rather than by log2(), so
/// there is no floating-point boundary to get wrong either.
inline int bench_stages(int max_batch) {
    if (max_batch < 1)
        throw std::runtime_error("--max-batch must be at least 1, got " +
                                 std::to_string(max_batch));
    if (max_batch > kMaxBatchCeiling)
        throw std::runtime_error(
            "--max-batch " + std::to_string(max_batch) + " is beyond the " +
            std::to_string(kMaxBatchCeiling) + " this command will sweep. The" + 
            " largest tier any shipped design declares is 128, so the stages" + 
            " above that measure how the host chunks a request rather than the" + 
            " array -- and an accidental extra zero should be an error, not an" + 
            " afternoon.");
    if ((max_batch & (max_batch - 1)) != 0)
        throw std::runtime_error(
            "--max-batch must be a power of two (1, 2, 4, 8 ... ), got " +
            std::to_string(max_batch) +
            ". The sweep doubles from 1, so a limit it cannot land on would"
            " silently shorten the experiment: 3 would run 1 and 2 and never 3.");
    int stages = 1;
    for (int b = 1; b < max_batch; b <<= 1) ++stages;
    return stages;
}

/// Resolve a REST task name to the enum, reusing the server's own table.
///
/// Reused rather than re-listed: a second copy of this vocabulary would drift,
/// and then the benchmark would measure a prompt the endpoint cannot be asked
/// for. Empty means "query", which is what /v1/embeddings resolves an
/// unspecified request to.
inline embedding_task_type_t task_from_name(const std::string& name) {
    if (name.empty()) return task_query;
    for (const auto& kv : openai_compat::task_names())
        if (name == kv.first) return kv.second;
    throw std::runtime_error(
        "unknown task prompt '" + name + "'. Valid names: " +
        openai_compat::task_names_csv());
}

/// Build the sweep plan from an optional config file plus the CLI values.
///
/// The file wins per key and the CLI fills the rest. That differs DELIBERATELY
/// from `oflm bench`, whose --bench-iterations never reaches a file-supplied
/// config at all -- a trap its own README documents.
///
/// `task_explicit` exists because a task supplied through the config file is
/// just as explicit as one on the command line, and the first version only
/// looked at the CLI string. Without it, a config that names a task read as no
/// task at all, and the policy check below would have demanded one anyway.
inline EmbedBenchPlan make_embed_bench_plan(const nlohmann::json& cfg, bool have_cfg,
                                            int cli_iterations, int cli_max_batch,
                                            const std::string& cli_prompt_name) {
    EmbedBenchPlan plan;
    plan.max_batch = cli_max_batch;
    plan.iterations = cli_iterations;
    plan.task_name = cli_prompt_name;
    plan.task_explicit = !cli_prompt_name.empty();
    plan.corpus = builtin_corpus();
    plan.corpus_source = CorpusSource::BuiltIn;

    if (have_cfg) {
        // The ROOT has to be an object before any key is read. nlohmann's
        // contains() is `is_object() && ...`, so a file holding `[]`, `null`
        // or a scalar made every key below invisible and the sweep ran on the
        // CLI defaults with nothing to say the file had been ignored.
        if (!cfg.is_object())
            throw std::runtime_error(
                "bench config: the file must hold a JSON object with the keys "
                "max_batch, iterations, task and texts; its root is " +
                std::string(cfg.type_name()) +
                ". Every key would otherwise have been read as absent and the "
                "run would have used the command-line values instead.");
        if (cfg.contains("max_batch")) {
            if (!cfg["max_batch"].is_number_integer())
                throw std::runtime_error("bench config: \"max_batch\" must be an integer");
            plan.max_batch = cfg["max_batch"].get<int>();
        }
        if (cfg.contains("iterations")) {
            if (!cfg["iterations"].is_number_integer())
                throw std::runtime_error("bench config: \"iterations\" must be an integer");
            plan.iterations = cfg["iterations"].get<int>();
        }
        if (cfg.contains("task")) {
            if (!cfg["task"].is_string())
                throw std::runtime_error("bench config: \"task\" must be a string");
            plan.task_name = cfg["task"].get<std::string>();
            plan.task_explicit = !plan.task_name.empty();
        }
        if (cfg.contains("texts")) {
            if (!cfg["texts"].is_array())
                throw std::runtime_error("bench config: \"texts\" must be an array of strings");
            std::vector<std::string> custom;
            for (const auto& t : cfg["texts"]) {
                if (!t.is_string())
                    throw std::runtime_error(
                        "bench config: every entry of \"texts\" must be a string");
                custom.push_back(t.get<std::string>());
            }
            if (custom.empty())
                throw std::runtime_error("bench config: \"texts\" contained no strings");
            plan.corpus = std::move(custom);
            plan.corpus_source = CorpusSource::Config;
        }
    }

    if (plan.iterations < 1)
        throw std::runtime_error("--bench-iterations must be at least 1, got " +
                                 std::to_string(plan.iterations));
    (void)bench_stages(plan.max_batch);   // refuse a bad limit here, not mid-run
    (void)task_from_name(plan.task_name); // and an unknown task name here too
    return plan;
}

/// Turn the endpoint's own three-state task policy into a refusal message, or
/// an empty string when the request is fine.
///
/// Shares openai_compat::task_policy() with /v1/embeddings ON PURPOSE: a task
/// this benchmark accepts has to be one a client could also have asked for.
/// Without this the benchmark timed nomic under an implicit `query` while the
/// endpoint answered 400 to the same request -- measuring something nobody can
/// send.
inline std::string task_policy_refusal(bool supports_prompts, bool declares_names,
                                       bool task_given,
                                       const std::vector<std::string>& declared) {
    switch (openai_compat::task_policy(supports_prompts, declares_names, task_given)) {
        case openai_compat::TaskPolicy::Ok:
            return std::string();
        case openai_compat::TaskPolicy::Required: {
            std::string have;
            for (const auto& n : declared) have += (have.empty() ? "" : ", ") + n;
            return "this model declares task prompts [" + have +
                   "] and /v1/embeddings requires one, so benchmarking it without "
                   "one would time a request no client can send. Pass "
                   "--prompt-name (or \"task\" in the config file); valid names: " +
                   openai_compat::task_names_csv();
        }
        case openai_compat::TaskPolicy::NotSupported:
            return "this model has no task-prompt concept, and /v1/embeddings "
                   "refuses a prompt for it. Drop --prompt-name (or the config's "
                   "\"task\") rather than measuring a request that would be "
                   "rejected.";
    }
    return std::string();
}

}  // namespace benchmarking

/// \file benchmark_embed.hpp
/// \brief `oflm bench-embed` -- the embedding sibling of `oflm bench`.
///
/// SPDX-License-Identifier: MIT
///
/// WHY A SECOND BENCHMARK RATHER THAN A FLAG ON THE FIRST.
///
/// benchmarking.hpp sweeps CONTEXT LENGTH and reports TTFT, prefill tok/s and
/// decode tok/s. Not one of those three exists for an encoder: there is no
/// first token, no prefill/decode split, and the sequence length is fixed by
/// the compiled design rather than by the request. The axis that costs here is
/// the BATCH, so that is what this sweeps -- 1, 2, 4 ... max_batch, the same
/// doubling shape, hardest stage first for the same reason.
///
/// WHAT IT IS ACTUALLY FOR. The one property that decides this engine's
/// throughput is whether a caller batches, and until now nothing measured it.
/// AutoEmbeddingModel::embed() takes one text so /v1/embeddings loops;
/// embed_batch() encodes a whole tier per dispatch. On bge-base that is 405 ms
/// against 70 ms for sixteen texts -- a number that lived in a README because
/// no committed tool produced it. So every stage here times BOTH paths and
/// prints the ratio.
///
/// That gap is invisible to every other check in this tree, and the reason is
/// worth stating: BOTH PATHS RETURN THE SAME VECTORS. Batching is a scheduling
/// choice, not an arithmetic one, so no accuracy gate, no cosine and no
/// bit-identity test can see the slow one. The only symptom is time. This file
/// checks the vectors agree anyway, because a pure stopwatch would not have
/// noticed if the fast path were wrong.
///
/// RULE ON THE NUMBERS. Everything here is WALL CLOCK, end to end: tokenizer,
/// the host half of the encode, the array, pooling and the normalise. It is a
/// throughput and latency claim about the whole pipeline and NEVER an NPU
/// kernel claim -- the array is shared, and a wall-clock reading measures how
/// busy the machine was as much as how good the kernels are. The footer under
/// every table says so, and it is not decoration.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "benchmarking.hpp"                            // statistic_t + helpers
#include "AutoEmbeddingModel/all_embedding_model.hpp"  // get_auto_embedding_model
#include "openai_compat.hpp"                           // task_names, task_policy
#include "model_downloader.hpp"
#include "model_list.hpp"
#include "nlohmann/json.hpp"

namespace benchmarking {

/// One stage of the sweep: everything measured at a single batch size.
struct EmbedStage_t {
    int         batch = 0;
    statistic_t batched_s;      ///< one embed_batch() call
    statistic_t texts_per_s;    ///< batch / batched_s, per iteration
    statistic_t tokens_per_s;   ///< only when the backend reports a count
    statistic_t looped_s;       ///< the same texts, one embed() call at a time
    bool        have_tokens = false;
    float       speedup = 0.0f; ///< looped average over batched average
};

struct EmbedBenchResults_t {
    std::vector<EmbedStage_t> stages;   ///< ascending batch size
    std::string task_name;              ///< the REST task name requested
    std::string prompt_applied;         ///< container prompt, "" when none
    int    iterations = 0;
    int    corpus_texts = 0;
    /// Largest absolute difference between one text's vector from the batched
    /// path and from the looped one. 0.0 is the expected value; -1.0 means the
    /// comparison could not be made.
    double agreement = -1.0;
};

/// The built-in corpus.
///
/// Sixteen distinct sentences, cycled to fill a batch. Distinctness does not
/// change the timing -- there is no data-dependent branching in an encoder --
/// but the cycling is printed rather than left implicit, because a reader who
/// assumes 128 unique documents is reading a different experiment than the one
/// that ran. They sit in the length range a RAG chunk does; the design pads
/// every row to its own compiled `seq` regardless, which is exactly why text
/// length is not the interesting axis here.
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

/// Fill `n` texts by cycling the corpus.
inline std::vector<std::string> take_texts(const std::vector<std::string>& corpus, int n) {
    std::vector<std::string> out;
    out.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; i++)
        out.push_back(corpus[static_cast<size_t>(i) % corpus.size()]);
    return out;
}

inline double bench_now_s() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

/// Resolve a REST task name to the enum, reusing the server's own table.
///
/// Reused rather than re-listed on purpose: a second copy of this vocabulary
/// would drift, and then `bench-embed` would measure a prompt the endpoint
/// cannot be asked for. Empty means "query", which is what the endpoint
/// resolves an unspecified request to.
inline embedding_task_type_t task_from_name(const std::string& name) {
    if (name.empty()) return task_query;
    for (const auto& kv : openai_compat::task_names())
        if (name == kv.first) return kv.second;
    throw std::runtime_error(
        "unknown task prompt '" + name + "'. Valid names: " +
        openai_compat::task_names_csv());
}

inline void write_embed_bench_csv(const EmbedBenchResults_t& results,
                                  const std::string& model_tag,
                                  const std::string& output_dir = ".") {
    // "bench_embed_", not "bench_": the LLM benchmark writes
    // bench_<tag>_<date>[_<cpu>].csv, and benchmarking both sides of one model
    // on one day would otherwise have the second run silently overwrite the
    // first. Same convention, distinct namespace.
    const std::string safe_tag = sanitize_model_tag_for_filename(model_tag);
    const std::string date_stamp = get_date_yyyymmdd();
    const std::string cpu_name = sanitize_model_tag_for_filename(get_cpu_name());
    std::string filename = output_dir;
    if (!filename.empty() && filename.back() != '/' && filename.back() != '\\')
        filename += "/";
    filename += "bench_embed_" + safe_tag + "_" + date_stamp;
    if (!cpu_name.empty()) filename += "_" + cpu_name;
    filename += ".csv";

    std::ofstream out(filename, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        header_print_r("ERROR", "Failed to open output file: " + filename);
        return;
    }

    out << "batch,"
           "batched_avg_s,batched_std_s,batched_min_s,batched_max_s,"
           "texts_avg_per_s,texts_std_per_s,texts_min_per_s,texts_max_per_s,"
           "tokens_avg_per_s,tokens_std_per_s,tokens_min_per_s,tokens_max_per_s,"
           "looped_avg_s,looped_std_s,batch_speedup\n";

    for (const EmbedStage_t& st : results.stages) {
        out << st.batch << ","
            << format_float_csv(st.batched_s.average, 6) << ","
            << format_float_csv(st.batched_s.std_variance, 6) << ","
            << format_float_csv(st.batched_s.min, 6) << ","
            << format_float_csv(st.batched_s.max, 6) << ","
            << format_float_csv(st.texts_per_s.average, 2) << ","
            << format_float_csv(st.texts_per_s.std_variance, 2) << ","
            << format_float_csv(st.texts_per_s.min, 2) << ","
            << format_float_csv(st.texts_per_s.max, 2) << ",";
        if (st.have_tokens) {
            out << format_float_csv(st.tokens_per_s.average, 2) << ","
                << format_float_csv(st.tokens_per_s.std_variance, 2) << ","
                << format_float_csv(st.tokens_per_s.min, 2) << ","
                << format_float_csv(st.tokens_per_s.max, 2) << ",";
        } else {
            // Empty rather than 0: the backend did not report a count, and a
            // zero here would read as one.
            out << ",,,,";
        }
        out << format_float_csv(st.looped_s.average, 6) << ","
            << format_float_csv(st.looped_s.std_variance, 6) << ","
            << format_float_csv(st.speedup, 3) << "\n";
    }
    out.close();
    header_print("OFLM", "Wrote " + filename);
}

inline void print_embed_result(const EmbedBenchResults_t& results,
                               const std::string& model_tag) {
    header_print("OFLM", "=== Embedding Benchmark Results ===");
    std::cout << "\n";
    std::cout << std::setw(7)  << "Batch"       << " | "
              << std::setw(19) << "Batched (s)" << " | "
              << std::setw(10) << "Looped (s)"  << " | "
              << std::setw(8)  << "Speedup"     << " | "
              << std::setw(19) << "Texts/s"     << " | "
              << std::setw(17) << "Tokens/s"    << "\n";
    std::cout << std::string(98, '-') << "\n";

    for (const EmbedStage_t& st : results.stages) {
        std::cout << std::setw(7) << st.batch << " | ";
        std::cout << std::setw(11) << std::fixed << std::setprecision(4)
                  << st.batched_s.average << " +- " << std::setw(4)
                  << std::fixed << std::setprecision(4)
                  << st.batched_s.std_variance << " | ";
        std::cout << std::setw(10) << std::fixed << std::setprecision(4)
                  << st.looped_s.average << " | ";
        std::ostringstream sp;
        sp << std::fixed << std::setprecision(2) << st.speedup << "x";
        std::cout << std::setw(8) << sp.str() << " | ";
        std::cout << std::setw(11) << std::fixed << std::setprecision(1)
                  << st.texts_per_s.average << " +- " << std::setw(4)
                  << std::fixed << std::setprecision(1)
                  << st.texts_per_s.std_variance << " | ";
        if (st.have_tokens) {
            std::cout << std::setw(11) << std::fixed << std::setprecision(0)
                      << st.tokens_per_s.average << " +- " << std::setw(3)
                      << std::fixed << std::setprecision(0)
                      << st.tokens_per_s.std_variance;
        } else {
            std::cout << std::setw(17) << "not reported";
        }
        std::cout << "\n";
    }
    std::cout << std::string(98, '-') << "\n";

    std::cout << "  " << model_tag << ", task " << results.task_name;
    if (results.prompt_applied.empty())
        std::cout << " (model declares no prompt table -- no prefix applied)";
    else
        std::cout << " -> container prompt \"" << results.prompt_applied << "\"";
    std::cout << "\n";
    std::cout << "  corpus: built-in, " << results.corpus_texts
              << " distinct texts cycled to fill each batch; "
              << results.iterations << " iterations, 1 warm-up discarded\n";
    if (results.agreement == 0.0)
        std::cout << "  batched and looped paths returned BIT-IDENTICAL vectors\n";
    else if (results.agreement > 0.0)
        std::cout << "  WARNING: batched and looped paths DISAGREE, max abs diff "
                  << std::scientific << std::setprecision(3) << results.agreement
                  << std::fixed << "\n";
    std::cout << "  Speedup is the looped average over the batched average -- what a caller\n"
                 "  gains by sending one request with N inputs instead of N requests.\n";
    std::cout << "  Wall clock, end to end (tokenizer + host + array + pooling).\n"
                 "  NOT an NPU kernel claim: the array is shared, so wall clock also measures\n"
                 "  how busy the machine was. Quiesce the NPU before comparing two runs.\n";
    std::cout << "\n";
}

/// Run the sweep.
///
/// \param bench_config_file optional JSON: {"max_batch":N, "iterations":N,
///        "task":"document", "texts":[...]}. `max_batch` and `iterations` in
///        the file win over the CLI when present; when absent the CLI value is
///        used. That differs DELIBERATELY from `oflm bench`, whose
///        --bench-iterations never reaches a file-supplied config at all -- a
///        trap its own README documents. Same file shape, better rule.
inline EmbedBenchResults_t run_embed_benchmarks(const std::string& model_tag,
                                                const std::string& bench_config_file,
                                                model_list& availble_models,
                                                ModelDownloader& downloader,
                                                int iterations,
                                                int max_batch,
                                                const std::string& prompt_name,
                                                bool preemption,
                                                bool modelscope) {
    EmbedBenchResults_t results;
    std::vector<std::string> corpus = builtin_corpus();
    std::string task_name = prompt_name;

    if (!bench_config_file.empty()) {
        std::ifstream input_file(bench_config_file);
        if (!input_file.is_open())
            throw std::runtime_error("Failed to open bench config: " + bench_config_file);
        const nlohmann::json cfg = nlohmann::json::parse(input_file);
        input_file.close();
        if (cfg.contains("max_batch") && cfg["max_batch"].is_number_integer())
            max_batch = cfg["max_batch"].get<int>();
        if (cfg.contains("iterations") && cfg["iterations"].is_number_integer())
            iterations = cfg["iterations"].get<int>();
        if (cfg.contains("task") && cfg["task"].is_string())
            task_name = cfg["task"].get<std::string>();
        if (cfg.contains("texts") && cfg["texts"].is_array() && !cfg["texts"].empty()) {
            corpus.clear();
            for (const auto& t : cfg["texts"])
                if (t.is_string()) corpus.push_back(t.get<std::string>());
            if (corpus.empty())
                throw std::runtime_error("bench config \"texts\" contained no strings");
        }
    }

    if (max_batch < 1)
        throw std::runtime_error("--max-batch must be at least 1");
    if (iterations < 1)
        throw std::runtime_error("--bench-iterations must be at least 1");

    const embedding_task_type_t task = task_from_name(task_name);
    if (task_name.empty()) task_name = "query";

    // The model has to be on disk. Say what is happening rather than failing
    // inside load_model with a path that means nothing to the reader.
    switch (downloader.is_model_downloaded(model_tag)) {
        case ModelDownloader::ModelStatus::Ready:
            break;
        case ModelDownloader::ModelStatus::Missing:
        case ModelDownloader::ModelStatus::Outdated:
            header_print("OFLM", "Model not present or outdated -- pulling '" + model_tag + "'");
            if (!downloader.pull_model(model_tag, modelscope))
                throw std::runtime_error("failed to pull '" + model_tag +
                                         "'. Run `oflm pull " + model_tag + "` and retry.");
            break;
        case ModelDownloader::ModelStatus::Incompatible:
            throw std::runtime_error("'" + model_tag +
                                     "' is not compatible with this build of OFLM");
    }

    oflm_rt::device npu_device_inst = oflm_rt::device(0);

    // Same sequence RestHandler::ensure_embed_model_loaded runs, so the
    // benchmark exercises the load path the server exercises.
    auto [resolved_tag, engine] = get_auto_embedding_model(model_tag, &npu_device_inst);
    if (engine == nullptr)
        throw std::runtime_error(
            "cannot benchmark '" + model_tag + "': no embedding backend claimed it. "
            "Refusing to benchmark a substitute, which would report the wrong "
            "model's numbers under the right model's name.");
    auto [new_tag, model_info] = availble_models.get_model_info(resolved_tag);
    engine->load_model(availble_models.get_model_path(new_tag), model_info, preemption);

    // Whether this model takes a prompt at all, decided by the same predicate
    // the endpoint uses -- so a task this benchmark accepts is one a client
    // could also have asked for.
    const std::vector<std::string> declared = engine->prompt_names();
    const openai_compat::TaskPolicy policy = openai_compat::task_policy(
        engine->supports_task_prompts(), !declared.empty(), !prompt_name.empty());
    if (policy == openai_compat::TaskPolicy::NotSupported)
        header_print("OFLM", "'" + new_tag + "' has no task-prompt concept; the "
                             "requested task is not applied to it");
    for (const auto& kv : openai_compat::task_names())
        if (kv.second == task &&
            std::find(declared.begin(), declared.end(), kv.first) != declared.end()) {
            results.prompt_applied = kv.first;
            break;
        }

    const int stages = static_cast<int>(std::floor(std::log2((double)max_batch))) + 1;
    header_print("OFLM", "Starting embedding benchmark: " + std::to_string(stages) +
                         " stages up to batch " + std::to_string(1 << (stages - 1)) +
                         ", " + std::to_string(iterations) + " iterations");

    std::vector<std::vector<float>> batched_s(stages), texts_ps(stages),
                                    tokens_ps(stages), looped_s(stages);
    std::vector<bool> have_tokens(stages, false);

    // ---- warm-up, discarded ----
    //
    // The LLM benchmark has none and does not need one. This does: the FIRST
    // call on a model with no `.npue` container yet PACKS ONE from the
    // checkpoint, which takes tens of seconds for a 100M model. Without a
    // discarded iteration that lands inside iteration 1 and the average is
    // nonsense -- and it would read as a slow model rather than a one-off.
    {
        const int warm = 1 << (stages - 1);
        header_print("OFLM", "Warm-up at batch " + std::to_string(warm) + " (discarded)");
        std::vector<std::string> texts = take_texts(corpus, warm);
        int64_t tok = -1;
        const std::vector<float> hot = engine->embed_batch(texts, task, &tok);

        // Do the two paths agree? A stopwatch cannot tell, and the whole
        // reason this benchmark exists is a difference no accuracy gate sees.
        // Compare the FIRST text through both -- if batching changed the
        // arithmetic, this is where it shows.
        std::string one = corpus[0];
        const std::vector<float> single = engine->embed(one, task);
        double worst = 0.0;
        if (!single.empty() && hot.size() >= single.size()) {
            for (size_t i = 0; i < single.size(); i++)
                worst = std::max(worst, std::abs((double)hot[i] - (double)single[i]));
            results.agreement = worst;
        }
        if (results.agreement > 0.0)
            header_print_r("WARN", "batched and looped vectors differ by " +
                                   std::to_string(results.agreement) +
                                   " -- the speedup below is not a like-for-like "
                                   "comparison");
    }

    // Hardest stage first, like the LLM benchmark: a run killed early has then
    // measured the expensive end rather than nothing.
    for (int it = 0; it < iterations; it++) {
        for (int s = stages - 1; s >= 0; s--) {
            const int n = 1 << s;
            header_print("OFLM", "batch " + std::to_string(n) + ", iteration " +
                                 std::to_string(it + 1) + "...");
            std::vector<std::string> texts = take_texts(corpus, n);

            int64_t tok = -1;
            const double b0 = bench_now_s();
            (void)engine->embed_batch(texts, task, &tok);
            const double b1 = bench_now_s();

            const double l0 = bench_now_s();
            for (std::string& t : texts) (void)engine->embed(t, task);
            const double l1 = bench_now_s();

            const double bs = b1 - b0;
            batched_s[s].push_back((float)bs);
            looped_s[s].push_back((float)(l1 - l0));
            if (bs > 0.0) texts_ps[s].push_back((float)((double)n / bs));
            if (tok > 0 && bs > 0.0) {
                tokens_ps[s].push_back((float)((double)tok / bs));
                have_tokens[s] = true;
            }
            // Same one-second gap the LLM benchmark leaves, same reason.
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    for (int s = 0; s < stages; s++) {
        EmbedStage_t st;
        st.batch = 1 << s;
        st.batched_s.calculate_statistics(batched_s[s]);
        st.texts_per_s.calculate_statistics(texts_ps[s]);
        st.looped_s.calculate_statistics(looped_s[s]);
        st.have_tokens = have_tokens[s];
        if (st.have_tokens) st.tokens_per_s.calculate_statistics(tokens_ps[s]);
        st.speedup = (st.batched_s.average > 0.0f)
                         ? st.looped_s.average / st.batched_s.average
                         : 0.0f;
        results.stages.push_back(st);
    }

    results.task_name = task_name;
    results.iterations = iterations;
    results.corpus_texts = (int)corpus.size();

    engine.reset();

    print_embed_result(results, new_tag);
    write_embed_bench_csv(results, new_tag, ".");
    return results;
}

}  // namespace benchmarking

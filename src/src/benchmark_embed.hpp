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
/// AutoEmbeddingModel::embed() takes one text so a naive caller loops;
/// embed_batch() encodes a whole tier per dispatch. On bge-base that is 405 ms
/// against 70 ms for sixteen texts -- a number that lived in a README because
/// no committed tool produced it. So every stage here times BOTH paths and
/// prints the ratio.
///
/// That gap is invisible to every other check in this tree, and the reason is
/// worth stating: BOTH PATHS RETURN THE SAME VECTORS. Batching is a scheduling
/// choice, not an arithmetic one, so no accuracy gate, no cosine and no
/// bit-identity test can see the slow one. The only symptom is time. This file
/// checks the vectors agree anyway -- over EVERY row, see the identity gate --
/// because a pure stopwatch would not have noticed if the fast path were wrong.
///
/// RULE ON THE NUMBERS. Everything here is WALL CLOCK, end to end: tokenizer,
/// the host half of the encode, the array, pooling and the normalise. It is a
/// throughput and latency claim about the whole pipeline and NEVER an NPU
/// kernel claim -- the array is shared, and a wall-clock reading measures how
/// busy the machine was as much as how good the kernels are. The footer under
/// every table says so, and it is not decoration.
///
/// The pure half of this -- the sweep plan, the stage count, the task lookup,
/// the corpus -- lives in benchmark_embed_util.hpp so it can be unit-tested
/// without a device. See benchmark_embed_test.cpp.
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
#include "benchmark_embed_util.hpp"                    // the pure, tested half
#include "AutoEmbeddingModel/all_embedding_model.hpp"  // get_auto_embedding_model
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
    std::string prompt_applied;         ///< container prompt name, when there is one
    PrefixKind  prefix_kind = PrefixKind::None;
    int    iterations = 0;
    int    corpus_texts = 0;
    CorpusSource corpus_source = CorpusSource::BuiltIn;
    /// The identity gate: how many vectors were compared between the batched
    /// and the looped path, and the largest absolute difference found. -1.0
    /// means the comparison did not run.
    double agreement = -1.0;
    size_t agreement_vectors = 0;
};

inline double bench_now_s() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
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

    // Provenance first. Two runs of one model differing only in --prompt-name,
    // or one of them using a config-supplied corpus, used to write
    // indistinguishable files -- and the numbers really do differ. `#` lines
    // are skippable (pandas: comment="#"); `oflm bench`'s CSV has none, so
    // nothing depends on their absence.
    out << "# oflm bench-embed\n"
        << "# model=" << model_tag << "\n"
        << "# task=" << results.task_name << "\n"
        << "# prefix=" << (results.prefix_kind == PrefixKind::ContainerPrompt
                               ? (results.prompt_applied.empty()
                                      ? "container-prompt-unnamed"
                                      : results.prompt_applied)
                               : results.prefix_kind == PrefixKind::BackendHardcoded
                                     ? "backend-hardcoded"
                                     : "none") << "\n"
        << "# corpus=" << corpus_source_name(results.corpus_source)
        << " texts=" << results.corpus_texts << "\n"
        << "# iterations=" << results.iterations << " warmup=1\n"
        << "# identity_gate_vectors=" << results.agreement_vectors
        << " max_abs_diff=" << results.agreement << "\n"
        << "# wall clock, end to end; NOT an NPU kernel claim\n";
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
    switch (results.prefix_kind) {
        case PrefixKind::ContainerPrompt:
            if (results.prompt_applied.empty())
                std::cout << " -> a prompt from the container's own table that this"
                             " build cannot name";
            else
                std::cout << " -> container prompt \"" << results.prompt_applied << "\"";
            break;
        case PrefixKind::BackendHardcoded:
            // OpenGemma declares no prompt NAMES and still prefixes every text.
            // Saying "no prefix applied" here, as the first version did, was
            // simply false.
            std::cout << " -> a prefix the backend hardcodes per task"
                         " (it declares no prompt names)";
            break;
        case PrefixKind::None:
            std::cout << " (this model has no task-prompt concept; no prefix applied)";
            break;
    }
    std::cout << "\n";
    std::cout << "  corpus: " << corpus_source_name(results.corpus_source) << ", "
              << results.corpus_texts
              << " distinct texts cycled to fill each batch; "
              << results.iterations << " iterations, 1 warm-up discarded\n";
    if (results.agreement == 0.0)
        std::cout << "  identity gate: all " << results.agreement_vectors
                  << " vectors of the largest batch are BIT-IDENTICAL to the same"
                     " text embedded alone\n";
    else if (results.agreement > 0.0)
        std::cout << "  WARNING: batched and looped paths DISAGREE over "
                  << results.agreement_vectors << " vectors, max abs diff "
                  << std::scientific << std::setprecision(3) << results.agreement
                  << std::fixed << "\n";
    else
        std::cout << "  identity gate: DID NOT RUN\n";
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
///        "task":"document", "texts":[...]}. See make_embed_bench_plan() for
///        the precedence rules.
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

    nlohmann::json cfg;
    bool have_cfg = false;
    if (!bench_config_file.empty()) {
        std::ifstream input_file(bench_config_file);
        if (!input_file.is_open())
            throw std::runtime_error("Failed to open bench config: " + bench_config_file);
        cfg = nlohmann::json::parse(input_file);
        input_file.close();
        have_cfg = true;
    }

    // Everything the run depends on, decided and validated before anything is
    // downloaded, loaded or timed.
    const EmbedBenchPlan plan = make_embed_bench_plan(cfg, have_cfg, iterations,
                                                      max_batch, prompt_name);
    const embedding_task_type_t task = task_from_name(plan.task_name);
    const int stages = bench_stages(plan.max_batch);

    // Canonicalise the tag FIRST. model_list::all_tags accepts the shorthand
    // ("bge-base"), so main.cpp's model-support check passes it, but
    // get_auto_embedding_model() matches on the full tag -- so the shorthand
    // used to clear every check and then fail as an unknown embedding model.
    auto [canonical_tag, model_info] = availble_models.get_model_info(model_tag);

    oflm_rt::device npu_device_inst = oflm_rt::device(0);

    // Resolve the BACKEND before touching the network. A valid chat tag such as
    // llama3.2:1b otherwise triggered a multi-gigabyte download and only then
    // failed as an unknown embedding model. Constructing the backend is cheap
    // -- it stores a tag; load_model() is what opens anything.
    auto [resolved_tag, engine] = get_auto_embedding_model(canonical_tag, &npu_device_inst);
    if (engine == nullptr)
        throw std::runtime_error(
            "cannot benchmark '" + canonical_tag + "': no embedding backend claimed it. "
            "Refusing to benchmark a substitute, which would report the wrong "
            "model's numbers under the right model's name.");

    switch (downloader.is_model_downloaded(resolved_tag)) {
        case ModelDownloader::ModelStatus::Ready:
            break;
        case ModelDownloader::ModelStatus::Missing:
        case ModelDownloader::ModelStatus::Outdated:
            header_print("OFLM", "Model not present or outdated -- pulling '" +
                                 resolved_tag + "'");
            if (!downloader.pull_model(resolved_tag, modelscope))
                throw std::runtime_error("failed to pull '" + resolved_tag +
                                         "'. Run `oflm pull " + resolved_tag +
                                         "` and retry.");
            break;
        case ModelDownloader::ModelStatus::Incompatible:
            throw std::runtime_error("'" + resolved_tag +
                                     "' is not compatible with this build of OFLM");
    }

    // Same sequence RestHandler::ensure_embed_model_loaded runs, so the
    // benchmark exercises the load path the server exercises. NOTE that this is
    // also where a missing `.npue` container gets PACKED, by find_container()
    // inside load_model() -- i.e. before the warm-up below, and outside every
    // timed iteration already.
    engine->load_model(availble_models.get_model_path(resolved_tag), model_info,
                       preemption);

    // Does the endpoint accept the request this is about to time? Decided by
    // the endpoint's own predicate, so the answer cannot drift from it.
    const std::vector<std::string> declared = engine->prompt_names();
    const std::string refusal = task_policy_refusal(
        engine->supports_task_prompts(), !declared.empty(), plan.task_explicit,
        declared);
    if (!refusal.empty())
        throw std::runtime_error("cannot benchmark '" + resolved_tag + "': " + refusal);

    // What prefix actually gets applied -- three cases, because "declares no
    // prompt names" means two different things.
    if (!declared.empty()) {
        // A prompt IS applied -- the engine would have thrown otherwise. Set the
        // kind first and fill the name only if this build can name it: the
        // adapter matches candidates such as "Retrieval" that
        // openai_compat::task_names() does not list, and a failed name lookup is
        // not evidence that no prefix ran.
        results.prefix_kind = PrefixKind::ContainerPrompt;
        for (const auto& kv : openai_compat::task_names())
            if (kv.second == task &&
                std::find(declared.begin(), declared.end(), kv.first) != declared.end()) {
                results.prompt_applied = kv.first;
                break;
            }
    } else if (engine->supports_task_prompts()) {
        results.prefix_kind = PrefixKind::BackendHardcoded;
    } else {
        results.prefix_kind = PrefixKind::None;
    }

    header_print("OFLM", "Starting embedding benchmark: " + std::to_string(stages) +
                         " stages up to batch " + std::to_string(1 << (stages - 1)) +
                         ", " + std::to_string(plan.iterations) + " iterations");

    std::vector<std::vector<float>> batched_s(stages), texts_ps(stages),
                                    tokens_ps(stages), looped_s(stages);
    std::vector<bool> have_tokens(stages, false);

    // ---- warm-up and identity gate, both discarded from the timings ----
    //
    // WHAT THE WARM-UP IS NOT FOR. An earlier version of this comment, and of
    // the README, claimed it kept `.npue` packing out of iteration 1. That was
    // wrong: load_model() above calls find_container(), which packs, so packing
    // was already outside the timed loop. What this call actually excludes is
    // first-call runtime cost -- faulting in the mmapped container, the
    // tokenizer's first use, and the lanes' first dispatch.
    //
    // THE IDENTITY GATE. Every row of the largest batch is compared against the
    // same text embedded alone. The first version compared only the first row,
    // which is a probe whose coverage nobody checked: a batch-specific
    // ordering, truncation or write error in any later row would still have
    // printed BIT-IDENTICAL. It costs N extra single calls, once.
    {
        const int warm = 1 << (stages - 1);
        header_print("OFLM", "Warm-up and identity gate at batch " +
                             std::to_string(warm) + " (discarded from the timings)");
        const std::vector<std::string> texts = take_texts(plan.corpus, warm);
        int64_t tok = -1;
        const std::vector<float> hot = engine->embed_batch(texts, task, &tok);
        const size_t dim = openai_compat::embedding_batch_dim(hot.size(), texts.size());

        double worst = 0.0;
        for (size_t i = 0; i < texts.size(); ++i) {
            std::string one = texts[i];
            const std::vector<float> single = engine->embed(one, task);
            if (single.size() != dim)
                throw std::runtime_error(
                    "embed() returned " + std::to_string(single.size()) +
                    " floats while embed_batch() implies " + std::to_string(dim) +
                    ". The two paths do not agree on the vector width, so the"
                    " speedup below would not be a like-for-like comparison.");
            for (size_t k = 0; k < dim; ++k)
                worst = std::max(worst, std::abs(static_cast<double>(hot[i * dim + k]) -
                                                 static_cast<double>(single[k])));
            ++results.agreement_vectors;
        }
        results.agreement = worst;
        if (worst != 0.0)
            header_print_r("WARN", "batched and looped vectors differ by " +
                                   std::to_string(worst) +
                                   " -- the speedup below is not a like-for-like "
                                   "comparison");
    }

    // Hardest stage first, like the LLM benchmark: a run killed early has then
    // measured the expensive end rather than nothing.
    for (int it = 0; it < plan.iterations; it++) {
        for (int s = stages - 1; s >= 0; s--) {
            const int n = 1 << s;
            header_print("OFLM", "batch " + std::to_string(n) + ", iteration " +
                                 std::to_string(it + 1) + "...");
            std::vector<std::string> texts = take_texts(plan.corpus, n);

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

    results.task_name = plan.task_name.empty() ? std::string("query") : plan.task_name;
    results.iterations = plan.iterations;
    results.corpus_texts = (int)plan.corpus.size();
    results.corpus_source = plan.corpus_source;

    engine.reset();

    print_embed_result(results, resolved_tag);
    write_embed_bench_csv(results, resolved_tag, ".");
    return results;
}

}  // namespace benchmarking

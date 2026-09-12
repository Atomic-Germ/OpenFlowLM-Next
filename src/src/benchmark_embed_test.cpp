/// \file benchmark_embed_test.cpp
/// \brief Unit tests for the pure half of `oflm bench-embed`.
///
/// SPDX-License-Identifier: MIT
///
/// WHY THESE EXIST. Every assertion below corresponds to a way this command
/// could produce a number that looks fine and is not:
///
///   * `--max-batch 3` swept 1 and 2 and never 3, so the option documented as
///     "the largest batch swept" quietly swept something else.
///   * a `task` in the config file read as "no task given", so a model that
///     requires a prompt was benchmarked without one -- timing a request the
///     endpoint answers 400 to.
///   * a custom corpus was reported as the built-in one, making the run's own
///     reproducibility note false.
///   * a shorthand tag (`bge-base`) passed the command's model check and then
///     failed as an unknown embedding model.
///   * a concatenated batch result was sliced by a derived width with nothing
///     checking the division.
///
/// None of those is a crash and none changes a vector, so no smoke test and no
/// accuracy gate could see any of them. They are all decided before a single
/// measurement is taken, which is exactly why they can be tested with no
/// device, no weights and no network.
///
///   ctest --test-dir src/build -R bench_embed
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "AutoModel/model_families.hpp"   // is_chat_model
#include "benchmark_embed_util.hpp"
#include "model_list.hpp"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using namespace benchmarking;
using njson = nlohmann::json;   // NOT the global `json`, which is ordered_json

static int failures = 0;
static int checks = 0;

static void ok(bool cond, const std::string& what) {
    ++checks;
    if (cond) {
        std::printf("ok    %s\n", what.c_str());
    } else {
        ++failures;
        std::printf("FAIL  %s\n", what.c_str());
    }
}

static void eqi(long long got, long long want, const std::string& what) {
    ok(got == want, what + (got == want ? "" : "  (got " + std::to_string(got) +
                                              ", want " + std::to_string(want) + ")"));
}

static void eqs(const std::string& got, const std::string& want, const std::string& what) {
    ok(got == want, what + (got == want ? "" : "  (got \"" + got + "\", want \"" + want + "\")"));
}

/// Run `fn` and report whether it threw, and whether the message mentions
/// `needle`. A refusal whose message does not say what to do is only half a
/// refusal, and several of these messages are the only guidance a user gets.
template <typename F>
static void throws_with(F fn, const std::string& needle, const std::string& what) {
    ++checks;
    try {
        fn();
        ++failures;
        std::printf("FAIL  %s  (did not throw)\n", what.c_str());
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        if (msg.find(needle) != std::string::npos) {
            std::printf("ok    %s\n", what.c_str());
        } else {
            ++failures;
            std::printf("FAIL  %s  (threw, but the message does not mention \"%s\": %s)\n",
                        what.c_str(), needle.c_str(), msg.c_str());
        }
    }
}

template <typename F>
static void does_not_throw(F fn, const std::string& what) {
    ++checks;
    try {
        fn();
        std::printf("ok    %s\n", what.c_str());
    } catch (const std::exception& e) {
        ++failures;
        std::printf("FAIL  %s  (threw: %s)\n", what.c_str(), e.what());
    }
}

// ---------------------------------------------------------------------------
// bench_stages -- the sweep's shape, and the limits it must refuse
// ---------------------------------------------------------------------------
static void test_bench_stages() {
    std::printf("\n-- bench_stages --\n");
    eqi(bench_stages(1), 1, "max-batch 1 -> 1 stage");
    eqi(bench_stages(2), 2, "max-batch 2 -> 2 stages");
    eqi(bench_stages(4), 3, "max-batch 4 -> 3 stages");
    eqi(bench_stages(32), 6, "max-batch 32 -> 6 stages");
    eqi(bench_stages(128), 8, "max-batch 128 -> 8 stages");
    eqi(bench_stages(1024), 11, "max-batch 1024 -> 11 stages");

    // The whole point: a limit the doubling cannot land on is refused, not
    // silently rounded down. 3 used to give two stages and never run 3.
    throws_with([] { bench_stages(3); }, "power of two", "max-batch 3 refuses");
    throws_with([] { bench_stages(100); }, "power of two", "max-batch 100 refuses");
    throws_with([] { bench_stages(129); }, "power of two", "max-batch 129 refuses");
    throws_with([] { bench_stages(0); }, "at least 1", "max-batch 0 refuses");
    throws_with([] { bench_stages(-8); }, "at least 1", "max-batch -8 refuses");

    // A power of two can still be absurd. 65536 is 17 stages of up to 65,536
    // texts: it does not fail fast, it looks like a hang.
    eqi(bench_stages(kMaxBatchCeiling), 14, "the ceiling itself is accepted");
    throws_with([] { bench_stages(kMaxBatchCeiling * 2); }, "beyond the",
                "a power of two above the ceiling refuses");
    throws_with([] { bench_stages(1 << 20); }, "beyond the",
                "an accidental extra zero refuses rather than running for an afternoon");

    // And the refusal has to explain itself, because "3 is not a power of two"
    // is not obviously a problem until you know the sweep doubles.
    throws_with([] { bench_stages(3); }, "never 3",
                "the refusal names what would have happened instead");

    // Every accepted value must actually reach its own limit -- the property
    // the old log2() version violated.
    bool reaches = true;
    for (int b = 1; b <= 1024; b <<= 1)
        if ((1 << (bench_stages(b) - 1)) != b) reaches = false;
    ok(reaches, "every accepted max-batch is the largest stage the sweep runs");
}

// ---------------------------------------------------------------------------
// task_from_name -- prompt_name, the field with a wrong answer that looks right
// ---------------------------------------------------------------------------
static void test_task_from_name() {
    std::printf("\n-- task_from_name --\n");
    ok(task_from_name("") == task_query,
       "an omitted prompt-name means query, as /v1/embeddings resolves it");
    ok(task_from_name("query") == task_query, "query");
    ok(task_from_name("search_query") == task_query, "search_query is an alias of query");
    ok(task_from_name("document") == task_document, "document");
    ok(task_from_name("search_document") == task_document, "search_document");
    ok(task_from_name("clustering") == task_clustering, "clustering");
    ok(task_from_name("Classification") == task_classification, "Classification");
    ok(task_from_name("STS") == task_sentence_similarity, "STS");

    // Every name the endpoint accepts must be accepted here too, or the
    // benchmark cannot measure a request a client can send.
    bool all_endpoint_names_resolve = true;
    for (const auto& kv : openai_compat::task_names()) {
        try {
            if (task_from_name(kv.first) != kv.second) all_endpoint_names_resolve = false;
        } catch (const std::exception&) {
            all_endpoint_names_resolve = false;
        }
    }
    ok(all_endpoint_names_resolve,
       "every name in the endpoint's own table resolves to the same task here");

    throws_with([] { task_from_name("tullball"); }, "unknown task prompt",
                "an unknown prompt-name refuses");
    // Case matters in this vocabulary ("Classification" vs "classification" are
    // both present, "QUERY" is not), so a near-miss must not be accepted.
    throws_with([] { task_from_name("QUERY"); }, "unknown task prompt",
                "prompt-name matching is case-sensitive");
    throws_with([] { task_from_name("query "); }, "unknown task prompt",
                "a trailing space is not silently trimmed");
    // The refusal must list the valid names -- it is the only place a user
    // learns them.
    throws_with([] { task_from_name("nope"); }, "search_document",
                "the refusal lists the valid names");
}

// ---------------------------------------------------------------------------
// take_texts -- the corpus, cycled
// ---------------------------------------------------------------------------
static void test_take_texts() {
    std::printf("\n-- take_texts --\n");
    const std::vector<std::string> three = {"a", "b", "c"};
    eqi((long long)take_texts(three, 0).size(), 0, "n=0 gives nothing");
    eqi((long long)take_texts(three, 2).size(), 2, "n below the corpus size");
    eqi((long long)take_texts(three, 7).size(), 7, "n above the corpus size");
    eqs(take_texts(three, 7)[3], "a", "cycling wraps at the corpus length");
    eqs(take_texts(three, 7)[6], "a", "and keeps wrapping");
    eqs(take_texts(three, 2)[1], "b", "order is preserved");

    eqi((long long)builtin_corpus().size(), 16, "the built-in corpus has 16 texts");
    bool distinct = true;
    for (size_t i = 0; i < builtin_corpus().size(); ++i)
        for (size_t j = i + 1; j < builtin_corpus().size(); ++j)
            if (builtin_corpus()[i] == builtin_corpus()[j]) distinct = false;
    ok(distinct, "the built-in corpus texts are actually distinct");

    throws_with([] { take_texts({}, 4); }, "corpus is empty", "an empty corpus refuses");
    throws_with([&three] { take_texts(three, -1); }, "negative", "a negative count refuses");
}

// ---------------------------------------------------------------------------
// make_embed_bench_plan -- the config file, and the CLI it must not shadow
// ---------------------------------------------------------------------------
static void test_plan() {
    std::printf("\n-- make_embed_bench_plan --\n");

    // No config: the CLI decides everything.
    {
        const EmbedBenchPlan p = make_embed_bench_plan(njson::object(), false, 3, 32, "");
        eqi(p.iterations, 3, "no config: iterations from the CLI");
        eqi(p.max_batch, 32, "no config: max_batch from the CLI");
        eqs(p.task_name, "", "no config: no task name");
        ok(!p.task_explicit, "no config and no --prompt-name: the task is not explicit");
        ok(p.corpus_source == CorpusSource::BuiltIn, "no config: the built-in corpus");
        eqi((long long)p.corpus.size(), 16, "no config: 16 built-in texts");
    }

    // A config wins per key, and fills nothing it does not mention. `oflm
    // bench` ignores --bench-iterations entirely once -i is given; this must
    // not.
    {
        njson cfg = {{"max_batch", 16}};
        const EmbedBenchPlan p = make_embed_bench_plan(cfg, true, 7, 128, "");
        eqi(p.max_batch, 16, "the config's max_batch wins");
        eqi(p.iterations, 7, "the CLI's iterations still applies when the config omits it");
    }
    {
        njson cfg = {{"iterations", 5}};
        const EmbedBenchPlan p = make_embed_bench_plan(cfg, true, 2, 8, "");
        eqi(p.iterations, 5, "the config's iterations wins");
        eqi(p.max_batch, 8, "the CLI's max_batch still applies when the config omits it");
    }

    // THE BUG: a task in the config file is just as explicit as one on the
    // command line. Reading only the CLI string made a config-supplied task
    // invisible, so a model that requires a prompt was refused -- or worse,
    // benchmarked without one.
    {
        njson cfg = {{"task", "document"}};
        const EmbedBenchPlan p = make_embed_bench_plan(cfg, true, 2, 8, "");
        eqs(p.task_name, "document", "the config's task is read");
        ok(p.task_explicit, "a task from the CONFIG FILE counts as explicit");
    }
    {
        const EmbedBenchPlan p = make_embed_bench_plan(njson::object(), false, 2, 8, "document");
        ok(p.task_explicit, "a task from --prompt-name counts as explicit");
    }
    {
        njson cfg = {{"task", "clustering"}};
        const EmbedBenchPlan p = make_embed_bench_plan(cfg, true, 2, 8, "query");
        eqs(p.task_name, "clustering", "the config's task overrides --prompt-name");
    }

    // A custom corpus must be REPORTED as custom; labelling it "built-in" made
    // the run's own reproducibility note false.
    {
        njson cfg = {{"texts", {"one", "two"}}};
        const EmbedBenchPlan p = make_embed_bench_plan(cfg, true, 2, 8, "");
        eqi((long long)p.corpus.size(), 2, "the config's texts replace the corpus");
        ok(p.corpus_source == CorpusSource::Config, "and the source is recorded as the config");
        eqs(corpus_source_name(p.corpus_source), "from the config file",
            "which is what gets printed");
        eqs(corpus_source_name(CorpusSource::BuiltIn), "built-in",
            "and the built-in label is unchanged");
    }

    // Wrong syntax in the config: refuse, and say which key.
    throws_with([] { njson c = {{"max_batch", "128"}};
                     make_embed_bench_plan(c, true, 2, 8, ""); },
                "max_batch", "a string max_batch refuses");
    throws_with([] { njson c = {{"iterations", 1.5}};
                     make_embed_bench_plan(c, true, 2, 8, ""); },
                "iterations", "a non-integer iterations refuses");
    throws_with([] { njson c = {{"task", 7}};
                     make_embed_bench_plan(c, true, 2, 8, ""); },
                "task", "a non-string task refuses");
    throws_with([] { njson c = {{"texts", "not an array"}};
                     make_embed_bench_plan(c, true, 2, 8, ""); },
                "texts", "a string texts refuses");
    throws_with([] { njson c = {{"texts", {1, 2, 3}}};
                     make_embed_bench_plan(c, true, 2, 8, ""); },
                "must be a string", "texts of numbers refuses");
    throws_with([] { njson c = {{"texts", njson::array()}};
                     make_embed_bench_plan(c, true, 2, 8, ""); },
                "no strings", "an empty texts array refuses");

    // A bad limit or a bad task name is caught while planning, not after the
    // model has been downloaded and loaded.
    throws_with([] { njson c = {{"max_batch", 3}};
                     make_embed_bench_plan(c, true, 2, 8, ""); },
                "power of two", "a non-power-of-two in the config refuses at plan time");
    throws_with([] { njson c = {{"task", "tullball"}};
                     make_embed_bench_plan(c, true, 2, 8, ""); },
                "unknown task prompt", "an unknown task in the config refuses at plan time");
    throws_with([] { make_embed_bench_plan(njson::object(), false, 0, 8, ""); },
                "at least 1", "zero iterations refuses");
    throws_with([] { make_embed_bench_plan(njson::object(), false, 2, 1 << 20, ""); },
                "beyond the", "a max_batch above the ceiling refuses at plan time");

    // A config file whose ROOT is not an object was silently ignored:
    // nlohmann contains() is `is_object() && ...`, so every key read as
    // absent and the sweep ran on the CLI defaults with no sign of it.
    throws_with([] { make_embed_bench_plan(njson::array(), true, 2, 8, ""); },
                "must hold a JSON object", "a config whose root is an array refuses");
    throws_with([] { make_embed_bench_plan(njson(), true, 2, 8, ""); },
                "must hold a JSON object", "a null config root refuses");
    throws_with([] { make_embed_bench_plan(njson(42), true, 2, 8, ""); },
                "must hold a JSON object", "a scalar config root refuses");
    throws_with([] { make_embed_bench_plan(njson::array(), true, 2, 8, ""); },
                "command-line values instead",
                "and the refusal says what would have happened silently");
    // ...while an OBJECT root with no keys at all is fine: it means "use the
    // CLI values", said explicitly.
    does_not_throw([] { make_embed_bench_plan(njson::object(), true, 2, 8, ""); },
                   "an empty object config is accepted");

    // An unknown key is NOT an error: the shipped config carries a `_comment`,
    // and `oflm bench`'s configs do too.
    does_not_throw([] { njson c = {{"_comment", "why this file exists"}, {"max_batch", 8}};
                        make_embed_bench_plan(c, true, 2, 8, ""); },
                   "an unknown key such as _comment is ignored, not refused");
}

// ---------------------------------------------------------------------------
// task_policy_refusal -- the three backend shapes this tree actually has
// ---------------------------------------------------------------------------
static void test_task_policy_refusal() {
    std::printf("\n-- task_policy_refusal --\n");
    const std::vector<std::string> none;
    const std::vector<std::string> nomic = {"search_query", "search_document",
                                            "clustering", "classification"};

    // The BERT family: no prompt names, no task concept.
    ok(task_policy_refusal(false, false, false, none).empty(),
       "bge/minilm without a task: fine");
    throws_with([&] { const std::string r = task_policy_refusal(false, false, true, none);
                      if (!r.empty()) throw std::runtime_error(r); },
                "no task-prompt concept",
                "bge/minilm WITH a task: refused, as the endpoint refuses it");

    // nomic: declares names, and /v1/embeddings requires one.
    throws_with([&] { const std::string r = task_policy_refusal(true, true, false, nomic);
                      if (!r.empty()) throw std::runtime_error(r); },
                "requires one",
                "nomic without a task: refused rather than timed under an implicit query");
    throws_with([&] { const std::string r = task_policy_refusal(true, true, false, nomic);
                      if (!r.empty()) throw std::runtime_error(r); },
                "search_document",
                "and the refusal names what the model itself offers");
    ok(task_policy_refusal(true, true, true, nomic).empty(),
       "nomic with a task: fine");

    // OpenGemma: declares NO names and still prefixes every text. This is the
    // case that collapsing "no names" into "no prefix" got wrong.
    ok(task_policy_refusal(true, false, false, none).empty(),
       "embed-gemma without a task: fine, it has a per-task default");
    ok(task_policy_refusal(true, false, true, none).empty(),
       "embed-gemma with a task: fine, it honours tasks without declaring names");

    // The invariant: nothing lets a task reach a backend that would drop it.
    bool never_dropped = true;
    for (bool declares : {true, false})
        if (task_policy_refusal(false, declares, true, none).empty()) never_dropped = false;
    ok(never_dropped, "no combination lets a task reach a backend that ignores it");
}

// ---------------------------------------------------------------------------
// embedding_batch_dim -- slicing a concatenated result
// ---------------------------------------------------------------------------
static void test_batch_dim() {
    std::printf("\n-- embedding_batch_dim --\n");
    eqi((long long)openai_compat::embedding_batch_dim(768, 1), 768, "one 768-wide vector");
    eqi((long long)openai_compat::embedding_batch_dim(768 * 16, 16), 768, "sixteen of them");
    eqi((long long)openai_compat::embedding_batch_dim(384 * 128, 128), 384, "a 384-wide model");

    throws_with([] { openai_compat::embedding_batch_dim(768 * 16 - 1, 16); },
                "does not divide evenly", "a truncated result refuses");
    throws_with([] { openai_compat::embedding_batch_dim(1000, 16); },
                "does not divide evenly", "a result that does not divide refuses");
    throws_with([] { openai_compat::embedding_batch_dim(0, 4); },
                "does not divide evenly", "an empty result refuses");
    throws_with([] { openai_compat::embedding_batch_dim(768, 0); },
                "zero inputs", "zero inputs refuses");

    // The refusal must say why it will not guess, because a caller who "fixes"
    // it by dividing anyway reintroduces the exact defect.
    throws_with([] { openai_compat::embedding_batch_dim(1000, 16); },
                "wrong inputs", "the refusal explains what a mis-split would return");
}

// ---------------------------------------------------------------------------
// The model catalogue: wrong model, and the shorthand this command must accept
// ---------------------------------------------------------------------------
static void test_model_tags(const std::string& list_path) {
    std::printf("\n-- model tags (%s) --\n", list_path.c_str());
    std::string path = list_path, exe_dir = ".";   // the ctor takes non-const references
    model_list ml(path, exe_dir);

    // The seven tags the embedding registry serves, and the shorthand a user
    // will type. `all_tags` accepts the shorthand, so main.cpp's model check
    // passes it -- which is why bench-embed has to canonicalise before asking
    // the registry, and why this asserts the canonical form is what we expect.
    const std::vector<std::pair<const char*, const char*>> shorthand = {
        {"bge-base", "bge-base:en-v1.5"},
        {"bge-small", "bge-small:en-v1.5"},
        {"bge-large", "bge-large:en-v1.5"},
        {"all-minilm", "all-minilm:l6-v2"},
        {"nomic-embed-text", "nomic-embed-text:v1.5"},
        {"gte-multilingual", "gte-multilingual:base"},
        {"embed-gemma", "embed-gemma:300m"},
    };
    for (const auto& [shortf, full] : shorthand) {
        // Absence is a FAILURE, not a skip. This list is the set of tags the
        // embedding registry advertises, so a tag missing from model_list.json
        // means `bench-embed <tag>` fails its own model check -- and a test
        // that skips it reports success for a command that cannot run.
        ok(ml.is_model_supported(full),
           std::string(full) + " is present in model_list.json");
        if (!ml.is_model_supported(full)) continue;
        ok(ml.is_model_supported(shortf),
           std::string("the shorthand '") + shortf + "' is accepted by the model list");
        auto [canonical, info] = ml.get_model_info(shortf);
        eqs(canonical, full,
            std::string("'") + shortf + "' canonicalises to '" + full + "'");
        (void)info;

        // An embedding model must NOT look like a chat model: that is what
        // keeps `bench` and `bench-embed` from being pointed at each other's
        // models.
        ok(!is_chat_model(full, ml),
           std::string(full) + " is not a chat model");
    }

    // ... and the converse. A chat tag is a valid model_list entry, so only the
    // embedding registry can reject it -- which is the reason bench-embed
    // resolves the backend BEFORE it downloads anything.
    if (ml.is_model_supported("llama3.2:1b")) {
        ok(is_chat_model("llama3.2:1b", ml), "llama3.2:1b IS a chat model");
    } else {
        std::printf("skip  llama3.2:1b is not in this model_list.json\n");
    }

    ok(!ml.is_model_supported("finnes-ikke:1b"), "a nonexistent tag is not supported");
    ok(!ml.is_model_supported("model-faker"),
       "'model-faker' -- what an omitted positional looks like -- is not a real tag");
}

int main(int argc, char** argv) {
    const std::string list_path = argc > 1 ? argv[1] : "model_list.json";

    test_bench_stages();
    test_task_from_name();
    test_take_texts();
    test_plan();
    test_task_policy_refusal();
    test_batch_dim();

    if (fs::exists(list_path)) {
        test_model_tags(list_path);
    } else {
        std::printf("\nFATAL model_list.json not found at '%s' -- pass its path as argv[1]\n",
                    list_path.c_str());
        return 2;
    }

    std::printf("\n%s (%d checks, %d failures)\n", failures ? "FAILED" : "PASS",
                checks, failures);
    return failures ? 1 : 0;
}

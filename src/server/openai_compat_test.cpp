/// \file openai_compat_test.cpp
/// \brief Unit tests for the OpenAI wire vocabulary and the chat-model predicate.
///
/// Every defect #52 fixes has the same shape: a wrong answer that is WELL FORMED,
/// so nothing downstream can tell. HTTP 200 with an error body parses. A vector
/// under the wrong task prompt is correctly normed. A Llama3 renamed to
/// "llama3.2:1b" answers fluently. That is why these assertions are worth having
/// and why an end-to-end smoke test would not have caught any of them: the server
/// was never down.
///
/// No device, no model weights, no network -- the four things under test are pure
/// functions plus one that reads model_list.json off disk.
///
///   Standalone:  see src/open_qwen36/build.cmd's sibling invocation, or
///                ctest --test-dir build -R openai_compat
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "AutoModel/model_families.hpp"
#include "server/openai_compat.hpp"

namespace fs = std::filesystem;
using openai_compat::ModelLoad;

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

static void eq(const std::string& got, const std::string& want, const std::string& what) {
    ok(got == want, what + (got == want ? "" : "  (got \"" + got + "\", want \"" + want + "\")"));
}

static void eqi(int got, int want, const std::string& what) {
    ok(got == want, what + (got == want ? "" : "  (got " + std::to_string(got) +
                                              ", want " + std::to_string(want) + ")"));
}

// ---------------------------------------------------------------------------
// finish_reason: the OpenAI schema's enum is {stop, length, tool_calls,
// content_filter, function_call}. The engine's own vocabulary is wider.
// ---------------------------------------------------------------------------
static void test_finish_reason() {
    std::printf("\n-- finish_reason --\n");
    eq(openai_compat::finish_reason(EOT_DETECTED), "stop", "EOT -> stop");
    eq(openai_compat::finish_reason(MAX_LENGTH_REACHED), "length", "max length -> length");
    eq(openai_compat::finish_reason(TOOL_DETECTED), "tool_calls", "tool -> tool_calls");

    // The two that used to escape onto the wire, and the whole reason this
    // function exists rather than a call to stop_reason_to_string().
    eq(openai_compat::finish_reason(CANCEL_DETECTED), "stop", "cancel -> stop, NOT \"cancel\"");
    eq(openai_compat::finish_reason(ERROR_DETECTED), "stop", "error -> stop, NOT \"error\"");
    eq(stop_reason_to_string(CANCEL_DETECTED), "cancel", "...the engine spelling is still \"cancel\"");

    // Exhaustive: nothing in the enum may map outside the schema.
    const stop_reason_t all[] = {EOT_DETECTED, MAX_LENGTH_REACHED, ERROR_DETECTED,
                                 CANCEL_DETECTED, TOOL_DETECTED};
    bool every_value_legal = true;
    for (stop_reason_t r : all) {
        const std::string v = openai_compat::finish_reason(r);
        if (v != "stop" && v != "length" && v != "tool_calls" &&
            v != "content_filter" && v != "function_call")
            every_value_legal = false;
    }
    ok(every_value_legal, "every stop_reason_t maps inside OpenAI's finish_reason enum");
    // An out-of-range value (a future enumerator) must not fall out either.
    eq(openai_compat::finish_reason(static_cast<stop_reason_t>(99)), "stop",
       "an unknown reason is \"stop\", not \"UNKNOWN\"");
}

// ---------------------------------------------------------------------------
// status_for: the defect was that exactly 400 was recognised and a handler's own
// 500 went out as HTTP 200 with an error body.
// ---------------------------------------------------------------------------
static void test_status_for() {
    using openai_compat::status_for;
    using nlohmann::json;
    std::printf("\n-- status_for --\n");

    eqi(status_for(json{{"choices", json::array()}}), 200, "a normal response keeps its status");
    eqi(status_for(json{{"error", "a bare string"}}), 200,
        "a non-object 'error' is not an error body (the old shape this never handled)");

    eqi(status_for(json{{"error", {{"code", 400}}}}), 400, "numeric 400");
    eqi(status_for(json{{"error", {{"code", 500}}}}), 500, "numeric 500 -- the regression under test");
    eqi(status_for(json{{"error", {{"code", 404}}}}), 404, "numeric 404");
    eqi(status_for(json{{"error", {{"code", 599}}}}), 599, "the top of the honoured range");
    eqi(status_for(json{{"error", {{"code", 399}}}}), 500,
        "a numeric code below 400 is not a status; the body is still an error");
    eqi(status_for(json{{"error", {{"code", 600}}}}), 500, "...and neither is one above 599");

    // Our own bodies carry a STRING code, so `type` is what classifies them.
    eqi(status_for(json{{"error", {{"type", "invalid_request_error"}, {"code", "model_not_found"}}}}),
        400, "string code + invalid_request_error -> 400");
    eqi(status_for(json{{"error", {{"type", "server_error"}, {"code", "model_load_failed"}}}}),
        500, "string code + server_error -> 500");
    eqi(status_for(json{{"error", {{"type", "rate_limit_error"}}}}), 429, "rate_limit_error -> 429");
    eqi(status_for(json{{"error", {{"type", "not_found_error"}}}}), 404, "not_found_error -> 404");
    eqi(status_for(json{{"error", {{"type", "authentication_error"}}}}), 401, "authentication_error -> 401");
    eqi(status_for(json{{"error", {{"type", "permission_error"}}}}), 403, "permission_error -> 403");
    eqi(status_for(json{{"error", {{"type", "something_new"}}}}), 500,
        "an unclassifiable error is 500 -- 200 is the one answer certainly wrong");
    eqi(status_for(json{{"error", {{"message", "no type, no code"}}}}), 500,
        "an error object with neither is still not a success");

    // The property that matters more than any single row.
    const nlohmann::json bodies[] = {
        json{{"error", {{"code", 500}}}},
        json{{"error", {{"type", "server_error"}}}},
        json{{"error", {{"type", "invalid_request_error"}, {"code", "model_not_found"}}}},
        json{{"error", {{"message", "bare"}}}},
        json{{"error", {{"type", "unrecognised"}, {"code", "also_unrecognised"}}}},
    };
    bool never_200 = true;
    for (const auto& b : bodies) if (status_for(b) == 200) never_200 = false;
    ok(never_200, "NO error object is ever answered 200");
}

// ---------------------------------------------------------------------------
// model_error: shape, and the promise that a client can tell whose fault it is.
// ---------------------------------------------------------------------------
static void test_model_error() {
    std::printf("\n-- model_error --\n");
    const auto unknown = openai_compat::model_error(ModelLoad::Unknown, "nope:1b");
    const auto notchat = openai_compat::model_error(ModelLoad::NotChatModel, "embed-gemma:300m");
    const auto nomodel = openai_compat::model_error(ModelLoad::NoModel, "model-faker");
    const auto failed  = openai_compat::model_error(ModelLoad::LoadFailed, "granite:3b");

    eqi(openai_compat::status_for(unknown), 400, "unknown tag -> 400");
    eqi(openai_compat::status_for(notchat), 400, "not a chat model -> 400");
    eqi(openai_compat::status_for(nomodel), 400, "no model loaded -> 400");
    eqi(openai_compat::status_for(failed), 500,
        "a load failure is OURS -> 500, so a client does not retry a 400 forever");

    for (const auto& b : {unknown, notchat, nomodel, failed}) {
        ok(b["error"].contains("message") && b["error"]["message"].is_string() &&
           !b["error"]["message"].get<std::string>().empty(), "the body carries a message");
        eq(b["error"]["param"].get<std::string>(), "model", "param names the field");
        ok(b["error"]["code"].is_string(), "code is a STRING (the OpenAI shape), not a number");
    }
    ok(unknown["error"]["message"].get<std::string>().find("nope:1b") != std::string::npos,
       "the message names the model the client asked for");
    // Review on #52: this phrasing reads as if a substitute were an option somewhere.
    bool phrase_gone = true;
    for (const auto& b : {unknown, notchat, nomodel, failed})
        if (b["error"]["message"].get<std::string>().find("substitute") != std::string::npos)
            phrase_gone = false;
    ok(phrase_gone, "no \"substitute\" phrasing, per review");
}

// ---------------------------------------------------------------------------
// is_chat_model: the predicate the server asks BEFORE it evicts. Reads the real
// model_list.json, because the whole defect was that the catalogue and the engine
// families disagree and only the catalogue was consulted.
// ---------------------------------------------------------------------------
static void test_is_chat_model(const std::string& list_path) {
    std::printf("\n-- is_chat_model (%s) --\n", list_path.c_str());
    std::string path = list_path, exe_dir = ".";   // the ctor takes non-const references
    model_list ml(path, exe_dir);

    ok(ml.is_model_supported("llama3.2:1b"), "precondition: llama3.2:1b is in the list");
    ok(is_chat_model("llama3.2:1b", ml), "llama3.2:1b IS a chat model");

    // The two that passed is_model_supported() and were served by a renamed Llama3.
    for (const char* tag : {"embed-gemma:300m", "whisper-v3:turbo"}) {
        if (!ml.is_model_supported(tag)) {
            std::printf("skip  %s is not in this model_list.json\n", tag);
            continue;
        }
        ok(!is_chat_model(tag, ml),
           std::string(tag) + " is in the model list and is NOT a chat model");
    }

    ok(!is_chat_model("definitely-not-a-model:9000b", ml), "an unknown tag is not a chat model");
    ok(!is_chat_model("", ml), "the empty tag is not a chat model");

    // Alias spellings. The defect was that is_model_supported() is an exact set
    // lookup, so "Ollama/<tag>" read as unknown, and a BARE tag never equalled the
    // resolved "<tag>:<size>" that current_model_tag holds -- so every bare-tag
    // request after the first evicted the model and reloaded it from disk.
    //
    // The bare tag is DISCOVERED rather than named, so this cannot quietly skip
    // when the catalogue changes (it already did once: the hardcoded "granite" is
    // not in this build's list).
    std::string bare;
    for (const auto& t : ml.all_tags)
        if (t.find(':') == std::string::npos && is_chat_model(t, ml)) { bare = t; break; }
    ok(!bare.empty(), "the catalogue has at least one bare chat tag to test with");
    if (!bare.empty()) {
        const std::string resolved = ml.rectify_model_tag(bare);
        std::printf("      using bare tag \"%s\" -> \"%s\"\n", bare.c_str(), resolved.c_str());
        ok(resolved.find(':') != std::string::npos, "a bare tag resolves to <type>:<size>");
        eq(ml.cut_tag("Ollama/" + resolved), resolved, "cut_tag strips a client prefix");
        eq(ml.rectify_model_tag(bare), ml.rectify_model_tag(resolved),
           "a bare tag and its resolved form normalise to ONE string");
        eq(ml.rectify_model_tag(ml.cut_tag("Ollama/" + bare)), resolved,
           "...and so does the prefixed spelling");
        ok(is_chat_model(resolved, ml), "the normalised tag is still a chat model");
        // The exact lookup that made the prefixed spelling read as unknown.
        ok(!ml.is_model_supported("Ollama/" + resolved),
           "is_model_supported() alone still rejects the prefixed spelling -- which is "
           "why ensure_model_loaded() must normalise BEFORE it asks");
        ok(ml.is_model_supported(ml.cut_tag("Ollama/" + resolved)),
           "...and accepts it once normalised");
    }
}

int main(int argc, char** argv) {
    std::string list_path = argc > 1 ? argv[1] : "model_list.json";
    if (!fs::exists(list_path)) {
        std::printf("FATAL model_list.json not found at '%s' -- pass its path as argv[1]\n",
                    list_path.c_str());
        return 2;
    }

    test_finish_reason();
    test_status_for();
    test_model_error();
    test_is_chat_model(list_path);

    std::printf("\n%s (%d checks, %d failures)\n", failures ? "FAILED" : "PASS", checks, failures);
    return failures ? 1 : 0;
}

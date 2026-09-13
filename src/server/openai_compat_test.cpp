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
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

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
    // THE ROW THAT ASSERTED THE BUG. It read 200 with the comment "a non-object
    // 'error' is not an error body", which is false: 22 catch blocks in
    // rest_handler.cpp emit exactly {"error": e.what()}. The rule and this test
    // were written together, and both covered error OBJECTS while the server was
    // emitting error BODIES.
    eqi(status_for(json{{"error", "a bare string"}}), 500,
        "a flat {\"error\": \"...\"} is 500 -- the shape the catch blocks use");
    eqi(status_for(json{{"error", "Max length reached"}}), 500,
        "...whatever the text says");

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

    // The property that matters more than any single row -- stated over error
    // BODIES, not error objects. The narrower version passed while the flat shape
    // sailed through at 200.
    const nlohmann::json bodies[] = {
        json{{"error", {{"code", 500}}}},
        json{{"error", {{"type", "server_error"}}}},
        json{{"error", {{"type", "invalid_request_error"}, {"code", "model_not_found"}}}},
        json{{"error", {{"message", "bare"}}}},
        json{{"error", {{"type", "unrecognised"}, {"code", "also_unrecognised"}}}},
        json{{"error", "a flat string from a catch block"}},
        json{{"error", nullptr}},
        json{{"error", json::array({"odd", "but still an error key"})}},
        json{{"error", 42}},
    };
    bool never_200 = true;
    for (const auto& b : bodies) if (status_for(b) == 200) never_200 = false;
    ok(never_200, "NO response carrying an 'error' key is ever answered 200");

    // And a success is still a success -- the rule must not catch ordinary bodies.
    bool success_untouched = status_for(json{{"choices", json::array()}}) == 200 &&
                             status_for(json::object()) == 200 &&
                             status_for(json{{"data", json::array()}, {"model", "m"}}) == 200;
    ok(success_untouched, "a body with no 'error' key keeps its 200");
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


// ---------------------------------------------------------------------------
// preflight: the `model` field, before any eviction.
//
// This one exists because of a REGRESSION. The handlers resolve the field as
// `request.value("model", current_model_tag)`, which makes an OMITTED "model" and
// an explicit `""` the same string -- so a sentinel branch added to answer "no
// model is loaded" started serving an explicit `""` with whatever was resident.
// It had previously been refused as unknown. Presence is an argument now, and
// these rows are the reason.
// ---------------------------------------------------------------------------
static void test_preflight() {
    using openai_compat::preflight;
    using P = openai_compat::Preflight;
    const bool SENT = true, ABSENT = false, LOADED = true, EMPTY = false;
    std::printf("\n-- preflight --\n");

    // The regression, both spellings, with a model resident.
    ok(preflight(SENT, "", "qwen3:4b", LOADED) == P::BadModelValue,
       "explicit model:\"\" is REFUSED, not served by the loaded model");
    ok(preflight(SENT, "model-faker", "qwen3:4b", LOADED) == P::BadModelValue,
       "explicit model:\"model-faker\" is REFUSED too");
    // ...and with nothing resident, so it cannot be mistaken for the NoModel case.
    ok(preflight(SENT, "", "", EMPTY) == P::BadModelValue,
       "explicit model:\"\" is refused as a VALUE, not reported as \"none loaded\"");

    // The case the sentinel branch was actually added for.
    ok(preflight(ABSENT, "model-faker", "model-faker", EMPTY) == P::NoModel,
       "no model field and none loaded -> NoModel");
    ok(preflight(ABSENT, "", "", EMPTY) == P::NoModel,
       "...same when the current tag is still the initial empty string");
    ok(preflight(ABSENT, "model-faker", "model-faker", LOADED) == P::Ok,
       "no model field but one IS loaded -> serve it");

    // A tag match is not proof of a loaded engine (a failed load leaves the
    // sentinel, and an Incompatible one used to leave a half-built engine).
    ok(preflight(SENT, "qwen3:4b", "qwen3:4b", LOADED) == P::Ok,
       "the loaded model, named -> Ok");
    ok(preflight(SENT, "qwen3:4b", "qwen3:4b", EMPTY) == P::NeedsLoad,
       "the same tag with NO engine must load, not dereference null");
    ok(preflight(SENT, "llama3.2:1b", "qwen3:4b", LOADED) == P::NeedsLoad,
       "a different model -> load it");
    ok(preflight(ABSENT, "qwen3:4b", "qwen3:4b", LOADED) == P::Ok,
       "field omitted, resolved to the loaded tag -> Ok");

    // The property, over the whole grid: nothing a client SENDS as the sentinel is
    // ever served, and nothing is ever served without an engine.
    bool never_served = true;
    for (const char* tag : {"", "model-faker"})
        for (bool loaded : {true, false})
            for (const char* cur : {"", "model-faker", "qwen3:4b"})
                if (preflight(SENT, tag, cur, loaded) != P::BadModelValue) never_served = false;
    ok(never_served, "an explicitly sent sentinel is NEVER Ok, whatever is loaded");

    bool never_ok_without_engine = true;
    for (bool sent : {true, false})
        for (const char* tag : {"", "model-faker", "qwen3:4b", "llama3.2:1b"})
            for (const char* cur : {"", "model-faker", "qwen3:4b"})
                if (preflight(sent, tag, cur, false) == P::Ok) never_ok_without_engine = false;
    ok(never_ok_without_engine, "preflight never says Ok while no engine is loaded");
}


// ---------------------------------------------------------------------------
// resolve_task: "prompt_name", its alias "task_type", and what happens when a
// request carries both. Third round of bugs in this one validation -- the
// vocabulary it quoted, the field it named, and now the alias it never read.
// ---------------------------------------------------------------------------
static void test_resolve_task() {
    using openai_compat::resolve_task;
    using S = openai_compat::TaskResolution::Status;
    using json = openai_compat::json;
    std::printf("\n-- resolve_task --\n");

    ok(resolve_task(json{{"input", "hi"}}).status == S::Absent, "no task field -> Absent");

    auto q = resolve_task(json{{"prompt_name", "search_document"}});
    ok(q.status == S::Ok && q.task == task_document, "prompt_name resolves");
    auto a = resolve_task(json{{"task_type", "search_document"}});
    ok(a.status == S::Ok && a.task == task_document, "the task_type alias resolves the same");

    // The field the CLIENT sent has to be the one named back.
    eq(resolve_task(json{{"task_type", 7}}).field, "task_type", "a bad task_type names task_type");
    eq(resolve_task(json{{"prompt_name", 7}}).field, "prompt_name", "...and prompt_name names prompt_name");
    ok(resolve_task(json{{"task_type", 7}}).status == S::NotAString, "a number is NotAString");
    ok(resolve_task(json{{"prompt_name", "nonsense"}}).status == S::Unknown, "an unlisted name is Unknown");
    eq(resolve_task(json{{"prompt_name", "nonsense"}}).value, "nonsense", "...and the value is reported back");

    // THE BUG: both present, and only the first was ever looked at.
    ok(resolve_task(json{{"prompt_name", "query"}, {"task_type", 7}}).status == S::NotAString,
       "prompt_name valid + task_type a NUMBER is still refused");
    ok(resolve_task(json{{"prompt_name", "query"}, {"task_type", "nonsense"}}).status == S::Unknown,
       "prompt_name valid + task_type UNKNOWN is still refused");
    ok(resolve_task(json{{"prompt_name", "query"}, {"task_type", "document"}}).status == S::Conflict,
       "two aliases that DISAGREE are refused, not resolved by precedence");
    auto agree = resolve_task(json{{"prompt_name", "query"}, {"task_type", "search_query"}});
    ok(agree.status == S::Ok && agree.task == task_query,
       "two spellings of the SAME task are fine");

    // Every listed name resolves, and the csv the errors quote is exactly this list.
    bool all_resolve = true;
    for (const auto& kv : openai_compat::task_names()) {
        auto r = resolve_task(json{{"prompt_name", kv.first}});
        if (r.status != S::Ok || r.task != kv.second) all_resolve = false;
    }
    ok(all_resolve, "every name in task_names() resolves to its own enum");
    for (const auto& kv : openai_compat::task_names())
        if (openai_compat::task_names_csv().find(kv.first) == std::string::npos) all_resolve = false;
    ok(all_resolve, "...and every one of them appears in the list the errors quote");
}


// ---------------------------------------------------------------------------
// task_policy: required, refused, or fine.
//
// The bug: prompt_for() returns an empty prefix when the model declares no
// prompts, so an explicit prompt_name on the BERT family came back 200 with an
// UNPREFIXED vector. src/open_npue_adapter/README.md:288 says that is an error.
// The trap next to it: an empty prompt table does NOT mean "no task concept" --
// OpenGemma_Embedding declares no names and still prefixes per task.
// ---------------------------------------------------------------------------
static void test_task_policy() {
    using openai_compat::task_policy;
    using P = openai_compat::TaskPolicy;
    const bool SUPPORTS = true, NONE = false, DECLARES = true, NODECL = false;
    const bool GIVEN = true, OMITTED = false;
    std::printf("\n-- task_policy --\n");

    // BERT / gte: no task concept at all.
    ok(task_policy(NONE, NODECL, GIVEN) == P::NotSupported,
       "a prompt on a model with no task concept is REFUSED, not ignored");
    ok(task_policy(NONE, NODECL, OMITTED) == P::Ok,
       "...and omitting it is fine");

    // nomic / EmbeddingGemma: a declared table, so the choice is the client's.
    ok(task_policy(SUPPORTS, DECLARES, OMITTED) == P::Required,
       "a model that declares prompts requires one");
    ok(task_policy(SUPPORTS, DECLARES, GIVEN) == P::Ok,
       "...and accepts one");

    // OpenGemma: prefixes hardcoded, table empty. The case a naive
    // "empty table means no tasks" rule would have broken.
    ok(task_policy(SUPPORTS, NODECL, GIVEN) == P::Ok,
       "hardcoded prefixes accept a task even with an EMPTY prompt table");
    ok(task_policy(SUPPORTS, NODECL, OMITTED) == P::Ok,
       "...and do not require one");

    // The property: a task is never accepted by something that would drop it.
    bool never_silently_dropped = true;
    for (bool declares : {true, false})
        if (task_policy(false, declares, true) != P::NotSupported) never_silently_dropped = false;
    ok(never_silently_dropped,
       "no combination lets a task reach a backend that does not honour it");
}

// ---------------------------------------------------------------------------
// Errors in a streaming handler (#64).
//
// The bug: an exception out of generate() after tokens had streamed went through
// send_response(). The 200 and the chunked headers were already on the wire, so the
// body was never written (HttpSession skips write_response() once streaming) and the
// stream was never ended -- the client waited for a [DONE] that did not come.
// ---------------------------------------------------------------------------
static void test_stream_errors() {
    using openai_compat::error_route;
    using openai_compat::stream_error_frames;
    using openai_compat::StreamState;
    using openai_compat::StreamWire;
    using R = openai_compat::ErrorRoute;
    using json = openai_compat::json;
    std::printf("\n-- stream errors --\n");

    using openai_compat::send_tracked;
    int sent = 0;
    auto send_ok = [&] { ++sent; };
    auto send_throws = [&] { ++sent; throw std::runtime_error("write failed"); };

    StreamState s;
    ok(error_route(s) == R::Body,
       "nothing sent yet: an ordinary error body, which is still correct before the first frame");
    send_tracked(s, false, send_ok);
    ok(error_route(s) == R::Frame, "after a token frame: the error goes in-stream");
    send_tracked(s, false, send_ok);
    ok(error_route(s) == R::Frame, "...after any number of them");
    send_tracked(s, true, send_ok);
    ok(error_route(s) == R::Unreportable,
       "after the final frame: nothing, because the stream has ended and the queue has moved on");
    send_tracked(s, false, send_ok);
    ok(error_route(s) == R::Unreportable, "...and a frame after the final one does not reopen it");
    eqi(sent, 4, "send_tracked calls the send exactly once per frame");

    StreamState only_final;
    send_tracked(only_final, true, send_ok);
    ok(error_route(only_final) == R::Unreportable,
       "a final frame that is also the first still ends the stream");

    // The review finding: a FINAL send that throws has not ended the stream and has
    // not advanced the NPU queue. Recording it as closed before the send made the
    // handler drop the error -- and nothing ever released the NPU.
    auto threw = [](StreamState& st, bool is_final, const std::function<void()>& send) {
        try {
            send_tracked(st, is_final, send);
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };
    StreamState final_throws;
    send_tracked(final_throws, false, send_ok);
    ok(threw(final_throws, true, send_throws), "a throwing send propagates its exception");
    ok(error_route(final_throws) == R::Frame,
       "a FINAL send that threw leaves the stream open, so the error still goes in-stream");
    StreamState first_throws;
    ok(threw(first_throws, false, send_throws), "...a throwing first send propagates too");
    ok(error_route(first_throws) == R::Frame,
       "a first send that threw counts as opened: its headers may already be on the wire");

    // SSE: OpenAI's error object, then the terminator.
    auto sse = stream_error_frames(StreamWire::Sse, "NPU fault");
    eqi(static_cast<int>(sse.size()), 2, "SSE: an error event and a terminator");
    if (sse.size() == 2) {
        const std::string& ev = sse[0];
        const bool framed = ev.rfind("data: ", 0) == 0 && ev.size() >= 8 &&
                            ev.compare(ev.size() - 2, 2, "\n\n") == 0;
        ok(framed, "SSE: the error is one `data: ...\\n\\n` event");
        if (framed) {
            json body = json::parse(ev.substr(6, ev.size() - 8), nullptr, false);
            ok(body.is_object() && body.contains("error") && body["error"].is_object(),
               "SSE: the event carries an error OBJECT, which is what OpenAI clients look for");
            if (body.is_object() && body.contains("error") && body["error"].is_object()) {
                eq(body["error"].value("message", ""), "NPU fault", "SSE: the message is the exception's");
                eq(body["error"].value("type", ""), "server_error", "SSE: type server_error");
                eqi(openai_compat::status_for(body), 500,
                    "SSE: the same body read as a response is a 500, as the Body route would say");
            }
        }
        eq(sse[1], "data: [DONE]\n\n", "SSE: the stream ends with [DONE], last");
    }

    // A newline pair in the message must not end the event early.
    auto sse_nl = stream_error_frames(StreamWire::Sse, "line one\n\nline two");
    ok(!sse_nl.empty() && sse_nl[0].find("\n\n") == sse_nl[0].size() - 2,
       "SSE: a blank line in the message does not split the event");

    // NDJSON: Ollama's {"error": "<text>"}, one line, no SSE prefix.
    auto nd = stream_error_frames(StreamWire::Ndjson, "NPU fault\nsecond line");
    eqi(static_cast<int>(nd.size()), 1, "NDJSON: one line, which the caller sends as final");
    if (nd.size() == 1) {
        const std::string& line = nd[0];
        ok(!line.empty() && line.back() == '\n' && line.find('\n') == line.size() - 1,
           "NDJSON: exactly one newline, at the end, even when the message has one");
        ok(line.rfind("data: ", 0) != 0, "NDJSON: no SSE prefix");
        json body = json::parse(line, nullptr, false);
        ok(body.is_object() && body.contains("error") && body["error"].is_string(),
           "NDJSON: {\"error\": \"<text>\"}, the shape Ollama's client checks each line for");
        if (body.is_object() && body.contains("error") && body["error"].is_string())
            eq(body["error"].get<std::string>(), "NPU fault\nsecond line", "NDJSON: the message survives");
    }

    // what() can quote invalid UTF-8 (json::parse_error does). The error path must
    // not throw on the thing it is reporting.
    const std::string bad = std::string("parse error, last read: '") + '\xE5' + "'";
    for (StreamWire w : {StreamWire::Sse, StreamWire::Ndjson}) {
        const char* name = w == StreamWire::Sse ? "SSE" : "NDJSON";
        bool threw = false;
        std::vector<std::string> frames;
        try {
            frames = stream_error_frames(w, bad);
        } catch (...) {
            threw = true;
        }
        ok(!threw, std::string(name) + ": invalid UTF-8 in the message does not throw");
        ok(!frames.empty() && frames[0].find("\xEF\xBF\xBD") != std::string::npos,
           std::string(name) + ": ...it is replaced with U+FFFD");
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
    test_preflight();
    test_resolve_task();
    test_task_policy();
    test_stream_errors();
    test_is_chat_model(list_path);

    std::printf("\n%s (%d checks, %d failures)\n", failures ? "FAILED" : "PASS", checks, failures);
    return failures ? 1 : 0;
}

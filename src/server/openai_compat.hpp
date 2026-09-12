/// \file openai_compat.hpp
/// \brief The OpenAI wire vocabulary: finish_reason, error bodies, HTTP status.
///
/// These three are here rather than inside a handler for one reason: EVERY defect
/// #52 fixes is a case where one code path answered correctly and another, saying
/// the same thing, did not. The non-streaming responder mapped finish_reason and
/// the streaming one did not; one call site built a 400 body and the next three
/// had their own copies; the status mapper recognised 400 and let 500 through as
/// 200. A shared definition is what makes "the server always says this" a fact
/// rather than an intention, and these are pure functions so a test can hold them
/// to it without a server, a device or a model.
#pragma once

#include <string>

#include <utility>
#include <vector>

#include "AutoEmbeddingModel/auto_embedding_model.hpp"   // embedding_task_type_t
#include "AutoModel/stop_reason.hpp"
#include "nlohmann/json.hpp"

namespace openai_compat {

/// The server's own json type (server.hpp, rest_handler.hpp). Building a plain
/// nlohmann::json here and handing it to send_response() compiles and works, but
/// it converts -- and an ordered_json built from an unordered one loses the key
/// order the rest of the wire format keeps.
using json = nlohmann::ordered_json;

/// OpenAI's finish_reason vocabulary is {stop, length, tool_calls,
/// content_filter, function_call}. `stop_reason_to_string()` also yields
/// "cancel", "error" and "UNKNOWN", which are not in it, so a response must map
/// rather than print. Anything without an OpenAI equivalent stays "stop", which
/// is what the handlers emitted for every outcome before #52.
inline const char* finish_reason(stop_reason_t reason) {
    switch (reason) {
        case MAX_LENGTH_REACHED: return "length";
        case TOOL_DETECTED:      return "tool_calls";
        default:                 return "stop";
    }
}

/// Why ensure_model_loaded() did not leave a model serving. The distinction is
/// not cosmetic: Unknown / NotChatModel / NoModel are the CLIENT's mistake and
/// answer 400, while LoadFailed is ours and answers 500. A client that retries a
/// 400 forever is being told the wrong thing about whose problem it is.
enum class ModelLoad { Ok, Unknown, NotChatModel, NoModel, LoadFailed };

/// The error body for a non-Ok outcome.
///
/// Deliberately does NOT say "no substitute was used": per review on #52 that
/// reads as if answering with a different model were an option somewhere.
inline json model_error(ModelLoad why, const std::string& model) {
    switch (why) {
        case ModelLoad::Unknown:
            return json{{"error", {
                {"message", "model '" + model + "' is not in this build's model list"},
                {"type", "invalid_request_error"}, {"param", "model"}, {"code", "model_not_found"}}}};
        case ModelLoad::NotChatModel:
            return json{{"error", {
                {"message", "model '" + model + "' is not a chat model; this endpoint serves "
                            "text generation only"},
                {"type", "invalid_request_error"}, {"param", "model"}, {"code", "model_not_found"}}}};
        case ModelLoad::NoModel:
            return json{{"error", {
                {"message", "no chat model is loaded: this server was started without one. "
                            "Name a model in the request's 'model' field, or start oflm serve "
                            "with a model tag."},
                {"type", "invalid_request_error"}, {"param", "model"}, {"code", "model_not_found"}}}};
        case ModelLoad::LoadFailed:
        default:
            return json{{"error", {
                {"message", "model '" + model + "' is known to this build but could not be "
                            "loaded; the server log says why"},
                {"type", "server_error"}, {"param", "model"}, {"code", "model_load_failed"}}}};
    }
}

/// The HTTP status a response body deserves, or `fallback` when it is not an error.
///
/// The rules, in order:
///   1. a top-level `error` that is NOT an object is `{"error": "<text>"}` -- the
///      shape 22 catch blocks in rest_handler.cpp still use. 500;
///   2. a numeric `code` in 400-599 is a status and is taken as given;
///   3. otherwise the `type` classifies it -- our own errors carry a STRING code
///      ("model_not_found"), so the type is the only thing that can;
///   4. an error object this server built but cannot classify is 500, because 200
///      is the one answer that is certainly wrong.
///
/// Rule 1 was missing, and the test asserted its absence. The first version of
/// this recognised a numeric 400 and nothing else; the second added the type but
/// still returned the 200 fallback for a flat string, so the promised invariant
/// covered error OBJECTS while the handlers were emitting error BODIES.
inline int status_for(const json& response_data, int fallback = 200) {
    if (!response_data.contains("error")) return fallback;
    const json& err = response_data["error"];
    if (!err.is_object()) return 500;   // {"error": "<what() text>"}
    if (err.contains("code") && err["code"].is_number_integer()) {
        const int c = err["code"].get<int>();
        if (c >= 400 && c <= 599) return c;
    }
    if (err.contains("type") && err["type"].is_string()) {
        const std::string t = err["type"].get<std::string>();
        if (t == "invalid_request_error") return 400;
        if (t == "authentication_error")  return 401;
        if (t == "permission_error")      return 403;
        if (t == "not_found_error")       return 404;
        if (t == "rate_limit_error")      return 429;
        return 500;
    }
    return 500;
}

/// What to do about a request's `model` field, BEFORE any eviction or loading.
///
/// A pure function because the case that made it one was a regression: the handlers
/// resolve `model` as `request.value("model", current_model_tag)`, which makes an
/// OMITTED field and an explicit `""` the same string. Treating both as "no model
/// named" meant an explicit `""` or `"model-faker"` -- previously refused -- was
/// served by whatever happened to be loaded. Presence has to be carried in, and the
/// rule is small enough to be worth stating once and testing.
enum class Preflight {
    Ok,             ///< the loaded engine already serves this request
    NoModel,        ///< nothing is loaded and the request named nothing
    BadModelValue,  ///< the client SENT a sentinel or an empty string
    NeedsLoad       ///< resolve, evict, load
};

/// \param field_present  the request actually carried a "model" key
/// \param requested      that value (or the current tag when absent), normalised
/// \param current        the tag the loaded engine was loaded for
/// \param engine_loaded  a model is actually resident
inline Preflight preflight(bool field_present, const std::string& requested,
                           const std::string& current, bool engine_loaded) {
    const bool sentinel = requested.empty() || requested == "model-faker";
    if (sentinel) {
        // Sent deliberately, it is not a model name and must not resolve to one.
        if (field_present) return Preflight::BadModelValue;
        // Omitted, and the server was started without a model.
        return engine_loaded ? Preflight::Ok : Preflight::NoModel;
    }
    // A tag MATCH is not proof of a loaded model: current_model_tag is also
    // "model-faker" after a failed load, and it starts empty.
    if (requested == current && engine_loaded) return Preflight::Ok;
    return Preflight::NeedsLoad;
}

/// The REST names for an embedding task, and the enum each maps to.
///
/// NOT the container's vocabulary: a container declares names like "Retrieval",
/// and these map onto those in NpueEmbedding::prompt_for(). Quoting the wrong one
/// back to a client is how the "requires a task prompt" error came to name values
/// the validator then rejected.
inline const std::vector<std::pair<const char*, embedding_task_type_t>>& task_names() {
    static const std::vector<std::pair<const char*, embedding_task_type_t>> kTasks = {
        {"query", task_query}, {"search_query", task_query},
        {"Retrieval-query", task_query},
        {"document", task_document}, {"search_document", task_document},
        {"Retrieval-document", task_document},
        {"clustering", task_clustering}, {"Clustering", task_clustering},
        {"classification", task_classification},
        {"Classification", task_classification},
        {"MultilabelClassification", task_multilabel_classification},
        {"STS", task_sentence_similarity},
        {"sentence_similarity", task_sentence_similarity},
        {"Summarization", task_summarization},
        {"summarization", task_summarization},
        {"BitextMining", task_bitextmining},
        {"bitextmining", task_bitextmining},
        {"code_retrieval", task_code_retrieval},
        {"search_result", task_search_result},
    };
    return kTasks;
}

/// Those names as one comma-separated string, for an error message.
inline std::string task_names_csv() {
    std::string s;
    for (const auto& kv : task_names()) s += (s.empty() ? "" : ", ") + std::string(kv.first);
    return s;
}

/// The outcome of reading a request's task prompt.
struct TaskResolution {
    enum class Status { Ok, Absent, NotAString, Unknown, Conflict };
    Status status = Status::Absent;
    embedding_task_type_t task = task_query;  ///< valid only when Ok
    std::string field;                        ///< which key this is about
    std::string value;                        ///< the offending value, when Unknown
};

/// Read "prompt_name", or its accepted alias "task_type".
///
/// BOTH are checked. The first version took prompt_name when present and never
/// looked at task_type, so `{prompt_name:"query", task_type:7}` was accepted with
/// an invalid value sitting in the request. Sending both is fine when they agree;
/// disagreeing is a client bug and is refused rather than resolved by precedence.
inline TaskResolution resolve_task(const json& request) {
    TaskResolution out;
    bool have = false;
    for (const char* field : {"prompt_name", "task_type"}) {
        if (!request.contains(field)) continue;
        const json& f = request.at(field);
        if (!f.is_string()) return {TaskResolution::Status::NotAString, task_query, field, ""};
        const std::string want = f.get<std::string>();
        const auto& tbl = task_names();
        auto hit = tbl.end();
        for (auto it = tbl.begin(); it != tbl.end(); ++it)
            if (want == it->first) { hit = it; break; }
        if (hit == tbl.end()) return {TaskResolution::Status::Unknown, task_query, field, want};
        if (have && hit->second != out.task)
            return {TaskResolution::Status::Conflict, task_query, field, want};
        out.task = hit->second;
        out.field = field;
        have = true;
    }
    out.status = have ? TaskResolution::Status::Ok : TaskResolution::Status::Absent;
    return out;
}

/// Whether a request's task prompt is required, refused, or fine.
///
/// Three states because an empty prompt table means two different things: a model
/// with no task concept (the BERT family), and one whose prefixes are hardcoded
/// rather than declared (OpenGemma). Inferring from the table alone silently
/// dropped an explicit prompt on the first and would have broken the second.
enum class TaskPolicy {
    Ok,            ///< what the request carries is acceptable
    Required,      ///< the model declares prompts and the request named none
    NotSupported   ///< the model has no task concept and the request named one
};

inline TaskPolicy task_policy(bool supports_prompts, bool declares_names, bool task_given) {
    if (task_given && !supports_prompts) return TaskPolicy::NotSupported;
    if (!task_given && declares_names)   return TaskPolicy::Required;
    return TaskPolicy::Ok;
}

/// What a streaming response has already handed to the transport (#64).
///
/// The first frame sends the 200 and the chunked headers, so from then on
/// send_response() cannot reach the client: HttpSession skips write_response() for a
/// streaming session, so an error body went nowhere and the connection was never
/// terminated -- or, for a queued request, a second HTTP response was written into
/// the chunked body. The final frame ends the stream, and it is also what advances
/// the NPU queue, so nothing may follow it either.
struct StreamState {
    bool opened = false;  ///< a frame has been, or was being, handed to the transport
    bool closed = false;  ///< a final frame's send has RETURNED
};

/// Send one frame through `send()` and record it in `s`.
///
/// `opened` is set before the send, because once a send has started the headers may be
/// on the wire. `closed` is set only after it returns: a final send that throws has
/// neither ended the stream nor advanced the NPU queue, so the error must still go
/// in-stream. Marking it closed first dropped that error as unreportable, and the queue
/// never moved again. Exceptions propagate unchanged.
template <class Send>
inline void send_tracked(StreamState& s, bool is_final, Send&& send) {
    s.opened = true;
    send();
    if (is_final) s.closed = true;
}

/// Where an error raised by a streaming handler has to go.
enum class ErrorRoute {
    Body,         ///< nothing is on the wire yet: an ordinary error body, with its status
    Frame,        ///< the stream is open: report in-stream, then end the stream
    Unreportable  ///< the stream has ended: there is no transport left to report on
};

inline ErrorRoute error_route(const StreamState& s) {
    if (!s.opened) return ErrorRoute::Body;
    return s.closed ? ErrorRoute::Unreportable : ErrorRoute::Frame;
}

/// The two streaming formats this server speaks.
enum class StreamWire {
    Sse,     ///< /v1/*: `data: <json>\n\n` events, terminated by `data: [DONE]`
    Ndjson   ///< /api/*: one JSON object per line, no terminator event
};

/// The frames that report `message` inside an open stream, in order. The caller sends
/// the last one as final.
///
/// SSE carries OpenAI's error object, which is what its clients look for in a stream
/// (openai-python raises APIError on a data event with an "error" key), followed by
/// [DONE] so a client waiting for the terminator gets one. NDJSON carries Ollama's
/// `{"error": "<text>"}`, the shape its client checks for on every line.
///
/// Serialised with error_handler_t::replace: `message` is an exception's what(), and
/// a json::parse_error quotes the bytes it choked on. A strict dump() would throw on
/// invalid UTF-8 from inside the catch block that is reporting it.
inline std::vector<std::string> stream_error_frames(StreamWire wire, const std::string& message) {
    if (wire == StreamWire::Sse) {
        const json body = {{"error", {
            {"message", message},
            {"type", "server_error"},
            {"code", 500}
        }}};
        return {"data: " + body.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n",
                "data: [DONE]\n\n"};
    }
    const json body = {{"error", message}};
    return {body.dump(-1, ' ', false, json::error_handler_t::replace) + "\n"};
}

}  // namespace openai_compat

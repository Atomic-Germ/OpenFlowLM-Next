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
///   1. a numeric `code` in 400-599 is a status and is taken as given;
///   2. otherwise the `type` classifies it -- our own errors carry a STRING code
///      ("model_not_found"), so the type is the only thing that can;
///   3. an error object this server built but cannot classify is 500, because 200
///      is the one answer that is certainly wrong.
///
/// The first version of this recognised a numeric 400 and nothing else, and
/// because that test sat inside the `if`, a handler's own 500 fell through BOTH
/// branches and went out as HTTP 200 with an error body.
inline int status_for(const json& response_data, int fallback = 200) {
    if (!response_data.contains("error") || !response_data["error"].is_object()) return fallback;
    const json& err = response_data["error"];
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

}  // namespace openai_compat

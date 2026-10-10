/// \file openai_compat.hpp
/// \brief The OpenAI wire vocabulary: finish_reason, error bodies, HTTP status, and the
///        Images API's request rules.
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

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>

#include <utility>
#include <vector>

#include "AutoEmbeddingModel/auto_embedding_model.hpp"   // embedding_task_type_t
#include "AutoModel/request_error.hpp"
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

/// The width of one embedding inside a concatenated batch result.
///
/// embed_batch() returns every vector end to end, so the caller has to slice.
/// The failure that matters is not a crash: a wrong width, or a wrong order,
/// hands a caller a correctly shaped, correctly normed, deterministic vector
/// for somebody else's text, and nothing downstream can detect it.
///
/// `expected_dim` is the backend's own vector width
/// (AutoEmbeddingModel::embedding_dim()). When it is known the result must be
/// exactly `n_inputs * expected_dim` floats, which also catches a backend that
/// returns a whole number of vectors but the wrong number of them -- half a
/// batch divides evenly. When it is 0 the backend does not report a width, and
/// the only check left is that the result divides; that cannot catch the
/// half-a-batch case.
inline size_t embedding_batch_dim(size_t flat_size, size_t n_inputs,
                                  size_t expected_dim = 0) {
    if (n_inputs == 0)
        throw std::runtime_error(
            "embedding_batch_dim: asked for the vector width of zero inputs");
    if (expected_dim != 0) {
        if (flat_size != n_inputs * expected_dim)
            throw std::runtime_error(
                "embedding backend returned " + std::to_string(flat_size) +
                " floats for " + std::to_string(n_inputs) + " inputs, but its"
                " vectors are " + std::to_string(expected_dim) + " wide, so it"
                " should have returned " + std::to_string(n_inputs * expected_dim) +
                ". Refusing to slice: a mis-split returns correctly shaped,"
                " correctly normed vectors for the wrong inputs.");
        return expected_dim;
    }
    if (flat_size == 0 || flat_size % n_inputs != 0)
        throw std::runtime_error(
            "embedding backend returned " + std::to_string(flat_size) +
            " floats for " + std::to_string(n_inputs) +
            " inputs, which does not divide evenly. Refusing to guess the"
            " vector width: a mis-split returns correctly shaped, correctly"
            " normed vectors for the wrong inputs.");
    return flat_size / n_inputs;
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

/// The JSON type a required request field must have.
enum class FieldType { String, Array };

/// Check a required field BEFORE a handler reads it.
///
/// `request["x"]` on a const json without the key is undefined behaviour, and on
/// this build it segfaults: `POST {}` took the server down on /api/show,
/// /api/generate and /v1/completions (#70). Returns an empty json when the field
/// is present with the right type, otherwise the 400 body to send.
inline json require_field(const json& request, const char* field, FieldType type) {
    const char* want = type == FieldType::String ? "a string" : "an array";
    if (!request.is_object())
        return json{{"error", {
            {"message", "the request body must be a JSON object."},
            {"type", "invalid_request_error"}, {"param", ""}, {"code", "invalid_value"}}}};
    if (!request.contains(field))
        return json{{"error", {
            {"message", std::string(field) + " is required and must be " + want + "."},
            {"type", "invalid_request_error"}, {"param", field},
            {"code", "missing_required_parameter"}}}};
    const json& v = request[field];
    const bool right = type == FieldType::String ? v.is_string() : v.is_array();
    if (!right)
        return json{{"error", {
            {"message", std::string(field) + " must be " + want + "."},
            {"type", "invalid_request_error"}, {"param", field}, {"code", "invalid_value"}}}};
    return json();
}

/// The body for an exception a handler caught (#135). e.what() never goes to the
/// client: nlohmann quotes the request's own bytes into a parse error ("last read:
/// ...") and names its type-system internals, and an engine's text can name paths.
/// The caller logs e.what(); the client gets one of two fixed bodies.
///
/// Only a request_error is the client's fault (400): it is thrown where the fault
/// is known, such as a chat template refusing the conversation. Neither the type
/// nor the catch site can say more -- a json::exception comes as readily from the
/// model list as from a request field, and an insert() can fail on the NPU -- so
/// everything else is the server's (500).
inline json exception_body(const std::exception& e) {
    if (dynamic_cast<const request_error*>(&e) != nullptr)
        return json{{"error", {{"message", "Invalid request"}, {"type", "invalid_request_error"},
                               {"code", "invalid_value"}}}};
    return json{{"error", {{"message", "Internal error"}, {"type", "server_error"}}}};
}

// ---------------------------------------------------------------------------------
// The Images API (/v1/images/generations, /v1/images/edits). specs/server-api:
// SERVER-IMAGES-PARAMS, SERVER-IMAGES-SIZE.
// ---------------------------------------------------------------------------------

/// A 400 naming the field it is about.
inline json invalid_param(const std::string& param, const std::string& message,
                          const char* code = "invalid_value") {
    return json{{"error", {
        {"message", message}, {"type", "invalid_request_error"}, {"param", param}, {"code", code}}}};
}

/// The error for a tag that is in the model list but makes no images.
inline json not_image_model_error(const std::string& model) {
    return invalid_param("model", "model '" + model + "' is not an image model; this endpoint "
                                  "serves image generation only", "model_not_found");
}

/// A request's image controls, validated. Everything but the model: the handler
/// resolves that first, because the sizes it may ask for are the model's.
struct ImagesRequest {
    std::string prompt;
    int n = 1;
    int size = 0;                       ///< the square's side in pixels
    std::string output_format = "png";  ///< "png" | "jpeg"
    int jpeg_quality = 90;              ///< output_compression, for jpeg
    bool seeded = false;                ///< false: the handler picks a random seed
    uint64_t seed = 0;
    int steps = 0;                      ///< 0: the model's own count
    std::vector<std::string> ignored;   ///< controls accepted and ignored, for the log line
};

constexpr int kImagesMaxN = 10;
constexpr int kImagesMaxSteps = 50;

/// The sampler names that mean flow-match Euler, the only sampler klein has.
inline const std::vector<std::string>& euler_sampler_names() {
    static const std::vector<std::string> kNames = {
        "euler", "Euler", "Euler a", "flowmatch_euler", "FlowMatchEulerDiscreteScheduler"};
    return kNames;
}

/// "512x512, 1024x1024" for a list of square sides.
inline std::string image_sizes_csv(const std::vector<int>& sizes) {
    std::string s;
    for (int r : sizes) s += (s.empty() ? "" : ", ") + std::to_string(r) + "x" + std::to_string(r);
    return s;
}

/// Parse and check a request's image controls into `out`. Returns an empty json when
/// the request is acceptable, else the 400 body to send.
///
/// \param sizes      the model's resolutions (square sides)
/// \param auto_size  what `"auto"` and an omitted `size` mean
///
/// The rules (SERVER-IMAGES-PARAMS):
///   - `null` is the same as leaving a field out: clients send it for "unset";
///   - each alias pair is ONE field. Both spellings in one request must agree, or it is
///     a 400 naming the second -- neither silently wins;
///   - `cfg_scale`/`guidance_scale` and `negative_prompt` are type-checked and then
///     ignored (klein is guidance-distilled; A1111 clients always send them);
///   - `sampler`/`sampler_name` accepts the Euler family and refuses anything else;
///   - `seed: -1` is A1111's "random";
///   - fields this server does not know are ignored.
inline json images_request(const json& req, const std::vector<int>& sizes, int auto_size,
                           ImagesRequest& out) {
    out = ImagesRequest{};
    if (json err = require_field(req, "prompt", FieldType::String); !err.is_null()) return err;
    out.prompt = req.at("prompt").get<std::string>();

    auto has = [&](const char* f) { return req.contains(f) && !req.at(f).is_null(); };
    // One field under two names: the first present spelling, and a 400 if both are
    // present and differ. *field is the name the value came from.
    auto alias = [&](const char* a, const char* b, const json** v, std::string* field) -> json {
        *v = nullptr;
        if (has(a)) { *v = &req.at(a); *field = a; }
        if (has(b)) {
            if (*v && **v != req.at(b))
                return invalid_param(b, std::string(a) + " and " + b + " are the same field and "
                                        "the request gives them different values");
            if (!*v) { *v = &req.at(b); *field = b; }
        }
        return json();
    };
    // An integer in [lo, hi]; a float, even an integral one, is refused.
    auto int_in = [&](const json& v, const std::string& f, long long lo, long long hi, long long* got) -> json {
        if (!v.is_number_integer())
            return invalid_param(f, f + " must be an integer.");
        long long x = v.is_number_unsigned() && v.get<uint64_t>() > static_cast<uint64_t>(hi)
                          ? hi + 1 : v.get<long long>();
        if (x < lo || x > hi)
            return invalid_param(f, f + " must be between " + std::to_string(lo) + " and " +
                                    std::to_string(hi) + ".");
        *got = x;
        return json();
    };

    long long x = 0;
    if (has("n")) {
        if (json e = int_in(req.at("n"), "n", 1, kImagesMaxN, &x); !e.is_null()) return e;
        out.n = static_cast<int>(x);
    }

    out.size = auto_size;
    if (has("size")) {
        const json& v = req.at("size");
        if (!v.is_string()) return invalid_param("size", "size must be a string: \"WxH\" or \"auto\".");
        const std::string s = v.get<std::string>();
        if (s != "auto") {
            size_t xpos = s.find('x');
            auto digits = [](const std::string& t) {
                return !t.empty() && t.size() <= 5 &&
                       std::all_of(t.begin(), t.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
            };
            if (xpos == std::string::npos || !digits(s.substr(0, xpos)) || !digits(s.substr(xpos + 1)))
                return invalid_param("size", "size must be \"WxH\" (e.g. \"1024x1024\") or \"auto\", not \"" +
                                             s + "\".");
            int w = std::stoi(s.substr(0, xpos)), h = std::stoi(s.substr(xpos + 1));
            if (w != h || std::find(sizes.begin(), sizes.end(), w) == sizes.end())
                return invalid_param("size", "size " + s + " is not supported by this model; supported: " +
                                             image_sizes_csv(sizes) + " (or \"auto\").");
            out.size = w;
        }
    }

    if (has("output_format")) {
        const json& v = req.at("output_format");
        const std::string f = v.is_string() ? v.get<std::string>() : std::string();
        if (f == "webp")
            return invalid_param("output_format", "output_format webp is not implemented in this "
                                                  "server; use png or jpeg.", "not_implemented");
        if (f != "png" && f != "jpeg")
            return invalid_param("output_format", "output_format must be png or jpeg.");
        out.output_format = f;
    }
    if (has("output_compression")) {
        if (json e = int_in(req.at("output_compression"), "output_compression", 0, 100, &x); !e.is_null()) return e;
        out.jpeg_quality = std::max(1, static_cast<int>(x));
    }
    if (has("response_format")) {
        const json& v = req.at("response_format");
        const std::string f = v.is_string() ? v.get<std::string>() : std::string();
        if (f == "url")
            return invalid_param("response_format", "response_format url is not offered: this server "
                                                    "returns b64_json only.", "not_implemented");
        if (f != "b64_json") return invalid_param("response_format", "response_format must be b64_json.");
    }
    if (has("stream")) {
        const json& v = req.at("stream");
        if (!v.is_boolean()) return invalid_param("stream", "stream must be a boolean.");
        if (v.get<bool>())
            return invalid_param("stream", "streaming (partial images) is not offered: each preview "
                                           "costs a full VAE decode on the NPU.", "not_implemented");
    }
    if (has("partial_images")) {
        if (json e = int_in(req.at("partial_images"), "partial_images", 0, 3, &x); !e.is_null()) return e;
        if (x > 0)
            return invalid_param("partial_images", "partial_images is not offered: each preview costs "
                                                   "a full VAE decode on the NPU.", "not_implemented");
    }

    if (has("seed")) {
        const json& v = req.at("seed");
        if (!v.is_number_integer()) return invalid_param("seed", "seed must be an integer.");
        if (v.is_number_unsigned()) {
            out.seeded = true;
            out.seed = v.get<uint64_t>();
        } else if (v.get<long long>() == -1) {
            out.seeded = false;                      // A1111's "random"
        } else if (v.get<long long>() < 0) {
            return invalid_param("seed", "seed must be a non-negative integer, or -1 for a random one.");
        } else {
            out.seeded = true;
            out.seed = static_cast<uint64_t>(v.get<long long>());
        }
    }

    const json* v = nullptr;
    std::string field;
    if (json e = alias("steps", "num_inference_steps", &v, &field); !e.is_null()) return e;
    if (v) {
        if (json e = int_in(*v, field, 1, kImagesMaxSteps, &x); !e.is_null()) return e;
        out.steps = static_cast<int>(x);
    }

    if (json e = alias("cfg_scale", "guidance_scale", &v, &field); !e.is_null()) return e;
    if (v) {
        if (!v->is_number()) return invalid_param(field, field + " must be a number.");
        out.ignored.push_back(field);
    }
    if (has("negative_prompt")) {
        if (!req.at("negative_prompt").is_string())
            return invalid_param("negative_prompt", "negative_prompt must be a string.");
        out.ignored.push_back("negative_prompt");
    }

    if (json e = alias("sampler", "sampler_name", &v, &field); !e.is_null()) return e;
    if (v) {
        if (!v->is_string()) return invalid_param(field, field + " must be a string.");
        const auto& names = euler_sampler_names();
        if (std::find(names.begin(), names.end(), v->get<std::string>()) == names.end()) {
            std::string list;
            for (const auto& s : names) list += (list.empty() ? "" : ", ") + s;
            return invalid_param(field, field + " '" + v->get<std::string>() + "' is not available: this "
                                        "model's only sampler is flow-match Euler (" + list + ").");
        }
    }
    return json();
}

/// A multipart form's text fields as the JSON images_request() reads. Form values are
/// all strings; the numeric and boolean controls are converted when they parse as
/// such, and left as strings otherwise, so a bad one is refused as the wrong type.
inline json images_form_json(const std::vector<std::pair<std::string, std::string>>& fields) {
    static const char* kInts[] = {"n", "seed", "steps", "num_inference_steps", "output_compression",
                                  "partial_images"};
    static const char* kNumbers[] = {"cfg_scale", "guidance_scale"};
    json out = json::object();
    for (const auto& [k, s] : fields) {
        json v = s;
        auto is = [&](const char* const* names, size_t count) {
            return std::find_if(names, names + count, [&](const char* n) { return k == n; }) != names + count;
        };
        if (is(kInts, std::size(kInts)) || is(kNumbers, std::size(kNumbers))) {
            json p = json::parse(s, nullptr, false);
            if (!p.is_discarded() && p.is_number() && (!is(kInts, std::size(kInts)) || p.is_number_integer()))
                v = p;
        } else if (k == "stream" && (s == "true" || s == "false")) {
            v = s == "true";
        }
        out[k] = v;
    }
    return out;
}

}  // namespace openai_compat

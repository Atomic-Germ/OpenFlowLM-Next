/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file rest_handler.cpp
 * \brief RestHandler class and related declarations
 * \author OpenFlowLM Team
 * \date 2025-08-05
 *  \version 0.9.24
 */
#include "rest_handler.hpp"
#include "wstream_buf.hpp"
#include "streaming_ostream.hpp"
#include "streaming_ostream_openai.hpp"
#include "image/image_reader.hpp"
#include <sstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <locale>
#include <random>
#include "server.hpp"

///@brief Normalize messages by merging consecutive user messages (like Ollama does)
///@param messages the original messages
///@return normalized messages with consecutive user messages merged
static json normalize_messages(json messages) {
    if (messages.empty()) return messages;

    json normalized = nlohmann::ordered_json::array();

    for (size_t i = 0; i < messages.size(); i++) {
        auto current_msg = messages[i];
        std::string role = current_msg.value("role", "");

        if (role == "user") {
            json merged_content_array = json::array();

            // Merge all consecutive user messages into array format
            while (i < messages.size() && messages[i].value("role", "") == "user") {
                if (messages[i].contains("content")) {
                    if (messages[i]["content"].is_array()) {
                        for (auto& item : messages[i]["content"]) {
                            merged_content_array.push_back(item);
                        }
                    }
                    else if (messages[i]["content"].is_string()) {
                        std::string text = messages[i]["content"].get<std::string>();
                        if (!text.empty()) {
                            nlohmann::ordered_json text_item;
                            text_item["type"] = "text";
                            text_item["text"] = text;
                            merged_content_array.push_back(text_item);
                        }
                    }
                }
                if (i + 1 < messages.size() && messages[i + 1].value("role", "") == "user") i++;
                else break;
            }

            current_msg["content"] = merged_content_array;
        }
        else if (role == "system") {
            json merged_content_array = json::array();

            // Merge all consecutive system messages into array format
            while (i < messages.size() && messages[i].value("role", "") == "system") {
                if (messages[i].contains("content")) {
                    if (messages[i]["content"].is_array()) {
                        for (auto& item : messages[i]["content"]) {
                            merged_content_array.push_back(item);
                        }
                    }
                    else if (messages[i]["content"].is_string()) {
                        std::string text = messages[i]["content"].get<std::string>();
                        if (!text.empty()) {
                            json text_item;
                            text_item["type"] = "text";
                            text_item["text"] = text;
                            merged_content_array.push_back(text_item);
                        }
                    }
                }
                if (i + 1 < messages.size() && messages[i + 1].value("role", "") == "system") i++;
                else break;
            }

            current_msg["content"] = merged_content_array;
        }
        else if (role == "assistant") {
            // Strip prior assistant "thinking" / reasoning fields so they are not
            // fed back into the model on subsequent turns.
            if (current_msg.contains("thinking")) {
                current_msg.erase("thinking");
            }
            if (current_msg.contains("reasoning")) {
                current_msg.erase("reasoning");
            }
            if (current_msg.contains("reasoning_content")) {
                current_msg.erase("reasoning_content");
            }
            // Also strip <think>...</think> blocks from string content if present.
            // if (current_msg.contains("content") && current_msg["content"].is_string()) {
            //     std::string text = current_msg["content"].get<std::string>();
            //     size_t start = text.find("<think>");
            //     while (start != std::string::npos) {
            //         size_t end = text.find("</think>", start);
            //         if (end == std::string::npos) {
            //             text.erase(start);
            //             break;
            //         }
            //         text.erase(start, (end + std::string("</think>").size()) - start);
            //         start = text.find("<think>");
            //     }
            //     // Trim leading whitespace/newlines left after removal.
            //     size_t first = text.find_first_not_of(" \t\r\n");
            //     if (first == std::string::npos) text.clear();
            //     else if (first > 0) text.erase(0, first);
            //     current_msg["content"] = text;
            // }
        }
        normalized.push_back(current_msg);
    }

    return normalized;
}

static json normalize_template(json messages) {
    json template_message = json::array();

    for (auto& message : messages) {
        json new_message = message;
        std::string merged_text;
        nlohmann::ordered_json::array_t merged_images;
        nlohmann::ordered_json::array_t merged_audio;

        if (message["content"].is_string()) {
            // Simple format: just text
            merged_text = message["content"].get<std::string>();
        }
        else if (message["content"].is_array()) {
            // Structured format: extract text and image URLs
            for (auto& contentItem : message["content"]) {
                if (contentItem.contains("type") && contentItem["type"] == "text") {
                    merged_text += contentItem["text"].get<std::string>();
                }
                else if (contentItem.contains("type") && contentItem["type"] == "image_url") {
                    std::string image_url = contentItem["image_url"]["url"].get<std::string>();
                    const std::vector<std::string> prefixes = {
                        "data:image/png;base64,",
                        "data:image/jpeg;base64,",
                        "data:image/jpg;base64,"
                    };
                    for (const auto& prefix : prefixes) {
                        if (image_url.substr(0, prefix.length()) == prefix) {
                            image_url = image_url.substr(prefix.length());
                            break;
                        }
                    }
                    if (image_url.substr(0, 5) == "data:") {
                        header_print("Warning", "Unsupported image format, skipping this image.");
                        continue;
                    }
                    if (image_url.empty()) {
                        header_print("Warning", "Empty image, skipping this image.");
                        continue;
                    }
                    merged_images.push_back(image_url);
                }
                else if (contentItem.contains("type") && contentItem["type"] == "input_audio") {
                    std::string audio_base64 = contentItem["input_audio"]["data"].get<std::string>();
                    if (audio_base64.empty()) {
                        header_print("Warning", "Empty audio, skipping this audio.");
                        continue;
                    }
                    merged_audio.push_back(audio_base64);
                }
            }
        }

        //new_message["role"] = message["role"];
        new_message["content"] = merged_text;
        if (!merged_images.empty()) {
            new_message["images"] = merged_images;
        }
        if (!merged_audio.empty()) {
            new_message["audios"] = merged_audio;
        }
        

        template_message.push_back(new_message);
    }

    return template_message;
}


///@brief Try to parse a JSON value from a string, returning the original string on failure.
static json try_parse_json_value(const std::string& s) {
    try {
        return json::parse(s);
    }
    catch (...) {
        return json(s);
    }
}

///@brief Convert OpenAI-style assistant tool_calls + following tool messages into the
/// Gemma4 chat-template format, which expects a single assistant message containing
/// both `tool_calls` and `tool_responses` (with `{name, response}` entries).
static json convert_tool_responses_gemma4(json messages) {
    json converted_messages = json::array();

    size_t i = 0;
    while (i < messages.size()) {
        const auto& msg = messages[i];

        if (msg.value("role", "") != "assistant" || !msg.contains("tool_calls") ||
            !msg.at("tool_calls").is_array() || msg.at("tool_calls").empty()) {
            converted_messages.push_back(msg);
            i++;
            continue;
        }

        // Normalize tool_calls: ensure function.arguments is a JSON object (not a string).
        json normalized_tool_calls = json::array();
        // Map tool_call_id -> function name, for matching tool responses below.
        std::unordered_map<std::string, std::string> id_to_name;
        // Preserve call order so unmatched tool responses can fall back positionally.
        std::vector<std::string> call_names_in_order;

        for (const auto& tc : msg.at("tool_calls")) {
            json new_tc = tc;
            std::string name;
            if (new_tc.contains("function") && new_tc["function"].is_object()) {
                auto& fn = new_tc["function"];
                if (fn.contains("name") && fn["name"].is_string()) {
                    name = fn["name"].get<std::string>();
                }
                if (fn.contains("arguments") && fn["arguments"].is_string()) {
                    fn["arguments"] = try_parse_json_value(fn["arguments"].get<std::string>());
                }
            }
            if (new_tc.contains("id") && new_tc["id"].is_string()) {
                id_to_name[new_tc["id"].get<std::string>()] = name;
            }
            call_names_in_order.push_back(name);
            normalized_tool_calls.push_back(new_tc);
        }

        // Collect consecutive following "tool" role messages as tool_responses.
        json tool_responses = json::array();
        size_t j = i + 1;
        size_t response_index = 0;
        while (j < messages.size() && messages[j].value("role", "") == "tool") {
            const auto& tool_msg = messages[j];

            // Resolve the function name: prefer matching by tool_call_id, then by
            // an explicit name field, then by positional order of the tool_calls.
            std::string name;
            if (tool_msg.contains("tool_call_id") && tool_msg["tool_call_id"].is_string()) {
                auto it = id_to_name.find(tool_msg["tool_call_id"].get<std::string>());
                if (it != id_to_name.end()) name = it->second;
            }
            if (name.empty() && tool_msg.contains("name") && tool_msg["name"].is_string()) {
                name = tool_msg["name"].get<std::string>();
            }
            if (name.empty() && response_index < call_names_in_order.size()) {
                name = call_names_in_order[response_index];
            }

            // Parse content into a JSON value if possible.
            json response_value;
            if (tool_msg.contains("content")) {
                const auto& content = tool_msg.at("content");
                if (content.is_string()) {
                    response_value = try_parse_json_value(content.get<std::string>());
                }
                else {
                    response_value = content;
                }
            }
            else {
                response_value = nullptr;
            }

            tool_responses.push_back({
                {"name", name},
                {"response", response_value},
            });

            response_index++;
            j++;
        }

        json merged = {
            {"role", "assistant"},
            {"tool_calls", normalized_tool_calls},
        };
        if (!tool_responses.empty()) {
            merged["tool_responses"] = tool_responses;
        }
        if (msg.contains("content") && !msg.at("content").is_null()) {
            merged["content"] = msg.at("content");
        }
        if (msg.contains("reasoning_content") && !msg.at("reasoning_content").is_null()) {
            merged["reasoning_content"] = msg.at("reasoning_content");
        }

        converted_messages.push_back(merged);
        i = j;
    }

    return converted_messages;
}

///@brief RestHandler constructor
///@param models the model list
///@param downloader the downloader
///@param default_tag the default tag
///@param asr whether to enable asr
///@param embed whether to enable embedding

///@return the rest handler
RestHandler::RestHandler(model_list& models, ModelDownloader& downloader, program_args_t& args)
    : supported_models(models), downloader(downloader), default_model_tag(args.model_tag), current_model_tag(""), modelscope(args.modelscope), asr(args.asr), embed(args.embed), embedding_model_tag(args.embedding_model), img_pre_resize(args.img_pre_resize), preemption(args.preemption){
    this->npu_device_inst = oflm_rt::device(0);

    if (args.ctx_length != -1) {
        this->ctx_length = args.ctx_length >= 512 ? args.ctx_length : 512;
    } else {
        this->ctx_length = -1;
    }
    if (args.prefill_chunk_len != -1) {
        this->prefill_chunk_len = args.prefill_chunk_len >= 512 ? args.prefill_chunk_len : 512;
    }
    else {
        this->prefill_chunk_len = -1;
    }
    
    // Initialize chat bot with default model
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
    if (this->asr) {
        std::string whisper_tag = "whisper-v3:turbo";
        ensure_asr_model_loaded(whisper_tag);
    }
    if (this->embed) {
        // --embeddingmodel, defaulting to the historical tag so an
        // existing `--embed 1` command line is unchanged. An unknown
        // tag is an error from get_auto_embedding_model(), not a
        // silent fallback: serving one model's embeddings under
        // another's name produces a correctly shaped, correctly normed
        // vector that nothing downstream can tell is wrong.
        std::string embed_tag = this->embedding_model_tag.empty()
                                    ? std::string("embed-gemma:300m")
                                    : this->embedding_model_tag;
        ensure_embed_model_loaded(embed_tag);
    }
#else
    if (this->asr) {
        header_print("Error", "ASR models are not supported in this build");
    }
    if (this->embed) {
        header_print("Error", "Embedding models are not supported in this build");
    }
#endif

    if (default_model_tag != "model-faker") {
        if (!supported_models.is_model_supported(default_model_tag)) {
            header_print("Warning", "Default model tag '" << default_model_tag << "' is not supported. Falling back to 'llama3.2:1b'.");
            this->default_model_tag = "llama3.2:1b";
        }
        if (ensure_model_loaded(default_model_tag) != ModelLoad::Ok) {
            header_print("Error", "Failed to load default model: " + default_model_tag);
        }
    }
    else {
        this->current_model_tag = "model-faker";
    }
    this->prompt_cache = PromptCache();
}

///@brief RestHandler destructor
///@return the rest handler
RestHandler::~RestHandler() = default;

///@brief Ensure the model is loaded
///@param model_tag the model tag
RestHandler::ModelLoad RestHandler::ensure_model_loaded(const std::string& model_tag,
                                                        bool model_field_present) {
    // Normalise FIRST, because both the comparison and the lookup below are exact.
    // Clients send three spellings of one model -- "granite", "granite:3b" and
    // "Ollama/granite:3b" -- and all_tags holds the first two only, while
    // current_model_tag always holds the resolved "granite:3b". The prefixed form was
    // therefore refused as unknown, and a bare "granite" compared unequal to the
    // "granite:3b" it had itself just loaded: every bare-tag request after the first
    // took the switch path below, evicted the model and reloaded it from disk. The
    // only symptom was latency.
    //
    // rectify_model_tag() indexes config["models"][type], so it is only safe once the
    // type is known to exist -- hence the support check around it rather than after.
    std::string ensure_tag = this->supported_models.cut_tag(model_tag);
    if (this->supported_models.is_model_supported(ensure_tag)) {
        ensure_tag = this->supported_models.rectify_model_tag(ensure_tag);
    }
    // "model-faker" is the sentinel for `oflm serve` started without a chat model,
    // and the handlers default `model` to current_model_tag -- so a request naming no
    // model arrives here as the sentinel. Say so, rather than looking it up and
    // reporting it as a typo.
    switch (openai_compat::preflight(model_field_present, ensure_tag, current_model_tag,
                                     auto_chat_engine != nullptr)) {
        case openai_compat::Preflight::Ok:      return ModelLoad::Ok;
        case openai_compat::Preflight::NoModel: return ModelLoad::NoModel;
        case openai_compat::Preflight::BadModelValue:
            header_print("ERROR", "request set 'model' to '" + ensure_tag +
                                  "', which is not a model name -- refusing");
            return ModelLoad::Unknown;
        case openai_compat::Preflight::NeedsLoad: break;
    }
    {
        // Checked BEFORE anything is unloaded. The old order reset the engine first
        // and only then resolved the tag, so a request for a model that does not
        // exist evicted the served one and was answered by the substitute.
        if (!this->supported_models.is_model_supported(ensure_tag)) {
            header_print("ERROR", "unknown model '" + ensure_tag + "' -- refusing; '" +
                                  current_model_tag + "' stays loaded");
            return ModelLoad::Unknown;
        }
        // ... and the same question one level down. `embed-gemma:300m` and
        // `whisper-v3:turbo` ARE in the model list, so the check above passes them;
        // the factory can only refuse them by returning null, and it is called after
        // the loaded engine has already been reset. Asking here keeps the promise the
        // comment above makes -- nothing is unloaded for a request that cannot be served.
        if (!is_chat_model(ensure_tag, this->supported_models)) {
            header_print("ERROR", "model '" + ensure_tag + "' is not a chat model -- refusing; '" +
                                  current_model_tag + "' stays loaded");
            return ModelLoad::NotChatModel;
        }
        // One request naming another model evicts the loaded one, and may pull it first.
        // That is the intended behaviour, but it used to happen with no output at all --
        // a typo in a client's model field took the served model off the NPU and cost a
        // full reload, and the operator's only evidence was the latency.
        if (!current_model_tag.empty() && current_model_tag != "model-faker") {
            header_print("OFLM", "request asked for '" + ensure_tag + "' while '" +
                                 current_model_tag + "' is loaded -- switching; the "
                                 "previous model leaves the NPU and must be reloaded");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (auto_chat_engine != nullptr) {
            auto_chat_engine.reset();
        }
        std::pair<std::string, std::unique_ptr<AutoModel>> auto_model = get_auto_model(ensure_tag, this->supported_models, &this->npu_device_inst);
        auto_chat_engine = std::move(auto_model.second);
        ensure_tag = auto_model.first;
        if (auto_chat_engine == nullptr) {
            // The factory's contract is null-on-failure and this was the one caller that
            // did not honour it -- configure_parameter() below is a dereference. It was
            // unreachable while null meant only "unsupported tag" (checked above); it
            // stopped being unreachable the moment null also meant "not a chat model".
            header_print("ERROR", "no engine for '" + ensure_tag + "'; nothing is loaded now");
            this->current_model_tag = "model-faker";
            return ModelLoad::NotChatModel;
        }
        switch (downloader.is_model_downloaded(ensure_tag)) {
            case ModelDownloader::ModelStatus::Ready:
                break;
            case ModelDownloader::ModelStatus::Outdated:
            case ModelDownloader::ModelStatus::Missing:
                downloader.pull_model(ensure_tag, this->modelscope);
                break;
            case ModelDownloader::ModelStatus::Incompatible:
                // auto_chat_engine holds a freshly CONSTRUCTED engine that was never
                // loaded, and current_model_tag still names the model just evicted --
                // so the next request for that tag took the fast path straight into it.
                // Leave the same state the load-failure catch below leaves.
                this->auto_chat_engine.reset();
                this->current_model_tag = "model-faker";
                return ModelLoad::LoadFailed;
            }
        auto [new_ensure_tag, model_info] = supported_models.get_model_info(ensure_tag);
        auto_chat_engine->configure_parameter("img_pre_resize", this->img_pre_resize);
        try {
            auto_chat_engine->load_model(supported_models.get_model_path(new_ensure_tag), model_info, ctx_length, preemption);
        }
        catch (const std::exception& e) {
            header_print("ERROR", "Failed to load model: " + std::string(e.what()));
            this->auto_chat_engine.reset();
            this->npu_device_inst.reset();
            this->npu_device_inst = oflm_rt::device(0);
            this->current_model_tag = "model-faker";
            return ModelLoad::LoadFailed;
        }
        
        if (this->prefill_chunk_len == -1) {
            this->prefill_chunk_len = model_info["max_prefill_len"].get<int>();;
        }
        current_model_tag = ensure_tag;
    }
    return ModelLoad::Ok;
}

///@brief Ensure the asr model is loaded
///@param model_tag the model tag
void RestHandler::ensure_asr_model_loaded(const std::string& model_tag) {
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
    std::string ensure_tag = model_tag;
    switch (downloader.is_model_downloaded(ensure_tag)) {
        case ModelDownloader::ModelStatus::Ready:
            break;
        case ModelDownloader::ModelStatus::Outdated:
        case ModelDownloader::ModelStatus::Missing:
            downloader.pull_model(ensure_tag, modelscope);
            break;
        case ModelDownloader::ModelStatus::Incompatible:
            header_print("ERROR", "Whisper is incompatible with this version of OpenFlowLM, skipping... ");
            this->asr = false;
            return;
    }
    this->whisper_engine = std::make_unique<Whisper>(&this->npu_device_inst);
    auto [new_ensure_tag, whisper_model_info] = this->supported_models.get_model_info(ensure_tag);
    std::string whisper_model_path = this->supported_models.get_model_path(new_ensure_tag);
    try {
        this->whisper_engine->load_model(whisper_model_path, whisper_model_info, this->preemption);
    }
    catch (const std::exception& e) {
        header_print("ERROR", "Failed to load ASR model: " + std::string(e.what()));
        exit(EXIT_FAILURE);
    }
#else
    throw std::runtime_error("ASR models are not supported in this build");
#endif
}

///@brief Ensure the embed model is loaded
///@param model_tag the model tag
void RestHandler::ensure_embed_model_loaded(const std::string& model_tag) {
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
    std::string ensure_tag = model_tag;
    switch (this->downloader.is_model_downloaded(ensure_tag)) {
        case ModelDownloader::ModelStatus::Ready:
            break;
        case ModelDownloader::ModelStatus::Outdated:
        case ModelDownloader::ModelStatus::Missing:
            this->downloader.pull_model(ensure_tag, this->modelscope);
            break;
        case ModelDownloader::ModelStatus::Incompatible:
            header_print("ERROR", "EmbeddingGemma is incompatible with this version of OpenFlowLM, skipping... ");
            this->embed = false;
            return;
    }
    // `resolved_tag` rather than `embedding_model_tag`: the latter is now a
    // member (the --embeddingmodel value), and shadowing it here would compile
    // fine while making the two impossible to tell apart at a glance.
    auto [resolved_tag, auto_embedding_engine] = get_auto_embedding_model(ensure_tag, &this->npu_device_inst);
    this->auto_embedding_engine = std::move(auto_embedding_engine);
    auto [new_embedding_model_tag, embedding_model_info] = this->supported_models.get_model_info(resolved_tag);
    std::string embedding_model_path = this->supported_models.get_model_path(new_embedding_model_tag);
    try {
        this->auto_embedding_engine->load_model(embedding_model_path, embedding_model_info, this->preemption);
    }
    catch (const std::exception& e) {
        header_print("ERROR", "Failed to load embedding model: " + std::string(e.what()));
        exit(EXIT_FAILURE);
    }
#else
    throw std::runtime_error("Embedding models are not supported in this build");
#endif
}

///@brief Configure chat engine parameters from options and request
///@param options the options JSON object
///@param request the request JSON object
void RestHandler::configure_chat_engine_parameters(const json& options, const json& request) {
    if (request.contains("temperature")) {
        float temperature = request["temperature"];
        auto_chat_engine->set_temperature(temperature);
    }
    if (request.contains("top_p")) {
        float top_p = request["top_p"];
        auto_chat_engine->set_topp(top_p);
    }
    if (request.contains("top_k")) {
        int top_k = request["top_k"];
        auto_chat_engine->set_topk(top_k);
    }
    if (request.contains("min_p")) {
        int min_p = request["min_p"];
        auto_chat_engine->set_minp(min_p);
    }
    if (request.contains("presence_penalty")) {
        float presence_penalty = request["presence_penalty"];
        auto_chat_engine->set_presence_penalty(presence_penalty);
    }
    if (request.contains("frequency_penalty")) {
        float frequency_penalty = request["frequency_penalty"];
        auto_chat_engine->set_frequency_penalty(frequency_penalty);
    }
    if (request.contains("repetition_penalty")) {
        float repetition_penalty = request["repetition_penalty"];
        auto_chat_engine->set_repetition_penalty(repetition_penalty);
    }
    if (request.contains("think")) {
        bool enable_thinking = request["think"];
        auto_chat_engine->configure_parameter("enable_think", enable_thinking);
    }
    if (request.contains("reasoning_effort")) {
        std::string reasoning_effort = request["reasoning_effort"];
        auto_chat_engine->configure_parameter("reasoning_effort", reasoning_effort);
    }

    if (request.contains("image-max-tokens")) {
        int image_max_tokens = request["image-max-tokens"];
        auto_chat_engine->configure_parameter("image_max_tokens", image_max_tokens);
    }
}

json RestHandler::build_nstream_response(std::string response_text,
                                         stop_reason_t stop_reason) {
    // Get tool info
    NonStreamResult result = auto_chat_engine->parse_nstream_content(response_text);

    json message;
    message["role"] = "assistant";

    bool is_reasoning = !result.reasoning_content.empty();
    bool is_tool_call = !result.tool_calls_list.empty() || !result.tool_name.empty();

    if (is_reasoning) {
        message["reasoning_content"] = result.reasoning_content;
    }

    if (is_tool_call) {
        json tool_calls_json = json::array();
        if (!result.tool_calls_list.empty()) {
            int idx = 0;
            for (const auto& tc : result.tool_calls_list) {
                tool_calls_json.push_back({
                    {"index", idx++},
                    {"id", "call_" + std::to_string(std::time(nullptr)) + "_" + std::to_string(idx)},
                    {"type", "function"},
                    {"function", {
                        {"name", tc.first},
                        {"arguments", tc.second}
                    }}
                });
            }
        } else {
            tool_calls_json.push_back({
                {"index", 0},
                {"id", "call_" + std::to_string(std::time(nullptr))},
                {"type", "function"},
                {"function", {
                    {"name", result.tool_name},
                    {"arguments", result.tool_args}
                }}
            });
        }
        message["tool_calls"] = tool_calls_json;
        if (!result.content.empty()) {
            message["content"] = result.content;
        }
    }
    else {
        message["content"] = result.content;
    }


    // Construct the final choice object
    return json::array({
        {
            {"index", 0},
            {"message", message},
            {"logprobs", nullptr},
            {"finish_reason", is_tool_call ? "tool_calls"
                                           : openai_compat::finish_reason(stop_reason)}
        }
    });
}

///@brief Handle the show request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_show(const json& request,
    std::function<void(const json&)> send_response,
    StreamResponseCallback send_streaming_response) {
    try {
        std::string model = request["model"];
        json info = {
            {"modelfile", ""},
            {"parameters", ""},
            {"template", ""},
            {"details", {
                {"parent_model", ""},
                {"format", ""},
                {"family", ""},
                {"families", {""}},
                {"parameter_size", ""},
                {"quantization_level", ""}
                }
            },
            {"model_info", {
                {"general.architecture", "oflm" }
            }},
            {"capabilities", {"chat", "vision", "completion"}}
        };


        send_response(info);
    }
    catch (const std::exception& e) {
        json error_response = { {"error", e.what()} };
        send_response(error_response);
    }
}

///@brief Handle the generate request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_generate(const json& request,
                                 std::function<void(const json&)> send_response,
                                 StreamResponseCallback send_streaming_response,
                                 std::shared_ptr<CancellationToken> cancellation_token) {
    try {
        std::string prompt = request["prompt"];
        bool stream = request.value("stream", true);
        std::string model = request.value("model", current_model_tag);
        json options = request.value("options", json::object());
       
        int length_limit = request.value("max_tokens", 4096);
        auto load_start_time = time_utils::now();
        // TODO: Use Another Check Function avoid loading again
        if (const ModelLoad why = ensure_model_loaded(model, request.contains("model")); why != ModelLoad::Ok) {
            send_response(openai_compat::model_error(why, model));
            return;
        }
        auto load_end_time = time_utils::now();
      
        chat_meta_info_t meta_info;
        lm_uniform_input_t uniformed_input;
        meta_info.max_prefill_len = this->prefill_chunk_len;
        meta_info.load_duration = (uint64_t)time_utils::duration_ns(load_start_time, load_end_time).first;
        header_print("OFLM", "Start generating...");
        
        if (stream) {
            // Streaming response using streaming_ostream
            auto total_start_time = time_utils::now();
            streaming_ostream ostream(model, send_streaming_response, false);
            uniformed_input.prompt = prompt;
            try {
                bool success = auto_chat_engine->insert(meta_info, uniformed_input);
                if (!success){
                    json error_response = {{"error", {
                        {"message", "the prompt does not fit this model's context window"},
                        {"type", "invalid_request_error"},
                        {"param", "messages"},
                        {"code", "context_length_exceeded"}
                    }}};
                    send_response(error_response);
                    this->auto_chat_engine->clear_context();
                    return;
                }
            } catch (const std::exception& e) {
                json error_response = {{"error", e.what()}};
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            try {
                auto_chat_engine->generate(meta_info, length_limit, ostream);
            } catch (const std::exception& e) {
                json error_response = {{"error", e.what()}};
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            auto total_end_time = time_utils::now();
            auto history = this->auto_chat_engine->get_history();
            // std::cout << "history: " << history.first << std::endl;
            meta_info.total_duration = (uint64_t)time_utils::duration_ns(total_start_time, total_end_time).first;
            ostream.finalize_generate(meta_info, history.second);
        } else {
            // Non-streaming response
            std::stringstream ss;
            wstream_buf obuf(ss);
            std::ostream ostream(&obuf);
            uniformed_input.prompt = prompt;
            try {
                bool success = auto_chat_engine->insert(meta_info, uniformed_input);
                if (!success){
                    json error_response = {{"error", {
                        {"message", "the prompt does not fit this model's context window"},
                        {"type", "invalid_request_error"},
                        {"param", "messages"},
                        {"code", "context_length_exceeded"}
                    }}};
                    send_response(error_response);
                    this->auto_chat_engine->clear_context();
                    return;
                }
            } catch (const std::exception& e) {
                json error_response = {{"error", e.what()}};
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            try {
                auto_chat_engine->generate(meta_info, length_limit, ostream);
            } catch (const std::exception& e) {
                json error_response = {{"error", e.what()}};
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            std::string response_text = ss.str();
            auto history = this->auto_chat_engine->get_history();
            json response = {
                {"model", model},
                {"response", response_text},
                {"context", history.second},
                {"done", true},
                {"prompt_eval_count", meta_info.prompt_tokens},
                {"eval_count", meta_info.generated_tokens},
                {"total_duration", meta_info.total_duration},
                {"load_duration", meta_info.load_duration},
                {"prompt_eval_duration", meta_info.prefill_duration},
                {"eval_duration", meta_info.decoding_duration},
                {"done_reason", stop_reason_to_string(meta_info.stop_reason)}
            };
            // std::cout << "history: " << history.first << std::endl;
            send_response(response);
        }
    } catch (const std::exception& e) {
        json error_response = {{"error", e.what()}};
        send_response(error_response);
    }
}

///@brief Handle the chat request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_chat(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response,
                             std::shared_ptr<CancellationToken> cancellation_token) {
    try {
        nlohmann::ordered_json messages = request["messages"];
        bool stream = request.value("stream", false);
        std::string model = request.value("model", current_model_tag);
        json options = request.value("options", json::object());
        int length_limit = options.value("num_predict", 4096);

        auto load_start_time = time_utils::now();
        if (const ModelLoad why = ensure_model_loaded(model, request.contains("model")); why != ModelLoad::Ok) {
            send_response(openai_compat::model_error(why, model));
            return;
        }
        auto load_end_time = time_utils::now();
       
        configure_chat_engine_parameters(options, request);

        // messages = normalize_messages(messages);
        
        chat_meta_info_t meta_info;
        lm_uniform_input_t uniformed_input;
        meta_info.load_duration = (uint64_t)time_utils::duration_ns(load_start_time, load_end_time).first;
        meta_info.max_prefill_len = this->prefill_chunk_len;
        header_print("OFLM", "Start generating...");
        if (stream) {
            // Streaming response using streaming_ostream
            auto total_start_time = time_utils::now();
            streaming_ostream ostream(model, send_streaming_response, true);  // true for chat format
            uniformed_input.messages = messages;
            try {
                bool success = auto_chat_engine->insert(meta_info, uniformed_input);
                if (!success){
                    json error_response = {{"error", {
                        {"message", "the prompt does not fit this model's context window"},
                        {"type", "invalid_request_error"},
                        {"param", "messages"},
                        {"code", "context_length_exceeded"}
                    }}};
                    send_response(error_response);
                    this->auto_chat_engine->clear_context();
                    return;
                }
            } catch (const std::exception& e) {
                json error_response = {{"error", e.what()}};
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            try {
                bool success = auto_chat_engine->insert(meta_info, uniformed_input);
                if (!success){
                    json error_response = {{"error", {
                        {"message", "the prompt does not fit this model's context window"},
                        {"type", "invalid_request_error"},
                        {"param", "messages"},
                        {"code", "context_length_exceeded"}
                    }}};
                    send_response(error_response);
                    this->auto_chat_engine->clear_context();
                    return;
                }
            } catch (const std::exception& e) {
                json error_response = {{"error", e.what()}};
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            auto total_end_time = time_utils::now();
            meta_info.total_duration = (uint64_t)time_utils::duration_ns(total_start_time, total_end_time).first;
            
            ostream.finalize_chat(meta_info);
            // auto history = this->chat_engine->get_history();
            // std::cout << "history: " << history.first << std::endl;
            this->auto_chat_engine->clear_context();
        } else {
            // Non-streaming response
            uniformed_input.messages = messages;
            auto total_start_time = time_utils::now();
            nullstream nstream;
            //std::string response_text = auto_chat_engine->generate_with_prompt(meta_info, uniformed_input, length_limit, std::cout);
            std::string response_text;
            try {
                response_text = auto_chat_engine->generate_with_prompt(meta_info, uniformed_input, length_limit, nstream);
            } catch (const std::exception& e) {
                json error_response = {{"error", e.what()}};
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            //std::string response_text = chat_engine->generate_with_prompt(meta_info, prompts, length_limit, std::cout, payload);
            auto total_end_time = time_utils::now();
            meta_info.total_duration = (uint64_t)time_utils::duration_ns(total_start_time, total_end_time).first;
            
            json response = {
                {"model", model},
                {"message", {
                    {"role", "assistant"},
                    {"content", response_text},
                    {"images", nullptr}
                }},
                {"done", true},
                {"prompt_eval_count", meta_info.prompt_tokens},
                {"eval_count", meta_info.generated_tokens},
                {"total_duration", meta_info.total_duration},
                {"load_duration", meta_info.load_duration},
                {"prompt_eval_duration", meta_info.prefill_duration},
                {"eval_duration", meta_info.decoding_duration},
                {"done_reason", stop_reason_to_string(meta_info.stop_reason)}
            };
            send_response(response);
            
            // auto history = this->chat_engine->get_history();
            // std::cout << "history: " << history.first << std::endl;
            this->auto_chat_engine->clear_context();
        }
    } catch (const std::exception& e) {
        json error_response = {{"error", e.what()}};
        send_response(error_response);
    }
}

///@brief Handle the embeddings request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_embeddings(const json& request,
                                   std::function<void(const json&)> send_response,
                                   StreamResponseCallback send_streaming_response) {
    try {
        std::string model = request["model"];

        // THE `model` FIELD USED TO BE ECHOED AND OTHERWISE IGNORED, which is
        // the worst version of a wrong answer: the response ASSERTED it was
        // something it was not.
        //
        // One embedding model is loaded per server (the engine's geometry is
        // process-wide). A request naming any other one was served by the
        // loaded model anyway, and the reply came back labelled with the tag
        // that had been ASKED for -- so a client comparing response.model to
        // its request saw agreement. Measured on a server started with
        // --embeddingmodel bge-base:en-v1.5: asking for gte-multilingual:base
        // returned bge-base's vectors, byte for byte, under the name
        // "gte-multilingual:base". A RAG deployment embedding documents with
        // one model and queries with another, against one oflm, would retrieve
        // nonsense with no signal anywhere.
        //
        // It refuses now, and names what IS loaded. An unknown model is an
        // error every OpenAI client already understands.
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
        if (this->auto_embedding_engine) {
            const std::string loaded = this->auto_embedding_engine->get_current_model();
            if (!model.empty() && !loaded.empty() && model != loaded) {
                json err = { {"error", {
                    {"message", "this server has '" + loaded + "' loaded, not '" +
                                model + "'. One embedding model is loaded per "
                                "server; start another with --embeddingmodel " +
                                model + " to serve it."},
                    {"type", "invalid_request_error"},
                    {"param", "model"},
                    {"code", "model_not_found"}
                }} };
                send_response(err);
                return;
            }
        }
#endif

        // The task prompt. nomic-embed-text and friends prepend a per-task prefix, and
        // which one is chosen changes the vector materially -- measured on this server,
        // search_query against search_document on the same text is cosine 0.914, not 1.
        // This handler used to pass task_query unconditionally and ignore the request, so
        // every DOCUMENT was embedded as a QUERY and no caller could tell: the vector is
        // correctly shaped, correctly normed and deterministic either way.
        // The task prompt. nomic-embed-text and friends prepend a per-task prefix, and
        // which one is chosen changes the vector materially -- measured on this server,
        // search_query against search_document on the same text is cosine 0.914, not 1.
        // This handler used to pass task_query unconditionally and ignore the request, so
        // every DOCUMENT was embedded as a QUERY and no caller could tell.
        embedding_task_type_t task_type = embedding_task_type_t::task_query;
        const std::string accepted = openai_compat::task_names_csv();
        std::vector<std::string> declared;
        bool supports_prompts = false;
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
        if (this->auto_embedding_engine) {
            declared = this->auto_embedding_engine->prompt_names();
            supports_prompts = this->auto_embedding_engine->supports_task_prompts();
        }
#endif
        const openai_compat::TaskResolution tr = openai_compat::resolve_task(request);
        using TRS = openai_compat::TaskResolution::Status;

        const openai_compat::TaskPolicy policy =
            openai_compat::task_policy(supports_prompts, !declared.empty(), tr.status != TRS::Absent);
        if (policy == openai_compat::TaskPolicy::NotSupported) {
            // prompt_for() returns an empty prefix for a model with no prompt table,
            // so this used to answer 200 with an UNPREFIXED vector -- correctly
            // shaped, correctly normed, and not what was asked for.
            // src/open_npue_adapter/README.md: "model has no prompts, a prompt is
            // given | error".
            send_response(json{{"error", {
                {"message", "model '" + model + "' has no task prompts; remove '" + tr.field +
                            "'. Passing one would be ignored, and the vector would come back "
                            "correctly shaped and unprefixed with nothing to show it."},
                {"type", "invalid_request_error"}, {"param", tr.field}, {"code", "invalid_value"}}}});
            return;
        }
        if (policy == openai_compat::TaskPolicy::Required) {
            std::string names;
            for (const auto& n : declared) names += (names.empty() ? "" : ", ") + n;
            // Quote the REST vocabulary, not `declared`: the validator only accepts
            // the former, so naming the latter sent clients to values it refuses.
            json err = { {"error", {
                {"message", "this model requires a task prompt: pass 'prompt_name' as "
                            "one of [" + accepted + "] (this model declares the prompts [" +
                            names + "], which those names map onto). Refusing to pick one -- "
                            "the prefix changes the vector (search_query against "
                            "search_document on the same text is cosine 0.914 here), and the "
                            "result is correctly shaped, correctly normed and deterministic "
                            "either way, so nothing downstream can tell the wrong one was used."},
                {"type", "invalid_request_error"},
                {"param", "prompt_name"},
                {"code", "missing_required_parameter"}
            }} };
            send_response(err);
            return;
        }
        if (tr.status == TRS::NotAString) {
            send_response(json{{"error", {
                {"message", "'" + tr.field + "' must be a string, one of [" + accepted + "]"},
                {"type", "invalid_request_error"}, {"param", tr.field}, {"code", "invalid_value"}}}});
            return;
        }
        if (tr.status == TRS::Unknown) {
            send_response(json{{"error", {
                {"message", "unknown " + tr.field + " '" + tr.value + "'. Known: [" + accepted +
                            "]. Refusing to substitute one: an embedding under the wrong task "
                            "prompt is correctly shaped and correctly normed, so nothing "
                            "downstream can tell it is wrong."},
                {"type", "invalid_request_error"}, {"param", tr.field}, {"code", "invalid_value"}}}});
            return;
        }
        if (tr.status == TRS::Conflict) {
            send_response(json{{"error", {
                {"message", "'prompt_name' and 'task_type' are aliases and disagree "
                            "('task_type' says '" + tr.value + "'). Send one, or send the same "
                            "task in both."},
                {"type", "invalid_request_error"}, {"param", tr.field}, {"code", "invalid_value"}}}});
            return;
        }
        if (tr.status == TRS::Ok) task_type = tr.task;

        // `input` is REQUIRED, and this guard is not cosmetic: `request` is a
        // const json&, so reading a missing key through operator[] is undefined
        // behaviour. In practice it answered 200 with an empty data array, which
        // made a malformed request look exactly like one that asked for nothing.
        if (!request.contains("input")) {
            send_response(json{{"error", {
                {"message", "input is required: a string, or an array of strings."},
                {"type", "invalid_request_error"},
                {"param", "input"},
                {"code", "missing_required_parameter"}}}});
            return;
        }

        std::vector<std::string> inputs;

        // The TYPE has to be checked too, not just the presence. `input: null`,
        // a number or an object left `inputs` empty and answered 200 with an
        // empty `data` -- indistinguishable from a request that asked for
        // nothing. An array holding a non-string threw out of
        // get<std::string>() into the function-level catch, which is a 500 for
        // what is plainly a client error. Both are 400s, named, before any work.
        const json& input_field = request.at("input");
        if (input_field.is_string()) {
            inputs.push_back(input_field.get<std::string>());
        }
        else if (input_field.is_array()) {
            for (size_t i = 0; i < input_field.size(); ++i) {
                if (!input_field[i].is_string()) {
                    send_response(json{{"error", {
                        {"message", "input[" + std::to_string(i) + "] is not a string."
                                    " input must be a string, or an array of strings."},
                        {"type", "invalid_request_error"},
                        {"param", "input[" + std::to_string(i) + "]"},
                        {"code", "invalid_value"}}}});
                    return;
                }
                inputs.push_back(input_field[i].get<std::string>());
            }
        }
        else {
            // An empty ARRAY is deliberately still 200 with an empty list: that
            // request is well formed and its answer is correct. This branch is
            // for null, numbers, booleans and objects, which are not.
            send_response(json{{"error", {
                {"message", "input must be a string, or an array of strings."},
                {"type", "invalid_request_error"},
                {"param", "input"},
                {"code", "invalid_value"}}}});
            return;
        }

        json response;
        // -1 means the backend did not report a count; see the usage field below.
        int64_t prompt_tokens = -1;
        if (this->embed) {
            json embedding_data = json::array();
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
            try {
                // ONE call for the whole array, not one per input.
                //
                // AutoEmbeddingModel::embed_batch() defaults to exactly the loop
                // this replaces, so a backend that does not override it behaves
                // identically. NpueEmbedding does override it and encodes a whole
                // tier of sequences per dispatch: measured 5-10x faster on all six
                // of its models, peaking at 10.3x
                // (docs/docs/benchmarks/embeddings_results.md).
                //
                // The vectors are BIT-IDENTICAL either way. Batching is a
                // scheduling choice, not an arithmetic one, which is precisely why
                // nothing here could ever have flagged the loop: no accuracy check,
                // cosine or byte comparison can tell the slow path from the fast
                // one. The only symptom was time, and the endpoint measured none.
                if (!inputs.empty()) {
                    const std::vector<float> flat =
                        this->auto_embedding_engine->embed_batch(inputs, task_type,
                                                                &prompt_tokens);
                    // The vectors come back concatenated, so the width has to be
                    // derived -- and therefore CHECKED. A mis-split returns
                    // correctly shaped, correctly normed, deterministic vectors for
                    // the wrong inputs, which is the one failure nothing downstream
                    // can see. openai_compat::embedding_batch_dim() refuses rather
                    // than dividing and hoping, and openai_compat_test holds it to
                    // that without a device -- one definition, shared with
                    // bench-embed, so the two cannot drift.
                    const size_t dim =
                        openai_compat::embedding_batch_dim(flat.size(), inputs.size());
                    for (size_t i = 0; i < inputs.size(); ++i) {
                        embedding_data.push_back({
                            {"object", "embedding"},
                            {"embedding", std::vector<float>(
                                 flat.begin() + static_cast<std::ptrdiff_t>(i * dim),
                                 flat.begin() + static_cast<std::ptrdiff_t>((i + 1) * dim))},
                            {"index", i}
                        });
                    }
                    // One line, not one per input with the text echoed back. The
                    // old print put the full text of every request on the console.
                    header_print("OFLM", "embedded " + std::to_string(inputs.size()) +
                                         " input(s), " + std::to_string(dim) + " dims");
                }
            } catch (const TaskPromptUnavailable& e) {
                // The model has prompts but none serves this task -- README.md:288's
                // "model has prompts, task maps to none of them | error naming what the
                // model does offer". The engine refuses on purpose; this used to reach
                // the function-level catch as {"error": <string>} and go out as 200.
                send_response(json{{"error", {
                    {"message", std::string(e.what())},
                    {"type", "invalid_request_error"},
                    {"param", tr.field.empty() ? std::string("prompt_name") : tr.field},
                    {"code", "invalid_value"}}}});
                return;
            }
#else
            throw std::runtime_error("Embedding models are not supported in this build");
#endif

            response = {
                {"object", "list"},
                {"data", embedding_data},
                {"model", model},
                {"usage", {
                    // A real count when the backend reports one. 0 still means
                    // NOT REPORTED -- which is what every request got before this,
                    // on both backends. embed_batch() yields -1 rather than 0 for a
                    // backend that does not count, because a zero there would read
                    // as a real number.
                    {"prompt_tokens", prompt_tokens < 0 ? 0 : prompt_tokens},
                    {"total_tokens",  prompt_tokens < 0 ? 0 : prompt_tokens}
                }}
            };
        }
        else {
            header_print("Warning", "No embedding model loaded");
        }
        send_response(response);
    }
    catch (const std::exception& e) {
        json error_response = {{"error", e.what()}};
        send_response(error_response);
    }
}

///@brief Handle the models request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_models(const json& request,
                               std::function<void(const json&)> send_response,
                               StreamResponseCallback send_streaming_response) {
    try {
        json models = supported_models.get_all_models_ollama();
        send_response(models);
    } catch (const std::exception& e) {
        json error_response = {{"error", e.what()}};
        send_response(error_response);
    }
}

///@brief Handle the version request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_version(const json& request,
                                std::function<void(const json&)> send_response,
                                StreamResponseCallback send_streaming_response) {
    std::string version = __OFLM_VERSION__;
    
    json response = {{"version", version}};
    send_response(response);
}

///@brief Handle the models request (open ai)
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_models_openai(const json& request,
    std::function<void(const json&)> send_response,
    StreamResponseCallback send_streaming_response) {
    try {
        json models = supported_models.get_all_models_openai();
        send_response(models);
    }
    catch (const std::exception& e) {
        json error_response = { {"error", e.what()} };
        send_response(error_response);
    }
}

///@brief Handle the ps request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_ps(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response) {
    try {
        // Generate expires_at timestamp (1 hour from now)
        auto now = std::chrono::system_clock::now();
        auto expires_time = now + std::chrono::hours(1);
        auto expires_time_t = std::chrono::system_clock::to_time_t(expires_time);
        auto expires_tp = std::chrono::system_clock::from_time_t(expires_time_t);
        auto fractional_seconds = std::chrono::duration_cast<std::chrono::microseconds>(expires_time - expires_tp).count();
        
        // Get local time and timezone offset
        std::tm* local_tm = std::localtime(&expires_time_t);
        std::tm* utc_tm = std::gmtime(&expires_time_t);
        
        // Calculate timezone offset in minutes
        int offset_minutes = (local_tm->tm_hour - utc_tm->tm_hour) * 60 + (local_tm->tm_min - utc_tm->tm_min);
        if (local_tm->tm_mday != utc_tm->tm_mday) {
            offset_minutes += (local_tm->tm_mday > utc_tm->tm_mday) ? 1440 : -1440;
        }
        
        std::stringstream expires_ss;
        expires_ss.imbue(std::locale::classic()); // Use C locale to avoid commas
        expires_ss << std::put_time(local_tm, "%Y-%m-%dT%H:%M:%S");
        expires_ss << "." << std::setfill('0') << std::setw(5) << (fractional_seconds / 40); // 5 decimal places
        
        // Format timezone offset
        int offset_hours = offset_minutes / 60;
        int offset_mins = abs(offset_minutes % 60);
        expires_ss << (offset_minutes >= 0 ? "+" : "-") 
                  << std::setfill('0') << std::setw(2) << abs(offset_hours)
                  << ":" << std::setfill('0') << std::setw(2) << offset_mins;
        
        std::string expires_at = expires_ss.str();
        
        auto [new_current_model_tag, model_info] = supported_models.get_model_info(current_model_tag);
        json response = {
            {"models", json::array({
                {
                    {"name", current_model_tag},
                    {"model", current_model_tag},
                    {"size", model_info["size"]},
                    {"details", model_info["details"]},
                    {"expires_at", expires_at},
                }
            })}
        };
        // std::cout << "response: " << response.dump(4) << std::endl;
        send_response(response);
    } catch (const std::exception& e) {
        json error_response = {{"error", e.what()}};
        send_response(error_response);
    }
}

///@brief Handle the pull request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_pull(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Pull operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the push request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_push(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Push operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the delete request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_delete(const json& request,
                               std::function<void(const json&)> send_response,
                               StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Delete operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the copy request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_copy(const json& request,
                             std::function<void(const json&)> send_response,
                             StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Copy operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the create request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_create(const json& request,
                               std::function<void(const json&)> send_response,
                               StreamResponseCallback send_streaming_response) {
    json error_response = {{"error", "Create operation not implemented"}};
    send_response(error_response);
}

///@brief Handle the openai chat completion request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_openai_chat_completion(const json& request,
                                               std::function<void(const json&)> send_response,
                                               StreamResponseCallback send_streaming_response,
                                               std::shared_ptr<CancellationToken> cancellation_token) {
    static std::string model_used_for_last_message = "model-faker";
    try {
        // Extract OpenAI-style parameters
        json current_messages = request["messages"];
        std::string model = request.value("model", current_model_tag);
        bool stream = request.value("stream", false);
        int length_limit = request.value("max_tokens", request.value("max_completion_tokens", 4096));
        json tools = request.value("tools", json::array());
        json options = request.value("options", json::object());

        auto load_start_time = time_utils::now();
        if (const ModelLoad why = ensure_model_loaded(model, request.contains("model")); why != ModelLoad::Ok) {
            send_response(openai_compat::model_error(why, model));
            return;
        }
        auto load_end_time = time_utils::now();

        configure_chat_engine_parameters(options, request);

        current_messages = normalize_messages(current_messages);
        current_messages = normalize_template(current_messages);

        // see if we can use prompt cache
        chat_meta_info_t meta_info;
        bool can_use_prompt_cache = false;
        if (model != model_used_for_last_message) { // switch models will clear context
            this->prompt_cache.update_message_checksum(current_messages);
            this->prompt_cache.update_tool_checksum(tools);
            model_used_for_last_message = model;
        }
        else {
            cache_match_info_t cache_info;
            can_use_prompt_cache = prompt_cache.can_use_cache(current_messages, auto_chat_engine->get_chat_template_type(), tools, cache_info);
            if (can_use_prompt_cache) {
                meta_info.restore_allowed = true;
                header_print("OFLM", "Use cached prompt!");
                header_print("OFLM", "Matched " + std::to_string(cache_info.matched_rounds) +
                    " out of " + std::to_string(cache_info.total_rounds) + " messages (" +
                    std::to_string(cache_info.total_rounds - cache_info.matched_rounds) + " new to prefill).");
            }
            else {
                // cannot use cache, clear and re-insert all
                header_print("OFLM", "Prompt cache miss.");
                header_print("OFLM", "Clearing context...");
                auto_chat_engine->clear_context();
            }
        }

        if (model.starts_with("gemma4-it")) {
            current_messages = convert_tool_responses_gemma4(current_messages);
        }

        // std::cout << "OFLM current_messages: \n" << current_messages.dump(4) << std::endl;

        lm_uniform_input_t uniformed_input;
        uniformed_input.messages = current_messages;
        uniformed_input.tools = tools;
        meta_info.load_duration = (uint64_t)time_utils::duration_ns(load_start_time, load_end_time).first;
        meta_info.max_prefill_len = this->prefill_chunk_len;
        if (stream){
            // Create a wrapper callback that passes the pre-formatted SSE string directly
            cancellation_token->reset();
            auto_chat_engine->reset_parser();
            auto openai_stream_callback = [&send_streaming_response](const std::string& data, bool is_final) {
                json data_json = data;
                send_streaming_response(data_json, is_final);
                };
            streaming_ostream_openai_chat ostream(model, auto_chat_engine.get(), openai_stream_callback);  // streaming in chat completion format

            header_print("OFLM", "Start prefill...");
            try {
                bool success = auto_chat_engine->insert(meta_info, uniformed_input, [&] { return cancellation_token->cancelled(); });
                if (!success) {
                    if (meta_info.stop_reason == CANCEL_DETECTED || cancellation_token->cancelled()) {
                        meta_info.stop_reason = CANCEL_DETECTED;
                        header_print("❌ ", "Prefill Cancelled!");
                        ostream.finalize(meta_info);
                        this->auto_chat_engine->clear_context();
                        this->prompt_cache.reset();
                        return;
                    }

                    json error_response = {
                        {"error", {
                        {"message", "Max length reached!"},
                        {"type", "model_error"},
                        {"code", 400}
                        }}
                    };
                    send_response(error_response);
                    this->auto_chat_engine->clear_context();
                    this->prompt_cache.reset();
                    return;
                }
            } catch (const std::exception& e) {
                json error_response = {{"error", e.what()}};
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                this->prompt_cache.reset();
                return;
            }
            header_print("OFLM", "Start generating...");
            try {
                auto_chat_engine->generate(meta_info, length_limit, ostream, [&] { return cancellation_token->cancelled(); });
            } catch (const std::exception& e) {
                json error_response = {{"error", e.what()}};
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                this->prompt_cache.reset();
                return;
            }
            if (meta_info.stop_reason == CANCEL_DETECTED) {
                header_print("❌ ", "Generation Cancelled!");
                this->prompt_cache.reset();
            }
                        
            ostream.finalize(meta_info);
        }
        else {
            nullstream nstream;
            json response;
            std::string response_text;
            header_print("OFLM", "Start prefill...");
            try {
                bool success = auto_chat_engine->insert(meta_info, uniformed_input, [&] { return cancellation_token->cancelled(); });
                if (!success) {
                    if (meta_info.stop_reason == CANCEL_DETECTED || cancellation_token->cancelled()) {
                        meta_info.stop_reason = CANCEL_DETECTED;
                        header_print("❌ ", "Prefill Cancelled!");
                        send_response(response);
                        this->auto_chat_engine->clear_context();
                        this->prompt_cache.reset();
                        return;
                    }
                    json error_response = {
                        {"error", {
                        {"message", "Max length reached!"},
                        {"type", "model_error"},
                        {"code", 400}
                        }}
                    };
                    send_response(error_response);
                    this->auto_chat_engine->clear_context();
                    this->prompt_cache.reset();
                    return;
                }
            } catch (const std::exception& e) {
                json error_response = {{"error", e.what()}};
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                this->prompt_cache.reset();
                return;
            }
            header_print("OFLM", "Start generating...");
            try {
                response_text = auto_chat_engine->generate(meta_info, length_limit, nstream, [&] { return cancellation_token->cancelled(); });
            } catch (const std::exception& e) {
                json error_response = {{"error", e.what()}};
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                this->prompt_cache.reset();
                return;
            }
            // check response_text
            // meta_info.stop_reason is MAX_LENGTH_REACHED when generation stopped at
            // max_tokens. It was computed and then dropped, so every truncated answer
            // reported finish_reason "stop" and no client could see the cut.
            json choices = build_nstream_response(response_text, meta_info.stop_reason);
            response = {
                {"id", "openflowlm-chat-completion"},
                {"object", "chat.completion"},
                {"created", static_cast<long long>(std::time(nullptr))},
                {"model", model},
                {"choices", choices},
                {"usage", {
                    {"prompt_tokens", meta_info.prompt_tokens},
                    {"completion_tokens", meta_info.generated_tokens},
                    {"total_tokens", meta_info.prompt_tokens + meta_info.generated_tokens},
                    {"kv_token_occupancy_rate_percentage", (float)this->auto_chat_engine->get_current_context_length() / (float)this->auto_chat_engine->get_max_length() * 100},
                    {"load_duration", static_cast<double>(meta_info.load_duration) / 1'000'000'000},
                    {"prefill_duration_ttft", static_cast<double>(meta_info.prefill_duration) / 1'000'000'000},
                    {"decoding_duration", static_cast<double>(meta_info.decoding_duration) / 1'000'000'000},
                    {"prefill_speed_tps", static_cast<double>(meta_info.prompt_tokens) / static_cast<double>(meta_info.prefill_duration) * 1'000'000'000},
                    {"decoding_speed_tps", static_cast<double>(meta_info.generated_tokens) / static_cast<double>(meta_info.decoding_duration) * 1'000'000'000},
                }},
                {"service_tier", "default"}
            };
            if (meta_info.stop_reason == CANCEL_DETECTED) {
                header_print("❌ ", "Generation Cancelled!");
                this->prompt_cache.reset();
            }
            send_response(response);
        }

    } catch (const std::exception& e) {
        json error_response = {
            {"error", {
                {"message", e.what()},
                {"type", "server_error"},
                {"code", 500}
            }}
        };
        send_response(error_response);
    }
}

///@brief Handle the openai audio transcriptions request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_openai_audio_transcriptions(const json& request,
                                        std::function<void(const json&)> send_response,
                                        StreamResponseCallback send_streaming_response,
                                        std::shared_ptr<CancellationToken> cancellation_token) {
    try {
        std::string model = request["model"];
        std::string file_content = request["file"].get<std::string>();
        std::vector<uint8_t> audio_raw(file_content.begin(), file_content.end());
        bool stream = request.value("stream", false);
        json response;
        if (this->asr) {
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
            this->whisper_engine->load_audio(audio_raw);
            header_print("OFLM", "Transforming audio to text...");
            // Show text
            std::cout << "Audio content: " << std::flush;
            std::pair<std::string, std::string> audio_result = this->whisper_engine->generate(Whisper::whisper_task_type_t::e_transcribe, true, false, std::cout);
            std::string audio_context = audio_result.first;
            std::cout << std::endl;
#else
            throw std::runtime_error("ASR models are not supported in this build");
            std::string audio_context;
#endif

            response = {
                {"model", model},
                {"text", audio_context}
                //{"usage", {
                //    {"type", "tokens"},
                //    {"input_tokens", 0},
                //    {"input_tokens_details", json::array({
                //        {
                //            {"text_tokens", 0},
                //            {"audio_tokens", 0}
                //        }
                //    })},
                //    {"output_tokens", 0},
                //    {"total_tokens", 0}
                //}}
            };
        }
        else {
            header_print("Warning", "No asr model loaded, cannot load audio file");
        }
        send_response(response);
        //this->whisper_engine->clear_context();
    }
    catch (const std::exception& e) {
        json error_response = {
            {"error", {
                {"message", e.what()},
                {"type", "server_error"},
                {"code", 500}
            }}
        };
        send_response(error_response);
    }
}

///@brief Handle the openai completion request
///@param request the request
///@param send_response the send response
///@param send_streaming_response the send streaming response
void RestHandler::handle_openai_completion(const json& request,
    std::function<void(const json&)> send_response,
    StreamResponseCallback send_streaming_response,
    std::shared_ptr<CancellationToken> cancellation_token) {
    try {
        // Extract OpenAI-style parameters
        std::string prompt = request["prompt"];
        std::string model = request.value("model", current_model_tag);
        std::string reasoning_effort = request.value("reasoning_effort", "medium");
        bool stream = request.value("stream", false);
        json options = request.value("options", json::object());

        // The is_model_supported() throw that used to sit here ran BEFORE
        // ensure_model_loaded(), so /v1/completions never reached the structured
        // model_not_found response below -- the outer catch turned an unknown model
        // into a generic server_error. ensure_model_loaded() asks the same question
        // and answers it properly.

        int length_limit = request.value("max_tokens", 4096);

         if (const ModelLoad why = ensure_model_loaded(model, request.contains("model")); why != ModelLoad::Ok) {
            send_response(openai_compat::model_error(why, model));
            return;
        }

        configure_chat_engine_parameters(options, request);

        chat_meta_info_t meta_info;
        meta_info.max_prefill_len = this->prefill_chunk_len;
        lm_uniform_input_t uniformed_input;
        header_print("OFLM", "Start generating...");

        if (stream) {
            // Create a wrapper callback that passes the pre-formatted SSE string directly
            auto openai_stream_callback = [&send_streaming_response](const std::string& data, bool is_final) {
                json data_json = data;
                send_streaming_response(data_json, is_final);
                };
            streaming_ostream_openai ostream(model, openai_stream_callback);  // streaming in completion format
            uniformed_input.prompt = prompt;
            try {
                bool success = auto_chat_engine->insert(meta_info, uniformed_input);
                if (!success) {
                    json error_response = {{"error", {
                        {"message", "the prompt does not fit this model's context window"},
                        {"type", "invalid_request_error"},
                        {"param", "messages"},
                        {"code", "context_length_exceeded"}
                    }}};
                    send_response(error_response);
                    this->auto_chat_engine->clear_context();
                    return;
                }
            } catch (const std::exception& e) {
                json error_response = {{"error", e.what()}};
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            try {
                auto_chat_engine->generate(meta_info, length_limit, ostream);
            } catch (const std::exception& e) {
                json error_response = {{"error", e.what()}};
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            ostream.finalize(meta_info);

            this->auto_chat_engine->clear_context();
        }
        else {
            std::stringstream ss;
            wstream_buf obuf(ss);
            std::ostream ostream(&obuf);
            uniformed_input.prompt = prompt;
            try {
                bool success = auto_chat_engine->insert(meta_info, uniformed_input);
                if (!success) {
                    json error_response = {{"error", {
                        {"message", "the prompt does not fit this model's context window"},
                        {"type", "invalid_request_error"},
                        {"param", "messages"},
                        {"code", "context_length_exceeded"}
                    }}};
                    send_response(error_response);
                    this->auto_chat_engine->clear_context();
                    return;
                }
            } catch (const std::exception& e) {
                json error_response = {{"error", e.what()}};
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            try {
                auto_chat_engine->generate(meta_info, length_limit, ostream);
            } catch (const std::exception& e) {
                json error_response = {{"error", e.what()}};
                send_response(error_response);
                this->auto_chat_engine->clear_context();
                return;
            }
            std::string response_text = ss.str();
            auto history = this->auto_chat_engine->get_history();

            json response = {
                {"id", "openflowlm-chat-completion"},
                {"object", "text_completion"},
                {"created", (int)std::time(nullptr)},
                {"model", model},
                {"choices", json::array({
                    {
                        {"text", response_text},
                        {"index", 0},
                        {"logprobs", nullptr},
                        {"finish_reason", openai_compat::finish_reason(meta_info.stop_reason)}
                    }
                })},
                {"usage", {
                    {"prompt_tokens", meta_info.prompt_tokens},
                    {"completion_tokens", meta_info.generated_tokens},
                    {"total_tokens", meta_info.prompt_tokens + meta_info.generated_tokens}
                }}
            };
            send_response(response);
        }
    }
    catch (const std::exception& e) {
        json error_response = {
            {"error", {
                {"message", e.what()},
                {"type", "server_error"},
                {"code", 500}
            }}
        };
        send_response(error_response);
    }
}

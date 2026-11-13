/// \file tool_messages.hpp
/// \brief Tool-call message rewriting for chat templates that carry the response on the assistant turn.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

#include "minja/chat-template.hpp"

namespace tool_messages {
using json = nlohmann::ordered_json;

///@brief Try to parse a JSON value from a string, returning the original string on failure.
inline json try_parse_json_value(const std::string& s) {
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
inline json convert_tool_responses_gemma4(json messages) {
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


///@brief OpenAI clients send tool_calls[].function.arguments as a JSON string; the Gemma 4 template raises on a string,
/// so deserialize it (what minja's polyfill_object_arguments would do, which that template must not get).
inline json deserialize_tool_arguments(json messages) {
    for (auto& message : messages) {
        if (!message.is_object() || !message.contains("tool_calls") || !message["tool_calls"].is_array()) continue;
        for (auto& tool_call : message["tool_calls"]) {
            if (!tool_call.is_object() || !tool_call.contains("function")) continue;
            auto& function = tool_call["function"];
            if (!function.is_object() || !function.contains("arguments") || !function["arguments"].is_string()) continue;
            const auto raw = function["arguments"].get<std::string>();
            if (raw.empty()) { function["arguments"] = json::object(); continue; }
            auto parsed = json::parse(raw, nullptr, /* allow_exceptions= */ false);
            if (parsed.is_object()) function["arguments"] = parsed;
        }
    }
    return messages;
}

///@brief The Gemma 4 template emits unquoted keys, which minja's capability probe does not recognize, so its polyfills
/// would move a tool call out of the model turn into a JSON blob after the response. It handles tools natively.
inline minja::chat_template_options gemma4_template_options() {
    minja::chat_template_options opt;
    opt.polyfill_tools = false;
    opt.polyfill_tool_calls = false;
    opt.polyfill_tool_responses = false;
    opt.polyfill_object_arguments = false;
    return opt;
}

}  // namespace tool_messages

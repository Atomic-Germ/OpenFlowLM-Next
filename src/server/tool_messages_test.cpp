// Traces: OPEN-SERVE-TOOL-FOLLOWUP (canonical spec: specs/open-engine/spec.md)
// A tool-call exchange, as an OpenAI client sends it back, rendered through the Gemma 4 chat template with the
// server's rewriting: the follow-up prompt must keep the assistant's call and carry the response after it.
//   cl /std:c++20 /EHsc /utf-8 /I src/include /I <nlohmann include dir> src/server/tool_messages_test.cpp
//   tool_messages_test <the model's chat_template.jinja>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "minja/chat-template.hpp"
#include "tool_messages.hpp"

using tool_messages::json;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: tool_messages_test <chat_template.jinja>\n");
        return 2;
    }
    std::ifstream f(argv[1], std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    minja::chat_template tmpl(ss.str(), "<bos>", "<eos>");

    json tools = json::array({{{"type", "function"},
                               {"function", {{"name", "get_price"}, {"description", "Unit price of an item"},
                                             {"parameters", {{"type", "object"},
                                                             {"properties", {{"item", {{"type", "string"}}}}},
                                                             {"required", json::array({"item"})}}}}}}});
    json user = {{"role", "user"}, {"content", "What does a widget cost?"}};
    json call = {{"role", "assistant"}, {"content", nullptr},
                 {"tool_calls", json::array({{{"id", "call_1"}, {"type", "function"},
                                              {"function", {{"name", "get_price"}, {"arguments", "{\"item\":\"widget\"}"}}}}})}};
    json reply = {{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "{\"price\": 18}"}};

    auto render = [&](json messages, bool convert) {
        if (convert) messages = tool_messages::convert_tool_responses_gemma4(messages);
        for (auto& m : messages)
            if (m.contains("content") && m["content"].is_null()) m["content"] = "";
        minja::chat_template_inputs in;
        const auto opts = tool_messages::gemma4_template_options();
        in.messages = tool_messages::deserialize_tool_arguments(messages);
        in.tools = tools;
        in.add_generation_prompt = true;
        in.extra_context["enable_thinking"] = false;
        return tmpl.apply(in, opts);
    };

    const std::string first = render(json::array({user}), true);
    const std::string follow = render(json::array({user, call, reply}), true);
    std::printf("---- follow-up prompt, from the model turn on:\n%s\n----\n", follow.substr(follow.rfind("<|turn>model")).c_str());

    const std::string want_call = "<|tool_call>call:get_price{item:<|\"|>widget<|\"|>}<tool_call|>";
    check(follow.find(want_call) != std::string::npos, "the follow-up prompt keeps the assistant's tool call");
    check(follow.find("<|tool_response>response:get_price{price:18}<tool_response|>") != std::string::npos,
          "and carries the tool response");
    const size_t at_call = follow.find(want_call), at_resp = follow.find("<|tool_response>");
    check(at_call != std::string::npos && at_resp != std::string::npos && at_call < at_resp, "the call comes before the response");
    check(follow.compare(0, first.size() - 0, first) == 0 || follow.rfind("<|turn>model\n") != std::string::npos,
          "the follow-up starts from the same prefix as the first prompt");
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

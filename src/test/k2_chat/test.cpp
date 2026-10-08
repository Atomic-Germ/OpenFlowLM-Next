// Traces: TOOLS-K2-HISTORY, TOOLS-K2-REASONING, TOOLS-K2-CALLS (canonical spec: specs/tool-calling/spec.md)
#include "AutoModel/k2_chat.hpp"
#include <iostream>
#include <string>

using k2_chat::json;

static int failures = 0;

#define CHECK(cond)                                                               \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond << "\n"; \
            ++failures;                                                           \
        }                                                                         \
    } while (0)

static const json TOOLS = json::parse(R"([
  {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
    "city": {"type": "string"}, "zip": {"type": "string"}, "days": {"type": "integer"},
    "tags": {"type": "array"}, "note": {"type": ["string", "null"]}}}}}
])");

static void test_history() {
    json msgs = json::parse(R"([
      {"role": "user", "content": "hi"},
      {"role": "assistant", "content": "hello"},
      {"role": "assistant", "content": "x", "reasoning_content": "kept"},
      {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
      {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "type": "function",
        "function": {"name": "get_weather", "arguments": "not json"}}]}
    ])");
    json out = k2_chat::prepare_messages(msgs);
    CHECK(!out[0].contains("reasoning_content"));
    CHECK(out[1]["reasoning_content"] == "");
    CHECK(out[2]["reasoning_content"] == "kept");
    CHECK(out[3]["tool_calls"][0]["function"]["arguments"] == json({{"city", "Paris"}}));
    CHECK(out[4]["tool_calls"][0]["function"]["arguments"] == "not json");
}

static void test_reasoning_split() {
    auto r = k2_chat::parse_response("Let me think.\n</ifm|think>\n\nAn NPU is a chip.", {});
    CHECK(r.reasoning == "Let me think.");
    CHECK(r.content == "An NPU is a chip.");
    CHECK(r.calls.empty());

    auto fast = k2_chat::parse_response("quick</ifm|think_fast>Done.", {});
    CHECK(fast.reasoning == "quick");
    CHECK(fast.content == "Done.");

    auto cut = k2_chat::parse_response("still thinking when max_tokens hit", {});
    CHECK(cut.reasoning == "still thinking when max_tokens hit");
    CHECK(cut.content.empty());

    auto blank = k2_chat::parse_response("\n</ifm|think>Answer", {});
    CHECK(blank.reasoning.empty());
    CHECK(blank.content == "Answer");
}

static void test_calls() {
    const auto types = k2_chat::param_types(TOOLS);
    CHECK(types.at("get_weather").at("note") == "string");
    const std::string text =
        "Need weather.</ifm|think>\n<ifm|tool_calls>\n<ifm|tool_call>get_weather\n"
        "<ifm|arg_key>city</ifm|arg_key>\n<ifm|arg_value>Paris</ifm|arg_value>\n"
        "<ifm|arg_key>zip</ifm|arg_key>\n<ifm|arg_value>75001</ifm|arg_value>\n"
        "<ifm|arg_key>days</ifm|arg_key>\n<ifm|arg_value>3</ifm|arg_value>\n"
        "<ifm|arg_key>tags</ifm|arg_key>\n<ifm|arg_value>[\"a\", \"b\"]</ifm|arg_value>\n"
        "</ifm|tool_call>\n<ifm|tool_call>get_time\n</ifm|tool_call>\n</ifm|tool_calls>";
    auto r = k2_chat::parse_response(text, types);
    CHECK(r.reasoning == "Need weather.");
    CHECK(r.content.empty());
    CHECK(r.calls.size() == 2);
    CHECK(r.calls[0].first == "get_weather");
    CHECK(r.calls[0].second == json({{"city", "Paris"}, {"zip", "75001"}, {"days", 3}, {"tags", {"a", "b"}}}));
    CHECK(r.calls[1].first == "get_time");
    CHECK(r.calls[1].second == json::object());

    auto unknown = k2_chat::parse_response(
        "</ifm|think><ifm|tool_call>other\n<ifm|arg_key>n</ifm|arg_key>\n<ifm|arg_value>42</ifm|arg_value>\n</ifm|tool_call>", types);
    CHECK(unknown.calls.size() == 1);
    CHECK(unknown.calls[0].second == json({{"n", 42}}));

    auto as_json = k2_chat::parse_response(
        "</ifm|think><ifm|tool_calls>\n<ifm|tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}</ifm|tool_call>\n</ifm|tool_calls>", types);
    CHECK(as_json.calls.size() == 1);
    CHECK(as_json.calls[0].first == "get_weather");
    CHECK(as_json.calls[0].second == json({{"city", "Oslo"}}));

    auto before = k2_chat::parse_response("ok</ifm|think>Checking.\n<ifm|tool_calls>\n<ifm|tool_call>get_time\n</ifm|tool_call>\n</ifm|tool_calls>", types);
    CHECK(before.content == "Checking.");
    CHECK(before.calls.size() == 1);
}

static void test_stream_matches_whole() {
    const auto types = k2_chat::param_types(TOOLS);
    const std::string text =
        "Plan it.</ifm|think>\n\nCalling <ifm|tool_calls>\n<ifm|tool_call>get_weather\n"
        "<ifm|arg_key>city</ifm|arg_key>\n<ifm|arg_value>New <York></ifm|arg_value>\n</ifm|tool_call>\n"
        "</ifm|tool_calls>";
    const auto whole = k2_chat::parse_response(text, types);
    for (size_t step : {1, 2, 3, 7}) {
        k2_chat::StreamParser p;
        p.reset();
        std::string reasoning, content;
        std::vector<std::pair<std::string, json>> calls;
        auto take = [&](const k2_chat::Event& ev) {
            if (ev.kind == k2_chat::Event::Reasoning) reasoning += ev.text;
            if (ev.kind == k2_chat::Event::Content) content += ev.text;
            if (ev.kind == k2_chat::Event::Tool) calls.emplace_back(ev.name, ev.args);
            CHECK(ev.text.find("<ifm|") == std::string::npos && ev.text.find("</ifm|") == std::string::npos);
        };
        for (size_t i = 0; i < text.size(); i += step) take(p.feed(text.substr(i, step), false, types));
        take(p.feed("", true, types));
        CHECK(k2_chat::trim(reasoning) == whole.reasoning);
        CHECK(k2_chat::trim(content) == whole.content);
        CHECK(calls == whole.calls);
    }
    CHECK(whole.calls.size() == 1);
    CHECK(whole.calls[0].second == json({{"city", "New <York>"}}));
    CHECK(whole.content == "Calling");
}

int main() {
    test_history();
    test_reasoning_split();
    test_calls();
    test_stream_matches_whole();
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "k2_chat: all checks passed\n";
    return 0;
}

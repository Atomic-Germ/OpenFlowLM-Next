// Traces: TOOLS-GEMMA4-ENVELOPE, TOOLS-GEMMA4-SCHEMA-TYPES (canonical spec: specs/tool-calling/spec.md)
// Unit tests for the shared Gemma 4 tool-call parser. Header-only, no NPU, no model.
#include "AutoModel/gemma4_tool_parser.hpp"
#include <cstdlib>
#include <iostream>
#include <string>

using gemma4_tools::json;

static int failures = 0;

#define CHECK(cond)                                                               \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond << "\n"; \
            ++failures;                                                           \
        }                                                                         \
    } while (0)

static void test_direct_call() {
    auto [name, args] = gemma4_tools::parse_tool_call("call:get_current_weather{location:<|\"|>Paris<|\"|>}");
    CHECK(name == "get_current_weather");
    CHECK(args == json({{"location", "Paris"}}));
}

static void test_no_args() {
    auto [name, args] = gemma4_tools::parse_tool_call("call:get_time{}");
    CHECK(name == "get_time");
    CHECK(args.is_object() && args.empty());
}

static void test_nested_values() {
    auto [name, args] = gemma4_tools::parse_tool_call(
        "call:create_event{attendees:[{email:<|\"|>a@x.com<|\"|>,optional:true},{email:<|\"|>b@x.com<|\"|>}],"
        "title:<|\"|>Sync<|\"|>,when:{date:<|\"|>2026-09-12<|\"|>,time:<|\"|>10:00<|\"|>},count:3}");
    CHECK(name == "create_event");
    CHECK(args["attendees"].size() == 2);
    CHECK(args["attendees"][0]["optional"] == true);
    CHECK(args["when"]["time"] == "10:00");
    CHECK(args["count"] == 3);
}

static void test_envelope_from_issue_722() {
    // the trace from ROCm/FastFlowLM#722, verbatim shape
    auto [name, args] = gemma4_tools::parse_tool_call(
        "call:tool_call{args:{query:\"тренировки на пресс или ноги\"},id:<|\"|>memory_search<|\"|>}");
    CHECK(name == "memory_search");
    CHECK(args == json({{"query", "тренировки на пресс или ноги"}}));
}

static void test_envelope_name_arguments_variant() {
    auto [name, args] = gemma4_tools::parse_tool_call(
        "call:function{name:<|\"|>lookup_item_price<|\"|>,arguments:{item:<|\"|>widget<|\"|>}}");
    CHECK(name == "lookup_item_price");
    CHECK(args == json({{"item", "widget"}}));
}

static void test_envelope_without_inner_name_is_left_alone() {
    // a real tool that happens to be called tool_call with plain args must not be rewritten
    auto [name, args] = gemma4_tools::parse_tool_call("call:tool_call{query:<|\"|>x<|\"|>}");
    CHECK(name == "tool_call");
    CHECK(args == json({{"query", "x"}}));
}

static void test_schema_type_array_becomes_nullable_scalar() {
    json tools = json::parse(R"([{"type":"function","function":{"name":"search_notes","parameters":{
        "type":"object","properties":{
            "query":{"type":"string"},
            "folder":{"type":["string","null"],"description":"Folder or null"},
            "tags":{"type":"array","items":{"type":["integer","null"]}},
            "opts":{"type":"object","properties":{"deep":{"type":["null","boolean"]}}}
        },"required":["query"]}}}])");
    json out = gemma4_tools::normalize_tools(tools);
    auto& props = out[0]["function"]["parameters"]["properties"];
    CHECK(props["query"]["type"] == "string");
    CHECK(props["folder"]["type"] == "string");
    CHECK(props["folder"]["nullable"] == true);
    CHECK(props["folder"]["description"] == "Folder or null");
    CHECK(props["tags"]["items"]["type"] == "integer");
    CHECK(props["tags"]["items"]["nullable"] == true);
    CHECK(props["opts"]["properties"]["deep"]["type"] == "boolean");
    CHECK(props["opts"]["properties"]["deep"]["nullable"] == true);
    // untouched input: scalar types and the original object are not modified
    CHECK(tools[0]["function"]["parameters"]["properties"]["folder"]["type"].is_array());
}

static void test_schema_without_arrays_is_unchanged() {
    json tools = json::parse(R"([{"type":"function","function":{"name":"f","parameters":{
        "type":"object","properties":{"a":{"type":"string","enum":["x","y"]}},"required":["a"]}}}])");
    CHECK(gemma4_tools::normalize_tools(tools) == tools);
}

int main() {
    test_direct_call();
    test_no_args();
    test_nested_values();
    test_envelope_from_issue_722();
    test_envelope_name_arguments_variant();
    test_envelope_without_inner_name_is_left_alone();
    test_schema_type_array_becomes_nullable_scalar();
    test_schema_without_arrays_is_unchanged();
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "gemma4_tool_parser: all checks passed\n";
    return EXIT_SUCCESS;
}

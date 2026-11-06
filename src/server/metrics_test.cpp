/// \file metrics_test.cpp
/// \brief Unit tests for Prometheus/OpenMetrics text exposition and OTLP/JSON serialization.
#include <cstdio>
#include <string>

#include "server/metrics.hpp"

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

static void contains(const std::string& haystack, const std::string& needle, const std::string& what) {
    ok(haystack.find(needle) != std::string::npos,
       what + (haystack.find(needle) != std::string::npos ? "" : "  (missing \"" + needle + "\")"));
}

static void test_prometheus_exposition() {
    std::printf("\n-- prometheus exposition --\n");
    oflm::metrics::Collector col;
    col.record_http_request("POST /v1/chat/completions", 200);
    col.record_http_request("POST /v1/chat/completions", 400);
    col.record_generation("qwen3:1.7b", 120, 45, 250'000'000ULL, 1'500'000'000ULL, 50'000'000ULL, 1'800'000'000ULL);
    col.record_embeddings("embed-gemma:300m", 32);

    oflm::metrics::LiveState live;
    live.active_connections = 3;
    live.active_requests = 1;
    live.npu_available = false;
    live.npu_active_requests = 1;
    live.npu_queued_requests = 2;
    live.loaded_model = "qwen3:1.7b";

    const auto snap = col.snapshot(live, 1700000000000000000ULL);
    const std::string prom = oflm::metrics::render_prometheus(snap);

    contains(prom, "# TYPE oflm_active_connections gauge\noflm_active_connections 3\n", "active connections gauge");
    contains(prom, "oflm_npu_available 0\n", "npu busy is 0");
    contains(prom, "oflm_npu_queued_requests 2\n", "npu queue gauge");
    contains(prom, "oflm_loaded_model_info{model=\"qwen3:1.7b\"} 1\n", "loaded model info metric");
    contains(prom, "oflm_http_requests_total 2\n", "total http requests counter");
    contains(prom, "oflm_http_errors_total 1\n", "total http errors counter");
    contains(prom, "oflm_tokens_prompt_total{model=\"qwen3:1.7b\"} 120\n", "prompt tokens counter");
    contains(prom, "oflm_tokens_generated_total{model=\"qwen3:1.7b\"} 45\n", "generated tokens counter");
    contains(prom, "gen_ai_client_token_usage_total{gen_ai_request_model=\"qwen3:1.7b\",gen_ai_token_type=\"input\"} 120\n",
             "OTel GenAI input tokens in Prometheus format");
    contains(prom, "gen_ai_client_token_usage_total{gen_ai_request_model=\"qwen3:1.7b\",gen_ai_token_type=\"output\"} 45\n",
             "OTel GenAI output tokens in Prometheus format");
    contains(prom, "oflm_prefill_duration_seconds_total{model=\"qwen3:1.7b\"} 0.250000\n", "prefill duration in seconds");
    contains(prom, "oflm_decoding_duration_seconds_total{model=\"qwen3:1.7b\"} 1.500000\n", "decoding duration in seconds");
    contains(prom, "oflm_embeddings_tokens_total{model=\"embed-gemma:300m\"} 32\n", "embedding tokens counter");
}

static void test_otlp_json() {
    std::printf("\n-- otlp json --\n");
    oflm::metrics::Collector col;
    col.record_http_request("POST /v1/chat/completions", 200);
    col.record_generation("llama3.2:1b", 64, 128, 100'000'000ULL, 800'000'000ULL, 0ULL, 900'000'000ULL);

    oflm::metrics::LiveState live;
    live.active_connections = 1;
    live.npu_available = true;
    live.loaded_model = "llama3.2:1b";

    const auto snap = col.snapshot(live, 1700000000000000000ULL);
    const auto j = oflm::metrics::render_otlp_json(snap);

    ok(j.contains("resourceMetrics") && j["resourceMetrics"].is_array() && j["resourceMetrics"].size() == 1,
       "top-level ExportMetricsServiceRequest has resourceMetrics[1]");
    const auto& rm = j["resourceMetrics"][0];
    ok(rm.contains("scopeMetrics") && rm["scopeMetrics"].is_array() && rm["scopeMetrics"].size() == 1,
       "resourceMetrics[0] has scopeMetrics[1]");
    const auto& metrics = rm["scopeMetrics"][0]["metrics"];
    ok(metrics.is_array() && !metrics.empty(), "scopeMetrics[0].metrics is non-empty array");

    bool found_token_usage = false;
    bool found_npu_avail = false;
    for (const auto& m : metrics) {
        if (m["name"] == "gen_ai.client.token.usage") {
            found_token_usage = true;
            ok(m.contains("sum") && m["sum"]["aggregationTemporality"] == 2 && m["sum"]["isMonotonic"] == true,
               "gen_ai.client.token.usage is cumulative monotonic sum");
            ok(m["sum"]["dataPoints"].size() == 2, "gen_ai.client.token.usage has input + output dataPoints");
        }
        if (m["name"] == "oflm.npu.available") {
            found_npu_avail = true;
            ok(m["gauge"]["dataPoints"][0]["asInt"] == "1", "oflm.npu.available gauge is \"1\"");
        }
    }
    ok(found_token_usage, "OTLP JSON includes gen_ai.client.token.usage");
    ok(found_npu_avail, "OTLP JSON includes oflm.npu.available");
}

static void test_content_negotiation() {
    std::printf("\n-- content negotiation --\n");
    using oflm::metrics::wants_otlp_json;
    ok(!wants_otlp_json("/metrics", "*/*"), "GET /metrics defaults to Prometheus text");
    ok(!wants_otlp_json("/metrics", "text/plain; version=0.0.4"), "GET /metrics with text/plain is Prometheus text");
    ok(wants_otlp_json("/metrics", "application/json"), "GET /metrics with Accept: application/json is OTLP JSON");
    ok(wants_otlp_json("/metrics?format=otlp", "*/*"), "GET /metrics?format=otlp is OTLP JSON");
    ok(wants_otlp_json("/api/metrics", "*/*"), "GET /api/metrics defaults to OTLP JSON");
    ok(wants_otlp_json("/v1/metrics", "*/*"), "GET /v1/metrics defaults to OTLP JSON");
    ok(!wants_otlp_json("/api/metrics?format=prometheus", "application/json"),
       "query ?format=prometheus overrides path and Accept");
}

int main() {
    test_prometheus_exposition();
    test_otlp_json();
    test_content_negotiation();
    std::printf("\n%d/%d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}

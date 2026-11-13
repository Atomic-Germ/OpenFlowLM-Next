/// \file metrics_test.cpp
/// \brief Unit tests for Prometheus text exposition and the JSON summary.
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
    contains(prom, "oflm_prefill_duration_seconds_total{model=\"qwen3:1.7b\"} 0.250000\n", "prefill duration in seconds");
    contains(prom, "oflm_decoding_duration_seconds_total{model=\"qwen3:1.7b\"} 1.500000\n", "decoding duration in seconds");
    contains(prom, "oflm_total_duration_seconds_total{model=\"qwen3:1.7b\"} 1.800000\n", "total duration in seconds");
    contains(prom, "oflm_embeddings_tokens_total{model=\"embed-gemma:300m\"} 32\n", "embedding tokens counter");
}

static void test_json_summary() {
    std::printf("\n-- json summary --\n");
    oflm::metrics::Collector col;
    col.record_http_request("POST /v1/chat/completions", 200);
    col.record_http_request("unmatched", 404);
    col.record_generation("llama3.2:1b", 64, 128, 100'000'000ULL, 800'000'000ULL, 0ULL, 900'000'000ULL);

    oflm::metrics::LiveState live;
    live.active_connections = 1;
    live.npu_available = true;
    live.loaded_model = "llama3.2:1b";

    const auto snap = col.snapshot(live, 1700000000000000000ULL);
    const auto j = oflm::metrics::render_json_summary(snap);

    ok(j.contains("oflm") && j["oflm"].is_object(), "top-level oflm object exists");
    ok(!j.contains("resourceMetrics"), "not an OTLP payload");
    const auto& summary = j["oflm"];
    ok(summary["http_requests_total"] == 2, "http_requests_total is 2");
    ok(summary["http_errors_total"] == 1, "http_errors_total counts the 404");
    ok(summary["requests_by_route"]["unmatched"] == 1, "unmatched paths are bucketed");
    ok(summary["live"]["npu_available"] == true, "npu_available is true");
    ok(summary["live"]["loaded_model"] == "llama3.2:1b", "loaded model is reported");
    const auto& model = summary["models"]["llama3.2:1b"];
    ok(model["generation_requests_total"] == 1, "generation requests counter");
    ok(model["tokens_prompt_total"] == 64, "prompt tokens counter");
    ok(model["total_duration_seconds_total"] == 0.9, "total duration rendered in seconds");
}

static void test_content_negotiation() {
    std::printf("\n-- content negotiation --\n");
    using oflm::metrics::wants_json;
    ok(!wants_json("/metrics", "*/*"), "GET /metrics defaults to Prometheus text");
    ok(!wants_json("/metrics", "text/plain; version=0.0.4"), "GET /metrics with text/plain is Prometheus text");
    ok(wants_json("/metrics", "application/json"), "GET /metrics with Accept: application/json is the JSON summary");
    ok(wants_json("/metrics?format=json", "*/*"), "GET /metrics?format=json is the JSON summary");
    ok(wants_json("/api/metrics", "*/*"), "GET /api/metrics defaults to the JSON summary");
    ok(!wants_json("/api/metrics?format=prometheus", "application/json"),
       "query ?format=prometheus overrides path and Accept");
    ok(!wants_json("/metrics?format=prometheus", "application/json"),
       "query ?format=prometheus overrides Accept on /metrics");
}

int main() {
    test_prometheus_exposition();
    test_json_summary();
    test_content_negotiation();
    std::printf("\n%d/%d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}

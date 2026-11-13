/*!
 * \file metrics.hpp
 * \brief Prometheus text exposition and plain JSON summary for OpenFlowLM
 */
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <string_view>

namespace oflm::metrics {

using json = nlohmann::ordered_json;

/// Per-model cumulative LLM / embedding counters.
struct ModelStats {
    uint64_t generation_requests = 0;
    uint64_t prompt_tokens = 0;
    uint64_t completion_tokens = 0;
    uint64_t prefill_duration_ns = 0;
    uint64_t decoding_duration_ns = 0;
    uint64_t load_duration_ns = 0;
    uint64_t total_duration_ns = 0;
    uint64_t embedding_requests = 0;
    uint64_t embedding_tokens = 0;
};

/// Live server state supplied when a scrape occurs.
struct LiveState {
    size_t active_connections = 0;
    size_t active_requests = 0;
    bool npu_available = true;
    int npu_active_requests = 0;
    size_t npu_queued_requests = 0;
    std::string loaded_model;
};

/// Point-in-time snapshot used by the Prometheus and JSON renderers.
struct Snapshot {
    uint64_t start_time_unix_nano = 0;
    uint64_t now_unix_nano = 0;
    uint64_t http_requests_total = 0;
    uint64_t http_errors_total = 0;
    LiveState live;
    std::map<std::string, int64_t> requests_by_route;
    std::map<std::string, ModelStats> by_model;
};

inline uint64_t current_unix_nano() {
    auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

/// Thread-safe metrics collector.
class Collector {
public:
    Collector() : start_time_unix_nano_(current_unix_nano()) {}

    static Collector& instance() {
        static Collector inst;
        return inst;
    }

    void record_http_request(const std::string& method_and_path, int status_code) {
        http_requests_total_.fetch_add(1, std::memory_order_relaxed);
        if (status_code >= 400) {
            http_errors_total_.fetch_add(1, std::memory_order_relaxed);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        requests_by_route_[method_and_path] += 1;
    }

    void record_generation(const std::string& model,
                           uint64_t prompt_tokens,
                           uint64_t completion_tokens,
                           uint64_t prefill_duration_ns,
                           uint64_t decoding_duration_ns,
                           uint64_t load_duration_ns,
                           uint64_t total_duration_ns) {
        std::string key = model.empty() ? "unknown" : model;
        std::lock_guard<std::mutex> lock(mutex_);
        auto& s = by_model_[key];
        s.generation_requests += 1;
        s.prompt_tokens += prompt_tokens;
        s.completion_tokens += completion_tokens;
        s.prefill_duration_ns += prefill_duration_ns;
        s.decoding_duration_ns += decoding_duration_ns;
        s.load_duration_ns += load_duration_ns;
        s.total_duration_ns += total_duration_ns;
    }

    void record_embeddings(const std::string& model, uint64_t prompt_tokens) {
        std::string key = model.empty() ? "unknown" : model;
        std::lock_guard<std::mutex> lock(mutex_);
        auto& s = by_model_[key];
        s.embedding_requests += 1;
        s.embedding_tokens += prompt_tokens;
    }

    Snapshot snapshot(const LiveState& live, uint64_t now_nano = 0) const {
        Snapshot snap;
        snap.start_time_unix_nano = start_time_unix_nano_;
        snap.now_unix_nano = now_nano ? now_nano : current_unix_nano();
        snap.http_requests_total = http_requests_total_.load(std::memory_order_relaxed);
        snap.http_errors_total = http_errors_total_.load(std::memory_order_relaxed);
        snap.live = live;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            snap.requests_by_route = requests_by_route_;
            snap.by_model = by_model_;
        }
        return snap;
    }

private:
    uint64_t start_time_unix_nano_;
    std::atomic<uint64_t> http_requests_total_{0};
    std::atomic<uint64_t> http_errors_total_{0};
    mutable std::mutex mutex_;
    std::map<std::string, int64_t> requests_by_route_;
    std::map<std::string, ModelStats> by_model_;
};

/// Escape a Prometheus label value per the exposition specification (`\`, `"`, `\n`).
inline std::string escape_prometheus_label(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}

/// Format a Snapshot in Prometheus text exposition format (0.0.4).
/// Compatible with Prometheus and the OpenTelemetry Collector `prometheus` receiver.
inline std::string render_prometheus(const Snapshot& snap) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);

    out << "# HELP oflm_active_connections Currently active HTTP connections.\n"
        << "# TYPE oflm_active_connections gauge\n"
        << "oflm_active_connections " << snap.live.active_connections << "\n\n";

    out << "# HELP oflm_active_requests Currently executing HTTP requests.\n"
        << "# TYPE oflm_active_requests gauge\n"
        << "oflm_active_requests " << snap.live.active_requests << "\n\n";

    out << "# HELP oflm_npu_available Whether the NPU is currently available (1 = available, 0 = busy).\n"
        << "# TYPE oflm_npu_available gauge\n"
        << "oflm_npu_available " << (snap.live.npu_available ? 1 : 0) << "\n\n";

    out << "# HELP oflm_npu_active_requests Requests currently holding the NPU lock.\n"
        << "# TYPE oflm_npu_active_requests gauge\n"
        << "oflm_npu_active_requests " << snap.live.npu_active_requests << "\n\n";

    out << "# HELP oflm_npu_queued_requests Requests waiting in the NPU queue.\n"
        << "# TYPE oflm_npu_queued_requests gauge\n"
        << "oflm_npu_queued_requests " << snap.live.npu_queued_requests << "\n\n";

    out << "# HELP oflm_loaded_model_info Currently resident chat model on the NPU (1 = loaded).\n"
        << "# TYPE oflm_loaded_model_info gauge\n";
    if (!snap.live.loaded_model.empty()) {
        out << "oflm_loaded_model_info{model=\""
            << escape_prometheus_label(snap.live.loaded_model) << "\"} 1\n";
    }
    out << "\n";

    out << "# HELP oflm_http_requests_total Total HTTP requests processed by the server.\n"
        << "# TYPE oflm_http_requests_total counter\n"
        << "oflm_http_requests_total " << snap.http_requests_total << "\n\n";

    out << "# HELP oflm_http_errors_total Total HTTP requests resulting in status >= 400.\n"
        << "# TYPE oflm_http_errors_total counter\n"
        << "oflm_http_errors_total " << snap.http_errors_total << "\n\n";

    if (!snap.requests_by_route.empty()) {
        out << "# HELP oflm_http_route_requests_total Total HTTP requests by route.\n"
            << "# TYPE oflm_http_route_requests_total counter\n";
        for (const auto& [route, count] : snap.requests_by_route) {
            out << "oflm_http_route_requests_total{route=\""
                << escape_prometheus_label(route) << "\"} " << count << "\n";
        }
        out << "\n";
    }

    out << "# HELP oflm_generation_requests_total Total LLM generation requests completed.\n"
        << "# TYPE oflm_generation_requests_total counter\n";
    for (const auto& [model, s] : snap.by_model) {
        if (s.generation_requests > 0) {
            out << "oflm_generation_requests_total{model=\""
                << escape_prometheus_label(model) << "\"} " << s.generation_requests << "\n";
        }
    }
    out << "\n";

    out << "# HELP oflm_tokens_prompt_total Total prompt (input) tokens processed.\n"
        << "# TYPE oflm_tokens_prompt_total counter\n";
    for (const auto& [model, s] : snap.by_model) {
        if (s.generation_requests > 0) {
            out << "oflm_tokens_prompt_total{model=\""
                << escape_prometheus_label(model) << "\"} " << s.prompt_tokens << "\n";
        }
    }
    out << "\n";

    out << "# HELP oflm_tokens_generated_total Total completion (output) tokens generated.\n"
        << "# TYPE oflm_tokens_generated_total counter\n";
    for (const auto& [model, s] : snap.by_model) {
        if (s.generation_requests > 0) {
            out << "oflm_tokens_generated_total{model=\""
                << escape_prometheus_label(model) << "\"} " << s.completion_tokens << "\n";
        }
    }
    out << "\n";

    out << "# HELP oflm_prefill_duration_seconds_total Cumulative prompt prefill duration in seconds.\n"
        << "# TYPE oflm_prefill_duration_seconds_total counter\n";
    for (const auto& [model, s] : snap.by_model) {
        if (s.generation_requests > 0) {
            out << "oflm_prefill_duration_seconds_total{model=\""
                << escape_prometheus_label(model) << "\"} "
                << (static_cast<double>(s.prefill_duration_ns) / 1e9) << "\n";
        }
    }
    out << "\n";

    out << "# HELP oflm_decoding_duration_seconds_total Cumulative token decoding duration in seconds.\n"
        << "# TYPE oflm_decoding_duration_seconds_total counter\n";
    for (const auto& [model, s] : snap.by_model) {
        if (s.generation_requests > 0) {
            out << "oflm_decoding_duration_seconds_total{model=\""
                << escape_prometheus_label(model) << "\"} "
                << (static_cast<double>(s.decoding_duration_ns) / 1e9) << "\n";
        }
    }
    out << "\n";

    out << "# HELP oflm_load_duration_seconds_total Cumulative model load duration in seconds.\n"
        << "# TYPE oflm_load_duration_seconds_total counter\n";
    for (const auto& [model, s] : snap.by_model) {
        if (s.generation_requests > 0) {
            out << "oflm_load_duration_seconds_total{model=\""
                << escape_prometheus_label(model) << "\"} "
                << (static_cast<double>(s.load_duration_ns) / 1e9) << "\n";
        }
    }
    out << "\n";

    out << "# HELP oflm_total_duration_seconds_total Cumulative end-to-end generation duration in seconds.\n"
        << "# TYPE oflm_total_duration_seconds_total counter\n";
    for (const auto& [model, s] : snap.by_model) {
        if (s.generation_requests > 0) {
            out << "oflm_total_duration_seconds_total{model=\""
                << escape_prometheus_label(model) << "\"} "
                << (static_cast<double>(s.total_duration_ns) / 1e9) << "\n";
        }
    }
    out << "\n";

    out << "# HELP oflm_embeddings_requests_total Total embedding requests completed.\n"
        << "# TYPE oflm_embeddings_requests_total counter\n";
    for (const auto& [model, s] : snap.by_model) {
        if (s.embedding_requests > 0) {
            out << "oflm_embeddings_requests_total{model=\""
                << escape_prometheus_label(model) << "\"} " << s.embedding_requests << "\n";
        }
    }
    out << "\n";

    out << "# HELP oflm_embeddings_tokens_total Total tokens processed by embedding requests.\n"
        << "# TYPE oflm_embeddings_tokens_total counter\n";
    for (const auto& [model, s] : snap.by_model) {
        if (s.embedding_requests > 0) {
            out << "oflm_embeddings_tokens_total{model=\""
                << escape_prometheus_label(model) << "\"} " << s.embedding_tokens << "\n";
        }
    }

    return out.str();
}

/// Format a Snapshot as a plain JSON summary of the oflm.* instruments.
/// This is not an OTLP payload: OTLP is a push protocol, so nothing scrapes it
/// over a GET. Scrapers that speak OTLP should use the Prometheus endpoint with
/// an OpenTelemetry Collector `prometheus` receiver.
inline json render_json_summary(const Snapshot& snap) {
    auto by_model = json::object();
    for (const auto& [model, s] : snap.by_model) {
        by_model[model] = json{
            {"generation_requests_total", s.generation_requests},
            {"tokens_prompt_total", s.prompt_tokens},
            {"tokens_generated_total", s.completion_tokens},
            {"prefill_duration_seconds_total", static_cast<double>(s.prefill_duration_ns) / 1e9},
            {"decoding_duration_seconds_total", static_cast<double>(s.decoding_duration_ns) / 1e9},
            {"load_duration_seconds_total", static_cast<double>(s.load_duration_ns) / 1e9},
            {"total_duration_seconds_total", static_cast<double>(s.total_duration_ns) / 1e9},
            {"embeddings_requests_total", s.embedding_requests},
            {"embeddings_tokens_total", s.embedding_tokens},
        };
    }
    auto by_route = json::object();
    for (const auto& [route, count] : snap.requests_by_route) {
        by_route[route] = count;
    }
    json live = {
        {"active_connections", snap.live.active_connections},
        {"active_requests", snap.live.active_requests},
        {"npu_available", snap.live.npu_available},
        {"npu_active_requests", snap.live.npu_active_requests},
        {"npu_queued_requests", snap.live.npu_queued_requests},
    };
    if (!snap.live.loaded_model.empty()) {
        live["loaded_model"] = snap.live.loaded_model;
    }
    return json{
        {"oflm", json{
            {"uptime_seconds", static_cast<double>(snap.now_unix_nano - snap.start_time_unix_nano) / 1e9},
            {"http_requests_total", snap.http_requests_total},
            {"http_errors_total", snap.http_errors_total},
            {"requests_by_route", std::move(by_route)},
            {"live", std::move(live)},
            {"models", std::move(by_model)},
        }},
    };
}

/// Decide whether the caller asked for the JSON summary instead of Prometheus text.
/// - `/metrics` defaults to Prometheus text (`text/plain; version=0.0.4`), unless
///   `?format=json` is in the query string or `Accept: application/json`
///   is sent without `text/plain`.
/// - `/api/metrics` defaults to the JSON summary unless `?format=prometheus`
///   is in the query string.
inline bool wants_json(std::string_view target, std::string_view accept_header) {
    auto qpos = target.find('?');
    std::string_view path = (qpos == std::string_view::npos) ? target : target.substr(0, qpos);
    std::string_view query = (qpos == std::string_view::npos) ? std::string_view{} : target.substr(qpos + 1);

    if (query.find("format=prometheus") != std::string_view::npos ||
        query.find("format=prom") != std::string_view::npos) {
        return false;
    }
    if (query.find("format=json") != std::string_view::npos) {
        return true;
    }
    if (path == "/api/metrics") {
        return true;
    }
    if (accept_header.find("application/json") != std::string_view::npos &&
        accept_header.find("text/plain") == std::string_view::npos) {
        return true;
    }
    return false;
}

} // namespace oflm::metrics

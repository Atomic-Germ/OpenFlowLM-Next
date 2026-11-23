// Scoped, host-only fold evidence. An incomplete execution emits no successful row.
#pragma once

#include <cstdio>
#include <cstring>
#include <exception>
#include <optional>
#include <stdexcept>

#include "open_qwen36/prefill_precision.hpp"

namespace open_qwen36 {

class FoldProfile {
    struct Record {
        const char* phase;
        size_t real_tokens, physical_tokens;
        prefill_precision::FoldPhaseTiming timing;
    };
    FILE* sink_ = nullptr;
    bool first_ = true;
    std::optional<Record> record_;

    void end(bool complete) noexcept {
        if (complete) {
            if (first_) {
                first_ = false;
                std::fprintf(sink_, "{\"record\":\"header\",\"version\":1}\n");
            }
            const auto& r = *record_;
            const auto& t = r.timing;
            std::fprintf(sink_,
                "{\"record\":\"fold\",\"what\":\"%s\",\"real_tokens\":%zu,"
                "\"block_tokens\":%zu,\"calls\":%lld,\"split_ms\":%.3f,"
                "\"copy_ms\":%.3f,\"accumulate_ms\":%.3f,\"total_ms\":%.3f}\n",
                r.phase, r.real_tokens, r.physical_tokens, t.calls,
                t.split_ms, t.copy_ms, t.accumulate_ms,
                t.split_ms + t.copy_ms + t.accumulate_ms);
            std::fflush(sink_);
        }
        record_.reset();
    }

public:
    void attach(FILE* sink) {
        if (record_) throw std::logic_error("fold profile: cannot replace an active sink");
        sink_ = sink;
        first_ = true;
    }

    prefill_precision::FoldPhaseTiming* timing() noexcept {
        return record_ ? &record_->timing : nullptr;
    }

    class Scope {
        FoldProfile* owner_ = nullptr;
        bool* request_active_ = nullptr;
        int exceptions_ = 0;
        bool complete_ = false;

    public:
        Scope(FoldProfile& owner, const char* phase, size_t real, size_t physical,
              bool* request_active = nullptr) : request_active_(request_active) {
            if (!owner.sink_) return;
            if (owner.record_) throw std::logic_error("fold profile: nested execution unit");
            if (!phase || (std::strcmp(phase, "gemm-block") && std::strcmp(phase, "gemm-prompt")) ||
                !real || real > physical)
                throw std::invalid_argument("fold profile: invalid phase/token metadata");
            owner.record_.emplace(Record{phase, real, physical, {}});
            owner_ = &owner;
            exceptions_ = std::uncaught_exceptions();
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        ~Scope() {
            if (owner_) owner_->end(complete_ && std::uncaught_exceptions() == exceptions_);
            // The request profiler must not remain armed after a failed layer/tail.
            if (request_active_ && *request_active_) *request_active_ = false;
        }
        void complete() noexcept { complete_ = true; }
    };
};

}  // namespace open_qwen36

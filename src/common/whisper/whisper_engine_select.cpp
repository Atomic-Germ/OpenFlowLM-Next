/// \file whisper_engine_select.cpp
/// \brief The whisper_engine factory: picks open vs. closed and builds it (phase 3b, issue #72)
/// \note This is the ONLY place that decision is made. Both engines return a plausible
///       transcript for any input, so a wrong-but-silent choice here is invisible at the
///       API boundary -- every branch below logs which rule fired, and every refusal
///       names the file it could not find rather than falling back to the other engine.
#include "whisper/whisper_engine.hpp"
#include "utils/utils.hpp"

#ifdef OFLM_USE_OPEN_WHISPER
#include "open_whisper/engine_adapter.hpp"
#endif

#include <filesystem>
#include <stdexcept>

/// \brief Defined in whisper_engine_closed.cpp -- the only caller of the closed engine's
///        constructor now that make_whisper_engine() lives here instead.
std::unique_ptr<whisper_engine> make_closed_whisper_engine(const std::string& model_path,
                                                           Whisper_Config& config,
                                                           oflm_rt::device* device,
                                                           bool enable_preemption);

namespace {

bool has_file(const std::string& dir, const char* name) {
    std::error_code ec;
    return std::filesystem::is_regular_file(std::filesystem::path(dir) / name, ec);
}

#ifdef OFLM_USE_OPEN_WHISPER
/// \brief Mirrors ow::KernelSet::resolve_dir's own search order (OFLM_WHISPER_KERNELS_DIR,
///        else <model_dir>/open_kernels) WITHOUT opening a device or the kernel set's own
///        JSON, so auto-selection can ask "would an open engine even find one?" before
///        committing to either engine. engine_adapter.cpp calls the real resolve_dir right
///        afterwards, which is what actually validates it -- this is only a existence probe.
bool open_kernels_resolve(const std::string& model_dir) {
    const std::string hint = utils::getenv_oflm("OFLM_WHISPER_KERNELS_DIR");
    std::error_code ec;
    if (!hint.empty()) return std::filesystem::is_directory(hint, ec);
    return std::filesystem::is_directory(std::filesystem::path(model_dir) / "open_kernels", ec);
}

std::unique_ptr<whisper_engine> make_open_engine(const std::string& model_path, Whisper_Config& config) {
    // Whisper_Config::from_pretrained already rounded vocab_size up to a multiple of 32
    // (see lm_config.hpp) by the time this runs -- Whisper::load_model calls
    // from_pretrained() before make_whisper_engine(). Read it rather than hardcoding it:
    // OpenWhisperEngine's constructor checks it against the decoder's own compile-time
    // constant instead of assuming the two agree.
    const int64_t vocab_padded = static_cast<int64_t>(config.get<u32>("vocab_size"));
    return std::make_unique<open_whisper::OpenWhisperEngine>(model_path, "", vocab_padded);
}
#endif

} // namespace

std::unique_ptr<whisper_engine> make_whisper_engine(const std::string& model_path,
                                                    Whisper_Config& config,
                                                    oflm_rt::device* device,
                                                    bool enable_preemption) {
    const std::string want = utils::getenv_oflm("OFLM_WHISPER_ENGINE");

    if (want == "closed") {
        if (!has_file(model_path, "model.q4nx")) {
            throw std::runtime_error("OFLM_WHISPER_ENGINE=closed: " + model_path +
                                     "/model.q4nx not found");
        }
        header_print("OFLM", "Whisper engine: closed (OFLM_WHISPER_ENGINE=closed)");
        return make_closed_whisper_engine(model_path, config, device, enable_preemption);
    }

#ifdef OFLM_USE_OPEN_WHISPER
    if (want == "open") {
        if (!has_file(model_path, "model.open.safetensors")) {
            throw std::runtime_error("OFLM_WHISPER_ENGINE=open: " + model_path +
                                     "/model.open.safetensors not found");
        }
        if (!open_kernels_resolve(model_path)) {
            throw std::runtime_error(
                "OFLM_WHISPER_ENGINE=open: no whisper_gemm kernel set found (set "
                "OFLM_WHISPER_KERNELS_DIR, or place one at " + model_path + "/open_kernels)");
        }
        header_print("OFLM", "Whisper engine: open (OFLM_WHISPER_ENGINE=open)");
        return make_open_engine(model_path, config);
    }
    if (!want.empty()) {
        throw std::runtime_error("OFLM_WHISPER_ENGINE=" + want +
                                 ": expected 'open' or 'closed' (this build has both)");
    }

    // Unset: prefer open when ITS OWN container and a kernel set are both present; fall
    // back to closed when its weights are there; otherwise refuse, naming both missing
    // paths, rather than silently choosing whichever engine happens to construct without
    // throwing (the "fails open" class this project keeps finding -- see CLAUDE.md rule 8).
    const bool open_ready = has_file(model_path, "model.open.safetensors") &&
                            open_kernels_resolve(model_path);
    const bool closed_ready = has_file(model_path, "model.q4nx");
    if (open_ready) {
        header_print("OFLM", "Whisper engine: open (model.open.safetensors + a kernel set "
                             "found, OFLM_WHISPER_ENGINE unset)");
        return make_open_engine(model_path, config);
    }
    if (closed_ready) {
        header_print("OFLM", "Whisper engine: closed (model.q4nx found, OFLM_WHISPER_ENGINE "
                             "unset)");
        return make_closed_whisper_engine(model_path, config, device, enable_preemption);
    }
    throw std::runtime_error(
        "no usable Whisper weights in " + model_path + ": neither model.open.safetensors "
        "(with a kernel set -- OFLM_WHISPER_KERNELS_DIR or <model>/open_kernels) nor "
        "model.q4nx was found");
#else
    // HRX builds (and any build without open_whisper's sources) compile only this branch --
    // OFLM_USE_OPEN_WHISPER is off, so there is no open_whisper::OpenWhisperEngine to name.
    if (want == "open") {
        throw std::runtime_error("OFLM_WHISPER_ENGINE=open: this build has no open Whisper "
                                 "engine (built with OFLM_USE_HRX, or open_whisper's sources "
                                 "were not compiled in)");
    }
    if (!want.empty()) {
        throw std::runtime_error("OFLM_WHISPER_ENGINE=" + want +
                                 ": this build has only the closed Whisper engine");
    }
    if (!has_file(model_path, "model.q4nx")) {
        throw std::runtime_error("no usable Whisper weights in " + model_path +
                                 ": model.q4nx not found (this build has no open Whisper engine)");
    }
    header_print("OFLM", "Whisper engine: closed (only engine in this build)");
    return make_closed_whisper_engine(model_path, config, device, enable_preemption);
#endif
}

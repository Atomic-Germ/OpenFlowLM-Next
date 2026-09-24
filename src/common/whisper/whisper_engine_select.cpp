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

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

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
/// \brief The exporter's own format tag (open_kernels/export_whisper_kernels.py's `FORMAT`),
///        checked below so auto-discovery never accepts a whisper_kernels.json some future,
///        differently-shaped exporter wrote.
constexpr const char* kOpenKernelsFormat = "oflm-open-whisper-kernels-v1";

/// \brief Where the open engine's kernel set is, or "" -- the same search order the other
///        open engines use (open_qwen36/engine.cpp's find_kernels): OFLM_WHISPER_KERNELS_DIR,
///        then <model_dir>/open_kernels, then <root>/xclbins/<model_name>/open_kernels for
///        every xclbins root (the build tree's xclbins junction, the install tree, oflm-add's
///        user roots). The last is where export_whisper_kernels.py writes by default, so a
///        build of this tree finds its own kernels with no configuration. A candidate counts
///        only if whisper_kernels.json PARSES, names this exporter's own format, and reads
///        `"complete": true` (PR #111 review): an `--only` export -- a subset of the seven GEMM
///        streams, built for testing one stream in isolation -- writes a whisper_kernels.json
///        that exists and is valid JSON but is deliberately incomplete, and auto-discovery
///        picking it up would hand the engine a kernel set missing streams it needs. A
///        candidate that fails this check is skipped, NOT refused -- the next candidate in the
///        search order still gets a chance, since this is discovery, not validation of an
///        operator-named location (that is OFLM_WHISPER_KERNELS_DIR below, which is always used
///        as given and left for the engine's own construction to refuse).
/// \note *how names the rule that chose it, because every rule yields a working engine and a
///       set chosen against intent looks right.
std::string find_open_kernels(const std::string& model_dir, const Whisper_Config& config, std::string* how) {
    namespace fs = std::filesystem;
    auto ok = [&](const fs::path& d) {
        std::ifstream f(d / "whisper_kernels.json", std::ios::binary);
        if (!f) return false;
        try {
            nlohmann::json j;
            f >> j;
            return j.value("complete", false) && j.value("format", std::string()) == kOpenKernelsFormat;
        } catch (const nlohmann::json::exception&) {
            return false;
        }
    };
    const std::string env = utils::getenv_oflm("OFLM_WHISPER_KERNELS_DIR");
    if (!env.empty()) {
        // An explicit location is used as given (the engine then validates it and refuses a
        // wrong one), never silently replaced by a set found somewhere else -- including when
        // it is incomplete: an operator who names a directory explicitly gets ITS refusal
        // (e.g. from the streams the container actually needs), not a silent skip to whatever
        // the search order would otherwise have picked.
        *how = "OFLM_WHISPER_KERNELS_DIR";
        return env;
    }
    const fs::path local = fs::path(model_dir) / "open_kernels";
    if (ok(local)) { *how = "beside the model"; return local.string(); }
    std::vector<std::string> roots = utils::xclbin_roots();
    if (!config.exec_path.empty() && std::find(roots.begin(), roots.end(), config.exec_path) == roots.end())
        roots.push_back(config.exec_path);
    for (const std::string& r : roots) {
        const fs::path cand = fs::path(r) / "xclbins" / config.model_name / "open_kernels";
        if (ok(cand)) { *how = "an xclbins root"; return cand.string(); }
    }
    return {};
}

std::unique_ptr<whisper_engine> make_open_engine(const std::string& model_path, const std::string& kernels_dir,
                                                 Whisper_Config& config) {
    // Whisper_Config::from_pretrained already rounded vocab_size up to a multiple of 32
    // (see lm_config.hpp) by the time this runs -- Whisper::load_model calls
    // from_pretrained() before make_whisper_engine(). Read it rather than hardcoding it:
    // OpenWhisperEngine's constructor checks it against the decoder's own compile-time
    // constant instead of assuming the two agree.
    const int64_t vocab_padded = static_cast<int64_t>(config.get<u32>("vocab_size"));
    return std::make_unique<open_whisper::OpenWhisperEngine>(model_path, kernels_dir, vocab_padded);
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
        std::string how;
        const std::string kdir = find_open_kernels(model_path, config, &how);
        if (kdir.empty()) {
            throw std::runtime_error(
                "OFLM_WHISPER_ENGINE=open: no Whisper kernel set found (build one with "
                "open_kernels/export_whisper_kernels.py, which writes xclbins/" + config.model_name +
                "/open_kernels; or set OFLM_WHISPER_KERNELS_DIR; or place one at " + model_path +
                "/open_kernels)");
        }
        header_print("OFLM", "Whisper engine: open (OFLM_WHISPER_ENGINE=open), kernels " << kdir << " (" << how << ")");
        return make_open_engine(model_path, kdir, config);
    }
    if (!want.empty()) {
        throw std::runtime_error("OFLM_WHISPER_ENGINE=" + want +
                                 ": expected 'open' or 'closed' (this build has both)");
    }

    // Unset: prefer open when ITS OWN container and a kernel set are both present; fall
    // back to closed when its weights are there; otherwise refuse, naming both missing
    // paths, rather than silently choosing whichever engine happens to construct without
    // throwing (the "fails open" class this project keeps finding -- see CLAUDE.md rule 8).
    std::string how;
    const std::string kdir = has_file(model_path, "model.open.safetensors")
                                 ? find_open_kernels(model_path, config, &how) : std::string();
    const bool open_ready = !kdir.empty();
    const bool closed_ready = has_file(model_path, "model.q4nx");
    if (open_ready) {
        header_print("OFLM", "Whisper engine: open (model.open.safetensors + a kernel set "
                             "found, OFLM_WHISPER_ENGINE unset), kernels " << kdir << " (" << how << ")");
        return make_open_engine(model_path, kdir, config);
    }
    if (closed_ready) {
        header_print("OFLM", "Whisper engine: closed (model.q4nx found, OFLM_WHISPER_ENGINE "
                             "unset)");
        return make_closed_whisper_engine(model_path, config, device, enable_preemption);
    }
    throw std::runtime_error(
        "no usable Whisper weights in " + model_path + ": neither model.open.safetensors "
        "(with a kernel set -- xclbins/<model>/open_kernels, <model>/open_kernels or "
        "OFLM_WHISPER_KERNELS_DIR) nor "
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

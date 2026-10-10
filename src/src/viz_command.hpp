/// `oflm viz [<tag>]`: the flow-viz page for a model's open kernel set (specs/viz, VIZ-COMMAND).
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#else
#include <spawn.h>
#include <sys/wait.h>
extern char** environ;
#endif

#include "model_list.hpp"
#include "nlohmann/json.hpp"
#include "utils/utils.hpp"
#include "viz_render.hpp"

namespace viz_assets {
extern const unsigned char page[];
extern const size_t page_size;
extern const unsigned char fonts[];
extern const size_t fonts_size;
extern const unsigned char explain[];
extern const size_t explain_size;
}  // namespace viz_assets

namespace viz_command {
namespace fs = std::filesystem;

inline std::string asset(const unsigned char* p, size_t n) { return std::string(reinterpret_cast<const char*>(p), n); }

inline std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + p.string());
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

/// The engine's own search order (Engine::find_kernels): the override, beside the model, then every xclbins root.
inline fs::path find_set(const std::string& model_dir, const std::string& name, const std::string& override_dir, std::string* how) {
    std::error_code ec;
    auto is_set = [&](const fs::path& d) { return fs::is_regular_file(d / "manifest.json", ec); };
    if (!override_dir.empty()) {
        *how = "--kernels";
        return is_set(override_dir) ? fs::path(override_dir) : fs::path();
    }
    if (const char* env = std::getenv("OFLM_OPEN_KERNELS_DIR"); env && *env && is_set(env)) {
        *how = "OFLM_OPEN_KERNELS_DIR";
        return env;
    }
    if (fs::path local = fs::path(model_dir) / "open_kernels"; is_set(local)) {
        *how = "beside the model";
        return local;
    }
    for (const std::string& r : utils::xclbin_roots()) {
        fs::path c = fs::path(r) / "xclbins" / name / "open_kernels";
        if (is_set(c)) {
            *how = "an xclbins root";
            return c;
        }
    }
    return {};
}

inline bool open_in_browser(const fs::path& page) {
#ifdef _WIN32
    auto r = reinterpret_cast<intptr_t>(ShellExecuteW(nullptr, L"open", page.wstring().c_str(), nullptr, nullptr, SW_SHOWNORMAL));
    return r > 32;
#else
    std::string p = page.string();
    char* argv[] = {const_cast<char*>("xdg-open"), p.data(), nullptr};
    pid_t pid = 0;
    if (posix_spawnp(&pid, "xdg-open", nullptr, nullptr, argv, environ) != 0) return false;
    int status = 0;
    return waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0;
#endif
}

inline void usage() {
    std::printf(
        "Usage: oflm viz [<model_tag>] [--no-open] [-o <file.html>] [--kernels <dir>]\n\n"
        "  Writes an animated page of one decode step of the model -- every NPU dispatch, DMA stream,\n"
        "  core function and host stage of its open kernel set -- and opens it in the default browser.\n"
        "  Placement, data paths and DMA order come from the compiled kernels; durations are modelled.\n\n"
        "  oflm viz                       list the models whose kernel set carries a viz\n"
        "  oflm viz qwen3.6-moe:35b-a3b   write and open the page\n"
        "  --no-open                      only print where the page was written\n"
        "  -o <file.html>                 write the page there instead of <models dir>/viz/\n"
        "  --kernels <dir>                read this kernel set instead of the model's own\n");
}

inline int run(int argc, char** argv) {
    std::string tag, out, kernels;
    bool open = true;
    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else if (a == "--no-open") {
            open = false;
        } else if ((a == "-o" || a == "--out" || a == "--kernels") && i + 1 < argc) {
            (a == "--kernels" ? kernels : out) = argv[++i];
        } else if (!a.empty() && a[0] != '-' && tag.empty()) {
            tag = a;
        } else {
            std::fprintf(stderr, "oflm viz: unexpected argument '%s'\n", a.c_str());
            usage();
            return 1;
        }
    }
    std::string config_path, models_dir;
    try {
        config_path = utils::find_model_list();
        models_dir = utils::get_models_directory();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "oflm viz: %s\n", e.what());
        return 1;
    }
    model_list models(config_path, models_dir);

    if (tag.empty()) {
        std::vector<std::string> tags(models.all_tags.begin(), models.all_tags.end());
        std::sort(tags.begin(), tags.end());
        int n = 0;
        for (const auto& t : tags) {
            if (t.find(':') == std::string::npos) continue;
            auto [resolved, info] = models.get_model_info(t);
            std::string how;
            fs::path set = find_set(models.get_model_path(t), info.value("name", std::string()), kernels, &how);
            if (!set.empty() && fs::is_regular_file(set / "viz.json")) {
                std::printf("  %s\n", t.c_str());
                ++n;
            }
        }
        if (!n) std::printf("No installed kernel set carries a viz.json yet; `oflm viz <tag>` says why for one model.\n");
        return 0;
    }
    if (!models.is_model_supported(tag)) {
        std::fprintf(stderr, "oflm viz: '%s' is not in the model list; `oflm list` shows the tags\n", tag.c_str());
        return 1;
    }
    auto [resolved, info] = models.get_model_info(tag);
    const std::string name = info.value("name", std::string());
    const std::string family = info.contains("details") ? info["details"].value("family", std::string("?")) : "?";
    std::string how;
    fs::path set = find_set(models.get_model_path(resolved), name, kernels, &how);
    if (set.empty()) {
        std::fprintf(stderr,
                     "oflm viz: %s (%s, family %s) has no open kernel set here. A model on oflm's precompiled kernels "
                     "cannot be inspected, so flow-viz is not implemented for it; an open-kernel model gets its set "
                     "from `oflm pull %s` or `oflm add`.\n",
                     resolved.c_str(), name.c_str(), family.c_str(), resolved.c_str());
        return 1;
    }
    if (!fs::is_regular_file(set / "viz.json")) {
        std::fprintf(stderr,
                     "oflm viz: the kernel set at %s predates flow-viz (it has no viz.json). Update oflm, or re-export "
                     "the set from a build tree: python open_kernels/export_qwen36_kernels.py --no-build --out <set>\n",
                     set.string().c_str());
        return 1;
    }
    nlohmann::json model = {{"tag", resolved}, {"set", set.string()}, {"found", how}};
    nlohmann::json entry;
    for (const char* k : {"name", "details", "label", "size", "footprint", "default_context_length"})
        if (info.contains(k)) entry[k] = info[k];
    model["entry"] = entry;
    fs::path dest = out.empty() ? fs::path(models_dir) / "viz" / (name + ".html") : fs::path(out);
    try {
        std::string html = render(asset(viz_assets::page, viz_assets::page_size), asset(viz_assets::fonts, viz_assets::fonts_size),
                                  read_file(set / "viz.json"), asset(viz_assets::explain, viz_assets::explain_size), model.dump());
        std::error_code ec;
        if (dest.has_parent_path()) fs::create_directories(dest.parent_path(), ec);
        std::ofstream f(dest, std::ios::binary);
        if (!f || !f.write(html.data(), static_cast<std::streamsize>(html.size())))
            throw std::runtime_error("cannot write " + dest.string());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "oflm viz: %s\n", e.what());
        return 1;
    }
    dest.make_preferred();
    set.make_preferred();
    std::printf("flow-viz: %s\n  kernels: %s (%s)\n", dest.string().c_str(), set.string().c_str(), how.c_str());
    if (open && !open_in_browser(dest)) std::printf("  could not start a browser here; open the file above in one\n");
    return 0;
}

}  // namespace viz_command

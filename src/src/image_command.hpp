/// \file image_command.hpp
/// \brief `oflm image <tag> "<prompt>" [--image in.png]` -- one image from the open
///        diffusion engine (src/open_diffusion; spec: specs/open-diffusion/spec.md,
///        OPEN-DIFFUSION-CLI, OPEN-DIFFUSION-EDIT).
///
/// Everything the run depends on is checked before anything is downloaded or loaded: the
/// tag names an image model, the output extension is one the engine encodes, the size is
/// one the registry lists -- and with --image, that the reference decodes and can be
/// prepared (reference.hpp). Then bench-embed's sequence: pull if missing, load, run.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_downloader.hpp"
#include "model_list.hpp"
#include "open_diffusion/engine.hpp"
#include "open_diffusion/prompt.hpp"
#include "open_diffusion/reference.hpp"
#include "program_args.hpp"
#include "tokenizer/tokenizer.hpp"
#include "utils/utils.hpp"

namespace image_command {

inline int run(const program_args_t& a, model_list& models, ModelDownloader& downloader) {
    namespace fs = std::filesystem;
    if (std::string why; !open_diffusion::available(&why)) {
        header_print("ERROR", "oflm image: " + why);
        return 1;
    }
    // get_model_info falls back to llama3.2:1b for an unknown size; main.cpp has already
    // refused a tag that is not in the list, so this is the tag's own entry
    auto [tag, info] = models.get_model_info(a.model_tag);
    if (!info.value("image", false)) {
        header_print("ERROR", "'" + tag + "' is not an image model; `oflm list` shows which are");
        return 1;
    }
    const bool edit = !a.image_ref.empty();
    std::vector<int> sizes = info.value(edit ? "image_edit_sizes" : "image_sizes", std::vector<int>{});
    auto list = [&] {
        std::string have;
        for (int s : sizes) have += (have.empty() ? "" : ", ") + std::to_string(s);
        return have;
    };
    if (edit && sizes.empty()) {
        header_print("ERROR", "'" + tag + "' has no edit configurations (edits are not implemented for it)");
        return 1;
    }
    // an edit's reference: read, and prepared to the size now, so a bad file stops the run here
    std::vector<uint8_t> ref_rgb;
    int size = a.image_size;
    if (edit) {
        std::ifstream rf(a.image_ref, std::ios::binary);
        if (!rf) {
            header_print("ERROR", "cannot open the reference image " + a.image_ref);
            return 1;
        }
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(rf)), std::istreambuf_iterator<char>());
        int w = 0, h = 0;
        if (!a.image_size_given && open_diffusion::reference_dims(bytes.data(), bytes.size(), &w, &h))
            size = open_diffusion::default_edit_size(w, h, sizes);
        if (std::find(sizes.begin(), sizes.end(), size) == sizes.end()) {
            header_print("ERROR", "unsupported --size " + std::to_string(size) + " for an edit with " + tag +
                                  " (supported: " + list() + "; the output is square, the reference's size)");
            return 1;
        }
        try {
            auto prepared = open_diffusion::prepare_reference(bytes.data(), bytes.size(), size);
            header_print("OFLM", prepared.describe());
            ref_rgb = std::move(prepared.rgb);
        } catch (const open_diffusion::ReferenceError& e) {
            header_print("ERROR", std::string(e.what()));
            return 1;
        }
    } else if (std::find(sizes.begin(), sizes.end(), size) == sizes.end()) {
        header_print("ERROR", "unsupported --size " + std::to_string(size) + " for " + tag +
                              " (supported: " + list() + ")");
        return 1;
    }
    uint64_t seed;
    if (a.image_seed.empty()) {
        std::random_device rd;
        seed = (static_cast<uint64_t>(rd()) << 32) | rd();
    } else {
        try {
            size_t used = 0;
            seed = std::stoull(a.image_seed, &used);
            if (used != a.image_seed.size()) throw std::invalid_argument(a.image_seed);
        } catch (const std::exception&) {
            header_print("ERROR", "--seed must be a non-negative integer, not '" + a.image_seed + "'");
            return 1;
        }
    }
    std::string out = a.image_out.empty() ? "oflm-" + std::to_string(seed) + ".png" : a.image_out;
    std::string format = open_diffusion::format_for_path(out);
    if (format.empty()) {
        header_print("ERROR", "unsupported output '" + out + "': use .png, .jpg or .jpeg");
        return 1;
    }

    switch (downloader.is_model_downloaded(tag)) {
        case ModelDownloader::ModelStatus::Ready:
            break;
        case ModelDownloader::ModelStatus::Missing:
        case ModelDownloader::ModelStatus::Outdated:
            header_print("OFLM", "Model not present or outdated -- pulling '" + tag + "'");
            if (!downloader.pull_model(tag, a.modelscope))
                throw std::runtime_error("failed to pull '" + tag + "'. Run `oflm pull " + tag + "` and retry.");
            break;
        case ModelDownloader::ModelStatus::Incompatible:
            throw std::runtime_error("'" + tag + "' is not compatible with this build of OFLM");
    }
    const std::string model_dir = models.get_model_path(tag);

    std::string how;
    std::string kernels = open_diffusion::find_kernels(
        model_dir, utils::getenv_oflm("OFLM_DIFFUSION_KERNELS_DIR"), utils::xclbin_roots(), &how);
    if (kernels.empty())
        throw std::runtime_error("no kernel set for " + tag + ": looked for open_kernels beside the model "
                                 "and under every xclbins root; set OFLM_DIFFUSION_KERNELS_DIR");
    header_print("OFLM", "Kernels (" + how + "): " + kernels);

    // Tokenizer's constructor exits the process on a missing file: check first
    if (!fs::is_regular_file(fs::path(model_dir) / "tokenizer.json"))
        throw std::runtime_error((fs::path(model_dir) / "tokenizer.json").string() +
                                 " is missing; run `oflm pull " + tag + " --force`");
    Tokenizer tok(model_dir);
    std::optional<open_diffusion::Engine> engine;
    if (edit) {
        engine.emplace(model_dir, kernels);
        engine->select(size, 0, true);
    } else {
        engine.emplace(model_dir, kernels, size);    // the size known up front: see engine.hpp
    }
    open_diffusion::Engine& eng = *engine;
    eng.set_tokens(open_diffusion::prompt_ids(tok, eng.prompt_template(), a.image_prompt, eng.max_tokens()));
    eng.set_noise(eng.seeded_noise(seed));
    if (edit) eng.set_reference(ref_rgb);
    open_diffusion::Timing t = eng.run();
    std::vector<uint8_t> bytes = eng.encode(format);

    std::ofstream f(out, std::ios::binary);
    if (!f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error("cannot write " + out);
    f.close();

    double text = 0, steps = 0, vae = 0, encode = 0;
    for (const auto& [phase, s] : t.phases) {
        if (phase.rfind("step", 0) == 0) steps += s;
        else if (phase == "vae") vae = s;
        else if (phase == "encode") encode = s;   // an edit's reference through the VAE encoder
        else text += s;                          // the text encoder and the conditioning
    }
    char line[200];
    if (edit)
        std::snprintf(line, sizeof line, "%.1f s on the NPU (text %.2f, encode %.2f, steps %.2f, vae %.2f)",
                      t.total_s, text, encode, steps, vae);
    else
        std::snprintf(line, sizeof line, "%.1f s on the NPU (text %.2f, steps %.2f, vae %.2f)",
                      t.total_s, text, steps, vae);
    header_print("OFLM", "Wrote " + fs::absolute(out).string());
    header_print("OFLM", "Seed " + std::to_string(seed));
    header_print("OFLM", std::string(line));
    return 0;
}

}  // namespace image_command

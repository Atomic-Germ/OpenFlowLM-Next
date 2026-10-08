// open_diffusion: FLUX.2 [klein] 4B text-to-image with every op on the NPU.
//
// The engine replays a bundle (utilities/dit-chain/export_bundle.py): the schedule
// open_kernels/klein_pipeline.py plans -- text encoder, conditioning, 4 denoising steps,
// VAE; 1050 dispatches over six kernel sets -- as XRT runs built once at load. Per image
// the host only writes the prompt's 512 embedding rows and the noise, patches te_attn's
// valid_len, and reads the RGBA back. Runs on one hardware context are queued back to
// back; before the next kernel set the host blocks on the last one (XRT's wait sleeps).
//
// Two directories: the model (q4nx-build --open-diffusion: bundle.json, the schedules,
// weights.bin, the embedding table) and a kernel set (export_dit_kernels.py --install:
// diffusion_kernels.json and the six sets). They must carry the same layout hash: the
// packed weights and the schedule are only valid against the streams they were made for.
//
// Tokenizing is the caller's: the engine takes token ids (prompt.hpp templates and
// tokenizes in the main build; utilities/dit-chain/klein_tokens.py writes them for the
// standalone CLI).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace open_diffusion {

struct Timing {
    double total_s = 0;                                  // first start to last completion
    std::vector<std::pair<std::string, double>> phases;  // cond, text, step0..3, vae
    std::vector<double> op_ms;                           // per op, with profile only
};

// The installed kernel set's manifest format (export_dit_kernels.py's MANIFEST_FORMAT).
constexpr const char* kKernelsFormat = "oflm-open-diffusion-kernels-v1";

// Whether dir holds a complete kernel set of this format and layout; *why says why not.
bool kernels_usable(const std::string& dir, const std::string& layout, std::string* why);

// The model's kernel set, or "": env_dir if given (used as named -- the engine then
// refuses it if wrong), else <model_dir>/open_kernels, else <root>/xclbins/<family>/
// open_kernels for each root, the first usable one. *how names the rule that chose it.
std::string find_kernels(const std::string& model_dir, const std::string& env_dir,
                         const std::vector<std::string>& roots, std::string* how);

class Engine {
public:
    // model_dir: q4nx-build --open-diffusion's output; kernels_dir: an installed kernel
    // set (find_kernels); size: a resolution the model has.
    Engine(const std::string& model_dir, const std::string& kernels_dir, int size);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    int size() const;
    int image_tokens() const;       // (size / 16)^2
    int latent_channels() const;    // 128
    int max_tokens() const;         // 512
    int pad_id() const;
    const std::string& prompt_template() const;   // "{prompt}" marks the user text

    // ids: the chat-templated prompt's tokens, unpadded, 1..max_tokens(); padded here.
    void set_tokens(const std::vector<int64_t>& ids);
    // The initial latents, packed [image_tokens, 128] bf16 bits.
    void set_noise(const std::vector<uint16_t>& bf16_bits);
    // bf16 bits of N(0, 1) samples from a seed (the engine's own generator).
    std::vector<uint16_t> seeded_noise(uint64_t seed) const;

    // Every op in order. profile: a blocking wait after each op (per-op times).
    Timing run(bool profile = false);
    // The image, [size, size, 3] RGB8.
    std::vector<uint8_t> rgb();
    // The image as "png" or "jpeg" file bytes (jpeg_quality 1..100).
    std::vector<uint8_t> encode(const std::string& format, int jpeg_quality = 90);
    // Per-op description for a profile: (kernel set, stream, phase).
    std::vector<std::tuple<std::string, std::string, std::string>> ops() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// "png" or "jpeg" for a path's extension (.png, .jpg, .jpeg; any case), else "".
std::string format_for_path(const std::string& path);

}  // namespace open_diffusion

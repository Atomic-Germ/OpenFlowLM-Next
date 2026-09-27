// open_diffusion: FLUX.2 [klein] 4B text-to-image with every op on the NPU.
//
// The engine replays a bundle (utilities/dit-chain/export_bundle.py): the schedule
// open_kernels/klein_pipeline.py plans -- text encoder, conditioning, 4 denoising steps,
// VAE; 1050 dispatches over six kernel sets -- as XRT runs built once at load. Per image
// the host only writes the prompt's 512 embedding rows and the noise, patches te_attn's
// valid_len, and reads the RGBA back. Runs on one hardware context are queued back to
// back; before the next kernel set the host blocks on the last one (XRT's wait sleeps).
//
// Tokenizing is the caller's: the engine takes token ids (the server has the HF
// tokenizer; utilities/dit-chain/klein_tokens.py writes them for the CLI).
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

class Engine {
public:
    // bundle_dir: export_bundle.py's output; size: a resolution the bundle has.
    Engine(const std::string& bundle_dir, int size);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    int size() const;
    int image_tokens() const;       // (size / 16)^2
    int latent_channels() const;    // 128
    int max_tokens() const;         // 512
    int pad_id() const;
    const std::string& prompt_template() const;   // "{prompt}" marks the user text

    // ids: the chat-templated prompt's tokens, at most max_tokens(); padded here.
    void set_tokens(const std::vector<int64_t>& ids);
    // The initial latents, packed [image_tokens, 128] bf16 bits.
    void set_noise(const std::vector<uint16_t>& bf16_bits);
    // bf16 bits of N(0, 1) samples from a seed (the engine's own generator).
    std::vector<uint16_t> seeded_noise(uint64_t seed) const;

    // Every op in order. profile: a blocking wait after each op (per-op times).
    Timing run(bool profile = false);
    // The image, [size, size, 3] RGB8.
    std::vector<uint8_t> rgb();
    // Per-op description for a profile: (kernel set, stream, phase).
    std::vector<std::tuple<std::string, std::string, std::string>> ops() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// An 8-bit RGB PNG with stored (uncompressed) deflate blocks: no zlib.
void write_png_rgb(const std::string& path, const uint8_t* rgb, int width, int height);

}  // namespace open_diffusion

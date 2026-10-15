// open_diffusion on a build without the engine (OFLM_USE_HRX): the interface engine.hpp
// declares, answering "not implemented". engine.cpp drives XRT directly -- six hardware
// contexts in one process and sub-buffer views -- which the HRX shim does not offer, so
// CMake compiles this file in its place and the callers need no build flag of their own.
#include "engine.hpp"

#include <stdexcept>

namespace open_diffusion {
namespace {

const char* kWhy =
    "image generation is not implemented in this build: the open diffusion engine runs on "
    "XRT, and this is an HRX build";

[[noreturn]] void unavailable() { throw std::runtime_error(kWhy); }

}  // namespace

bool available(std::string* why) {
    if (why) *why = kWhy;
    return false;
}

bool kernels_usable(const std::string&, const std::string&, std::string* why) {
    if (why) *why = kWhy;
    return false;
}

std::string find_kernels(const std::string&, const std::string&, const std::vector<std::string>&,
                         std::string*) {
    unavailable();
}

struct Engine::Impl {};

Engine::Engine(const std::string&, const std::string&, const oflm_rt::device*) { unavailable(); }
Engine::Engine(const std::string&, const std::string&, int) { unavailable(); }
Engine::~Engine() = default;

std::vector<int> Engine::sizes() const { unavailable(); }
std::vector<int> Engine::edit_sizes() const { unavailable(); }
int Engine::default_steps() const { unavailable(); }
void Engine::select(int, int, bool) { unavailable(); }
int Engine::size() const { unavailable(); }
bool Engine::editing() const { unavailable(); }
int Engine::steps() const { unavailable(); }
int Engine::image_tokens() const { unavailable(); }
int Engine::latent_channels() const { unavailable(); }
int Engine::max_tokens() const { unavailable(); }
int Engine::pad_id() const { unavailable(); }
const std::string& Engine::prompt_template() const { unavailable(); }
void Engine::set_tokens(const std::vector<int64_t>&) { unavailable(); }
void Engine::set_noise(const std::vector<uint16_t>&) { unavailable(); }
void Engine::set_reference(const std::vector<uint8_t>&) { unavailable(); }
std::vector<uint16_t> Engine::seeded_noise(uint64_t) const { unavailable(); }
Timing Engine::run(bool) { unavailable(); }
std::vector<uint8_t> Engine::rgb() { unavailable(); }
std::vector<uint8_t> Engine::encode(const std::string&, int) { unavailable(); }
std::vector<std::tuple<std::string, std::string, std::string>> Engine::ops() const { unavailable(); }

std::string format_for_path(const std::string&) { unavailable(); }

}  // namespace open_diffusion

// open_diffusion prompt: see prompt.hpp.
#include "prompt.hpp"

#include <stdexcept>

#include "tokenizer/tokenizer.hpp"

namespace open_diffusion {

std::vector<int64_t> prompt_ids(Tokenizer& tok, const std::string& templ, const std::string& prompt,
                                int max_tokens) {
    static const std::string kSlot = "{prompt}";
    auto at = templ.find(kSlot);
    if (at == std::string::npos) throw std::runtime_error("the prompt template has no {prompt}");
    std::string text = templ.substr(0, at) + prompt + templ.substr(at + kSlot.size());
    std::vector<int> ids = tok.encode(text);
    if (static_cast<int>(ids.size()) > max_tokens) ids.resize(static_cast<size_t>(max_tokens));
    return std::vector<int64_t>(ids.begin(), ids.end());
}

}  // namespace open_diffusion

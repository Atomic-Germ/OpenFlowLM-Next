// open_diffusion prompt: text to the engine's token ids, in the main build (it needs the
// HF tokenizer the rest of oflm uses; the standalone CLI takes ids instead).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

class Tokenizer;

namespace open_diffusion {

// bundle.json's prompt_template with "{prompt}" replaced by prompt, tokenized without
// special tokens (the template carries its own), truncated to max_tokens after templating:
// klein_pipeline.token_ids, which the goldens (specs/open-diffusion/tests) come from.
// Unpadded; Engine::set_tokens pads.
std::vector<int64_t> prompt_ids(Tokenizer& tok, const std::string& templ, const std::string& prompt,
                                int max_tokens);

}  // namespace open_diffusion

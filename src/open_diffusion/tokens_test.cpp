// Traces: OPEN-DIFFUSION-TOKENS (canonical spec: specs/open-diffusion/spec.md)
//
// open_diffusion_tokens_test GOLDENS.json [MODEL_DIR]
//
// Every case's prompt through oflm's own prompt path (prompt_ids over the main build's
// Tokenizer and the model's bundle.json template) must give exactly the ids
// klein_pipeline.token_ids gave (utilities/dit-chain/klein_tokens.py --goldens). A wrong id
// is silent: the image just drifts. MODEL_DIR defaults to the installed model; without it
// the test skips (CTest SKIP, 77), naming the path: the model is a gigabyte install, not a CI
// fixture, and ctest reports a skip as a skip, never as a pass.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "prompt.hpp"
#include "tokenizer/tokenizer.hpp"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: open_diffusion_tokens_test GOLDENS.json [MODEL_DIR]\n");
        return 2;
    }
    fs::path model;
    if (argc > 2) {
        model = argv[2];
    } else {
        const char* root = std::getenv("OFLM_MODEL_PATH");
#ifdef _WIN32
        const char* home = std::getenv("USERPROFILE");
        const fs::path dflt = ".oflm";
#else
        const char* home = std::getenv("HOME");
        const fs::path dflt = fs::path(".config") / "oflm";
#endif
        // utils::get_models_directory's root, then model_list.json's "models" subdirectory
        model = root ? fs::path(root) : fs::path(home ? home : ".") / dflt;
        model /= fs::path("models") / "FLUX.2-klein-4B-NPU2";
    }
    for (const char* f : {"tokenizer.json", "bundle.json"}) {
        if (!fs::is_regular_file(model / f)) {
            std::fprintf(stderr, "SKIP: %s is missing (install flux2-klein:4b or pass MODEL_DIR)\n",
                         (model / f).string().c_str());
            return 77;  // CTest SKIP_RETURN_CODE: the model is an install, not a CI fixture
        }
    }
    nlohmann::json goldens = nlohmann::json::parse(std::ifstream(argv[1]));
    nlohmann::json bundle = nlohmann::json::parse(std::ifstream(model / "bundle.json"));
    const std::string templ = bundle.at("prompt_template").get<std::string>();
    const int max_tokens = bundle.at("max_tokens").get<int>();
    if (max_tokens != goldens.at("max_tokens").get<int>()) {
        std::fprintf(stderr, "FAIL: the model's max_tokens %d is not the goldens' %d\n", max_tokens,
                     goldens.at("max_tokens").get<int>());
        return 1;
    }

    Tokenizer tok(model.string());
    int failed = 0, n = 0;
    for (const auto& c : goldens.at("cases")) {
        ++n;
        const std::string prompt = c.at("prompt").get<std::string>();
        const auto want = c.at("ids").get<std::vector<int64_t>>();
        const auto got = open_diffusion::prompt_ids(tok, templ, prompt, max_tokens);
        if (got == want) continue;
        ++failed;
        size_t i = 0;
        while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
        std::fprintf(stderr, "FAIL case %d (%zu chars): %zu ids, want %zu; first difference at %zu\n", n,
                     prompt.size(), got.size(), want.size(), i);
    }
    std::printf("%s: %d of %d cases match\n", failed ? "FAIL" : "OK", n - failed, n);
    return failed ? 1 : 0;
}

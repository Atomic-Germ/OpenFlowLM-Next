// Reference logits and greedy continuations from llama.cpp (b10944) for the prefill accuracy study.
// llama_ref_dump --model m.gguf --ids-list prompts.txt --out dir [--n-decode 64] [--ngl 0] [--threads 8] [--first 0] [--count N]
// prompts.txt: one prompt per line, comma-separated token ids. Writes <out>/p<k>_logits.f32 (n x V float32, every position)
// and <out>/p<k>_gen.txt (the greedy continuation).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include "llama.h"
#include "ggml-backend.h"

int main(int argc, char** argv) {
    std::string model, list, out;
    int n_decode = 64, ngl = 0, threads = 8, first = 0, count = 1 << 30, n_ctx = 2304;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--model") model = v;
        else if (k == "--ids-list") list = v;
        else if (k == "--out") out = v;
        else if (k == "--n-decode") n_decode = std::atoi(v.c_str());
        else if (k == "--ngl") ngl = std::atoi(v.c_str());
        else if (k == "--threads") threads = std::atoi(v.c_str());
        else if (k == "--first") first = std::atoi(v.c_str());
        else if (k == "--count") count = std::atoi(v.c_str());
        else if (k == "--n-ctx") n_ctx = std::atoi(v.c_str());
    }
    if (model.empty() || list.empty() || out.empty()) { std::fprintf(stderr, "usage: see header\n"); return 2; }
    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model* m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) { std::fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(m);
    const int V = llama_vocab_n_tokens(vocab);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = 2048; cp.n_ubatch = 512; cp.n_threads = threads; cp.n_threads_batch = threads;
    llama_context* ctx = llama_init_from_model(m, cp);
    if (!ctx) { std::fprintf(stderr, "context failed\n"); return 1; }
    std::ifstream f(list);
    std::string line;
    int k = -1;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        ++k;
        if (k < first || k >= first + count) continue;
        std::vector<llama_token> ids;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) ids.push_back(std::atoi(tok.c_str()));
        const int n = static_cast<int>(ids.size());
        llama_memory_clear(llama_get_memory(ctx), true);
        std::string lp = out + "/p" + std::to_string(k) + "_logits.f32";
        FILE* lf = std::fopen(lp.c_str(), "wb");
        const int UB = 512;
        llama_batch b = llama_batch_init(UB, 0, 1);
        for (int s = 0; s < n; s += UB) {
            const int c = std::min(UB, n - s);
            b.n_tokens = c;
            for (int i = 0; i < c; ++i) {
                b.token[i] = ids[s + i]; b.pos[i] = s + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
            }
            if (llama_decode(ctx, b)) { std::fprintf(stderr, "decode failed at %d\n", s); return 1; }
            for (int i = 0; i < c; ++i) std::fwrite(llama_get_logits_ith(ctx, i), 4, V, lf);
        }
        std::fclose(lf);
        std::vector<int> gen;
        int pos = n;
        const float* last = llama_get_logits_ith(ctx, (n - 1) % UB);
        std::vector<float> cur(last, last + V);
        for (int g = 0; g < n_decode; ++g) {
            int best = static_cast<int>(std::max_element(cur.begin(), cur.end()) - cur.begin());
            gen.push_back(best);
            if (g + 1 == n_decode) break;
            b.n_tokens = 1; b.token[0] = best; b.pos[0] = pos++; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
            if (llama_decode(ctx, b)) { std::fprintf(stderr, "decode failed (gen)\n"); return 1; }
            const float* l = llama_get_logits_ith(ctx, 0);
            cur.assign(l, l + V);
        }
        llama_batch_free(b);
        std::ofstream gf(out + "/p" + std::to_string(k) + "_gen.txt");
        for (size_t i = 0; i < gen.size(); ++i) gf << (i ? "," : "") << gen[i];
        gf << "\n";
        std::fprintf(stderr, "prompt %d: %d tokens, V=%d\n", k, n, V);
    }
    llama_free(ctx);
    llama_model_free(m);
    return 0;
}

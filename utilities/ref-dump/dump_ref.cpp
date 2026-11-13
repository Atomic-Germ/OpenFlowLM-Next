/// \file dump_ref.cpp
/// \brief llama.cpp reference dumper for the open engines' bring-up: final logits
///        per position AND every transformer layer's output ("l_out-<il>").
///
/// The open engines can dump a layer's residual stream as it runs
/// (open_qwen36_cli under OFLM_OPEN_LAYER_NAN=1 OFLM_OPEN_LAYER_DUMP=<prefix>),
/// and comparing the logits alone cannot say WHICH layer diverged. This tool
/// produces the other side of that comparison from llama.cpp itself, through
/// the public ggml_backend_sched eval callback: no reimplementation of any
/// model, and the same tensors llama.cpp's own graph computes.
///
///   dump_ref --model <model.gguf> --ids 1,2,3 --out /tmp/ref/y
///
/// Writes, for every position p in 0..P-1:
///   <out>_p<p>.bin   f32[vocab]   final logits at p (llama_get_logits_ith)
/// and, for every layer l:
///   <out>_l<l>.bin   f32[n_tokens][hidden]   layer l's output, one hidden
///                    vector per token, in position order (the tensor is
///                    ne[0]=hidden, ne[1]=n_tokens, so token p starts at
///                    p*hidden)
///   <out>_meta.txt   "tokens <P> hidden <H> layers <L> vocab <V>"
///
/// All prompt tokens are decoded in ONE llama_decode, so every l_out tensor is
/// computed once over the whole prompt and token p's vector is directly
/// comparable to the engine's per-step dump at position p.
///
/// Build: see build.sh next to this file (links the installed libllama).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"
#include "ggml-backend.h"

namespace {

struct DumpState {
    std::string prefix;
    int hidden = 0;
    int layers = 0;
    int tokens = 0;
    std::vector<char> seen;      ///< per layer: wrote it already this decode
    std::vector<char> buffer;    ///< scratch for the backend copy
    std::string filter;          ///< --tensors: substring every dumped name must contain
};

bool on_tensor(struct ggml_tensor* t, bool ask, void* user_data) {
    if (ask || t == nullptr) return true;       // ask=true: we want to see it
    auto* st = static_cast<DumpState*>(user_data);
    const char* name = ggml_get_name(t);
    if (name == nullptr) return true;

    // Per-layer output: "l_out-<il>". The residual stream after layer il,
    // exactly what the engine's xres holds when it has run layer il.
    if (std::strncmp(name, "l_out-", 6) != 0) {
        // --tensors: any other named tensor, dumped whole under its own name.
        // Bring-up uses it to walk one layer stage by stage (linear_attn_qkv_mixed-0,
        // conv_output_silu-0, q_conv-0, final_output-0, ...).
        if (st->filter.empty() || std::string(name).find(st->filter) == std::string::npos) return true;
        size_t bytes = ggml_nbytes(t);
        if (st->buffer.size() < bytes) st->buffer.resize(bytes);
        ggml_backend_tensor_get(t, st->buffer.data(), 0, bytes);
        char path[1024];
        std::snprintf(path, sizeof path, "%s_t_%s.bin", st->prefix.c_str(), name);
        if (std::FILE* f = std::fopen(path, "wb")) {
            std::fwrite(st->buffer.data(), 1, bytes, f);
            std::fclose(f);
        }
        return true;
    }
    const int il = std::atoi(name + 6);
    if (il < 0 || il >= st->layers) return true;
    if (st->seen[static_cast<size_t>(il)]) return true;   // one write per layer per decode
    st->seen[static_cast<size_t>(il)] = 1;

    const int hidden = static_cast<int>(t->ne[0]);
    const int ntok = static_cast<int>(t->ne[1]);
    if (hidden != st->hidden) {
        std::fprintf(stderr, "dump_ref: layer %d has hidden %d, expected %d\n", il, hidden, st->hidden);
        return true;
    }
    const size_t bytes = static_cast<size_t>(hidden) * static_cast<size_t>(ntok) * sizeof(float);
    if (st->buffer.size() < bytes) st->buffer.resize(bytes);
    ggml_backend_tensor_get(t, st->buffer.data(), 0, bytes);

    char path[1024];
    std::snprintf(path, sizeof path, "%s_l%d.bin", st->prefix.c_str(), il);
    std::FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::fprintf(stderr, "dump_ref: cannot write %s\n", path);
        return true;
    }
    std::fwrite(st->buffer.data(), 1, bytes, f);
    std::fclose(f);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* out_prefix = "/tmp/ref/y";
    const char* ids_str = nullptr;
    const char* tensor_filter = nullptr;
    int n_ctx = 4096;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model")) model_path = argv[++i];
        else if (!std::strcmp(argv[i], "--ids")) ids_str = argv[++i];
        else if (!std::strcmp(argv[i], "--out")) out_prefix = argv[++i];
        else if (!std::strcmp(argv[i], "--tensors")) tensor_filter = argv[++i];
        else if (!std::strcmp(argv[i], "--n-ctx")) n_ctx = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (!model_path || !ids_str) {
        std::fprintf(stderr, "usage: dump_ref --model <model.gguf> --ids <csv> [--out <prefix>] [--n-ctx N]\n");
        return 2;
    }

    std::vector<llama_token> tokens;
    {
        std::string s(ids_str);
        size_t at = 0;
        while (at <= s.size()) {
            const size_t comma = s.find(',', at);
            const std::string piece = s.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
            if (!piece.empty()) tokens.push_back(static_cast<llama_token>(std::atoi(piece.c_str())));
            if (comma == std::string::npos) break;
            at = comma + 1;
        }
    }
    if (tokens.empty()) { std::fprintf(stderr, "dump_ref: no token ids\n"); return 2; }

    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) { std::fprintf(stderr, "dump_ref: failed to load %s\n", model_path); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    DumpState st;
    st.prefix = out_prefix;
    st.hidden = llama_model_n_embd(model);
    st.layers = llama_model_n_layer(model);
    st.tokens = static_cast<int>(tokens.size());
    if (tensor_filter) st.filter = tensor_filter;
    st.seen.assign(static_cast<size_t>(st.layers), 0);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(std::max(n_ctx, static_cast<int>(tokens.size())));
    cparams.n_batch = static_cast<uint32_t>(std::max(n_ctx, static_cast<int>(tokens.size())));
    cparams.n_ubatch = cparams.n_batch;
    cparams.cb_eval = on_tensor;
    cparams.cb_eval_user_data = &st;
    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) { std::fprintf(stderr, "dump_ref: failed to create context\n"); llama_model_free(model); return 1; }

    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;        // logits at every position
    }
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) { std::fprintf(stderr, "dump_ref: llama_decode failed (%d)\n", rc); llama_free(ctx); llama_model_free(model); return 1; }

    char path[1024];
    for (int p = 0; p < st.tokens; ++p) {
        const float* logits = llama_get_logits_ith(ctx, p);
        if (!logits) { std::fprintf(stderr, "dump_ref: no logits at position %d\n", p); continue; }
        std::snprintf(path, sizeof path, "%s_p%d.bin", out_prefix, p);
        std::FILE* f = std::fopen(path, "wb");
        if (!f) { std::fprintf(stderr, "dump_ref: cannot write %s\n", path); continue; }
        std::fwrite(logits, sizeof(float), static_cast<size_t>(n_vocab), f);
        std::fclose(f);
    }

    std::snprintf(path, sizeof path, "%s_meta.txt", out_prefix);
    if (std::FILE* f = std::fopen(path, "w")) {
        std::fprintf(f, "tokens %d hidden %d layers %d vocab %d\n", st.tokens, st.hidden, st.layers, n_vocab);
        std::fclose(f);
    }

    int wrote = 0;
    for (char c : st.seen) wrote += c != 0;
    std::fprintf(stderr, "dump_ref: %d positions, %d/%d layer dumps, hidden %d, vocab %d -> %s\n",
                 st.tokens, wrote, st.layers, st.hidden, n_vocab, out_prefix);

    llama_free(ctx);
    llama_model_free(model);
    return wrote == st.layers ? 0 : 1;
}

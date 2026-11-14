/// \file npue_embedding.cpp
/// \brief The only translation unit that sees the NpuEmbeddings engine.
/// \date 2026-09-03
///
/// SPDX-License-Identifier: MIT
///
/// THIS FILE IS FORK-OWNED; src/open_npue/ is a synced copy from upstream and
/// must not be edited here. The separation is deliberate: the sync tool
/// overwrites everything under open_npue/ and knows nothing about this file.
///
/// It is listed in OPEN_NPUE_SOURCES in src/CMakeLists.txt, which means it
/// carries the engine's own compile flags -- `/arch:AVX2` (or `-mavx2 -mfma`)
/// and the open_npue include directory. Both matter, and the header explains
/// why at length: the engine is header-defined and half of an encode is AVX2
/// intrinsics behind `#if defined(__AVX2__)`, so a host translation unit that
/// includes it WITHOUT the flag instantiates the scalar path, and the linker
/// keeps a mixture of the two. Measured cost of that mistake: 1-cos 1.04e-04
/// on bge-base against the same engine in its own binary, on identical bytes.

#include "AutoEmbeddingModel/npue_embedding.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>

#include "gguf/gguf_file.hpp"
#include "utils/utils.hpp"
#include "npue_encoder.hpp"
#include "npue_pack.hpp"

namespace {

/// Map the caller's task onto one of THIS model's prompt names.
///
/// THE MOST DANGEROUS SEAM IN THE WHOLE INTEGRATION, and the reason it refuses
/// rather than guessing: a wrongly-prefixed embedding is correctly shaped,
/// correctly normed and deterministic. Nothing downstream can tell that the
/// answer is for a different task.
///
/// Models differ in what they offer -- the BERT family has no prompt concept
/// at all, nomic has four names, EmbeddingGemma has fourteen -- so this asks
/// the CONTAINER what exists and takes the first candidate the model actually
/// declares. If a model has prompts and none of the candidates is among them,
/// that is a mapping this build does not know, and it throws naming what the
/// model does offer.
std::string prompt_for(const std::vector<std::string>& names,
                       embedding_task_type_t task_type) {
    if (names.empty()) return std::string();

    std::vector<std::string> candidates;
    switch (task_type) {
        case task_query:
            candidates = {"query", "search_query", "Retrieval-query",
                          "Retrieval"};                        break;
        case task_document:
            candidates = {"document", "search_document",
                          "Retrieval-document", "Retrieval"};  break;
        case task_clustering:
            candidates = {"clustering", "Clustering"};         break;
        case task_classification:
            candidates = {"classification", "Classification"}; break;
        case task_multilabel_classification:
            candidates = {"MultilabelClassification",
                          "classification", "Classification"}; break;
        case task_sentence_similarity:
            candidates = {"STS", "classification", "search_query"}; break;
        case task_summarization:
            candidates = {"Summarization", "clustering"};      break;
        case task_bitextmining:
            candidates = {"BitextMining", "search_query"};     break;
        case task_code_retrieval:
            candidates = {"Retrieval-query", "search_query", "query"}; break;
        case task_search_result:
            candidates = {"search_document", "document",
                          "Retrieval-document"};               break;
        default:
            candidates = {"search_query", "query"};            break;
    }
    for (const auto& c : candidates)
        if (std::find(names.begin(), names.end(), c) != names.end()) return c;

    std::string have;
    for (const auto& n : names) have += (have.empty() ? "" : ", ") + n;
    throw TaskPromptUnavailable(
        "this model declares task prompts [" + have +
        "] and none of them matches the requested task. Refusing to pick one: "
        "a wrongly-prefixed embedding is correctly shaped and correctly "
        "normed, so nothing downstream could tell the answer is for a "
        "different task.");
}

int int_or(const json& j, const char* key, int fallback) {
    if (j.contains(key) && j[key].is_number_integer()) return j[key].get<int>();
    return fallback;
}

// ---- GGUF -> the packer's checkpoint ------------------------------------
//
// The NpuEmbeddings packers read a HuggingFace checkpoint: an F32 safetensors
// with HF tensor names, config.json, vocab.txt and 1_Pooling/config.json. A
// GGUF holds the same weights in llama.cpp's names and block layouts, plus
// enough metadata to describe the model. The helpers below turn one into the
// other in a staging directory, so `oflm add <gguf repo>` works for these
// families exactly as it does for the chat families -- and
// npue::prepare_model_auto() is still the only place that decides which
// architecture to pack. Nothing under src/open_npue/ is touched.

namespace fs = std::filesystem;

/// The GGUF a directory carries, or "" when it carries none. `model.gguf` is
/// the install convention; otherwise the single *.gguf (mmproj excluded) is
/// taken, the same rule the open_embedding engine uses.
std::string find_gguf(const fs::path& dir) {
    std::error_code ec;
    if (fs::is_regular_file(dir / "model.gguf", ec))
        return (dir / "model.gguf").string();
    std::vector<fs::path> found;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end;
         it.increment(ec)) {
        if (it->path().extension() != ".gguf") continue;
        const std::string n = it->path().filename().string();
        if (n.rfind("mmproj", 0) == 0) continue;   // the vision tower
        found.push_back(it->path());
    }
    if (found.size() == 1) return found.front().string();
    if (found.size() > 1) {
        std::string names;
        for (const auto& f : found)
            names += (names.empty() ? "" : ", ") + f.filename().string();
        throw std::runtime_error(
            "NpueEmbedding: " + dir.string() + " holds several .gguf files (" +
            names + "); name the weights one as model.gguf");
    }
    return std::string();
}

struct NamePair {
    const char* gguf;
    const char* hf;
};

const char* lookup_pair(const NamePair* table, size_t n,
                        const std::string& key) {
    for (size_t i = 0; i < n; ++i)
        if (key == table[i].gguf) return table[i].hf;
    return nullptr;
}

/// llama.cpp tensor name -> HuggingFace name, for the two architectures the
/// NpuEmbeddings packers cover. Empty when this tensor is not part of the
/// packer's input (a pooler, say), which the caller skips with a note.
///
/// The names are llama.cpp's own conversion output, read off real bge /
/// MiniLM / nomic GGUFs rather than inferred: `token_embd` ->
/// `embeddings.word_embeddings`, `blk.N.attn_output_norm` -> the post-attention
/// LayerNorm, `blk.N.layer_output_norm` -> the post-FFN one.
std::string hf_embed_name(const std::string& arch, const std::string& g) {
    static const NamePair bert_top[] = {
        {"token_embd.weight", "embeddings.word_embeddings.weight"},
        {"position_embd.weight", "embeddings.position_embeddings.weight"},
        {"token_types.weight", "embeddings.token_type_embeddings.weight"},
        {"token_embd_norm.weight", "embeddings.LayerNorm.weight"},
        {"token_embd_norm.bias", "embeddings.LayerNorm.bias"},
    };
    static const NamePair bert_layer[] = {
        {"attn_q.weight", "attention.self.query.weight"},
        {"attn_q.bias", "attention.self.query.bias"},
        {"attn_k.weight", "attention.self.key.weight"},
        {"attn_k.bias", "attention.self.key.bias"},
        {"attn_v.weight", "attention.self.value.weight"},
        {"attn_v.bias", "attention.self.value.bias"},
        {"attn_output.weight", "attention.output.dense.weight"},
        {"attn_output.bias", "attention.output.dense.bias"},
        {"attn_output_norm.weight", "attention.output.LayerNorm.weight"},
        {"attn_output_norm.bias", "attention.output.LayerNorm.bias"},
        {"ffn_up.weight", "intermediate.dense.weight"},
        {"ffn_up.bias", "intermediate.dense.bias"},
        {"ffn_down.weight", "output.dense.weight"},
        {"ffn_down.bias", "output.dense.bias"},
        {"layer_output_norm.weight", "output.LayerNorm.weight"},
        {"layer_output_norm.bias", "output.LayerNorm.bias"},
    };
    static const NamePair nomic_top[] = {
        {"token_embd.weight", "embeddings.word_embeddings.weight"},
        {"token_types.weight", "embeddings.token_type_embeddings.weight"},
        {"token_embd_norm.weight", "emb_ln.weight"},
        {"token_embd_norm.bias", "emb_ln.bias"},
    };
    static const NamePair nomic_layer[] = {
        {"attn_qkv.weight", "attn.Wqkv.weight"},
        {"attn_output.weight", "attn.out_proj.weight"},
        {"attn_output_norm.weight", "norm1.weight"},
        {"attn_output_norm.bias", "norm1.bias"},
        {"ffn_up.weight", "mlp.fc11.weight"},
        {"ffn_gate.weight", "mlp.fc12.weight"},
        {"ffn_down.weight", "mlp.fc2.weight"},
        {"layer_output_norm.weight", "norm2.weight"},
        {"layer_output_norm.bias", "norm2.bias"},
    };

    const bool bert = arch == "bert";
    const bool nomic = arch == "nomic-bert";
    if (!bert && !nomic)
        throw std::runtime_error(
            "NpueEmbedding: GGUF architecture '" + arch + "' is not one this "
            "backend packs (supported: bert, nomic-bert). The NpuEmbeddings "
            "packers read a HuggingFace checkpoint, so a GGUF of another arch "
            "has no name mapping here.");

    if (g.rfind("blk.", 0) != 0) {
        const char* hf = bert
            ? lookup_pair(bert_top, sizeof bert_top / sizeof bert_top[0], g)
            : lookup_pair(nomic_top, sizeof nomic_top / sizeof nomic_top[0], g);
        return hf ? std::string(hf) : std::string();
    }
    const size_t dot = g.find('.', 4);
    if (dot == std::string::npos) return std::string();
    const std::string idx = g.substr(4, dot - 4);
    const std::string suffix = g.substr(dot + 1);
    if (bert) {
        const char* hf = lookup_pair(bert_layer,
                                     sizeof bert_layer / sizeof bert_layer[0],
                                     suffix);
        return hf ? "encoder.layer." + idx + "." + hf : std::string();
    }
    const char* hf = lookup_pair(nomic_layer,
                                 sizeof nomic_layer / sizeof nomic_layer[0],
                                 suffix);
    return hf ? "encoder.layers." + idx + "." + hf : std::string();
}

/// One tensor of the staged safetensors: its HF name and ROW-MAJOR shape.
/// GGUF stores dims fastest-first, so the shape is the reverse of `dims`.
struct StTensor {
    std::string name;       // HuggingFace name in the staged file
    std::string gguf_name;  // the source tensor
    std::vector<int64_t> shape;
    uint64_t count = 0;
};

/// Minimal safetensors writer: 8-byte little-endian header length, that many
/// bytes of JSON, then the tensor data in header order. Only F32 is emitted --
/// the packers read F32, and widening here is exact for every source type
/// except the block quants, which are dequantized by the GGUF reader.
void write_f32_safetensors(const fs::path& out,
                           const std::vector<StTensor>& tensors,
                           const std::function<std::vector<float>(const StTensor&)>& data) {
    std::string hdr = "{";
    uint64_t off = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const StTensor& t = tensors[i];
        const uint64_t bytes = t.count * 4;
        if (i) hdr += ',';
        hdr += "\"" + t.name + "\":{\"dtype\":\"F32\",\"shape\":[";
        for (size_t d = 0; d < t.shape.size(); ++d) {
            if (d) hdr += ',';
            hdr += std::to_string(t.shape[d]);
        }
        hdr += "],\"data_offsets\":[" + std::to_string(off) + "," +
               std::to_string(off + bytes) + "]}";
        off += bytes;
    }
    hdr += '}';
    while (hdr.size() % 8) hdr += ' ';

    std::ofstream f(out, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("NpueEmbedding: cannot write " + out.string());
    const uint64_t hlen = hdr.size();
    f.write(reinterpret_cast<const char*>(&hlen), 8);
    f.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
    for (const StTensor& t : tensors) {
        std::vector<float> v = data(t);
        if (v.size() != t.count)
            throw std::runtime_error("NpueEmbedding: " + t.name + " decoded to " +
                                     std::to_string(v.size()) + " values, expected " +
                                     std::to_string(t.count));
        f.write(reinterpret_cast<const char*>(v.data()),
                static_cast<std::streamsize>(v.size() * 4));
    }
    if (!f) throw std::runtime_error("NpueEmbedding: write failed for " + out.string());
}

uint64_t req_u64(const open_qwen36::GgufFile& g, const std::string& key) {
    if (!g.kv_has(key))
        throw std::runtime_error(
            "NpueEmbedding: the GGUF has no '" + key + "' -- cannot describe "
            "the checkpoint to the packer");
    return g.kv_u64(key);
}

/// The config.json the packers read, from the GGUF's own metadata. `model_type`
/// is what npue::prepare_model_auto() dispatches on; the geometry keys are
/// what each packer asserts against the tensors.
json derive_embed_config(const open_qwen36::GgufFile& g, const std::string& arch,
                         const std::string& model_type, uint64_t vocab,
                         uint64_t type_vocab) {
    json c = json::object();
    c["model_type"] = model_type;
    c["hidden_size"] = req_u64(g, arch + ".embedding_length");
    c["num_hidden_layers"] = req_u64(g, arch + ".block_count");
    c["num_attention_heads"] = req_u64(g, arch + ".attention.head_count");
    c["intermediate_size"] = req_u64(g, arch + ".feed_forward_length");
    c["vocab_size"] = vocab;
    c["type_vocab_size"] = type_vocab;
    if (g.kv_has(arch + ".attention.layer_norm_epsilon"))
        c["layer_norm_eps"] = g.kv_f64(arch + ".attention.layer_norm_epsilon");
    if (g.kv_has(arch + ".context_length"))
        c["max_position_embeddings"] = g.kv_u64(arch + ".context_length");
    if (model_type == "nomic_bert") {
        const uint64_t hidden = c["hidden_size"].get<uint64_t>();
        const uint64_t heads = c["num_attention_heads"].get<uint64_t>();
        c["head_dim"] = heads ? hidden / heads : 0;
        // nomic's packer asserts these are the values it was validated
        // against; they are properties of the architecture, not knobs.
        if (c.contains("layer_norm_eps")) c["layer_norm_epsilon"] = c["layer_norm_eps"];
        c["rotary_emb_base"] = g.kv_has(arch + ".rope.freq_base")
            ? json(g.kv_f64(arch + ".rope.freq_base")) : json(1000.0);
        c["rotary_emb_fraction"] = 1.0;
        c["rotary_emb_interleaved"] = false;
        c["qkv_proj_bias"] = false;
        c["mlp_fc1_bias"] = false;
        c["mlp_fc2_bias"] = false;
        c["prenorm"] = false;
        c["activation_function"] = "swiglu";
        c["hidden_act"] = "silu";
    }
    return c;
}

/// 1_Pooling/config.json from llama.cpp's `*.pooling_type` (1 = mean, 2 = CLS).
/// This is the same decision the packer's resolve_pooling() reads; the GGUF
/// carries it, so a GGUF-only install needs no 1_Pooling file.
json derive_pooling_config(const open_qwen36::GgufFile& g, const std::string& arch,
                           uint64_t hidden) {
    const uint64_t p = req_u64(g, arch + ".pooling_type");
    if (p != 1 && p != 2)
        throw std::runtime_error(
            "NpueEmbedding: the GGUF's " + arch + ".pooling_type is " +
            std::to_string(p) + "; this runtime implements mean (1) and CLS "
            "(2) pooling only. Supply 1_Pooling/config.json if the model "
            "pools differently.");
    json c = json::object();
    c["word_embedding_dimension"] = hidden;
    c["pooling_mode_cls_token"] = p == 2;
    c["pooling_mode_mean_tokens"] = p == 1;
    c["pooling_mode_max_tokens"] = false;
    c["pooling_mode_mean_sqrt_len_tokens"] = false;
    return c;
}

/// One GGUF vocab token -> the WordPiece piece a BERT vocab.txt holds.
///
/// llama.cpp stores a WordPiece vocab in sentencepiece's spelling: a word-start
/// piece carries U+2581 and a continuation piece carries no marker, while the
/// HF file uses no marker on a word start and '##' on a continuation. The
/// mapping is therefore reversible, and is verified exactly against the base
/// repos' own vocab.txt for bge-base and nomic-embed-text.
std::string restore_wordpiece(const std::string& t) {
    if (t.size() >= 3 && t.front() == '[' && t.back() == ']') {
        const std::string inner = t.substr(1, t.size() - 2);
        if (inner == "PAD" || inner == "UNK" || inner == "CLS" ||
            inner == "SEP" || inner == "MASK" ||
            inner.rfind("unused", 0) == 0)
            return t;
    }
    static const std::string kSpmSpace = "\xE2\x96\x81";   // U+2581
    if (t.rfind(kSpmSpace, 0) == 0) return t.substr(kSpmSpace.size());
    return "##" + t;
}

/// Build the packer's checkpoint from a GGUF, in <dir>/.npue_gguf. Returns the
/// staging directory; the caller removes it once the container is written.
std::string stage_gguf_checkpoint(const fs::path& dir, const std::string& gguf_path) {
    const fs::path stage = dir / ".npue_gguf";
    std::error_code ec;
    fs::remove_all(stage, ec);
    fs::create_directories(stage, ec);
    if (ec)
        throw std::runtime_error("NpueEmbedding: cannot create " + stage.string() +
                                 ": " + ec.message());

    open_qwen36::GgufFile g(gguf_path);
    const std::string arch = g.kv_str("general.architecture");
    const std::string model_type =
        arch == "bert" ? "bert" : arch == "nomic-bert" ? "nomic_bert" : "";
    if (model_type.empty())
        throw std::runtime_error(
            "NpueEmbedding: GGUF architecture '" + arch + "' is not one this "
            "backend packs (supported: bert, nomic-bert).");

    // The tensor is the authority on the two vocabulary sizes: nomic's config
    // says 30528 while its GGUF (and its tokenizer) stop at 30522, and a
    // container built from the larger number would describe rows it does not
    // hold.
    auto rows = [&](const char* name, uint64_t fallback) -> uint64_t {
        if (!g.has(name)) return fallback;
        const auto& t = g.tensor(name);
        return t.dims.size() >= 2 ? t.dims[1] : fallback;
    };
    const uint64_t vocab = rows("token_embd.weight", 0);
    const uint64_t type_vocab = rows("token_types.weight", 0);
    if (!vocab)
        throw std::runtime_error("NpueEmbedding: the GGUF has no token_embd.weight");

    // Tensors, in the packer's own names.
    std::vector<StTensor> tensors;
    std::vector<std::string> skipped;
    for (const std::string& name : g.tensor_names()) {
        const std::string hf = hf_embed_name(arch, name);
        if (hf.empty()) { skipped.push_back(name); continue; }
        const auto& t = g.tensor(name);
        StTensor st;
        st.name = hf;
        st.gguf_name = name;
        for (auto it = t.dims.rbegin(); it != t.dims.rend(); ++it)
            st.shape.push_back(static_cast<int64_t>(*it));
        st.count = t.nvalues();
        tensors.push_back(std::move(st));
    }
    if (!skipped.empty()) {
        std::string names;
        for (const auto& n : skipped) names += (names.empty() ? "" : ", ") + n;
        header_print("NPUE", "GGUF tensors not used by the packer: " + names);
    }
    write_f32_safetensors(stage / "model.safetensors", tensors,
                          [&](const StTensor& t) { return g.f32_all(t.gguf_name); });

    // config.json: the GGUF is the authority for the geometry, so its values
    // overwrite whatever the repo shipped. Other keys the base config carries
    // (architectures, dropouts) are kept -- nothing reads them, but dropping
    // them would make the staging file a worse description of the model.
    json cfg = json::object();
    if (fs::is_regular_file(dir / "config.json", ec)) {
        std::ifstream cf(dir / "config.json");
        std::stringstream ss;
        ss << cf.rdbuf();
        const json parsed = json::parse(ss.str(), nullptr, false);
        if (parsed.is_object()) cfg = parsed;
    }
    cfg.update(derive_embed_config(g, arch, model_type, vocab, type_vocab));
    std::ofstream(stage / "config.json") << cfg.dump(1) << "\n";

    // 1_Pooling/config.json: prefer the GGUF's own pooling_type, which is what
    // llama.cpp used for this model; fall back to the checkpoint's file when
    // the GGUF does not name one.
    const fs::path pool_dir = stage / "1_Pooling";
    fs::create_directories(pool_dir, ec);
    if (g.kv_has(arch + ".pooling_type")) {
        const uint64_t hidden = req_u64(g, arch + ".embedding_length");
        std::ofstream(pool_dir / "config.json")
            << derive_pooling_config(g, arch, hidden).dump(1) << "\n";
    } else if (fs::is_regular_file(dir / "1_Pooling" / "config.json", ec)) {
        fs::copy_file(dir / "1_Pooling" / "config.json",
                      pool_dir / "config.json", fs::copy_options::overwrite_existing, ec);
        if (ec)
            throw std::runtime_error("NpueEmbedding: cannot copy 1_Pooling/config.json: " +
                                     ec.message());
    } else {
        throw std::runtime_error(
            "NpueEmbedding: the GGUF names no " + arch + ".pooling_type and "
            "there is no 1_Pooling/config.json -- cannot tell whether this "
            "model pools by mean or by CLS, and the two are different models.");
    }

    // vocab.txt: the base repo's own file when present; otherwise rebuilt from
    // the GGUF's WordPiece vocab (see restore_wordpiece).
    if (fs::is_regular_file(dir / "vocab.txt", ec)) {
        fs::copy_file(dir / "vocab.txt", stage / "vocab.txt",
                      fs::copy_options::overwrite_existing, ec);
        if (ec)
            throw std::runtime_error("NpueEmbedding: cannot copy vocab.txt: " + ec.message());
    } else {
        const std::string tok_model =
            g.kv_has("tokenizer.ggml.model") ? g.kv_str("tokenizer.ggml.model") : "";
        if (tok_model != "bert")
            throw std::runtime_error(
                "NpueEmbedding: this GGUF's tokenizer is '" + tok_model +
                "', not WordPiece, and there is no vocab.txt beside it. The "
                "packer needs the WordPiece vocab the checkpoint shipped; "
                "fetch vocab.txt from the base repository named in the GGUF "
                "repo's README.");
        const std::vector<std::string> tokens = g.kv_str_array("tokenizer.ggml.tokens");
        if (tokens.empty())
            throw std::runtime_error(
                "NpueEmbedding: the GGUF carries no tokenizer.ggml.tokens and "
                "there is no vocab.txt beside it");
        std::ofstream vf(stage / "vocab.txt", std::ios::binary | std::ios::trunc);
        for (const std::string& t : tokens)
            vf << restore_wordpiece(t) << '\n';
    }

    header_print("NPUE", "staged the GGUF as a " +
                         std::to_string(tensors.size()) + "-tensor F32 checkpoint in " +
                         stage.string());
    return stage.string();
}

/// The `.npue` container: the checkpoint's weights, pre-tiled for the array.
///
/// PACKED ON FIRST RUN when it is absent, from the model author's own files in
/// the same directory. NOTHING IS RE-HOSTED: the weights a user gets are the
/// author's bytes with the author's hash, and the container -- those same
/// weights pre-tiled for the array -- is produced locally from them.
///
/// The packing DECISION (which architecture, which tile width, which pooling,
/// which source repository) is npue::prepare_model_auto() in the engine and not
/// here. That separation is the point: a second copy of that decision in this
/// repository would have to agree byte for byte with upstream's, and upstream's
/// byte-identity gate would stop being evidence about these containers the
/// moment the two diverged. So this calls one function and chooses nothing.
std::string find_container(const std::filesystem::path& dir, const json& info) {
    namespace fs = std::filesystem;
    if (info.contains("npue_container") && info["npue_container"].is_string()) {
        const fs::path p = dir / info["npue_container"].get<std::string>();
        if (fs::is_regular_file(p)) return p.string();
    }
    std::vector<fs::path> found;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end;
         it.increment(ec)) {
        if (it->path().extension() == ".npue") found.push_back(it->path());
    }
    if (found.size() == 1) return found.front().string();
    if (found.size() > 1) {
        // Two containers are two different datapaths (bf16 vs int8) or two
        // different sequence lengths, and they are not interchangeable.
        // Picking one by sort order is how upstream's tasks/0104 nearly
        // shipped the wrong datapath; name the choice instead.
        std::string names;
        for (const auto& f : found)
            names += (names.empty() ? "" : ", ") + f.filename().string();
        throw std::runtime_error(
            "NpueEmbedding: " + dir.string() + " holds several .npue "
            "containers (" + names + ") and they are not interchangeable -- "
            "they differ in datapath or sequence length. Name one with "
            "\"npue_container\" in the model entry.");
    }
    // Nothing packed yet: do it now, once, from the checkpoint beside us. A
    // first run therefore costs a pack -- tens of seconds for a 100M model --
    // and every run after it mmaps the result.
    //
    // A GGUF install has no model.safetensors: the packers read a HuggingFace
    // checkpoint, so one is staged from the GGUF first and the packer's own
    // input contract is left untouched.
    const bool have_st = fs::is_regular_file(dir / "model.safetensors", ec);
    const std::string gguf = have_st ? std::string() : find_gguf(dir);
    std::string staged;
    if (!have_st) {
        if (gguf.empty())
            throw std::runtime_error(
                "NpueEmbedding: " + dir.string() + " has neither a .npue "
                "container nor a checkpoint to pack one from (model.safetensors "
                "or a .gguf). The container is the checkpoint's weights "
                "pre-tiled for the array; it is produced locally from the "
                "model author's own files, which means those files have to be "
                "here.");
        header_print("NPUE", "no container yet -- packing one from the GGUF in " +
                             dir.string() + " (first run only)");
        staged = stage_gguf_checkpoint(dir, gguf);
    } else {
        if (!fs::is_regular_file(dir / "config.json", ec))
            throw std::runtime_error(
                "NpueEmbedding: " + dir.string() + " has neither a .npue "
                "container nor a config.json to pack one from. The container "
                "is the checkpoint's weights pre-tiled for the array; it is "
                "produced locally from the model author's own files, which "
                "means those files have to be here.");
        header_print("NPUE", "no container yet -- packing one from the checkpoint "
                             "in " + dir.string() + " (first run only)");
    }
    npue::PrepareOptions po;
    po.checkpoint_dir = staged.empty() ? dir.string() : staged;
    // The container is named after the MODEL directory, not after the staging
    // directory the GGUF was unpacked into.
    po.out_path = (dir / (dir.filename().string() + ".npue")).string();
    // WHICH REPOSITORY THESE WEIGHTS CAME FROM, and this tree is the
    // authority on it: ModelDownloader fetched them from the entry's own
    // `url`. Upstream's packer falls back to a CHECKPOINT.json side-car, which
    // is a file NpuEmbeddings writes and a HuggingFace checkpoint does not
    // have -- so without this, packing a model fetched by `oflm pull` refuses.
    // It refuses rather than guessing because a container that misattributes
    // its own weights is a licensing statement, and the fix is to tell it,
    // not to loosen it.
    if (info.contains("npue_source_repo") &&
        info["npue_source_repo"].is_string()) {
        po.source_repo = info["npue_source_repo"].get<std::string>();
    } else if (info.contains("url") && info["url"].is_string()) {
        const std::string url = info["url"].get<std::string>();
        const std::string host = "huggingface.co/";
        const size_t i = url.find(host);
        if (i != std::string::npos) po.source_repo = url.substr(i + host.size());
    }
    // tile_n is a property of the MODEL's widths, not a preference: the design
    // asserts N % (tile_n * n_cols) == 0, so bge-large's N in {1024,3072,4096}
    // needs 32 where hidden-768 models take 48. The model entry names it when
    // it is not the default.
    if (info.contains("npue_tile_n") && info["npue_tile_n"].is_number_integer())
        po.tile_n = info["npue_tile_n"].get<int>();
    po.log = [](const std::string& s) { header_print("NPUE", s); };
    std::string out;
    try {
        out = npue::prepare_model_auto(po);
    } catch (...) {
        if (!staged.empty()) fs::remove_all(staged, ec);
        throw;
    }
    if (!staged.empty()) fs::remove_all(staged, ec);
    return out;
}

/// The compiled design set. Same two-tier convention open_embedding uses: a
/// model-local copy wins, so a brand-new geometry can ship its kernels inside
/// its own model directory before it is promoted to a maintained family;
/// otherwise the app's installed xclbin tree.
///
/// The family name is a GEOMETRY, not a model: one BERT-768 set also serves
/// gte-multilingual and nomic, because their GEMM shapes match bit for bit.
std::string find_artifacts(const std::filesystem::path& dir, const json& info) {
    namespace fs = std::filesystem;
    auto has_design = [](const fs::path& d) {
        std::error_code ec;
        return fs::is_regular_file(d / "gemm_rtp" / "design.json", ec);
    };

    // 1) A model can ship its own design set alongside the checkpoint. This lets a
    //    brand-new geometry be promoted model-by-model before it joins a family.
    const fs::path local = dir / "npue_designs";
    if (has_design(local)) return local.string();

    // 2) Otherwise the design set is keyed by GEMM geometry (family) and ships
    //    pre-built with the package. The family name is a GEOMETRY, not a model:
    //    one BERT-768 set also serves gte-multilingual and nomic, because their
    //    GEMM shapes match bit for bit. Scan EVERY known xclbin root so the set
    //    is found wherever the package put it -- the user tree (flm-add install),
    //    the relocatable install prefix (chat kernels), or the separately-shipped
    //    embedding design tree -- without a manual symlink or FLM_XCLBIN_PATH.
    if (info.contains("npue_design_family") &&
        info["npue_design_family"].is_string()) {
        const std::string family =
            info["npue_design_family"].get<std::string>();
        for (const std::string& root : utils::xclbin_roots()) {
            const fs::path cand = fs::path(root) / "xclbins" / family;
            if (has_design(cand)) return cand.string();
        }
    }
    throw std::runtime_error(
        "NpueEmbedding: no design set for this model.\n"
        "Looked for " + local.string() + "/gemm_rtp/design.json and, for the "
        "family named by the model entry's \"npue_design_family\", under every "
        "installed xclbin tree.\n"
        "A design set is four instruction streams over one xclbin, compiled "
        "for this model's GEMM geometry -- one set serves every model whose "
        "shapes match, which is why it is keyed by geometry rather than by "
        "model name.\n"
        "The design sets ship pre-built in the distributed package (the "
        "embedding xclbin tree). If missing, see "
        "npu_offload/gemm_rtp/README.md for the one command that builds this "
        "family.");
}

}  // namespace

struct NpueEmbedding::Impl {
    std::string tag;
    std::vector<std::string> prompt_names;
    std::unique_ptr<npue::enc::Embedder> emb;
};

NpueEmbedding::NpueEmbedding(oflm_rt::device* npu_device_inst, std::string tag)
    : AutoEmbeddingModel(npu_device_inst, tag), impl_(new Impl) {
    impl_->tag = std::move(tag);
}

NpueEmbedding::~NpueEmbedding() = default;

std::vector<std::string> NpueEmbedding::prompt_names() const {
    return impl_->prompt_names;
}

void NpueEmbedding::load_model(std::string model_path, json model_info,
                               bool enable_preemption) {
    (void)enable_preemption;   // the engine holds one hw_context, always
    namespace fs = std::filesystem;
    this->model_path = std::move(model_path);

    const fs::path dir(this->model_path);
    npue::enc::EmbedderOptions opt;
    opt.npue_path = find_container(dir, model_info);
    opt.artifacts_dir = find_artifacts(dir, model_info);
    // Threads and lanes are the host-side shape of the work, not model facts,
    // so they come from model_info if it says anything and from sensible
    // defaults otherwise. `lanes` overlaps one lane's host work with another's
    // array work; the array serialises either way.
    opt.threads = int_or(model_info, "npue_threads",
                         std::max(1u, std::thread::hardware_concurrency()));
    opt.lanes = int_or(model_info, "npue_lanes", 2);

    impl_->emb = std::make_unique<npue::enc::Embedder>(opt);
    impl_->prompt_names = impl_->emb->prompt_names();
    this->is_model_loaded = true;

    header_print("NPUE", "loaded " + impl_->emb->name() + " (" +
                         impl_->emb->source_repo() + "), hidden " +
                         std::to_string(impl_->emb->hidden()) + ", seq " +
                         std::to_string(impl_->emb->seq()) + ", " +
                         impl_->emb->datapath());
}

std::vector<float> NpueEmbedding::embed(std::string& text,
                                        embedding_task_type_t task_type) {
    if (!impl_->emb)
        throw std::runtime_error(
            "NpueEmbedding: embed() called before load_model()");
    std::vector<std::string> one{text};
    return impl_->emb->embed(one, prompt_for(impl_->prompt_names, task_type));
}

std::vector<float> NpueEmbedding::embed_batch(
        const std::vector<std::string>& texts,
        embedding_task_type_t task_type, int64_t* tokens) {
    if (!impl_->emb)
        throw std::runtime_error(
            "NpueEmbedding: embed_batch() called before load_model()");
    return impl_->emb->embed(texts, prompt_for(impl_->prompt_names, task_type),
                             tokens);
}

int64_t NpueEmbedding::hidden() const {
    return impl_->emb ? impl_->emb->hidden() : 0;
}

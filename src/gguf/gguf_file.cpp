/// \file gguf_file.cpp
/// \brief GGUF parsing and the per-token embedding dequant (see gguf_file.hpp).
#include "gguf/gguf_file.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <regex>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace open_qwen36 {

namespace {

namespace fs = std::filesystem;

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("gguf: " + what); }

uint64_t rd_u64(const uint8_t*& p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    p += 8;
    return v;
}
uint32_t rd_u32(const uint8_t*& p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    p += 4;
    return v;
}
uint8_t rd_u8(const uint8_t*& p) { return *p++; }

std::string rd_str(const uint8_t*& p, const uint8_t* end) {
    const uint64_t n = rd_u64(p);
    if (n > static_cast<uint64_t>(end - p)) fail("string runs past EOF");
    std::string s(reinterpret_cast<const char*>(p), n);
    p += n;
    return s;
}

// One metadata value, recorded as its string form (the config derivation's
// view); arrays and other composite values are skipped wholesale.
void rd_value(uint32_t t, const uint8_t*& p, const uint8_t* end, std::string& out) {
    switch (t) {
        case 0: out = std::to_string(rd_u8(p)); break;
        case 1: out = std::to_string(static_cast<int8_t>(rd_u8(p))); break;
        case 2: { uint16_t v; std::memcpy(&v, p, 2); p += 2; out = std::to_string(v); break; }
        case 3: { int16_t v; std::memcpy(&v, p, 2); p += 2; out = std::to_string(v); break; }
        case 4: out = std::to_string(rd_u32(p)); break;
        case 5: { int32_t v; std::memcpy(&v, p, 4); p += 4; out = std::to_string(v); break; }
        case 6: { float v; std::memcpy(&v, p, 4); p += 4; out = std::to_string(v); break; }
        case 7: out = std::to_string(rd_u8(p) != 0); break;
        case 8: out = rd_str(p, end); break;
        case 9: {  // array: element type + count
            const uint32_t et = rd_u32(p);
            const uint64_t n = rd_u64(p);
            // A SMALL array is KEPT, comma-joined: llama.cpp writes per-layer counts as an
            // array (`lfm2.attention.head_count_kv` is one value per block, 0 for the
            // short-conv layers), and the engine needs the attention layers' value. A big
            // one (the tokenizer's vocab) is skipped wholesale, as it always was: keeping it
            // would put megabytes into `meta_` for nothing.
            const bool keep = n <= 64;
            std::string joined;
            for (uint64_t i = 0; i < n; ++i) {
                std::string one;
                rd_value(et, p, end, one);
                if (keep) {
                    if (i) joined += ',';
                    joined += one;
                }
            }
            if (keep) out = joined;
            break;
        }
        case 10: out = std::to_string(rd_u64(p)); break;
        case 11: { int64_t v; std::memcpy(&v, p, 8); p += 8; out = std::to_string(v); break; }
        case 12: { double v; std::memcpy(&v, p, 8); p += 8; out = std::to_string(v); break; }
        default: fail("unknown metadata type " + std::to_string(t));
    }
}

/// Exact tensor byte count for a value count (0 when the type's size is
/// unknown to this reader; the pack ops refuse those with a proper message).
uint64_t gguf_type_bytes(uint32_t t, uint64_t values) {
    // block quants: 18 B / 32 q4_0 | 20 B / 32 q4_1 | 22 B / 32 q5_0 |
    // 24 B / 32 q5_1 | 34 B / 32 q8_0 | 144 B / 256 q4_k | 176 B / 256 q5_k |
    // 210 B / 256 q6_k. Unsloth's UD-Q4_K_XL mixes Q4_K with Q5_1 on the MoE
    // down projections, so both 5-bit families must be sized here as well.
    switch (t) {
        case 0: return values * 4;                        // f32
        case 1: case 30: return values * 2;               // f16 / bf16
        case 2: return values / 32 * 18;                  // q4_0
        case 3: return values / 32 * 20;                  // q4_1
        case 6: return values / 32 * 22;                  // q5_0
        case 7: return values / 32 * 24;                  // q5_1
        case 8: return values / 32 * 34;                  // q8_0
        case 12: return values / 256 * 144;               // q4_k
        case 14: return values / 256 * 210;               // q6_k
        case 13: return values / 256 * 176;               // q5_k
        case 20: return values / 32 * 18;                 // iq4_nl
        case 23: return values / 256 * 136;               // iq4_xs
        default: return 0;                                // unsupported
    }
}

// llama.cpp's iq4_nl codebook: the sixteen values a 4-bit code indexes. iq4_xs
// uses the same table (it is a finer-grained scale on top of it).
static const int8_t kIQ4NLValues[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

inline float bf16_to_f32(uint16_t u) {
    uint32_t w = static_cast<uint32_t>(u) << 16;
    float f;
    std::memcpy(&f, &w, 4);
    return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);  // round to nearest even
}

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

struct ParsedHeader {
    std::map<std::string, std::string> meta;
    std::map<std::string, GgufFile::TensorInfo> tensors;
    size_t data_base = 0;
    uint64_t split_no = 0;
    uint64_t split_count = 0;
};

uint64_t meta_u64(const std::map<std::string, std::string>& meta, const std::string& key) {
    auto it = meta.find(key);
    if (it == meta.end()) return 0;
    try {
        return std::stoull(it->second);
    } catch (const std::logic_error&) {
        fail("metadata " + key + " is not a number (" + it->second + ")");
    }
}

/// Parse one shard's header. Every shard carries its own alignment and tensor
/// offset table; `split.no` / `split.count` say which part it is when the model
/// was exported in pieces.
ParsedHeader read_header(const uint8_t* map, size_t size, const std::string& path) {
    ParsedHeader h;
    if (size < 24 || map[0] != 'G' || map[1] != 'G' || map[2] != 'U' || map[3] != 'F')
        fail(path + " is not a GGUF file");
    const uint8_t* end = map + size;
    const uint8_t* q = map + 4;
    const uint32_t version = rd_u32(q);
    if (version < 2) fail(path + ": GGUF v" + std::to_string(version) + " unsupported (need v2+)");
    const uint64_t ntensors = rd_u64(q);
    const uint64_t nkv = rd_u64(q);
    for (uint64_t i = 0; i < nkv; ++i) {
        const std::string key = rd_str(q, end);
        const uint32_t t = rd_u32(q);
        std::string v;
        rd_value(t, q, end, v);
        h.meta[key] = v;
    }
    h.split_no = meta_u64(h.meta, "split.no");
    h.split_count = meta_u64(h.meta, "split.count");
    uint64_t align = 32;
    auto it = h.meta.find("general.alignment");
    if (it != h.meta.end()) {
        try {
            align = std::stoull(it->second);
        } catch (const std::logic_error&) {
            fail(path + ": general.alignment is not a number (" + it->second + ")");
        }
    }
    if (align == 0 || (align & (align - 1)) != 0)
        fail(path + ": general.alignment must be a non-zero power of two (got " + std::to_string(align) + ")");
    const uint8_t* tq = q;
    for (uint64_t i = 0; i < ntensors; ++i) {
        const std::string name = rd_str(tq, end);
        const uint32_t nd = rd_u32(tq);
        GgufFile::TensorInfo t;
        t.dims.resize(nd);
        for (uint32_t d = 0; d < nd; ++d) t.dims[d] = rd_u64(tq);
        t.type = static_cast<GgufFile::Type>(rd_u32(tq));
        t.offset = rd_u64(tq);
        h.tensors.emplace(name, std::move(t));
    }
    // The data section starts AFTER the tensor-info table, aligned. Offsets in
    // each tensor are relative to THIS shard's aligned data section, not to the
    // start of the file -- checking against `size` alone hid truncated shards.
    h.data_base = static_cast<size_t>(tq - map);
    h.data_base += static_cast<size_t>((align - h.data_base % align) % align);
    if (h.data_base > size) fail(path + ": GGUF header runs past EOF");
    const size_t data_bytes = size - h.data_base;
    for (const auto& [name, t] : h.tensors) {
        const uint64_t nb = gguf_type_bytes(static_cast<uint32_t>(t.type), t.nvalues());
        if (nb && (t.offset > data_bytes || nb > data_bytes - t.offset))
            fail("tensor " + name + " runs past EOF in " + path);
    }
    return h;
}

std::string split_name(const std::string& stem, uint64_t i, uint64_t count) {
    const auto five = [](uint64_t v) {
        std::string s = std::to_string(v);
        return std::string(5 - std::min<size_t>(5, s.size()), '0') + s;
    };
    return stem + "-" + five(i) + "-of-" + five(count) + ".gguf";
}

uint64_t shard_count_from_name(const std::string& path) {
    static const std::regex part(R"(-(\d{5})-of-(\d{5})\.gguf$)", std::regex::icase);
    std::smatch m;
    const std::string leaf = fs::path(path).filename().string();
    return std::regex_search(leaf, m, part) ? std::stoull(m[2].str()) : 0;
}

/// All shard paths for a model, in split order. The installed form is
/// `model.gguf` plus `model.gguf-00002-of-00004` siblings; a source repo's
/// `Name-00001-of-00004.gguf` spelling is understood too.
std::vector<std::string> split_paths(const std::string& path, uint64_t count) {
    if (count < 1 || count > 100000) fail(path + ": split.count " + std::to_string(count) + " is implausible");
    const fs::path fp(path);
    const std::string leaf = fp.filename().string();
    static const std::regex part(R"(^(.*)-(\d{5})-of-(\d{5})\.gguf$)", std::regex::icase);
    std::smatch m;
    if (std::regex_match(leaf, m, part)) {
        const std::string stem = m[1].str();
        const uint64_t idx = std::stoull(m[2].str()), named = std::stoull(m[3].str());
        if (named != count) fail(path + ": split.count is " + std::to_string(count) +
                                " but the name says " + std::to_string(named));
        std::vector<std::string> out(count);
        for (uint64_t i = 1; i <= count; ++i)
            out[i - 1] = (fp.parent_path() / split_name(stem, i, count)).string();
        if (idx < 1 || idx > count) fail(path + ": shard number " + std::to_string(idx) + " is out of range");
        out[idx - 1] = path;                         // the path the caller actually opened
        return out;
    }
    static const std::regex sibling(R"(^model\.gguf-(\d{5})-of-(\d{5})$)", std::regex::icase);
    if (leaf == "model.gguf" || std::regex_match(leaf, m, sibling)) {
        std::vector<std::string> out(count);
        out[0] = (fp.parent_path() / "model.gguf").string();
        for (uint64_t i = 2; i <= count; ++i) {
            // Installed siblings keep the .gguf on the PRIMARY file only:
            // model.gguf-00002-of-00004 (no trailing .gguf).
            std::string sib = split_name("model.gguf", i, count);
            sib.resize(sib.size() - 5);                 // drop the appended ".gguf"
            out[i - 1] = (fp.parent_path() / sib).string();
        }
        if (std::regex_match(leaf, m, sibling)) {
            const uint64_t idx = std::stoull(m[1].str()), named = std::stoull(m[2].str());
            if (named != count || idx < 2 || idx > count)
                fail(path + ": shard name disagrees with split.count " + std::to_string(count));
            out[idx - 1] = path;
        }
        return out;
    }
    fail(path + ": a split GGUF must be model.gguf with numbered siblings, or "
         "Name-00001-of-000NN.gguf");
}

}  // namespace

const char* GgufFile::type_name(Type t) {
    switch (t) {
        case Type::F32: return "F32"; case Type::F16: return "F16"; case Type::Q4_0: return "Q4_0";
        case Type::Q4_1: return "Q4_1"; case Type::Q5_0: return "Q5_0"; case Type::Q5_1: return "Q5_1";
        case Type::Q8_0: return "Q8_0"; case Type::Q8_1: return "Q8_1"; case Type::Q2_K: return "Q2_K";
        case Type::Q3_K: return "Q3_K"; case Type::Q4_K: return "Q4_K"; case Type::Q5_K: return "Q5_K";
        case Type::Q6_K: return "Q6_K"; case Type::Q8_K: return "Q8_K"; case Type::IQ2_XXS: return "IQ2_XXS";
        case Type::IQ2_XS: return "IQ2_XS"; case Type::IQ3_XXS: return "IQ3_XXS"; case Type::IQ1_S: return "IQ1_S";
        case Type::IQ4_NL: return "IQ4_NL"; case Type::IQ3_S: return "IQ3_S"; case Type::IQ2_S: return "IQ2_S";
        case Type::IQ4_XS: return "IQ4_XS"; case Type::I8: return "I8"; case Type::I16: return "I16";
        case Type::I32: return "I32"; case Type::I64: return "I64"; case Type::F64: return "F64";
        case Type::IQ1_M: return "IQ1_M"; case Type::BF16: return "BF16";
    }
    return "type?";
}

GgufFile::GgufFile(const std::string& path) : path_(path) {
    // Peek at the named file: its split.count (or its -NNNNN-of-NNNNN name)
    // decides whether we open one mapping or a whole set. Unsloth's big UD
    // exports are split into several GGUFs, each with its own tensor table;
    // llama.cpp offsets are relative to that shard's own aligned data section.
    std::vector<std::string> paths{path};
    {
        Shard probe;
        map_shard(path, probe);
        try {
            ParsedHeader h = read_header(probe.map, probe.size, path);
            const uint64_t count = h.split_count ? h.split_count : shard_count_from_name(path);
            if (count > 1) paths = split_paths(path, count);
            else if (h.split_no > 1 || shard_count_from_name(path) > 1)
                fail(path + ": a shard without split.count cannot be opened alone");
        } catch (...) {
            unmap_shard(probe);
            throw;
        }
        unmap_shard(probe);
    }
    // Unsloth writes split.no 0-based (0..count-1), llama.cpp's writer uses
    // 1-based. The -NNNNN-of-NNNNN file name fixes the order, so take the
    // convention from the first shard that declares it and require the rest to
    // agree. -1 means no shard has declared it yet.
    int no_base = -1;
    try {
        for (size_t i = 0; i < paths.size(); ++i) {
            Shard s;
            map_shard(paths[i], s);
            try {
                ParsedHeader h = read_header(s.map, s.size, paths[i]);
                s.data_base = h.data_base;
                if (h.meta.count("split.no")) {
                    if (no_base < 0) {
                        if (h.split_no == 0) no_base = 0;
                        else if (h.split_no == 1) no_base = 1;
                        else fail(paths[i] + ": split.no " + std::to_string(h.split_no) +
                                  " is neither 0- nor 1-based");
                    }
                    const uint64_t want = i + static_cast<uint64_t>(no_base);
                    if (h.split_no != want)
                        fail(paths[i] + ": split.no is " + std::to_string(h.split_no) +
                             ", expected " + std::to_string(want));
                }
                if (h.split_count && h.split_count != paths.size())
                    fail(paths[i] + ": split.count is " + std::to_string(h.split_count) +
                         ", expected " + std::to_string(paths.size()));
                if (i == 0) meta_ = std::move(h.meta);
                else for (const auto& [k, v] : h.meta) meta_.emplace(k, v);
                for (auto& [name, t] : h.tensors) {
                    t.shard = static_cast<uint32_t>(i);
                    if (!tensors_.emplace(name, std::move(t)).second)
                        fail("tensor " + name + " appears in more than one shard of " + path);
                }
                shards_.push_back(std::move(s));
            } catch (...) {
                unmap_shard(s);
                throw;
            }
        }
        sandwich_ = tensors_.count("blk.0.post_ffw_norm.weight") != 0;
        // The pre-FFN norm: llama.cpp calls it ffn_norm (llama, qwen3) or
        // post_attention_norm (gemma, qwen35) -- the file says which.
        post_norm_ = tensors_.count("blk.0.post_attention_norm.weight") != 0;
    } catch (...) {
        for (auto& s : shards_) unmap_shard(s);
        shards_.clear();
        throw;
    }
}

GgufFile::~GgufFile() {
    for (auto& s : shards_) unmap_shard(s);
}

void GgufFile::drop_pages() {
    for (auto& s : shards_) drop_shard_pages(s);
}

void GgufFile::map_shard(const std::string& path, Shard& s) {
    s.path = path;
#ifdef _WIN32
    HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) throw std::runtime_error("gguf: cannot open " + path);
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(f, &sz)) { CloseHandle(f); throw std::runtime_error("gguf: size of " + path); }
    HANDLE m = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) { CloseHandle(f); throw std::runtime_error("gguf: cannot map " + path); }
    const void* p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (!p) { CloseHandle(m); CloseHandle(f); throw std::runtime_error("gguf: cannot map view of " + path); }
    s.file = f;
    s.mapping = m;
    s.map = static_cast<const uint8_t*>(p);
    s.size = static_cast<size_t>(sz.QuadPart);
#else
    s.fd = ::open(path.c_str(), O_RDONLY);
    if (s.fd < 0) throw std::runtime_error("gguf: cannot open " + path);
    struct stat st;
    if (fstat(s.fd, &st) != 0) { ::close(s.fd); s.fd = -1; throw std::runtime_error("gguf: size of " + path); }
    void* p = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, s.fd, 0);
    if (p == MAP_FAILED) { ::close(s.fd); s.fd = -1; throw std::runtime_error("gguf: cannot map " + path); }
    s.map = static_cast<const uint8_t*>(p);
    s.size = static_cast<size_t>(st.st_size);
#endif
}

void GgufFile::unmap_shard(Shard& s) {
#ifdef _WIN32
    if (s.map) UnmapViewOfFile(const_cast<uint8_t*>(s.map));
    if (s.mapping) CloseHandle(static_cast<HANDLE>(s.mapping));
    if (s.file) CloseHandle(static_cast<HANDLE>(s.file));
#else
    if (s.map) munmap(const_cast<uint8_t*>(s.map), s.size);
    if (s.fd >= 0) ::close(s.fd);
#endif
    s.map = nullptr;
    s.size = 0;
    s.data_base = 0;
#ifdef _WIN32
    s.file = nullptr;
    s.mapping = nullptr;
#else
    s.fd = -1;
#endif
}

void GgufFile::drop_shard_pages(Shard& s) {
#ifdef _WIN32
    UnmapViewOfFile(const_cast<uint8_t*>(s.map));
    const void* p = MapViewOfFile(static_cast<HANDLE>(s.mapping), FILE_MAP_READ, 0, 0, 0);
    if (!p) throw std::runtime_error("gguf: cannot remap " + s.path);
    s.map = static_cast<const uint8_t*>(p);
#else
    madvise(const_cast<uint8_t*>(s.map), s.size, MADV_DONTNEED);
#endif
}

std::string GgufFile::gguf_name(const std::string& name, bool sandwich, bool post_norm) {
    // The manifest's packing plan names tensors the q4nx/HF way; a GGUF uses
    // llama.cpp's conventions (utilities/q4nx-build/configs/*.json hold the
    // same map on the conversion side).
    if (name == "model.embed_tokens.weight") return "token_embd.weight";
    // LFM2's container spells the table `model.token_embd.weight` (utilities/q4nx-build/
    // configs/lfm2.json maps it to the same `token_embd.weight`), so both container
    // spellings resolve to the one GGUF tensor.
    if (name == "model.token_embd.weight") return "token_embd.weight";
    if (name == "model.norm.weight") return "output_norm.weight";
    if (name == "lm_head.weight") return "output.weight";
    std::string n = name;
    // Both container spellings: the dense families use `model.layers.{l}.`, the MoE
    // container (utilities/q4nx-build/q4nx/models/qwen35moe.py) uses `model.layer.{l}.`.
    // They name the same GGUF tensors, so both resolve to `blk.{l}.` -- exactly as the
    // two token-table spellings above do.
    const std::string layers = "model.layers.";
    const std::string layer = "model.layer.";
    if (n.rfind(layers, 0) == 0 || n.rfind(layer, 0) == 0) {
        n = "blk." + n.substr(n.rfind(layers, 0) == 0 ? layers.size() : layer.size());
        const auto dot = [&](const std::string& from, const std::string& to) {
            const std::string a = "." + from;
            const auto pos = n.find(a);
            if (pos != std::string::npos) n.replace(pos, a.size(), to);
        };
        dot("self_attn.q_proj", ".attn_q");
        dot("self_attn.k_proj", ".attn_k");
        dot("self_attn.v_proj", ".attn_v");
        dot("self_attn.o_proj", ".attn_output");
        dot("self_attn.q_norm", ".attn_q_norm");
        dot("self_attn.k_norm", ".attn_k_norm");
        dot("input_layernorm", ".attn_norm");
        // The qwen35 family's linear attention (Gated DeltaNet), which llama.cpp
        // names ssm_*/attn_qkv (the map the conversion side holds in
        // utilities/q4nx-build/configs/qwen3.5_*.json). The .bf16 spelling is
        // the container's own (the tensor is bf16 there), so it is mapped first.
        dot("linear_attn.qkv_proj", ".attn_qkv");
        dot("self_attn.gate_proj", ".attn_gate");
        dot("linear_attn.ssm_alpha_proj.bf16", ".ssm_alpha");
        dot("linear_attn.ssm_alpha_proj", ".ssm_alpha");
        dot("linear_attn.ssm_beta_proj.bf16", ".ssm_beta");
        dot("linear_attn.ssm_beta_proj", ".ssm_beta");
        dot("linear_attn.ssm_a", ".ssm_a");
        dot("linear_attn.ssm_dt.bias", ".ssm_dt.bias");
        dot("linear_attn.ssm_conv1d.weight", ".ssm_conv1d.weight");
        dot("linear_attn.ssm_norm.weight", ".ssm_norm.weight");
        dot("linear_attn.ssm_out_proj.weight", ".ssm_out.weight");
        // HF's post_attention_layernorm is the pre-FFN norm in Llama, but the
        // real post-attention norm in Gemma and in qwen35 -- the file says
        // which (it has one name or the other, never both).
        dot("post_attention_layernorm", (post_norm || sandwich) ? ".post_attention_norm" : ".ffn_norm");
        dot("pre_feedforward_layernorm", ".ffn_norm");
        dot("post_feedforward_layernorm", ".post_ffw_norm");
        dot("mlp.gate_proj", ".ffn_gate");
        dot("mlp.up_proj", ".ffn_up");
        dot("mlp.down_proj", ".ffn_down");
        // The MoE block (qwen3.6-moe / qwen3.5-moe): the container flattens the routed
        // experts into one [E*n, k] tensor per projection, the GGUF keeps them 3-D
        // [k, n, E] under llama.cpp's *_exps names. The shared expert is *_shexp, and
        // its scalar gate plus the router are f32 vectors/matrices, not chunks.
        dot("mlp.gate_exps_proj", ".ffn_gate_exps");
        dot("mlp.up_exps_proj", ".ffn_up_exps");
        dot("mlp.down_exps_proj", ".ffn_down_exps");
        dot("mlp.share_gate_exps_proj", ".ffn_gate_shexp");
        dot("mlp.share_up_exps_proj", ".ffn_up_shexp");
        dot("mlp.share_down_exps_proj", ".ffn_down_shexp");
        dot("moe_router", ".ffn_gate_inp");
        dot("shared_expert_gate", ".ffn_gate_inp_shexp");
    }
    return n;
}

const GgufFile::TensorInfo& GgufFile::tensor(const std::string& name) const {
    std::string n = gguf_name(name, sandwich_, post_norm_);
    auto it = tensors_.find(n);
    if (it == tensors_.end() && n == "output.weight") {
        // tied embeddings: the GGUF has no output.weight; the (untied) rows
        // of token_embd serve as the lm head (same vocab x hidden shape)
        n = "token_embd.weight";
        it = tensors_.find(n);
    }
    if (it == tensors_.end() && n == "output_norm.weight") {
        // The final norm, by SHAPE rather than by architecture name: llama.cpp's arches do
        // not agree on it (Qwen/Llama/Gemma: output_norm.weight; LFM2: token_embd_norm.weight),
        // but the structure is identical in every one -- a top-level 1-D tensor of `hidden`
        // values that is not the embedding table. A name table here would need an entry per
        // architecture; this needs none.
        n = final_norm_name();
        it = n.empty() ? tensors_.end() : tensors_.find(n);
    }
    if (it == tensors_.end() && n.rfind("blk.", 0) == 0) {
        // Fused QKV (Phi-3: blk.<l>.attn_qkv.weight holds [q | k | v]) and fused
        // gate|up (Phi-3: blk.<l>.ffn_up.weight holds [gate | up]).
        for (const char* sfx : {".attn_q.weight", ".attn_k.weight", ".attn_v.weight"}) {
            const std::string s(sfx);
            if (n.size() > s.size() && n.compare(n.size() - s.size(), s.size(), s) == 0) {
                it = tensors_.find(n.substr(0, n.size() - s.size()) + ".attn_qkv.weight");
                break;
            }
        }
        const std::string gate_sfx = ".ffn_gate.weight";
        if (it == tensors_.end() && n.size() > gate_sfx.size() &&
            n.compare(n.size() - gate_sfx.size(), gate_sfx.size(), gate_sfx) == 0) {
            it = tensors_.find(n.substr(0, n.size() - gate_sfx.size()) + ".ffn_up.weight");
        }
    }
    if (it == tensors_.end()) fail("no tensor " + name + " in " + path_);
    return it->second;
}

bool GgufFile::has(const std::string& name) const {
    std::string n = gguf_name(name, sandwich_, post_norm_);
    if (tensors_.count(n)) return true;
    if (n.rfind("blk.", 0) == 0) {
        for (const char* sfx : {".attn_q.weight", ".attn_k.weight", ".attn_v.weight"}) {
            const std::string s(sfx);
            if (n.size() > s.size() && n.compare(n.size() - s.size(), s.size(), s) == 0)
                return tensors_.count(n.substr(0, n.size() - s.size()) + ".attn_qkv.weight") != 0;
        }
        const std::string gate_sfx = ".ffn_gate.weight";
        if (n.size() > gate_sfx.size() && n.compare(n.size() - gate_sfx.size(), gate_sfx.size(), gate_sfx) == 0)
            return tensors_.count(n.substr(0, n.size() - gate_sfx.size()) + ".ffn_up.weight") != 0;
    }
    return false;
}

std::string GgufFile::final_norm_name() const {
    auto emb = tensors_.find("token_embd.weight");
    if (emb == tensors_.end() || emb->second.dims.empty()) return {};
    const uint64_t hidden = emb->second.dims[0];          // fastest dim of [hidden, vocab]
    std::vector<std::string> found;
    for (const auto& [n, t] : tensors_) {
        if (n.rfind("blk.", 0) == 0) continue;             // per-layer tensors
        if (n == "token_embd.weight") continue;            // the embedding table itself
        if (t.dims.size() != 1 || t.dims[0] != hidden) continue;
        found.push_back(n);
    }
    if (found.size() == 1) return found[0];
    // Several candidates: a norm is the one whose name says so. Still shape-first -- the
    // candidates all have the right shape, and the name only breaks the tie.
    std::string norm;
    for (const auto& n : found)
        if (n.find("norm") != std::string::npos) {
            if (!norm.empty()) return {};                  // ambiguous: refuse, do not guess
            norm = n;
        }
    return norm;
}


const uint8_t* GgufFile::raw(const std::string& name, size_t* nbytes) const {
    const TensorInfo& t = tensor(name);
    const uint64_t nb = gguf_type_bytes(static_cast<uint32_t>(t.type), t.nvalues());
    if (nbytes) *nbytes = static_cast<size_t>(nb);
    if (t.shard >= shards_.size()) fail("tensor " + name + " has no mapped shard");
    const Shard& s = shards_[t.shard];
    return s.map + s.data_base + t.offset;
}

std::vector<std::string> GgufFile::tensor_names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto& [name, info] : tensors_) out.push_back(name);
    return out;
}

const TensorMeta& GgufFile::meta(const std::string& name) const {
    auto cached = meta_cache_.find(name);
    if (cached != meta_cache_.end()) return cached->second;
    const TensorInfo& t = tensor(name);
    TensorMeta m;
    m.dtype = type_name(t.type);
    // GGUF dims are fastest-first; the host stages speak row-major [rows, cols].
    for (auto d = t.dims.rbegin(); d != t.dims.rend(); ++d) m.shape.push_back(static_cast<size_t>(*d));
    return meta_cache_.emplace(name, std::move(m)).first->second;
}

namespace {
// A whole tensor decoded to f32, row by row (the GGUF's fastest dim is the row).
std::vector<float> gguf_dequant_all(const GgufFile& g, const std::string& name) {
    const GgufFile::TensorInfo& t = g.tensor(name);
    const size_t cols = static_cast<size_t>(t.dims[0]);
    const size_t nrows = t.dims.size() >= 2 ? static_cast<size_t>(t.dims[1]) : 1;
    std::vector<float> out(cols * nrows);
    for (size_t r = 0; r < nrows; ++r) g.embed_row(name, r, cols, out.data() + r * cols);
    return out;
}
}  // namespace

std::vector<float> GgufFile::bf16(const std::string& name) const { return gguf_dequant_all(*this, name); }
std::vector<float> GgufFile::f32(const std::string& name) const { return gguf_dequant_all(*this, name); }

std::string GgufFile::kv_str(const std::string& key) const {
    auto it = meta_.find(key);
    if (it == meta_.end()) fail("no metadata key " + key + " in " + path_);
    return it->second;
}

uint64_t GgufFile::kv_u64(const std::string& key) const {
    const std::string s = kv_str(key);
    try {
        return std::stoull(s);
    } catch (const std::logic_error&) {
        // A PER-LAYER array (llama.cpp writes some keys one value per block). The spec
        // carries ONE count for the whole model, so the values must agree; zero entries are
        // the layers that do not use the key at all -- LFM2's short-conv blocks have no kv
        // heads, so `lfm2.attention.head_count_kv` is [0, 0, 8, 0, ...] and the answer is 8.
        uint64_t seen = 0;
        bool any = false;
        size_t i = 0;
        while (i <= s.size()) {
            const size_t j = s.find(',', i);
            const std::string part = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
            if (!part.empty()) {
                uint64_t v = 0;
                try {
                    v = std::stoull(part);
                } catch (const std::logic_error&) {
                    fail("metadata " + key + " is not a number (" + s + ")");
                }
                if (v) {
                    if (any && v != seen)
                        fail("metadata " + key + " differs per layer (" + s + "); the kernels "
                             "run one value for the whole model");
                    seen = v;
                    any = true;
                }
            }
            if (j == std::string::npos) break;
            i = j + 1;
        }
        if (any) return seen;
        fail("metadata " + key + " is not a number (" + s + ")");
    }
}

double GgufFile::kv_f64(const std::string& key) const {
    try {
        return std::stod(kv_str(key));
    } catch (const std::logic_error&) {
        fail("metadata " + key + " is not a number (" + kv_str(key) + ")");
    }
}

std::vector<std::string> GgufFile::kv_str_array(const std::string& key) const {
    // `meta_` deliberately drops big arrays (the tokenizer's vocab is megabytes of
    // text nothing else reads), so walk the primary shard's header again to the key.
    // Called once per GGUF->checkpoint conversion, never on a hot path.
    if (shards_.empty() || shards_[0].size < 24) return {};
    const Shard& s = shards_[0];
    const uint8_t* end = s.map + s.size;
    const uint8_t* q = s.map + 4;
    (void)rd_u32(q);                 // version
    (void)rd_u64(q);                 // tensor count
    const uint64_t nkv = rd_u64(q);
    for (uint64_t i = 0; i < nkv; ++i) {
        const std::string k = rd_str(q, end);
        const uint32_t t = rd_u32(q);
        if (k != key) {
            std::string v;
            rd_value(t, q, end, v);
            continue;
        }
        if (t != 9) fail("metadata " + key + " is not an array");
        const uint32_t et = rd_u32(q);
        const uint64_t n = rd_u64(q);
        if (et != 8) fail("metadata " + key + " is not a string array");
        std::vector<std::string> out;
        out.reserve(static_cast<size_t>(n));
        for (uint64_t j = 0; j < n; ++j) out.push_back(rd_str(q, end));
        return out;
    }
    return {};
}

/// One row of a tensor, `dim` values wide, from a tensor whose fastest dim is `dim`.
/// The caller has already checked that the row exists: embed_row for the model's
/// [rows, cols] tensors, f32_all for any rank (the vision projector's patch embed is
/// [P, P, C, hidden], and llama.cpp stores the conv kernel with the patch width
/// fastest -- a row of it is a patch column, not a hidden row).
void GgufFile::dequant_row(const TensorInfo& t, size_t row, size_t dim, float* out) const {
    if (t.shard >= shards_.size()) fail("tensor row has no mapped shard");
    const Shard& s = shards_[t.shard];
    const uint8_t* base = s.map + s.data_base + t.offset;
    switch (t.type) {
        case Type::F32: {
            std::memcpy(out, base + row * dim * 4, dim * 4);
            break;
        }
        case Type::BF16: {
            const uint8_t* p = base + row * dim * 2;
            for (size_t i = 0; i < dim; ++i) {
                uint16_t u;
                std::memcpy(&u, p + 2 * i, 2);
                out[i] = bf16_to_f32(u);
            }
            break;
        }
        case Type::F16: {
            const uint8_t* p = base + row * dim * 2;
            for (size_t i = 0; i < dim; ++i) {
                uint16_t u;
                std::memcpy(&u, p + 2 * i, 2);
                out[i] = fp16_to_f32(u);
            }
            break;
        }
        case Type::Q8_0: {
            // fp16 scale + 32 int8 values per block
            const uint8_t* p = base + row * (dim / 32) * 34;
            for (size_t b = 0; b < dim / 32; ++b, p += 34) {
                uint16_t u;
                std::memcpy(&u, p, 2);
                const float d = fp16_to_f32(u);
                for (size_t j = 0; j < 32; ++j) out[b * 32 + j] = d * static_cast<int8_t>(p[2 + j]);
            }
            break;
        }
        case Type::Q4_0: {
            // fp16 scale + 16 nibble bytes; GGML's values are d * (q - 8)
            const uint8_t* p = base + row * (dim / 32) * 18;
            for (size_t b = 0; b < dim / 32; ++b, p += 18) {
                uint16_t u;
                std::memcpy(&u, p, 2);
                const float d = fp16_to_f32(u);
                for (size_t j = 0; j < 16; ++j) {
                    out[b * 32 + j] = d * (static_cast<int>(p[2 + j] & 0xF) - 8);
                    out[b * 32 + 16 + j] = d * (static_cast<int>(p[2 + j] >> 4) - 8);
                }
            }
            break;
        }
        case Type::Q4_1: {
            // fp16 scale + fp16 min + 16 nibble bytes per block
            const uint8_t* p = base + row * (dim / 32) * 20;
            for (size_t b = 0; b < dim / 32; ++b, p += 20) {
                uint16_t ud, um;
                std::memcpy(&ud, p, 2);
                std::memcpy(&um, p + 2, 2);
                const float d = fp16_to_f32(ud), m = fp16_to_f32(um);
                for (size_t j = 0; j < 16; ++j) {
                    out[b * 32 + j] = d * (p[4 + j] & 0xF) + m;
                    out[b * 32 + 16 + j] = d * (p[4 + j] >> 4) + m;
                }
            }
            break;
        }
        case Type::Q5_0: {
            // llama.cpp's q5_0: fp16 d, 16 nibble bytes, 4 bytes of high bits
            // (32 bits, one per value). 22 B per 32 values. The high-bit layout
            // is bitwise, not a second nibble array: values 0..15 take bit j of
            // qh, values 16+j take bit 16+j (llama.cpp extracts both with the
            // shifts below).
            const uint8_t* p = base + row * (dim / 32) * 22;
            for (size_t b = 0; b < dim / 32; ++b, p += 22) {
                uint16_t u;
                std::memcpy(&u, p, 2);
                const float d = fp16_to_f32(u);
                uint32_t qh;
                std::memcpy(&qh, p + 18, 4);
                // llama.cpp emits these INTERLEAVED (y[2*j], y[2*j+1]), unlike
                // q4_0/q4_1's low nibble block then high nibble block.
                for (size_t j = 0; j < 16; ++j) {
                    const uint8_t xh0 = static_cast<uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
                    const uint8_t xh1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10);
                    out[b * 32 + 2 * j] = static_cast<float>(
                        static_cast<int>((p[2 + j] & 0xF) | xh0) - 16) * d;
                    out[b * 32 + 2 * j + 1] = static_cast<float>(
                        static_cast<int>((p[2 + j] >> 4) | xh1) - 16) * d;
                }
            }
            break;
        }
        case Type::Q5_1: {
            // Unsloth's UD-Q4_K_XL keeps the MoE down projections at q5_1:
            // fp16 d, fp16 m, 16 nibble bytes, 4 bytes of high bits (24 B/32).
            // Values are d * (nibble | high) + m; layout as q5_0 above.
            const uint8_t* p = base + row * (dim / 32) * 24;
            for (size_t b = 0; b < dim / 32; ++b, p += 24) {
                uint16_t ud, um;
                std::memcpy(&ud, p, 2);
                std::memcpy(&um, p + 2, 2);
                const float d = fp16_to_f32(ud), m = fp16_to_f32(um);
                uint32_t qh;
                std::memcpy(&qh, p + 20, 4);
                for (size_t j = 0; j < 16; ++j) {
                    const uint8_t xh0 = static_cast<uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
                    const uint8_t xh1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10);
                    const int v0 = (p[4 + j] & 0xF) | xh0;
                    const int v1 = (p[4 + j] >> 4) | xh1;
                    out[b * 32 + 2 * j] = static_cast<float>(v0) * d + m;
                    out[b * 32 + 2 * j + 1] = static_cast<float>(v1) * d + m;
                }
            }
            break;
        }
        case Type::Q4_K: {
            // super-block of 256: fp16 d, fp16 dmin, 12 bytes of 6-bit scales
            // (8 scale/min pairs over 8 blocks of 32), then 128 bytes of nibbles
            const uint8_t* p = base + row * (dim / 256) * 144;
            for (size_t sb = 0; sb < dim / 256; ++sb, p += 144) {
                uint16_t ud, um;
                std::memcpy(&ud, p, 2);
                std::memcpy(&um, p + 2, 2);
                const float d = fp16_to_f32(ud), min = fp16_to_f32(um);
                const uint8_t* scales = p + 4;
                const uint8_t* qs = p + 16;
                int is = 0;
                uint8_t sc, m;
                float* y = out + sb * 256;
                for (int j = 0; j < 256; j += 64) {
                    get_scale_min_k4(is + 0, scales, &sc, &m);
                    const float d1 = d * sc, m1 = min * m;
                    get_scale_min_k4(is + 1, scales, &sc, &m);
                    const float d2 = d * sc, m2 = min * m;
                    for (int l = 0; l < 32; ++l) y[j + l] = d1 * (qs[j / 2 + l] & 0xF) - m1;
                    for (int l = 0; l < 32; ++l) y[j + 32 + l] = d2 * (qs[j / 2 + l] >> 4) - m2;
                    is += 2;
                }
            }
            break;
        }
        case Type::Q5_K: {
            // super-block of 256: fp16 d, fp16 dmin, 12 B of 6-bit scales (eight
            // scale/min pairs over eight blocks of 32), 32 B of the FIFTH bit (one
            // per value), then 128 B of 4-bit codes. The value is
            // d*scale*(nibble | fifth<<4) - dmin*min.
            const uint8_t* p = base + row * (dim / 256) * 176;
            for (size_t sb = 0; sb < dim / 256; ++sb, p += 176) {
                uint16_t ud, um;
                std::memcpy(&ud, p, 2);
                std::memcpy(&um, p + 2, 2);
                const float d = fp16_to_f32(ud), min = fp16_to_f32(um);
                const uint8_t* scales = p + 4;
                const uint8_t* qh = p + 16;                 // the fifth bit, one per value
                const uint8_t* qs = p + 48;                 // the 4-bit codes
                float* y = out + sb * 256;
                int is = 0;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc, m;
                    get_scale_min_k4(is + 0, scales, &sc, &m);
                    const float d1 = d * sc, m1 = min * m;
                    get_scale_min_k4(is + 1, scales, &sc, &m);
                    const float d2 = d * sc, m2 = min * m;
                    for (int l = 0; l < 32; ++l) {
                        const int v0 = j + l, v1 = j + 32 + l;
                        const uint8_t byte = qs[j / 2 + l];
                        const uint8_t n0 = byte & 0xF, n1 = byte >> 4;
                        // the fifth bit: byte (v % 32) holds the bits of the values
                        // v, v+32, v+64, ... -- bit (v / 32) of it
                        const uint8_t h0 = (qh[v0 % 32] >> (v0 / 32)) & 1;
                        const uint8_t h1 = (qh[v1 % 32] >> (v1 / 32)) & 1;
                        y[v0] = d1 * static_cast<float>(n0 | (h0 << 4)) - m1;
                        y[v1] = d2 * static_cast<float>(n1 | (h1 << 4)) - m2;
                    }
                    is += 2;
                }
            }
            break;
        }
        case Type::Q6_K: {
            // super-block of 256: 128 B lower nibbles, 64 B upper 2 bits,
            // 16 int8 scales (per 16 values), fp16 super-block scale
            const uint8_t* p = base + row * (dim / 256) * 210;
            for (size_t sb = 0; sb < dim / 256; ++sb, p += 210) {
                uint16_t u;
                std::memcpy(&u, p + 208, 2);
                const float d = fp16_to_f32(u);
                const uint8_t* ql = p;
                const uint8_t* qh = p + 128;
                const int8_t* sc = reinterpret_cast<const int8_t*>(p + 192);
                float* y = out + sb * 256;
                for (int n = 0; n < 256; n += 128) {
                    for (int l = 0; l < 32; ++l) {
                        const int is = l / 16;
                        const int8_t q1 = static_cast<int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int8_t q3 = static_cast<int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        y[n + l + 0] = d * sc[is + 0] * q1;
                        y[n + l + 32] = d * sc[is + 2] * q2;
                        y[n + l + 64] = d * sc[is + 4] * q3;
                        y[n + l + 96] = d * sc[is + 6] * q4;
                    }
                    ql += 64;
                    qh += 32;   // 32 qh bytes per 128 values (4 values' high bits per byte)
                    sc += 8;
                }
            }
            break;
        }
        case Type::IQ4_NL: {
            // 4-bit "nibble lookup" (llama.cpp's iq4_nl): each code indexes the
            // sixteen-entry codebook, scaled by the block's fp16 d. 18 B per 32 values.
            const uint8_t* p = base + row * (dim / 32) * 18;
            for (size_t b = 0; b < dim / 32; ++b, p += 18) {
                uint16_t u;
                std::memcpy(&u, p, 2);
                const float d = fp16_to_f32(u);
                for (size_t j = 0; j < 16; ++j) {
                    out[b * 32 + j] = d * kIQ4NLValues[p[2 + j] & 0xF];
                    out[b * 32 + 16 + j] = d * kIQ4NLValues[p[2 + j] >> 4];
                }
            }
            break;
        }
        case Type::IQ4_XS: {
            // llama.cpp's iq4_xs: a super-block of 256 values in 136 B --
            //   fp16 d | 2 B of 2-bit scale highs | 4 B of 4-bit scale lows | 128 B of codes
            // Eight groups of 32 values, each with a signed 6-bit scale (the low nibble
            // | the high two bits << 4, minus 32) times d; the codes index the same
            // sixteen-entry codebook as iq4_nl.
            const uint8_t* p = base + row * (dim / 256) * 136;
            for (size_t sb = 0; sb < dim / 256; ++sb, p += 136) {
                uint16_t ud, sh;
                std::memcpy(&ud, p, 2);
                std::memcpy(&sh, p + 2, 2);
                const float d = fp16_to_f32(ud);
                const uint8_t* sl = p + 4;               // four bytes: eight scale lows
                const uint8_t* qs = p + 8;               // 128 bytes: 256 codes
                float* y = out + sb * 256;
                for (int g = 0; g < 8; ++g) {
                    const uint8_t lo = (g & 1) ? static_cast<uint8_t>(sl[g / 2] >> 4)
                                               : static_cast<uint8_t>(sl[g / 2] & 0xF);
                    const uint8_t hi = static_cast<uint8_t>((sh >> (2 * g)) & 0x3);
                    const float dl = d * static_cast<float>(static_cast<int>(lo | (hi << 4)) - 32);
                    const uint8_t* q = qs + g * 16;
                    float* yg = y + g * 32;
                    for (int j = 0; j < 16; ++j) {
                        yg[j] = dl * kIQ4NLValues[q[j] & 0xF];
                        yg[j + 16] = dl * kIQ4NLValues[q[j] >> 4];
                    }
                }
            }
            break;
        }
        default:
            fail(std::string("the embedding is ") + type_name(t.type) +
                 "; this reader dequantizes F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/"
                 "Q6_K/IQ4_NL/IQ4_XS rows "
                 "(convert the model with q4nx-build, or quantize the embedding to a supported type)");
    }
}

void GgufFile::embed_row(const std::string& name, size_t row, size_t dim, float* out) const {
    const TensorInfo& t = tensor(name);
    const uint64_t cols = t.dims[0];               // fastest dim: the hidden size
    if (cols != dim) fail(name + " has " + std::to_string(cols) + " cols, the row wants " + std::to_string(dim));
    // A row index is a FLAT row of the tensor: for a 2-D [rows, cols] it is dims[1], but a
    // MoE expert tensor is 3-D ([in, n, experts]) and its rows are n * experts. Counting
    // only dims[1] refused every expert but the first (the packer reads them by flat row).
    const uint64_t nrows = cols ? t.nvalues() / cols : 0;
    if (row >= nrows) fail("row " + std::to_string(row) + " past " + name);
    dequant_row(t, row, dim, out);
}

std::vector<float> GgufFile::f32_all(const std::string& name) const {
    const TensorInfo& t = tensor(name);
    const size_t n = static_cast<size_t>(t.nvalues());
    std::vector<float> out(n);
    if (!n) return out;
    // The fastest dim is the row width; the rows are the product of the rest, whatever
    // the rank. A 4-D conv kernel therefore decodes as rows of P values -- which is what
    // its layout means -- not as the dims[1] rows a [rows, cols] accessor assumes.
    const size_t cols = static_cast<size_t>(t.dims.empty() ? n : t.dims[0]);
    if (!cols || n % cols)
        fail(name + ": " + std::to_string(n) + " values is not whole rows of " + std::to_string(cols));
    for (size_t r = 0; r < n / cols; ++r) dequant_row(t, r, cols, out.data() + r * cols);
    return out;
}

}  // namespace open_qwen36

/// \file gguf_file.cpp
/// \brief GGUF parsing and the per-token embedding dequant (see gguf_file.hpp).
#include "open_qwen36/gguf_file.hpp"

#include <cstring>
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
        case 9: {  // array: element type + count; the values are skipped
            const uint32_t et = rd_u32(p);
            const uint64_t n = rd_u64(p);
            for (uint64_t i = 0; i < n; ++i) {
                std::string skip;
                rd_value(et, p, end, skip);
            }
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
    // block quants: 18 B / 32 q4_0 | 20 B / 32 q4_1 | 34 B / 32 q8_0 |
    // 144 B / 256 q4_k | 210 B / 256 q6_k
    switch (t) {
        case 0: case 6: case 30: return values * 4;      // f32 / bf16
        case 1: return values * 2;                        // f16
        case 2: return values / 32 * 18;                  // q4_0
        case 3: return values / 32 * 20;                  // q4_1
        case 8: return values / 32 * 34;                  // q8_0
        case 14: case 15: return values / 256 * 144;      // q4_k_s / q4_k_m
        case 18: return values / 256 * 210;               // q6_k
        default: return 0;                                // unsupported
    }
}

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

}  // namespace

const char* GgufFile::type_name(Type t) {
    switch (t) {
        case Type::F32: return "F32"; case Type::F16: return "F16"; case Type::Q4_0: return "Q4_0";
        case Type::Q4_1: return "Q4_1"; case Type::Q5_0: return "Q5_0"; case Type::Q5_1: return "Q5_1";
        case Type::Q8_0: return "Q8_0"; case Type::Q8_1: return "Q8_1"; case Type::Q2_K: return "Q2_K";
        case Type::Q3_K_S: return "Q3_K_S"; case Type::Q3_K_M: return "Q3_K_M"; case Type::Q3_K_L: return "Q3_K_L";
        case Type::Q4_K_S: return "Q4_K_S"; case Type::Q4_K_M: return "Q4_K_M"; case Type::Q5_K_S: return "Q5_K_S";
        case Type::Q5_K_M: return "Q5_K_M"; case Type::Q6_K: return "Q6_K"; case Type::IQ2_XXS: return "IQ2_XXS";
        case Type::IQ2_XS: return "IQ2_XS"; case Type::Q2_K_S: return "Q2_K_S"; case Type::Q3_K_XS: return "Q3_K_XS";
        case Type::IQ3_XXS: return "IQ3_XXS"; case Type::Q8_K: return "Q8_K"; case Type::IQ4_NL: return "IQ4_NL";
        case Type::IQ4_XS: return "IQ4_XS"; case Type::BF16: return "BF16";
    }
    return "type?";
}

GgufFile::GgufFile(const std::string& path) : path_(path) {
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
    file_ = f;
    mapping_ = m;
    map_ = static_cast<const uint8_t*>(p);
    map_size_ = static_cast<size_t>(sz.QuadPart);
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("gguf: cannot open " + path);
    struct stat st;
    if (fstat(fd_, &st) != 0) { ::close(fd_); throw std::runtime_error("gguf: size of " + path); }
    void* p = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd_, 0);
    if (p == MAP_FAILED) { ::close(fd_); throw std::runtime_error("gguf: cannot map " + path); }
    map_ = static_cast<const uint8_t*>(p);
    map_size_ = static_cast<size_t>(st.st_size);
#endif
    try {
        const uint8_t* end = map_ + map_size_;
        if (map_size_ < 24 || map_[0] != 'G' || map_[1] != 'G' || map_[2] != 'U' || map_[3] != 'F') fail(path + " is not a GGUF file");
        const uint8_t* q = map_ + 4;
        const uint32_t version = rd_u32(q);
        if (version < 2) fail(path + ": GGUF v" + std::to_string(version) + " unsupported (need v2+)");
        const uint64_t ntensors = rd_u64(q);
        const uint64_t nkv = rd_u64(q);
        for (uint64_t i = 0; i < nkv; ++i) {
            const std::string key = rd_str(q, end);
            const uint32_t t = rd_u32(q);
            std::string v;
            rd_value(t, q, end, v);
            meta_[key] = v;
        }
        for (uint64_t i = 0; i < ntensors; ++i) {
            const std::string name = rd_str(q, end);
            const uint32_t nd = rd_u32(q);
            TensorInfo t;
            t.dims.resize(nd);
            for (uint32_t d = 0; d < nd; ++d) t.dims[d] = rd_u64(q);
            const uint32_t ty = rd_u32(q);
            t.type = static_cast<Type>(ty);
            t.offset = rd_u64(q);
            const uint64_t nb = gguf_type_bytes(ty, t.nvalues());
            if (nb && t.offset + nb > map_size_) fail("tensor " + name + " runs past EOF in " + path);
            tensors_.emplace(name, std::move(t));
        }
        uint64_t align = 32;
        if (version == 2) align = rd_u32(q);       // v3 removed it; fixed at 32
        data_base_ = static_cast<size_t>(q - map_);
        data_base_ += static_cast<size_t>((align - data_base_ % align) % align);
    } catch (...) {
#ifdef _WIN32
        if (map_) UnmapViewOfFile(map_);
        if (mapping_) CloseHandle(static_cast<HANDLE>(mapping_));
        if (file_) CloseHandle(static_cast<HANDLE>(file_));
#else
        if (map_) munmap(const_cast<uint8_t*>(map_), map_size_);
        if (fd_ >= 0) ::close(fd_);
#endif
        throw;
    }
}

GgufFile::~GgufFile() {
#ifdef _WIN32
    if (map_) UnmapViewOfFile(map_);
    if (mapping_) CloseHandle(static_cast<HANDLE>(mapping_));
    if (file_) CloseHandle(static_cast<HANDLE>(file_));
#else
    if (map_) munmap(const_cast<uint8_t*>(map_), map_size_);
    if (fd_ >= 0) ::close(fd_);
#endif
}

void GgufFile::drop_pages() {
#ifdef _WIN32
    UnmapViewOfFile(map_);
    const void* p = MapViewOfFile(static_cast<HANDLE>(mapping_), FILE_MAP_READ, 0, 0, 0);
    if (!p) throw std::runtime_error("gguf: cannot remap " + path_);
    map_ = static_cast<const uint8_t*>(p);
#else
    madvise(const_cast<uint8_t*>(map_), map_size_, MADV_DONTNEED);
#endif
}

std::string GgufFile::gguf_name(const std::string& name) {
    // The manifest's packing plan names tensors the q4nx/HF way; a GGUF uses
    // llama.cpp's conventions (utilities/q4nx-build/configs/*.json hold the
    // same map on the conversion side).
    if (name == "model.embed_tokens.weight") return "token_embd.weight";
    if (name == "model.norm.weight") return "output_norm.weight";
    if (name == "lm_head.weight") return "output.weight";
    std::string n = name;
    const std::string layers = "model.layers.";
    if (n.rfind(layers, 0) == 0) {
        n = "blk." + n.substr(layers.size());
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
        dot("self_attn.post_attention_layernorm", ".post_attention_norm");   // gemma3
        dot("input_layernorm", ".attn_norm");
        dot("post_attention_layernorm", ".ffn_norm");
        dot("pre_feedforward_layernorm", ".pre_ffn_norm");                   // gemma3
        dot("post_feedforward_layernorm", ".post_ffn_norm");                 // gemma3
        dot("mlp.gate_proj", ".ffn_gate");
        dot("mlp.up_proj", ".ffn_up");
        dot("mlp.down_proj", ".ffn_down");
    }
    return n;
}

const GgufFile::TensorInfo& GgufFile::tensor(const std::string& name) const {
    std::string n = gguf_name(name);
    auto it = tensors_.find(n);
    if (it == tensors_.end() && n == "output.weight") {
        // tied embeddings: the GGUF has no output.weight; the (untied) rows
        // of token_embd serve as the lm head (same vocab x hidden shape)
        n = "token_embd.weight";
        it = tensors_.find(n);
    }
    if (it == tensors_.end()) fail("no tensor " + name + " in " + path_);
    return it->second;
}

const uint8_t* GgufFile::raw(const std::string& name, size_t* nbytes) const {
    const TensorInfo& t = tensor(name);
    const uint64_t nb = gguf_type_bytes(static_cast<uint32_t>(t.type), t.nvalues());
    if (nbytes) *nbytes = static_cast<size_t>(nb);
    return map_ + data_base_ + t.offset;
}

std::string GgufFile::kv_str(const std::string& key) const {
    auto it = meta_.find(key);
    if (it == meta_.end()) fail("no metadata key " + key + " in " + path_);
    return it->second;
}

uint64_t GgufFile::kv_u64(const std::string& key) const {
    try {
        return std::stoull(kv_str(key));
    } catch (const std::logic_error&) {
        fail("metadata " + key + " is not a number (" + kv_str(key) + ")");
    }
}

double GgufFile::kv_f64(const std::string& key) const {
    try {
        return std::stod(kv_str(key));
    } catch (const std::logic_error&) {
        fail("metadata " + key + " is not a number (" + kv_str(key) + ")");
    }
}

void GgufFile::embed_row(const std::string& name, size_t row, size_t dim, float* out) const {
    const TensorInfo& t = tensor(name);
    const uint64_t cols = t.dims[0];               // fastest dim: the hidden size
    if (cols != dim) fail(name + " has " + std::to_string(cols) + " cols, the row wants " + std::to_string(dim));
    if (row >= t.dims[1]) fail("row " + std::to_string(row) + " past " + name);
    const uint8_t* base = map_ + data_base_ + t.offset;
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
            // fp16 scale + 16 nibble bytes per block (min = 0)
            const uint8_t* p = base + row * (dim / 32) * 18;
            for (size_t b = 0; b < dim / 32; ++b, p += 18) {
                uint16_t u;
                std::memcpy(&u, p, 2);
                const float d = fp16_to_f32(u);
                for (size_t j = 0; j < 16; ++j) {
                    out[b * 32 + j] = d * (p[2 + j] & 0xF);
                    out[b * 32 + 16 + j] = d * (p[2 + j] >> 4);
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
        case Type::Q4_K_S:
        case Type::Q4_K_M: {
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
                    qh += 16;
                    sc += 8;
                }
            }
            break;
        }
        default:
            fail(std::string("the embedding is ") + type_name(t.type) +
                 "; this reader dequantizes F32/F16/BF16/Q4_0/Q4_1/Q8_0/Q4_K/Q6_K rows "
                 "(convert the model with q4nx-build, or quantize the embedding to a supported type)");
    }
}

}  // namespace open_qwen36

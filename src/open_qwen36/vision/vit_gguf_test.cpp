/// vit_gguf_test: the vision tower read from a llama.cpp `mmproj` GGUF.
///
/// A GGUF model ships its projector as `mmproj-*.gguf` -- there is no
/// `vision_weight.q4nx` and, on Qwen3-VL-4B, no `vision_config` either. The file is
/// the whole description: `clip.vision.*` states the geometry, the patch embed's
/// shape states the channels and the temporal split, and the deepstack indexes are
/// in the tensor names.
///
/// This test writes such a file ITSELF, with values that encode their own position,
/// so every transpose and every split the loader performs is checked:
///
///   * llama.cpp stores a matmul weight [in, out]; the tower keeps [out, in].
///   * the patch embed is one [P, P, C, hidden] conv per temporal slice, and the
///     tower's patch vector is (C, T, P_h, P_w).
///   * the deepstack mergers normalise over `hidden * merge^2`, not `hidden`.
///
/// It also pins the refusals: a file that is not a projector, a projector this
/// reader does not run, and one extra tensor (a missing piece otherwise reads as
/// plausible numbers rather than as an error).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "open_qwen36/q4nx_file.hpp"   // bf16_to_f32, for the debug/compare
#include "open_qwen36/vision/vit.hpp"

namespace {

int g_failed = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

// ---------------------------------------------------------------- a tiny GGUF writer
// Only what a projector needs: string/u32/f32 metadata and f32/f16 tensors. Offsets
// are exact (this reader takes them as given), and the data section starts at the
// 32-byte alignment GGUF v3 uses.

enum : uint32_t { kU32 = 4, kF32 = 6, kStr = 8, kF16 = 1, kF32T = 0 };

void put_u32(std::vector<uint8_t>& b, uint32_t v) { b.insert(b.end(), reinterpret_cast<uint8_t*>(&v), reinterpret_cast<uint8_t*>(&v) + 4); }
void put_u64(std::vector<uint8_t>& b, uint64_t v) { b.insert(b.end(), reinterpret_cast<uint8_t*>(&v), reinterpret_cast<uint8_t*>(&v) + 8); }

void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put_u64(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

/// float -> IEEE half, round to nearest even (the subset a test needs: no infinities).
uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127;
    uint32_t man = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0) return static_cast<uint16_t>(sign);            // zero / subnormal -> 0
    if (exp > 15) return static_cast<uint16_t>(sign | 0x7C00u);                 // overflow -> inf
    if (exp < -14) {                                                            // subnormal
        const uint32_t shift = static_cast<uint32_t>(-exp - 1);
        const uint32_t m = (man | 0x800000u) >> (shift + 13);
        return static_cast<uint16_t>(sign | m);
    }
    const uint32_t half_exp = static_cast<uint32_t>(exp + 15) << 10;
    const uint32_t half_man = (man + 0x1000u) >> 13;                            // round to nearest
    uint32_t out = half_exp | half_man;
    if (half_man == 0x400u) out = (half_exp + 0x400u);                          // mantissa carried
    return static_cast<uint16_t>(sign | out);
}

struct Tensor {
    std::string name;
    std::vector<uint64_t> dims;   // fastest first
    uint32_t type = kF32T;
    std::vector<float> f32;
};

struct Meta {
    std::string key;
    uint32_t type = kU32;
    std::string str;
    uint32_t u32 = 0;
    float f32 = 0.f;
};

struct Gguf {
    std::vector<Meta> kv;
    std::vector<Tensor> tensors;

    void m_str(const std::string& k, const std::string& v) { Meta m; m.key = k; m.type = kStr; m.str = v; kv.push_back(m); }
    void m_u32(const std::string& k, uint32_t v) { Meta m; m.key = k; m.type = kU32; m.u32 = v; kv.push_back(m); }
    void m_f32(const std::string& k, float v) { Meta m; m.key = k; m.type = kF32; m.f32 = v; kv.push_back(m); }

    void t_f32(const std::string& n, std::vector<uint64_t> dims, std::vector<float> v) {
        Tensor t; t.name = n; t.dims = std::move(dims); t.type = kF32T; t.f32 = std::move(v); tensors.push_back(std::move(t));
    }
    void t_f16(const std::string& n, std::vector<uint64_t> dims, std::vector<float> v) {
        Tensor t; t.name = n; t.dims = std::move(dims); t.type = kF16; t.f32 = std::move(v); tensors.push_back(std::move(t));
    }

    void write(const std::string& path) const {
        std::vector<uint8_t> head = {'G', 'G', 'U', 'F'};
        put_u32(head, 3);
        put_u64(head, tensors.size());
        put_u64(head, kv.size());
        for (const Meta& m : kv) {
            put_str(head, m.key);
            put_u32(head, m.type);
            switch (m.type) {
                case kStr: put_str(head, m.str); break;
                case kU32: put_u32(head, m.u32); break;
                case kF32: {
                    const uint32_t u = [&] { uint32_t x; std::memcpy(&x, &m.f32, 4); return x; }();
                    put_u32(head, u);
                    break;
                }
                default: break;
            }
        }
        std::vector<uint64_t> offsets(tensors.size(), 0);
        uint64_t off = 0;
        for (size_t i = 0; i < tensors.size(); ++i) {
            const Tensor& t = tensors[i];
            put_str(head, t.name);
            put_u32(head, static_cast<uint32_t>(t.dims.size()));
            for (uint64_t d : t.dims) put_u64(head, d);
            put_u32(head, t.type);
            put_u64(head, off);
            offsets[i] = off;
            off += t.f32.size() * (t.type == kF16 ? 2 : 4);
        }
        while (head.size() % 32) head.push_back(0);
        std::vector<uint8_t> data;
        for (const Tensor& t : tensors) {
            for (float v : t.f32) {
                uint16_t h = t.type == kF16 ? f32_to_f16(v) : 0;
                if (t.type == kF32T) {
                    uint32_t u;
                    std::memcpy(&u, &v, 4);
                    data.insert(data.end(), reinterpret_cast<uint8_t*>(&u), reinterpret_cast<uint8_t*>(&u) + 4);
                } else {
                    data.insert(data.end(), reinterpret_cast<uint8_t*>(&h), reinterpret_cast<uint8_t*>(&h) + 2);
                }
            }
        }
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(head.data()), static_cast<std::streamsize>(head.size()));
        f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
};

// ---------------------------------------------------------------- the synthetic tower
struct Shape {
    int depth = 2, hidden = 8, heads = 2, inter = 16, out = 6, patch = 4, temporal = 2, merge = 2,
        channels = 3, npos = 9;
};

/// Distinct, order-revealing values: a transposed read cannot look plausible.
float val(size_t seed, size_t i) {
    return static_cast<float>((i * 37 + seed * 101 + 11) % 521) / 64.0f - 4.0f;
}

Gguf make_projector(const Shape& s, size_t seed = 1) {
    Gguf g;
    const int H = s.hidden, I = s.inter, M = s.merge * s.merge, P = s.patch, C = s.channels, T = s.temporal;
    g.m_str("general.architecture", "clip");
    g.m_u32("clip.has_vision_encoder", 1);
    g.m_str("clip.projector_type", "qwen3vl_merger");
    g.m_u32("clip.vision.block_count", static_cast<uint32_t>(s.depth));
    g.m_u32("clip.vision.embedding_length", static_cast<uint32_t>(H));
    g.m_u32("clip.vision.attention.head_count", static_cast<uint32_t>(s.heads));
    g.m_u32("clip.vision.feed_forward_length", static_cast<uint32_t>(I));
    g.m_u32("clip.vision.projection_dim", static_cast<uint32_t>(s.out));
    g.m_u32("clip.vision.patch_size", static_cast<uint32_t>(P));
    g.m_u32("clip.vision.spatial_merge_size", static_cast<uint32_t>(s.merge));
    g.m_f32("clip.vision.attention.layer_norm_epsilon", 1e-6f);

    for (int t = 0; t < T; ++t) {
        std::vector<float> v(static_cast<size_t>(P) * P * C * H);
        for (size_t i = 0; i < v.size(); ++i) v[i] = val(seed + 3 * t, i);
        g.t_f16(t == 0 ? "v.patch_embd.weight" : "v.patch_embd.weight." + std::to_string(t),
                {static_cast<uint64_t>(P), static_cast<uint64_t>(P), static_cast<uint64_t>(C),
                 static_cast<uint64_t>(H)},
                v);
    }
    {
        std::vector<float> b(H);
        for (int i = 0; i < H; ++i) b[i] = val(seed + 7, i);
        g.t_f32("v.patch_embd.bias", {static_cast<uint64_t>(H)}, b);
    }
    {
        std::vector<float> v(static_cast<size_t>(H) * s.npos);
        for (size_t i = 0; i < v.size(); ++i) v[i] = val(seed + 11, i);
        g.t_f32("v.position_embd.weight", {static_cast<uint64_t>(H), static_cast<uint64_t>(s.npos)}, v);
    }
    for (int l = 0; l < s.depth; ++l) {
        const std::string b = "v.blk." + std::to_string(l) + ".";
        auto linear = [&](const std::string& n, int in, int out, size_t sd) {
            std::vector<float> v(static_cast<size_t>(in) * out);
            for (size_t i = 0; i < v.size(); ++i) v[i] = val(sd, i);
            g.t_f32(n + ".weight", {static_cast<uint64_t>(in), static_cast<uint64_t>(out)}, v);
            std::vector<float> bias(out);
            for (int i = 0; i < out; ++i) bias[i] = val(sd + 1, i);
            g.t_f32(n + ".bias", {static_cast<uint64_t>(out)}, bias);
        };
        for (const char* nm : {"ln1", "ln2"}) {
            std::vector<float> v(H);
            for (int i = 0; i < H; ++i) v[i] = val(seed + 13 + l, i);
            g.t_f32(b + nm + ".weight", {static_cast<uint64_t>(H)}, v);
            std::vector<float> vb(H);
            for (int i = 0; i < H; ++i) vb[i] = val(seed + 17 + l, i);
            g.t_f32(b + nm + ".bias", {static_cast<uint64_t>(H)}, vb);
        }
        linear(b + "attn_qkv", H, 3 * H, seed + 19);
        linear(b + "attn_out", H, H, seed + 23);
        linear(b + "ffn_up", H, I, seed + 29);
        linear(b + "ffn_down", I, H, seed + 31);
    }
    for (const char* nm : {"weight", "bias"}) {
        std::vector<float> v(H);
        for (int i = 0; i < H; ++i) v[i] = val(seed + 37, i);
        g.t_f32(std::string("v.post_ln.") + nm, {static_cast<uint64_t>(H)}, v);
    }
    auto merger = [&](const std::string& pre, int in, int out, size_t sd) {
        std::vector<float> v(static_cast<size_t>(in) * out);
        for (size_t i = 0; i < v.size(); ++i) v[i] = val(sd, i);
        g.t_f32(pre + ".weight", {static_cast<uint64_t>(in), static_cast<uint64_t>(out)}, v);
        std::vector<float> b(out);
        for (int i = 0; i < out; ++i) b[i] = val(sd + 1, i);
        g.t_f32(pre + ".bias", {static_cast<uint64_t>(out)}, b);
    };
    merger("mm.0", H * M, H * M, seed + 41);
    merger("mm.2", H * M, s.out, seed + 43);
    // One deepstack merger, hanging off block 1 (the names state which block).
    {
        std::vector<float> n(static_cast<size_t>(H) * M);
        for (size_t i = 0; i < n.size(); ++i) n[i] = val(seed + 47, i);
        g.t_f32("v.deepstack.1.norm.weight", {static_cast<uint64_t>(H) * M}, n);
        std::vector<float> nb(static_cast<size_t>(H) * M);
        for (size_t i = 0; i < nb.size(); ++i) nb[i] = val(seed + 53, i);
        g.t_f32("v.deepstack.1.norm.bias", {static_cast<uint64_t>(H) * M}, nb);
        merger("v.deepstack.1.fc1", H * M, H * M, seed + 59);
        merger("v.deepstack.1.fc2", H * M, s.out, seed + 61);
    }
    return g;
}

/// The f32 the loader should end up with for a ggml [in, out] weight.
bool near(float got, float want, float tol = 1e-6f) { return std::fabs(got - want) <= tol * (1.f + std::fabs(want)); }

float bf16_round(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;      // round to nearest even, then truncate
    u <<= 16;
    float out;
    std::memcpy(&out, &u, 4);
    return out;
}

// ------------------------------------------------------- Qwen2.5-VL's windowed tower
// Same file format, different tower: split q/k/v, RMSNorm (no bias), SwiGLU, a merger whose
// norm is over `hidden`, and attention inside 112-px windows except in the blocks the
// pattern names. Its metadata has NO merge size and NO FFN width, so those come from the
// shapes -- which is what this checks.
Gguf make_projector_qwen25(const Shape& s, size_t seed = 1) {
    Gguf g;
    const int H = s.hidden, I = s.inter, M = s.merge * s.merge, P = s.patch, C = s.channels, T = s.temporal;
    g.m_str("general.architecture", "clip");
    g.m_u32("clip.has_vision_encoder", 1);
    g.m_str("clip.projector_type", "qwen2.5vl_merger");
    g.m_u32("clip.vision.block_count", static_cast<uint32_t>(s.depth));
    g.m_u32("clip.vision.embedding_length", static_cast<uint32_t>(H));
    g.m_u32("clip.vision.attention.head_count", static_cast<uint32_t>(s.heads));
    g.m_u32("clip.vision.projection_dim", static_cast<uint32_t>(s.out));
    g.m_u32("clip.vision.patch_size", static_cast<uint32_t>(P));
    g.m_u32("clip.vision.feed_forward_length", 0);          // as the real file: the shapes say it
    g.m_u32("clip.vision.n_wa_pattern", static_cast<uint32_t>(s.depth));   // every depth-th block is full
    g.m_f32("clip.vision.attention.layer_norm_epsilon", 1e-6f);

    for (int t = 0; t < T; ++t) {
        std::vector<float> v(static_cast<size_t>(P) * P * C * H);
        for (size_t i = 0; i < v.size(); ++i) v[i] = val(seed + 3 * t, i);
        g.t_f16(t == 0 ? "v.patch_embd.weight" : "v.patch_embd.weight." + std::to_string(t),
                {static_cast<uint64_t>(P), static_cast<uint64_t>(P), static_cast<uint64_t>(C),
                 static_cast<uint64_t>(H)}, v);
    }
    {   // llama.cpp writes this one and this tower does not read it (2-D RoPE)
        std::vector<float> v(100);
        for (size_t i = 0; i < v.size(); ++i) v[i] = val(seed + 5, i);
        g.t_f32("v.position_embd.weight", {10, 10}, v);
    }
    {
        std::vector<float> v(H);
        for (int i = 0; i < H; ++i) v[i] = val(seed + 7, i);
        g.t_f32("v.post_ln.weight", {static_cast<uint64_t>(H)}, v);
    }
    auto linear = [&](const std::string& n, int in, int out, size_t sd) {
        std::vector<float> v(static_cast<size_t>(in) * out);
        for (size_t i = 0; i < v.size(); ++i) v[i] = val(sd, i);
        g.t_f32(n + ".weight", {static_cast<uint64_t>(in), static_cast<uint64_t>(out)}, v);
        std::vector<float> b(out);
        for (int i = 0; i < out; ++i) b[i] = val(sd + 1, i);
        g.t_f32(n + ".bias", {static_cast<uint64_t>(out)}, b);
    };
    for (int l = 0; l < s.depth; ++l) {
        const std::string b = "v.blk." + std::to_string(l) + ".";
        for (const char* nm : {"ln1", "ln2"}) {
            std::vector<float> v(H);
            for (int i = 0; i < H; ++i) v[i] = val(seed + 11 + l, i);
            g.t_f32(b + nm + ".weight", {static_cast<uint64_t>(H)}, v);   // RMSNorm: no bias
        }
        linear(b + "attn_q", H, H, seed + 13);
        linear(b + "attn_k", H, H, seed + 17);
        linear(b + "attn_v", H, H, seed + 19);
        linear(b + "attn_out", H, H, seed + 23);
        linear(b + "ffn_gate", H, I, seed + 29);
        linear(b + "ffn_up", H, I, seed + 31);
        linear(b + "ffn_down", I, H, seed + 37);
    }
    auto merger = [&](const std::string& pre, int in, int out, size_t sd) {
        std::vector<float> v(static_cast<size_t>(in) * out);
        for (size_t i = 0; i < v.size(); ++i) v[i] = val(sd, i);
        g.t_f32(pre + ".weight", {static_cast<uint64_t>(in), static_cast<uint64_t>(out)}, v);
        std::vector<float> b(out);
        for (int i = 0; i < out; ++i) b[i] = val(sd + 1, i);
        g.t_f32(pre + ".bias", {static_cast<uint64_t>(out)}, b);
    };
    merger("mm.0", H * M, H * M, seed + 41);
    merger("mm.2", H * M, s.out, seed + 43);
    return g;
}

int run_qwen25() {
    using namespace open_qwen36::vision;
    Shape s;
    s.depth = 2;
    s.hidden = 8;
    s.heads = 2;
    s.inter = 20;
    s.out = 6;
    s.patch = 2;
    s.temporal = 2;
    s.merge = 2;
    s.channels = 3;
    s.npos = 0;
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "oflm-vit-gguf-test" / "qwen25";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "mmproj.gguf").string();
    make_projector_qwen25(s).write(path);
    // the window is NOT in the projector; config.json states it (and this is the shape a
    // GGUF install keeps from the base repo or writes from the projector)
    {
        std::ofstream f((dir / "config.json").string());
        f << R"({"model_type":"qwen2_5_vl","vision_config":{"window_size":8,"fullatt_block_indexes":[1]}})";
    }

    const VitConfig cfg = VitConfig::from_gguf(path);
    check(cfg.family == VitFamily::Qwen25VL, "qwen2.5: family is the windowed tower");
    check(cfg.depth == 2 && cfg.hidden == 8 && cfg.heads == 2 && cfg.head_dim == 4, "qwen2.5: geometry from clip.*");
    check(cfg.inter == 20, "qwen2.5: FFN width from the block's shape (the metadata says 0)");
    check(cfg.merge == 2 && cfg.out == 6, "qwen2.5: merge from the merger's width, out from projection_dim");
    check(cfg.window == 8 && cfg.fullatt.size() == 1 && cfg.fullatt[0] == 1, "qwen2.5: window + full-attention blocks from config.json");
    check(cfg.temporal == 2 && cfg.channels == 3 && cfg.npos == 0, "qwen2.5: temporal slices, channels, no position table");

    const VitWeights w = load_vit(path, cfg);
    {
        const Gguf fresh = make_projector_qwen25(s);   // kept alive: `t` points into it
        const Tensor* t = nullptr;
        for (const Tensor& x : fresh.tensors)
            if (x.name == "v.blk.0.attn_k.weight") t = &x;
        bool ok = t != nullptr;
        if (ok)
            for (int o = 0; ok && o < s.hidden; ++o)
                for (int i = 0; ok && i < s.hidden; ++i)
                    ok = near(open_qwen36::bf16_to_f32(w.blocks[0].k.w[static_cast<size_t>(o) * s.hidden + i]),
                              bf16_round(t->f32[static_cast<size_t>(i) + static_cast<size_t>(s.hidden) * o]));
        check(ok, "qwen2.5: split k_proj transposed to [out, in]");
    }
    check(w.blocks[0].ln1_b.empty(), "qwen2.5: RMSNorm has no bias");
    check(w.blocks[0].gate.out == s.inter && w.blocks[0].down.in == s.inter, "qwen2.5: SwiGLU gate/up/down shapes");
    check(w.merger_ln_w.size() == static_cast<size_t>(s.hidden), "qwen2.5: merger norm over hidden");
    check(w.merger_fc2.out == s.out && w.merger_fc2.in == s.hidden * 4, "qwen2.5: merger fc2 [out, hidden*merge^2]");

    {
        // a projector whose window cannot be found is refused, not guessed
        const std::filesystem::path lonely = std::filesystem::temp_directory_path() / "oflm-vit-gguf-test" / "qwen25-nowindow";
        std::filesystem::create_directories(lonely);
        const std::string p2 = (lonely / "mmproj.gguf").string();
        Gguf g2 = make_projector_qwen25(s);
        for (Meta& m : g2.kv)
            if (m.key == "clip.vision.n_wa_pattern") m.u32 = 0;   // no pattern either
        g2.write(p2);
        bool threw = false;
        try {
            VitConfig::from_gguf(p2);
        } catch (const std::exception& e) {
            threw = std::string(e.what()).find("window") != std::string::npos;
        }
        check(threw, "qwen2.5: refuses a projector with no window anywhere");
    }
    return g_failed;
}

int run() {
    using namespace open_qwen36::vision;
    const Shape s;
    const Gguf g = make_projector(s);
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "oflm-vit-gguf-test";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "mmproj.gguf").string();
    g.write(path);
    std::printf("wrote %s (%zu tensors)\n", path.c_str(), g.tensors.size());

    const VitConfig cfg = VitConfig::from_gguf(path);
    check(cfg.family == VitFamily::Qwen3VL, "family is Qwen3-VL");
    check(cfg.depth == s.depth && cfg.hidden == s.hidden && cfg.heads == s.heads, "depth / hidden / heads from clip.*");
    check(cfg.head_dim == s.hidden / s.heads, "head_dim = hidden / heads");
    check(cfg.inter == s.inter && cfg.out == s.out, "FFN width and projection dim from clip.*");
    check(cfg.patch == s.patch && cfg.merge == s.merge && cfg.channels == s.channels, "patch / merge / channels");
    check(cfg.temporal == s.temporal, "temporal patch size = the number of patch_embd slices");
    check(cfg.npos == s.npos, "position rows from the tensor's own shape");
    check(cfg.deepstack.size() == 1 && cfg.deepstack[0] == 1, "deepstack index from the tensor name");

    const VitWeights w = load_vit(path, cfg);
    const int H = s.hidden, I = s.inter, M = s.merge * s.merge, P = s.patch, C = s.channels, T = s.temporal;

    // The patch embed: [P, P, C, H] with the patch WIDTH fastest -> (C, T, P_h, P_w).
    {
        bool ok = true;
        int bad = 0;
        for (int t = 0; t < T && ok; ++t) {
            const std::string nm = t == 0 ? std::string("v.patch_embd.weight")
                                          : "v.patch_embd.weight." + std::to_string(t);
            const Tensor* tp = nullptr;
            for (const Tensor& x : g.tensors)
                if (x.name == nm) tp = &x;
            if (!tp) { ok = false; break; }
            const std::vector<float>& src = tp->f32;
            for (int h = 0; h < H; ++h)
                for (int c = 0; c < C; ++c)
                    for (int ph = 0; ph < P; ++ph)
                        for (int pw = 0; pw < P; ++pw) {
                            const size_t d = ((static_cast<size_t>(h) * C + c) * T + t) * P * P + ph * P + pw;
                            const float raw = src[static_cast<size_t>(pw) + P * (ph + static_cast<size_t>(P) * (c + static_cast<size_t>(C) * h))];
                            const float want = bf16_round(open_qwen36::fp16_to_f32(f32_to_f16(raw)));
                            if (!near(open_qwen36::bf16_to_f32(w.patch.w[d]), want)) {
                                ok = false;
                                if (++bad < 3) std::printf("     patch mismatch at (%d,%d,%d,%d,%d)\n", h, c, t, ph, pw);
                            }
                        }
        }
        check(ok, "patch embed: one conv per temporal slice, reassembled as (C, T, P_h, P_w)");
    }

    // attn_qkv: ggml [hidden, 3*hidden] -> [3*hidden, hidden].
    {
        const Tensor* t = nullptr;
        for (const Tensor& x : g.tensors)
            if (x.name == "v.blk.0.attn_qkv.weight") t = &x;
        bool ok = t != nullptr;
        if (ok)
            for (int o = 0; ok && o < 3 * H; ++o)
                for (int i = 0; ok && i < H; ++i)
                    ok = near(open_qwen36::bf16_to_f32(w.blocks[0].qkv.w[static_cast<size_t>(o) * H + i]),
                              bf16_round(t->f32[static_cast<size_t>(i) + static_cast<size_t>(H) * o]));
        check(ok, "attn_qkv: [in, out] -> [out, in], q|k|v kept in order");
    }
    // mm.2: ggml [H*M, out] -> [out, H*M].
    {
        const Tensor* t = nullptr;
        for (const Tensor& x : g.tensors)
            if (x.name == "mm.2.weight") t = &x;
        bool ok = t != nullptr;
        if (ok)
            for (int o = 0; ok && o < s.out; ++o)
                for (int i = 0; ok && i < H * M; ++i)
                    ok = near(open_qwen36::bf16_to_f32(w.merger_fc2.w[static_cast<size_t>(o) * H * M + i]),
                              bf16_round(t->f32[static_cast<size_t>(i) + static_cast<size_t>(H * M) * o]));
        check(ok, "merger fc2: [in, out] -> [out, in]");
    }
    check(w.merger_ln_w.size() == static_cast<size_t>(H), "the tower merger normalises over hidden (before the 2x2 concat)");
    check(w.deepstack.size() == 1 && w.deepstack[0].ln_w.size() == static_cast<size_t>(H) * M,
          "the deepstack merger normalises over hidden * merge^2");
    check(w.pos.size() == static_cast<size_t>(s.npos) * H, "position table read at its own shape");
    check(w.blocks[0].fc2.out == H && w.blocks[0].fc2.in == I, "FFN down is [hidden, inter] after the transpose");

    // Refusals: each is a file this reader must not believe.
    {
        Gguf other = make_projector(s);
        other.kv[0].str = "llama";
        const std::string p2 = (dir / "not-clip.gguf").string();
        other.write(p2);
        bool threw = false;
        try {
            VitConfig::from_gguf(p2);
        } catch (const std::exception& e) {
            threw = std::string(e.what()).find("not a vision projector") != std::string::npos;
        }
        check(threw, "refuses a GGUF that is not a vision projector");
    }
    {
        Gguf other = make_projector(s);
        for (Meta& m : other.kv)
            if (m.key == "clip.projector_type") m.str = "siglip_merger";
        const std::string p2 = (dir / "other-projector.gguf").string();
        other.write(p2);
        bool threw = false;
        try {
            VitConfig::from_gguf(p2);
        } catch (const std::exception& e) {
            threw = std::string(e.what()).find("this reader runs") != std::string::npos;
        }
        check(threw, "refuses a projector this reader does not run, naming the ones it does");
    }
    {
        Gguf other = make_projector(s);
        other.t_f32("v.unexpected.weight", {1}, {1.f});
        const std::string p2 = (dir / "extra-tensor.gguf").string();
        other.write(p2);
        bool threw = false;
        try {
            load_vit(p2, cfg);
        } catch (const std::exception& e) {
            threw = std::string(e.what()).find("tensors where this tower reads") != std::string::npos;
        }
        check(threw, "refuses a file with a tensor the tower does not read");
    }

    std::printf("%s\n", g_failed ? "FAILED" : "all checks passed");
    return g_failed ? 1 : 0;
}

}  // namespace

int main() {
    const int a = run();
    const int b = run_qwen25();
    return (a || b) ? 1 : 0;
}

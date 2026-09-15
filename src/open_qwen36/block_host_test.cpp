// Traces: OPEN-PREFILL-BATCH (canonical spec: specs/open-engine/spec.md)
// The block prefill's host stages against open_kernels/model/replica_block.py's fixture:
//   python open_kernels/model/replica_block.py --fixture <dir>
//   block_host_test.exe <dir>
// Every stage runs on the fixture's inputs and is compared with the numpy result written
// beside them (f32 arrays; inv_freq f64; idx i32). No XRT, no model.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "open_qwen36/block_host.hpp"
#include "open_qwen36/q4nx_file.hpp"

using namespace open_qwen36;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

template <class T>
std::vector<T> read(const std::string& dir, const std::string& name) {
    std::ifstream f(dir + "/" + name, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("no " + name);
    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<T> v(static_cast<size_t>(n) / sizeof(T));
    f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

std::map<std::string, size_t> shapes(const std::string& dir) {
    std::ifstream f(dir + "/shapes.txt");
    std::map<std::string, size_t> m;
    std::string k;
    size_t v;
    while (f >> k >> v) m[k] = v;
    return m;
}

// max |a - b| against the reference's own scale
double maxrel(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e9;
    double m = 0, scale = 1e-30;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
        scale = std::max(scale, std::fabs(static_cast<double>(b[i])));
    }
    return m / scale;
}

std::vector<uint16_t> to_bf16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = f32_to_bf16(v[i]);
    return o;
}

std::vector<float> from_bf16(const std::vector<uint16_t>& v) {
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = bf16_to_f32(v[i]);
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: block_host_test <fixture dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    const auto S = shapes(dir);
    const size_t T = S.at("T"), t_real = S.at("t_real"), hid = S.at("hid");
    const double tol = 1e-4;

    // ---- rmsnorm: a row of ones with a unit weight is 1 / sqrt(1 + eps)
    {
        std::vector<float> x(2 * hid, 1.f), w(hid, 2.f), out(2 * hid);
        host::rmsnorm_rows(x.data(), 2, hid, w.data(), 1e-6, out.data());
        check(std::fabs(out[0] - 2.f / std::sqrt(1.f + 1e-6f)) < 1e-6 && out[hid + 3] == out[0], "rmsnorm_rows");
    }

    // ---- the DeltaNet stage
    {
        host::DeltaGeom g;
        g.T = T; g.t_real = t_real; g.hid = hid;
        g.key_heads = S.at("key_heads"); g.value_heads = S.at("value_heads"); g.head_dim = S.at("head_dim");
        g.taps = S.at("taps"); g.lanes = S.at("lanes"); g.s_rows = S.at("s_rows");
        auto qkv = read<float>(dir, "qkv.f32"), z = read<float>(dir, "z.f32"), xn = read<float>(dir, "xn.f32");
        auto convw = read<float>(dir, "convw.f32"), Wa = read<float>(dir, "Wa.f32"), Wb = read<float>(dir, "Wb.f32");
        auto A = read<float>(dir, "A.f32"), dtb = read<float>(dir, "dtb.f32"), nw = read<float>(dir, "nw.f32");
        auto conv_state = to_bf16(read<float>(dir, "conv_state.f32"));
        auto Sst = read<float>(dir, "S.f32");
        std::vector<float> og(T * g.value_heads * g.head_dim);
        host::deltanet_block(g, qkv.data(), z.data(), xn.data(), convw.data(), Wa.data(), Wb.data(), A.data(), dtb.data(),
                             nw.data(), conv_state.data(), Sst.data(), og.data());
        const double e1 = maxrel(og, read<float>(dir, "out_og_lin.f32"));
        const double e2 = maxrel(Sst, read<float>(dir, "out_S.f32"));
        const auto cs_ref = to_bf16(read<float>(dir, "out_conv_state.f32"));
        check(e1 < 1e-3, "deltanet_block: og vs replica_block (maxrel " + std::to_string(e1) + ")");
        check(e2 < 1e-3, "deltanet_block: S vs replica_block (maxrel " + std::to_string(e2) + ")");
        check(conv_state == cs_ref, "deltanet_block: the conv state rows are the reference's, bit for bit (bf16)");
        bool zero_tail = true;
        for (size_t i = t_real * g.value_heads * g.head_dim; i < og.size(); ++i) zero_tail = zero_tail && og[i] == 0.f;
        check(zero_tail, "deltanet_block: og past t_real is zero");
    }

    // ---- the attention stage
    {
        host::AttnGeom g;
        g.T = T; g.t_real = t_real; g.nh = S.at("nh"); g.kvh = S.at("kvh"); g.hd = S.at("hd"); g.rot = S.at("rot");
        g.pos0 = S.at("pos0");
        auto q = read<float>(dir, "q.f32"), k = read<float>(dir, "k.f32"), v = read<float>(dir, "v.f32");
        auto gate = read<float>(dir, "gate.f32"), qn = read<float>(dir, "qn.f32"), kn = read<float>(dir, "kn.f32");
        auto inv_freq = read<double>(dir, "inv_freq.f64");
        auto kv = to_bf16(read<float>(dir, "kv.f32"));
        const size_t kv_row = 2 * g.kvh * g.hd;
        std::vector<float> og(T * g.nh * g.hd);
        host::attention_block(g, q.data(), k.data(), v.data(), gate.data(), qn.data(), kn.data(), inv_freq.data(),
                              kv.data(), kv_row, og.data());
        const double e1 = maxrel(og, read<float>(dir, "out_og_att.f32"));
        const auto kv_ref = read<float>(dir, "out_kv.f32");
        const double e2 = maxrel(from_bf16(kv), kv_ref);
        check(e1 < 1e-3, "attention_block: og vs replica_block (maxrel " + std::to_string(e1) + ")");
        check(e2 < 4e-3, "attention_block: the KV rows vs replica_block, within a bf16 ulp (maxrel " + std::to_string(e2) + ")");
        bool untouched = true;
        const auto kv_in = read<float>(dir, "kv.f32");
        for (size_t i = 0; i < g.pos0 * kv_row; ++i) untouched = untouched && bf16_to_f32(kv[i]) == kv_in[i];
        for (size_t i = (g.pos0 + t_real) * kv_row; i < kv.size(); ++i) untouched = untouched && bf16_to_f32(kv[i]) == kv_in[i];
        check(untouched, "attention_block: rows before the block and past t_real are untouched");
    }

    // ---- the router
    {
        const size_t E = S.at("E"), topk = S.at("topk");
        auto xm = read<float>(dir, "xm.f32"), Wr = read<float>(dir, "Wr.f32");
        std::vector<float> probs(T * E), w(T * topk);
        std::vector<int32_t> idx(T * topk);
        host::router_block(T, hid, E, topk, xm.data(), Wr.data(), probs.data(), idx.data(), w.data());
        const double e1 = maxrel(probs, read<float>(dir, "out_probs.f32"));
        const double e2 = maxrel(w, read<float>(dir, "out_w.f32"));
        check(e1 < 1e-3 && e2 < 1e-3, "router_block: probabilities and top-k weights vs replica_block");
        check(idx == read<int32_t>(dir, "out_idx.i32"), "router_block: the top-k ids");
    }

    // ---- the GEMM operand helpers against the obvious loops
    {
        const size_t T2 = 64, K2 = 128, N2 = 96;
        std::vector<float> x(T2 * K2), y(N2 * T2), yt(T2 * N2);
        for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>((i * 7919) % 1000) / 37.f - 13.f;
        for (size_t i = 0; i < y.size(); ++i) y[i] = static_cast<float>(i % 251) - 100.f;
        std::vector<uint16_t> tiled(K2 * T2), ref(K2 * T2);
        host::tile_x(x.data(), T2, K2, tiled.data());
        size_t w = 0;
        for (size_t kb = 0; kb < K2 / 64; ++kb)
            for (size_t nb = 0; nb < T2 / 32; ++nb)
                for (size_t si = 0; si < 8; ++si)
                    for (size_t ti = 0; ti < 4; ++ti)
                        for (size_t s = 0; s < 8; ++s)
                            for (size_t t = 0; t < 8; ++t)
                                ref[w++] = f32_to_bf16(x[(nb * 32 + ti * 8 + t) * K2 + kb * 64 + si * 8 + s]);
        check(tiled == ref, "tile_x: the GEMM's k,n tiled bf16 layout");
        host::transpose(y.data(), N2, T2, yt.data());
        bool ok = true;
        for (size_t n = 0; n < N2; ++n)
            for (size_t t = 0; t < T2; ++t) ok = ok && yt[t * N2 + n] == y[n * T2 + t];
        check(ok, "transpose: [N, T] -> [T, N]");
    }

    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

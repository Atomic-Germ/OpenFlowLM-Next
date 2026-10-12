// Traces: OPEN-PREFILL-BATCH (canonical spec: specs/open-engine/spec.md)
// The block prefill's host stages against open_kernels/model/replica_block.py's fixture:
//   python open_kernels/model/replica_block.py --fixture <dir>
//   block_host_test.exe <dir>
// Every stage runs on the fixture's inputs and is compared with the numpy result written
// beside them (f32 arrays; inv_freq f64; idx i32). No XRT, no model.
#include <algorithm>
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

float bf16r(float x) { return bf16_to_f32(f32_to_bf16(x)); }
float silu(float x) { return x / (1.0f + std::exp(-x)); }
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float softplus(float x) { return x > 0 ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x)); }

void rms_vec(const float* x, size_t d, const float* w, double eps, float* out) {
    double ss = 0;
    for (size_t j = 0; j < d; ++j) ss += static_cast<double>(x[j]) * x[j];
    const float r = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(d) + eps));
    for (size_t j = 0; j < d; ++j) out[j] = x[j] * r * w[j];
}

// block_host.cpp's deltanet_block with no intrinsics anywhere: the oracle its AVX2 conv and
// alpha / beta loops have to match bit for bit.
void deltanet_scalar(const host::DeltaGeom& g, const float* qkv, const float* z, const float* xn,
                     const float* convw, const float* Wa, const float* Wb, const float* A, const float* dtb,
                     const float* nw, uint16_t* conv_state, float* S, float* og) {
    const size_t dim = g.head_dim, key_w = g.key_heads * dim, vw = g.value_heads * dim, nch = 2 * key_w + vw;
    const size_t grp = g.value_heads / g.key_heads, R = g.t_real, pre = g.taps - 1;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(dim));
    std::fill(og, og + g.T * vw, 0.f);

    std::vector<float> carry(pre * nch);
    for (size_t i = 0; i < carry.size(); ++i) carry[i] = bf16_to_f32(conv_state[i]);
    std::vector<float> Q(R * key_w), Kk(R * key_w), V(R * vw), decay(R * g.value_heads), beta(R * g.value_heads);
    std::vector<float> c(nch), al(g.lanes), be(g.lanes);
    for (size_t t = 0; t < R; ++t) {
        std::fill(c.begin(), c.end(), 0.f);
        for (size_t r = 0; r < g.taps; ++r) {
            const long long s = static_cast<long long>(t) - static_cast<long long>(pre) + static_cast<long long>(r);
            const float* cw = convw + r * nch;
            if (s < 0) {
                const float* v = carry.data() + (static_cast<size_t>(s) + pre) * nch;
                for (size_t j = 0; j < nch; ++j) c[j] += cw[j] * v[j];
            } else {
                const float* v = qkv + static_cast<size_t>(s) * nch;
                for (size_t j = 0; j < nch; ++j) c[j] += cw[j] * bf16r(v[j]);
            }
        }
        for (size_t j = 0; j < nch; ++j) c[j] = silu(c[j]);
        for (size_t hh = 0; hh < g.key_heads; ++hh)
            for (int which = 0; which < 2; ++which) {
                const float* src = c.data() + which * key_w + hh * dim;
                float* dst = (which ? Kk : Q).data() + t * key_w + hh * dim;
                double ss = 0;
                for (size_t j = 0; j < dim; ++j) ss += static_cast<double>(src[j]) * src[j];
                const float r = static_cast<float>(1.0 / std::sqrt(ss + 1e-6));
                for (size_t j = 0; j < dim; ++j) dst[j] = src[j] * r;
            }
        std::copy(c.begin() + 2 * key_w, c.end(), V.begin() + t * vw);
        const float* x = xn + t * g.hid;
        for (size_t h = 0; h < g.lanes; ++h) {
            float a = 0.f, b = 0.f;
            for (size_t i = 0; i < g.hid; ++i) {
                a += x[i] * Wa[i * g.lanes + h];
                b += x[i] * Wb[i * g.lanes + h];
            }
            al[h] = a;
            be[h] = b;
        }
        for (size_t h = 0; h < g.value_heads; ++h) {
            decay[t * g.value_heads + h] = std::exp(A[h] * softplus(al[h] + dtb[h]));
            beta[t * g.value_heads + h] = sigmoid(be[h]);
        }
    }
    for (size_t r = 0; r < pre; ++r) {
        const long long s = static_cast<long long>(R) - static_cast<long long>(pre) + static_cast<long long>(r);
        if (s < 0) {
            std::copy(conv_state + (R + r) * nch, conv_state + (R + r + 1) * nch, conv_state + r * nch);
        } else {
            const float* row = qkv + static_cast<size_t>(s) * nch;
            for (size_t j = 0; j < nch; ++j) conv_state[r * nch + j] = f32_to_bf16(row[j]);
        }
    }
    std::vector<float> tv(dim), delta(dim), o(dim), on(dim);
    for (size_t h = 0; h < g.value_heads; ++h) {
        float* Sh = S + h * g.s_rows * dim;
        for (size_t t = 0; t < R; ++t) {
            const float* kk = Kk.data() + t * key_w + (h / grp) * dim;
            const float* qq = Q.data() + t * key_w + (h / grp) * dim;
            const float* v = V.data() + t * vw + h * dim;
            const float dc = decay[t * g.value_heads + h], bt = beta[t * g.value_heads + h];
            std::fill(tv.begin(), tv.end(), 0.f);
            for (size_t i = 0; i < dim; ++i) {
                const float* Si = Sh + i * dim;
                const float ki = kk[i];
                for (size_t j = 0; j < dim; ++j) tv[j] += ki * (Si[j] * dc);
            }
            for (size_t j = 0; j < dim; ++j) delta[j] = bt * (v[j] - tv[j]);
            std::fill(o.begin(), o.end(), 0.f);
            for (size_t i = 0; i < dim; ++i) {
                float* Si = Sh + i * dim;
                const float ki = kk[i], qi = qq[i];
                for (size_t j = 0; j < dim; ++j) {
                    const float s = Si[j] * dc + ki * delta[j];
                    Si[j] = s;
                    o[j] += s * qi;
                }
            }
            for (size_t j = 0; j < dim; ++j) o[j] *= inv_sqrt;
            rms_vec(o.data(), dim, nw, g.eps, on.data());
            float* out = og + t * vw + h * dim;
            const float* zz = z + t * vw + h * dim;
            for (size_t j = 0; j < dim; ++j) out[j] = on[j] * silu(zz[j]);
        }
    }
}

// A float built bit by bit so the dropped half decides the bf16 round: lo 0x8000 is an exact
// tie, settled by bit 16 alone -- odd at i % 8 == 1 (up), even at i % 8 == 2 (down).
float round_probe(size_t i) {
    static const uint16_t lo[] = {0x0000, 0x8000, 0x8000, 0x7FFF, 0x8001, 0xFFFF, 0x4000, 0xC000};
    const uint32_t u = (i % 3 == 0 ? 0x80000000u : 0u) | (static_cast<uint32_t>(124 + i % 9) << 23) |
                       (static_cast<uint32_t>((i * 37) % 128) << 16) | lo[i % 8];
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

float spread(size_t i, float amp, float off) {
    return amp * (static_cast<float>((i * 7919) % 2003) / 1001.f - 1.f) + off;
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

    // ---- the DeltaNet stage against the obvious scalar loops, bit for bit
    // Its conv (over the bf16 round trip) and its alpha / beta projection run as AVX2 where
    // the width fits, and the claim is that they are the scalar form's bits exactly. The
    // fixture above cannot hold them to that: it compares against a tolerance, and the
    // model's own widths are multiples of 8 and 16, so no vector remainder ever runs. These
    // widths leave one, drop below one 16-lane group, and feed the conv exact bf16 ties.
    // value_heads runs past lane 16 on purpose: lanes past it are padding nothing reads, so a
    // lane count alone would leave the projection's scalar tail with no way to reach og.
    {
        struct Case { size_t key_heads, value_heads, head_dim, taps, hid, lanes, s_rows, T, t_real; };
        static const Case cases[] = {
            {1, 19, 5, 4, 7, 19, 6, 6, 5},   // nch 105: one channel past the last 8-wide step; three lanes past the group
            {2, 4, 3, 4, 9, 4, 3, 4, 2},     // nch 24: no remainder; lanes 4: the projection all scalar; t_real < taps - 1
            {1, 19, 3, 4, 11, 32, 4, 5, 5},  // nch 63: seven past the step; lanes 32: two whole groups, no remainder
        };
        for (const auto& cse : cases) {
            host::DeltaGeom g;
            g.T = cse.T; g.t_real = cse.t_real; g.hid = cse.hid;
            g.key_heads = cse.key_heads; g.value_heads = cse.value_heads; g.head_dim = cse.head_dim;
            g.taps = cse.taps; g.lanes = cse.lanes; g.s_rows = cse.s_rows;
            const size_t dim = g.head_dim, vw = g.value_heads * dim, nch = 2 * g.key_heads * dim + vw;
            std::vector<float> qkv(g.T * nch), z(g.T * vw), xn(g.T * g.hid), convw(g.taps * nch);
            std::vector<float> Wa(g.hid * g.lanes), Wb(g.hid * g.lanes), A(g.value_heads), dtb(g.value_heads), nw(dim);
            for (size_t i = 0; i < qkv.size(); ++i) qkv[i] = round_probe(i);
            for (size_t i = 0; i < z.size(); ++i) z[i] = spread(i + 1, 1.5f, 0.f);
            for (size_t i = 0; i < xn.size(); ++i) xn[i] = spread(i + 2, 1.0f, 0.f);
            for (size_t i = 0; i < convw.size(); ++i) convw[i] = spread(i + 3, 0.4f, 0.f);
            for (size_t i = 0; i < Wa.size(); ++i) Wa[i] = spread(i + 4, 0.3f, 0.f);
            for (size_t i = 0; i < Wb.size(); ++i) Wb[i] = spread(i + 5, 0.3f, 0.f);
            for (size_t i = 0; i < A.size(); ++i) A[i] = spread(i + 6, 0.4f, -0.5f);
            for (size_t i = 0; i < dtb.size(); ++i) dtb[i] = spread(i + 7, 0.5f, 0.f);
            for (size_t i = 0; i < nw.size(); ++i) nw[i] = spread(i + 8, 0.2f, 1.f);
            std::vector<uint16_t> st0((g.taps - 1) * nch);
            for (size_t i = 0; i < st0.size(); ++i) st0[i] = f32_to_bf16(spread(i + 9, 0.6f, 0.f));
            std::vector<float> S0(g.value_heads * g.s_rows * dim);
            for (size_t i = 0; i < S0.size(); ++i) S0[i] = spread(i + 10, 0.2f, 0.f);

            auto st_v = st0, st_r = st0;
            auto S_v = S0, S_r = S0;
            std::vector<float> og_v(g.T * vw, 3.f), og_r(g.T * vw, 7.f);
            host::deltanet_block(g, qkv.data(), z.data(), xn.data(), convw.data(), Wa.data(), Wb.data(), A.data(),
                                 dtb.data(), nw.data(), st_v.data(), S_v.data(), og_v.data());
            deltanet_scalar(g, qkv.data(), z.data(), xn.data(), convw.data(), Wa.data(), Wb.data(), A.data(),
                            dtb.data(), nw.data(), st_r.data(), S_r.data(), og_r.data());
            const bool same = std::memcmp(og_v.data(), og_r.data(), og_v.size() * sizeof(float)) == 0 &&
                              std::memcmp(S_v.data(), S_r.data(), S_v.size() * sizeof(float)) == 0 && st_v == st_r;
            check(same, "deltanet_block: the scalar loops' bits exactly, at nch " + std::to_string(nch) +
                            " and lanes " + std::to_string(g.lanes));
        }
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

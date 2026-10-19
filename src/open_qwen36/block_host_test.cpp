// Traces: OPEN-PREFILL-BATCH, OPEN-SPEC-VERIFY (canonical spec: specs/open-engine/spec.md)
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
        // qkv and z read in place out of one token-major GEMM output [T, nch + vw] (qkv_ld / z_ld)
        // give the packed call's every bit (Traces: OPEN-GEMM-T2)
        {
            const size_t vw = g.value_heads * g.head_dim, nch = qkv.size() / T, ld = nch + vw;
            std::vector<float> y(T * ld);
            for (size_t t = 0; t < T; ++t) {
                std::memcpy(y.data() + t * ld, qkv.data() + t * nch, nch * 4);
                std::memcpy(y.data() + t * ld + nch, z.data() + t * vw, vw * 4);
            }
            auto cs2 = to_bf16(read<float>(dir, "conv_state.f32"));
            auto S2 = read<float>(dir, "S.f32");
            std::vector<float> og2(og.size(), 7.f);
            host::DeltaGeom g2 = g;
            g2.qkv_ld = g2.z_ld = ld;
            host::deltanet_block(g2, y.data(), y.data() + nch, xn.data(), convw.data(), Wa.data(), Wb.data(), A.data(),
                                 dtb.data(), nw.data(), cs2.data(), S2.data(), og2.data());
            check(og2 == og && S2 == Sst && cs2 == conv_state,
                  "deltanet_block: qkv / z at a token-major y's row stride, bit-identical to the packed call");
        }
    }

    // ---- the verify route's rollback (OPEN-SPEC-VERIFY)
    // A verify runs B tokens through the layer and keeps only the accepted prefix, so the
    // DeltaNet state has to come back to token k. The gate is that it lands exactly where k
    // single-token blocks carrying the state would have left it -- bit for bit, not within a
    // tolerance, because a speculative step that is only nearly lossless is not lossless.
    {
        host::DeltaGeom g;
        g.T = T; g.t_real = t_real; g.hid = hid;
        g.key_heads = S.at("key_heads"); g.value_heads = S.at("value_heads"); g.head_dim = S.at("head_dim");
        g.taps = S.at("taps"); g.lanes = S.at("lanes"); g.s_rows = S.at("s_rows");
        const size_t dim = g.head_dim, key_w = g.key_heads * dim, vw = g.value_heads * dim;
        const size_t nch = 2 * key_w + vw, s_elems = g.value_heads * g.s_rows * dim;
        auto qkv = read<float>(dir, "qkv.f32"), z = read<float>(dir, "z.f32"), xn = read<float>(dir, "xn.f32");
        auto convw = read<float>(dir, "convw.f32"), Wa = read<float>(dir, "Wa.f32"), Wb = read<float>(dir, "Wb.f32");
        auto A = read<float>(dir, "A.f32"), dtb = read<float>(dir, "dtb.f32"), nw = read<float>(dir, "nw.f32");
        const auto cs0 = to_bf16(read<float>(dir, "conv_state.f32"));
        const auto S0 = read<float>(dir, "S.f32");

        auto run_block = [&](const host::DeltaGeom& gg, const float* qkv_p, const float* z_p, const float* xn_p,
                             std::vector<uint16_t>& cs, std::vector<float>& St, host::DeltaTape* tape) {
            std::vector<float> og(gg.T * vw);
            host::deltanet_block(gg, qkv_p, z_p, xn_p, convw.data(), Wa.data(), Wb.data(), A.data(), dtb.data(),
                                 nw.data(), cs.data(), St.data(), og.data(), nullptr, tape);
        };

        // the whole block, taped
        auto cs_b = cs0;
        auto S_b = S0;
        host::DeltaTape tape;
        run_block(g, qkv.data(), z.data(), xn.data(), cs_b, S_b, &tape);
        check(tape.t_real == t_real && tape.key.size() >= t_real * key_w && tape.S0 == S0 &&
                  tape.conv_state0 == cs0,
              "DeltaTape: the block-start state and the per-token rows the replay reads");

        for (size_t k : {size_t{1}, t_real / 2, t_real - 1, t_real}) {
            auto cs_k = cs0;
            auto S_k = S0;
            host::deltanet_rollback(g, tape, k, qkv.data(), cs_k.data(), S_k.data());

            // the same k tokens one block at a time, the state carried between them
            host::DeltaGeom g1 = g;
            g1.T = 1; g1.t_real = 1;
            auto cs_s = cs0;
            auto S_s = S0;
            for (size_t t = 0; t < k; ++t)
                run_block(g1, qkv.data() + t * nch, z.data() + t * vw, xn.data() + t * hid, cs_s, S_s, nullptr);

            const std::string at = " (k = " + std::to_string(k) + ")";
            check(cs_k == cs_s, "deltanet_rollback: the conv state is the sequential one, bit for bit" + at);
            check(std::memcmp(S_k.data(), S_s.data(), s_elems * sizeof(float)) == 0,
                  "deltanet_rollback: S is the sequential one, bit for bit" + at);
            if (k == t_real) {
                check(cs_k == cs_b && std::memcmp(S_k.data(), S_b.data(), s_elems * sizeof(float)) == 0,
                      "deltanet_rollback: accepting the whole block is the block's own final state, bit for bit");
            }
        }

        // the old checks let both through: they only required each array to be at least big enough
        auto throws = [&](const host::DeltaGeom& gg, const host::DeltaTape& tt) {
            auto cs = cs0;
            auto St = S0;
            try { host::deltanet_rollback(gg, tt, 1, qkv.data(), cs.data(), St.data()); } catch (const std::runtime_error&) { return true; }
            return false;
        };
        host::DeltaGeom g_short = g;
        g_short.t_real = t_real - 1;
        check(throws(g_short, tape), "deltanet_rollback: a tape from a block of a different t_real throws");
        host::DeltaTape tape_long = tape;
        tape_long.decay.push_back(0.f);
        check(throws(g, tape_long), "deltanet_rollback: a tape with an oversized per-token array throws");
        check(!throws(g, tape), "deltanet_rollback: the matching tape does not throw");
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
        // q / k / v / gate read in place out of one token-major GEMM output [T, 2 qw + 2 kvw]
        // (Traces: OPEN-GEMM-T2): attention_block and attention_prep give the packed calls' bits
        {
            const size_t qw = g.nh * g.hd, kvw = g.kvh * g.hd, ld = 2 * qw + 2 * kvw;
            std::vector<float> y(T * ld);
            for (size_t t = 0; t < T; ++t) {
                std::memcpy(y.data() + t * ld, q.data() + t * qw, qw * 4);
                std::memcpy(y.data() + t * ld + qw, k.data() + t * kvw, kvw * 4);
                std::memcpy(y.data() + t * ld + qw + kvw, v.data() + t * kvw, kvw * 4);
                std::memcpy(y.data() + t * ld + qw + 2 * kvw, gate.data() + t * qw, qw * 4);
            }
            host::AttnGeom g2 = g;
            g2.q_ld = g2.k_ld = g2.v_ld = g2.g_ld = ld;
            auto kv2 = to_bf16(read<float>(dir, "kv.f32"));
            std::vector<float> og2(og.size());
            host::attention_block(g2, y.data(), y.data() + qw, y.data() + qw + kvw, y.data() + qw + 2 * kvw, qn.data(),
                                  kn.data(), inv_freq.data(), kv2.data(), kv_row, og2.data());
            check(og2 == og && kv2 == kv, "attention_block: inputs at a token-major y's row stride, bit-identical");
            auto kv3 = to_bf16(read<float>(dir, "kv.f32")), kv4 = kv3;
            std::vector<float> Q3(T * qw), Q4(T * qw);
            host::attention_prep(g, q.data(), k.data(), v.data(), qn.data(), kn.data(), inv_freq.data(), kv3.data(), kv_row,
                                 Q3.data());
            host::attention_prep(g2, y.data(), y.data() + qw, y.data() + qw + kvw, qn.data(), kn.data(), inv_freq.data(),
                                 kv4.data(), kv_row, Q4.data());
            check(Q3 == Q4 && kv3 == kv4, "attention_prep: inputs at a token-major y's row stride, bit-identical");
            // the NPU route's per-group host steps against the loops core.cpp used to run inline
            // (Traces: OPEN-PREFILL-ATTN)
            const size_t grp = g.nh / g.kvh;
            for (size_t gh = 0; gh < g.kvh; ++gh) {
                std::vector<uint16_t> qb(grp * T * g.hd), qb_ref(grp * T * g.hd);
                host::attn_group_queries(Q3.data(), T, g.nh, g.kvh, g.hd, gh, qb.data());
                for (size_t hl = 0; hl < grp; ++hl)
                    for (size_t t = 0; t < T; ++t)
                        for (size_t j = 0; j < g.hd; ++j)
                            qb_ref[(hl * T + t) * g.hd + j] = f32_to_bf16(Q3[t * qw + (gh * grp + hl) * g.hd + j]);
                std::vector<float> acc(grp * T * g.hd), lsum(grp * T), o1(T * qw, 0.f), o2(T * qw, 0.f);
                for (size_t i = 0; i < acc.size(); ++i) acc[i] = static_cast<float>((i * 7) % 13) - 6.f;
                for (size_t i = 0; i < lsum.size(); ++i) lsum[i] = 1.f + static_cast<float>(i % 5);
                host::attn_group_out(acc.data(), lsum.data(), y.data() + qw + 2 * kvw, T, t_real, g.nh, g.kvh, g.hd, gh,
                                     ld, o1.data());
                for (size_t hl = 0; hl < grp; ++hl)
                    for (size_t t = 0; t < t_real; ++t) {
                        const size_t r = hl * T + t, h = gh * grp + hl;
                        const float inv = 1.0f / lsum[r];
                        for (size_t j = 0; j < g.hd; ++j)
                            o2[t * qw + h * g.hd + j] =
                                acc[r * g.hd + j] * inv / (1.0f + std::exp(-gate[t * qw + h * g.hd + j]));
                    }
                check(qb == qb_ref && o1 == o2,
                      "attn_group_queries / attn_group_out: the inline loops' bits, group " + std::to_string(gh));
            }
        }
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
        // the fused rotated-basis path against hadamard_rows on a copy + tile_x, bit for bit,
        // with and without the sign vector (Traces: OPEN-HADAMARD)
        for (int with_signs = 0; with_signs < 2; ++with_signs)
            for (size_t tk : {size_t{64}, size_t{128}}) {
                const size_t T3 = 64, K3 = 256, blk = 128;
                std::vector<float> x3(T3 * K3), sg(K3), cp;
                for (size_t i = 0; i < x3.size(); ++i) x3[i] = static_cast<float>((i * 104729) % 2001) / 91.f - 11.f;
                for (size_t j = 0; j < K3; ++j) sg[j] = ((j * 37) % 5) < 2 ? -1.f : 1.f;
                cp = x3;
                host::hadamard_rows(cp.data(), T3, K3, blk, with_signs ? sg.data() : nullptr);
                std::vector<uint16_t> want(K3 * T3), got(K3 * T3), ref3(K3 * T3);
                host::tile_x(cp.data(), T3, K3, want.data(), tk);
                const std::vector<float> before = x3;
                host::hadamard_tile_x(x3.data(), T3, K3, blk, with_signs ? sg.data() : nullptr, got.data(), tk);
                check(got == want && x3 == before,
                      std::string("hadamard_tile_x: bit-identical to hadamard_rows + tile_x, tk ") + std::to_string(tk) +
                          (with_signs ? ", signed" : ""));
                // tile_x at this tile width against the obvious loops (OPEN-GEMM-T2's 128-k tiles)
                size_t w3 = 0;
                for (size_t kb = 0; kb < K3 / tk; ++kb)
                    for (size_t nb = 0; nb < T3 / 32; ++nb)
                        for (size_t si = 0; si < tk / 8; ++si)
                            for (size_t ti = 0; ti < 4; ++ti)
                                for (size_t s8 = 0; s8 < 8; ++s8)
                                    for (size_t t8 = 0; t8 < 8; ++t8)
                                        ref3[w3++] = f32_to_bf16(cp[(nb * 32 + ti * 8 + t8) * K3 + kb * tk + si * 8 + s8]);
                if (!with_signs) check(want == ref3, "tile_x: the k,n tiled layout at tk " + std::to_string(tk));
            }
        // the transform at the production block (1024, 128-k tiles), signed, against the two-step form
        {
            const size_t T3 = 64, K3 = 2048, blk = 1024;
            std::vector<float> x3(T3 * K3), sg(K3), cp;
            for (size_t i = 0; i < x3.size(); ++i) x3[i] = static_cast<float>((i * 7919) % 3001) / 77.f - 19.f;
            for (size_t j = 0; j < K3; ++j) sg[j] = ((j * 13) % 7) < 3 ? -1.f : 1.f;
            cp = x3;
            host::hadamard_rows(cp.data(), T3, K3, blk, sg.data());
            std::vector<uint16_t> want(K3 * T3), got(K3 * T3);
            host::tile_x(cp.data(), T3, K3, want.data(), 128);
            host::hadamard_tile_x(x3.data(), T3, K3, blk, sg.data(), got.data(), 128);
            check(got == want, "hadamard_tile_x: bit-identical to hadamard_rows + tile_x at block 1024, tk 128");
            // the FFN's down projection input made on the fly from the up|gate output [T, 2 ff],
            // against Core::ffn_block's loop (scalar, as MSVC compiles it there) + hadamard_tile_x
            const size_t ff = K3;
            std::vector<float> ug(T3 * 2 * ff), h(T3 * ff);
            for (size_t i = 0; i < ug.size(); ++i) ug[i] = static_cast<float>((i * 104729) % 4001) / 500.f - 4.f;
            for (size_t t = 0; t < T3; ++t) {
                const float* u = ug.data() + t * 2 * ff;
                const float* gg = u + ff;
#pragma loop(no_vector)
                for (size_t j = 0; j < ff; ++j) h[t * ff + j] = gg[j] / (1.f + std::exp(-gg[j])) * u[j];
            }
            std::vector<uint16_t> want2(K3 * T3), got2(K3 * T3);
            host::hadamard_tile_x(h.data(), T3, ff, blk, nullptr, want2.data(), 128);
            host::hadamard_tile_swiglu(ug.data(), T3, ff, 2 * ff, blk, got2.data(), 128);
            check(got2 == want2, "hadamard_tile_swiglu: bit-identical to the silu(g) * u loop + hadamard_tile_x");
            // the bfp16 activation tiles (OPEN-GEMM-T2, GQP_XBFP): every block is the conversion of
            // the very bf16 values the bf16 tiles hold -- E the block's largest exponent, each int8
            // mantissa the 8-bit significand shifted right by E - e + 1, nearest even, and E + 1 for
            // the block when a value would round out of int8 (the NPU's own conversion, matched byte
            // for byte by open_kernels/designs/bfp_cvt) -- laid out [token block pair][k block]
            // [token block % 2] x [exponent | 8 mantissas] x 8 tokens
            auto ref_tiles = [&](const std::vector<uint16_t>& bf, size_t Tn, size_t Kn) {
                const size_t NBn = Tn / 32;
                std::vector<uint8_t> o(Kn * Tn * 9 / 8);
                for (size_t kb = 0; kb < Kn / 128; ++kb)
                    for (size_t nb = 0; nb < NBn; ++nb)
                        for (size_t ti = 0; ti < 4; ++ti)
                            for (size_t si = 0; si < 16; ++si) {
                                uint8_t* v = o.data() + (kb * NBn + nb) * 4608 + ((ti / 2) * 32 + si * 2 + ti % 2) * 72;
                                for (size_t tt = 0; tt < 8; ++tt) {
                                    uint16_t b8[8];
                                    int e[8], E = 0;
                                    for (size_t s8 = 0; s8 < 8; ++s8) {
                                        b8[s8] = bf[(kb * NBn + nb) * 4096 + (si * 4 + ti) * 64 + s8 * 8 + tt];
                                        e[s8] = (b8[s8] >> 7) & 0xFF;
                                        E = std::max(E, e[s8]);
                                    }
                                    auto mant = [&](int Eb, size_t s8) {
                                        const int sig = e[s8] ? ((b8[s8] & 0x7F) | 0x80) : 0;
                                        const int sh = std::min(Eb - e[s8] + 1, 9);
                                        int q = sig >> sh;
                                        const int rem = sig & ((1 << sh) - 1), half = 1 << (sh - 1);
                                        if (rem > half || (rem == half && (q & 1))) ++q;
                                        return (b8[s8] & 0x8000) ? -q : q;
                                    };
                                    bool over = false;          // a value rounding out of int8 bumps the block
                                    for (size_t s8 = 0; s8 < 8; ++s8) {
                                        const int q = mant(E, s8);
                                        over = over || q > 127 || q < -128;
                                    }
                                    if (over) ++E;
                                    v[tt * 9] = static_cast<uint8_t>(E);
                                    for (size_t s8 = 0; s8 < 8; ++s8)
                                        v[tt * 9 + 1 + s8] = static_cast<uint8_t>(static_cast<int8_t>(mant(E, s8)));
                                }
                            }
                return o;
            };
            std::vector<uint8_t> bfp(K3 * T3 * 9 / 8), bfp2(K3 * T3 * 9 / 8);
            host::hadamard_tile_x_bfp(x3.data(), T3, K3, blk, sg.data(), bfp.data());
            host::hadamard_tile_swiglu_bfp(ug.data(), T3, ff, 2 * ff, blk, bfp2.data());
            check(bfp == ref_tiles(want, T3, K3), "hadamard_tile_x_bfp: the bf16 tiles' values as bfp16 blocks");
            check(bfp2 == ref_tiles(want2, T3, ff), "hadamard_tile_swiglu_bfp: the bf16 tiles' values as bfp16 blocks");
        }
        host::transpose(y.data(), N2, T2, yt.data());
        bool ok = true;
        for (size_t n = 0; n < N2; ++n)
            for (size_t t = 0; t < T2; ++t) ok = ok && yt[t * N2 + n] == y[n * T2 + t];
        check(ok, "transpose: [N, T] -> [T, N]");
        // split_rows on the token-major y gives what transpose_parts gives on y [N, T]
        // (Traces: OPEN-GEMM-T2)
        {
            std::vector<float> a(T2 * 40), b(T2 * 56), a2(T2 * 40), b2(T2 * 56);
            const host::TransposePart pt[2] = {{a.data(), 0, 40}, {b.data(), 40, 56}};
            const host::TransposePart pr[2] = {{a2.data(), 0, 40}, {b2.data(), 40, 56}};
            host::transpose_parts(y.data(), T2, pt, 2);
            host::split_rows(yt.data(), T2, N2, pr, 2);
            check(a == a2 && b == b2, "split_rows: a token-major y's column ranges");
        }
    }

    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

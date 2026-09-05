// stream_patch: rewrite words of a compiled mlir-aie instruction stream so one
// program per layer type serves every layer and every token.
//
// An instruction stream is a sequence of ops. The ones that matter here are
// op 1 (a BD blockwrite: 8 registers from +4) and op 0x81 (a DDR patch:
// register at +6, host buffer arg index at +8, byte offset into that buffer
// at +10). Every weight fill and every cache transfer is one 0x81, so
// re-pointing a DMA is one word plus an instruction-BO sync.
//
// Shared by open_kernels/harness/run_kernel.cpp (the .cfg driver) and
// src/open_qwen36 (the engine). Header-only, no XRT dependency: callers own
// the instruction BO and pass its mapped words. Ported from phlegm's
// npu-engine/src/decode.rs (moe2_patch_table / attn_patch_table).
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace stream_patch {

// --- layer-pool geometry (open_kernels/model/pools.py) ----------------------
// Routed experts: up/gate as interleaved 160 KB stripes from 0, four of each
// per expert; down slices from kMoePoolDown, 640 KB per expert; then the
// shared expert's up/gate/down at fixed offsets.
constexpr uint64_t kMoeStripe = 163840;
constexpr uint64_t kMoeUpBytes = 4 * kMoeStripe;  // one expert's up (or gate, or down)
constexpr uint64_t kMoeExpertBytes = 3 * kMoeUpBytes;
constexpr uint64_t kMoeDownCore = 81920;  // down rows one core streams
constexpr uint64_t kMoePoolDown = 335544320;
constexpr uint64_t kMoePoolShareUp = 503316480;
constexpr uint64_t kMoePoolShareGate = 503971840;
constexpr uint64_t kMoePoolShareDown = 504627200;
// designs/layer_x/layout.py: the KV cache row [K_t | V_t] and the position
// record (pos, nf, RoPE cos/sin). The cache capacity is not a kernel property:
// the window length, the new row's offset and the record's offset are all
// runtime-patched words, so it is whatever the KV and ptab buffers were sized to.
constexpr uint64_t kAttnKvRow = 2048;
constexpr uint64_t kAttnPtabRow = 1024;

/// Word length of the instruction-stream op starting with word `w`.
inline size_t op_len(uint32_t w) {
    switch (w) {
        case 0: return 6;
        case 1: return 12;
        case 3: return 7;
        case 0x80: return 4;
        case 0x81: return 12;
        default: return 1;
    }
}

/// One patchable weight fill: the word to rewrite, which routed slot it stands
/// for (bit 8 set = the down slice rather than a gate/up stripe) plus, for
/// `moeroute`, the fill's kind and position inside its stripe.
struct MoePatch {
    size_t word;
    size_t slot;
    uint64_t core = 0;
    uint8_t kind = 0;
    uint64_t intra = 0;
};

/// An `attnpos` patch: the word, and which of the three per-token quantities it
/// carries (0 = the KV window fill's BD length, 1 = the new row's drain offset,
/// 2 = the position record's fill offset) with the offset word's flag bits.
struct AttnPatch {
    size_t word;
    uint8_t kind;
    uint32_t flags;
};

/// `moeroute`'s table: the moe_experts design's weight fills, built against a
/// host-concatenated `[up | gate | down]` per expert slot, so each fill's
/// static offset names its (slot, core, kind) and its position inside that
/// stripe. 144 fills (4 cores x 3) or 216 (the balanced 8-core split).
inline std::vector<MoePatch> moe_table(const std::vector<uint32_t>& w, const std::string& kn) {
    std::vector<MoePatch> t;
    for (size_t i = 4; i < w.size(); i += op_len(w[i])) {
        if (w[i] != 0x81 || i + 11 >= w.size() || w[i + 8] != 0) continue;
        uint64_t off = w[i + 10];
        uint64_t rem = off % kMoeExpertBytes;
        MoePatch p{i + 10, static_cast<size_t>(off / kMoeExpertBytes)};
        if (rem < kMoeUpBytes) {
            p.kind = 0; p.core = rem / kMoeStripe; p.intra = rem % kMoeStripe;
        } else if (rem < 2 * kMoeUpBytes) {
            p.kind = 1; p.core = (rem - kMoeUpBytes) / kMoeStripe; p.intra = (rem - kMoeUpBytes) % kMoeStripe;
        } else {
            p.kind = 2; p.core = (rem - 2 * kMoeUpBytes) / kMoeDownCore;
            p.intra = (rem - 2 * kMoeUpBytes) % kMoeDownCore;
        }
        t.push_back(p);
    }
    if (t.size() != 144 && t.size() != 216)
        throw std::runtime_error("moeroute: " + kn + " has " + std::to_string(t.size()) +
                                 " weight fills, expected 144 or 216");
    return t;
}

/// `moeroute2`'s table: for designs whose fills already read the layer pool
/// (layer_x), routed slot j is compiled as expert j, so every DDR patch on
/// arg 0 landing in expert 0..7's stripe set or down slice is a placeholder;
/// the shared expert, qkv, o, ... are real offsets and are left alone.
inline std::vector<MoePatch> moe2_table(const std::vector<uint32_t>& w, const std::string& kn) {
    std::vector<MoePatch> t;
    for (size_t i = 4; i < w.size(); i += op_len(w[i])) {
        if (w[i] != 0x81 || i + 11 >= w.size() || w[i + 8] != 0) continue;
        uint64_t off = w[i + 10];
        if (off < kMoePoolDown) {
            uint64_t e = off / (8 * kMoeStripe);
            if (e < 8) t.push_back({i + 10, static_cast<size_t>(e), 0, 0, off % (8 * kMoeStripe)});
        } else if (off < kMoePoolShareUp) {
            uint64_t e = (off - kMoePoolDown) / kMoeUpBytes;
            if (e < 8)
                t.push_back({i + 10, static_cast<size_t>(e) | 0x100, 0, 0, (off - kMoePoolDown) % kMoeUpBytes});
        }
    }
    if (t.empty()) throw std::runtime_error("moeroute2: " + kn + " has no routed-expert fills");
    return t;
}

/// `attnpos`'s table for an `ax0` stream (built for the placeholder position
/// 1): the KV window fill is the patch on arg 3 at offset 0 — its length lives
/// in the BD blockwrite just before it, not in the patch — the new row's drain
/// is arg 3 at one row, the position record is arg 5.
inline std::vector<AttnPatch> attn_table(const std::vector<uint32_t>& w, const std::string& kn) {
    std::vector<AttnPatch> t;
    size_t bd_write = 0;
    bool have_bd = false;
    for (size_t i = 4; i < w.size(); i += op_len(w[i])) {
        if (w[i] == 1) { bd_write = i; have_bd = true; }
        if (w[i] != 0x81 || i + 11 >= w.size()) continue;
        // The firmware translates only the first 5 buffer args into the AIE
        // address space; patches on args 5+ carry that translation folded
        // into the offset word as +0x80000000 (mlir-aie kDDRAIEAddrOffset).
        // Keep the bit, rewrite the low 31.
        uint32_t reg = w[i + 6], arg = w[i + 8];
        uint64_t off = w[i + 10] & 0x7fffffffu;
        uint32_t flags = w[i + 10] & 0x80000000u;
        if (arg == 3 && off == 0) {
            if (!have_bd || bd_write + 2 >= w.size() || w[bd_write + 2] + 4 != reg)
                throw std::runtime_error("attnpos: " + kn + ": no BD write before the KV window fill");
            t.push_back({bd_write + 4, 0, 0});
        } else if (arg == 3 && off == kAttnKvRow) {
            t.push_back({i + 10, 1, flags});
        } else if (arg == 3) {
            throw std::runtime_error("attnpos: " + kn + ": unexpected kv transfer at offset " + std::to_string(off));
        } else if (arg == 5) {
            t.push_back({i + 10, 2, flags});
        }
    }
    for (uint8_t kind = 0; kind < 3; ++kind) {
        size_t n = 0;
        for (const auto& p : t) n += (p.kind == kind);
        if (n != 1)
            throw std::runtime_error("attnpos: " + kn + ": " + std::to_string(n) + " patches of kind " +
                                     std::to_string(kind) + ", expected 1 (window fill, row drain, record fill)");
    }
    return t;
}

/// Point a `moeroute` table's fills at experts idx[0..7] (slot 8 = the shared expert).
inline void moe_apply(uint32_t* iw, const std::vector<MoePatch>& table, const uint32_t* idx) {
    for (const auto& p : table) {
        uint64_t base;
        if (p.slot < 8) {
            uint64_t e = idx[p.slot];
            base = p.kind == 0   ? (8 * e + 2 * p.core) * kMoeStripe
                   : p.kind == 1 ? (8 * e + 2 * p.core + 1) * kMoeStripe
                                 : kMoePoolDown + e * kMoeUpBytes + p.core * kMoeDownCore;
        } else {
            base = p.kind == 0   ? kMoePoolShareUp + p.core * kMoeStripe
                   : p.kind == 1 ? kMoePoolShareGate + p.core * kMoeStripe
                                 : kMoePoolShareDown + p.core * kMoeDownCore;
        }
        iw[p.word] = static_cast<uint32_t>(base + p.intra);
    }
}

/// Point a `moeroute2` table's fills at experts idx[0..7].
inline void moe2_apply(uint32_t* iw, const std::vector<MoePatch>& table, const uint32_t* idx) {
    for (const auto& p : table) {
        uint64_t e = idx[p.slot & 0xff];
        iw[p.word] = static_cast<uint32_t>((p.slot & 0x100) == 0 ? 8 * e * kMoeStripe + p.intra
                                                                  : kMoePoolDown + e * kMoeUpBytes + p.intra);
    }
}

/// Set the cache position: the window fill reads rows [0, max(pos, 1)), the
/// new row lands at row pos, the RoPE record is ptab row pos.
inline void attn_apply(uint32_t* iw, const std::vector<AttnPatch>& table, uint64_t pos) {
    uint64_t nf = pos ? pos : 1;
    for (const auto& p : table) {
        uint64_t v = p.kind == 0   ? nf * kAttnKvRow / 4  // a BD length is in words
                     : p.kind == 1 ? pos * kAttnKvRow
                                   : pos * kAttnPtabRow;
        iw[p.word] = static_cast<uint32_t>(v) | p.flags;
    }
}

}  // namespace stream_patch

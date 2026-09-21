// stream_patch: rewrite words of a compiled mlir-aie instruction stream so one
// program per layer type serves every layer and every token.
//
// An instruction stream is a sequence of ops. The ones that matter here are
// op 1 (a BD blockwrite: 8 registers from +4) and op 0x81 (a DDR patch:
// register at +6, host buffer arg index at +8, byte offset into that buffer
// at +10). Every weight fill and every cache transfer is one 0x81, so
// re-pointing a DMA is one word plus an instruction-BO sync.
//
// The pool geometry the tables are decoded against (where the routed experts'
// stripes and down slices lie, the KV row and position-record sizes) is a
// parameter: the engine passes the manifest's (open_kernels/recipes ->
// manifest.json `layout.moe`), the .cfg harness the defaults, which are the
// Qwen3.6-27B's (open_kernels/recipes/qwen36moe.py for the 27B spec).
//
// Shared by open_kernels/harness/run_kernel.cpp (the .cfg driver) and
// src/open_qwen36 (the engine). Header-only, no XRT dependency: callers own
// the instruction BO and pass its mapped words. Ported from phlegm's
// npu-engine/src/decode.rs (moe2_patch_table / attn_patch_table).
#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace stream_patch {

/// Layer-pool geometry of the MoE block: routed experts' up/gate as
/// interleaved stripes from 0 (2 * up_bytes per expert), down slices from
/// pool_down (up_bytes each), then the shared expert's up/gate/down.
struct MoeGeometry {
    uint64_t stripe = 163840;        ///< one 128-row up (or gate) stripe
    uint64_t up_bytes = 655360;      ///< one expert's up (= gate = down)
    uint64_t down_core = 81920;      ///< down rows one core streams
    uint64_t pool_down = 335544320;
    uint64_t share_up = 503316480;
    uint64_t share_gate = 503971840;
    uint64_t share_down = 504627200;
    unsigned experts = 256;
    unsigned topk = 8;               ///< routed slots (the placeholder experts 0..topk-1)
    uint64_t expert_stripes() const { return 2 * up_bytes; }     ///< one expert's up + gate stripes
    uint64_t expert_bytes() const { return 3 * up_bytes; }       ///< moe_table's host-concatenated [up | gate | down]
};

/// The KV cache row [K_t | V_t] and the position record (pos, nf, RoPE
/// cos/sin). The cache capacity is not a kernel property: the window length,
/// the new row's offset and the record's offset are all runtime-patched
/// words, so it is whatever the KV and ptab buffers were sized to.
struct AttnGeometry {
    uint64_t kv_row = 2048;
    uint64_t ptab_row = 1024;
    uint64_t window = 0;             ///< rows of a sliding window (0 = every cached row); Gemma's local layers
};

/// The cached rows position `pos` attends to: [start, pos), streamed as nf rows (>= 1: position 0
/// streams one dummy row the core masks). Mirrors recipes/pack.py window_rows.
inline void attn_window(uint64_t pos, uint64_t window, uint64_t* start, uint64_t* nf) {
    uint64_t s = (window && pos + 1 > window) ? pos + 1 - window : 0;
    *start = s;
    uint64_t valid = pos - s;
    *nf = valid ? valid : 1;
}

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

/// An `attnpos` patch: the word, and which of the four per-token quantities it
/// carries (0 = the KV window fill's BD length, 1 = the new row's drain offset,
/// 2 = the position record's fill offset, 3 = the KV window fill's offset -- the
/// window's first row, 0 unless the layer has a sliding window) with the offset
/// word's flag bits.
struct AttnPatch {
    size_t word;
    uint8_t kind;
    uint32_t flags;
};

/// `moeroute`'s table: the moe_experts design's weight fills, built against a
/// host-concatenated `[up | gate | down]` per expert slot, so each fill's
/// static offset names its (slot, core, kind) and its position inside that
/// stripe. 144 fills (4 cores x 3) or 216 (the balanced 8-core split).
inline std::vector<MoePatch> moe_table(const std::vector<uint32_t>& w, const std::string& kn,
                                       const MoeGeometry& g = MoeGeometry{}) {
    std::vector<MoePatch> t;
    for (size_t i = 4; i < w.size(); i += op_len(w[i])) {
        if (w[i] != 0x81 || i + 11 >= w.size() || w[i + 8] != 0) continue;
        uint64_t off = w[i + 10];
        uint64_t rem = off % g.expert_bytes();
        MoePatch p{i + 10, static_cast<size_t>(off / g.expert_bytes())};
        if (rem < g.up_bytes) {
            p.kind = 0; p.core = rem / g.stripe; p.intra = rem % g.stripe;
        } else if (rem < 2 * g.up_bytes) {
            p.kind = 1; p.core = (rem - g.up_bytes) / g.stripe; p.intra = (rem - g.up_bytes) % g.stripe;
        } else {
            p.kind = 2; p.core = (rem - 2 * g.up_bytes) / g.down_core;
            p.intra = (rem - 2 * g.up_bytes) % g.down_core;
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
/// arg 0 landing in expert 0..topk-1's stripe set or down slice is a
/// placeholder; the shared expert, qkv, o, ... are real offsets and are left alone.
inline std::vector<MoePatch> moe2_table(const std::vector<uint32_t>& w, const std::string& kn,
                                        const MoeGeometry& g = MoeGeometry{}) {
    std::vector<MoePatch> t;
    for (size_t i = 4; i < w.size(); i += op_len(w[i])) {
        if (w[i] != 0x81 || i + 11 >= w.size() || w[i + 8] != 0) continue;
        uint64_t off = w[i + 10];
        if (off < g.pool_down) {
            uint64_t e = off / g.expert_stripes();
            if (e < g.topk) t.push_back({i + 10, static_cast<size_t>(e), 0, 0, off % g.expert_stripes()});
        } else if (off < g.share_up) {
            uint64_t e = (off - g.pool_down) / g.up_bytes;
            if (e < g.topk)
                t.push_back({i + 10, static_cast<size_t>(e) | 0x100, 0, 0, (off - g.pool_down) % g.up_bytes});
        }
    }
    if (t.empty()) throw std::runtime_error("moeroute2: " + kn + " has no routed-expert fills");
    return t;
}

/// `attnpos`'s table for an `ax0` stream (built for the placeholder position
/// 1): the KV window fill is the patch on arg 3 at offset 0 — its length lives
/// in the BD blockwrite just before it, not in the patch — the new row's drain
/// is arg 3 at one row, the position record is arg 5.
inline std::vector<AttnPatch> attn_table(const std::vector<uint32_t>& w, const std::string& kn,
                                         const AttnGeometry& g = AttnGeometry{}) {
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
            t.push_back({i + 10, 3, flags});
        } else if (arg == 3 && off == g.kv_row) {
            t.push_back({i + 10, 1, flags});
        } else if (arg == 3) {
            throw std::runtime_error("attnpos: " + kn + ": unexpected kv transfer at offset " + std::to_string(off));
        } else if (arg == 5) {
            t.push_back({i + 10, 2, flags});
        }
    }
    for (uint8_t kind = 0; kind < 4; ++kind) {
        size_t n = 0;
        for (const auto& p : t) n += (p.kind == kind);
        if (n != 1)
            throw std::runtime_error("attnpos: " + kn + ": " + std::to_string(n) + " patches of kind " +
                                     std::to_string(kind) + ", expected 1 (window length, row drain, record fill, window offset)");
    }
    return t;
}

/// Bytes this instruction stream moves between DDR and the array, per host
/// buffer argument.
///
/// Every DDR transfer is a BD blockwrite (op 1: eight registers from +4, the
/// FIRST of them the transfer length in WORDS) followed by the DDR patch
/// (op 0x81) that points it at a buffer argument -- `attn_table` above relies
/// on exactly that pairing to find the KV window fill's length, and
/// `attn_apply` writes `nf * kv_row / 4` into it. Summing the pairs is a
/// dispatch's own DMA traffic, which is what a GB/s reading needs.
///
/// Read it AFTER patching (`iw()`, not `words`): a runtime-sized transfer --
/// the KV window is the one that matters -- then counts what this position
/// actually streams, not what the stream was compiled for.
///
/// A patch this cannot pair with a blockwrite is skipped rather than guessed
/// at, so the total is a floor. Validate a design's number against something
/// independently known (the q8 lm head's vocab x K, a projection's pack-op
/// chunk count) before quoting a rate off it.
inline std::map<uint32_t, uint64_t> ddr_bytes(const uint32_t* w, size_t n) {
    std::map<uint32_t, uint64_t> per_arg;
    size_t bd_write = 0;
    bool have_bd = false;
    for (size_t i = 4; i < n; i += op_len(w[i])) {
        if (w[i] == 1) { bd_write = i; have_bd = true; continue; }
        if (w[i] != 0x81 || i + 11 >= n) continue;
        if (!have_bd || bd_write + 4 >= n || w[bd_write + 2] + 4 != w[i + 6]) continue;
        per_arg[w[i + 8]] += static_cast<uint64_t>(w[bd_write + 4]) * 4;
    }
    return per_arg;
}

/// The same, totalled over every buffer argument.
inline uint64_t ddr_bytes_total(const uint32_t* w, size_t n) {
    uint64_t t = 0;
    for (const auto& kv : ddr_bytes(w, n)) t += kv.second;
    return t;
}

/// Point a `moeroute` table's fills at experts idx[0..topk-1] (slot topk = the shared expert).
inline void moe_apply(uint32_t* iw, const std::vector<MoePatch>& table, const uint32_t* idx,
                      const MoeGeometry& g = MoeGeometry{}) {
    for (const auto& p : table) {
        uint64_t base;
        if (p.slot < g.topk) {
            uint64_t e = idx[p.slot];
            base = p.kind == 0   ? (e * g.expert_stripes() + 2 * p.core * g.stripe)
                   : p.kind == 1 ? (e * g.expert_stripes() + (2 * p.core + 1) * g.stripe)
                                 : g.pool_down + e * g.up_bytes + p.core * g.down_core;
        } else {
            base = p.kind == 0   ? g.share_up + p.core * g.stripe
                   : p.kind == 1 ? g.share_gate + p.core * g.stripe
                                 : g.share_down + p.core * g.down_core;
        }
        iw[p.word] = static_cast<uint32_t>(base + p.intra);
    }
}

/// Point a `moeroute2` table's fills at experts idx[0..topk-1].
inline void moe2_apply(uint32_t* iw, const std::vector<MoePatch>& table, const uint32_t* idx,
                       const MoeGeometry& g = MoeGeometry{}) {
    for (const auto& p : table) {
        uint64_t e = idx[p.slot & 0xff];
        iw[p.word] = static_cast<uint32_t>((p.slot & 0x100) == 0 ? e * g.expert_stripes() + p.intra
                                                                  : g.pool_down + e * g.up_bytes + p.intra);
    }
}

/// Set the cache position: the window fill reads rows [start, start + nf) (attn_window: every
/// cached row, or the sliding window's), the new row lands at row pos, the RoPE record is ptab
/// row pos (whose [valid | nf] counts match the same window).
inline void attn_apply(uint32_t* iw, const std::vector<AttnPatch>& table, uint64_t pos,
                       const AttnGeometry& g = AttnGeometry{}) {
    uint64_t start, nf;
    attn_window(pos, g.window, &start, &nf);
    for (const auto& p : table) {
        uint64_t v = p.kind == 0   ? nf * g.kv_row / 4  // a BD length is in words
                     : p.kind == 1 ? pos * g.kv_row
                     : p.kind == 2 ? pos * g.ptab_row
                                   : start * g.kv_row;
        iw[p.word] = static_cast<uint32_t>(v) | p.flags;
    }
}

}  // namespace stream_patch

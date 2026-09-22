#pragma once
// On-device routed-expert routing: the control words the MoE router core emits so the
// shim retargets and enqueues the routed-expert descriptors it never started.
//
// The fused whole-layer design configures the routed slots' fills but does NOT enqueue
// them (open_kernels/ironutil.py: configure_only_fill); the router helper core, which
// already computes the top-8, writes a TileControl packet stream into DDR and a shim
// MM2S channel streams it into the shim's own TileControl port, rewriting each pinned
// descriptor's DDR address and pushing it (designs/expert_fetch, all spikes PASS).
//
// Packet encoding (recovered from the spikes; the two whose comments compute the
// header by hand are core_ctrlpkt.mlir and minpkt.mlir -- see
// designs/layer_x/ondv_ctrl_ref.py, whose self-test reproduces them):
//
//   hdr = stream_id<<24 | opcode<<22 | (beats-1)<<20 | address
//   bit 31 = parity, 1 iff popcount(bits[30:0]) is EVEN          (opcode 0 = write)
//
// A retarget + enqueue of descriptor `bd` at 64-bit DDR byte address `addr` is:
//
//   hdr(w1_reg, 2 beats), addr_lo, addr_hi, hdr(queue_reg, 1 beat), 0x80000000 | bd
//
// The last word's leading 1 is the task queue's hardware start flag, not a parity bit.

#include <stdint.h>

// ---- TileControl packet words ------------------------------------------------
static inline int32_t ondv_hdr(uint32_t address, unsigned beats) {
  uint32_t w = (((beats - 1u) & 3u) << 20) | (address & 0xFFFFFu);
  unsigned pc = 0;
  for (uint32_t v = w; v; v &= v - 1u) ++pc;
  if ((pc & 1u) == 0u) w |= 0x80000000u;
  return (int32_t)w;
}

// BD n registers live at 0x1D000 + 0x20*n: w0 len @+0, w1 addr_low @+4, w2 addr_high @+8.
static inline uint32_t ondv_bd_w1(unsigned bd) { return 0x1D000u + 0x20u * bd + 4u; }

// MM2S task queues on a shim tile: ch0 @ 0x1D214, ch1 @ 0x1D21C.
static inline uint32_t ondv_mm2s_queue(unsigned ch) { return ch == 0u ? 0x1D214u : 0x1D21Cu; }

// ---- the descriptors xcommon pins -------------------------------------------
static constexpr unsigned kOndvBdUp = 8;
static constexpr unsigned kOndvBdGate = 9;
static constexpr unsigned kOndvBdDown = 10;

// ---- the 35B recipe's MoE geometry (recipes/qwen36moe.py `Common`) ----------
// STRIPE/PAIR/UP_BYTES/DOWN_PER_CORE*DOWN_BAND/POOL_DOWN and the stripe split.
static constexpr unsigned kOndvCores = 8;
static constexpr unsigned kOndvRouted = 8;          // NE, the routed slots
static constexpr unsigned kOndvSpp = 4;             // STRIPES_PER_PROJ
static constexpr unsigned kOndvCps = 2;             // CORES_PER_STRIPE
static constexpr uint32_t kOndvStripe = 163840u;    // one 128-row up (or gate) stripe
static constexpr uint32_t kOndvPair = 10240u;       // the two chunks of a half at one k-tile
static constexpr uint32_t kOndvUpBytes = 655360u;   // one expert's up (= gate = down)
static constexpr uint32_t kOndvDownCore = 81920u;   // DOWN_PER_CORE * DOWN_BAND
static constexpr uint32_t kOndvPoolDown = 335544320u;

// The MM2S channel each column's w fifo landed on, i.e. its queue register. Taken from
// the built design's `aie.shim_dma_allocation` table for @w0..@w7 (w0 ch1, w1 ch0, w2
// ch1, w3 ch1, w4 ch1, w5 ch0, w6 ch0, w7 ch0). This is a property of the design's
// shim budget, not of the model: re-derive it (tools/check_ondv_channels.py) if the
// design's fifo set changes, or the packets push the wrong column's queue.
static constexpr uint32_t kOndvQueue[kOndvCores] = {
    0x1D21Cu, 0x1D214u, 0x1D21Cu, 0x1D21Cu, 0x1D21Cu, 0x1D214u, 0x1D214u, 0x1D214u};

// One routed slot's control words for one column: three address packets (one per
// descriptor) and ONE push packet -- the three pushes all target the same task-queue
// register, so a single 3-beat write enqueues BD 8, 9 and 10 in order. That is 13 words
// per (slot, column) and FOUR shim descriptors, which is what makes a wave fit a tile's
// 16-BD pool (three columns x 4 + the three pinned routed descriptors = 15).
static inline void ondv_words(int32_t *w, unsigned queue, uint64_t up, uint64_t gate,
                              uint64_t down) {
  const uint32_t addr[3] = {(uint32_t)(up & 0xFFFFFFFCu), (uint32_t)(gate & 0xFFFFFFFCu),
                            (uint32_t)(down & 0xFFFFFFFCu)};
  const uint32_t hi[3] = {(uint32_t)((up >> 32) & 0xFFFFu), (uint32_t)((gate >> 32) & 0xFFFFu),
                          (uint32_t)((down >> 32) & 0xFFFFu)};
  const unsigned bd[3] = {kOndvBdUp, kOndvBdGate, kOndvBdDown};
  for (unsigned i = 0; i < 3; ++i) {
    w[i * 3 + 0] = ondv_hdr(ondv_bd_w1(bd[i]), 2);
    w[i * 3 + 1] = (int32_t)addr[i];
    w[i * 3 + 2] = (int32_t)hi[i];
  }
  w[9] = ondv_hdr(queue, 3);                       // push BD 8, 9, 10 in one packet
  w[10] = (int32_t)(0x80000000u | kOndvBdUp);
  w[11] = (int32_t)(0x80000000u | kOndvBdGate);
  w[12] = (int32_t)(0x80000000u | kOndvBdDown);
}

// The expert's byte offset of routed slot k's up stripe for column c (xcommon.moe_sequence's
// `up`), plus the gate one stripe later and the down slice at POOL_DOWN.
static inline uint32_t ondv_up_off(unsigned expert, unsigned c) {
  return (2u * kOndvSpp * expert + 2u * (c / kOndvCps)) * kOndvStripe + (c % kOndvCps) * kOndvPair;
}
static inline uint32_t ondv_down_off(unsigned expert, unsigned c) {
  return kOndvPoolDown + expert * kOndvUpBytes + c * kOndvDownCore;
}

// Emit the whole stream: for every routed slot k and column c, 13 words --
// [up addr(3)][gate addr(3)][down addr(3)][push(4)] -- at out[(k*kOndvCores + c)*13].
// `idx` is the router's top-8 index output (out + kE, int32[8]);
// (base_hi<<32|base_lo) is the pool BO's DDR address (bo.address() + 0x8000_0000)
// with no offset.
static inline void ondv_ctrl_impl(const int32_t *__restrict idx, uint32_t base_lo, uint32_t base_hi,
                                  int32_t *__restrict out) {
  const uint64_t base = ((uint64_t)base_hi << 32) | (uint64_t)base_lo;
  for (unsigned k = 0; k < kOndvRouted; ++k) {
    const unsigned e = (unsigned)idx[k];
    for (unsigned c = 0; c < kOndvCores; ++c) {
      const uint32_t up = ondv_up_off(e, c);
      ondv_words(out + (k * kOndvCores + c) * 13, kOndvQueue[c], base + up, base + up + kOndvStripe,
                 base + ondv_down_off(e, c));
    }
  }
}

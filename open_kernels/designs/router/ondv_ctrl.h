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
//   hdr(w1_reg, 2 beats), addr_lo, addr_hi, hdr(queue_reg, 1 beat), bd
//
// The last word is the MM2S task-queue INSERTION command: Start_BD_ID in bits [3:0]
// (the register write itself is the start). Bit 31 is Enable_Token_Issue on AIE2 -- the
// managed dma_start_task path sets it only when it wants a completion token, never for a
// plain fill: issuing a token on every routed push back-pressures the shim channel
// (DMA_Task_Token_Stall) once the token queue fills, which is the fused design's TDR.

#include <stdint.h>

// ---- TileControl packet words ------------------------------------------------
// A control packet's WIRE form (mlir-aie lib/Targets/AIETargetNPU.cpp
// AIETranslateControlPacketsToUI32Vec):
//   stream_header(1 word) + ctrl_header(1 word) + data(size words)
// The stream header carries the DESTINATION shim's controller_id
// ((pkt_type & 7) << 12 | (pkt_id & 0xff)); the stream switch parses it out of the DATA
// stream, so the BD is NOT packet-stamped and every control header must be preceded by
// one. (A packet-stamped BD with no stream header is the spike's shape and does not
// deliver here: the descriptor is never retargeted.) The ctrl header itself is ondv_hdr.
static inline int32_t ondv_parity(uint32_t w) {
  unsigned pc = 0;
  for (uint32_t v = w; v; v &= v - 1u) ++pc;
  if ((pc & 1u) == 0u) w |= 0x80000000u;
  return (int32_t)w;
}

static inline int32_t ondv_hdr(uint32_t address, unsigned beats) {
  return ondv_parity((((beats - 1u) & 3u) << 20) | (address & 0xFFFFFu));
}

// The destination shims' controller_id is 15 for every column (the placer's value).
static constexpr unsigned kOndvPktId = 15;
static inline int32_t ondv_stream_hdr() {
  return ondv_parity(/*pkt_type 0*/ (unsigned)(kOndvPktId & 0xffu));
}

// BD n registers live at 0x1D000 + 0x20*n: w0 len @+0, w1 addr_low @+4, w2 addr_high @+8.
static inline uint32_t ondv_bd_w1(unsigned bd) { return 0x1D000u + 0x20u * bd + 4u; }

// MM2S task queues on a shim tile: ch0 @ 0x1D214, ch1 @ 0x1D21C.
static inline uint32_t ondv_mm2s_queue(unsigned ch) { return ch == 0u ? 0x1D214u : 0x1D21Cu; }

// The pinned routed descriptors' physical BD ids. Overridable so a design whose w
// channel already occupies 8/9/10 can move them (the packet retargets whatever id it
// is told, so the header and designs/layer_x/xcommon.py ONDV_BD_* must agree).
#ifndef ONDV_BD_UP
#define ONDV_BD_UP 8
#endif
#ifndef ONDV_BD_GATE
#define ONDV_BD_GATE 9
#endif
#ifndef ONDV_BD_DOWN
#define ONDV_BD_DOWN 10
#endif

// ---- the descriptors xcommon pins -------------------------------------------
static constexpr unsigned kOndvBdUp = ONDV_BD_UP;
static constexpr unsigned kOndvBdGate = ONDV_BD_GATE;
static constexpr unsigned kOndvBdDown = ONDV_BD_DOWN;

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

// One routed slot's SEVEN words: stream hdr, retarget ctrl hdr + 2 address words, stream
// hdr, queue-push ctrl hdr + the bd data word.
static constexpr unsigned kOndvWords = 7;
static inline void ondv_words(int32_t *w, unsigned bd, uint32_t queue, uint64_t addr) {
  w[0] = ondv_stream_hdr();                              // stream header (routing)
  w[1] = ondv_hdr(ondv_bd_w1(bd), 2);                    // ctrl: write w1, 2 beats
  w[2] = (int32_t)(uint32_t)(addr & 0xFFFFFFFCu);        // w1: addr_low, bits 1:0 zero
  w[3] = (int32_t)(uint32_t)((addr >> 32) & 0xFFFFu);    // w2: addr_high[15:0]
  w[4] = ondv_stream_hdr();                              // stream header (routing)
  w[5] = ondv_hdr(queue, 1);                             // ctrl: push queue, 1 beat
  w[6] = (int32_t)bd;                                    // queue push: Start_BD_ID (no token)
}

// The expert's byte offset of routed slot k's up stripe for column c (xcommon.moe_sequence's
// `up`), plus the gate one stripe later and the down slice at POOL_DOWN.
static inline uint32_t ondv_up_off(unsigned expert, unsigned c) {
  return (2u * kOndvSpp * expert + 2u * (c / kOndvCps)) * kOndvStripe + (c % kOndvCps) * kOndvPair;
}
static inline uint32_t ondv_down_off(unsigned expert, unsigned c) {
  return kOndvPoolDown + expert * kOndvUpBytes + c * kOndvDownCore;
}

// Emit the whole stream: for every routed slot k and column c, 15 words --
// up(5), gate(5), down(5) -- at out[(k*kOndvCores + c)*15]. `idx` is the router's
// top-8 index output (out + kE, int32[8]); (base_hi<<32|base_lo) is the pool BO's
// DDR address (bo.address() + 0x8000_0000) with no offset.
// One COLUMN's 120 words (8 slots x 15) -- the per-column entry the unblock needs: each main
// core emits its OWN column's control words to its OWN shim's TileControl (a core's packet
// reaches its own column on the South port), so no cross-column routing, no control overlay
// and no shim MM2S channel are required.
static inline void ondv_ctrl_col_impl(const int32_t *__restrict idx, uint32_t base_lo,
                                      uint32_t base_hi, unsigned col, uint32_t queue,
                                      int32_t *__restrict out) {
  const uint64_t base = ((uint64_t)base_hi << 32) | (uint64_t)base_lo;
  for (unsigned k = 0; k < kOndvRouted; ++k) {
    const unsigned e = (unsigned)idx[k];
    const uint32_t up = ondv_up_off(e, col);
    int32_t *w = out + k * (3u * kOndvWords);
    ondv_words(w + 0 * kOndvWords, kOndvBdUp, queue, base + up);
    ondv_words(w + 1 * kOndvWords, kOndvBdGate, queue, base + up + kOndvStripe);
    ondv_words(w + 2 * kOndvWords, kOndvBdDown, queue, base + ondv_down_off(e, col));
  }
}

static inline void ondv_ctrl_impl(const int32_t *__restrict idx, uint32_t base_lo, uint32_t base_hi,
                                  int32_t *__restrict out) {
  const uint64_t base = ((uint64_t)base_hi << 32) | (uint64_t)base_lo;
  for (unsigned k = 0; k < kOndvRouted; ++k) {
    const unsigned e = (unsigned)idx[k];
    for (unsigned c = 0; c < kOndvCores; ++c) {
      // COLUMN-major: column c's 8 slots (8 x 15 words) are contiguous, so one packet BD
      // per column can carry them (the corrected, core-sourced control shape)
      int32_t *w = out + (c * kOndvRouted + k) * (3u * kOndvWords);
      const uint32_t up = ondv_up_off(e, c);
      ondv_words(w + 0 * kOndvWords, kOndvBdUp, kOndvQueue[c], base + up);
      ondv_words(w + 1 * kOndvWords, kOndvBdGate, kOndvQueue[c], base + up + kOndvStripe);
      ondv_words(w + 2 * kOndvWords, kOndvBdDown, kOndvQueue[c], base + ondv_down_off(e, c));
    }
  }
}

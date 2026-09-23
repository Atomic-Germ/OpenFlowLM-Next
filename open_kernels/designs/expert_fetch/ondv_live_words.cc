// Copyright (c) 2026 bong-water-water-bong
// The core half of the on-device-routing liveness probe: build the TileControl packet
// words that retarget + enqueue a shim descriptor.
//
// ENCODING (from mlir-aie's own control-packet emitter, lib/Targets/AIETargetNPU.cpp):
//   each control packet is  stream_header(1 word) + ctrl_header(1 word) + data(size words)
//   stream_header = (pkt_type<<12 | pkt_id), bit31 = 1 iff popcount(bits[30:0]) is EVEN
//   ctrl_header   = stream_id<<24 | opcode<<22 | (beats-1)<<20 | address, same parity bit
//   and the DMA BD sends this stream RAW (no enable_packet): the stream switch parses the
//   embedded stream headers itself. (The earlier packet-stamped-BD shape was the spike's
//   and is not what the compiler emits; it does not deliver here.)
// BD n's registers live at 0x1D000 + 0x20*n (w0 len @+0, w1 addr_low @+4, w2 addr_high @+8);
// a shim's MM2S task queues are ch0 @ 0x1D214, ch1 @ 0x1D21C. The last data word is the
// queue-INSERTION command (Start_BD_ID in bits [3:0]); bit 31 is Enable_Token_Issue on
// AIE2 and must stay 0 (a token per push back-pressures the channel).
#include <stdint.h>

#define WS 4096u   // the probe's slab descriptor length in 32-bit words (16 KB)
#ifndef LP_MAT
#define LP_MAT 1   // matrices (15-word blocks) per packet BD (fused design: 2-3)
#endif

static inline int32_t lp_parity(uint32_t w) {
  unsigned pc = 0;
  for (uint32_t v = w; v; v &= v - 1u) ++pc;
  if ((pc & 1u) == 0u) w |= 0x80000000u;
  return (int32_t)w;
}
static inline int32_t lp_stream_hdr(uint32_t pkt_id) {
  return lp_parity(pkt_id & 0xffu);                 // pkt_type 0
}
static inline int32_t lp_ctrl_hdr(uint32_t address, unsigned beats) {
  return lp_parity((((beats - 1u) & 3u) << 20) | (address & 0xFFFFFu));
}

// cfg[0..1] = the retarget address (lo, hi) the harness wrote with `poolbase`
// (the destination BO's device address + 0x8000_0000). bd = the descriptor's physical
// index; queue = the shim MM2S queue register that owns it; pkt = the destination shim's
// controller_id (15 on this design).
extern "C" {
void ondv_live_words(const uint32_t *__restrict cfg, int32_t *__restrict w,
                     int32_t bd, int32_t queue) {
  const uint32_t pkt = 15u;
  for (int m = 0; m < LP_MAT; ++m) {
  int32_t *o = w + m * 15;
  o[0] = lp_stream_hdr(pkt);                                 // stream header
  o[1] = lp_ctrl_hdr(0x1D000u + 0x20u * (uint32_t)bd + 0u, 4); // ctrl: write w0..w3, 4 beats
  o[2] = (int32_t)WS;                                        // w0 Buffer_Length
  o[3] = (int32_t)(cfg[0] & 0xFFFFFFFCu);                    // w1 addr_low
  o[4] = (int32_t)(cfg[1] & 0xFFFFu);                        // w2 addr_high[15:0]
  o[5] = 0;                                                  // w3
  o[6] = lp_stream_hdr(pkt);                                 // stream header
  o[7] = lp_ctrl_hdr(0x1D000u + 0x20u * (uint32_t)bd + 16u, 4); // ctrl: write w4..w7, 4 beats
  o[8] = 0;                                                  // w4
  o[9] = 0;                                                  // w5
  o[10] = 0;                                                 // w6
  o[11] = (int32_t)0x02000000u;                              // w7 Valid_BD
  o[12] = lp_stream_hdr(pkt);                                // stream header
  o[13] = lp_ctrl_hdr((uint32_t)queue, 1);                   // ctrl: queue push, 1 beat
  o[14] = (int32_t)((uint32_t)bd & 0xFu);
  }
}
}

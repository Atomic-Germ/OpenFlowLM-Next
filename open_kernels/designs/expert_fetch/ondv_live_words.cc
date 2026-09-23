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
// a shim's MM2S task queues are ch0 @ 0x1D214, ch1 @ 0x1D21C. The last data word's leading
// 1 is the task queue's hardware start flag.
#include <stdint.h>

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
  w[0] = lp_stream_hdr(pkt);                                 // stream header
  w[1] = lp_ctrl_hdr(0x1D000u + 0x20u * (uint32_t)bd + 4u, 2); // ctrl: write w1, 2 beats
  w[2] = (int32_t)(cfg[0] & 0xFFFFFFFCu);
  w[3] = (int32_t)(cfg[1] & 0xFFFFu);                        // addr_high[15:0]
  w[4] = lp_stream_hdr(pkt);                                 // stream header
  w[5] = lp_ctrl_hdr((uint32_t)queue, 1);                    // ctrl: queue push, 1 beat
  w[6] = (int32_t)(0x80000000u | (uint32_t)bd);
}
}

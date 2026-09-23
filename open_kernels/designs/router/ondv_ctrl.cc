#include "ondv_ctrl.h"

// router.h's output layout: [p fp32[256] @0][idx int32[8] @1024 B][w fp32[8] @1056 B].
static constexpr unsigned kOndvIdxOff = 1024;

extern "C" {
// One entry point per .cc (IRON compiles the source once per ExternalFunction).
//
// rout: the router's 4 KB output (idx read at +1024 B)
// cfg : [base_lo, base_hi] -- the pool BO's DDR address (bo.address() + 0x8000_0000),
//       read once per dispatch from a host-filled element (the MoE pool is one BO per
//       layer, so this is a per-ELf constant the driver writes before the runlist)
// out : 8 slots x 8 columns x 15 words (see ondv_ctrl_impl)
void ondv_ctrl(const uint8_t *__restrict rout, const uint32_t *__restrict cfg,
               int32_t *__restrict out) {
  ondv_ctrl_impl((const int32_t *)(rout + kOndvIdxOff), cfg[0], cfg[1], out);
}

// The per-column entry the unblock uses: column `col`'s 120 words only (see ondv_ctrl.h).
// One main core runs this for its own column and sends the result to its own shim's
// TileControl with a packet-stamped BD on its own MM2S.
void ondv_ctrl_col(const uint8_t *__restrict rout, const uint32_t *__restrict cfg,
                   int32_t col, int32_t *__restrict out) {
  ondv_ctrl_col_impl((const int32_t *)(rout + kOndvIdxOff), cfg[0], cfg[1], (unsigned)col, out);
}
}

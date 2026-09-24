#include <stdint.h>

// ONDV_DBG: what the emitter core actually read. One entry point per .cc (see ondv_ctrl_col.cc).
// dbg (64 words): [0..7] rout words 0..7, [8..15] rout+1024 (the top-8 idx), [16..31] cfg
// words 0..15, [32..63] ctrl words 0..31 (after ondv_ctrl_col wrote them).
extern "C" {
void ondv_echo(const uint32_t *__restrict rout, const uint32_t *__restrict cfg,
               const uint32_t *__restrict ctrl, uint32_t *__restrict dbg) {
  for (int i = 0; i < 8; ++i) dbg[i] = rout[i];
  for (int i = 0; i < 8; ++i) dbg[8 + i] = rout[256 + i];
  for (int i = 0; i < 16; ++i) dbg[16 + i] = cfg[i];
  for (int i = 0; i < 32; ++i) dbg[32 + i] = ctrl[i];
}
}

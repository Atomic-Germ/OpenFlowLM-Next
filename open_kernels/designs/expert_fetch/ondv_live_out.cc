// Copyright (c) 2026 bong-water-water-bong
// The probe's observation: report what the retargeted descriptor delivered.
#include <stdint.h>

// The observation: the first and last word of what the descriptor actually delivered.
// All-ones zero => the descriptor never fired; all-ones one => it fired with the slab
// the packet pointed it at rather than the one it was configured with.
extern "C" {
void ondv_live_out(const uint32_t *__restrict slab, uint32_t *__restrict out, int32_t n) {
  out[0] = slab[0];
  out[1] = slab[n - 1];
}
}

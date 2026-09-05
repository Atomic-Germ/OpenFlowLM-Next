// og (bf16[4096]) arrives as two 4 KB act elements: blocks 0..63 from the first.
#include "gemv_q4.h"

extern "C" {
GEMV_Q4_PREP_BLOCKS_ENTRY(4096, 0, 64)
}

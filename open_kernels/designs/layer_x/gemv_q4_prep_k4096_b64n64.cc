// og (bf16[4096]) arrives as two 4 KB act elements: blocks 64..127 from the second.
#include "gemv_q4.h"

extern "C" {
GEMV_Q4_PREP_BLOCKS_ENTRY(4096, 64, 64)
}

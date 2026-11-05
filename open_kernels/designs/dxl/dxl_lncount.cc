// The ln core's add-norm count from the dispatch's mode element (0 verify, 1 draft, 2 head), the first element of
// its stream as of the main cores' x: 2 L for a layer, none for the head. An rtp word written in-context can land
// after the core has read it (dxl_mode.cc), and the layer and head streams share the image.
#include <stdint.h>

extern "C" {
void dxl_lncount(const uint8_t *__restrict e, int32_t *__restrict cnt) {
    cnt[0] = reinterpret_cast<const int32_t *>(e)[0] == 2 ? 0 : 2 * DXL_L;
}
}

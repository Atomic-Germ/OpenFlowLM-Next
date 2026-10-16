// The dispatch's mode (0 verify, 1 draft, 2 head) rides in its first x element: an rtp word written in-context lands after
// the cores have read it (a mode switch only worked across a context reload).
#include <stdint.h>

extern "C" {
void dxl_mode(const int32_t *__restrict x, int32_t *__restrict rtp) { rtp[0] = x[0]; }
}

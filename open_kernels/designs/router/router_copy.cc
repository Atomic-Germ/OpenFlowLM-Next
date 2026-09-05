#include "router.h"
extern "C" {
void router_copy_x(const bfloat16 *__restrict src, bfloat16 *__restrict dst) { router_copy_x_impl(src, dst); }
}

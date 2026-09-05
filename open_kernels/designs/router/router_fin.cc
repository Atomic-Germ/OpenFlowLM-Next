#include "router.h"
extern "C" {
void router_fin(const float *__restrict acc, float *__restrict out) { router_fin_impl(acc, out); }
}

#include "router.h"
extern "C" {
void router_fin(const float *__restrict acc, float *__restrict out) { router_fin_impl(acc, out); }
void router_fin_bias(const float *__restrict acc, const float *__restrict bias, float *__restrict out) { router_fin_impl(acc, bias, out); }
}

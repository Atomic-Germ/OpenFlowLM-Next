#include "router.h"
extern "C" {
void router_acc(const bfloat16 *__restrict W, const bfloat16 *__restrict x, float *__restrict acc, int rb) {
  router_acc_impl(W, x, acc, rb);
}
}

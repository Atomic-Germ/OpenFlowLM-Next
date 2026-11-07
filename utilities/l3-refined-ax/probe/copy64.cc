#include <aie_api/aie.hpp>
extern "C" void copy64(const float *__restrict x, float *__restrict y) {
  aie::store_v(y,aie::load_v<32>(x));
  aie::store_v(y+32,aie::load_v<32>(x+32));
}

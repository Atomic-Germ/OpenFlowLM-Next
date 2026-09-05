//===- rot13.cc ---------------------------------------------*- C++ -*-===//
//
// OpenFFLM PoC -- the readable proof that we can put our own code on the array.
// SPDX-License-Identifier: Apache-2.0
//
// ROT13 over an ASCII tile. Chosen because it is self-inverse: running the
// design twice must return the input byte for byte, so the test needs no
// tolerance, and BOTH the intermediate and the final result are printable.
// A wrong kernel produces visible garbage, not a small numeric error.
//
// Everything here is vector integer work. No scalar float anywhere -- on this
// toolchain a scalar float op lowers to __mulsf3 and costs ~1617x
// (../NpuEmbeddings/research/notes/0001).

#include "aie_kernel_utils.h"   // event0()/event1(), for hardware trace
#include <aie_api/aie.hpp>
#include <stdint.h>

// 64 int8 lanes is one full 512-bit AIE2P vector register.
static constexpr unsigned kVec = 64;

template <unsigned N>
static inline void rot13_impl(const int8_t *__restrict in,
                            int8_t *__restrict out) {
  event0();

  auto it_in = aie::begin_restrict_vector<kVec>((int8_t *)in);
  auto it_out = aie::begin_restrict_vector<kVec>(out);

  // No chess_* pragmas: Peano drops them silently (trap 5b).
  for (unsigned i = 0; i < N; i += kVec) {
    aie::vector<int8_t, kVec> c = *it_in++;

    // Neither branch can overflow int8: +13 is applied only up to 'm' (109)
    // giving 122, and -13 only from 'n' (110) giving 97.
    aie::vector<int8_t, kVec> up = aie::add(c, (int8_t)13);
    aie::vector<int8_t, kVec> dn = aie::sub(c, (int8_t)13);

    auto first = (aie::ge(c, (int8_t)'a') & aie::le(c, (int8_t)'m')) |
                 (aie::ge(c, (int8_t)'A') & aie::le(c, (int8_t)'M'));
    auto second = (aie::ge(c, (int8_t)'n') & aie::le(c, (int8_t)'z')) |
                  (aie::ge(c, (int8_t)'N') & aie::le(c, (int8_t)'Z'));

    // aie::select is out[i] = m[i] == 0 ? v1[i] : v2[i]; everything that is
    // not a letter falls through both selects unchanged.
    aie::vector<int8_t, kVec> r = aie::select(c, up, first);
    *it_out++ = aie::select(r, dn, second);
  }

  event1();
}

extern "C" {

// One entry point per tile size. The IRON design declares the same length as a
// CompileTime parameter; a mismatch here compiles clean and hangs (trap 8).
void rot13_1024(const int8_t *__restrict in, int8_t *__restrict out) {
  rot13_impl<1024>(in, out);
}

}  // extern "C"

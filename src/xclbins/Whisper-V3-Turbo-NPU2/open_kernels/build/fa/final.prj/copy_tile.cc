//===- attn_cascade_wrap.cc -----------------------------------*- C++ -*-===//
// SPDX-License-Identifier: MIT
//
// Thin cascade-I/O layer for the IRON port of AMD's attn_npu2.py FlashAttention
// (NpuEmbeddings task 0180 Part 18 / fa_iron). attn_npu2.cc's compute kernels
// (matmul, softmax, merge arithmetic) are reused UNMODIFIED -- this file
// #includes it verbatim and adds only the cascade-port transfer, which AIR's
// own compiler emits as inline `aie.put_cascade`/`aie.get_cascade` MLIR ops
// around calls to those unmodified kernels (confirmed by reading
// pass_057_after_cse.mlir). IRON's CascadeFlow model requires the linked
// KERNEL to issue the transfer itself (aie.iron.dataflow.cascadeflow only
// declares the topology edge) -- attn_npu2.cc has no put_mcd/get_scd calls
// anywhere, so this file supplies them.
//
// aie2p's put_mcd/get_scd (aie2p_aie_api_compat.h) are declared only for
// int32/cint16/cint32/cacc64-family vector types -- no bf16 or f32 overload.
// Verified (cascade_probe.cc, this session) that a same-width bitcast
// (v32bfloat16 <-> v16int32, v16float <-> v16int32) compiles clean under our
// pinned Peano 21 and emits real `vmov mcd,x0` / `vmov x0,scd` cascade-port
// instructions (llvm-objdump -d), not a numeric conversion -- the cascade
// port moves raw 512-bit register contents regardless of the C++ type on
// either end, which is the same trick aie_kernels/aie2/cascade_mm.cc uses
// for its int32-typed GEMM accumulators.
//===----------------------------------------------------------------------===//

#include "attn_npu2.cc"

extern "C" {

// bf16, 32 lanes/transfer (512 bit). n must be a multiple of 32.
static inline void cascade_get_bf16_n(bfloat16 *__restrict dst, int n) {
  for (int i = 0; i < n; i += 32) {
    v16int32 vi = get_scd_v16int32();
    *(v32bfloat16 *)(dst + i) = (v32bfloat16)vi;
  }
}
static inline void cascade_put_bf16_n(const bfloat16 *__restrict src, int n) {
  for (int i = 0; i < n; i += 32) {
    v32bfloat16 v = *(v32bfloat16 *)(src + i);
    put_mcd((v16int32)v);
  }
}
// f32, 16 lanes/transfer (512 bit). n must be a multiple of 16.
static inline void cascade_get_f32_n(float *__restrict dst, int n) {
  for (int i = 0; i < n; i += 16) {
    v16int32 vi = get_scd_v16int32();
    *(v16float *)(dst + i) = (v16float)vi;
  }
}
static inline void cascade_put_f32_n(const float *__restrict src, int n) {
  for (int i = 0; i < n; i += 16) {
    v16float v = *(v16float *)(src + i);
    put_mcd((v16int32)v);
  }
}

// gp: [lqp, dv] bf16 (lqp*dv elems). up: [lqp,1] bf16. sp: [lqp,1] f32
// (FP32_STATE / variant MF -- the task's pinned configuration).
constexpr int GP_ELEMS = lqp * dv;
constexpr int UP_ELEMS = lqp;
constexpr int SP_ELEMS = lqp;
static_assert(GP_ELEMS % 32 == 0, "gp must tile into 32-lane cascade beats");
static_assert(UP_ELEMS % 32 == 0, "up must tile into 32-lane cascade beats");
static_assert(SP_ELEMS % 16 == 0, "sp(f32) must tile into 16-lane cascade beats");

// One "get" of the neighbour's partials: gp_c, up_c, sp_c, in that order --
// matches attn_npu2.py merge()'s cascade_gp.get / cascade_up.get / cascade_sp.get.
void cascade_get3(bfloat16 *gp_c, bfloat16 *up_c, float *sp_c) {
  cascade_get_bf16_n(gp_c, GP_ELEMS);
  cascade_get_bf16_n(up_c, UP_ELEMS);
  cascade_get_f32_n(sp_c, SP_ELEMS);
}

// One "put" onto the cascade toward the next stage south: gp, up, sp, in that
// order -- matches cascade_gp.put / cascade_up.put / cascade_sp.put.
void cascade_put3(const bfloat16 *gp, const bfloat16 *up, const float *sp) {
  cascade_put_bf16_n(gp, GP_ELEMS);
  cascade_put_bf16_n(up, UP_ELEMS);
  cascade_put_f32_n(sp, SP_ELEMS);
}

} // extern "C"

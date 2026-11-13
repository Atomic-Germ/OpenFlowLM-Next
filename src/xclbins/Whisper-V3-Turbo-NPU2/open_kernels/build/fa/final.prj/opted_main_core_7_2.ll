; ModuleID = '/home/atomic-germ/Projects/OpenFlowLM-Next/src/xclbins/Whisper-V3-Turbo-NPU2/open_kernels/build/fa/final.prj/peano-linked_main_core_7_2.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@st_7_0 = external global [64 x float]
@rl_7_0 = external global [64 x bfloat]
@rc_7_0 = external global [64 x bfloat]
@up_s_7_0 = external global [64 x bfloat]
@sp_c_7_0 = external global [64 x float]
@up_c_7_0 = external global [64 x bfloat]
@gp_c_7_0 = external global [64 x [64 x bfloat]]
@s_tmp_f32_7_0 = external global [64 x float]
@r_tmp_7_0 = external global [64 x bfloat]
@s_tmp_7_0 = external global [64 x bfloat]
@sp_7_0 = external global [64 x float]
@up_7_0 = external global [64 x bfloat]
@gp_7_0 = external global [64 x [64 x bfloat]]
@g_7_0 = external global [64 x [64 x bfloat]]
@q_saved_7_0 = external global [64 x [64 x bfloat]]
@GP_L1L2_7_buff_0 = external global [64 x [64 x bfloat]]
@QK_L2L1_h1_s0_3_cons_buff_0 = external global [64 x [64 x bfloat]]
@V_L2L1_h1_s0_3_cons_buff_0 = external global [64 x [64 x bfloat]]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @copy_tile(ptr, ptr) local_unnamed_addr

declare void @matmul_a_b_bf16(ptr, ptr, ptr) local_unnamed_addr

declare void @matmul_g_b_bf16(ptr, ptr, ptr) local_unnamed_addr

declare void @zero_fill_gp_bf16(ptr) local_unnamed_addr

declare void @zero_fill_g_bf16(ptr) local_unnamed_addr

declare void @neg_inf_fill_up_bf16(ptr) local_unnamed_addr

declare void @maximum_up_u_bf16(ptr, ptr) local_unnamed_addr

declare void @exp_up_minus_u(ptr, ptr, ptr) local_unnamed_addr

declare void @mul_r_gp(ptr, ptr) local_unnamed_addr

declare void @add_gp_g(ptr, ptr) local_unnamed_addr

declare void @vector_copy_32elems(i32, ptr, ptr) local_unnamed_addr

declare void @apply_length_mask(ptr, i32) local_unnamed_addr

declare void @fused_softmax(ptr, ptr, ptr, ptr) local_unnamed_addr

declare void @zero_fill_sp_f32(ptr) local_unnamed_addr

declare void @widen_bf16_to_f32(ptr, ptr) local_unnamed_addr

declare void @vector_copy_32elems_f32(i32, ptr, ptr) local_unnamed_addr

declare void @accum_sp_r_s_f32(ptr, ptr, ptr) local_unnamed_addr

declare void @div_gp_sp_f32(ptr, ptr) local_unnamed_addr

declare void @cascade_get3(ptr, ptr, ptr) local_unnamed_addr

define void @core_7_2() local_unnamed_addr {
  br label %.preheader2

.preheader2:                                      ; preds = %0, %10
  %1 = phi i64 [ 0, %0 ], [ %11, %10 ]
  br label %.preheader

.preheader:                                       ; preds = %.preheader2, %7
  %2 = phi i64 [ 0, %.preheader2 ], [ %8, %7 ]
  br label %3

3:                                                ; preds = %.preheader, %3
  %4 = phi i64 [ 0, %.preheader ], [ %5, %3 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @copy_tile(ptr nonnull @QK_L2L1_h1_s0_3_cons_buff_0, ptr nonnull @q_saved_7_0)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @zero_fill_gp_bf16(ptr nonnull @gp_7_0)
  tail call void @zero_fill_sp_f32(ptr nonnull @sp_7_0)
  tail call void @neg_inf_fill_up_bf16(ptr nonnull @up_7_0)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @zero_fill_g_bf16(ptr nonnull @g_7_0)
  tail call void @matmul_a_b_bf16(ptr nonnull @q_saved_7_0, ptr nonnull @QK_L2L1_h1_s0_3_cons_buff_0, ptr nonnull @g_7_0)
  tail call void @apply_length_mask(ptr nonnull @g_7_0, i32 0)
  tail call void @fused_softmax(ptr nonnull @g_7_0, ptr nonnull @up_7_0, ptr nonnull @s_tmp_7_0, ptr nonnull @r_tmp_7_0)
  tail call void @mul_r_gp(ptr nonnull @r_tmp_7_0, ptr nonnull @gp_7_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @g_7_0, ptr nonnull @V_L2L1_h1_s0_3_cons_buff_0, ptr nonnull @gp_7_0)
  tail call void @widen_bf16_to_f32(ptr nonnull @s_tmp_7_0, ptr nonnull @s_tmp_f32_7_0)
  tail call void @accum_sp_r_s_f32(ptr nonnull @sp_7_0, ptr nonnull @r_tmp_7_0, ptr nonnull @s_tmp_f32_7_0)
  tail call void @vector_copy_32elems_f32(i32 0, ptr nonnull @s_tmp_f32_7_0, ptr nonnull @sp_7_0)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @zero_fill_g_bf16(ptr nonnull @g_7_0)
  tail call void @matmul_a_b_bf16(ptr nonnull @q_saved_7_0, ptr nonnull @QK_L2L1_h1_s0_3_cons_buff_0, ptr nonnull @g_7_0)
  tail call void @apply_length_mask(ptr nonnull @g_7_0, i32 1)
  tail call void @fused_softmax(ptr nonnull @g_7_0, ptr nonnull @up_7_0, ptr nonnull @s_tmp_7_0, ptr nonnull @r_tmp_7_0)
  tail call void @mul_r_gp(ptr nonnull @r_tmp_7_0, ptr nonnull @gp_7_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @g_7_0, ptr nonnull @V_L2L1_h1_s0_3_cons_buff_0, ptr nonnull @gp_7_0)
  tail call void @widen_bf16_to_f32(ptr nonnull @s_tmp_7_0, ptr nonnull @s_tmp_f32_7_0)
  tail call void @accum_sp_r_s_f32(ptr nonnull @sp_7_0, ptr nonnull @r_tmp_7_0, ptr nonnull @s_tmp_f32_7_0)
  tail call void @vector_copy_32elems_f32(i32 0, ptr nonnull @s_tmp_f32_7_0, ptr nonnull @sp_7_0)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @zero_fill_g_bf16(ptr nonnull @g_7_0)
  tail call void @matmul_a_b_bf16(ptr nonnull @q_saved_7_0, ptr nonnull @QK_L2L1_h1_s0_3_cons_buff_0, ptr nonnull @g_7_0)
  tail call void @apply_length_mask(ptr nonnull @g_7_0, i32 2)
  tail call void @fused_softmax(ptr nonnull @g_7_0, ptr nonnull @up_7_0, ptr nonnull @s_tmp_7_0, ptr nonnull @r_tmp_7_0)
  tail call void @mul_r_gp(ptr nonnull @r_tmp_7_0, ptr nonnull @gp_7_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @g_7_0, ptr nonnull @V_L2L1_h1_s0_3_cons_buff_0, ptr nonnull @gp_7_0)
  tail call void @widen_bf16_to_f32(ptr nonnull @s_tmp_7_0, ptr nonnull @s_tmp_f32_7_0)
  tail call void @accum_sp_r_s_f32(ptr nonnull @sp_7_0, ptr nonnull @r_tmp_7_0, ptr nonnull @s_tmp_f32_7_0)
  tail call void @vector_copy_32elems_f32(i32 0, ptr nonnull @s_tmp_f32_7_0, ptr nonnull @sp_7_0)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @zero_fill_g_bf16(ptr nonnull @g_7_0)
  tail call void @matmul_a_b_bf16(ptr nonnull @q_saved_7_0, ptr nonnull @QK_L2L1_h1_s0_3_cons_buff_0, ptr nonnull @g_7_0)
  tail call void @apply_length_mask(ptr nonnull @g_7_0, i32 3)
  tail call void @fused_softmax(ptr nonnull @g_7_0, ptr nonnull @up_7_0, ptr nonnull @s_tmp_7_0, ptr nonnull @r_tmp_7_0)
  tail call void @mul_r_gp(ptr nonnull @r_tmp_7_0, ptr nonnull @gp_7_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @g_7_0, ptr nonnull @V_L2L1_h1_s0_3_cons_buff_0, ptr nonnull @gp_7_0)
  tail call void @widen_bf16_to_f32(ptr nonnull @s_tmp_7_0, ptr nonnull @s_tmp_f32_7_0)
  tail call void @accum_sp_r_s_f32(ptr nonnull @sp_7_0, ptr nonnull @r_tmp_7_0, ptr nonnull @s_tmp_f32_7_0)
  tail call void @vector_copy_32elems_f32(i32 0, ptr nonnull @s_tmp_f32_7_0, ptr nonnull @sp_7_0)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @zero_fill_g_bf16(ptr nonnull @g_7_0)
  tail call void @matmul_a_b_bf16(ptr nonnull @q_saved_7_0, ptr nonnull @QK_L2L1_h1_s0_3_cons_buff_0, ptr nonnull @g_7_0)
  tail call void @apply_length_mask(ptr nonnull @g_7_0, i32 4)
  tail call void @fused_softmax(ptr nonnull @g_7_0, ptr nonnull @up_7_0, ptr nonnull @s_tmp_7_0, ptr nonnull @r_tmp_7_0)
  tail call void @mul_r_gp(ptr nonnull @r_tmp_7_0, ptr nonnull @gp_7_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @g_7_0, ptr nonnull @V_L2L1_h1_s0_3_cons_buff_0, ptr nonnull @gp_7_0)
  tail call void @widen_bf16_to_f32(ptr nonnull @s_tmp_7_0, ptr nonnull @s_tmp_f32_7_0)
  tail call void @accum_sp_r_s_f32(ptr nonnull @sp_7_0, ptr nonnull @r_tmp_7_0, ptr nonnull @s_tmp_f32_7_0)
  tail call void @vector_copy_32elems_f32(i32 0, ptr nonnull @s_tmp_f32_7_0, ptr nonnull @sp_7_0)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @zero_fill_g_bf16(ptr nonnull @g_7_0)
  tail call void @matmul_a_b_bf16(ptr nonnull @q_saved_7_0, ptr nonnull @QK_L2L1_h1_s0_3_cons_buff_0, ptr nonnull @g_7_0)
  tail call void @apply_length_mask(ptr nonnull @g_7_0, i32 5)
  tail call void @fused_softmax(ptr nonnull @g_7_0, ptr nonnull @up_7_0, ptr nonnull @s_tmp_7_0, ptr nonnull @r_tmp_7_0)
  tail call void @mul_r_gp(ptr nonnull @r_tmp_7_0, ptr nonnull @gp_7_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @g_7_0, ptr nonnull @V_L2L1_h1_s0_3_cons_buff_0, ptr nonnull @gp_7_0)
  tail call void @widen_bf16_to_f32(ptr nonnull @s_tmp_7_0, ptr nonnull @s_tmp_f32_7_0)
  tail call void @accum_sp_r_s_f32(ptr nonnull @sp_7_0, ptr nonnull @r_tmp_7_0, ptr nonnull @s_tmp_f32_7_0)
  tail call void @vector_copy_32elems_f32(i32 0, ptr nonnull @s_tmp_f32_7_0, ptr nonnull @sp_7_0)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @cascade_get3(ptr nonnull @gp_c_7_0, ptr nonnull @up_c_7_0, ptr nonnull @sp_c_7_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @up_7_0, ptr nonnull @up_s_7_0)
  tail call void @maximum_up_u_bf16(ptr nonnull @up_c_7_0, ptr nonnull @up_7_0)
  tail call void @exp_up_minus_u(ptr nonnull @up_c_7_0, ptr nonnull @up_7_0, ptr nonnull @rc_7_0)
  tail call void @exp_up_minus_u(ptr nonnull @up_s_7_0, ptr nonnull @up_7_0, ptr nonnull @rl_7_0)
  tail call void @mul_r_gp(ptr nonnull @rc_7_0, ptr nonnull @gp_c_7_0)
  tail call void @mul_r_gp(ptr nonnull @rl_7_0, ptr nonnull @gp_7_0)
  tail call void @add_gp_g(ptr nonnull @gp_7_0, ptr nonnull @gp_c_7_0)
  tail call void @zero_fill_sp_f32(ptr nonnull @st_7_0)
  tail call void @accum_sp_r_s_f32(ptr nonnull @sp_c_7_0, ptr nonnull @rc_7_0, ptr nonnull @st_7_0)
  tail call void @accum_sp_r_s_f32(ptr nonnull @sp_7_0, ptr nonnull @rl_7_0, ptr nonnull @st_7_0)
  tail call void @vector_copy_32elems_f32(i32 0, ptr nonnull @st_7_0, ptr nonnull @sp_c_7_0)
  tail call void @div_gp_sp_f32(ptr nonnull @sp_c_7_0, ptr nonnull @gp_c_7_0)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @copy_tile(ptr nonnull @gp_c_7_0, ptr nonnull @GP_L1L2_7_buff_0)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  %5 = add nuw nsw i64 %4, 1
  %6 = icmp samesign ult i64 %4, 5
  br i1 %6, label %3, label %7

7:                                                ; preds = %3
  %8 = add nuw nsw i64 %2, 1
  %9 = icmp samesign ult i64 %2, 9
  br i1 %9, label %.preheader, label %10

10:                                               ; preds = %7
  %11 = add nuw nsw i64 %1, 1
  %.not = icmp eq i64 %11, 9223372036854775807
  br i1 %.not, label %12, label %.preheader2

12:                                               ; preds = %10
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

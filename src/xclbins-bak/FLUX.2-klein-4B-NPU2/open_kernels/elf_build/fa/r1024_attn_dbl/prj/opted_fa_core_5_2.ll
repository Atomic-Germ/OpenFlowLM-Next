; ModuleID = '/home/atomic-germ/Projects/Next/src/xclbins/FLUX.2-klein-4B-NPU2/open_kernels/elf_build/fa/r1024_attn_dbl/prj/peano-linked_fa_core_5_2.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@rtp_1_4 = global [8 x i32] zeroinitializer
@sp_1_4 = external global [256 x float]
@up_1_4 = external global [32 x bfloat]
@g_1_4 = external global [2048 x bfloat]
@q_saved_1_4 = external global [4096 x bfloat]
@A_L2L1_1_4_cons_buff_1 = external global [64 x [64 x bfloat]]
@A_L2L1_1_4_cons_buff_0 = external global [64 x [64 x bfloat]]
@B_L2L1_1_4_cons_buff_1 = external global [64 x [64 x bfloat]]
@B_L2L1_1_4_cons_buff_0 = external global [64 x [64 x bfloat]]
@O_L1L2_5_0_buff_0 = external global [4096 x bfloat]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @capture_q(ptr, ptr, i32, i32) local_unnamed_addr

declare void @pass_init(ptr, ptr, ptr) local_unnamed_addr

declare void @zero_g(ptr) local_unnamed_addr

declare void @qk(ptr, ptr, ptr, i32) local_unnamed_addr

declare void @pv(ptr, ptr, ptr, i32) local_unnamed_addr

declare void @apply_mask(ptr, ptr, i32, i32) local_unnamed_addr

declare void @softmax_step(ptr, ptr, ptr, ptr) local_unnamed_addr

declare void @finalize(ptr, ptr) local_unnamed_addr

define void @core_5_2() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %._crit_edge5
  %2 = phi i64 [ 0, %0 ], [ %23, %._crit_edge5 ]
  tail call void @llvm.aie2p.acquire(i32 54, i32 1)
  %3 = load i32, ptr @rtp_1_4, align 16
  %4 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_1_4, i20 4), align 4
  %5 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_1_4, i20 8), align 8
  %6 = sext i32 %3 to i64
  %7 = sext i32 %4 to i64
  %8 = sext i32 %5 to i64
  %9 = icmp sgt i32 %3, 0
  br i1 %9, label %.preheader.lr.ph, label %._crit_edge5

.preheader.lr.ph:                                 ; preds = %1
  %10 = icmp sgt i32 %4, 0
  %11 = icmp sgt i32 %5, 0
  br label %.preheader

.preheader:                                       ; preds = %.preheader.lr.ph, %._crit_edge4
  %12 = phi i64 [ 0, %.preheader.lr.ph ], [ %21, %._crit_edge4 ]
  br i1 %10, label %.lr.ph3, label %._crit_edge4

.lr.ph3:                                          ; preds = %.preheader, %._crit_edge
  %13 = phi i64 [ %19, %._crit_edge ], [ 0, %.preheader ]
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @capture_q(ptr nonnull @A_L2L1_1_4_cons_buff_0, ptr nonnull @q_saved_1_4, i32 0, i32 0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @capture_q(ptr nonnull @A_L2L1_1_4_cons_buff_1, ptr nonnull @q_saved_1_4, i32 1, i32 0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @pass_init(ptr nonnull @O_L1L2_5_0_buff_0, ptr nonnull @up_1_4, ptr nonnull @sp_1_4)
  %14 = trunc i64 %13 to i32
  br i1 %11, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %.lr.ph3, %.lr.ph
  %15 = phi i64 [ %17, %.lr.ph ], [ 0, %.lr.ph3 ]
  tail call void @zero_g(ptr nonnull @g_1_4)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @qk(ptr nonnull @q_saved_1_4, ptr nonnull @A_L2L1_1_4_cons_buff_0, ptr nonnull @g_1_4, i32 0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @qk(ptr nonnull @q_saved_1_4, ptr nonnull @A_L2L1_1_4_cons_buff_1, ptr nonnull @g_1_4, i32 1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %16 = trunc i64 %15 to i32
  tail call void @apply_mask(ptr nonnull @g_1_4, ptr nonnull @rtp_1_4, i32 %14, i32 %16)
  tail call void @softmax_step(ptr nonnull @g_1_4, ptr nonnull @up_1_4, ptr nonnull @sp_1_4, ptr nonnull @O_L1L2_5_0_buff_0)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @pv(ptr nonnull @g_1_4, ptr nonnull @B_L2L1_1_4_cons_buff_0, ptr nonnull @O_L1L2_5_0_buff_0, i32 0)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  tail call void @pv(ptr nonnull @g_1_4, ptr nonnull @B_L2L1_1_4_cons_buff_1, ptr nonnull @O_L1L2_5_0_buff_0, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %17 = add nuw nsw i64 %15, 1
  %18 = icmp slt i64 %17, %8
  br i1 %18, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph, %.lr.ph3
  tail call void @finalize(ptr nonnull @sp_1_4, ptr nonnull @O_L1L2_5_0_buff_0)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  %19 = add nuw nsw i64 %13, 1
  %20 = icmp slt i64 %19, %7
  br i1 %20, label %.lr.ph3, label %._crit_edge4

._crit_edge4:                                     ; preds = %._crit_edge, %.preheader
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp slt i64 %21, %6
  br i1 %22, label %.preheader, label %._crit_edge5

._crit_edge5:                                     ; preds = %._crit_edge4, %1
  tail call void @llvm.aie2p.release(i32 54, i32 1)
  %23 = add nuw nsw i64 %2, 1
  %.not = icmp eq i64 %23, 9223372036854775807
  br i1 %.not, label %24, label %1

24:                                               ; preds = %._crit_edge5
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

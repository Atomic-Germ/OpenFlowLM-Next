; ModuleID = '/home/atomic-germ/Projects/OpenFlowLM-Next/src/xclbins/FLUX.2-klein-4B-NPU2/open_kernels/build/r1024_x_emb/final.prj/peano-linked_main_core_7_2.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@rtp_3_4 = global [4 x i32] zeroinitializer
@A_L2L1_3_4_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_3_4_cons_buff_0 = external global [32 x [64 x bfloat]]
@B_L2L1_4_3_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_4_3_cons_buff_0 = external global [64 x [16 x i72]]
@C_L1L2_4_3_buff_0 = external global [128 x [128 x bfloat]]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @dit_zero_c(ptr) local_unnamed_addr

declare void @dit_matmul_quarter(ptr, ptr, ptr) local_unnamed_addr

declare void @dit_swiglu_epi(ptr, ptr, i32) local_unnamed_addr

define void @core_7_2() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %._crit_edge6
  %2 = phi i32 [ 0, %0 ], [ %.lcssa3, %._crit_edge6 ]
  %3 = phi i64 [ 0, %0 ], [ %23, %._crit_edge6 ]
  tail call void @llvm.aie2p.acquire(i32 54, i32 1)
  %4 = load i32, ptr @rtp_3_4, align 4
  %5 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_3_4, i20 4), align 4
  %6 = sext i32 %4 to i64
  %7 = sext i32 %5 to i64
  %8 = icmp sgt i32 %4, 0
  br i1 %8, label %.lr.ph5, label %._crit_edge6

.lr.ph5:                                          ; preds = %1
  %9 = icmp sgt i32 %5, 0
  br label %10

10:                                               ; preds = %.lr.ph5, %._crit_edge
  %11 = phi i32 [ %2, %.lr.ph5 ], [ %.lcssa, %._crit_edge ]
  %12 = phi i64 [ 0, %.lr.ph5 ], [ %21, %._crit_edge ]
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @dit_zero_c(ptr nonnull @C_L1L2_4_3_buff_0)
  br i1 %9, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %10, %.lr.ph
  %13 = phi i32 [ %17, %.lr.ph ], [ %11, %10 ]
  %14 = phi i64 [ %18, %.lr.ph ], [ 0, %10 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond = icmp eq i32 %13, 1
  %spec.select = select i1 %cond, ptr @B_L2L1_4_3_cons_buff_1, ptr @B_L2L1_4_3_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @dit_matmul_quarter(ptr nonnull @A_L2L1_3_4_cons_buff_0, ptr nonnull %spec.select, ptr nonnull @C_L1L2_4_3_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @dit_matmul_quarter(ptr nonnull @A_L2L1_3_4_cons_buff_1, ptr nonnull %spec.select, ptr nonnull @C_L1L2_4_3_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @dit_matmul_quarter(ptr nonnull @A_L2L1_3_4_cons_buff_0, ptr nonnull %spec.select, ptr nonnull @C_L1L2_4_3_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @dit_matmul_quarter(ptr nonnull @A_L2L1_3_4_cons_buff_1, ptr nonnull %spec.select, ptr nonnull @C_L1L2_4_3_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %15 = add i32 %13, 1
  %16 = icmp sgt i32 %15, 1
  %17 = select i1 %16, i32 0, i32 %15
  %18 = add nuw nsw i64 %14, 1
  %19 = icmp slt i64 %18, %7
  br i1 %19, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph, %10
  %.lcssa = phi i32 [ %11, %10 ], [ %17, %.lr.ph ]
  %20 = trunc i64 %12 to i32
  tail call void @dit_swiglu_epi(ptr nonnull @C_L1L2_4_3_buff_0, ptr nonnull @rtp_3_4, i32 %20)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  %21 = add nuw nsw i64 %12, 1
  %22 = icmp slt i64 %21, %6
  br i1 %22, label %10, label %._crit_edge6

._crit_edge6:                                     ; preds = %._crit_edge, %1
  %.lcssa3 = phi i32 [ %2, %1 ], [ %.lcssa, %._crit_edge ]
  tail call void @llvm.aie2p.release(i32 54, i32 1)
  %23 = add nuw nsw i64 %3, 1
  %.not = icmp eq i64 %23, 9223372036854775807
  br i1 %.not, label %24, label %1

24:                                               ; preds = %._crit_edge6
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

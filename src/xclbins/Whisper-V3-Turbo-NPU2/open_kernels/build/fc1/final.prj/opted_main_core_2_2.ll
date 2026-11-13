; ModuleID = '/home/atomic-germ/Projects/OpenFlowLM-Next/src/xclbins/Whisper-V3-Turbo-NPU2/open_kernels/build/fc1/final.prj/peano-linked_main_core_2_2.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@rtp_1_0 = local_unnamed_addr global [2 x i32] zeroinitializer
@A_L2L1_1_0_cons_buff_1 = external global [64 x [64 x bfloat]]
@A_L2L1_1_0_cons_buff_0 = external global [64 x [64 x bfloat]]
@B_L2L1_0_1_cons_buff_1 = external global [64 x [32 x bfloat]]
@B_L2L1_0_1_cons_buff_0 = external global [64 x [32 x bfloat]]
@C_L1L2_0_1_buff_1 = external global [64 x [32 x float]]
@C_L1L2_0_1_buff_0 = external global [64 x [32 x float]]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @zero_f32(ptr) local_unnamed_addr

declare void @be894551_matmul_bf16_f32(ptr, ptr, ptr) local_unnamed_addr

define void @core_2_2() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %._crit_edge13
  %2 = phi i32 [ 0, %0 ], [ %.lcssa7, %._crit_edge13 ]
  %3 = phi i32 [ 0, %0 ], [ %.lcssa8, %._crit_edge13 ]
  %4 = phi i32 [ 0, %0 ], [ %.lcssa9, %._crit_edge13 ]
  %5 = phi i64 [ 0, %0 ], [ %34, %._crit_edge13 ]
  tail call void @llvm.aie2p.acquire(i32 54, i32 1)
  %6 = load i32, ptr @rtp_1_0, align 4
  %7 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_1_0, i20 4), align 4
  %8 = sext i32 %6 to i64
  %9 = sext i32 %7 to i64
  %10 = icmp sgt i32 %6, 0
  br i1 %10, label %.lr.ph12, label %._crit_edge13

.lr.ph12:                                         ; preds = %1
  %11 = icmp sgt i32 %7, 0
  br label %12

12:                                               ; preds = %.lr.ph12, %._crit_edge
  %13 = phi i32 [ %2, %.lr.ph12 ], [ %.lcssa, %._crit_edge ]
  %14 = phi i32 [ %3, %.lr.ph12 ], [ %.lcssa6, %._crit_edge ]
  %15 = phi i32 [ %4, %.lr.ph12 ], [ %31, %._crit_edge ]
  %16 = phi i64 [ 0, %.lr.ph12 ], [ %32, %._crit_edge ]
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  %cond = icmp eq i32 %15, 1
  %spec.select = select i1 %cond, ptr @C_L1L2_0_1_buff_1, ptr @C_L1L2_0_1_buff_0
  tail call void @zero_f32(ptr nonnull %spec.select)
  br i1 %11, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %12, %.lr.ph
  %17 = phi i32 [ %26, %.lr.ph ], [ %13, %12 ]
  %18 = phi i32 [ %23, %.lr.ph ], [ %14, %12 ]
  %19 = phi i64 [ %27, %.lr.ph ], [ 0, %12 ]
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  %cond1 = icmp eq i32 %18, 1
  %spec.select5 = select i1 %cond1, ptr @A_L2L1_1_0_cons_buff_1, ptr @A_L2L1_1_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond2 = icmp eq i32 %17, 1
  %20 = select i1 %cond2, ptr @B_L2L1_0_1_cons_buff_1, ptr @B_L2L1_0_1_cons_buff_0
  tail call void @be894551_matmul_bf16_f32(ptr nonnull %spec.select5, ptr nonnull %20, ptr nonnull %spec.select)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  %21 = add i32 %18, 1
  %22 = icmp sgt i32 %21, 1
  %23 = select i1 %22, i32 0, i32 %21
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %24 = add i32 %17, 1
  %25 = icmp sgt i32 %24, 1
  %26 = select i1 %25, i32 0, i32 %24
  %27 = add nuw nsw i64 %19, 1
  %28 = icmp slt i64 %27, %9
  br i1 %28, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph, %12
  %.lcssa6 = phi i32 [ %14, %12 ], [ %23, %.lr.ph ]
  %.lcssa = phi i32 [ %13, %12 ], [ %26, %.lr.ph ]
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  %29 = add i32 %15, 1
  %30 = icmp sgt i32 %29, 1
  %31 = select i1 %30, i32 0, i32 %29
  %32 = add nuw nsw i64 %16, 1
  %33 = icmp slt i64 %32, %8
  br i1 %33, label %12, label %._crit_edge13

._crit_edge13:                                    ; preds = %._crit_edge, %1
  %.lcssa9 = phi i32 [ %4, %1 ], [ %31, %._crit_edge ]
  %.lcssa8 = phi i32 [ %3, %1 ], [ %.lcssa6, %._crit_edge ]
  %.lcssa7 = phi i32 [ %2, %1 ], [ %.lcssa, %._crit_edge ]
  tail call void @llvm.aie2p.release(i32 54, i32 1)
  %34 = add nuw nsw i64 %5, 1
  %.not = icmp eq i64 %34, 9223372036854775807
  br i1 %.not, label %35, label %1

35:                                               ; preds = %._crit_edge13
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

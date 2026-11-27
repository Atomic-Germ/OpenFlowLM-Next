; ModuleID = '/home/atomic-germ/Projects/Next/src/xclbins/FLUX.2-klein-4B-NPU2/open_kernels/elf_build/conv/vae1024_conv00/prj/peano-linked_conv_core_6_4.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@rtp_6_2 = local_unnamed_addr global [4 x i32] zeroinitializer
@a_6_2_1 = external global [32 x [64 x bfloat]]
@a_6_2_0 = external global [32 x [64 x bfloat]]
@B_L2L1_6_2_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_6_2_cons_buff_0 = external global [64 x [16 x i72]]
@C_L1L2_6_2_buff_0 = external global [128 x [128 x bfloat]]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @dit_conv_bias(ptr, ptr) local_unnamed_addr

declare void @dit_matmul_quarter(ptr, ptr, ptr) local_unnamed_addr

define void @core_6_4() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %._crit_edge8
  %2 = phi i32 [ 0, %0 ], [ %.lcssa5, %._crit_edge8 ]
  %3 = phi i64 [ 0, %0 ], [ %25, %._crit_edge8 ]
  tail call void @llvm.aie2p.acquire(i32 54, i32 1)
  %4 = load i32, ptr @rtp_6_2, align 4
  %5 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_6_2, i20 4), align 4
  %6 = sext i32 %4 to i64
  %7 = sext i32 %5 to i64
  %8 = icmp sgt i32 %4, 0
  br i1 %8, label %.lr.ph7, label %._crit_edge8

.lr.ph7:                                          ; preds = %1
  %9 = icmp sgt i32 %5, 0
  br label %10

10:                                               ; preds = %.lr.ph7, %._crit_edge
  %11 = phi i32 [ %2, %.lr.ph7 ], [ %.lcssa, %._crit_edge ]
  %12 = phi i64 [ 0, %.lr.ph7 ], [ %23, %._crit_edge ]
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond = icmp eq i32 %11, 1
  %spec.select = select i1 %cond, ptr @B_L2L1_6_2_cons_buff_1, ptr @B_L2L1_6_2_cons_buff_0
  tail call void @dit_conv_bias(ptr nonnull %spec.select, ptr nonnull @C_L1L2_6_2_buff_0)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %13 = add i32 %11, 1
  %14 = icmp sgt i32 %13, 1
  %15 = select i1 %14, i32 0, i32 %13
  br i1 %9, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %10, %.lr.ph
  %16 = phi i32 [ %20, %.lr.ph ], [ %15, %10 ]
  %17 = phi i64 [ %21, %.lr.ph ], [ 0, %10 ]
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %cond1 = icmp eq i32 %16, 1
  %spec.select4 = select i1 %cond1, ptr @B_L2L1_6_2_cons_buff_1, ptr @B_L2L1_6_2_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @dit_matmul_quarter(ptr nonnull @a_6_2_0, ptr nonnull %spec.select4, ptr nonnull @C_L1L2_6_2_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @dit_matmul_quarter(ptr nonnull @a_6_2_1, ptr nonnull %spec.select4, ptr nonnull @C_L1L2_6_2_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @dit_matmul_quarter(ptr nonnull @a_6_2_0, ptr nonnull %spec.select4, ptr nonnull @C_L1L2_6_2_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @dit_matmul_quarter(ptr nonnull @a_6_2_1, ptr nonnull %spec.select4, ptr nonnull @C_L1L2_6_2_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %18 = add i32 %16, 1
  %19 = icmp sgt i32 %18, 1
  %20 = select i1 %19, i32 0, i32 %18
  %21 = add nuw nsw i64 %17, 1
  %22 = icmp slt i64 %21, %7
  br i1 %22, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph, %10
  %.lcssa = phi i32 [ %15, %10 ], [ %20, %.lr.ph ]
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  %23 = add nuw nsw i64 %12, 1
  %24 = icmp slt i64 %23, %6
  br i1 %24, label %10, label %._crit_edge8

._crit_edge8:                                     ; preds = %._crit_edge, %1
  %.lcssa5 = phi i32 [ %2, %1 ], [ %.lcssa, %._crit_edge ]
  tail call void @llvm.aie2p.release(i32 54, i32 1)
  %25 = add nuw nsw i64 %3, 1
  %.not = icmp eq i64 %25, 9223372036854775807
  br i1 %.not, label %26, label %1

26:                                               ; preds = %._crit_edge8
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

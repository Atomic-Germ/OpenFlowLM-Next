; ModuleID = '/home/atomic-germ/Projects/Next/src/xclbins/FLUX.2-klein-4B-NPU2/open_kernels/elf_build/vew/vae1024_vew11/prj/peano-linked_vew_core_0_2.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@rtp_0_0 = global [12 x i32] zeroinitializer
@par_0_0 = external global [2624 x float]
@A_L2L1_0_0_cons_buff_1 = external global [4096 x bfloat]
@A_L2L1_0_0_cons_buff_0 = external global [4096 x bfloat]
@B_L2L1_0_0_cons_buff_0 = external global [4096 x bfloat]
@Y_L1L2_0_0_buff_1 = external global [4096 x bfloat]
@Y_L1L2_0_0_buff_0 = external global [4096 x bfloat]
@Z_L1L2_0_0_buff_0 = external global [4096 x bfloat]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @vew_begin(ptr, ptr) local_unnamed_addr

declare void @vew_param(ptr, ptr, i32, i32, ptr) local_unnamed_addr

declare void @vew_stats(ptr, ptr, ptr) local_unnamed_addr

declare void @vew_stats_out(ptr, ptr, ptr) local_unnamed_addr

declare void @vew_apply(ptr, ptr, ptr, ptr) local_unnamed_addr

declare void @vew_add(ptr, ptr, ptr, ptr, ptr) local_unnamed_addr

declare void @vew_rgba(ptr, ptr, ptr) local_unnamed_addr

define void @core_0_2() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %._crit_edge52
  %2 = phi i32 [ 0, %0 ], [ %.lcssa27, %._crit_edge52 ]
  %3 = phi i32 [ 0, %0 ], [ %.lcssa28, %._crit_edge52 ]
  %4 = phi i64 [ 0, %0 ], [ %87, %._crit_edge52 ]
  tail call void @llvm.aie2p.acquire(i32 56, i32 1)
  tail call void @vew_begin(ptr nonnull @par_0_0, ptr nonnull @rtp_0_0)
  %5 = load i32, ptr @rtp_0_0, align 16
  %6 = sext i32 %5 to i64
  %7 = icmp sgt i32 %5, 0
  br i1 %7, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %1, %.lr.ph
  %8 = phi i32 [ %13, %.lr.ph ], [ %3, %1 ]
  %9 = phi i64 [ %14, %.lr.ph ], [ 0, %1 ]
  tail call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %cond = icmp eq i32 %8, 1
  %spec.select = select i1 %cond, ptr @A_L2L1_0_0_cons_buff_1, ptr @A_L2L1_0_0_cons_buff_0
  %10 = trunc i64 %9 to i32
  tail call void @vew_param(ptr nonnull %spec.select, ptr nonnull @par_0_0, i32 %10, i32 -1, ptr nonnull @rtp_0_0)
  tail call void @llvm.aie2p.release(i32 54, i32 1)
  %11 = add i32 %8, 1
  %12 = icmp sgt i32 %11, 1
  %13 = select i1 %12, i32 0, i32 %11
  %14 = add nuw nsw i64 %9, 1
  %15 = icmp slt i64 %14, %6
  br i1 %15, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph, %1
  %.lcssa = phi i32 [ %3, %1 ], [ %13, %.lr.ph ]
  %16 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_0_0, i20 4), align 4
  %17 = sext i32 %16 to i64
  %18 = icmp sgt i32 %16, 0
  br i1 %18, label %.lr.ph30, label %._crit_edge31

.lr.ph30:                                         ; preds = %._crit_edge, %.lr.ph30
  %19 = phi i32 [ %23, %.lr.ph30 ], [ %.lcssa, %._crit_edge ]
  %20 = phi i64 [ %24, %.lr.ph30 ], [ 0, %._crit_edge ]
  tail call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %cond1 = icmp eq i32 %19, 1
  %spec.select16 = select i1 %cond1, ptr @A_L2L1_0_0_cons_buff_1, ptr @A_L2L1_0_0_cons_buff_0
  tail call void @vew_stats(ptr nonnull %spec.select16, ptr nonnull @par_0_0, ptr nonnull @rtp_0_0)
  tail call void @llvm.aie2p.release(i32 54, i32 1)
  %21 = add i32 %19, 1
  %22 = icmp sgt i32 %21, 1
  %23 = select i1 %22, i32 0, i32 %21
  %24 = add nuw nsw i64 %20, 1
  %25 = icmp slt i64 %24, %17
  br i1 %25, label %.lr.ph30, label %._crit_edge31

._crit_edge31:                                    ; preds = %.lr.ph30, %._crit_edge
  %.lcssa21 = phi i32 [ %.lcssa, %._crit_edge ], [ %23, %.lr.ph30 ]
  %26 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_0_0, i20 8), align 8
  %27 = sext i32 %26 to i64
  %28 = icmp sgt i32 %26, 0
  br i1 %28, label %.lr.ph34, label %._crit_edge35

.lr.ph34:                                         ; preds = %._crit_edge31, %.lr.ph34
  %29 = phi i32 [ %33, %.lr.ph34 ], [ %2, %._crit_edge31 ]
  %30 = phi i64 [ %34, %.lr.ph34 ], [ 0, %._crit_edge31 ]
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %cond2 = icmp eq i32 %29, 1
  %spec.select17 = select i1 %cond2, ptr @Y_L1L2_0_0_buff_1, ptr @Y_L1L2_0_0_buff_0
  tail call void @vew_stats_out(ptr nonnull %spec.select17, ptr nonnull @par_0_0, ptr nonnull @rtp_0_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %31 = add i32 %29, 1
  %32 = icmp sgt i32 %31, 1
  %33 = select i1 %32, i32 0, i32 %31
  %34 = add nuw nsw i64 %30, 1
  %35 = icmp slt i64 %34, %27
  br i1 %35, label %.lr.ph34, label %._crit_edge35

._crit_edge35:                                    ; preds = %.lr.ph34, %._crit_edge31
  %.lcssa22 = phi i32 [ %2, %._crit_edge31 ], [ %33, %.lr.ph34 ]
  %36 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_0_0, i20 12), align 4
  %37 = sext i32 %36 to i64
  %38 = icmp sgt i32 %36, 0
  br i1 %38, label %.lr.ph38, label %._crit_edge39

.lr.ph38:                                         ; preds = %._crit_edge35, %.lr.ph38
  %39 = phi i32 [ %48, %.lr.ph38 ], [ %.lcssa22, %._crit_edge35 ]
  %40 = phi i32 [ %45, %.lr.ph38 ], [ %.lcssa21, %._crit_edge35 ]
  %41 = phi i64 [ %49, %.lr.ph38 ], [ 0, %._crit_edge35 ]
  tail call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %cond3 = icmp eq i32 %40, 1
  %spec.select18 = select i1 %cond3, ptr @A_L2L1_0_0_cons_buff_1, ptr @A_L2L1_0_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %cond4 = icmp eq i32 %39, 1
  %42 = select i1 %cond4, ptr @Y_L1L2_0_0_buff_1, ptr @Y_L1L2_0_0_buff_0
  tail call void @vew_apply(ptr nonnull %spec.select18, ptr nonnull %42, ptr nonnull @par_0_0, ptr nonnull @rtp_0_0)
  tail call void @llvm.aie2p.release(i32 54, i32 1)
  %43 = add i32 %40, 1
  %44 = icmp sgt i32 %43, 1
  %45 = select i1 %44, i32 0, i32 %43
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %46 = add i32 %39, 1
  %47 = icmp sgt i32 %46, 1
  %48 = select i1 %47, i32 0, i32 %46
  %49 = add nuw nsw i64 %41, 1
  %50 = icmp slt i64 %49, %37
  br i1 %50, label %.lr.ph38, label %._crit_edge39

._crit_edge39:                                    ; preds = %.lr.ph38, %._crit_edge35
  %.lcssa24 = phi i32 [ %.lcssa21, %._crit_edge35 ], [ %45, %.lr.ph38 ]
  %.lcssa23 = phi i32 [ %.lcssa22, %._crit_edge35 ], [ %48, %.lr.ph38 ]
  %51 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_0_0, i20 16), align 16
  %52 = sext i32 %51 to i64
  %53 = icmp sgt i32 %51, 0
  br i1 %53, label %.lr.ph43, label %._crit_edge44

.lr.ph43:                                         ; preds = %._crit_edge39, %.lr.ph43
  %54 = phi i32 [ %63, %.lr.ph43 ], [ %.lcssa23, %._crit_edge39 ]
  %55 = phi i32 [ %60, %.lr.ph43 ], [ %.lcssa24, %._crit_edge39 ]
  %56 = phi i64 [ %64, %.lr.ph43 ], [ 0, %._crit_edge39 ]
  tail call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %cond5 = icmp eq i32 %55, 1
  %spec.select19 = select i1 %cond5, ptr @A_L2L1_0_0_cons_buff_1, ptr @A_L2L1_0_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %cond6 = icmp eq i32 %54, 1
  %57 = select i1 %cond6, ptr @Y_L1L2_0_0_buff_1, ptr @Y_L1L2_0_0_buff_0
  tail call void @vew_add(ptr nonnull %spec.select19, ptr nonnull @B_L2L1_0_0_cons_buff_0, ptr nonnull %57, ptr nonnull @par_0_0, ptr nonnull @rtp_0_0)
  tail call void @llvm.aie2p.release(i32 54, i32 1)
  %58 = add i32 %55, 1
  %59 = icmp sgt i32 %58, 1
  %60 = select i1 %59, i32 0, i32 %58
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %61 = add i32 %54, 1
  %62 = icmp sgt i32 %61, 1
  %63 = select i1 %62, i32 0, i32 %61
  %64 = add nuw nsw i64 %56, 1
  %65 = icmp slt i64 %64, %52
  br i1 %65, label %.lr.ph43, label %._crit_edge44

._crit_edge44:                                    ; preds = %.lr.ph43, %._crit_edge39
  %.lcssa26 = phi i32 [ %.lcssa24, %._crit_edge39 ], [ %60, %.lr.ph43 ]
  %.lcssa25 = phi i32 [ %.lcssa23, %._crit_edge39 ], [ %63, %.lr.ph43 ]
  %66 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_0_0, i20 20), align 4
  %67 = sext i32 %66 to i64
  %68 = icmp sgt i32 %66, 0
  br i1 %68, label %.lr.ph48, label %._crit_edge49

.lr.ph48:                                         ; preds = %._crit_edge44, %.lr.ph48
  %69 = phi i64 [ %70, %.lr.ph48 ], [ 0, %._crit_edge44 ]
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @vew_stats_out(ptr nonnull @Z_L1L2_0_0_buff_0, ptr nonnull @par_0_0, ptr nonnull @rtp_0_0)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  %70 = add nuw nsw i64 %69, 1
  %71 = icmp slt i64 %70, %67
  br i1 %71, label %.lr.ph48, label %._crit_edge49

._crit_edge49:                                    ; preds = %.lr.ph48, %._crit_edge44
  %72 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_0_0, i20 24), align 8
  %73 = sext i32 %72 to i64
  %74 = icmp sgt i32 %72, 0
  br i1 %74, label %.lr.ph51, label %._crit_edge52

.lr.ph51:                                         ; preds = %._crit_edge49, %.lr.ph51
  %75 = phi i32 [ %84, %.lr.ph51 ], [ %.lcssa25, %._crit_edge49 ]
  %76 = phi i32 [ %81, %.lr.ph51 ], [ %.lcssa26, %._crit_edge49 ]
  %77 = phi i64 [ %85, %.lr.ph51 ], [ 0, %._crit_edge49 ]
  tail call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %cond7 = icmp eq i32 %76, 1
  %spec.select20 = select i1 %cond7, ptr @A_L2L1_0_0_cons_buff_1, ptr @A_L2L1_0_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %cond8 = icmp eq i32 %75, 1
  %78 = select i1 %cond8, ptr @Y_L1L2_0_0_buff_1, ptr @Y_L1L2_0_0_buff_0
  tail call void @vew_rgba(ptr nonnull %spec.select20, ptr nonnull %78, ptr nonnull @rtp_0_0)
  tail call void @llvm.aie2p.release(i32 54, i32 1)
  %79 = add i32 %76, 1
  %80 = icmp sgt i32 %79, 1
  %81 = select i1 %80, i32 0, i32 %79
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %82 = add i32 %75, 1
  %83 = icmp sgt i32 %82, 1
  %84 = select i1 %83, i32 0, i32 %82
  %85 = add nuw nsw i64 %77, 1
  %86 = icmp slt i64 %85, %73
  br i1 %86, label %.lr.ph51, label %._crit_edge52

._crit_edge52:                                    ; preds = %.lr.ph51, %._crit_edge49
  %.lcssa28 = phi i32 [ %.lcssa26, %._crit_edge49 ], [ %81, %.lr.ph51 ]
  %.lcssa27 = phi i32 [ %.lcssa25, %._crit_edge49 ], [ %84, %.lr.ph51 ]
  tail call void @llvm.aie2p.release(i32 56, i32 1)
  %87 = add nuw nsw i64 %4, 1
  %.not = icmp eq i64 %87, 9223372036854775807
  br i1 %.not, label %88, label %1

88:                                               ; preds = %._crit_edge52
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

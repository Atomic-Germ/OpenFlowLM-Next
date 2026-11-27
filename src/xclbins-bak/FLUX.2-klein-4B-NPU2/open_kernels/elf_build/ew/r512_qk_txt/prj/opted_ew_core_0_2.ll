; ModuleID = '/home/atomic-germ/Projects/Next/src/xclbins/FLUX.2-klein-4B-NPU2/open_kernels/elf_build/ew/r512_qk_txt/prj/peano-linked_ew_core_0_2.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@rtp_0_0 = global [24 x i32] zeroinitializer
@par_0_0 = external global [9216 x bfloat]
@A_L2L1_0_0_cons_buff_1 = external global [3072 x bfloat]
@A_L2L1_0_0_cons_buff_0 = external global [3072 x bfloat]
@B_L2L1_0_0_cons_buff_0 = external global [3072 x bfloat]
@Y_L1L2_0_0_buff_1 = external global [3072 x bfloat]
@Y_L1L2_0_0_buff_0 = external global [3072 x bfloat]
@Z_L1L2_0_0_buff_0 = external global [3072 x bfloat]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.release(i32, i32) #0

declare void @load_param(ptr, ptr, i32, i32, ptr) local_unnamed_addr

declare void @ln_mod_row(ptr, ptr, ptr, ptr) local_unnamed_addr

declare void @res_ln_mod_row(ptr, ptr, ptr, ptr, ptr, ptr) local_unnamed_addr

declare void @qk_row(ptr, ptr, ptr, ptr, ptr, ptr, i32) local_unnamed_addr

declare void @bin_row(ptr, ptr, ptr, ptr, ptr) local_unnamed_addr

declare void @un_row(ptr, ptr, ptr) local_unnamed_addr

define void @core_0_2() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %._crit_edge54
  %2 = phi i32 [ 0, %0 ], [ %.lcssa30, %._crit_edge54 ]
  %3 = phi i32 [ 0, %0 ], [ %.lcssa31, %._crit_edge54 ]
  %4 = phi i64 [ 0, %0 ], [ %92, %._crit_edge54 ]
  tail call void @llvm.aie2p.acquire(i32 56, i32 1)
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
  tail call void @load_param(ptr nonnull %spec.select, ptr nonnull @par_0_0, i32 %10, i32 -1, ptr nonnull @rtp_0_0)
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
  br i1 %18, label %.lr.ph33, label %._crit_edge34

.lr.ph33:                                         ; preds = %._crit_edge, %.lr.ph33
  %19 = phi i32 [ %28, %.lr.ph33 ], [ %2, %._crit_edge ]
  %20 = phi i32 [ %25, %.lr.ph33 ], [ %.lcssa, %._crit_edge ]
  %21 = phi i64 [ %29, %.lr.ph33 ], [ 0, %._crit_edge ]
  tail call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %cond1 = icmp eq i32 %20, 1
  %spec.select17 = select i1 %cond1, ptr @A_L2L1_0_0_cons_buff_1, ptr @A_L2L1_0_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %cond2 = icmp eq i32 %19, 1
  %22 = select i1 %cond2, ptr @Y_L1L2_0_0_buff_1, ptr @Y_L1L2_0_0_buff_0
  tail call void @ln_mod_row(ptr nonnull %spec.select17, ptr nonnull %22, ptr nonnull @par_0_0, ptr nonnull @rtp_0_0)
  tail call void @llvm.aie2p.release(i32 54, i32 1)
  %23 = add i32 %20, 1
  %24 = icmp sgt i32 %23, 1
  %25 = select i1 %24, i32 0, i32 %23
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %26 = add i32 %19, 1
  %27 = icmp sgt i32 %26, 1
  %28 = select i1 %27, i32 0, i32 %26
  %29 = add nuw nsw i64 %21, 1
  %30 = icmp slt i64 %29, %17
  br i1 %30, label %.lr.ph33, label %._crit_edge34

._crit_edge34:                                    ; preds = %.lr.ph33, %._crit_edge
  %.lcssa23 = phi i32 [ %.lcssa, %._crit_edge ], [ %25, %.lr.ph33 ]
  %.lcssa22 = phi i32 [ %2, %._crit_edge ], [ %28, %.lr.ph33 ]
  %31 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_0_0, i20 8), align 8
  %32 = sext i32 %31 to i64
  %33 = icmp sgt i32 %31, 0
  br i1 %33, label %.lr.ph38, label %._crit_edge39

.lr.ph38:                                         ; preds = %._crit_edge34, %.lr.ph38
  %34 = phi i32 [ %43, %.lr.ph38 ], [ %.lcssa22, %._crit_edge34 ]
  %35 = phi i32 [ %40, %.lr.ph38 ], [ %.lcssa23, %._crit_edge34 ]
  %36 = phi i64 [ %44, %.lr.ph38 ], [ 0, %._crit_edge34 ]
  tail call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %cond3 = icmp eq i32 %35, 1
  %spec.select18 = select i1 %cond3, ptr @A_L2L1_0_0_cons_buff_1, ptr @A_L2L1_0_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %cond4 = icmp eq i32 %34, 1
  %37 = select i1 %cond4, ptr @Y_L1L2_0_0_buff_1, ptr @Y_L1L2_0_0_buff_0
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @res_ln_mod_row(ptr nonnull %spec.select18, ptr nonnull @B_L2L1_0_0_cons_buff_0, ptr nonnull %37, ptr nonnull @Z_L1L2_0_0_buff_0, ptr nonnull @par_0_0, ptr nonnull @rtp_0_0)
  tail call void @llvm.aie2p.release(i32 54, i32 1)
  %38 = add i32 %35, 1
  %39 = icmp sgt i32 %38, 1
  %40 = select i1 %39, i32 0, i32 %38
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %41 = add i32 %34, 1
  %42 = icmp sgt i32 %41, 1
  %43 = select i1 %42, i32 0, i32 %41
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  %44 = add nuw nsw i64 %36, 1
  %45 = icmp slt i64 %44, %32
  br i1 %45, label %.lr.ph38, label %._crit_edge39

._crit_edge39:                                    ; preds = %.lr.ph38, %._crit_edge34
  %.lcssa25 = phi i32 [ %.lcssa23, %._crit_edge34 ], [ %40, %.lr.ph38 ]
  %.lcssa24 = phi i32 [ %.lcssa22, %._crit_edge34 ], [ %43, %.lr.ph38 ]
  %46 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_0_0, i20 12), align 4
  %47 = sext i32 %46 to i64
  %48 = icmp sgt i32 %46, 0
  br i1 %48, label %.lr.ph43, label %._crit_edge44

.lr.ph43:                                         ; preds = %._crit_edge39, %.lr.ph43
  %49 = phi i32 [ %59, %.lr.ph43 ], [ %.lcssa24, %._crit_edge39 ]
  %50 = phi i32 [ %56, %.lr.ph43 ], [ %.lcssa25, %._crit_edge39 ]
  %51 = phi i64 [ %60, %.lr.ph43 ], [ 0, %._crit_edge39 ]
  tail call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %cond5 = icmp eq i32 %50, 1
  %spec.select19 = select i1 %cond5, ptr @A_L2L1_0_0_cons_buff_1, ptr @A_L2L1_0_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %cond6 = icmp eq i32 %49, 1
  %52 = select i1 %cond6, ptr @Y_L1L2_0_0_buff_1, ptr @Y_L1L2_0_0_buff_0
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  %53 = trunc i64 %51 to i32
  tail call void @qk_row(ptr nonnull %spec.select19, ptr nonnull @B_L2L1_0_0_cons_buff_0, ptr nonnull %52, ptr nonnull @Z_L1L2_0_0_buff_0, ptr nonnull @par_0_0, ptr nonnull @rtp_0_0, i32 %53)
  tail call void @llvm.aie2p.release(i32 54, i32 1)
  %54 = add i32 %50, 1
  %55 = icmp sgt i32 %54, 1
  %56 = select i1 %55, i32 0, i32 %54
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %57 = add i32 %49, 1
  %58 = icmp sgt i32 %57, 1
  %59 = select i1 %58, i32 0, i32 %57
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  %60 = add nuw nsw i64 %51, 1
  %61 = icmp slt i64 %60, %47
  br i1 %61, label %.lr.ph43, label %._crit_edge44

._crit_edge44:                                    ; preds = %.lr.ph43, %._crit_edge39
  %.lcssa27 = phi i32 [ %.lcssa25, %._crit_edge39 ], [ %56, %.lr.ph43 ]
  %.lcssa26 = phi i32 [ %.lcssa24, %._crit_edge39 ], [ %59, %.lr.ph43 ]
  %62 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_0_0, i20 16), align 16
  %63 = sext i32 %62 to i64
  %64 = icmp sgt i32 %62, 0
  br i1 %64, label %.lr.ph48, label %._crit_edge49

.lr.ph48:                                         ; preds = %._crit_edge44, %.lr.ph48
  %65 = phi i32 [ %74, %.lr.ph48 ], [ %.lcssa26, %._crit_edge44 ]
  %66 = phi i32 [ %71, %.lr.ph48 ], [ %.lcssa27, %._crit_edge44 ]
  %67 = phi i64 [ %75, %.lr.ph48 ], [ 0, %._crit_edge44 ]
  tail call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %cond7 = icmp eq i32 %66, 1
  %spec.select20 = select i1 %cond7, ptr @A_L2L1_0_0_cons_buff_1, ptr @A_L2L1_0_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %cond8 = icmp eq i32 %65, 1
  %68 = select i1 %cond8, ptr @Y_L1L2_0_0_buff_1, ptr @Y_L1L2_0_0_buff_0
  tail call void @bin_row(ptr nonnull %spec.select20, ptr nonnull @B_L2L1_0_0_cons_buff_0, ptr nonnull %68, ptr nonnull @par_0_0, ptr nonnull @rtp_0_0)
  tail call void @llvm.aie2p.release(i32 54, i32 1)
  %69 = add i32 %66, 1
  %70 = icmp sgt i32 %69, 1
  %71 = select i1 %70, i32 0, i32 %69
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %72 = add i32 %65, 1
  %73 = icmp sgt i32 %72, 1
  %74 = select i1 %73, i32 0, i32 %72
  %75 = add nuw nsw i64 %67, 1
  %76 = icmp slt i64 %75, %63
  br i1 %76, label %.lr.ph48, label %._crit_edge49

._crit_edge49:                                    ; preds = %.lr.ph48, %._crit_edge44
  %.lcssa29 = phi i32 [ %.lcssa27, %._crit_edge44 ], [ %71, %.lr.ph48 ]
  %.lcssa28 = phi i32 [ %.lcssa26, %._crit_edge44 ], [ %74, %.lr.ph48 ]
  %77 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_0_0, i20 20), align 4
  %78 = sext i32 %77 to i64
  %79 = icmp sgt i32 %77, 0
  br i1 %79, label %.lr.ph53, label %._crit_edge54

.lr.ph53:                                         ; preds = %._crit_edge49, %.lr.ph53
  %80 = phi i32 [ %89, %.lr.ph53 ], [ %.lcssa28, %._crit_edge49 ]
  %81 = phi i32 [ %86, %.lr.ph53 ], [ %.lcssa29, %._crit_edge49 ]
  %82 = phi i64 [ %90, %.lr.ph53 ], [ 0, %._crit_edge49 ]
  tail call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %cond9 = icmp eq i32 %81, 1
  %spec.select21 = select i1 %cond9, ptr @A_L2L1_0_0_cons_buff_1, ptr @A_L2L1_0_0_cons_buff_0
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %cond10 = icmp eq i32 %80, 1
  %83 = select i1 %cond10, ptr @Y_L1L2_0_0_buff_1, ptr @Y_L1L2_0_0_buff_0
  tail call void @un_row(ptr nonnull %spec.select21, ptr nonnull %83, ptr nonnull @rtp_0_0)
  tail call void @llvm.aie2p.release(i32 54, i32 1)
  %84 = add i32 %81, 1
  %85 = icmp sgt i32 %84, 1
  %86 = select i1 %85, i32 0, i32 %84
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %87 = add i32 %80, 1
  %88 = icmp sgt i32 %87, 1
  %89 = select i1 %88, i32 0, i32 %87
  %90 = add nuw nsw i64 %82, 1
  %91 = icmp slt i64 %90, %78
  br i1 %91, label %.lr.ph53, label %._crit_edge54

._crit_edge54:                                    ; preds = %.lr.ph53, %._crit_edge49
  %.lcssa31 = phi i32 [ %.lcssa29, %._crit_edge49 ], [ %86, %.lr.ph53 ]
  %.lcssa30 = phi i32 [ %.lcssa28, %._crit_edge49 ], [ %89, %.lr.ph53 ]
  tail call void @llvm.aie2p.release(i32 56, i32 1)
  %92 = add nuw nsw i64 %4, 1
  %.not = icmp eq i64 %92, 9223372036854775807
  br i1 %.not, label %93, label %1

93:                                               ; preds = %._crit_edge54
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

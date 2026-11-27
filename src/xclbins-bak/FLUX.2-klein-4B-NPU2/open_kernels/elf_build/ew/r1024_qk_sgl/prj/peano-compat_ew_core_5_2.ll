; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@rtp_7_1 = external global [24 x i32]
@par_7_1 = external global [9216 x bfloat]
@rtp_7_0 = external global [24 x i32]
@par_7_0 = external global [9216 x bfloat]
@rtp_6_1 = external global [24 x i32]
@par_6_1 = external global [9216 x bfloat]
@rtp_6_0 = external global [24 x i32]
@par_6_0 = external global [9216 x bfloat]
@rtp_5_1 = external global [24 x i32]
@par_5_1 = external global [9216 x bfloat]
@rtp_5_0 = global [24 x i32] zeroinitializer
@par_5_0 = external global [9216 x bfloat]
@rtp_4_1 = external global [24 x i32]
@par_4_1 = external global [9216 x bfloat]
@rtp_4_0 = external global [24 x i32]
@par_4_0 = external global [9216 x bfloat]
@rtp_3_1 = external global [24 x i32]
@par_3_1 = external global [9216 x bfloat]
@rtp_3_0 = external global [24 x i32]
@par_3_0 = external global [9216 x bfloat]
@rtp_2_1 = external global [24 x i32]
@par_2_1 = external global [9216 x bfloat]
@rtp_2_0 = external global [24 x i32]
@par_2_0 = external global [9216 x bfloat]
@rtp_1_1 = external global [24 x i32]
@par_1_1 = external global [9216 x bfloat]
@rtp_1_0 = external global [24 x i32]
@par_1_0 = external global [9216 x bfloat]
@rtp_0_1 = external global [24 x i32]
@par_0_1 = external global [9216 x bfloat]
@rtp_0_0 = external global [24 x i32]
@par_0_0 = external global [9216 x bfloat]
@A_L3L2_0_cons_buff_1 = external global [6144 x bfloat]
@A_L3L2_0_cons_buff_0 = external global [6144 x bfloat]
@A_L2L1_0_0_cons_buff_1 = external global [3072 x bfloat]
@A_L2L1_0_0_cons_buff_0 = external global [3072 x bfloat]
@A_L3L2_1_cons_buff_1 = external global [6144 x bfloat]
@A_L3L2_1_cons_buff_0 = external global [6144 x bfloat]
@A_L3L2_2_cons_buff_1 = external global [6144 x bfloat]
@A_L3L2_2_cons_buff_0 = external global [6144 x bfloat]
@A_L2L1_0_1_cons_buff_1 = external global [3072 x bfloat]
@A_L2L1_0_1_cons_buff_0 = external global [3072 x bfloat]
@A_L3L2_3_cons_buff_1 = external global [6144 x bfloat]
@A_L3L2_3_cons_buff_0 = external global [6144 x bfloat]
@A_L2L1_1_0_cons_buff_1 = external global [3072 x bfloat]
@A_L2L1_1_0_cons_buff_0 = external global [3072 x bfloat]
@A_L3L2_4_cons_buff_1 = external global [6144 x bfloat]
@A_L3L2_4_cons_buff_0 = external global [6144 x bfloat]
@A_L3L2_5_cons_buff_1 = external global [6144 x bfloat]
@A_L3L2_5_cons_buff_0 = external global [6144 x bfloat]
@A_L2L1_1_1_cons_buff_1 = external global [3072 x bfloat]
@A_L2L1_1_1_cons_buff_0 = external global [3072 x bfloat]
@A_L3L2_6_cons_buff_1 = external global [6144 x bfloat]
@A_L3L2_6_cons_buff_0 = external global [6144 x bfloat]
@A_L2L1_2_0_cons_buff_1 = external global [3072 x bfloat]
@A_L2L1_2_0_cons_buff_0 = external global [3072 x bfloat]
@A_L3L2_7_cons_buff_1 = external global [6144 x bfloat]
@A_L3L2_7_cons_buff_0 = external global [6144 x bfloat]
@B_L3L2_0_cons_buff_1 = external global [6144 x bfloat]
@B_L3L2_0_cons_buff_0 = external global [6144 x bfloat]
@A_L2L1_2_1_cons_buff_1 = external global [3072 x bfloat]
@A_L2L1_2_1_cons_buff_0 = external global [3072 x bfloat]
@B_L3L2_1_cons_buff_1 = external global [6144 x bfloat]
@B_L3L2_1_cons_buff_0 = external global [6144 x bfloat]
@A_L2L1_3_0_cons_buff_1 = external global [3072 x bfloat]
@A_L2L1_3_0_cons_buff_0 = external global [3072 x bfloat]
@B_L3L2_2_cons_buff_1 = external global [6144 x bfloat]
@B_L3L2_2_cons_buff_0 = external global [6144 x bfloat]
@B_L3L2_3_cons_buff_1 = external global [6144 x bfloat]
@B_L3L2_3_cons_buff_0 = external global [6144 x bfloat]
@A_L2L1_3_1_cons_buff_1 = external global [3072 x bfloat]
@A_L2L1_3_1_cons_buff_0 = external global [3072 x bfloat]
@B_L3L2_4_cons_buff_1 = external global [6144 x bfloat]
@B_L3L2_4_cons_buff_0 = external global [6144 x bfloat]
@A_L2L1_4_0_cons_buff_1 = external global [3072 x bfloat]
@A_L2L1_4_0_cons_buff_0 = external global [3072 x bfloat]
@B_L3L2_5_cons_buff_1 = external global [6144 x bfloat]
@B_L3L2_5_cons_buff_0 = external global [6144 x bfloat]
@B_L3L2_6_cons_buff_1 = external global [6144 x bfloat]
@B_L3L2_6_cons_buff_0 = external global [6144 x bfloat]
@A_L2L1_4_1_cons_buff_1 = external global [3072 x bfloat]
@A_L2L1_4_1_cons_buff_0 = external global [3072 x bfloat]
@B_L3L2_7_cons_buff_1 = external global [6144 x bfloat]
@B_L3L2_7_cons_buff_0 = external global [6144 x bfloat]
@A_L2L1_5_0_cons_buff_1 = external global [3072 x bfloat]
@A_L2L1_5_0_cons_buff_0 = external global [3072 x bfloat]
@Y_L2L3_0_buff_1 = external global [6144 x bfloat]
@Y_L2L3_0_buff_0 = external global [6144 x bfloat]
@Y_L2L3_1_buff_1 = external global [6144 x bfloat]
@Y_L2L3_1_buff_0 = external global [6144 x bfloat]
@A_L2L1_5_1_cons_buff_1 = external global [3072 x bfloat]
@A_L2L1_5_1_cons_buff_0 = external global [3072 x bfloat]
@Y_L2L3_2_buff_1 = external global [6144 x bfloat]
@Y_L2L3_2_buff_0 = external global [6144 x bfloat]
@A_L2L1_6_0_cons_buff_1 = external global [3072 x bfloat]
@A_L2L1_6_0_cons_buff_0 = external global [3072 x bfloat]
@Y_L2L3_3_buff_1 = external global [6144 x bfloat]
@Y_L2L3_3_buff_0 = external global [6144 x bfloat]
@Y_L2L3_4_buff_1 = external global [6144 x bfloat]
@Y_L2L3_4_buff_0 = external global [6144 x bfloat]
@A_L2L1_6_1_cons_buff_1 = external global [3072 x bfloat]
@A_L2L1_6_1_cons_buff_0 = external global [3072 x bfloat]
@Y_L2L3_5_buff_1 = external global [6144 x bfloat]
@Y_L2L3_5_buff_0 = external global [6144 x bfloat]
@A_L2L1_7_0_cons_buff_1 = external global [3072 x bfloat]
@A_L2L1_7_0_cons_buff_0 = external global [3072 x bfloat]
@Y_L2L3_6_buff_1 = external global [6144 x bfloat]
@Y_L2L3_6_buff_0 = external global [6144 x bfloat]
@Y_L2L3_7_buff_1 = external global [6144 x bfloat]
@Y_L2L3_7_buff_0 = external global [6144 x bfloat]
@A_L2L1_7_1_cons_buff_1 = external global [3072 x bfloat]
@A_L2L1_7_1_cons_buff_0 = external global [3072 x bfloat]
@Z_L2L3_0_buff_1 = external global [6144 x bfloat]
@Z_L2L3_0_buff_0 = external global [6144 x bfloat]
@B_L2L1_0_0_cons_buff_0 = external global [3072 x bfloat]
@Z_L2L3_1_buff_1 = external global [6144 x bfloat]
@Z_L2L3_1_buff_0 = external global [6144 x bfloat]
@Z_L2L3_2_buff_1 = external global [6144 x bfloat]
@Z_L2L3_2_buff_0 = external global [6144 x bfloat]
@B_L2L1_0_1_cons_buff_0 = external global [3072 x bfloat]
@Z_L2L3_3_buff_1 = external global [6144 x bfloat]
@Z_L2L3_3_buff_0 = external global [6144 x bfloat]
@B_L2L1_1_0_cons_buff_0 = external global [3072 x bfloat]
@Z_L2L3_4_buff_1 = external global [6144 x bfloat]
@Z_L2L3_4_buff_0 = external global [6144 x bfloat]
@Z_L2L3_5_buff_1 = external global [6144 x bfloat]
@Z_L2L3_5_buff_0 = external global [6144 x bfloat]
@B_L2L1_1_1_cons_buff_0 = external global [3072 x bfloat]
@Z_L2L3_6_buff_1 = external global [6144 x bfloat]
@Z_L2L3_6_buff_0 = external global [6144 x bfloat]
@B_L2L1_2_0_cons_buff_0 = external global [3072 x bfloat]
@Z_L2L3_7_buff_1 = external global [6144 x bfloat]
@Z_L2L3_7_buff_0 = external global [6144 x bfloat]
@B_L2L1_2_1_cons_buff_0 = external global [3072 x bfloat]
@B_L2L1_3_0_cons_buff_0 = external global [3072 x bfloat]
@B_L2L1_3_1_cons_buff_0 = external global [3072 x bfloat]
@B_L2L1_4_0_cons_buff_0 = external global [3072 x bfloat]
@B_L2L1_4_1_cons_buff_0 = external global [3072 x bfloat]
@B_L2L1_5_0_cons_buff_0 = external global [3072 x bfloat]
@B_L2L1_5_1_cons_buff_0 = external global [3072 x bfloat]
@B_L2L1_6_0_cons_buff_0 = external global [3072 x bfloat]
@B_L2L1_6_1_cons_buff_0 = external global [3072 x bfloat]
@B_L2L1_7_0_cons_buff_0 = external global [3072 x bfloat]
@B_L2L1_7_1_cons_buff_0 = external global [3072 x bfloat]
@Y_L1L2_0_0_buff_1 = external global [3072 x bfloat]
@Y_L1L2_0_0_buff_0 = external global [3072 x bfloat]
@Y_L1L2_0_1_buff_1 = external global [3072 x bfloat]
@Y_L1L2_0_1_buff_0 = external global [3072 x bfloat]
@Y_L1L2_1_0_buff_1 = external global [3072 x bfloat]
@Y_L1L2_1_0_buff_0 = external global [3072 x bfloat]
@Y_L1L2_1_1_buff_1 = external global [3072 x bfloat]
@Y_L1L2_1_1_buff_0 = external global [3072 x bfloat]
@Y_L1L2_2_0_buff_1 = external global [3072 x bfloat]
@Y_L1L2_2_0_buff_0 = external global [3072 x bfloat]
@Y_L1L2_2_1_buff_1 = external global [3072 x bfloat]
@Y_L1L2_2_1_buff_0 = external global [3072 x bfloat]
@Y_L1L2_3_0_buff_1 = external global [3072 x bfloat]
@Y_L1L2_3_0_buff_0 = external global [3072 x bfloat]
@Y_L1L2_3_1_buff_1 = external global [3072 x bfloat]
@Y_L1L2_3_1_buff_0 = external global [3072 x bfloat]
@Y_L1L2_4_0_buff_1 = external global [3072 x bfloat]
@Y_L1L2_4_0_buff_0 = external global [3072 x bfloat]
@Y_L1L2_4_1_buff_1 = external global [3072 x bfloat]
@Y_L1L2_4_1_buff_0 = external global [3072 x bfloat]
@Y_L1L2_5_0_buff_1 = external global [3072 x bfloat]
@Y_L1L2_5_0_buff_0 = external global [3072 x bfloat]
@Y_L1L2_5_1_buff_1 = external global [3072 x bfloat]
@Y_L1L2_5_1_buff_0 = external global [3072 x bfloat]
@Y_L1L2_6_0_buff_1 = external global [3072 x bfloat]
@Y_L1L2_6_0_buff_0 = external global [3072 x bfloat]
@Y_L1L2_6_1_buff_1 = external global [3072 x bfloat]
@Y_L1L2_6_1_buff_0 = external global [3072 x bfloat]
@Y_L1L2_7_0_buff_1 = external global [3072 x bfloat]
@Y_L1L2_7_0_buff_0 = external global [3072 x bfloat]
@Y_L1L2_7_1_buff_1 = external global [3072 x bfloat]
@Y_L1L2_7_1_buff_0 = external global [3072 x bfloat]
@Z_L1L2_0_0_buff_0 = external global [3072 x bfloat]
@Z_L1L2_0_1_buff_0 = external global [3072 x bfloat]
@Z_L1L2_1_0_buff_0 = external global [3072 x bfloat]
@Z_L1L2_1_1_buff_0 = external global [3072 x bfloat]
@Z_L1L2_2_0_buff_0 = external global [3072 x bfloat]
@Z_L1L2_2_1_buff_0 = external global [3072 x bfloat]
@Z_L1L2_3_0_buff_0 = external global [3072 x bfloat]
@Z_L1L2_3_1_buff_0 = external global [3072 x bfloat]
@Z_L1L2_4_0_buff_0 = external global [3072 x bfloat]
@Z_L1L2_4_1_buff_0 = external global [3072 x bfloat]
@Z_L1L2_5_0_buff_0 = external global [3072 x bfloat]
@Z_L1L2_5_1_buff_0 = external global [3072 x bfloat]
@Z_L1L2_6_0_buff_0 = external global [3072 x bfloat]
@Z_L1L2_6_1_buff_0 = external global [3072 x bfloat]
@Z_L1L2_7_0_buff_0 = external global [3072 x bfloat]
@Z_L1L2_7_1_buff_0 = external global [3072 x bfloat]

declare void @debug_i32(i32)

; Unknown intrinsic
declare void @llvm.aie2p.event(i32)

; Unknown intrinsic
declare void @llvm.aie2p.put.ms(i32, i32)

; Unknown intrinsic
declare { i32, i32 } @llvm.aie2p.get.ss()

; Unknown intrinsic
declare void @llvm.aie2p.mcd.write.vec(<16 x i32>, i32)

; Unknown intrinsic
declare <16 x i32> @llvm.aie2p.scd.read.vec(i32)

; Unknown intrinsic
declare void @llvm.aie2p.acquire(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.release(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2p.set.ctrl.reg(i32, i32)

declare void @load_param(ptr, ptr, i32, i32, ptr)

declare void @ln_mod_row(ptr, ptr, ptr, ptr)

declare void @res_ln_mod_row(ptr, ptr, ptr, ptr, ptr, ptr)

declare void @qk_row(ptr, ptr, ptr, ptr, ptr, ptr, i32)

declare void @bin_row(ptr, ptr, ptr, ptr, ptr)

declare void @un_row(ptr, ptr, ptr)

define void @core_5_2() {
  br label %1

1:                                                ; preds = %155, %0
  %2 = phi i64 [ %156, %155 ], [ 0, %0 ]
  %3 = phi i32 [ %134, %155 ], [ 0, %0 ]
  %4 = phi i32 [ %135, %155 ], [ 0, %0 ]
  %5 = icmp slt i64 %2, 9223372036854775807
  br i1 %5, label %6, label %157

6:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 56, i32 1)
  %7 = load i32, ptr @rtp_5_0
  %8 = sext i32 %7 to i64
  br label %9

9:                                                ; preds = %18, %6
  %10 = phi i64 [ %23, %18 ], [ 0, %6 ]
  %11 = phi i32 [ %22, %18 ], [ %3, %6 ]
  %12 = icmp slt i64 %10, %8
  br i1 %12, label %13, label %24

13:                                               ; preds = %9
  call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %14 = sext i32 %11 to i64
  switch i64 %14, label %15 [
    i64 0, label %158
    i64 1, label %160
  ]

15:                                               ; preds = %158, %160, %13
  %16 = phi ptr [ %161, %160 ], [ %159, %158 ], [ @A_L2L1_5_0_cons_buff_0, %13 ]
  %17 = getelementptr [3072 x bfloat], ptr %16, i32 0, i32 0
  br label %18

18:                                               ; preds = %15
  %19 = trunc i64 %10 to i32
  call void @load_param(ptr %17, ptr @par_5_0, i32 %19, i32 -1, ptr @rtp_5_0)
  call void @llvm.aie2p.release(i32 54, i32 1)
  %20 = add i32 %11, 1
  %21 = icmp sge i32 %20, 2
  %22 = select i1 %21, i32 0, i32 %20
  %23 = add i64 %10, 1
  br label %9

24:                                               ; preds = %9
  %25 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_5_0, i64 4)
  %26 = sext i32 %25 to i64
  br label %27

27:                                               ; preds = %42, %24
  %28 = phi i64 [ %49, %42 ], [ 0, %24 ]
  %29 = phi i32 [ %45, %42 ], [ %11, %24 ]
  %30 = phi i32 [ %48, %42 ], [ %4, %24 ]
  %31 = icmp slt i64 %28, %26
  br i1 %31, label %32, label %50

32:                                               ; preds = %27
  call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %33 = sext i32 %29 to i64
  switch i64 %33, label %34 [
    i64 0, label %162
    i64 1, label %164
  ]

34:                                               ; preds = %162, %164, %32
  %35 = phi ptr [ %165, %164 ], [ %163, %162 ], [ @A_L2L1_5_0_cons_buff_0, %32 ]
  %36 = getelementptr [3072 x bfloat], ptr %35, i32 0, i32 0
  br label %37

37:                                               ; preds = %34
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %38 = sext i32 %30 to i64
  switch i64 %38, label %39 [
    i64 0, label %166
    i64 1, label %168
  ]

39:                                               ; preds = %166, %168, %37
  %40 = phi ptr [ %169, %168 ], [ %167, %166 ], [ @Y_L1L2_5_0_buff_0, %37 ]
  %41 = getelementptr [3072 x bfloat], ptr %40, i32 0, i32 0
  br label %42

42:                                               ; preds = %39
  call void @ln_mod_row(ptr %36, ptr %41, ptr @par_5_0, ptr @rtp_5_0)
  call void @llvm.aie2p.release(i32 54, i32 1)
  %43 = add i32 %29, 1
  %44 = icmp sge i32 %43, 2
  %45 = select i1 %44, i32 0, i32 %43
  call void @llvm.aie2p.release(i32 51, i32 1)
  %46 = add i32 %30, 1
  %47 = icmp sge i32 %46, 2
  %48 = select i1 %47, i32 0, i32 %46
  %49 = add i64 %28, 1
  br label %27

50:                                               ; preds = %27
  %51 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_5_0, i64 8)
  %52 = sext i32 %51 to i64
  br label %53

53:                                               ; preds = %68, %50
  %54 = phi i64 [ %75, %68 ], [ 0, %50 ]
  %55 = phi i32 [ %71, %68 ], [ %29, %50 ]
  %56 = phi i32 [ %74, %68 ], [ %30, %50 ]
  %57 = icmp slt i64 %54, %52
  br i1 %57, label %58, label %76

58:                                               ; preds = %53
  call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %59 = sext i32 %55 to i64
  switch i64 %59, label %60 [
    i64 0, label %170
    i64 1, label %172
  ]

60:                                               ; preds = %170, %172, %58
  %61 = phi ptr [ %173, %172 ], [ %171, %170 ], [ @A_L2L1_5_0_cons_buff_0, %58 ]
  %62 = getelementptr [3072 x bfloat], ptr %61, i32 0, i32 0
  br label %63

63:                                               ; preds = %60
  call void @llvm.aie2p.acquire(i32 53, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %64 = sext i32 %56 to i64
  switch i64 %64, label %65 [
    i64 0, label %174
    i64 1, label %176
  ]

65:                                               ; preds = %174, %176, %63
  %66 = phi ptr [ %177, %176 ], [ %175, %174 ], [ @Y_L1L2_5_0_buff_0, %63 ]
  %67 = getelementptr [3072 x bfloat], ptr %66, i32 0, i32 0
  br label %68

68:                                               ; preds = %65
  call void @llvm.aie2p.acquire(i32 48, i32 -1)
  call void @res_ln_mod_row(ptr %62, ptr @B_L2L1_5_0_cons_buff_0, ptr %67, ptr @Z_L1L2_5_0_buff_0, ptr @par_5_0, ptr @rtp_5_0)
  call void @llvm.aie2p.release(i32 54, i32 1)
  %69 = add i32 %55, 1
  %70 = icmp sge i32 %69, 2
  %71 = select i1 %70, i32 0, i32 %69
  call void @llvm.aie2p.release(i32 52, i32 1)
  call void @llvm.aie2p.release(i32 51, i32 1)
  %72 = add i32 %56, 1
  %73 = icmp sge i32 %72, 2
  %74 = select i1 %73, i32 0, i32 %72
  call void @llvm.aie2p.release(i32 49, i32 1)
  %75 = add i64 %54, 1
  br label %53

76:                                               ; preds = %53
  %77 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_5_0, i64 12)
  %78 = sext i32 %77 to i64
  br label %79

79:                                               ; preds = %94, %76
  %80 = phi i64 [ %102, %94 ], [ 0, %76 ]
  %81 = phi i32 [ %98, %94 ], [ %55, %76 ]
  %82 = phi i32 [ %101, %94 ], [ %56, %76 ]
  %83 = icmp slt i64 %80, %78
  br i1 %83, label %84, label %103

84:                                               ; preds = %79
  call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %85 = sext i32 %81 to i64
  switch i64 %85, label %86 [
    i64 0, label %178
    i64 1, label %180
  ]

86:                                               ; preds = %178, %180, %84
  %87 = phi ptr [ %181, %180 ], [ %179, %178 ], [ @A_L2L1_5_0_cons_buff_0, %84 ]
  %88 = getelementptr [3072 x bfloat], ptr %87, i32 0, i32 0
  br label %89

89:                                               ; preds = %86
  call void @llvm.aie2p.acquire(i32 53, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %90 = sext i32 %82 to i64
  switch i64 %90, label %91 [
    i64 0, label %182
    i64 1, label %184
  ]

91:                                               ; preds = %182, %184, %89
  %92 = phi ptr [ %185, %184 ], [ %183, %182 ], [ @Y_L1L2_5_0_buff_0, %89 ]
  %93 = getelementptr [3072 x bfloat], ptr %92, i32 0, i32 0
  br label %94

94:                                               ; preds = %91
  call void @llvm.aie2p.acquire(i32 48, i32 -1)
  %95 = trunc i64 %80 to i32
  call void @qk_row(ptr %88, ptr @B_L2L1_5_0_cons_buff_0, ptr %93, ptr @Z_L1L2_5_0_buff_0, ptr @par_5_0, ptr @rtp_5_0, i32 %95)
  call void @llvm.aie2p.release(i32 54, i32 1)
  %96 = add i32 %81, 1
  %97 = icmp sge i32 %96, 2
  %98 = select i1 %97, i32 0, i32 %96
  call void @llvm.aie2p.release(i32 52, i32 1)
  call void @llvm.aie2p.release(i32 51, i32 1)
  %99 = add i32 %82, 1
  %100 = icmp sge i32 %99, 2
  %101 = select i1 %100, i32 0, i32 %99
  call void @llvm.aie2p.release(i32 49, i32 1)
  %102 = add i64 %80, 1
  br label %79

103:                                              ; preds = %79
  %104 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_5_0, i64 16)
  %105 = sext i32 %104 to i64
  br label %106

106:                                              ; preds = %121, %103
  %107 = phi i64 [ %128, %121 ], [ 0, %103 ]
  %108 = phi i32 [ %124, %121 ], [ %81, %103 ]
  %109 = phi i32 [ %127, %121 ], [ %82, %103 ]
  %110 = icmp slt i64 %107, %105
  br i1 %110, label %111, label %129

111:                                              ; preds = %106
  call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %112 = sext i32 %108 to i64
  switch i64 %112, label %113 [
    i64 0, label %186
    i64 1, label %188
  ]

113:                                              ; preds = %186, %188, %111
  %114 = phi ptr [ %189, %188 ], [ %187, %186 ], [ @A_L2L1_5_0_cons_buff_0, %111 ]
  %115 = getelementptr [3072 x bfloat], ptr %114, i32 0, i32 0
  br label %116

116:                                              ; preds = %113
  call void @llvm.aie2p.acquire(i32 53, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %117 = sext i32 %109 to i64
  switch i64 %117, label %118 [
    i64 0, label %190
    i64 1, label %192
  ]

118:                                              ; preds = %190, %192, %116
  %119 = phi ptr [ %193, %192 ], [ %191, %190 ], [ @Y_L1L2_5_0_buff_0, %116 ]
  %120 = getelementptr [3072 x bfloat], ptr %119, i32 0, i32 0
  br label %121

121:                                              ; preds = %118
  call void @bin_row(ptr %115, ptr @B_L2L1_5_0_cons_buff_0, ptr %120, ptr @par_5_0, ptr @rtp_5_0)
  call void @llvm.aie2p.release(i32 54, i32 1)
  %122 = add i32 %108, 1
  %123 = icmp sge i32 %122, 2
  %124 = select i1 %123, i32 0, i32 %122
  call void @llvm.aie2p.release(i32 52, i32 1)
  call void @llvm.aie2p.release(i32 51, i32 1)
  %125 = add i32 %109, 1
  %126 = icmp sge i32 %125, 2
  %127 = select i1 %126, i32 0, i32 %125
  %128 = add i64 %107, 1
  br label %106

129:                                              ; preds = %106
  %130 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_5_0, i64 20)
  %131 = sext i32 %130 to i64
  br label %132

132:                                              ; preds = %147, %129
  %133 = phi i64 [ %154, %147 ], [ 0, %129 ]
  %134 = phi i32 [ %150, %147 ], [ %108, %129 ]
  %135 = phi i32 [ %153, %147 ], [ %109, %129 ]
  %136 = icmp slt i64 %133, %131
  br i1 %136, label %137, label %155

137:                                              ; preds = %132
  call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %138 = sext i32 %134 to i64
  switch i64 %138, label %139 [
    i64 0, label %194
    i64 1, label %196
  ]

139:                                              ; preds = %194, %196, %137
  %140 = phi ptr [ %197, %196 ], [ %195, %194 ], [ @A_L2L1_5_0_cons_buff_0, %137 ]
  %141 = getelementptr [3072 x bfloat], ptr %140, i32 0, i32 0
  br label %142

142:                                              ; preds = %139
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %143 = sext i32 %135 to i64
  switch i64 %143, label %144 [
    i64 0, label %198
    i64 1, label %200
  ]

144:                                              ; preds = %198, %200, %142
  %145 = phi ptr [ %201, %200 ], [ %199, %198 ], [ @Y_L1L2_5_0_buff_0, %142 ]
  %146 = getelementptr [3072 x bfloat], ptr %145, i32 0, i32 0
  br label %147

147:                                              ; preds = %144
  call void @un_row(ptr %141, ptr %146, ptr @rtp_5_0)
  call void @llvm.aie2p.release(i32 54, i32 1)
  %148 = add i32 %134, 1
  %149 = icmp sge i32 %148, 2
  %150 = select i1 %149, i32 0, i32 %148
  call void @llvm.aie2p.release(i32 51, i32 1)
  %151 = add i32 %135, 1
  %152 = icmp sge i32 %151, 2
  %153 = select i1 %152, i32 0, i32 %151
  %154 = add i64 %133, 1
  br label %132

155:                                              ; preds = %132
  call void @llvm.aie2p.release(i32 56, i32 1)
  %156 = add i64 %2, 1
  br label %1

157:                                              ; preds = %1
  ret void

158:                                              ; preds = %13
  %159 = phi ptr [ @A_L2L1_5_0_cons_buff_0, %13 ]
  br label %15

160:                                              ; preds = %13
  %161 = phi ptr [ @A_L2L1_5_0_cons_buff_1, %13 ]
  br label %15

162:                                              ; preds = %32
  %163 = phi ptr [ @A_L2L1_5_0_cons_buff_0, %32 ]
  br label %34

164:                                              ; preds = %32
  %165 = phi ptr [ @A_L2L1_5_0_cons_buff_1, %32 ]
  br label %34

166:                                              ; preds = %37
  %167 = phi ptr [ @Y_L1L2_5_0_buff_0, %37 ]
  br label %39

168:                                              ; preds = %37
  %169 = phi ptr [ @Y_L1L2_5_0_buff_1, %37 ]
  br label %39

170:                                              ; preds = %58
  %171 = phi ptr [ @A_L2L1_5_0_cons_buff_0, %58 ]
  br label %60

172:                                              ; preds = %58
  %173 = phi ptr [ @A_L2L1_5_0_cons_buff_1, %58 ]
  br label %60

174:                                              ; preds = %63
  %175 = phi ptr [ @Y_L1L2_5_0_buff_0, %63 ]
  br label %65

176:                                              ; preds = %63
  %177 = phi ptr [ @Y_L1L2_5_0_buff_1, %63 ]
  br label %65

178:                                              ; preds = %84
  %179 = phi ptr [ @A_L2L1_5_0_cons_buff_0, %84 ]
  br label %86

180:                                              ; preds = %84
  %181 = phi ptr [ @A_L2L1_5_0_cons_buff_1, %84 ]
  br label %86

182:                                              ; preds = %89
  %183 = phi ptr [ @Y_L1L2_5_0_buff_0, %89 ]
  br label %91

184:                                              ; preds = %89
  %185 = phi ptr [ @Y_L1L2_5_0_buff_1, %89 ]
  br label %91

186:                                              ; preds = %111
  %187 = phi ptr [ @A_L2L1_5_0_cons_buff_0, %111 ]
  br label %113

188:                                              ; preds = %111
  %189 = phi ptr [ @A_L2L1_5_0_cons_buff_1, %111 ]
  br label %113

190:                                              ; preds = %116
  %191 = phi ptr [ @Y_L1L2_5_0_buff_0, %116 ]
  br label %118

192:                                              ; preds = %116
  %193 = phi ptr [ @Y_L1L2_5_0_buff_1, %116 ]
  br label %118

194:                                              ; preds = %137
  %195 = phi ptr [ @A_L2L1_5_0_cons_buff_0, %137 ]
  br label %139

196:                                              ; preds = %137
  %197 = phi ptr [ @A_L2L1_5_0_cons_buff_1, %137 ]
  br label %139

198:                                              ; preds = %142
  %199 = phi ptr [ @Y_L1L2_5_0_buff_0, %142 ]
  br label %144

200:                                              ; preds = %142
  %201 = phi ptr [ @Y_L1L2_5_0_buff_1, %142 ]
  br label %144
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

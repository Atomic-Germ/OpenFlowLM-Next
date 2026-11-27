; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@rtp_7_1 = external global [12 x i32]
@par_7_1 = external global [2624 x float]
@rtp_7_0 = external global [12 x i32]
@par_7_0 = external global [2624 x float]
@rtp_6_1 = external global [12 x i32]
@par_6_1 = external global [2624 x float]
@rtp_6_0 = external global [12 x i32]
@par_6_0 = external global [2624 x float]
@rtp_5_1 = external global [12 x i32]
@par_5_1 = external global [2624 x float]
@rtp_5_0 = global [12 x i32] zeroinitializer
@par_5_0 = external global [2624 x float]
@rtp_4_1 = external global [12 x i32]
@par_4_1 = external global [2624 x float]
@rtp_4_0 = external global [12 x i32]
@par_4_0 = external global [2624 x float]
@rtp_3_1 = external global [12 x i32]
@par_3_1 = external global [2624 x float]
@rtp_3_0 = external global [12 x i32]
@par_3_0 = external global [2624 x float]
@rtp_2_1 = external global [12 x i32]
@par_2_1 = external global [2624 x float]
@rtp_2_0 = external global [12 x i32]
@par_2_0 = external global [2624 x float]
@rtp_1_1 = external global [12 x i32]
@par_1_1 = external global [2624 x float]
@rtp_1_0 = external global [12 x i32]
@par_1_0 = external global [2624 x float]
@rtp_0_1 = external global [12 x i32]
@par_0_1 = external global [2624 x float]
@rtp_0_0 = external global [12 x i32]
@par_0_0 = external global [2624 x float]
@A_L3L2_0_cons_buff_1 = external global [8192 x bfloat]
@A_L3L2_0_cons_buff_0 = external global [8192 x bfloat]
@A_L2L1_0_0_cons_buff_1 = external global [4096 x bfloat]
@A_L2L1_0_0_cons_buff_0 = external global [4096 x bfloat]
@A_L3L2_1_cons_buff_1 = external global [8192 x bfloat]
@A_L3L2_1_cons_buff_0 = external global [8192 x bfloat]
@A_L3L2_2_cons_buff_1 = external global [8192 x bfloat]
@A_L3L2_2_cons_buff_0 = external global [8192 x bfloat]
@A_L2L1_0_1_cons_buff_1 = external global [4096 x bfloat]
@A_L2L1_0_1_cons_buff_0 = external global [4096 x bfloat]
@A_L3L2_3_cons_buff_1 = external global [8192 x bfloat]
@A_L3L2_3_cons_buff_0 = external global [8192 x bfloat]
@A_L2L1_1_0_cons_buff_1 = external global [4096 x bfloat]
@A_L2L1_1_0_cons_buff_0 = external global [4096 x bfloat]
@A_L3L2_4_cons_buff_1 = external global [8192 x bfloat]
@A_L3L2_4_cons_buff_0 = external global [8192 x bfloat]
@A_L3L2_5_cons_buff_1 = external global [8192 x bfloat]
@A_L3L2_5_cons_buff_0 = external global [8192 x bfloat]
@A_L2L1_1_1_cons_buff_1 = external global [4096 x bfloat]
@A_L2L1_1_1_cons_buff_0 = external global [4096 x bfloat]
@A_L3L2_6_cons_buff_1 = external global [8192 x bfloat]
@A_L3L2_6_cons_buff_0 = external global [8192 x bfloat]
@A_L2L1_2_0_cons_buff_1 = external global [4096 x bfloat]
@A_L2L1_2_0_cons_buff_0 = external global [4096 x bfloat]
@A_L3L2_7_cons_buff_1 = external global [8192 x bfloat]
@A_L3L2_7_cons_buff_0 = external global [8192 x bfloat]
@B_L3L2_0_cons_buff_1 = external global [8192 x bfloat]
@B_L3L2_0_cons_buff_0 = external global [8192 x bfloat]
@A_L2L1_2_1_cons_buff_1 = external global [4096 x bfloat]
@A_L2L1_2_1_cons_buff_0 = external global [4096 x bfloat]
@B_L3L2_1_cons_buff_1 = external global [8192 x bfloat]
@B_L3L2_1_cons_buff_0 = external global [8192 x bfloat]
@A_L2L1_3_0_cons_buff_1 = external global [4096 x bfloat]
@A_L2L1_3_0_cons_buff_0 = external global [4096 x bfloat]
@B_L3L2_2_cons_buff_1 = external global [8192 x bfloat]
@B_L3L2_2_cons_buff_0 = external global [8192 x bfloat]
@B_L3L2_3_cons_buff_1 = external global [8192 x bfloat]
@B_L3L2_3_cons_buff_0 = external global [8192 x bfloat]
@A_L2L1_3_1_cons_buff_1 = external global [4096 x bfloat]
@A_L2L1_3_1_cons_buff_0 = external global [4096 x bfloat]
@B_L3L2_4_cons_buff_1 = external global [8192 x bfloat]
@B_L3L2_4_cons_buff_0 = external global [8192 x bfloat]
@A_L2L1_4_0_cons_buff_1 = external global [4096 x bfloat]
@A_L2L1_4_0_cons_buff_0 = external global [4096 x bfloat]
@B_L3L2_5_cons_buff_1 = external global [8192 x bfloat]
@B_L3L2_5_cons_buff_0 = external global [8192 x bfloat]
@B_L3L2_6_cons_buff_1 = external global [8192 x bfloat]
@B_L3L2_6_cons_buff_0 = external global [8192 x bfloat]
@A_L2L1_4_1_cons_buff_1 = external global [4096 x bfloat]
@A_L2L1_4_1_cons_buff_0 = external global [4096 x bfloat]
@B_L3L2_7_cons_buff_1 = external global [8192 x bfloat]
@B_L3L2_7_cons_buff_0 = external global [8192 x bfloat]
@A_L2L1_5_0_cons_buff_1 = external global [4096 x bfloat]
@A_L2L1_5_0_cons_buff_0 = external global [4096 x bfloat]
@Y_L2L3_0_buff_1 = external global [8192 x bfloat]
@Y_L2L3_0_buff_0 = external global [8192 x bfloat]
@Y_L2L3_1_buff_1 = external global [8192 x bfloat]
@Y_L2L3_1_buff_0 = external global [8192 x bfloat]
@A_L2L1_5_1_cons_buff_1 = external global [4096 x bfloat]
@A_L2L1_5_1_cons_buff_0 = external global [4096 x bfloat]
@Y_L2L3_2_buff_1 = external global [8192 x bfloat]
@Y_L2L3_2_buff_0 = external global [8192 x bfloat]
@A_L2L1_6_0_cons_buff_1 = external global [4096 x bfloat]
@A_L2L1_6_0_cons_buff_0 = external global [4096 x bfloat]
@Y_L2L3_3_buff_1 = external global [8192 x bfloat]
@Y_L2L3_3_buff_0 = external global [8192 x bfloat]
@Y_L2L3_4_buff_1 = external global [8192 x bfloat]
@Y_L2L3_4_buff_0 = external global [8192 x bfloat]
@A_L2L1_6_1_cons_buff_1 = external global [4096 x bfloat]
@A_L2L1_6_1_cons_buff_0 = external global [4096 x bfloat]
@Y_L2L3_5_buff_1 = external global [8192 x bfloat]
@Y_L2L3_5_buff_0 = external global [8192 x bfloat]
@A_L2L1_7_0_cons_buff_1 = external global [4096 x bfloat]
@A_L2L1_7_0_cons_buff_0 = external global [4096 x bfloat]
@Y_L2L3_6_buff_1 = external global [8192 x bfloat]
@Y_L2L3_6_buff_0 = external global [8192 x bfloat]
@Y_L2L3_7_buff_1 = external global [8192 x bfloat]
@Y_L2L3_7_buff_0 = external global [8192 x bfloat]
@A_L2L1_7_1_cons_buff_1 = external global [4096 x bfloat]
@A_L2L1_7_1_cons_buff_0 = external global [4096 x bfloat]
@Z_L2L3_0_buff_1 = external global [8192 x bfloat]
@Z_L2L3_0_buff_0 = external global [8192 x bfloat]
@B_L2L1_0_0_cons_buff_0 = external global [4096 x bfloat]
@Z_L2L3_1_buff_1 = external global [8192 x bfloat]
@Z_L2L3_1_buff_0 = external global [8192 x bfloat]
@Z_L2L3_2_buff_1 = external global [8192 x bfloat]
@Z_L2L3_2_buff_0 = external global [8192 x bfloat]
@B_L2L1_0_1_cons_buff_0 = external global [4096 x bfloat]
@Z_L2L3_3_buff_1 = external global [8192 x bfloat]
@Z_L2L3_3_buff_0 = external global [8192 x bfloat]
@B_L2L1_1_0_cons_buff_0 = external global [4096 x bfloat]
@Z_L2L3_4_buff_1 = external global [8192 x bfloat]
@Z_L2L3_4_buff_0 = external global [8192 x bfloat]
@Z_L2L3_5_buff_1 = external global [8192 x bfloat]
@Z_L2L3_5_buff_0 = external global [8192 x bfloat]
@B_L2L1_1_1_cons_buff_0 = external global [4096 x bfloat]
@Z_L2L3_6_buff_1 = external global [8192 x bfloat]
@Z_L2L3_6_buff_0 = external global [8192 x bfloat]
@B_L2L1_2_0_cons_buff_0 = external global [4096 x bfloat]
@Z_L2L3_7_buff_1 = external global [8192 x bfloat]
@Z_L2L3_7_buff_0 = external global [8192 x bfloat]
@B_L2L1_2_1_cons_buff_0 = external global [4096 x bfloat]
@B_L2L1_3_0_cons_buff_0 = external global [4096 x bfloat]
@B_L2L1_3_1_cons_buff_0 = external global [4096 x bfloat]
@B_L2L1_4_0_cons_buff_0 = external global [4096 x bfloat]
@B_L2L1_4_1_cons_buff_0 = external global [4096 x bfloat]
@B_L2L1_5_0_cons_buff_0 = external global [4096 x bfloat]
@B_L2L1_5_1_cons_buff_0 = external global [4096 x bfloat]
@B_L2L1_6_0_cons_buff_0 = external global [4096 x bfloat]
@B_L2L1_6_1_cons_buff_0 = external global [4096 x bfloat]
@B_L2L1_7_0_cons_buff_0 = external global [4096 x bfloat]
@B_L2L1_7_1_cons_buff_0 = external global [4096 x bfloat]
@Y_L1L2_0_0_buff_1 = external global [4096 x bfloat]
@Y_L1L2_0_0_buff_0 = external global [4096 x bfloat]
@Y_L1L2_0_1_buff_1 = external global [4096 x bfloat]
@Y_L1L2_0_1_buff_0 = external global [4096 x bfloat]
@Y_L1L2_1_0_buff_1 = external global [4096 x bfloat]
@Y_L1L2_1_0_buff_0 = external global [4096 x bfloat]
@Y_L1L2_1_1_buff_1 = external global [4096 x bfloat]
@Y_L1L2_1_1_buff_0 = external global [4096 x bfloat]
@Y_L1L2_2_0_buff_1 = external global [4096 x bfloat]
@Y_L1L2_2_0_buff_0 = external global [4096 x bfloat]
@Y_L1L2_2_1_buff_1 = external global [4096 x bfloat]
@Y_L1L2_2_1_buff_0 = external global [4096 x bfloat]
@Y_L1L2_3_0_buff_1 = external global [4096 x bfloat]
@Y_L1L2_3_0_buff_0 = external global [4096 x bfloat]
@Y_L1L2_3_1_buff_1 = external global [4096 x bfloat]
@Y_L1L2_3_1_buff_0 = external global [4096 x bfloat]
@Y_L1L2_4_0_buff_1 = external global [4096 x bfloat]
@Y_L1L2_4_0_buff_0 = external global [4096 x bfloat]
@Y_L1L2_4_1_buff_1 = external global [4096 x bfloat]
@Y_L1L2_4_1_buff_0 = external global [4096 x bfloat]
@Y_L1L2_5_0_buff_1 = external global [4096 x bfloat]
@Y_L1L2_5_0_buff_0 = external global [4096 x bfloat]
@Y_L1L2_5_1_buff_1 = external global [4096 x bfloat]
@Y_L1L2_5_1_buff_0 = external global [4096 x bfloat]
@Y_L1L2_6_0_buff_1 = external global [4096 x bfloat]
@Y_L1L2_6_0_buff_0 = external global [4096 x bfloat]
@Y_L1L2_6_1_buff_1 = external global [4096 x bfloat]
@Y_L1L2_6_1_buff_0 = external global [4096 x bfloat]
@Y_L1L2_7_0_buff_1 = external global [4096 x bfloat]
@Y_L1L2_7_0_buff_0 = external global [4096 x bfloat]
@Y_L1L2_7_1_buff_1 = external global [4096 x bfloat]
@Y_L1L2_7_1_buff_0 = external global [4096 x bfloat]
@Z_L1L2_0_0_buff_0 = external global [4096 x bfloat]
@Z_L1L2_0_1_buff_0 = external global [4096 x bfloat]
@Z_L1L2_1_0_buff_0 = external global [4096 x bfloat]
@Z_L1L2_1_1_buff_0 = external global [4096 x bfloat]
@Z_L1L2_2_0_buff_0 = external global [4096 x bfloat]
@Z_L1L2_2_1_buff_0 = external global [4096 x bfloat]
@Z_L1L2_3_0_buff_0 = external global [4096 x bfloat]
@Z_L1L2_3_1_buff_0 = external global [4096 x bfloat]
@Z_L1L2_4_0_buff_0 = external global [4096 x bfloat]
@Z_L1L2_4_1_buff_0 = external global [4096 x bfloat]
@Z_L1L2_5_0_buff_0 = external global [4096 x bfloat]
@Z_L1L2_5_1_buff_0 = external global [4096 x bfloat]
@Z_L1L2_6_0_buff_0 = external global [4096 x bfloat]
@Z_L1L2_6_1_buff_0 = external global [4096 x bfloat]
@Z_L1L2_7_0_buff_0 = external global [4096 x bfloat]
@Z_L1L2_7_1_buff_0 = external global [4096 x bfloat]

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

declare void @vew_begin(ptr, ptr)

declare void @vew_param(ptr, ptr, i32, i32, ptr)

declare void @vew_stats(ptr, ptr, ptr)

declare void @vew_stats_out(ptr, ptr, ptr)

declare void @vew_apply(ptr, ptr, ptr, ptr)

declare void @vew_add(ptr, ptr, ptr, ptr, ptr)

declare void @vew_rgba(ptr, ptr, ptr)

define void @core_5_2() {
  br label %1

1:                                                ; preds = %144, %0
  %2 = phi i64 [ %145, %144 ], [ 0, %0 ]
  %3 = phi i32 [ %123, %144 ], [ 0, %0 ]
  %4 = phi i32 [ %124, %144 ], [ 0, %0 ]
  %5 = icmp slt i64 %2, 9223372036854775807
  br i1 %5, label %6, label %146

6:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 56, i32 1)
  call void @vew_begin(ptr @par_5_0, ptr @rtp_5_0)
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
    i64 0, label %147
    i64 1, label %149
  ]

15:                                               ; preds = %147, %149, %13
  %16 = phi ptr [ %150, %149 ], [ %148, %147 ], [ @A_L2L1_5_0_cons_buff_0, %13 ]
  %17 = getelementptr [4096 x bfloat], ptr %16, i32 0, i32 0
  br label %18

18:                                               ; preds = %15
  %19 = trunc i64 %10 to i32
  call void @vew_param(ptr %17, ptr @par_5_0, i32 %19, i32 -1, ptr @rtp_5_0)
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

27:                                               ; preds = %36, %24
  %28 = phi i64 [ %40, %36 ], [ 0, %24 ]
  %29 = phi i32 [ %39, %36 ], [ %11, %24 ]
  %30 = icmp slt i64 %28, %26
  br i1 %30, label %31, label %41

31:                                               ; preds = %27
  call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %32 = sext i32 %29 to i64
  switch i64 %32, label %33 [
    i64 0, label %151
    i64 1, label %153
  ]

33:                                               ; preds = %151, %153, %31
  %34 = phi ptr [ %154, %153 ], [ %152, %151 ], [ @A_L2L1_5_0_cons_buff_0, %31 ]
  %35 = getelementptr [4096 x bfloat], ptr %34, i32 0, i32 0
  br label %36

36:                                               ; preds = %33
  call void @vew_stats(ptr %35, ptr @par_5_0, ptr @rtp_5_0)
  call void @llvm.aie2p.release(i32 54, i32 1)
  %37 = add i32 %29, 1
  %38 = icmp sge i32 %37, 2
  %39 = select i1 %38, i32 0, i32 %37
  %40 = add i64 %28, 1
  br label %27

41:                                               ; preds = %27
  %42 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_5_0, i64 8)
  %43 = sext i32 %42 to i64
  br label %44

44:                                               ; preds = %53, %41
  %45 = phi i64 [ %57, %53 ], [ 0, %41 ]
  %46 = phi i32 [ %56, %53 ], [ %4, %41 ]
  %47 = icmp slt i64 %45, %43
  br i1 %47, label %48, label %58

48:                                               ; preds = %44
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %49 = sext i32 %46 to i64
  switch i64 %49, label %50 [
    i64 0, label %155
    i64 1, label %157
  ]

50:                                               ; preds = %155, %157, %48
  %51 = phi ptr [ %158, %157 ], [ %156, %155 ], [ @Y_L1L2_5_0_buff_0, %48 ]
  %52 = getelementptr [4096 x bfloat], ptr %51, i32 0, i32 0
  br label %53

53:                                               ; preds = %50
  call void @vew_stats_out(ptr %52, ptr @par_5_0, ptr @rtp_5_0)
  call void @llvm.aie2p.release(i32 51, i32 1)
  %54 = add i32 %46, 1
  %55 = icmp sge i32 %54, 2
  %56 = select i1 %55, i32 0, i32 %54
  %57 = add i64 %45, 1
  br label %44

58:                                               ; preds = %44
  %59 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_5_0, i64 12)
  %60 = sext i32 %59 to i64
  br label %61

61:                                               ; preds = %76, %58
  %62 = phi i64 [ %83, %76 ], [ 0, %58 ]
  %63 = phi i32 [ %79, %76 ], [ %29, %58 ]
  %64 = phi i32 [ %82, %76 ], [ %46, %58 ]
  %65 = icmp slt i64 %62, %60
  br i1 %65, label %66, label %84

66:                                               ; preds = %61
  call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %67 = sext i32 %63 to i64
  switch i64 %67, label %68 [
    i64 0, label %159
    i64 1, label %161
  ]

68:                                               ; preds = %159, %161, %66
  %69 = phi ptr [ %162, %161 ], [ %160, %159 ], [ @A_L2L1_5_0_cons_buff_0, %66 ]
  %70 = getelementptr [4096 x bfloat], ptr %69, i32 0, i32 0
  br label %71

71:                                               ; preds = %68
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %72 = sext i32 %64 to i64
  switch i64 %72, label %73 [
    i64 0, label %163
    i64 1, label %165
  ]

73:                                               ; preds = %163, %165, %71
  %74 = phi ptr [ %166, %165 ], [ %164, %163 ], [ @Y_L1L2_5_0_buff_0, %71 ]
  %75 = getelementptr [4096 x bfloat], ptr %74, i32 0, i32 0
  br label %76

76:                                               ; preds = %73
  call void @vew_apply(ptr %70, ptr %75, ptr @par_5_0, ptr @rtp_5_0)
  call void @llvm.aie2p.release(i32 54, i32 1)
  %77 = add i32 %63, 1
  %78 = icmp sge i32 %77, 2
  %79 = select i1 %78, i32 0, i32 %77
  call void @llvm.aie2p.release(i32 51, i32 1)
  %80 = add i32 %64, 1
  %81 = icmp sge i32 %80, 2
  %82 = select i1 %81, i32 0, i32 %80
  %83 = add i64 %62, 1
  br label %61

84:                                               ; preds = %61
  %85 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_5_0, i64 16)
  %86 = sext i32 %85 to i64
  br label %87

87:                                               ; preds = %102, %84
  %88 = phi i64 [ %109, %102 ], [ 0, %84 ]
  %89 = phi i32 [ %105, %102 ], [ %63, %84 ]
  %90 = phi i32 [ %108, %102 ], [ %64, %84 ]
  %91 = icmp slt i64 %88, %86
  br i1 %91, label %92, label %110

92:                                               ; preds = %87
  call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %93 = sext i32 %89 to i64
  switch i64 %93, label %94 [
    i64 0, label %167
    i64 1, label %169
  ]

94:                                               ; preds = %167, %169, %92
  %95 = phi ptr [ %170, %169 ], [ %168, %167 ], [ @A_L2L1_5_0_cons_buff_0, %92 ]
  %96 = getelementptr [4096 x bfloat], ptr %95, i32 0, i32 0
  br label %97

97:                                               ; preds = %94
  call void @llvm.aie2p.acquire(i32 53, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %98 = sext i32 %90 to i64
  switch i64 %98, label %99 [
    i64 0, label %171
    i64 1, label %173
  ]

99:                                               ; preds = %171, %173, %97
  %100 = phi ptr [ %174, %173 ], [ %172, %171 ], [ @Y_L1L2_5_0_buff_0, %97 ]
  %101 = getelementptr [4096 x bfloat], ptr %100, i32 0, i32 0
  br label %102

102:                                              ; preds = %99
  call void @vew_add(ptr %96, ptr @B_L2L1_5_0_cons_buff_0, ptr %101, ptr @par_5_0, ptr @rtp_5_0)
  call void @llvm.aie2p.release(i32 54, i32 1)
  %103 = add i32 %89, 1
  %104 = icmp sge i32 %103, 2
  %105 = select i1 %104, i32 0, i32 %103
  call void @llvm.aie2p.release(i32 52, i32 1)
  call void @llvm.aie2p.release(i32 51, i32 1)
  %106 = add i32 %90, 1
  %107 = icmp sge i32 %106, 2
  %108 = select i1 %107, i32 0, i32 %106
  %109 = add i64 %88, 1
  br label %87

110:                                              ; preds = %87
  %111 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_5_0, i64 20)
  %112 = sext i32 %111 to i64
  br label %113

113:                                              ; preds = %116, %110
  %114 = phi i64 [ %117, %116 ], [ 0, %110 ]
  %115 = icmp slt i64 %114, %112
  br i1 %115, label %116, label %118

116:                                              ; preds = %113
  call void @llvm.aie2p.acquire(i32 48, i32 -1)
  call void @vew_stats_out(ptr @Z_L1L2_5_0_buff_0, ptr @par_5_0, ptr @rtp_5_0)
  call void @llvm.aie2p.release(i32 49, i32 1)
  %117 = add i64 %114, 1
  br label %113

118:                                              ; preds = %113
  %119 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_5_0, i64 24)
  %120 = sext i32 %119 to i64
  br label %121

121:                                              ; preds = %136, %118
  %122 = phi i64 [ %143, %136 ], [ 0, %118 ]
  %123 = phi i32 [ %139, %136 ], [ %89, %118 ]
  %124 = phi i32 [ %142, %136 ], [ %90, %118 ]
  %125 = icmp slt i64 %122, %120
  br i1 %125, label %126, label %144

126:                                              ; preds = %121
  call void @llvm.aie2p.acquire(i32 55, i32 -1)
  %127 = sext i32 %123 to i64
  switch i64 %127, label %128 [
    i64 0, label %175
    i64 1, label %177
  ]

128:                                              ; preds = %175, %177, %126
  %129 = phi ptr [ %178, %177 ], [ %176, %175 ], [ @A_L2L1_5_0_cons_buff_0, %126 ]
  %130 = getelementptr [4096 x bfloat], ptr %129, i32 0, i32 0
  br label %131

131:                                              ; preds = %128
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  %132 = sext i32 %124 to i64
  switch i64 %132, label %133 [
    i64 0, label %179
    i64 1, label %181
  ]

133:                                              ; preds = %179, %181, %131
  %134 = phi ptr [ %182, %181 ], [ %180, %179 ], [ @Y_L1L2_5_0_buff_0, %131 ]
  %135 = getelementptr [4096 x bfloat], ptr %134, i32 0, i32 0
  br label %136

136:                                              ; preds = %133
  call void @vew_rgba(ptr %130, ptr %135, ptr @rtp_5_0)
  call void @llvm.aie2p.release(i32 54, i32 1)
  %137 = add i32 %123, 1
  %138 = icmp sge i32 %137, 2
  %139 = select i1 %138, i32 0, i32 %137
  call void @llvm.aie2p.release(i32 51, i32 1)
  %140 = add i32 %124, 1
  %141 = icmp sge i32 %140, 2
  %142 = select i1 %141, i32 0, i32 %140
  %143 = add i64 %122, 1
  br label %121

144:                                              ; preds = %121
  call void @llvm.aie2p.release(i32 56, i32 1)
  %145 = add i64 %2, 1
  br label %1

146:                                              ; preds = %1
  ret void

147:                                              ; preds = %13
  %148 = phi ptr [ @A_L2L1_5_0_cons_buff_0, %13 ]
  br label %15

149:                                              ; preds = %13
  %150 = phi ptr [ @A_L2L1_5_0_cons_buff_1, %13 ]
  br label %15

151:                                              ; preds = %31
  %152 = phi ptr [ @A_L2L1_5_0_cons_buff_0, %31 ]
  br label %33

153:                                              ; preds = %31
  %154 = phi ptr [ @A_L2L1_5_0_cons_buff_1, %31 ]
  br label %33

155:                                              ; preds = %48
  %156 = phi ptr [ @Y_L1L2_5_0_buff_0, %48 ]
  br label %50

157:                                              ; preds = %48
  %158 = phi ptr [ @Y_L1L2_5_0_buff_1, %48 ]
  br label %50

159:                                              ; preds = %66
  %160 = phi ptr [ @A_L2L1_5_0_cons_buff_0, %66 ]
  br label %68

161:                                              ; preds = %66
  %162 = phi ptr [ @A_L2L1_5_0_cons_buff_1, %66 ]
  br label %68

163:                                              ; preds = %71
  %164 = phi ptr [ @Y_L1L2_5_0_buff_0, %71 ]
  br label %73

165:                                              ; preds = %71
  %166 = phi ptr [ @Y_L1L2_5_0_buff_1, %71 ]
  br label %73

167:                                              ; preds = %92
  %168 = phi ptr [ @A_L2L1_5_0_cons_buff_0, %92 ]
  br label %94

169:                                              ; preds = %92
  %170 = phi ptr [ @A_L2L1_5_0_cons_buff_1, %92 ]
  br label %94

171:                                              ; preds = %97
  %172 = phi ptr [ @Y_L1L2_5_0_buff_0, %97 ]
  br label %99

173:                                              ; preds = %97
  %174 = phi ptr [ @Y_L1L2_5_0_buff_1, %97 ]
  br label %99

175:                                              ; preds = %126
  %176 = phi ptr [ @A_L2L1_5_0_cons_buff_0, %126 ]
  br label %128

177:                                              ; preds = %126
  %178 = phi ptr [ @A_L2L1_5_0_cons_buff_1, %126 ]
  br label %128

179:                                              ; preds = %131
  %180 = phi ptr [ @Y_L1L2_5_0_buff_0, %131 ]
  br label %133

181:                                              ; preds = %131
  %182 = phi ptr [ @Y_L1L2_5_0_buff_1, %131 ]
  br label %133
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

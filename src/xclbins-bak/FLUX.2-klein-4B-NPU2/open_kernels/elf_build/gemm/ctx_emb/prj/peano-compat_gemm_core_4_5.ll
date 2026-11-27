; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@rtp_3_7 = external global [4 x i32]
@rtp_3_6 = external global [4 x i32]
@rtp_3_5 = external global [4 x i32]
@rtp_3_4 = external global [4 x i32]
@rtp_3_3 = external global [4 x i32]
@rtp_3_2 = external global [4 x i32]
@rtp_3_1 = external global [4 x i32]
@rtp_3_0 = external global [4 x i32]
@rtp_2_7 = external global [4 x i32]
@rtp_2_6 = external global [4 x i32]
@rtp_2_5 = external global [4 x i32]
@rtp_2_4 = external global [4 x i32]
@rtp_2_3 = global [4 x i32] zeroinitializer
@rtp_2_2 = external global [4 x i32]
@rtp_2_1 = external global [4 x i32]
@rtp_2_0 = external global [4 x i32]
@rtp_1_7 = external global [4 x i32]
@rtp_1_6 = external global [4 x i32]
@rtp_1_5 = external global [4 x i32]
@rtp_1_4 = external global [4 x i32]
@rtp_1_3 = external global [4 x i32]
@rtp_1_2 = external global [4 x i32]
@rtp_1_1 = external global [4 x i32]
@rtp_1_0 = external global [4 x i32]
@rtp_0_7 = external global [4 x i32]
@rtp_0_6 = external global [4 x i32]
@rtp_0_5 = external global [4 x i32]
@rtp_0_4 = external global [4 x i32]
@rtp_0_3 = external global [4 x i32]
@rtp_0_2 = external global [4 x i32]
@rtp_0_1 = external global [4 x i32]
@rtp_0_0 = external global [4 x i32]
@A_L3L2_0_cons_buff_1 = external global [128 x [512 x bfloat]]
@A_L3L2_0_cons_buff_0 = external global [128 x [512 x bfloat]]
@A_L2L1_0_7_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_0_7_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_0_6_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_0_6_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_0_5_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_0_5_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_0_4_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_0_4_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_0_3_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_0_3_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_0_2_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_0_2_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_0_1_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_0_1_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_0_0_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_0_0_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L3L2_1_cons_buff_1 = external global [128 x [512 x bfloat]]
@A_L3L2_1_cons_buff_0 = external global [128 x [512 x bfloat]]
@A_L3L2_2_cons_buff_1 = external global [128 x [512 x bfloat]]
@A_L3L2_2_cons_buff_0 = external global [128 x [512 x bfloat]]
@A_L2L1_1_7_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_1_7_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_1_6_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_1_6_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_1_5_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_1_5_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_1_4_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_1_4_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_1_3_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_1_3_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_1_2_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_1_2_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_1_1_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_1_1_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_1_0_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_1_0_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L3L2_3_cons_buff_1 = external global [128 x [512 x bfloat]]
@A_L3L2_3_cons_buff_0 = external global [128 x [512 x bfloat]]
@C_L2L3_0_buff_1 = external global [512 x [128 x bfloat]]
@C_L2L3_0_buff_0 = external global [512 x [128 x bfloat]]
@A_L2L1_2_7_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_2_7_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_2_6_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_2_6_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_2_5_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_2_5_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_2_4_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_2_4_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_2_3_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_2_3_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_2_2_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_2_2_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_2_1_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_2_1_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_2_0_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_2_0_cons_buff_0 = external global [32 x [64 x bfloat]]
@C_L2L3_1_buff_1 = external global [512 x [128 x bfloat]]
@C_L2L3_1_buff_0 = external global [512 x [128 x bfloat]]
@C_L2L3_2_buff_1 = external global [512 x [128 x bfloat]]
@C_L2L3_2_buff_0 = external global [512 x [128 x bfloat]]
@A_L2L1_3_7_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_3_7_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_3_6_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_3_6_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_3_5_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_3_5_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_3_4_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_3_4_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_3_3_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_3_3_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_3_2_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_3_2_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_3_1_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_3_1_cons_buff_0 = external global [32 x [64 x bfloat]]
@A_L2L1_3_0_cons_buff_1 = external global [32 x [64 x bfloat]]
@A_L2L1_3_0_cons_buff_0 = external global [32 x [64 x bfloat]]
@C_L2L3_3_buff_1 = external global [512 x [128 x bfloat]]
@C_L2L3_3_buff_0 = external global [512 x [128 x bfloat]]
@C_L2L3_4_buff_1 = external global [512 x [128 x bfloat]]
@C_L2L3_4_buff_0 = external global [512 x [128 x bfloat]]
@B_L2L1_0_3_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_0_3_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_0_2_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_0_2_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_0_1_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_0_1_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_0_0_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_0_0_cons_buff_0 = external global [64 x [16 x i72]]
@C_L2L3_5_buff_1 = external global [512 x [128 x bfloat]]
@C_L2L3_5_buff_0 = external global [512 x [128 x bfloat]]
@C_L2L3_6_buff_1 = external global [512 x [128 x bfloat]]
@C_L2L3_6_buff_0 = external global [512 x [128 x bfloat]]
@B_L2L1_1_3_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_1_3_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_1_2_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_1_2_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_1_1_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_1_1_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_1_0_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_1_0_cons_buff_0 = external global [64 x [16 x i72]]
@C_L2L3_7_buff_1 = external global [512 x [128 x bfloat]]
@C_L2L3_7_buff_0 = external global [512 x [128 x bfloat]]
@B_L2L1_2_3_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_2_3_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_2_2_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_2_2_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_2_1_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_2_1_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_2_0_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_2_0_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_3_3_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_3_3_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_3_2_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_3_2_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_3_1_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_3_1_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_3_0_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_3_0_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_4_3_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_4_3_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_4_2_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_4_2_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_4_1_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_4_1_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_4_0_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_4_0_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_5_3_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_5_3_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_5_2_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_5_2_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_5_1_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_5_1_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_5_0_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_5_0_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_6_3_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_6_3_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_6_2_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_6_2_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_6_1_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_6_1_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_6_0_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_6_0_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_7_3_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_7_3_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_7_2_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_7_2_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_7_1_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_7_1_cons_buff_0 = external global [64 x [16 x i72]]
@B_L2L1_7_0_cons_buff_1 = external global [64 x [16 x i72]]
@B_L2L1_7_0_cons_buff_0 = external global [64 x [16 x i72]]
@C_L1L2_0_0_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_0_1_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_0_2_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_0_3_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_1_0_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_1_1_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_1_2_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_1_3_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_2_0_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_2_1_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_2_2_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_2_3_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_3_0_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_3_1_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_3_2_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_3_3_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_4_0_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_4_1_buff_0 = external global [128 x [128 x bfloat]]
@B_L3L2_0_cons_buff_1 = external global [64 x [16 x i72]]
@B_L3L2_0_cons_buff_0 = external global [64 x [16 x i72]]
@C_L1L2_4_2_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_4_3_buff_0 = external global [128 x [128 x bfloat]]
@B_L3L2_1_cons_buff_1 = external global [64 x [16 x i72]]
@B_L3L2_1_cons_buff_0 = external global [64 x [16 x i72]]
@C_L1L2_5_0_buff_0 = external global [128 x [128 x bfloat]]
@B_L3L2_2_cons_buff_1 = external global [64 x [16 x i72]]
@B_L3L2_2_cons_buff_0 = external global [64 x [16 x i72]]
@C_L1L2_5_1_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_5_2_buff_0 = external global [128 x [128 x bfloat]]
@B_L3L2_3_cons_buff_1 = external global [64 x [16 x i72]]
@B_L3L2_3_cons_buff_0 = external global [64 x [16 x i72]]
@C_L1L2_5_3_buff_0 = external global [128 x [128 x bfloat]]
@B_L3L2_4_cons_buff_1 = external global [64 x [16 x i72]]
@B_L3L2_4_cons_buff_0 = external global [64 x [16 x i72]]
@C_L1L2_6_0_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_6_1_buff_0 = external global [128 x [128 x bfloat]]
@B_L3L2_5_cons_buff_1 = external global [64 x [16 x i72]]
@B_L3L2_5_cons_buff_0 = external global [64 x [16 x i72]]
@C_L1L2_6_2_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_6_3_buff_0 = external global [128 x [128 x bfloat]]
@B_L3L2_6_cons_buff_1 = external global [64 x [16 x i72]]
@B_L3L2_6_cons_buff_0 = external global [64 x [16 x i72]]
@C_L1L2_7_0_buff_0 = external global [128 x [128 x bfloat]]
@B_L3L2_7_cons_buff_1 = external global [64 x [16 x i72]]
@B_L3L2_7_cons_buff_0 = external global [64 x [16 x i72]]
@C_L1L2_7_1_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_7_2_buff_0 = external global [128 x [128 x bfloat]]
@C_L1L2_7_3_buff_0 = external global [128 x [128 x bfloat]]

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

declare void @dit_zero_c(ptr)

declare void @dit_matmul_quarter(ptr, ptr, ptr)

declare void @dit_swiglu_epi(ptr, ptr, i32)

define void @core_4_5() {
  br label %1

1:                                                ; preds = %32, %0
  %2 = phi i64 [ %33, %32 ], [ 0, %0 ]
  %3 = phi i32 [ %12, %32 ], [ 0, %0 ]
  %4 = icmp slt i64 %2, 9223372036854775807
  br i1 %4, label %5, label %34

5:                                                ; preds = %1
  call void @llvm.aie2p.acquire(i32 54, i32 1)
  %6 = load i32, ptr @rtp_2_3
  %7 = load i32, ptr getelementptr inbounds nuw (i8, ptr @rtp_2_3, i64 4)
  %8 = sext i32 %6 to i64
  %9 = sext i32 %7 to i64
  br label %10

10:                                               ; preds = %29, %5
  %11 = phi i64 [ %31, %29 ], [ 0, %5 ]
  %12 = phi i32 [ %17, %29 ], [ %3, %5 ]
  %13 = icmp slt i64 %11, %8
  br i1 %13, label %14, label %32

14:                                               ; preds = %10
  call void @llvm.aie2p.acquire(i32 48, i32 -1)
  call void @dit_zero_c(ptr @C_L1L2_3_2_buff_0)
  br label %15

15:                                               ; preds = %24, %14
  %16 = phi i64 [ %28, %24 ], [ 0, %14 ]
  %17 = phi i32 [ %27, %24 ], [ %12, %14 ]
  %18 = icmp slt i64 %16, %9
  br i1 %18, label %19, label %29

19:                                               ; preds = %15
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  %20 = sext i32 %17 to i64
  switch i64 %20, label %21 [
    i64 0, label %35
    i64 1, label %37
  ]

21:                                               ; preds = %35, %37, %19
  %22 = phi ptr [ %38, %37 ], [ %36, %35 ], [ @B_L2L1_3_2_cons_buff_0, %19 ]
  %23 = getelementptr [64 x [16 x i72]], ptr %22, i32 0, i32 0, i32 0
  br label %24

24:                                               ; preds = %21
  call void @llvm.aie2p.acquire(i32 53, i32 -1)
  call void @dit_matmul_quarter(ptr @A_L2L1_2_3_cons_buff_0, ptr %23, ptr @C_L1L2_3_2_buff_0)
  call void @llvm.aie2p.release(i32 52, i32 1)
  call void @llvm.aie2p.acquire(i32 53, i32 -1)
  call void @dit_matmul_quarter(ptr @A_L2L1_2_3_cons_buff_1, ptr %23, ptr @C_L1L2_3_2_buff_0)
  call void @llvm.aie2p.release(i32 52, i32 1)
  call void @llvm.aie2p.acquire(i32 53, i32 -1)
  call void @dit_matmul_quarter(ptr @A_L2L1_2_3_cons_buff_0, ptr %23, ptr @C_L1L2_3_2_buff_0)
  call void @llvm.aie2p.release(i32 52, i32 1)
  call void @llvm.aie2p.acquire(i32 53, i32 -1)
  call void @dit_matmul_quarter(ptr @A_L2L1_2_3_cons_buff_1, ptr %23, ptr @C_L1L2_3_2_buff_0)
  call void @llvm.aie2p.release(i32 52, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %25 = add i32 %17, 1
  %26 = icmp sge i32 %25, 2
  %27 = select i1 %26, i32 0, i32 %25
  %28 = add i64 %16, 1
  br label %15

29:                                               ; preds = %15
  %30 = trunc i64 %11 to i32
  call void @dit_swiglu_epi(ptr @C_L1L2_3_2_buff_0, ptr @rtp_2_3, i32 %30)
  call void @llvm.aie2p.release(i32 49, i32 1)
  %31 = add i64 %11, 1
  br label %10

32:                                               ; preds = %10
  call void @llvm.aie2p.release(i32 54, i32 1)
  %33 = add i64 %2, 1
  br label %1

34:                                               ; preds = %1
  ret void

35:                                               ; preds = %19
  %36 = phi ptr [ @B_L2L1_3_2_cons_buff_0, %19 ]
  br label %21

37:                                               ; preds = %19
  %38 = phi ptr [ @B_L2L1_3_2_cons_buff_1, %19 ]
  br label %21
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

module {
  aie.device(npu2) {
    %logical_core = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_0 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_1 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_2 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_3 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_4 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_5 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_6 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_7 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_8 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_9 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_10 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_11 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_12 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_13 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_14 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_15 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_16 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_17 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_18 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_19 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_20 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_21 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_22 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_23 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_24 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_25 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_26 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_27 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_28 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_29 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_30 = aie.logical_tile<CoreTile>(?, ?)
    %logical_mem = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_31 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_32 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_33 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_34 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_35 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_36 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_37 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_38 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_39 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_40 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_41 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_42 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_43 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_44 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_45 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_46 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_47 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_48 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_49 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_50 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_51 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_52 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_53 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_54 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_55 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_56 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_57 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_58 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_59 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_60 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_61 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_62 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_63 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_64 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_65 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_66 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_67 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_68 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @A_L2L1_0(%logical_mem dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%logical_core, %logical_core_0, %logical_core_1, %logical_core_2, %logical_core_3, %logical_core_4, %logical_core_5, %logical_core_6}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>>  
    aie.objectfifo @A_L3L2_0(%logical_shim_noc, {%logical_mem}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@A_L3L2_0] -> [@A_L2L1_0]([] [0])
    aie.objectfifo @A_L2L1_1(%logical_mem_31 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%logical_core_7, %logical_core_8, %logical_core_9, %logical_core_10, %logical_core_11, %logical_core_12, %logical_core_13, %logical_core_14}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>>  
    aie.objectfifo @A_L3L2_1(%logical_shim_noc_34, {%logical_mem_31}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@A_L3L2_1] -> [@A_L2L1_1]([] [0])
    aie.objectfifo @A_L2L1_2(%logical_mem_32 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%logical_core_15, %logical_core_16, %logical_core_17, %logical_core_18, %logical_core_19, %logical_core_20, %logical_core_21, %logical_core_22}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>>  
    aie.objectfifo @A_L3L2_2(%logical_shim_noc_35, {%logical_mem_32}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@A_L3L2_2] -> [@A_L2L1_2]([] [0])
    aie.objectfifo @A_L2L1_3(%logical_mem_33 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%logical_core_23, %logical_core_24, %logical_core_25, %logical_core_26, %logical_core_27, %logical_core_28, %logical_core_29, %logical_core_30}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>>  
    aie.objectfifo @A_L3L2_3(%logical_shim_noc_36, {%logical_mem_33}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@A_L3L2_3] -> [@A_L2L1_3]([] [0])
    aie.objectfifo @B_L2L1_0(%logical_mem_37, {%logical_core, %logical_core_7, %logical_core_15, %logical_core_23}, 2 : i32) : !aie.objectfifo<memref<64x32xbf16>>  
    aie.objectfifo @B_L3L2_0(%logical_shim_noc_45, {%logical_mem_37}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>>  
    aie.objectfifo.link [@B_L3L2_0] -> [@B_L2L1_0]([] [0])
    aie.objectfifo @B_L2L1_1(%logical_mem_38, {%logical_core_0, %logical_core_8, %logical_core_16, %logical_core_24}, 2 : i32) : !aie.objectfifo<memref<64x32xbf16>>  
    aie.objectfifo @B_L3L2_1(%logical_shim_noc_46, {%logical_mem_38}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>>  
    aie.objectfifo.link [@B_L3L2_1] -> [@B_L2L1_1]([] [0])
    aie.objectfifo @B_L2L1_2(%logical_mem_39, {%logical_core_1, %logical_core_9, %logical_core_17, %logical_core_25}, 2 : i32) : !aie.objectfifo<memref<64x32xbf16>>  
    aie.objectfifo @B_L3L2_2(%logical_shim_noc_47, {%logical_mem_39}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>>  
    aie.objectfifo.link [@B_L3L2_2] -> [@B_L2L1_2]([] [0])
    aie.objectfifo @B_L2L1_3(%logical_mem_40, {%logical_core_2, %logical_core_10, %logical_core_18, %logical_core_26}, 2 : i32) : !aie.objectfifo<memref<64x32xbf16>>  
    aie.objectfifo @B_L3L2_3(%logical_shim_noc_48, {%logical_mem_40}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>>  
    aie.objectfifo.link [@B_L3L2_3] -> [@B_L2L1_3]([] [0])
    aie.objectfifo @B_L2L1_4(%logical_mem_41, {%logical_core_3, %logical_core_11, %logical_core_19, %logical_core_27}, 2 : i32) : !aie.objectfifo<memref<64x32xbf16>>  
    aie.objectfifo @B_L3L2_4(%logical_shim_noc_49, {%logical_mem_41}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>>  
    aie.objectfifo.link [@B_L3L2_4] -> [@B_L2L1_4]([] [0])
    aie.objectfifo @B_L2L1_5(%logical_mem_42, {%logical_core_4, %logical_core_12, %logical_core_20, %logical_core_28}, 2 : i32) : !aie.objectfifo<memref<64x32xbf16>>  
    aie.objectfifo @B_L3L2_5(%logical_shim_noc_50, {%logical_mem_42}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>>  
    aie.objectfifo.link [@B_L3L2_5] -> [@B_L2L1_5]([] [0])
    aie.objectfifo @B_L2L1_6(%logical_mem_43, {%logical_core_5, %logical_core_13, %logical_core_21, %logical_core_29}, 2 : i32) : !aie.objectfifo<memref<64x32xbf16>>  
    aie.objectfifo @B_L3L2_6(%logical_shim_noc_51, {%logical_mem_43}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>>  
    aie.objectfifo.link [@B_L3L2_6] -> [@B_L2L1_6]([] [0])
    aie.objectfifo @B_L2L1_7(%logical_mem_44, {%logical_core_6, %logical_core_14, %logical_core_22, %logical_core_30}, 2 : i32) : !aie.objectfifo<memref<64x32xbf16>>  
    aie.objectfifo @B_L3L2_7(%logical_shim_noc_52, {%logical_mem_44}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>>  
    aie.objectfifo.link [@B_L3L2_7] -> [@B_L2L1_7]([] [0])
    aie.objectfifo @C_L1L2_0_0(%logical_core, {%logical_mem_53}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_0_1(%logical_core_7, {%logical_mem_53}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_0_2(%logical_core_15, {%logical_mem_53}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_0_3(%logical_core_23, {%logical_mem_53}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L2L3_0(%logical_mem_53 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_61}, 2 : i32) : !aie.objectfifo<memref<8192xf32>>  
    aie.objectfifo.link [@C_L1L2_0_0, @C_L1L2_0_1, @C_L1L2_0_2, @C_L1L2_0_3] -> [@C_L2L3_0]([0, 2048, 4096, 6144] [])
    aie.objectfifo @C_L1L2_1_0(%logical_core_0, {%logical_mem_54}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_1_1(%logical_core_8, {%logical_mem_54}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_1_2(%logical_core_16, {%logical_mem_54}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_1_3(%logical_core_24, {%logical_mem_54}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L2L3_1(%logical_mem_54 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_62}, 2 : i32) : !aie.objectfifo<memref<8192xf32>>  
    aie.objectfifo.link [@C_L1L2_1_0, @C_L1L2_1_1, @C_L1L2_1_2, @C_L1L2_1_3] -> [@C_L2L3_1]([0, 2048, 4096, 6144] [])
    aie.objectfifo @C_L1L2_2_0(%logical_core_1, {%logical_mem_55}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_2_1(%logical_core_9, {%logical_mem_55}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_2_2(%logical_core_17, {%logical_mem_55}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_2_3(%logical_core_25, {%logical_mem_55}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L2L3_2(%logical_mem_55 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_63}, 2 : i32) : !aie.objectfifo<memref<8192xf32>>  
    aie.objectfifo.link [@C_L1L2_2_0, @C_L1L2_2_1, @C_L1L2_2_2, @C_L1L2_2_3] -> [@C_L2L3_2]([0, 2048, 4096, 6144] [])
    aie.objectfifo @C_L1L2_3_0(%logical_core_2, {%logical_mem_56}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_3_1(%logical_core_10, {%logical_mem_56}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_3_2(%logical_core_18, {%logical_mem_56}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_3_3(%logical_core_26, {%logical_mem_56}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L2L3_3(%logical_mem_56 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_64}, 2 : i32) : !aie.objectfifo<memref<8192xf32>>  
    aie.objectfifo.link [@C_L1L2_3_0, @C_L1L2_3_1, @C_L1L2_3_2, @C_L1L2_3_3] -> [@C_L2L3_3]([0, 2048, 4096, 6144] [])
    aie.objectfifo @C_L1L2_4_0(%logical_core_3, {%logical_mem_57}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_4_1(%logical_core_11, {%logical_mem_57}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_4_2(%logical_core_19, {%logical_mem_57}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_4_3(%logical_core_27, {%logical_mem_57}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L2L3_4(%logical_mem_57 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_65}, 2 : i32) : !aie.objectfifo<memref<8192xf32>>  
    aie.objectfifo.link [@C_L1L2_4_0, @C_L1L2_4_1, @C_L1L2_4_2, @C_L1L2_4_3] -> [@C_L2L3_4]([0, 2048, 4096, 6144] [])
    aie.objectfifo @C_L1L2_5_0(%logical_core_4, {%logical_mem_58}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_5_1(%logical_core_12, {%logical_mem_58}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_5_2(%logical_core_20, {%logical_mem_58}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_5_3(%logical_core_28, {%logical_mem_58}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L2L3_5(%logical_mem_58 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_66}, 2 : i32) : !aie.objectfifo<memref<8192xf32>>  
    aie.objectfifo.link [@C_L1L2_5_0, @C_L1L2_5_1, @C_L1L2_5_2, @C_L1L2_5_3] -> [@C_L2L3_5]([0, 2048, 4096, 6144] [])
    aie.objectfifo @C_L1L2_6_0(%logical_core_5, {%logical_mem_59}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_6_1(%logical_core_13, {%logical_mem_59}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_6_2(%logical_core_21, {%logical_mem_59}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_6_3(%logical_core_29, {%logical_mem_59}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L2L3_6(%logical_mem_59 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_67}, 2 : i32) : !aie.objectfifo<memref<8192xf32>>  
    aie.objectfifo.link [@C_L1L2_6_0, @C_L1L2_6_1, @C_L1L2_6_2, @C_L1L2_6_3] -> [@C_L2L3_6]([0, 2048, 4096, 6144] [])
    aie.objectfifo @C_L1L2_7_0(%logical_core_6, {%logical_mem_60}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_7_1(%logical_core_14, {%logical_mem_60}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_7_2(%logical_core_22, {%logical_mem_60}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L1L2_7_3(%logical_core_30, {%logical_mem_60}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>>  
    aie.objectfifo @C_L2L3_7(%logical_mem_60 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_68}, 2 : i32) : !aie.objectfifo<memref<8192xf32>>  
    aie.objectfifo.link [@C_L1L2_7_0, @C_L1L2_7_1, @C_L1L2_7_2, @C_L1L2_7_3] -> [@C_L2L3_7]([0, 2048, 4096, 6144] [])
    func.func private @zero_f32(memref<2048xf32>) attributes {link_with = "matmul_bf16_f32_be894551.o"}
    func.func private @be894551_matmul_bf16_f32(memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) attributes {link_with = "matmul_bf16_f32_be894551.o"}
    %rtp_0_0 = aie.buffer(%logical_core) {sym_name = "rtp_0_0"} : memref<2xi32> = dense<0>
    %rtp_0_1 = aie.buffer(%logical_core_0) {sym_name = "rtp_0_1"} : memref<2xi32> = dense<0>
    %rtp_0_2 = aie.buffer(%logical_core_1) {sym_name = "rtp_0_2"} : memref<2xi32> = dense<0>
    %rtp_0_3 = aie.buffer(%logical_core_2) {sym_name = "rtp_0_3"} : memref<2xi32> = dense<0>
    %rtp_0_4 = aie.buffer(%logical_core_3) {sym_name = "rtp_0_4"} : memref<2xi32> = dense<0>
    %rtp_0_5 = aie.buffer(%logical_core_4) {sym_name = "rtp_0_5"} : memref<2xi32> = dense<0>
    %rtp_0_6 = aie.buffer(%logical_core_5) {sym_name = "rtp_0_6"} : memref<2xi32> = dense<0>
    %rtp_0_7 = aie.buffer(%logical_core_6) {sym_name = "rtp_0_7"} : memref<2xi32> = dense<0>
    %rtp_1_0 = aie.buffer(%logical_core_7) {sym_name = "rtp_1_0"} : memref<2xi32> = dense<0>
    %rtp_1_1 = aie.buffer(%logical_core_8) {sym_name = "rtp_1_1"} : memref<2xi32> = dense<0>
    %rtp_1_2 = aie.buffer(%logical_core_9) {sym_name = "rtp_1_2"} : memref<2xi32> = dense<0>
    %rtp_1_3 = aie.buffer(%logical_core_10) {sym_name = "rtp_1_3"} : memref<2xi32> = dense<0>
    %rtp_1_4 = aie.buffer(%logical_core_11) {sym_name = "rtp_1_4"} : memref<2xi32> = dense<0>
    %rtp_1_5 = aie.buffer(%logical_core_12) {sym_name = "rtp_1_5"} : memref<2xi32> = dense<0>
    %rtp_1_6 = aie.buffer(%logical_core_13) {sym_name = "rtp_1_6"} : memref<2xi32> = dense<0>
    %rtp_1_7 = aie.buffer(%logical_core_14) {sym_name = "rtp_1_7"} : memref<2xi32> = dense<0>
    %rtp_2_0 = aie.buffer(%logical_core_15) {sym_name = "rtp_2_0"} : memref<2xi32> = dense<0>
    %rtp_2_1 = aie.buffer(%logical_core_16) {sym_name = "rtp_2_1"} : memref<2xi32> = dense<0>
    %rtp_2_2 = aie.buffer(%logical_core_17) {sym_name = "rtp_2_2"} : memref<2xi32> = dense<0>
    %rtp_2_3 = aie.buffer(%logical_core_18) {sym_name = "rtp_2_3"} : memref<2xi32> = dense<0>
    %rtp_2_4 = aie.buffer(%logical_core_19) {sym_name = "rtp_2_4"} : memref<2xi32> = dense<0>
    %rtp_2_5 = aie.buffer(%logical_core_20) {sym_name = "rtp_2_5"} : memref<2xi32> = dense<0>
    %rtp_2_6 = aie.buffer(%logical_core_21) {sym_name = "rtp_2_6"} : memref<2xi32> = dense<0>
    %rtp_2_7 = aie.buffer(%logical_core_22) {sym_name = "rtp_2_7"} : memref<2xi32> = dense<0>
    %rtp_3_0 = aie.buffer(%logical_core_23) {sym_name = "rtp_3_0"} : memref<2xi32> = dense<0>
    %rtp_3_1 = aie.buffer(%logical_core_24) {sym_name = "rtp_3_1"} : memref<2xi32> = dense<0>
    %rtp_3_2 = aie.buffer(%logical_core_25) {sym_name = "rtp_3_2"} : memref<2xi32> = dense<0>
    %rtp_3_3 = aie.buffer(%logical_core_26) {sym_name = "rtp_3_3"} : memref<2xi32> = dense<0>
    %rtp_3_4 = aie.buffer(%logical_core_27) {sym_name = "rtp_3_4"} : memref<2xi32> = dense<0>
    %rtp_3_5 = aie.buffer(%logical_core_28) {sym_name = "rtp_3_5"} : memref<2xi32> = dense<0>
    %rtp_3_6 = aie.buffer(%logical_core_29) {sym_name = "rtp_3_6"} : memref<2xi32> = dense<0>
    %rtp_3_7 = aie.buffer(%logical_core_30) {sym_name = "rtp_3_7"} : memref<2xi32> = dense<0>
    %0 = aie.lock(%logical_core)
    %1 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%0, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_0_0[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_0_0[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%0, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %2 = aie.lock(%logical_core_0)
    %3 = aie.core(%logical_core_0) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%2, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_0_1[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_0_1[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%2, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %4 = aie.lock(%logical_core_1)
    %5 = aie.core(%logical_core_1) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%4, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_0_2[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_0_2[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%4, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %6 = aie.lock(%logical_core_2)
    %7 = aie.core(%logical_core_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%6, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_0_3[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_0_3[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%6, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %8 = aie.lock(%logical_core_3)
    %9 = aie.core(%logical_core_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%8, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_0_4[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_0_4[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_4(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%8, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %10 = aie.lock(%logical_core_4)
    %11 = aie.core(%logical_core_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%10, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_0_5[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_0_5[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_5(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%10, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %12 = aie.lock(%logical_core_5)
    %13 = aie.core(%logical_core_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%12, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_0_6[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_0_6[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_6(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%12, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %14 = aie.lock(%logical_core_6)
    %15 = aie.core(%logical_core_6) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%14, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_0_7[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_0_7[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_7(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%14, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %16 = aie.lock(%logical_core_7)
    %17 = aie.core(%logical_core_7) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%16, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_1_0[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_1_0[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%16, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %18 = aie.lock(%logical_core_8)
    %19 = aie.core(%logical_core_8) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%18, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_1_1[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_1_1[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%18, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %20 = aie.lock(%logical_core_9)
    %21 = aie.core(%logical_core_9) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%20, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_1_2[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_1_2[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_1(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%20, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %22 = aie.lock(%logical_core_10)
    %23 = aie.core(%logical_core_10) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%22, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_1_3[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_1_3[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_1(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%22, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %24 = aie.lock(%logical_core_11)
    %25 = aie.core(%logical_core_11) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%24, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_1_4[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_1_4[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_4(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_1(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%24, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %26 = aie.lock(%logical_core_12)
    %27 = aie.core(%logical_core_12) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%26, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_1_5[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_1_5[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_5(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_1(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%26, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %28 = aie.lock(%logical_core_13)
    %29 = aie.core(%logical_core_13) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%28, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_1_6[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_1_6[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_6(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_1(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%28, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %30 = aie.lock(%logical_core_14)
    %31 = aie.core(%logical_core_14) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%30, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_1_7[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_1_7[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_7(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_1(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%30, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %32 = aie.lock(%logical_core_15)
    %33 = aie.core(%logical_core_15) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%32, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_2_0[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_2_0[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%32, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %34 = aie.lock(%logical_core_16)
    %35 = aie.core(%logical_core_16) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%34, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_2_1[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_2_1[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%34, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %36 = aie.lock(%logical_core_17)
    %37 = aie.core(%logical_core_17) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%36, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_2_2[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_2_2[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_2(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%36, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %38 = aie.lock(%logical_core_18)
    %39 = aie.core(%logical_core_18) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%38, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_2_3[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_2_3[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_2(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%38, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %40 = aie.lock(%logical_core_19)
    %41 = aie.core(%logical_core_19) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%40, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_2_4[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_2_4[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_4(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_2(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%40, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %42 = aie.lock(%logical_core_20)
    %43 = aie.core(%logical_core_20) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%42, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_2_5[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_2_5[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_5(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_2(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%42, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %44 = aie.lock(%logical_core_21)
    %45 = aie.core(%logical_core_21) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%44, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_2_6[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_2_6[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_6(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_2(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%44, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %46 = aie.lock(%logical_core_22)
    %47 = aie.core(%logical_core_22) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%46, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_2_7[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_2_7[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_7(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_2(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%46, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %48 = aie.lock(%logical_core_23)
    %49 = aie.core(%logical_core_23) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%48, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_3_0[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_3_0[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%48, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %50 = aie.lock(%logical_core_24)
    %51 = aie.core(%logical_core_24) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%50, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_3_1[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_3_1[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%50, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %52 = aie.lock(%logical_core_25)
    %53 = aie.core(%logical_core_25) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%52, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_3_2[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_3_2[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_3(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%52, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %54 = aie.lock(%logical_core_26)
    %55 = aie.core(%logical_core_26) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%54, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_3_3[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_3_3[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_3(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%54, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %56 = aie.lock(%logical_core_27)
    %57 = aie.core(%logical_core_27) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%56, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_3_4[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_3_4[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_4(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_3(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%56, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %58 = aie.lock(%logical_core_28)
    %59 = aie.core(%logical_core_28) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%58, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_3_5[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_3_5[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_5(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_3(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%58, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %60 = aie.lock(%logical_core_29)
    %61 = aie.core(%logical_core_29) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%60, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_3_6[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_3_6[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_6(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_3(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%60, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %62 = aie.lock(%logical_core_30)
    %63 = aie.core(%logical_core_30) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%62, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_3_7[%c0_69] : memref<2xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_3_7[%c1_70] : memref<2xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          %collapse_shape = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
          func.call @zero_f32(%collapse_shape) : (memref<2048xf32>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %70 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %71 = aie.objectfifo.subview.access %70[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %72 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %73 = aie.objectfifo.subview.access %72[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            %collapse_shape_76 = memref.collapse_shape %71 [[0, 1]] : memref<64x64xbf16> into memref<4096xbf16>
            %collapse_shape_77 = memref.collapse_shape %73 [[0, 1]] : memref<64x32xbf16> into memref<2048xbf16>
            %collapse_shape_78 = memref.collapse_shape %68 [[0, 1]] : memref<64x32xf32> into memref<2048xf32>
            func.call @be894551_matmul_bf16_f32(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78) : (memref<4096xbf16>, memref<2048xbf16>, memref<2048xf32>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_7(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_3(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%62, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3328 : i32}
    aie.runtime_sequence(%arg0: memref<1966080xbf16>, %arg1: memref<4915200xbf16>, %arg2: memref<5898240xf32>) {
      %c90_i32 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_0_0, 0, %c90_i32) : i32
      %c20_i32 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_0_0, 1, %c20_i32) : i32
      %c90_i32_69 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_0_1, 0, %c90_i32_69) : i32
      %c20_i32_70 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_0_1, 1, %c20_i32_70) : i32
      %c90_i32_71 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_0_2, 0, %c90_i32_71) : i32
      %c20_i32_72 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_0_2, 1, %c20_i32_72) : i32
      %c90_i32_73 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_0_3, 0, %c90_i32_73) : i32
      %c20_i32_74 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_0_3, 1, %c20_i32_74) : i32
      %c90_i32_75 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_0_4, 0, %c90_i32_75) : i32
      %c20_i32_76 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_0_4, 1, %c20_i32_76) : i32
      %c90_i32_77 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_0_5, 0, %c90_i32_77) : i32
      %c20_i32_78 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_0_5, 1, %c20_i32_78) : i32
      %c90_i32_79 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_0_6, 0, %c90_i32_79) : i32
      %c20_i32_80 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_0_6, 1, %c20_i32_80) : i32
      %c90_i32_81 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_0_7, 0, %c90_i32_81) : i32
      %c20_i32_82 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_0_7, 1, %c20_i32_82) : i32
      %c90_i32_83 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_1_0, 0, %c90_i32_83) : i32
      %c20_i32_84 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_1_0, 1, %c20_i32_84) : i32
      %c90_i32_85 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_1_1, 0, %c90_i32_85) : i32
      %c20_i32_86 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_1_1, 1, %c20_i32_86) : i32
      %c90_i32_87 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_1_2, 0, %c90_i32_87) : i32
      %c20_i32_88 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_1_2, 1, %c20_i32_88) : i32
      %c90_i32_89 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_1_3, 0, %c90_i32_89) : i32
      %c20_i32_90 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_1_3, 1, %c20_i32_90) : i32
      %c90_i32_91 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_1_4, 0, %c90_i32_91) : i32
      %c20_i32_92 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_1_4, 1, %c20_i32_92) : i32
      %c90_i32_93 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_1_5, 0, %c90_i32_93) : i32
      %c20_i32_94 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_1_5, 1, %c20_i32_94) : i32
      %c90_i32_95 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_1_6, 0, %c90_i32_95) : i32
      %c20_i32_96 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_1_6, 1, %c20_i32_96) : i32
      %c90_i32_97 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_1_7, 0, %c90_i32_97) : i32
      %c20_i32_98 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_1_7, 1, %c20_i32_98) : i32
      %c90_i32_99 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_2_0, 0, %c90_i32_99) : i32
      %c20_i32_100 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_2_0, 1, %c20_i32_100) : i32
      %c90_i32_101 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_2_1, 0, %c90_i32_101) : i32
      %c20_i32_102 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_2_1, 1, %c20_i32_102) : i32
      %c90_i32_103 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_2_2, 0, %c90_i32_103) : i32
      %c20_i32_104 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_2_2, 1, %c20_i32_104) : i32
      %c90_i32_105 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_2_3, 0, %c90_i32_105) : i32
      %c20_i32_106 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_2_3, 1, %c20_i32_106) : i32
      %c90_i32_107 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_2_4, 0, %c90_i32_107) : i32
      %c20_i32_108 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_2_4, 1, %c20_i32_108) : i32
      %c90_i32_109 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_2_5, 0, %c90_i32_109) : i32
      %c20_i32_110 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_2_5, 1, %c20_i32_110) : i32
      %c90_i32_111 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_2_6, 0, %c90_i32_111) : i32
      %c20_i32_112 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_2_6, 1, %c20_i32_112) : i32
      %c90_i32_113 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_2_7, 0, %c90_i32_113) : i32
      %c20_i32_114 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_2_7, 1, %c20_i32_114) : i32
      %c90_i32_115 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_3_0, 0, %c90_i32_115) : i32
      %c20_i32_116 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_3_0, 1, %c20_i32_116) : i32
      %c90_i32_117 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_3_1, 0, %c90_i32_117) : i32
      %c20_i32_118 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_3_1, 1, %c20_i32_118) : i32
      %c90_i32_119 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_3_2, 0, %c90_i32_119) : i32
      %c20_i32_120 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_3_2, 1, %c20_i32_120) : i32
      %c90_i32_121 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_3_3, 0, %c90_i32_121) : i32
      %c20_i32_122 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_3_3, 1, %c20_i32_122) : i32
      %c90_i32_123 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_3_4, 0, %c90_i32_123) : i32
      %c20_i32_124 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_3_4, 1, %c20_i32_124) : i32
      %c90_i32_125 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_3_5, 0, %c90_i32_125) : i32
      %c20_i32_126 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_3_5, 1, %c20_i32_126) : i32
      %c90_i32_127 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_3_6, 0, %c90_i32_127) : i32
      %c20_i32_128 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_3_6, 1, %c20_i32_128) : i32
      %c90_i32_129 = arith.constant 90 : i32
      aiex.npu.rtp_write(@rtp_3_7, 0, %c90_i32_129) : i32
      %c20_i32_130 = arith.constant 20 : i32
      aiex.npu.rtp_write(@rtp_3_7, 1, %c20_i32_130) : i32
      aiex.set_lock(%0, 1)
      aiex.set_lock(%2, 1)
      aiex.set_lock(%4, 1)
      aiex.set_lock(%6, 1)
      aiex.set_lock(%8, 1)
      aiex.set_lock(%10, 1)
      aiex.set_lock(%12, 1)
      aiex.set_lock(%14, 1)
      aiex.set_lock(%16, 1)
      aiex.set_lock(%18, 1)
      aiex.set_lock(%20, 1)
      aiex.set_lock(%22, 1)
      aiex.set_lock(%24, 1)
      aiex.set_lock(%26, 1)
      aiex.set_lock(%28, 1)
      aiex.set_lock(%30, 1)
      aiex.set_lock(%32, 1)
      aiex.set_lock(%34, 1)
      aiex.set_lock(%36, 1)
      aiex.set_lock(%38, 1)
      aiex.set_lock(%40, 1)
      aiex.set_lock(%42, 1)
      aiex.set_lock(%44, 1)
      aiex.set_lock(%46, 1)
      aiex.set_lock(%48, 1)
      aiex.set_lock(%50, 1)
      aiex.set_lock(%52, 1)
      aiex.set_lock(%54, 1)
      aiex.set_lock(%56, 1)
      aiex.set_lock(%58, 1)
      aiex.set_lock(%60, 1)
      aiex.set_lock(%62, 1)
      %64 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 0 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 0 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 0 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 327680 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%67)
      %68 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 0 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 32 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 81920 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 2048 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 409600 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 2048 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 64 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 163840 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 4096 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 491520 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 4096 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 96 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 245760 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 6144 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 573440 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 6144 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 128 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%84)
      %85 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 8192 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 8192 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 160 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 10240 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 10240 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 192 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 12288 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 12288 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 224 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 14336 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 14336 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%95)
      %96 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 1966080 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 655360 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%97)
      %98 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 0 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 983040 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 0 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%100)
      %101 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 1966112 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%101)
      %102 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 737280 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%102)
      %103 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 2048 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%103)
      %104 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 1064960 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%104)
      %105 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 2048 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%105)
      %106 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 1966144 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%106)
      %107 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 819200 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%107)
      %108 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 4096 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%108)
      %109 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 1146880 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%109)
      %110 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 4096 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%110)
      %111 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 1966176 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%111)
      %112 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 901120 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%112)
      %113 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 6144 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%113)
      %114 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 1228800 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%114)
      %115 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 6144 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%115)
      %116 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 1966208 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%116)
      %117 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 8192 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%117)
      %118 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 8192 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%118)
      %119 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 1966240 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%119)
      %120 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 10240 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%120)
      %121 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 10240 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%121)
      %122 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 1966272 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%122)
      %123 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 12288 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%123)
      %124 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 12288 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%124)
      %125 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 1966304 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%125)
      %126 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 14336 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%126)
      %127 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 14336 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%127)
      aiex.dma_await_task(%64)
      aiex.dma_await_task(%69)
      aiex.dma_await_task(%74)
      aiex.dma_await_task(%79)
      aiex.dma_await_task(%84)
      aiex.dma_await_task(%87)
      aiex.dma_await_task(%90)
      aiex.dma_await_task(%93)
      aiex.dma_free_task(%64)
      aiex.dma_free_task(%65)
      aiex.dma_free_task(%66)
      aiex.dma_free_task(%67)
      aiex.dma_free_task(%68)
      aiex.dma_free_task(%69)
      aiex.dma_free_task(%70)
      aiex.dma_free_task(%71)
      aiex.dma_free_task(%72)
      aiex.dma_free_task(%73)
      aiex.dma_free_task(%74)
      aiex.dma_free_task(%75)
      aiex.dma_free_task(%76)
      aiex.dma_free_task(%77)
      aiex.dma_free_task(%78)
      aiex.dma_free_task(%79)
      aiex.dma_free_task(%80)
      aiex.dma_free_task(%81)
      aiex.dma_free_task(%82)
      aiex.dma_free_task(%83)
      aiex.dma_free_task(%84)
      aiex.dma_free_task(%85)
      aiex.dma_free_task(%86)
      aiex.dma_free_task(%87)
      aiex.dma_free_task(%88)
      aiex.dma_free_task(%89)
      aiex.dma_free_task(%90)
      aiex.dma_free_task(%91)
      aiex.dma_free_task(%92)
      aiex.dma_free_task(%93)
      aiex.dma_free_task(%94)
      aiex.dma_free_task(%95)
      %128 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 3932160 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%128)
      %129 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 1310720 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%129)
      %130 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 0 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%130)
      %131 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 1638400 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%131)
      %132 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 0 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%132)
      %133 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 3932192 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%133)
      %134 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 1392640 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%134)
      %135 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 2048 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%135)
      %136 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 1720320 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%136)
      %137 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 2048 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%137)
      %138 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 3932224 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%138)
      %139 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 1474560 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%139)
      %140 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 4096 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%140)
      %141 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 1802240 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%141)
      %142 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 4096 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%142)
      %143 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 3932256 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%143)
      %144 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 1556480 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%144)
      %145 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 6144 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%145)
      %146 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<1966080xbf16> offset = 1884160 len = 81920 sizes = [15, 20, 64, 64] strides = [0, 64, 1280, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%146)
      %147 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 6144 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%147)
      %148 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 3932288 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%148)
      %149 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 8192 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%149)
      %150 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 8192 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%150)
      %151 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 3932320 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%151)
      %152 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 10240 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%152)
      %153 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 10240 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%153)
      %154 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 3932352 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%154)
      %155 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 12288 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%155)
      %156 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 12288 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%156)
      %157 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<5898240xf32> offset = 3932384 len = 122880 sizes = [2, 15, 256, 32] strides = [983040, 256, 3840, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%157)
      %158 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 14336 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%158)
      %159 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4915200xbf16> offset = 14336 len = 40960 sizes = [15, 20, 64, 32] strides = [16384, 245760, 32, 1])
        aie.end
      } {repeat_count = 14 : i32}
      aiex.dma_start_task(%159)
      aiex.dma_await_task(%96)
      aiex.dma_await_task(%101)
      aiex.dma_await_task(%106)
      aiex.dma_await_task(%111)
      aiex.dma_await_task(%116)
      aiex.dma_await_task(%119)
      aiex.dma_await_task(%122)
      aiex.dma_await_task(%125)
      aiex.dma_free_task(%96)
      aiex.dma_free_task(%97)
      aiex.dma_free_task(%98)
      aiex.dma_free_task(%99)
      aiex.dma_free_task(%100)
      aiex.dma_free_task(%101)
      aiex.dma_free_task(%102)
      aiex.dma_free_task(%103)
      aiex.dma_free_task(%104)
      aiex.dma_free_task(%105)
      aiex.dma_free_task(%106)
      aiex.dma_free_task(%107)
      aiex.dma_free_task(%108)
      aiex.dma_free_task(%109)
      aiex.dma_free_task(%110)
      aiex.dma_free_task(%111)
      aiex.dma_free_task(%112)
      aiex.dma_free_task(%113)
      aiex.dma_free_task(%114)
      aiex.dma_free_task(%115)
      aiex.dma_free_task(%116)
      aiex.dma_free_task(%117)
      aiex.dma_free_task(%118)
      aiex.dma_free_task(%119)
      aiex.dma_free_task(%120)
      aiex.dma_free_task(%121)
      aiex.dma_free_task(%122)
      aiex.dma_free_task(%123)
      aiex.dma_free_task(%124)
      aiex.dma_free_task(%125)
      aiex.dma_free_task(%126)
      aiex.dma_free_task(%127)
      aiex.dma_await_task(%128)
      aiex.dma_await_task(%133)
      aiex.dma_await_task(%138)
      aiex.dma_await_task(%143)
      aiex.dma_await_task(%148)
      aiex.dma_await_task(%151)
      aiex.dma_await_task(%154)
      aiex.dma_await_task(%157)
      aiex.dma_free_task(%128)
      aiex.dma_free_task(%129)
      aiex.dma_free_task(%130)
      aiex.dma_free_task(%131)
      aiex.dma_free_task(%132)
      aiex.dma_free_task(%133)
      aiex.dma_free_task(%134)
      aiex.dma_free_task(%135)
      aiex.dma_free_task(%136)
      aiex.dma_free_task(%137)
      aiex.dma_free_task(%138)
      aiex.dma_free_task(%139)
      aiex.dma_free_task(%140)
      aiex.dma_free_task(%141)
      aiex.dma_free_task(%142)
      aiex.dma_free_task(%143)
      aiex.dma_free_task(%144)
      aiex.dma_free_task(%145)
      aiex.dma_free_task(%146)
      aiex.dma_free_task(%147)
      aiex.dma_free_task(%148)
      aiex.dma_free_task(%149)
      aiex.dma_free_task(%150)
      aiex.dma_free_task(%151)
      aiex.dma_free_task(%152)
      aiex.dma_free_task(%153)
      aiex.dma_free_task(%154)
      aiex.dma_free_task(%155)
      aiex.dma_free_task(%156)
      aiex.dma_free_task(%157)
      aiex.dma_free_task(%158)
      aiex.dma_free_task(%159)
    }
  }
}

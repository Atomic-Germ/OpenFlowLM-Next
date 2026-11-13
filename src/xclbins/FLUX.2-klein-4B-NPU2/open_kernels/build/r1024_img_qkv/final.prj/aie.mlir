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
    aie.objectfifo @A_L2L1_0(%logical_mem dimensionsToStream [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>], {%logical_core dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_0 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_1 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_2 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_3 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_4 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_5 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_6 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>>  
    aie.objectfifo @A_L3L2_0(%logical_shim_noc, {%logical_mem dimensionsFromStream [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<128x512xbf16>>  
    aie.objectfifo.link [@A_L3L2_0] -> [@A_L2L1_0]([] [0])
    aie.objectfifo @A_L2L1_1(%logical_mem_31 dimensionsToStream [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>], {%logical_core_7 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_8 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_9 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_10 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_11 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_12 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_13 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_14 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>>  
    aie.objectfifo @A_L3L2_1(%logical_shim_noc_34, {%logical_mem_31 dimensionsFromStream [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<128x512xbf16>>  
    aie.objectfifo.link [@A_L3L2_1] -> [@A_L2L1_1]([] [0])
    aie.objectfifo @A_L2L1_2(%logical_mem_32 dimensionsToStream [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>], {%logical_core_15 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_16 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_17 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_18 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_19 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_20 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_21 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_22 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>>  
    aie.objectfifo @A_L3L2_2(%logical_shim_noc_35, {%logical_mem_32 dimensionsFromStream [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<128x512xbf16>>  
    aie.objectfifo.link [@A_L3L2_2] -> [@A_L2L1_2]([] [0])
    aie.objectfifo @A_L2L1_3(%logical_mem_33 dimensionsToStream [<size = 32, stride = 2048>, <size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>], {%logical_core_23 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_24 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_25 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_26 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_27 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_28 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_29 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>], %logical_core_30 dimensionsFromStream [<size = 8, stride = 64>, <size = 4, stride = 512>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<32x64xbf16>>  
    aie.objectfifo @A_L3L2_3(%logical_shim_noc_36, {%logical_mem_33 dimensionsFromStream [<size = 128, stride = 64>, <size = 8, stride = 8192>, <size = 64, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<128x512xbf16>>  
    aie.objectfifo.link [@A_L3L2_3] -> [@A_L2L1_3]([] [0])
    aie.objectfifo @B_L2L1_0(%logical_mem_37, {%logical_core, %logical_core_7, %logical_core_15, %logical_core_23}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo @B_L3L2_0(%logical_shim_noc_45, {%logical_mem_37}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo.link [@B_L3L2_0] -> [@B_L2L1_0]([] [0])
    aie.objectfifo @B_L2L1_1(%logical_mem_38, {%logical_core_0, %logical_core_8, %logical_core_16, %logical_core_24}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo @B_L3L2_1(%logical_shim_noc_46, {%logical_mem_38}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo.link [@B_L3L2_1] -> [@B_L2L1_1]([] [0])
    aie.objectfifo @B_L2L1_2(%logical_mem_39, {%logical_core_1, %logical_core_9, %logical_core_17, %logical_core_25}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo @B_L3L2_2(%logical_shim_noc_47, {%logical_mem_39}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo.link [@B_L3L2_2] -> [@B_L2L1_2]([] [0])
    aie.objectfifo @B_L2L1_3(%logical_mem_40, {%logical_core_2, %logical_core_10, %logical_core_18, %logical_core_26}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo @B_L3L2_3(%logical_shim_noc_48, {%logical_mem_40}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo.link [@B_L3L2_3] -> [@B_L2L1_3]([] [0])
    aie.objectfifo @B_L2L1_4(%logical_mem_41, {%logical_core_3, %logical_core_11, %logical_core_19, %logical_core_27}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo @B_L3L2_4(%logical_shim_noc_49, {%logical_mem_41}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo.link [@B_L3L2_4] -> [@B_L2L1_4]([] [0])
    aie.objectfifo @B_L2L1_5(%logical_mem_42, {%logical_core_4, %logical_core_12, %logical_core_20, %logical_core_28}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo @B_L3L2_5(%logical_shim_noc_50, {%logical_mem_42}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo.link [@B_L3L2_5] -> [@B_L2L1_5]([] [0])
    aie.objectfifo @B_L2L1_6(%logical_mem_43, {%logical_core_5, %logical_core_13, %logical_core_21, %logical_core_29}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo @B_L3L2_6(%logical_shim_noc_51, {%logical_mem_43}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo.link [@B_L3L2_6] -> [@B_L2L1_6]([] [0])
    aie.objectfifo @B_L2L1_7(%logical_mem_44, {%logical_core_6, %logical_core_14, %logical_core_22, %logical_core_30}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo @B_L3L2_7(%logical_shim_noc_52, {%logical_mem_44}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo.link [@B_L3L2_7] -> [@B_L2L1_7]([] [0])
    aie.objectfifo @C_L1L2_0_0(%logical_core, {%logical_mem_53}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_0_1(%logical_core_7, {%logical_mem_53}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_0_2(%logical_core_15, {%logical_mem_53}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_0_3(%logical_core_23, {%logical_mem_53}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L2L3_0(%logical_mem_53 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_61}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>>  
    aie.objectfifo.link [@C_L1L2_0_0, @C_L1L2_0_1, @C_L1L2_0_2, @C_L1L2_0_3] -> [@C_L2L3_0]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_1_0(%logical_core_0, {%logical_mem_54}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_1_1(%logical_core_8, {%logical_mem_54}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_1_2(%logical_core_16, {%logical_mem_54}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_1_3(%logical_core_24, {%logical_mem_54}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L2L3_1(%logical_mem_54 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_62}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>>  
    aie.objectfifo.link [@C_L1L2_1_0, @C_L1L2_1_1, @C_L1L2_1_2, @C_L1L2_1_3] -> [@C_L2L3_1]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_2_0(%logical_core_1, {%logical_mem_55}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_2_1(%logical_core_9, {%logical_mem_55}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_2_2(%logical_core_17, {%logical_mem_55}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_2_3(%logical_core_25, {%logical_mem_55}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L2L3_2(%logical_mem_55 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_63}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>>  
    aie.objectfifo.link [@C_L1L2_2_0, @C_L1L2_2_1, @C_L1L2_2_2, @C_L1L2_2_3] -> [@C_L2L3_2]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_3_0(%logical_core_2, {%logical_mem_56}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_3_1(%logical_core_10, {%logical_mem_56}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_3_2(%logical_core_18, {%logical_mem_56}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_3_3(%logical_core_26, {%logical_mem_56}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L2L3_3(%logical_mem_56 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_64}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>>  
    aie.objectfifo.link [@C_L1L2_3_0, @C_L1L2_3_1, @C_L1L2_3_2, @C_L1L2_3_3] -> [@C_L2L3_3]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_4_0(%logical_core_3, {%logical_mem_57}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_4_1(%logical_core_11, {%logical_mem_57}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_4_2(%logical_core_19, {%logical_mem_57}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_4_3(%logical_core_27, {%logical_mem_57}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L2L3_4(%logical_mem_57 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_65}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>>  
    aie.objectfifo.link [@C_L1L2_4_0, @C_L1L2_4_1, @C_L1L2_4_2, @C_L1L2_4_3] -> [@C_L2L3_4]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_5_0(%logical_core_4, {%logical_mem_58}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_5_1(%logical_core_12, {%logical_mem_58}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_5_2(%logical_core_20, {%logical_mem_58}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_5_3(%logical_core_28, {%logical_mem_58}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L2L3_5(%logical_mem_58 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_66}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>>  
    aie.objectfifo.link [@C_L1L2_5_0, @C_L1L2_5_1, @C_L1L2_5_2, @C_L1L2_5_3] -> [@C_L2L3_5]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_6_0(%logical_core_5, {%logical_mem_59}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_6_1(%logical_core_13, {%logical_mem_59}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_6_2(%logical_core_21, {%logical_mem_59}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_6_3(%logical_core_29, {%logical_mem_59}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L2L3_6(%logical_mem_59 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_67}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>>  
    aie.objectfifo.link [@C_L1L2_6_0, @C_L1L2_6_1, @C_L1L2_6_2, @C_L1L2_6_3] -> [@C_L2L3_6]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_7_0(%logical_core_6, {%logical_mem_60}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_7_1(%logical_core_14, {%logical_mem_60}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_7_2(%logical_core_22, {%logical_mem_60}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_7_3(%logical_core_30, {%logical_mem_60}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L2L3_7(%logical_mem_60 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_68}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>>  
    aie.objectfifo.link [@C_L1L2_7_0, @C_L1L2_7_1, @C_L1L2_7_2, @C_L1L2_7_3] -> [@C_L2L3_7]([0, 16384, 32768, 49152] [])
    func.func private @dit_zero_c(memref<128x128xbf16>) attributes {link_with = "dit_zero_c.o"}
    func.func private @dit_matmul_quarter(memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) attributes {link_with = "dit_matmul_quarter.o"}
    func.func private @dit_swiglu_epi(memref<128x128xbf16>, memref<4xi32>, i32) attributes {link_with = "dit_swiglu_epi.o"}
    %rtp_0_0 = aie.buffer(%logical_core) {sym_name = "rtp_0_0"} : memref<4xi32> = dense<0>
    %rtp_0_1 = aie.buffer(%logical_core_0) {sym_name = "rtp_0_1"} : memref<4xi32> = dense<0>
    %rtp_0_2 = aie.buffer(%logical_core_1) {sym_name = "rtp_0_2"} : memref<4xi32> = dense<0>
    %rtp_0_3 = aie.buffer(%logical_core_2) {sym_name = "rtp_0_3"} : memref<4xi32> = dense<0>
    %rtp_0_4 = aie.buffer(%logical_core_3) {sym_name = "rtp_0_4"} : memref<4xi32> = dense<0>
    %rtp_0_5 = aie.buffer(%logical_core_4) {sym_name = "rtp_0_5"} : memref<4xi32> = dense<0>
    %rtp_0_6 = aie.buffer(%logical_core_5) {sym_name = "rtp_0_6"} : memref<4xi32> = dense<0>
    %rtp_0_7 = aie.buffer(%logical_core_6) {sym_name = "rtp_0_7"} : memref<4xi32> = dense<0>
    %rtp_1_0 = aie.buffer(%logical_core_7) {sym_name = "rtp_1_0"} : memref<4xi32> = dense<0>
    %rtp_1_1 = aie.buffer(%logical_core_8) {sym_name = "rtp_1_1"} : memref<4xi32> = dense<0>
    %rtp_1_2 = aie.buffer(%logical_core_9) {sym_name = "rtp_1_2"} : memref<4xi32> = dense<0>
    %rtp_1_3 = aie.buffer(%logical_core_10) {sym_name = "rtp_1_3"} : memref<4xi32> = dense<0>
    %rtp_1_4 = aie.buffer(%logical_core_11) {sym_name = "rtp_1_4"} : memref<4xi32> = dense<0>
    %rtp_1_5 = aie.buffer(%logical_core_12) {sym_name = "rtp_1_5"} : memref<4xi32> = dense<0>
    %rtp_1_6 = aie.buffer(%logical_core_13) {sym_name = "rtp_1_6"} : memref<4xi32> = dense<0>
    %rtp_1_7 = aie.buffer(%logical_core_14) {sym_name = "rtp_1_7"} : memref<4xi32> = dense<0>
    %rtp_2_0 = aie.buffer(%logical_core_15) {sym_name = "rtp_2_0"} : memref<4xi32> = dense<0>
    %rtp_2_1 = aie.buffer(%logical_core_16) {sym_name = "rtp_2_1"} : memref<4xi32> = dense<0>
    %rtp_2_2 = aie.buffer(%logical_core_17) {sym_name = "rtp_2_2"} : memref<4xi32> = dense<0>
    %rtp_2_3 = aie.buffer(%logical_core_18) {sym_name = "rtp_2_3"} : memref<4xi32> = dense<0>
    %rtp_2_4 = aie.buffer(%logical_core_19) {sym_name = "rtp_2_4"} : memref<4xi32> = dense<0>
    %rtp_2_5 = aie.buffer(%logical_core_20) {sym_name = "rtp_2_5"} : memref<4xi32> = dense<0>
    %rtp_2_6 = aie.buffer(%logical_core_21) {sym_name = "rtp_2_6"} : memref<4xi32> = dense<0>
    %rtp_2_7 = aie.buffer(%logical_core_22) {sym_name = "rtp_2_7"} : memref<4xi32> = dense<0>
    %rtp_3_0 = aie.buffer(%logical_core_23) {sym_name = "rtp_3_0"} : memref<4xi32> = dense<0>
    %rtp_3_1 = aie.buffer(%logical_core_24) {sym_name = "rtp_3_1"} : memref<4xi32> = dense<0>
    %rtp_3_2 = aie.buffer(%logical_core_25) {sym_name = "rtp_3_2"} : memref<4xi32> = dense<0>
    %rtp_3_3 = aie.buffer(%logical_core_26) {sym_name = "rtp_3_3"} : memref<4xi32> = dense<0>
    %rtp_3_4 = aie.buffer(%logical_core_27) {sym_name = "rtp_3_4"} : memref<4xi32> = dense<0>
    %rtp_3_5 = aie.buffer(%logical_core_28) {sym_name = "rtp_3_5"} : memref<4xi32> = dense<0>
    %rtp_3_6 = aie.buffer(%logical_core_29) {sym_name = "rtp_3_6"} : memref<4xi32> = dense<0>
    %rtp_3_7 = aie.buffer(%logical_core_30) {sym_name = "rtp_3_7"} : memref<4xi32> = dense<0>
    %0 = aie.lock(%logical_core)
    %1 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%0, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_0_0[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_0_0[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_0_0, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%0, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %2 = aie.lock(%logical_core_0)
    %3 = aie.core(%logical_core_0) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%2, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_0_1[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_0_1[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_0_1, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%2, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %4 = aie.lock(%logical_core_1)
    %5 = aie.core(%logical_core_1) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%4, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_0_2[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_0_2[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_0_2, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%4, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %6 = aie.lock(%logical_core_2)
    %7 = aie.core(%logical_core_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%6, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_0_3[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_0_3[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_0_3, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%6, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %8 = aie.lock(%logical_core_3)
    %9 = aie.core(%logical_core_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%8, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_0_4[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_0_4[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_4(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_0_4, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%8, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %10 = aie.lock(%logical_core_4)
    %11 = aie.core(%logical_core_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%10, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_0_5[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_0_5[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_5(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_0_5, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%10, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %12 = aie.lock(%logical_core_5)
    %13 = aie.core(%logical_core_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%12, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_0_6[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_0_6[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_6(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_0_6, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%12, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %14 = aie.lock(%logical_core_6)
    %15 = aie.core(%logical_core_6) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%14, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_0_7[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_0_7[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_7(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_0_7, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%14, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %16 = aie.lock(%logical_core_7)
    %17 = aie.core(%logical_core_7) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%16, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_1_0[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_1_0[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_1_0, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%16, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %18 = aie.lock(%logical_core_8)
    %19 = aie.core(%logical_core_8) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%18, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_1_1[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_1_1[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_1_1, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%18, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %20 = aie.lock(%logical_core_9)
    %21 = aie.core(%logical_core_9) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%20, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_1_2[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_1_2[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_1_2, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_2_1(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%20, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %22 = aie.lock(%logical_core_10)
    %23 = aie.core(%logical_core_10) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%22, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_1_3[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_1_3[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_1_3, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_3_1(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%22, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %24 = aie.lock(%logical_core_11)
    %25 = aie.core(%logical_core_11) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%24, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_1_4[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_1_4[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_4(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_1_4, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_4_1(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%24, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %26 = aie.lock(%logical_core_12)
    %27 = aie.core(%logical_core_12) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%26, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_1_5[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_1_5[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_5(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_1_5, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_5_1(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%26, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %28 = aie.lock(%logical_core_13)
    %29 = aie.core(%logical_core_13) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%28, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_1_6[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_1_6[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_6(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_1_6, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_6_1(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%28, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %30 = aie.lock(%logical_core_14)
    %31 = aie.core(%logical_core_14) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%30, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_1_7[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_1_7[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_7(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_1_7, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_7_1(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%30, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %32 = aie.lock(%logical_core_15)
    %33 = aie.core(%logical_core_15) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%32, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_2_0[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_2_0[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_2_0, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%32, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %34 = aie.lock(%logical_core_16)
    %35 = aie.core(%logical_core_16) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%34, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_2_1[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_2_1[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_2_1, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%34, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %36 = aie.lock(%logical_core_17)
    %37 = aie.core(%logical_core_17) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%36, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_2_2[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_2_2[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_2_2, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_2_2(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%36, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %38 = aie.lock(%logical_core_18)
    %39 = aie.core(%logical_core_18) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%38, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_2_3[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_2_3[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_2_3, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_3_2(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%38, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %40 = aie.lock(%logical_core_19)
    %41 = aie.core(%logical_core_19) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%40, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_2_4[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_2_4[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_4(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_2_4, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_4_2(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%40, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %42 = aie.lock(%logical_core_20)
    %43 = aie.core(%logical_core_20) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%42, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_2_5[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_2_5[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_5(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_2_5, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_5_2(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%42, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %44 = aie.lock(%logical_core_21)
    %45 = aie.core(%logical_core_21) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%44, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_2_6[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_2_6[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_6(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_2_6, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_6_2(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%44, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %46 = aie.lock(%logical_core_22)
    %47 = aie.core(%logical_core_22) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%46, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_2_7[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_2_7[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_7(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_2_7, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_7_2(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%46, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %48 = aie.lock(%logical_core_23)
    %49 = aie.core(%logical_core_23) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%48, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_3_0[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_3_0[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_3_0, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%48, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %50 = aie.lock(%logical_core_24)
    %51 = aie.core(%logical_core_24) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%50, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_3_1[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_3_1[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_3_1, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%50, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %52 = aie.lock(%logical_core_25)
    %53 = aie.core(%logical_core_25) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%52, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_3_2[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_3_2[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_3_2, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_2_3(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%52, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %54 = aie.lock(%logical_core_26)
    %55 = aie.core(%logical_core_26) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%54, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_3_3[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_3_3[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_3_3, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_3_3(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%54, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %56 = aie.lock(%logical_core_27)
    %57 = aie.core(%logical_core_27) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%56, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_3_4[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_3_4[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_4(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_3_4, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_4_3(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%56, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %58 = aie.lock(%logical_core_28)
    %59 = aie.core(%logical_core_28) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%58, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_3_5[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_3_5[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_5(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_3_5, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_5_3(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%58, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %60 = aie.lock(%logical_core_29)
    %61 = aie.core(%logical_core_29) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%60, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_3_6[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_3_6[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_6(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_3_6, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_6_3(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%60, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %62 = aie.lock(%logical_core_30)
    %63 = aie.core(%logical_core_30) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%62, Acquire, %c1_i32)
        %c0_69 = arith.constant 0 : index
        %64 = memref.load %rtp_3_7[%c0_69] : memref<4xi32>
        %c1_70 = arith.constant 1 : index
        %65 = memref.load %rtp_3_7[%c1_70] : memref<4xi32>
        %c0_71 = arith.constant 0 : index
        %66 = arith.index_cast %64 : i32 to index
        %c1_72 = arith.constant 1 : index
        scf.for %arg1 = %c0_71 to %66 step %c1_72 {
          %67 = aie.objectfifo.acquire @C_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %68 = aie.objectfifo.subview.access %67[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          func.call @dit_zero_c(%68) : (memref<128x128xbf16>) -> ()
          %c0_74 = arith.constant 0 : index
          %69 = arith.index_cast %65 : i32 to index
          %c1_75 = arith.constant 1 : index
          scf.for %arg2 = %c0_74 to %69 step %c1_75 {
            %71 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %72 = aie.objectfifo.subview.access %71[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %73 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %74 = aie.objectfifo.subview.access %73[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%74, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %75 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %76 = aie.objectfifo.subview.access %75[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%76, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %77 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %78 = aie.objectfifo.subview.access %77[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%78, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            %79 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<32x64xbf16>>
            %80 = aie.objectfifo.subview.access %79[0] : !aie.objectfifosubview<memref<32x64xbf16>> -> memref<32x64xbf16>
            func.call @dit_matmul_quarter(%80, %72, %68) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_7(Consume, 1)
          }
          %70 = arith.index_cast %arg1 : index to i32
          func.call @dit_swiglu_epi(%68, %rtp_3_7, %70) : (memref<128x128xbf16>, memref<4xi32>, i32) -> ()
          aie.objectfifo.release @C_L1L2_7_3(Produce, 1)
        }
        %c1_i32_73 = arith.constant 1 : i32
        aie.use_lock(%62, Release, %c1_i32_73)
      }
      aie.end
    } {stack_size = 3840 : i32}
    aie.runtime_sequence(%arg0: memref<12582912xbf16>, %arg1: memref<3538944x!aiex.bfp<"v8bfp16ebs8">>, %arg2: memref<37748736xbf16>) {
      %c72_i32 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_0_0, 0, %c72_i32) : i32
      %c48_i32 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_0_0, 1, %c48_i32) : i32
      %c9_i32 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_0, 2, %c9_i32) : i32
      %c9_i32_69 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_0, 3, %c9_i32_69) : i32
      %c72_i32_70 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_0_1, 0, %c72_i32_70) : i32
      %c48_i32_71 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_0_1, 1, %c48_i32_71) : i32
      %c9_i32_72 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_1, 2, %c9_i32_72) : i32
      %c9_i32_73 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_1, 3, %c9_i32_73) : i32
      %c72_i32_74 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_0_2, 0, %c72_i32_74) : i32
      %c48_i32_75 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_0_2, 1, %c48_i32_75) : i32
      %c9_i32_76 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_2, 2, %c9_i32_76) : i32
      %c9_i32_77 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_2, 3, %c9_i32_77) : i32
      %c72_i32_78 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_0_3, 0, %c72_i32_78) : i32
      %c48_i32_79 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_0_3, 1, %c48_i32_79) : i32
      %c9_i32_80 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_3, 2, %c9_i32_80) : i32
      %c9_i32_81 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_3, 3, %c9_i32_81) : i32
      %c72_i32_82 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_0_4, 0, %c72_i32_82) : i32
      %c48_i32_83 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_0_4, 1, %c48_i32_83) : i32
      %c9_i32_84 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_4, 2, %c9_i32_84) : i32
      %c9_i32_85 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_4, 3, %c9_i32_85) : i32
      %c72_i32_86 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_0_5, 0, %c72_i32_86) : i32
      %c48_i32_87 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_0_5, 1, %c48_i32_87) : i32
      %c9_i32_88 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_5, 2, %c9_i32_88) : i32
      %c9_i32_89 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_5, 3, %c9_i32_89) : i32
      %c72_i32_90 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_0_6, 0, %c72_i32_90) : i32
      %c48_i32_91 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_0_6, 1, %c48_i32_91) : i32
      %c9_i32_92 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_6, 2, %c9_i32_92) : i32
      %c9_i32_93 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_6, 3, %c9_i32_93) : i32
      %c72_i32_94 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_0_7, 0, %c72_i32_94) : i32
      %c48_i32_95 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_0_7, 1, %c48_i32_95) : i32
      %c9_i32_96 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_7, 2, %c9_i32_96) : i32
      %c9_i32_97 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_7, 3, %c9_i32_97) : i32
      %c72_i32_98 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_1_0, 0, %c72_i32_98) : i32
      %c48_i32_99 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_1_0, 1, %c48_i32_99) : i32
      %c9_i32_100 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_0, 2, %c9_i32_100) : i32
      %c9_i32_101 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_0, 3, %c9_i32_101) : i32
      %c72_i32_102 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_1_1, 0, %c72_i32_102) : i32
      %c48_i32_103 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_1_1, 1, %c48_i32_103) : i32
      %c9_i32_104 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_1, 2, %c9_i32_104) : i32
      %c9_i32_105 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_1, 3, %c9_i32_105) : i32
      %c72_i32_106 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_1_2, 0, %c72_i32_106) : i32
      %c48_i32_107 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_1_2, 1, %c48_i32_107) : i32
      %c9_i32_108 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_2, 2, %c9_i32_108) : i32
      %c9_i32_109 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_2, 3, %c9_i32_109) : i32
      %c72_i32_110 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_1_3, 0, %c72_i32_110) : i32
      %c48_i32_111 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_1_3, 1, %c48_i32_111) : i32
      %c9_i32_112 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_3, 2, %c9_i32_112) : i32
      %c9_i32_113 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_3, 3, %c9_i32_113) : i32
      %c72_i32_114 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_1_4, 0, %c72_i32_114) : i32
      %c48_i32_115 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_1_4, 1, %c48_i32_115) : i32
      %c9_i32_116 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_4, 2, %c9_i32_116) : i32
      %c9_i32_117 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_4, 3, %c9_i32_117) : i32
      %c72_i32_118 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_1_5, 0, %c72_i32_118) : i32
      %c48_i32_119 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_1_5, 1, %c48_i32_119) : i32
      %c9_i32_120 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_5, 2, %c9_i32_120) : i32
      %c9_i32_121 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_5, 3, %c9_i32_121) : i32
      %c72_i32_122 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_1_6, 0, %c72_i32_122) : i32
      %c48_i32_123 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_1_6, 1, %c48_i32_123) : i32
      %c9_i32_124 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_6, 2, %c9_i32_124) : i32
      %c9_i32_125 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_6, 3, %c9_i32_125) : i32
      %c72_i32_126 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_1_7, 0, %c72_i32_126) : i32
      %c48_i32_127 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_1_7, 1, %c48_i32_127) : i32
      %c9_i32_128 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_7, 2, %c9_i32_128) : i32
      %c9_i32_129 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_7, 3, %c9_i32_129) : i32
      %c72_i32_130 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_2_0, 0, %c72_i32_130) : i32
      %c48_i32_131 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_2_0, 1, %c48_i32_131) : i32
      %c9_i32_132 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_0, 2, %c9_i32_132) : i32
      %c9_i32_133 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_0, 3, %c9_i32_133) : i32
      %c72_i32_134 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_2_1, 0, %c72_i32_134) : i32
      %c48_i32_135 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_2_1, 1, %c48_i32_135) : i32
      %c9_i32_136 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_1, 2, %c9_i32_136) : i32
      %c9_i32_137 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_1, 3, %c9_i32_137) : i32
      %c72_i32_138 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_2_2, 0, %c72_i32_138) : i32
      %c48_i32_139 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_2_2, 1, %c48_i32_139) : i32
      %c9_i32_140 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_2, 2, %c9_i32_140) : i32
      %c9_i32_141 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_2, 3, %c9_i32_141) : i32
      %c72_i32_142 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_2_3, 0, %c72_i32_142) : i32
      %c48_i32_143 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_2_3, 1, %c48_i32_143) : i32
      %c9_i32_144 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_3, 2, %c9_i32_144) : i32
      %c9_i32_145 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_3, 3, %c9_i32_145) : i32
      %c72_i32_146 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_2_4, 0, %c72_i32_146) : i32
      %c48_i32_147 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_2_4, 1, %c48_i32_147) : i32
      %c9_i32_148 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_4, 2, %c9_i32_148) : i32
      %c9_i32_149 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_4, 3, %c9_i32_149) : i32
      %c72_i32_150 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_2_5, 0, %c72_i32_150) : i32
      %c48_i32_151 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_2_5, 1, %c48_i32_151) : i32
      %c9_i32_152 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_5, 2, %c9_i32_152) : i32
      %c9_i32_153 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_5, 3, %c9_i32_153) : i32
      %c72_i32_154 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_2_6, 0, %c72_i32_154) : i32
      %c48_i32_155 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_2_6, 1, %c48_i32_155) : i32
      %c9_i32_156 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_6, 2, %c9_i32_156) : i32
      %c9_i32_157 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_6, 3, %c9_i32_157) : i32
      %c72_i32_158 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_2_7, 0, %c72_i32_158) : i32
      %c48_i32_159 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_2_7, 1, %c48_i32_159) : i32
      %c9_i32_160 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_7, 2, %c9_i32_160) : i32
      %c9_i32_161 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_7, 3, %c9_i32_161) : i32
      %c72_i32_162 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_3_0, 0, %c72_i32_162) : i32
      %c48_i32_163 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_3_0, 1, %c48_i32_163) : i32
      %c9_i32_164 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_0, 2, %c9_i32_164) : i32
      %c9_i32_165 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_0, 3, %c9_i32_165) : i32
      %c72_i32_166 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_3_1, 0, %c72_i32_166) : i32
      %c48_i32_167 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_3_1, 1, %c48_i32_167) : i32
      %c9_i32_168 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_1, 2, %c9_i32_168) : i32
      %c9_i32_169 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_1, 3, %c9_i32_169) : i32
      %c72_i32_170 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_3_2, 0, %c72_i32_170) : i32
      %c48_i32_171 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_3_2, 1, %c48_i32_171) : i32
      %c9_i32_172 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_2, 2, %c9_i32_172) : i32
      %c9_i32_173 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_2, 3, %c9_i32_173) : i32
      %c72_i32_174 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_3_3, 0, %c72_i32_174) : i32
      %c48_i32_175 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_3_3, 1, %c48_i32_175) : i32
      %c9_i32_176 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_3, 2, %c9_i32_176) : i32
      %c9_i32_177 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_3, 3, %c9_i32_177) : i32
      %c72_i32_178 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_3_4, 0, %c72_i32_178) : i32
      %c48_i32_179 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_3_4, 1, %c48_i32_179) : i32
      %c9_i32_180 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_4, 2, %c9_i32_180) : i32
      %c9_i32_181 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_4, 3, %c9_i32_181) : i32
      %c72_i32_182 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_3_5, 0, %c72_i32_182) : i32
      %c48_i32_183 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_3_5, 1, %c48_i32_183) : i32
      %c9_i32_184 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_5, 2, %c9_i32_184) : i32
      %c9_i32_185 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_5, 3, %c9_i32_185) : i32
      %c72_i32_186 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_3_6, 0, %c72_i32_186) : i32
      %c48_i32_187 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_3_6, 1, %c48_i32_187) : i32
      %c9_i32_188 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_6, 2, %c9_i32_188) : i32
      %c9_i32_189 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_6, 3, %c9_i32_189) : i32
      %c72_i32_190 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_3_7, 0, %c72_i32_190) : i32
      %c48_i32_191 = arith.constant 48 : i32
      aiex.npu.rtp_write(@rtp_3_7, 1, %c48_i32_191) : i32
      %c9_i32_192 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_7, 2, %c9_i32_192) : i32
      %c9_i32_193 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_7, 3, %c9_i32_193) : i32
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
      %64 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 0 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 393216 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 786432 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1179648 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%67)
      %68 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 49152 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 98304 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 147456 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 196608 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 245760 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 294912 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 344064 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 0 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 128 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 256 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 384 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 512 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 640 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 768 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 896 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 0 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%84)
      %85 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 393216 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 786432 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1179648 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 393216 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 442368 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 491520 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 540672 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 589824 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 638976 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 688128 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 737280 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%95)
      %96 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 1024 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 1152 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%97)
      %98 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 1280 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 1408 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 1536 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%100)
      %101 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 1664 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%101)
      %102 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 1792 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%102)
      %103 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 1920 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%103)
      %104 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 0 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%104)
      %105 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 393216 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%105)
      %106 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 786432 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%106)
      %107 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1179648 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%107)
      %108 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 786432 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%108)
      %109 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 835584 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%109)
      %110 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 884736 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%110)
      %111 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 933888 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%111)
      %112 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 983040 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%112)
      %113 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1032192 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%113)
      %114 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1081344 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%114)
      %115 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1130496 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%115)
      %116 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 2048 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%116)
      %117 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 2176 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%117)
      %118 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 2304 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%118)
      %119 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 2432 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%119)
      %120 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 2560 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%120)
      %121 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 2688 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%121)
      %122 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 2816 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%122)
      %123 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 2944 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%123)
      %124 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 0 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%124)
      %125 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 393216 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%125)
      %126 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 786432 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%126)
      %127 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1179648 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%127)
      %128 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1179648 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%128)
      %129 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1228800 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%129)
      %130 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1277952 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%130)
      %131 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1327104 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%131)
      %132 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1376256 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%132)
      %133 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1425408 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%133)
      %134 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1474560 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%134)
      %135 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1523712 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%135)
      %136 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 3072 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%136)
      %137 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 3200 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%137)
      %138 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 3328 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%138)
      %139 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 3456 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%139)
      %140 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 3584 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%140)
      %141 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 3712 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%141)
      %142 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 3840 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%142)
      %143 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 3968 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%143)
      aiex.dma_await_task(%76)
      aiex.dma_await_task(%77)
      aiex.dma_await_task(%78)
      aiex.dma_await_task(%79)
      aiex.dma_await_task(%80)
      aiex.dma_await_task(%81)
      aiex.dma_await_task(%82)
      aiex.dma_await_task(%83)
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
      aiex.dma_await_task(%96)
      aiex.dma_await_task(%97)
      aiex.dma_await_task(%98)
      aiex.dma_await_task(%99)
      aiex.dma_await_task(%100)
      aiex.dma_await_task(%101)
      aiex.dma_await_task(%102)
      aiex.dma_await_task(%103)
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
      aiex.dma_free_task(%96)
      aiex.dma_free_task(%97)
      aiex.dma_free_task(%98)
      aiex.dma_free_task(%99)
      aiex.dma_free_task(%100)
      aiex.dma_free_task(%101)
      aiex.dma_free_task(%102)
      aiex.dma_free_task(%103)
      %144 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 0 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%144)
      %145 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 393216 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%145)
      %146 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 786432 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%146)
      %147 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1179648 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%147)
      %148 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1572864 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%148)
      %149 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1622016 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%149)
      %150 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1671168 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%150)
      %151 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1720320 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%151)
      %152 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1769472 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%152)
      %153 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1818624 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%153)
      %154 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1867776 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%154)
      %155 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1916928 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%155)
      %156 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4096 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%156)
      %157 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4224 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%157)
      %158 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4352 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%158)
      %159 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4480 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%159)
      %160 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4608 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%160)
      %161 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4736 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%161)
      %162 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4864 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%162)
      %163 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4992 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%163)
      %164 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 0 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%164)
      %165 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 393216 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%165)
      %166 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 786432 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%166)
      %167 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1179648 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%167)
      %168 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1966080 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%168)
      %169 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2015232 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%169)
      %170 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2064384 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%170)
      %171 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2113536 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%171)
      %172 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2162688 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%172)
      %173 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2211840 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%173)
      %174 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2260992 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%174)
      %175 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2310144 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%175)
      %176 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 5120 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%176)
      %177 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 5248 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%177)
      %178 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 5376 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%178)
      %179 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 5504 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%179)
      %180 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 5632 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%180)
      %181 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 5760 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%181)
      %182 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 5888 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%182)
      %183 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 6016 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%183)
      aiex.dma_await_task(%116)
      aiex.dma_await_task(%117)
      aiex.dma_await_task(%118)
      aiex.dma_await_task(%119)
      aiex.dma_await_task(%120)
      aiex.dma_await_task(%121)
      aiex.dma_await_task(%122)
      aiex.dma_await_task(%123)
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
      aiex.dma_await_task(%136)
      aiex.dma_await_task(%137)
      aiex.dma_await_task(%138)
      aiex.dma_await_task(%139)
      aiex.dma_await_task(%140)
      aiex.dma_await_task(%141)
      aiex.dma_await_task(%142)
      aiex.dma_await_task(%143)
      aiex.dma_free_task(%124)
      aiex.dma_free_task(%125)
      aiex.dma_free_task(%126)
      aiex.dma_free_task(%127)
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
      %184 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 0 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%184)
      %185 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 393216 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%185)
      %186 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 786432 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%186)
      %187 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1179648 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%187)
      %188 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2359296 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%188)
      %189 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2408448 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%189)
      %190 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2457600 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%190)
      %191 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2506752 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%191)
      %192 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2555904 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%192)
      %193 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2605056 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%193)
      %194 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2654208 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%194)
      %195 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2703360 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%195)
      %196 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 6144 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%196)
      %197 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 6272 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%197)
      %198 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 6400 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%198)
      %199 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 6528 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%199)
      %200 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 6656 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%200)
      %201 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 6784 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%201)
      %202 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 6912 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%202)
      %203 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 7040 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%203)
      %204 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 0 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%204)
      %205 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 393216 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%205)
      %206 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 786432 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%206)
      %207 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1179648 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%207)
      %208 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2752512 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%208)
      %209 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2801664 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%209)
      %210 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2850816 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%210)
      %211 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2899968 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%211)
      %212 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2949120 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%212)
      %213 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2998272 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%213)
      %214 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3047424 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%214)
      %215 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3096576 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%215)
      %216 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 7168 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%216)
      %217 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 7296 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%217)
      %218 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 7424 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%218)
      %219 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 7552 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%219)
      %220 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 7680 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%220)
      %221 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 7808 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%221)
      %222 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 7936 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%222)
      %223 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 8064 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%223)
      aiex.dma_await_task(%156)
      aiex.dma_await_task(%157)
      aiex.dma_await_task(%158)
      aiex.dma_await_task(%159)
      aiex.dma_await_task(%160)
      aiex.dma_await_task(%161)
      aiex.dma_await_task(%162)
      aiex.dma_await_task(%163)
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
      aiex.dma_free_task(%160)
      aiex.dma_free_task(%161)
      aiex.dma_free_task(%162)
      aiex.dma_free_task(%163)
      aiex.dma_await_task(%176)
      aiex.dma_await_task(%177)
      aiex.dma_await_task(%178)
      aiex.dma_await_task(%179)
      aiex.dma_await_task(%180)
      aiex.dma_await_task(%181)
      aiex.dma_await_task(%182)
      aiex.dma_await_task(%183)
      aiex.dma_free_task(%164)
      aiex.dma_free_task(%165)
      aiex.dma_free_task(%166)
      aiex.dma_free_task(%167)
      aiex.dma_free_task(%168)
      aiex.dma_free_task(%169)
      aiex.dma_free_task(%170)
      aiex.dma_free_task(%171)
      aiex.dma_free_task(%172)
      aiex.dma_free_task(%173)
      aiex.dma_free_task(%174)
      aiex.dma_free_task(%175)
      aiex.dma_free_task(%176)
      aiex.dma_free_task(%177)
      aiex.dma_free_task(%178)
      aiex.dma_free_task(%179)
      aiex.dma_free_task(%180)
      aiex.dma_free_task(%181)
      aiex.dma_free_task(%182)
      aiex.dma_free_task(%183)
      %224 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 0 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%224)
      %225 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 393216 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%225)
      %226 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 786432 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%226)
      %227 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1179648 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%227)
      %228 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3145728 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%228)
      %229 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3194880 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%229)
      %230 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3244032 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%230)
      %231 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3293184 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%231)
      %232 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3342336 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%232)
      %233 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3391488 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%233)
      %234 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3440640 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%234)
      %235 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3489792 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%235)
      %236 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 8192 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%236)
      %237 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 8320 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%237)
      %238 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 8448 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%238)
      %239 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 8576 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%239)
      %240 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 8704 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%240)
      %241 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 8832 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%241)
      %242 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 8960 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%242)
      %243 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9088 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%243)
      %244 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1572864 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%244)
      %245 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1966080 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%245)
      %246 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 2359296 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%246)
      %247 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 2752512 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%247)
      %248 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%248)
      %249 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 49152 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%249)
      %250 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 98304 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%250)
      %251 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 147456 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%251)
      %252 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 196608 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%252)
      %253 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 245760 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%253)
      %254 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 294912 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%254)
      %255 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 344064 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%255)
      %256 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4718592 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%256)
      %257 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4718720 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%257)
      %258 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4718848 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%258)
      %259 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4718976 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%259)
      %260 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4719104 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%260)
      %261 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4719232 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%261)
      %262 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4719360 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%262)
      %263 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4719488 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%263)
      aiex.dma_await_task(%196)
      aiex.dma_await_task(%197)
      aiex.dma_await_task(%198)
      aiex.dma_await_task(%199)
      aiex.dma_await_task(%200)
      aiex.dma_await_task(%201)
      aiex.dma_await_task(%202)
      aiex.dma_await_task(%203)
      aiex.dma_free_task(%184)
      aiex.dma_free_task(%185)
      aiex.dma_free_task(%186)
      aiex.dma_free_task(%187)
      aiex.dma_free_task(%188)
      aiex.dma_free_task(%189)
      aiex.dma_free_task(%190)
      aiex.dma_free_task(%191)
      aiex.dma_free_task(%192)
      aiex.dma_free_task(%193)
      aiex.dma_free_task(%194)
      aiex.dma_free_task(%195)
      aiex.dma_free_task(%196)
      aiex.dma_free_task(%197)
      aiex.dma_free_task(%198)
      aiex.dma_free_task(%199)
      aiex.dma_free_task(%200)
      aiex.dma_free_task(%201)
      aiex.dma_free_task(%202)
      aiex.dma_free_task(%203)
      aiex.dma_await_task(%216)
      aiex.dma_await_task(%217)
      aiex.dma_await_task(%218)
      aiex.dma_await_task(%219)
      aiex.dma_await_task(%220)
      aiex.dma_await_task(%221)
      aiex.dma_await_task(%222)
      aiex.dma_await_task(%223)
      aiex.dma_free_task(%204)
      aiex.dma_free_task(%205)
      aiex.dma_free_task(%206)
      aiex.dma_free_task(%207)
      aiex.dma_free_task(%208)
      aiex.dma_free_task(%209)
      aiex.dma_free_task(%210)
      aiex.dma_free_task(%211)
      aiex.dma_free_task(%212)
      aiex.dma_free_task(%213)
      aiex.dma_free_task(%214)
      aiex.dma_free_task(%215)
      aiex.dma_free_task(%216)
      aiex.dma_free_task(%217)
      aiex.dma_free_task(%218)
      aiex.dma_free_task(%219)
      aiex.dma_free_task(%220)
      aiex.dma_free_task(%221)
      aiex.dma_free_task(%222)
      aiex.dma_free_task(%223)
      %264 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1572864 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%264)
      %265 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1966080 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%265)
      %266 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 2359296 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%266)
      %267 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 2752512 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%267)
      %268 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 393216 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%268)
      %269 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 442368 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%269)
      %270 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 491520 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%270)
      %271 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 540672 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%271)
      %272 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 589824 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%272)
      %273 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 638976 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%273)
      %274 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 688128 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%274)
      %275 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 737280 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%275)
      %276 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4719616 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%276)
      %277 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4719744 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%277)
      %278 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4719872 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%278)
      %279 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4720000 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%279)
      %280 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4720128 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%280)
      %281 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4720256 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%281)
      %282 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4720384 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%282)
      %283 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4720512 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%283)
      %284 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1572864 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%284)
      %285 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1966080 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%285)
      %286 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 2359296 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%286)
      %287 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 2752512 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%287)
      %288 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 786432 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%288)
      %289 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 835584 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%289)
      %290 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 884736 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%290)
      %291 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 933888 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%291)
      %292 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 983040 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%292)
      %293 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1032192 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%293)
      %294 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1081344 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%294)
      %295 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1130496 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%295)
      %296 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4720640 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%296)
      %297 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4720768 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%297)
      %298 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4720896 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%298)
      %299 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4721024 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%299)
      %300 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4721152 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%300)
      %301 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4721280 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%301)
      %302 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4721408 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%302)
      %303 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4721536 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%303)
      aiex.dma_await_task(%236)
      aiex.dma_await_task(%237)
      aiex.dma_await_task(%238)
      aiex.dma_await_task(%239)
      aiex.dma_await_task(%240)
      aiex.dma_await_task(%241)
      aiex.dma_await_task(%242)
      aiex.dma_await_task(%243)
      aiex.dma_free_task(%224)
      aiex.dma_free_task(%225)
      aiex.dma_free_task(%226)
      aiex.dma_free_task(%227)
      aiex.dma_free_task(%228)
      aiex.dma_free_task(%229)
      aiex.dma_free_task(%230)
      aiex.dma_free_task(%231)
      aiex.dma_free_task(%232)
      aiex.dma_free_task(%233)
      aiex.dma_free_task(%234)
      aiex.dma_free_task(%235)
      aiex.dma_free_task(%236)
      aiex.dma_free_task(%237)
      aiex.dma_free_task(%238)
      aiex.dma_free_task(%239)
      aiex.dma_free_task(%240)
      aiex.dma_free_task(%241)
      aiex.dma_free_task(%242)
      aiex.dma_free_task(%243)
      aiex.dma_await_task(%256)
      aiex.dma_await_task(%257)
      aiex.dma_await_task(%258)
      aiex.dma_await_task(%259)
      aiex.dma_await_task(%260)
      aiex.dma_await_task(%261)
      aiex.dma_await_task(%262)
      aiex.dma_await_task(%263)
      aiex.dma_free_task(%244)
      aiex.dma_free_task(%245)
      aiex.dma_free_task(%246)
      aiex.dma_free_task(%247)
      aiex.dma_free_task(%248)
      aiex.dma_free_task(%249)
      aiex.dma_free_task(%250)
      aiex.dma_free_task(%251)
      aiex.dma_free_task(%252)
      aiex.dma_free_task(%253)
      aiex.dma_free_task(%254)
      aiex.dma_free_task(%255)
      aiex.dma_free_task(%256)
      aiex.dma_free_task(%257)
      aiex.dma_free_task(%258)
      aiex.dma_free_task(%259)
      aiex.dma_free_task(%260)
      aiex.dma_free_task(%261)
      aiex.dma_free_task(%262)
      aiex.dma_free_task(%263)
      %304 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1572864 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%304)
      %305 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1966080 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%305)
      %306 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 2359296 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%306)
      %307 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 2752512 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%307)
      %308 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1179648 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%308)
      %309 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1228800 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%309)
      %310 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1277952 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%310)
      %311 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1327104 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%311)
      %312 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1376256 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%312)
      %313 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1425408 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%313)
      %314 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1474560 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%314)
      %315 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1523712 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%315)
      %316 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4721664 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%316)
      %317 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4721792 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%317)
      %318 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4721920 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%318)
      %319 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4722048 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%319)
      %320 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4722176 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%320)
      %321 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4722304 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%321)
      %322 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4722432 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%322)
      %323 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4722560 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%323)
      %324 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1572864 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%324)
      %325 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1966080 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%325)
      %326 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 2359296 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%326)
      %327 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 2752512 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%327)
      %328 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1572864 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%328)
      %329 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1622016 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%329)
      %330 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1671168 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%330)
      %331 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1720320 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%331)
      %332 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1769472 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%332)
      %333 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1818624 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%333)
      %334 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1867776 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%334)
      %335 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1916928 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%335)
      %336 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4722688 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%336)
      %337 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4722816 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%337)
      %338 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4722944 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%338)
      %339 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4723072 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%339)
      %340 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4723200 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%340)
      %341 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4723328 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%341)
      %342 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4723456 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%342)
      %343 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4723584 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%343)
      aiex.dma_await_task(%276)
      aiex.dma_await_task(%277)
      aiex.dma_await_task(%278)
      aiex.dma_await_task(%279)
      aiex.dma_await_task(%280)
      aiex.dma_await_task(%281)
      aiex.dma_await_task(%282)
      aiex.dma_await_task(%283)
      aiex.dma_free_task(%264)
      aiex.dma_free_task(%265)
      aiex.dma_free_task(%266)
      aiex.dma_free_task(%267)
      aiex.dma_free_task(%268)
      aiex.dma_free_task(%269)
      aiex.dma_free_task(%270)
      aiex.dma_free_task(%271)
      aiex.dma_free_task(%272)
      aiex.dma_free_task(%273)
      aiex.dma_free_task(%274)
      aiex.dma_free_task(%275)
      aiex.dma_free_task(%276)
      aiex.dma_free_task(%277)
      aiex.dma_free_task(%278)
      aiex.dma_free_task(%279)
      aiex.dma_free_task(%280)
      aiex.dma_free_task(%281)
      aiex.dma_free_task(%282)
      aiex.dma_free_task(%283)
      aiex.dma_await_task(%296)
      aiex.dma_await_task(%297)
      aiex.dma_await_task(%298)
      aiex.dma_await_task(%299)
      aiex.dma_await_task(%300)
      aiex.dma_await_task(%301)
      aiex.dma_await_task(%302)
      aiex.dma_await_task(%303)
      aiex.dma_free_task(%284)
      aiex.dma_free_task(%285)
      aiex.dma_free_task(%286)
      aiex.dma_free_task(%287)
      aiex.dma_free_task(%288)
      aiex.dma_free_task(%289)
      aiex.dma_free_task(%290)
      aiex.dma_free_task(%291)
      aiex.dma_free_task(%292)
      aiex.dma_free_task(%293)
      aiex.dma_free_task(%294)
      aiex.dma_free_task(%295)
      aiex.dma_free_task(%296)
      aiex.dma_free_task(%297)
      aiex.dma_free_task(%298)
      aiex.dma_free_task(%299)
      aiex.dma_free_task(%300)
      aiex.dma_free_task(%301)
      aiex.dma_free_task(%302)
      aiex.dma_free_task(%303)
      %344 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1572864 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%344)
      %345 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1966080 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%345)
      %346 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 2359296 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%346)
      %347 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 2752512 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%347)
      %348 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1966080 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%348)
      %349 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2015232 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%349)
      %350 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2064384 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%350)
      %351 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2113536 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%351)
      %352 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2162688 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%352)
      %353 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2211840 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%353)
      %354 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2260992 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%354)
      %355 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2310144 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%355)
      %356 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4723712 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%356)
      %357 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4723840 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%357)
      %358 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4723968 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%358)
      %359 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4724096 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%359)
      %360 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4724224 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%360)
      %361 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4724352 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%361)
      %362 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4724480 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%362)
      %363 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4724608 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%363)
      %364 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1572864 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%364)
      %365 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1966080 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%365)
      %366 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 2359296 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%366)
      %367 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 2752512 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%367)
      %368 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2359296 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%368)
      %369 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2408448 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%369)
      %370 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2457600 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%370)
      %371 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2506752 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%371)
      %372 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2555904 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%372)
      %373 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2605056 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%373)
      %374 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2654208 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%374)
      %375 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2703360 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%375)
      %376 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4724736 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%376)
      %377 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4724864 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%377)
      %378 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4724992 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%378)
      %379 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4725120 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%379)
      %380 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4725248 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%380)
      %381 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4725376 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%381)
      %382 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4725504 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%382)
      %383 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4725632 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%383)
      aiex.dma_await_task(%316)
      aiex.dma_await_task(%317)
      aiex.dma_await_task(%318)
      aiex.dma_await_task(%319)
      aiex.dma_await_task(%320)
      aiex.dma_await_task(%321)
      aiex.dma_await_task(%322)
      aiex.dma_await_task(%323)
      aiex.dma_free_task(%304)
      aiex.dma_free_task(%305)
      aiex.dma_free_task(%306)
      aiex.dma_free_task(%307)
      aiex.dma_free_task(%308)
      aiex.dma_free_task(%309)
      aiex.dma_free_task(%310)
      aiex.dma_free_task(%311)
      aiex.dma_free_task(%312)
      aiex.dma_free_task(%313)
      aiex.dma_free_task(%314)
      aiex.dma_free_task(%315)
      aiex.dma_free_task(%316)
      aiex.dma_free_task(%317)
      aiex.dma_free_task(%318)
      aiex.dma_free_task(%319)
      aiex.dma_free_task(%320)
      aiex.dma_free_task(%321)
      aiex.dma_free_task(%322)
      aiex.dma_free_task(%323)
      aiex.dma_await_task(%336)
      aiex.dma_await_task(%337)
      aiex.dma_await_task(%338)
      aiex.dma_await_task(%339)
      aiex.dma_await_task(%340)
      aiex.dma_await_task(%341)
      aiex.dma_await_task(%342)
      aiex.dma_await_task(%343)
      aiex.dma_free_task(%324)
      aiex.dma_free_task(%325)
      aiex.dma_free_task(%326)
      aiex.dma_free_task(%327)
      aiex.dma_free_task(%328)
      aiex.dma_free_task(%329)
      aiex.dma_free_task(%330)
      aiex.dma_free_task(%331)
      aiex.dma_free_task(%332)
      aiex.dma_free_task(%333)
      aiex.dma_free_task(%334)
      aiex.dma_free_task(%335)
      aiex.dma_free_task(%336)
      aiex.dma_free_task(%337)
      aiex.dma_free_task(%338)
      aiex.dma_free_task(%339)
      aiex.dma_free_task(%340)
      aiex.dma_free_task(%341)
      aiex.dma_free_task(%342)
      aiex.dma_free_task(%343)
      %384 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1572864 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%384)
      %385 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1966080 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%385)
      %386 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 2359296 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%386)
      %387 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 2752512 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%387)
      %388 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2752512 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%388)
      %389 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2801664 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%389)
      %390 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2850816 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%390)
      %391 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2899968 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%391)
      %392 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2949120 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%392)
      %393 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2998272 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%393)
      %394 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3047424 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%394)
      %395 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3096576 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%395)
      %396 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4725760 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%396)
      %397 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4725888 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%397)
      %398 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4726016 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%398)
      %399 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4726144 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%399)
      %400 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4726272 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%400)
      %401 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4726400 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%401)
      %402 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4726528 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%402)
      %403 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4726656 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%403)
      %404 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1572864 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%404)
      %405 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 1966080 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%405)
      %406 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 2359296 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%406)
      %407 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 2752512 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%407)
      %408 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3145728 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%408)
      %409 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3194880 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%409)
      %410 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3244032 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%410)
      %411 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3293184 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%411)
      %412 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3342336 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%412)
      %413 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3391488 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%413)
      %414 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3440640 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%414)
      %415 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3489792 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%415)
      %416 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4726784 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%416)
      %417 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4726912 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%417)
      %418 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4727040 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%418)
      %419 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4727168 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%419)
      %420 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4727296 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%420)
      %421 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4727424 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%421)
      %422 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4727552 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%422)
      %423 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 4727680 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%423)
      aiex.dma_await_task(%356)
      aiex.dma_await_task(%357)
      aiex.dma_await_task(%358)
      aiex.dma_await_task(%359)
      aiex.dma_await_task(%360)
      aiex.dma_await_task(%361)
      aiex.dma_await_task(%362)
      aiex.dma_await_task(%363)
      aiex.dma_free_task(%344)
      aiex.dma_free_task(%345)
      aiex.dma_free_task(%346)
      aiex.dma_free_task(%347)
      aiex.dma_free_task(%348)
      aiex.dma_free_task(%349)
      aiex.dma_free_task(%350)
      aiex.dma_free_task(%351)
      aiex.dma_free_task(%352)
      aiex.dma_free_task(%353)
      aiex.dma_free_task(%354)
      aiex.dma_free_task(%355)
      aiex.dma_free_task(%356)
      aiex.dma_free_task(%357)
      aiex.dma_free_task(%358)
      aiex.dma_free_task(%359)
      aiex.dma_free_task(%360)
      aiex.dma_free_task(%361)
      aiex.dma_free_task(%362)
      aiex.dma_free_task(%363)
      aiex.dma_await_task(%376)
      aiex.dma_await_task(%377)
      aiex.dma_await_task(%378)
      aiex.dma_await_task(%379)
      aiex.dma_await_task(%380)
      aiex.dma_await_task(%381)
      aiex.dma_await_task(%382)
      aiex.dma_await_task(%383)
      aiex.dma_free_task(%364)
      aiex.dma_free_task(%365)
      aiex.dma_free_task(%366)
      aiex.dma_free_task(%367)
      aiex.dma_free_task(%368)
      aiex.dma_free_task(%369)
      aiex.dma_free_task(%370)
      aiex.dma_free_task(%371)
      aiex.dma_free_task(%372)
      aiex.dma_free_task(%373)
      aiex.dma_free_task(%374)
      aiex.dma_free_task(%375)
      aiex.dma_free_task(%376)
      aiex.dma_free_task(%377)
      aiex.dma_free_task(%378)
      aiex.dma_free_task(%379)
      aiex.dma_free_task(%380)
      aiex.dma_free_task(%381)
      aiex.dma_free_task(%382)
      aiex.dma_free_task(%383)
      %424 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3145728 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%424)
      %425 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3538944 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%425)
      %426 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3932160 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%426)
      %427 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 4325376 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%427)
      %428 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%428)
      %429 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 49152 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%429)
      %430 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 98304 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%430)
      %431 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 147456 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%431)
      %432 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 196608 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%432)
      %433 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 245760 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%433)
      %434 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 294912 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%434)
      %435 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 344064 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%435)
      %436 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9437184 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%436)
      %437 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9437312 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%437)
      %438 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9437440 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%438)
      %439 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9437568 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%439)
      %440 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9437696 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%440)
      %441 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9437824 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%441)
      %442 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9437952 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%442)
      %443 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9438080 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%443)
      %444 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3145728 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%444)
      %445 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3538944 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%445)
      %446 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3932160 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%446)
      %447 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 4325376 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%447)
      %448 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 393216 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%448)
      %449 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 442368 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%449)
      %450 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 491520 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%450)
      %451 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 540672 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%451)
      %452 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 589824 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%452)
      %453 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 638976 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%453)
      %454 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 688128 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%454)
      %455 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 737280 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%455)
      %456 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9438208 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%456)
      %457 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9438336 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%457)
      %458 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9438464 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%458)
      %459 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9438592 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%459)
      %460 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9438720 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%460)
      %461 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9438848 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%461)
      %462 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9438976 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%462)
      %463 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9439104 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%463)
      aiex.dma_await_task(%396)
      aiex.dma_await_task(%397)
      aiex.dma_await_task(%398)
      aiex.dma_await_task(%399)
      aiex.dma_await_task(%400)
      aiex.dma_await_task(%401)
      aiex.dma_await_task(%402)
      aiex.dma_await_task(%403)
      aiex.dma_free_task(%384)
      aiex.dma_free_task(%385)
      aiex.dma_free_task(%386)
      aiex.dma_free_task(%387)
      aiex.dma_free_task(%388)
      aiex.dma_free_task(%389)
      aiex.dma_free_task(%390)
      aiex.dma_free_task(%391)
      aiex.dma_free_task(%392)
      aiex.dma_free_task(%393)
      aiex.dma_free_task(%394)
      aiex.dma_free_task(%395)
      aiex.dma_free_task(%396)
      aiex.dma_free_task(%397)
      aiex.dma_free_task(%398)
      aiex.dma_free_task(%399)
      aiex.dma_free_task(%400)
      aiex.dma_free_task(%401)
      aiex.dma_free_task(%402)
      aiex.dma_free_task(%403)
      aiex.dma_await_task(%416)
      aiex.dma_await_task(%417)
      aiex.dma_await_task(%418)
      aiex.dma_await_task(%419)
      aiex.dma_await_task(%420)
      aiex.dma_await_task(%421)
      aiex.dma_await_task(%422)
      aiex.dma_await_task(%423)
      aiex.dma_free_task(%404)
      aiex.dma_free_task(%405)
      aiex.dma_free_task(%406)
      aiex.dma_free_task(%407)
      aiex.dma_free_task(%408)
      aiex.dma_free_task(%409)
      aiex.dma_free_task(%410)
      aiex.dma_free_task(%411)
      aiex.dma_free_task(%412)
      aiex.dma_free_task(%413)
      aiex.dma_free_task(%414)
      aiex.dma_free_task(%415)
      aiex.dma_free_task(%416)
      aiex.dma_free_task(%417)
      aiex.dma_free_task(%418)
      aiex.dma_free_task(%419)
      aiex.dma_free_task(%420)
      aiex.dma_free_task(%421)
      aiex.dma_free_task(%422)
      aiex.dma_free_task(%423)
      %464 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3145728 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%464)
      %465 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3538944 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%465)
      %466 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3932160 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%466)
      %467 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 4325376 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%467)
      %468 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 786432 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%468)
      %469 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 835584 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%469)
      %470 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 884736 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%470)
      %471 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 933888 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%471)
      %472 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 983040 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%472)
      %473 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1032192 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%473)
      %474 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1081344 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%474)
      %475 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1130496 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%475)
      %476 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9439232 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%476)
      %477 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9439360 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%477)
      %478 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9439488 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%478)
      %479 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9439616 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%479)
      %480 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9439744 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%480)
      %481 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9439872 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%481)
      %482 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9440000 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%482)
      %483 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9440128 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%483)
      %484 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3145728 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%484)
      %485 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3538944 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%485)
      %486 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3932160 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%486)
      %487 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 4325376 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%487)
      %488 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1179648 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%488)
      %489 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1228800 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%489)
      %490 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1277952 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%490)
      %491 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1327104 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%491)
      %492 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1376256 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%492)
      %493 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1425408 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%493)
      %494 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1474560 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%494)
      %495 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1523712 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%495)
      %496 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9440256 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%496)
      %497 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9440384 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%497)
      %498 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9440512 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%498)
      %499 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9440640 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%499)
      %500 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9440768 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%500)
      %501 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9440896 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%501)
      %502 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9441024 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%502)
      %503 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9441152 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%503)
      aiex.dma_await_task(%436)
      aiex.dma_await_task(%437)
      aiex.dma_await_task(%438)
      aiex.dma_await_task(%439)
      aiex.dma_await_task(%440)
      aiex.dma_await_task(%441)
      aiex.dma_await_task(%442)
      aiex.dma_await_task(%443)
      aiex.dma_free_task(%424)
      aiex.dma_free_task(%425)
      aiex.dma_free_task(%426)
      aiex.dma_free_task(%427)
      aiex.dma_free_task(%428)
      aiex.dma_free_task(%429)
      aiex.dma_free_task(%430)
      aiex.dma_free_task(%431)
      aiex.dma_free_task(%432)
      aiex.dma_free_task(%433)
      aiex.dma_free_task(%434)
      aiex.dma_free_task(%435)
      aiex.dma_free_task(%436)
      aiex.dma_free_task(%437)
      aiex.dma_free_task(%438)
      aiex.dma_free_task(%439)
      aiex.dma_free_task(%440)
      aiex.dma_free_task(%441)
      aiex.dma_free_task(%442)
      aiex.dma_free_task(%443)
      aiex.dma_await_task(%456)
      aiex.dma_await_task(%457)
      aiex.dma_await_task(%458)
      aiex.dma_await_task(%459)
      aiex.dma_await_task(%460)
      aiex.dma_await_task(%461)
      aiex.dma_await_task(%462)
      aiex.dma_await_task(%463)
      aiex.dma_free_task(%444)
      aiex.dma_free_task(%445)
      aiex.dma_free_task(%446)
      aiex.dma_free_task(%447)
      aiex.dma_free_task(%448)
      aiex.dma_free_task(%449)
      aiex.dma_free_task(%450)
      aiex.dma_free_task(%451)
      aiex.dma_free_task(%452)
      aiex.dma_free_task(%453)
      aiex.dma_free_task(%454)
      aiex.dma_free_task(%455)
      aiex.dma_free_task(%456)
      aiex.dma_free_task(%457)
      aiex.dma_free_task(%458)
      aiex.dma_free_task(%459)
      aiex.dma_free_task(%460)
      aiex.dma_free_task(%461)
      aiex.dma_free_task(%462)
      aiex.dma_free_task(%463)
      %504 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3145728 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%504)
      %505 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3538944 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%505)
      %506 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3932160 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%506)
      %507 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 4325376 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%507)
      %508 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1572864 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%508)
      %509 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1622016 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%509)
      %510 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1671168 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%510)
      %511 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1720320 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%511)
      %512 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1769472 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%512)
      %513 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1818624 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%513)
      %514 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1867776 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%514)
      %515 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1916928 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%515)
      %516 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9441280 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%516)
      %517 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9441408 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%517)
      %518 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9441536 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%518)
      %519 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9441664 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%519)
      %520 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9441792 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%520)
      %521 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9441920 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%521)
      %522 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9442048 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%522)
      %523 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9442176 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%523)
      %524 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3145728 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%524)
      %525 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3538944 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%525)
      %526 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3932160 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%526)
      %527 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 4325376 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%527)
      %528 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1966080 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%528)
      %529 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2015232 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%529)
      %530 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2064384 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%530)
      %531 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2113536 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%531)
      %532 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2162688 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%532)
      %533 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2211840 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%533)
      %534 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2260992 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%534)
      %535 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2310144 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%535)
      %536 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9442304 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%536)
      %537 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9442432 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%537)
      %538 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9442560 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%538)
      %539 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9442688 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%539)
      %540 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9442816 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%540)
      %541 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9442944 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%541)
      %542 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9443072 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%542)
      %543 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9443200 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%543)
      aiex.dma_await_task(%476)
      aiex.dma_await_task(%477)
      aiex.dma_await_task(%478)
      aiex.dma_await_task(%479)
      aiex.dma_await_task(%480)
      aiex.dma_await_task(%481)
      aiex.dma_await_task(%482)
      aiex.dma_await_task(%483)
      aiex.dma_free_task(%464)
      aiex.dma_free_task(%465)
      aiex.dma_free_task(%466)
      aiex.dma_free_task(%467)
      aiex.dma_free_task(%468)
      aiex.dma_free_task(%469)
      aiex.dma_free_task(%470)
      aiex.dma_free_task(%471)
      aiex.dma_free_task(%472)
      aiex.dma_free_task(%473)
      aiex.dma_free_task(%474)
      aiex.dma_free_task(%475)
      aiex.dma_free_task(%476)
      aiex.dma_free_task(%477)
      aiex.dma_free_task(%478)
      aiex.dma_free_task(%479)
      aiex.dma_free_task(%480)
      aiex.dma_free_task(%481)
      aiex.dma_free_task(%482)
      aiex.dma_free_task(%483)
      aiex.dma_await_task(%496)
      aiex.dma_await_task(%497)
      aiex.dma_await_task(%498)
      aiex.dma_await_task(%499)
      aiex.dma_await_task(%500)
      aiex.dma_await_task(%501)
      aiex.dma_await_task(%502)
      aiex.dma_await_task(%503)
      aiex.dma_free_task(%484)
      aiex.dma_free_task(%485)
      aiex.dma_free_task(%486)
      aiex.dma_free_task(%487)
      aiex.dma_free_task(%488)
      aiex.dma_free_task(%489)
      aiex.dma_free_task(%490)
      aiex.dma_free_task(%491)
      aiex.dma_free_task(%492)
      aiex.dma_free_task(%493)
      aiex.dma_free_task(%494)
      aiex.dma_free_task(%495)
      aiex.dma_free_task(%496)
      aiex.dma_free_task(%497)
      aiex.dma_free_task(%498)
      aiex.dma_free_task(%499)
      aiex.dma_free_task(%500)
      aiex.dma_free_task(%501)
      aiex.dma_free_task(%502)
      aiex.dma_free_task(%503)
      %544 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3145728 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%544)
      %545 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3538944 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%545)
      %546 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3932160 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%546)
      %547 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 4325376 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%547)
      %548 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2359296 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%548)
      %549 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2408448 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%549)
      %550 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2457600 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%550)
      %551 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2506752 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%551)
      %552 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2555904 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%552)
      %553 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2605056 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%553)
      %554 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2654208 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%554)
      %555 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2703360 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%555)
      %556 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9443328 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%556)
      %557 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9443456 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%557)
      %558 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9443584 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%558)
      %559 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9443712 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%559)
      %560 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9443840 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%560)
      %561 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9443968 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%561)
      %562 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9444096 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%562)
      %563 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9444224 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%563)
      %564 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3145728 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%564)
      %565 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3538944 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%565)
      %566 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3932160 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%566)
      %567 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 4325376 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%567)
      %568 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2752512 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%568)
      %569 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2801664 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%569)
      %570 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2850816 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%570)
      %571 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2899968 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%571)
      %572 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2949120 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%572)
      %573 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2998272 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%573)
      %574 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3047424 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%574)
      %575 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3096576 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%575)
      %576 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9444352 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%576)
      %577 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9444480 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%577)
      %578 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9444608 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%578)
      %579 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9444736 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%579)
      %580 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9444864 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%580)
      %581 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9444992 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%581)
      %582 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9445120 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%582)
      %583 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9445248 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%583)
      aiex.dma_await_task(%516)
      aiex.dma_await_task(%517)
      aiex.dma_await_task(%518)
      aiex.dma_await_task(%519)
      aiex.dma_await_task(%520)
      aiex.dma_await_task(%521)
      aiex.dma_await_task(%522)
      aiex.dma_await_task(%523)
      aiex.dma_free_task(%504)
      aiex.dma_free_task(%505)
      aiex.dma_free_task(%506)
      aiex.dma_free_task(%507)
      aiex.dma_free_task(%508)
      aiex.dma_free_task(%509)
      aiex.dma_free_task(%510)
      aiex.dma_free_task(%511)
      aiex.dma_free_task(%512)
      aiex.dma_free_task(%513)
      aiex.dma_free_task(%514)
      aiex.dma_free_task(%515)
      aiex.dma_free_task(%516)
      aiex.dma_free_task(%517)
      aiex.dma_free_task(%518)
      aiex.dma_free_task(%519)
      aiex.dma_free_task(%520)
      aiex.dma_free_task(%521)
      aiex.dma_free_task(%522)
      aiex.dma_free_task(%523)
      aiex.dma_await_task(%536)
      aiex.dma_await_task(%537)
      aiex.dma_await_task(%538)
      aiex.dma_await_task(%539)
      aiex.dma_await_task(%540)
      aiex.dma_await_task(%541)
      aiex.dma_await_task(%542)
      aiex.dma_await_task(%543)
      aiex.dma_free_task(%524)
      aiex.dma_free_task(%525)
      aiex.dma_free_task(%526)
      aiex.dma_free_task(%527)
      aiex.dma_free_task(%528)
      aiex.dma_free_task(%529)
      aiex.dma_free_task(%530)
      aiex.dma_free_task(%531)
      aiex.dma_free_task(%532)
      aiex.dma_free_task(%533)
      aiex.dma_free_task(%534)
      aiex.dma_free_task(%535)
      aiex.dma_free_task(%536)
      aiex.dma_free_task(%537)
      aiex.dma_free_task(%538)
      aiex.dma_free_task(%539)
      aiex.dma_free_task(%540)
      aiex.dma_free_task(%541)
      aiex.dma_free_task(%542)
      aiex.dma_free_task(%543)
      %584 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3145728 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%584)
      %585 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3538944 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%585)
      %586 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 3932160 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%586)
      %587 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 4325376 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%587)
      %588 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3145728 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%588)
      %589 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3194880 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%589)
      %590 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3244032 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%590)
      %591 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3293184 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%591)
      %592 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3342336 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%592)
      %593 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3391488 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%593)
      %594 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3440640 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%594)
      %595 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3489792 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%595)
      %596 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9445376 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%596)
      %597 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9445504 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%597)
      %598 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9445632 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%598)
      %599 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9445760 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%599)
      %600 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9445888 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%600)
      %601 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9446016 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%601)
      %602 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9446144 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%602)
      %603 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 9446272 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%603)
      %604 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 4718592 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%604)
      %605 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5111808 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%605)
      %606 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5505024 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%606)
      %607 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5898240 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%607)
      %608 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%608)
      %609 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 49152 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%609)
      %610 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 98304 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%610)
      %611 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 147456 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%611)
      %612 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 196608 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%612)
      %613 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 245760 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%613)
      %614 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 294912 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%614)
      %615 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 344064 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%615)
      %616 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14155776 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%616)
      %617 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14155904 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%617)
      %618 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14156032 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%618)
      %619 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14156160 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%619)
      %620 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14156288 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%620)
      %621 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14156416 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%621)
      %622 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14156544 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%622)
      %623 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14156672 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%623)
      aiex.dma_await_task(%556)
      aiex.dma_await_task(%557)
      aiex.dma_await_task(%558)
      aiex.dma_await_task(%559)
      aiex.dma_await_task(%560)
      aiex.dma_await_task(%561)
      aiex.dma_await_task(%562)
      aiex.dma_await_task(%563)
      aiex.dma_free_task(%544)
      aiex.dma_free_task(%545)
      aiex.dma_free_task(%546)
      aiex.dma_free_task(%547)
      aiex.dma_free_task(%548)
      aiex.dma_free_task(%549)
      aiex.dma_free_task(%550)
      aiex.dma_free_task(%551)
      aiex.dma_free_task(%552)
      aiex.dma_free_task(%553)
      aiex.dma_free_task(%554)
      aiex.dma_free_task(%555)
      aiex.dma_free_task(%556)
      aiex.dma_free_task(%557)
      aiex.dma_free_task(%558)
      aiex.dma_free_task(%559)
      aiex.dma_free_task(%560)
      aiex.dma_free_task(%561)
      aiex.dma_free_task(%562)
      aiex.dma_free_task(%563)
      aiex.dma_await_task(%576)
      aiex.dma_await_task(%577)
      aiex.dma_await_task(%578)
      aiex.dma_await_task(%579)
      aiex.dma_await_task(%580)
      aiex.dma_await_task(%581)
      aiex.dma_await_task(%582)
      aiex.dma_await_task(%583)
      aiex.dma_free_task(%564)
      aiex.dma_free_task(%565)
      aiex.dma_free_task(%566)
      aiex.dma_free_task(%567)
      aiex.dma_free_task(%568)
      aiex.dma_free_task(%569)
      aiex.dma_free_task(%570)
      aiex.dma_free_task(%571)
      aiex.dma_free_task(%572)
      aiex.dma_free_task(%573)
      aiex.dma_free_task(%574)
      aiex.dma_free_task(%575)
      aiex.dma_free_task(%576)
      aiex.dma_free_task(%577)
      aiex.dma_free_task(%578)
      aiex.dma_free_task(%579)
      aiex.dma_free_task(%580)
      aiex.dma_free_task(%581)
      aiex.dma_free_task(%582)
      aiex.dma_free_task(%583)
      %624 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 4718592 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%624)
      %625 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5111808 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%625)
      %626 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5505024 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%626)
      %627 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5898240 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%627)
      %628 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 393216 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%628)
      %629 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 442368 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%629)
      %630 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 491520 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%630)
      %631 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 540672 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%631)
      %632 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 589824 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%632)
      %633 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 638976 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%633)
      %634 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 688128 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%634)
      %635 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 737280 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%635)
      %636 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14156800 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%636)
      %637 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14156928 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%637)
      %638 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14157056 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%638)
      %639 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14157184 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%639)
      %640 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14157312 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%640)
      %641 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14157440 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%641)
      %642 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14157568 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%642)
      %643 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14157696 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%643)
      %644 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 4718592 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%644)
      %645 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5111808 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%645)
      %646 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5505024 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%646)
      %647 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5898240 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%647)
      %648 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 786432 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%648)
      %649 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 835584 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%649)
      %650 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 884736 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%650)
      %651 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 933888 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%651)
      %652 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 983040 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%652)
      %653 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1032192 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%653)
      %654 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1081344 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%654)
      %655 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1130496 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%655)
      %656 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14157824 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%656)
      %657 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14157952 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%657)
      %658 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14158080 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%658)
      %659 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14158208 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%659)
      %660 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14158336 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%660)
      %661 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14158464 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%661)
      %662 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14158592 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%662)
      %663 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14158720 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%663)
      aiex.dma_await_task(%596)
      aiex.dma_await_task(%597)
      aiex.dma_await_task(%598)
      aiex.dma_await_task(%599)
      aiex.dma_await_task(%600)
      aiex.dma_await_task(%601)
      aiex.dma_await_task(%602)
      aiex.dma_await_task(%603)
      aiex.dma_free_task(%584)
      aiex.dma_free_task(%585)
      aiex.dma_free_task(%586)
      aiex.dma_free_task(%587)
      aiex.dma_free_task(%588)
      aiex.dma_free_task(%589)
      aiex.dma_free_task(%590)
      aiex.dma_free_task(%591)
      aiex.dma_free_task(%592)
      aiex.dma_free_task(%593)
      aiex.dma_free_task(%594)
      aiex.dma_free_task(%595)
      aiex.dma_free_task(%596)
      aiex.dma_free_task(%597)
      aiex.dma_free_task(%598)
      aiex.dma_free_task(%599)
      aiex.dma_free_task(%600)
      aiex.dma_free_task(%601)
      aiex.dma_free_task(%602)
      aiex.dma_free_task(%603)
      aiex.dma_await_task(%616)
      aiex.dma_await_task(%617)
      aiex.dma_await_task(%618)
      aiex.dma_await_task(%619)
      aiex.dma_await_task(%620)
      aiex.dma_await_task(%621)
      aiex.dma_await_task(%622)
      aiex.dma_await_task(%623)
      aiex.dma_free_task(%604)
      aiex.dma_free_task(%605)
      aiex.dma_free_task(%606)
      aiex.dma_free_task(%607)
      aiex.dma_free_task(%608)
      aiex.dma_free_task(%609)
      aiex.dma_free_task(%610)
      aiex.dma_free_task(%611)
      aiex.dma_free_task(%612)
      aiex.dma_free_task(%613)
      aiex.dma_free_task(%614)
      aiex.dma_free_task(%615)
      aiex.dma_free_task(%616)
      aiex.dma_free_task(%617)
      aiex.dma_free_task(%618)
      aiex.dma_free_task(%619)
      aiex.dma_free_task(%620)
      aiex.dma_free_task(%621)
      aiex.dma_free_task(%622)
      aiex.dma_free_task(%623)
      %664 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 4718592 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%664)
      %665 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5111808 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%665)
      %666 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5505024 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%666)
      %667 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5898240 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%667)
      %668 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1179648 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%668)
      %669 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1228800 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%669)
      %670 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1277952 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%670)
      %671 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1327104 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%671)
      %672 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1376256 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%672)
      %673 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1425408 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%673)
      %674 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1474560 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%674)
      %675 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1523712 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%675)
      %676 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14158848 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%676)
      %677 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14158976 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%677)
      %678 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14159104 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%678)
      %679 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14159232 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%679)
      %680 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14159360 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%680)
      %681 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14159488 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%681)
      %682 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14159616 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%682)
      %683 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14159744 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%683)
      %684 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 4718592 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%684)
      %685 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5111808 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%685)
      %686 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5505024 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%686)
      %687 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5898240 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%687)
      %688 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1572864 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%688)
      %689 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1622016 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%689)
      %690 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1671168 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%690)
      %691 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1720320 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%691)
      %692 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1769472 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%692)
      %693 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1818624 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%693)
      %694 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1867776 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%694)
      %695 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1916928 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%695)
      %696 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14159872 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%696)
      %697 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14160000 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%697)
      %698 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14160128 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%698)
      %699 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14160256 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%699)
      %700 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14160384 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%700)
      %701 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14160512 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%701)
      %702 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14160640 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%702)
      %703 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14160768 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%703)
      aiex.dma_await_task(%636)
      aiex.dma_await_task(%637)
      aiex.dma_await_task(%638)
      aiex.dma_await_task(%639)
      aiex.dma_await_task(%640)
      aiex.dma_await_task(%641)
      aiex.dma_await_task(%642)
      aiex.dma_await_task(%643)
      aiex.dma_free_task(%624)
      aiex.dma_free_task(%625)
      aiex.dma_free_task(%626)
      aiex.dma_free_task(%627)
      aiex.dma_free_task(%628)
      aiex.dma_free_task(%629)
      aiex.dma_free_task(%630)
      aiex.dma_free_task(%631)
      aiex.dma_free_task(%632)
      aiex.dma_free_task(%633)
      aiex.dma_free_task(%634)
      aiex.dma_free_task(%635)
      aiex.dma_free_task(%636)
      aiex.dma_free_task(%637)
      aiex.dma_free_task(%638)
      aiex.dma_free_task(%639)
      aiex.dma_free_task(%640)
      aiex.dma_free_task(%641)
      aiex.dma_free_task(%642)
      aiex.dma_free_task(%643)
      aiex.dma_await_task(%656)
      aiex.dma_await_task(%657)
      aiex.dma_await_task(%658)
      aiex.dma_await_task(%659)
      aiex.dma_await_task(%660)
      aiex.dma_await_task(%661)
      aiex.dma_await_task(%662)
      aiex.dma_await_task(%663)
      aiex.dma_free_task(%644)
      aiex.dma_free_task(%645)
      aiex.dma_free_task(%646)
      aiex.dma_free_task(%647)
      aiex.dma_free_task(%648)
      aiex.dma_free_task(%649)
      aiex.dma_free_task(%650)
      aiex.dma_free_task(%651)
      aiex.dma_free_task(%652)
      aiex.dma_free_task(%653)
      aiex.dma_free_task(%654)
      aiex.dma_free_task(%655)
      aiex.dma_free_task(%656)
      aiex.dma_free_task(%657)
      aiex.dma_free_task(%658)
      aiex.dma_free_task(%659)
      aiex.dma_free_task(%660)
      aiex.dma_free_task(%661)
      aiex.dma_free_task(%662)
      aiex.dma_free_task(%663)
      %704 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 4718592 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%704)
      %705 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5111808 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%705)
      %706 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5505024 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%706)
      %707 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5898240 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%707)
      %708 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1966080 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%708)
      %709 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2015232 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%709)
      %710 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2064384 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%710)
      %711 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2113536 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%711)
      %712 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2162688 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%712)
      %713 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2211840 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%713)
      %714 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2260992 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%714)
      %715 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2310144 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%715)
      %716 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14160896 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%716)
      %717 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14161024 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%717)
      %718 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14161152 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%718)
      %719 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14161280 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%719)
      %720 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14161408 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%720)
      %721 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14161536 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%721)
      %722 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14161664 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%722)
      %723 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14161792 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%723)
      %724 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 4718592 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%724)
      %725 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5111808 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%725)
      %726 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5505024 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%726)
      %727 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5898240 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%727)
      %728 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2359296 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%728)
      %729 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2408448 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%729)
      %730 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2457600 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%730)
      %731 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2506752 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%731)
      %732 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2555904 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%732)
      %733 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2605056 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%733)
      %734 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2654208 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%734)
      %735 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2703360 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%735)
      %736 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14161920 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%736)
      %737 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14162048 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%737)
      %738 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14162176 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%738)
      %739 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14162304 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%739)
      %740 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14162432 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%740)
      %741 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14162560 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%741)
      %742 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14162688 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%742)
      %743 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14162816 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%743)
      aiex.dma_await_task(%676)
      aiex.dma_await_task(%677)
      aiex.dma_await_task(%678)
      aiex.dma_await_task(%679)
      aiex.dma_await_task(%680)
      aiex.dma_await_task(%681)
      aiex.dma_await_task(%682)
      aiex.dma_await_task(%683)
      aiex.dma_free_task(%664)
      aiex.dma_free_task(%665)
      aiex.dma_free_task(%666)
      aiex.dma_free_task(%667)
      aiex.dma_free_task(%668)
      aiex.dma_free_task(%669)
      aiex.dma_free_task(%670)
      aiex.dma_free_task(%671)
      aiex.dma_free_task(%672)
      aiex.dma_free_task(%673)
      aiex.dma_free_task(%674)
      aiex.dma_free_task(%675)
      aiex.dma_free_task(%676)
      aiex.dma_free_task(%677)
      aiex.dma_free_task(%678)
      aiex.dma_free_task(%679)
      aiex.dma_free_task(%680)
      aiex.dma_free_task(%681)
      aiex.dma_free_task(%682)
      aiex.dma_free_task(%683)
      aiex.dma_await_task(%696)
      aiex.dma_await_task(%697)
      aiex.dma_await_task(%698)
      aiex.dma_await_task(%699)
      aiex.dma_await_task(%700)
      aiex.dma_await_task(%701)
      aiex.dma_await_task(%702)
      aiex.dma_await_task(%703)
      aiex.dma_free_task(%684)
      aiex.dma_free_task(%685)
      aiex.dma_free_task(%686)
      aiex.dma_free_task(%687)
      aiex.dma_free_task(%688)
      aiex.dma_free_task(%689)
      aiex.dma_free_task(%690)
      aiex.dma_free_task(%691)
      aiex.dma_free_task(%692)
      aiex.dma_free_task(%693)
      aiex.dma_free_task(%694)
      aiex.dma_free_task(%695)
      aiex.dma_free_task(%696)
      aiex.dma_free_task(%697)
      aiex.dma_free_task(%698)
      aiex.dma_free_task(%699)
      aiex.dma_free_task(%700)
      aiex.dma_free_task(%701)
      aiex.dma_free_task(%702)
      aiex.dma_free_task(%703)
      %744 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 4718592 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%744)
      %745 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5111808 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%745)
      %746 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5505024 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%746)
      %747 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5898240 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%747)
      %748 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2752512 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%748)
      %749 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2801664 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%749)
      %750 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2850816 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%750)
      %751 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2899968 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%751)
      %752 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2949120 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%752)
      %753 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2998272 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%753)
      %754 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3047424 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%754)
      %755 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3096576 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%755)
      %756 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14162944 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%756)
      %757 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14163072 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%757)
      %758 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14163200 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%758)
      %759 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14163328 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%759)
      %760 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14163456 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%760)
      %761 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14163584 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%761)
      %762 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14163712 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%762)
      %763 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14163840 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%763)
      %764 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 4718592 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%764)
      %765 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5111808 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%765)
      %766 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5505024 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%766)
      %767 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 5898240 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%767)
      %768 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3145728 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%768)
      %769 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3194880 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%769)
      %770 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3244032 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%770)
      %771 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3293184 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%771)
      %772 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3342336 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%772)
      %773 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3391488 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%773)
      %774 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3440640 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%774)
      %775 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3489792 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%775)
      %776 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14163968 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%776)
      %777 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14164096 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%777)
      %778 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14164224 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%778)
      %779 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14164352 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%779)
      %780 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14164480 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%780)
      %781 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14164608 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%781)
      %782 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14164736 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%782)
      %783 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 14164864 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%783)
      aiex.dma_await_task(%716)
      aiex.dma_await_task(%717)
      aiex.dma_await_task(%718)
      aiex.dma_await_task(%719)
      aiex.dma_await_task(%720)
      aiex.dma_await_task(%721)
      aiex.dma_await_task(%722)
      aiex.dma_await_task(%723)
      aiex.dma_free_task(%704)
      aiex.dma_free_task(%705)
      aiex.dma_free_task(%706)
      aiex.dma_free_task(%707)
      aiex.dma_free_task(%708)
      aiex.dma_free_task(%709)
      aiex.dma_free_task(%710)
      aiex.dma_free_task(%711)
      aiex.dma_free_task(%712)
      aiex.dma_free_task(%713)
      aiex.dma_free_task(%714)
      aiex.dma_free_task(%715)
      aiex.dma_free_task(%716)
      aiex.dma_free_task(%717)
      aiex.dma_free_task(%718)
      aiex.dma_free_task(%719)
      aiex.dma_free_task(%720)
      aiex.dma_free_task(%721)
      aiex.dma_free_task(%722)
      aiex.dma_free_task(%723)
      aiex.dma_await_task(%736)
      aiex.dma_await_task(%737)
      aiex.dma_await_task(%738)
      aiex.dma_await_task(%739)
      aiex.dma_await_task(%740)
      aiex.dma_await_task(%741)
      aiex.dma_await_task(%742)
      aiex.dma_await_task(%743)
      aiex.dma_free_task(%724)
      aiex.dma_free_task(%725)
      aiex.dma_free_task(%726)
      aiex.dma_free_task(%727)
      aiex.dma_free_task(%728)
      aiex.dma_free_task(%729)
      aiex.dma_free_task(%730)
      aiex.dma_free_task(%731)
      aiex.dma_free_task(%732)
      aiex.dma_free_task(%733)
      aiex.dma_free_task(%734)
      aiex.dma_free_task(%735)
      aiex.dma_free_task(%736)
      aiex.dma_free_task(%737)
      aiex.dma_free_task(%738)
      aiex.dma_free_task(%739)
      aiex.dma_free_task(%740)
      aiex.dma_free_task(%741)
      aiex.dma_free_task(%742)
      aiex.dma_free_task(%743)
      %784 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 6291456 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%784)
      %785 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 6684672 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%785)
      %786 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7077888 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%786)
      %787 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7471104 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%787)
      %788 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%788)
      %789 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 49152 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%789)
      %790 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 98304 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%790)
      %791 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 147456 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%791)
      %792 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 196608 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%792)
      %793 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 245760 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%793)
      %794 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 294912 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%794)
      %795 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 344064 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%795)
      %796 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18874368 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%796)
      %797 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18874496 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%797)
      %798 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18874624 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%798)
      %799 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18874752 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%799)
      %800 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18874880 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%800)
      %801 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18875008 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%801)
      %802 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18875136 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%802)
      %803 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18875264 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%803)
      %804 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 6291456 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%804)
      %805 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 6684672 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%805)
      %806 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7077888 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%806)
      %807 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7471104 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%807)
      %808 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 393216 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%808)
      %809 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 442368 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%809)
      %810 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 491520 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%810)
      %811 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 540672 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%811)
      %812 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 589824 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%812)
      %813 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 638976 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%813)
      %814 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 688128 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%814)
      %815 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 737280 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%815)
      %816 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18875392 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%816)
      %817 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18875520 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%817)
      %818 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18875648 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%818)
      %819 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18875776 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%819)
      %820 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18875904 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%820)
      %821 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18876032 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%821)
      %822 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18876160 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%822)
      %823 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18876288 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%823)
      aiex.dma_await_task(%756)
      aiex.dma_await_task(%757)
      aiex.dma_await_task(%758)
      aiex.dma_await_task(%759)
      aiex.dma_await_task(%760)
      aiex.dma_await_task(%761)
      aiex.dma_await_task(%762)
      aiex.dma_await_task(%763)
      aiex.dma_free_task(%744)
      aiex.dma_free_task(%745)
      aiex.dma_free_task(%746)
      aiex.dma_free_task(%747)
      aiex.dma_free_task(%748)
      aiex.dma_free_task(%749)
      aiex.dma_free_task(%750)
      aiex.dma_free_task(%751)
      aiex.dma_free_task(%752)
      aiex.dma_free_task(%753)
      aiex.dma_free_task(%754)
      aiex.dma_free_task(%755)
      aiex.dma_free_task(%756)
      aiex.dma_free_task(%757)
      aiex.dma_free_task(%758)
      aiex.dma_free_task(%759)
      aiex.dma_free_task(%760)
      aiex.dma_free_task(%761)
      aiex.dma_free_task(%762)
      aiex.dma_free_task(%763)
      aiex.dma_await_task(%776)
      aiex.dma_await_task(%777)
      aiex.dma_await_task(%778)
      aiex.dma_await_task(%779)
      aiex.dma_await_task(%780)
      aiex.dma_await_task(%781)
      aiex.dma_await_task(%782)
      aiex.dma_await_task(%783)
      aiex.dma_free_task(%764)
      aiex.dma_free_task(%765)
      aiex.dma_free_task(%766)
      aiex.dma_free_task(%767)
      aiex.dma_free_task(%768)
      aiex.dma_free_task(%769)
      aiex.dma_free_task(%770)
      aiex.dma_free_task(%771)
      aiex.dma_free_task(%772)
      aiex.dma_free_task(%773)
      aiex.dma_free_task(%774)
      aiex.dma_free_task(%775)
      aiex.dma_free_task(%776)
      aiex.dma_free_task(%777)
      aiex.dma_free_task(%778)
      aiex.dma_free_task(%779)
      aiex.dma_free_task(%780)
      aiex.dma_free_task(%781)
      aiex.dma_free_task(%782)
      aiex.dma_free_task(%783)
      %824 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 6291456 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%824)
      %825 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 6684672 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%825)
      %826 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7077888 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%826)
      %827 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7471104 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%827)
      %828 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 786432 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%828)
      %829 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 835584 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%829)
      %830 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 884736 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%830)
      %831 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 933888 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%831)
      %832 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 983040 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%832)
      %833 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1032192 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%833)
      %834 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1081344 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%834)
      %835 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1130496 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%835)
      %836 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18876416 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%836)
      %837 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18876544 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%837)
      %838 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18876672 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%838)
      %839 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18876800 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%839)
      %840 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18876928 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%840)
      %841 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18877056 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%841)
      %842 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18877184 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%842)
      %843 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18877312 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%843)
      %844 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 6291456 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%844)
      %845 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 6684672 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%845)
      %846 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7077888 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%846)
      %847 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7471104 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%847)
      %848 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1179648 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%848)
      %849 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1228800 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%849)
      %850 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1277952 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%850)
      %851 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1327104 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%851)
      %852 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1376256 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%852)
      %853 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1425408 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%853)
      %854 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1474560 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%854)
      %855 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1523712 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%855)
      %856 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18877440 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%856)
      %857 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18877568 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%857)
      %858 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18877696 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%858)
      %859 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18877824 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%859)
      %860 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18877952 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%860)
      %861 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18878080 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%861)
      %862 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18878208 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%862)
      %863 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18878336 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%863)
      aiex.dma_await_task(%796)
      aiex.dma_await_task(%797)
      aiex.dma_await_task(%798)
      aiex.dma_await_task(%799)
      aiex.dma_await_task(%800)
      aiex.dma_await_task(%801)
      aiex.dma_await_task(%802)
      aiex.dma_await_task(%803)
      aiex.dma_free_task(%784)
      aiex.dma_free_task(%785)
      aiex.dma_free_task(%786)
      aiex.dma_free_task(%787)
      aiex.dma_free_task(%788)
      aiex.dma_free_task(%789)
      aiex.dma_free_task(%790)
      aiex.dma_free_task(%791)
      aiex.dma_free_task(%792)
      aiex.dma_free_task(%793)
      aiex.dma_free_task(%794)
      aiex.dma_free_task(%795)
      aiex.dma_free_task(%796)
      aiex.dma_free_task(%797)
      aiex.dma_free_task(%798)
      aiex.dma_free_task(%799)
      aiex.dma_free_task(%800)
      aiex.dma_free_task(%801)
      aiex.dma_free_task(%802)
      aiex.dma_free_task(%803)
      aiex.dma_await_task(%816)
      aiex.dma_await_task(%817)
      aiex.dma_await_task(%818)
      aiex.dma_await_task(%819)
      aiex.dma_await_task(%820)
      aiex.dma_await_task(%821)
      aiex.dma_await_task(%822)
      aiex.dma_await_task(%823)
      aiex.dma_free_task(%804)
      aiex.dma_free_task(%805)
      aiex.dma_free_task(%806)
      aiex.dma_free_task(%807)
      aiex.dma_free_task(%808)
      aiex.dma_free_task(%809)
      aiex.dma_free_task(%810)
      aiex.dma_free_task(%811)
      aiex.dma_free_task(%812)
      aiex.dma_free_task(%813)
      aiex.dma_free_task(%814)
      aiex.dma_free_task(%815)
      aiex.dma_free_task(%816)
      aiex.dma_free_task(%817)
      aiex.dma_free_task(%818)
      aiex.dma_free_task(%819)
      aiex.dma_free_task(%820)
      aiex.dma_free_task(%821)
      aiex.dma_free_task(%822)
      aiex.dma_free_task(%823)
      %864 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 6291456 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%864)
      %865 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 6684672 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%865)
      %866 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7077888 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%866)
      %867 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7471104 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%867)
      %868 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1572864 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%868)
      %869 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1622016 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%869)
      %870 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1671168 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%870)
      %871 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1720320 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%871)
      %872 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1769472 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%872)
      %873 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1818624 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%873)
      %874 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1867776 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%874)
      %875 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1916928 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%875)
      %876 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18878464 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%876)
      %877 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18878592 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%877)
      %878 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18878720 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%878)
      %879 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18878848 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%879)
      %880 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18878976 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%880)
      %881 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18879104 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%881)
      %882 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18879232 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%882)
      %883 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18879360 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%883)
      %884 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 6291456 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%884)
      %885 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 6684672 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%885)
      %886 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7077888 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%886)
      %887 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7471104 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%887)
      %888 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1966080 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%888)
      %889 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2015232 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%889)
      %890 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2064384 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%890)
      %891 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2113536 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%891)
      %892 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2162688 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%892)
      %893 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2211840 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%893)
      %894 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2260992 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%894)
      %895 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2310144 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%895)
      %896 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18879488 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%896)
      %897 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18879616 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%897)
      %898 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18879744 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%898)
      %899 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18879872 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%899)
      %900 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18880000 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%900)
      %901 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18880128 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%901)
      %902 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18880256 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%902)
      %903 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18880384 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%903)
      aiex.dma_await_task(%836)
      aiex.dma_await_task(%837)
      aiex.dma_await_task(%838)
      aiex.dma_await_task(%839)
      aiex.dma_await_task(%840)
      aiex.dma_await_task(%841)
      aiex.dma_await_task(%842)
      aiex.dma_await_task(%843)
      aiex.dma_free_task(%824)
      aiex.dma_free_task(%825)
      aiex.dma_free_task(%826)
      aiex.dma_free_task(%827)
      aiex.dma_free_task(%828)
      aiex.dma_free_task(%829)
      aiex.dma_free_task(%830)
      aiex.dma_free_task(%831)
      aiex.dma_free_task(%832)
      aiex.dma_free_task(%833)
      aiex.dma_free_task(%834)
      aiex.dma_free_task(%835)
      aiex.dma_free_task(%836)
      aiex.dma_free_task(%837)
      aiex.dma_free_task(%838)
      aiex.dma_free_task(%839)
      aiex.dma_free_task(%840)
      aiex.dma_free_task(%841)
      aiex.dma_free_task(%842)
      aiex.dma_free_task(%843)
      aiex.dma_await_task(%856)
      aiex.dma_await_task(%857)
      aiex.dma_await_task(%858)
      aiex.dma_await_task(%859)
      aiex.dma_await_task(%860)
      aiex.dma_await_task(%861)
      aiex.dma_await_task(%862)
      aiex.dma_await_task(%863)
      aiex.dma_free_task(%844)
      aiex.dma_free_task(%845)
      aiex.dma_free_task(%846)
      aiex.dma_free_task(%847)
      aiex.dma_free_task(%848)
      aiex.dma_free_task(%849)
      aiex.dma_free_task(%850)
      aiex.dma_free_task(%851)
      aiex.dma_free_task(%852)
      aiex.dma_free_task(%853)
      aiex.dma_free_task(%854)
      aiex.dma_free_task(%855)
      aiex.dma_free_task(%856)
      aiex.dma_free_task(%857)
      aiex.dma_free_task(%858)
      aiex.dma_free_task(%859)
      aiex.dma_free_task(%860)
      aiex.dma_free_task(%861)
      aiex.dma_free_task(%862)
      aiex.dma_free_task(%863)
      %904 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 6291456 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%904)
      %905 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 6684672 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%905)
      %906 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7077888 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%906)
      %907 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7471104 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%907)
      %908 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2359296 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%908)
      %909 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2408448 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%909)
      %910 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2457600 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%910)
      %911 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2506752 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%911)
      %912 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2555904 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%912)
      %913 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2605056 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%913)
      %914 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2654208 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%914)
      %915 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2703360 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%915)
      %916 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18880512 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%916)
      %917 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18880640 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%917)
      %918 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18880768 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%918)
      %919 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18880896 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%919)
      %920 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18881024 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%920)
      %921 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18881152 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%921)
      %922 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18881280 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%922)
      %923 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18881408 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%923)
      %924 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 6291456 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%924)
      %925 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 6684672 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%925)
      %926 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7077888 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%926)
      %927 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7471104 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%927)
      %928 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2752512 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%928)
      %929 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2801664 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%929)
      %930 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2850816 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%930)
      %931 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2899968 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%931)
      %932 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2949120 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%932)
      %933 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2998272 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%933)
      %934 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3047424 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%934)
      %935 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3096576 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%935)
      %936 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18881536 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%936)
      %937 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18881664 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%937)
      %938 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18881792 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%938)
      %939 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18881920 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%939)
      %940 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18882048 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%940)
      %941 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18882176 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%941)
      %942 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18882304 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%942)
      %943 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18882432 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%943)
      aiex.dma_await_task(%876)
      aiex.dma_await_task(%877)
      aiex.dma_await_task(%878)
      aiex.dma_await_task(%879)
      aiex.dma_await_task(%880)
      aiex.dma_await_task(%881)
      aiex.dma_await_task(%882)
      aiex.dma_await_task(%883)
      aiex.dma_free_task(%864)
      aiex.dma_free_task(%865)
      aiex.dma_free_task(%866)
      aiex.dma_free_task(%867)
      aiex.dma_free_task(%868)
      aiex.dma_free_task(%869)
      aiex.dma_free_task(%870)
      aiex.dma_free_task(%871)
      aiex.dma_free_task(%872)
      aiex.dma_free_task(%873)
      aiex.dma_free_task(%874)
      aiex.dma_free_task(%875)
      aiex.dma_free_task(%876)
      aiex.dma_free_task(%877)
      aiex.dma_free_task(%878)
      aiex.dma_free_task(%879)
      aiex.dma_free_task(%880)
      aiex.dma_free_task(%881)
      aiex.dma_free_task(%882)
      aiex.dma_free_task(%883)
      aiex.dma_await_task(%896)
      aiex.dma_await_task(%897)
      aiex.dma_await_task(%898)
      aiex.dma_await_task(%899)
      aiex.dma_await_task(%900)
      aiex.dma_await_task(%901)
      aiex.dma_await_task(%902)
      aiex.dma_await_task(%903)
      aiex.dma_free_task(%884)
      aiex.dma_free_task(%885)
      aiex.dma_free_task(%886)
      aiex.dma_free_task(%887)
      aiex.dma_free_task(%888)
      aiex.dma_free_task(%889)
      aiex.dma_free_task(%890)
      aiex.dma_free_task(%891)
      aiex.dma_free_task(%892)
      aiex.dma_free_task(%893)
      aiex.dma_free_task(%894)
      aiex.dma_free_task(%895)
      aiex.dma_free_task(%896)
      aiex.dma_free_task(%897)
      aiex.dma_free_task(%898)
      aiex.dma_free_task(%899)
      aiex.dma_free_task(%900)
      aiex.dma_free_task(%901)
      aiex.dma_free_task(%902)
      aiex.dma_free_task(%903)
      %944 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 6291456 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%944)
      %945 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 6684672 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%945)
      %946 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7077888 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%946)
      %947 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7471104 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%947)
      %948 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3145728 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%948)
      %949 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3194880 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%949)
      %950 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3244032 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%950)
      %951 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3293184 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%951)
      %952 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3342336 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%952)
      %953 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3391488 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%953)
      %954 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3440640 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%954)
      %955 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3489792 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%955)
      %956 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18882560 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%956)
      %957 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18882688 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%957)
      %958 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18882816 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%958)
      %959 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18882944 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%959)
      %960 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18883072 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%960)
      %961 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18883200 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%961)
      %962 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18883328 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%962)
      %963 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 18883456 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%963)
      %964 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7864320 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%964)
      %965 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 8257536 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%965)
      %966 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 8650752 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%966)
      %967 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9043968 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%967)
      %968 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%968)
      %969 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 49152 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%969)
      %970 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 98304 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%970)
      %971 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 147456 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%971)
      %972 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 196608 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%972)
      %973 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 245760 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%973)
      %974 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 294912 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%974)
      %975 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 344064 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%975)
      %976 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23592960 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%976)
      %977 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23593088 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%977)
      %978 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23593216 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%978)
      %979 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23593344 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%979)
      %980 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23593472 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%980)
      %981 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23593600 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%981)
      %982 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23593728 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%982)
      %983 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23593856 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%983)
      aiex.dma_await_task(%916)
      aiex.dma_await_task(%917)
      aiex.dma_await_task(%918)
      aiex.dma_await_task(%919)
      aiex.dma_await_task(%920)
      aiex.dma_await_task(%921)
      aiex.dma_await_task(%922)
      aiex.dma_await_task(%923)
      aiex.dma_free_task(%904)
      aiex.dma_free_task(%905)
      aiex.dma_free_task(%906)
      aiex.dma_free_task(%907)
      aiex.dma_free_task(%908)
      aiex.dma_free_task(%909)
      aiex.dma_free_task(%910)
      aiex.dma_free_task(%911)
      aiex.dma_free_task(%912)
      aiex.dma_free_task(%913)
      aiex.dma_free_task(%914)
      aiex.dma_free_task(%915)
      aiex.dma_free_task(%916)
      aiex.dma_free_task(%917)
      aiex.dma_free_task(%918)
      aiex.dma_free_task(%919)
      aiex.dma_free_task(%920)
      aiex.dma_free_task(%921)
      aiex.dma_free_task(%922)
      aiex.dma_free_task(%923)
      aiex.dma_await_task(%936)
      aiex.dma_await_task(%937)
      aiex.dma_await_task(%938)
      aiex.dma_await_task(%939)
      aiex.dma_await_task(%940)
      aiex.dma_await_task(%941)
      aiex.dma_await_task(%942)
      aiex.dma_await_task(%943)
      aiex.dma_free_task(%924)
      aiex.dma_free_task(%925)
      aiex.dma_free_task(%926)
      aiex.dma_free_task(%927)
      aiex.dma_free_task(%928)
      aiex.dma_free_task(%929)
      aiex.dma_free_task(%930)
      aiex.dma_free_task(%931)
      aiex.dma_free_task(%932)
      aiex.dma_free_task(%933)
      aiex.dma_free_task(%934)
      aiex.dma_free_task(%935)
      aiex.dma_free_task(%936)
      aiex.dma_free_task(%937)
      aiex.dma_free_task(%938)
      aiex.dma_free_task(%939)
      aiex.dma_free_task(%940)
      aiex.dma_free_task(%941)
      aiex.dma_free_task(%942)
      aiex.dma_free_task(%943)
      %984 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7864320 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%984)
      %985 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 8257536 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%985)
      %986 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 8650752 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%986)
      %987 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9043968 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%987)
      %988 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 393216 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%988)
      %989 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 442368 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%989)
      %990 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 491520 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%990)
      %991 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 540672 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%991)
      %992 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 589824 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%992)
      %993 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 638976 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%993)
      %994 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 688128 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%994)
      %995 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 737280 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%995)
      %996 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23593984 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%996)
      %997 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23594112 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%997)
      %998 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23594240 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%998)
      %999 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23594368 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%999)
      %1000 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23594496 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1000)
      %1001 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23594624 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1001)
      %1002 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23594752 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1002)
      %1003 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23594880 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1003)
      %1004 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7864320 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1004)
      %1005 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 8257536 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1005)
      %1006 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 8650752 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1006)
      %1007 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9043968 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1007)
      %1008 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 786432 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1008)
      %1009 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 835584 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1009)
      %1010 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 884736 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1010)
      %1011 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 933888 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1011)
      %1012 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 983040 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1012)
      %1013 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1032192 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1013)
      %1014 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1081344 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1014)
      %1015 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1130496 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1015)
      %1016 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23595008 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1016)
      %1017 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23595136 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1017)
      %1018 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23595264 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1018)
      %1019 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23595392 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1019)
      %1020 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23595520 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1020)
      %1021 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23595648 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1021)
      %1022 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23595776 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1022)
      %1023 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23595904 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1023)
      aiex.dma_await_task(%956)
      aiex.dma_await_task(%957)
      aiex.dma_await_task(%958)
      aiex.dma_await_task(%959)
      aiex.dma_await_task(%960)
      aiex.dma_await_task(%961)
      aiex.dma_await_task(%962)
      aiex.dma_await_task(%963)
      aiex.dma_free_task(%944)
      aiex.dma_free_task(%945)
      aiex.dma_free_task(%946)
      aiex.dma_free_task(%947)
      aiex.dma_free_task(%948)
      aiex.dma_free_task(%949)
      aiex.dma_free_task(%950)
      aiex.dma_free_task(%951)
      aiex.dma_free_task(%952)
      aiex.dma_free_task(%953)
      aiex.dma_free_task(%954)
      aiex.dma_free_task(%955)
      aiex.dma_free_task(%956)
      aiex.dma_free_task(%957)
      aiex.dma_free_task(%958)
      aiex.dma_free_task(%959)
      aiex.dma_free_task(%960)
      aiex.dma_free_task(%961)
      aiex.dma_free_task(%962)
      aiex.dma_free_task(%963)
      aiex.dma_await_task(%976)
      aiex.dma_await_task(%977)
      aiex.dma_await_task(%978)
      aiex.dma_await_task(%979)
      aiex.dma_await_task(%980)
      aiex.dma_await_task(%981)
      aiex.dma_await_task(%982)
      aiex.dma_await_task(%983)
      aiex.dma_free_task(%964)
      aiex.dma_free_task(%965)
      aiex.dma_free_task(%966)
      aiex.dma_free_task(%967)
      aiex.dma_free_task(%968)
      aiex.dma_free_task(%969)
      aiex.dma_free_task(%970)
      aiex.dma_free_task(%971)
      aiex.dma_free_task(%972)
      aiex.dma_free_task(%973)
      aiex.dma_free_task(%974)
      aiex.dma_free_task(%975)
      aiex.dma_free_task(%976)
      aiex.dma_free_task(%977)
      aiex.dma_free_task(%978)
      aiex.dma_free_task(%979)
      aiex.dma_free_task(%980)
      aiex.dma_free_task(%981)
      aiex.dma_free_task(%982)
      aiex.dma_free_task(%983)
      %1024 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7864320 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1024)
      %1025 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 8257536 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1025)
      %1026 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 8650752 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1026)
      %1027 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9043968 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1027)
      %1028 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1179648 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1028)
      %1029 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1228800 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1029)
      %1030 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1277952 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1030)
      %1031 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1327104 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1031)
      %1032 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1376256 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1032)
      %1033 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1425408 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1033)
      %1034 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1474560 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1034)
      %1035 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1523712 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1035)
      %1036 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23596032 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1036)
      %1037 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23596160 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1037)
      %1038 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23596288 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1038)
      %1039 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23596416 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1039)
      %1040 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23596544 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1040)
      %1041 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23596672 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1041)
      %1042 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23596800 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1042)
      %1043 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23596928 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1043)
      %1044 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7864320 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1044)
      %1045 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 8257536 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1045)
      %1046 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 8650752 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1046)
      %1047 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9043968 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1047)
      %1048 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1572864 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1048)
      %1049 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1622016 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1049)
      %1050 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1671168 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1050)
      %1051 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1720320 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1051)
      %1052 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1769472 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1052)
      %1053 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1818624 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1053)
      %1054 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1867776 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1054)
      %1055 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1916928 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1055)
      %1056 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23597056 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1056)
      %1057 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23597184 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1057)
      %1058 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23597312 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1058)
      %1059 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23597440 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1059)
      %1060 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23597568 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1060)
      %1061 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23597696 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1061)
      %1062 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23597824 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1062)
      %1063 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23597952 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1063)
      aiex.dma_await_task(%996)
      aiex.dma_await_task(%997)
      aiex.dma_await_task(%998)
      aiex.dma_await_task(%999)
      aiex.dma_await_task(%1000)
      aiex.dma_await_task(%1001)
      aiex.dma_await_task(%1002)
      aiex.dma_await_task(%1003)
      aiex.dma_free_task(%984)
      aiex.dma_free_task(%985)
      aiex.dma_free_task(%986)
      aiex.dma_free_task(%987)
      aiex.dma_free_task(%988)
      aiex.dma_free_task(%989)
      aiex.dma_free_task(%990)
      aiex.dma_free_task(%991)
      aiex.dma_free_task(%992)
      aiex.dma_free_task(%993)
      aiex.dma_free_task(%994)
      aiex.dma_free_task(%995)
      aiex.dma_free_task(%996)
      aiex.dma_free_task(%997)
      aiex.dma_free_task(%998)
      aiex.dma_free_task(%999)
      aiex.dma_free_task(%1000)
      aiex.dma_free_task(%1001)
      aiex.dma_free_task(%1002)
      aiex.dma_free_task(%1003)
      aiex.dma_await_task(%1016)
      aiex.dma_await_task(%1017)
      aiex.dma_await_task(%1018)
      aiex.dma_await_task(%1019)
      aiex.dma_await_task(%1020)
      aiex.dma_await_task(%1021)
      aiex.dma_await_task(%1022)
      aiex.dma_await_task(%1023)
      aiex.dma_free_task(%1004)
      aiex.dma_free_task(%1005)
      aiex.dma_free_task(%1006)
      aiex.dma_free_task(%1007)
      aiex.dma_free_task(%1008)
      aiex.dma_free_task(%1009)
      aiex.dma_free_task(%1010)
      aiex.dma_free_task(%1011)
      aiex.dma_free_task(%1012)
      aiex.dma_free_task(%1013)
      aiex.dma_free_task(%1014)
      aiex.dma_free_task(%1015)
      aiex.dma_free_task(%1016)
      aiex.dma_free_task(%1017)
      aiex.dma_free_task(%1018)
      aiex.dma_free_task(%1019)
      aiex.dma_free_task(%1020)
      aiex.dma_free_task(%1021)
      aiex.dma_free_task(%1022)
      aiex.dma_free_task(%1023)
      %1064 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7864320 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1064)
      %1065 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 8257536 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1065)
      %1066 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 8650752 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1066)
      %1067 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9043968 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1067)
      %1068 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1966080 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1068)
      %1069 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2015232 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1069)
      %1070 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2064384 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1070)
      %1071 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2113536 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1071)
      %1072 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2162688 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1072)
      %1073 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2211840 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1073)
      %1074 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2260992 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1074)
      %1075 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2310144 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1075)
      %1076 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23598080 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1076)
      %1077 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23598208 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1077)
      %1078 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23598336 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1078)
      %1079 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23598464 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1079)
      %1080 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23598592 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1080)
      %1081 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23598720 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1081)
      %1082 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23598848 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1082)
      %1083 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23598976 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1083)
      %1084 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7864320 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1084)
      %1085 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 8257536 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1085)
      %1086 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 8650752 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1086)
      %1087 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9043968 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1087)
      %1088 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2359296 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1088)
      %1089 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2408448 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1089)
      %1090 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2457600 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1090)
      %1091 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2506752 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1091)
      %1092 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2555904 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1092)
      %1093 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2605056 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1093)
      %1094 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2654208 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1094)
      %1095 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2703360 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1095)
      %1096 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23599104 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1096)
      %1097 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23599232 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1097)
      %1098 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23599360 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1098)
      %1099 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23599488 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1099)
      %1100 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23599616 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1100)
      %1101 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23599744 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1101)
      %1102 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23599872 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1102)
      %1103 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23600000 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1103)
      aiex.dma_await_task(%1036)
      aiex.dma_await_task(%1037)
      aiex.dma_await_task(%1038)
      aiex.dma_await_task(%1039)
      aiex.dma_await_task(%1040)
      aiex.dma_await_task(%1041)
      aiex.dma_await_task(%1042)
      aiex.dma_await_task(%1043)
      aiex.dma_free_task(%1024)
      aiex.dma_free_task(%1025)
      aiex.dma_free_task(%1026)
      aiex.dma_free_task(%1027)
      aiex.dma_free_task(%1028)
      aiex.dma_free_task(%1029)
      aiex.dma_free_task(%1030)
      aiex.dma_free_task(%1031)
      aiex.dma_free_task(%1032)
      aiex.dma_free_task(%1033)
      aiex.dma_free_task(%1034)
      aiex.dma_free_task(%1035)
      aiex.dma_free_task(%1036)
      aiex.dma_free_task(%1037)
      aiex.dma_free_task(%1038)
      aiex.dma_free_task(%1039)
      aiex.dma_free_task(%1040)
      aiex.dma_free_task(%1041)
      aiex.dma_free_task(%1042)
      aiex.dma_free_task(%1043)
      aiex.dma_await_task(%1056)
      aiex.dma_await_task(%1057)
      aiex.dma_await_task(%1058)
      aiex.dma_await_task(%1059)
      aiex.dma_await_task(%1060)
      aiex.dma_await_task(%1061)
      aiex.dma_await_task(%1062)
      aiex.dma_await_task(%1063)
      aiex.dma_free_task(%1044)
      aiex.dma_free_task(%1045)
      aiex.dma_free_task(%1046)
      aiex.dma_free_task(%1047)
      aiex.dma_free_task(%1048)
      aiex.dma_free_task(%1049)
      aiex.dma_free_task(%1050)
      aiex.dma_free_task(%1051)
      aiex.dma_free_task(%1052)
      aiex.dma_free_task(%1053)
      aiex.dma_free_task(%1054)
      aiex.dma_free_task(%1055)
      aiex.dma_free_task(%1056)
      aiex.dma_free_task(%1057)
      aiex.dma_free_task(%1058)
      aiex.dma_free_task(%1059)
      aiex.dma_free_task(%1060)
      aiex.dma_free_task(%1061)
      aiex.dma_free_task(%1062)
      aiex.dma_free_task(%1063)
      %1104 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7864320 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1104)
      %1105 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 8257536 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1105)
      %1106 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 8650752 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1106)
      %1107 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9043968 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1107)
      %1108 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2752512 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1108)
      %1109 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2801664 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1109)
      %1110 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2850816 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1110)
      %1111 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2899968 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1111)
      %1112 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2949120 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1112)
      %1113 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2998272 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1113)
      %1114 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3047424 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1114)
      %1115 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3096576 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1115)
      %1116 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23600128 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1116)
      %1117 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23600256 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1117)
      %1118 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23600384 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1118)
      %1119 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23600512 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1119)
      %1120 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23600640 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1120)
      %1121 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23600768 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1121)
      %1122 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23600896 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1122)
      %1123 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23601024 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1123)
      %1124 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 7864320 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1124)
      %1125 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 8257536 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1125)
      %1126 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 8650752 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1126)
      %1127 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9043968 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1127)
      %1128 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3145728 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1128)
      %1129 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3194880 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1129)
      %1130 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3244032 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1130)
      %1131 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3293184 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1131)
      %1132 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3342336 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1132)
      %1133 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3391488 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1133)
      %1134 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3440640 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1134)
      %1135 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3489792 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1135)
      %1136 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23601152 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1136)
      %1137 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23601280 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1137)
      %1138 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23601408 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1138)
      %1139 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23601536 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1139)
      %1140 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23601664 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1140)
      %1141 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23601792 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1141)
      %1142 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23601920 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1142)
      %1143 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 23602048 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1143)
      aiex.dma_await_task(%1076)
      aiex.dma_await_task(%1077)
      aiex.dma_await_task(%1078)
      aiex.dma_await_task(%1079)
      aiex.dma_await_task(%1080)
      aiex.dma_await_task(%1081)
      aiex.dma_await_task(%1082)
      aiex.dma_await_task(%1083)
      aiex.dma_free_task(%1064)
      aiex.dma_free_task(%1065)
      aiex.dma_free_task(%1066)
      aiex.dma_free_task(%1067)
      aiex.dma_free_task(%1068)
      aiex.dma_free_task(%1069)
      aiex.dma_free_task(%1070)
      aiex.dma_free_task(%1071)
      aiex.dma_free_task(%1072)
      aiex.dma_free_task(%1073)
      aiex.dma_free_task(%1074)
      aiex.dma_free_task(%1075)
      aiex.dma_free_task(%1076)
      aiex.dma_free_task(%1077)
      aiex.dma_free_task(%1078)
      aiex.dma_free_task(%1079)
      aiex.dma_free_task(%1080)
      aiex.dma_free_task(%1081)
      aiex.dma_free_task(%1082)
      aiex.dma_free_task(%1083)
      aiex.dma_await_task(%1096)
      aiex.dma_await_task(%1097)
      aiex.dma_await_task(%1098)
      aiex.dma_await_task(%1099)
      aiex.dma_await_task(%1100)
      aiex.dma_await_task(%1101)
      aiex.dma_await_task(%1102)
      aiex.dma_await_task(%1103)
      aiex.dma_free_task(%1084)
      aiex.dma_free_task(%1085)
      aiex.dma_free_task(%1086)
      aiex.dma_free_task(%1087)
      aiex.dma_free_task(%1088)
      aiex.dma_free_task(%1089)
      aiex.dma_free_task(%1090)
      aiex.dma_free_task(%1091)
      aiex.dma_free_task(%1092)
      aiex.dma_free_task(%1093)
      aiex.dma_free_task(%1094)
      aiex.dma_free_task(%1095)
      aiex.dma_free_task(%1096)
      aiex.dma_free_task(%1097)
      aiex.dma_free_task(%1098)
      aiex.dma_free_task(%1099)
      aiex.dma_free_task(%1100)
      aiex.dma_free_task(%1101)
      aiex.dma_free_task(%1102)
      aiex.dma_free_task(%1103)
      %1144 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9437184 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1144)
      %1145 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9830400 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1145)
      %1146 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 10223616 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1146)
      %1147 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 10616832 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1147)
      %1148 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1148)
      %1149 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 49152 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1149)
      %1150 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 98304 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1150)
      %1151 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 147456 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1151)
      %1152 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 196608 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1152)
      %1153 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 245760 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1153)
      %1154 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 294912 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1154)
      %1155 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 344064 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1155)
      %1156 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28311552 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1156)
      %1157 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28311680 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1157)
      %1158 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28311808 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1158)
      %1159 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28311936 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1159)
      %1160 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28312064 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1160)
      %1161 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28312192 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1161)
      %1162 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28312320 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1162)
      %1163 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28312448 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1163)
      %1164 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9437184 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1164)
      %1165 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9830400 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1165)
      %1166 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 10223616 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1166)
      %1167 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 10616832 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1167)
      %1168 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 393216 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1168)
      %1169 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 442368 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1169)
      %1170 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 491520 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1170)
      %1171 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 540672 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1171)
      %1172 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 589824 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1172)
      %1173 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 638976 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1173)
      %1174 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 688128 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1174)
      %1175 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 737280 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1175)
      %1176 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28312576 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1176)
      %1177 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28312704 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1177)
      %1178 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28312832 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1178)
      %1179 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28312960 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1179)
      %1180 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28313088 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1180)
      %1181 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28313216 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1181)
      %1182 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28313344 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1182)
      %1183 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28313472 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1183)
      aiex.dma_await_task(%1116)
      aiex.dma_await_task(%1117)
      aiex.dma_await_task(%1118)
      aiex.dma_await_task(%1119)
      aiex.dma_await_task(%1120)
      aiex.dma_await_task(%1121)
      aiex.dma_await_task(%1122)
      aiex.dma_await_task(%1123)
      aiex.dma_free_task(%1104)
      aiex.dma_free_task(%1105)
      aiex.dma_free_task(%1106)
      aiex.dma_free_task(%1107)
      aiex.dma_free_task(%1108)
      aiex.dma_free_task(%1109)
      aiex.dma_free_task(%1110)
      aiex.dma_free_task(%1111)
      aiex.dma_free_task(%1112)
      aiex.dma_free_task(%1113)
      aiex.dma_free_task(%1114)
      aiex.dma_free_task(%1115)
      aiex.dma_free_task(%1116)
      aiex.dma_free_task(%1117)
      aiex.dma_free_task(%1118)
      aiex.dma_free_task(%1119)
      aiex.dma_free_task(%1120)
      aiex.dma_free_task(%1121)
      aiex.dma_free_task(%1122)
      aiex.dma_free_task(%1123)
      aiex.dma_await_task(%1136)
      aiex.dma_await_task(%1137)
      aiex.dma_await_task(%1138)
      aiex.dma_await_task(%1139)
      aiex.dma_await_task(%1140)
      aiex.dma_await_task(%1141)
      aiex.dma_await_task(%1142)
      aiex.dma_await_task(%1143)
      aiex.dma_free_task(%1124)
      aiex.dma_free_task(%1125)
      aiex.dma_free_task(%1126)
      aiex.dma_free_task(%1127)
      aiex.dma_free_task(%1128)
      aiex.dma_free_task(%1129)
      aiex.dma_free_task(%1130)
      aiex.dma_free_task(%1131)
      aiex.dma_free_task(%1132)
      aiex.dma_free_task(%1133)
      aiex.dma_free_task(%1134)
      aiex.dma_free_task(%1135)
      aiex.dma_free_task(%1136)
      aiex.dma_free_task(%1137)
      aiex.dma_free_task(%1138)
      aiex.dma_free_task(%1139)
      aiex.dma_free_task(%1140)
      aiex.dma_free_task(%1141)
      aiex.dma_free_task(%1142)
      aiex.dma_free_task(%1143)
      %1184 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9437184 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1184)
      %1185 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9830400 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1185)
      %1186 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 10223616 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1186)
      %1187 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 10616832 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1187)
      %1188 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 786432 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1188)
      %1189 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 835584 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1189)
      %1190 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 884736 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1190)
      %1191 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 933888 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1191)
      %1192 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 983040 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1192)
      %1193 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1032192 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1193)
      %1194 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1081344 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1194)
      %1195 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1130496 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1195)
      %1196 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28313600 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1196)
      %1197 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28313728 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1197)
      %1198 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28313856 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1198)
      %1199 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28313984 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1199)
      %1200 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28314112 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1200)
      %1201 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28314240 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1201)
      %1202 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28314368 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1202)
      %1203 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28314496 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1203)
      %1204 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9437184 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1204)
      %1205 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9830400 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1205)
      %1206 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 10223616 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1206)
      %1207 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 10616832 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1207)
      %1208 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1179648 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1208)
      %1209 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1228800 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1209)
      %1210 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1277952 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1210)
      %1211 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1327104 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1211)
      %1212 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1376256 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1212)
      %1213 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1425408 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1213)
      %1214 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1474560 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1214)
      %1215 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1523712 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1215)
      %1216 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28314624 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1216)
      %1217 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28314752 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1217)
      %1218 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28314880 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1218)
      %1219 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28315008 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1219)
      %1220 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28315136 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1220)
      %1221 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28315264 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1221)
      %1222 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28315392 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1222)
      %1223 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28315520 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1223)
      aiex.dma_await_task(%1156)
      aiex.dma_await_task(%1157)
      aiex.dma_await_task(%1158)
      aiex.dma_await_task(%1159)
      aiex.dma_await_task(%1160)
      aiex.dma_await_task(%1161)
      aiex.dma_await_task(%1162)
      aiex.dma_await_task(%1163)
      aiex.dma_free_task(%1144)
      aiex.dma_free_task(%1145)
      aiex.dma_free_task(%1146)
      aiex.dma_free_task(%1147)
      aiex.dma_free_task(%1148)
      aiex.dma_free_task(%1149)
      aiex.dma_free_task(%1150)
      aiex.dma_free_task(%1151)
      aiex.dma_free_task(%1152)
      aiex.dma_free_task(%1153)
      aiex.dma_free_task(%1154)
      aiex.dma_free_task(%1155)
      aiex.dma_free_task(%1156)
      aiex.dma_free_task(%1157)
      aiex.dma_free_task(%1158)
      aiex.dma_free_task(%1159)
      aiex.dma_free_task(%1160)
      aiex.dma_free_task(%1161)
      aiex.dma_free_task(%1162)
      aiex.dma_free_task(%1163)
      aiex.dma_await_task(%1176)
      aiex.dma_await_task(%1177)
      aiex.dma_await_task(%1178)
      aiex.dma_await_task(%1179)
      aiex.dma_await_task(%1180)
      aiex.dma_await_task(%1181)
      aiex.dma_await_task(%1182)
      aiex.dma_await_task(%1183)
      aiex.dma_free_task(%1164)
      aiex.dma_free_task(%1165)
      aiex.dma_free_task(%1166)
      aiex.dma_free_task(%1167)
      aiex.dma_free_task(%1168)
      aiex.dma_free_task(%1169)
      aiex.dma_free_task(%1170)
      aiex.dma_free_task(%1171)
      aiex.dma_free_task(%1172)
      aiex.dma_free_task(%1173)
      aiex.dma_free_task(%1174)
      aiex.dma_free_task(%1175)
      aiex.dma_free_task(%1176)
      aiex.dma_free_task(%1177)
      aiex.dma_free_task(%1178)
      aiex.dma_free_task(%1179)
      aiex.dma_free_task(%1180)
      aiex.dma_free_task(%1181)
      aiex.dma_free_task(%1182)
      aiex.dma_free_task(%1183)
      %1224 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9437184 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1224)
      %1225 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9830400 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1225)
      %1226 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 10223616 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1226)
      %1227 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 10616832 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1227)
      %1228 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1572864 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1228)
      %1229 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1622016 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1229)
      %1230 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1671168 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1230)
      %1231 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1720320 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1231)
      %1232 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1769472 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1232)
      %1233 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1818624 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1233)
      %1234 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1867776 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1234)
      %1235 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1916928 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1235)
      %1236 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28315648 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1236)
      %1237 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28315776 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1237)
      %1238 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28315904 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1238)
      %1239 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28316032 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1239)
      %1240 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28316160 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1240)
      %1241 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28316288 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1241)
      %1242 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28316416 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1242)
      %1243 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28316544 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1243)
      %1244 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9437184 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1244)
      %1245 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9830400 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1245)
      %1246 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 10223616 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1246)
      %1247 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 10616832 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1247)
      %1248 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1966080 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1248)
      %1249 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2015232 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1249)
      %1250 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2064384 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1250)
      %1251 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2113536 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1251)
      %1252 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2162688 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1252)
      %1253 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2211840 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1253)
      %1254 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2260992 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1254)
      %1255 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2310144 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1255)
      %1256 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28316672 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1256)
      %1257 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28316800 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1257)
      %1258 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28316928 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1258)
      %1259 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28317056 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1259)
      %1260 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28317184 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1260)
      %1261 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28317312 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1261)
      %1262 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28317440 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1262)
      %1263 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28317568 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1263)
      aiex.dma_await_task(%1196)
      aiex.dma_await_task(%1197)
      aiex.dma_await_task(%1198)
      aiex.dma_await_task(%1199)
      aiex.dma_await_task(%1200)
      aiex.dma_await_task(%1201)
      aiex.dma_await_task(%1202)
      aiex.dma_await_task(%1203)
      aiex.dma_free_task(%1184)
      aiex.dma_free_task(%1185)
      aiex.dma_free_task(%1186)
      aiex.dma_free_task(%1187)
      aiex.dma_free_task(%1188)
      aiex.dma_free_task(%1189)
      aiex.dma_free_task(%1190)
      aiex.dma_free_task(%1191)
      aiex.dma_free_task(%1192)
      aiex.dma_free_task(%1193)
      aiex.dma_free_task(%1194)
      aiex.dma_free_task(%1195)
      aiex.dma_free_task(%1196)
      aiex.dma_free_task(%1197)
      aiex.dma_free_task(%1198)
      aiex.dma_free_task(%1199)
      aiex.dma_free_task(%1200)
      aiex.dma_free_task(%1201)
      aiex.dma_free_task(%1202)
      aiex.dma_free_task(%1203)
      aiex.dma_await_task(%1216)
      aiex.dma_await_task(%1217)
      aiex.dma_await_task(%1218)
      aiex.dma_await_task(%1219)
      aiex.dma_await_task(%1220)
      aiex.dma_await_task(%1221)
      aiex.dma_await_task(%1222)
      aiex.dma_await_task(%1223)
      aiex.dma_free_task(%1204)
      aiex.dma_free_task(%1205)
      aiex.dma_free_task(%1206)
      aiex.dma_free_task(%1207)
      aiex.dma_free_task(%1208)
      aiex.dma_free_task(%1209)
      aiex.dma_free_task(%1210)
      aiex.dma_free_task(%1211)
      aiex.dma_free_task(%1212)
      aiex.dma_free_task(%1213)
      aiex.dma_free_task(%1214)
      aiex.dma_free_task(%1215)
      aiex.dma_free_task(%1216)
      aiex.dma_free_task(%1217)
      aiex.dma_free_task(%1218)
      aiex.dma_free_task(%1219)
      aiex.dma_free_task(%1220)
      aiex.dma_free_task(%1221)
      aiex.dma_free_task(%1222)
      aiex.dma_free_task(%1223)
      %1264 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9437184 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1264)
      %1265 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9830400 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1265)
      %1266 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 10223616 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1266)
      %1267 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 10616832 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1267)
      %1268 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2359296 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1268)
      %1269 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2408448 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1269)
      %1270 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2457600 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1270)
      %1271 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2506752 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1271)
      %1272 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2555904 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1272)
      %1273 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2605056 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1273)
      %1274 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2654208 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1274)
      %1275 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2703360 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1275)
      %1276 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28317696 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1276)
      %1277 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28317824 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1277)
      %1278 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28317952 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1278)
      %1279 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28318080 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1279)
      %1280 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28318208 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1280)
      %1281 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28318336 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1281)
      %1282 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28318464 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1282)
      %1283 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28318592 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1283)
      %1284 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9437184 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1284)
      %1285 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9830400 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1285)
      %1286 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 10223616 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1286)
      %1287 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 10616832 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1287)
      %1288 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2752512 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1288)
      %1289 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2801664 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1289)
      %1290 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2850816 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1290)
      %1291 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2899968 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1291)
      %1292 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2949120 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1292)
      %1293 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2998272 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1293)
      %1294 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3047424 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1294)
      %1295 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3096576 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1295)
      %1296 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28318720 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1296)
      %1297 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28318848 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1297)
      %1298 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28318976 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1298)
      %1299 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28319104 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1299)
      %1300 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28319232 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1300)
      %1301 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28319360 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1301)
      %1302 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28319488 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1302)
      %1303 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28319616 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1303)
      aiex.dma_await_task(%1236)
      aiex.dma_await_task(%1237)
      aiex.dma_await_task(%1238)
      aiex.dma_await_task(%1239)
      aiex.dma_await_task(%1240)
      aiex.dma_await_task(%1241)
      aiex.dma_await_task(%1242)
      aiex.dma_await_task(%1243)
      aiex.dma_free_task(%1224)
      aiex.dma_free_task(%1225)
      aiex.dma_free_task(%1226)
      aiex.dma_free_task(%1227)
      aiex.dma_free_task(%1228)
      aiex.dma_free_task(%1229)
      aiex.dma_free_task(%1230)
      aiex.dma_free_task(%1231)
      aiex.dma_free_task(%1232)
      aiex.dma_free_task(%1233)
      aiex.dma_free_task(%1234)
      aiex.dma_free_task(%1235)
      aiex.dma_free_task(%1236)
      aiex.dma_free_task(%1237)
      aiex.dma_free_task(%1238)
      aiex.dma_free_task(%1239)
      aiex.dma_free_task(%1240)
      aiex.dma_free_task(%1241)
      aiex.dma_free_task(%1242)
      aiex.dma_free_task(%1243)
      aiex.dma_await_task(%1256)
      aiex.dma_await_task(%1257)
      aiex.dma_await_task(%1258)
      aiex.dma_await_task(%1259)
      aiex.dma_await_task(%1260)
      aiex.dma_await_task(%1261)
      aiex.dma_await_task(%1262)
      aiex.dma_await_task(%1263)
      aiex.dma_free_task(%1244)
      aiex.dma_free_task(%1245)
      aiex.dma_free_task(%1246)
      aiex.dma_free_task(%1247)
      aiex.dma_free_task(%1248)
      aiex.dma_free_task(%1249)
      aiex.dma_free_task(%1250)
      aiex.dma_free_task(%1251)
      aiex.dma_free_task(%1252)
      aiex.dma_free_task(%1253)
      aiex.dma_free_task(%1254)
      aiex.dma_free_task(%1255)
      aiex.dma_free_task(%1256)
      aiex.dma_free_task(%1257)
      aiex.dma_free_task(%1258)
      aiex.dma_free_task(%1259)
      aiex.dma_free_task(%1260)
      aiex.dma_free_task(%1261)
      aiex.dma_free_task(%1262)
      aiex.dma_free_task(%1263)
      %1304 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9437184 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1304)
      %1305 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 9830400 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1305)
      %1306 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 10223616 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1306)
      %1307 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 10616832 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1307)
      %1308 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3145728 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1308)
      %1309 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3194880 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1309)
      %1310 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3244032 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1310)
      %1311 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3293184 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1311)
      %1312 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3342336 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1312)
      %1313 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3391488 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1313)
      %1314 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3440640 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1314)
      %1315 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3489792 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1315)
      %1316 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28319744 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1316)
      %1317 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28319872 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1317)
      %1318 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28320000 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1318)
      %1319 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28320128 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1319)
      %1320 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28320256 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1320)
      %1321 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28320384 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1321)
      %1322 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28320512 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1322)
      %1323 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 28320640 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1323)
      %1324 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11010048 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1324)
      %1325 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11403264 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1325)
      %1326 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11796480 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1326)
      %1327 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 12189696 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1327)
      %1328 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1328)
      %1329 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 49152 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1329)
      %1330 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 98304 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1330)
      %1331 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 147456 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1331)
      %1332 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 196608 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1332)
      %1333 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 245760 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1333)
      %1334 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 294912 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1334)
      %1335 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 344064 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1335)
      %1336 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33030144 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1336)
      %1337 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33030272 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1337)
      %1338 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33030400 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1338)
      %1339 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33030528 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1339)
      %1340 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33030656 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1340)
      %1341 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33030784 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1341)
      %1342 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33030912 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1342)
      %1343 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33031040 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1343)
      aiex.dma_await_task(%1276)
      aiex.dma_await_task(%1277)
      aiex.dma_await_task(%1278)
      aiex.dma_await_task(%1279)
      aiex.dma_await_task(%1280)
      aiex.dma_await_task(%1281)
      aiex.dma_await_task(%1282)
      aiex.dma_await_task(%1283)
      aiex.dma_free_task(%1264)
      aiex.dma_free_task(%1265)
      aiex.dma_free_task(%1266)
      aiex.dma_free_task(%1267)
      aiex.dma_free_task(%1268)
      aiex.dma_free_task(%1269)
      aiex.dma_free_task(%1270)
      aiex.dma_free_task(%1271)
      aiex.dma_free_task(%1272)
      aiex.dma_free_task(%1273)
      aiex.dma_free_task(%1274)
      aiex.dma_free_task(%1275)
      aiex.dma_free_task(%1276)
      aiex.dma_free_task(%1277)
      aiex.dma_free_task(%1278)
      aiex.dma_free_task(%1279)
      aiex.dma_free_task(%1280)
      aiex.dma_free_task(%1281)
      aiex.dma_free_task(%1282)
      aiex.dma_free_task(%1283)
      aiex.dma_await_task(%1296)
      aiex.dma_await_task(%1297)
      aiex.dma_await_task(%1298)
      aiex.dma_await_task(%1299)
      aiex.dma_await_task(%1300)
      aiex.dma_await_task(%1301)
      aiex.dma_await_task(%1302)
      aiex.dma_await_task(%1303)
      aiex.dma_free_task(%1284)
      aiex.dma_free_task(%1285)
      aiex.dma_free_task(%1286)
      aiex.dma_free_task(%1287)
      aiex.dma_free_task(%1288)
      aiex.dma_free_task(%1289)
      aiex.dma_free_task(%1290)
      aiex.dma_free_task(%1291)
      aiex.dma_free_task(%1292)
      aiex.dma_free_task(%1293)
      aiex.dma_free_task(%1294)
      aiex.dma_free_task(%1295)
      aiex.dma_free_task(%1296)
      aiex.dma_free_task(%1297)
      aiex.dma_free_task(%1298)
      aiex.dma_free_task(%1299)
      aiex.dma_free_task(%1300)
      aiex.dma_free_task(%1301)
      aiex.dma_free_task(%1302)
      aiex.dma_free_task(%1303)
      %1344 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11010048 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1344)
      %1345 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11403264 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1345)
      %1346 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11796480 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1346)
      %1347 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 12189696 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1347)
      %1348 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 393216 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1348)
      %1349 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 442368 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1349)
      %1350 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 491520 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1350)
      %1351 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 540672 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1351)
      %1352 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 589824 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1352)
      %1353 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 638976 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1353)
      %1354 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 688128 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1354)
      %1355 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 737280 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1355)
      %1356 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33031168 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1356)
      %1357 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33031296 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1357)
      %1358 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33031424 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1358)
      %1359 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33031552 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1359)
      %1360 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33031680 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1360)
      %1361 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33031808 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1361)
      %1362 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33031936 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1362)
      %1363 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33032064 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1363)
      %1364 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11010048 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1364)
      %1365 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11403264 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1365)
      %1366 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11796480 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1366)
      %1367 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 12189696 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1367)
      %1368 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 786432 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1368)
      %1369 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 835584 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1369)
      %1370 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 884736 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1370)
      %1371 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 933888 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1371)
      %1372 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 983040 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1372)
      %1373 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1032192 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1373)
      %1374 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1081344 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1374)
      %1375 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1130496 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1375)
      %1376 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33032192 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1376)
      %1377 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33032320 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1377)
      %1378 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33032448 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1378)
      %1379 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33032576 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1379)
      %1380 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33032704 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1380)
      %1381 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33032832 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1381)
      %1382 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33032960 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1382)
      %1383 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33033088 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1383)
      aiex.dma_await_task(%1316)
      aiex.dma_await_task(%1317)
      aiex.dma_await_task(%1318)
      aiex.dma_await_task(%1319)
      aiex.dma_await_task(%1320)
      aiex.dma_await_task(%1321)
      aiex.dma_await_task(%1322)
      aiex.dma_await_task(%1323)
      aiex.dma_free_task(%1304)
      aiex.dma_free_task(%1305)
      aiex.dma_free_task(%1306)
      aiex.dma_free_task(%1307)
      aiex.dma_free_task(%1308)
      aiex.dma_free_task(%1309)
      aiex.dma_free_task(%1310)
      aiex.dma_free_task(%1311)
      aiex.dma_free_task(%1312)
      aiex.dma_free_task(%1313)
      aiex.dma_free_task(%1314)
      aiex.dma_free_task(%1315)
      aiex.dma_free_task(%1316)
      aiex.dma_free_task(%1317)
      aiex.dma_free_task(%1318)
      aiex.dma_free_task(%1319)
      aiex.dma_free_task(%1320)
      aiex.dma_free_task(%1321)
      aiex.dma_free_task(%1322)
      aiex.dma_free_task(%1323)
      aiex.dma_await_task(%1336)
      aiex.dma_await_task(%1337)
      aiex.dma_await_task(%1338)
      aiex.dma_await_task(%1339)
      aiex.dma_await_task(%1340)
      aiex.dma_await_task(%1341)
      aiex.dma_await_task(%1342)
      aiex.dma_await_task(%1343)
      aiex.dma_free_task(%1324)
      aiex.dma_free_task(%1325)
      aiex.dma_free_task(%1326)
      aiex.dma_free_task(%1327)
      aiex.dma_free_task(%1328)
      aiex.dma_free_task(%1329)
      aiex.dma_free_task(%1330)
      aiex.dma_free_task(%1331)
      aiex.dma_free_task(%1332)
      aiex.dma_free_task(%1333)
      aiex.dma_free_task(%1334)
      aiex.dma_free_task(%1335)
      aiex.dma_free_task(%1336)
      aiex.dma_free_task(%1337)
      aiex.dma_free_task(%1338)
      aiex.dma_free_task(%1339)
      aiex.dma_free_task(%1340)
      aiex.dma_free_task(%1341)
      aiex.dma_free_task(%1342)
      aiex.dma_free_task(%1343)
      %1384 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11010048 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1384)
      %1385 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11403264 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1385)
      %1386 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11796480 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1386)
      %1387 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 12189696 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1387)
      %1388 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1179648 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1388)
      %1389 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1228800 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1389)
      %1390 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1277952 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1390)
      %1391 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1327104 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1391)
      %1392 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1376256 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1392)
      %1393 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1425408 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1393)
      %1394 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1474560 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1394)
      %1395 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1523712 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1395)
      %1396 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33033216 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1396)
      %1397 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33033344 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1397)
      %1398 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33033472 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1398)
      %1399 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33033600 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1399)
      %1400 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33033728 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1400)
      %1401 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33033856 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1401)
      %1402 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33033984 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1402)
      %1403 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33034112 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1403)
      %1404 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11010048 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1404)
      %1405 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11403264 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1405)
      %1406 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11796480 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1406)
      %1407 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 12189696 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1407)
      %1408 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1572864 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1408)
      %1409 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1622016 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1409)
      %1410 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1671168 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1410)
      %1411 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1720320 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1411)
      %1412 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1769472 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1412)
      %1413 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1818624 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1413)
      %1414 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1867776 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1414)
      %1415 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1916928 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1415)
      %1416 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33034240 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1416)
      %1417 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33034368 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1417)
      %1418 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33034496 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1418)
      %1419 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33034624 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1419)
      %1420 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33034752 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1420)
      %1421 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33034880 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1421)
      %1422 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33035008 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1422)
      %1423 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33035136 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1423)
      aiex.dma_await_task(%1356)
      aiex.dma_await_task(%1357)
      aiex.dma_await_task(%1358)
      aiex.dma_await_task(%1359)
      aiex.dma_await_task(%1360)
      aiex.dma_await_task(%1361)
      aiex.dma_await_task(%1362)
      aiex.dma_await_task(%1363)
      aiex.dma_free_task(%1344)
      aiex.dma_free_task(%1345)
      aiex.dma_free_task(%1346)
      aiex.dma_free_task(%1347)
      aiex.dma_free_task(%1348)
      aiex.dma_free_task(%1349)
      aiex.dma_free_task(%1350)
      aiex.dma_free_task(%1351)
      aiex.dma_free_task(%1352)
      aiex.dma_free_task(%1353)
      aiex.dma_free_task(%1354)
      aiex.dma_free_task(%1355)
      aiex.dma_free_task(%1356)
      aiex.dma_free_task(%1357)
      aiex.dma_free_task(%1358)
      aiex.dma_free_task(%1359)
      aiex.dma_free_task(%1360)
      aiex.dma_free_task(%1361)
      aiex.dma_free_task(%1362)
      aiex.dma_free_task(%1363)
      aiex.dma_await_task(%1376)
      aiex.dma_await_task(%1377)
      aiex.dma_await_task(%1378)
      aiex.dma_await_task(%1379)
      aiex.dma_await_task(%1380)
      aiex.dma_await_task(%1381)
      aiex.dma_await_task(%1382)
      aiex.dma_await_task(%1383)
      aiex.dma_free_task(%1364)
      aiex.dma_free_task(%1365)
      aiex.dma_free_task(%1366)
      aiex.dma_free_task(%1367)
      aiex.dma_free_task(%1368)
      aiex.dma_free_task(%1369)
      aiex.dma_free_task(%1370)
      aiex.dma_free_task(%1371)
      aiex.dma_free_task(%1372)
      aiex.dma_free_task(%1373)
      aiex.dma_free_task(%1374)
      aiex.dma_free_task(%1375)
      aiex.dma_free_task(%1376)
      aiex.dma_free_task(%1377)
      aiex.dma_free_task(%1378)
      aiex.dma_free_task(%1379)
      aiex.dma_free_task(%1380)
      aiex.dma_free_task(%1381)
      aiex.dma_free_task(%1382)
      aiex.dma_free_task(%1383)
      %1424 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11010048 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1424)
      %1425 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11403264 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1425)
      %1426 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11796480 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1426)
      %1427 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 12189696 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1427)
      %1428 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 1966080 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1428)
      %1429 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2015232 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1429)
      %1430 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2064384 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1430)
      %1431 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2113536 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1431)
      %1432 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2162688 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1432)
      %1433 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2211840 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1433)
      %1434 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2260992 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1434)
      %1435 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2310144 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1435)
      %1436 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33035264 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1436)
      %1437 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33035392 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1437)
      %1438 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33035520 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1438)
      %1439 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33035648 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1439)
      %1440 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33035776 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1440)
      %1441 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33035904 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1441)
      %1442 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33036032 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1442)
      %1443 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33036160 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1443)
      %1444 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11010048 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1444)
      %1445 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11403264 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1445)
      %1446 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11796480 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1446)
      %1447 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 12189696 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1447)
      %1448 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2359296 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1448)
      %1449 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2408448 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1449)
      %1450 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2457600 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1450)
      %1451 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2506752 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1451)
      %1452 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2555904 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1452)
      %1453 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2605056 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1453)
      %1454 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2654208 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1454)
      %1455 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2703360 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1455)
      %1456 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33036288 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1456)
      %1457 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33036416 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1457)
      %1458 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33036544 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1458)
      %1459 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33036672 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1459)
      %1460 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33036800 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1460)
      %1461 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33036928 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1461)
      %1462 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33037056 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1462)
      %1463 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33037184 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1463)
      aiex.dma_await_task(%1396)
      aiex.dma_await_task(%1397)
      aiex.dma_await_task(%1398)
      aiex.dma_await_task(%1399)
      aiex.dma_await_task(%1400)
      aiex.dma_await_task(%1401)
      aiex.dma_await_task(%1402)
      aiex.dma_await_task(%1403)
      aiex.dma_free_task(%1384)
      aiex.dma_free_task(%1385)
      aiex.dma_free_task(%1386)
      aiex.dma_free_task(%1387)
      aiex.dma_free_task(%1388)
      aiex.dma_free_task(%1389)
      aiex.dma_free_task(%1390)
      aiex.dma_free_task(%1391)
      aiex.dma_free_task(%1392)
      aiex.dma_free_task(%1393)
      aiex.dma_free_task(%1394)
      aiex.dma_free_task(%1395)
      aiex.dma_free_task(%1396)
      aiex.dma_free_task(%1397)
      aiex.dma_free_task(%1398)
      aiex.dma_free_task(%1399)
      aiex.dma_free_task(%1400)
      aiex.dma_free_task(%1401)
      aiex.dma_free_task(%1402)
      aiex.dma_free_task(%1403)
      aiex.dma_await_task(%1416)
      aiex.dma_await_task(%1417)
      aiex.dma_await_task(%1418)
      aiex.dma_await_task(%1419)
      aiex.dma_await_task(%1420)
      aiex.dma_await_task(%1421)
      aiex.dma_await_task(%1422)
      aiex.dma_await_task(%1423)
      aiex.dma_free_task(%1404)
      aiex.dma_free_task(%1405)
      aiex.dma_free_task(%1406)
      aiex.dma_free_task(%1407)
      aiex.dma_free_task(%1408)
      aiex.dma_free_task(%1409)
      aiex.dma_free_task(%1410)
      aiex.dma_free_task(%1411)
      aiex.dma_free_task(%1412)
      aiex.dma_free_task(%1413)
      aiex.dma_free_task(%1414)
      aiex.dma_free_task(%1415)
      aiex.dma_free_task(%1416)
      aiex.dma_free_task(%1417)
      aiex.dma_free_task(%1418)
      aiex.dma_free_task(%1419)
      aiex.dma_free_task(%1420)
      aiex.dma_free_task(%1421)
      aiex.dma_free_task(%1422)
      aiex.dma_free_task(%1423)
      %1464 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11010048 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1464)
      %1465 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11403264 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1465)
      %1466 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11796480 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1466)
      %1467 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 12189696 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1467)
      %1468 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2752512 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1468)
      %1469 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2801664 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1469)
      %1470 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2850816 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1470)
      %1471 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2899968 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1471)
      %1472 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2949120 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1472)
      %1473 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 2998272 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1473)
      %1474 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3047424 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1474)
      %1475 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3096576 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1475)
      %1476 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33037312 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1476)
      %1477 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33037440 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1477)
      %1478 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33037568 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1478)
      %1479 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33037696 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1479)
      %1480 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33037824 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1480)
      %1481 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33037952 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1481)
      %1482 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33038080 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1482)
      %1483 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33038208 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1483)
      %1484 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11010048 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1484)
      %1485 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11403264 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1485)
      %1486 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 11796480 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1486)
      %1487 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<12582912xbf16> offset = 12189696 len = 393216 sizes = [1, 6, 128, 512] strides = [0, 512, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1487)
      %1488 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3145728 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1488)
      %1489 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3194880 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1489)
      %1490 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3244032 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1490)
      %1491 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3293184 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1491)
      %1492 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3342336 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1492)
      %1493 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3391488 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1493)
      %1494 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3440640 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1494)
      %1495 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<3538944x!aiex.bfp<"v8bfp16ebs8">> offset = 3489792 len = 49152 sizes = [1, 1, 1, 49152] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1495)
      %1496 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33038336 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1496)
      %1497 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33038464 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1497)
      %1498 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33038592 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1498)
      %1499 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33038720 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1499)
      %1500 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33038848 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1500)
      %1501 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33038976 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1501)
      %1502 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33039104 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1502)
      %1503 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<37748736xbf16> offset = 33039232 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 9216, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1503)
      aiex.dma_await_task(%1436)
      aiex.dma_await_task(%1437)
      aiex.dma_await_task(%1438)
      aiex.dma_await_task(%1439)
      aiex.dma_await_task(%1440)
      aiex.dma_await_task(%1441)
      aiex.dma_await_task(%1442)
      aiex.dma_await_task(%1443)
      aiex.dma_free_task(%1424)
      aiex.dma_free_task(%1425)
      aiex.dma_free_task(%1426)
      aiex.dma_free_task(%1427)
      aiex.dma_free_task(%1428)
      aiex.dma_free_task(%1429)
      aiex.dma_free_task(%1430)
      aiex.dma_free_task(%1431)
      aiex.dma_free_task(%1432)
      aiex.dma_free_task(%1433)
      aiex.dma_free_task(%1434)
      aiex.dma_free_task(%1435)
      aiex.dma_free_task(%1436)
      aiex.dma_free_task(%1437)
      aiex.dma_free_task(%1438)
      aiex.dma_free_task(%1439)
      aiex.dma_free_task(%1440)
      aiex.dma_free_task(%1441)
      aiex.dma_free_task(%1442)
      aiex.dma_free_task(%1443)
      aiex.dma_await_task(%1456)
      aiex.dma_await_task(%1457)
      aiex.dma_await_task(%1458)
      aiex.dma_await_task(%1459)
      aiex.dma_await_task(%1460)
      aiex.dma_await_task(%1461)
      aiex.dma_await_task(%1462)
      aiex.dma_await_task(%1463)
      aiex.dma_free_task(%1444)
      aiex.dma_free_task(%1445)
      aiex.dma_free_task(%1446)
      aiex.dma_free_task(%1447)
      aiex.dma_free_task(%1448)
      aiex.dma_free_task(%1449)
      aiex.dma_free_task(%1450)
      aiex.dma_free_task(%1451)
      aiex.dma_free_task(%1452)
      aiex.dma_free_task(%1453)
      aiex.dma_free_task(%1454)
      aiex.dma_free_task(%1455)
      aiex.dma_free_task(%1456)
      aiex.dma_free_task(%1457)
      aiex.dma_free_task(%1458)
      aiex.dma_free_task(%1459)
      aiex.dma_free_task(%1460)
      aiex.dma_free_task(%1461)
      aiex.dma_free_task(%1462)
      aiex.dma_free_task(%1463)
      aiex.dma_await_task(%1476)
      aiex.dma_await_task(%1477)
      aiex.dma_await_task(%1478)
      aiex.dma_await_task(%1479)
      aiex.dma_await_task(%1480)
      aiex.dma_await_task(%1481)
      aiex.dma_await_task(%1482)
      aiex.dma_await_task(%1483)
      aiex.dma_free_task(%1464)
      aiex.dma_free_task(%1465)
      aiex.dma_free_task(%1466)
      aiex.dma_free_task(%1467)
      aiex.dma_free_task(%1468)
      aiex.dma_free_task(%1469)
      aiex.dma_free_task(%1470)
      aiex.dma_free_task(%1471)
      aiex.dma_free_task(%1472)
      aiex.dma_free_task(%1473)
      aiex.dma_free_task(%1474)
      aiex.dma_free_task(%1475)
      aiex.dma_free_task(%1476)
      aiex.dma_free_task(%1477)
      aiex.dma_free_task(%1478)
      aiex.dma_free_task(%1479)
      aiex.dma_free_task(%1480)
      aiex.dma_free_task(%1481)
      aiex.dma_free_task(%1482)
      aiex.dma_free_task(%1483)
      aiex.dma_await_task(%1496)
      aiex.dma_await_task(%1497)
      aiex.dma_await_task(%1498)
      aiex.dma_await_task(%1499)
      aiex.dma_await_task(%1500)
      aiex.dma_await_task(%1501)
      aiex.dma_await_task(%1502)
      aiex.dma_await_task(%1503)
      aiex.dma_free_task(%1484)
      aiex.dma_free_task(%1485)
      aiex.dma_free_task(%1486)
      aiex.dma_free_task(%1487)
      aiex.dma_free_task(%1488)
      aiex.dma_free_task(%1489)
      aiex.dma_free_task(%1490)
      aiex.dma_free_task(%1491)
      aiex.dma_free_task(%1492)
      aiex.dma_free_task(%1493)
      aiex.dma_free_task(%1494)
      aiex.dma_free_task(%1495)
      aiex.dma_free_task(%1496)
      aiex.dma_free_task(%1497)
      aiex.dma_free_task(%1498)
      aiex.dma_free_task(%1499)
      aiex.dma_free_task(%1500)
      aiex.dma_free_task(%1501)
      aiex.dma_free_task(%1502)
      aiex.dma_free_task(%1503)
    }
  }
}

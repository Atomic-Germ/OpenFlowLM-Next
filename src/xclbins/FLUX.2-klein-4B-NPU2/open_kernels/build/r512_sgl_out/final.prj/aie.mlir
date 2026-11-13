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
    aie.runtime_sequence(%arg0: memref<51904512xbf16>, %arg1: memref<4718592x!aiex.bfp<"v8bfp16ebs8">>, %arg2: memref<4718592xbf16>) {
      %c9_i32 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_0, 0, %c9_i32) : i32
      %c192_i32 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_0_0, 1, %c192_i32) : i32
      %c3_i32 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_0, 2, %c3_i32) : i32
      %c3_i32_69 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_0, 3, %c3_i32_69) : i32
      %c9_i32_70 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_1, 0, %c9_i32_70) : i32
      %c192_i32_71 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_0_1, 1, %c192_i32_71) : i32
      %c3_i32_72 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_1, 2, %c3_i32_72) : i32
      %c3_i32_73 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_1, 3, %c3_i32_73) : i32
      %c9_i32_74 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_2, 0, %c9_i32_74) : i32
      %c192_i32_75 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_0_2, 1, %c192_i32_75) : i32
      %c3_i32_76 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_2, 2, %c3_i32_76) : i32
      %c3_i32_77 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_2, 3, %c3_i32_77) : i32
      %c9_i32_78 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_3, 0, %c9_i32_78) : i32
      %c192_i32_79 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_0_3, 1, %c192_i32_79) : i32
      %c3_i32_80 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_3, 2, %c3_i32_80) : i32
      %c3_i32_81 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_3, 3, %c3_i32_81) : i32
      %c9_i32_82 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_4, 0, %c9_i32_82) : i32
      %c192_i32_83 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_0_4, 1, %c192_i32_83) : i32
      %c3_i32_84 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_4, 2, %c3_i32_84) : i32
      %c3_i32_85 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_4, 3, %c3_i32_85) : i32
      %c9_i32_86 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_5, 0, %c9_i32_86) : i32
      %c192_i32_87 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_0_5, 1, %c192_i32_87) : i32
      %c3_i32_88 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_5, 2, %c3_i32_88) : i32
      %c3_i32_89 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_5, 3, %c3_i32_89) : i32
      %c9_i32_90 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_6, 0, %c9_i32_90) : i32
      %c192_i32_91 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_0_6, 1, %c192_i32_91) : i32
      %c3_i32_92 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_6, 2, %c3_i32_92) : i32
      %c3_i32_93 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_6, 3, %c3_i32_93) : i32
      %c9_i32_94 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_0_7, 0, %c9_i32_94) : i32
      %c192_i32_95 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_0_7, 1, %c192_i32_95) : i32
      %c3_i32_96 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_7, 2, %c3_i32_96) : i32
      %c3_i32_97 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_7, 3, %c3_i32_97) : i32
      %c9_i32_98 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_0, 0, %c9_i32_98) : i32
      %c192_i32_99 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_1_0, 1, %c192_i32_99) : i32
      %c3_i32_100 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_0, 2, %c3_i32_100) : i32
      %c3_i32_101 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_0, 3, %c3_i32_101) : i32
      %c9_i32_102 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_1, 0, %c9_i32_102) : i32
      %c192_i32_103 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_1_1, 1, %c192_i32_103) : i32
      %c3_i32_104 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_1, 2, %c3_i32_104) : i32
      %c3_i32_105 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_1, 3, %c3_i32_105) : i32
      %c9_i32_106 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_2, 0, %c9_i32_106) : i32
      %c192_i32_107 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_1_2, 1, %c192_i32_107) : i32
      %c3_i32_108 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_2, 2, %c3_i32_108) : i32
      %c3_i32_109 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_2, 3, %c3_i32_109) : i32
      %c9_i32_110 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_3, 0, %c9_i32_110) : i32
      %c192_i32_111 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_1_3, 1, %c192_i32_111) : i32
      %c3_i32_112 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_3, 2, %c3_i32_112) : i32
      %c3_i32_113 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_3, 3, %c3_i32_113) : i32
      %c9_i32_114 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_4, 0, %c9_i32_114) : i32
      %c192_i32_115 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_1_4, 1, %c192_i32_115) : i32
      %c3_i32_116 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_4, 2, %c3_i32_116) : i32
      %c3_i32_117 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_4, 3, %c3_i32_117) : i32
      %c9_i32_118 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_5, 0, %c9_i32_118) : i32
      %c192_i32_119 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_1_5, 1, %c192_i32_119) : i32
      %c3_i32_120 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_5, 2, %c3_i32_120) : i32
      %c3_i32_121 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_5, 3, %c3_i32_121) : i32
      %c9_i32_122 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_6, 0, %c9_i32_122) : i32
      %c192_i32_123 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_1_6, 1, %c192_i32_123) : i32
      %c3_i32_124 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_6, 2, %c3_i32_124) : i32
      %c3_i32_125 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_6, 3, %c3_i32_125) : i32
      %c9_i32_126 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_1_7, 0, %c9_i32_126) : i32
      %c192_i32_127 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_1_7, 1, %c192_i32_127) : i32
      %c3_i32_128 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_7, 2, %c3_i32_128) : i32
      %c3_i32_129 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_7, 3, %c3_i32_129) : i32
      %c9_i32_130 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_0, 0, %c9_i32_130) : i32
      %c192_i32_131 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_2_0, 1, %c192_i32_131) : i32
      %c3_i32_132 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_2_0, 2, %c3_i32_132) : i32
      %c3_i32_133 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_2_0, 3, %c3_i32_133) : i32
      %c9_i32_134 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_1, 0, %c9_i32_134) : i32
      %c192_i32_135 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_2_1, 1, %c192_i32_135) : i32
      %c3_i32_136 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_2_1, 2, %c3_i32_136) : i32
      %c3_i32_137 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_2_1, 3, %c3_i32_137) : i32
      %c9_i32_138 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_2, 0, %c9_i32_138) : i32
      %c192_i32_139 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_2_2, 1, %c192_i32_139) : i32
      %c3_i32_140 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_2_2, 2, %c3_i32_140) : i32
      %c3_i32_141 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_2_2, 3, %c3_i32_141) : i32
      %c9_i32_142 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_3, 0, %c9_i32_142) : i32
      %c192_i32_143 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_2_3, 1, %c192_i32_143) : i32
      %c3_i32_144 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_2_3, 2, %c3_i32_144) : i32
      %c3_i32_145 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_2_3, 3, %c3_i32_145) : i32
      %c9_i32_146 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_4, 0, %c9_i32_146) : i32
      %c192_i32_147 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_2_4, 1, %c192_i32_147) : i32
      %c3_i32_148 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_2_4, 2, %c3_i32_148) : i32
      %c3_i32_149 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_2_4, 3, %c3_i32_149) : i32
      %c9_i32_150 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_5, 0, %c9_i32_150) : i32
      %c192_i32_151 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_2_5, 1, %c192_i32_151) : i32
      %c3_i32_152 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_2_5, 2, %c3_i32_152) : i32
      %c3_i32_153 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_2_5, 3, %c3_i32_153) : i32
      %c9_i32_154 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_6, 0, %c9_i32_154) : i32
      %c192_i32_155 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_2_6, 1, %c192_i32_155) : i32
      %c3_i32_156 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_2_6, 2, %c3_i32_156) : i32
      %c3_i32_157 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_2_6, 3, %c3_i32_157) : i32
      %c9_i32_158 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_2_7, 0, %c9_i32_158) : i32
      %c192_i32_159 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_2_7, 1, %c192_i32_159) : i32
      %c3_i32_160 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_2_7, 2, %c3_i32_160) : i32
      %c3_i32_161 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_2_7, 3, %c3_i32_161) : i32
      %c9_i32_162 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_0, 0, %c9_i32_162) : i32
      %c192_i32_163 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_3_0, 1, %c192_i32_163) : i32
      %c3_i32_164 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_3_0, 2, %c3_i32_164) : i32
      %c3_i32_165 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_3_0, 3, %c3_i32_165) : i32
      %c9_i32_166 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_1, 0, %c9_i32_166) : i32
      %c192_i32_167 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_3_1, 1, %c192_i32_167) : i32
      %c3_i32_168 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_3_1, 2, %c3_i32_168) : i32
      %c3_i32_169 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_3_1, 3, %c3_i32_169) : i32
      %c9_i32_170 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_2, 0, %c9_i32_170) : i32
      %c192_i32_171 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_3_2, 1, %c192_i32_171) : i32
      %c3_i32_172 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_3_2, 2, %c3_i32_172) : i32
      %c3_i32_173 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_3_2, 3, %c3_i32_173) : i32
      %c9_i32_174 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_3, 0, %c9_i32_174) : i32
      %c192_i32_175 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_3_3, 1, %c192_i32_175) : i32
      %c3_i32_176 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_3_3, 2, %c3_i32_176) : i32
      %c3_i32_177 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_3_3, 3, %c3_i32_177) : i32
      %c9_i32_178 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_4, 0, %c9_i32_178) : i32
      %c192_i32_179 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_3_4, 1, %c192_i32_179) : i32
      %c3_i32_180 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_3_4, 2, %c3_i32_180) : i32
      %c3_i32_181 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_3_4, 3, %c3_i32_181) : i32
      %c9_i32_182 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_5, 0, %c9_i32_182) : i32
      %c192_i32_183 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_3_5, 1, %c192_i32_183) : i32
      %c3_i32_184 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_3_5, 2, %c3_i32_184) : i32
      %c3_i32_185 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_3_5, 3, %c3_i32_185) : i32
      %c9_i32_186 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_6, 0, %c9_i32_186) : i32
      %c192_i32_187 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_3_6, 1, %c192_i32_187) : i32
      %c3_i32_188 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_3_6, 2, %c3_i32_188) : i32
      %c3_i32_189 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_3_6, 3, %c3_i32_189) : i32
      %c9_i32_190 = arith.constant 9 : i32
      aiex.npu.rtp_write(@rtp_3_7, 0, %c9_i32_190) : i32
      %c192_i32_191 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_3_7, 1, %c192_i32_191) : i32
      %c3_i32_192 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_3_7, 2, %c3_i32_192) : i32
      %c3_i32_193 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_3_7, 3, %c3_i32_193) : i32
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
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 9216 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 4334592 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 8659968 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 12985344 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%67)
      %68 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 196608 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 393216 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 589824 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 786432 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 983040 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 1179648 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 1376256 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 0 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 128 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 256 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 384 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 512 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 640 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 768 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 896 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 9216 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%84)
      %85 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 4334592 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 8659968 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 12985344 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 1572864 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 1769472 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 1966080 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 2162688 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 2359296 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 2555904 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 2752512 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 2949120 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%95)
      %96 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1024 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1152 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%97)
      %98 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1280 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1408 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1536 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%100)
      %101 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1664 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%101)
      %102 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1792 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%102)
      %103 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1920 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%103)
      %104 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 9216 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%104)
      %105 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 4334592 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%105)
      %106 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 8659968 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%106)
      %107 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 12985344 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%107)
      %108 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 3145728 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%108)
      %109 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 3342336 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%109)
      %110 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 3538944 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%110)
      %111 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 3735552 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%111)
      %112 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 3932160 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%112)
      %113 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 4128768 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%113)
      %114 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 4325376 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%114)
      %115 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 4521984 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%115)
      %116 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2048 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%116)
      %117 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2176 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%117)
      %118 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2304 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%118)
      %119 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2432 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%119)
      %120 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2560 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%120)
      %121 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2688 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%121)
      %122 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2816 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%122)
      %123 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2944 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%123)
      %124 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 17310720 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%124)
      %125 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 21636096 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%125)
      %126 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 25961472 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%126)
      %127 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 30286848 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%127)
      %128 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%128)
      %129 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 196608 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%129)
      %130 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 393216 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%130)
      %131 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 589824 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%131)
      %132 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 786432 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%132)
      %133 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 983040 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%133)
      %134 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 1179648 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%134)
      %135 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 1376256 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%135)
      %136 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1572864 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%136)
      %137 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1572992 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%137)
      %138 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1573120 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%138)
      %139 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1573248 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%139)
      %140 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1573376 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%140)
      %141 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1573504 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%141)
      %142 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1573632 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%142)
      %143 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1573760 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
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
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 17310720 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%144)
      %145 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 21636096 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%145)
      %146 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 25961472 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%146)
      %147 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 30286848 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%147)
      %148 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 1572864 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%148)
      %149 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 1769472 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%149)
      %150 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 1966080 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%150)
      %151 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 2162688 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%151)
      %152 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 2359296 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%152)
      %153 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 2555904 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%153)
      %154 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 2752512 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%154)
      %155 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 2949120 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%155)
      %156 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1573888 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%156)
      %157 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1574016 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%157)
      %158 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1574144 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%158)
      %159 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1574272 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%159)
      %160 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1574400 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%160)
      %161 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1574528 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%161)
      %162 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1574656 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%162)
      %163 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1574784 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%163)
      %164 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 17310720 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%164)
      %165 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 21636096 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%165)
      %166 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 25961472 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%166)
      %167 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 30286848 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%167)
      %168 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 3145728 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%168)
      %169 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 3342336 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%169)
      %170 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 3538944 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%170)
      %171 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 3735552 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%171)
      %172 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 3932160 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%172)
      %173 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 4128768 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%173)
      %174 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 4325376 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%174)
      %175 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 4521984 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%175)
      %176 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1574912 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%176)
      %177 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1575040 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%177)
      %178 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1575168 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%178)
      %179 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1575296 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%179)
      %180 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1575424 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%180)
      %181 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1575552 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%181)
      %182 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1575680 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%182)
      %183 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1575808 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
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
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 34612224 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%184)
      %185 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 38937600 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%185)
      %186 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 43262976 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%186)
      %187 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 47588352 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%187)
      %188 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%188)
      %189 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 196608 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%189)
      %190 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 393216 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%190)
      %191 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 589824 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%191)
      %192 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 786432 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%192)
      %193 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 983040 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%193)
      %194 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 1179648 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%194)
      %195 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 1376256 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%195)
      %196 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3145728 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%196)
      %197 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3145856 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%197)
      %198 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3145984 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%198)
      %199 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3146112 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%199)
      %200 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3146240 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%200)
      %201 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3146368 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%201)
      %202 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3146496 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%202)
      %203 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3146624 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%203)
      %204 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 34612224 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%204)
      %205 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 38937600 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%205)
      %206 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 43262976 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%206)
      %207 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 47588352 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%207)
      %208 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 1572864 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%208)
      %209 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 1769472 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%209)
      %210 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 1966080 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%210)
      %211 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 2162688 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%211)
      %212 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 2359296 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%212)
      %213 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 2555904 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%213)
      %214 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 2752512 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%214)
      %215 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 2949120 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%215)
      %216 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3146752 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%216)
      %217 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3146880 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%217)
      %218 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3147008 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%218)
      %219 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3147136 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%219)
      %220 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3147264 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%220)
      %221 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3147392 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%221)
      %222 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3147520 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%222)
      %223 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3147648 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
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
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 34612224 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%224)
      %225 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 38937600 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%225)
      %226 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 43262976 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%226)
      %227 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<51904512xbf16> offset = 47588352 len = 65536 sizes = [24, 128, 8, 64] strides = [1024, 33792, 128, 1])
        aie.end
      } {repeat_count = 23 : i32}
      aiex.dma_start_task(%227)
      %228 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 3145728 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%228)
      %229 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 3342336 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%229)
      %230 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 3538944 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%230)
      %231 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 3735552 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%231)
      %232 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 3932160 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%232)
      %233 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 4128768 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%233)
      %234 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 4325376 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%234)
      %235 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4718592x!aiex.bfp<"v8bfp16ebs8">> offset = 4521984 len = 196608 sizes = [1, 1, 1, 196608] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%235)
      %236 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3147776 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%236)
      %237 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3147904 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%237)
      %238 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3148032 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%238)
      %239 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3148160 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%239)
      %240 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3148288 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%240)
      %241 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3148416 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%241)
      %242 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3148544 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%242)
      %243 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 3148672 len = 65536 sizes = [1, 1, 512, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%243)
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
    }
  }
}

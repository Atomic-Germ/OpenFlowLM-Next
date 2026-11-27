module {
  aie.device(npu2) @main {
    aie.runtime_sequence @cfg_conv1() {
      aiex.configure @conv1 {
      }
    }
  }
  aie.device(npu2) @conv1 {
    %logical_core = aie.logical_tile<CoreTile>(0, 2)
    %logical_core_0 = aie.logical_tile<CoreTile>(0, 3)
    %logical_core_1 = aie.logical_tile<CoreTile>(0, 4)
    %logical_core_2 = aie.logical_tile<CoreTile>(0, 5)
    %logical_core_3 = aie.logical_tile<CoreTile>(1, 2)
    %logical_core_4 = aie.logical_tile<CoreTile>(1, 3)
    %logical_core_5 = aie.logical_tile<CoreTile>(1, 4)
    %logical_core_6 = aie.logical_tile<CoreTile>(1, 5)
    %logical_core_7 = aie.logical_tile<CoreTile>(2, 2)
    %logical_core_8 = aie.logical_tile<CoreTile>(2, 3)
    %logical_core_9 = aie.logical_tile<CoreTile>(2, 4)
    %logical_core_10 = aie.logical_tile<CoreTile>(2, 5)
    %logical_core_11 = aie.logical_tile<CoreTile>(3, 2)
    %logical_core_12 = aie.logical_tile<CoreTile>(3, 3)
    %logical_core_13 = aie.logical_tile<CoreTile>(3, 4)
    %logical_core_14 = aie.logical_tile<CoreTile>(3, 5)
    %logical_core_15 = aie.logical_tile<CoreTile>(4, 2)
    %logical_core_16 = aie.logical_tile<CoreTile>(4, 3)
    %logical_core_17 = aie.logical_tile<CoreTile>(4, 4)
    %logical_core_18 = aie.logical_tile<CoreTile>(4, 5)
    %logical_core_19 = aie.logical_tile<CoreTile>(5, 2)
    %logical_core_20 = aie.logical_tile<CoreTile>(5, 3)
    %logical_core_21 = aie.logical_tile<CoreTile>(5, 4)
    %logical_core_22 = aie.logical_tile<CoreTile>(5, 5)
    %logical_core_23 = aie.logical_tile<CoreTile>(6, 2)
    %logical_core_24 = aie.logical_tile<CoreTile>(6, 3)
    %logical_core_25 = aie.logical_tile<CoreTile>(6, 4)
    %logical_core_26 = aie.logical_tile<CoreTile>(6, 5)
    %logical_core_27 = aie.logical_tile<CoreTile>(7, 2)
    %logical_core_28 = aie.logical_tile<CoreTile>(7, 3)
    %logical_core_29 = aie.logical_tile<CoreTile>(7, 4)
    %logical_core_30 = aie.logical_tile<CoreTile>(7, 5)
    %logical_mem = aie.logical_tile<MemTile>(0, 1)
    %logical_mem_31 = aie.logical_tile<MemTile>(1, 1)
    %logical_mem_32 = aie.logical_tile<MemTile>(2, 1)
    %logical_mem_33 = aie.logical_tile<MemTile>(3, 1)
    %logical_mem_34 = aie.logical_tile<MemTile>(4, 1)
    %logical_mem_35 = aie.logical_tile<MemTile>(5, 1)
    %logical_mem_36 = aie.logical_tile<MemTile>(6, 1)
    %logical_mem_37 = aie.logical_tile<MemTile>(7, 1)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_38 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_39 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_40 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_41 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_42 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_43 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_44 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_45 = aie.logical_tile<MemTile>(0, 1)
    %logical_mem_46 = aie.logical_tile<MemTile>(1, 1)
    %logical_mem_47 = aie.logical_tile<MemTile>(2, 1)
    %logical_mem_48 = aie.logical_tile<MemTile>(3, 1)
    %logical_mem_49 = aie.logical_tile<MemTile>(4, 1)
    %logical_mem_50 = aie.logical_tile<MemTile>(5, 1)
    %logical_mem_51 = aie.logical_tile<MemTile>(6, 1)
    %logical_mem_52 = aie.logical_tile<MemTile>(7, 1)
    %logical_shim_noc_53 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_54 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_55 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_56 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_57 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_58 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_59 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_60 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_61 = aie.logical_tile<MemTile>(0, 1)
    %logical_shim_noc_62 = aie.logical_tile<ShimNOCTile>(0, 0)
    %logical_mem_63 = aie.logical_tile<MemTile>(1, 1)
    %logical_shim_noc_64 = aie.logical_tile<ShimNOCTile>(1, 0)
    %logical_mem_65 = aie.logical_tile<MemTile>(2, 1)
    %logical_shim_noc_66 = aie.logical_tile<ShimNOCTile>(2, 0)
    %logical_mem_67 = aie.logical_tile<MemTile>(3, 1)
    %logical_shim_noc_68 = aie.logical_tile<ShimNOCTile>(3, 0)
    %logical_mem_69 = aie.logical_tile<MemTile>(4, 1)
    %logical_shim_noc_70 = aie.logical_tile<ShimNOCTile>(4, 0)
    %logical_mem_71 = aie.logical_tile<MemTile>(5, 1)
    %logical_shim_noc_72 = aie.logical_tile<ShimNOCTile>(5, 0)
    %logical_mem_73 = aie.logical_tile<MemTile>(6, 1)
    %logical_shim_noc_74 = aie.logical_tile<ShimNOCTile>(6, 0)
    %logical_mem_75 = aie.logical_tile<MemTile>(7, 1)
    %logical_shim_noc_76 = aie.logical_tile<ShimNOCTile>(7, 0)
    aie.objectfifo @B_L2L1_0(%logical_mem, {%logical_core, %logical_core_0, %logical_core_1, %logical_core_2}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo @B_L3L2_0(%logical_shim_noc, {%logical_mem}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo.link [@B_L3L2_0] -> [@B_L2L1_0]([] [0])
    aie.objectfifo @B_L2L1_1(%logical_mem_31, {%logical_core_3, %logical_core_4, %logical_core_5, %logical_core_6}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo @B_L3L2_1(%logical_shim_noc_38, {%logical_mem_31}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo.link [@B_L3L2_1] -> [@B_L2L1_1]([] [0])
    aie.objectfifo @B_L2L1_2(%logical_mem_32, {%logical_core_7, %logical_core_8, %logical_core_9, %logical_core_10}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo @B_L3L2_2(%logical_shim_noc_39, {%logical_mem_32}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo.link [@B_L3L2_2] -> [@B_L2L1_2]([] [0])
    aie.objectfifo @B_L2L1_3(%logical_mem_33, {%logical_core_11, %logical_core_12, %logical_core_13, %logical_core_14}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo @B_L3L2_3(%logical_shim_noc_40, {%logical_mem_33}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo.link [@B_L3L2_3] -> [@B_L2L1_3]([] [0])
    aie.objectfifo @B_L2L1_4(%logical_mem_34, {%logical_core_15, %logical_core_16, %logical_core_17, %logical_core_18}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo @B_L3L2_4(%logical_shim_noc_41, {%logical_mem_34}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo.link [@B_L3L2_4] -> [@B_L2L1_4]([] [0])
    aie.objectfifo @B_L2L1_5(%logical_mem_35, {%logical_core_19, %logical_core_20, %logical_core_21, %logical_core_22}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo @B_L3L2_5(%logical_shim_noc_42, {%logical_mem_35}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo.link [@B_L3L2_5] -> [@B_L2L1_5]([] [0])
    aie.objectfifo @B_L2L1_6(%logical_mem_36, {%logical_core_23, %logical_core_24, %logical_core_25, %logical_core_26}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo @B_L3L2_6(%logical_shim_noc_43, {%logical_mem_36}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo.link [@B_L3L2_6] -> [@B_L2L1_6]([] [0])
    aie.objectfifo @B_L2L1_7(%logical_mem_37, {%logical_core_27, %logical_core_28, %logical_core_29, %logical_core_30}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo @B_L3L2_7(%logical_shim_noc_44, {%logical_mem_37}, 2 : i32) : !aie.objectfifo<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>  
    aie.objectfifo.link [@B_L3L2_7] -> [@B_L2L1_7]([] [0])
    aie.objectfifo @C_L1L2_0_0(%logical_core, {%logical_mem_45}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_0_1(%logical_core_0, {%logical_mem_45}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_0_2(%logical_core_1, {%logical_mem_45}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_0_3(%logical_core_2, {%logical_mem_45}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L2L3_0(%logical_mem_45 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_53}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>>  
    aie.objectfifo.link [@C_L1L2_0_0, @C_L1L2_0_1, @C_L1L2_0_2, @C_L1L2_0_3] -> [@C_L2L3_0]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_1_0(%logical_core_3, {%logical_mem_46}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_1_1(%logical_core_4, {%logical_mem_46}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_1_2(%logical_core_5, {%logical_mem_46}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_1_3(%logical_core_6, {%logical_mem_46}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L2L3_1(%logical_mem_46 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_54}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>>  
    aie.objectfifo.link [@C_L1L2_1_0, @C_L1L2_1_1, @C_L1L2_1_2, @C_L1L2_1_3] -> [@C_L2L3_1]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_2_0(%logical_core_7, {%logical_mem_47}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_2_1(%logical_core_8, {%logical_mem_47}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_2_2(%logical_core_9, {%logical_mem_47}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_2_3(%logical_core_10, {%logical_mem_47}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L2L3_2(%logical_mem_47 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_55}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>>  
    aie.objectfifo.link [@C_L1L2_2_0, @C_L1L2_2_1, @C_L1L2_2_2, @C_L1L2_2_3] -> [@C_L2L3_2]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_3_0(%logical_core_11, {%logical_mem_48}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_3_1(%logical_core_12, {%logical_mem_48}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_3_2(%logical_core_13, {%logical_mem_48}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_3_3(%logical_core_14, {%logical_mem_48}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L2L3_3(%logical_mem_48 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_56}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>>  
    aie.objectfifo.link [@C_L1L2_3_0, @C_L1L2_3_1, @C_L1L2_3_2, @C_L1L2_3_3] -> [@C_L2L3_3]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_4_0(%logical_core_15, {%logical_mem_49}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_4_1(%logical_core_16, {%logical_mem_49}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_4_2(%logical_core_17, {%logical_mem_49}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_4_3(%logical_core_18, {%logical_mem_49}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L2L3_4(%logical_mem_49 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_57}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>>  
    aie.objectfifo.link [@C_L1L2_4_0, @C_L1L2_4_1, @C_L1L2_4_2, @C_L1L2_4_3] -> [@C_L2L3_4]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_5_0(%logical_core_19, {%logical_mem_50}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_5_1(%logical_core_20, {%logical_mem_50}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_5_2(%logical_core_21, {%logical_mem_50}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_5_3(%logical_core_22, {%logical_mem_50}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L2L3_5(%logical_mem_50 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_58}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>>  
    aie.objectfifo.link [@C_L1L2_5_0, @C_L1L2_5_1, @C_L1L2_5_2, @C_L1L2_5_3] -> [@C_L2L3_5]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_6_0(%logical_core_23, {%logical_mem_51}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_6_1(%logical_core_24, {%logical_mem_51}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_6_2(%logical_core_25, {%logical_mem_51}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_6_3(%logical_core_26, {%logical_mem_51}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L2L3_6(%logical_mem_51 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_59}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>>  
    aie.objectfifo.link [@C_L1L2_6_0, @C_L1L2_6_1, @C_L1L2_6_2, @C_L1L2_6_3] -> [@C_L2L3_6]([0, 16384, 32768, 49152] [])
    aie.objectfifo @C_L1L2_7_0(%logical_core_27, {%logical_mem_52}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_7_1(%logical_core_28, {%logical_mem_52}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_7_2(%logical_core_29, {%logical_mem_52}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L1L2_7_3(%logical_core_30, {%logical_mem_52}, 1 : i32) : !aie.objectfifo<memref<128x128xbf16>>  
    aie.objectfifo @C_L2L3_7(%logical_mem_52 dimensionsToStream [<size = 16, stride = 1024>, <size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_60}, 2 : i32) : !aie.objectfifo<memref<512x128xbf16>>  
    aie.objectfifo.link [@C_L1L2_7_0, @C_L1L2_7_1, @C_L1L2_7_2, @C_L1L2_7_3] -> [@C_L2L3_7]([0, 16384, 32768, 49152] [])
    aie.flow(%logical_mem_61, DMA : 0, %logical_core, DMA : 0)
    aie.flow(%logical_mem_61, DMA : 2, %logical_core_0, DMA : 0)
    aie.flow(%logical_mem_61, DMA : 4, %logical_core_1, DMA : 0)
    aie.flow(%logical_mem_61, DMA : 1, %logical_core_2, DMA : 0)
    aie.flow(%logical_shim_noc_62, DMA : 0, %logical_mem_61, DMA : 0)
    aie.shim_dma_allocation @A_0(%logical_shim_noc_62, MM2S, 0)
    aie.flow(%logical_mem_63, DMA : 0, %logical_core_3, DMA : 0)
    aie.flow(%logical_mem_63, DMA : 2, %logical_core_4, DMA : 0)
    aie.flow(%logical_mem_63, DMA : 4, %logical_core_5, DMA : 0)
    aie.flow(%logical_mem_63, DMA : 1, %logical_core_6, DMA : 0)
    aie.flow(%logical_shim_noc_64, DMA : 0, %logical_mem_63, DMA : 0)
    aie.shim_dma_allocation @A_1(%logical_shim_noc_64, MM2S, 0)
    aie.flow(%logical_mem_65, DMA : 0, %logical_core_7, DMA : 0)
    aie.flow(%logical_mem_65, DMA : 2, %logical_core_8, DMA : 0)
    aie.flow(%logical_mem_65, DMA : 4, %logical_core_9, DMA : 0)
    aie.flow(%logical_mem_65, DMA : 1, %logical_core_10, DMA : 0)
    aie.flow(%logical_shim_noc_66, DMA : 0, %logical_mem_65, DMA : 0)
    aie.shim_dma_allocation @A_2(%logical_shim_noc_66, MM2S, 0)
    aie.flow(%logical_mem_67, DMA : 0, %logical_core_11, DMA : 0)
    aie.flow(%logical_mem_67, DMA : 2, %logical_core_12, DMA : 0)
    aie.flow(%logical_mem_67, DMA : 4, %logical_core_13, DMA : 0)
    aie.flow(%logical_mem_67, DMA : 1, %logical_core_14, DMA : 0)
    aie.flow(%logical_shim_noc_68, DMA : 0, %logical_mem_67, DMA : 0)
    aie.shim_dma_allocation @A_3(%logical_shim_noc_68, MM2S, 0)
    aie.flow(%logical_mem_69, DMA : 0, %logical_core_15, DMA : 0)
    aie.flow(%logical_mem_69, DMA : 2, %logical_core_16, DMA : 0)
    aie.flow(%logical_mem_69, DMA : 4, %logical_core_17, DMA : 0)
    aie.flow(%logical_mem_69, DMA : 1, %logical_core_18, DMA : 0)
    aie.flow(%logical_shim_noc_70, DMA : 0, %logical_mem_69, DMA : 0)
    aie.shim_dma_allocation @A_4(%logical_shim_noc_70, MM2S, 0)
    aie.flow(%logical_mem_71, DMA : 0, %logical_core_19, DMA : 0)
    aie.flow(%logical_mem_71, DMA : 2, %logical_core_20, DMA : 0)
    aie.flow(%logical_mem_71, DMA : 4, %logical_core_21, DMA : 0)
    aie.flow(%logical_mem_71, DMA : 1, %logical_core_22, DMA : 0)
    aie.flow(%logical_shim_noc_72, DMA : 0, %logical_mem_71, DMA : 0)
    aie.shim_dma_allocation @A_5(%logical_shim_noc_72, MM2S, 0)
    aie.flow(%logical_mem_73, DMA : 0, %logical_core_23, DMA : 0)
    aie.flow(%logical_mem_73, DMA : 2, %logical_core_24, DMA : 0)
    aie.flow(%logical_mem_73, DMA : 4, %logical_core_25, DMA : 0)
    aie.flow(%logical_mem_73, DMA : 1, %logical_core_26, DMA : 0)
    aie.flow(%logical_shim_noc_74, DMA : 0, %logical_mem_73, DMA : 0)
    aie.shim_dma_allocation @A_6(%logical_shim_noc_74, MM2S, 0)
    aie.flow(%logical_mem_75, DMA : 0, %logical_core_27, DMA : 0)
    aie.flow(%logical_mem_75, DMA : 2, %logical_core_28, DMA : 0)
    aie.flow(%logical_mem_75, DMA : 4, %logical_core_29, DMA : 0)
    aie.flow(%logical_mem_75, DMA : 1, %logical_core_30, DMA : 0)
    aie.flow(%logical_shim_noc_76, DMA : 0, %logical_mem_75, DMA : 0)
    aie.shim_dma_allocation @A_7(%logical_shim_noc_76, MM2S, 0)
    %band_full_0_0 = aie.lock(%logical_mem_61) {init = 0 : i32, sym_name = "band_full_0_0"}
    %band_full_0_1 = aie.lock(%logical_mem_61) {init = 0 : i32, sym_name = "band_full_0_1"}
    %band_free_0_0 = aie.lock(%logical_mem_61) {init = 4 : i32, sym_name = "band_free_0_0"}
    %band_free_0_1 = aie.lock(%logical_mem_61) {init = 4 : i32, sym_name = "band_free_0_1"}
    %a_prod_0_0 = aie.lock(%logical_core) {init = 2 : i32, sym_name = "a_prod_0_0"}
    %a_cons_0_0 = aie.lock(%logical_core) {init = 0 : i32, sym_name = "a_cons_0_0"}
    %a_prod_0_1 = aie.lock(%logical_core_0) {init = 2 : i32, sym_name = "a_prod_0_1"}
    %a_cons_0_1 = aie.lock(%logical_core_0) {init = 0 : i32, sym_name = "a_cons_0_1"}
    %a_prod_0_2 = aie.lock(%logical_core_1) {init = 2 : i32, sym_name = "a_prod_0_2"}
    %a_cons_0_2 = aie.lock(%logical_core_1) {init = 0 : i32, sym_name = "a_cons_0_2"}
    %a_prod_0_3 = aie.lock(%logical_core_2) {init = 2 : i32, sym_name = "a_prod_0_3"}
    %a_cons_0_3 = aie.lock(%logical_core_2) {init = 0 : i32, sym_name = "a_cons_0_3"}
    %band_full_1_0 = aie.lock(%logical_mem_63) {init = 0 : i32, sym_name = "band_full_1_0"}
    %band_full_1_1 = aie.lock(%logical_mem_63) {init = 0 : i32, sym_name = "band_full_1_1"}
    %band_free_1_0 = aie.lock(%logical_mem_63) {init = 4 : i32, sym_name = "band_free_1_0"}
    %band_free_1_1 = aie.lock(%logical_mem_63) {init = 4 : i32, sym_name = "band_free_1_1"}
    %a_prod_1_0 = aie.lock(%logical_core_3) {init = 2 : i32, sym_name = "a_prod_1_0"}
    %a_cons_1_0 = aie.lock(%logical_core_3) {init = 0 : i32, sym_name = "a_cons_1_0"}
    %a_prod_1_1 = aie.lock(%logical_core_4) {init = 2 : i32, sym_name = "a_prod_1_1"}
    %a_cons_1_1 = aie.lock(%logical_core_4) {init = 0 : i32, sym_name = "a_cons_1_1"}
    %a_prod_1_2 = aie.lock(%logical_core_5) {init = 2 : i32, sym_name = "a_prod_1_2"}
    %a_cons_1_2 = aie.lock(%logical_core_5) {init = 0 : i32, sym_name = "a_cons_1_2"}
    %a_prod_1_3 = aie.lock(%logical_core_6) {init = 2 : i32, sym_name = "a_prod_1_3"}
    %a_cons_1_3 = aie.lock(%logical_core_6) {init = 0 : i32, sym_name = "a_cons_1_3"}
    %band_full_2_0 = aie.lock(%logical_mem_65) {init = 0 : i32, sym_name = "band_full_2_0"}
    %band_full_2_1 = aie.lock(%logical_mem_65) {init = 0 : i32, sym_name = "band_full_2_1"}
    %band_free_2_0 = aie.lock(%logical_mem_65) {init = 4 : i32, sym_name = "band_free_2_0"}
    %band_free_2_1 = aie.lock(%logical_mem_65) {init = 4 : i32, sym_name = "band_free_2_1"}
    %a_prod_2_0 = aie.lock(%logical_core_7) {init = 2 : i32, sym_name = "a_prod_2_0"}
    %a_cons_2_0 = aie.lock(%logical_core_7) {init = 0 : i32, sym_name = "a_cons_2_0"}
    %a_prod_2_1 = aie.lock(%logical_core_8) {init = 2 : i32, sym_name = "a_prod_2_1"}
    %a_cons_2_1 = aie.lock(%logical_core_8) {init = 0 : i32, sym_name = "a_cons_2_1"}
    %a_prod_2_2 = aie.lock(%logical_core_9) {init = 2 : i32, sym_name = "a_prod_2_2"}
    %a_cons_2_2 = aie.lock(%logical_core_9) {init = 0 : i32, sym_name = "a_cons_2_2"}
    %a_prod_2_3 = aie.lock(%logical_core_10) {init = 2 : i32, sym_name = "a_prod_2_3"}
    %a_cons_2_3 = aie.lock(%logical_core_10) {init = 0 : i32, sym_name = "a_cons_2_3"}
    %band_full_3_0 = aie.lock(%logical_mem_67) {init = 0 : i32, sym_name = "band_full_3_0"}
    %band_full_3_1 = aie.lock(%logical_mem_67) {init = 0 : i32, sym_name = "band_full_3_1"}
    %band_free_3_0 = aie.lock(%logical_mem_67) {init = 4 : i32, sym_name = "band_free_3_0"}
    %band_free_3_1 = aie.lock(%logical_mem_67) {init = 4 : i32, sym_name = "band_free_3_1"}
    %a_prod_3_0 = aie.lock(%logical_core_11) {init = 2 : i32, sym_name = "a_prod_3_0"}
    %a_cons_3_0 = aie.lock(%logical_core_11) {init = 0 : i32, sym_name = "a_cons_3_0"}
    %a_prod_3_1 = aie.lock(%logical_core_12) {init = 2 : i32, sym_name = "a_prod_3_1"}
    %a_cons_3_1 = aie.lock(%logical_core_12) {init = 0 : i32, sym_name = "a_cons_3_1"}
    %a_prod_3_2 = aie.lock(%logical_core_13) {init = 2 : i32, sym_name = "a_prod_3_2"}
    %a_cons_3_2 = aie.lock(%logical_core_13) {init = 0 : i32, sym_name = "a_cons_3_2"}
    %a_prod_3_3 = aie.lock(%logical_core_14) {init = 2 : i32, sym_name = "a_prod_3_3"}
    %a_cons_3_3 = aie.lock(%logical_core_14) {init = 0 : i32, sym_name = "a_cons_3_3"}
    %band_full_4_0 = aie.lock(%logical_mem_69) {init = 0 : i32, sym_name = "band_full_4_0"}
    %band_full_4_1 = aie.lock(%logical_mem_69) {init = 0 : i32, sym_name = "band_full_4_1"}
    %band_free_4_0 = aie.lock(%logical_mem_69) {init = 4 : i32, sym_name = "band_free_4_0"}
    %band_free_4_1 = aie.lock(%logical_mem_69) {init = 4 : i32, sym_name = "band_free_4_1"}
    %a_prod_4_0 = aie.lock(%logical_core_15) {init = 2 : i32, sym_name = "a_prod_4_0"}
    %a_cons_4_0 = aie.lock(%logical_core_15) {init = 0 : i32, sym_name = "a_cons_4_0"}
    %a_prod_4_1 = aie.lock(%logical_core_16) {init = 2 : i32, sym_name = "a_prod_4_1"}
    %a_cons_4_1 = aie.lock(%logical_core_16) {init = 0 : i32, sym_name = "a_cons_4_1"}
    %a_prod_4_2 = aie.lock(%logical_core_17) {init = 2 : i32, sym_name = "a_prod_4_2"}
    %a_cons_4_2 = aie.lock(%logical_core_17) {init = 0 : i32, sym_name = "a_cons_4_2"}
    %a_prod_4_3 = aie.lock(%logical_core_18) {init = 2 : i32, sym_name = "a_prod_4_3"}
    %a_cons_4_3 = aie.lock(%logical_core_18) {init = 0 : i32, sym_name = "a_cons_4_3"}
    %band_full_5_0 = aie.lock(%logical_mem_71) {init = 0 : i32, sym_name = "band_full_5_0"}
    %band_full_5_1 = aie.lock(%logical_mem_71) {init = 0 : i32, sym_name = "band_full_5_1"}
    %band_free_5_0 = aie.lock(%logical_mem_71) {init = 4 : i32, sym_name = "band_free_5_0"}
    %band_free_5_1 = aie.lock(%logical_mem_71) {init = 4 : i32, sym_name = "band_free_5_1"}
    %a_prod_5_0 = aie.lock(%logical_core_19) {init = 2 : i32, sym_name = "a_prod_5_0"}
    %a_cons_5_0 = aie.lock(%logical_core_19) {init = 0 : i32, sym_name = "a_cons_5_0"}
    %a_prod_5_1 = aie.lock(%logical_core_20) {init = 2 : i32, sym_name = "a_prod_5_1"}
    %a_cons_5_1 = aie.lock(%logical_core_20) {init = 0 : i32, sym_name = "a_cons_5_1"}
    %a_prod_5_2 = aie.lock(%logical_core_21) {init = 2 : i32, sym_name = "a_prod_5_2"}
    %a_cons_5_2 = aie.lock(%logical_core_21) {init = 0 : i32, sym_name = "a_cons_5_2"}
    %a_prod_5_3 = aie.lock(%logical_core_22) {init = 2 : i32, sym_name = "a_prod_5_3"}
    %a_cons_5_3 = aie.lock(%logical_core_22) {init = 0 : i32, sym_name = "a_cons_5_3"}
    %band_full_6_0 = aie.lock(%logical_mem_73) {init = 0 : i32, sym_name = "band_full_6_0"}
    %band_full_6_1 = aie.lock(%logical_mem_73) {init = 0 : i32, sym_name = "band_full_6_1"}
    %band_free_6_0 = aie.lock(%logical_mem_73) {init = 4 : i32, sym_name = "band_free_6_0"}
    %band_free_6_1 = aie.lock(%logical_mem_73) {init = 4 : i32, sym_name = "band_free_6_1"}
    %a_prod_6_0 = aie.lock(%logical_core_23) {init = 2 : i32, sym_name = "a_prod_6_0"}
    %a_cons_6_0 = aie.lock(%logical_core_23) {init = 0 : i32, sym_name = "a_cons_6_0"}
    %a_prod_6_1 = aie.lock(%logical_core_24) {init = 2 : i32, sym_name = "a_prod_6_1"}
    %a_cons_6_1 = aie.lock(%logical_core_24) {init = 0 : i32, sym_name = "a_cons_6_1"}
    %a_prod_6_2 = aie.lock(%logical_core_25) {init = 2 : i32, sym_name = "a_prod_6_2"}
    %a_cons_6_2 = aie.lock(%logical_core_25) {init = 0 : i32, sym_name = "a_cons_6_2"}
    %a_prod_6_3 = aie.lock(%logical_core_26) {init = 2 : i32, sym_name = "a_prod_6_3"}
    %a_cons_6_3 = aie.lock(%logical_core_26) {init = 0 : i32, sym_name = "a_cons_6_3"}
    %band_full_7_0 = aie.lock(%logical_mem_75) {init = 0 : i32, sym_name = "band_full_7_0"}
    %band_full_7_1 = aie.lock(%logical_mem_75) {init = 0 : i32, sym_name = "band_full_7_1"}
    %band_free_7_0 = aie.lock(%logical_mem_75) {init = 4 : i32, sym_name = "band_free_7_0"}
    %band_free_7_1 = aie.lock(%logical_mem_75) {init = 4 : i32, sym_name = "band_free_7_1"}
    %a_prod_7_0 = aie.lock(%logical_core_27) {init = 2 : i32, sym_name = "a_prod_7_0"}
    %a_cons_7_0 = aie.lock(%logical_core_27) {init = 0 : i32, sym_name = "a_cons_7_0"}
    %a_prod_7_1 = aie.lock(%logical_core_28) {init = 2 : i32, sym_name = "a_prod_7_1"}
    %a_cons_7_1 = aie.lock(%logical_core_28) {init = 0 : i32, sym_name = "a_cons_7_1"}
    %a_prod_7_2 = aie.lock(%logical_core_29) {init = 2 : i32, sym_name = "a_prod_7_2"}
    %a_cons_7_2 = aie.lock(%logical_core_29) {init = 0 : i32, sym_name = "a_cons_7_2"}
    %a_prod_7_3 = aie.lock(%logical_core_30) {init = 2 : i32, sym_name = "a_prod_7_3"}
    %a_cons_7_3 = aie.lock(%logical_core_30) {init = 0 : i32, sym_name = "a_cons_7_3"}
    %band_0_0 = aie.buffer(%logical_mem_61) {sym_name = "band_0_0"} : memref<32768xbf16> 
    %band_0_1 = aie.buffer(%logical_mem_61) {sym_name = "band_0_1"} : memref<32768xbf16> 
    %a_0_0_0 = aie.buffer(%logical_core) {sym_name = "a_0_0_0"} : memref<32x64xbf16> 
    %a_0_0_1 = aie.buffer(%logical_core) {sym_name = "a_0_0_1"} : memref<32x64xbf16> 
    %a_0_1_0 = aie.buffer(%logical_core_0) {sym_name = "a_0_1_0"} : memref<32x64xbf16> 
    %a_0_1_1 = aie.buffer(%logical_core_0) {sym_name = "a_0_1_1"} : memref<32x64xbf16> 
    %a_0_2_0 = aie.buffer(%logical_core_1) {sym_name = "a_0_2_0"} : memref<32x64xbf16> 
    %a_0_2_1 = aie.buffer(%logical_core_1) {sym_name = "a_0_2_1"} : memref<32x64xbf16> 
    %a_0_3_0 = aie.buffer(%logical_core_2) {sym_name = "a_0_3_0"} : memref<32x64xbf16> 
    %a_0_3_1 = aie.buffer(%logical_core_2) {sym_name = "a_0_3_1"} : memref<32x64xbf16> 
    %band_1_0 = aie.buffer(%logical_mem_63) {sym_name = "band_1_0"} : memref<32768xbf16> 
    %band_1_1 = aie.buffer(%logical_mem_63) {sym_name = "band_1_1"} : memref<32768xbf16> 
    %a_1_0_0 = aie.buffer(%logical_core_3) {sym_name = "a_1_0_0"} : memref<32x64xbf16> 
    %a_1_0_1 = aie.buffer(%logical_core_3) {sym_name = "a_1_0_1"} : memref<32x64xbf16> 
    %a_1_1_0 = aie.buffer(%logical_core_4) {sym_name = "a_1_1_0"} : memref<32x64xbf16> 
    %a_1_1_1 = aie.buffer(%logical_core_4) {sym_name = "a_1_1_1"} : memref<32x64xbf16> 
    %a_1_2_0 = aie.buffer(%logical_core_5) {sym_name = "a_1_2_0"} : memref<32x64xbf16> 
    %a_1_2_1 = aie.buffer(%logical_core_5) {sym_name = "a_1_2_1"} : memref<32x64xbf16> 
    %a_1_3_0 = aie.buffer(%logical_core_6) {sym_name = "a_1_3_0"} : memref<32x64xbf16> 
    %a_1_3_1 = aie.buffer(%logical_core_6) {sym_name = "a_1_3_1"} : memref<32x64xbf16> 
    %band_2_0 = aie.buffer(%logical_mem_65) {sym_name = "band_2_0"} : memref<32768xbf16> 
    %band_2_1 = aie.buffer(%logical_mem_65) {sym_name = "band_2_1"} : memref<32768xbf16> 
    %a_2_0_0 = aie.buffer(%logical_core_7) {sym_name = "a_2_0_0"} : memref<32x64xbf16> 
    %a_2_0_1 = aie.buffer(%logical_core_7) {sym_name = "a_2_0_1"} : memref<32x64xbf16> 
    %a_2_1_0 = aie.buffer(%logical_core_8) {sym_name = "a_2_1_0"} : memref<32x64xbf16> 
    %a_2_1_1 = aie.buffer(%logical_core_8) {sym_name = "a_2_1_1"} : memref<32x64xbf16> 
    %a_2_2_0 = aie.buffer(%logical_core_9) {sym_name = "a_2_2_0"} : memref<32x64xbf16> 
    %a_2_2_1 = aie.buffer(%logical_core_9) {sym_name = "a_2_2_1"} : memref<32x64xbf16> 
    %a_2_3_0 = aie.buffer(%logical_core_10) {sym_name = "a_2_3_0"} : memref<32x64xbf16> 
    %a_2_3_1 = aie.buffer(%logical_core_10) {sym_name = "a_2_3_1"} : memref<32x64xbf16> 
    %band_3_0 = aie.buffer(%logical_mem_67) {sym_name = "band_3_0"} : memref<32768xbf16> 
    %band_3_1 = aie.buffer(%logical_mem_67) {sym_name = "band_3_1"} : memref<32768xbf16> 
    %a_3_0_0 = aie.buffer(%logical_core_11) {sym_name = "a_3_0_0"} : memref<32x64xbf16> 
    %a_3_0_1 = aie.buffer(%logical_core_11) {sym_name = "a_3_0_1"} : memref<32x64xbf16> 
    %a_3_1_0 = aie.buffer(%logical_core_12) {sym_name = "a_3_1_0"} : memref<32x64xbf16> 
    %a_3_1_1 = aie.buffer(%logical_core_12) {sym_name = "a_3_1_1"} : memref<32x64xbf16> 
    %a_3_2_0 = aie.buffer(%logical_core_13) {sym_name = "a_3_2_0"} : memref<32x64xbf16> 
    %a_3_2_1 = aie.buffer(%logical_core_13) {sym_name = "a_3_2_1"} : memref<32x64xbf16> 
    %a_3_3_0 = aie.buffer(%logical_core_14) {sym_name = "a_3_3_0"} : memref<32x64xbf16> 
    %a_3_3_1 = aie.buffer(%logical_core_14) {sym_name = "a_3_3_1"} : memref<32x64xbf16> 
    %band_4_0 = aie.buffer(%logical_mem_69) {sym_name = "band_4_0"} : memref<32768xbf16> 
    %band_4_1 = aie.buffer(%logical_mem_69) {sym_name = "band_4_1"} : memref<32768xbf16> 
    %a_4_0_0 = aie.buffer(%logical_core_15) {sym_name = "a_4_0_0"} : memref<32x64xbf16> 
    %a_4_0_1 = aie.buffer(%logical_core_15) {sym_name = "a_4_0_1"} : memref<32x64xbf16> 
    %a_4_1_0 = aie.buffer(%logical_core_16) {sym_name = "a_4_1_0"} : memref<32x64xbf16> 
    %a_4_1_1 = aie.buffer(%logical_core_16) {sym_name = "a_4_1_1"} : memref<32x64xbf16> 
    %a_4_2_0 = aie.buffer(%logical_core_17) {sym_name = "a_4_2_0"} : memref<32x64xbf16> 
    %a_4_2_1 = aie.buffer(%logical_core_17) {sym_name = "a_4_2_1"} : memref<32x64xbf16> 
    %a_4_3_0 = aie.buffer(%logical_core_18) {sym_name = "a_4_3_0"} : memref<32x64xbf16> 
    %a_4_3_1 = aie.buffer(%logical_core_18) {sym_name = "a_4_3_1"} : memref<32x64xbf16> 
    %band_5_0 = aie.buffer(%logical_mem_71) {sym_name = "band_5_0"} : memref<32768xbf16> 
    %band_5_1 = aie.buffer(%logical_mem_71) {sym_name = "band_5_1"} : memref<32768xbf16> 
    %a_5_0_0 = aie.buffer(%logical_core_19) {sym_name = "a_5_0_0"} : memref<32x64xbf16> 
    %a_5_0_1 = aie.buffer(%logical_core_19) {sym_name = "a_5_0_1"} : memref<32x64xbf16> 
    %a_5_1_0 = aie.buffer(%logical_core_20) {sym_name = "a_5_1_0"} : memref<32x64xbf16> 
    %a_5_1_1 = aie.buffer(%logical_core_20) {sym_name = "a_5_1_1"} : memref<32x64xbf16> 
    %a_5_2_0 = aie.buffer(%logical_core_21) {sym_name = "a_5_2_0"} : memref<32x64xbf16> 
    %a_5_2_1 = aie.buffer(%logical_core_21) {sym_name = "a_5_2_1"} : memref<32x64xbf16> 
    %a_5_3_0 = aie.buffer(%logical_core_22) {sym_name = "a_5_3_0"} : memref<32x64xbf16> 
    %a_5_3_1 = aie.buffer(%logical_core_22) {sym_name = "a_5_3_1"} : memref<32x64xbf16> 
    %band_6_0 = aie.buffer(%logical_mem_73) {sym_name = "band_6_0"} : memref<32768xbf16> 
    %band_6_1 = aie.buffer(%logical_mem_73) {sym_name = "band_6_1"} : memref<32768xbf16> 
    %a_6_0_0 = aie.buffer(%logical_core_23) {sym_name = "a_6_0_0"} : memref<32x64xbf16> 
    %a_6_0_1 = aie.buffer(%logical_core_23) {sym_name = "a_6_0_1"} : memref<32x64xbf16> 
    %a_6_1_0 = aie.buffer(%logical_core_24) {sym_name = "a_6_1_0"} : memref<32x64xbf16> 
    %a_6_1_1 = aie.buffer(%logical_core_24) {sym_name = "a_6_1_1"} : memref<32x64xbf16> 
    %a_6_2_0 = aie.buffer(%logical_core_25) {sym_name = "a_6_2_0"} : memref<32x64xbf16> 
    %a_6_2_1 = aie.buffer(%logical_core_25) {sym_name = "a_6_2_1"} : memref<32x64xbf16> 
    %a_6_3_0 = aie.buffer(%logical_core_26) {sym_name = "a_6_3_0"} : memref<32x64xbf16> 
    %a_6_3_1 = aie.buffer(%logical_core_26) {sym_name = "a_6_3_1"} : memref<32x64xbf16> 
    %band_7_0 = aie.buffer(%logical_mem_75) {sym_name = "band_7_0"} : memref<32768xbf16> 
    %band_7_1 = aie.buffer(%logical_mem_75) {sym_name = "band_7_1"} : memref<32768xbf16> 
    %a_7_0_0 = aie.buffer(%logical_core_27) {sym_name = "a_7_0_0"} : memref<32x64xbf16> 
    %a_7_0_1 = aie.buffer(%logical_core_27) {sym_name = "a_7_0_1"} : memref<32x64xbf16> 
    %a_7_1_0 = aie.buffer(%logical_core_28) {sym_name = "a_7_1_0"} : memref<32x64xbf16> 
    %a_7_1_1 = aie.buffer(%logical_core_28) {sym_name = "a_7_1_1"} : memref<32x64xbf16> 
    %a_7_2_0 = aie.buffer(%logical_core_29) {sym_name = "a_7_2_0"} : memref<32x64xbf16> 
    %a_7_2_1 = aie.buffer(%logical_core_29) {sym_name = "a_7_2_1"} : memref<32x64xbf16> 
    %a_7_3_0 = aie.buffer(%logical_core_30) {sym_name = "a_7_3_0"} : memref<32x64xbf16> 
    %a_7_3_1 = aie.buffer(%logical_core_30) {sym_name = "a_7_3_1"} : memref<32x64xbf16> 
    func.func private @dit_conv_bias(memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) attributes {link_with = "dit_conv_bias.o"}
    func.func private @dit_matmul_quarter(memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) attributes {link_with = "dit_matmul_quarter.o"}
    %rtp_0_0 = aie.buffer(%logical_core) {sym_name = "rtp_0_0"} : memref<4xi32> = dense<0>
    %rtp_0_1 = aie.buffer(%logical_core_0) {sym_name = "rtp_0_1"} : memref<4xi32> = dense<0>
    %rtp_0_2 = aie.buffer(%logical_core_1) {sym_name = "rtp_0_2"} : memref<4xi32> = dense<0>
    %rtp_0_3 = aie.buffer(%logical_core_2) {sym_name = "rtp_0_3"} : memref<4xi32> = dense<0>
    %rtp_1_0 = aie.buffer(%logical_core_3) {sym_name = "rtp_1_0"} : memref<4xi32> = dense<0>
    %rtp_1_1 = aie.buffer(%logical_core_4) {sym_name = "rtp_1_1"} : memref<4xi32> = dense<0>
    %rtp_1_2 = aie.buffer(%logical_core_5) {sym_name = "rtp_1_2"} : memref<4xi32> = dense<0>
    %rtp_1_3 = aie.buffer(%logical_core_6) {sym_name = "rtp_1_3"} : memref<4xi32> = dense<0>
    %rtp_2_0 = aie.buffer(%logical_core_7) {sym_name = "rtp_2_0"} : memref<4xi32> = dense<0>
    %rtp_2_1 = aie.buffer(%logical_core_8) {sym_name = "rtp_2_1"} : memref<4xi32> = dense<0>
    %rtp_2_2 = aie.buffer(%logical_core_9) {sym_name = "rtp_2_2"} : memref<4xi32> = dense<0>
    %rtp_2_3 = aie.buffer(%logical_core_10) {sym_name = "rtp_2_3"} : memref<4xi32> = dense<0>
    %rtp_3_0 = aie.buffer(%logical_core_11) {sym_name = "rtp_3_0"} : memref<4xi32> = dense<0>
    %rtp_3_1 = aie.buffer(%logical_core_12) {sym_name = "rtp_3_1"} : memref<4xi32> = dense<0>
    %rtp_3_2 = aie.buffer(%logical_core_13) {sym_name = "rtp_3_2"} : memref<4xi32> = dense<0>
    %rtp_3_3 = aie.buffer(%logical_core_14) {sym_name = "rtp_3_3"} : memref<4xi32> = dense<0>
    %rtp_4_0 = aie.buffer(%logical_core_15) {sym_name = "rtp_4_0"} : memref<4xi32> = dense<0>
    %rtp_4_1 = aie.buffer(%logical_core_16) {sym_name = "rtp_4_1"} : memref<4xi32> = dense<0>
    %rtp_4_2 = aie.buffer(%logical_core_17) {sym_name = "rtp_4_2"} : memref<4xi32> = dense<0>
    %rtp_4_3 = aie.buffer(%logical_core_18) {sym_name = "rtp_4_3"} : memref<4xi32> = dense<0>
    %rtp_5_0 = aie.buffer(%logical_core_19) {sym_name = "rtp_5_0"} : memref<4xi32> = dense<0>
    %rtp_5_1 = aie.buffer(%logical_core_20) {sym_name = "rtp_5_1"} : memref<4xi32> = dense<0>
    %rtp_5_2 = aie.buffer(%logical_core_21) {sym_name = "rtp_5_2"} : memref<4xi32> = dense<0>
    %rtp_5_3 = aie.buffer(%logical_core_22) {sym_name = "rtp_5_3"} : memref<4xi32> = dense<0>
    %rtp_6_0 = aie.buffer(%logical_core_23) {sym_name = "rtp_6_0"} : memref<4xi32> = dense<0>
    %rtp_6_1 = aie.buffer(%logical_core_24) {sym_name = "rtp_6_1"} : memref<4xi32> = dense<0>
    %rtp_6_2 = aie.buffer(%logical_core_25) {sym_name = "rtp_6_2"} : memref<4xi32> = dense<0>
    %rtp_6_3 = aie.buffer(%logical_core_26) {sym_name = "rtp_6_3"} : memref<4xi32> = dense<0>
    %rtp_7_0 = aie.buffer(%logical_core_27) {sym_name = "rtp_7_0"} : memref<4xi32> = dense<0>
    %rtp_7_1 = aie.buffer(%logical_core_28) {sym_name = "rtp_7_1"} : memref<4xi32> = dense<0>
    %rtp_7_2 = aie.buffer(%logical_core_29) {sym_name = "rtp_7_2"} : memref<4xi32> = dense<0>
    %rtp_7_3 = aie.buffer(%logical_core_30) {sym_name = "rtp_7_3"} : memref<4xi32> = dense<0>
    %lock_0_2 = aie.lock(%logical_core)
    %core_0_2 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_0_2, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_0_0[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_0_0[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_0_0, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_0_0_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_0_0, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_0_0, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_0_0_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_0_0, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_0_0, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_0_0_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_0_0, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_0_0, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_0_0_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_0_0, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_0_2, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_0_3 = aie.lock(%logical_core_0)
    %core_0_3 = aie.core(%logical_core_0) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_0_3, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_0_1[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_0_1[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_0_1, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_0_1_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_0_1, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_0_1, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_0_1_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_0_1, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_0_1, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_0_1_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_0_1, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_0_1, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_0_1_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_0_1, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_0_3, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_0_4 = aie.lock(%logical_core_1)
    %core_0_4 = aie.core(%logical_core_1) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_0_4, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_0_2[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_0_2[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_0_2, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_0_2_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_0_2, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_0_2, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_0_2_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_0_2, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_0_2, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_0_2_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_0_2, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_0_2, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_0_2_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_0_2, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_0_4, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_0_5 = aie.lock(%logical_core_2)
    %core_0_5 = aie.core(%logical_core_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_0_5, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_0_3[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_0_3[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_0(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_0_3, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_0_3_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_0_3, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_0_3, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_0_3_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_0_3, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_0_3, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_0_3_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_0_3, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_0_3, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_0_3_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_0_3, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_0_5, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_1_2 = aie.lock(%logical_core_3)
    %core_1_2 = aie.core(%logical_core_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_1_2, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_1_0[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_1_0[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_1_0, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_1_0_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_1_0, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_1_0, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_1_0_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_1_0, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_1_0, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_1_0_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_1_0, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_1_0, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_1_0_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_1_0, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_1_2, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_1_3 = aie.lock(%logical_core_4)
    %core_1_3 = aie.core(%logical_core_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_1_3, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_1_1[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_1_1[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_1_1, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_1_1_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_1_1, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_1_1, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_1_1_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_1_1, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_1_1, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_1_1_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_1_1, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_1_1, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_1_1_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_1_1, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_1_3, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_1_4 = aie.lock(%logical_core_5)
    %core_1_4 = aie.core(%logical_core_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_1_4, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_1_2[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_1_2[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_1_2, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_1_2_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_1_2, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_1_2, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_1_2_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_1_2, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_1_2, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_1_2_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_1_2, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_1_2, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_1_2_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_1_2, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_1_4, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_1_5 = aie.lock(%logical_core_6)
    %core_1_5 = aie.core(%logical_core_6) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_1_5, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_1_3[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_1_3[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_1(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_1_3, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_1_3_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_1_3, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_1_3, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_1_3_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_1_3, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_1_3, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_1_3_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_1_3, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_1_3, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_1_3_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_1_3, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_1_5, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_2_2 = aie.lock(%logical_core_7)
    %core_2_2 = aie.core(%logical_core_7) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_2_2, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_2_0[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_2_0[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_2_0, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_2_0_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_2_0, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_2_0, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_2_0_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_2_0, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_2_0, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_2_0_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_2_0, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_2_0, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_2_0_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_2_0, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_2_2, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_2_3 = aie.lock(%logical_core_8)
    %core_2_3 = aie.core(%logical_core_8) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_2_3, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_2_1[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_2_1[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_2_1, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_2_1_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_2_1, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_2_1, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_2_1_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_2_1, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_2_1, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_2_1_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_2_1, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_2_1, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_2_1_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_2_1, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_1(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_2_3, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_2_4 = aie.lock(%logical_core_9)
    %core_2_4 = aie.core(%logical_core_9) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_2_4, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_2_2[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_2_2[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_2_2, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_2_2_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_2_2, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_2_2, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_2_2_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_2_2, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_2_2, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_2_2_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_2_2, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_2_2, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_2_2_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_2_2, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_2(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_2_4, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_2_5 = aie.lock(%logical_core_10)
    %core_2_5 = aie.core(%logical_core_10) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_2_5, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_2_3[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_2_3[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_2(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_2_3, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_2_3_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_2_3, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_2_3, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_2_3_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_2_3, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_2_3, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_2_3_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_2_3, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_2_3, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_2_3_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_2_3, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_3(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_2_5, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_3_2 = aie.lock(%logical_core_11)
    %core_3_2 = aie.core(%logical_core_11) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_3_2, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_3_0[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_3_0[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_3_0, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_3_0_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_3_0, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_3_0, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_3_0_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_3_0, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_3_0, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_3_0_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_3_0, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_3_0, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_3_0_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_3_0, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_3_2, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_3_3 = aie.lock(%logical_core_12)
    %core_3_3 = aie.core(%logical_core_12) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_3_3, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_3_1[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_3_1[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_3_1, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_3_1_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_3_1, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_3_1, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_3_1_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_3_1, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_3_1, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_3_1_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_3_1, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_3_1, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_3_1_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_3_1, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_1(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_3_3, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_3_4 = aie.lock(%logical_core_13)
    %core_3_4 = aie.core(%logical_core_13) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_3_4, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_3_2[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_3_2[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_3_2, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_3_2_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_3_2, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_3_2, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_3_2_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_3_2, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_3_2, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_3_2_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_3_2, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_3_2, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_3_2_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_3_2, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_2(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_3_4, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_3_5 = aie.lock(%logical_core_14)
    %core_3_5 = aie.core(%logical_core_14) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_3_5, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_3_3[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_3_3[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_3(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_3_3, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_3_3_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_3_3, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_3_3, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_3_3_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_3_3, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_3_3, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_3_3_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_3_3, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_3_3, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_3_3_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_3_3, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_3(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_3_5, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_4_2 = aie.lock(%logical_core_15)
    %core_4_2 = aie.core(%logical_core_15) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_4_2, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_4_0[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_4_0[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_4_0, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_4_0_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_4_0, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_4_0, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_4_0_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_4_0, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_4_0, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_4_0_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_4_0, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_4_0, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_4_0_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_4_0, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_4(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_4_2, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_4_3 = aie.lock(%logical_core_16)
    %core_4_3 = aie.core(%logical_core_16) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_4_3, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_4_1[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_4_1[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_4_1, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_4_1_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_4_1, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_4_1, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_4_1_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_4_1, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_4_1, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_4_1_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_4_1, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_4_1, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_4_1_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_4_1, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_4(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_1(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_4_3, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_4_4 = aie.lock(%logical_core_17)
    %core_4_4 = aie.core(%logical_core_17) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_4_4, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_4_2[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_4_2[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_4_2, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_4_2_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_4_2, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_4_2, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_4_2_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_4_2, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_4_2, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_4_2_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_4_2, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_4_2, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_4_2_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_4_2, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_4(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_2(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_4_4, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_4_5 = aie.lock(%logical_core_18)
    %core_4_5 = aie.core(%logical_core_18) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_4_5, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_4_3[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_4_3[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_4(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_4_3, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_4_3_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_4_3, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_4_3, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_4_3_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_4_3, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_4_3, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_4_3_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_4_3, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_4_3, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_4_3_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_4_3, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_4(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_3(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_4_5, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_5_2 = aie.lock(%logical_core_19)
    %core_5_2 = aie.core(%logical_core_19) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_5_2, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_5_0[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_5_0[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_5_0, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_5_0_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_5_0, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_5_0, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_5_0_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_5_0, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_5_0, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_5_0_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_5_0, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_5_0, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_5_0_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_5_0, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_5(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_5_2, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_5_3 = aie.lock(%logical_core_20)
    %core_5_3 = aie.core(%logical_core_20) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_5_3, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_5_1[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_5_1[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_5_1, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_5_1_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_5_1, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_5_1, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_5_1_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_5_1, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_5_1, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_5_1_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_5_1, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_5_1, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_5_1_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_5_1, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_5(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_1(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_5_3, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_5_4 = aie.lock(%logical_core_21)
    %core_5_4 = aie.core(%logical_core_21) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_5_4, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_5_2[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_5_2[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_5_2, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_5_2_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_5_2, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_5_2, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_5_2_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_5_2, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_5_2, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_5_2_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_5_2, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_5_2, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_5_2_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_5_2, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_5(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_2(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_5_4, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_5_5 = aie.lock(%logical_core_22)
    %core_5_5 = aie.core(%logical_core_22) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_5_5, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_5_3[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_5_3[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_5(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_5_3, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_5_3_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_5_3, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_5_3, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_5_3_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_5_3, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_5_3, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_5_3_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_5_3, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_5_3, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_5_3_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_5_3, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_5(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_3(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_5_5, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_6_2 = aie.lock(%logical_core_23)
    %core_6_2 = aie.core(%logical_core_23) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_6_2, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_6_0[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_6_0[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_6_0, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_6_0_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_6_0, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_6_0, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_6_0_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_6_0, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_6_0, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_6_0_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_6_0, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_6_0, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_6_0_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_6_0, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_6(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_6_2, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_6_3 = aie.lock(%logical_core_24)
    %core_6_3 = aie.core(%logical_core_24) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_6_3, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_6_1[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_6_1[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_6_1, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_6_1_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_6_1, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_6_1, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_6_1_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_6_1, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_6_1, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_6_1_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_6_1, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_6_1, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_6_1_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_6_1, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_6(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_1(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_6_3, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_6_4 = aie.lock(%logical_core_25)
    %core_6_4 = aie.core(%logical_core_25) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_6_4, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_6_2[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_6_2[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_6_2, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_6_2_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_6_2, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_6_2, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_6_2_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_6_2, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_6_2, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_6_2_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_6_2, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_6_2, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_6_2_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_6_2, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_6(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_2(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_6_4, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_6_5 = aie.lock(%logical_core_26)
    %core_6_5 = aie.core(%logical_core_26) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_6_5, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_6_3[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_6_3[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_6(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_6_3, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_6_3_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_6_3, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_6_3, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_6_3_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_6_3, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_6_3, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_6_3_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_6_3, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_6_3, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_6_3_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_6_3, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_6(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_3(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_6_5, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_7_2 = aie.lock(%logical_core_27)
    %core_7_2 = aie.core(%logical_core_27) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_7_2, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_7_0[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_7_0[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_7_0, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_7_0_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_7_0, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_7_0, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_7_0_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_7_0, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_7_0, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_7_0_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_7_0, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_7_0, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_7_0_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_7_0, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_7(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_7_2, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_7_3 = aie.lock(%logical_core_28)
    %core_7_3 = aie.core(%logical_core_28) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_7_3, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_7_1[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_7_1[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_7_1, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_7_1_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_7_1, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_7_1, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_7_1_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_7_1, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_7_1, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_7_1_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_7_1, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_7_1, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_7_1_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_7_1, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_7(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_1(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_7_3, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_7_4 = aie.lock(%logical_core_29)
    %core_7_4 = aie.core(%logical_core_29) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_7_4, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_7_2[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_7_2[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_7_2, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_7_2_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_7_2, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_7_2, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_7_2_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_7_2, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_7_2, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_7_2_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_7_2, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_7_2, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_7_2_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_7_2, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_7(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_2(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_7_4, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %lock_7_5 = aie.lock(%logical_core_30)
    %core_7_5 = aie.core(%logical_core_30) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_7_5, Acquire, %c1_i32)
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_7_3[%c0_77] : memref<4xi32>
        %c1_78 = arith.constant 1 : index
        %1 = memref.load %rtp_7_3[%c1_78] : memref<4xi32>
        %c0_79 = arith.constant 0 : index
        %2 = arith.index_cast %0 : i32 to index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %2 step %c1_80 {
          %3 = aie.objectfifo.acquire @C_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<128x128xbf16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<128x128xbf16>> -> memref<128x128xbf16>
          %5 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
          func.call @dit_conv_bias(%6, %4) : (memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
          aie.objectfifo.release @B_L2L1_7(Consume, 1)
          %c0_82 = arith.constant 0 : index
          %7 = arith.index_cast %1 : i32 to index
          %c1_83 = arith.constant 1 : index
          scf.for %arg2 = %c0_82 to %7 step %c1_83 {
            %8 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<64x16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<64x16x!aiex.bfp<"v8bfp16ebs8">>
            %c1_i32_84 = arith.constant 1 : i32
            aie.use_lock(%a_cons_7_3, AcquireGreaterEqual, %c1_i32_84)
            func.call @dit_matmul_quarter(%a_7_3_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_85 = arith.constant 1 : i32
            aie.use_lock(%a_prod_7_3, Release, %c1_i32_85)
            %c1_i32_86 = arith.constant 1 : i32
            aie.use_lock(%a_cons_7_3, AcquireGreaterEqual, %c1_i32_86)
            func.call @dit_matmul_quarter(%a_7_3_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_87 = arith.constant 1 : i32
            aie.use_lock(%a_prod_7_3, Release, %c1_i32_87)
            %c1_i32_88 = arith.constant 1 : i32
            aie.use_lock(%a_cons_7_3, AcquireGreaterEqual, %c1_i32_88)
            func.call @dit_matmul_quarter(%a_7_3_0, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_89 = arith.constant 1 : i32
            aie.use_lock(%a_prod_7_3, Release, %c1_i32_89)
            %c1_i32_90 = arith.constant 1 : i32
            aie.use_lock(%a_cons_7_3, AcquireGreaterEqual, %c1_i32_90)
            func.call @dit_matmul_quarter(%a_7_3_1, %9, %4) : (memref<32x64xbf16>, memref<64x16x!aiex.bfp<"v8bfp16ebs8">>, memref<128x128xbf16>) -> ()
            %c1_i32_91 = arith.constant 1 : i32
            aie.use_lock(%a_prod_7_3, Release, %c1_i32_91)
            aie.objectfifo.release @B_L2L1_7(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_3(Produce, 1)
        }
        %c1_i32_81 = arith.constant 1 : i32
        aie.use_lock(%lock_7_5, Release, %c1_i32_81)
      }
      aie.end
    } {stack_size = 3840 : i32}
    %memtile_dma_0_1 = aie.memtile_dma(%logical_mem_61) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb11
      %c4_i32 = arith.constant 4 : i32
      aie.use_lock(%band_free_0_0, AcquireGreaterEqual, %c4_i32)
      aie.dma_bd(%band_0_0 : memref<32768xbf16> len = 32768 sizes = [4, 128, 8, 8] strides = [1024, 8, 4096, 1])
      %c4_i32_77 = arith.constant 4 : i32
      aie.use_lock(%band_full_0_0, Release, %c4_i32_77)
      aie.next_bd ^bb11
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb12
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%band_0_0 : memref<32768xbf16> len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_0, Release, %c1_i32_78)
      aie.next_bd ^bb12
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb13
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_0, AcquireGreaterEqual, %c1_i32_79)
      aie.dma_bd(%band_0_0 : memref<32768xbf16> offset = 1024 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_80 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_0, Release, %c1_i32_80)
      aie.next_bd ^bb13
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 4, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb14
      %c1_i32_81 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_0, AcquireGreaterEqual, %c1_i32_81)
      aie.dma_bd(%band_0_0 : memref<32768xbf16> offset = 2048 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_82 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_0, Release, %c1_i32_82)
      aie.next_bd ^bb14
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb15
      %c1_i32_83 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_0, AcquireGreaterEqual, %c1_i32_83)
      aie.dma_bd(%band_0_0 : memref<32768xbf16> offset = 3072 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_84 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_0, Release, %c1_i32_84)
      aie.next_bd ^bb15
    ^bb10:  // pred: ^bb8
      aie.end
    ^bb11:  // pred: ^bb1
      %c4_i32_85 = arith.constant 4 : i32
      aie.use_lock(%band_free_0_1, AcquireGreaterEqual, %c4_i32_85)
      aie.dma_bd(%band_0_1 : memref<32768xbf16> len = 32768 sizes = [4, 128, 8, 8] strides = [1024, 8, 4096, 1])
      %c4_i32_86 = arith.constant 4 : i32
      aie.use_lock(%band_full_0_1, Release, %c4_i32_86)
      aie.next_bd ^bb1
    ^bb12:  // pred: ^bb3
      %c1_i32_87 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_1, AcquireGreaterEqual, %c1_i32_87)
      aie.dma_bd(%band_0_1 : memref<32768xbf16> len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_88 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_1, Release, %c1_i32_88)
      aie.next_bd ^bb3
    ^bb13:  // pred: ^bb5
      %c1_i32_89 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_1, AcquireGreaterEqual, %c1_i32_89)
      aie.dma_bd(%band_0_1 : memref<32768xbf16> offset = 1024 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_90 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_1, Release, %c1_i32_90)
      aie.next_bd ^bb5
    ^bb14:  // pred: ^bb7
      %c1_i32_91 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_1, AcquireGreaterEqual, %c1_i32_91)
      aie.dma_bd(%band_0_1 : memref<32768xbf16> offset = 2048 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_92 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_1, Release, %c1_i32_92)
      aie.next_bd ^bb7
    ^bb15:  // pred: ^bb9
      %c1_i32_93 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_1, AcquireGreaterEqual, %c1_i32_93)
      aie.dma_bd(%band_0_1 : memref<32768xbf16> offset = 3072 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_94 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_1, Release, %c1_i32_94)
      aie.next_bd ^bb9
    }
    %mem_0_2 = aie.mem(%logical_core) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_0_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_0_0_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_0_0, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_0_0, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_0_0_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_0_0, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_0_3 = aie.mem(%logical_core_0) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_0_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_0_1_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_0_1, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_0_1, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_0_1_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_0_1, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_0_4 = aie.mem(%logical_core_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_0_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_0_2_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_0_2, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_0_2, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_0_2_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_0_2, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_0_5 = aie.mem(%logical_core_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_0_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_0_3_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_0_3, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_0_3, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_0_3_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_0_3, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %memtile_dma_1_1 = aie.memtile_dma(%logical_mem_63) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb11
      %c4_i32 = arith.constant 4 : i32
      aie.use_lock(%band_free_1_0, AcquireGreaterEqual, %c4_i32)
      aie.dma_bd(%band_1_0 : memref<32768xbf16> len = 32768 sizes = [4, 128, 8, 8] strides = [1024, 8, 4096, 1])
      %c4_i32_77 = arith.constant 4 : i32
      aie.use_lock(%band_full_1_0, Release, %c4_i32_77)
      aie.next_bd ^bb11
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb12
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%band_1_0 : memref<32768xbf16> len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_0, Release, %c1_i32_78)
      aie.next_bd ^bb12
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb13
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_0, AcquireGreaterEqual, %c1_i32_79)
      aie.dma_bd(%band_1_0 : memref<32768xbf16> offset = 1024 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_80 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_0, Release, %c1_i32_80)
      aie.next_bd ^bb13
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 4, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb14
      %c1_i32_81 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_0, AcquireGreaterEqual, %c1_i32_81)
      aie.dma_bd(%band_1_0 : memref<32768xbf16> offset = 2048 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_82 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_0, Release, %c1_i32_82)
      aie.next_bd ^bb14
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb15
      %c1_i32_83 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_0, AcquireGreaterEqual, %c1_i32_83)
      aie.dma_bd(%band_1_0 : memref<32768xbf16> offset = 3072 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_84 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_0, Release, %c1_i32_84)
      aie.next_bd ^bb15
    ^bb10:  // pred: ^bb8
      aie.end
    ^bb11:  // pred: ^bb1
      %c4_i32_85 = arith.constant 4 : i32
      aie.use_lock(%band_free_1_1, AcquireGreaterEqual, %c4_i32_85)
      aie.dma_bd(%band_1_1 : memref<32768xbf16> len = 32768 sizes = [4, 128, 8, 8] strides = [1024, 8, 4096, 1])
      %c4_i32_86 = arith.constant 4 : i32
      aie.use_lock(%band_full_1_1, Release, %c4_i32_86)
      aie.next_bd ^bb1
    ^bb12:  // pred: ^bb3
      %c1_i32_87 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_1, AcquireGreaterEqual, %c1_i32_87)
      aie.dma_bd(%band_1_1 : memref<32768xbf16> len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_88 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_1, Release, %c1_i32_88)
      aie.next_bd ^bb3
    ^bb13:  // pred: ^bb5
      %c1_i32_89 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_1, AcquireGreaterEqual, %c1_i32_89)
      aie.dma_bd(%band_1_1 : memref<32768xbf16> offset = 1024 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_90 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_1, Release, %c1_i32_90)
      aie.next_bd ^bb5
    ^bb14:  // pred: ^bb7
      %c1_i32_91 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_1, AcquireGreaterEqual, %c1_i32_91)
      aie.dma_bd(%band_1_1 : memref<32768xbf16> offset = 2048 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_92 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_1, Release, %c1_i32_92)
      aie.next_bd ^bb7
    ^bb15:  // pred: ^bb9
      %c1_i32_93 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_1, AcquireGreaterEqual, %c1_i32_93)
      aie.dma_bd(%band_1_1 : memref<32768xbf16> offset = 3072 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_94 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_1, Release, %c1_i32_94)
      aie.next_bd ^bb9
    }
    %mem_1_2 = aie.mem(%logical_core_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_1_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_1_0_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_1_0, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_1_0, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_1_0_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_1_0, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_1_3 = aie.mem(%logical_core_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_1_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_1_1_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_1_1, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_1_1, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_1_1_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_1_1, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_1_4 = aie.mem(%logical_core_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_1_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_1_2_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_1_2, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_1_2, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_1_2_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_1_2, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_1_5 = aie.mem(%logical_core_6) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_1_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_1_3_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_1_3, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_1_3, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_1_3_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_1_3, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %memtile_dma_2_1 = aie.memtile_dma(%logical_mem_65) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb11
      %c4_i32 = arith.constant 4 : i32
      aie.use_lock(%band_free_2_0, AcquireGreaterEqual, %c4_i32)
      aie.dma_bd(%band_2_0 : memref<32768xbf16> len = 32768 sizes = [4, 128, 8, 8] strides = [1024, 8, 4096, 1])
      %c4_i32_77 = arith.constant 4 : i32
      aie.use_lock(%band_full_2_0, Release, %c4_i32_77)
      aie.next_bd ^bb11
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb12
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%band_2_0 : memref<32768xbf16> len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_0, Release, %c1_i32_78)
      aie.next_bd ^bb12
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb13
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_0, AcquireGreaterEqual, %c1_i32_79)
      aie.dma_bd(%band_2_0 : memref<32768xbf16> offset = 1024 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_80 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_0, Release, %c1_i32_80)
      aie.next_bd ^bb13
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 4, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb14
      %c1_i32_81 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_0, AcquireGreaterEqual, %c1_i32_81)
      aie.dma_bd(%band_2_0 : memref<32768xbf16> offset = 2048 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_82 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_0, Release, %c1_i32_82)
      aie.next_bd ^bb14
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb15
      %c1_i32_83 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_0, AcquireGreaterEqual, %c1_i32_83)
      aie.dma_bd(%band_2_0 : memref<32768xbf16> offset = 3072 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_84 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_0, Release, %c1_i32_84)
      aie.next_bd ^bb15
    ^bb10:  // pred: ^bb8
      aie.end
    ^bb11:  // pred: ^bb1
      %c4_i32_85 = arith.constant 4 : i32
      aie.use_lock(%band_free_2_1, AcquireGreaterEqual, %c4_i32_85)
      aie.dma_bd(%band_2_1 : memref<32768xbf16> len = 32768 sizes = [4, 128, 8, 8] strides = [1024, 8, 4096, 1])
      %c4_i32_86 = arith.constant 4 : i32
      aie.use_lock(%band_full_2_1, Release, %c4_i32_86)
      aie.next_bd ^bb1
    ^bb12:  // pred: ^bb3
      %c1_i32_87 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_1, AcquireGreaterEqual, %c1_i32_87)
      aie.dma_bd(%band_2_1 : memref<32768xbf16> len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_88 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_1, Release, %c1_i32_88)
      aie.next_bd ^bb3
    ^bb13:  // pred: ^bb5
      %c1_i32_89 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_1, AcquireGreaterEqual, %c1_i32_89)
      aie.dma_bd(%band_2_1 : memref<32768xbf16> offset = 1024 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_90 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_1, Release, %c1_i32_90)
      aie.next_bd ^bb5
    ^bb14:  // pred: ^bb7
      %c1_i32_91 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_1, AcquireGreaterEqual, %c1_i32_91)
      aie.dma_bd(%band_2_1 : memref<32768xbf16> offset = 2048 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_92 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_1, Release, %c1_i32_92)
      aie.next_bd ^bb7
    ^bb15:  // pred: ^bb9
      %c1_i32_93 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_1, AcquireGreaterEqual, %c1_i32_93)
      aie.dma_bd(%band_2_1 : memref<32768xbf16> offset = 3072 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_94 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_1, Release, %c1_i32_94)
      aie.next_bd ^bb9
    }
    %mem_2_2 = aie.mem(%logical_core_7) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_2_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_2_0_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_2_0, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_2_0, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_2_0_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_2_0, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_2_3 = aie.mem(%logical_core_8) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_2_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_2_1_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_2_1, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_2_1, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_2_1_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_2_1, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_2_4 = aie.mem(%logical_core_9) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_2_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_2_2_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_2_2, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_2_2, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_2_2_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_2_2, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_2_5 = aie.mem(%logical_core_10) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_2_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_2_3_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_2_3, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_2_3, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_2_3_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_2_3, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %memtile_dma_3_1 = aie.memtile_dma(%logical_mem_67) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb11
      %c4_i32 = arith.constant 4 : i32
      aie.use_lock(%band_free_3_0, AcquireGreaterEqual, %c4_i32)
      aie.dma_bd(%band_3_0 : memref<32768xbf16> len = 32768 sizes = [4, 128, 8, 8] strides = [1024, 8, 4096, 1])
      %c4_i32_77 = arith.constant 4 : i32
      aie.use_lock(%band_full_3_0, Release, %c4_i32_77)
      aie.next_bd ^bb11
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb12
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%band_3_0 : memref<32768xbf16> len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_0, Release, %c1_i32_78)
      aie.next_bd ^bb12
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb13
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_0, AcquireGreaterEqual, %c1_i32_79)
      aie.dma_bd(%band_3_0 : memref<32768xbf16> offset = 1024 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_80 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_0, Release, %c1_i32_80)
      aie.next_bd ^bb13
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 4, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb14
      %c1_i32_81 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_0, AcquireGreaterEqual, %c1_i32_81)
      aie.dma_bd(%band_3_0 : memref<32768xbf16> offset = 2048 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_82 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_0, Release, %c1_i32_82)
      aie.next_bd ^bb14
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb15
      %c1_i32_83 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_0, AcquireGreaterEqual, %c1_i32_83)
      aie.dma_bd(%band_3_0 : memref<32768xbf16> offset = 3072 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_84 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_0, Release, %c1_i32_84)
      aie.next_bd ^bb15
    ^bb10:  // pred: ^bb8
      aie.end
    ^bb11:  // pred: ^bb1
      %c4_i32_85 = arith.constant 4 : i32
      aie.use_lock(%band_free_3_1, AcquireGreaterEqual, %c4_i32_85)
      aie.dma_bd(%band_3_1 : memref<32768xbf16> len = 32768 sizes = [4, 128, 8, 8] strides = [1024, 8, 4096, 1])
      %c4_i32_86 = arith.constant 4 : i32
      aie.use_lock(%band_full_3_1, Release, %c4_i32_86)
      aie.next_bd ^bb1
    ^bb12:  // pred: ^bb3
      %c1_i32_87 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_1, AcquireGreaterEqual, %c1_i32_87)
      aie.dma_bd(%band_3_1 : memref<32768xbf16> len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_88 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_1, Release, %c1_i32_88)
      aie.next_bd ^bb3
    ^bb13:  // pred: ^bb5
      %c1_i32_89 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_1, AcquireGreaterEqual, %c1_i32_89)
      aie.dma_bd(%band_3_1 : memref<32768xbf16> offset = 1024 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_90 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_1, Release, %c1_i32_90)
      aie.next_bd ^bb5
    ^bb14:  // pred: ^bb7
      %c1_i32_91 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_1, AcquireGreaterEqual, %c1_i32_91)
      aie.dma_bd(%band_3_1 : memref<32768xbf16> offset = 2048 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_92 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_1, Release, %c1_i32_92)
      aie.next_bd ^bb7
    ^bb15:  // pred: ^bb9
      %c1_i32_93 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_1, AcquireGreaterEqual, %c1_i32_93)
      aie.dma_bd(%band_3_1 : memref<32768xbf16> offset = 3072 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_94 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_1, Release, %c1_i32_94)
      aie.next_bd ^bb9
    }
    %mem_3_2 = aie.mem(%logical_core_11) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_3_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_3_0_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_3_0, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_3_0, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_3_0_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_3_0, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_3_3 = aie.mem(%logical_core_12) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_3_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_3_1_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_3_1, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_3_1, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_3_1_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_3_1, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_3_4 = aie.mem(%logical_core_13) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_3_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_3_2_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_3_2, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_3_2, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_3_2_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_3_2, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_3_5 = aie.mem(%logical_core_14) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_3_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_3_3_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_3_3, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_3_3, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_3_3_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_3_3, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %memtile_dma_4_1 = aie.memtile_dma(%logical_mem_69) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb11
      %c4_i32 = arith.constant 4 : i32
      aie.use_lock(%band_free_4_0, AcquireGreaterEqual, %c4_i32)
      aie.dma_bd(%band_4_0 : memref<32768xbf16> len = 32768 sizes = [4, 128, 8, 8] strides = [1024, 8, 4096, 1])
      %c4_i32_77 = arith.constant 4 : i32
      aie.use_lock(%band_full_4_0, Release, %c4_i32_77)
      aie.next_bd ^bb11
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb12
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%band_4_0 : memref<32768xbf16> len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_0, Release, %c1_i32_78)
      aie.next_bd ^bb12
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb13
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_0, AcquireGreaterEqual, %c1_i32_79)
      aie.dma_bd(%band_4_0 : memref<32768xbf16> offset = 1024 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_80 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_0, Release, %c1_i32_80)
      aie.next_bd ^bb13
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 4, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb14
      %c1_i32_81 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_0, AcquireGreaterEqual, %c1_i32_81)
      aie.dma_bd(%band_4_0 : memref<32768xbf16> offset = 2048 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_82 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_0, Release, %c1_i32_82)
      aie.next_bd ^bb14
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb15
      %c1_i32_83 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_0, AcquireGreaterEqual, %c1_i32_83)
      aie.dma_bd(%band_4_0 : memref<32768xbf16> offset = 3072 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_84 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_0, Release, %c1_i32_84)
      aie.next_bd ^bb15
    ^bb10:  // pred: ^bb8
      aie.end
    ^bb11:  // pred: ^bb1
      %c4_i32_85 = arith.constant 4 : i32
      aie.use_lock(%band_free_4_1, AcquireGreaterEqual, %c4_i32_85)
      aie.dma_bd(%band_4_1 : memref<32768xbf16> len = 32768 sizes = [4, 128, 8, 8] strides = [1024, 8, 4096, 1])
      %c4_i32_86 = arith.constant 4 : i32
      aie.use_lock(%band_full_4_1, Release, %c4_i32_86)
      aie.next_bd ^bb1
    ^bb12:  // pred: ^bb3
      %c1_i32_87 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_1, AcquireGreaterEqual, %c1_i32_87)
      aie.dma_bd(%band_4_1 : memref<32768xbf16> len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_88 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_1, Release, %c1_i32_88)
      aie.next_bd ^bb3
    ^bb13:  // pred: ^bb5
      %c1_i32_89 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_1, AcquireGreaterEqual, %c1_i32_89)
      aie.dma_bd(%band_4_1 : memref<32768xbf16> offset = 1024 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_90 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_1, Release, %c1_i32_90)
      aie.next_bd ^bb5
    ^bb14:  // pred: ^bb7
      %c1_i32_91 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_1, AcquireGreaterEqual, %c1_i32_91)
      aie.dma_bd(%band_4_1 : memref<32768xbf16> offset = 2048 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_92 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_1, Release, %c1_i32_92)
      aie.next_bd ^bb7
    ^bb15:  // pred: ^bb9
      %c1_i32_93 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_1, AcquireGreaterEqual, %c1_i32_93)
      aie.dma_bd(%band_4_1 : memref<32768xbf16> offset = 3072 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_94 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_1, Release, %c1_i32_94)
      aie.next_bd ^bb9
    }
    %mem_4_2 = aie.mem(%logical_core_15) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_4_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_4_0_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_4_0, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_4_0, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_4_0_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_4_0, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_4_3 = aie.mem(%logical_core_16) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_4_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_4_1_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_4_1, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_4_1, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_4_1_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_4_1, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_4_4 = aie.mem(%logical_core_17) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_4_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_4_2_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_4_2, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_4_2, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_4_2_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_4_2, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_4_5 = aie.mem(%logical_core_18) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_4_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_4_3_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_4_3, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_4_3, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_4_3_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_4_3, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %memtile_dma_5_1 = aie.memtile_dma(%logical_mem_71) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb11
      %c4_i32 = arith.constant 4 : i32
      aie.use_lock(%band_free_5_0, AcquireGreaterEqual, %c4_i32)
      aie.dma_bd(%band_5_0 : memref<32768xbf16> len = 32768 sizes = [4, 128, 8, 8] strides = [1024, 8, 4096, 1])
      %c4_i32_77 = arith.constant 4 : i32
      aie.use_lock(%band_full_5_0, Release, %c4_i32_77)
      aie.next_bd ^bb11
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb12
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%band_5_0 : memref<32768xbf16> len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_0, Release, %c1_i32_78)
      aie.next_bd ^bb12
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb13
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_0, AcquireGreaterEqual, %c1_i32_79)
      aie.dma_bd(%band_5_0 : memref<32768xbf16> offset = 1024 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_80 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_0, Release, %c1_i32_80)
      aie.next_bd ^bb13
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 4, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb14
      %c1_i32_81 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_0, AcquireGreaterEqual, %c1_i32_81)
      aie.dma_bd(%band_5_0 : memref<32768xbf16> offset = 2048 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_82 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_0, Release, %c1_i32_82)
      aie.next_bd ^bb14
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb15
      %c1_i32_83 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_0, AcquireGreaterEqual, %c1_i32_83)
      aie.dma_bd(%band_5_0 : memref<32768xbf16> offset = 3072 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_84 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_0, Release, %c1_i32_84)
      aie.next_bd ^bb15
    ^bb10:  // pred: ^bb8
      aie.end
    ^bb11:  // pred: ^bb1
      %c4_i32_85 = arith.constant 4 : i32
      aie.use_lock(%band_free_5_1, AcquireGreaterEqual, %c4_i32_85)
      aie.dma_bd(%band_5_1 : memref<32768xbf16> len = 32768 sizes = [4, 128, 8, 8] strides = [1024, 8, 4096, 1])
      %c4_i32_86 = arith.constant 4 : i32
      aie.use_lock(%band_full_5_1, Release, %c4_i32_86)
      aie.next_bd ^bb1
    ^bb12:  // pred: ^bb3
      %c1_i32_87 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_1, AcquireGreaterEqual, %c1_i32_87)
      aie.dma_bd(%band_5_1 : memref<32768xbf16> len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_88 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_1, Release, %c1_i32_88)
      aie.next_bd ^bb3
    ^bb13:  // pred: ^bb5
      %c1_i32_89 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_1, AcquireGreaterEqual, %c1_i32_89)
      aie.dma_bd(%band_5_1 : memref<32768xbf16> offset = 1024 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_90 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_1, Release, %c1_i32_90)
      aie.next_bd ^bb5
    ^bb14:  // pred: ^bb7
      %c1_i32_91 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_1, AcquireGreaterEqual, %c1_i32_91)
      aie.dma_bd(%band_5_1 : memref<32768xbf16> offset = 2048 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_92 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_1, Release, %c1_i32_92)
      aie.next_bd ^bb7
    ^bb15:  // pred: ^bb9
      %c1_i32_93 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_1, AcquireGreaterEqual, %c1_i32_93)
      aie.dma_bd(%band_5_1 : memref<32768xbf16> offset = 3072 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_94 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_1, Release, %c1_i32_94)
      aie.next_bd ^bb9
    }
    %mem_5_2 = aie.mem(%logical_core_19) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_5_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_5_0_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_5_0, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_5_0, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_5_0_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_5_0, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_5_3 = aie.mem(%logical_core_20) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_5_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_5_1_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_5_1, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_5_1, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_5_1_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_5_1, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_5_4 = aie.mem(%logical_core_21) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_5_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_5_2_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_5_2, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_5_2, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_5_2_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_5_2, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_5_5 = aie.mem(%logical_core_22) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_5_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_5_3_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_5_3, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_5_3, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_5_3_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_5_3, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %memtile_dma_6_1 = aie.memtile_dma(%logical_mem_73) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb11
      %c4_i32 = arith.constant 4 : i32
      aie.use_lock(%band_free_6_0, AcquireGreaterEqual, %c4_i32)
      aie.dma_bd(%band_6_0 : memref<32768xbf16> len = 32768 sizes = [4, 128, 8, 8] strides = [1024, 8, 4096, 1])
      %c4_i32_77 = arith.constant 4 : i32
      aie.use_lock(%band_full_6_0, Release, %c4_i32_77)
      aie.next_bd ^bb11
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb12
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%band_6_0 : memref<32768xbf16> len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_0, Release, %c1_i32_78)
      aie.next_bd ^bb12
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb13
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_0, AcquireGreaterEqual, %c1_i32_79)
      aie.dma_bd(%band_6_0 : memref<32768xbf16> offset = 1024 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_80 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_0, Release, %c1_i32_80)
      aie.next_bd ^bb13
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 4, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb14
      %c1_i32_81 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_0, AcquireGreaterEqual, %c1_i32_81)
      aie.dma_bd(%band_6_0 : memref<32768xbf16> offset = 2048 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_82 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_0, Release, %c1_i32_82)
      aie.next_bd ^bb14
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb15
      %c1_i32_83 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_0, AcquireGreaterEqual, %c1_i32_83)
      aie.dma_bd(%band_6_0 : memref<32768xbf16> offset = 3072 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_84 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_0, Release, %c1_i32_84)
      aie.next_bd ^bb15
    ^bb10:  // pred: ^bb8
      aie.end
    ^bb11:  // pred: ^bb1
      %c4_i32_85 = arith.constant 4 : i32
      aie.use_lock(%band_free_6_1, AcquireGreaterEqual, %c4_i32_85)
      aie.dma_bd(%band_6_1 : memref<32768xbf16> len = 32768 sizes = [4, 128, 8, 8] strides = [1024, 8, 4096, 1])
      %c4_i32_86 = arith.constant 4 : i32
      aie.use_lock(%band_full_6_1, Release, %c4_i32_86)
      aie.next_bd ^bb1
    ^bb12:  // pred: ^bb3
      %c1_i32_87 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_1, AcquireGreaterEqual, %c1_i32_87)
      aie.dma_bd(%band_6_1 : memref<32768xbf16> len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_88 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_1, Release, %c1_i32_88)
      aie.next_bd ^bb3
    ^bb13:  // pred: ^bb5
      %c1_i32_89 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_1, AcquireGreaterEqual, %c1_i32_89)
      aie.dma_bd(%band_6_1 : memref<32768xbf16> offset = 1024 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_90 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_1, Release, %c1_i32_90)
      aie.next_bd ^bb5
    ^bb14:  // pred: ^bb7
      %c1_i32_91 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_1, AcquireGreaterEqual, %c1_i32_91)
      aie.dma_bd(%band_6_1 : memref<32768xbf16> offset = 2048 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_92 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_1, Release, %c1_i32_92)
      aie.next_bd ^bb7
    ^bb15:  // pred: ^bb9
      %c1_i32_93 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_1, AcquireGreaterEqual, %c1_i32_93)
      aie.dma_bd(%band_6_1 : memref<32768xbf16> offset = 3072 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_94 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_1, Release, %c1_i32_94)
      aie.next_bd ^bb9
    }
    %mem_6_2 = aie.mem(%logical_core_23) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_6_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_6_0_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_6_0, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_6_0, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_6_0_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_6_0, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_6_3 = aie.mem(%logical_core_24) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_6_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_6_1_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_6_1, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_6_1, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_6_1_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_6_1, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_6_4 = aie.mem(%logical_core_25) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_6_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_6_2_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_6_2, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_6_2, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_6_2_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_6_2, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_6_5 = aie.mem(%logical_core_26) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_6_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_6_3_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_6_3, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_6_3, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_6_3_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_6_3, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %memtile_dma_7_1 = aie.memtile_dma(%logical_mem_75) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb11
      %c4_i32 = arith.constant 4 : i32
      aie.use_lock(%band_free_7_0, AcquireGreaterEqual, %c4_i32)
      aie.dma_bd(%band_7_0 : memref<32768xbf16> len = 32768 sizes = [4, 128, 8, 8] strides = [1024, 8, 4096, 1])
      %c4_i32_77 = arith.constant 4 : i32
      aie.use_lock(%band_full_7_0, Release, %c4_i32_77)
      aie.next_bd ^bb11
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb12
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%band_7_0 : memref<32768xbf16> len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_0, Release, %c1_i32_78)
      aie.next_bd ^bb12
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb13
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_0, AcquireGreaterEqual, %c1_i32_79)
      aie.dma_bd(%band_7_0 : memref<32768xbf16> offset = 1024 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_80 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_0, Release, %c1_i32_80)
      aie.next_bd ^bb13
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 4, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb14
      %c1_i32_81 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_0, AcquireGreaterEqual, %c1_i32_81)
      aie.dma_bd(%band_7_0 : memref<32768xbf16> offset = 2048 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_82 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_0, Release, %c1_i32_82)
      aie.next_bd ^bb14
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb15
      %c1_i32_83 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_0, AcquireGreaterEqual, %c1_i32_83)
      aie.dma_bd(%band_7_0 : memref<32768xbf16> offset = 3072 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_84 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_0, Release, %c1_i32_84)
      aie.next_bd ^bb15
    ^bb10:  // pred: ^bb8
      aie.end
    ^bb11:  // pred: ^bb1
      %c4_i32_85 = arith.constant 4 : i32
      aie.use_lock(%band_free_7_1, AcquireGreaterEqual, %c4_i32_85)
      aie.dma_bd(%band_7_1 : memref<32768xbf16> len = 32768 sizes = [4, 128, 8, 8] strides = [1024, 8, 4096, 1])
      %c4_i32_86 = arith.constant 4 : i32
      aie.use_lock(%band_full_7_1, Release, %c4_i32_86)
      aie.next_bd ^bb1
    ^bb12:  // pred: ^bb3
      %c1_i32_87 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_1, AcquireGreaterEqual, %c1_i32_87)
      aie.dma_bd(%band_7_1 : memref<32768xbf16> len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_88 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_1, Release, %c1_i32_88)
      aie.next_bd ^bb3
    ^bb13:  // pred: ^bb5
      %c1_i32_89 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_1, AcquireGreaterEqual, %c1_i32_89)
      aie.dma_bd(%band_7_1 : memref<32768xbf16> offset = 1024 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_90 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_1, Release, %c1_i32_90)
      aie.next_bd ^bb5
    ^bb14:  // pred: ^bb7
      %c1_i32_91 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_1, AcquireGreaterEqual, %c1_i32_91)
      aie.dma_bd(%band_7_1 : memref<32768xbf16> offset = 2048 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_92 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_1, Release, %c1_i32_92)
      aie.next_bd ^bb7
    ^bb15:  // pred: ^bb9
      %c1_i32_93 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_1, AcquireGreaterEqual, %c1_i32_93)
      aie.dma_bd(%band_7_1 : memref<32768xbf16> offset = 3072 len = 8192 sizes = [4, 8, 256] strides = [256, 4096, 1])
      %c1_i32_94 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_1, Release, %c1_i32_94)
      aie.next_bd ^bb9
    }
    %mem_7_2 = aie.mem(%logical_core_27) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_7_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_7_0_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_7_0, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_7_0, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_7_0_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_7_0, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_7_3 = aie.mem(%logical_core_28) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_7_1, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_7_1_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_7_1, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_7_1, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_7_1_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_7_1, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_7_4 = aie.mem(%logical_core_29) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_7_2, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_7_2_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_7_2, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_7_2, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_7_2_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_7_2, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    %mem_7_5 = aie.mem(%logical_core_30) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb3
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%a_prod_7_3, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%a_7_3_0 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_77 = arith.constant 1 : i32
      aie.use_lock(%a_cons_7_3, Release, %c1_i32_77)
      aie.next_bd ^bb3
    ^bb2:  // pred: ^bb0
      aie.end
    ^bb3:  // pred: ^bb1
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%a_prod_7_3, AcquireGreaterEqual, %c1_i32_78)
      aie.dma_bd(%a_7_3_1 : memref<32x64xbf16> len = 2048 sizes = [8, 4, 64] strides = [64, 512, 1])
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%a_cons_7_3, Release, %c1_i32_79)
      aie.next_bd ^bb1
    }
    aie.runtime_sequence @vae1024_conv100(%arg0: memref<532480xbf16>, %arg1: memref<12288x!aiex.bfp<"v8bfp16ebs8">>, %arg2: memref<4326400xbf16>) {
      %c8_i32 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_0_0, 0, %c8_i32) : i32
      %c2_i32 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_0_0, 1, %c2_i32) : i32
      %c8_i32_77 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_0_1, 0, %c8_i32_77) : i32
      %c2_i32_78 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_0_1, 1, %c2_i32_78) : i32
      %c8_i32_79 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_0_2, 0, %c8_i32_79) : i32
      %c2_i32_80 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_0_2, 1, %c2_i32_80) : i32
      %c8_i32_81 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_0_3, 0, %c8_i32_81) : i32
      %c2_i32_82 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_0_3, 1, %c2_i32_82) : i32
      %c8_i32_83 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_1_0, 0, %c8_i32_83) : i32
      %c2_i32_84 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_1_0, 1, %c2_i32_84) : i32
      %c8_i32_85 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_1_1, 0, %c8_i32_85) : i32
      %c2_i32_86 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_1_1, 1, %c2_i32_86) : i32
      %c8_i32_87 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_1_2, 0, %c8_i32_87) : i32
      %c2_i32_88 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_1_2, 1, %c2_i32_88) : i32
      %c8_i32_89 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_1_3, 0, %c8_i32_89) : i32
      %c2_i32_90 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_1_3, 1, %c2_i32_90) : i32
      %c8_i32_91 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_2_0, 0, %c8_i32_91) : i32
      %c2_i32_92 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_2_0, 1, %c2_i32_92) : i32
      %c8_i32_93 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_2_1, 0, %c8_i32_93) : i32
      %c2_i32_94 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_2_1, 1, %c2_i32_94) : i32
      %c8_i32_95 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_2_2, 0, %c8_i32_95) : i32
      %c2_i32_96 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_2_2, 1, %c2_i32_96) : i32
      %c8_i32_97 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_2_3, 0, %c8_i32_97) : i32
      %c2_i32_98 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_2_3, 1, %c2_i32_98) : i32
      %c8_i32_99 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_3_0, 0, %c8_i32_99) : i32
      %c2_i32_100 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_3_0, 1, %c2_i32_100) : i32
      %c8_i32_101 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_3_1, 0, %c8_i32_101) : i32
      %c2_i32_102 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_3_1, 1, %c2_i32_102) : i32
      %c8_i32_103 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_3_2, 0, %c8_i32_103) : i32
      %c2_i32_104 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_3_2, 1, %c2_i32_104) : i32
      %c8_i32_105 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_3_3, 0, %c8_i32_105) : i32
      %c2_i32_106 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_3_3, 1, %c2_i32_106) : i32
      %c8_i32_107 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_4_0, 0, %c8_i32_107) : i32
      %c2_i32_108 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_4_0, 1, %c2_i32_108) : i32
      %c8_i32_109 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_4_1, 0, %c8_i32_109) : i32
      %c2_i32_110 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_4_1, 1, %c2_i32_110) : i32
      %c8_i32_111 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_4_2, 0, %c8_i32_111) : i32
      %c2_i32_112 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_4_2, 1, %c2_i32_112) : i32
      %c8_i32_113 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_4_3, 0, %c8_i32_113) : i32
      %c2_i32_114 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_4_3, 1, %c2_i32_114) : i32
      %c8_i32_115 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_5_0, 0, %c8_i32_115) : i32
      %c2_i32_116 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_5_0, 1, %c2_i32_116) : i32
      %c8_i32_117 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_5_1, 0, %c8_i32_117) : i32
      %c2_i32_118 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_5_1, 1, %c2_i32_118) : i32
      %c8_i32_119 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_5_2, 0, %c8_i32_119) : i32
      %c2_i32_120 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_5_2, 1, %c2_i32_120) : i32
      %c8_i32_121 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_5_3, 0, %c8_i32_121) : i32
      %c2_i32_122 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_5_3, 1, %c2_i32_122) : i32
      %c8_i32_123 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_6_0, 0, %c8_i32_123) : i32
      %c2_i32_124 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_6_0, 1, %c2_i32_124) : i32
      %c8_i32_125 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_6_1, 0, %c8_i32_125) : i32
      %c2_i32_126 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_6_1, 1, %c2_i32_126) : i32
      %c8_i32_127 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_6_2, 0, %c8_i32_127) : i32
      %c2_i32_128 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_6_2, 1, %c2_i32_128) : i32
      %c8_i32_129 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_6_3, 0, %c8_i32_129) : i32
      %c2_i32_130 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_6_3, 1, %c2_i32_130) : i32
      %c8_i32_131 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_7_0, 0, %c8_i32_131) : i32
      %c2_i32_132 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_7_0, 1, %c2_i32_132) : i32
      %c8_i32_133 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_7_1, 0, %c8_i32_133) : i32
      %c2_i32_134 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_7_1, 1, %c2_i32_134) : i32
      %c8_i32_135 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_7_2, 0, %c8_i32_135) : i32
      %c2_i32_136 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_7_2, 1, %c2_i32_136) : i32
      %c8_i32_137 = arith.constant 8 : i32
      aiex.npu.rtp_write(@rtp_7_3, 0, %c8_i32_137) : i32
      %c2_i32_138 = arith.constant 2 : i32
      aiex.npu.rtp_write(@rtp_7_3, 1, %c2_i32_138) : i32
      aiex.set_lock(%lock_0_2, 1)
      aiex.set_lock(%lock_0_3, 1)
      aiex.set_lock(%lock_0_4, 1)
      aiex.set_lock(%lock_0_5, 1)
      aiex.set_lock(%lock_1_2, 1)
      aiex.set_lock(%lock_1_3, 1)
      aiex.set_lock(%lock_1_4, 1)
      aiex.set_lock(%lock_1_5, 1)
      aiex.set_lock(%lock_2_2, 1)
      aiex.set_lock(%lock_2_3, 1)
      aiex.set_lock(%lock_2_4, 1)
      aiex.set_lock(%lock_2_5, 1)
      aiex.set_lock(%lock_3_2, 1)
      aiex.set_lock(%lock_3_3, 1)
      aiex.set_lock(%lock_3_4, 1)
      aiex.set_lock(%lock_3_5, 1)
      aiex.set_lock(%lock_4_2, 1)
      aiex.set_lock(%lock_4_3, 1)
      aiex.set_lock(%lock_4_4, 1)
      aiex.set_lock(%lock_4_5, 1)
      aiex.set_lock(%lock_5_2, 1)
      aiex.set_lock(%lock_5_3, 1)
      aiex.set_lock(%lock_5_4, 1)
      aiex.set_lock(%lock_5_5, 1)
      aiex.set_lock(%lock_6_2, 1)
      aiex.set_lock(%lock_6_3, 1)
      aiex.set_lock(%lock_6_4, 1)
      aiex.set_lock(%lock_6_5, 1)
      aiex.set_lock(%lock_7_2, 1)
      aiex.set_lock(%lock_7_3, 1)
      aiex.set_lock(%lock_7_4, 1)
      aiex.set_lock(%lock_7_5, 1)
      %0 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 0 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 32768 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 65536 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 98304 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 131072 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 163840 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 196608 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 229376 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 33408 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 299648 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 565888 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 832128 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 1098368 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 1364608 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 1630848 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 1897088 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 262144 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 294912 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 327680 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 360448 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 393216 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 425984 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 458752 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 491520 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 2163328 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 2429568 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 2695808 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 2962048 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 3228288 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 3494528 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 3760768 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 4027008 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 0 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 3072 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 32768 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%50)
      %51 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 3072 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 65536 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 3072 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 98304 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 3072 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 131072 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 3072 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 163840 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 3072 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 196608 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 3072 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 229376 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 3072 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%63)
      %64 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 33536 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 299776 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 566016 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 832256 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%67)
      %68 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 1098496 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 1364736 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 1630976 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 1897216 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 262144 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 3072 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 294912 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 3072 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 327680 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 3072 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 360448 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 3072 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 393216 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 3072 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 425984 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 3072 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 458752 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%84)
      %85 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 3072 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 491520 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 3072 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 2163456 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 2429696 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 2695936 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 2962176 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 3228416 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 3494656 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 3760896 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 4027136 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%95)
      aiex.dma_await_task(%16)
      aiex.dma_await_task(%17)
      aiex.dma_await_task(%18)
      aiex.dma_await_task(%19)
      aiex.dma_await_task(%20)
      aiex.dma_await_task(%21)
      aiex.dma_await_task(%22)
      aiex.dma_await_task(%23)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%9)
      aiex.dma_free_task(%10)
      aiex.dma_free_task(%11)
      aiex.dma_free_task(%12)
      aiex.dma_free_task(%13)
      aiex.dma_free_task(%14)
      aiex.dma_free_task(%15)
      aiex.dma_free_task(%16)
      aiex.dma_free_task(%17)
      aiex.dma_free_task(%18)
      aiex.dma_free_task(%19)
      aiex.dma_free_task(%20)
      aiex.dma_free_task(%21)
      aiex.dma_free_task(%22)
      aiex.dma_free_task(%23)
      aiex.dma_await_task(%40)
      aiex.dma_await_task(%41)
      aiex.dma_await_task(%42)
      aiex.dma_await_task(%43)
      aiex.dma_await_task(%44)
      aiex.dma_await_task(%45)
      aiex.dma_await_task(%46)
      aiex.dma_await_task(%47)
      aiex.dma_free_task(%24)
      aiex.dma_free_task(%25)
      aiex.dma_free_task(%26)
      aiex.dma_free_task(%27)
      aiex.dma_free_task(%28)
      aiex.dma_free_task(%29)
      aiex.dma_free_task(%30)
      aiex.dma_free_task(%31)
      aiex.dma_free_task(%32)
      aiex.dma_free_task(%33)
      aiex.dma_free_task(%34)
      aiex.dma_free_task(%35)
      aiex.dma_free_task(%36)
      aiex.dma_free_task(%37)
      aiex.dma_free_task(%38)
      aiex.dma_free_task(%39)
      aiex.dma_free_task(%40)
      aiex.dma_free_task(%41)
      aiex.dma_free_task(%42)
      aiex.dma_free_task(%43)
      aiex.dma_free_task(%44)
      aiex.dma_free_task(%45)
      aiex.dma_free_task(%46)
      aiex.dma_free_task(%47)
      %96 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 0 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 6144 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%97)
      %98 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 32768 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 6144 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 65536 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%100)
      %101 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 6144 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%101)
      %102 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 98304 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%102)
      %103 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 6144 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%103)
      %104 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 131072 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%104)
      %105 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 6144 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%105)
      %106 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 163840 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%106)
      %107 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 6144 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%107)
      %108 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 196608 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%108)
      %109 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 6144 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%109)
      %110 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 229376 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%110)
      %111 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 6144 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%111)
      %112 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 66688 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%112)
      %113 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 332928 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%113)
      %114 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 599168 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%114)
      %115 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 865408 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%115)
      %116 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 1131648 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%116)
      %117 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 1397888 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%117)
      %118 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 1664128 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%118)
      %119 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 1930368 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%119)
      %120 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 262144 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%120)
      %121 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 6144 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%121)
      %122 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 294912 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%122)
      %123 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 6144 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%123)
      %124 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 327680 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%124)
      %125 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 6144 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%125)
      %126 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 360448 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%126)
      %127 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 6144 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%127)
      %128 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 393216 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%128)
      %129 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 6144 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%129)
      %130 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 425984 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%130)
      %131 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 6144 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%131)
      %132 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 458752 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%132)
      %133 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 6144 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%133)
      %134 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 491520 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%134)
      %135 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 6144 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%135)
      %136 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 2196608 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%136)
      %137 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 2462848 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%137)
      %138 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 2729088 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%138)
      %139 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 2995328 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%139)
      %140 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 3261568 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%140)
      %141 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 3527808 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%141)
      %142 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 3794048 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%142)
      %143 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 4060288 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%143)
      aiex.dma_await_task(%64)
      aiex.dma_await_task(%65)
      aiex.dma_await_task(%66)
      aiex.dma_await_task(%67)
      aiex.dma_await_task(%68)
      aiex.dma_await_task(%69)
      aiex.dma_await_task(%70)
      aiex.dma_await_task(%71)
      aiex.dma_free_task(%48)
      aiex.dma_free_task(%49)
      aiex.dma_free_task(%50)
      aiex.dma_free_task(%51)
      aiex.dma_free_task(%52)
      aiex.dma_free_task(%53)
      aiex.dma_free_task(%54)
      aiex.dma_free_task(%55)
      aiex.dma_free_task(%56)
      aiex.dma_free_task(%57)
      aiex.dma_free_task(%58)
      aiex.dma_free_task(%59)
      aiex.dma_free_task(%60)
      aiex.dma_free_task(%61)
      aiex.dma_free_task(%62)
      aiex.dma_free_task(%63)
      aiex.dma_free_task(%64)
      aiex.dma_free_task(%65)
      aiex.dma_free_task(%66)
      aiex.dma_free_task(%67)
      aiex.dma_free_task(%68)
      aiex.dma_free_task(%69)
      aiex.dma_free_task(%70)
      aiex.dma_free_task(%71)
      aiex.dma_await_task(%88)
      aiex.dma_await_task(%89)
      aiex.dma_await_task(%90)
      aiex.dma_await_task(%91)
      aiex.dma_await_task(%92)
      aiex.dma_await_task(%93)
      aiex.dma_await_task(%94)
      aiex.dma_await_task(%95)
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
      %144 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 0 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%144)
      %145 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 9216 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%145)
      %146 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 32768 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%146)
      %147 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 9216 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%147)
      %148 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 65536 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%148)
      %149 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 9216 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%149)
      %150 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 98304 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%150)
      %151 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 9216 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%151)
      %152 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 131072 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%152)
      %153 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 9216 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%153)
      %154 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 163840 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%154)
      %155 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 9216 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%155)
      %156 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 196608 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%156)
      %157 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 9216 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%157)
      %158 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 229376 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%158)
      %159 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 9216 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%159)
      %160 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 66816 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%160)
      %161 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 333056 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%161)
      %162 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 599296 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%162)
      %163 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 865536 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%163)
      %164 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 1131776 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%164)
      %165 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 1398016 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%165)
      %166 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 1664256 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%166)
      %167 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 1930496 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%167)
      %168 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 262144 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%168)
      %169 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 9216 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%169)
      %170 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 294912 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%170)
      %171 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 9216 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%171)
      %172 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 327680 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%172)
      %173 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 9216 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%173)
      %174 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 360448 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%174)
      %175 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 9216 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%175)
      %176 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 393216 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%176)
      %177 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 9216 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%177)
      %178 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 425984 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%178)
      %179 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 9216 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%179)
      %180 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 458752 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%180)
      %181 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 9216 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%181)
      %182 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<532480xbf16> offset = 491520 len = 32768 sizes = [2, 4, 128, 64] strides = [64, 8192, 128, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%182)
      %183 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<12288x!aiex.bfp<"v8bfp16ebs8">> offset = 9216 len = 3072 sizes = [1, 1, 1, 3072] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%183)
      %184 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 2196736 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%184)
      %185 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 2462976 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%185)
      %186 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 2729216 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%186)
      %187 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 2995456 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%187)
      %188 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 3261696 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%188)
      %189 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 3527936 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%189)
      %190 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 3794176 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%190)
      %191 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4326400xbf16> offset = 4060416 len = 16384 sizes = [4, 2, 64, 128] strides = [66560, 16512, 256, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%191)
      aiex.dma_await_task(%112)
      aiex.dma_await_task(%113)
      aiex.dma_await_task(%114)
      aiex.dma_await_task(%115)
      aiex.dma_await_task(%116)
      aiex.dma_await_task(%117)
      aiex.dma_await_task(%118)
      aiex.dma_await_task(%119)
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
      aiex.dma_await_task(%136)
      aiex.dma_await_task(%137)
      aiex.dma_await_task(%138)
      aiex.dma_await_task(%139)
      aiex.dma_await_task(%140)
      aiex.dma_await_task(%141)
      aiex.dma_await_task(%142)
      aiex.dma_await_task(%143)
      aiex.dma_free_task(%120)
      aiex.dma_free_task(%121)
      aiex.dma_free_task(%122)
      aiex.dma_free_task(%123)
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
      aiex.dma_await_task(%160)
      aiex.dma_await_task(%161)
      aiex.dma_await_task(%162)
      aiex.dma_await_task(%163)
      aiex.dma_await_task(%164)
      aiex.dma_await_task(%165)
      aiex.dma_await_task(%166)
      aiex.dma_await_task(%167)
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
      aiex.dma_free_task(%164)
      aiex.dma_free_task(%165)
      aiex.dma_free_task(%166)
      aiex.dma_free_task(%167)
      aiex.dma_await_task(%184)
      aiex.dma_await_task(%185)
      aiex.dma_await_task(%186)
      aiex.dma_await_task(%187)
      aiex.dma_await_task(%188)
      aiex.dma_await_task(%189)
      aiex.dma_await_task(%190)
      aiex.dma_await_task(%191)
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
      aiex.dma_free_task(%184)
      aiex.dma_free_task(%185)
      aiex.dma_free_task(%186)
      aiex.dma_free_task(%187)
      aiex.dma_free_task(%188)
      aiex.dma_free_task(%189)
      aiex.dma_free_task(%190)
      aiex.dma_free_task(%191)
    }
  }
}

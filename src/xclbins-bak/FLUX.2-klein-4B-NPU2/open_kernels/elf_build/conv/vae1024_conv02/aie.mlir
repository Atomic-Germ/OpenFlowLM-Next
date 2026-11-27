module {
  aie.device(npu2) @main {
    aie.runtime_sequence @cfg_conv() {
      aiex.configure @conv {
      }
    }
  }
  aie.device(npu2) @conv {
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
    %band_free_0_0 = aie.lock(%logical_mem_61) {init = 12 : i32, sym_name = "band_free_0_0"}
    %band_free_0_1 = aie.lock(%logical_mem_61) {init = 12 : i32, sym_name = "band_free_0_1"}
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
    %band_free_1_0 = aie.lock(%logical_mem_63) {init = 12 : i32, sym_name = "band_free_1_0"}
    %band_free_1_1 = aie.lock(%logical_mem_63) {init = 12 : i32, sym_name = "band_free_1_1"}
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
    %band_free_2_0 = aie.lock(%logical_mem_65) {init = 12 : i32, sym_name = "band_free_2_0"}
    %band_free_2_1 = aie.lock(%logical_mem_65) {init = 12 : i32, sym_name = "band_free_2_1"}
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
    %band_free_3_0 = aie.lock(%logical_mem_67) {init = 12 : i32, sym_name = "band_free_3_0"}
    %band_free_3_1 = aie.lock(%logical_mem_67) {init = 12 : i32, sym_name = "band_free_3_1"}
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
    %band_free_4_0 = aie.lock(%logical_mem_69) {init = 12 : i32, sym_name = "band_free_4_0"}
    %band_free_4_1 = aie.lock(%logical_mem_69) {init = 12 : i32, sym_name = "band_free_4_1"}
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
    %band_free_5_0 = aie.lock(%logical_mem_71) {init = 12 : i32, sym_name = "band_free_5_0"}
    %band_free_5_1 = aie.lock(%logical_mem_71) {init = 12 : i32, sym_name = "band_free_5_1"}
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
    %band_free_6_0 = aie.lock(%logical_mem_73) {init = 12 : i32, sym_name = "band_free_6_0"}
    %band_free_6_1 = aie.lock(%logical_mem_73) {init = 12 : i32, sym_name = "band_free_6_1"}
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
    %band_free_7_0 = aie.lock(%logical_mem_75) {init = 12 : i32, sym_name = "band_free_7_0"}
    %band_free_7_1 = aie.lock(%logical_mem_75) {init = 12 : i32, sym_name = "band_free_7_1"}
    %a_prod_7_0 = aie.lock(%logical_core_27) {init = 2 : i32, sym_name = "a_prod_7_0"}
    %a_cons_7_0 = aie.lock(%logical_core_27) {init = 0 : i32, sym_name = "a_cons_7_0"}
    %a_prod_7_1 = aie.lock(%logical_core_28) {init = 2 : i32, sym_name = "a_prod_7_1"}
    %a_cons_7_1 = aie.lock(%logical_core_28) {init = 0 : i32, sym_name = "a_cons_7_1"}
    %a_prod_7_2 = aie.lock(%logical_core_29) {init = 2 : i32, sym_name = "a_prod_7_2"}
    %a_cons_7_2 = aie.lock(%logical_core_29) {init = 0 : i32, sym_name = "a_cons_7_2"}
    %a_prod_7_3 = aie.lock(%logical_core_30) {init = 2 : i32, sym_name = "a_prod_7_3"}
    %a_cons_7_3 = aie.lock(%logical_core_30) {init = 0 : i32, sym_name = "a_cons_7_3"}
    %band_0_0 = aie.buffer(%logical_mem_61) {sym_name = "band_0_0"} : memref<49920xbf16> 
    %band_0_1 = aie.buffer(%logical_mem_61) {sym_name = "band_0_1"} : memref<49920xbf16> 
    %a_0_0_0 = aie.buffer(%logical_core) {sym_name = "a_0_0_0"} : memref<32x64xbf16> 
    %a_0_0_1 = aie.buffer(%logical_core) {sym_name = "a_0_0_1"} : memref<32x64xbf16> 
    %a_0_1_0 = aie.buffer(%logical_core_0) {sym_name = "a_0_1_0"} : memref<32x64xbf16> 
    %a_0_1_1 = aie.buffer(%logical_core_0) {sym_name = "a_0_1_1"} : memref<32x64xbf16> 
    %a_0_2_0 = aie.buffer(%logical_core_1) {sym_name = "a_0_2_0"} : memref<32x64xbf16> 
    %a_0_2_1 = aie.buffer(%logical_core_1) {sym_name = "a_0_2_1"} : memref<32x64xbf16> 
    %a_0_3_0 = aie.buffer(%logical_core_2) {sym_name = "a_0_3_0"} : memref<32x64xbf16> 
    %a_0_3_1 = aie.buffer(%logical_core_2) {sym_name = "a_0_3_1"} : memref<32x64xbf16> 
    %band_1_0 = aie.buffer(%logical_mem_63) {sym_name = "band_1_0"} : memref<49920xbf16> 
    %band_1_1 = aie.buffer(%logical_mem_63) {sym_name = "band_1_1"} : memref<49920xbf16> 
    %a_1_0_0 = aie.buffer(%logical_core_3) {sym_name = "a_1_0_0"} : memref<32x64xbf16> 
    %a_1_0_1 = aie.buffer(%logical_core_3) {sym_name = "a_1_0_1"} : memref<32x64xbf16> 
    %a_1_1_0 = aie.buffer(%logical_core_4) {sym_name = "a_1_1_0"} : memref<32x64xbf16> 
    %a_1_1_1 = aie.buffer(%logical_core_4) {sym_name = "a_1_1_1"} : memref<32x64xbf16> 
    %a_1_2_0 = aie.buffer(%logical_core_5) {sym_name = "a_1_2_0"} : memref<32x64xbf16> 
    %a_1_2_1 = aie.buffer(%logical_core_5) {sym_name = "a_1_2_1"} : memref<32x64xbf16> 
    %a_1_3_0 = aie.buffer(%logical_core_6) {sym_name = "a_1_3_0"} : memref<32x64xbf16> 
    %a_1_3_1 = aie.buffer(%logical_core_6) {sym_name = "a_1_3_1"} : memref<32x64xbf16> 
    %band_2_0 = aie.buffer(%logical_mem_65) {sym_name = "band_2_0"} : memref<49920xbf16> 
    %band_2_1 = aie.buffer(%logical_mem_65) {sym_name = "band_2_1"} : memref<49920xbf16> 
    %a_2_0_0 = aie.buffer(%logical_core_7) {sym_name = "a_2_0_0"} : memref<32x64xbf16> 
    %a_2_0_1 = aie.buffer(%logical_core_7) {sym_name = "a_2_0_1"} : memref<32x64xbf16> 
    %a_2_1_0 = aie.buffer(%logical_core_8) {sym_name = "a_2_1_0"} : memref<32x64xbf16> 
    %a_2_1_1 = aie.buffer(%logical_core_8) {sym_name = "a_2_1_1"} : memref<32x64xbf16> 
    %a_2_2_0 = aie.buffer(%logical_core_9) {sym_name = "a_2_2_0"} : memref<32x64xbf16> 
    %a_2_2_1 = aie.buffer(%logical_core_9) {sym_name = "a_2_2_1"} : memref<32x64xbf16> 
    %a_2_3_0 = aie.buffer(%logical_core_10) {sym_name = "a_2_3_0"} : memref<32x64xbf16> 
    %a_2_3_1 = aie.buffer(%logical_core_10) {sym_name = "a_2_3_1"} : memref<32x64xbf16> 
    %band_3_0 = aie.buffer(%logical_mem_67) {sym_name = "band_3_0"} : memref<49920xbf16> 
    %band_3_1 = aie.buffer(%logical_mem_67) {sym_name = "band_3_1"} : memref<49920xbf16> 
    %a_3_0_0 = aie.buffer(%logical_core_11) {sym_name = "a_3_0_0"} : memref<32x64xbf16> 
    %a_3_0_1 = aie.buffer(%logical_core_11) {sym_name = "a_3_0_1"} : memref<32x64xbf16> 
    %a_3_1_0 = aie.buffer(%logical_core_12) {sym_name = "a_3_1_0"} : memref<32x64xbf16> 
    %a_3_1_1 = aie.buffer(%logical_core_12) {sym_name = "a_3_1_1"} : memref<32x64xbf16> 
    %a_3_2_0 = aie.buffer(%logical_core_13) {sym_name = "a_3_2_0"} : memref<32x64xbf16> 
    %a_3_2_1 = aie.buffer(%logical_core_13) {sym_name = "a_3_2_1"} : memref<32x64xbf16> 
    %a_3_3_0 = aie.buffer(%logical_core_14) {sym_name = "a_3_3_0"} : memref<32x64xbf16> 
    %a_3_3_1 = aie.buffer(%logical_core_14) {sym_name = "a_3_3_1"} : memref<32x64xbf16> 
    %band_4_0 = aie.buffer(%logical_mem_69) {sym_name = "band_4_0"} : memref<49920xbf16> 
    %band_4_1 = aie.buffer(%logical_mem_69) {sym_name = "band_4_1"} : memref<49920xbf16> 
    %a_4_0_0 = aie.buffer(%logical_core_15) {sym_name = "a_4_0_0"} : memref<32x64xbf16> 
    %a_4_0_1 = aie.buffer(%logical_core_15) {sym_name = "a_4_0_1"} : memref<32x64xbf16> 
    %a_4_1_0 = aie.buffer(%logical_core_16) {sym_name = "a_4_1_0"} : memref<32x64xbf16> 
    %a_4_1_1 = aie.buffer(%logical_core_16) {sym_name = "a_4_1_1"} : memref<32x64xbf16> 
    %a_4_2_0 = aie.buffer(%logical_core_17) {sym_name = "a_4_2_0"} : memref<32x64xbf16> 
    %a_4_2_1 = aie.buffer(%logical_core_17) {sym_name = "a_4_2_1"} : memref<32x64xbf16> 
    %a_4_3_0 = aie.buffer(%logical_core_18) {sym_name = "a_4_3_0"} : memref<32x64xbf16> 
    %a_4_3_1 = aie.buffer(%logical_core_18) {sym_name = "a_4_3_1"} : memref<32x64xbf16> 
    %band_5_0 = aie.buffer(%logical_mem_71) {sym_name = "band_5_0"} : memref<49920xbf16> 
    %band_5_1 = aie.buffer(%logical_mem_71) {sym_name = "band_5_1"} : memref<49920xbf16> 
    %a_5_0_0 = aie.buffer(%logical_core_19) {sym_name = "a_5_0_0"} : memref<32x64xbf16> 
    %a_5_0_1 = aie.buffer(%logical_core_19) {sym_name = "a_5_0_1"} : memref<32x64xbf16> 
    %a_5_1_0 = aie.buffer(%logical_core_20) {sym_name = "a_5_1_0"} : memref<32x64xbf16> 
    %a_5_1_1 = aie.buffer(%logical_core_20) {sym_name = "a_5_1_1"} : memref<32x64xbf16> 
    %a_5_2_0 = aie.buffer(%logical_core_21) {sym_name = "a_5_2_0"} : memref<32x64xbf16> 
    %a_5_2_1 = aie.buffer(%logical_core_21) {sym_name = "a_5_2_1"} : memref<32x64xbf16> 
    %a_5_3_0 = aie.buffer(%logical_core_22) {sym_name = "a_5_3_0"} : memref<32x64xbf16> 
    %a_5_3_1 = aie.buffer(%logical_core_22) {sym_name = "a_5_3_1"} : memref<32x64xbf16> 
    %band_6_0 = aie.buffer(%logical_mem_73) {sym_name = "band_6_0"} : memref<49920xbf16> 
    %band_6_1 = aie.buffer(%logical_mem_73) {sym_name = "band_6_1"} : memref<49920xbf16> 
    %a_6_0_0 = aie.buffer(%logical_core_23) {sym_name = "a_6_0_0"} : memref<32x64xbf16> 
    %a_6_0_1 = aie.buffer(%logical_core_23) {sym_name = "a_6_0_1"} : memref<32x64xbf16> 
    %a_6_1_0 = aie.buffer(%logical_core_24) {sym_name = "a_6_1_0"} : memref<32x64xbf16> 
    %a_6_1_1 = aie.buffer(%logical_core_24) {sym_name = "a_6_1_1"} : memref<32x64xbf16> 
    %a_6_2_0 = aie.buffer(%logical_core_25) {sym_name = "a_6_2_0"} : memref<32x64xbf16> 
    %a_6_2_1 = aie.buffer(%logical_core_25) {sym_name = "a_6_2_1"} : memref<32x64xbf16> 
    %a_6_3_0 = aie.buffer(%logical_core_26) {sym_name = "a_6_3_0"} : memref<32x64xbf16> 
    %a_6_3_1 = aie.buffer(%logical_core_26) {sym_name = "a_6_3_1"} : memref<32x64xbf16> 
    %band_7_0 = aie.buffer(%logical_mem_75) {sym_name = "band_7_0"} : memref<49920xbf16> 
    %band_7_1 = aie.buffer(%logical_mem_75) {sym_name = "band_7_1"} : memref<49920xbf16> 
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
      %c12_i32 = arith.constant 12 : i32
      aie.use_lock(%band_free_0_0, AcquireGreaterEqual, %c12_i32)
      aie.dma_bd(%band_0_0 : memref<49920xbf16> len = 49920 sizes = [6, 130, 8, 8] strides = [1040, 8, 6240, 1])
      %c12_i32_77 = arith.constant 12 : i32
      aie.use_lock(%band_full_0_0, Release, %c12_i32_77)
      aie.next_bd ^bb11
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb16
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%band_0_0 : memref<49920xbf16> len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_0, Release, %c1_i32_78)
      aie.next_bd ^bb12
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb21
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_0, AcquireGreaterEqual, %c1_i32_79)
      aie.dma_bd(%band_0_0 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_80 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_0, Release, %c1_i32_80)
      aie.next_bd ^bb17
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 4, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb26
      %c1_i32_81 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_0, AcquireGreaterEqual, %c1_i32_81)
      aie.dma_bd(%band_0_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_82 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_0, Release, %c1_i32_82)
      aie.next_bd ^bb22
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb31
      %c1_i32_83 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_0, AcquireGreaterEqual, %c1_i32_83)
      aie.dma_bd(%band_0_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_84 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_0, Release, %c1_i32_84)
      aie.next_bd ^bb27
    ^bb10:  // pred: ^bb8
      aie.end
    ^bb11:  // pred: ^bb1
      %c12_i32_85 = arith.constant 12 : i32
      aie.use_lock(%band_free_0_1, AcquireGreaterEqual, %c12_i32_85)
      aie.dma_bd(%band_0_1 : memref<49920xbf16> len = 49920 sizes = [6, 130, 8, 8] strides = [1040, 8, 6240, 1])
      %c12_i32_86 = arith.constant 12 : i32
      aie.use_lock(%band_full_0_1, Release, %c12_i32_86)
      aie.next_bd ^bb1
    ^bb12:  // pred: ^bb3
      %c1_i32_87 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_0, AcquireGreaterEqual, %c1_i32_87)
      aie.dma_bd(%band_0_0 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_88 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_0, Release, %c1_i32_88)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      %c1_i32_89 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_0, AcquireGreaterEqual, %c1_i32_89)
      aie.dma_bd(%band_0_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_90 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_0, Release, %c1_i32_90)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      %c1_i32_91 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_1, AcquireGreaterEqual, %c1_i32_91)
      aie.dma_bd(%band_0_1 : memref<49920xbf16> len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_92 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_1, Release, %c1_i32_92)
      aie.next_bd ^bb15
    ^bb15:  // pred: ^bb14
      %c1_i32_93 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_1, AcquireGreaterEqual, %c1_i32_93)
      aie.dma_bd(%band_0_1 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_94 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_1, Release, %c1_i32_94)
      aie.next_bd ^bb16
    ^bb16:  // pred: ^bb15
      %c1_i32_95 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_1, AcquireGreaterEqual, %c1_i32_95)
      aie.dma_bd(%band_0_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_96 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_1, Release, %c1_i32_96)
      aie.next_bd ^bb3
    ^bb17:  // pred: ^bb5
      %c1_i32_97 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_0, AcquireGreaterEqual, %c1_i32_97)
      aie.dma_bd(%band_0_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_98 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_0, Release, %c1_i32_98)
      aie.next_bd ^bb18
    ^bb18:  // pred: ^bb17
      %c1_i32_99 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_0, AcquireGreaterEqual, %c1_i32_99)
      aie.dma_bd(%band_0_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_100 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_0, Release, %c1_i32_100)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      %c1_i32_101 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_1, AcquireGreaterEqual, %c1_i32_101)
      aie.dma_bd(%band_0_1 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_102 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_1, Release, %c1_i32_102)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      %c1_i32_103 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_1, AcquireGreaterEqual, %c1_i32_103)
      aie.dma_bd(%band_0_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_104 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_1, Release, %c1_i32_104)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      %c1_i32_105 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_1, AcquireGreaterEqual, %c1_i32_105)
      aie.dma_bd(%band_0_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_106 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_1, Release, %c1_i32_106)
      aie.next_bd ^bb5
    ^bb22:  // pred: ^bb7
      %c1_i32_107 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_0, AcquireGreaterEqual, %c1_i32_107)
      aie.dma_bd(%band_0_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_108 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_0, Release, %c1_i32_108)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      %c1_i32_109 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_0, AcquireGreaterEqual, %c1_i32_109)
      aie.dma_bd(%band_0_0 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_110 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_0, Release, %c1_i32_110)
      aie.next_bd ^bb24
    ^bb24:  // pred: ^bb23
      %c1_i32_111 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_1, AcquireGreaterEqual, %c1_i32_111)
      aie.dma_bd(%band_0_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_112 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_1, Release, %c1_i32_112)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      %c1_i32_113 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_1, AcquireGreaterEqual, %c1_i32_113)
      aie.dma_bd(%band_0_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_114 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_1, Release, %c1_i32_114)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      %c1_i32_115 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_1, AcquireGreaterEqual, %c1_i32_115)
      aie.dma_bd(%band_0_1 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_116 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_1, Release, %c1_i32_116)
      aie.next_bd ^bb7
    ^bb27:  // pred: ^bb9
      %c1_i32_117 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_0, AcquireGreaterEqual, %c1_i32_117)
      aie.dma_bd(%band_0_0 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_118 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_0, Release, %c1_i32_118)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      %c1_i32_119 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_0, AcquireGreaterEqual, %c1_i32_119)
      aie.dma_bd(%band_0_0 : memref<49920xbf16> offset = 5200 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_120 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_0, Release, %c1_i32_120)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      %c1_i32_121 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_1, AcquireGreaterEqual, %c1_i32_121)
      aie.dma_bd(%band_0_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_122 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_1, Release, %c1_i32_122)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      %c1_i32_123 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_1, AcquireGreaterEqual, %c1_i32_123)
      aie.dma_bd(%band_0_1 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_124 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_1, Release, %c1_i32_124)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      %c1_i32_125 = arith.constant 1 : i32
      aie.use_lock(%band_full_0_1, AcquireGreaterEqual, %c1_i32_125)
      aie.dma_bd(%band_0_1 : memref<49920xbf16> offset = 5200 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_126 = arith.constant 1 : i32
      aie.use_lock(%band_free_0_1, Release, %c1_i32_126)
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
      %c12_i32 = arith.constant 12 : i32
      aie.use_lock(%band_free_1_0, AcquireGreaterEqual, %c12_i32)
      aie.dma_bd(%band_1_0 : memref<49920xbf16> len = 49920 sizes = [6, 130, 8, 8] strides = [1040, 8, 6240, 1])
      %c12_i32_77 = arith.constant 12 : i32
      aie.use_lock(%band_full_1_0, Release, %c12_i32_77)
      aie.next_bd ^bb11
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb16
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%band_1_0 : memref<49920xbf16> len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_0, Release, %c1_i32_78)
      aie.next_bd ^bb12
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb21
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_0, AcquireGreaterEqual, %c1_i32_79)
      aie.dma_bd(%band_1_0 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_80 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_0, Release, %c1_i32_80)
      aie.next_bd ^bb17
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 4, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb26
      %c1_i32_81 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_0, AcquireGreaterEqual, %c1_i32_81)
      aie.dma_bd(%band_1_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_82 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_0, Release, %c1_i32_82)
      aie.next_bd ^bb22
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb31
      %c1_i32_83 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_0, AcquireGreaterEqual, %c1_i32_83)
      aie.dma_bd(%band_1_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_84 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_0, Release, %c1_i32_84)
      aie.next_bd ^bb27
    ^bb10:  // pred: ^bb8
      aie.end
    ^bb11:  // pred: ^bb1
      %c12_i32_85 = arith.constant 12 : i32
      aie.use_lock(%band_free_1_1, AcquireGreaterEqual, %c12_i32_85)
      aie.dma_bd(%band_1_1 : memref<49920xbf16> len = 49920 sizes = [6, 130, 8, 8] strides = [1040, 8, 6240, 1])
      %c12_i32_86 = arith.constant 12 : i32
      aie.use_lock(%band_full_1_1, Release, %c12_i32_86)
      aie.next_bd ^bb1
    ^bb12:  // pred: ^bb3
      %c1_i32_87 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_0, AcquireGreaterEqual, %c1_i32_87)
      aie.dma_bd(%band_1_0 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_88 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_0, Release, %c1_i32_88)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      %c1_i32_89 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_0, AcquireGreaterEqual, %c1_i32_89)
      aie.dma_bd(%band_1_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_90 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_0, Release, %c1_i32_90)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      %c1_i32_91 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_1, AcquireGreaterEqual, %c1_i32_91)
      aie.dma_bd(%band_1_1 : memref<49920xbf16> len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_92 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_1, Release, %c1_i32_92)
      aie.next_bd ^bb15
    ^bb15:  // pred: ^bb14
      %c1_i32_93 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_1, AcquireGreaterEqual, %c1_i32_93)
      aie.dma_bd(%band_1_1 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_94 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_1, Release, %c1_i32_94)
      aie.next_bd ^bb16
    ^bb16:  // pred: ^bb15
      %c1_i32_95 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_1, AcquireGreaterEqual, %c1_i32_95)
      aie.dma_bd(%band_1_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_96 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_1, Release, %c1_i32_96)
      aie.next_bd ^bb3
    ^bb17:  // pred: ^bb5
      %c1_i32_97 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_0, AcquireGreaterEqual, %c1_i32_97)
      aie.dma_bd(%band_1_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_98 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_0, Release, %c1_i32_98)
      aie.next_bd ^bb18
    ^bb18:  // pred: ^bb17
      %c1_i32_99 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_0, AcquireGreaterEqual, %c1_i32_99)
      aie.dma_bd(%band_1_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_100 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_0, Release, %c1_i32_100)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      %c1_i32_101 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_1, AcquireGreaterEqual, %c1_i32_101)
      aie.dma_bd(%band_1_1 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_102 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_1, Release, %c1_i32_102)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      %c1_i32_103 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_1, AcquireGreaterEqual, %c1_i32_103)
      aie.dma_bd(%band_1_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_104 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_1, Release, %c1_i32_104)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      %c1_i32_105 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_1, AcquireGreaterEqual, %c1_i32_105)
      aie.dma_bd(%band_1_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_106 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_1, Release, %c1_i32_106)
      aie.next_bd ^bb5
    ^bb22:  // pred: ^bb7
      %c1_i32_107 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_0, AcquireGreaterEqual, %c1_i32_107)
      aie.dma_bd(%band_1_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_108 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_0, Release, %c1_i32_108)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      %c1_i32_109 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_0, AcquireGreaterEqual, %c1_i32_109)
      aie.dma_bd(%band_1_0 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_110 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_0, Release, %c1_i32_110)
      aie.next_bd ^bb24
    ^bb24:  // pred: ^bb23
      %c1_i32_111 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_1, AcquireGreaterEqual, %c1_i32_111)
      aie.dma_bd(%band_1_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_112 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_1, Release, %c1_i32_112)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      %c1_i32_113 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_1, AcquireGreaterEqual, %c1_i32_113)
      aie.dma_bd(%band_1_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_114 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_1, Release, %c1_i32_114)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      %c1_i32_115 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_1, AcquireGreaterEqual, %c1_i32_115)
      aie.dma_bd(%band_1_1 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_116 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_1, Release, %c1_i32_116)
      aie.next_bd ^bb7
    ^bb27:  // pred: ^bb9
      %c1_i32_117 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_0, AcquireGreaterEqual, %c1_i32_117)
      aie.dma_bd(%band_1_0 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_118 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_0, Release, %c1_i32_118)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      %c1_i32_119 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_0, AcquireGreaterEqual, %c1_i32_119)
      aie.dma_bd(%band_1_0 : memref<49920xbf16> offset = 5200 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_120 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_0, Release, %c1_i32_120)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      %c1_i32_121 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_1, AcquireGreaterEqual, %c1_i32_121)
      aie.dma_bd(%band_1_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_122 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_1, Release, %c1_i32_122)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      %c1_i32_123 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_1, AcquireGreaterEqual, %c1_i32_123)
      aie.dma_bd(%band_1_1 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_124 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_1, Release, %c1_i32_124)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      %c1_i32_125 = arith.constant 1 : i32
      aie.use_lock(%band_full_1_1, AcquireGreaterEqual, %c1_i32_125)
      aie.dma_bd(%band_1_1 : memref<49920xbf16> offset = 5200 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_126 = arith.constant 1 : i32
      aie.use_lock(%band_free_1_1, Release, %c1_i32_126)
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
      %c12_i32 = arith.constant 12 : i32
      aie.use_lock(%band_free_2_0, AcquireGreaterEqual, %c12_i32)
      aie.dma_bd(%band_2_0 : memref<49920xbf16> len = 49920 sizes = [6, 130, 8, 8] strides = [1040, 8, 6240, 1])
      %c12_i32_77 = arith.constant 12 : i32
      aie.use_lock(%band_full_2_0, Release, %c12_i32_77)
      aie.next_bd ^bb11
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb16
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%band_2_0 : memref<49920xbf16> len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_0, Release, %c1_i32_78)
      aie.next_bd ^bb12
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb21
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_0, AcquireGreaterEqual, %c1_i32_79)
      aie.dma_bd(%band_2_0 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_80 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_0, Release, %c1_i32_80)
      aie.next_bd ^bb17
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 4, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb26
      %c1_i32_81 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_0, AcquireGreaterEqual, %c1_i32_81)
      aie.dma_bd(%band_2_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_82 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_0, Release, %c1_i32_82)
      aie.next_bd ^bb22
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb31
      %c1_i32_83 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_0, AcquireGreaterEqual, %c1_i32_83)
      aie.dma_bd(%band_2_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_84 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_0, Release, %c1_i32_84)
      aie.next_bd ^bb27
    ^bb10:  // pred: ^bb8
      aie.end
    ^bb11:  // pred: ^bb1
      %c12_i32_85 = arith.constant 12 : i32
      aie.use_lock(%band_free_2_1, AcquireGreaterEqual, %c12_i32_85)
      aie.dma_bd(%band_2_1 : memref<49920xbf16> len = 49920 sizes = [6, 130, 8, 8] strides = [1040, 8, 6240, 1])
      %c12_i32_86 = arith.constant 12 : i32
      aie.use_lock(%band_full_2_1, Release, %c12_i32_86)
      aie.next_bd ^bb1
    ^bb12:  // pred: ^bb3
      %c1_i32_87 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_0, AcquireGreaterEqual, %c1_i32_87)
      aie.dma_bd(%band_2_0 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_88 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_0, Release, %c1_i32_88)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      %c1_i32_89 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_0, AcquireGreaterEqual, %c1_i32_89)
      aie.dma_bd(%band_2_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_90 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_0, Release, %c1_i32_90)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      %c1_i32_91 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_1, AcquireGreaterEqual, %c1_i32_91)
      aie.dma_bd(%band_2_1 : memref<49920xbf16> len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_92 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_1, Release, %c1_i32_92)
      aie.next_bd ^bb15
    ^bb15:  // pred: ^bb14
      %c1_i32_93 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_1, AcquireGreaterEqual, %c1_i32_93)
      aie.dma_bd(%band_2_1 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_94 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_1, Release, %c1_i32_94)
      aie.next_bd ^bb16
    ^bb16:  // pred: ^bb15
      %c1_i32_95 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_1, AcquireGreaterEqual, %c1_i32_95)
      aie.dma_bd(%band_2_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_96 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_1, Release, %c1_i32_96)
      aie.next_bd ^bb3
    ^bb17:  // pred: ^bb5
      %c1_i32_97 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_0, AcquireGreaterEqual, %c1_i32_97)
      aie.dma_bd(%band_2_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_98 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_0, Release, %c1_i32_98)
      aie.next_bd ^bb18
    ^bb18:  // pred: ^bb17
      %c1_i32_99 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_0, AcquireGreaterEqual, %c1_i32_99)
      aie.dma_bd(%band_2_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_100 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_0, Release, %c1_i32_100)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      %c1_i32_101 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_1, AcquireGreaterEqual, %c1_i32_101)
      aie.dma_bd(%band_2_1 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_102 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_1, Release, %c1_i32_102)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      %c1_i32_103 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_1, AcquireGreaterEqual, %c1_i32_103)
      aie.dma_bd(%band_2_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_104 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_1, Release, %c1_i32_104)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      %c1_i32_105 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_1, AcquireGreaterEqual, %c1_i32_105)
      aie.dma_bd(%band_2_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_106 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_1, Release, %c1_i32_106)
      aie.next_bd ^bb5
    ^bb22:  // pred: ^bb7
      %c1_i32_107 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_0, AcquireGreaterEqual, %c1_i32_107)
      aie.dma_bd(%band_2_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_108 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_0, Release, %c1_i32_108)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      %c1_i32_109 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_0, AcquireGreaterEqual, %c1_i32_109)
      aie.dma_bd(%band_2_0 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_110 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_0, Release, %c1_i32_110)
      aie.next_bd ^bb24
    ^bb24:  // pred: ^bb23
      %c1_i32_111 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_1, AcquireGreaterEqual, %c1_i32_111)
      aie.dma_bd(%band_2_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_112 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_1, Release, %c1_i32_112)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      %c1_i32_113 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_1, AcquireGreaterEqual, %c1_i32_113)
      aie.dma_bd(%band_2_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_114 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_1, Release, %c1_i32_114)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      %c1_i32_115 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_1, AcquireGreaterEqual, %c1_i32_115)
      aie.dma_bd(%band_2_1 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_116 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_1, Release, %c1_i32_116)
      aie.next_bd ^bb7
    ^bb27:  // pred: ^bb9
      %c1_i32_117 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_0, AcquireGreaterEqual, %c1_i32_117)
      aie.dma_bd(%band_2_0 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_118 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_0, Release, %c1_i32_118)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      %c1_i32_119 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_0, AcquireGreaterEqual, %c1_i32_119)
      aie.dma_bd(%band_2_0 : memref<49920xbf16> offset = 5200 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_120 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_0, Release, %c1_i32_120)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      %c1_i32_121 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_1, AcquireGreaterEqual, %c1_i32_121)
      aie.dma_bd(%band_2_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_122 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_1, Release, %c1_i32_122)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      %c1_i32_123 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_1, AcquireGreaterEqual, %c1_i32_123)
      aie.dma_bd(%band_2_1 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_124 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_1, Release, %c1_i32_124)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      %c1_i32_125 = arith.constant 1 : i32
      aie.use_lock(%band_full_2_1, AcquireGreaterEqual, %c1_i32_125)
      aie.dma_bd(%band_2_1 : memref<49920xbf16> offset = 5200 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_126 = arith.constant 1 : i32
      aie.use_lock(%band_free_2_1, Release, %c1_i32_126)
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
      %c12_i32 = arith.constant 12 : i32
      aie.use_lock(%band_free_3_0, AcquireGreaterEqual, %c12_i32)
      aie.dma_bd(%band_3_0 : memref<49920xbf16> len = 49920 sizes = [6, 130, 8, 8] strides = [1040, 8, 6240, 1])
      %c12_i32_77 = arith.constant 12 : i32
      aie.use_lock(%band_full_3_0, Release, %c12_i32_77)
      aie.next_bd ^bb11
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb16
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%band_3_0 : memref<49920xbf16> len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_0, Release, %c1_i32_78)
      aie.next_bd ^bb12
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb21
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_0, AcquireGreaterEqual, %c1_i32_79)
      aie.dma_bd(%band_3_0 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_80 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_0, Release, %c1_i32_80)
      aie.next_bd ^bb17
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 4, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb26
      %c1_i32_81 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_0, AcquireGreaterEqual, %c1_i32_81)
      aie.dma_bd(%band_3_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_82 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_0, Release, %c1_i32_82)
      aie.next_bd ^bb22
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb31
      %c1_i32_83 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_0, AcquireGreaterEqual, %c1_i32_83)
      aie.dma_bd(%band_3_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_84 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_0, Release, %c1_i32_84)
      aie.next_bd ^bb27
    ^bb10:  // pred: ^bb8
      aie.end
    ^bb11:  // pred: ^bb1
      %c12_i32_85 = arith.constant 12 : i32
      aie.use_lock(%band_free_3_1, AcquireGreaterEqual, %c12_i32_85)
      aie.dma_bd(%band_3_1 : memref<49920xbf16> len = 49920 sizes = [6, 130, 8, 8] strides = [1040, 8, 6240, 1])
      %c12_i32_86 = arith.constant 12 : i32
      aie.use_lock(%band_full_3_1, Release, %c12_i32_86)
      aie.next_bd ^bb1
    ^bb12:  // pred: ^bb3
      %c1_i32_87 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_0, AcquireGreaterEqual, %c1_i32_87)
      aie.dma_bd(%band_3_0 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_88 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_0, Release, %c1_i32_88)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      %c1_i32_89 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_0, AcquireGreaterEqual, %c1_i32_89)
      aie.dma_bd(%band_3_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_90 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_0, Release, %c1_i32_90)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      %c1_i32_91 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_1, AcquireGreaterEqual, %c1_i32_91)
      aie.dma_bd(%band_3_1 : memref<49920xbf16> len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_92 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_1, Release, %c1_i32_92)
      aie.next_bd ^bb15
    ^bb15:  // pred: ^bb14
      %c1_i32_93 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_1, AcquireGreaterEqual, %c1_i32_93)
      aie.dma_bd(%band_3_1 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_94 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_1, Release, %c1_i32_94)
      aie.next_bd ^bb16
    ^bb16:  // pred: ^bb15
      %c1_i32_95 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_1, AcquireGreaterEqual, %c1_i32_95)
      aie.dma_bd(%band_3_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_96 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_1, Release, %c1_i32_96)
      aie.next_bd ^bb3
    ^bb17:  // pred: ^bb5
      %c1_i32_97 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_0, AcquireGreaterEqual, %c1_i32_97)
      aie.dma_bd(%band_3_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_98 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_0, Release, %c1_i32_98)
      aie.next_bd ^bb18
    ^bb18:  // pred: ^bb17
      %c1_i32_99 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_0, AcquireGreaterEqual, %c1_i32_99)
      aie.dma_bd(%band_3_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_100 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_0, Release, %c1_i32_100)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      %c1_i32_101 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_1, AcquireGreaterEqual, %c1_i32_101)
      aie.dma_bd(%band_3_1 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_102 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_1, Release, %c1_i32_102)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      %c1_i32_103 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_1, AcquireGreaterEqual, %c1_i32_103)
      aie.dma_bd(%band_3_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_104 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_1, Release, %c1_i32_104)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      %c1_i32_105 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_1, AcquireGreaterEqual, %c1_i32_105)
      aie.dma_bd(%band_3_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_106 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_1, Release, %c1_i32_106)
      aie.next_bd ^bb5
    ^bb22:  // pred: ^bb7
      %c1_i32_107 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_0, AcquireGreaterEqual, %c1_i32_107)
      aie.dma_bd(%band_3_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_108 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_0, Release, %c1_i32_108)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      %c1_i32_109 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_0, AcquireGreaterEqual, %c1_i32_109)
      aie.dma_bd(%band_3_0 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_110 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_0, Release, %c1_i32_110)
      aie.next_bd ^bb24
    ^bb24:  // pred: ^bb23
      %c1_i32_111 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_1, AcquireGreaterEqual, %c1_i32_111)
      aie.dma_bd(%band_3_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_112 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_1, Release, %c1_i32_112)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      %c1_i32_113 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_1, AcquireGreaterEqual, %c1_i32_113)
      aie.dma_bd(%band_3_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_114 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_1, Release, %c1_i32_114)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      %c1_i32_115 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_1, AcquireGreaterEqual, %c1_i32_115)
      aie.dma_bd(%band_3_1 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_116 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_1, Release, %c1_i32_116)
      aie.next_bd ^bb7
    ^bb27:  // pred: ^bb9
      %c1_i32_117 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_0, AcquireGreaterEqual, %c1_i32_117)
      aie.dma_bd(%band_3_0 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_118 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_0, Release, %c1_i32_118)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      %c1_i32_119 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_0, AcquireGreaterEqual, %c1_i32_119)
      aie.dma_bd(%band_3_0 : memref<49920xbf16> offset = 5200 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_120 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_0, Release, %c1_i32_120)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      %c1_i32_121 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_1, AcquireGreaterEqual, %c1_i32_121)
      aie.dma_bd(%band_3_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_122 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_1, Release, %c1_i32_122)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      %c1_i32_123 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_1, AcquireGreaterEqual, %c1_i32_123)
      aie.dma_bd(%band_3_1 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_124 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_1, Release, %c1_i32_124)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      %c1_i32_125 = arith.constant 1 : i32
      aie.use_lock(%band_full_3_1, AcquireGreaterEqual, %c1_i32_125)
      aie.dma_bd(%band_3_1 : memref<49920xbf16> offset = 5200 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_126 = arith.constant 1 : i32
      aie.use_lock(%band_free_3_1, Release, %c1_i32_126)
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
      %c12_i32 = arith.constant 12 : i32
      aie.use_lock(%band_free_4_0, AcquireGreaterEqual, %c12_i32)
      aie.dma_bd(%band_4_0 : memref<49920xbf16> len = 49920 sizes = [6, 130, 8, 8] strides = [1040, 8, 6240, 1])
      %c12_i32_77 = arith.constant 12 : i32
      aie.use_lock(%band_full_4_0, Release, %c12_i32_77)
      aie.next_bd ^bb11
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb16
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%band_4_0 : memref<49920xbf16> len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_0, Release, %c1_i32_78)
      aie.next_bd ^bb12
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb21
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_0, AcquireGreaterEqual, %c1_i32_79)
      aie.dma_bd(%band_4_0 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_80 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_0, Release, %c1_i32_80)
      aie.next_bd ^bb17
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 4, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb26
      %c1_i32_81 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_0, AcquireGreaterEqual, %c1_i32_81)
      aie.dma_bd(%band_4_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_82 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_0, Release, %c1_i32_82)
      aie.next_bd ^bb22
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb31
      %c1_i32_83 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_0, AcquireGreaterEqual, %c1_i32_83)
      aie.dma_bd(%band_4_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_84 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_0, Release, %c1_i32_84)
      aie.next_bd ^bb27
    ^bb10:  // pred: ^bb8
      aie.end
    ^bb11:  // pred: ^bb1
      %c12_i32_85 = arith.constant 12 : i32
      aie.use_lock(%band_free_4_1, AcquireGreaterEqual, %c12_i32_85)
      aie.dma_bd(%band_4_1 : memref<49920xbf16> len = 49920 sizes = [6, 130, 8, 8] strides = [1040, 8, 6240, 1])
      %c12_i32_86 = arith.constant 12 : i32
      aie.use_lock(%band_full_4_1, Release, %c12_i32_86)
      aie.next_bd ^bb1
    ^bb12:  // pred: ^bb3
      %c1_i32_87 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_0, AcquireGreaterEqual, %c1_i32_87)
      aie.dma_bd(%band_4_0 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_88 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_0, Release, %c1_i32_88)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      %c1_i32_89 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_0, AcquireGreaterEqual, %c1_i32_89)
      aie.dma_bd(%band_4_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_90 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_0, Release, %c1_i32_90)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      %c1_i32_91 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_1, AcquireGreaterEqual, %c1_i32_91)
      aie.dma_bd(%band_4_1 : memref<49920xbf16> len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_92 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_1, Release, %c1_i32_92)
      aie.next_bd ^bb15
    ^bb15:  // pred: ^bb14
      %c1_i32_93 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_1, AcquireGreaterEqual, %c1_i32_93)
      aie.dma_bd(%band_4_1 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_94 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_1, Release, %c1_i32_94)
      aie.next_bd ^bb16
    ^bb16:  // pred: ^bb15
      %c1_i32_95 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_1, AcquireGreaterEqual, %c1_i32_95)
      aie.dma_bd(%band_4_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_96 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_1, Release, %c1_i32_96)
      aie.next_bd ^bb3
    ^bb17:  // pred: ^bb5
      %c1_i32_97 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_0, AcquireGreaterEqual, %c1_i32_97)
      aie.dma_bd(%band_4_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_98 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_0, Release, %c1_i32_98)
      aie.next_bd ^bb18
    ^bb18:  // pred: ^bb17
      %c1_i32_99 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_0, AcquireGreaterEqual, %c1_i32_99)
      aie.dma_bd(%band_4_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_100 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_0, Release, %c1_i32_100)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      %c1_i32_101 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_1, AcquireGreaterEqual, %c1_i32_101)
      aie.dma_bd(%band_4_1 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_102 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_1, Release, %c1_i32_102)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      %c1_i32_103 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_1, AcquireGreaterEqual, %c1_i32_103)
      aie.dma_bd(%band_4_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_104 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_1, Release, %c1_i32_104)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      %c1_i32_105 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_1, AcquireGreaterEqual, %c1_i32_105)
      aie.dma_bd(%band_4_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_106 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_1, Release, %c1_i32_106)
      aie.next_bd ^bb5
    ^bb22:  // pred: ^bb7
      %c1_i32_107 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_0, AcquireGreaterEqual, %c1_i32_107)
      aie.dma_bd(%band_4_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_108 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_0, Release, %c1_i32_108)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      %c1_i32_109 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_0, AcquireGreaterEqual, %c1_i32_109)
      aie.dma_bd(%band_4_0 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_110 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_0, Release, %c1_i32_110)
      aie.next_bd ^bb24
    ^bb24:  // pred: ^bb23
      %c1_i32_111 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_1, AcquireGreaterEqual, %c1_i32_111)
      aie.dma_bd(%band_4_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_112 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_1, Release, %c1_i32_112)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      %c1_i32_113 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_1, AcquireGreaterEqual, %c1_i32_113)
      aie.dma_bd(%band_4_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_114 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_1, Release, %c1_i32_114)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      %c1_i32_115 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_1, AcquireGreaterEqual, %c1_i32_115)
      aie.dma_bd(%band_4_1 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_116 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_1, Release, %c1_i32_116)
      aie.next_bd ^bb7
    ^bb27:  // pred: ^bb9
      %c1_i32_117 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_0, AcquireGreaterEqual, %c1_i32_117)
      aie.dma_bd(%band_4_0 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_118 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_0, Release, %c1_i32_118)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      %c1_i32_119 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_0, AcquireGreaterEqual, %c1_i32_119)
      aie.dma_bd(%band_4_0 : memref<49920xbf16> offset = 5200 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_120 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_0, Release, %c1_i32_120)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      %c1_i32_121 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_1, AcquireGreaterEqual, %c1_i32_121)
      aie.dma_bd(%band_4_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_122 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_1, Release, %c1_i32_122)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      %c1_i32_123 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_1, AcquireGreaterEqual, %c1_i32_123)
      aie.dma_bd(%band_4_1 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_124 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_1, Release, %c1_i32_124)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      %c1_i32_125 = arith.constant 1 : i32
      aie.use_lock(%band_full_4_1, AcquireGreaterEqual, %c1_i32_125)
      aie.dma_bd(%band_4_1 : memref<49920xbf16> offset = 5200 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_126 = arith.constant 1 : i32
      aie.use_lock(%band_free_4_1, Release, %c1_i32_126)
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
      %c12_i32 = arith.constant 12 : i32
      aie.use_lock(%band_free_5_0, AcquireGreaterEqual, %c12_i32)
      aie.dma_bd(%band_5_0 : memref<49920xbf16> len = 49920 sizes = [6, 130, 8, 8] strides = [1040, 8, 6240, 1])
      %c12_i32_77 = arith.constant 12 : i32
      aie.use_lock(%band_full_5_0, Release, %c12_i32_77)
      aie.next_bd ^bb11
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb16
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%band_5_0 : memref<49920xbf16> len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_0, Release, %c1_i32_78)
      aie.next_bd ^bb12
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb21
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_0, AcquireGreaterEqual, %c1_i32_79)
      aie.dma_bd(%band_5_0 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_80 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_0, Release, %c1_i32_80)
      aie.next_bd ^bb17
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 4, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb26
      %c1_i32_81 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_0, AcquireGreaterEqual, %c1_i32_81)
      aie.dma_bd(%band_5_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_82 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_0, Release, %c1_i32_82)
      aie.next_bd ^bb22
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb31
      %c1_i32_83 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_0, AcquireGreaterEqual, %c1_i32_83)
      aie.dma_bd(%band_5_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_84 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_0, Release, %c1_i32_84)
      aie.next_bd ^bb27
    ^bb10:  // pred: ^bb8
      aie.end
    ^bb11:  // pred: ^bb1
      %c12_i32_85 = arith.constant 12 : i32
      aie.use_lock(%band_free_5_1, AcquireGreaterEqual, %c12_i32_85)
      aie.dma_bd(%band_5_1 : memref<49920xbf16> len = 49920 sizes = [6, 130, 8, 8] strides = [1040, 8, 6240, 1])
      %c12_i32_86 = arith.constant 12 : i32
      aie.use_lock(%band_full_5_1, Release, %c12_i32_86)
      aie.next_bd ^bb1
    ^bb12:  // pred: ^bb3
      %c1_i32_87 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_0, AcquireGreaterEqual, %c1_i32_87)
      aie.dma_bd(%band_5_0 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_88 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_0, Release, %c1_i32_88)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      %c1_i32_89 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_0, AcquireGreaterEqual, %c1_i32_89)
      aie.dma_bd(%band_5_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_90 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_0, Release, %c1_i32_90)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      %c1_i32_91 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_1, AcquireGreaterEqual, %c1_i32_91)
      aie.dma_bd(%band_5_1 : memref<49920xbf16> len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_92 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_1, Release, %c1_i32_92)
      aie.next_bd ^bb15
    ^bb15:  // pred: ^bb14
      %c1_i32_93 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_1, AcquireGreaterEqual, %c1_i32_93)
      aie.dma_bd(%band_5_1 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_94 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_1, Release, %c1_i32_94)
      aie.next_bd ^bb16
    ^bb16:  // pred: ^bb15
      %c1_i32_95 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_1, AcquireGreaterEqual, %c1_i32_95)
      aie.dma_bd(%band_5_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_96 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_1, Release, %c1_i32_96)
      aie.next_bd ^bb3
    ^bb17:  // pred: ^bb5
      %c1_i32_97 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_0, AcquireGreaterEqual, %c1_i32_97)
      aie.dma_bd(%band_5_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_98 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_0, Release, %c1_i32_98)
      aie.next_bd ^bb18
    ^bb18:  // pred: ^bb17
      %c1_i32_99 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_0, AcquireGreaterEqual, %c1_i32_99)
      aie.dma_bd(%band_5_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_100 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_0, Release, %c1_i32_100)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      %c1_i32_101 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_1, AcquireGreaterEqual, %c1_i32_101)
      aie.dma_bd(%band_5_1 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_102 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_1, Release, %c1_i32_102)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      %c1_i32_103 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_1, AcquireGreaterEqual, %c1_i32_103)
      aie.dma_bd(%band_5_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_104 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_1, Release, %c1_i32_104)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      %c1_i32_105 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_1, AcquireGreaterEqual, %c1_i32_105)
      aie.dma_bd(%band_5_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_106 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_1, Release, %c1_i32_106)
      aie.next_bd ^bb5
    ^bb22:  // pred: ^bb7
      %c1_i32_107 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_0, AcquireGreaterEqual, %c1_i32_107)
      aie.dma_bd(%band_5_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_108 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_0, Release, %c1_i32_108)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      %c1_i32_109 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_0, AcquireGreaterEqual, %c1_i32_109)
      aie.dma_bd(%band_5_0 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_110 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_0, Release, %c1_i32_110)
      aie.next_bd ^bb24
    ^bb24:  // pred: ^bb23
      %c1_i32_111 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_1, AcquireGreaterEqual, %c1_i32_111)
      aie.dma_bd(%band_5_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_112 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_1, Release, %c1_i32_112)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      %c1_i32_113 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_1, AcquireGreaterEqual, %c1_i32_113)
      aie.dma_bd(%band_5_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_114 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_1, Release, %c1_i32_114)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      %c1_i32_115 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_1, AcquireGreaterEqual, %c1_i32_115)
      aie.dma_bd(%band_5_1 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_116 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_1, Release, %c1_i32_116)
      aie.next_bd ^bb7
    ^bb27:  // pred: ^bb9
      %c1_i32_117 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_0, AcquireGreaterEqual, %c1_i32_117)
      aie.dma_bd(%band_5_0 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_118 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_0, Release, %c1_i32_118)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      %c1_i32_119 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_0, AcquireGreaterEqual, %c1_i32_119)
      aie.dma_bd(%band_5_0 : memref<49920xbf16> offset = 5200 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_120 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_0, Release, %c1_i32_120)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      %c1_i32_121 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_1, AcquireGreaterEqual, %c1_i32_121)
      aie.dma_bd(%band_5_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_122 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_1, Release, %c1_i32_122)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      %c1_i32_123 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_1, AcquireGreaterEqual, %c1_i32_123)
      aie.dma_bd(%band_5_1 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_124 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_1, Release, %c1_i32_124)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      %c1_i32_125 = arith.constant 1 : i32
      aie.use_lock(%band_full_5_1, AcquireGreaterEqual, %c1_i32_125)
      aie.dma_bd(%band_5_1 : memref<49920xbf16> offset = 5200 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_126 = arith.constant 1 : i32
      aie.use_lock(%band_free_5_1, Release, %c1_i32_126)
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
      %c12_i32 = arith.constant 12 : i32
      aie.use_lock(%band_free_6_0, AcquireGreaterEqual, %c12_i32)
      aie.dma_bd(%band_6_0 : memref<49920xbf16> len = 49920 sizes = [6, 130, 8, 8] strides = [1040, 8, 6240, 1])
      %c12_i32_77 = arith.constant 12 : i32
      aie.use_lock(%band_full_6_0, Release, %c12_i32_77)
      aie.next_bd ^bb11
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb16
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%band_6_0 : memref<49920xbf16> len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_0, Release, %c1_i32_78)
      aie.next_bd ^bb12
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb21
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_0, AcquireGreaterEqual, %c1_i32_79)
      aie.dma_bd(%band_6_0 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_80 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_0, Release, %c1_i32_80)
      aie.next_bd ^bb17
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 4, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb26
      %c1_i32_81 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_0, AcquireGreaterEqual, %c1_i32_81)
      aie.dma_bd(%band_6_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_82 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_0, Release, %c1_i32_82)
      aie.next_bd ^bb22
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb31
      %c1_i32_83 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_0, AcquireGreaterEqual, %c1_i32_83)
      aie.dma_bd(%band_6_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_84 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_0, Release, %c1_i32_84)
      aie.next_bd ^bb27
    ^bb10:  // pred: ^bb8
      aie.end
    ^bb11:  // pred: ^bb1
      %c12_i32_85 = arith.constant 12 : i32
      aie.use_lock(%band_free_6_1, AcquireGreaterEqual, %c12_i32_85)
      aie.dma_bd(%band_6_1 : memref<49920xbf16> len = 49920 sizes = [6, 130, 8, 8] strides = [1040, 8, 6240, 1])
      %c12_i32_86 = arith.constant 12 : i32
      aie.use_lock(%band_full_6_1, Release, %c12_i32_86)
      aie.next_bd ^bb1
    ^bb12:  // pred: ^bb3
      %c1_i32_87 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_0, AcquireGreaterEqual, %c1_i32_87)
      aie.dma_bd(%band_6_0 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_88 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_0, Release, %c1_i32_88)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      %c1_i32_89 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_0, AcquireGreaterEqual, %c1_i32_89)
      aie.dma_bd(%band_6_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_90 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_0, Release, %c1_i32_90)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      %c1_i32_91 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_1, AcquireGreaterEqual, %c1_i32_91)
      aie.dma_bd(%band_6_1 : memref<49920xbf16> len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_92 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_1, Release, %c1_i32_92)
      aie.next_bd ^bb15
    ^bb15:  // pred: ^bb14
      %c1_i32_93 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_1, AcquireGreaterEqual, %c1_i32_93)
      aie.dma_bd(%band_6_1 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_94 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_1, Release, %c1_i32_94)
      aie.next_bd ^bb16
    ^bb16:  // pred: ^bb15
      %c1_i32_95 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_1, AcquireGreaterEqual, %c1_i32_95)
      aie.dma_bd(%band_6_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_96 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_1, Release, %c1_i32_96)
      aie.next_bd ^bb3
    ^bb17:  // pred: ^bb5
      %c1_i32_97 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_0, AcquireGreaterEqual, %c1_i32_97)
      aie.dma_bd(%band_6_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_98 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_0, Release, %c1_i32_98)
      aie.next_bd ^bb18
    ^bb18:  // pred: ^bb17
      %c1_i32_99 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_0, AcquireGreaterEqual, %c1_i32_99)
      aie.dma_bd(%band_6_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_100 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_0, Release, %c1_i32_100)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      %c1_i32_101 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_1, AcquireGreaterEqual, %c1_i32_101)
      aie.dma_bd(%band_6_1 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_102 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_1, Release, %c1_i32_102)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      %c1_i32_103 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_1, AcquireGreaterEqual, %c1_i32_103)
      aie.dma_bd(%band_6_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_104 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_1, Release, %c1_i32_104)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      %c1_i32_105 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_1, AcquireGreaterEqual, %c1_i32_105)
      aie.dma_bd(%band_6_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_106 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_1, Release, %c1_i32_106)
      aie.next_bd ^bb5
    ^bb22:  // pred: ^bb7
      %c1_i32_107 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_0, AcquireGreaterEqual, %c1_i32_107)
      aie.dma_bd(%band_6_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_108 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_0, Release, %c1_i32_108)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      %c1_i32_109 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_0, AcquireGreaterEqual, %c1_i32_109)
      aie.dma_bd(%band_6_0 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_110 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_0, Release, %c1_i32_110)
      aie.next_bd ^bb24
    ^bb24:  // pred: ^bb23
      %c1_i32_111 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_1, AcquireGreaterEqual, %c1_i32_111)
      aie.dma_bd(%band_6_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_112 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_1, Release, %c1_i32_112)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      %c1_i32_113 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_1, AcquireGreaterEqual, %c1_i32_113)
      aie.dma_bd(%band_6_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_114 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_1, Release, %c1_i32_114)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      %c1_i32_115 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_1, AcquireGreaterEqual, %c1_i32_115)
      aie.dma_bd(%band_6_1 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_116 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_1, Release, %c1_i32_116)
      aie.next_bd ^bb7
    ^bb27:  // pred: ^bb9
      %c1_i32_117 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_0, AcquireGreaterEqual, %c1_i32_117)
      aie.dma_bd(%band_6_0 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_118 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_0, Release, %c1_i32_118)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      %c1_i32_119 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_0, AcquireGreaterEqual, %c1_i32_119)
      aie.dma_bd(%band_6_0 : memref<49920xbf16> offset = 5200 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_120 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_0, Release, %c1_i32_120)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      %c1_i32_121 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_1, AcquireGreaterEqual, %c1_i32_121)
      aie.dma_bd(%band_6_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_122 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_1, Release, %c1_i32_122)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      %c1_i32_123 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_1, AcquireGreaterEqual, %c1_i32_123)
      aie.dma_bd(%band_6_1 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_124 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_1, Release, %c1_i32_124)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      %c1_i32_125 = arith.constant 1 : i32
      aie.use_lock(%band_full_6_1, AcquireGreaterEqual, %c1_i32_125)
      aie.dma_bd(%band_6_1 : memref<49920xbf16> offset = 5200 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_126 = arith.constant 1 : i32
      aie.use_lock(%band_free_6_1, Release, %c1_i32_126)
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
      %c12_i32 = arith.constant 12 : i32
      aie.use_lock(%band_free_7_0, AcquireGreaterEqual, %c12_i32)
      aie.dma_bd(%band_7_0 : memref<49920xbf16> len = 49920 sizes = [6, 130, 8, 8] strides = [1040, 8, 6240, 1])
      %c12_i32_77 = arith.constant 12 : i32
      aie.use_lock(%band_full_7_0, Release, %c12_i32_77)
      aie.next_bd ^bb11
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb16
      %c1_i32 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_0, AcquireGreaterEqual, %c1_i32)
      aie.dma_bd(%band_7_0 : memref<49920xbf16> len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_78 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_0, Release, %c1_i32_78)
      aie.next_bd ^bb12
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 2, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb21
      %c1_i32_79 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_0, AcquireGreaterEqual, %c1_i32_79)
      aie.dma_bd(%band_7_0 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_80 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_0, Release, %c1_i32_80)
      aie.next_bd ^bb17
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 4, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb26
      %c1_i32_81 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_0, AcquireGreaterEqual, %c1_i32_81)
      aie.dma_bd(%band_7_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_82 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_0, Release, %c1_i32_82)
      aie.next_bd ^bb22
    ^bb8:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 1, ^bb9, ^bb10)
    ^bb9:  // 2 preds: ^bb8, ^bb31
      %c1_i32_83 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_0, AcquireGreaterEqual, %c1_i32_83)
      aie.dma_bd(%band_7_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_84 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_0, Release, %c1_i32_84)
      aie.next_bd ^bb27
    ^bb10:  // pred: ^bb8
      aie.end
    ^bb11:  // pred: ^bb1
      %c12_i32_85 = arith.constant 12 : i32
      aie.use_lock(%band_free_7_1, AcquireGreaterEqual, %c12_i32_85)
      aie.dma_bd(%band_7_1 : memref<49920xbf16> len = 49920 sizes = [6, 130, 8, 8] strides = [1040, 8, 6240, 1])
      %c12_i32_86 = arith.constant 12 : i32
      aie.use_lock(%band_full_7_1, Release, %c12_i32_86)
      aie.next_bd ^bb1
    ^bb12:  // pred: ^bb3
      %c1_i32_87 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_0, AcquireGreaterEqual, %c1_i32_87)
      aie.dma_bd(%band_7_0 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_88 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_0, Release, %c1_i32_88)
      aie.next_bd ^bb13
    ^bb13:  // pred: ^bb12
      %c1_i32_89 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_0, AcquireGreaterEqual, %c1_i32_89)
      aie.dma_bd(%band_7_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_90 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_0, Release, %c1_i32_90)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      %c1_i32_91 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_1, AcquireGreaterEqual, %c1_i32_91)
      aie.dma_bd(%band_7_1 : memref<49920xbf16> len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_92 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_1, Release, %c1_i32_92)
      aie.next_bd ^bb15
    ^bb15:  // pred: ^bb14
      %c1_i32_93 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_1, AcquireGreaterEqual, %c1_i32_93)
      aie.dma_bd(%band_7_1 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_94 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_1, Release, %c1_i32_94)
      aie.next_bd ^bb16
    ^bb16:  // pred: ^bb15
      %c1_i32_95 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_1, AcquireGreaterEqual, %c1_i32_95)
      aie.dma_bd(%band_7_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_96 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_1, Release, %c1_i32_96)
      aie.next_bd ^bb3
    ^bb17:  // pred: ^bb5
      %c1_i32_97 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_0, AcquireGreaterEqual, %c1_i32_97)
      aie.dma_bd(%band_7_0 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_98 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_0, Release, %c1_i32_98)
      aie.next_bd ^bb18
    ^bb18:  // pred: ^bb17
      %c1_i32_99 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_0, AcquireGreaterEqual, %c1_i32_99)
      aie.dma_bd(%band_7_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_100 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_0, Release, %c1_i32_100)
      aie.next_bd ^bb19
    ^bb19:  // pred: ^bb18
      %c1_i32_101 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_1, AcquireGreaterEqual, %c1_i32_101)
      aie.dma_bd(%band_7_1 : memref<49920xbf16> offset = 1040 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_102 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_1, Release, %c1_i32_102)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      %c1_i32_103 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_1, AcquireGreaterEqual, %c1_i32_103)
      aie.dma_bd(%band_7_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_104 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_1, Release, %c1_i32_104)
      aie.next_bd ^bb21
    ^bb21:  // pred: ^bb20
      %c1_i32_105 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_1, AcquireGreaterEqual, %c1_i32_105)
      aie.dma_bd(%band_7_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_106 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_1, Release, %c1_i32_106)
      aie.next_bd ^bb5
    ^bb22:  // pred: ^bb7
      %c1_i32_107 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_0, AcquireGreaterEqual, %c1_i32_107)
      aie.dma_bd(%band_7_0 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_108 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_0, Release, %c1_i32_108)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      %c1_i32_109 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_0, AcquireGreaterEqual, %c1_i32_109)
      aie.dma_bd(%band_7_0 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_110 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_0, Release, %c1_i32_110)
      aie.next_bd ^bb24
    ^bb24:  // pred: ^bb23
      %c1_i32_111 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_1, AcquireGreaterEqual, %c1_i32_111)
      aie.dma_bd(%band_7_1 : memref<49920xbf16> offset = 2080 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_112 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_1, Release, %c1_i32_112)
      aie.next_bd ^bb25
    ^bb25:  // pred: ^bb24
      %c1_i32_113 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_1, AcquireGreaterEqual, %c1_i32_113)
      aie.dma_bd(%band_7_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_114 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_1, Release, %c1_i32_114)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      %c1_i32_115 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_1, AcquireGreaterEqual, %c1_i32_115)
      aie.dma_bd(%band_7_1 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_116 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_1, Release, %c1_i32_116)
      aie.next_bd ^bb7
    ^bb27:  // pred: ^bb9
      %c1_i32_117 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_0, AcquireGreaterEqual, %c1_i32_117)
      aie.dma_bd(%band_7_0 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_118 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_0, Release, %c1_i32_118)
      aie.next_bd ^bb28
    ^bb28:  // pred: ^bb27
      %c1_i32_119 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_0, AcquireGreaterEqual, %c1_i32_119)
      aie.dma_bd(%band_7_0 : memref<49920xbf16> offset = 5200 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_120 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_0, Release, %c1_i32_120)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      %c1_i32_121 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_1, AcquireGreaterEqual, %c1_i32_121)
      aie.dma_bd(%band_7_1 : memref<49920xbf16> offset = 3120 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_122 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_1, Release, %c1_i32_122)
      aie.next_bd ^bb30
    ^bb30:  // pred: ^bb29
      %c1_i32_123 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_1, AcquireGreaterEqual, %c1_i32_123)
      aie.dma_bd(%band_7_1 : memref<49920xbf16> offset = 4160 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_124 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_1, Release, %c1_i32_124)
      aie.next_bd ^bb31
    ^bb31:  // pred: ^bb30
      %c1_i32_125 = arith.constant 1 : i32
      aie.use_lock(%band_full_7_1, AcquireGreaterEqual, %c1_i32_125)
      aie.dma_bd(%band_7_1 : memref<49920xbf16> offset = 5200 len = 24576 sizes = [3, 4, 8, 256] strides = [8, 256, 6240, 1])
      %c1_i32_126 = arith.constant 1 : i32
      aie.use_lock(%band_free_7_1, Release, %c1_i32_126)
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
    aie.runtime_sequence @vae1024_conv02(%arg0: memref<8652800xbf16>, %arg1: memref<1196032x!aiex.bfp<"v8bfp16ebs8">>, %arg2: memref<34080768xbf16>) {
      %c64_i32 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_0_0, 0, %c64_i32) : i32
      %c72_i32 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_0_0, 1, %c72_i32) : i32
      %c64_i32_77 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_0_1, 0, %c64_i32_77) : i32
      %c72_i32_78 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_0_1, 1, %c72_i32_78) : i32
      %c64_i32_79 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_0_2, 0, %c64_i32_79) : i32
      %c72_i32_80 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_0_2, 1, %c72_i32_80) : i32
      %c64_i32_81 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_0_3, 0, %c64_i32_81) : i32
      %c72_i32_82 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_0_3, 1, %c72_i32_82) : i32
      %c64_i32_83 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_1_0, 0, %c64_i32_83) : i32
      %c72_i32_84 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_1_0, 1, %c72_i32_84) : i32
      %c64_i32_85 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_1_1, 0, %c64_i32_85) : i32
      %c72_i32_86 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_1_1, 1, %c72_i32_86) : i32
      %c64_i32_87 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_1_2, 0, %c64_i32_87) : i32
      %c72_i32_88 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_1_2, 1, %c72_i32_88) : i32
      %c64_i32_89 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_1_3, 0, %c64_i32_89) : i32
      %c72_i32_90 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_1_3, 1, %c72_i32_90) : i32
      %c64_i32_91 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_2_0, 0, %c64_i32_91) : i32
      %c72_i32_92 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_2_0, 1, %c72_i32_92) : i32
      %c64_i32_93 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_2_1, 0, %c64_i32_93) : i32
      %c72_i32_94 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_2_1, 1, %c72_i32_94) : i32
      %c64_i32_95 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_2_2, 0, %c64_i32_95) : i32
      %c72_i32_96 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_2_2, 1, %c72_i32_96) : i32
      %c64_i32_97 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_2_3, 0, %c64_i32_97) : i32
      %c72_i32_98 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_2_3, 1, %c72_i32_98) : i32
      %c64_i32_99 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_3_0, 0, %c64_i32_99) : i32
      %c72_i32_100 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_3_0, 1, %c72_i32_100) : i32
      %c64_i32_101 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_3_1, 0, %c64_i32_101) : i32
      %c72_i32_102 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_3_1, 1, %c72_i32_102) : i32
      %c64_i32_103 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_3_2, 0, %c64_i32_103) : i32
      %c72_i32_104 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_3_2, 1, %c72_i32_104) : i32
      %c64_i32_105 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_3_3, 0, %c64_i32_105) : i32
      %c72_i32_106 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_3_3, 1, %c72_i32_106) : i32
      %c64_i32_107 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_4_0, 0, %c64_i32_107) : i32
      %c72_i32_108 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_4_0, 1, %c72_i32_108) : i32
      %c64_i32_109 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_4_1, 0, %c64_i32_109) : i32
      %c72_i32_110 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_4_1, 1, %c72_i32_110) : i32
      %c64_i32_111 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_4_2, 0, %c64_i32_111) : i32
      %c72_i32_112 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_4_2, 1, %c72_i32_112) : i32
      %c64_i32_113 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_4_3, 0, %c64_i32_113) : i32
      %c72_i32_114 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_4_3, 1, %c72_i32_114) : i32
      %c64_i32_115 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_5_0, 0, %c64_i32_115) : i32
      %c72_i32_116 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_5_0, 1, %c72_i32_116) : i32
      %c64_i32_117 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_5_1, 0, %c64_i32_117) : i32
      %c72_i32_118 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_5_1, 1, %c72_i32_118) : i32
      %c64_i32_119 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_5_2, 0, %c64_i32_119) : i32
      %c72_i32_120 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_5_2, 1, %c72_i32_120) : i32
      %c64_i32_121 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_5_3, 0, %c64_i32_121) : i32
      %c72_i32_122 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_5_3, 1, %c72_i32_122) : i32
      %c64_i32_123 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_6_0, 0, %c64_i32_123) : i32
      %c72_i32_124 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_6_0, 1, %c72_i32_124) : i32
      %c64_i32_125 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_6_1, 0, %c64_i32_125) : i32
      %c72_i32_126 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_6_1, 1, %c72_i32_126) : i32
      %c64_i32_127 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_6_2, 0, %c64_i32_127) : i32
      %c72_i32_128 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_6_2, 1, %c72_i32_128) : i32
      %c64_i32_129 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_6_3, 0, %c64_i32_129) : i32
      %c72_i32_130 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_6_3, 1, %c72_i32_130) : i32
      %c64_i32_131 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_7_0, 0, %c64_i32_131) : i32
      %c72_i32_132 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_7_0, 1, %c72_i32_132) : i32
      %c64_i32_133 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_7_1, 0, %c64_i32_133) : i32
      %c72_i32_134 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_7_1, 1, %c72_i32_134) : i32
      %c64_i32_135 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_7_2, 0, %c64_i32_135) : i32
      %c72_i32_136 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_7_2, 1, %c72_i32_136) : i32
      %c64_i32_137 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_7_3, 0, %c64_i32_137) : i32
      %c72_i32_138 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_7_3, 1, %c72_i32_138) : i32
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
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 266240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 532480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 798720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1064960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1331200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1597440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1863680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 132608 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 1189376 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 2246144 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 3302912 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 4359680 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 5416448 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 6473216 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 7529984 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2129920 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2396160 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2662400 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2928640 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3194880 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3461120 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3727360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3993600 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 8586752 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 9643520 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 10700288 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 11757056 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 12813824 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 13870592 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 14927360 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 15984128 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4259840 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4526080 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%50)
      %51 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4792320 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5058560 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5324800 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5591040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5857280 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6123520 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%63)
      %64 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 17040896 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 18097664 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 19154432 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 20211200 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%67)
      %68 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 21267968 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 22324736 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 23381504 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 24438272 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6389760 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6656000 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6922240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7188480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7454720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7720960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7987200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%84)
      %85 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 8253440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 25495040 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 26551808 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 27608576 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 28665344 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 29722112 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 30778880 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 31835648 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 32892416 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
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
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%97)
      %98 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 266240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 532480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%100)
      %101 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%101)
      %102 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 798720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%102)
      %103 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%103)
      %104 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1064960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%104)
      %105 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%105)
      %106 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1331200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%106)
      %107 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%107)
      %108 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1597440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%108)
      %109 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%109)
      %110 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1863680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%110)
      %111 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%111)
      %112 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 132736 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%112)
      %113 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 1189504 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%113)
      %114 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 2246272 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%114)
      %115 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 3303040 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%115)
      %116 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 4359808 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%116)
      %117 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 5416576 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%117)
      %118 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 6473344 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%118)
      %119 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 7530112 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%119)
      %120 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2129920 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%120)
      %121 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%121)
      %122 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2396160 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%122)
      %123 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%123)
      %124 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2662400 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%124)
      %125 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%125)
      %126 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2928640 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%126)
      %127 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%127)
      %128 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3194880 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%128)
      %129 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%129)
      %130 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3461120 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%130)
      %131 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%131)
      %132 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3727360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%132)
      %133 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%133)
      %134 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3993600 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%134)
      %135 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%135)
      %136 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 8586880 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%136)
      %137 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 9643648 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%137)
      %138 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 10700416 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%138)
      %139 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 11757184 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%139)
      %140 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 12813952 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%140)
      %141 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 13870720 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%141)
      %142 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 14927488 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%142)
      %143 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 15984256 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
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
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4259840 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%144)
      %145 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%145)
      %146 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4526080 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%146)
      %147 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%147)
      %148 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4792320 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%148)
      %149 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%149)
      %150 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5058560 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%150)
      %151 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%151)
      %152 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5324800 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%152)
      %153 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%153)
      %154 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5591040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%154)
      %155 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%155)
      %156 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5857280 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%156)
      %157 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%157)
      %158 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6123520 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%158)
      %159 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%159)
      %160 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 17041024 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%160)
      %161 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 18097792 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%161)
      %162 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 19154560 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%162)
      %163 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 20211328 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%163)
      %164 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 21268096 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%164)
      %165 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 22324864 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%165)
      %166 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 23381632 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%166)
      %167 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 24438400 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%167)
      %168 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6389760 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%168)
      %169 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%169)
      %170 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6656000 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%170)
      %171 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%171)
      %172 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6922240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%172)
      %173 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%173)
      %174 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7188480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%174)
      %175 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%175)
      %176 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7454720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%176)
      %177 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%177)
      %178 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7720960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%178)
      %179 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%179)
      %180 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7987200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%180)
      %181 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%181)
      %182 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 8253440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%182)
      %183 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%183)
      %184 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 25495168 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%184)
      %185 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 26551936 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%185)
      %186 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 27608704 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%186)
      %187 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 28665472 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%187)
      %188 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 29722240 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%188)
      %189 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 30779008 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%189)
      %190 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 31835776 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%190)
      %191 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 32892544 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
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
      %192 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%192)
      %193 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%193)
      %194 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 266240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%194)
      %195 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%195)
      %196 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 532480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%196)
      %197 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%197)
      %198 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 798720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%198)
      %199 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%199)
      %200 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1064960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%200)
      %201 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%201)
      %202 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1331200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%202)
      %203 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%203)
      %204 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1597440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%204)
      %205 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%205)
      %206 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1863680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%206)
      %207 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%207)
      %208 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 132864 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%208)
      %209 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 1189632 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%209)
      %210 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 2246400 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%210)
      %211 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 3303168 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%211)
      %212 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 4359936 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%212)
      %213 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 5416704 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%213)
      %214 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 6473472 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%214)
      %215 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 7530240 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%215)
      %216 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2129920 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%216)
      %217 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%217)
      %218 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2396160 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%218)
      %219 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%219)
      %220 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2662400 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%220)
      %221 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%221)
      %222 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2928640 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%222)
      %223 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%223)
      %224 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3194880 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%224)
      %225 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%225)
      %226 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3461120 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%226)
      %227 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%227)
      %228 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3727360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%228)
      %229 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%229)
      %230 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3993600 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%230)
      %231 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%231)
      %232 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 8587008 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%232)
      %233 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 9643776 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%233)
      %234 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 10700544 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%234)
      %235 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 11757312 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%235)
      %236 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 12814080 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%236)
      %237 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 13870848 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%237)
      %238 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 14927616 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%238)
      %239 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 15984384 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%239)
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
      %240 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4259840 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%240)
      %241 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%241)
      %242 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4526080 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%242)
      %243 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%243)
      %244 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4792320 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%244)
      %245 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%245)
      %246 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5058560 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%246)
      %247 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%247)
      %248 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5324800 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%248)
      %249 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%249)
      %250 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5591040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%250)
      %251 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%251)
      %252 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5857280 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%252)
      %253 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%253)
      %254 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6123520 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%254)
      %255 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%255)
      %256 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 17041152 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%256)
      %257 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 18097920 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%257)
      %258 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 19154688 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%258)
      %259 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 20211456 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%259)
      %260 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 21268224 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%260)
      %261 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 22324992 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%261)
      %262 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 23381760 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%262)
      %263 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 24438528 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%263)
      %264 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6389760 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%264)
      %265 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%265)
      %266 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6656000 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%266)
      %267 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%267)
      %268 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6922240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%268)
      %269 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%269)
      %270 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7188480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%270)
      %271 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%271)
      %272 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7454720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%272)
      %273 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%273)
      %274 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7720960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%274)
      %275 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%275)
      %276 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7987200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%276)
      %277 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%277)
      %278 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 8253440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%278)
      %279 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%279)
      %280 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 25495296 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%280)
      %281 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 26552064 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%281)
      %282 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 27608832 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%282)
      %283 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 28665600 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%283)
      %284 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 29722368 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%284)
      %285 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 30779136 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%285)
      %286 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 31835904 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%286)
      %287 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 32892672 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%287)
      aiex.dma_await_task(%208)
      aiex.dma_await_task(%209)
      aiex.dma_await_task(%210)
      aiex.dma_await_task(%211)
      aiex.dma_await_task(%212)
      aiex.dma_await_task(%213)
      aiex.dma_await_task(%214)
      aiex.dma_await_task(%215)
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
      aiex.dma_await_task(%232)
      aiex.dma_await_task(%233)
      aiex.dma_await_task(%234)
      aiex.dma_await_task(%235)
      aiex.dma_await_task(%236)
      aiex.dma_await_task(%237)
      aiex.dma_await_task(%238)
      aiex.dma_await_task(%239)
      aiex.dma_free_task(%216)
      aiex.dma_free_task(%217)
      aiex.dma_free_task(%218)
      aiex.dma_free_task(%219)
      aiex.dma_free_task(%220)
      aiex.dma_free_task(%221)
      aiex.dma_free_task(%222)
      aiex.dma_free_task(%223)
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
      %288 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%288)
      %289 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%289)
      %290 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 266240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%290)
      %291 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%291)
      %292 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 532480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%292)
      %293 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%293)
      %294 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 798720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%294)
      %295 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%295)
      %296 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1064960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%296)
      %297 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%297)
      %298 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1331200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%298)
      %299 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%299)
      %300 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1597440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%300)
      %301 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%301)
      %302 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1863680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%302)
      %303 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%303)
      %304 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 132992 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%304)
      %305 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 1189760 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%305)
      %306 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 2246528 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%306)
      %307 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 3303296 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%307)
      %308 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 4360064 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%308)
      %309 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 5416832 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%309)
      %310 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 6473600 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%310)
      %311 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 7530368 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%311)
      %312 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2129920 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%312)
      %313 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%313)
      %314 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2396160 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%314)
      %315 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%315)
      %316 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2662400 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%316)
      %317 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%317)
      %318 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2928640 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%318)
      %319 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%319)
      %320 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3194880 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%320)
      %321 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%321)
      %322 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3461120 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%322)
      %323 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%323)
      %324 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3727360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%324)
      %325 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%325)
      %326 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3993600 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%326)
      %327 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%327)
      %328 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 8587136 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%328)
      %329 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 9643904 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%329)
      %330 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 10700672 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%330)
      %331 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 11757440 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%331)
      %332 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 12814208 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%332)
      %333 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 13870976 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%333)
      %334 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 14927744 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%334)
      %335 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 15984512 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%335)
      aiex.dma_await_task(%256)
      aiex.dma_await_task(%257)
      aiex.dma_await_task(%258)
      aiex.dma_await_task(%259)
      aiex.dma_await_task(%260)
      aiex.dma_await_task(%261)
      aiex.dma_await_task(%262)
      aiex.dma_await_task(%263)
      aiex.dma_free_task(%240)
      aiex.dma_free_task(%241)
      aiex.dma_free_task(%242)
      aiex.dma_free_task(%243)
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
      aiex.dma_await_task(%280)
      aiex.dma_await_task(%281)
      aiex.dma_await_task(%282)
      aiex.dma_await_task(%283)
      aiex.dma_await_task(%284)
      aiex.dma_await_task(%285)
      aiex.dma_await_task(%286)
      aiex.dma_await_task(%287)
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
      aiex.dma_free_task(%284)
      aiex.dma_free_task(%285)
      aiex.dma_free_task(%286)
      aiex.dma_free_task(%287)
      %336 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4259840 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%336)
      %337 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%337)
      %338 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4526080 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%338)
      %339 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%339)
      %340 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4792320 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%340)
      %341 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%341)
      %342 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5058560 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%342)
      %343 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%343)
      %344 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5324800 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%344)
      %345 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%345)
      %346 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5591040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%346)
      %347 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%347)
      %348 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5857280 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%348)
      %349 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%349)
      %350 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6123520 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%350)
      %351 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%351)
      %352 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 17041280 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%352)
      %353 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 18098048 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%353)
      %354 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 19154816 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%354)
      %355 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 20211584 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%355)
      %356 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 21268352 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%356)
      %357 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 22325120 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%357)
      %358 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 23381888 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%358)
      %359 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 24438656 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%359)
      %360 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6389760 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%360)
      %361 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%361)
      %362 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6656000 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%362)
      %363 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%363)
      %364 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6922240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%364)
      %365 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%365)
      %366 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7188480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%366)
      %367 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%367)
      %368 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7454720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%368)
      %369 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%369)
      %370 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7720960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%370)
      %371 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%371)
      %372 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7987200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%372)
      %373 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%373)
      %374 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 8253440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%374)
      %375 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%375)
      %376 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 25495424 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%376)
      %377 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 26552192 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%377)
      %378 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 27608960 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%378)
      %379 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 28665728 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%379)
      %380 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 29722496 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%380)
      %381 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 30779264 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%381)
      %382 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 31836032 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%382)
      %383 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 32892800 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%383)
      aiex.dma_await_task(%304)
      aiex.dma_await_task(%305)
      aiex.dma_await_task(%306)
      aiex.dma_await_task(%307)
      aiex.dma_await_task(%308)
      aiex.dma_await_task(%309)
      aiex.dma_await_task(%310)
      aiex.dma_await_task(%311)
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
      aiex.dma_free_task(%304)
      aiex.dma_free_task(%305)
      aiex.dma_free_task(%306)
      aiex.dma_free_task(%307)
      aiex.dma_free_task(%308)
      aiex.dma_free_task(%309)
      aiex.dma_free_task(%310)
      aiex.dma_free_task(%311)
      aiex.dma_await_task(%328)
      aiex.dma_await_task(%329)
      aiex.dma_await_task(%330)
      aiex.dma_await_task(%331)
      aiex.dma_await_task(%332)
      aiex.dma_await_task(%333)
      aiex.dma_await_task(%334)
      aiex.dma_await_task(%335)
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
      %384 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%384)
      %385 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%385)
      %386 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 266240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%386)
      %387 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%387)
      %388 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 532480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%388)
      %389 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%389)
      %390 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 798720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%390)
      %391 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%391)
      %392 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1064960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%392)
      %393 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%393)
      %394 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1331200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%394)
      %395 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%395)
      %396 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1597440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%396)
      %397 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%397)
      %398 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1863680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%398)
      %399 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%399)
      %400 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 133120 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%400)
      %401 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 1189888 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%401)
      %402 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 2246656 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%402)
      %403 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 3303424 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%403)
      %404 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 4360192 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%404)
      %405 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 5416960 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%405)
      %406 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 6473728 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%406)
      %407 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 7530496 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%407)
      %408 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2129920 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%408)
      %409 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%409)
      %410 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2396160 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%410)
      %411 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%411)
      %412 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2662400 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%412)
      %413 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%413)
      %414 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2928640 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%414)
      %415 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%415)
      %416 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3194880 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%416)
      %417 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%417)
      %418 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3461120 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%418)
      %419 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%419)
      %420 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3727360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%420)
      %421 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%421)
      %422 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3993600 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%422)
      %423 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%423)
      %424 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 8587264 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%424)
      %425 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 9644032 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%425)
      %426 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 10700800 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%426)
      %427 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 11757568 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%427)
      %428 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 12814336 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%428)
      %429 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 13871104 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%429)
      %430 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 14927872 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%430)
      %431 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 15984640 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%431)
      aiex.dma_await_task(%352)
      aiex.dma_await_task(%353)
      aiex.dma_await_task(%354)
      aiex.dma_await_task(%355)
      aiex.dma_await_task(%356)
      aiex.dma_await_task(%357)
      aiex.dma_await_task(%358)
      aiex.dma_await_task(%359)
      aiex.dma_free_task(%336)
      aiex.dma_free_task(%337)
      aiex.dma_free_task(%338)
      aiex.dma_free_task(%339)
      aiex.dma_free_task(%340)
      aiex.dma_free_task(%341)
      aiex.dma_free_task(%342)
      aiex.dma_free_task(%343)
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
      aiex.dma_await_task(%376)
      aiex.dma_await_task(%377)
      aiex.dma_await_task(%378)
      aiex.dma_await_task(%379)
      aiex.dma_await_task(%380)
      aiex.dma_await_task(%381)
      aiex.dma_await_task(%382)
      aiex.dma_await_task(%383)
      aiex.dma_free_task(%360)
      aiex.dma_free_task(%361)
      aiex.dma_free_task(%362)
      aiex.dma_free_task(%363)
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
      %432 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4259840 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%432)
      %433 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%433)
      %434 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4526080 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%434)
      %435 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%435)
      %436 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4792320 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%436)
      %437 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%437)
      %438 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5058560 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%438)
      %439 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%439)
      %440 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5324800 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%440)
      %441 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%441)
      %442 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5591040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%442)
      %443 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%443)
      %444 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5857280 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%444)
      %445 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%445)
      %446 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6123520 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%446)
      %447 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%447)
      %448 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 17041408 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%448)
      %449 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 18098176 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%449)
      %450 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 19154944 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%450)
      %451 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 20211712 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%451)
      %452 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 21268480 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%452)
      %453 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 22325248 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%453)
      %454 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 23382016 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%454)
      %455 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 24438784 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%455)
      %456 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6389760 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%456)
      %457 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%457)
      %458 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6656000 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%458)
      %459 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%459)
      %460 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6922240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%460)
      %461 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%461)
      %462 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7188480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%462)
      %463 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%463)
      %464 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7454720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%464)
      %465 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%465)
      %466 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7720960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%466)
      %467 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%467)
      %468 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7987200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%468)
      %469 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%469)
      %470 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 8253440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%470)
      %471 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%471)
      %472 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 25495552 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%472)
      %473 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 26552320 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%473)
      %474 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 27609088 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%474)
      %475 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 28665856 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%475)
      %476 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 29722624 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%476)
      %477 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 30779392 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%477)
      %478 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 31836160 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%478)
      %479 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 32892928 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%479)
      aiex.dma_await_task(%400)
      aiex.dma_await_task(%401)
      aiex.dma_await_task(%402)
      aiex.dma_await_task(%403)
      aiex.dma_await_task(%404)
      aiex.dma_await_task(%405)
      aiex.dma_await_task(%406)
      aiex.dma_await_task(%407)
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
      aiex.dma_free_task(%404)
      aiex.dma_free_task(%405)
      aiex.dma_free_task(%406)
      aiex.dma_free_task(%407)
      aiex.dma_await_task(%424)
      aiex.dma_await_task(%425)
      aiex.dma_await_task(%426)
      aiex.dma_await_task(%427)
      aiex.dma_await_task(%428)
      aiex.dma_await_task(%429)
      aiex.dma_await_task(%430)
      aiex.dma_await_task(%431)
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
      aiex.dma_free_task(%424)
      aiex.dma_free_task(%425)
      aiex.dma_free_task(%426)
      aiex.dma_free_task(%427)
      aiex.dma_free_task(%428)
      aiex.dma_free_task(%429)
      aiex.dma_free_task(%430)
      aiex.dma_free_task(%431)
      %480 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%480)
      %481 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%481)
      %482 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 266240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%482)
      %483 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%483)
      %484 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 532480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%484)
      %485 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%485)
      %486 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 798720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%486)
      %487 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%487)
      %488 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1064960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%488)
      %489 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%489)
      %490 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1331200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%490)
      %491 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%491)
      %492 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1597440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%492)
      %493 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%493)
      %494 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1863680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%494)
      %495 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%495)
      %496 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 133248 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%496)
      %497 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 1190016 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%497)
      %498 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 2246784 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%498)
      %499 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 3303552 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%499)
      %500 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 4360320 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%500)
      %501 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 5417088 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%501)
      %502 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 6473856 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%502)
      %503 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 7530624 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%503)
      %504 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2129920 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%504)
      %505 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%505)
      %506 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2396160 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%506)
      %507 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%507)
      %508 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2662400 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%508)
      %509 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%509)
      %510 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2928640 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%510)
      %511 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%511)
      %512 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3194880 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%512)
      %513 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%513)
      %514 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3461120 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%514)
      %515 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%515)
      %516 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3727360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%516)
      %517 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%517)
      %518 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3993600 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%518)
      %519 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%519)
      %520 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 8587392 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%520)
      %521 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 9644160 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%521)
      %522 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 10700928 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%522)
      %523 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 11757696 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%523)
      %524 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 12814464 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%524)
      %525 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 13871232 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%525)
      %526 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 14928000 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%526)
      %527 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 15984768 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%527)
      aiex.dma_await_task(%448)
      aiex.dma_await_task(%449)
      aiex.dma_await_task(%450)
      aiex.dma_await_task(%451)
      aiex.dma_await_task(%452)
      aiex.dma_await_task(%453)
      aiex.dma_await_task(%454)
      aiex.dma_await_task(%455)
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
      aiex.dma_await_task(%472)
      aiex.dma_await_task(%473)
      aiex.dma_await_task(%474)
      aiex.dma_await_task(%475)
      aiex.dma_await_task(%476)
      aiex.dma_await_task(%477)
      aiex.dma_await_task(%478)
      aiex.dma_await_task(%479)
      aiex.dma_free_task(%456)
      aiex.dma_free_task(%457)
      aiex.dma_free_task(%458)
      aiex.dma_free_task(%459)
      aiex.dma_free_task(%460)
      aiex.dma_free_task(%461)
      aiex.dma_free_task(%462)
      aiex.dma_free_task(%463)
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
      %528 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4259840 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%528)
      %529 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%529)
      %530 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4526080 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%530)
      %531 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%531)
      %532 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4792320 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%532)
      %533 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%533)
      %534 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5058560 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%534)
      %535 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%535)
      %536 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5324800 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%536)
      %537 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%537)
      %538 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5591040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%538)
      %539 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%539)
      %540 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5857280 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%540)
      %541 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%541)
      %542 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6123520 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%542)
      %543 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%543)
      %544 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 17041536 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%544)
      %545 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 18098304 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%545)
      %546 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 19155072 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%546)
      %547 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 20211840 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%547)
      %548 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 21268608 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%548)
      %549 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 22325376 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%549)
      %550 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 23382144 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%550)
      %551 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 24438912 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%551)
      %552 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6389760 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%552)
      %553 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%553)
      %554 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6656000 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%554)
      %555 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%555)
      %556 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6922240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%556)
      %557 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%557)
      %558 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7188480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%558)
      %559 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%559)
      %560 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7454720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%560)
      %561 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%561)
      %562 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7720960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%562)
      %563 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%563)
      %564 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7987200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%564)
      %565 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%565)
      %566 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 8253440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%566)
      %567 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%567)
      %568 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 25495680 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%568)
      %569 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 26552448 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%569)
      %570 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 27609216 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%570)
      %571 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 28665984 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%571)
      %572 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 29722752 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%572)
      %573 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 30779520 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%573)
      %574 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 31836288 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%574)
      %575 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 32893056 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%575)
      aiex.dma_await_task(%496)
      aiex.dma_await_task(%497)
      aiex.dma_await_task(%498)
      aiex.dma_await_task(%499)
      aiex.dma_await_task(%500)
      aiex.dma_await_task(%501)
      aiex.dma_await_task(%502)
      aiex.dma_await_task(%503)
      aiex.dma_free_task(%480)
      aiex.dma_free_task(%481)
      aiex.dma_free_task(%482)
      aiex.dma_free_task(%483)
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
      aiex.dma_await_task(%520)
      aiex.dma_await_task(%521)
      aiex.dma_await_task(%522)
      aiex.dma_await_task(%523)
      aiex.dma_await_task(%524)
      aiex.dma_await_task(%525)
      aiex.dma_await_task(%526)
      aiex.dma_await_task(%527)
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
      aiex.dma_free_task(%524)
      aiex.dma_free_task(%525)
      aiex.dma_free_task(%526)
      aiex.dma_free_task(%527)
      %576 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%576)
      %577 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%577)
      %578 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 266240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%578)
      %579 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%579)
      %580 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 532480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%580)
      %581 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%581)
      %582 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 798720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%582)
      %583 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%583)
      %584 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1064960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%584)
      %585 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%585)
      %586 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1331200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%586)
      %587 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%587)
      %588 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1597440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%588)
      %589 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%589)
      %590 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1863680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%590)
      %591 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%591)
      %592 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 133376 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%592)
      %593 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 1190144 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%593)
      %594 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 2246912 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%594)
      %595 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 3303680 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%595)
      %596 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 4360448 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%596)
      %597 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 5417216 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%597)
      %598 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 6473984 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%598)
      %599 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 7530752 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%599)
      %600 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2129920 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%600)
      %601 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%601)
      %602 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2396160 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%602)
      %603 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%603)
      %604 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2662400 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%604)
      %605 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%605)
      %606 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2928640 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%606)
      %607 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%607)
      %608 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3194880 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%608)
      %609 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%609)
      %610 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3461120 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%610)
      %611 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%611)
      %612 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3727360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%612)
      %613 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%613)
      %614 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3993600 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%614)
      %615 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%615)
      %616 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 8587520 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%616)
      %617 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 9644288 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%617)
      %618 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 10701056 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%618)
      %619 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 11757824 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%619)
      %620 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 12814592 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%620)
      %621 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 13871360 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%621)
      %622 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 14928128 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%622)
      %623 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 15984896 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%623)
      aiex.dma_await_task(%544)
      aiex.dma_await_task(%545)
      aiex.dma_await_task(%546)
      aiex.dma_await_task(%547)
      aiex.dma_await_task(%548)
      aiex.dma_await_task(%549)
      aiex.dma_await_task(%550)
      aiex.dma_await_task(%551)
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
      aiex.dma_free_task(%544)
      aiex.dma_free_task(%545)
      aiex.dma_free_task(%546)
      aiex.dma_free_task(%547)
      aiex.dma_free_task(%548)
      aiex.dma_free_task(%549)
      aiex.dma_free_task(%550)
      aiex.dma_free_task(%551)
      aiex.dma_await_task(%568)
      aiex.dma_await_task(%569)
      aiex.dma_await_task(%570)
      aiex.dma_await_task(%571)
      aiex.dma_await_task(%572)
      aiex.dma_await_task(%573)
      aiex.dma_await_task(%574)
      aiex.dma_await_task(%575)
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
      %624 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4259840 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%624)
      %625 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%625)
      %626 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4526080 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%626)
      %627 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%627)
      %628 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4792320 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%628)
      %629 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%629)
      %630 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5058560 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%630)
      %631 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%631)
      %632 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5324800 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%632)
      %633 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%633)
      %634 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5591040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%634)
      %635 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%635)
      %636 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5857280 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%636)
      %637 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%637)
      %638 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6123520 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%638)
      %639 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%639)
      %640 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 17041664 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%640)
      %641 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 18098432 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%641)
      %642 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 19155200 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%642)
      %643 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 20211968 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%643)
      %644 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 21268736 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%644)
      %645 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 22325504 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%645)
      %646 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 23382272 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%646)
      %647 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 24439040 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%647)
      %648 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6389760 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%648)
      %649 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%649)
      %650 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6656000 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%650)
      %651 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%651)
      %652 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6922240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%652)
      %653 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%653)
      %654 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7188480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%654)
      %655 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%655)
      %656 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7454720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%656)
      %657 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%657)
      %658 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7720960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%658)
      %659 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%659)
      %660 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7987200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%660)
      %661 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%661)
      %662 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 8253440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%662)
      %663 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%663)
      %664 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 25495808 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%664)
      %665 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 26552576 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%665)
      %666 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 27609344 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%666)
      %667 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 28666112 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%667)
      %668 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 29722880 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%668)
      %669 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 30779648 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%669)
      %670 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 31836416 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%670)
      %671 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 32893184 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%671)
      aiex.dma_await_task(%592)
      aiex.dma_await_task(%593)
      aiex.dma_await_task(%594)
      aiex.dma_await_task(%595)
      aiex.dma_await_task(%596)
      aiex.dma_await_task(%597)
      aiex.dma_await_task(%598)
      aiex.dma_await_task(%599)
      aiex.dma_free_task(%576)
      aiex.dma_free_task(%577)
      aiex.dma_free_task(%578)
      aiex.dma_free_task(%579)
      aiex.dma_free_task(%580)
      aiex.dma_free_task(%581)
      aiex.dma_free_task(%582)
      aiex.dma_free_task(%583)
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
      aiex.dma_await_task(%616)
      aiex.dma_await_task(%617)
      aiex.dma_await_task(%618)
      aiex.dma_await_task(%619)
      aiex.dma_await_task(%620)
      aiex.dma_await_task(%621)
      aiex.dma_await_task(%622)
      aiex.dma_await_task(%623)
      aiex.dma_free_task(%600)
      aiex.dma_free_task(%601)
      aiex.dma_free_task(%602)
      aiex.dma_free_task(%603)
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
      %672 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%672)
      %673 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%673)
      %674 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 266240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%674)
      %675 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%675)
      %676 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 532480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%676)
      %677 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%677)
      %678 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 798720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%678)
      %679 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%679)
      %680 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1064960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%680)
      %681 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%681)
      %682 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1331200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%682)
      %683 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%683)
      %684 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1597440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%684)
      %685 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%685)
      %686 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1863680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%686)
      %687 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%687)
      %688 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 133504 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%688)
      %689 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 1190272 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%689)
      %690 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 2247040 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%690)
      %691 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 3303808 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%691)
      %692 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 4360576 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%692)
      %693 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 5417344 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%693)
      %694 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 6474112 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%694)
      %695 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 7530880 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%695)
      %696 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2129920 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%696)
      %697 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%697)
      %698 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2396160 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%698)
      %699 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%699)
      %700 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2662400 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%700)
      %701 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%701)
      %702 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2928640 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%702)
      %703 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%703)
      %704 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3194880 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%704)
      %705 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%705)
      %706 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3461120 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%706)
      %707 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%707)
      %708 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3727360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%708)
      %709 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%709)
      %710 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3993600 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%710)
      %711 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%711)
      %712 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 8587648 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%712)
      %713 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 9644416 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%713)
      %714 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 10701184 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%714)
      %715 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 11757952 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%715)
      %716 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 12814720 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%716)
      %717 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 13871488 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%717)
      %718 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 14928256 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%718)
      %719 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 15985024 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%719)
      aiex.dma_await_task(%640)
      aiex.dma_await_task(%641)
      aiex.dma_await_task(%642)
      aiex.dma_await_task(%643)
      aiex.dma_await_task(%644)
      aiex.dma_await_task(%645)
      aiex.dma_await_task(%646)
      aiex.dma_await_task(%647)
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
      aiex.dma_free_task(%644)
      aiex.dma_free_task(%645)
      aiex.dma_free_task(%646)
      aiex.dma_free_task(%647)
      aiex.dma_await_task(%664)
      aiex.dma_await_task(%665)
      aiex.dma_await_task(%666)
      aiex.dma_await_task(%667)
      aiex.dma_await_task(%668)
      aiex.dma_await_task(%669)
      aiex.dma_await_task(%670)
      aiex.dma_await_task(%671)
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
      aiex.dma_free_task(%664)
      aiex.dma_free_task(%665)
      aiex.dma_free_task(%666)
      aiex.dma_free_task(%667)
      aiex.dma_free_task(%668)
      aiex.dma_free_task(%669)
      aiex.dma_free_task(%670)
      aiex.dma_free_task(%671)
      %720 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4259840 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%720)
      %721 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%721)
      %722 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4526080 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%722)
      %723 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%723)
      %724 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4792320 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%724)
      %725 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%725)
      %726 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5058560 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%726)
      %727 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%727)
      %728 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5324800 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%728)
      %729 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%729)
      %730 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5591040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%730)
      %731 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%731)
      %732 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5857280 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%732)
      %733 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%733)
      %734 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6123520 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%734)
      %735 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%735)
      %736 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 17041792 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%736)
      %737 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 18098560 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%737)
      %738 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 19155328 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%738)
      %739 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 20212096 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%739)
      %740 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 21268864 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%740)
      %741 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 22325632 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%741)
      %742 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 23382400 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%742)
      %743 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 24439168 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%743)
      %744 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6389760 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%744)
      %745 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%745)
      %746 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6656000 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%746)
      %747 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%747)
      %748 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6922240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%748)
      %749 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%749)
      %750 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7188480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%750)
      %751 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%751)
      %752 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7454720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%752)
      %753 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%753)
      %754 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7720960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%754)
      %755 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%755)
      %756 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7987200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%756)
      %757 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%757)
      %758 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 8253440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%758)
      %759 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%759)
      %760 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 25495936 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%760)
      %761 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 26552704 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%761)
      %762 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 27609472 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%762)
      %763 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 28666240 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%763)
      %764 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 29723008 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%764)
      %765 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 30779776 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%765)
      %766 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 31836544 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%766)
      %767 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 32893312 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%767)
      aiex.dma_await_task(%688)
      aiex.dma_await_task(%689)
      aiex.dma_await_task(%690)
      aiex.dma_await_task(%691)
      aiex.dma_await_task(%692)
      aiex.dma_await_task(%693)
      aiex.dma_await_task(%694)
      aiex.dma_await_task(%695)
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
      aiex.dma_await_task(%712)
      aiex.dma_await_task(%713)
      aiex.dma_await_task(%714)
      aiex.dma_await_task(%715)
      aiex.dma_await_task(%716)
      aiex.dma_await_task(%717)
      aiex.dma_await_task(%718)
      aiex.dma_await_task(%719)
      aiex.dma_free_task(%696)
      aiex.dma_free_task(%697)
      aiex.dma_free_task(%698)
      aiex.dma_free_task(%699)
      aiex.dma_free_task(%700)
      aiex.dma_free_task(%701)
      aiex.dma_free_task(%702)
      aiex.dma_free_task(%703)
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
      %768 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%768)
      %769 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%769)
      %770 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 266240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%770)
      %771 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%771)
      %772 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 532480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%772)
      %773 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%773)
      %774 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 798720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%774)
      %775 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%775)
      %776 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1064960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%776)
      %777 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%777)
      %778 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1331200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%778)
      %779 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%779)
      %780 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1597440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%780)
      %781 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%781)
      %782 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1863680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%782)
      %783 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%783)
      %784 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 264704 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%784)
      %785 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 1321472 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%785)
      %786 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 2378240 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%786)
      %787 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 3435008 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%787)
      %788 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 4491776 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%788)
      %789 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 5548544 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%789)
      %790 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 6605312 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%790)
      %791 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 7662080 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%791)
      %792 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2129920 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%792)
      %793 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%793)
      %794 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2396160 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%794)
      %795 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%795)
      %796 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2662400 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%796)
      %797 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%797)
      %798 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2928640 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%798)
      %799 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%799)
      %800 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3194880 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%800)
      %801 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%801)
      %802 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3461120 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%802)
      %803 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%803)
      %804 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3727360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%804)
      %805 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%805)
      %806 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3993600 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%806)
      %807 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%807)
      %808 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 8718848 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%808)
      %809 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 9775616 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%809)
      %810 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 10832384 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%810)
      %811 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 11889152 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%811)
      %812 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 12945920 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%812)
      %813 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 14002688 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%813)
      %814 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 15059456 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%814)
      %815 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 16116224 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%815)
      aiex.dma_await_task(%736)
      aiex.dma_await_task(%737)
      aiex.dma_await_task(%738)
      aiex.dma_await_task(%739)
      aiex.dma_await_task(%740)
      aiex.dma_await_task(%741)
      aiex.dma_await_task(%742)
      aiex.dma_await_task(%743)
      aiex.dma_free_task(%720)
      aiex.dma_free_task(%721)
      aiex.dma_free_task(%722)
      aiex.dma_free_task(%723)
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
      aiex.dma_await_task(%760)
      aiex.dma_await_task(%761)
      aiex.dma_await_task(%762)
      aiex.dma_await_task(%763)
      aiex.dma_await_task(%764)
      aiex.dma_await_task(%765)
      aiex.dma_await_task(%766)
      aiex.dma_await_task(%767)
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
      aiex.dma_free_task(%764)
      aiex.dma_free_task(%765)
      aiex.dma_free_task(%766)
      aiex.dma_free_task(%767)
      %816 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4259840 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%816)
      %817 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%817)
      %818 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4526080 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%818)
      %819 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%819)
      %820 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4792320 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%820)
      %821 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%821)
      %822 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5058560 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%822)
      %823 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%823)
      %824 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5324800 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%824)
      %825 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%825)
      %826 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5591040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%826)
      %827 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%827)
      %828 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5857280 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%828)
      %829 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%829)
      %830 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6123520 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%830)
      %831 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%831)
      %832 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 17172992 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%832)
      %833 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 18229760 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%833)
      %834 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 19286528 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%834)
      %835 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 20343296 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%835)
      %836 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 21400064 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%836)
      %837 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 22456832 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%837)
      %838 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 23513600 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%838)
      %839 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 24570368 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%839)
      %840 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6389760 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%840)
      %841 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%841)
      %842 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6656000 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%842)
      %843 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%843)
      %844 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6922240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%844)
      %845 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%845)
      %846 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7188480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%846)
      %847 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%847)
      %848 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7454720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%848)
      %849 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%849)
      %850 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7720960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%850)
      %851 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%851)
      %852 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7987200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%852)
      %853 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%853)
      %854 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 8253440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%854)
      %855 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%855)
      %856 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 25627136 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%856)
      %857 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 26683904 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%857)
      %858 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 27740672 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%858)
      %859 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 28797440 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%859)
      %860 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 29854208 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%860)
      %861 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 30910976 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%861)
      %862 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 31967744 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%862)
      %863 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 33024512 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%863)
      aiex.dma_await_task(%784)
      aiex.dma_await_task(%785)
      aiex.dma_await_task(%786)
      aiex.dma_await_task(%787)
      aiex.dma_await_task(%788)
      aiex.dma_await_task(%789)
      aiex.dma_await_task(%790)
      aiex.dma_await_task(%791)
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
      aiex.dma_free_task(%784)
      aiex.dma_free_task(%785)
      aiex.dma_free_task(%786)
      aiex.dma_free_task(%787)
      aiex.dma_free_task(%788)
      aiex.dma_free_task(%789)
      aiex.dma_free_task(%790)
      aiex.dma_free_task(%791)
      aiex.dma_await_task(%808)
      aiex.dma_await_task(%809)
      aiex.dma_await_task(%810)
      aiex.dma_await_task(%811)
      aiex.dma_await_task(%812)
      aiex.dma_await_task(%813)
      aiex.dma_await_task(%814)
      aiex.dma_await_task(%815)
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
      %864 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%864)
      %865 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%865)
      %866 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 266240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%866)
      %867 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%867)
      %868 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 532480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%868)
      %869 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%869)
      %870 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 798720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%870)
      %871 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%871)
      %872 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1064960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%872)
      %873 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%873)
      %874 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1331200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%874)
      %875 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%875)
      %876 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1597440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%876)
      %877 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%877)
      %878 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1863680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%878)
      %879 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%879)
      %880 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 264832 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%880)
      %881 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 1321600 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%881)
      %882 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 2378368 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%882)
      %883 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 3435136 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%883)
      %884 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 4491904 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%884)
      %885 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 5548672 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%885)
      %886 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 6605440 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%886)
      %887 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 7662208 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%887)
      %888 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2129920 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%888)
      %889 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%889)
      %890 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2396160 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%890)
      %891 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%891)
      %892 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2662400 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%892)
      %893 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%893)
      %894 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2928640 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%894)
      %895 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%895)
      %896 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3194880 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%896)
      %897 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%897)
      %898 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3461120 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%898)
      %899 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%899)
      %900 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3727360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%900)
      %901 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%901)
      %902 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3993600 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%902)
      %903 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%903)
      %904 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 8718976 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%904)
      %905 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 9775744 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%905)
      %906 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 10832512 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%906)
      %907 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 11889280 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%907)
      %908 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 12946048 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%908)
      %909 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 14002816 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%909)
      %910 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 15059584 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%910)
      %911 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 16116352 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%911)
      aiex.dma_await_task(%832)
      aiex.dma_await_task(%833)
      aiex.dma_await_task(%834)
      aiex.dma_await_task(%835)
      aiex.dma_await_task(%836)
      aiex.dma_await_task(%837)
      aiex.dma_await_task(%838)
      aiex.dma_await_task(%839)
      aiex.dma_free_task(%816)
      aiex.dma_free_task(%817)
      aiex.dma_free_task(%818)
      aiex.dma_free_task(%819)
      aiex.dma_free_task(%820)
      aiex.dma_free_task(%821)
      aiex.dma_free_task(%822)
      aiex.dma_free_task(%823)
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
      aiex.dma_await_task(%856)
      aiex.dma_await_task(%857)
      aiex.dma_await_task(%858)
      aiex.dma_await_task(%859)
      aiex.dma_await_task(%860)
      aiex.dma_await_task(%861)
      aiex.dma_await_task(%862)
      aiex.dma_await_task(%863)
      aiex.dma_free_task(%840)
      aiex.dma_free_task(%841)
      aiex.dma_free_task(%842)
      aiex.dma_free_task(%843)
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
      %912 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4259840 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%912)
      %913 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%913)
      %914 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4526080 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%914)
      %915 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%915)
      %916 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4792320 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%916)
      %917 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%917)
      %918 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5058560 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%918)
      %919 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%919)
      %920 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5324800 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%920)
      %921 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%921)
      %922 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5591040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%922)
      %923 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%923)
      %924 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5857280 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%924)
      %925 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%925)
      %926 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6123520 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%926)
      %927 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%927)
      %928 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 17173120 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%928)
      %929 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 18229888 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%929)
      %930 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 19286656 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%930)
      %931 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 20343424 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%931)
      %932 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 21400192 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%932)
      %933 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 22456960 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%933)
      %934 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 23513728 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%934)
      %935 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 24570496 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%935)
      %936 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6389760 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%936)
      %937 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%937)
      %938 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6656000 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%938)
      %939 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%939)
      %940 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6922240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%940)
      %941 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%941)
      %942 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7188480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%942)
      %943 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%943)
      %944 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7454720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%944)
      %945 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%945)
      %946 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7720960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%946)
      %947 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%947)
      %948 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7987200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%948)
      %949 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%949)
      %950 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 8253440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%950)
      %951 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%951)
      %952 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 25627264 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%952)
      %953 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 26684032 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%953)
      %954 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 27740800 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%954)
      %955 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 28797568 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%955)
      %956 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 29854336 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%956)
      %957 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 30911104 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%957)
      %958 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 31967872 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%958)
      %959 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 33024640 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%959)
      aiex.dma_await_task(%880)
      aiex.dma_await_task(%881)
      aiex.dma_await_task(%882)
      aiex.dma_await_task(%883)
      aiex.dma_await_task(%884)
      aiex.dma_await_task(%885)
      aiex.dma_await_task(%886)
      aiex.dma_await_task(%887)
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
      aiex.dma_free_task(%884)
      aiex.dma_free_task(%885)
      aiex.dma_free_task(%886)
      aiex.dma_free_task(%887)
      aiex.dma_await_task(%904)
      aiex.dma_await_task(%905)
      aiex.dma_await_task(%906)
      aiex.dma_await_task(%907)
      aiex.dma_await_task(%908)
      aiex.dma_await_task(%909)
      aiex.dma_await_task(%910)
      aiex.dma_await_task(%911)
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
      aiex.dma_free_task(%904)
      aiex.dma_free_task(%905)
      aiex.dma_free_task(%906)
      aiex.dma_free_task(%907)
      aiex.dma_free_task(%908)
      aiex.dma_free_task(%909)
      aiex.dma_free_task(%910)
      aiex.dma_free_task(%911)
      %960 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%960)
      %961 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%961)
      %962 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 266240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%962)
      %963 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%963)
      %964 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 532480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%964)
      %965 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%965)
      %966 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 798720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%966)
      %967 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%967)
      %968 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1064960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%968)
      %969 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%969)
      %970 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1331200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%970)
      %971 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%971)
      %972 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1597440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%972)
      %973 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%973)
      %974 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1863680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%974)
      %975 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%975)
      %976 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 264960 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%976)
      %977 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 1321728 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%977)
      %978 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 2378496 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%978)
      %979 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 3435264 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%979)
      %980 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 4492032 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%980)
      %981 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 5548800 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%981)
      %982 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 6605568 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%982)
      %983 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 7662336 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%983)
      %984 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2129920 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%984)
      %985 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%985)
      %986 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2396160 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%986)
      %987 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%987)
      %988 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2662400 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%988)
      %989 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%989)
      %990 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2928640 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%990)
      %991 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%991)
      %992 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3194880 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%992)
      %993 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%993)
      %994 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3461120 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%994)
      %995 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%995)
      %996 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3727360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%996)
      %997 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%997)
      %998 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3993600 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%998)
      %999 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%999)
      %1000 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 8719104 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1000)
      %1001 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 9775872 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1001)
      %1002 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 10832640 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1002)
      %1003 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 11889408 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1003)
      %1004 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 12946176 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1004)
      %1005 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 14002944 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1005)
      %1006 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 15059712 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1006)
      %1007 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 16116480 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1007)
      aiex.dma_await_task(%928)
      aiex.dma_await_task(%929)
      aiex.dma_await_task(%930)
      aiex.dma_await_task(%931)
      aiex.dma_await_task(%932)
      aiex.dma_await_task(%933)
      aiex.dma_await_task(%934)
      aiex.dma_await_task(%935)
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
      aiex.dma_await_task(%952)
      aiex.dma_await_task(%953)
      aiex.dma_await_task(%954)
      aiex.dma_await_task(%955)
      aiex.dma_await_task(%956)
      aiex.dma_await_task(%957)
      aiex.dma_await_task(%958)
      aiex.dma_await_task(%959)
      aiex.dma_free_task(%936)
      aiex.dma_free_task(%937)
      aiex.dma_free_task(%938)
      aiex.dma_free_task(%939)
      aiex.dma_free_task(%940)
      aiex.dma_free_task(%941)
      aiex.dma_free_task(%942)
      aiex.dma_free_task(%943)
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
      %1008 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4259840 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1008)
      %1009 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1009)
      %1010 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4526080 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1010)
      %1011 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1011)
      %1012 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4792320 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1012)
      %1013 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1013)
      %1014 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5058560 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1014)
      %1015 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1015)
      %1016 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5324800 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1016)
      %1017 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1017)
      %1018 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5591040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1018)
      %1019 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1019)
      %1020 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5857280 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1020)
      %1021 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1021)
      %1022 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6123520 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1022)
      %1023 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1023)
      %1024 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 17173248 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1024)
      %1025 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 18230016 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1025)
      %1026 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 19286784 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1026)
      %1027 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 20343552 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1027)
      %1028 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 21400320 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1028)
      %1029 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 22457088 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1029)
      %1030 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 23513856 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1030)
      %1031 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 24570624 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1031)
      %1032 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6389760 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1032)
      %1033 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1033)
      %1034 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6656000 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1034)
      %1035 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1035)
      %1036 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6922240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1036)
      %1037 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1037)
      %1038 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7188480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1038)
      %1039 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1039)
      %1040 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7454720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1040)
      %1041 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1041)
      %1042 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7720960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1042)
      %1043 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1043)
      %1044 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7987200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1044)
      %1045 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1045)
      %1046 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 8253440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1046)
      %1047 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1047)
      %1048 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 25627392 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1048)
      %1049 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 26684160 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1049)
      %1050 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 27740928 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1050)
      %1051 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 28797696 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1051)
      %1052 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 29854464 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1052)
      %1053 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 30911232 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1053)
      %1054 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 31968000 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1054)
      %1055 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 33024768 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1055)
      aiex.dma_await_task(%976)
      aiex.dma_await_task(%977)
      aiex.dma_await_task(%978)
      aiex.dma_await_task(%979)
      aiex.dma_await_task(%980)
      aiex.dma_await_task(%981)
      aiex.dma_await_task(%982)
      aiex.dma_await_task(%983)
      aiex.dma_free_task(%960)
      aiex.dma_free_task(%961)
      aiex.dma_free_task(%962)
      aiex.dma_free_task(%963)
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
      aiex.dma_await_task(%1000)
      aiex.dma_await_task(%1001)
      aiex.dma_await_task(%1002)
      aiex.dma_await_task(%1003)
      aiex.dma_await_task(%1004)
      aiex.dma_await_task(%1005)
      aiex.dma_await_task(%1006)
      aiex.dma_await_task(%1007)
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
      aiex.dma_free_task(%1004)
      aiex.dma_free_task(%1005)
      aiex.dma_free_task(%1006)
      aiex.dma_free_task(%1007)
      %1056 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1056)
      %1057 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1057)
      %1058 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 266240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1058)
      %1059 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1059)
      %1060 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 532480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1060)
      %1061 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1061)
      %1062 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 798720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1062)
      %1063 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1063)
      %1064 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1064960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1064)
      %1065 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1065)
      %1066 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1331200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1066)
      %1067 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1067)
      %1068 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1597440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1068)
      %1069 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1069)
      %1070 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1863680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1070)
      %1071 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1071)
      %1072 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 265088 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1072)
      %1073 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 1321856 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1073)
      %1074 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 2378624 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1074)
      %1075 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 3435392 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1075)
      %1076 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 4492160 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1076)
      %1077 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 5548928 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1077)
      %1078 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 6605696 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1078)
      %1079 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 7662464 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1079)
      %1080 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2129920 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1080)
      %1081 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1081)
      %1082 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2396160 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1082)
      %1083 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1083)
      %1084 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2662400 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1084)
      %1085 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1085)
      %1086 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2928640 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1086)
      %1087 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1087)
      %1088 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3194880 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1088)
      %1089 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1089)
      %1090 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3461120 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1090)
      %1091 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1091)
      %1092 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3727360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1092)
      %1093 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1093)
      %1094 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3993600 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1094)
      %1095 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1095)
      %1096 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 8719232 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1096)
      %1097 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 9776000 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1097)
      %1098 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 10832768 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1098)
      %1099 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 11889536 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1099)
      %1100 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 12946304 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1100)
      %1101 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 14003072 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1101)
      %1102 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 15059840 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1102)
      %1103 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 16116608 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1103)
      aiex.dma_await_task(%1024)
      aiex.dma_await_task(%1025)
      aiex.dma_await_task(%1026)
      aiex.dma_await_task(%1027)
      aiex.dma_await_task(%1028)
      aiex.dma_await_task(%1029)
      aiex.dma_await_task(%1030)
      aiex.dma_await_task(%1031)
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
      aiex.dma_free_task(%1024)
      aiex.dma_free_task(%1025)
      aiex.dma_free_task(%1026)
      aiex.dma_free_task(%1027)
      aiex.dma_free_task(%1028)
      aiex.dma_free_task(%1029)
      aiex.dma_free_task(%1030)
      aiex.dma_free_task(%1031)
      aiex.dma_await_task(%1048)
      aiex.dma_await_task(%1049)
      aiex.dma_await_task(%1050)
      aiex.dma_await_task(%1051)
      aiex.dma_await_task(%1052)
      aiex.dma_await_task(%1053)
      aiex.dma_await_task(%1054)
      aiex.dma_await_task(%1055)
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
      %1104 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4259840 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1104)
      %1105 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1105)
      %1106 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4526080 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1106)
      %1107 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1107)
      %1108 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4792320 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1108)
      %1109 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1109)
      %1110 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5058560 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1110)
      %1111 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1111)
      %1112 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5324800 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1112)
      %1113 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1113)
      %1114 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5591040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1114)
      %1115 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1115)
      %1116 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5857280 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1116)
      %1117 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1117)
      %1118 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6123520 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1118)
      %1119 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1119)
      %1120 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 17173376 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1120)
      %1121 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 18230144 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1121)
      %1122 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 19286912 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1122)
      %1123 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 20343680 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1123)
      %1124 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 21400448 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1124)
      %1125 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 22457216 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1125)
      %1126 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 23513984 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1126)
      %1127 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 24570752 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1127)
      %1128 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6389760 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1128)
      %1129 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1129)
      %1130 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6656000 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1130)
      %1131 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1131)
      %1132 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6922240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1132)
      %1133 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1133)
      %1134 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7188480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1134)
      %1135 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1135)
      %1136 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7454720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1136)
      %1137 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1137)
      %1138 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7720960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1138)
      %1139 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1139)
      %1140 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7987200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1140)
      %1141 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1141)
      %1142 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 8253440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1142)
      %1143 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1143)
      %1144 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 25627520 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1144)
      %1145 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 26684288 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1145)
      %1146 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 27741056 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1146)
      %1147 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 28797824 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1147)
      %1148 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 29854592 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1148)
      %1149 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 30911360 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1149)
      %1150 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 31968128 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1150)
      %1151 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 33024896 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1151)
      aiex.dma_await_task(%1072)
      aiex.dma_await_task(%1073)
      aiex.dma_await_task(%1074)
      aiex.dma_await_task(%1075)
      aiex.dma_await_task(%1076)
      aiex.dma_await_task(%1077)
      aiex.dma_await_task(%1078)
      aiex.dma_await_task(%1079)
      aiex.dma_free_task(%1056)
      aiex.dma_free_task(%1057)
      aiex.dma_free_task(%1058)
      aiex.dma_free_task(%1059)
      aiex.dma_free_task(%1060)
      aiex.dma_free_task(%1061)
      aiex.dma_free_task(%1062)
      aiex.dma_free_task(%1063)
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
      aiex.dma_await_task(%1096)
      aiex.dma_await_task(%1097)
      aiex.dma_await_task(%1098)
      aiex.dma_await_task(%1099)
      aiex.dma_await_task(%1100)
      aiex.dma_await_task(%1101)
      aiex.dma_await_task(%1102)
      aiex.dma_await_task(%1103)
      aiex.dma_free_task(%1080)
      aiex.dma_free_task(%1081)
      aiex.dma_free_task(%1082)
      aiex.dma_free_task(%1083)
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
      %1152 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1152)
      %1153 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1153)
      %1154 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 266240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1154)
      %1155 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1155)
      %1156 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 532480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1156)
      %1157 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1157)
      %1158 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 798720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1158)
      %1159 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1159)
      %1160 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1064960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1160)
      %1161 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1161)
      %1162 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1331200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1162)
      %1163 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1163)
      %1164 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1597440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1164)
      %1165 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1165)
      %1166 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1863680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1166)
      %1167 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1167)
      %1168 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 265216 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1168)
      %1169 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 1321984 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1169)
      %1170 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 2378752 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1170)
      %1171 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 3435520 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1171)
      %1172 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 4492288 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1172)
      %1173 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 5549056 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1173)
      %1174 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 6605824 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1174)
      %1175 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 7662592 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1175)
      %1176 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2129920 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1176)
      %1177 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1177)
      %1178 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2396160 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1178)
      %1179 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1179)
      %1180 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2662400 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1180)
      %1181 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1181)
      %1182 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2928640 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1182)
      %1183 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1183)
      %1184 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3194880 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1184)
      %1185 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1185)
      %1186 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3461120 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1186)
      %1187 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1187)
      %1188 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3727360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1188)
      %1189 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1189)
      %1190 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3993600 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1190)
      %1191 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1191)
      %1192 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 8719360 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1192)
      %1193 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 9776128 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1193)
      %1194 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 10832896 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1194)
      %1195 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 11889664 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1195)
      %1196 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 12946432 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1196)
      %1197 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 14003200 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1197)
      %1198 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 15059968 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1198)
      %1199 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 16116736 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1199)
      aiex.dma_await_task(%1120)
      aiex.dma_await_task(%1121)
      aiex.dma_await_task(%1122)
      aiex.dma_await_task(%1123)
      aiex.dma_await_task(%1124)
      aiex.dma_await_task(%1125)
      aiex.dma_await_task(%1126)
      aiex.dma_await_task(%1127)
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
      aiex.dma_free_task(%1124)
      aiex.dma_free_task(%1125)
      aiex.dma_free_task(%1126)
      aiex.dma_free_task(%1127)
      aiex.dma_await_task(%1144)
      aiex.dma_await_task(%1145)
      aiex.dma_await_task(%1146)
      aiex.dma_await_task(%1147)
      aiex.dma_await_task(%1148)
      aiex.dma_await_task(%1149)
      aiex.dma_await_task(%1150)
      aiex.dma_await_task(%1151)
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
      aiex.dma_free_task(%1144)
      aiex.dma_free_task(%1145)
      aiex.dma_free_task(%1146)
      aiex.dma_free_task(%1147)
      aiex.dma_free_task(%1148)
      aiex.dma_free_task(%1149)
      aiex.dma_free_task(%1150)
      aiex.dma_free_task(%1151)
      %1200 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4259840 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1200)
      %1201 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1201)
      %1202 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4526080 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1202)
      %1203 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1203)
      %1204 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4792320 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1204)
      %1205 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1205)
      %1206 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5058560 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1206)
      %1207 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1207)
      %1208 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5324800 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1208)
      %1209 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1209)
      %1210 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5591040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1210)
      %1211 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1211)
      %1212 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5857280 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1212)
      %1213 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1213)
      %1214 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6123520 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1214)
      %1215 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1215)
      %1216 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 17173504 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1216)
      %1217 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 18230272 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1217)
      %1218 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 19287040 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1218)
      %1219 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 20343808 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1219)
      %1220 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 21400576 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1220)
      %1221 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 22457344 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1221)
      %1222 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 23514112 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1222)
      %1223 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 24570880 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1223)
      %1224 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6389760 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1224)
      %1225 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1225)
      %1226 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6656000 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1226)
      %1227 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1227)
      %1228 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6922240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1228)
      %1229 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1229)
      %1230 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7188480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1230)
      %1231 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1231)
      %1232 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7454720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1232)
      %1233 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1233)
      %1234 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7720960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1234)
      %1235 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1235)
      %1236 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7987200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1236)
      %1237 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1237)
      %1238 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 8253440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1238)
      %1239 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1239)
      %1240 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 25627648 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1240)
      %1241 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 26684416 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1241)
      %1242 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 27741184 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1242)
      %1243 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 28797952 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1243)
      %1244 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 29854720 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1244)
      %1245 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 30911488 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1245)
      %1246 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 31968256 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1246)
      %1247 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 33025024 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1247)
      aiex.dma_await_task(%1168)
      aiex.dma_await_task(%1169)
      aiex.dma_await_task(%1170)
      aiex.dma_await_task(%1171)
      aiex.dma_await_task(%1172)
      aiex.dma_await_task(%1173)
      aiex.dma_await_task(%1174)
      aiex.dma_await_task(%1175)
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
      aiex.dma_await_task(%1192)
      aiex.dma_await_task(%1193)
      aiex.dma_await_task(%1194)
      aiex.dma_await_task(%1195)
      aiex.dma_await_task(%1196)
      aiex.dma_await_task(%1197)
      aiex.dma_await_task(%1198)
      aiex.dma_await_task(%1199)
      aiex.dma_free_task(%1176)
      aiex.dma_free_task(%1177)
      aiex.dma_free_task(%1178)
      aiex.dma_free_task(%1179)
      aiex.dma_free_task(%1180)
      aiex.dma_free_task(%1181)
      aiex.dma_free_task(%1182)
      aiex.dma_free_task(%1183)
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
      %1248 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1248)
      %1249 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1249)
      %1250 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 266240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1250)
      %1251 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1251)
      %1252 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 532480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1252)
      %1253 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1253)
      %1254 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 798720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1254)
      %1255 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1255)
      %1256 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1064960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1256)
      %1257 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1257)
      %1258 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1331200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1258)
      %1259 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1259)
      %1260 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1597440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1260)
      %1261 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1261)
      %1262 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1863680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1262)
      %1263 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1263)
      %1264 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 265344 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1264)
      %1265 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 1322112 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1265)
      %1266 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 2378880 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1266)
      %1267 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 3435648 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1267)
      %1268 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 4492416 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1268)
      %1269 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 5549184 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1269)
      %1270 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 6605952 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1270)
      %1271 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 7662720 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1271)
      %1272 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2129920 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1272)
      %1273 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1273)
      %1274 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2396160 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1274)
      %1275 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1275)
      %1276 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2662400 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1276)
      %1277 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1277)
      %1278 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2928640 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1278)
      %1279 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1279)
      %1280 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3194880 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1280)
      %1281 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1281)
      %1282 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3461120 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1282)
      %1283 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1283)
      %1284 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3727360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1284)
      %1285 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1285)
      %1286 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3993600 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1286)
      %1287 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1287)
      %1288 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 8719488 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1288)
      %1289 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 9776256 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1289)
      %1290 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 10833024 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1290)
      %1291 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 11889792 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1291)
      %1292 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 12946560 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1292)
      %1293 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 14003328 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1293)
      %1294 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 15060096 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1294)
      %1295 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 16116864 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1295)
      aiex.dma_await_task(%1216)
      aiex.dma_await_task(%1217)
      aiex.dma_await_task(%1218)
      aiex.dma_await_task(%1219)
      aiex.dma_await_task(%1220)
      aiex.dma_await_task(%1221)
      aiex.dma_await_task(%1222)
      aiex.dma_await_task(%1223)
      aiex.dma_free_task(%1200)
      aiex.dma_free_task(%1201)
      aiex.dma_free_task(%1202)
      aiex.dma_free_task(%1203)
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
      aiex.dma_await_task(%1240)
      aiex.dma_await_task(%1241)
      aiex.dma_await_task(%1242)
      aiex.dma_await_task(%1243)
      aiex.dma_await_task(%1244)
      aiex.dma_await_task(%1245)
      aiex.dma_await_task(%1246)
      aiex.dma_await_task(%1247)
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
      aiex.dma_free_task(%1244)
      aiex.dma_free_task(%1245)
      aiex.dma_free_task(%1246)
      aiex.dma_free_task(%1247)
      %1296 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4259840 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1296)
      %1297 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1297)
      %1298 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4526080 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1298)
      %1299 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1299)
      %1300 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4792320 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1300)
      %1301 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1301)
      %1302 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5058560 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1302)
      %1303 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1303)
      %1304 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5324800 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1304)
      %1305 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1305)
      %1306 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5591040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1306)
      %1307 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1307)
      %1308 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5857280 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1308)
      %1309 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1309)
      %1310 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6123520 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1310)
      %1311 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1311)
      %1312 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 17173632 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1312)
      %1313 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 18230400 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1313)
      %1314 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 19287168 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1314)
      %1315 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 20343936 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1315)
      %1316 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 21400704 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1316)
      %1317 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 22457472 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1317)
      %1318 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 23514240 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1318)
      %1319 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 24571008 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1319)
      %1320 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6389760 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1320)
      %1321 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1321)
      %1322 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6656000 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1322)
      %1323 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1323)
      %1324 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6922240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1324)
      %1325 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1325)
      %1326 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7188480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1326)
      %1327 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1327)
      %1328 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7454720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1328)
      %1329 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1329)
      %1330 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7720960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1330)
      %1331 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1331)
      %1332 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7987200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1332)
      %1333 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1333)
      %1334 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 8253440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1334)
      %1335 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1335)
      %1336 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 25627776 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1336)
      %1337 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 26684544 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1337)
      %1338 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 27741312 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1338)
      %1339 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 28798080 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1339)
      %1340 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 29854848 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1340)
      %1341 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 30911616 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1341)
      %1342 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 31968384 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1342)
      %1343 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 33025152 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1343)
      aiex.dma_await_task(%1264)
      aiex.dma_await_task(%1265)
      aiex.dma_await_task(%1266)
      aiex.dma_await_task(%1267)
      aiex.dma_await_task(%1268)
      aiex.dma_await_task(%1269)
      aiex.dma_await_task(%1270)
      aiex.dma_await_task(%1271)
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
      aiex.dma_free_task(%1264)
      aiex.dma_free_task(%1265)
      aiex.dma_free_task(%1266)
      aiex.dma_free_task(%1267)
      aiex.dma_free_task(%1268)
      aiex.dma_free_task(%1269)
      aiex.dma_free_task(%1270)
      aiex.dma_free_task(%1271)
      aiex.dma_await_task(%1288)
      aiex.dma_await_task(%1289)
      aiex.dma_await_task(%1290)
      aiex.dma_await_task(%1291)
      aiex.dma_await_task(%1292)
      aiex.dma_await_task(%1293)
      aiex.dma_await_task(%1294)
      aiex.dma_await_task(%1295)
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
      %1344 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1344)
      %1345 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1345)
      %1346 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 266240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1346)
      %1347 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1347)
      %1348 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 532480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1348)
      %1349 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1349)
      %1350 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 798720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1350)
      %1351 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1351)
      %1352 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1064960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1352)
      %1353 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1353)
      %1354 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1331200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1354)
      %1355 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1355)
      %1356 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1597440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1356)
      %1357 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1357)
      %1358 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1863680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1358)
      %1359 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1359)
      %1360 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 265472 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1360)
      %1361 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 1322240 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1361)
      %1362 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 2379008 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1362)
      %1363 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 3435776 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1363)
      %1364 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 4492544 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1364)
      %1365 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 5549312 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1365)
      %1366 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 6606080 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1366)
      %1367 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 7662848 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1367)
      %1368 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2129920 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1368)
      %1369 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1369)
      %1370 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2396160 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1370)
      %1371 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1371)
      %1372 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2662400 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1372)
      %1373 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1373)
      %1374 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2928640 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1374)
      %1375 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1375)
      %1376 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3194880 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1376)
      %1377 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1377)
      %1378 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3461120 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1378)
      %1379 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1379)
      %1380 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3727360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1380)
      %1381 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1381)
      %1382 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3993600 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1382)
      %1383 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1383)
      %1384 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 8719616 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1384)
      %1385 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 9776384 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1385)
      %1386 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 10833152 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1386)
      %1387 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 11889920 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1387)
      %1388 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 12946688 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1388)
      %1389 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 14003456 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1389)
      %1390 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 15060224 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1390)
      %1391 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 16116992 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1391)
      aiex.dma_await_task(%1312)
      aiex.dma_await_task(%1313)
      aiex.dma_await_task(%1314)
      aiex.dma_await_task(%1315)
      aiex.dma_await_task(%1316)
      aiex.dma_await_task(%1317)
      aiex.dma_await_task(%1318)
      aiex.dma_await_task(%1319)
      aiex.dma_free_task(%1296)
      aiex.dma_free_task(%1297)
      aiex.dma_free_task(%1298)
      aiex.dma_free_task(%1299)
      aiex.dma_free_task(%1300)
      aiex.dma_free_task(%1301)
      aiex.dma_free_task(%1302)
      aiex.dma_free_task(%1303)
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
      aiex.dma_await_task(%1336)
      aiex.dma_await_task(%1337)
      aiex.dma_await_task(%1338)
      aiex.dma_await_task(%1339)
      aiex.dma_await_task(%1340)
      aiex.dma_await_task(%1341)
      aiex.dma_await_task(%1342)
      aiex.dma_await_task(%1343)
      aiex.dma_free_task(%1320)
      aiex.dma_free_task(%1321)
      aiex.dma_free_task(%1322)
      aiex.dma_free_task(%1323)
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
      %1392 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4259840 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1392)
      %1393 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1393)
      %1394 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4526080 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1394)
      %1395 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1395)
      %1396 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4792320 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1396)
      %1397 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1397)
      %1398 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5058560 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1398)
      %1399 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1399)
      %1400 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5324800 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1400)
      %1401 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1401)
      %1402 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5591040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1402)
      %1403 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1403)
      %1404 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5857280 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1404)
      %1405 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1405)
      %1406 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6123520 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1406)
      %1407 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1407)
      %1408 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 17173760 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1408)
      %1409 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 18230528 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1409)
      %1410 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 19287296 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1410)
      %1411 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 20344064 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1411)
      %1412 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 21400832 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1412)
      %1413 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 22457600 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1413)
      %1414 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 23514368 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1414)
      %1415 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 24571136 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1415)
      %1416 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6389760 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1416)
      %1417 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1417)
      %1418 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6656000 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1418)
      %1419 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1419)
      %1420 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6922240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1420)
      %1421 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1421)
      %1422 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7188480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1422)
      %1423 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1423)
      %1424 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7454720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1424)
      %1425 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1425)
      %1426 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7720960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1426)
      %1427 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1427)
      %1428 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7987200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1428)
      %1429 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1429)
      %1430 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 8253440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1430)
      %1431 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1431)
      %1432 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 25627904 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1432)
      %1433 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 26684672 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1433)
      %1434 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 27741440 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1434)
      %1435 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 28798208 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1435)
      %1436 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 29854976 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1436)
      %1437 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 30911744 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1437)
      %1438 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 31968512 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1438)
      %1439 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 33025280 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1439)
      aiex.dma_await_task(%1360)
      aiex.dma_await_task(%1361)
      aiex.dma_await_task(%1362)
      aiex.dma_await_task(%1363)
      aiex.dma_await_task(%1364)
      aiex.dma_await_task(%1365)
      aiex.dma_await_task(%1366)
      aiex.dma_await_task(%1367)
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
      aiex.dma_free_task(%1364)
      aiex.dma_free_task(%1365)
      aiex.dma_free_task(%1366)
      aiex.dma_free_task(%1367)
      aiex.dma_await_task(%1384)
      aiex.dma_await_task(%1385)
      aiex.dma_await_task(%1386)
      aiex.dma_await_task(%1387)
      aiex.dma_await_task(%1388)
      aiex.dma_await_task(%1389)
      aiex.dma_await_task(%1390)
      aiex.dma_await_task(%1391)
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
      aiex.dma_free_task(%1384)
      aiex.dma_free_task(%1385)
      aiex.dma_free_task(%1386)
      aiex.dma_free_task(%1387)
      aiex.dma_free_task(%1388)
      aiex.dma_free_task(%1389)
      aiex.dma_free_task(%1390)
      aiex.dma_free_task(%1391)
      %1440 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1440)
      %1441 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1441)
      %1442 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 266240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1442)
      %1443 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1443)
      %1444 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 532480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1444)
      %1445 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1445)
      %1446 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 798720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1446)
      %1447 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1447)
      %1448 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1064960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1448)
      %1449 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1449)
      %1450 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1331200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1450)
      %1451 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1451)
      %1452 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1597440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1452)
      %1453 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1453)
      %1454 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 1863680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1454)
      %1455 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1455)
      %1456 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 265600 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1456)
      %1457 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 1322368 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1457)
      %1458 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 2379136 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1458)
      %1459 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 3435904 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1459)
      %1460 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 4492672 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1460)
      %1461 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 5549440 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1461)
      %1462 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 6606208 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1462)
      %1463 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 7662976 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1463)
      %1464 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2129920 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1464)
      %1465 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1465)
      %1466 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2396160 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1466)
      %1467 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1467)
      %1468 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2662400 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1468)
      %1469 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1469)
      %1470 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 2928640 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1470)
      %1471 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1471)
      %1472 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3194880 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1472)
      %1473 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1473)
      %1474 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3461120 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1474)
      %1475 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1475)
      %1476 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3727360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1476)
      %1477 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1477)
      %1478 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 3993600 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1478)
      %1479 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1479)
      %1480 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 8719744 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1480)
      %1481 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 9776512 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1481)
      %1482 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 10833280 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1482)
      %1483 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 11890048 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1483)
      %1484 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 12946816 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1484)
      %1485 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 14003584 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1485)
      %1486 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 15060352 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1486)
      %1487 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 16117120 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1487)
      aiex.dma_await_task(%1408)
      aiex.dma_await_task(%1409)
      aiex.dma_await_task(%1410)
      aiex.dma_await_task(%1411)
      aiex.dma_await_task(%1412)
      aiex.dma_await_task(%1413)
      aiex.dma_await_task(%1414)
      aiex.dma_await_task(%1415)
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
      aiex.dma_await_task(%1432)
      aiex.dma_await_task(%1433)
      aiex.dma_await_task(%1434)
      aiex.dma_await_task(%1435)
      aiex.dma_await_task(%1436)
      aiex.dma_await_task(%1437)
      aiex.dma_await_task(%1438)
      aiex.dma_await_task(%1439)
      aiex.dma_free_task(%1416)
      aiex.dma_free_task(%1417)
      aiex.dma_free_task(%1418)
      aiex.dma_free_task(%1419)
      aiex.dma_free_task(%1420)
      aiex.dma_free_task(%1421)
      aiex.dma_free_task(%1422)
      aiex.dma_free_task(%1423)
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
      %1488 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4259840 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1488)
      %1489 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1489)
      %1490 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4526080 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1490)
      %1491 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1491)
      %1492 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 4792320 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1492)
      %1493 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1493)
      %1494 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5058560 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1494)
      %1495 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1495)
      %1496 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5324800 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1496)
      %1497 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1497)
      %1498 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5591040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1498)
      %1499 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1499)
      %1500 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 5857280 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1500)
      %1501 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1501)
      %1502 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6123520 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1502)
      %1503 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1503)
      %1504 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 17173888 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1504)
      %1505 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 18230656 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1505)
      %1506 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 19287424 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1506)
      %1507 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 20344192 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1507)
      %1508 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 21400960 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1508)
      %1509 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 22457728 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1509)
      %1510 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 23514496 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1510)
      %1511 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 24571264 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1511)
      %1512 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6389760 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1512)
      %1513 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1513)
      %1514 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6656000 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1514)
      %1515 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1515)
      %1516 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 6922240 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1516)
      %1517 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1517)
      %1518 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7188480 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1518)
      %1519 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1519)
      %1520 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7454720 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1520)
      %1521 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1521)
      %1522 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7720960 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1522)
      %1523 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1523)
      %1524 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 7987200 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1524)
      %1525 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1525)
      %1526 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<8652800xbf16> offset = 8253440 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 66560, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1526)
      %1527 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1527)
      %1528 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 25628032 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1528)
      %1529 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 26684800 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1529)
      %1530 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 27741568 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1530)
      %1531 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 28798336 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1531)
      %1532 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 29855104 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1532)
      %1533 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 30911872 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1533)
      %1534 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 31968640 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1534)
      %1535 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<34080768xbf16> offset = 33025408 len = 65536 sizes = [1, 4, 128, 128] strides = [0, 264192, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1535)
      aiex.dma_await_task(%1456)
      aiex.dma_await_task(%1457)
      aiex.dma_await_task(%1458)
      aiex.dma_await_task(%1459)
      aiex.dma_await_task(%1460)
      aiex.dma_await_task(%1461)
      aiex.dma_await_task(%1462)
      aiex.dma_await_task(%1463)
      aiex.dma_free_task(%1440)
      aiex.dma_free_task(%1441)
      aiex.dma_free_task(%1442)
      aiex.dma_free_task(%1443)
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
      aiex.dma_await_task(%1480)
      aiex.dma_await_task(%1481)
      aiex.dma_await_task(%1482)
      aiex.dma_await_task(%1483)
      aiex.dma_await_task(%1484)
      aiex.dma_await_task(%1485)
      aiex.dma_await_task(%1486)
      aiex.dma_await_task(%1487)
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
      aiex.dma_free_task(%1484)
      aiex.dma_free_task(%1485)
      aiex.dma_free_task(%1486)
      aiex.dma_free_task(%1487)
      aiex.dma_await_task(%1504)
      aiex.dma_await_task(%1505)
      aiex.dma_await_task(%1506)
      aiex.dma_await_task(%1507)
      aiex.dma_await_task(%1508)
      aiex.dma_await_task(%1509)
      aiex.dma_await_task(%1510)
      aiex.dma_await_task(%1511)
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
      aiex.dma_free_task(%1504)
      aiex.dma_free_task(%1505)
      aiex.dma_free_task(%1506)
      aiex.dma_free_task(%1507)
      aiex.dma_free_task(%1508)
      aiex.dma_free_task(%1509)
      aiex.dma_free_task(%1510)
      aiex.dma_free_task(%1511)
      aiex.dma_await_task(%1528)
      aiex.dma_await_task(%1529)
      aiex.dma_await_task(%1530)
      aiex.dma_await_task(%1531)
      aiex.dma_await_task(%1532)
      aiex.dma_await_task(%1533)
      aiex.dma_await_task(%1534)
      aiex.dma_await_task(%1535)
      aiex.dma_free_task(%1512)
      aiex.dma_free_task(%1513)
      aiex.dma_free_task(%1514)
      aiex.dma_free_task(%1515)
      aiex.dma_free_task(%1516)
      aiex.dma_free_task(%1517)
      aiex.dma_free_task(%1518)
      aiex.dma_free_task(%1519)
      aiex.dma_free_task(%1520)
      aiex.dma_free_task(%1521)
      aiex.dma_free_task(%1522)
      aiex.dma_free_task(%1523)
      aiex.dma_free_task(%1524)
      aiex.dma_free_task(%1525)
      aiex.dma_free_task(%1526)
      aiex.dma_free_task(%1527)
      aiex.dma_free_task(%1528)
      aiex.dma_free_task(%1529)
      aiex.dma_free_task(%1530)
      aiex.dma_free_task(%1531)
      aiex.dma_free_task(%1532)
      aiex.dma_free_task(%1533)
      aiex.dma_free_task(%1534)
      aiex.dma_free_task(%1535)
    }
  }
}

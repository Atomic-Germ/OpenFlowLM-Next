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
    aie.runtime_sequence @vae512_conv02(%arg0: memref<4528128xbf16>, %arg1: memref<1196032x!aiex.bfp<"v8bfp16ebs8">>, %arg2: memref<17305600xbf16>) {
      %c32_i32 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_0_0, 0, %c32_i32) : i32
      %c72_i32 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_0_0, 1, %c72_i32) : i32
      %c32_i32_77 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_0_1, 0, %c32_i32_77) : i32
      %c72_i32_78 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_0_1, 1, %c72_i32_78) : i32
      %c32_i32_79 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_0_2, 0, %c32_i32_79) : i32
      %c72_i32_80 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_0_2, 1, %c72_i32_80) : i32
      %c32_i32_81 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_0_3, 0, %c32_i32_81) : i32
      %c72_i32_82 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_0_3, 1, %c72_i32_82) : i32
      %c32_i32_83 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_1_0, 0, %c32_i32_83) : i32
      %c72_i32_84 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_1_0, 1, %c72_i32_84) : i32
      %c32_i32_85 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_1_1, 0, %c32_i32_85) : i32
      %c72_i32_86 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_1_1, 1, %c72_i32_86) : i32
      %c32_i32_87 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_1_2, 0, %c32_i32_87) : i32
      %c72_i32_88 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_1_2, 1, %c72_i32_88) : i32
      %c32_i32_89 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_1_3, 0, %c32_i32_89) : i32
      %c72_i32_90 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_1_3, 1, %c72_i32_90) : i32
      %c32_i32_91 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_2_0, 0, %c32_i32_91) : i32
      %c72_i32_92 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_2_0, 1, %c72_i32_92) : i32
      %c32_i32_93 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_2_1, 0, %c32_i32_93) : i32
      %c72_i32_94 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_2_1, 1, %c72_i32_94) : i32
      %c32_i32_95 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_2_2, 0, %c32_i32_95) : i32
      %c72_i32_96 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_2_2, 1, %c72_i32_96) : i32
      %c32_i32_97 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_2_3, 0, %c32_i32_97) : i32
      %c72_i32_98 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_2_3, 1, %c72_i32_98) : i32
      %c32_i32_99 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_3_0, 0, %c32_i32_99) : i32
      %c72_i32_100 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_3_0, 1, %c72_i32_100) : i32
      %c32_i32_101 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_3_1, 0, %c32_i32_101) : i32
      %c72_i32_102 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_3_1, 1, %c72_i32_102) : i32
      %c32_i32_103 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_3_2, 0, %c32_i32_103) : i32
      %c72_i32_104 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_3_2, 1, %c72_i32_104) : i32
      %c32_i32_105 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_3_3, 0, %c32_i32_105) : i32
      %c72_i32_106 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_3_3, 1, %c72_i32_106) : i32
      %c32_i32_107 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_4_0, 0, %c32_i32_107) : i32
      %c72_i32_108 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_4_0, 1, %c72_i32_108) : i32
      %c32_i32_109 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_4_1, 0, %c32_i32_109) : i32
      %c72_i32_110 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_4_1, 1, %c72_i32_110) : i32
      %c32_i32_111 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_4_2, 0, %c32_i32_111) : i32
      %c72_i32_112 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_4_2, 1, %c72_i32_112) : i32
      %c32_i32_113 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_4_3, 0, %c32_i32_113) : i32
      %c72_i32_114 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_4_3, 1, %c72_i32_114) : i32
      %c32_i32_115 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_5_0, 0, %c32_i32_115) : i32
      %c72_i32_116 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_5_0, 1, %c72_i32_116) : i32
      %c32_i32_117 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_5_1, 0, %c32_i32_117) : i32
      %c72_i32_118 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_5_1, 1, %c72_i32_118) : i32
      %c32_i32_119 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_5_2, 0, %c32_i32_119) : i32
      %c72_i32_120 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_5_2, 1, %c72_i32_120) : i32
      %c32_i32_121 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_5_3, 0, %c32_i32_121) : i32
      %c72_i32_122 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_5_3, 1, %c72_i32_122) : i32
      %c32_i32_123 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_6_0, 0, %c32_i32_123) : i32
      %c72_i32_124 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_6_0, 1, %c72_i32_124) : i32
      %c32_i32_125 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_6_1, 0, %c32_i32_125) : i32
      %c72_i32_126 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_6_1, 1, %c72_i32_126) : i32
      %c32_i32_127 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_6_2, 0, %c32_i32_127) : i32
      %c72_i32_128 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_6_2, 1, %c72_i32_128) : i32
      %c32_i32_129 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_6_3, 0, %c32_i32_129) : i32
      %c72_i32_130 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_6_3, 1, %c72_i32_130) : i32
      %c32_i32_131 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_7_0, 0, %c32_i32_131) : i32
      %c72_i32_132 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_7_0, 1, %c72_i32_132) : i32
      %c32_i32_133 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_7_1, 0, %c32_i32_133) : i32
      %c72_i32_134 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_7_1, 1, %c72_i32_134) : i32
      %c32_i32_135 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_7_2, 0, %c32_i32_135) : i32
      %c72_i32_136 = arith.constant 72 : i32
      aiex.npu.rtp_write(@rtp_7_2, 1, %c72_i32_136) : i32
      %c32_i32_137 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_7_3, 0, %c32_i32_137) : i32
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
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 270336 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 540672 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 811008 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1081344 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1351680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1622016 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1892352 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 133632 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 1198592 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 2263552 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 3328512 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 4393472 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 5458432 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 6523392 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 7588352 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2162688 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2433024 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2703360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2973696 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3244032 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3514368 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3784704 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 4055040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 0 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 8653312 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 9718272 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 10783232 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 11848192 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 12913152 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 13978112 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 15043072 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 16108032 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 270336 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%50)
      %51 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 540672 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 811008 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1081344 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1351680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1622016 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1892352 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%63)
      %64 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 133760 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 1198720 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 2263680 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 3328640 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%67)
      %68 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 4393600 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 5458560 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 6523520 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 7588480 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2162688 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2433024 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2703360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2973696 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3244032 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3514368 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3784704 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%84)
      %85 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 4055040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 74752 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 8653440 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 9718400 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 10783360 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 11848320 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 12913280 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 13978240 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 15043200 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 16108160 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
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
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%97)
      %98 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 270336 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 540672 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%100)
      %101 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%101)
      %102 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 811008 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%102)
      %103 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%103)
      %104 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1081344 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%104)
      %105 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%105)
      %106 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1351680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%106)
      %107 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%107)
      %108 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1622016 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%108)
      %109 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%109)
      %110 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1892352 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%110)
      %111 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%111)
      %112 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 133888 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%112)
      %113 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 1198848 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%113)
      %114 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 2263808 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%114)
      %115 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 3328768 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%115)
      %116 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 4393728 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%116)
      %117 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 5458688 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%117)
      %118 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 6523648 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%118)
      %119 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 7588608 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%119)
      %120 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2162688 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%120)
      %121 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%121)
      %122 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2433024 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%122)
      %123 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%123)
      %124 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2703360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%124)
      %125 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%125)
      %126 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2973696 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%126)
      %127 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%127)
      %128 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3244032 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%128)
      %129 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%129)
      %130 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3514368 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%130)
      %131 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%131)
      %132 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3784704 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%132)
      %133 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%133)
      %134 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 4055040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%134)
      %135 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 149504 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%135)
      %136 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 8653568 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%136)
      %137 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 9718528 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%137)
      %138 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 10783488 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%138)
      %139 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 11848448 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%139)
      %140 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 12913408 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%140)
      %141 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 13978368 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%141)
      %142 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 15043328 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%142)
      %143 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 16108288 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
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
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%144)
      %145 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%145)
      %146 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 270336 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%146)
      %147 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%147)
      %148 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 540672 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%148)
      %149 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%149)
      %150 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 811008 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%150)
      %151 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%151)
      %152 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1081344 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%152)
      %153 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%153)
      %154 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1351680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%154)
      %155 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%155)
      %156 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1622016 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%156)
      %157 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%157)
      %158 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1892352 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%158)
      %159 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%159)
      %160 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 134016 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%160)
      %161 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 1198976 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%161)
      %162 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 2263936 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%162)
      %163 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 3328896 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%163)
      %164 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 4393856 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%164)
      %165 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 5458816 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%165)
      %166 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 6523776 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%166)
      %167 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 7588736 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%167)
      %168 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2162688 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%168)
      %169 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%169)
      %170 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2433024 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%170)
      %171 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%171)
      %172 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2703360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%172)
      %173 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%173)
      %174 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2973696 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%174)
      %175 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%175)
      %176 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3244032 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%176)
      %177 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%177)
      %178 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3514368 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%178)
      %179 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%179)
      %180 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3784704 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%180)
      %181 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%181)
      %182 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 4055040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%182)
      %183 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 224256 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%183)
      %184 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 8653696 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%184)
      %185 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 9718656 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%185)
      %186 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 10783616 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%186)
      %187 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 11848576 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%187)
      %188 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 12913536 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%188)
      %189 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 13978496 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%189)
      %190 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 15043456 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%190)
      %191 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 16108416 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
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
      %192 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%192)
      %193 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%193)
      %194 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 270336 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%194)
      %195 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%195)
      %196 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 540672 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%196)
      %197 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%197)
      %198 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 811008 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%198)
      %199 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%199)
      %200 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1081344 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%200)
      %201 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%201)
      %202 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1351680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%202)
      %203 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%203)
      %204 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1622016 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%204)
      %205 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%205)
      %206 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1892352 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%206)
      %207 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%207)
      %208 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 134144 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%208)
      %209 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 1199104 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%209)
      %210 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 2264064 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%210)
      %211 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 3329024 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%211)
      %212 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 4393984 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%212)
      %213 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 5458944 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%213)
      %214 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 6523904 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%214)
      %215 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 7588864 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%215)
      %216 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2162688 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%216)
      %217 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%217)
      %218 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2433024 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%218)
      %219 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%219)
      %220 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2703360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%220)
      %221 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%221)
      %222 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2973696 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%222)
      %223 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%223)
      %224 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3244032 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%224)
      %225 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%225)
      %226 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3514368 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%226)
      %227 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%227)
      %228 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3784704 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%228)
      %229 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%229)
      %230 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 4055040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%230)
      %231 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 299008 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%231)
      %232 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 8653824 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%232)
      %233 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 9718784 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%233)
      %234 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 10783744 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%234)
      %235 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 11848704 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%235)
      %236 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 12913664 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%236)
      %237 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 13978624 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%237)
      %238 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 15043584 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%238)
      %239 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 16108544 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
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
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%240)
      %241 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%241)
      %242 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 270336 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%242)
      %243 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%243)
      %244 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 540672 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%244)
      %245 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%245)
      %246 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 811008 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%246)
      %247 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%247)
      %248 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1081344 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%248)
      %249 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%249)
      %250 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1351680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%250)
      %251 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%251)
      %252 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1622016 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%252)
      %253 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%253)
      %254 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1892352 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%254)
      %255 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%255)
      %256 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 134272 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%256)
      %257 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 1199232 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%257)
      %258 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 2264192 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%258)
      %259 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 3329152 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%259)
      %260 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 4394112 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%260)
      %261 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 5459072 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%261)
      %262 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 6524032 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%262)
      %263 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 7588992 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%263)
      %264 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2162688 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%264)
      %265 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%265)
      %266 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2433024 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%266)
      %267 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%267)
      %268 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2703360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%268)
      %269 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%269)
      %270 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2973696 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%270)
      %271 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%271)
      %272 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3244032 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%272)
      %273 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%273)
      %274 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3514368 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%274)
      %275 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%275)
      %276 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3784704 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%276)
      %277 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%277)
      %278 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 4055040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%278)
      %279 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 373760 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%279)
      %280 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 8653952 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%280)
      %281 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 9718912 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%281)
      %282 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 10783872 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%282)
      %283 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 11848832 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%283)
      %284 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 12913792 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%284)
      %285 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 13978752 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%285)
      %286 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 15043712 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%286)
      %287 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 16108672 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
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
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%288)
      %289 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%289)
      %290 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 270336 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%290)
      %291 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%291)
      %292 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 540672 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%292)
      %293 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%293)
      %294 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 811008 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%294)
      %295 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%295)
      %296 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1081344 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%296)
      %297 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%297)
      %298 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1351680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%298)
      %299 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%299)
      %300 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1622016 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%300)
      %301 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%301)
      %302 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1892352 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%302)
      %303 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%303)
      %304 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 134400 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%304)
      %305 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 1199360 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%305)
      %306 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 2264320 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%306)
      %307 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 3329280 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%307)
      %308 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 4394240 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%308)
      %309 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 5459200 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%309)
      %310 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 6524160 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%310)
      %311 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 7589120 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%311)
      %312 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2162688 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%312)
      %313 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%313)
      %314 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2433024 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%314)
      %315 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%315)
      %316 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2703360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%316)
      %317 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%317)
      %318 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2973696 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%318)
      %319 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%319)
      %320 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3244032 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%320)
      %321 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%321)
      %322 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3514368 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%322)
      %323 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%323)
      %324 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3784704 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%324)
      %325 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%325)
      %326 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 4055040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%326)
      %327 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 448512 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%327)
      %328 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 8654080 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%328)
      %329 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 9719040 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%329)
      %330 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 10784000 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%330)
      %331 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 11848960 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%331)
      %332 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 12913920 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%332)
      %333 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 13978880 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%333)
      %334 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 15043840 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%334)
      %335 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 16108800 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
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
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%336)
      %337 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%337)
      %338 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 270336 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%338)
      %339 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%339)
      %340 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 540672 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%340)
      %341 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%341)
      %342 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 811008 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%342)
      %343 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%343)
      %344 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1081344 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%344)
      %345 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%345)
      %346 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1351680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%346)
      %347 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%347)
      %348 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1622016 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%348)
      %349 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%349)
      %350 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1892352 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%350)
      %351 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%351)
      %352 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 134528 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%352)
      %353 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 1199488 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%353)
      %354 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 2264448 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%354)
      %355 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 3329408 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%355)
      %356 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 4394368 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%356)
      %357 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 5459328 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%357)
      %358 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 6524288 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%358)
      %359 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 7589248 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%359)
      %360 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2162688 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%360)
      %361 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%361)
      %362 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2433024 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%362)
      %363 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%363)
      %364 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2703360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%364)
      %365 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%365)
      %366 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2973696 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%366)
      %367 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%367)
      %368 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3244032 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%368)
      %369 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%369)
      %370 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3514368 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%370)
      %371 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%371)
      %372 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3784704 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%372)
      %373 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%373)
      %374 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 4055040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%374)
      %375 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 523264 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%375)
      %376 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 8654208 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%376)
      %377 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 9719168 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%377)
      %378 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 10784128 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%378)
      %379 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 11849088 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%379)
      %380 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 12914048 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%380)
      %381 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 13979008 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%381)
      %382 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 15043968 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%382)
      %383 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 16108928 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
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
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%384)
      %385 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%385)
      %386 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 270336 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%386)
      %387 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%387)
      %388 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 540672 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%388)
      %389 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%389)
      %390 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 811008 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%390)
      %391 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%391)
      %392 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1081344 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%392)
      %393 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%393)
      %394 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1351680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%394)
      %395 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%395)
      %396 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1622016 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%396)
      %397 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%397)
      %398 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1892352 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%398)
      %399 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%399)
      %400 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 266752 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%400)
      %401 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 1331712 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%401)
      %402 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 2396672 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%402)
      %403 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 3461632 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%403)
      %404 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 4526592 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%404)
      %405 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 5591552 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%405)
      %406 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 6656512 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%406)
      %407 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 7721472 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%407)
      %408 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2162688 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%408)
      %409 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%409)
      %410 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2433024 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%410)
      %411 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%411)
      %412 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2703360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%412)
      %413 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%413)
      %414 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2973696 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%414)
      %415 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%415)
      %416 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3244032 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%416)
      %417 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%417)
      %418 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3514368 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%418)
      %419 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%419)
      %420 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3784704 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%420)
      %421 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%421)
      %422 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 4055040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%422)
      %423 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 598016 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%423)
      %424 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 8786432 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%424)
      %425 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 9851392 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%425)
      %426 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 10916352 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%426)
      %427 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 11981312 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%427)
      %428 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 13046272 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%428)
      %429 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 14111232 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%429)
      %430 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 15176192 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%430)
      %431 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 16241152 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
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
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%432)
      %433 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%433)
      %434 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 270336 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%434)
      %435 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%435)
      %436 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 540672 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%436)
      %437 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%437)
      %438 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 811008 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%438)
      %439 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%439)
      %440 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1081344 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%440)
      %441 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%441)
      %442 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1351680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%442)
      %443 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%443)
      %444 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1622016 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%444)
      %445 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%445)
      %446 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1892352 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%446)
      %447 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%447)
      %448 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 266880 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%448)
      %449 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 1331840 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%449)
      %450 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 2396800 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%450)
      %451 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 3461760 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%451)
      %452 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 4526720 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%452)
      %453 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 5591680 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%453)
      %454 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 6656640 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%454)
      %455 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 7721600 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%455)
      %456 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2162688 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%456)
      %457 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%457)
      %458 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2433024 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%458)
      %459 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%459)
      %460 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2703360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%460)
      %461 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%461)
      %462 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2973696 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%462)
      %463 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%463)
      %464 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3244032 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%464)
      %465 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%465)
      %466 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3514368 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%466)
      %467 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%467)
      %468 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3784704 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%468)
      %469 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%469)
      %470 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 4055040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%470)
      %471 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 672768 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%471)
      %472 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 8786560 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%472)
      %473 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 9851520 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%473)
      %474 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 10916480 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%474)
      %475 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 11981440 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%475)
      %476 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 13046400 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%476)
      %477 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 14111360 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%477)
      %478 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 15176320 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%478)
      %479 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 16241280 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
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
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%480)
      %481 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%481)
      %482 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 270336 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%482)
      %483 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%483)
      %484 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 540672 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%484)
      %485 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%485)
      %486 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 811008 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%486)
      %487 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%487)
      %488 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1081344 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%488)
      %489 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%489)
      %490 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1351680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%490)
      %491 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%491)
      %492 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1622016 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%492)
      %493 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%493)
      %494 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1892352 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%494)
      %495 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%495)
      %496 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 267008 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%496)
      %497 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 1331968 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%497)
      %498 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 2396928 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%498)
      %499 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 3461888 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%499)
      %500 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 4526848 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%500)
      %501 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 5591808 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%501)
      %502 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 6656768 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%502)
      %503 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 7721728 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%503)
      %504 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2162688 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%504)
      %505 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%505)
      %506 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2433024 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%506)
      %507 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%507)
      %508 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2703360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%508)
      %509 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%509)
      %510 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2973696 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%510)
      %511 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%511)
      %512 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3244032 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%512)
      %513 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%513)
      %514 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3514368 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%514)
      %515 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%515)
      %516 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3784704 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%516)
      %517 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%517)
      %518 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 4055040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%518)
      %519 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 747520 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%519)
      %520 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 8786688 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%520)
      %521 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 9851648 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%521)
      %522 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 10916608 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%522)
      %523 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 11981568 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%523)
      %524 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 13046528 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%524)
      %525 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 14111488 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%525)
      %526 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 15176448 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%526)
      %527 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 16241408 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
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
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%528)
      %529 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%529)
      %530 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 270336 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%530)
      %531 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%531)
      %532 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 540672 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%532)
      %533 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%533)
      %534 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 811008 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%534)
      %535 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%535)
      %536 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1081344 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%536)
      %537 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%537)
      %538 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1351680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%538)
      %539 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%539)
      %540 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1622016 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%540)
      %541 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%541)
      %542 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1892352 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%542)
      %543 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%543)
      %544 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 267136 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%544)
      %545 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 1332096 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%545)
      %546 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 2397056 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%546)
      %547 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 3462016 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%547)
      %548 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 4526976 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%548)
      %549 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 5591936 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%549)
      %550 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 6656896 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%550)
      %551 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 7721856 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%551)
      %552 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2162688 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%552)
      %553 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%553)
      %554 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2433024 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%554)
      %555 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%555)
      %556 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2703360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%556)
      %557 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%557)
      %558 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2973696 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%558)
      %559 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%559)
      %560 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3244032 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%560)
      %561 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%561)
      %562 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3514368 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%562)
      %563 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%563)
      %564 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3784704 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%564)
      %565 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%565)
      %566 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 4055040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%566)
      %567 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 822272 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%567)
      %568 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 8786816 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%568)
      %569 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 9851776 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%569)
      %570 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 10916736 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%570)
      %571 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 11981696 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%571)
      %572 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 13046656 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%572)
      %573 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 14111616 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%573)
      %574 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 15176576 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%574)
      %575 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 16241536 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
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
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%576)
      %577 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%577)
      %578 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 270336 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%578)
      %579 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%579)
      %580 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 540672 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%580)
      %581 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%581)
      %582 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 811008 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%582)
      %583 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%583)
      %584 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1081344 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%584)
      %585 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%585)
      %586 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1351680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%586)
      %587 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%587)
      %588 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1622016 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%588)
      %589 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%589)
      %590 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1892352 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%590)
      %591 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%591)
      %592 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 267264 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%592)
      %593 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 1332224 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%593)
      %594 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 2397184 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%594)
      %595 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 3462144 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%595)
      %596 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 4527104 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%596)
      %597 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 5592064 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%597)
      %598 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 6657024 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%598)
      %599 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 7721984 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%599)
      %600 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2162688 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%600)
      %601 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%601)
      %602 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2433024 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%602)
      %603 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%603)
      %604 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2703360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%604)
      %605 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%605)
      %606 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2973696 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%606)
      %607 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%607)
      %608 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3244032 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%608)
      %609 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%609)
      %610 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3514368 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%610)
      %611 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%611)
      %612 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3784704 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%612)
      %613 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%613)
      %614 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 4055040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%614)
      %615 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 897024 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%615)
      %616 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 8786944 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%616)
      %617 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 9851904 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%617)
      %618 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 10916864 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%618)
      %619 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 11981824 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%619)
      %620 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 13046784 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%620)
      %621 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 14111744 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%621)
      %622 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 15176704 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%622)
      %623 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 16241664 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
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
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%624)
      %625 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%625)
      %626 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 270336 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%626)
      %627 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%627)
      %628 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 540672 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%628)
      %629 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%629)
      %630 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 811008 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%630)
      %631 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%631)
      %632 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1081344 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%632)
      %633 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%633)
      %634 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1351680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%634)
      %635 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%635)
      %636 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1622016 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%636)
      %637 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%637)
      %638 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1892352 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%638)
      %639 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%639)
      %640 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 267392 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%640)
      %641 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 1332352 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%641)
      %642 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 2397312 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%642)
      %643 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 3462272 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%643)
      %644 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 4527232 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%644)
      %645 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 5592192 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%645)
      %646 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 6657152 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%646)
      %647 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 7722112 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%647)
      %648 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2162688 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%648)
      %649 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%649)
      %650 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2433024 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%650)
      %651 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%651)
      %652 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2703360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%652)
      %653 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%653)
      %654 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2973696 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%654)
      %655 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%655)
      %656 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3244032 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%656)
      %657 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%657)
      %658 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3514368 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%658)
      %659 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%659)
      %660 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3784704 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%660)
      %661 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%661)
      %662 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 4055040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%662)
      %663 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 971776 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%663)
      %664 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 8787072 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%664)
      %665 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 9852032 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%665)
      %666 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 10916992 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%666)
      %667 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 11981952 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%667)
      %668 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 13046912 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%668)
      %669 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 14111872 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%669)
      %670 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 15176832 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%670)
      %671 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 16241792 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
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
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%672)
      %673 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%673)
      %674 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 270336 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%674)
      %675 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%675)
      %676 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 540672 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%676)
      %677 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%677)
      %678 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 811008 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%678)
      %679 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%679)
      %680 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1081344 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%680)
      %681 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%681)
      %682 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1351680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%682)
      %683 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%683)
      %684 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1622016 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%684)
      %685 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%685)
      %686 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1892352 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%686)
      %687 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%687)
      %688 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 267520 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%688)
      %689 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 1332480 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%689)
      %690 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 2397440 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%690)
      %691 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 3462400 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%691)
      %692 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 4527360 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%692)
      %693 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 5592320 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%693)
      %694 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 6657280 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%694)
      %695 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 7722240 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%695)
      %696 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2162688 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%696)
      %697 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%697)
      %698 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2433024 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%698)
      %699 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%699)
      %700 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2703360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%700)
      %701 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%701)
      %702 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2973696 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%702)
      %703 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%703)
      %704 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3244032 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%704)
      %705 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%705)
      %706 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3514368 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%706)
      %707 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%707)
      %708 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3784704 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%708)
      %709 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%709)
      %710 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 4055040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%710)
      %711 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1046528 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%711)
      %712 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 8787200 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%712)
      %713 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 9852160 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%713)
      %714 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 10917120 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%714)
      %715 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 11982080 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%715)
      %716 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 13047040 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%716)
      %717 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 14112000 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%717)
      %718 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 15176960 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%718)
      %719 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 16241920 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
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
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 0 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%720)
      %721 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%721)
      %722 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 270336 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%722)
      %723 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%723)
      %724 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 540672 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%724)
      %725 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%725)
      %726 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 811008 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%726)
      %727 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%727)
      %728 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1081344 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%728)
      %729 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%729)
      %730 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1351680 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%730)
      %731 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%731)
      %732 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1622016 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%732)
      %733 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%733)
      %734 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 1892352 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%734)
      %735 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%735)
      %736 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 267648 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%736)
      %737 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 1332608 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%737)
      %738 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 2397568 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%738)
      %739 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 3462528 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%739)
      %740 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 4527488 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%740)
      %741 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 5592448 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%741)
      %742 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 6657408 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%742)
      %743 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 7722368 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%743)
      %744 = aiex.dma_configure_task_for @A_0 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2162688 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%744)
      %745 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%745)
      %746 = aiex.dma_configure_task_for @A_1 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2433024 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%746)
      %747 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%747)
      %748 = aiex.dma_configure_task_for @A_2 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2703360 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%748)
      %749 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%749)
      %750 = aiex.dma_configure_task_for @A_3 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 2973696 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%750)
      %751 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%751)
      %752 = aiex.dma_configure_task_for @A_4 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3244032 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%752)
      %753 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%753)
      %754 = aiex.dma_configure_task_for @A_5 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3514368 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%754)
      %755 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%755)
      %756 = aiex.dma_configure_task_for @A_6 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 3784704 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%756)
      %757 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%757)
      %758 = aiex.dma_configure_task_for @A_7 {
        aie.dma_bd(%arg0 : memref<4528128xbf16> offset = 4055040 len = 49920 sizes = [8, 6, 130, 64] strides = [64, 67584, 512, 1])
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%758)
      %759 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<1196032x!aiex.bfp<"v8bfp16ebs8">> offset = 1121280 len = 74752 sizes = [1, 1, 1, 74752] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%759)
      %760 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 8787328 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%760)
      %761 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 9852288 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%761)
      %762 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 10917248 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%762)
      %763 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 11982208 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%763)
      %764 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 13047168 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%764)
      %765 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 14112128 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%765)
      %766 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 15177088 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%766)
      %767 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<17305600xbf16> offset = 16242048 len = 16384 sizes = [4, 2, 64, 128] strides = [266240, 66048, 1024, 1])
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
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
    }
  }
}

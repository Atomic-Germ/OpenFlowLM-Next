module {
  aie.device(npu2) @main {
    aie.runtime_sequence @cfg_vew() {
      aiex.configure @vew {
      }
    }
  }
  aie.device(npu2) @vew {
    %logical_core = aie.logical_tile<CoreTile>(0, 2)
    %logical_core_0 = aie.logical_tile<CoreTile>(0, 3)
    %logical_core_1 = aie.logical_tile<CoreTile>(1, 2)
    %logical_core_2 = aie.logical_tile<CoreTile>(1, 3)
    %logical_core_3 = aie.logical_tile<CoreTile>(2, 2)
    %logical_core_4 = aie.logical_tile<CoreTile>(2, 3)
    %logical_core_5 = aie.logical_tile<CoreTile>(3, 2)
    %logical_core_6 = aie.logical_tile<CoreTile>(3, 3)
    %logical_core_7 = aie.logical_tile<CoreTile>(4, 2)
    %logical_core_8 = aie.logical_tile<CoreTile>(4, 3)
    %logical_core_9 = aie.logical_tile<CoreTile>(5, 2)
    %logical_core_10 = aie.logical_tile<CoreTile>(5, 3)
    %logical_core_11 = aie.logical_tile<CoreTile>(6, 2)
    %logical_core_12 = aie.logical_tile<CoreTile>(6, 3)
    %logical_core_13 = aie.logical_tile<CoreTile>(7, 2)
    %logical_core_14 = aie.logical_tile<CoreTile>(7, 3)
    %logical_mem = aie.logical_tile<MemTile>(0, 1)
    %logical_mem_15 = aie.logical_tile<MemTile>(1, 1)
    %logical_mem_16 = aie.logical_tile<MemTile>(2, 1)
    %logical_mem_17 = aie.logical_tile<MemTile>(3, 1)
    %logical_mem_18 = aie.logical_tile<MemTile>(4, 1)
    %logical_mem_19 = aie.logical_tile<MemTile>(5, 1)
    %logical_mem_20 = aie.logical_tile<MemTile>(6, 1)
    %logical_mem_21 = aie.logical_tile<MemTile>(7, 1)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(0, 0)
    %logical_shim_noc_22 = aie.logical_tile<ShimNOCTile>(1, 0)
    %logical_shim_noc_23 = aie.logical_tile<ShimNOCTile>(2, 0)
    %logical_shim_noc_24 = aie.logical_tile<ShimNOCTile>(3, 0)
    %logical_shim_noc_25 = aie.logical_tile<ShimNOCTile>(4, 0)
    %logical_shim_noc_26 = aie.logical_tile<ShimNOCTile>(5, 0)
    %logical_shim_noc_27 = aie.logical_tile<ShimNOCTile>(6, 0)
    %logical_shim_noc_28 = aie.logical_tile<ShimNOCTile>(7, 0)
    %logical_mem_29 = aie.logical_tile<MemTile>(0, 1)
    %logical_mem_30 = aie.logical_tile<MemTile>(1, 1)
    %logical_mem_31 = aie.logical_tile<MemTile>(2, 1)
    %logical_mem_32 = aie.logical_tile<MemTile>(3, 1)
    %logical_mem_33 = aie.logical_tile<MemTile>(4, 1)
    %logical_mem_34 = aie.logical_tile<MemTile>(5, 1)
    %logical_mem_35 = aie.logical_tile<MemTile>(6, 1)
    %logical_mem_36 = aie.logical_tile<MemTile>(7, 1)
    %logical_shim_noc_37 = aie.logical_tile<ShimNOCTile>(0, 0)
    %logical_shim_noc_38 = aie.logical_tile<ShimNOCTile>(1, 0)
    %logical_shim_noc_39 = aie.logical_tile<ShimNOCTile>(2, 0)
    %logical_shim_noc_40 = aie.logical_tile<ShimNOCTile>(3, 0)
    %logical_shim_noc_41 = aie.logical_tile<ShimNOCTile>(4, 0)
    %logical_shim_noc_42 = aie.logical_tile<ShimNOCTile>(5, 0)
    %logical_shim_noc_43 = aie.logical_tile<ShimNOCTile>(6, 0)
    %logical_shim_noc_44 = aie.logical_tile<ShimNOCTile>(7, 0)
    %logical_mem_45 = aie.logical_tile<MemTile>(0, 1)
    %logical_mem_46 = aie.logical_tile<MemTile>(1, 1)
    %logical_mem_47 = aie.logical_tile<MemTile>(2, 1)
    %logical_mem_48 = aie.logical_tile<MemTile>(3, 1)
    %logical_mem_49 = aie.logical_tile<MemTile>(4, 1)
    %logical_mem_50 = aie.logical_tile<MemTile>(5, 1)
    %logical_mem_51 = aie.logical_tile<MemTile>(6, 1)
    %logical_mem_52 = aie.logical_tile<MemTile>(7, 1)
    %logical_shim_noc_53 = aie.logical_tile<ShimNOCTile>(0, 0)
    %logical_shim_noc_54 = aie.logical_tile<ShimNOCTile>(1, 0)
    %logical_shim_noc_55 = aie.logical_tile<ShimNOCTile>(2, 0)
    %logical_shim_noc_56 = aie.logical_tile<ShimNOCTile>(3, 0)
    %logical_shim_noc_57 = aie.logical_tile<ShimNOCTile>(4, 0)
    %logical_shim_noc_58 = aie.logical_tile<ShimNOCTile>(5, 0)
    %logical_shim_noc_59 = aie.logical_tile<ShimNOCTile>(6, 0)
    %logical_shim_noc_60 = aie.logical_tile<ShimNOCTile>(7, 0)
    %logical_mem_61 = aie.logical_tile<MemTile>(0, 1)
    %logical_mem_62 = aie.logical_tile<MemTile>(1, 1)
    %logical_mem_63 = aie.logical_tile<MemTile>(2, 1)
    %logical_mem_64 = aie.logical_tile<MemTile>(3, 1)
    %logical_mem_65 = aie.logical_tile<MemTile>(4, 1)
    %logical_mem_66 = aie.logical_tile<MemTile>(5, 1)
    %logical_mem_67 = aie.logical_tile<MemTile>(6, 1)
    %logical_mem_68 = aie.logical_tile<MemTile>(7, 1)
    %logical_shim_noc_69 = aie.logical_tile<ShimNOCTile>(0, 0)
    %logical_shim_noc_70 = aie.logical_tile<ShimNOCTile>(1, 0)
    %logical_shim_noc_71 = aie.logical_tile<ShimNOCTile>(2, 0)
    %logical_shim_noc_72 = aie.logical_tile<ShimNOCTile>(3, 0)
    %logical_shim_noc_73 = aie.logical_tile<ShimNOCTile>(4, 0)
    %logical_shim_noc_74 = aie.logical_tile<ShimNOCTile>(5, 0)
    %logical_shim_noc_75 = aie.logical_tile<ShimNOCTile>(6, 0)
    %logical_shim_noc_76 = aie.logical_tile<ShimNOCTile>(7, 0)
    aie.objectfifo @A_L2L1_0_0(%logical_mem, {%logical_core}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @A_L3L2_0(%logical_shim_noc, {%logical_mem}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo @A_L2L1_0_1(%logical_mem, {%logical_core_0}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@A_L3L2_0] -> [@A_L2L1_0_0, @A_L2L1_0_1]([] [0, 4096])
    aie.objectfifo @A_L2L1_1_0(%logical_mem_15, {%logical_core_1}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @A_L3L2_1(%logical_shim_noc_22, {%logical_mem_15}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo @A_L2L1_1_1(%logical_mem_15, {%logical_core_2}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@A_L3L2_1] -> [@A_L2L1_1_0, @A_L2L1_1_1]([] [0, 4096])
    aie.objectfifo @A_L2L1_2_0(%logical_mem_16, {%logical_core_3}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @A_L3L2_2(%logical_shim_noc_23, {%logical_mem_16}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo @A_L2L1_2_1(%logical_mem_16, {%logical_core_4}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@A_L3L2_2] -> [@A_L2L1_2_0, @A_L2L1_2_1]([] [0, 4096])
    aie.objectfifo @A_L2L1_3_0(%logical_mem_17, {%logical_core_5}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @A_L3L2_3(%logical_shim_noc_24, {%logical_mem_17}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo @A_L2L1_3_1(%logical_mem_17, {%logical_core_6}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@A_L3L2_3] -> [@A_L2L1_3_0, @A_L2L1_3_1]([] [0, 4096])
    aie.objectfifo @A_L2L1_4_0(%logical_mem_18, {%logical_core_7}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @A_L3L2_4(%logical_shim_noc_25, {%logical_mem_18}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo @A_L2L1_4_1(%logical_mem_18, {%logical_core_8}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@A_L3L2_4] -> [@A_L2L1_4_0, @A_L2L1_4_1]([] [0, 4096])
    aie.objectfifo @A_L2L1_5_0(%logical_mem_19, {%logical_core_9}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @A_L3L2_5(%logical_shim_noc_26, {%logical_mem_19}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo @A_L2L1_5_1(%logical_mem_19, {%logical_core_10}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@A_L3L2_5] -> [@A_L2L1_5_0, @A_L2L1_5_1]([] [0, 4096])
    aie.objectfifo @A_L2L1_6_0(%logical_mem_20, {%logical_core_11}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @A_L3L2_6(%logical_shim_noc_27, {%logical_mem_20}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo @A_L2L1_6_1(%logical_mem_20, {%logical_core_12}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@A_L3L2_6] -> [@A_L2L1_6_0, @A_L2L1_6_1]([] [0, 4096])
    aie.objectfifo @A_L2L1_7_0(%logical_mem_21, {%logical_core_13}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @A_L3L2_7(%logical_shim_noc_28, {%logical_mem_21}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo @A_L2L1_7_1(%logical_mem_21, {%logical_core_14}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@A_L3L2_7] -> [@A_L2L1_7_0, @A_L2L1_7_1]([] [0, 4096])
    aie.objectfifo @B_L2L1_0_0(%logical_mem_29, {%logical_core}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @B_L3L2_0(%logical_shim_noc_37, {%logical_mem_29}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo @B_L2L1_0_1(%logical_mem_29, {%logical_core_0}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@B_L3L2_0] -> [@B_L2L1_0_0, @B_L2L1_0_1]([] [0, 4096])
    aie.objectfifo @B_L2L1_1_0(%logical_mem_30, {%logical_core_1}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @B_L3L2_1(%logical_shim_noc_38, {%logical_mem_30}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo @B_L2L1_1_1(%logical_mem_30, {%logical_core_2}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@B_L3L2_1] -> [@B_L2L1_1_0, @B_L2L1_1_1]([] [0, 4096])
    aie.objectfifo @B_L2L1_2_0(%logical_mem_31, {%logical_core_3}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @B_L3L2_2(%logical_shim_noc_39, {%logical_mem_31}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo @B_L2L1_2_1(%logical_mem_31, {%logical_core_4}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@B_L3L2_2] -> [@B_L2L1_2_0, @B_L2L1_2_1]([] [0, 4096])
    aie.objectfifo @B_L2L1_3_0(%logical_mem_32, {%logical_core_5}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @B_L3L2_3(%logical_shim_noc_40, {%logical_mem_32}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo @B_L2L1_3_1(%logical_mem_32, {%logical_core_6}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@B_L3L2_3] -> [@B_L2L1_3_0, @B_L2L1_3_1]([] [0, 4096])
    aie.objectfifo @B_L2L1_4_0(%logical_mem_33, {%logical_core_7}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @B_L3L2_4(%logical_shim_noc_41, {%logical_mem_33}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo @B_L2L1_4_1(%logical_mem_33, {%logical_core_8}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@B_L3L2_4] -> [@B_L2L1_4_0, @B_L2L1_4_1]([] [0, 4096])
    aie.objectfifo @B_L2L1_5_0(%logical_mem_34, {%logical_core_9}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @B_L3L2_5(%logical_shim_noc_42, {%logical_mem_34}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo @B_L2L1_5_1(%logical_mem_34, {%logical_core_10}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@B_L3L2_5] -> [@B_L2L1_5_0, @B_L2L1_5_1]([] [0, 4096])
    aie.objectfifo @B_L2L1_6_0(%logical_mem_35, {%logical_core_11}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @B_L3L2_6(%logical_shim_noc_43, {%logical_mem_35}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo @B_L2L1_6_1(%logical_mem_35, {%logical_core_12}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@B_L3L2_6] -> [@B_L2L1_6_0, @B_L2L1_6_1]([] [0, 4096])
    aie.objectfifo @B_L2L1_7_0(%logical_mem_36, {%logical_core_13}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @B_L3L2_7(%logical_shim_noc_44, {%logical_mem_36}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo @B_L2L1_7_1(%logical_mem_36, {%logical_core_14}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo.link [@B_L3L2_7] -> [@B_L2L1_7_0, @B_L2L1_7_1]([] [0, 4096])
    aie.objectfifo @Y_L1L2_0_0(%logical_core, {%logical_mem_45}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Y_L1L2_0_1(%logical_core_0, {%logical_mem_45}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Y_L2L3_0(%logical_mem_45, {%logical_shim_noc_53}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo.link [@Y_L1L2_0_0, @Y_L1L2_0_1] -> [@Y_L2L3_0]([0, 4096] [])
    aie.objectfifo @Y_L1L2_1_0(%logical_core_1, {%logical_mem_46}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Y_L1L2_1_1(%logical_core_2, {%logical_mem_46}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Y_L2L3_1(%logical_mem_46, {%logical_shim_noc_54}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo.link [@Y_L1L2_1_0, @Y_L1L2_1_1] -> [@Y_L2L3_1]([0, 4096] [])
    aie.objectfifo @Y_L1L2_2_0(%logical_core_3, {%logical_mem_47}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Y_L1L2_2_1(%logical_core_4, {%logical_mem_47}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Y_L2L3_2(%logical_mem_47, {%logical_shim_noc_55}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo.link [@Y_L1L2_2_0, @Y_L1L2_2_1] -> [@Y_L2L3_2]([0, 4096] [])
    aie.objectfifo @Y_L1L2_3_0(%logical_core_5, {%logical_mem_48}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Y_L1L2_3_1(%logical_core_6, {%logical_mem_48}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Y_L2L3_3(%logical_mem_48, {%logical_shim_noc_56}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo.link [@Y_L1L2_3_0, @Y_L1L2_3_1] -> [@Y_L2L3_3]([0, 4096] [])
    aie.objectfifo @Y_L1L2_4_0(%logical_core_7, {%logical_mem_49}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Y_L1L2_4_1(%logical_core_8, {%logical_mem_49}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Y_L2L3_4(%logical_mem_49, {%logical_shim_noc_57}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo.link [@Y_L1L2_4_0, @Y_L1L2_4_1] -> [@Y_L2L3_4]([0, 4096] [])
    aie.objectfifo @Y_L1L2_5_0(%logical_core_9, {%logical_mem_50}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Y_L1L2_5_1(%logical_core_10, {%logical_mem_50}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Y_L2L3_5(%logical_mem_50, {%logical_shim_noc_58}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo.link [@Y_L1L2_5_0, @Y_L1L2_5_1] -> [@Y_L2L3_5]([0, 4096] [])
    aie.objectfifo @Y_L1L2_6_0(%logical_core_11, {%logical_mem_51}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Y_L1L2_6_1(%logical_core_12, {%logical_mem_51}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Y_L2L3_6(%logical_mem_51, {%logical_shim_noc_59}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo.link [@Y_L1L2_6_0, @Y_L1L2_6_1] -> [@Y_L2L3_6]([0, 4096] [])
    aie.objectfifo @Y_L1L2_7_0(%logical_core_13, {%logical_mem_52}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Y_L1L2_7_1(%logical_core_14, {%logical_mem_52}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Y_L2L3_7(%logical_mem_52, {%logical_shim_noc_60}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo.link [@Y_L1L2_7_0, @Y_L1L2_7_1] -> [@Y_L2L3_7]([0, 4096] [])
    aie.objectfifo @Z_L1L2_0_0(%logical_core, {%logical_mem_61}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Z_L1L2_0_1(%logical_core_0, {%logical_mem_61}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Z_L2L3_0(%logical_mem_61, {%logical_shim_noc_69}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo.link [@Z_L1L2_0_0, @Z_L1L2_0_1] -> [@Z_L2L3_0]([0, 4096] [])
    aie.objectfifo @Z_L1L2_1_0(%logical_core_1, {%logical_mem_62}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Z_L1L2_1_1(%logical_core_2, {%logical_mem_62}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Z_L2L3_1(%logical_mem_62, {%logical_shim_noc_70}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo.link [@Z_L1L2_1_0, @Z_L1L2_1_1] -> [@Z_L2L3_1]([0, 4096] [])
    aie.objectfifo @Z_L1L2_2_0(%logical_core_3, {%logical_mem_63}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Z_L1L2_2_1(%logical_core_4, {%logical_mem_63}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Z_L2L3_2(%logical_mem_63, {%logical_shim_noc_71}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo.link [@Z_L1L2_2_0, @Z_L1L2_2_1] -> [@Z_L2L3_2]([0, 4096] [])
    aie.objectfifo @Z_L1L2_3_0(%logical_core_5, {%logical_mem_64}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Z_L1L2_3_1(%logical_core_6, {%logical_mem_64}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Z_L2L3_3(%logical_mem_64, {%logical_shim_noc_72}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo.link [@Z_L1L2_3_0, @Z_L1L2_3_1] -> [@Z_L2L3_3]([0, 4096] [])
    aie.objectfifo @Z_L1L2_4_0(%logical_core_7, {%logical_mem_65}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Z_L1L2_4_1(%logical_core_8, {%logical_mem_65}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Z_L2L3_4(%logical_mem_65, {%logical_shim_noc_73}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo.link [@Z_L1L2_4_0, @Z_L1L2_4_1] -> [@Z_L2L3_4]([0, 4096] [])
    aie.objectfifo @Z_L1L2_5_0(%logical_core_9, {%logical_mem_66}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Z_L1L2_5_1(%logical_core_10, {%logical_mem_66}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Z_L2L3_5(%logical_mem_66, {%logical_shim_noc_74}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo.link [@Z_L1L2_5_0, @Z_L1L2_5_1] -> [@Z_L2L3_5]([0, 4096] [])
    aie.objectfifo @Z_L1L2_6_0(%logical_core_11, {%logical_mem_67}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Z_L1L2_6_1(%logical_core_12, {%logical_mem_67}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Z_L2L3_6(%logical_mem_67, {%logical_shim_noc_75}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo.link [@Z_L1L2_6_0, @Z_L1L2_6_1] -> [@Z_L2L3_6]([0, 4096] [])
    aie.objectfifo @Z_L1L2_7_0(%logical_core_13, {%logical_mem_68}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Z_L1L2_7_1(%logical_core_14, {%logical_mem_68}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @Z_L2L3_7(%logical_mem_68, {%logical_shim_noc_76}, 2 : i32) : !aie.objectfifo<memref<8192xbf16>>  
    aie.objectfifo.link [@Z_L1L2_7_0, @Z_L1L2_7_1] -> [@Z_L2L3_7]([0, 4096] [])
    func.func private @vew_begin(memref<2624xf32>, memref<12xi32>) attributes {link_with = "vew_begin.o"}
    func.func private @vew_param(memref<4096xbf16>, memref<2624xf32>, i32, i32, memref<12xi32>) attributes {link_with = "vew_begin.o"}
    func.func private @vew_stats(memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) attributes {link_with = "vew_begin.o"}
    func.func private @vew_stats_out(memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) attributes {link_with = "vew_begin.o"}
    func.func private @vew_apply(memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) attributes {link_with = "vew_begin.o"}
    func.func private @vew_add(memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) attributes {link_with = "vew_begin.o"}
    func.func private @vew_rgba(memref<4096xbf16>, memref<4096xbf16>, memref<12xi32>) attributes {link_with = "vew_begin.o"}
    %par_0_0 = aie.buffer(%logical_core) {sym_name = "par_0_0"} : memref<2624xf32> 
    %rtp_0_0 = aie.buffer(%logical_core) {sym_name = "rtp_0_0"} : memref<12xi32> = dense<0>
    %par_0_1 = aie.buffer(%logical_core_0) {sym_name = "par_0_1"} : memref<2624xf32> 
    %rtp_0_1 = aie.buffer(%logical_core_0) {sym_name = "rtp_0_1"} : memref<12xi32> = dense<0>
    %par_1_0 = aie.buffer(%logical_core_1) {sym_name = "par_1_0"} : memref<2624xf32> 
    %rtp_1_0 = aie.buffer(%logical_core_1) {sym_name = "rtp_1_0"} : memref<12xi32> = dense<0>
    %par_1_1 = aie.buffer(%logical_core_2) {sym_name = "par_1_1"} : memref<2624xf32> 
    %rtp_1_1 = aie.buffer(%logical_core_2) {sym_name = "rtp_1_1"} : memref<12xi32> = dense<0>
    %par_2_0 = aie.buffer(%logical_core_3) {sym_name = "par_2_0"} : memref<2624xf32> 
    %rtp_2_0 = aie.buffer(%logical_core_3) {sym_name = "rtp_2_0"} : memref<12xi32> = dense<0>
    %par_2_1 = aie.buffer(%logical_core_4) {sym_name = "par_2_1"} : memref<2624xf32> 
    %rtp_2_1 = aie.buffer(%logical_core_4) {sym_name = "rtp_2_1"} : memref<12xi32> = dense<0>
    %par_3_0 = aie.buffer(%logical_core_5) {sym_name = "par_3_0"} : memref<2624xf32> 
    %rtp_3_0 = aie.buffer(%logical_core_5) {sym_name = "rtp_3_0"} : memref<12xi32> = dense<0>
    %par_3_1 = aie.buffer(%logical_core_6) {sym_name = "par_3_1"} : memref<2624xf32> 
    %rtp_3_1 = aie.buffer(%logical_core_6) {sym_name = "rtp_3_1"} : memref<12xi32> = dense<0>
    %par_4_0 = aie.buffer(%logical_core_7) {sym_name = "par_4_0"} : memref<2624xf32> 
    %rtp_4_0 = aie.buffer(%logical_core_7) {sym_name = "rtp_4_0"} : memref<12xi32> = dense<0>
    %par_4_1 = aie.buffer(%logical_core_8) {sym_name = "par_4_1"} : memref<2624xf32> 
    %rtp_4_1 = aie.buffer(%logical_core_8) {sym_name = "rtp_4_1"} : memref<12xi32> = dense<0>
    %par_5_0 = aie.buffer(%logical_core_9) {sym_name = "par_5_0"} : memref<2624xf32> 
    %rtp_5_0 = aie.buffer(%logical_core_9) {sym_name = "rtp_5_0"} : memref<12xi32> = dense<0>
    %par_5_1 = aie.buffer(%logical_core_10) {sym_name = "par_5_1"} : memref<2624xf32> 
    %rtp_5_1 = aie.buffer(%logical_core_10) {sym_name = "rtp_5_1"} : memref<12xi32> = dense<0>
    %par_6_0 = aie.buffer(%logical_core_11) {sym_name = "par_6_0"} : memref<2624xf32> 
    %rtp_6_0 = aie.buffer(%logical_core_11) {sym_name = "rtp_6_0"} : memref<12xi32> = dense<0>
    %par_6_1 = aie.buffer(%logical_core_12) {sym_name = "par_6_1"} : memref<2624xf32> 
    %rtp_6_1 = aie.buffer(%logical_core_12) {sym_name = "rtp_6_1"} : memref<12xi32> = dense<0>
    %par_7_0 = aie.buffer(%logical_core_13) {sym_name = "par_7_0"} : memref<2624xf32> 
    %rtp_7_0 = aie.buffer(%logical_core_13) {sym_name = "rtp_7_0"} : memref<12xi32> = dense<0>
    %par_7_1 = aie.buffer(%logical_core_14) {sym_name = "par_7_1"} : memref<2624xf32> 
    %rtp_7_1 = aie.buffer(%logical_core_14) {sym_name = "rtp_7_1"} : memref<12xi32> = dense<0>
    %lock_0_2 = aie.lock(%logical_core)
    %core_0_2 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_0_2, Acquire, %c1_i32)
        func.call @vew_begin(%par_0_0, %rtp_0_0) : (memref<2624xf32>, memref<12xi32>) -> ()
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_0_0[%c0_77] : memref<12xi32>
        %c0_78 = arith.constant 0 : index
        %1 = arith.index_cast %0 : i32 to index
        %c1_79 = arith.constant 1 : index
        scf.for %arg1 = %c0_78 to %1 step %c1_79 {
          %14 = aie.objectfifo.acquire @A_L2L1_0_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = arith.index_cast %arg1 : index to i32
          %c-1_i32 = arith.constant -1 : i32
          func.call @vew_param(%15, %par_0_0, %16, %c-1_i32, %rtp_0_0) : (memref<4096xbf16>, memref<2624xf32>, i32, i32, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_0_0(Consume, 1)
        }
        %c1_80 = arith.constant 1 : index
        %2 = memref.load %rtp_0_0[%c1_80] : memref<12xi32>
        %c0_81 = arith.constant 0 : index
        %3 = arith.index_cast %2 : i32 to index
        %c1_82 = arith.constant 1 : index
        scf.for %arg1 = %c0_81 to %3 step %c1_82 {
          %14 = aie.objectfifo.acquire @A_L2L1_0_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats(%15, %par_0_0, %rtp_0_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_0_0(Consume, 1)
        }
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp_0_0[%c2] : memref<12xi32>
        %c0_83 = arith.constant 0 : index
        %5 = arith.index_cast %4 : i32 to index
        %c1_84 = arith.constant 1 : index
        scf.for %arg1 = %c0_83 to %5 step %c1_84 {
          %14 = aie.objectfifo.acquire @Y_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_0_0, %rtp_0_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Y_L1L2_0_0(Produce, 1)
        }
        %c3 = arith.constant 3 : index
        %6 = memref.load %rtp_0_0[%c3] : memref<12xi32>
        %c0_85 = arith.constant 0 : index
        %7 = arith.index_cast %6 : i32 to index
        %c1_86 = arith.constant 1 : index
        scf.for %arg1 = %c0_85 to %7 step %c1_86 {
          %14 = aie.objectfifo.acquire @A_L2L1_0_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_apply(%15, %17, %par_0_0, %rtp_0_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_0_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_0_0(Produce, 1)
        }
        %c4 = arith.constant 4 : index
        %8 = memref.load %rtp_0_0[%c4] : memref<12xi32>
        %c0_87 = arith.constant 0 : index
        %9 = arith.index_cast %8 : i32 to index
        %c1_88 = arith.constant 1 : index
        scf.for %arg1 = %c0_87 to %9 step %c1_88 {
          %14 = aie.objectfifo.acquire @A_L2L1_0_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @B_L2L1_0_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %18 = aie.objectfifo.acquire @Y_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_add(%15, %17, %19, %par_0_0, %rtp_0_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_0_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_0_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_0_0(Produce, 1)
        }
        %c5 = arith.constant 5 : index
        %10 = memref.load %rtp_0_0[%c5] : memref<12xi32>
        %c0_89 = arith.constant 0 : index
        %11 = arith.index_cast %10 : i32 to index
        %c1_90 = arith.constant 1 : index
        scf.for %arg1 = %c0_89 to %11 step %c1_90 {
          %14 = aie.objectfifo.acquire @Z_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_0_0, %rtp_0_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Z_L1L2_0_0(Produce, 1)
        }
        %c6 = arith.constant 6 : index
        %12 = memref.load %rtp_0_0[%c6] : memref<12xi32>
        %c0_91 = arith.constant 0 : index
        %13 = arith.index_cast %12 : i32 to index
        %c1_92 = arith.constant 1 : index
        scf.for %arg1 = %c0_91 to %13 step %c1_92 {
          %14 = aie.objectfifo.acquire @A_L2L1_0_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_rgba(%15, %17, %rtp_0_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_0_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_0_0(Produce, 1)
        }
        %c1_i32_93 = arith.constant 1 : i32
        aie.use_lock(%lock_0_2, Release, %c1_i32_93)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %lock_0_3 = aie.lock(%logical_core_0)
    %core_0_3 = aie.core(%logical_core_0) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_0_3, Acquire, %c1_i32)
        func.call @vew_begin(%par_0_1, %rtp_0_1) : (memref<2624xf32>, memref<12xi32>) -> ()
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_0_1[%c0_77] : memref<12xi32>
        %c0_78 = arith.constant 0 : index
        %1 = arith.index_cast %0 : i32 to index
        %c1_79 = arith.constant 1 : index
        scf.for %arg1 = %c0_78 to %1 step %c1_79 {
          %14 = aie.objectfifo.acquire @A_L2L1_0_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = arith.index_cast %arg1 : index to i32
          %c0_i32 = arith.constant 0 : i32
          func.call @vew_param(%15, %par_0_1, %16, %c0_i32, %rtp_0_1) : (memref<4096xbf16>, memref<2624xf32>, i32, i32, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_0_1(Consume, 1)
        }
        %c1_80 = arith.constant 1 : index
        %2 = memref.load %rtp_0_1[%c1_80] : memref<12xi32>
        %c0_81 = arith.constant 0 : index
        %3 = arith.index_cast %2 : i32 to index
        %c1_82 = arith.constant 1 : index
        scf.for %arg1 = %c0_81 to %3 step %c1_82 {
          %14 = aie.objectfifo.acquire @A_L2L1_0_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats(%15, %par_0_1, %rtp_0_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_0_1(Consume, 1)
        }
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp_0_1[%c2] : memref<12xi32>
        %c0_83 = arith.constant 0 : index
        %5 = arith.index_cast %4 : i32 to index
        %c1_84 = arith.constant 1 : index
        scf.for %arg1 = %c0_83 to %5 step %c1_84 {
          %14 = aie.objectfifo.acquire @Y_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_0_1, %rtp_0_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Y_L1L2_0_1(Produce, 1)
        }
        %c3 = arith.constant 3 : index
        %6 = memref.load %rtp_0_1[%c3] : memref<12xi32>
        %c0_85 = arith.constant 0 : index
        %7 = arith.index_cast %6 : i32 to index
        %c1_86 = arith.constant 1 : index
        scf.for %arg1 = %c0_85 to %7 step %c1_86 {
          %14 = aie.objectfifo.acquire @A_L2L1_0_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_apply(%15, %17, %par_0_1, %rtp_0_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_0_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_0_1(Produce, 1)
        }
        %c4 = arith.constant 4 : index
        %8 = memref.load %rtp_0_1[%c4] : memref<12xi32>
        %c0_87 = arith.constant 0 : index
        %9 = arith.index_cast %8 : i32 to index
        %c1_88 = arith.constant 1 : index
        scf.for %arg1 = %c0_87 to %9 step %c1_88 {
          %14 = aie.objectfifo.acquire @A_L2L1_0_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @B_L2L1_0_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %18 = aie.objectfifo.acquire @Y_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_add(%15, %17, %19, %par_0_1, %rtp_0_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_0_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_0_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_0_1(Produce, 1)
        }
        %c5 = arith.constant 5 : index
        %10 = memref.load %rtp_0_1[%c5] : memref<12xi32>
        %c0_89 = arith.constant 0 : index
        %11 = arith.index_cast %10 : i32 to index
        %c1_90 = arith.constant 1 : index
        scf.for %arg1 = %c0_89 to %11 step %c1_90 {
          %14 = aie.objectfifo.acquire @Z_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_0_1, %rtp_0_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Z_L1L2_0_1(Produce, 1)
        }
        %c6 = arith.constant 6 : index
        %12 = memref.load %rtp_0_1[%c6] : memref<12xi32>
        %c0_91 = arith.constant 0 : index
        %13 = arith.index_cast %12 : i32 to index
        %c1_92 = arith.constant 1 : index
        scf.for %arg1 = %c0_91 to %13 step %c1_92 {
          %14 = aie.objectfifo.acquire @A_L2L1_0_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_rgba(%15, %17, %rtp_0_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_0_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_0_1(Produce, 1)
        }
        %c1_i32_93 = arith.constant 1 : i32
        aie.use_lock(%lock_0_3, Release, %c1_i32_93)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %lock_1_2 = aie.lock(%logical_core_1)
    %core_1_2 = aie.core(%logical_core_1) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_1_2, Acquire, %c1_i32)
        func.call @vew_begin(%par_1_0, %rtp_1_0) : (memref<2624xf32>, memref<12xi32>) -> ()
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_1_0[%c0_77] : memref<12xi32>
        %c0_78 = arith.constant 0 : index
        %1 = arith.index_cast %0 : i32 to index
        %c1_79 = arith.constant 1 : index
        scf.for %arg1 = %c0_78 to %1 step %c1_79 {
          %14 = aie.objectfifo.acquire @A_L2L1_1_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = arith.index_cast %arg1 : index to i32
          %c-1_i32 = arith.constant -1 : i32
          func.call @vew_param(%15, %par_1_0, %16, %c-1_i32, %rtp_1_0) : (memref<4096xbf16>, memref<2624xf32>, i32, i32, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_1_0(Consume, 1)
        }
        %c1_80 = arith.constant 1 : index
        %2 = memref.load %rtp_1_0[%c1_80] : memref<12xi32>
        %c0_81 = arith.constant 0 : index
        %3 = arith.index_cast %2 : i32 to index
        %c1_82 = arith.constant 1 : index
        scf.for %arg1 = %c0_81 to %3 step %c1_82 {
          %14 = aie.objectfifo.acquire @A_L2L1_1_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats(%15, %par_1_0, %rtp_1_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_1_0(Consume, 1)
        }
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp_1_0[%c2] : memref<12xi32>
        %c0_83 = arith.constant 0 : index
        %5 = arith.index_cast %4 : i32 to index
        %c1_84 = arith.constant 1 : index
        scf.for %arg1 = %c0_83 to %5 step %c1_84 {
          %14 = aie.objectfifo.acquire @Y_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_1_0, %rtp_1_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Y_L1L2_1_0(Produce, 1)
        }
        %c3 = arith.constant 3 : index
        %6 = memref.load %rtp_1_0[%c3] : memref<12xi32>
        %c0_85 = arith.constant 0 : index
        %7 = arith.index_cast %6 : i32 to index
        %c1_86 = arith.constant 1 : index
        scf.for %arg1 = %c0_85 to %7 step %c1_86 {
          %14 = aie.objectfifo.acquire @A_L2L1_1_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_apply(%15, %17, %par_1_0, %rtp_1_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_1_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_1_0(Produce, 1)
        }
        %c4 = arith.constant 4 : index
        %8 = memref.load %rtp_1_0[%c4] : memref<12xi32>
        %c0_87 = arith.constant 0 : index
        %9 = arith.index_cast %8 : i32 to index
        %c1_88 = arith.constant 1 : index
        scf.for %arg1 = %c0_87 to %9 step %c1_88 {
          %14 = aie.objectfifo.acquire @A_L2L1_1_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @B_L2L1_1_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %18 = aie.objectfifo.acquire @Y_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_add(%15, %17, %19, %par_1_0, %rtp_1_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_1_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_1_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_1_0(Produce, 1)
        }
        %c5 = arith.constant 5 : index
        %10 = memref.load %rtp_1_0[%c5] : memref<12xi32>
        %c0_89 = arith.constant 0 : index
        %11 = arith.index_cast %10 : i32 to index
        %c1_90 = arith.constant 1 : index
        scf.for %arg1 = %c0_89 to %11 step %c1_90 {
          %14 = aie.objectfifo.acquire @Z_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_1_0, %rtp_1_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Z_L1L2_1_0(Produce, 1)
        }
        %c6 = arith.constant 6 : index
        %12 = memref.load %rtp_1_0[%c6] : memref<12xi32>
        %c0_91 = arith.constant 0 : index
        %13 = arith.index_cast %12 : i32 to index
        %c1_92 = arith.constant 1 : index
        scf.for %arg1 = %c0_91 to %13 step %c1_92 {
          %14 = aie.objectfifo.acquire @A_L2L1_1_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_rgba(%15, %17, %rtp_1_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_1_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_1_0(Produce, 1)
        }
        %c1_i32_93 = arith.constant 1 : i32
        aie.use_lock(%lock_1_2, Release, %c1_i32_93)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %lock_1_3 = aie.lock(%logical_core_2)
    %core_1_3 = aie.core(%logical_core_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_1_3, Acquire, %c1_i32)
        func.call @vew_begin(%par_1_1, %rtp_1_1) : (memref<2624xf32>, memref<12xi32>) -> ()
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_1_1[%c0_77] : memref<12xi32>
        %c0_78 = arith.constant 0 : index
        %1 = arith.index_cast %0 : i32 to index
        %c1_79 = arith.constant 1 : index
        scf.for %arg1 = %c0_78 to %1 step %c1_79 {
          %14 = aie.objectfifo.acquire @A_L2L1_1_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = arith.index_cast %arg1 : index to i32
          %c0_i32 = arith.constant 0 : i32
          func.call @vew_param(%15, %par_1_1, %16, %c0_i32, %rtp_1_1) : (memref<4096xbf16>, memref<2624xf32>, i32, i32, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_1_1(Consume, 1)
        }
        %c1_80 = arith.constant 1 : index
        %2 = memref.load %rtp_1_1[%c1_80] : memref<12xi32>
        %c0_81 = arith.constant 0 : index
        %3 = arith.index_cast %2 : i32 to index
        %c1_82 = arith.constant 1 : index
        scf.for %arg1 = %c0_81 to %3 step %c1_82 {
          %14 = aie.objectfifo.acquire @A_L2L1_1_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats(%15, %par_1_1, %rtp_1_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_1_1(Consume, 1)
        }
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp_1_1[%c2] : memref<12xi32>
        %c0_83 = arith.constant 0 : index
        %5 = arith.index_cast %4 : i32 to index
        %c1_84 = arith.constant 1 : index
        scf.for %arg1 = %c0_83 to %5 step %c1_84 {
          %14 = aie.objectfifo.acquire @Y_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_1_1, %rtp_1_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Y_L1L2_1_1(Produce, 1)
        }
        %c3 = arith.constant 3 : index
        %6 = memref.load %rtp_1_1[%c3] : memref<12xi32>
        %c0_85 = arith.constant 0 : index
        %7 = arith.index_cast %6 : i32 to index
        %c1_86 = arith.constant 1 : index
        scf.for %arg1 = %c0_85 to %7 step %c1_86 {
          %14 = aie.objectfifo.acquire @A_L2L1_1_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_apply(%15, %17, %par_1_1, %rtp_1_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_1_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_1_1(Produce, 1)
        }
        %c4 = arith.constant 4 : index
        %8 = memref.load %rtp_1_1[%c4] : memref<12xi32>
        %c0_87 = arith.constant 0 : index
        %9 = arith.index_cast %8 : i32 to index
        %c1_88 = arith.constant 1 : index
        scf.for %arg1 = %c0_87 to %9 step %c1_88 {
          %14 = aie.objectfifo.acquire @A_L2L1_1_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @B_L2L1_1_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %18 = aie.objectfifo.acquire @Y_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_add(%15, %17, %19, %par_1_1, %rtp_1_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_1_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_1_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_1_1(Produce, 1)
        }
        %c5 = arith.constant 5 : index
        %10 = memref.load %rtp_1_1[%c5] : memref<12xi32>
        %c0_89 = arith.constant 0 : index
        %11 = arith.index_cast %10 : i32 to index
        %c1_90 = arith.constant 1 : index
        scf.for %arg1 = %c0_89 to %11 step %c1_90 {
          %14 = aie.objectfifo.acquire @Z_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_1_1, %rtp_1_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Z_L1L2_1_1(Produce, 1)
        }
        %c6 = arith.constant 6 : index
        %12 = memref.load %rtp_1_1[%c6] : memref<12xi32>
        %c0_91 = arith.constant 0 : index
        %13 = arith.index_cast %12 : i32 to index
        %c1_92 = arith.constant 1 : index
        scf.for %arg1 = %c0_91 to %13 step %c1_92 {
          %14 = aie.objectfifo.acquire @A_L2L1_1_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_rgba(%15, %17, %rtp_1_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_1_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_1_1(Produce, 1)
        }
        %c1_i32_93 = arith.constant 1 : i32
        aie.use_lock(%lock_1_3, Release, %c1_i32_93)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %lock_2_2 = aie.lock(%logical_core_3)
    %core_2_2 = aie.core(%logical_core_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_2_2, Acquire, %c1_i32)
        func.call @vew_begin(%par_2_0, %rtp_2_0) : (memref<2624xf32>, memref<12xi32>) -> ()
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_2_0[%c0_77] : memref<12xi32>
        %c0_78 = arith.constant 0 : index
        %1 = arith.index_cast %0 : i32 to index
        %c1_79 = arith.constant 1 : index
        scf.for %arg1 = %c0_78 to %1 step %c1_79 {
          %14 = aie.objectfifo.acquire @A_L2L1_2_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = arith.index_cast %arg1 : index to i32
          %c-1_i32 = arith.constant -1 : i32
          func.call @vew_param(%15, %par_2_0, %16, %c-1_i32, %rtp_2_0) : (memref<4096xbf16>, memref<2624xf32>, i32, i32, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_2_0(Consume, 1)
        }
        %c1_80 = arith.constant 1 : index
        %2 = memref.load %rtp_2_0[%c1_80] : memref<12xi32>
        %c0_81 = arith.constant 0 : index
        %3 = arith.index_cast %2 : i32 to index
        %c1_82 = arith.constant 1 : index
        scf.for %arg1 = %c0_81 to %3 step %c1_82 {
          %14 = aie.objectfifo.acquire @A_L2L1_2_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats(%15, %par_2_0, %rtp_2_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_2_0(Consume, 1)
        }
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp_2_0[%c2] : memref<12xi32>
        %c0_83 = arith.constant 0 : index
        %5 = arith.index_cast %4 : i32 to index
        %c1_84 = arith.constant 1 : index
        scf.for %arg1 = %c0_83 to %5 step %c1_84 {
          %14 = aie.objectfifo.acquire @Y_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_2_0, %rtp_2_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Y_L1L2_2_0(Produce, 1)
        }
        %c3 = arith.constant 3 : index
        %6 = memref.load %rtp_2_0[%c3] : memref<12xi32>
        %c0_85 = arith.constant 0 : index
        %7 = arith.index_cast %6 : i32 to index
        %c1_86 = arith.constant 1 : index
        scf.for %arg1 = %c0_85 to %7 step %c1_86 {
          %14 = aie.objectfifo.acquire @A_L2L1_2_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_apply(%15, %17, %par_2_0, %rtp_2_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_2_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_2_0(Produce, 1)
        }
        %c4 = arith.constant 4 : index
        %8 = memref.load %rtp_2_0[%c4] : memref<12xi32>
        %c0_87 = arith.constant 0 : index
        %9 = arith.index_cast %8 : i32 to index
        %c1_88 = arith.constant 1 : index
        scf.for %arg1 = %c0_87 to %9 step %c1_88 {
          %14 = aie.objectfifo.acquire @A_L2L1_2_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @B_L2L1_2_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %18 = aie.objectfifo.acquire @Y_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_add(%15, %17, %19, %par_2_0, %rtp_2_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_2_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_2_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_2_0(Produce, 1)
        }
        %c5 = arith.constant 5 : index
        %10 = memref.load %rtp_2_0[%c5] : memref<12xi32>
        %c0_89 = arith.constant 0 : index
        %11 = arith.index_cast %10 : i32 to index
        %c1_90 = arith.constant 1 : index
        scf.for %arg1 = %c0_89 to %11 step %c1_90 {
          %14 = aie.objectfifo.acquire @Z_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_2_0, %rtp_2_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Z_L1L2_2_0(Produce, 1)
        }
        %c6 = arith.constant 6 : index
        %12 = memref.load %rtp_2_0[%c6] : memref<12xi32>
        %c0_91 = arith.constant 0 : index
        %13 = arith.index_cast %12 : i32 to index
        %c1_92 = arith.constant 1 : index
        scf.for %arg1 = %c0_91 to %13 step %c1_92 {
          %14 = aie.objectfifo.acquire @A_L2L1_2_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_rgba(%15, %17, %rtp_2_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_2_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_2_0(Produce, 1)
        }
        %c1_i32_93 = arith.constant 1 : i32
        aie.use_lock(%lock_2_2, Release, %c1_i32_93)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %lock_2_3 = aie.lock(%logical_core_4)
    %core_2_3 = aie.core(%logical_core_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_2_3, Acquire, %c1_i32)
        func.call @vew_begin(%par_2_1, %rtp_2_1) : (memref<2624xf32>, memref<12xi32>) -> ()
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_2_1[%c0_77] : memref<12xi32>
        %c0_78 = arith.constant 0 : index
        %1 = arith.index_cast %0 : i32 to index
        %c1_79 = arith.constant 1 : index
        scf.for %arg1 = %c0_78 to %1 step %c1_79 {
          %14 = aie.objectfifo.acquire @A_L2L1_2_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = arith.index_cast %arg1 : index to i32
          %c0_i32 = arith.constant 0 : i32
          func.call @vew_param(%15, %par_2_1, %16, %c0_i32, %rtp_2_1) : (memref<4096xbf16>, memref<2624xf32>, i32, i32, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_2_1(Consume, 1)
        }
        %c1_80 = arith.constant 1 : index
        %2 = memref.load %rtp_2_1[%c1_80] : memref<12xi32>
        %c0_81 = arith.constant 0 : index
        %3 = arith.index_cast %2 : i32 to index
        %c1_82 = arith.constant 1 : index
        scf.for %arg1 = %c0_81 to %3 step %c1_82 {
          %14 = aie.objectfifo.acquire @A_L2L1_2_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats(%15, %par_2_1, %rtp_2_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_2_1(Consume, 1)
        }
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp_2_1[%c2] : memref<12xi32>
        %c0_83 = arith.constant 0 : index
        %5 = arith.index_cast %4 : i32 to index
        %c1_84 = arith.constant 1 : index
        scf.for %arg1 = %c0_83 to %5 step %c1_84 {
          %14 = aie.objectfifo.acquire @Y_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_2_1, %rtp_2_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Y_L1L2_2_1(Produce, 1)
        }
        %c3 = arith.constant 3 : index
        %6 = memref.load %rtp_2_1[%c3] : memref<12xi32>
        %c0_85 = arith.constant 0 : index
        %7 = arith.index_cast %6 : i32 to index
        %c1_86 = arith.constant 1 : index
        scf.for %arg1 = %c0_85 to %7 step %c1_86 {
          %14 = aie.objectfifo.acquire @A_L2L1_2_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_apply(%15, %17, %par_2_1, %rtp_2_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_2_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_2_1(Produce, 1)
        }
        %c4 = arith.constant 4 : index
        %8 = memref.load %rtp_2_1[%c4] : memref<12xi32>
        %c0_87 = arith.constant 0 : index
        %9 = arith.index_cast %8 : i32 to index
        %c1_88 = arith.constant 1 : index
        scf.for %arg1 = %c0_87 to %9 step %c1_88 {
          %14 = aie.objectfifo.acquire @A_L2L1_2_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @B_L2L1_2_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %18 = aie.objectfifo.acquire @Y_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_add(%15, %17, %19, %par_2_1, %rtp_2_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_2_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_2_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_2_1(Produce, 1)
        }
        %c5 = arith.constant 5 : index
        %10 = memref.load %rtp_2_1[%c5] : memref<12xi32>
        %c0_89 = arith.constant 0 : index
        %11 = arith.index_cast %10 : i32 to index
        %c1_90 = arith.constant 1 : index
        scf.for %arg1 = %c0_89 to %11 step %c1_90 {
          %14 = aie.objectfifo.acquire @Z_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_2_1, %rtp_2_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Z_L1L2_2_1(Produce, 1)
        }
        %c6 = arith.constant 6 : index
        %12 = memref.load %rtp_2_1[%c6] : memref<12xi32>
        %c0_91 = arith.constant 0 : index
        %13 = arith.index_cast %12 : i32 to index
        %c1_92 = arith.constant 1 : index
        scf.for %arg1 = %c0_91 to %13 step %c1_92 {
          %14 = aie.objectfifo.acquire @A_L2L1_2_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_rgba(%15, %17, %rtp_2_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_2_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_2_1(Produce, 1)
        }
        %c1_i32_93 = arith.constant 1 : i32
        aie.use_lock(%lock_2_3, Release, %c1_i32_93)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %lock_3_2 = aie.lock(%logical_core_5)
    %core_3_2 = aie.core(%logical_core_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_3_2, Acquire, %c1_i32)
        func.call @vew_begin(%par_3_0, %rtp_3_0) : (memref<2624xf32>, memref<12xi32>) -> ()
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_3_0[%c0_77] : memref<12xi32>
        %c0_78 = arith.constant 0 : index
        %1 = arith.index_cast %0 : i32 to index
        %c1_79 = arith.constant 1 : index
        scf.for %arg1 = %c0_78 to %1 step %c1_79 {
          %14 = aie.objectfifo.acquire @A_L2L1_3_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = arith.index_cast %arg1 : index to i32
          %c-1_i32 = arith.constant -1 : i32
          func.call @vew_param(%15, %par_3_0, %16, %c-1_i32, %rtp_3_0) : (memref<4096xbf16>, memref<2624xf32>, i32, i32, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_3_0(Consume, 1)
        }
        %c1_80 = arith.constant 1 : index
        %2 = memref.load %rtp_3_0[%c1_80] : memref<12xi32>
        %c0_81 = arith.constant 0 : index
        %3 = arith.index_cast %2 : i32 to index
        %c1_82 = arith.constant 1 : index
        scf.for %arg1 = %c0_81 to %3 step %c1_82 {
          %14 = aie.objectfifo.acquire @A_L2L1_3_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats(%15, %par_3_0, %rtp_3_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_3_0(Consume, 1)
        }
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp_3_0[%c2] : memref<12xi32>
        %c0_83 = arith.constant 0 : index
        %5 = arith.index_cast %4 : i32 to index
        %c1_84 = arith.constant 1 : index
        scf.for %arg1 = %c0_83 to %5 step %c1_84 {
          %14 = aie.objectfifo.acquire @Y_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_3_0, %rtp_3_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Y_L1L2_3_0(Produce, 1)
        }
        %c3 = arith.constant 3 : index
        %6 = memref.load %rtp_3_0[%c3] : memref<12xi32>
        %c0_85 = arith.constant 0 : index
        %7 = arith.index_cast %6 : i32 to index
        %c1_86 = arith.constant 1 : index
        scf.for %arg1 = %c0_85 to %7 step %c1_86 {
          %14 = aie.objectfifo.acquire @A_L2L1_3_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_apply(%15, %17, %par_3_0, %rtp_3_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_3_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_3_0(Produce, 1)
        }
        %c4 = arith.constant 4 : index
        %8 = memref.load %rtp_3_0[%c4] : memref<12xi32>
        %c0_87 = arith.constant 0 : index
        %9 = arith.index_cast %8 : i32 to index
        %c1_88 = arith.constant 1 : index
        scf.for %arg1 = %c0_87 to %9 step %c1_88 {
          %14 = aie.objectfifo.acquire @A_L2L1_3_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @B_L2L1_3_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %18 = aie.objectfifo.acquire @Y_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_add(%15, %17, %19, %par_3_0, %rtp_3_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_3_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_3_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_3_0(Produce, 1)
        }
        %c5 = arith.constant 5 : index
        %10 = memref.load %rtp_3_0[%c5] : memref<12xi32>
        %c0_89 = arith.constant 0 : index
        %11 = arith.index_cast %10 : i32 to index
        %c1_90 = arith.constant 1 : index
        scf.for %arg1 = %c0_89 to %11 step %c1_90 {
          %14 = aie.objectfifo.acquire @Z_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_3_0, %rtp_3_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Z_L1L2_3_0(Produce, 1)
        }
        %c6 = arith.constant 6 : index
        %12 = memref.load %rtp_3_0[%c6] : memref<12xi32>
        %c0_91 = arith.constant 0 : index
        %13 = arith.index_cast %12 : i32 to index
        %c1_92 = arith.constant 1 : index
        scf.for %arg1 = %c0_91 to %13 step %c1_92 {
          %14 = aie.objectfifo.acquire @A_L2L1_3_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_rgba(%15, %17, %rtp_3_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_3_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_3_0(Produce, 1)
        }
        %c1_i32_93 = arith.constant 1 : i32
        aie.use_lock(%lock_3_2, Release, %c1_i32_93)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %lock_3_3 = aie.lock(%logical_core_6)
    %core_3_3 = aie.core(%logical_core_6) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_3_3, Acquire, %c1_i32)
        func.call @vew_begin(%par_3_1, %rtp_3_1) : (memref<2624xf32>, memref<12xi32>) -> ()
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_3_1[%c0_77] : memref<12xi32>
        %c0_78 = arith.constant 0 : index
        %1 = arith.index_cast %0 : i32 to index
        %c1_79 = arith.constant 1 : index
        scf.for %arg1 = %c0_78 to %1 step %c1_79 {
          %14 = aie.objectfifo.acquire @A_L2L1_3_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = arith.index_cast %arg1 : index to i32
          %c0_i32 = arith.constant 0 : i32
          func.call @vew_param(%15, %par_3_1, %16, %c0_i32, %rtp_3_1) : (memref<4096xbf16>, memref<2624xf32>, i32, i32, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_3_1(Consume, 1)
        }
        %c1_80 = arith.constant 1 : index
        %2 = memref.load %rtp_3_1[%c1_80] : memref<12xi32>
        %c0_81 = arith.constant 0 : index
        %3 = arith.index_cast %2 : i32 to index
        %c1_82 = arith.constant 1 : index
        scf.for %arg1 = %c0_81 to %3 step %c1_82 {
          %14 = aie.objectfifo.acquire @A_L2L1_3_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats(%15, %par_3_1, %rtp_3_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_3_1(Consume, 1)
        }
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp_3_1[%c2] : memref<12xi32>
        %c0_83 = arith.constant 0 : index
        %5 = arith.index_cast %4 : i32 to index
        %c1_84 = arith.constant 1 : index
        scf.for %arg1 = %c0_83 to %5 step %c1_84 {
          %14 = aie.objectfifo.acquire @Y_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_3_1, %rtp_3_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Y_L1L2_3_1(Produce, 1)
        }
        %c3 = arith.constant 3 : index
        %6 = memref.load %rtp_3_1[%c3] : memref<12xi32>
        %c0_85 = arith.constant 0 : index
        %7 = arith.index_cast %6 : i32 to index
        %c1_86 = arith.constant 1 : index
        scf.for %arg1 = %c0_85 to %7 step %c1_86 {
          %14 = aie.objectfifo.acquire @A_L2L1_3_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_apply(%15, %17, %par_3_1, %rtp_3_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_3_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_3_1(Produce, 1)
        }
        %c4 = arith.constant 4 : index
        %8 = memref.load %rtp_3_1[%c4] : memref<12xi32>
        %c0_87 = arith.constant 0 : index
        %9 = arith.index_cast %8 : i32 to index
        %c1_88 = arith.constant 1 : index
        scf.for %arg1 = %c0_87 to %9 step %c1_88 {
          %14 = aie.objectfifo.acquire @A_L2L1_3_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @B_L2L1_3_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %18 = aie.objectfifo.acquire @Y_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_add(%15, %17, %19, %par_3_1, %rtp_3_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_3_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_3_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_3_1(Produce, 1)
        }
        %c5 = arith.constant 5 : index
        %10 = memref.load %rtp_3_1[%c5] : memref<12xi32>
        %c0_89 = arith.constant 0 : index
        %11 = arith.index_cast %10 : i32 to index
        %c1_90 = arith.constant 1 : index
        scf.for %arg1 = %c0_89 to %11 step %c1_90 {
          %14 = aie.objectfifo.acquire @Z_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_3_1, %rtp_3_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Z_L1L2_3_1(Produce, 1)
        }
        %c6 = arith.constant 6 : index
        %12 = memref.load %rtp_3_1[%c6] : memref<12xi32>
        %c0_91 = arith.constant 0 : index
        %13 = arith.index_cast %12 : i32 to index
        %c1_92 = arith.constant 1 : index
        scf.for %arg1 = %c0_91 to %13 step %c1_92 {
          %14 = aie.objectfifo.acquire @A_L2L1_3_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_rgba(%15, %17, %rtp_3_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_3_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_3_1(Produce, 1)
        }
        %c1_i32_93 = arith.constant 1 : i32
        aie.use_lock(%lock_3_3, Release, %c1_i32_93)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %lock_4_2 = aie.lock(%logical_core_7)
    %core_4_2 = aie.core(%logical_core_7) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_4_2, Acquire, %c1_i32)
        func.call @vew_begin(%par_4_0, %rtp_4_0) : (memref<2624xf32>, memref<12xi32>) -> ()
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_4_0[%c0_77] : memref<12xi32>
        %c0_78 = arith.constant 0 : index
        %1 = arith.index_cast %0 : i32 to index
        %c1_79 = arith.constant 1 : index
        scf.for %arg1 = %c0_78 to %1 step %c1_79 {
          %14 = aie.objectfifo.acquire @A_L2L1_4_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = arith.index_cast %arg1 : index to i32
          %c-1_i32 = arith.constant -1 : i32
          func.call @vew_param(%15, %par_4_0, %16, %c-1_i32, %rtp_4_0) : (memref<4096xbf16>, memref<2624xf32>, i32, i32, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_4_0(Consume, 1)
        }
        %c1_80 = arith.constant 1 : index
        %2 = memref.load %rtp_4_0[%c1_80] : memref<12xi32>
        %c0_81 = arith.constant 0 : index
        %3 = arith.index_cast %2 : i32 to index
        %c1_82 = arith.constant 1 : index
        scf.for %arg1 = %c0_81 to %3 step %c1_82 {
          %14 = aie.objectfifo.acquire @A_L2L1_4_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats(%15, %par_4_0, %rtp_4_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_4_0(Consume, 1)
        }
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp_4_0[%c2] : memref<12xi32>
        %c0_83 = arith.constant 0 : index
        %5 = arith.index_cast %4 : i32 to index
        %c1_84 = arith.constant 1 : index
        scf.for %arg1 = %c0_83 to %5 step %c1_84 {
          %14 = aie.objectfifo.acquire @Y_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_4_0, %rtp_4_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Y_L1L2_4_0(Produce, 1)
        }
        %c3 = arith.constant 3 : index
        %6 = memref.load %rtp_4_0[%c3] : memref<12xi32>
        %c0_85 = arith.constant 0 : index
        %7 = arith.index_cast %6 : i32 to index
        %c1_86 = arith.constant 1 : index
        scf.for %arg1 = %c0_85 to %7 step %c1_86 {
          %14 = aie.objectfifo.acquire @A_L2L1_4_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_apply(%15, %17, %par_4_0, %rtp_4_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_4_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_4_0(Produce, 1)
        }
        %c4 = arith.constant 4 : index
        %8 = memref.load %rtp_4_0[%c4] : memref<12xi32>
        %c0_87 = arith.constant 0 : index
        %9 = arith.index_cast %8 : i32 to index
        %c1_88 = arith.constant 1 : index
        scf.for %arg1 = %c0_87 to %9 step %c1_88 {
          %14 = aie.objectfifo.acquire @A_L2L1_4_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @B_L2L1_4_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %18 = aie.objectfifo.acquire @Y_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_add(%15, %17, %19, %par_4_0, %rtp_4_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_4_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_4_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_4_0(Produce, 1)
        }
        %c5 = arith.constant 5 : index
        %10 = memref.load %rtp_4_0[%c5] : memref<12xi32>
        %c0_89 = arith.constant 0 : index
        %11 = arith.index_cast %10 : i32 to index
        %c1_90 = arith.constant 1 : index
        scf.for %arg1 = %c0_89 to %11 step %c1_90 {
          %14 = aie.objectfifo.acquire @Z_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_4_0, %rtp_4_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Z_L1L2_4_0(Produce, 1)
        }
        %c6 = arith.constant 6 : index
        %12 = memref.load %rtp_4_0[%c6] : memref<12xi32>
        %c0_91 = arith.constant 0 : index
        %13 = arith.index_cast %12 : i32 to index
        %c1_92 = arith.constant 1 : index
        scf.for %arg1 = %c0_91 to %13 step %c1_92 {
          %14 = aie.objectfifo.acquire @A_L2L1_4_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_rgba(%15, %17, %rtp_4_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_4_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_4_0(Produce, 1)
        }
        %c1_i32_93 = arith.constant 1 : i32
        aie.use_lock(%lock_4_2, Release, %c1_i32_93)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %lock_4_3 = aie.lock(%logical_core_8)
    %core_4_3 = aie.core(%logical_core_8) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_4_3, Acquire, %c1_i32)
        func.call @vew_begin(%par_4_1, %rtp_4_1) : (memref<2624xf32>, memref<12xi32>) -> ()
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_4_1[%c0_77] : memref<12xi32>
        %c0_78 = arith.constant 0 : index
        %1 = arith.index_cast %0 : i32 to index
        %c1_79 = arith.constant 1 : index
        scf.for %arg1 = %c0_78 to %1 step %c1_79 {
          %14 = aie.objectfifo.acquire @A_L2L1_4_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = arith.index_cast %arg1 : index to i32
          %c0_i32 = arith.constant 0 : i32
          func.call @vew_param(%15, %par_4_1, %16, %c0_i32, %rtp_4_1) : (memref<4096xbf16>, memref<2624xf32>, i32, i32, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_4_1(Consume, 1)
        }
        %c1_80 = arith.constant 1 : index
        %2 = memref.load %rtp_4_1[%c1_80] : memref<12xi32>
        %c0_81 = arith.constant 0 : index
        %3 = arith.index_cast %2 : i32 to index
        %c1_82 = arith.constant 1 : index
        scf.for %arg1 = %c0_81 to %3 step %c1_82 {
          %14 = aie.objectfifo.acquire @A_L2L1_4_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats(%15, %par_4_1, %rtp_4_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_4_1(Consume, 1)
        }
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp_4_1[%c2] : memref<12xi32>
        %c0_83 = arith.constant 0 : index
        %5 = arith.index_cast %4 : i32 to index
        %c1_84 = arith.constant 1 : index
        scf.for %arg1 = %c0_83 to %5 step %c1_84 {
          %14 = aie.objectfifo.acquire @Y_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_4_1, %rtp_4_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Y_L1L2_4_1(Produce, 1)
        }
        %c3 = arith.constant 3 : index
        %6 = memref.load %rtp_4_1[%c3] : memref<12xi32>
        %c0_85 = arith.constant 0 : index
        %7 = arith.index_cast %6 : i32 to index
        %c1_86 = arith.constant 1 : index
        scf.for %arg1 = %c0_85 to %7 step %c1_86 {
          %14 = aie.objectfifo.acquire @A_L2L1_4_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_apply(%15, %17, %par_4_1, %rtp_4_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_4_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_4_1(Produce, 1)
        }
        %c4 = arith.constant 4 : index
        %8 = memref.load %rtp_4_1[%c4] : memref<12xi32>
        %c0_87 = arith.constant 0 : index
        %9 = arith.index_cast %8 : i32 to index
        %c1_88 = arith.constant 1 : index
        scf.for %arg1 = %c0_87 to %9 step %c1_88 {
          %14 = aie.objectfifo.acquire @A_L2L1_4_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @B_L2L1_4_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %18 = aie.objectfifo.acquire @Y_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_add(%15, %17, %19, %par_4_1, %rtp_4_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_4_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_4_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_4_1(Produce, 1)
        }
        %c5 = arith.constant 5 : index
        %10 = memref.load %rtp_4_1[%c5] : memref<12xi32>
        %c0_89 = arith.constant 0 : index
        %11 = arith.index_cast %10 : i32 to index
        %c1_90 = arith.constant 1 : index
        scf.for %arg1 = %c0_89 to %11 step %c1_90 {
          %14 = aie.objectfifo.acquire @Z_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_4_1, %rtp_4_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Z_L1L2_4_1(Produce, 1)
        }
        %c6 = arith.constant 6 : index
        %12 = memref.load %rtp_4_1[%c6] : memref<12xi32>
        %c0_91 = arith.constant 0 : index
        %13 = arith.index_cast %12 : i32 to index
        %c1_92 = arith.constant 1 : index
        scf.for %arg1 = %c0_91 to %13 step %c1_92 {
          %14 = aie.objectfifo.acquire @A_L2L1_4_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_rgba(%15, %17, %rtp_4_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_4_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_4_1(Produce, 1)
        }
        %c1_i32_93 = arith.constant 1 : i32
        aie.use_lock(%lock_4_3, Release, %c1_i32_93)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %lock_5_2 = aie.lock(%logical_core_9)
    %core_5_2 = aie.core(%logical_core_9) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_5_2, Acquire, %c1_i32)
        func.call @vew_begin(%par_5_0, %rtp_5_0) : (memref<2624xf32>, memref<12xi32>) -> ()
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_5_0[%c0_77] : memref<12xi32>
        %c0_78 = arith.constant 0 : index
        %1 = arith.index_cast %0 : i32 to index
        %c1_79 = arith.constant 1 : index
        scf.for %arg1 = %c0_78 to %1 step %c1_79 {
          %14 = aie.objectfifo.acquire @A_L2L1_5_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = arith.index_cast %arg1 : index to i32
          %c-1_i32 = arith.constant -1 : i32
          func.call @vew_param(%15, %par_5_0, %16, %c-1_i32, %rtp_5_0) : (memref<4096xbf16>, memref<2624xf32>, i32, i32, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_5_0(Consume, 1)
        }
        %c1_80 = arith.constant 1 : index
        %2 = memref.load %rtp_5_0[%c1_80] : memref<12xi32>
        %c0_81 = arith.constant 0 : index
        %3 = arith.index_cast %2 : i32 to index
        %c1_82 = arith.constant 1 : index
        scf.for %arg1 = %c0_81 to %3 step %c1_82 {
          %14 = aie.objectfifo.acquire @A_L2L1_5_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats(%15, %par_5_0, %rtp_5_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_5_0(Consume, 1)
        }
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp_5_0[%c2] : memref<12xi32>
        %c0_83 = arith.constant 0 : index
        %5 = arith.index_cast %4 : i32 to index
        %c1_84 = arith.constant 1 : index
        scf.for %arg1 = %c0_83 to %5 step %c1_84 {
          %14 = aie.objectfifo.acquire @Y_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_5_0, %rtp_5_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Y_L1L2_5_0(Produce, 1)
        }
        %c3 = arith.constant 3 : index
        %6 = memref.load %rtp_5_0[%c3] : memref<12xi32>
        %c0_85 = arith.constant 0 : index
        %7 = arith.index_cast %6 : i32 to index
        %c1_86 = arith.constant 1 : index
        scf.for %arg1 = %c0_85 to %7 step %c1_86 {
          %14 = aie.objectfifo.acquire @A_L2L1_5_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_apply(%15, %17, %par_5_0, %rtp_5_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_5_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_5_0(Produce, 1)
        }
        %c4 = arith.constant 4 : index
        %8 = memref.load %rtp_5_0[%c4] : memref<12xi32>
        %c0_87 = arith.constant 0 : index
        %9 = arith.index_cast %8 : i32 to index
        %c1_88 = arith.constant 1 : index
        scf.for %arg1 = %c0_87 to %9 step %c1_88 {
          %14 = aie.objectfifo.acquire @A_L2L1_5_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @B_L2L1_5_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %18 = aie.objectfifo.acquire @Y_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_add(%15, %17, %19, %par_5_0, %rtp_5_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_5_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_5_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_5_0(Produce, 1)
        }
        %c5 = arith.constant 5 : index
        %10 = memref.load %rtp_5_0[%c5] : memref<12xi32>
        %c0_89 = arith.constant 0 : index
        %11 = arith.index_cast %10 : i32 to index
        %c1_90 = arith.constant 1 : index
        scf.for %arg1 = %c0_89 to %11 step %c1_90 {
          %14 = aie.objectfifo.acquire @Z_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_5_0, %rtp_5_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Z_L1L2_5_0(Produce, 1)
        }
        %c6 = arith.constant 6 : index
        %12 = memref.load %rtp_5_0[%c6] : memref<12xi32>
        %c0_91 = arith.constant 0 : index
        %13 = arith.index_cast %12 : i32 to index
        %c1_92 = arith.constant 1 : index
        scf.for %arg1 = %c0_91 to %13 step %c1_92 {
          %14 = aie.objectfifo.acquire @A_L2L1_5_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_rgba(%15, %17, %rtp_5_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_5_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_5_0(Produce, 1)
        }
        %c1_i32_93 = arith.constant 1 : i32
        aie.use_lock(%lock_5_2, Release, %c1_i32_93)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %lock_5_3 = aie.lock(%logical_core_10)
    %core_5_3 = aie.core(%logical_core_10) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_5_3, Acquire, %c1_i32)
        func.call @vew_begin(%par_5_1, %rtp_5_1) : (memref<2624xf32>, memref<12xi32>) -> ()
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_5_1[%c0_77] : memref<12xi32>
        %c0_78 = arith.constant 0 : index
        %1 = arith.index_cast %0 : i32 to index
        %c1_79 = arith.constant 1 : index
        scf.for %arg1 = %c0_78 to %1 step %c1_79 {
          %14 = aie.objectfifo.acquire @A_L2L1_5_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = arith.index_cast %arg1 : index to i32
          %c0_i32 = arith.constant 0 : i32
          func.call @vew_param(%15, %par_5_1, %16, %c0_i32, %rtp_5_1) : (memref<4096xbf16>, memref<2624xf32>, i32, i32, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_5_1(Consume, 1)
        }
        %c1_80 = arith.constant 1 : index
        %2 = memref.load %rtp_5_1[%c1_80] : memref<12xi32>
        %c0_81 = arith.constant 0 : index
        %3 = arith.index_cast %2 : i32 to index
        %c1_82 = arith.constant 1 : index
        scf.for %arg1 = %c0_81 to %3 step %c1_82 {
          %14 = aie.objectfifo.acquire @A_L2L1_5_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats(%15, %par_5_1, %rtp_5_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_5_1(Consume, 1)
        }
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp_5_1[%c2] : memref<12xi32>
        %c0_83 = arith.constant 0 : index
        %5 = arith.index_cast %4 : i32 to index
        %c1_84 = arith.constant 1 : index
        scf.for %arg1 = %c0_83 to %5 step %c1_84 {
          %14 = aie.objectfifo.acquire @Y_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_5_1, %rtp_5_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Y_L1L2_5_1(Produce, 1)
        }
        %c3 = arith.constant 3 : index
        %6 = memref.load %rtp_5_1[%c3] : memref<12xi32>
        %c0_85 = arith.constant 0 : index
        %7 = arith.index_cast %6 : i32 to index
        %c1_86 = arith.constant 1 : index
        scf.for %arg1 = %c0_85 to %7 step %c1_86 {
          %14 = aie.objectfifo.acquire @A_L2L1_5_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_apply(%15, %17, %par_5_1, %rtp_5_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_5_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_5_1(Produce, 1)
        }
        %c4 = arith.constant 4 : index
        %8 = memref.load %rtp_5_1[%c4] : memref<12xi32>
        %c0_87 = arith.constant 0 : index
        %9 = arith.index_cast %8 : i32 to index
        %c1_88 = arith.constant 1 : index
        scf.for %arg1 = %c0_87 to %9 step %c1_88 {
          %14 = aie.objectfifo.acquire @A_L2L1_5_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @B_L2L1_5_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %18 = aie.objectfifo.acquire @Y_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_add(%15, %17, %19, %par_5_1, %rtp_5_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_5_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_5_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_5_1(Produce, 1)
        }
        %c5 = arith.constant 5 : index
        %10 = memref.load %rtp_5_1[%c5] : memref<12xi32>
        %c0_89 = arith.constant 0 : index
        %11 = arith.index_cast %10 : i32 to index
        %c1_90 = arith.constant 1 : index
        scf.for %arg1 = %c0_89 to %11 step %c1_90 {
          %14 = aie.objectfifo.acquire @Z_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_5_1, %rtp_5_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Z_L1L2_5_1(Produce, 1)
        }
        %c6 = arith.constant 6 : index
        %12 = memref.load %rtp_5_1[%c6] : memref<12xi32>
        %c0_91 = arith.constant 0 : index
        %13 = arith.index_cast %12 : i32 to index
        %c1_92 = arith.constant 1 : index
        scf.for %arg1 = %c0_91 to %13 step %c1_92 {
          %14 = aie.objectfifo.acquire @A_L2L1_5_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_rgba(%15, %17, %rtp_5_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_5_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_5_1(Produce, 1)
        }
        %c1_i32_93 = arith.constant 1 : i32
        aie.use_lock(%lock_5_3, Release, %c1_i32_93)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %lock_6_2 = aie.lock(%logical_core_11)
    %core_6_2 = aie.core(%logical_core_11) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_6_2, Acquire, %c1_i32)
        func.call @vew_begin(%par_6_0, %rtp_6_0) : (memref<2624xf32>, memref<12xi32>) -> ()
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_6_0[%c0_77] : memref<12xi32>
        %c0_78 = arith.constant 0 : index
        %1 = arith.index_cast %0 : i32 to index
        %c1_79 = arith.constant 1 : index
        scf.for %arg1 = %c0_78 to %1 step %c1_79 {
          %14 = aie.objectfifo.acquire @A_L2L1_6_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = arith.index_cast %arg1 : index to i32
          %c-1_i32 = arith.constant -1 : i32
          func.call @vew_param(%15, %par_6_0, %16, %c-1_i32, %rtp_6_0) : (memref<4096xbf16>, memref<2624xf32>, i32, i32, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_6_0(Consume, 1)
        }
        %c1_80 = arith.constant 1 : index
        %2 = memref.load %rtp_6_0[%c1_80] : memref<12xi32>
        %c0_81 = arith.constant 0 : index
        %3 = arith.index_cast %2 : i32 to index
        %c1_82 = arith.constant 1 : index
        scf.for %arg1 = %c0_81 to %3 step %c1_82 {
          %14 = aie.objectfifo.acquire @A_L2L1_6_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats(%15, %par_6_0, %rtp_6_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_6_0(Consume, 1)
        }
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp_6_0[%c2] : memref<12xi32>
        %c0_83 = arith.constant 0 : index
        %5 = arith.index_cast %4 : i32 to index
        %c1_84 = arith.constant 1 : index
        scf.for %arg1 = %c0_83 to %5 step %c1_84 {
          %14 = aie.objectfifo.acquire @Y_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_6_0, %rtp_6_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Y_L1L2_6_0(Produce, 1)
        }
        %c3 = arith.constant 3 : index
        %6 = memref.load %rtp_6_0[%c3] : memref<12xi32>
        %c0_85 = arith.constant 0 : index
        %7 = arith.index_cast %6 : i32 to index
        %c1_86 = arith.constant 1 : index
        scf.for %arg1 = %c0_85 to %7 step %c1_86 {
          %14 = aie.objectfifo.acquire @A_L2L1_6_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_apply(%15, %17, %par_6_0, %rtp_6_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_6_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_6_0(Produce, 1)
        }
        %c4 = arith.constant 4 : index
        %8 = memref.load %rtp_6_0[%c4] : memref<12xi32>
        %c0_87 = arith.constant 0 : index
        %9 = arith.index_cast %8 : i32 to index
        %c1_88 = arith.constant 1 : index
        scf.for %arg1 = %c0_87 to %9 step %c1_88 {
          %14 = aie.objectfifo.acquire @A_L2L1_6_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @B_L2L1_6_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %18 = aie.objectfifo.acquire @Y_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_add(%15, %17, %19, %par_6_0, %rtp_6_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_6_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_6_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_6_0(Produce, 1)
        }
        %c5 = arith.constant 5 : index
        %10 = memref.load %rtp_6_0[%c5] : memref<12xi32>
        %c0_89 = arith.constant 0 : index
        %11 = arith.index_cast %10 : i32 to index
        %c1_90 = arith.constant 1 : index
        scf.for %arg1 = %c0_89 to %11 step %c1_90 {
          %14 = aie.objectfifo.acquire @Z_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_6_0, %rtp_6_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Z_L1L2_6_0(Produce, 1)
        }
        %c6 = arith.constant 6 : index
        %12 = memref.load %rtp_6_0[%c6] : memref<12xi32>
        %c0_91 = arith.constant 0 : index
        %13 = arith.index_cast %12 : i32 to index
        %c1_92 = arith.constant 1 : index
        scf.for %arg1 = %c0_91 to %13 step %c1_92 {
          %14 = aie.objectfifo.acquire @A_L2L1_6_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_rgba(%15, %17, %rtp_6_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_6_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_6_0(Produce, 1)
        }
        %c1_i32_93 = arith.constant 1 : i32
        aie.use_lock(%lock_6_2, Release, %c1_i32_93)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %lock_6_3 = aie.lock(%logical_core_12)
    %core_6_3 = aie.core(%logical_core_12) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_6_3, Acquire, %c1_i32)
        func.call @vew_begin(%par_6_1, %rtp_6_1) : (memref<2624xf32>, memref<12xi32>) -> ()
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_6_1[%c0_77] : memref<12xi32>
        %c0_78 = arith.constant 0 : index
        %1 = arith.index_cast %0 : i32 to index
        %c1_79 = arith.constant 1 : index
        scf.for %arg1 = %c0_78 to %1 step %c1_79 {
          %14 = aie.objectfifo.acquire @A_L2L1_6_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = arith.index_cast %arg1 : index to i32
          %c0_i32 = arith.constant 0 : i32
          func.call @vew_param(%15, %par_6_1, %16, %c0_i32, %rtp_6_1) : (memref<4096xbf16>, memref<2624xf32>, i32, i32, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_6_1(Consume, 1)
        }
        %c1_80 = arith.constant 1 : index
        %2 = memref.load %rtp_6_1[%c1_80] : memref<12xi32>
        %c0_81 = arith.constant 0 : index
        %3 = arith.index_cast %2 : i32 to index
        %c1_82 = arith.constant 1 : index
        scf.for %arg1 = %c0_81 to %3 step %c1_82 {
          %14 = aie.objectfifo.acquire @A_L2L1_6_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats(%15, %par_6_1, %rtp_6_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_6_1(Consume, 1)
        }
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp_6_1[%c2] : memref<12xi32>
        %c0_83 = arith.constant 0 : index
        %5 = arith.index_cast %4 : i32 to index
        %c1_84 = arith.constant 1 : index
        scf.for %arg1 = %c0_83 to %5 step %c1_84 {
          %14 = aie.objectfifo.acquire @Y_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_6_1, %rtp_6_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Y_L1L2_6_1(Produce, 1)
        }
        %c3 = arith.constant 3 : index
        %6 = memref.load %rtp_6_1[%c3] : memref<12xi32>
        %c0_85 = arith.constant 0 : index
        %7 = arith.index_cast %6 : i32 to index
        %c1_86 = arith.constant 1 : index
        scf.for %arg1 = %c0_85 to %7 step %c1_86 {
          %14 = aie.objectfifo.acquire @A_L2L1_6_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_apply(%15, %17, %par_6_1, %rtp_6_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_6_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_6_1(Produce, 1)
        }
        %c4 = arith.constant 4 : index
        %8 = memref.load %rtp_6_1[%c4] : memref<12xi32>
        %c0_87 = arith.constant 0 : index
        %9 = arith.index_cast %8 : i32 to index
        %c1_88 = arith.constant 1 : index
        scf.for %arg1 = %c0_87 to %9 step %c1_88 {
          %14 = aie.objectfifo.acquire @A_L2L1_6_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @B_L2L1_6_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %18 = aie.objectfifo.acquire @Y_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_add(%15, %17, %19, %par_6_1, %rtp_6_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_6_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_6_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_6_1(Produce, 1)
        }
        %c5 = arith.constant 5 : index
        %10 = memref.load %rtp_6_1[%c5] : memref<12xi32>
        %c0_89 = arith.constant 0 : index
        %11 = arith.index_cast %10 : i32 to index
        %c1_90 = arith.constant 1 : index
        scf.for %arg1 = %c0_89 to %11 step %c1_90 {
          %14 = aie.objectfifo.acquire @Z_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_6_1, %rtp_6_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Z_L1L2_6_1(Produce, 1)
        }
        %c6 = arith.constant 6 : index
        %12 = memref.load %rtp_6_1[%c6] : memref<12xi32>
        %c0_91 = arith.constant 0 : index
        %13 = arith.index_cast %12 : i32 to index
        %c1_92 = arith.constant 1 : index
        scf.for %arg1 = %c0_91 to %13 step %c1_92 {
          %14 = aie.objectfifo.acquire @A_L2L1_6_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_rgba(%15, %17, %rtp_6_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_6_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_6_1(Produce, 1)
        }
        %c1_i32_93 = arith.constant 1 : i32
        aie.use_lock(%lock_6_3, Release, %c1_i32_93)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %lock_7_2 = aie.lock(%logical_core_13)
    %core_7_2 = aie.core(%logical_core_13) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_7_2, Acquire, %c1_i32)
        func.call @vew_begin(%par_7_0, %rtp_7_0) : (memref<2624xf32>, memref<12xi32>) -> ()
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_7_0[%c0_77] : memref<12xi32>
        %c0_78 = arith.constant 0 : index
        %1 = arith.index_cast %0 : i32 to index
        %c1_79 = arith.constant 1 : index
        scf.for %arg1 = %c0_78 to %1 step %c1_79 {
          %14 = aie.objectfifo.acquire @A_L2L1_7_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = arith.index_cast %arg1 : index to i32
          %c-1_i32 = arith.constant -1 : i32
          func.call @vew_param(%15, %par_7_0, %16, %c-1_i32, %rtp_7_0) : (memref<4096xbf16>, memref<2624xf32>, i32, i32, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_7_0(Consume, 1)
        }
        %c1_80 = arith.constant 1 : index
        %2 = memref.load %rtp_7_0[%c1_80] : memref<12xi32>
        %c0_81 = arith.constant 0 : index
        %3 = arith.index_cast %2 : i32 to index
        %c1_82 = arith.constant 1 : index
        scf.for %arg1 = %c0_81 to %3 step %c1_82 {
          %14 = aie.objectfifo.acquire @A_L2L1_7_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats(%15, %par_7_0, %rtp_7_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_7_0(Consume, 1)
        }
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp_7_0[%c2] : memref<12xi32>
        %c0_83 = arith.constant 0 : index
        %5 = arith.index_cast %4 : i32 to index
        %c1_84 = arith.constant 1 : index
        scf.for %arg1 = %c0_83 to %5 step %c1_84 {
          %14 = aie.objectfifo.acquire @Y_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_7_0, %rtp_7_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Y_L1L2_7_0(Produce, 1)
        }
        %c3 = arith.constant 3 : index
        %6 = memref.load %rtp_7_0[%c3] : memref<12xi32>
        %c0_85 = arith.constant 0 : index
        %7 = arith.index_cast %6 : i32 to index
        %c1_86 = arith.constant 1 : index
        scf.for %arg1 = %c0_85 to %7 step %c1_86 {
          %14 = aie.objectfifo.acquire @A_L2L1_7_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_apply(%15, %17, %par_7_0, %rtp_7_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_7_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_7_0(Produce, 1)
        }
        %c4 = arith.constant 4 : index
        %8 = memref.load %rtp_7_0[%c4] : memref<12xi32>
        %c0_87 = arith.constant 0 : index
        %9 = arith.index_cast %8 : i32 to index
        %c1_88 = arith.constant 1 : index
        scf.for %arg1 = %c0_87 to %9 step %c1_88 {
          %14 = aie.objectfifo.acquire @A_L2L1_7_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @B_L2L1_7_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %18 = aie.objectfifo.acquire @Y_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_add(%15, %17, %19, %par_7_0, %rtp_7_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_7_0(Consume, 1)
          aie.objectfifo.release @B_L2L1_7_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_7_0(Produce, 1)
        }
        %c5 = arith.constant 5 : index
        %10 = memref.load %rtp_7_0[%c5] : memref<12xi32>
        %c0_89 = arith.constant 0 : index
        %11 = arith.index_cast %10 : i32 to index
        %c1_90 = arith.constant 1 : index
        scf.for %arg1 = %c0_89 to %11 step %c1_90 {
          %14 = aie.objectfifo.acquire @Z_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_7_0, %rtp_7_0) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Z_L1L2_7_0(Produce, 1)
        }
        %c6 = arith.constant 6 : index
        %12 = memref.load %rtp_7_0[%c6] : memref<12xi32>
        %c0_91 = arith.constant 0 : index
        %13 = arith.index_cast %12 : i32 to index
        %c1_92 = arith.constant 1 : index
        scf.for %arg1 = %c0_91 to %13 step %c1_92 {
          %14 = aie.objectfifo.acquire @A_L2L1_7_0(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_rgba(%15, %17, %rtp_7_0) : (memref<4096xbf16>, memref<4096xbf16>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_7_0(Consume, 1)
          aie.objectfifo.release @Y_L1L2_7_0(Produce, 1)
        }
        %c1_i32_93 = arith.constant 1 : i32
        aie.use_lock(%lock_7_2, Release, %c1_i32_93)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %lock_7_3 = aie.lock(%logical_core_14)
    %core_7_3 = aie.core(%logical_core_14) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_7_3, Acquire, %c1_i32)
        func.call @vew_begin(%par_7_1, %rtp_7_1) : (memref<2624xf32>, memref<12xi32>) -> ()
        %c0_77 = arith.constant 0 : index
        %0 = memref.load %rtp_7_1[%c0_77] : memref<12xi32>
        %c0_78 = arith.constant 0 : index
        %1 = arith.index_cast %0 : i32 to index
        %c1_79 = arith.constant 1 : index
        scf.for %arg1 = %c0_78 to %1 step %c1_79 {
          %14 = aie.objectfifo.acquire @A_L2L1_7_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = arith.index_cast %arg1 : index to i32
          %c0_i32 = arith.constant 0 : i32
          func.call @vew_param(%15, %par_7_1, %16, %c0_i32, %rtp_7_1) : (memref<4096xbf16>, memref<2624xf32>, i32, i32, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_7_1(Consume, 1)
        }
        %c1_80 = arith.constant 1 : index
        %2 = memref.load %rtp_7_1[%c1_80] : memref<12xi32>
        %c0_81 = arith.constant 0 : index
        %3 = arith.index_cast %2 : i32 to index
        %c1_82 = arith.constant 1 : index
        scf.for %arg1 = %c0_81 to %3 step %c1_82 {
          %14 = aie.objectfifo.acquire @A_L2L1_7_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats(%15, %par_7_1, %rtp_7_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_7_1(Consume, 1)
        }
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtp_7_1[%c2] : memref<12xi32>
        %c0_83 = arith.constant 0 : index
        %5 = arith.index_cast %4 : i32 to index
        %c1_84 = arith.constant 1 : index
        scf.for %arg1 = %c0_83 to %5 step %c1_84 {
          %14 = aie.objectfifo.acquire @Y_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_7_1, %rtp_7_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Y_L1L2_7_1(Produce, 1)
        }
        %c3 = arith.constant 3 : index
        %6 = memref.load %rtp_7_1[%c3] : memref<12xi32>
        %c0_85 = arith.constant 0 : index
        %7 = arith.index_cast %6 : i32 to index
        %c1_86 = arith.constant 1 : index
        scf.for %arg1 = %c0_85 to %7 step %c1_86 {
          %14 = aie.objectfifo.acquire @A_L2L1_7_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_apply(%15, %17, %par_7_1, %rtp_7_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_7_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_7_1(Produce, 1)
        }
        %c4 = arith.constant 4 : index
        %8 = memref.load %rtp_7_1[%c4] : memref<12xi32>
        %c0_87 = arith.constant 0 : index
        %9 = arith.index_cast %8 : i32 to index
        %c1_88 = arith.constant 1 : index
        scf.for %arg1 = %c0_87 to %9 step %c1_88 {
          %14 = aie.objectfifo.acquire @A_L2L1_7_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @B_L2L1_7_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %18 = aie.objectfifo.acquire @Y_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_add(%15, %17, %19, %par_7_1, %rtp_7_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_7_1(Consume, 1)
          aie.objectfifo.release @B_L2L1_7_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_7_1(Produce, 1)
        }
        %c5 = arith.constant 5 : index
        %10 = memref.load %rtp_7_1[%c5] : memref<12xi32>
        %c0_89 = arith.constant 0 : index
        %11 = arith.index_cast %10 : i32 to index
        %c1_90 = arith.constant 1 : index
        scf.for %arg1 = %c0_89 to %11 step %c1_90 {
          %14 = aie.objectfifo.acquire @Z_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_stats_out(%15, %par_7_1, %rtp_7_1) : (memref<4096xbf16>, memref<2624xf32>, memref<12xi32>) -> ()
          aie.objectfifo.release @Z_L1L2_7_1(Produce, 1)
        }
        %c6 = arith.constant 6 : index
        %12 = memref.load %rtp_7_1[%c6] : memref<12xi32>
        %c0_91 = arith.constant 0 : index
        %13 = arith.index_cast %12 : i32 to index
        %c1_92 = arith.constant 1 : index
        scf.for %arg1 = %c0_91 to %13 step %c1_92 {
          %14 = aie.objectfifo.acquire @A_L2L1_7_1(Consume, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          %16 = aie.objectfifo.acquire @Y_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
          func.call @vew_rgba(%15, %17, %rtp_7_1) : (memref<4096xbf16>, memref<4096xbf16>, memref<12xi32>) -> ()
          aie.objectfifo.release @A_L2L1_7_1(Consume, 1)
          aie.objectfifo.release @Y_L1L2_7_1(Produce, 1)
        }
        %c1_i32_93 = arith.constant 1 : i32
        aie.use_lock(%lock_7_3, Release, %c1_i32_93)
      }
      aie.end
    } {stack_size = 2048 : i32}
    aie.runtime_sequence @vae512_vew05(%arg0: memref<4460544xbf16>, %arg1: memref<4460544xbf16>, %arg2: memref<77824xbf16>, %arg3: memref<4460544xbf16>) {
      %c0_i32 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_0, 0, %c0_i32) : i32
      %c0_i32_77 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_0, 1, %c0_i32_77) : i32
      %c0_i32_78 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_0, 2, %c0_i32_78) : i32
      %c0_i32_79 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_0, 3, %c0_i32_79) : i32
      %c32_i32 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_0_0, 4, %c32_i32) : i32
      %c0_i32_80 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_0, 5, %c0_i32_80) : i32
      %c0_i32_81 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_0, 6, %c0_i32_81) : i32
      %c512_i32 = arith.constant 512 : i32
      aiex.npu.rtp_write(@rtp_0_0, 7, %c512_i32) : i32
      %c0_i32_82 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_0, 8, %c0_i32_82) : i32
      %c0_i32_83 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_0, 9, %c0_i32_83) : i32
      %c4096_i32 = arith.constant 4096 : i32
      aiex.npu.rtp_write(@rtp_0_0, 10, %c4096_i32) : i32
      %c0_i32_84 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_0, 11, %c0_i32_84) : i32
      %c0_i32_85 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_1, 0, %c0_i32_85) : i32
      %c0_i32_86 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_1, 1, %c0_i32_86) : i32
      %c0_i32_87 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_1, 2, %c0_i32_87) : i32
      %c0_i32_88 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_1, 3, %c0_i32_88) : i32
      %c32_i32_89 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_0_1, 4, %c32_i32_89) : i32
      %c0_i32_90 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_1, 5, %c0_i32_90) : i32
      %c0_i32_91 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_1, 6, %c0_i32_91) : i32
      %c512_i32_92 = arith.constant 512 : i32
      aiex.npu.rtp_write(@rtp_0_1, 7, %c512_i32_92) : i32
      %c0_i32_93 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_1, 8, %c0_i32_93) : i32
      %c0_i32_94 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_1, 9, %c0_i32_94) : i32
      %c4096_i32_95 = arith.constant 4096 : i32
      aiex.npu.rtp_write(@rtp_0_1, 10, %c4096_i32_95) : i32
      %c0_i32_96 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_1, 11, %c0_i32_96) : i32
      %c0_i32_97 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_0, 0, %c0_i32_97) : i32
      %c0_i32_98 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_0, 1, %c0_i32_98) : i32
      %c0_i32_99 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_0, 2, %c0_i32_99) : i32
      %c0_i32_100 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_0, 3, %c0_i32_100) : i32
      %c32_i32_101 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_1_0, 4, %c32_i32_101) : i32
      %c0_i32_102 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_0, 5, %c0_i32_102) : i32
      %c0_i32_103 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_0, 6, %c0_i32_103) : i32
      %c512_i32_104 = arith.constant 512 : i32
      aiex.npu.rtp_write(@rtp_1_0, 7, %c512_i32_104) : i32
      %c0_i32_105 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_0, 8, %c0_i32_105) : i32
      %c0_i32_106 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_0, 9, %c0_i32_106) : i32
      %c4096_i32_107 = arith.constant 4096 : i32
      aiex.npu.rtp_write(@rtp_1_0, 10, %c4096_i32_107) : i32
      %c0_i32_108 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_0, 11, %c0_i32_108) : i32
      %c0_i32_109 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_1, 0, %c0_i32_109) : i32
      %c0_i32_110 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_1, 1, %c0_i32_110) : i32
      %c0_i32_111 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_1, 2, %c0_i32_111) : i32
      %c0_i32_112 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_1, 3, %c0_i32_112) : i32
      %c32_i32_113 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_1_1, 4, %c32_i32_113) : i32
      %c0_i32_114 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_1, 5, %c0_i32_114) : i32
      %c0_i32_115 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_1, 6, %c0_i32_115) : i32
      %c512_i32_116 = arith.constant 512 : i32
      aiex.npu.rtp_write(@rtp_1_1, 7, %c512_i32_116) : i32
      %c0_i32_117 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_1, 8, %c0_i32_117) : i32
      %c0_i32_118 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_1, 9, %c0_i32_118) : i32
      %c4096_i32_119 = arith.constant 4096 : i32
      aiex.npu.rtp_write(@rtp_1_1, 10, %c4096_i32_119) : i32
      %c0_i32_120 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_1, 11, %c0_i32_120) : i32
      %c0_i32_121 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_2_0, 0, %c0_i32_121) : i32
      %c0_i32_122 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_2_0, 1, %c0_i32_122) : i32
      %c0_i32_123 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_2_0, 2, %c0_i32_123) : i32
      %c0_i32_124 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_2_0, 3, %c0_i32_124) : i32
      %c32_i32_125 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_2_0, 4, %c32_i32_125) : i32
      %c0_i32_126 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_2_0, 5, %c0_i32_126) : i32
      %c0_i32_127 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_2_0, 6, %c0_i32_127) : i32
      %c512_i32_128 = arith.constant 512 : i32
      aiex.npu.rtp_write(@rtp_2_0, 7, %c512_i32_128) : i32
      %c0_i32_129 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_2_0, 8, %c0_i32_129) : i32
      %c0_i32_130 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_2_0, 9, %c0_i32_130) : i32
      %c4096_i32_131 = arith.constant 4096 : i32
      aiex.npu.rtp_write(@rtp_2_0, 10, %c4096_i32_131) : i32
      %c0_i32_132 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_2_0, 11, %c0_i32_132) : i32
      %c0_i32_133 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_2_1, 0, %c0_i32_133) : i32
      %c0_i32_134 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_2_1, 1, %c0_i32_134) : i32
      %c0_i32_135 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_2_1, 2, %c0_i32_135) : i32
      %c0_i32_136 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_2_1, 3, %c0_i32_136) : i32
      %c32_i32_137 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_2_1, 4, %c32_i32_137) : i32
      %c0_i32_138 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_2_1, 5, %c0_i32_138) : i32
      %c0_i32_139 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_2_1, 6, %c0_i32_139) : i32
      %c512_i32_140 = arith.constant 512 : i32
      aiex.npu.rtp_write(@rtp_2_1, 7, %c512_i32_140) : i32
      %c0_i32_141 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_2_1, 8, %c0_i32_141) : i32
      %c0_i32_142 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_2_1, 9, %c0_i32_142) : i32
      %c4096_i32_143 = arith.constant 4096 : i32
      aiex.npu.rtp_write(@rtp_2_1, 10, %c4096_i32_143) : i32
      %c0_i32_144 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_2_1, 11, %c0_i32_144) : i32
      %c0_i32_145 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_3_0, 0, %c0_i32_145) : i32
      %c0_i32_146 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_3_0, 1, %c0_i32_146) : i32
      %c0_i32_147 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_3_0, 2, %c0_i32_147) : i32
      %c0_i32_148 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_3_0, 3, %c0_i32_148) : i32
      %c32_i32_149 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_3_0, 4, %c32_i32_149) : i32
      %c0_i32_150 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_3_0, 5, %c0_i32_150) : i32
      %c0_i32_151 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_3_0, 6, %c0_i32_151) : i32
      %c512_i32_152 = arith.constant 512 : i32
      aiex.npu.rtp_write(@rtp_3_0, 7, %c512_i32_152) : i32
      %c0_i32_153 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_3_0, 8, %c0_i32_153) : i32
      %c0_i32_154 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_3_0, 9, %c0_i32_154) : i32
      %c4096_i32_155 = arith.constant 4096 : i32
      aiex.npu.rtp_write(@rtp_3_0, 10, %c4096_i32_155) : i32
      %c0_i32_156 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_3_0, 11, %c0_i32_156) : i32
      %c0_i32_157 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_3_1, 0, %c0_i32_157) : i32
      %c0_i32_158 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_3_1, 1, %c0_i32_158) : i32
      %c0_i32_159 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_3_1, 2, %c0_i32_159) : i32
      %c0_i32_160 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_3_1, 3, %c0_i32_160) : i32
      %c32_i32_161 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_3_1, 4, %c32_i32_161) : i32
      %c0_i32_162 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_3_1, 5, %c0_i32_162) : i32
      %c0_i32_163 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_3_1, 6, %c0_i32_163) : i32
      %c512_i32_164 = arith.constant 512 : i32
      aiex.npu.rtp_write(@rtp_3_1, 7, %c512_i32_164) : i32
      %c0_i32_165 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_3_1, 8, %c0_i32_165) : i32
      %c0_i32_166 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_3_1, 9, %c0_i32_166) : i32
      %c4096_i32_167 = arith.constant 4096 : i32
      aiex.npu.rtp_write(@rtp_3_1, 10, %c4096_i32_167) : i32
      %c0_i32_168 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_3_1, 11, %c0_i32_168) : i32
      %c0_i32_169 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_4_0, 0, %c0_i32_169) : i32
      %c0_i32_170 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_4_0, 1, %c0_i32_170) : i32
      %c0_i32_171 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_4_0, 2, %c0_i32_171) : i32
      %c0_i32_172 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_4_0, 3, %c0_i32_172) : i32
      %c32_i32_173 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_4_0, 4, %c32_i32_173) : i32
      %c0_i32_174 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_4_0, 5, %c0_i32_174) : i32
      %c0_i32_175 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_4_0, 6, %c0_i32_175) : i32
      %c512_i32_176 = arith.constant 512 : i32
      aiex.npu.rtp_write(@rtp_4_0, 7, %c512_i32_176) : i32
      %c0_i32_177 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_4_0, 8, %c0_i32_177) : i32
      %c0_i32_178 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_4_0, 9, %c0_i32_178) : i32
      %c4096_i32_179 = arith.constant 4096 : i32
      aiex.npu.rtp_write(@rtp_4_0, 10, %c4096_i32_179) : i32
      %c0_i32_180 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_4_0, 11, %c0_i32_180) : i32
      %c0_i32_181 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_4_1, 0, %c0_i32_181) : i32
      %c0_i32_182 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_4_1, 1, %c0_i32_182) : i32
      %c0_i32_183 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_4_1, 2, %c0_i32_183) : i32
      %c0_i32_184 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_4_1, 3, %c0_i32_184) : i32
      %c32_i32_185 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_4_1, 4, %c32_i32_185) : i32
      %c0_i32_186 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_4_1, 5, %c0_i32_186) : i32
      %c0_i32_187 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_4_1, 6, %c0_i32_187) : i32
      %c512_i32_188 = arith.constant 512 : i32
      aiex.npu.rtp_write(@rtp_4_1, 7, %c512_i32_188) : i32
      %c0_i32_189 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_4_1, 8, %c0_i32_189) : i32
      %c0_i32_190 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_4_1, 9, %c0_i32_190) : i32
      %c4096_i32_191 = arith.constant 4096 : i32
      aiex.npu.rtp_write(@rtp_4_1, 10, %c4096_i32_191) : i32
      %c0_i32_192 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_4_1, 11, %c0_i32_192) : i32
      %c0_i32_193 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_5_0, 0, %c0_i32_193) : i32
      %c0_i32_194 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_5_0, 1, %c0_i32_194) : i32
      %c0_i32_195 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_5_0, 2, %c0_i32_195) : i32
      %c0_i32_196 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_5_0, 3, %c0_i32_196) : i32
      %c32_i32_197 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_5_0, 4, %c32_i32_197) : i32
      %c0_i32_198 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_5_0, 5, %c0_i32_198) : i32
      %c0_i32_199 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_5_0, 6, %c0_i32_199) : i32
      %c512_i32_200 = arith.constant 512 : i32
      aiex.npu.rtp_write(@rtp_5_0, 7, %c512_i32_200) : i32
      %c0_i32_201 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_5_0, 8, %c0_i32_201) : i32
      %c0_i32_202 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_5_0, 9, %c0_i32_202) : i32
      %c4096_i32_203 = arith.constant 4096 : i32
      aiex.npu.rtp_write(@rtp_5_0, 10, %c4096_i32_203) : i32
      %c0_i32_204 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_5_0, 11, %c0_i32_204) : i32
      %c0_i32_205 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_5_1, 0, %c0_i32_205) : i32
      %c0_i32_206 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_5_1, 1, %c0_i32_206) : i32
      %c0_i32_207 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_5_1, 2, %c0_i32_207) : i32
      %c0_i32_208 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_5_1, 3, %c0_i32_208) : i32
      %c32_i32_209 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_5_1, 4, %c32_i32_209) : i32
      %c0_i32_210 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_5_1, 5, %c0_i32_210) : i32
      %c0_i32_211 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_5_1, 6, %c0_i32_211) : i32
      %c512_i32_212 = arith.constant 512 : i32
      aiex.npu.rtp_write(@rtp_5_1, 7, %c512_i32_212) : i32
      %c0_i32_213 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_5_1, 8, %c0_i32_213) : i32
      %c0_i32_214 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_5_1, 9, %c0_i32_214) : i32
      %c4096_i32_215 = arith.constant 4096 : i32
      aiex.npu.rtp_write(@rtp_5_1, 10, %c4096_i32_215) : i32
      %c0_i32_216 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_5_1, 11, %c0_i32_216) : i32
      %c0_i32_217 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_6_0, 0, %c0_i32_217) : i32
      %c0_i32_218 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_6_0, 1, %c0_i32_218) : i32
      %c0_i32_219 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_6_0, 2, %c0_i32_219) : i32
      %c0_i32_220 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_6_0, 3, %c0_i32_220) : i32
      %c32_i32_221 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_6_0, 4, %c32_i32_221) : i32
      %c0_i32_222 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_6_0, 5, %c0_i32_222) : i32
      %c0_i32_223 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_6_0, 6, %c0_i32_223) : i32
      %c512_i32_224 = arith.constant 512 : i32
      aiex.npu.rtp_write(@rtp_6_0, 7, %c512_i32_224) : i32
      %c0_i32_225 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_6_0, 8, %c0_i32_225) : i32
      %c0_i32_226 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_6_0, 9, %c0_i32_226) : i32
      %c4096_i32_227 = arith.constant 4096 : i32
      aiex.npu.rtp_write(@rtp_6_0, 10, %c4096_i32_227) : i32
      %c0_i32_228 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_6_0, 11, %c0_i32_228) : i32
      %c0_i32_229 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_6_1, 0, %c0_i32_229) : i32
      %c0_i32_230 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_6_1, 1, %c0_i32_230) : i32
      %c0_i32_231 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_6_1, 2, %c0_i32_231) : i32
      %c0_i32_232 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_6_1, 3, %c0_i32_232) : i32
      %c32_i32_233 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_6_1, 4, %c32_i32_233) : i32
      %c0_i32_234 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_6_1, 5, %c0_i32_234) : i32
      %c0_i32_235 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_6_1, 6, %c0_i32_235) : i32
      %c512_i32_236 = arith.constant 512 : i32
      aiex.npu.rtp_write(@rtp_6_1, 7, %c512_i32_236) : i32
      %c0_i32_237 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_6_1, 8, %c0_i32_237) : i32
      %c0_i32_238 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_6_1, 9, %c0_i32_238) : i32
      %c4096_i32_239 = arith.constant 4096 : i32
      aiex.npu.rtp_write(@rtp_6_1, 10, %c4096_i32_239) : i32
      %c0_i32_240 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_6_1, 11, %c0_i32_240) : i32
      %c0_i32_241 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_7_0, 0, %c0_i32_241) : i32
      %c0_i32_242 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_7_0, 1, %c0_i32_242) : i32
      %c0_i32_243 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_7_0, 2, %c0_i32_243) : i32
      %c0_i32_244 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_7_0, 3, %c0_i32_244) : i32
      %c32_i32_245 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_7_0, 4, %c32_i32_245) : i32
      %c0_i32_246 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_7_0, 5, %c0_i32_246) : i32
      %c0_i32_247 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_7_0, 6, %c0_i32_247) : i32
      %c512_i32_248 = arith.constant 512 : i32
      aiex.npu.rtp_write(@rtp_7_0, 7, %c512_i32_248) : i32
      %c0_i32_249 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_7_0, 8, %c0_i32_249) : i32
      %c0_i32_250 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_7_0, 9, %c0_i32_250) : i32
      %c4096_i32_251 = arith.constant 4096 : i32
      aiex.npu.rtp_write(@rtp_7_0, 10, %c4096_i32_251) : i32
      %c0_i32_252 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_7_0, 11, %c0_i32_252) : i32
      %c0_i32_253 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_7_1, 0, %c0_i32_253) : i32
      %c0_i32_254 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_7_1, 1, %c0_i32_254) : i32
      %c0_i32_255 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_7_1, 2, %c0_i32_255) : i32
      %c0_i32_256 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_7_1, 3, %c0_i32_256) : i32
      %c32_i32_257 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_7_1, 4, %c32_i32_257) : i32
      %c0_i32_258 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_7_1, 5, %c0_i32_258) : i32
      %c0_i32_259 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_7_1, 6, %c0_i32_259) : i32
      %c512_i32_260 = arith.constant 512 : i32
      aiex.npu.rtp_write(@rtp_7_1, 7, %c512_i32_260) : i32
      %c0_i32_261 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_7_1, 8, %c0_i32_261) : i32
      %c0_i32_262 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_7_1, 9, %c0_i32_262) : i32
      %c4096_i32_263 = arith.constant 4096 : i32
      aiex.npu.rtp_write(@rtp_7_1, 10, %c4096_i32_263) : i32
      %c0_i32_264 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_7_1, 11, %c0_i32_264) : i32
      aiex.set_lock(%lock_0_2, 1)
      aiex.set_lock(%lock_0_3, 1)
      aiex.set_lock(%lock_1_2, 1)
      aiex.set_lock(%lock_1_3, 1)
      aiex.set_lock(%lock_2_2, 1)
      aiex.set_lock(%lock_2_3, 1)
      aiex.set_lock(%lock_3_2, 1)
      aiex.set_lock(%lock_3_3, 1)
      aiex.set_lock(%lock_4_2, 1)
      aiex.set_lock(%lock_4_3, 1)
      aiex.set_lock(%lock_5_2, 1)
      aiex.set_lock(%lock_5_3, 1)
      aiex.set_lock(%lock_6_2, 1)
      aiex.set_lock(%lock_6_3, 1)
      aiex.set_lock(%lock_7_2, 1)
      aiex.set_lock(%lock_7_3, 1)
      %0 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4460544xbf16> offset = 68096 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4460544xbf16> offset = 68096 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @Y_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4460544xbf16> offset = 68096 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4460544xbf16> offset = 608768 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4460544xbf16> offset = 608768 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @Y_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4460544xbf16> offset = 608768 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<4460544xbf16> offset = 1149440 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4460544xbf16> offset = 1149440 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @Y_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4460544xbf16> offset = 1149440 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<4460544xbf16> offset = 1690112 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4460544xbf16> offset = 1690112 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @Y_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4460544xbf16> offset = 1690112 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @A_L3L2_4 {
        aie.dma_bd(%arg0 : memref<4460544xbf16> offset = 2230784 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4460544xbf16> offset = 2230784 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @Y_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4460544xbf16> offset = 2230784 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @A_L3L2_5 {
        aie.dma_bd(%arg0 : memref<4460544xbf16> offset = 2771456 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4460544xbf16> offset = 2771456 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @Y_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4460544xbf16> offset = 2771456 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @A_L3L2_6 {
        aie.dma_bd(%arg0 : memref<4460544xbf16> offset = 3312128 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4460544xbf16> offset = 3312128 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @Y_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4460544xbf16> offset = 3312128 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @A_L3L2_7 {
        aie.dma_bd(%arg0 : memref<4460544xbf16> offset = 3852800 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4460544xbf16> offset = 3852800 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @Y_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4460544xbf16> offset = 3852800 len = 262144 sizes = [1, 8, 32, 1024] strides = [0, 67584, 1024, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      aiex.dma_await_task(%2)
      aiex.dma_await_task(%5)
      aiex.dma_await_task(%8)
      aiex.dma_await_task(%11)
      aiex.dma_await_task(%14)
      aiex.dma_await_task(%17)
      aiex.dma_await_task(%20)
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
    }
  }
}

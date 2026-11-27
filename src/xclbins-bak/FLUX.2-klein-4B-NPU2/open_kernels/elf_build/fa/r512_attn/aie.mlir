module {
  aie.device(npu2) @main {
    aie.runtime_sequence @cfg_fa() {
      aiex.configure @fa {
      }
    }
  }
  aie.device(npu2) @fa {
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
    %logical_mem_31 = aie.logical_tile<MemTile>(4, 1)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(0, 0)
    %logical_shim_noc_32 = aie.logical_tile<ShimNOCTile>(4, 0)
    %logical_mem_33 = aie.logical_tile<MemTile>(1, 1)
    %logical_mem_34 = aie.logical_tile<MemTile>(5, 1)
    %logical_shim_noc_35 = aie.logical_tile<ShimNOCTile>(1, 0)
    %logical_shim_noc_36 = aie.logical_tile<ShimNOCTile>(5, 0)
    %logical_mem_37 = aie.logical_tile<MemTile>(0, 1)
    %logical_mem_38 = aie.logical_tile<MemTile>(1, 1)
    %logical_mem_39 = aie.logical_tile<MemTile>(2, 1)
    %logical_mem_40 = aie.logical_tile<MemTile>(3, 1)
    %logical_mem_41 = aie.logical_tile<MemTile>(4, 1)
    %logical_mem_42 = aie.logical_tile<MemTile>(5, 1)
    %logical_mem_43 = aie.logical_tile<MemTile>(6, 1)
    %logical_mem_44 = aie.logical_tile<MemTile>(7, 1)
    %logical_shim_noc_45 = aie.logical_tile<ShimNOCTile>(0, 0)
    %logical_shim_noc_46 = aie.logical_tile<ShimNOCTile>(1, 0)
    %logical_shim_noc_47 = aie.logical_tile<ShimNOCTile>(2, 0)
    %logical_shim_noc_48 = aie.logical_tile<ShimNOCTile>(3, 0)
    %logical_shim_noc_49 = aie.logical_tile<ShimNOCTile>(4, 0)
    %logical_shim_noc_50 = aie.logical_tile<ShimNOCTile>(5, 0)
    %logical_shim_noc_51 = aie.logical_tile<ShimNOCTile>(6, 0)
    %logical_shim_noc_52 = aie.logical_tile<ShimNOCTile>(7, 0)
    aie.objectfifo @A_L2L1_0(%logical_mem dimensionsToStream [<size = 16, stride = 8>, <size = 8, stride = 1024>, <size = 8, stride = 128>, <size = 8, stride = 1>], {%logical_core, %logical_core_0, %logical_core_1, %logical_core_2, %logical_core_3, %logical_core_4, %logical_core_5, %logical_core_6, %logical_core_7, %logical_core_8, %logical_core_9, %logical_core_10, %logical_core_11, %logical_core_12, %logical_core_13, %logical_core_14}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>>  
    aie.objectfifo @A_L3L2_0(%logical_shim_noc, {%logical_mem}, 2 : i32) : !aie.objectfifo<memref<64x128xbf16>>  
    aie.objectfifo.link [@A_L3L2_0] -> [@A_L2L1_0]([] [0])
    aie.objectfifo @A_L2L1_1(%logical_mem_31 dimensionsToStream [<size = 16, stride = 8>, <size = 8, stride = 1024>, <size = 8, stride = 128>, <size = 8, stride = 1>], {%logical_core_15, %logical_core_16, %logical_core_17, %logical_core_18, %logical_core_19, %logical_core_20, %logical_core_21, %logical_core_22, %logical_core_23, %logical_core_24, %logical_core_25, %logical_core_26, %logical_core_27, %logical_core_28, %logical_core_29, %logical_core_30}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>>  
    aie.objectfifo @A_L3L2_1(%logical_shim_noc_32, {%logical_mem_31}, 2 : i32) : !aie.objectfifo<memref<64x128xbf16>>  
    aie.objectfifo.link [@A_L3L2_1] -> [@A_L2L1_1]([] [0])
    aie.objectfifo @B_L2L1_0(%logical_mem_33 dimensionsToStream [<size = 16, stride = 8>, <size = 8, stride = 1024>, <size = 8, stride = 128>, <size = 8, stride = 1>], {%logical_core, %logical_core_0, %logical_core_1, %logical_core_2, %logical_core_3, %logical_core_4, %logical_core_5, %logical_core_6, %logical_core_7, %logical_core_8, %logical_core_9, %logical_core_10, %logical_core_11, %logical_core_12, %logical_core_13, %logical_core_14}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>>  
    aie.objectfifo @B_L3L2_0(%logical_shim_noc_35, {%logical_mem_33}, 2 : i32) : !aie.objectfifo<memref<64x128xbf16>>  
    aie.objectfifo.link [@B_L3L2_0] -> [@B_L2L1_0]([] [0])
    aie.objectfifo @B_L2L1_1(%logical_mem_34 dimensionsToStream [<size = 16, stride = 8>, <size = 8, stride = 1024>, <size = 8, stride = 128>, <size = 8, stride = 1>], {%logical_core_15, %logical_core_16, %logical_core_17, %logical_core_18, %logical_core_19, %logical_core_20, %logical_core_21, %logical_core_22, %logical_core_23, %logical_core_24, %logical_core_25, %logical_core_26, %logical_core_27, %logical_core_28, %logical_core_29, %logical_core_30}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>>  
    aie.objectfifo @B_L3L2_1(%logical_shim_noc_36, {%logical_mem_34}, 2 : i32) : !aie.objectfifo<memref<64x128xbf16>>  
    aie.objectfifo.link [@B_L3L2_1] -> [@B_L2L1_1]([] [0])
    aie.objectfifo @O_L1L2_0_0(%logical_core, {%logical_mem_37}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_0_1(%logical_core_0, {%logical_mem_37}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_0_2(%logical_core_1, {%logical_mem_37}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_0_3(%logical_core_2, {%logical_mem_37}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L2L3_0(%logical_mem_37 dimensionsToStream [<size = 32, stride = 8>, <size = 16, stride = 256>, <size = 8, stride = 1>], {%logical_shim_noc_45}, 2 : i32) : !aie.objectfifo<memref<16384xbf16>>  
    aie.objectfifo.link [@O_L1L2_0_0, @O_L1L2_0_1, @O_L1L2_0_2, @O_L1L2_0_3] -> [@O_L2L3_0]([0, 4096, 8192, 12288] [])
    aie.objectfifo @O_L1L2_1_0(%logical_core_3, {%logical_mem_38}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_1_1(%logical_core_4, {%logical_mem_38}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_1_2(%logical_core_5, {%logical_mem_38}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_1_3(%logical_core_6, {%logical_mem_38}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L2L3_1(%logical_mem_38 dimensionsToStream [<size = 32, stride = 8>, <size = 16, stride = 256>, <size = 8, stride = 1>], {%logical_shim_noc_46}, 2 : i32) : !aie.objectfifo<memref<16384xbf16>>  
    aie.objectfifo.link [@O_L1L2_1_0, @O_L1L2_1_1, @O_L1L2_1_2, @O_L1L2_1_3] -> [@O_L2L3_1]([0, 4096, 8192, 12288] [])
    aie.objectfifo @O_L1L2_2_0(%logical_core_7, {%logical_mem_39}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_2_1(%logical_core_8, {%logical_mem_39}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_2_2(%logical_core_9, {%logical_mem_39}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_2_3(%logical_core_10, {%logical_mem_39}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L2L3_2(%logical_mem_39 dimensionsToStream [<size = 32, stride = 8>, <size = 16, stride = 256>, <size = 8, stride = 1>], {%logical_shim_noc_47}, 2 : i32) : !aie.objectfifo<memref<16384xbf16>>  
    aie.objectfifo.link [@O_L1L2_2_0, @O_L1L2_2_1, @O_L1L2_2_2, @O_L1L2_2_3] -> [@O_L2L3_2]([0, 4096, 8192, 12288] [])
    aie.objectfifo @O_L1L2_3_0(%logical_core_11, {%logical_mem_40}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_3_1(%logical_core_12, {%logical_mem_40}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_3_2(%logical_core_13, {%logical_mem_40}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_3_3(%logical_core_14, {%logical_mem_40}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L2L3_3(%logical_mem_40 dimensionsToStream [<size = 32, stride = 8>, <size = 16, stride = 256>, <size = 8, stride = 1>], {%logical_shim_noc_48}, 2 : i32) : !aie.objectfifo<memref<16384xbf16>>  
    aie.objectfifo.link [@O_L1L2_3_0, @O_L1L2_3_1, @O_L1L2_3_2, @O_L1L2_3_3] -> [@O_L2L3_3]([0, 4096, 8192, 12288] [])
    aie.objectfifo @O_L1L2_4_0(%logical_core_15, {%logical_mem_41}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_4_1(%logical_core_16, {%logical_mem_41}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_4_2(%logical_core_17, {%logical_mem_41}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_4_3(%logical_core_18, {%logical_mem_41}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L2L3_4(%logical_mem_41 dimensionsToStream [<size = 32, stride = 8>, <size = 16, stride = 256>, <size = 8, stride = 1>], {%logical_shim_noc_49}, 2 : i32) : !aie.objectfifo<memref<16384xbf16>>  
    aie.objectfifo.link [@O_L1L2_4_0, @O_L1L2_4_1, @O_L1L2_4_2, @O_L1L2_4_3] -> [@O_L2L3_4]([0, 4096, 8192, 12288] [])
    aie.objectfifo @O_L1L2_5_0(%logical_core_19, {%logical_mem_42}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_5_1(%logical_core_20, {%logical_mem_42}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_5_2(%logical_core_21, {%logical_mem_42}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_5_3(%logical_core_22, {%logical_mem_42}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L2L3_5(%logical_mem_42 dimensionsToStream [<size = 32, stride = 8>, <size = 16, stride = 256>, <size = 8, stride = 1>], {%logical_shim_noc_50}, 2 : i32) : !aie.objectfifo<memref<16384xbf16>>  
    aie.objectfifo.link [@O_L1L2_5_0, @O_L1L2_5_1, @O_L1L2_5_2, @O_L1L2_5_3] -> [@O_L2L3_5]([0, 4096, 8192, 12288] [])
    aie.objectfifo @O_L1L2_6_0(%logical_core_23, {%logical_mem_43}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_6_1(%logical_core_24, {%logical_mem_43}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_6_2(%logical_core_25, {%logical_mem_43}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_6_3(%logical_core_26, {%logical_mem_43}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L2L3_6(%logical_mem_43 dimensionsToStream [<size = 32, stride = 8>, <size = 16, stride = 256>, <size = 8, stride = 1>], {%logical_shim_noc_51}, 2 : i32) : !aie.objectfifo<memref<16384xbf16>>  
    aie.objectfifo.link [@O_L1L2_6_0, @O_L1L2_6_1, @O_L1L2_6_2, @O_L1L2_6_3] -> [@O_L2L3_6]([0, 4096, 8192, 12288] [])
    aie.objectfifo @O_L1L2_7_0(%logical_core_27, {%logical_mem_44}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_7_1(%logical_core_28, {%logical_mem_44}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_7_2(%logical_core_29, {%logical_mem_44}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L1L2_7_3(%logical_core_30, {%logical_mem_44}, 1 : i32) : !aie.objectfifo<memref<4096xbf16>>  
    aie.objectfifo @O_L2L3_7(%logical_mem_44 dimensionsToStream [<size = 32, stride = 8>, <size = 16, stride = 256>, <size = 8, stride = 1>], {%logical_shim_noc_52}, 2 : i32) : !aie.objectfifo<memref<16384xbf16>>  
    aie.objectfifo.link [@O_L1L2_7_0, @O_L1L2_7_1, @O_L1L2_7_2, @O_L1L2_7_3] -> [@O_L2L3_7]([0, 4096, 8192, 12288] [])
    func.func private @capture_q(memref<64x64xbf16>, memref<4096xbf16>, i32, i32) attributes {link_with = "zero_g.o"}
    func.func private @pass_init(memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) attributes {link_with = "zero_g.o"}
    func.func private @zero_g(memref<2048xbf16>) attributes {link_with = "zero_g.o"}
    func.func private @qk(memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) attributes {link_with = "zero_g.o"}
    func.func private @pv(memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) attributes {link_with = "zero_g.o"}
    func.func private @apply_mask(memref<2048xbf16>, memref<8xi32>, i32, i32) attributes {link_with = "zero_g.o"}
    func.func private @softmax_step(memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) attributes {link_with = "zero_g.o"}
    func.func private @finalize(memref<256xf32>, memref<4096xbf16>) attributes {link_with = "zero_g.o"}
    %q_saved_0_0 = aie.buffer(%logical_core) {sym_name = "q_saved_0_0"} : memref<4096xbf16> 
    %g_0_0 = aie.buffer(%logical_core) {sym_name = "g_0_0"} : memref<2048xbf16> 
    %up_0_0 = aie.buffer(%logical_core) {sym_name = "up_0_0"} : memref<32xbf16> 
    %sp_0_0 = aie.buffer(%logical_core) {sym_name = "sp_0_0"} : memref<256xf32> 
    %rtp_0_0 = aie.buffer(%logical_core) {sym_name = "rtp_0_0"} : memref<8xi32> = dense<0>
    %q_saved_0_1 = aie.buffer(%logical_core_0) {sym_name = "q_saved_0_1"} : memref<4096xbf16> 
    %g_0_1 = aie.buffer(%logical_core_0) {sym_name = "g_0_1"} : memref<2048xbf16> 
    %up_0_1 = aie.buffer(%logical_core_0) {sym_name = "up_0_1"} : memref<32xbf16> 
    %sp_0_1 = aie.buffer(%logical_core_0) {sym_name = "sp_0_1"} : memref<256xf32> 
    %rtp_0_1 = aie.buffer(%logical_core_0) {sym_name = "rtp_0_1"} : memref<8xi32> = dense<0>
    %q_saved_0_2 = aie.buffer(%logical_core_1) {sym_name = "q_saved_0_2"} : memref<4096xbf16> 
    %g_0_2 = aie.buffer(%logical_core_1) {sym_name = "g_0_2"} : memref<2048xbf16> 
    %up_0_2 = aie.buffer(%logical_core_1) {sym_name = "up_0_2"} : memref<32xbf16> 
    %sp_0_2 = aie.buffer(%logical_core_1) {sym_name = "sp_0_2"} : memref<256xf32> 
    %rtp_0_2 = aie.buffer(%logical_core_1) {sym_name = "rtp_0_2"} : memref<8xi32> = dense<0>
    %q_saved_0_3 = aie.buffer(%logical_core_2) {sym_name = "q_saved_0_3"} : memref<4096xbf16> 
    %g_0_3 = aie.buffer(%logical_core_2) {sym_name = "g_0_3"} : memref<2048xbf16> 
    %up_0_3 = aie.buffer(%logical_core_2) {sym_name = "up_0_3"} : memref<32xbf16> 
    %sp_0_3 = aie.buffer(%logical_core_2) {sym_name = "sp_0_3"} : memref<256xf32> 
    %rtp_0_3 = aie.buffer(%logical_core_2) {sym_name = "rtp_0_3"} : memref<8xi32> = dense<0>
    %q_saved_0_4 = aie.buffer(%logical_core_3) {sym_name = "q_saved_0_4"} : memref<4096xbf16> 
    %g_0_4 = aie.buffer(%logical_core_3) {sym_name = "g_0_4"} : memref<2048xbf16> 
    %up_0_4 = aie.buffer(%logical_core_3) {sym_name = "up_0_4"} : memref<32xbf16> 
    %sp_0_4 = aie.buffer(%logical_core_3) {sym_name = "sp_0_4"} : memref<256xf32> 
    %rtp_0_4 = aie.buffer(%logical_core_3) {sym_name = "rtp_0_4"} : memref<8xi32> = dense<0>
    %q_saved_0_5 = aie.buffer(%logical_core_4) {sym_name = "q_saved_0_5"} : memref<4096xbf16> 
    %g_0_5 = aie.buffer(%logical_core_4) {sym_name = "g_0_5"} : memref<2048xbf16> 
    %up_0_5 = aie.buffer(%logical_core_4) {sym_name = "up_0_5"} : memref<32xbf16> 
    %sp_0_5 = aie.buffer(%logical_core_4) {sym_name = "sp_0_5"} : memref<256xf32> 
    %rtp_0_5 = aie.buffer(%logical_core_4) {sym_name = "rtp_0_5"} : memref<8xi32> = dense<0>
    %q_saved_0_6 = aie.buffer(%logical_core_5) {sym_name = "q_saved_0_6"} : memref<4096xbf16> 
    %g_0_6 = aie.buffer(%logical_core_5) {sym_name = "g_0_6"} : memref<2048xbf16> 
    %up_0_6 = aie.buffer(%logical_core_5) {sym_name = "up_0_6"} : memref<32xbf16> 
    %sp_0_6 = aie.buffer(%logical_core_5) {sym_name = "sp_0_6"} : memref<256xf32> 
    %rtp_0_6 = aie.buffer(%logical_core_5) {sym_name = "rtp_0_6"} : memref<8xi32> = dense<0>
    %q_saved_0_7 = aie.buffer(%logical_core_6) {sym_name = "q_saved_0_7"} : memref<4096xbf16> 
    %g_0_7 = aie.buffer(%logical_core_6) {sym_name = "g_0_7"} : memref<2048xbf16> 
    %up_0_7 = aie.buffer(%logical_core_6) {sym_name = "up_0_7"} : memref<32xbf16> 
    %sp_0_7 = aie.buffer(%logical_core_6) {sym_name = "sp_0_7"} : memref<256xf32> 
    %rtp_0_7 = aie.buffer(%logical_core_6) {sym_name = "rtp_0_7"} : memref<8xi32> = dense<0>
    %q_saved_0_8 = aie.buffer(%logical_core_7) {sym_name = "q_saved_0_8"} : memref<4096xbf16> 
    %g_0_8 = aie.buffer(%logical_core_7) {sym_name = "g_0_8"} : memref<2048xbf16> 
    %up_0_8 = aie.buffer(%logical_core_7) {sym_name = "up_0_8"} : memref<32xbf16> 
    %sp_0_8 = aie.buffer(%logical_core_7) {sym_name = "sp_0_8"} : memref<256xf32> 
    %rtp_0_8 = aie.buffer(%logical_core_7) {sym_name = "rtp_0_8"} : memref<8xi32> = dense<0>
    %q_saved_0_9 = aie.buffer(%logical_core_8) {sym_name = "q_saved_0_9"} : memref<4096xbf16> 
    %g_0_9 = aie.buffer(%logical_core_8) {sym_name = "g_0_9"} : memref<2048xbf16> 
    %up_0_9 = aie.buffer(%logical_core_8) {sym_name = "up_0_9"} : memref<32xbf16> 
    %sp_0_9 = aie.buffer(%logical_core_8) {sym_name = "sp_0_9"} : memref<256xf32> 
    %rtp_0_9 = aie.buffer(%logical_core_8) {sym_name = "rtp_0_9"} : memref<8xi32> = dense<0>
    %q_saved_0_10 = aie.buffer(%logical_core_9) {sym_name = "q_saved_0_10"} : memref<4096xbf16> 
    %g_0_10 = aie.buffer(%logical_core_9) {sym_name = "g_0_10"} : memref<2048xbf16> 
    %up_0_10 = aie.buffer(%logical_core_9) {sym_name = "up_0_10"} : memref<32xbf16> 
    %sp_0_10 = aie.buffer(%logical_core_9) {sym_name = "sp_0_10"} : memref<256xf32> 
    %rtp_0_10 = aie.buffer(%logical_core_9) {sym_name = "rtp_0_10"} : memref<8xi32> = dense<0>
    %q_saved_0_11 = aie.buffer(%logical_core_10) {sym_name = "q_saved_0_11"} : memref<4096xbf16> 
    %g_0_11 = aie.buffer(%logical_core_10) {sym_name = "g_0_11"} : memref<2048xbf16> 
    %up_0_11 = aie.buffer(%logical_core_10) {sym_name = "up_0_11"} : memref<32xbf16> 
    %sp_0_11 = aie.buffer(%logical_core_10) {sym_name = "sp_0_11"} : memref<256xf32> 
    %rtp_0_11 = aie.buffer(%logical_core_10) {sym_name = "rtp_0_11"} : memref<8xi32> = dense<0>
    %q_saved_0_12 = aie.buffer(%logical_core_11) {sym_name = "q_saved_0_12"} : memref<4096xbf16> 
    %g_0_12 = aie.buffer(%logical_core_11) {sym_name = "g_0_12"} : memref<2048xbf16> 
    %up_0_12 = aie.buffer(%logical_core_11) {sym_name = "up_0_12"} : memref<32xbf16> 
    %sp_0_12 = aie.buffer(%logical_core_11) {sym_name = "sp_0_12"} : memref<256xf32> 
    %rtp_0_12 = aie.buffer(%logical_core_11) {sym_name = "rtp_0_12"} : memref<8xi32> = dense<0>
    %q_saved_0_13 = aie.buffer(%logical_core_12) {sym_name = "q_saved_0_13"} : memref<4096xbf16> 
    %g_0_13 = aie.buffer(%logical_core_12) {sym_name = "g_0_13"} : memref<2048xbf16> 
    %up_0_13 = aie.buffer(%logical_core_12) {sym_name = "up_0_13"} : memref<32xbf16> 
    %sp_0_13 = aie.buffer(%logical_core_12) {sym_name = "sp_0_13"} : memref<256xf32> 
    %rtp_0_13 = aie.buffer(%logical_core_12) {sym_name = "rtp_0_13"} : memref<8xi32> = dense<0>
    %q_saved_0_14 = aie.buffer(%logical_core_13) {sym_name = "q_saved_0_14"} : memref<4096xbf16> 
    %g_0_14 = aie.buffer(%logical_core_13) {sym_name = "g_0_14"} : memref<2048xbf16> 
    %up_0_14 = aie.buffer(%logical_core_13) {sym_name = "up_0_14"} : memref<32xbf16> 
    %sp_0_14 = aie.buffer(%logical_core_13) {sym_name = "sp_0_14"} : memref<256xf32> 
    %rtp_0_14 = aie.buffer(%logical_core_13) {sym_name = "rtp_0_14"} : memref<8xi32> = dense<0>
    %q_saved_0_15 = aie.buffer(%logical_core_14) {sym_name = "q_saved_0_15"} : memref<4096xbf16> 
    %g_0_15 = aie.buffer(%logical_core_14) {sym_name = "g_0_15"} : memref<2048xbf16> 
    %up_0_15 = aie.buffer(%logical_core_14) {sym_name = "up_0_15"} : memref<32xbf16> 
    %sp_0_15 = aie.buffer(%logical_core_14) {sym_name = "sp_0_15"} : memref<256xf32> 
    %rtp_0_15 = aie.buffer(%logical_core_14) {sym_name = "rtp_0_15"} : memref<8xi32> = dense<0>
    %q_saved_1_0 = aie.buffer(%logical_core_15) {sym_name = "q_saved_1_0"} : memref<4096xbf16> 
    %g_1_0 = aie.buffer(%logical_core_15) {sym_name = "g_1_0"} : memref<2048xbf16> 
    %up_1_0 = aie.buffer(%logical_core_15) {sym_name = "up_1_0"} : memref<32xbf16> 
    %sp_1_0 = aie.buffer(%logical_core_15) {sym_name = "sp_1_0"} : memref<256xf32> 
    %rtp_1_0 = aie.buffer(%logical_core_15) {sym_name = "rtp_1_0"} : memref<8xi32> = dense<0>
    %q_saved_1_1 = aie.buffer(%logical_core_16) {sym_name = "q_saved_1_1"} : memref<4096xbf16> 
    %g_1_1 = aie.buffer(%logical_core_16) {sym_name = "g_1_1"} : memref<2048xbf16> 
    %up_1_1 = aie.buffer(%logical_core_16) {sym_name = "up_1_1"} : memref<32xbf16> 
    %sp_1_1 = aie.buffer(%logical_core_16) {sym_name = "sp_1_1"} : memref<256xf32> 
    %rtp_1_1 = aie.buffer(%logical_core_16) {sym_name = "rtp_1_1"} : memref<8xi32> = dense<0>
    %q_saved_1_2 = aie.buffer(%logical_core_17) {sym_name = "q_saved_1_2"} : memref<4096xbf16> 
    %g_1_2 = aie.buffer(%logical_core_17) {sym_name = "g_1_2"} : memref<2048xbf16> 
    %up_1_2 = aie.buffer(%logical_core_17) {sym_name = "up_1_2"} : memref<32xbf16> 
    %sp_1_2 = aie.buffer(%logical_core_17) {sym_name = "sp_1_2"} : memref<256xf32> 
    %rtp_1_2 = aie.buffer(%logical_core_17) {sym_name = "rtp_1_2"} : memref<8xi32> = dense<0>
    %q_saved_1_3 = aie.buffer(%logical_core_18) {sym_name = "q_saved_1_3"} : memref<4096xbf16> 
    %g_1_3 = aie.buffer(%logical_core_18) {sym_name = "g_1_3"} : memref<2048xbf16> 
    %up_1_3 = aie.buffer(%logical_core_18) {sym_name = "up_1_3"} : memref<32xbf16> 
    %sp_1_3 = aie.buffer(%logical_core_18) {sym_name = "sp_1_3"} : memref<256xf32> 
    %rtp_1_3 = aie.buffer(%logical_core_18) {sym_name = "rtp_1_3"} : memref<8xi32> = dense<0>
    %q_saved_1_4 = aie.buffer(%logical_core_19) {sym_name = "q_saved_1_4"} : memref<4096xbf16> 
    %g_1_4 = aie.buffer(%logical_core_19) {sym_name = "g_1_4"} : memref<2048xbf16> 
    %up_1_4 = aie.buffer(%logical_core_19) {sym_name = "up_1_4"} : memref<32xbf16> 
    %sp_1_4 = aie.buffer(%logical_core_19) {sym_name = "sp_1_4"} : memref<256xf32> 
    %rtp_1_4 = aie.buffer(%logical_core_19) {sym_name = "rtp_1_4"} : memref<8xi32> = dense<0>
    %q_saved_1_5 = aie.buffer(%logical_core_20) {sym_name = "q_saved_1_5"} : memref<4096xbf16> 
    %g_1_5 = aie.buffer(%logical_core_20) {sym_name = "g_1_5"} : memref<2048xbf16> 
    %up_1_5 = aie.buffer(%logical_core_20) {sym_name = "up_1_5"} : memref<32xbf16> 
    %sp_1_5 = aie.buffer(%logical_core_20) {sym_name = "sp_1_5"} : memref<256xf32> 
    %rtp_1_5 = aie.buffer(%logical_core_20) {sym_name = "rtp_1_5"} : memref<8xi32> = dense<0>
    %q_saved_1_6 = aie.buffer(%logical_core_21) {sym_name = "q_saved_1_6"} : memref<4096xbf16> 
    %g_1_6 = aie.buffer(%logical_core_21) {sym_name = "g_1_6"} : memref<2048xbf16> 
    %up_1_6 = aie.buffer(%logical_core_21) {sym_name = "up_1_6"} : memref<32xbf16> 
    %sp_1_6 = aie.buffer(%logical_core_21) {sym_name = "sp_1_6"} : memref<256xf32> 
    %rtp_1_6 = aie.buffer(%logical_core_21) {sym_name = "rtp_1_6"} : memref<8xi32> = dense<0>
    %q_saved_1_7 = aie.buffer(%logical_core_22) {sym_name = "q_saved_1_7"} : memref<4096xbf16> 
    %g_1_7 = aie.buffer(%logical_core_22) {sym_name = "g_1_7"} : memref<2048xbf16> 
    %up_1_7 = aie.buffer(%logical_core_22) {sym_name = "up_1_7"} : memref<32xbf16> 
    %sp_1_7 = aie.buffer(%logical_core_22) {sym_name = "sp_1_7"} : memref<256xf32> 
    %rtp_1_7 = aie.buffer(%logical_core_22) {sym_name = "rtp_1_7"} : memref<8xi32> = dense<0>
    %q_saved_1_8 = aie.buffer(%logical_core_23) {sym_name = "q_saved_1_8"} : memref<4096xbf16> 
    %g_1_8 = aie.buffer(%logical_core_23) {sym_name = "g_1_8"} : memref<2048xbf16> 
    %up_1_8 = aie.buffer(%logical_core_23) {sym_name = "up_1_8"} : memref<32xbf16> 
    %sp_1_8 = aie.buffer(%logical_core_23) {sym_name = "sp_1_8"} : memref<256xf32> 
    %rtp_1_8 = aie.buffer(%logical_core_23) {sym_name = "rtp_1_8"} : memref<8xi32> = dense<0>
    %q_saved_1_9 = aie.buffer(%logical_core_24) {sym_name = "q_saved_1_9"} : memref<4096xbf16> 
    %g_1_9 = aie.buffer(%logical_core_24) {sym_name = "g_1_9"} : memref<2048xbf16> 
    %up_1_9 = aie.buffer(%logical_core_24) {sym_name = "up_1_9"} : memref<32xbf16> 
    %sp_1_9 = aie.buffer(%logical_core_24) {sym_name = "sp_1_9"} : memref<256xf32> 
    %rtp_1_9 = aie.buffer(%logical_core_24) {sym_name = "rtp_1_9"} : memref<8xi32> = dense<0>
    %q_saved_1_10 = aie.buffer(%logical_core_25) {sym_name = "q_saved_1_10"} : memref<4096xbf16> 
    %g_1_10 = aie.buffer(%logical_core_25) {sym_name = "g_1_10"} : memref<2048xbf16> 
    %up_1_10 = aie.buffer(%logical_core_25) {sym_name = "up_1_10"} : memref<32xbf16> 
    %sp_1_10 = aie.buffer(%logical_core_25) {sym_name = "sp_1_10"} : memref<256xf32> 
    %rtp_1_10 = aie.buffer(%logical_core_25) {sym_name = "rtp_1_10"} : memref<8xi32> = dense<0>
    %q_saved_1_11 = aie.buffer(%logical_core_26) {sym_name = "q_saved_1_11"} : memref<4096xbf16> 
    %g_1_11 = aie.buffer(%logical_core_26) {sym_name = "g_1_11"} : memref<2048xbf16> 
    %up_1_11 = aie.buffer(%logical_core_26) {sym_name = "up_1_11"} : memref<32xbf16> 
    %sp_1_11 = aie.buffer(%logical_core_26) {sym_name = "sp_1_11"} : memref<256xf32> 
    %rtp_1_11 = aie.buffer(%logical_core_26) {sym_name = "rtp_1_11"} : memref<8xi32> = dense<0>
    %q_saved_1_12 = aie.buffer(%logical_core_27) {sym_name = "q_saved_1_12"} : memref<4096xbf16> 
    %g_1_12 = aie.buffer(%logical_core_27) {sym_name = "g_1_12"} : memref<2048xbf16> 
    %up_1_12 = aie.buffer(%logical_core_27) {sym_name = "up_1_12"} : memref<32xbf16> 
    %sp_1_12 = aie.buffer(%logical_core_27) {sym_name = "sp_1_12"} : memref<256xf32> 
    %rtp_1_12 = aie.buffer(%logical_core_27) {sym_name = "rtp_1_12"} : memref<8xi32> = dense<0>
    %q_saved_1_13 = aie.buffer(%logical_core_28) {sym_name = "q_saved_1_13"} : memref<4096xbf16> 
    %g_1_13 = aie.buffer(%logical_core_28) {sym_name = "g_1_13"} : memref<2048xbf16> 
    %up_1_13 = aie.buffer(%logical_core_28) {sym_name = "up_1_13"} : memref<32xbf16> 
    %sp_1_13 = aie.buffer(%logical_core_28) {sym_name = "sp_1_13"} : memref<256xf32> 
    %rtp_1_13 = aie.buffer(%logical_core_28) {sym_name = "rtp_1_13"} : memref<8xi32> = dense<0>
    %q_saved_1_14 = aie.buffer(%logical_core_29) {sym_name = "q_saved_1_14"} : memref<4096xbf16> 
    %g_1_14 = aie.buffer(%logical_core_29) {sym_name = "g_1_14"} : memref<2048xbf16> 
    %up_1_14 = aie.buffer(%logical_core_29) {sym_name = "up_1_14"} : memref<32xbf16> 
    %sp_1_14 = aie.buffer(%logical_core_29) {sym_name = "sp_1_14"} : memref<256xf32> 
    %rtp_1_14 = aie.buffer(%logical_core_29) {sym_name = "rtp_1_14"} : memref<8xi32> = dense<0>
    %q_saved_1_15 = aie.buffer(%logical_core_30) {sym_name = "q_saved_1_15"} : memref<4096xbf16> 
    %g_1_15 = aie.buffer(%logical_core_30) {sym_name = "g_1_15"} : memref<2048xbf16> 
    %up_1_15 = aie.buffer(%logical_core_30) {sym_name = "up_1_15"} : memref<32xbf16> 
    %sp_1_15 = aie.buffer(%logical_core_30) {sym_name = "sp_1_15"} : memref<256xf32> 
    %rtp_1_15 = aie.buffer(%logical_core_30) {sym_name = "rtp_1_15"} : memref<8xi32> = dense<0>
    %lock_0_2 = aie.lock(%logical_core)
    %core_0_2 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_0_2, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_0_0[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_0_0[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_0_0[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_60 = arith.constant 0 : i32
            func.call @capture_q(%8, %q_saved_0_0, %c0_i32, %c0_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c0_i32_62 = arith.constant 0 : i32
            func.call @capture_q(%10, %q_saved_0_0, %c1_i32_61, %c0_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            func.call @pass_init(%6, %up_0_0, %sp_0_0) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_0_0) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_0_0, %41, %g_0_0, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_0_0, %43, %g_0_0, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_0_0, %rtp_0_0, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_0_0, %up_0_0, %sp_0_0, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_0_0, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_0_0, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
            }
            func.call @finalize(%sp_0_0, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_0_0(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_0_2, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_0_3 = aie.lock(%logical_core_0)
    %core_0_3 = aie.core(%logical_core_0) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_0_3, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_0_1[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_0_1[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_0_1[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32_60 = arith.constant 1 : i32
            func.call @capture_q(%8, %q_saved_0_1, %c0_i32, %c1_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c1_i32_62 = arith.constant 1 : i32
            func.call @capture_q(%10, %q_saved_0_1, %c1_i32_61, %c1_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            func.call @pass_init(%6, %up_0_1, %sp_0_1) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_0_1) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_0_1, %41, %g_0_1, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_0_1, %43, %g_0_1, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_0_1, %rtp_0_1, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_0_1, %up_0_1, %sp_0_1, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_0_1, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_0_1, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
            }
            func.call @finalize(%sp_0_1, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_0_1(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_0_3, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_0_4 = aie.lock(%logical_core_1)
    %core_0_4 = aie.core(%logical_core_1) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_0_4, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_0_2[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_0_2[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_0_2[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_60 = arith.constant 0 : i32
            func.call @capture_q(%12, %q_saved_0_2, %c0_i32, %c0_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c0_i32_62 = arith.constant 0 : i32
            func.call @capture_q(%14, %q_saved_0_2, %c1_i32_61, %c0_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            func.call @pass_init(%6, %up_0_2, %sp_0_2) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_0_2) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_0_2, %41, %g_0_2, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_0_2, %43, %g_0_2, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_0_2, %rtp_0_2, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_0_2, %up_0_2, %sp_0_2, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_0_2, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_0_2, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
            }
            func.call @finalize(%sp_0_2, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_0_2(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_0_4, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_0_5 = aie.lock(%logical_core_2)
    %core_0_5 = aie.core(%logical_core_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_0_5, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_0_3[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_0_3[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_0_3[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32_60 = arith.constant 1 : i32
            func.call @capture_q(%12, %q_saved_0_3, %c0_i32, %c1_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c1_i32_62 = arith.constant 1 : i32
            func.call @capture_q(%14, %q_saved_0_3, %c1_i32_61, %c1_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            func.call @pass_init(%6, %up_0_3, %sp_0_3) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_0_3) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_0_3, %41, %g_0_3, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_0_3, %43, %g_0_3, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_0_3, %rtp_0_3, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_0_3, %up_0_3, %sp_0_3, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_0_3, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_0_3, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
            }
            func.call @finalize(%sp_0_3, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_0_3(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_0_5, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_1_2 = aie.lock(%logical_core_3)
    %core_1_2 = aie.core(%logical_core_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_1_2, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_0_4[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_0_4[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_0_4[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_60 = arith.constant 0 : i32
            func.call @capture_q(%16, %q_saved_0_4, %c0_i32, %c0_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c0_i32_62 = arith.constant 0 : i32
            func.call @capture_q(%18, %q_saved_0_4, %c1_i32_61, %c0_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            func.call @pass_init(%6, %up_0_4, %sp_0_4) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_0_4) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_0_4, %41, %g_0_4, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_0_4, %43, %g_0_4, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_0_4, %rtp_0_4, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_0_4, %up_0_4, %sp_0_4, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_0_4, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_0_4, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
            }
            func.call @finalize(%sp_0_4, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_1_0(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_1_2, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_1_3 = aie.lock(%logical_core_4)
    %core_1_3 = aie.core(%logical_core_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_1_3, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_0_5[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_0_5[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_0_5[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32_60 = arith.constant 1 : i32
            func.call @capture_q(%16, %q_saved_0_5, %c0_i32, %c1_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c1_i32_62 = arith.constant 1 : i32
            func.call @capture_q(%18, %q_saved_0_5, %c1_i32_61, %c1_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            func.call @pass_init(%6, %up_0_5, %sp_0_5) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_0_5) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_0_5, %41, %g_0_5, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_0_5, %43, %g_0_5, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_0_5, %rtp_0_5, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_0_5, %up_0_5, %sp_0_5, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_0_5, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_0_5, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
            }
            func.call @finalize(%sp_0_5, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_1_1(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_1_3, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_1_4 = aie.lock(%logical_core_5)
    %core_1_4 = aie.core(%logical_core_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_1_4, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_0_6[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_0_6[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_0_6[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_60 = arith.constant 0 : i32
            func.call @capture_q(%20, %q_saved_0_6, %c0_i32, %c0_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c0_i32_62 = arith.constant 0 : i32
            func.call @capture_q(%22, %q_saved_0_6, %c1_i32_61, %c0_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            func.call @pass_init(%6, %up_0_6, %sp_0_6) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_0_6) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_0_6, %41, %g_0_6, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_0_6, %43, %g_0_6, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_0_6, %rtp_0_6, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_0_6, %up_0_6, %sp_0_6, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_0_6, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_0_6, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
            }
            func.call @finalize(%sp_0_6, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_1_2(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_1_4, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_1_5 = aie.lock(%logical_core_6)
    %core_1_5 = aie.core(%logical_core_6) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_1_5, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_0_7[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_0_7[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_0_7[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32_60 = arith.constant 1 : i32
            func.call @capture_q(%20, %q_saved_0_7, %c0_i32, %c1_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c1_i32_62 = arith.constant 1 : i32
            func.call @capture_q(%22, %q_saved_0_7, %c1_i32_61, %c1_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            func.call @pass_init(%6, %up_0_7, %sp_0_7) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_0_7) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_0_7, %41, %g_0_7, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_0_7, %43, %g_0_7, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_0_7, %rtp_0_7, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_0_7, %up_0_7, %sp_0_7, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_0_7, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_0_7, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
            }
            func.call @finalize(%sp_0_7, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_1_3(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_1_5, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_2_2 = aie.lock(%logical_core_7)
    %core_2_2 = aie.core(%logical_core_7) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_2_2, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_0_8[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_0_8[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_0_8[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_60 = arith.constant 0 : i32
            func.call @capture_q(%24, %q_saved_0_8, %c0_i32, %c0_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c0_i32_62 = arith.constant 0 : i32
            func.call @capture_q(%26, %q_saved_0_8, %c1_i32_61, %c0_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            func.call @pass_init(%6, %up_0_8, %sp_0_8) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_0_8) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_0_8, %41, %g_0_8, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_0_8, %43, %g_0_8, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_0_8, %rtp_0_8, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_0_8, %up_0_8, %sp_0_8, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_0_8, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_0_8, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
            }
            func.call @finalize(%sp_0_8, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_2_0(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_2_2, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_2_3 = aie.lock(%logical_core_8)
    %core_2_3 = aie.core(%logical_core_8) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_2_3, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_0_9[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_0_9[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_0_9[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32_60 = arith.constant 1 : i32
            func.call @capture_q(%24, %q_saved_0_9, %c0_i32, %c1_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c1_i32_62 = arith.constant 1 : i32
            func.call @capture_q(%26, %q_saved_0_9, %c1_i32_61, %c1_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            func.call @pass_init(%6, %up_0_9, %sp_0_9) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_0_9) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_0_9, %41, %g_0_9, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_0_9, %43, %g_0_9, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_0_9, %rtp_0_9, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_0_9, %up_0_9, %sp_0_9, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_0_9, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_0_9, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
            }
            func.call @finalize(%sp_0_9, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_2_1(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_2_3, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_2_4 = aie.lock(%logical_core_9)
    %core_2_4 = aie.core(%logical_core_9) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_2_4, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_0_10[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_0_10[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_0_10[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_60 = arith.constant 0 : i32
            func.call @capture_q(%28, %q_saved_0_10, %c0_i32, %c0_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c0_i32_62 = arith.constant 0 : i32
            func.call @capture_q(%30, %q_saved_0_10, %c1_i32_61, %c0_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            func.call @pass_init(%6, %up_0_10, %sp_0_10) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_0_10) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_0_10, %41, %g_0_10, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_0_10, %43, %g_0_10, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_0_10, %rtp_0_10, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_0_10, %up_0_10, %sp_0_10, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_0_10, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_0_10, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
            }
            func.call @finalize(%sp_0_10, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_2_2(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_2_4, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_2_5 = aie.lock(%logical_core_10)
    %core_2_5 = aie.core(%logical_core_10) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_2_5, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_0_11[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_0_11[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_0_11[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32_60 = arith.constant 1 : i32
            func.call @capture_q(%28, %q_saved_0_11, %c0_i32, %c1_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c1_i32_62 = arith.constant 1 : i32
            func.call @capture_q(%30, %q_saved_0_11, %c1_i32_61, %c1_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            func.call @pass_init(%6, %up_0_11, %sp_0_11) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_0_11) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_0_11, %41, %g_0_11, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_0_11, %43, %g_0_11, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_0_11, %rtp_0_11, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_0_11, %up_0_11, %sp_0_11, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_0_11, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_0_11, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
            }
            func.call @finalize(%sp_0_11, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_2_3(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_2_5, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_3_2 = aie.lock(%logical_core_11)
    %core_3_2 = aie.core(%logical_core_11) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_3_2, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_0_12[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_0_12[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_0_12[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_60 = arith.constant 0 : i32
            func.call @capture_q(%32, %q_saved_0_12, %c0_i32, %c0_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c0_i32_62 = arith.constant 0 : i32
            func.call @capture_q(%34, %q_saved_0_12, %c1_i32_61, %c0_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            func.call @pass_init(%6, %up_0_12, %sp_0_12) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_0_12) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_0_12, %41, %g_0_12, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_0_12, %43, %g_0_12, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_0_12, %rtp_0_12, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_0_12, %up_0_12, %sp_0_12, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_0_12, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_0_12, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
            }
            func.call @finalize(%sp_0_12, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_3_0(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_3_2, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_3_3 = aie.lock(%logical_core_12)
    %core_3_3 = aie.core(%logical_core_12) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_3_3, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_0_13[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_0_13[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_0_13[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32_60 = arith.constant 1 : i32
            func.call @capture_q(%32, %q_saved_0_13, %c0_i32, %c1_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c1_i32_62 = arith.constant 1 : i32
            func.call @capture_q(%34, %q_saved_0_13, %c1_i32_61, %c1_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            func.call @pass_init(%6, %up_0_13, %sp_0_13) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_0_13) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_0_13, %41, %g_0_13, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_0_13, %43, %g_0_13, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_0_13, %rtp_0_13, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_0_13, %up_0_13, %sp_0_13, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_0_13, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_0_13, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
            }
            func.call @finalize(%sp_0_13, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_3_1(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_3_3, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_3_4 = aie.lock(%logical_core_13)
    %core_3_4 = aie.core(%logical_core_13) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_3_4, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_0_14[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_0_14[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_0_14[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_60 = arith.constant 0 : i32
            func.call @capture_q(%36, %q_saved_0_14, %c0_i32, %c0_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c0_i32_62 = arith.constant 0 : i32
            func.call @capture_q(%38, %q_saved_0_14, %c1_i32_61, %c0_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            func.call @pass_init(%6, %up_0_14, %sp_0_14) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_0_14) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_0_14, %41, %g_0_14, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_0_14, %43, %g_0_14, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_0_14, %rtp_0_14, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_0_14, %up_0_14, %sp_0_14, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_0_14, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_0_14, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
            }
            func.call @finalize(%sp_0_14, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_3_2(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_3_4, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_3_5 = aie.lock(%logical_core_14)
    %core_3_5 = aie.core(%logical_core_14) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_3_5, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_0_15[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_0_15[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_0_15[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32_60 = arith.constant 1 : i32
            func.call @capture_q(%36, %q_saved_0_15, %c0_i32, %c1_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c1_i32_62 = arith.constant 1 : i32
            func.call @capture_q(%38, %q_saved_0_15, %c1_i32_61, %c1_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
            func.call @pass_init(%6, %up_0_15, %sp_0_15) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_0_15) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_0_15, %41, %g_0_15, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_0_15, %43, %g_0_15, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_0(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_0_15, %rtp_0_15, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_0_15, %up_0_15, %sp_0_15, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_0_15, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_0_15, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_0(Consume, 1)
            }
            func.call @finalize(%sp_0_15, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_3_3(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_3_5, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_4_2 = aie.lock(%logical_core_15)
    %core_4_2 = aie.core(%logical_core_15) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_4_2, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_1_0[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_1_0[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_1_0[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_60 = arith.constant 0 : i32
            func.call @capture_q(%8, %q_saved_1_0, %c0_i32, %c0_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c0_i32_62 = arith.constant 0 : i32
            func.call @capture_q(%10, %q_saved_1_0, %c1_i32_61, %c0_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            func.call @pass_init(%6, %up_1_0, %sp_1_0) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_1_0) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_1_0, %41, %g_1_0, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_1_0, %43, %g_1_0, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_1_0, %rtp_1_0, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_1_0, %up_1_0, %sp_1_0, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_1_0, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_1_0, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
            }
            func.call @finalize(%sp_1_0, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_4_0(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_4_2, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_4_3 = aie.lock(%logical_core_16)
    %core_4_3 = aie.core(%logical_core_16) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_4_3, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_1_1[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_1_1[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_1_1[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32_60 = arith.constant 1 : i32
            func.call @capture_q(%8, %q_saved_1_1, %c0_i32, %c1_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c1_i32_62 = arith.constant 1 : i32
            func.call @capture_q(%10, %q_saved_1_1, %c1_i32_61, %c1_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            func.call @pass_init(%6, %up_1_1, %sp_1_1) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_1_1) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_1_1, %41, %g_1_1, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_1_1, %43, %g_1_1, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_1_1, %rtp_1_1, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_1_1, %up_1_1, %sp_1_1, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_1_1, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_1_1, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
            }
            func.call @finalize(%sp_1_1, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_4_1(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_4_3, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_4_4 = aie.lock(%logical_core_17)
    %core_4_4 = aie.core(%logical_core_17) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_4_4, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_1_2[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_1_2[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_1_2[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_60 = arith.constant 0 : i32
            func.call @capture_q(%12, %q_saved_1_2, %c0_i32, %c0_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c0_i32_62 = arith.constant 0 : i32
            func.call @capture_q(%14, %q_saved_1_2, %c1_i32_61, %c0_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            func.call @pass_init(%6, %up_1_2, %sp_1_2) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_1_2) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_1_2, %41, %g_1_2, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_1_2, %43, %g_1_2, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_1_2, %rtp_1_2, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_1_2, %up_1_2, %sp_1_2, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_1_2, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_1_2, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
            }
            func.call @finalize(%sp_1_2, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_4_2(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_4_4, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_4_5 = aie.lock(%logical_core_18)
    %core_4_5 = aie.core(%logical_core_18) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_4_5, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_1_3[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_1_3[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_1_3[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32_60 = arith.constant 1 : i32
            func.call @capture_q(%12, %q_saved_1_3, %c0_i32, %c1_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c1_i32_62 = arith.constant 1 : i32
            func.call @capture_q(%14, %q_saved_1_3, %c1_i32_61, %c1_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            func.call @pass_init(%6, %up_1_3, %sp_1_3) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_1_3) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_1_3, %41, %g_1_3, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_1_3, %43, %g_1_3, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_1_3, %rtp_1_3, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_1_3, %up_1_3, %sp_1_3, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_1_3, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_1_3, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
            }
            func.call @finalize(%sp_1_3, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_4_3(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_4_5, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_5_2 = aie.lock(%logical_core_19)
    %core_5_2 = aie.core(%logical_core_19) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_5_2, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_1_4[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_1_4[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_1_4[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_60 = arith.constant 0 : i32
            func.call @capture_q(%16, %q_saved_1_4, %c0_i32, %c0_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c0_i32_62 = arith.constant 0 : i32
            func.call @capture_q(%18, %q_saved_1_4, %c1_i32_61, %c0_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            func.call @pass_init(%6, %up_1_4, %sp_1_4) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_1_4) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_1_4, %41, %g_1_4, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_1_4, %43, %g_1_4, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_1_4, %rtp_1_4, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_1_4, %up_1_4, %sp_1_4, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_1_4, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_1_4, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
            }
            func.call @finalize(%sp_1_4, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_5_0(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_5_2, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_5_3 = aie.lock(%logical_core_20)
    %core_5_3 = aie.core(%logical_core_20) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_5_3, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_1_5[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_1_5[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_1_5[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32_60 = arith.constant 1 : i32
            func.call @capture_q(%16, %q_saved_1_5, %c0_i32, %c1_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c1_i32_62 = arith.constant 1 : i32
            func.call @capture_q(%18, %q_saved_1_5, %c1_i32_61, %c1_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            func.call @pass_init(%6, %up_1_5, %sp_1_5) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_1_5) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_1_5, %41, %g_1_5, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_1_5, %43, %g_1_5, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_1_5, %rtp_1_5, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_1_5, %up_1_5, %sp_1_5, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_1_5, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_1_5, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
            }
            func.call @finalize(%sp_1_5, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_5_1(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_5_3, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_5_4 = aie.lock(%logical_core_21)
    %core_5_4 = aie.core(%logical_core_21) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_5_4, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_1_6[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_1_6[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_1_6[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_60 = arith.constant 0 : i32
            func.call @capture_q(%20, %q_saved_1_6, %c0_i32, %c0_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c0_i32_62 = arith.constant 0 : i32
            func.call @capture_q(%22, %q_saved_1_6, %c1_i32_61, %c0_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            func.call @pass_init(%6, %up_1_6, %sp_1_6) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_1_6) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_1_6, %41, %g_1_6, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_1_6, %43, %g_1_6, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_1_6, %rtp_1_6, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_1_6, %up_1_6, %sp_1_6, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_1_6, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_1_6, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
            }
            func.call @finalize(%sp_1_6, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_5_2(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_5_4, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_5_5 = aie.lock(%logical_core_22)
    %core_5_5 = aie.core(%logical_core_22) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_5_5, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_1_7[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_1_7[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_1_7[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32_60 = arith.constant 1 : i32
            func.call @capture_q(%20, %q_saved_1_7, %c0_i32, %c1_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c1_i32_62 = arith.constant 1 : i32
            func.call @capture_q(%22, %q_saved_1_7, %c1_i32_61, %c1_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            func.call @pass_init(%6, %up_1_7, %sp_1_7) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_1_7) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_1_7, %41, %g_1_7, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_1_7, %43, %g_1_7, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_1_7, %rtp_1_7, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_1_7, %up_1_7, %sp_1_7, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_1_7, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_1_7, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
            }
            func.call @finalize(%sp_1_7, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_5_3(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_5_5, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_6_2 = aie.lock(%logical_core_23)
    %core_6_2 = aie.core(%logical_core_23) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_6_2, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_1_8[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_1_8[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_1_8[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_60 = arith.constant 0 : i32
            func.call @capture_q(%24, %q_saved_1_8, %c0_i32, %c0_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c0_i32_62 = arith.constant 0 : i32
            func.call @capture_q(%26, %q_saved_1_8, %c1_i32_61, %c0_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            func.call @pass_init(%6, %up_1_8, %sp_1_8) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_1_8) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_1_8, %41, %g_1_8, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_1_8, %43, %g_1_8, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_1_8, %rtp_1_8, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_1_8, %up_1_8, %sp_1_8, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_1_8, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_1_8, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
            }
            func.call @finalize(%sp_1_8, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_6_0(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_6_2, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_6_3 = aie.lock(%logical_core_24)
    %core_6_3 = aie.core(%logical_core_24) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_6_3, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_1_9[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_1_9[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_1_9[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32_60 = arith.constant 1 : i32
            func.call @capture_q(%24, %q_saved_1_9, %c0_i32, %c1_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c1_i32_62 = arith.constant 1 : i32
            func.call @capture_q(%26, %q_saved_1_9, %c1_i32_61, %c1_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            func.call @pass_init(%6, %up_1_9, %sp_1_9) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_1_9) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_1_9, %41, %g_1_9, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_1_9, %43, %g_1_9, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_1_9, %rtp_1_9, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_1_9, %up_1_9, %sp_1_9, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_1_9, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_1_9, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
            }
            func.call @finalize(%sp_1_9, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_6_1(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_6_3, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_6_4 = aie.lock(%logical_core_25)
    %core_6_4 = aie.core(%logical_core_25) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_6_4, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_1_10[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_1_10[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_1_10[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_60 = arith.constant 0 : i32
            func.call @capture_q(%28, %q_saved_1_10, %c0_i32, %c0_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c0_i32_62 = arith.constant 0 : i32
            func.call @capture_q(%30, %q_saved_1_10, %c1_i32_61, %c0_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            func.call @pass_init(%6, %up_1_10, %sp_1_10) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_1_10) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_1_10, %41, %g_1_10, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_1_10, %43, %g_1_10, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_1_10, %rtp_1_10, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_1_10, %up_1_10, %sp_1_10, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_1_10, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_1_10, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
            }
            func.call @finalize(%sp_1_10, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_6_2(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_6_4, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_6_5 = aie.lock(%logical_core_26)
    %core_6_5 = aie.core(%logical_core_26) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_6_5, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_1_11[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_1_11[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_1_11[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32_60 = arith.constant 1 : i32
            func.call @capture_q(%28, %q_saved_1_11, %c0_i32, %c1_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c1_i32_62 = arith.constant 1 : i32
            func.call @capture_q(%30, %q_saved_1_11, %c1_i32_61, %c1_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            func.call @pass_init(%6, %up_1_11, %sp_1_11) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_1_11) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_1_11, %41, %g_1_11, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_1_11, %43, %g_1_11, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_1_11, %rtp_1_11, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_1_11, %up_1_11, %sp_1_11, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_1_11, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_1_11, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
            }
            func.call @finalize(%sp_1_11, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_6_3(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_6_5, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_7_2 = aie.lock(%logical_core_27)
    %core_7_2 = aie.core(%logical_core_27) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_7_2, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_1_12[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_1_12[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_1_12[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_60 = arith.constant 0 : i32
            func.call @capture_q(%32, %q_saved_1_12, %c0_i32, %c0_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c0_i32_62 = arith.constant 0 : i32
            func.call @capture_q(%34, %q_saved_1_12, %c1_i32_61, %c0_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            func.call @pass_init(%6, %up_1_12, %sp_1_12) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_1_12) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_1_12, %41, %g_1_12, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_1_12, %43, %g_1_12, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_1_12, %rtp_1_12, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_1_12, %up_1_12, %sp_1_12, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_1_12, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_1_12, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
            }
            func.call @finalize(%sp_1_12, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_7_0(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_7_2, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_7_3 = aie.lock(%logical_core_28)
    %core_7_3 = aie.core(%logical_core_28) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_7_3, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_1_13[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_1_13[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_1_13[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32_60 = arith.constant 1 : i32
            func.call @capture_q(%32, %q_saved_1_13, %c0_i32, %c1_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c1_i32_62 = arith.constant 1 : i32
            func.call @capture_q(%34, %q_saved_1_13, %c1_i32_61, %c1_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            func.call @pass_init(%6, %up_1_13, %sp_1_13) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_1_13) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_1_13, %41, %g_1_13, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_1_13, %43, %g_1_13, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_1_13, %rtp_1_13, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_1_13, %up_1_13, %sp_1_13, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_1_13, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_1_13, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
            }
            func.call @finalize(%sp_1_13, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_7_1(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_7_3, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_7_4 = aie.lock(%logical_core_29)
    %core_7_4 = aie.core(%logical_core_29) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_7_4, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_1_14[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_1_14[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_1_14[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_60 = arith.constant 0 : i32
            func.call @capture_q(%36, %q_saved_1_14, %c0_i32, %c0_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c0_i32_62 = arith.constant 0 : i32
            func.call @capture_q(%38, %q_saved_1_14, %c1_i32_61, %c0_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            func.call @pass_init(%6, %up_1_14, %sp_1_14) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_1_14) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_1_14, %41, %g_1_14, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_1_14, %43, %g_1_14, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_1_14, %rtp_1_14, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_1_14, %up_1_14, %sp_1_14, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_1_14, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_1_14, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
            }
            func.call @finalize(%sp_1_14, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_7_2(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_7_4, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %lock_7_5 = aie.lock(%logical_core_30)
    %core_7_5 = aie.core(%logical_core_30) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c1_i32 = arith.constant 1 : i32
        aie.use_lock(%lock_7_5, Acquire, %c1_i32)
        %c0_53 = arith.constant 0 : index
        %0 = memref.load %rtp_1_15[%c0_53] : memref<8xi32>
        %c1_54 = arith.constant 1 : index
        %1 = memref.load %rtp_1_15[%c1_54] : memref<8xi32>
        %c2 = arith.constant 2 : index
        %2 = memref.load %rtp_1_15[%c2] : memref<8xi32>
        %c0_55 = arith.constant 0 : index
        %3 = arith.index_cast %0 : i32 to index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %3 step %c1_56 {
          %c0_58 = arith.constant 0 : index
          %4 = arith.index_cast %1 : i32 to index
          %c1_59 = arith.constant 1 : index
          scf.for %arg2 = %c0_58 to %4 step %c1_59 {
            %5 = aie.objectfifo.acquire @O_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<4096xbf16>>
            %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
            %7 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %9 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %11 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %13 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %15 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %17 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %19 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %21 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            %23 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %25 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %27 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %28 = aie.objectfifo.subview.access %27[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %29 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %30 = aie.objectfifo.subview.access %29[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %31 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %32 = aie.objectfifo.subview.access %31[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %33 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %34 = aie.objectfifo.subview.access %33[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %35 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %36 = aie.objectfifo.subview.access %35[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c0_i32 = arith.constant 0 : i32
            %c1_i32_60 = arith.constant 1 : i32
            func.call @capture_q(%36, %q_saved_1_15, %c0_i32, %c1_i32_60) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            %37 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %38 = aie.objectfifo.subview.access %37[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %c1_i32_61 = arith.constant 1 : i32
            %c1_i32_62 = arith.constant 1 : i32
            func.call @capture_q(%38, %q_saved_1_15, %c1_i32_61, %c1_i32_62) : (memref<64x64xbf16>, memref<4096xbf16>, i32, i32) -> ()
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
            func.call @pass_init(%6, %up_1_15, %sp_1_15) : (memref<4096xbf16>, memref<32xbf16>, memref<256xf32>) -> ()
            %c0_63 = arith.constant 0 : index
            %39 = arith.index_cast %2 : i32 to index
            %c1_64 = arith.constant 1 : index
            scf.for %arg3 = %c0_63 to %39 step %c1_64 {
              func.call @zero_g(%g_1_15) : (memref<2048xbf16>) -> ()
              %40 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %41 = aie.objectfifo.subview.access %40[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_65 = arith.constant 0 : i32
              func.call @qk(%q_saved_1_15, %41, %g_1_15, %c0_i32_65) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %42 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %43 = aie.objectfifo.subview.access %42[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_66 = arith.constant 1 : i32
              func.call @qk(%q_saved_1_15, %43, %g_1_15, %c1_i32_66) : (memref<4096xbf16>, memref<64x64xbf16>, memref<2048xbf16>, i32) -> ()
              aie.objectfifo.release @A_L2L1_1(Consume, 1)
              %44 = arith.index_cast %arg2 : index to i32
              %45 = arith.index_cast %arg3 : index to i32
              func.call @apply_mask(%g_1_15, %rtp_1_15, %44, %45) : (memref<2048xbf16>, memref<8xi32>, i32, i32) -> ()
              func.call @softmax_step(%g_1_15, %up_1_15, %sp_1_15, %6) : (memref<2048xbf16>, memref<32xbf16>, memref<256xf32>, memref<4096xbf16>) -> ()
              %46 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %47 = aie.objectfifo.subview.access %46[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c0_i32_67 = arith.constant 0 : i32
              func.call @pv(%g_1_15, %47, %6, %c0_i32_67) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
              %48 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
              %49 = aie.objectfifo.subview.access %48[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
              %c1_i32_68 = arith.constant 1 : i32
              func.call @pv(%g_1_15, %49, %6, %c1_i32_68) : (memref<2048xbf16>, memref<64x64xbf16>, memref<4096xbf16>, i32) -> ()
              aie.objectfifo.release @B_L2L1_1(Consume, 1)
            }
            func.call @finalize(%sp_1_15, %6) : (memref<256xf32>, memref<4096xbf16>) -> ()
            aie.objectfifo.release @O_L1L2_7_3(Produce, 1)
          }
        }
        %c1_i32_57 = arith.constant 1 : i32
        aie.use_lock(%lock_7_5, Release, %c1_i32_57)
      }
      aie.end
    } {stack_size = 4096 : i32}
    aie.runtime_sequence @r512_attn(%arg0: memref<4718592xbf16>, %arg1: memref<4718592xbf16>, %arg2: memref<4718592xbf16>, %arg3: memref<4718592xbf16>) {
      %c12_i32 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_0_0, 0, %c12_i32) : i32
      %c3_i32 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_0, 1, %c3_i32) : i32
      %c24_i32 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_0_0, 2, %c24_i32) : i32
      %c1536_i32 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_0_0, 3, %c1536_i32) : i32
      %c0_i32 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_0, 4, %c0_i32) : i32
      %c0_i32_53 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_0, 5, %c0_i32_53) : i32
      %c12_i32_54 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_0_1, 0, %c12_i32_54) : i32
      %c3_i32_55 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_1, 1, %c3_i32_55) : i32
      %c24_i32_56 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_0_1, 2, %c24_i32_56) : i32
      %c1536_i32_57 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_0_1, 3, %c1536_i32_57) : i32
      %c0_i32_58 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_1, 4, %c0_i32_58) : i32
      %c32_i32 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_0_1, 5, %c32_i32) : i32
      %c12_i32_59 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_0_2, 0, %c12_i32_59) : i32
      %c3_i32_60 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_2, 1, %c3_i32_60) : i32
      %c24_i32_61 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_0_2, 2, %c24_i32_61) : i32
      %c1536_i32_62 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_0_2, 3, %c1536_i32_62) : i32
      %c0_i32_63 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_2, 4, %c0_i32_63) : i32
      %c64_i32 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_0_2, 5, %c64_i32) : i32
      %c12_i32_64 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_0_3, 0, %c12_i32_64) : i32
      %c3_i32_65 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_3, 1, %c3_i32_65) : i32
      %c24_i32_66 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_0_3, 2, %c24_i32_66) : i32
      %c1536_i32_67 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_0_3, 3, %c1536_i32_67) : i32
      %c0_i32_68 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_3, 4, %c0_i32_68) : i32
      %c96_i32 = arith.constant 96 : i32
      aiex.npu.rtp_write(@rtp_0_3, 5, %c96_i32) : i32
      %c12_i32_69 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_0_4, 0, %c12_i32_69) : i32
      %c3_i32_70 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_4, 1, %c3_i32_70) : i32
      %c24_i32_71 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_0_4, 2, %c24_i32_71) : i32
      %c1536_i32_72 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_0_4, 3, %c1536_i32_72) : i32
      %c0_i32_73 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_4, 4, %c0_i32_73) : i32
      %c128_i32 = arith.constant 128 : i32
      aiex.npu.rtp_write(@rtp_0_4, 5, %c128_i32) : i32
      %c12_i32_74 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_0_5, 0, %c12_i32_74) : i32
      %c3_i32_75 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_5, 1, %c3_i32_75) : i32
      %c24_i32_76 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_0_5, 2, %c24_i32_76) : i32
      %c1536_i32_77 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_0_5, 3, %c1536_i32_77) : i32
      %c0_i32_78 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_5, 4, %c0_i32_78) : i32
      %c160_i32 = arith.constant 160 : i32
      aiex.npu.rtp_write(@rtp_0_5, 5, %c160_i32) : i32
      %c12_i32_79 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_0_6, 0, %c12_i32_79) : i32
      %c3_i32_80 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_6, 1, %c3_i32_80) : i32
      %c24_i32_81 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_0_6, 2, %c24_i32_81) : i32
      %c1536_i32_82 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_0_6, 3, %c1536_i32_82) : i32
      %c0_i32_83 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_6, 4, %c0_i32_83) : i32
      %c192_i32 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_0_6, 5, %c192_i32) : i32
      %c12_i32_84 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_0_7, 0, %c12_i32_84) : i32
      %c3_i32_85 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_7, 1, %c3_i32_85) : i32
      %c24_i32_86 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_0_7, 2, %c24_i32_86) : i32
      %c1536_i32_87 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_0_7, 3, %c1536_i32_87) : i32
      %c0_i32_88 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_7, 4, %c0_i32_88) : i32
      %c224_i32 = arith.constant 224 : i32
      aiex.npu.rtp_write(@rtp_0_7, 5, %c224_i32) : i32
      %c12_i32_89 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_0_8, 0, %c12_i32_89) : i32
      %c3_i32_90 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_8, 1, %c3_i32_90) : i32
      %c24_i32_91 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_0_8, 2, %c24_i32_91) : i32
      %c1536_i32_92 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_0_8, 3, %c1536_i32_92) : i32
      %c0_i32_93 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_8, 4, %c0_i32_93) : i32
      %c256_i32 = arith.constant 256 : i32
      aiex.npu.rtp_write(@rtp_0_8, 5, %c256_i32) : i32
      %c12_i32_94 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_0_9, 0, %c12_i32_94) : i32
      %c3_i32_95 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_9, 1, %c3_i32_95) : i32
      %c24_i32_96 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_0_9, 2, %c24_i32_96) : i32
      %c1536_i32_97 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_0_9, 3, %c1536_i32_97) : i32
      %c0_i32_98 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_9, 4, %c0_i32_98) : i32
      %c288_i32 = arith.constant 288 : i32
      aiex.npu.rtp_write(@rtp_0_9, 5, %c288_i32) : i32
      %c12_i32_99 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_0_10, 0, %c12_i32_99) : i32
      %c3_i32_100 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_10, 1, %c3_i32_100) : i32
      %c24_i32_101 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_0_10, 2, %c24_i32_101) : i32
      %c1536_i32_102 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_0_10, 3, %c1536_i32_102) : i32
      %c0_i32_103 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_10, 4, %c0_i32_103) : i32
      %c320_i32 = arith.constant 320 : i32
      aiex.npu.rtp_write(@rtp_0_10, 5, %c320_i32) : i32
      %c12_i32_104 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_0_11, 0, %c12_i32_104) : i32
      %c3_i32_105 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_11, 1, %c3_i32_105) : i32
      %c24_i32_106 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_0_11, 2, %c24_i32_106) : i32
      %c1536_i32_107 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_0_11, 3, %c1536_i32_107) : i32
      %c0_i32_108 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_11, 4, %c0_i32_108) : i32
      %c352_i32 = arith.constant 352 : i32
      aiex.npu.rtp_write(@rtp_0_11, 5, %c352_i32) : i32
      %c12_i32_109 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_0_12, 0, %c12_i32_109) : i32
      %c3_i32_110 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_12, 1, %c3_i32_110) : i32
      %c24_i32_111 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_0_12, 2, %c24_i32_111) : i32
      %c1536_i32_112 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_0_12, 3, %c1536_i32_112) : i32
      %c0_i32_113 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_12, 4, %c0_i32_113) : i32
      %c384_i32 = arith.constant 384 : i32
      aiex.npu.rtp_write(@rtp_0_12, 5, %c384_i32) : i32
      %c12_i32_114 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_0_13, 0, %c12_i32_114) : i32
      %c3_i32_115 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_13, 1, %c3_i32_115) : i32
      %c24_i32_116 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_0_13, 2, %c24_i32_116) : i32
      %c1536_i32_117 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_0_13, 3, %c1536_i32_117) : i32
      %c0_i32_118 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_13, 4, %c0_i32_118) : i32
      %c416_i32 = arith.constant 416 : i32
      aiex.npu.rtp_write(@rtp_0_13, 5, %c416_i32) : i32
      %c12_i32_119 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_0_14, 0, %c12_i32_119) : i32
      %c3_i32_120 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_14, 1, %c3_i32_120) : i32
      %c24_i32_121 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_0_14, 2, %c24_i32_121) : i32
      %c1536_i32_122 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_0_14, 3, %c1536_i32_122) : i32
      %c0_i32_123 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_14, 4, %c0_i32_123) : i32
      %c448_i32 = arith.constant 448 : i32
      aiex.npu.rtp_write(@rtp_0_14, 5, %c448_i32) : i32
      %c12_i32_124 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_0_15, 0, %c12_i32_124) : i32
      %c3_i32_125 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_0_15, 1, %c3_i32_125) : i32
      %c24_i32_126 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_0_15, 2, %c24_i32_126) : i32
      %c1536_i32_127 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_0_15, 3, %c1536_i32_127) : i32
      %c0_i32_128 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_0_15, 4, %c0_i32_128) : i32
      %c480_i32 = arith.constant 480 : i32
      aiex.npu.rtp_write(@rtp_0_15, 5, %c480_i32) : i32
      %c12_i32_129 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_1_0, 0, %c12_i32_129) : i32
      %c3_i32_130 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_0, 1, %c3_i32_130) : i32
      %c24_i32_131 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_1_0, 2, %c24_i32_131) : i32
      %c1536_i32_132 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_1_0, 3, %c1536_i32_132) : i32
      %c0_i32_133 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_0, 4, %c0_i32_133) : i32
      %c0_i32_134 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_0, 5, %c0_i32_134) : i32
      %c12_i32_135 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_1_1, 0, %c12_i32_135) : i32
      %c3_i32_136 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_1, 1, %c3_i32_136) : i32
      %c24_i32_137 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_1_1, 2, %c24_i32_137) : i32
      %c1536_i32_138 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_1_1, 3, %c1536_i32_138) : i32
      %c0_i32_139 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_1, 4, %c0_i32_139) : i32
      %c32_i32_140 = arith.constant 32 : i32
      aiex.npu.rtp_write(@rtp_1_1, 5, %c32_i32_140) : i32
      %c12_i32_141 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_1_2, 0, %c12_i32_141) : i32
      %c3_i32_142 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_2, 1, %c3_i32_142) : i32
      %c24_i32_143 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_1_2, 2, %c24_i32_143) : i32
      %c1536_i32_144 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_1_2, 3, %c1536_i32_144) : i32
      %c0_i32_145 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_2, 4, %c0_i32_145) : i32
      %c64_i32_146 = arith.constant 64 : i32
      aiex.npu.rtp_write(@rtp_1_2, 5, %c64_i32_146) : i32
      %c12_i32_147 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_1_3, 0, %c12_i32_147) : i32
      %c3_i32_148 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_3, 1, %c3_i32_148) : i32
      %c24_i32_149 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_1_3, 2, %c24_i32_149) : i32
      %c1536_i32_150 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_1_3, 3, %c1536_i32_150) : i32
      %c0_i32_151 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_3, 4, %c0_i32_151) : i32
      %c96_i32_152 = arith.constant 96 : i32
      aiex.npu.rtp_write(@rtp_1_3, 5, %c96_i32_152) : i32
      %c12_i32_153 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_1_4, 0, %c12_i32_153) : i32
      %c3_i32_154 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_4, 1, %c3_i32_154) : i32
      %c24_i32_155 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_1_4, 2, %c24_i32_155) : i32
      %c1536_i32_156 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_1_4, 3, %c1536_i32_156) : i32
      %c0_i32_157 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_4, 4, %c0_i32_157) : i32
      %c128_i32_158 = arith.constant 128 : i32
      aiex.npu.rtp_write(@rtp_1_4, 5, %c128_i32_158) : i32
      %c12_i32_159 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_1_5, 0, %c12_i32_159) : i32
      %c3_i32_160 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_5, 1, %c3_i32_160) : i32
      %c24_i32_161 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_1_5, 2, %c24_i32_161) : i32
      %c1536_i32_162 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_1_5, 3, %c1536_i32_162) : i32
      %c0_i32_163 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_5, 4, %c0_i32_163) : i32
      %c160_i32_164 = arith.constant 160 : i32
      aiex.npu.rtp_write(@rtp_1_5, 5, %c160_i32_164) : i32
      %c12_i32_165 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_1_6, 0, %c12_i32_165) : i32
      %c3_i32_166 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_6, 1, %c3_i32_166) : i32
      %c24_i32_167 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_1_6, 2, %c24_i32_167) : i32
      %c1536_i32_168 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_1_6, 3, %c1536_i32_168) : i32
      %c0_i32_169 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_6, 4, %c0_i32_169) : i32
      %c192_i32_170 = arith.constant 192 : i32
      aiex.npu.rtp_write(@rtp_1_6, 5, %c192_i32_170) : i32
      %c12_i32_171 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_1_7, 0, %c12_i32_171) : i32
      %c3_i32_172 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_7, 1, %c3_i32_172) : i32
      %c24_i32_173 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_1_7, 2, %c24_i32_173) : i32
      %c1536_i32_174 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_1_7, 3, %c1536_i32_174) : i32
      %c0_i32_175 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_7, 4, %c0_i32_175) : i32
      %c224_i32_176 = arith.constant 224 : i32
      aiex.npu.rtp_write(@rtp_1_7, 5, %c224_i32_176) : i32
      %c12_i32_177 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_1_8, 0, %c12_i32_177) : i32
      %c3_i32_178 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_8, 1, %c3_i32_178) : i32
      %c24_i32_179 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_1_8, 2, %c24_i32_179) : i32
      %c1536_i32_180 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_1_8, 3, %c1536_i32_180) : i32
      %c0_i32_181 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_8, 4, %c0_i32_181) : i32
      %c256_i32_182 = arith.constant 256 : i32
      aiex.npu.rtp_write(@rtp_1_8, 5, %c256_i32_182) : i32
      %c12_i32_183 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_1_9, 0, %c12_i32_183) : i32
      %c3_i32_184 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_9, 1, %c3_i32_184) : i32
      %c24_i32_185 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_1_9, 2, %c24_i32_185) : i32
      %c1536_i32_186 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_1_9, 3, %c1536_i32_186) : i32
      %c0_i32_187 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_9, 4, %c0_i32_187) : i32
      %c288_i32_188 = arith.constant 288 : i32
      aiex.npu.rtp_write(@rtp_1_9, 5, %c288_i32_188) : i32
      %c12_i32_189 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_1_10, 0, %c12_i32_189) : i32
      %c3_i32_190 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_10, 1, %c3_i32_190) : i32
      %c24_i32_191 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_1_10, 2, %c24_i32_191) : i32
      %c1536_i32_192 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_1_10, 3, %c1536_i32_192) : i32
      %c0_i32_193 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_10, 4, %c0_i32_193) : i32
      %c320_i32_194 = arith.constant 320 : i32
      aiex.npu.rtp_write(@rtp_1_10, 5, %c320_i32_194) : i32
      %c12_i32_195 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_1_11, 0, %c12_i32_195) : i32
      %c3_i32_196 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_11, 1, %c3_i32_196) : i32
      %c24_i32_197 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_1_11, 2, %c24_i32_197) : i32
      %c1536_i32_198 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_1_11, 3, %c1536_i32_198) : i32
      %c0_i32_199 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_11, 4, %c0_i32_199) : i32
      %c352_i32_200 = arith.constant 352 : i32
      aiex.npu.rtp_write(@rtp_1_11, 5, %c352_i32_200) : i32
      %c12_i32_201 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_1_12, 0, %c12_i32_201) : i32
      %c3_i32_202 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_12, 1, %c3_i32_202) : i32
      %c24_i32_203 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_1_12, 2, %c24_i32_203) : i32
      %c1536_i32_204 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_1_12, 3, %c1536_i32_204) : i32
      %c0_i32_205 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_12, 4, %c0_i32_205) : i32
      %c384_i32_206 = arith.constant 384 : i32
      aiex.npu.rtp_write(@rtp_1_12, 5, %c384_i32_206) : i32
      %c12_i32_207 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_1_13, 0, %c12_i32_207) : i32
      %c3_i32_208 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_13, 1, %c3_i32_208) : i32
      %c24_i32_209 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_1_13, 2, %c24_i32_209) : i32
      %c1536_i32_210 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_1_13, 3, %c1536_i32_210) : i32
      %c0_i32_211 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_13, 4, %c0_i32_211) : i32
      %c416_i32_212 = arith.constant 416 : i32
      aiex.npu.rtp_write(@rtp_1_13, 5, %c416_i32_212) : i32
      %c12_i32_213 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_1_14, 0, %c12_i32_213) : i32
      %c3_i32_214 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_14, 1, %c3_i32_214) : i32
      %c24_i32_215 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_1_14, 2, %c24_i32_215) : i32
      %c1536_i32_216 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_1_14, 3, %c1536_i32_216) : i32
      %c0_i32_217 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_14, 4, %c0_i32_217) : i32
      %c448_i32_218 = arith.constant 448 : i32
      aiex.npu.rtp_write(@rtp_1_14, 5, %c448_i32_218) : i32
      %c12_i32_219 = arith.constant 12 : i32
      aiex.npu.rtp_write(@rtp_1_15, 0, %c12_i32_219) : i32
      %c3_i32_220 = arith.constant 3 : i32
      aiex.npu.rtp_write(@rtp_1_15, 1, %c3_i32_220) : i32
      %c24_i32_221 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp_1_15, 2, %c24_i32_221) : i32
      %c1536_i32_222 = arith.constant 1536 : i32
      aiex.npu.rtp_write(@rtp_1_15, 3, %c1536_i32_222) : i32
      %c0_i32_223 = arith.constant 0 : i32
      aiex.npu.rtp_write(@rtp_1_15, 4, %c0_i32_223) : i32
      %c480_i32_224 = arith.constant 480 : i32
      aiex.npu.rtp_write(@rtp_1_15, 5, %c480_i32_224) : i32
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
      %0 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 0 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 786432 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 0 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 0 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 0 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 393216 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 786432 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1179648 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1536 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 787968 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1536 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1536 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1536 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 394752 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 787968 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1181184 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1572864 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2359296 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 0 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 0 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1572864 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1966080 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2359296 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2752512 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1574400 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2360832 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1536 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1536 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1574400 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1967616 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2360832 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2754048 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%31)
      aiex.dma_await_task(%4)
      aiex.dma_await_task(%5)
      aiex.dma_await_task(%6)
      aiex.dma_await_task(%7)
      aiex.dma_await_task(%12)
      aiex.dma_await_task(%13)
      aiex.dma_await_task(%14)
      aiex.dma_await_task(%15)
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
      %32 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3145728 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3932160 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 0 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 0 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3145728 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3538944 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3932160 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4325376 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3147264 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3933696 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1536 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1536 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3147264 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3540480 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3933696 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4326912 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%47)
      aiex.dma_await_task(%20)
      aiex.dma_await_task(%21)
      aiex.dma_await_task(%22)
      aiex.dma_await_task(%23)
      aiex.dma_await_task(%28)
      aiex.dma_await_task(%29)
      aiex.dma_await_task(%30)
      aiex.dma_await_task(%31)
      aiex.dma_free_task(%16)
      aiex.dma_free_task(%17)
      aiex.dma_free_task(%18)
      aiex.dma_free_task(%19)
      aiex.dma_free_task(%20)
      aiex.dma_free_task(%21)
      aiex.dma_free_task(%22)
      aiex.dma_free_task(%23)
      aiex.dma_free_task(%24)
      aiex.dma_free_task(%25)
      aiex.dma_free_task(%26)
      aiex.dma_free_task(%27)
      aiex.dma_free_task(%28)
      aiex.dma_free_task(%29)
      aiex.dma_free_task(%30)
      aiex.dma_free_task(%31)
      %48 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 128 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 786560 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 128 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%50)
      %51 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 128 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 128 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 393344 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 786560 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1179776 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1664 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 788096 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1664 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1664 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1664 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 394880 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 788096 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1181312 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%63)
      aiex.dma_await_task(%36)
      aiex.dma_await_task(%37)
      aiex.dma_await_task(%38)
      aiex.dma_await_task(%39)
      aiex.dma_await_task(%44)
      aiex.dma_await_task(%45)
      aiex.dma_await_task(%46)
      aiex.dma_await_task(%47)
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
      %64 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1572992 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2359424 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 128 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 128 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%67)
      %68 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1572992 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1966208 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2359424 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2752640 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1574528 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2360960 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1664 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1664 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1574528 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1967744 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2360960 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2754176 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%79)
      aiex.dma_await_task(%52)
      aiex.dma_await_task(%53)
      aiex.dma_await_task(%54)
      aiex.dma_await_task(%55)
      aiex.dma_await_task(%60)
      aiex.dma_await_task(%61)
      aiex.dma_await_task(%62)
      aiex.dma_await_task(%63)
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
      %80 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3145856 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3932288 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 128 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 128 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3145856 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%84)
      %85 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3539072 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3932288 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4325504 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3147392 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3933824 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1664 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1664 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3147392 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3540608 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3933824 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4327040 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%95)
      aiex.dma_await_task(%68)
      aiex.dma_await_task(%69)
      aiex.dma_await_task(%70)
      aiex.dma_await_task(%71)
      aiex.dma_await_task(%76)
      aiex.dma_await_task(%77)
      aiex.dma_await_task(%78)
      aiex.dma_await_task(%79)
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
      %96 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 256 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 786688 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%97)
      %98 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 256 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 256 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 256 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%100)
      %101 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 393472 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%101)
      %102 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 786688 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%102)
      %103 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1179904 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%103)
      %104 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1792 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%104)
      %105 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 788224 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%105)
      %106 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1792 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%106)
      %107 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1792 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%107)
      %108 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1792 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%108)
      %109 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 395008 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%109)
      %110 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 788224 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%110)
      %111 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1181440 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%111)
      aiex.dma_await_task(%84)
      aiex.dma_await_task(%85)
      aiex.dma_await_task(%86)
      aiex.dma_await_task(%87)
      aiex.dma_await_task(%92)
      aiex.dma_await_task(%93)
      aiex.dma_await_task(%94)
      aiex.dma_await_task(%95)
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
      %112 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1573120 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%112)
      %113 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2359552 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%113)
      %114 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 256 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%114)
      %115 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 256 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%115)
      %116 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1573120 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%116)
      %117 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1966336 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%117)
      %118 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2359552 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%118)
      %119 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2752768 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%119)
      %120 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1574656 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%120)
      %121 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2361088 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%121)
      %122 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1792 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%122)
      %123 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1792 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%123)
      %124 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1574656 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%124)
      %125 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1967872 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%125)
      %126 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2361088 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%126)
      %127 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2754304 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%127)
      aiex.dma_await_task(%100)
      aiex.dma_await_task(%101)
      aiex.dma_await_task(%102)
      aiex.dma_await_task(%103)
      aiex.dma_await_task(%108)
      aiex.dma_await_task(%109)
      aiex.dma_await_task(%110)
      aiex.dma_await_task(%111)
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
      %128 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3145984 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%128)
      %129 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3932416 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%129)
      %130 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 256 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%130)
      %131 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 256 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%131)
      %132 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3145984 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%132)
      %133 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3539200 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%133)
      %134 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3932416 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%134)
      %135 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4325632 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%135)
      %136 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3147520 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%136)
      %137 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3933952 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%137)
      %138 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1792 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%138)
      %139 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1792 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%139)
      %140 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3147520 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%140)
      %141 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3540736 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%141)
      %142 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3933952 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%142)
      %143 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4327168 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%143)
      aiex.dma_await_task(%116)
      aiex.dma_await_task(%117)
      aiex.dma_await_task(%118)
      aiex.dma_await_task(%119)
      aiex.dma_await_task(%124)
      aiex.dma_await_task(%125)
      aiex.dma_await_task(%126)
      aiex.dma_await_task(%127)
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
      %144 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 384 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%144)
      %145 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 786816 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%145)
      %146 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 384 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%146)
      %147 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 384 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%147)
      %148 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 384 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%148)
      %149 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 393600 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%149)
      %150 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 786816 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%150)
      %151 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1180032 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%151)
      %152 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1920 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%152)
      %153 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 788352 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%153)
      %154 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1920 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%154)
      %155 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1920 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%155)
      %156 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1920 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%156)
      %157 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 395136 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%157)
      %158 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 788352 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%158)
      %159 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1181568 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%159)
      aiex.dma_await_task(%132)
      aiex.dma_await_task(%133)
      aiex.dma_await_task(%134)
      aiex.dma_await_task(%135)
      aiex.dma_await_task(%140)
      aiex.dma_await_task(%141)
      aiex.dma_await_task(%142)
      aiex.dma_await_task(%143)
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
      %160 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1573248 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%160)
      %161 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2359680 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%161)
      %162 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 384 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%162)
      %163 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 384 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%163)
      %164 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1573248 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%164)
      %165 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1966464 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%165)
      %166 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2359680 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%166)
      %167 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2752896 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%167)
      %168 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1574784 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%168)
      %169 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2361216 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%169)
      %170 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1920 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%170)
      %171 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1920 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%171)
      %172 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1574784 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%172)
      %173 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1968000 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%173)
      %174 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2361216 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%174)
      %175 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2754432 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%175)
      aiex.dma_await_task(%148)
      aiex.dma_await_task(%149)
      aiex.dma_await_task(%150)
      aiex.dma_await_task(%151)
      aiex.dma_await_task(%156)
      aiex.dma_await_task(%157)
      aiex.dma_await_task(%158)
      aiex.dma_await_task(%159)
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
      %176 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3146112 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%176)
      %177 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3932544 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%177)
      %178 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 384 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%178)
      %179 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 384 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%179)
      %180 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3146112 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%180)
      %181 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3539328 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%181)
      %182 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3932544 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%182)
      %183 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4325760 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%183)
      %184 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3147648 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%184)
      %185 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3934080 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%185)
      %186 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1920 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%186)
      %187 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1920 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%187)
      %188 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3147648 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%188)
      %189 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3540864 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%189)
      %190 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3934080 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%190)
      %191 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4327296 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%191)
      aiex.dma_await_task(%164)
      aiex.dma_await_task(%165)
      aiex.dma_await_task(%166)
      aiex.dma_await_task(%167)
      aiex.dma_await_task(%172)
      aiex.dma_await_task(%173)
      aiex.dma_await_task(%174)
      aiex.dma_await_task(%175)
      aiex.dma_free_task(%160)
      aiex.dma_free_task(%161)
      aiex.dma_free_task(%162)
      aiex.dma_free_task(%163)
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
      %192 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 512 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%192)
      %193 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 786944 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%193)
      %194 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 512 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%194)
      %195 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 512 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%195)
      %196 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 512 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%196)
      %197 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 393728 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%197)
      %198 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 786944 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%198)
      %199 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1180160 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%199)
      %200 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2048 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%200)
      %201 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 788480 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%201)
      %202 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2048 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%202)
      %203 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2048 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%203)
      %204 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2048 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%204)
      %205 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 395264 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%205)
      %206 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 788480 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%206)
      %207 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1181696 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%207)
      aiex.dma_await_task(%180)
      aiex.dma_await_task(%181)
      aiex.dma_await_task(%182)
      aiex.dma_await_task(%183)
      aiex.dma_await_task(%188)
      aiex.dma_await_task(%189)
      aiex.dma_await_task(%190)
      aiex.dma_await_task(%191)
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
      %208 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1573376 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%208)
      %209 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2359808 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%209)
      %210 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 512 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%210)
      %211 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 512 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%211)
      %212 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1573376 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%212)
      %213 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1966592 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%213)
      %214 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2359808 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%214)
      %215 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2753024 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%215)
      %216 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1574912 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%216)
      %217 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2361344 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%217)
      %218 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2048 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%218)
      %219 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2048 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%219)
      %220 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1574912 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%220)
      %221 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1968128 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%221)
      %222 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2361344 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%222)
      %223 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2754560 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%223)
      aiex.dma_await_task(%196)
      aiex.dma_await_task(%197)
      aiex.dma_await_task(%198)
      aiex.dma_await_task(%199)
      aiex.dma_await_task(%204)
      aiex.dma_await_task(%205)
      aiex.dma_await_task(%206)
      aiex.dma_await_task(%207)
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
      %224 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3146240 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%224)
      %225 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3932672 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%225)
      %226 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 512 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%226)
      %227 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 512 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%227)
      %228 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3146240 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%228)
      %229 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3539456 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%229)
      %230 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3932672 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%230)
      %231 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4325888 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%231)
      %232 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3147776 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%232)
      %233 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3934208 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%233)
      %234 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2048 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%234)
      %235 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2048 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%235)
      %236 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3147776 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%236)
      %237 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3540992 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%237)
      %238 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3934208 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%238)
      %239 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4327424 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%239)
      aiex.dma_await_task(%212)
      aiex.dma_await_task(%213)
      aiex.dma_await_task(%214)
      aiex.dma_await_task(%215)
      aiex.dma_await_task(%220)
      aiex.dma_await_task(%221)
      aiex.dma_await_task(%222)
      aiex.dma_await_task(%223)
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
      %240 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 640 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%240)
      %241 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 787072 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%241)
      %242 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 640 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%242)
      %243 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 640 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%243)
      %244 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 640 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%244)
      %245 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 393856 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%245)
      %246 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 787072 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%246)
      %247 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1180288 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%247)
      %248 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2176 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%248)
      %249 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 788608 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%249)
      %250 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2176 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%250)
      %251 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2176 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%251)
      %252 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2176 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%252)
      %253 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 395392 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%253)
      %254 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 788608 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%254)
      %255 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1181824 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%255)
      aiex.dma_await_task(%228)
      aiex.dma_await_task(%229)
      aiex.dma_await_task(%230)
      aiex.dma_await_task(%231)
      aiex.dma_await_task(%236)
      aiex.dma_await_task(%237)
      aiex.dma_await_task(%238)
      aiex.dma_await_task(%239)
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
      %256 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1573504 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%256)
      %257 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2359936 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%257)
      %258 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 640 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%258)
      %259 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 640 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%259)
      %260 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1573504 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%260)
      %261 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1966720 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%261)
      %262 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2359936 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%262)
      %263 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2753152 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%263)
      %264 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1575040 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%264)
      %265 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2361472 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%265)
      %266 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2176 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%266)
      %267 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2176 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%267)
      %268 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1575040 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%268)
      %269 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1968256 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%269)
      %270 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2361472 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%270)
      %271 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2754688 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%271)
      aiex.dma_await_task(%244)
      aiex.dma_await_task(%245)
      aiex.dma_await_task(%246)
      aiex.dma_await_task(%247)
      aiex.dma_await_task(%252)
      aiex.dma_await_task(%253)
      aiex.dma_await_task(%254)
      aiex.dma_await_task(%255)
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
      %272 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3146368 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%272)
      %273 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3932800 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%273)
      %274 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 640 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%274)
      %275 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 640 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%275)
      %276 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3146368 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%276)
      %277 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3539584 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%277)
      %278 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3932800 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%278)
      %279 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4326016 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%279)
      %280 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3147904 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%280)
      %281 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3934336 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%281)
      %282 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2176 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%282)
      %283 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2176 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%283)
      %284 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3147904 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%284)
      %285 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3541120 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%285)
      %286 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3934336 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%286)
      %287 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4327552 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%287)
      aiex.dma_await_task(%260)
      aiex.dma_await_task(%261)
      aiex.dma_await_task(%262)
      aiex.dma_await_task(%263)
      aiex.dma_await_task(%268)
      aiex.dma_await_task(%269)
      aiex.dma_await_task(%270)
      aiex.dma_await_task(%271)
      aiex.dma_free_task(%256)
      aiex.dma_free_task(%257)
      aiex.dma_free_task(%258)
      aiex.dma_free_task(%259)
      aiex.dma_free_task(%260)
      aiex.dma_free_task(%261)
      aiex.dma_free_task(%262)
      aiex.dma_free_task(%263)
      aiex.dma_free_task(%264)
      aiex.dma_free_task(%265)
      aiex.dma_free_task(%266)
      aiex.dma_free_task(%267)
      aiex.dma_free_task(%268)
      aiex.dma_free_task(%269)
      aiex.dma_free_task(%270)
      aiex.dma_free_task(%271)
      %288 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 768 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%288)
      %289 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 787200 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%289)
      %290 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 768 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%290)
      %291 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 768 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%291)
      %292 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 768 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%292)
      %293 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 393984 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%293)
      %294 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 787200 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%294)
      %295 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1180416 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%295)
      %296 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2304 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%296)
      %297 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 788736 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%297)
      %298 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2304 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%298)
      %299 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2304 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%299)
      %300 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2304 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%300)
      %301 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 395520 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%301)
      %302 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 788736 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%302)
      %303 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1181952 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%303)
      aiex.dma_await_task(%276)
      aiex.dma_await_task(%277)
      aiex.dma_await_task(%278)
      aiex.dma_await_task(%279)
      aiex.dma_await_task(%284)
      aiex.dma_await_task(%285)
      aiex.dma_await_task(%286)
      aiex.dma_await_task(%287)
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
      %304 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1573632 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%304)
      %305 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2360064 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%305)
      %306 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 768 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%306)
      %307 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 768 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%307)
      %308 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1573632 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%308)
      %309 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1966848 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%309)
      %310 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2360064 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%310)
      %311 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2753280 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%311)
      %312 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1575168 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%312)
      %313 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2361600 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%313)
      %314 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2304 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%314)
      %315 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2304 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%315)
      %316 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1575168 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%316)
      %317 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1968384 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%317)
      %318 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2361600 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%318)
      %319 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2754816 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%319)
      aiex.dma_await_task(%292)
      aiex.dma_await_task(%293)
      aiex.dma_await_task(%294)
      aiex.dma_await_task(%295)
      aiex.dma_await_task(%300)
      aiex.dma_await_task(%301)
      aiex.dma_await_task(%302)
      aiex.dma_await_task(%303)
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
      %320 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3146496 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%320)
      %321 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3932928 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%321)
      %322 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 768 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%322)
      %323 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 768 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%323)
      %324 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3146496 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%324)
      %325 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3539712 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%325)
      %326 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3932928 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%326)
      %327 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4326144 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%327)
      %328 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3148032 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%328)
      %329 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3934464 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%329)
      %330 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2304 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%330)
      %331 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2304 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%331)
      %332 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3148032 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%332)
      %333 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3541248 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%333)
      %334 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3934464 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%334)
      %335 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4327680 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%335)
      aiex.dma_await_task(%308)
      aiex.dma_await_task(%309)
      aiex.dma_await_task(%310)
      aiex.dma_await_task(%311)
      aiex.dma_await_task(%316)
      aiex.dma_await_task(%317)
      aiex.dma_await_task(%318)
      aiex.dma_await_task(%319)
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
      %336 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 896 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%336)
      %337 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 787328 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%337)
      %338 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 896 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%338)
      %339 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 896 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%339)
      %340 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 896 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%340)
      %341 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 394112 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%341)
      %342 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 787328 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%342)
      %343 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1180544 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%343)
      %344 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2432 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%344)
      %345 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 788864 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%345)
      %346 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2432 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%346)
      %347 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2432 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%347)
      %348 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2432 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%348)
      %349 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 395648 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%349)
      %350 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 788864 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%350)
      %351 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1182080 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%351)
      aiex.dma_await_task(%324)
      aiex.dma_await_task(%325)
      aiex.dma_await_task(%326)
      aiex.dma_await_task(%327)
      aiex.dma_await_task(%332)
      aiex.dma_await_task(%333)
      aiex.dma_await_task(%334)
      aiex.dma_await_task(%335)
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
      %352 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1573760 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%352)
      %353 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2360192 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%353)
      %354 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 896 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%354)
      %355 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 896 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%355)
      %356 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1573760 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%356)
      %357 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1966976 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%357)
      %358 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2360192 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%358)
      %359 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2753408 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%359)
      %360 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1575296 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%360)
      %361 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2361728 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%361)
      %362 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2432 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%362)
      %363 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2432 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%363)
      %364 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1575296 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%364)
      %365 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1968512 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%365)
      %366 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2361728 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%366)
      %367 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2754944 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%367)
      aiex.dma_await_task(%340)
      aiex.dma_await_task(%341)
      aiex.dma_await_task(%342)
      aiex.dma_await_task(%343)
      aiex.dma_await_task(%348)
      aiex.dma_await_task(%349)
      aiex.dma_await_task(%350)
      aiex.dma_await_task(%351)
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
      %368 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3146624 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%368)
      %369 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3933056 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%369)
      %370 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 896 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%370)
      %371 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 896 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%371)
      %372 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3146624 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%372)
      %373 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3539840 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%373)
      %374 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3933056 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%374)
      %375 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4326272 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%375)
      %376 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3148160 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%376)
      %377 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3934592 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%377)
      %378 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2432 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%378)
      %379 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2432 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%379)
      %380 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3148160 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%380)
      %381 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3541376 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%381)
      %382 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3934592 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%382)
      %383 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4327808 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%383)
      aiex.dma_await_task(%356)
      aiex.dma_await_task(%357)
      aiex.dma_await_task(%358)
      aiex.dma_await_task(%359)
      aiex.dma_await_task(%364)
      aiex.dma_await_task(%365)
      aiex.dma_await_task(%366)
      aiex.dma_await_task(%367)
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
      aiex.dma_free_task(%364)
      aiex.dma_free_task(%365)
      aiex.dma_free_task(%366)
      aiex.dma_free_task(%367)
      %384 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1024 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%384)
      %385 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 787456 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%385)
      %386 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1024 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%386)
      %387 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1024 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%387)
      %388 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1024 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%388)
      %389 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 394240 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%389)
      %390 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 787456 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%390)
      %391 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1180672 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%391)
      %392 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2560 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%392)
      %393 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 788992 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%393)
      %394 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2560 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%394)
      %395 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2560 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%395)
      %396 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2560 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%396)
      %397 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 395776 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%397)
      %398 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 788992 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%398)
      %399 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1182208 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%399)
      aiex.dma_await_task(%372)
      aiex.dma_await_task(%373)
      aiex.dma_await_task(%374)
      aiex.dma_await_task(%375)
      aiex.dma_await_task(%380)
      aiex.dma_await_task(%381)
      aiex.dma_await_task(%382)
      aiex.dma_await_task(%383)
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
      %400 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1573888 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%400)
      %401 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2360320 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%401)
      %402 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1024 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%402)
      %403 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1024 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%403)
      %404 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1573888 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%404)
      %405 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1967104 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%405)
      %406 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2360320 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%406)
      %407 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2753536 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%407)
      %408 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1575424 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%408)
      %409 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2361856 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%409)
      %410 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2560 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%410)
      %411 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2560 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%411)
      %412 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1575424 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%412)
      %413 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1968640 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%413)
      %414 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2361856 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%414)
      %415 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2755072 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%415)
      aiex.dma_await_task(%388)
      aiex.dma_await_task(%389)
      aiex.dma_await_task(%390)
      aiex.dma_await_task(%391)
      aiex.dma_await_task(%396)
      aiex.dma_await_task(%397)
      aiex.dma_await_task(%398)
      aiex.dma_await_task(%399)
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
      %416 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3146752 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%416)
      %417 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3933184 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%417)
      %418 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1024 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%418)
      %419 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1024 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%419)
      %420 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3146752 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%420)
      %421 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3539968 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%421)
      %422 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3933184 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%422)
      %423 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4326400 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%423)
      %424 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3148288 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%424)
      %425 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3934720 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%425)
      %426 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2560 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%426)
      %427 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2560 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%427)
      %428 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3148288 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%428)
      %429 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3541504 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%429)
      %430 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3934720 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%430)
      %431 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4327936 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%431)
      aiex.dma_await_task(%404)
      aiex.dma_await_task(%405)
      aiex.dma_await_task(%406)
      aiex.dma_await_task(%407)
      aiex.dma_await_task(%412)
      aiex.dma_await_task(%413)
      aiex.dma_await_task(%414)
      aiex.dma_await_task(%415)
      aiex.dma_free_task(%400)
      aiex.dma_free_task(%401)
      aiex.dma_free_task(%402)
      aiex.dma_free_task(%403)
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
      %432 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1152 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%432)
      %433 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 787584 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%433)
      %434 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1152 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%434)
      %435 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1152 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%435)
      %436 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1152 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%436)
      %437 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 394368 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%437)
      %438 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 787584 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%438)
      %439 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1180800 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%439)
      %440 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2688 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%440)
      %441 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 789120 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%441)
      %442 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2688 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%442)
      %443 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2688 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%443)
      %444 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2688 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%444)
      %445 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 395904 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%445)
      %446 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 789120 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%446)
      %447 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1182336 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%447)
      aiex.dma_await_task(%420)
      aiex.dma_await_task(%421)
      aiex.dma_await_task(%422)
      aiex.dma_await_task(%423)
      aiex.dma_await_task(%428)
      aiex.dma_await_task(%429)
      aiex.dma_await_task(%430)
      aiex.dma_await_task(%431)
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
      %448 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1574016 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%448)
      %449 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2360448 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%449)
      %450 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1152 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%450)
      %451 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1152 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%451)
      %452 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1574016 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%452)
      %453 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1967232 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%453)
      %454 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2360448 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%454)
      %455 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2753664 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%455)
      %456 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1575552 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%456)
      %457 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2361984 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%457)
      %458 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2688 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%458)
      %459 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2688 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%459)
      %460 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1575552 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%460)
      %461 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1968768 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%461)
      %462 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2361984 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%462)
      %463 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2755200 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%463)
      aiex.dma_await_task(%436)
      aiex.dma_await_task(%437)
      aiex.dma_await_task(%438)
      aiex.dma_await_task(%439)
      aiex.dma_await_task(%444)
      aiex.dma_await_task(%445)
      aiex.dma_await_task(%446)
      aiex.dma_await_task(%447)
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
      %464 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3146880 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%464)
      %465 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3933312 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%465)
      %466 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1152 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%466)
      %467 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1152 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%467)
      %468 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3146880 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%468)
      %469 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3540096 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%469)
      %470 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3933312 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%470)
      %471 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4326528 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%471)
      %472 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3148416 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%472)
      %473 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3934848 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%473)
      %474 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2688 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%474)
      %475 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2688 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%475)
      %476 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3148416 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%476)
      %477 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3541632 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%477)
      %478 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3934848 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%478)
      %479 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4328064 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%479)
      aiex.dma_await_task(%452)
      aiex.dma_await_task(%453)
      aiex.dma_await_task(%454)
      aiex.dma_await_task(%455)
      aiex.dma_await_task(%460)
      aiex.dma_await_task(%461)
      aiex.dma_await_task(%462)
      aiex.dma_await_task(%463)
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
      %480 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1280 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%480)
      %481 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 787712 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%481)
      %482 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1280 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%482)
      %483 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1280 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%483)
      %484 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1280 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%484)
      %485 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 394496 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%485)
      %486 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 787712 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%486)
      %487 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1180928 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%487)
      %488 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2816 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%488)
      %489 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 789248 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%489)
      %490 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2816 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%490)
      %491 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2816 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%491)
      %492 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2816 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%492)
      %493 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 396032 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%493)
      %494 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 789248 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%494)
      %495 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1182464 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%495)
      aiex.dma_await_task(%468)
      aiex.dma_await_task(%469)
      aiex.dma_await_task(%470)
      aiex.dma_await_task(%471)
      aiex.dma_await_task(%476)
      aiex.dma_await_task(%477)
      aiex.dma_await_task(%478)
      aiex.dma_await_task(%479)
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
      %496 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1574144 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%496)
      %497 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2360576 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%497)
      %498 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1280 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%498)
      %499 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1280 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%499)
      %500 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1574144 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%500)
      %501 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1967360 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%501)
      %502 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2360576 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%502)
      %503 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2753792 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%503)
      %504 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1575680 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%504)
      %505 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2362112 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%505)
      %506 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2816 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%506)
      %507 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2816 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%507)
      %508 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1575680 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%508)
      %509 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1968896 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%509)
      %510 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2362112 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%510)
      %511 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2755328 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%511)
      aiex.dma_await_task(%484)
      aiex.dma_await_task(%485)
      aiex.dma_await_task(%486)
      aiex.dma_await_task(%487)
      aiex.dma_await_task(%492)
      aiex.dma_await_task(%493)
      aiex.dma_await_task(%494)
      aiex.dma_await_task(%495)
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
      %512 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3147008 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%512)
      %513 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3933440 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%513)
      %514 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1280 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%514)
      %515 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1280 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%515)
      %516 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3147008 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%516)
      %517 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3540224 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%517)
      %518 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3933440 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%518)
      %519 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4326656 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%519)
      %520 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3148544 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%520)
      %521 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3934976 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%521)
      %522 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2816 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%522)
      %523 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2816 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%523)
      %524 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3148544 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%524)
      %525 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3541760 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%525)
      %526 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3934976 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%526)
      %527 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4328192 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%527)
      aiex.dma_await_task(%500)
      aiex.dma_await_task(%501)
      aiex.dma_await_task(%502)
      aiex.dma_await_task(%503)
      aiex.dma_await_task(%508)
      aiex.dma_await_task(%509)
      aiex.dma_await_task(%510)
      aiex.dma_await_task(%511)
      aiex.dma_free_task(%496)
      aiex.dma_free_task(%497)
      aiex.dma_free_task(%498)
      aiex.dma_free_task(%499)
      aiex.dma_free_task(%500)
      aiex.dma_free_task(%501)
      aiex.dma_free_task(%502)
      aiex.dma_free_task(%503)
      aiex.dma_free_task(%504)
      aiex.dma_free_task(%505)
      aiex.dma_free_task(%506)
      aiex.dma_free_task(%507)
      aiex.dma_free_task(%508)
      aiex.dma_free_task(%509)
      aiex.dma_free_task(%510)
      aiex.dma_free_task(%511)
      %528 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1408 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%528)
      %529 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 787840 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%529)
      %530 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1408 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%530)
      %531 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1408 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%531)
      %532 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1408 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%532)
      %533 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 394624 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%533)
      %534 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 787840 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%534)
      %535 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1181056 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%535)
      %536 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2944 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%536)
      %537 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 789376 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%537)
      %538 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2944 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%538)
      %539 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2944 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%539)
      %540 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2944 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%540)
      %541 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 396160 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%541)
      %542 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 789376 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%542)
      %543 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1182592 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%543)
      aiex.dma_await_task(%516)
      aiex.dma_await_task(%517)
      aiex.dma_await_task(%518)
      aiex.dma_await_task(%519)
      aiex.dma_await_task(%524)
      aiex.dma_await_task(%525)
      aiex.dma_await_task(%526)
      aiex.dma_await_task(%527)
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
      %544 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1574272 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%544)
      %545 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2360704 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%545)
      %546 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1408 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%546)
      %547 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1408 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%547)
      %548 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1574272 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%548)
      %549 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1967488 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%549)
      %550 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2360704 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%550)
      %551 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2753920 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%551)
      %552 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 1575808 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%552)
      %553 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 2362240 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%553)
      %554 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2944 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%554)
      %555 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2944 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%555)
      %556 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1575808 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%556)
      %557 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 1969024 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%557)
      %558 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2362240 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%558)
      %559 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 2755456 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%559)
      aiex.dma_await_task(%532)
      aiex.dma_await_task(%533)
      aiex.dma_await_task(%534)
      aiex.dma_await_task(%535)
      aiex.dma_await_task(%540)
      aiex.dma_await_task(%541)
      aiex.dma_await_task(%542)
      aiex.dma_await_task(%543)
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
      %560 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3147136 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%560)
      %561 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3933568 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%561)
      %562 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 1408 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%562)
      %563 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 1408 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%563)
      %564 = aiex.dma_configure_task_for @O_L2L3_0 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3147136 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%564)
      %565 = aiex.dma_configure_task_for @O_L2L3_1 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3540352 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%565)
      %566 = aiex.dma_configure_task_for @O_L2L3_2 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3933568 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%566)
      %567 = aiex.dma_configure_task_for @O_L2L3_3 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4326784 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%567)
      %568 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3148672 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%568)
      %569 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4718592xbf16> offset = 3935104 len = 32768 sizes = [1, 1, 256, 128] strides = [0, 0, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%569)
      %570 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4718592xbf16> offset = 2944 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%570)
      %571 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg2 : memref<4718592xbf16> offset = 2944 len = 196608 sizes = [1, 24, 64, 128] strides = [0, 196608, 3072, 1])
        aie.end
      }
      aiex.dma_start_task(%571)
      %572 = aiex.dma_configure_task_for @O_L2L3_4 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3148672 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%572)
      %573 = aiex.dma_configure_task_for @O_L2L3_5 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3541888 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%573)
      %574 = aiex.dma_configure_task_for @O_L2L3_6 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 3935104 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%574)
      %575 = aiex.dma_configure_task_for @O_L2L3_7 {
        aie.dma_bd(%arg3 : memref<4718592xbf16> offset = 4328320 len = 16384 sizes = [1, 1, 128, 128] strides = [0, 0, 3072, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%575)
      aiex.dma_await_task(%548)
      aiex.dma_await_task(%549)
      aiex.dma_await_task(%550)
      aiex.dma_await_task(%551)
      aiex.dma_await_task(%556)
      aiex.dma_await_task(%557)
      aiex.dma_await_task(%558)
      aiex.dma_await_task(%559)
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
      aiex.dma_await_task(%564)
      aiex.dma_await_task(%565)
      aiex.dma_await_task(%566)
      aiex.dma_await_task(%567)
      aiex.dma_await_task(%572)
      aiex.dma_await_task(%573)
      aiex.dma_await_task(%574)
      aiex.dma_await_task(%575)
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
    }
  }
}

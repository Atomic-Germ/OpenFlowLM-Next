// Phase 0b: the DDR-bounce expert fetch, final proof.
//
// Established on this NPU so far:
//   - a core-issued control packet reaches another CORE's control port (write)
//   - packets fed by the SHIM's own DMA reach the SHIM's control port and can
//     read/write shim NoC-module registers, DMA descriptors included
//   - packets from the array do NOT reach the shim's control port
//   - FLM's fused layer kernel keeps 32 routed-expert descriptors that its
//     instruction stream writes but never enqueues
// So the zero-host-round-trip design is: the array writes control-packet
// words to DDR (S2MM), the shim DMA streams them into its own control port,
// and those packets rewrite an expert descriptor's address and enqueue it.
// This file proves the last link with host-built packet words (identical to
// what a core would write): slab BD 2 on MM2S ch1 is configured but never
// enqueued by the sequence; the packets set its DDR address to slab `idx` of
// %big and push it; the core sums the slab it receives.
//   out[0] == 4096 * (idx + 1)  ->  PASS
//
// Register facts used (AIE-ML NoC module, confirmed by readback):
//   BD n @ 0x1D000 + 0x20*n:  w0 len, w1 addr_low (bits 1:0 zero), w2 addr_high[15:0]
//   DDR address the firmware writes = bo.address() + 0x8000_0000
//   MM2S ch0/1 ctrl 0x1D210/0x1D218, queue 0x1D214/0x1D21C = 0x8000_0000 | bd
//   S2MM ch0 ctrl 0x1D200, queue 0x1D204
module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %core = aie.tile(0, 2)

    %slab_buf = aie.buffer(%core) {sym_name = "slab_buf"} : memref<4096xi32>
    %res_buf  = aie.buffer(%core) {sym_name = "res_buf"}  : memref<4xi32>
    %slab_empty = aie.lock(%core, 0) {init = 1 : i32, sym_name = "slab_empty"}
    %slab_full  = aie.lock(%core, 1) {init = 0 : i32, sym_name = "slab_full"}
    %res_empty  = aie.lock(%core, 2) {init = 1 : i32, sym_name = "res_empty"}
    %res_full   = aie.lock(%core, 3) {init = 0 : i32, sym_name = "res_full"}

    aie.flow(%shim, DMA : 1, %core, DMA : 0)    // slab: shim MM2S ch1 -> core
    aie.flow(%core, DMA : 0, %shim, DMA : 0)    // result: core -> shim S2MM ch0
    aie.packet_flow(0x1) {                       // control: shim MM2S ch0 -> shim ctrl port
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%shim, TileControl : 0>
    }
    aie.shim_dma_allocation @out0 (%shim, S2MM, 0)

    aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4096 = arith.constant 4096 : index
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      aie.use_lock(%slab_full, AcquireGreaterEqual, %one)
      %sum = scf.for %i = %c0 to %c4096 step %c1 iter_args(%acc = %zero) -> (i32) {
        %v = memref.load %slab_buf[%i] : memref<4096xi32>
        %a = arith.addi %acc, %v : i32
        scf.yield %a : i32
      }
      aie.use_lock(%slab_empty, Release, %one)
      aie.use_lock(%res_empty, AcquireGreaterEqual, %one)
      memref.store %sum, %res_buf[%c0] : memref<4xi32>
      aie.use_lock(%res_full, Release, %one)
      aie.end
    }

    aie.mem(%core) {
      %s2mm0 = aie.dma_start(S2MM, 0, ^slab, ^n1)
    ^slab:
      %l1 = arith.constant 1 : i32
      aie.use_lock(%slab_empty, AcquireGreaterEqual, %l1)
      aie.dma_bd(%slab_buf : memref<4096xi32> offset = 0 len = 4096)
      %l2 = arith.constant 1 : i32
      aie.use_lock(%slab_full, Release, %l2)
      aie.next_bd ^slab
    ^n1:
      %mm2s0 = aie.dma_start(MM2S, 0, ^res, ^end)
    ^res:
      %l3 = arith.constant 1 : i32
      aie.use_lock(%res_full, AcquireGreaterEqual, %l3)
      aie.dma_bd(%res_buf : memref<4xi32> offset = 0 len = 4)
      %l4 = arith.constant 1 : i32
      aie.use_lock(%res_empty, Release, %l4)
      aie.next_bd ^res
    ^end:
      aie.end
    }

    // BD words (linear, no packet): from the compiler's own output for the same shapes.
    memref.global "private" constant @bd_out  : memref<8xi32> = dense<[4,      0, 0,          0, 0xc0000000, 0x2000000, 0, 0x2000000]>
    memref.global "private" constant @bd_slab : memref<8xi32> = dense<[0x1000, 0, 0,          0, 0xc0000000, 0x2000000, 0, 0x2000000]>
    // ctrl BD: packet-enabled, pkt id 1 (word2 = 0x40090000 as in mlir-aie's add_one_ctrl_packet)
    memref.global "private" constant @bd_ctrl : memref<8xi32> = dense<[3,      0, 0x40090000, 0, 0x40000000, 0,         0, 0x2000000]>

    aie.runtime_sequence @seq(%big: memref<1048576xi32>, %ctrl: memref<8xi32>, %out: memref<4xi32>) {
      // ---- result path: BD1 on S2MM ch0, started now, awaited at the end
      %g_out = memref.get_global @bd_out : memref<8xi32>
      aiex.npu.blockwrite(%g_out) {address = 0x1d020 : ui32, column = 0 : i32, row = 0 : i32} : memref<8xi32>
      %z0 = arith.constant 0 : i32
      aiex.npu.address_patch(%z0 : i32) {addr = 0x1d024 : ui32, arg_idx = 2 : i32}
      %r_s2mm0_ctrl = arith.constant 0x1d200 : i32
      %v_ctrlid = arith.constant 0x400 : i32
      %m_ctrlid = arith.constant 0x00000F00 : i32
      aiex.npu.maskwrite32(%r_s2mm0_ctrl, %v_ctrlid, %m_ctrlid) {column = 0 : i32, row = 0 : i32} : i32, i32, i32
      %r_s2mm0_q = arith.constant 0x1d204 : i32
      %v_push1 = arith.constant 0x80000001 : i32
      aiex.npu.write32(%r_s2mm0_q, %v_push1) {column = 0 : i32, row = 0 : i32} : i32, i32

      // ---- slab path: BD2 on MM2S ch1, configured (address = %big + 0), NOT pushed
      %g_slab = memref.get_global @bd_slab : memref<8xi32>
      aiex.npu.blockwrite(%g_slab) {address = 0x1d040 : ui32, column = 0 : i32, row = 0 : i32} : memref<8xi32>
      %z1 = arith.constant 0 : i32
      aiex.npu.address_patch(%z1 : i32) {addr = 0x1d044 : ui32, arg_idx = 0 : i32}
      %r_mm2s1_ctrl = arith.constant 0x1d218 : i32
      aiex.npu.maskwrite32(%r_mm2s1_ctrl, %v_ctrlid, %m_ctrlid) {column = 0 : i32, row = 0 : i32} : i32, i32, i32

      // ---- control path: BD0 on MM2S ch0 streams host-built packets into the shim's ctrl port
      %g_ctrl = memref.get_global @bd_ctrl : memref<8xi32>
      aiex.npu.blockwrite(%g_ctrl) {address = 0x1d000 : ui32, column = 0 : i32, row = 0 : i32} : memref<8xi32>
      %r_mm2s0_ctrl = arith.constant 0x1d210 : i32
      aiex.npu.maskwrite32(%r_mm2s0_ctrl, %v_ctrlid, %m_ctrlid) {column = 0 : i32, row = 0 : i32} : i32, i32, i32
      %r_mm2s0_q = arith.constant 0x1d214 : i32
      %v_push0 = arith.constant 0x80000000 : i32
      // packet 1: 3 words at %ctrl+0  -> writes BD2 w1/w2 (new DDR address)
      %z2 = arith.constant 0 : i32
      aiex.npu.address_patch(%z2 : i32) {addr = 0x1d004 : ui32, arg_idx = 1 : i32}
      aiex.npu.write32(%r_mm2s0_q, %v_push0) {column = 0 : i32, row = 0 : i32} : i32, i32
      %s0 = arith.constant 0 : i32
      %s1 = arith.constant 1 : i32
      aiex.npu.sync(%s0, %s0, %s1, %s0, %s1, %s1) : i32, i32, i32, i32, i32, i32
      // packet 2: 2 words at %ctrl+12 -> pushes BD2 onto MM2S ch1's task queue
      %r_bd0_len = arith.constant 0x1d000 : i32
      %v_len2 = arith.constant 2 : i32
      aiex.npu.write32(%r_bd0_len, %v_len2) {column = 0 : i32, row = 0 : i32} : i32, i32
      %off12 = arith.constant 12 : i32
      aiex.npu.address_patch(%off12 : i32) {addr = 0x1d004 : ui32, arg_idx = 1 : i32}
      aiex.npu.write32(%r_mm2s0_q, %v_push0) {column = 0 : i32, row = 0 : i32} : i32, i32
      aiex.npu.sync(%s0, %s0, %s1, %s0, %s1, %s1) : i32, i32, i32, i32, i32, i32

      // ---- the slab now streams into the core; wait for its checksum
      aiex.npu.dma_wait {symbol = @out0}
    }
  }
}

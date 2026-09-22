// Phase 2: core-generated control packets (on-device routing loop).
//
// The ddr_bounce_fetch spike proved the shim->shim control-port link with
// HOST-built packet words. This file closes the last gap: a CORE writes the
// control-packet words to DDR (S2MM), then the shim's own MM2S ch0 streams
// them into its control port, which pushes the slab BD2 (configured but never
// enqueued). No host round-trip.
//
//   core: build ctrl words -> %ctrl_buf --(core MM2S ch1)--> shim S2MM ch1 -> DDR %ctrl
//   shim: MM2S ch0 streams %ctrl --packet_flow(0x1)--> shim TileControl
//         packet words push BD2 onto MM2S ch1's queue
//   slab %big[idx] --(shim MM2S ch1 BD2)--> core -> sum -> %out
//
//   out[0] == 4096 * (idx + 1)  ->  PASS
module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %core = aie.tile(0, 2)

    %slab_buf = aie.buffer(%core) {sym_name = "slab_buf"} : memref<4096xi32>
    %res_buf  = aie.buffer(%core) {sym_name = "res_buf"}  : memref<4xi32>
    %ctrl_buf = aie.buffer(%core) {sym_name = "ctrl_buf"} : memref<2xi32>
    %slab_empty = aie.lock(%core, 0) {init = 1 : i32, sym_name = "slab_empty"}
    %slab_full  = aie.lock(%core, 1) {init = 0 : i32, sym_name = "slab_full"}
    %res_empty  = aie.lock(%core, 2) {init = 1 : i32, sym_name = "res_empty"}
    %res_full   = aie.lock(%core, 3) {init = 0 : i32, sym_name = "res_full"}
    %ctrl_empty = aie.lock(%core, 4) {init = 1 : i32, sym_name = "ctrl_empty"}
    %ctrl_full  = aie.lock(%core, 5) {init = 0 : i32, sym_name = "ctrl_full"}

    aie.flow(%shim, DMA : 1, %core, DMA : 0)   // slab: shim MM2S ch1 -> core S2MM ch0
    aie.flow(%core, DMA : 0, %shim, DMA : 0)   // result: core MM2S ch0 -> shim S2MM ch0
    aie.flow(%core, DMA : 1, %shim, DMA : 1)   // ctrl words: core MM2S ch1 -> shim S2MM ch1
    aie.packet_flow(0x1) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%shim, TileControl : 0>
    }
    aie.shim_dma_allocation @ctrl_out (%shim, S2MM, 1)
    aie.shim_dma_allocation @out0 (%shim, S2MM, 0)

    aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4096 = arith.constant 4096 : index
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      // control words: hdr(write 0x1D21C, 1 beat) + push BD2 (0x80000002)
      %h_push = arith.constant 0x801D21C : i32
      %v_push = arith.constant 0x80000002 : i32
      aie.use_lock(%ctrl_empty, AcquireGreaterEqual, %one)
      memref.store %h_push, %ctrl_buf[%c0] : memref<2xi32>
      memref.store %v_push, %ctrl_buf[%c1] : memref<2xi32>
      aie.use_lock(%ctrl_full, Release, %one)
      // wait for the slab, sum it
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
      %mm2s0 = aie.dma_start(MM2S, 0, ^res, ^n2)
    ^res:
      %l3 = arith.constant 1 : i32
      aie.use_lock(%res_full, AcquireGreaterEqual, %l3)
      aie.dma_bd(%res_buf : memref<4xi32> offset = 0 len = 4)
      %l4 = arith.constant 1 : i32
      aie.use_lock(%res_empty, Release, %l4)
      aie.next_bd ^res
    ^n2:
      %mm2s1 = aie.dma_start(MM2S, 1, ^ctrl, ^end)
    ^ctrl:
      %l5 = arith.constant 1 : i32
      aie.use_lock(%ctrl_full, AcquireGreaterEqual, %l5)
      aie.dma_bd(%ctrl_buf : memref<2xi32> offset = 0 len = 2)
      %l6 = arith.constant 1 : i32
      aie.use_lock(%ctrl_empty, Release, %l6)
      aie.next_bd ^ctrl
    ^end:
      aie.end
    }

    // BD words (w0 len in i32, w1 addr_low, w2 addr_high; bytes 16-31 = the
    // 2D / packet / wrap fields the compiler writes for these shapes).
    memref.global "private" constant @bd_out  : memref<8xi32> = dense<[4, 0, 0, 0, 0xc0000000, 0x2000000, 0, 0x2000000]>
    memref.global "private" constant @bd_slab : memref<8xi32> = dense<[0x1000, 0, 0, 0, 0xc0000000, 0x2000000, 0, 0x2000000]>
    memref.global "private" constant @bd_ctrl : memref<8xi32> = dense<[3, 0, 0x40090000, 0, 0x40000000, 0, 0, 0x2000000]>
    memref.global "private" constant @bd_ctrlw : memref<8xi32> = dense<[2, 0, 0, 0, 0xc0000000, 0x2000000, 0, 0x2000000]>

    aie.runtime_sequence @seq(%big: memref<1048576xi32>, %ctrl: memref<2xi32>, %out: memref<4xi32>) {
      %v_ctrlid = arith.constant 0x400 : i32
      %m_ctrlid = arith.constant 0x00000F00 : i32

      // ---- ctrl-write path: BD3 on S2MM ch1 receives the core's control words -> DDR %ctrl
      %g_ctrlw = memref.get_global @bd_ctrlw : memref<8xi32>
      aiex.npu.blockwrite(%g_ctrlw) {address = 0x1d060 : ui32, column = 0 : i32, row = 0 : i32} : memref<8xi32>
      %z1 = arith.constant 0 : i32
      aiex.npu.address_patch(%z1 : i32) {addr = 0x1d064 : ui32, arg_idx = 1 : i32}
      %r_s2mm1_ctrl = arith.constant 0x1d208 : i32
      aiex.npu.maskwrite32(%r_s2mm1_ctrl, %v_ctrlid, %m_ctrlid) {column = 0 : i32, row = 0 : i32} : i32, i32, i32
      %r_s2mm1_q = arith.constant 0x1d20c : i32
      %v_push3 = arith.constant 0x80000003 : i32
      aiex.npu.write32(%r_s2mm1_q, %v_push3) {column = 0 : i32, row = 0 : i32} : i32, i32

      // ---- result path: BD1 on S2MM ch0 receives the core's sum -> %out
      %g_out = memref.get_global @bd_out : memref<8xi32>
      aiex.npu.blockwrite(%g_out) {address = 0x1d020 : ui32, column = 0 : i32, row = 0 : i32} : memref<8xi32>
      %z0 = arith.constant 0 : i32
      aiex.npu.address_patch(%z0 : i32) {addr = 0x1d024 : ui32, arg_idx = 2 : i32}
      %r_s2mm0_ctrl = arith.constant 0x1d200 : i32
      aiex.npu.maskwrite32(%r_s2mm0_ctrl, %v_ctrlid, %m_ctrlid) {column = 0 : i32, row = 0 : i32} : i32, i32, i32
      %r_s2mm0_q = arith.constant 0x1d204 : i32
      %v_push1 = arith.constant 0x80000001 : i32
      aiex.npu.write32(%r_s2mm0_q, %v_push1) {column = 0 : i32, row = 0 : i32} : i32, i32

      // ---- slab path: BD2 on MM2S ch1, configured (address = %big + 0), NOT pushed
      %g_slab = memref.get_global @bd_slab : memref<8xi32>
      aiex.npu.blockwrite(%g_slab) {address = 0x1d040 : ui32, column = 0 : i32, row = 0 : i32} : memref<8xi32>
      %z2 = arith.constant 0 : i32
      aiex.npu.address_patch(%z2 : i32) {addr = 0x1d044 : ui32, arg_idx = 0 : i32}
      %r_mm2s1_ctrl = arith.constant 0x1d218 : i32
      aiex.npu.maskwrite32(%r_mm2s1_ctrl, %v_ctrlid, %m_ctrlid) {column = 0 : i32, row = 0 : i32} : i32, i32, i32

      // ---- wait until the core's control words are in DDR %ctrl
      aiex.npu.dma_wait {symbol = @ctrl_out}

      // ---- control path: BD0 on MM2S ch0 streams the core's words into the shim ctrl port
      %g_ctrl = memref.get_global @bd_ctrl : memref<8xi32>
      aiex.npu.blockwrite(%g_ctrl) {address = 0x1d000 : ui32, column = 0 : i32, row = 0 : i32} : memref<8xi32>
      %z3 = arith.constant 0 : i32
      aiex.npu.address_patch(%z3 : i32) {addr = 0x1d004 : ui32, arg_idx = 1 : i32}
      %r_mm2s0_ctrl = arith.constant 0x1d210 : i32
      aiex.npu.maskwrite32(%r_mm2s0_ctrl, %v_ctrlid, %m_ctrlid) {column = 0 : i32, row = 0 : i32} : i32, i32, i32
      %r_mm2s0_q = arith.constant 0x1d214 : i32
      %v_push0 = arith.constant 0x80000000 : i32
      aiex.npu.write32(%r_mm2s0_q, %v_push0) {column = 0 : i32, row = 0 : i32} : i32, i32

      // the slab now streams into the core; wait for its checksum
      aiex.npu.dma_wait {symbol = @out0}
    }
  }
}

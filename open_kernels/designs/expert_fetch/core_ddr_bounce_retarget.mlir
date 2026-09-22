// Phase 2b: core-generated control packets that RETARGET a descriptor's DDR
// address (not just enqueue it), driven by a runtime input [base_low, base_high,
// idx]. This is the last primitive the fused whole-layer MoE needs: the router
// core computes the top-8 experts' pool addresses and writes control words that
// (a) rewrite each routed-expert descriptor's addr_low/addr_high and (b) enqueue
// it. No host round-trip.
//
//   core: read %in [base_low, base_high, idx] --(shim MM2S ch1 BD1)--> S2MM ch0
//         compute addr = base + idx*16384
//         write ctrl words [hdr(0x1D044,2 beats), addr_low, addr_high,
//                           hdr(0x1D21C,1 beat), 0x80000002]
//           --(core MM2S ch1)--> shim S2MM ch1 -> DDR %ctrl
//   shim: MM2S ch0 streams %ctrl --packet_flow(0x1)--> shim TileControl
//         packet rewrites BD2 w1/w2 + pushes BD2 to MM2S ch1 queue
//   slab %big[idx*4096 ..] --(shim MM2S ch1 BD2)--> core -> sum -> %out
//
//   out[0] == 4096 * (idx + 1)  ->  PASS
module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %core = aie.tile(0, 2)

    %slab_buf = aie.buffer(%core) {sym_name = "slab_buf"} : memref<4096xi32>
    %in_buf   = aie.buffer(%core) {sym_name = "in_buf"}   : memref<4xi32>
    %res_buf  = aie.buffer(%core) {sym_name = "res_buf"}  : memref<4xi32>
    %ctrl_buf = aie.buffer(%core) {sym_name = "ctrl_buf"} : memref<6xi32>
    %in_empty   = aie.lock(%core, 0) {init = 1 : i32, sym_name = "in_empty"}
    %in_full    = aie.lock(%core, 1) {init = 0 : i32, sym_name = "in_full"}
    %slab_empty = aie.lock(%core, 2) {init = 1 : i32, sym_name = "slab_empty"}
    %slab_full  = aie.lock(%core, 3) {init = 0 : i32, sym_name = "slab_full"}
    %res_empty  = aie.lock(%core, 4) {init = 1 : i32, sym_name = "res_empty"}
    %res_full   = aie.lock(%core, 5) {init = 0 : i32, sym_name = "res_full"}
    %ctrl_empty = aie.lock(%core, 6) {init = 1 : i32, sym_name = "ctrl_empty"}
    %ctrl_full  = aie.lock(%core, 7) {init = 0 : i32, sym_name = "ctrl_full"}

    aie.flow(%shim, DMA : 1, %core, DMA : 0)   // input+slab: shim MM2S ch1 -> core S2MM ch0
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
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c14 = arith.constant 14 : i32
      %c4096 = arith.constant 4096 : index
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32

      // ---- read [base_low, base_high, idx] and compute the retarget address
      aie.use_lock(%in_full, AcquireGreaterEqual, %one)
      %b0  = memref.load %in_buf[%c0] : memref<4xi32>
      %b1  = memref.load %in_buf[%c1] : memref<4xi32>
      %idx = memref.load %in_buf[%c2] : memref<4xi32>
      aie.use_lock(%in_empty, Release, %one)
      // addr = base + idx*4096*4 bytes = base + idx*16384 = base + (idx<<14)
      %sh14 = arith.shli %idx, %c14 : i32
      %al = arith.addi %b0, %sh14 : i32
      %lt = arith.cmpi ult, %al, %b0 : i32
      %carry = arith.extui %lt : i1 to i32
      %ah = arith.addi %b1, %carry : i32

      // ---- write control words: rewrite BD2 w1/w2 + enqueue BD2
      %hdr1 = arith.constant 0x0011D044 : i32   // write 0x1D044, 2 beats
      %hdr2 = arith.constant 0x801D21C : i32    // write 0x1D21C (MM2S ch1 queue), 1 beat
      %enq  = arith.constant 0x80000002 : i32   // push BD2
      aie.use_lock(%ctrl_empty, AcquireGreaterEqual, %one)
      memref.store %hdr1, %ctrl_buf[%c0] : memref<6xi32>
      memref.store %al,   %ctrl_buf[%c1] : memref<6xi32>
      memref.store %ah,   %ctrl_buf[%c2] : memref<6xi32>
      memref.store %hdr2, %ctrl_buf[%c3] : memref<6xi32>
      memref.store %enq,  %ctrl_buf[%c4] : memref<6xi32>
      aie.use_lock(%ctrl_full, Release, %one)

      // ---- wait for the slab, sum it
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
      %s2mm0 = aie.dma_start(S2MM, 0, ^in, ^n1)
    ^in:
      %l1 = arith.constant 1 : i32
      aie.use_lock(%in_empty, AcquireGreaterEqual, %l1)
      aie.dma_bd(%in_buf : memref<4xi32> offset = 0 len = 3)
      %l2 = arith.constant 1 : i32
      aie.use_lock(%in_full, Release, %l2)
      aie.next_bd ^slab
    ^slab:
      %l3 = arith.constant 1 : i32
      aie.use_lock(%slab_empty, AcquireGreaterEqual, %l3)
      aie.dma_bd(%slab_buf : memref<4096xi32> offset = 0 len = 4096)
      %l4 = arith.constant 1 : i32
      aie.use_lock(%slab_full, Release, %l4)
      aie.next_bd ^in
    ^n1:
      %mm2s0 = aie.dma_start(MM2S, 0, ^res, ^n2)
    ^res:
      %l5 = arith.constant 1 : i32
      aie.use_lock(%res_full, AcquireGreaterEqual, %l5)
      aie.dma_bd(%res_buf : memref<4xi32> offset = 0 len = 4)
      %l6 = arith.constant 1 : i32
      aie.use_lock(%res_empty, Release, %l6)
      aie.next_bd ^res
    ^n2:
      %mm2s1 = aie.dma_start(MM2S, 1, ^ctrl, ^end)
    ^ctrl:
      %l7 = arith.constant 1 : i32
      aie.use_lock(%ctrl_full, AcquireGreaterEqual, %l7)
      aie.dma_bd(%ctrl_buf : memref<6xi32> offset = 0 len = 5)
      %l8 = arith.constant 1 : i32
      aie.use_lock(%ctrl_empty, Release, %l8)
      aie.next_bd ^ctrl
    ^end:
      aie.end
    }

    // BD words (w0 len in i32, w1 addr_low, w2 addr_high; bytes 16-31 = 2D/packet/wrap).
    memref.global "private" constant @bd_in   : memref<8xi32> = dense<[3, 0, 0, 0, 0xc0000000, 0x2000000, 0, 0x2000000]>
    memref.global "private" constant @bd_slab : memref<8xi32> = dense<[0x1000, 0, 0, 0, 0xc0000000, 0x2000000, 0, 0x2000000]>
    memref.global "private" constant @bd_out  : memref<8xi32> = dense<[4, 0, 0, 0, 0xc0000000, 0x2000000, 0, 0x2000000]>
    memref.global "private" constant @bd_ctrl : memref<8xi32> = dense<[3, 0, 0x40090000, 0, 0x40000000, 0, 0, 0x2000000]>
    memref.global "private" constant @bd_ctrlw: memref<8xi32> = dense<[5, 0, 0, 0, 0xc0000000, 0x2000000, 0, 0x2000000]>

    aie.runtime_sequence @seq(%big: memref<1048576xi32>, %ctrl: memref<6xi32>, %out: memref<4xi32>, %in: memref<4xi32>) {
      %v_ctrlid = arith.constant 0x400 : i32
      %m_ctrlid = arith.constant 0x00000F00 : i32

      // ---- input BD1 on MM2S ch1 (reads %in -> core)
      %g_in = memref.get_global @bd_in : memref<8xi32>
      aiex.npu.blockwrite(%g_in) {address = 0x1d020 : ui32, column = 0 : i32, row = 0 : i32} : memref<8xi32>
      %zin = arith.constant 0 : i32
      aiex.npu.address_patch(%zin : i32) {addr = 0x1d024 : ui32, arg_idx = 3 : i32}
      %r_mm2s1_ctrl = arith.constant 0x1d218 : i32
      aiex.npu.maskwrite32(%r_mm2s1_ctrl, %v_ctrlid, %m_ctrlid) {column = 0 : i32, row = 0 : i32} : i32, i32, i32
      %r_mm2s1_q = arith.constant 0x1d21c : i32
      %v_push1 = arith.constant 0x80000001 : i32
      aiex.npu.write32(%r_mm2s1_q, %v_push1) {column = 0 : i32, row = 0 : i32} : i32, i32

      // ---- ctrl-write BD4 on S2MM ch1 (core's control words -> DDR %ctrl)
      %g_ctrlw = memref.get_global @bd_ctrlw : memref<8xi32>
      aiex.npu.blockwrite(%g_ctrlw) {address = 0x1d080 : ui32, column = 0 : i32, row = 0 : i32} : memref<8xi32>
      %zcw = arith.constant 0 : i32
      aiex.npu.address_patch(%zcw : i32) {addr = 0x1d084 : ui32, arg_idx = 1 : i32}
      %r_s2mm1_ctrl = arith.constant 0x1d208 : i32
      aiex.npu.maskwrite32(%r_s2mm1_ctrl, %v_ctrlid, %m_ctrlid) {column = 0 : i32, row = 0 : i32} : i32, i32, i32
      %r_s2mm1_q = arith.constant 0x1d20c : i32
      %v_push4 = arith.constant 0x80000004 : i32
      aiex.npu.write32(%r_s2mm1_q, %v_push4) {column = 0 : i32, row = 0 : i32} : i32, i32

      // ---- result BD3 on S2MM ch0 (core's sum -> %out)
      %g_out = memref.get_global @bd_out : memref<8xi32>
      aiex.npu.blockwrite(%g_out) {address = 0x1d060 : ui32, column = 0 : i32, row = 0 : i32} : memref<8xi32>
      %zo = arith.constant 0 : i32
      aiex.npu.address_patch(%zo : i32) {addr = 0x1d064 : ui32, arg_idx = 2 : i32}
      %r_s2mm0_ctrl = arith.constant 0x1d200 : i32
      aiex.npu.maskwrite32(%r_s2mm0_ctrl, %v_ctrlid, %m_ctrlid) {column = 0 : i32, row = 0 : i32} : i32, i32, i32
      %r_s2mm0_q = arith.constant 0x1d204 : i32
      %v_push3 = arith.constant 0x80000003 : i32
      aiex.npu.write32(%r_s2mm0_q, %v_push3) {column = 0 : i32, row = 0 : i32} : i32, i32

      // ---- slab BD2 on MM2S ch1, configured (address = %big + 0), NOT pushed
      %g_slab = memref.get_global @bd_slab : memref<8xi32>
      aiex.npu.blockwrite(%g_slab) {address = 0x1d040 : ui32, column = 0 : i32, row = 0 : i32} : memref<8xi32>
      %zs = arith.constant 0 : i32
      aiex.npu.address_patch(%zs : i32) {addr = 0x1d044 : ui32, arg_idx = 0 : i32}
      aiex.npu.maskwrite32(%r_mm2s1_ctrl, %v_ctrlid, %m_ctrlid) {column = 0 : i32, row = 0 : i32} : i32, i32, i32

      // ---- wait until the core's control words are in DDR %ctrl
      aiex.npu.dma_wait {symbol = @ctrl_out}

      // ---- control path: BD0 on MM2S ch0 streams the core's words into the shim ctrl port
      %g_ctrl = memref.get_global @bd_ctrl : memref<8xi32>
      aiex.npu.blockwrite(%g_ctrl) {address = 0x1d000 : ui32, column = 0 : i32, row = 0 : i32} : memref<8xi32>
      %r_mm2s0_ctrl = arith.constant 0x1d210 : i32
      aiex.npu.maskwrite32(%r_mm2s0_ctrl, %v_ctrlid, %m_ctrlid) {column = 0 : i32, row = 0 : i32} : i32, i32, i32
      %r_mm2s0_q = arith.constant 0x1d214 : i32
      %v_push0 = arith.constant 0x80000000 : i32
      // packet 1: 3 words at %ctrl+0 -> rewrites BD2 w1/w2 (new DDR address)
      %zc = arith.constant 0 : i32
      aiex.npu.address_patch(%zc : i32) {addr = 0x1d004 : ui32, arg_idx = 1 : i32}
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

      // the slab now streams into the core; wait for its checksum
      aiex.npu.dma_wait {symbol = @out0}
    }
  }
}

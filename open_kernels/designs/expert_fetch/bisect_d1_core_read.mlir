// Bisect C: CORE-sourced control-packet READ of a SHIM register, reply back
// into the core. The core asks the shim's control port for BD0 word0 (the
// slab descriptor's length, 0x1000 words in this design) with stream_id 5;
// the shim's TileControl emits the reply as packet id 5, routed into the
// core's S2MM1. The core forwards the reply words out.
//   out[0] == 0x1000  -> packets reach the shim control port and reads work
//   timeout           -> packets never reach the shim's control port (or the
//                        reply path is wrong): the shim is the problem, not
//                        the write op.
module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %dst = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %core = aie.tile(0, 2)

    %ctrl_buf = aie.buffer(%core) {sym_name = "ctrl_buf"} : memref<8xi32>
    %rsp_buf  = aie.buffer(%core) {sym_name = "rsp_buf"}  : memref<8xi32>
    %res_buf  = aie.buffer(%core) {sym_name = "res_buf"}  : memref<8xi32>
    %ctrl_empty = aie.lock(%core, 4) {init = 1 : i32, sym_name = "ctrl_empty"}
    %ctrl_full  = aie.lock(%core, 5) {init = 0 : i32, sym_name = "ctrl_full"}
    %rsp_empty  = aie.lock(%core, 6) {init = 1 : i32, sym_name = "rsp_empty"}
    %rsp_full   = aie.lock(%core, 7) {init = 0 : i32, sym_name = "rsp_full"}
    %res_empty  = aie.lock(%core, 2) {init = 1 : i32, sym_name = "res_empty"}
    %res_full   = aie.lock(%core, 3) {init = 0 : i32, sym_name = "res_full"}

    aie.flow(%core, DMA : 0, %shim, DMA : 0)          // result out
    aie.packet_flow(0x4) {                             // request: core -> shim ctrl port
      aie.packet_source<%core, DMA : 1>
      aie.packet_dest<%dst, TileControl : 0>
    }
    aie.packet_flow(0x5) {                             // reply: shim ctrl port -> core
      aie.packet_source<%dst, TileControl : 0>
      aie.packet_dest<%core, DMA : 1>
    }
    %marker = aie.lock(%dst, 5) {init = 42 : i32, sym_name = "marker"}
    aie.shim_dma_allocation @slab_in (%shim, MM2S, 0)
    aie.shim_dma_allocation @res_out (%shim, S2MM, 0)

    aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %one = arith.constant 1 : i32
      // READ header: stream_id 5 <<24 | opcode 1 <<22 | beats 3 <<20 | 0x1D000
      //   = 0x05000000 | 0x00400000 | 0x00300000 | 0x1D000 = 0x0571D000
      //   popcount(0x0571D000): 0x05->2, 0x71->4, 0xD0->3, 0x00->0 = 9 (odd) -> parity 0
      %h_read = arith.constant 0x8571f050 : i32   // read 4 words at (0,3) 0x1F050 (lock 5..8 values)
      %pad = arith.constant 0 : i32
      aie.use_lock(%ctrl_empty, AcquireGreaterEqual, %one)
      memref.store %h_read, %ctrl_buf[%c0] : memref<8xi32>
      memref.store %pad,    %ctrl_buf[%c1] : memref<8xi32>
      aie.use_lock(%ctrl_full, Release, %one)

      // wait for the 4-word reply
      aie.use_lock(%rsp_full, AcquireGreaterEqual, %one)
      aie.use_lock(%res_empty, AcquireGreaterEqual, %one)
      %r0 = memref.load %rsp_buf[%c0] : memref<8xi32>
      %r1 = memref.load %rsp_buf[%c1] : memref<8xi32>
      %r2 = memref.load %rsp_buf[%c2] : memref<8xi32>
      %r3 = memref.load %rsp_buf[%c3] : memref<8xi32>
      memref.store %r0, %res_buf[%c0] : memref<8xi32>
      memref.store %r1, %res_buf[%c1] : memref<8xi32>
      memref.store %r2, %res_buf[%c2] : memref<8xi32>
      memref.store %r3, %res_buf[%c3] : memref<8xi32>
      aie.use_lock(%rsp_empty, Release, %one)
      aie.use_lock(%res_full, Release, %one)
      aie.end
    }

    aie.mem(%core) {
      %mm2s0 = aie.dma_start(MM2S, 0, ^res, ^n1)
    ^res:
      %l3 = arith.constant 1 : i32
      aie.use_lock(%res_full, AcquireGreaterEqual, %l3)
      aie.dma_bd(%res_buf : memref<8xi32> offset = 0 len = 8)
      %l4 = arith.constant 1 : i32
      aie.use_lock(%res_empty, Release, %l4)
      aie.next_bd ^res
    ^n1:
      %mm2s1 = aie.dma_start(MM2S, 1, ^ctrl, ^n2)
    ^ctrl:
      %l5 = arith.constant 1 : i32
      aie.use_lock(%ctrl_full, AcquireGreaterEqual, %l5)
      aie.dma_bd(%ctrl_buf : memref<8xi32> offset = 0 len = 2) {packet = #aie.packet_info<pkt_id = 4, pkt_type = 0>}
      %l6 = arith.constant 1 : i32
      aie.use_lock(%ctrl_empty, Release, %l6)
      aie.next_bd ^ctrl
    ^n2:
      %s2mm1 = aie.dma_start(S2MM, 1, ^rsp, ^end)
    ^rsp:
      %l7 = arith.constant 1 : i32
      aie.use_lock(%rsp_empty, AcquireGreaterEqual, %l7)
      aie.dma_bd(%rsp_buf : memref<8xi32> offset = 0 len = 4)
      %l8 = arith.constant 1 : i32
      aie.use_lock(%rsp_full, Release, %l8)
      aie.next_bd ^rsp
    ^end:
      aie.end
    }

    aie.runtime_sequence @seq(%big: memref<1048576xi32>, %out: memref<8xi32>) {
      %t_slab = aiex.dma_configure_task_for @slab_in {
        aie.dma_bd(%big : memref<1048576xi32> offset = 0 len = 4096)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @res_out {
        aie.dma_bd(%out : memref<8xi32> offset = 0 len = 8)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
    }
  }
}

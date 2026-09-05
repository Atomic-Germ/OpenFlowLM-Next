// Bisect A: HOST-sourced control packet -> CORE tile TileControl.
// Same pattern as mlir-aie's add_one_ctrl_packet test (known-good on npu2):
// the packet writes the core's lock-5 value register (0x1F050) to 1, which
// releases the core; the core then publishes a magic word. If `out` == magic,
// our packet word encoding (header/parity/opcode/beats) is right.
module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}

    %res_buf = aie.buffer(%core) {sym_name = "res_buf"} : memref<4xi32>
    %res_empty = aie.lock(%core, 2) {init = 1 : i32, sym_name = "res_empty"}
    %res_full  = aie.lock(%core, 3) {init = 0 : i32, sym_name = "res_full"}
    %go        = aie.lock(%core, 5) {init = 0 : i32, sym_name = "go"}   // 0x1F050

    aie.flow(%core, DMA : 0, %shim, DMA : 0)
    aie.packet_flow(0x4) {
      aie.packet_source<%shim, DMA : 1>
      aie.packet_dest<%core, TileControl : 0>
    }
    aie.shim_dma_allocation @ctrl (%shim, MM2S, 1)
    aie.shim_dma_allocation @res_out (%shim, S2MM, 0)

    aie.core(%core) {
      %c0 = arith.constant 0 : index
      %one = arith.constant 1 : i32
      %magic = arith.constant 0xC0FFEE : i32
      aie.use_lock(%go, AcquireGreaterEqual, %one)        // released ONLY by the packet
      aie.use_lock(%res_empty, AcquireGreaterEqual, %one)
      memref.store %magic, %res_buf[%c0] : memref<4xi32>
      aie.use_lock(%res_full, Release, %one)
      aie.end
    }

    aie.mem(%core) {
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

    aie.runtime_sequence @seq(%ctrl: memref<8xi32>, %out: memref<4xi32>) {
      %t_ctrl = aiex.dma_configure_task_for @ctrl {
        aie.dma_bd(%ctrl : memref<8xi32> offset = 0 len = 8) {packet = #aie.packet_info<pkt_id = 4, pkt_type = 0>}
        aie.end
      } {issue_token = true}
      %t_out = aiex.dma_configure_task_for @res_out {
        aie.dma_bd(%out : memref<4xi32> offset = 0 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out)
      aiex.dma_start_task(%t_ctrl)
      aiex.dma_await_task(%t_ctrl)
      aiex.dma_await_task(%t_out)
    }
  }
}

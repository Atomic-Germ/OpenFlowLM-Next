// Bisect B: CORE-sourced control packet -> another CORE's TileControl.
// Core (0,2) builds the packet in its own memory and its tile DMA (MM2S1,
// packet id 4) sends it; core (0,3) is released by the write to its lock-5
// value register (0x1F050) and publishes a magic word. Passing this while
// core->SHIM fails would isolate the shim as the problematic destination.
module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %src  = aie.tile(0, 2)
    %dst  = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}

    // sender side
    %ctrl_buf  = aie.buffer(%src) {sym_name = "ctrl_buf"} : memref<8xi32>
    %ctrl_empty = aie.lock(%src, 4) {init = 1 : i32, sym_name = "ctrl_empty"}
    %ctrl_full  = aie.lock(%src, 5) {init = 0 : i32, sym_name = "ctrl_full"}
    // receiver side
    %res_buf   = aie.buffer(%dst) {sym_name = "res_buf"} : memref<4xi32>
    %res_empty = aie.lock(%dst, 2) {init = 1 : i32, sym_name = "res_empty"}
    %res_full  = aie.lock(%dst, 3) {init = 0 : i32, sym_name = "res_full"}
    %go        = aie.lock(%dst, 5) {init = 0 : i32, sym_name = "go"}   // 0x1F050 on (0,3)

    aie.flow(%dst, DMA : 0, %shim, DMA : 0)
    aie.packet_flow(0x4) {
      aie.packet_source<%src, DMA : 1>
      aie.packet_dest<%dst, TileControl : 0>
    }
    aie.shim_dma_allocation @res_out (%shim, S2MM, 0)

    aie.core(%src) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %one = arith.constant 1 : i32
      // hdr(0x1F050, 1 data word): popcount(0x1F050) = 7 (odd) -> parity bit 0
      %h = arith.constant 0x0001F050 : i32
      aie.use_lock(%ctrl_empty, AcquireGreaterEqual, %one)
      // 4 identical (idempotent) write packets = 8 words, in case tiny
      // transfers misbehave (LLMNpuTest: 128-byte shim transfers deliver zeros).
      memref.store %h,   %ctrl_buf[%c0] : memref<8xi32>
      memref.store %one, %ctrl_buf[%c1] : memref<8xi32>
      memref.store %h,   %ctrl_buf[%c2] : memref<8xi32>
      memref.store %one, %ctrl_buf[%c3] : memref<8xi32>
      memref.store %h,   %ctrl_buf[%c4] : memref<8xi32>
      memref.store %one, %ctrl_buf[%c5] : memref<8xi32>
      memref.store %h,   %ctrl_buf[%c6] : memref<8xi32>
      memref.store %one, %ctrl_buf[%c7] : memref<8xi32>
      aie.use_lock(%ctrl_full, Release, %one)
      aie.end
    }

    aie.mem(%src) {
      %mm2s1 = aie.dma_start(MM2S, 1, ^ctrl, ^end)
    ^ctrl:
      %l5 = arith.constant 1 : i32
      aie.use_lock(%ctrl_full, AcquireGreaterEqual, %l5)
      aie.dma_bd(%ctrl_buf : memref<8xi32> offset = 0 len = 8) {packet = #aie.packet_info<pkt_id = 4, pkt_type = 0>}
      %l6 = arith.constant 1 : i32
      aie.use_lock(%ctrl_empty, Release, %l6)
      aie.next_bd ^ctrl
    ^end:
      aie.end
    }

    aie.core(%dst) {
      %c0 = arith.constant 0 : index
      %one = arith.constant 1 : i32
      %magic = arith.constant 0xC0FFEE : i32
      aie.use_lock(%go, AcquireGreaterEqual, %one)
      aie.use_lock(%res_empty, AcquireGreaterEqual, %one)
      memref.store %magic, %res_buf[%c0] : memref<4xi32>
      aie.use_lock(%res_full, Release, %one)
      aie.end
    }

    aie.mem(%dst) {
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

    aie.runtime_sequence @seq(%out: memref<4xi32>) {
      %t_out = aiex.dma_configure_task_for @res_out {
        aie.dma_bd(%out : memref<4xi32> offset = 0 len = 4)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
    }
  }
}

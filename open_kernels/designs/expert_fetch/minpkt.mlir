//===- core_ctrlpkt.mlir -----------------------------------*- MLIR -*-===//
//
// Phase 0b spike, the decisive version: a CORE emits control packets that
// enqueue a SHIM DMA descriptor, pulling a slab of a large DDR buffer into
// the array with no host round-trip.
//
// Why this shape (and not shim -> its own TileControl, which timed out):
// the target model only allows a shim switchbox to reach its TileControl
// port from FIFO/South/West/North/East, not from its own DMA. Packets from
// a core tile arrive on South, which is legal -- and matches FastFlowLM,
// whose 32 routed-expert descriptors are written by the txn stream but never
// enqueued by it.
//
// Data flow:
//   DDR %big --(shim MM2S0, BD configured-not-started)--> core S2MM0 %slab_buf
//   core: build ctrl words -> %ctrl_buf --(core MM2S1, packet id 4)--> shim TileControl
//         then sum the slab -> %res_buf --(core MM2S0)--> shim S2MM0 --> DDR %out
//
// The control words enable the shim's MM2S ch0 and push BD 1 to its task
// queue (0x1D210 / 0x1D214, BD base 0x1D020 -- read off the compiler's own
// instruction stream for this design).
//
module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %core = aie.tile(0, 2)

    // ---- buffers ---------------------------------------------------------
    %slab_buf = aie.buffer(%core) {sym_name = "slab_buf"} : memref<4096xi32>
    %res_buf  = aie.buffer(%core) {sym_name = "res_buf"}  : memref<4xi32>
    %ctrl_buf = aie.buffer(%core) {sym_name = "ctrl_buf"} : memref<8xi32>

    // ---- locks (AIE2 semaphores) ----------------------------------------
    %slab_empty = aie.lock(%core, 0) {init = 1 : i32, sym_name = "slab_empty"}
    %slab_full  = aie.lock(%core, 1) {init = 0 : i32, sym_name = "slab_full"}
    %res_empty  = aie.lock(%core, 2) {init = 1 : i32, sym_name = "res_empty"}
    %res_full   = aie.lock(%core, 3) {init = 0 : i32, sym_name = "res_full"}
    %ctrl_empty = aie.lock(%core, 4) {init = 1 : i32, sym_name = "ctrl_empty"}
    %ctrl_full  = aie.lock(%core, 5) {init = 0 : i32, sym_name = "ctrl_full"}

    // ---- streams ---------------------------------------------------------
    aie.flow(%shim, DMA : 0, %core, DMA : 0)   // slab: shim -> core
    aie.flow(%core, DMA : 0, %shim, DMA : 0)   // result: core -> shim
    aie.packet_flow(0x4) {                      // control: core -> shim ctrl port
      aie.packet_source<%core, DMA : 1>
      aie.packet_dest<%shim, TileControl : 0>
    }

    aie.shim_dma_allocation @slab_in (%shim, MM2S, 0)
    aie.shim_dma_allocation @res_out (%shim, S2MM, 0)

    // ---- core ------------------------------------------------------------
    aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4096 = arith.constant 4096 : index
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32

      // One control packet: push the slab BD to the shim's MM2S ch0 task queue.
      // Header layout (mlir-aie AIETargetNPU.cpp):
      //   stream_id<<24 | opcode<<22 | (beats = n-1)<<20 | address, parity in bit 31
      //   opcode 0 = write; parity bit is 1 when the popcount of the rest is even.
      // hdr(0x1D214, 1 data word): popcount(0x1D214) = 7 (odd) -> bit31 = 0.
      // Data word = task queue value = enable_token_issue<<31 | start_bd_id;
      // the compiler's own stream uses 0x8000_0000 | bd, but nothing consumes
      // this task's completion token here, so push the BD id alone.
      // The slab BD is id 0 in this design (see build_core/insts.bin).
      %h_push = arith.constant 0x8001D204 : i32
      %v_push = arith.constant 0x80000001 : i32

      // MINIMAL PACKET TEST: the slab is host-started (proven good); the ONE
      // thing the core's control packet does is enqueue the RESULT descriptor
      // (shim S2MM ch0, BD 1) that the runtime sequence deliberately no longer
      // starts. If out.bin comes back with the right sum, a core-emitted
      // control packet reached the shim's control port and took effect.
      //   hdr(0x1D204): popcount 6 (even) -> parity bit 1 -> 0x8001D204
      //   value: enable_token_issue | start_bd_id = 0x8000_0001
      //     (token issue on, because the sequence still awaits this task's TCT)
      aie.use_lock(%ctrl_empty, AcquireGreaterEqual, %one)
      memref.store %h_push, %ctrl_buf[%c0] : memref<8xi32>
      memref.store %v_push, %ctrl_buf[%c1] : memref<8xi32>
      aie.use_lock(%ctrl_full, Release, %one)

      // The slab arrives because of the packet we just sent.
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

    // ---- core tile DMAs --------------------------------------------------
    aie.mem(%core) {
      %s2mm0 = aie.dma_start(S2MM, 0, ^slab, ^next1)
    ^slab:
      %l1 = arith.constant 1 : i32
      aie.use_lock(%slab_empty, AcquireGreaterEqual, %l1)
      aie.dma_bd(%slab_buf : memref<4096xi32> offset = 0 len = 4096)
      %l2 = arith.constant 1 : i32
      aie.use_lock(%slab_full, Release, %l2)
      aie.next_bd ^slab
    ^next1:
      %mm2s0 = aie.dma_start(MM2S, 0, ^res, ^next2)
    ^res:
      %l3 = arith.constant 1 : i32
      aie.use_lock(%res_full, AcquireGreaterEqual, %l3)
      aie.dma_bd(%res_buf : memref<4xi32> offset = 0 len = 4)
      %l4 = arith.constant 1 : i32
      aie.use_lock(%res_empty, Release, %l4)
      aie.next_bd ^res
    ^next2:
      %mm2s1 = aie.dma_start(MM2S, 1, ^ctrl, ^end)
    ^ctrl:
      %l5 = arith.constant 1 : i32
      aie.use_lock(%ctrl_full, AcquireGreaterEqual, %l5)
      aie.dma_bd(%ctrl_buf : memref<8xi32> offset = 0 len = 2) {packet = #aie.packet_info<pkt_id = 4, pkt_type = 0>}
      %l6 = arith.constant 1 : i32
      aie.use_lock(%ctrl_empty, Release, %l6)
      aie.next_bd ^ctrl
    ^end:
      aie.end
    }

    // ---- runtime sequence -------------------------------------------------
    aie.runtime_sequence @seq(%big: memref<1048576xi32>, %out: memref<4xi32>) {
      // Slab BD: configured, deliberately NOT started. The core's control
      // packet supplies the enqueue. Address comes from the compiler's own
      // ddr patch of %big (offset 0 for this first cut -- proving the enqueue
      // mechanism; the address rewrite is the next step).
      %t_slab = aiex.dma_configure_task_for @slab_in {
        aie.dma_bd(%big : memref<1048576xi32> offset = 0 len = 4096)
        aie.end
      }

      %t_out = aiex.dma_configure_task_for @res_out {
        aie.dma_bd(%out : memref<4xi32> offset = 0 len = 4)
        aie.end
      } {issue_token = true}

      aiex.dma_start_task(%t_slab)
      aiex.dma_await_task(%t_out)
    }
  }
}

//===- ctrlpkt_shim_bd.mlir --------------------------------*- MLIR -*-===//
//
// Phase 0b spike, step 1: can a control packet retarget a SHIM DMA buffer
// descriptor and trigger it, so that a data-dependent slab of a large DDR
// buffer streams into a core -- with no host round-trip per fetch?
//
// This is the mechanism FLM's fused decode layer kernel must be using for
// routed MoE experts: its control code writes 32 shim BDs pointing at pool
// offset 0 (8 experts x 4 gate/up stripes) and never enqueues them from the
// txn stream; every task-queue write in the txn names the *static* weight BDs
// only. So the expert BDs are retargeted + pushed from inside the array.
//
// Step 1 (this file) sends the control packets from the HOST's shim DMA into
// the same shim tile's TileControl port. It proves the packet encoding, the
// BD register layout and the task-queue push. Step 2 moves the packet source
// to a core tile (aie.packet_source<%core, DMA:1>), which is the part that
// actually removes the host from the loop.
//
// Register map (AIE-ML / AIE2P NOC module, confirmed against FLM's own txn):
//   BD n:            0x1D000 + 0x20*n   word0 = length (32b)
//                                       word1 = base_address_low  << 2
//                                       word2 = base_address_high (bits 15:0)
//   MM2S ch0 ctrl:   0x1D210            queue:  0x1D214
//   MM2S ch1 ctrl:   0x1D218            queue:  0x1D21C
// Task queue push = write 0x8000_0000 | bd_id.
//
// Data flow: DDR `big` (a 4 MB buffer the host fills with a known pattern,
// one distinct value per 64 KB slab) -> shim MM2S ch1 (BD 8) -> core tile
// (0,2) -> core sums the slab and writes the sum -> shim S2MM -> DDR `out`.
// The slab index comes from `idx` (a DDR word), read by the core, turned into
// a control packet that rewrites BD 8's address, then pushed.
//
// Build:  see build_mlir.sh in this directory.
//
module {
  aie.device(npu2_1col) {
    // controller_id must be set on any tile that receives control packets.
    %shim = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %core = aie.tile(0, 2)

    // ---- data path: shim -> core (slab), core -> shim (result) ----------
    aie.objectfifo @slab (%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo @res  (%core, {%shim}, 1 : i32) : !aie.objectfifo<memref<4xi32>>

    // ---- control path: host shim DMA -> this shim tile's control port ---
    // MM2S ch1 so the control stream does not share a channel with @slab's
    // data stream (which the compiler puts on ch0).
    aie.packet_flow(0x4) {
      aie.packet_source<%shim, DMA : 1>
      aie.packet_dest<%shim, TileControl : 0>
    }

    aie.shim_dma_allocation @ctrl (%shim, MM2S, 1)

    %c = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4096 = arith.constant 4096 : index
      %zero = arith.constant 0 : i32

      // consume one slab, sum it, publish the sum
      %in = aie.objectfifo.acquire @slab (Consume, 1) : !aie.objectfifosubview<memref<4096xi32>>
      %inb = aie.objectfifo.subview.access %in[0] : !aie.objectfifosubview<memref<4096xi32>> -> memref<4096xi32>
      %sum = scf.for %i = %c0 to %c4096 step %c1 iter_args(%acc = %zero) -> (i32) {
        %v = memref.load %inb[%i] : memref<4096xi32>
        %a = arith.addi %acc, %v : i32
        scf.yield %a : i32
      }
      %ov = aie.objectfifo.acquire @res (Produce, 1) : !aie.objectfifosubview<memref<4xi32>>
      %ob = aie.objectfifo.subview.access %ov[0] : !aie.objectfifosubview<memref<4xi32>> -> memref<4xi32>
      memref.store %sum, %ob[%c0] : memref<4xi32>
      aie.objectfifo.release @res (Produce, 1)
      aie.objectfifo.release @slab (Consume, 1)
      aie.end
    }

    // ---- runtime sequence ------------------------------------------------
    // %big  : the large "pool" (arg 0)
    // %ctrl : control packet words, built by the host (arg 1)
    // %out  : result (arg 2)
    aie.runtime_sequence @seq(%big: memref<1048576xi32>, %ctrl: memref<16xi32>, %out: memref<4xi32>) {
      // Send the host-built control packets into the shim's control port.
      // They rewrite BD 8's base address to point at the chosen slab of %big
      // and push it to the MM2S ch1 task queue.
      %t_ctrl = aiex.dma_configure_task_for @ctrl {
        // enable_packet + pkt_id 4 so the stream switch routes these words to
        // the shim's own TileControl port via packet_flow(0x4) above.
        aie.dma_bd(%ctrl : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_id = 4, pkt_type = 0>}
        aie.end
      } {issue_token = true}

      // Slab DMA: CONFIGURED but deliberately NOT started. The compiler emits
      // the full BD setup (length, stride, lock, packet fields) pointing at
      // %big offset 0; the control packet then rewrites only the address words
      // and pushes the task queue -- exactly the shape FLM's layer kernel uses
      // for its 32 routed-expert descriptors.
      %t_slab = aiex.dma_configure_task_for @slab {
        aie.dma_bd(%big : memref<1048576xi32> offset = 0 len = 4096)
        aie.end
      }

      // Result DMA (core -> DDR).
      %t_out = aiex.dma_configure_task_for @res {
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

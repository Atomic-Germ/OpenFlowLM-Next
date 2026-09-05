// Phase 0b, END-TO-END: the ARRAY chooses which slab of a DDR pool to fetch,
// with no host involvement between the choice and the fetch.
//
//   1. host: %ctrl[0..1] = DDR address of %big (+0x8000_0000 aperture), %ctrl[2] = idx
//   2. sequence streams %ctrl[0..7] into the core (params)           [shim MM2S1, BD3]
//   3. core: addr = base + idx*16384; builds the 5 control-packet words
//      [hdr(BD2.w1, 2 words), addr_lo, addr_hi, hdr(MM2S1 queue), 0x8000_0002]
//      and DMAs them to DDR %ctrl[8..12]                              [core MM2S1 -> shim S2MM1, BD4]
//   4. sequence waits for that write (TCT), then streams %ctrl[8..12] into the
//      shim's own control port                                        [shim MM2S0, BD0, packet id 1]
//   5. the packets retarget slab BD2 (MM2S1) and enqueue it
//   6. the slab lands in the core, which sums it -> %out               [core MM2S0 -> shim S2MM0, BD1]
//   out[0] == 4096*(idx+1)  ->  PASS
// In a MoE layer, step 3 is the router's top-k turned into 32 such packets.
module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
    %core = aie.tile(0, 2)

    %params_buf = aie.buffer(%core) {sym_name = "params_buf"} : memref<8xi32>
    %slab_buf   = aie.buffer(%core) {sym_name = "slab_buf"}   : memref<4096xi32>
    %pkt_buf    = aie.buffer(%core) {sym_name = "pkt_buf"}    : memref<8xi32>
    %res_buf    = aie.buffer(%core) {sym_name = "res_buf"}    : memref<4xi32>
    %params_empty = aie.lock(%core, 8) {init = 1 : i32, sym_name = "params_empty"}
    %params_full  = aie.lock(%core, 9) {init = 0 : i32, sym_name = "params_full"}
    %slab_empty = aie.lock(%core, 0) {init = 1 : i32, sym_name = "slab_empty"}
    %slab_full  = aie.lock(%core, 1) {init = 0 : i32, sym_name = "slab_full"}
    %res_empty  = aie.lock(%core, 2) {init = 1 : i32, sym_name = "res_empty"}
    %res_full   = aie.lock(%core, 3) {init = 0 : i32, sym_name = "res_full"}
    %pkt_empty  = aie.lock(%core, 4) {init = 1 : i32, sym_name = "pkt_empty"}
    %pkt_full   = aie.lock(%core, 5) {init = 0 : i32, sym_name = "pkt_full"}

    aie.flow(%shim, DMA : 1, %core, DMA : 0)    // params then slab: shim MM2S1 -> core S2MM0
    aie.flow(%core, DMA : 0, %shim, DMA : 0)    // result: core MM2S0 -> shim S2MM0
    aie.flow(%core, DMA : 1, %shim, DMA : 1)    // packet words to DDR: core MM2S1 -> shim S2MM1
    aie.packet_flow(0x1) {                       // control: shim MM2S0 -> shim ctrl port
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%shim, TileControl : 0>
    }
    aie.shim_dma_allocation @out0 (%shim, S2MM, 0)
    aie.shim_dma_allocation @pkt0 (%shim, S2MM, 1)

    aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      %slab_bytes = arith.constant 16384 : i32
      // control-packet headers (stream_id 0, opcode write, parity bit 31):
      //   0x1D044 (BD2 w1), 2 data words -> beats 1: 0x0011D044, popcount 7 -> bit31 0
      //   0x1D21C (MM2S1 queue), 1 word         : 0x0001D21C, popcount 8 -> bit31 1 => 0x8001D21
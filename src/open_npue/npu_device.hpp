//===- npu_device.hpp ---------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- dispatch a compiled design on the NPU through XRT.
// SPDX-License-Identifier: MIT
//
// One `Design` owns one xclbin, its instruction stream, and its buffers. The
// xclbin is loaded once and kept: F1 prescribes one resident xclbin, and
// tasks/0010 measured ~150 us of fixed cost per dispatch, so anything that
// reloads per call has already lost.
//
// The kernel signature comes from main_kernels.json in the build cache:
//   MLIR_AIE(opcode, instr*, ninstr, bo0, bo1, bo2, bo3, bo4)
// Buffer COUNT varies by design -- GEMM and GELU take (in, in, out) and
// (in, out); LayerNorm takes (in, params, out), because a core tile has only
// two input DMA channels and gamma+beta had to be packed into one buffer
// (tasks/0020). So buffers are a vector, sized from design.json.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// INSIDE npue, not at global scope. A host application can
// legitimately have its own `npu` -- OpenFlowLM-Next does,
// and MSVC refuses this declaration outright there (C2757).
// A three-letter top-level namespace from a library is a
// name nobody else can use. tasks/0156 B2.
namespace npue {
namespace npu {

// How data buffers are allocated. Set before any Design is constructed.
// See npu_device.cpp -- this exists to make "does allocation flavour affect
// DMA throughput" a measurement rather than an argument.
enum class BoMode { host_only, host_only_1m, ext, ext_1m };
void set_bo_mode(BoMode m);
BoMode bo_mode();
const char *bo_mode_name();
// Alignment of the last data buffer actually handed back, in bytes. Reported
// so the mode's effect on addresses is visible rather than assumed.
size_t last_bo_alignment();

// Metadata the build step wrote alongside the xclbin. The runtime asserts on it
// rather than trusting that the binary on disk matches what it is asked for --
// tools/export_xclbin.py once handed the same xclbin to four different designs,
// and only a check like this makes that loud.
struct DesignInfo {
  std::string name;
  std::string kind;                  // gemm | eltwise | layernorm | softmax
  int64_t M = 0, K = 0, N = 0;       // gemm only
  // The sequence length the design was compiled for. Distinct from the
  // container's max_seq_len, which is how many position embeddings were
  // packed; 0 means an export that predates the field.
  int64_t seq = 0;
  std::vector<size_t> buffer_bytes;  // in declaration order

  // What layout this design's B operand must be in, as the same sha256 that
  // tools/npue.py stamps into every tiled tensor. Empty means the design.json
  // did not say -- which the runtime treats as a failure, not as permission.
  std::string b_layout_hash;

  // Element size of C as the design actually emits it: 4 for fp32, 2 when the
  // design was exported with `--c-bf16` and narrows on the core after the fp32
  // K reduction (tasks/0045). READ, never assumed -- a bf16 artifact and an
  // fp32 one differ only here, and guessing wrong reads every result at the
  // wrong stride.
  //
  // A design.json with no "c_dtype" is an export that predates the field, and
  // every one of those IS fp32, so 4 is the correct reading of silence rather
  // than a default papering over a missing value.
  size_t c_elem_bytes = 4;

  // Element size of the A/B operands: 2 for bf16, 1 for int8 (tasks/0078).
  // READ from design.json's `a_dtype`, never assumed -- an int8 design needs
  // the host to quantise A on the way in and dequantise C on the way out, and
  // a bf16 runtime pointed at one would transfer plausible bytes and compute
  // nothing meaningful.
  size_t a_elem_bytes = 2;
  // True when C leaves the core as int32 rather than fp32/bf16, i.e. the
  // int8 datapath, whose accumulator carries no scale at all until the host
  // applies one.
  bool c_is_int = false;

  // The design's N-tile width and column count, READ from design.json
  // (b_layout's "tile_n", and "cols"), never assumed. 0 means an export that
  // predates the fields. Added for T47 (tasks/0124): --probe-streams
  // hardcoded 48 and 8 here, which silently inflated every published int8
  // GB/s figure by 1.57-1.85x -- bge-large ships tile_n 32 at bf16 and 64 at
  // int8, and neither is 48.
  int64_t tile_n = 0;
  int64_t cols = 0;

  // True when this design's MMAC was built with
  // emulate_bf16_mmul_with_bfp16 (tasks/0104, T23). This changes MMAC
  // precision, not operand storage or B's tiling, so a_elem_bytes,
  // c_elem_bytes and b_layout_hash are IDENTICAL between a bfp16 design and
  // a plain-bf16 one at the same geometry -- this is the only field that
  // tells them apart. READ from design.json's "emulate_bfp16", never
  // assumed.
  bool emulate_bfp16 = false;

  // ABSENT IS NOT THE SAME AS FALSE, and conflating them printed a lie the
  // first time this shipped: a pre-0104 directory that really was built
  // `--emulate-bfp16` has no such key, so it was reported as plain bf16
  // while running the emulated datapath -- caught by its `1-cos` reading
  // 3.397e-04 where plain bf16 gives 1.086e-05.
  //
  // Selection still treats absent as bf16, deliberately: every design set
  // this project has ever shipped predates the field, and refusing them
  // would break bge-small, whose plain-bf16 `artifacts_b128il` is exactly
  // such a directory. But REPORTING must distinguish the two, so an
  // unrecorded datapath reads as unknown rather than as a claim. Re-export a
  // set to give it the key.
  bool datapath_recorded = false;

  // WHICH RUNTIME-SEQUENCE BARRIER SCHEDULE THIS DESIGN WAS BUILT WITH
  // (T61-2, tasks/0152). `tb_max_n_rows` is how many row blocks the sequence
  // issues between TCT barriers (x2; 4 = 2 per ping-pong half), and
  // `tg_depth` is how many of those halves may be outstanding at once: 1 is
  // the pre-tasks/0152 schedule, which awaits each half immediately after
  // issuing it, so nothing is ever in flight across a barrier.
  //
  // Neither changes geometry, datapath, b_layout_hash or final.xclbin -- only
  // insts.bin -- so two directories differing ONLY in these are byte-for-byte
  // indistinguishable to every other check in this struct. That is trap 7c's
  // shape exactly, and it is why they are read and REPORTED rather than
  // assumed. `schedule_recorded` false means an export that predates
  // tasks/0152, which is the serial schedule at tb_max_n_rows = 4 -- but it
  // reads as UNRECORDED, never as a claim.
  int64_t tb_max_n_rows = 4;
  int64_t tg_depth = 1;
  bool schedule_recorded = false;

  // WHICH TOOLCHAIN BUILT THIS DESIGN (T39, tasks/0106). Read from a
  // toolchain.json the export tools write NEXT TO design.json -- not a key
  // inside design.json itself, so an older parser reading design.json still
  // works unchanged. tasks/0102's audit found kernel config hash `333c4d33`
  // names TWO different instruction streams (0 `crrnd` instructions before
  // the mlir-aie 1.3.4 -> 1.4.2 upgrade, 3 after) with nothing in the build
  // path able to tell them apart; this is what tells them apart from here on.
  //
  // "unavailable" (not "") is what a field reads when the export machine
  // could read the toolchain.json file but a specific value inside it
  // couldn't be established -- e.g. C:\dev\mlir-aie was not a git checkout,
  // or `git` itself was unavailable, at export time. The export must not
  // fail over provenance; a missing value is recorded as missing, not
  // guessed at and not a build error.
  std::string mlir_aie_version = "unavailable";
  std::string peano_version = "unavailable";
  std::string mlir_aie_git_head = "unavailable";

  // False when this design's directory has no toolchain.json at all --
  // i.e. it was exported before tasks/0106. Same ABSENT-IS-NOT-A-VALUE
  // discipline as datapath_recorded just above: an unrecorded toolchain
  // reads as UNRECORDED, never silently as "unavailable" (which claims the
  // export tried and failed) or omitted (which claims nobody thought to
  // ask).
  bool toolchain_recorded = false;
};

class Device {
public:
  Device();
  ~Device();
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  struct Impl;
  Impl *impl() const { return impl_.get(); }

private:
  std::unique_ptr<Impl> impl_;
};

// An in-flight dispatch: what `Design::submit()` hands back and
// `Design::wait()` blocks on.
//
// T61-1 (tasks/0152). `dispatch_only()` submits AND waits, so a caller that
// holds a lock across it holds it through the hardware. Splitting the two lets
// the shared state (the bind registers, the instruction slot) be locked while
// the wait -- which touches nothing shared -- is not. It also lets one lane's
// command sit in the hw_context queue behind another's, which is the point:
// the array picks up the next command without a host round trip in between.
//
// It owns an `xrt::run`, which references the buffer objects it was built
// from; those must not be rebound before the wait returns. Each lane binds its
// own slots, which is what makes that true.
class Dispatch {
public:
  Dispatch();
  ~Dispatch();
  Dispatch(Dispatch &&) noexcept;
  Dispatch &operator=(Dispatch &&) noexcept;
  Dispatch(const Dispatch &) = delete;
  Dispatch &operator=(const Dispatch &) = delete;
  bool valid() const { return static_cast<bool>(impl_); }

  struct Impl;

private:
  friend class Design;
  std::unique_ptr<Impl> impl_;
};

class Design {
public:
  // `dir` holds final.xclbin, insts.bin and design.json from
  // tools/export_xclbin.py.
  Design(Device &dev, const std::string &dir);
  ~Design();
  Design(const Design &) = delete;
  Design &operator=(const Design &) = delete;

  const DesignInfo &info() const { return info_; }

  // Submit and wait charged separately, summed across all dispatches. Public
  // because the answer decides what to optimise and hiding it would mean
  // guessing again.
  //
  // WRITTEN FROM SEVERAL LANES since tasks/0152 -- the wait no longer happens
  // under the caller's NPU mutex -- so every update goes through Design's own
  // stats lock. Read them after the lanes have joined.
  double t_submit = 0.0, t_wait = 0.0;
  int n_dispatch = 0;

  // Wall-clock time during which at least one dispatch of this design was IN
  // FLIGHT (submitted, not yet reaped) -- the union of the in-flight
  // intervals, not their sum.
  //
  // This exists because `t_submit + t_wait` stopped meaning "the array was
  // busy" the moment waits were allowed to overlap: four lanes each waiting
  // 1 ms on the same queued command sum to 4 ms across a 1 ms group. The union
  // is the honest host-side occupancy figure. It is still a HOST observation
  // of when the queue was non-empty, never an NPU performance claim (rule 1).
  double t_occupied = 0.0;

  // Host pointers in declaration order. Inputs are copied in and synced to the
  // device; outputs are synced back and copied out. Which is which is decided
  // by `output_index` -- everything else is an input.
  void run(const std::vector<const void *> &inputs, void *output);

  // Dispatches only, reusing whatever is already in the device buffers. This is
  // what makes a benchmark measure the NPU rather than the memcpy around it.
  void dispatch_only();

  // The two halves of dispatch_only(), so a caller can hold a lock across the
  // first and not the second. `submit()` reads the currently-bound slots and
  // the active instruction stream, so it must be called under whatever lock
  // guards bind(); `wait()` touches only the handle.
  Dispatch submit();
  void wait(Dispatch &h);

  // Load an ADDITIONAL instruction stream into this design's context and
  // return its slot id; slot 0 is the stream the design was constructed with.
  // bind_instr() selects which stream the next dispatch_only() replays.
  //
  // This is the mechanism behind the one-xclbin hypothesis (research/notes/
  // 0004 step 0): final.xclbin is the STATIC configuration and insts.bin is
  // the per-dispatch runtime sequence, so two operations whose static design
  // is identical are just two instruction streams over one hw_context -- and
  // switching between them should cost a dispatch, not a context switch.
  size_t load_instr(const std::string &path);
  void bind_instr(size_t slot);

  // Stage a buffer on the device once and keep it there; returns a slot id for
  // bind(). Slot 0 is always the design's own buffer.
  //
  // This exists for weights. Four GEMM designs serve six layers, so a design's
  // B buffer held a different layer's weights on every call and was refilled
  // from the mapped .npue each time -- 21 MB of memcpy per encode, of data that
  // never changes. Staging all 24 weight sets costs 21 MB of device buffers
  // once and removes that copy entirely (tasks/0024).
  size_t stage(size_t arg_index, const void *data, size_t bytes);

  // Allocate an EMPTY staged slot -- same slot semantics as stage(), no data.
  // This is what gives each pipeline of the two-encode overlap (tasks/0033)
  // its own A and C buffers on the shared design.
  size_t stage_alloc(size_t arg_index, size_t bytes);

  // Allocate `count` buffers of `chunk_bytes` through the same path the
  // encoder uses, touch each so the pages actually commit, and return how
  // many succeeded. bge-large needs ~1 GB of XRT buffers against MiniLM's
  // 175 MB, and finding the ceiling costs one command instead of a
  // four-xclbin build. The buffers are freed when `Design` is destroyed.
  size_t probe_alloc(size_t chunk_bytes, size_t count, bool verbose);

  // Host pointer of a SPECIFIC slot, independent of what is currently bound.
  // A pipeline converts into its own slot outside the dispatch lock; the
  // bind happens inside it.
  void *slot_ptr(size_t arg_index, size_t slot);

  // Choose which staged buffer argument `arg_index` dispatches with.
  void bind(size_t arg_index, size_t slot);

  // Direct access for callers that want to stage once and dispatch many times.
  // These act on whatever is currently bound to `index`.
  //
  // `bytes` = 0 syncs the whole buffer. The unified gemm_rtp design sizes its
  // buffers for the LARGEST stream (ffn_up's C is 50 MB), so a partial sync
  // of what the current stream actually touches is the difference between
  // syncing 6 MB and 50 MB on every qkv call.
  void *host_ptr(size_t index);
  void sync_to_device(size_t index, size_t bytes = 0);
  void sync_from_device(size_t index, size_t bytes = 0);

  // The same two syncs addressed by SLOT rather than by whatever is currently
  // bound -- the buffer-transfer analogue of slot_ptr(). A lane can then move
  // its own operand across the bus without holding the bind lock, which is
  // half of T61-1 (tasks/0152): `sync to device` was 2.4-3.9% of single-lane
  // wall and every microsecond of it was serialised against the array.
  void sync_slot_to_device(size_t arg_index, size_t slot, size_t bytes = 0);
  void sync_slot_from_device(size_t arg_index, size_t slot, size_t bytes = 0);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Device *dev_ = nullptr;      // kept so stage() can allocate more buffers
  DesignInfo info_;
  size_t output_index_ = 0;
};

}  // namespace npu
}  // namespace npue

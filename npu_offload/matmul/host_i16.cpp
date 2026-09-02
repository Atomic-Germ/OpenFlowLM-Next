// host_i16.cpp -*- C++ -*-
// Single-tile i16->i32 matmul host for the iron "matmul_i16" design.
// Integer-exact comparison against CPU reference.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int DIM_M = 64;
constexpr int DIM_K = 64;
constexpr int DIM_N = 64;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("matmul-i16-host");
  test_utils::add_default_options(options);
  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  xrt::device device;
  xrt::kernel kernel;
  test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                   vm["xclbin"].as<std::string>(),
                                   vm["kernel"].as<std::string>());

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(device, DIM_M * DIM_K * sizeof(int16_t),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b = xrt::bo(device, DIM_K * DIM_N * sizeof(int16_t),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_c = xrt::bo(device, DIM_M * DIM_N * sizeof(int32_t),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  memcpy(bo_instr.map<void *>(), instr_v.data(),
         instr_v.size() * sizeof(uint32_t));

  int16_t *a = bo_a.map<int16_t *>();
  int16_t *b = bo_b.map<int16_t *>();
  int32_t *c = bo_c.map<int32_t *>();

  for (int i = 0; i < DIM_M * DIM_K; i++)
    a[i] = (int16_t)(((i * 13) % 11) - 5);
  for (int i = 0; i < DIM_K * DIM_N; i++)
    b[i] = (int16_t)(((i * 17) % 9) - 4);
  memset(c, 0, DIM_M * DIM_N * sizeof(int32_t));

  std::vector<int64_t> ref(DIM_M * DIM_N, 0);
  for (int m = 0; m < DIM_M; m++)
    for (int n = 0; n < DIM_N; n++) {
      int64_t acc = 0;
      for (int k = 0; k < DIM_K; k++)
        acc += (int64_t)a[m * DIM_K + k] * (int64_t)b[k * DIM_N + n];
      ref[m * DIM_N + n] = acc;
    }

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_c);
  run.wait();
  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int errors = 0;
  int64_t max_err = 0;
  for (int i = 0; i < DIM_M * DIM_N; i++) {
    int64_t got = c[i];
    int64_t diff = std::abs(got - ref[i]);
    max_err = std::max(max_err, diff);
    if (diff != 0) errors++;
  }

  std::cout << "max_abs_err=" << max_err << "\n";
  if (!errors) {
    std::cout << "PASS!" << std::endl;
    return 0;
  }
  std::cout << errors << " mismatches." << std::endl;
  return 1;
}
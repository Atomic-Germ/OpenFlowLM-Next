// host_single_core.cpp -*- C++ -*-
// Single-core vectorized int16 matmul host (256x256x256) for the iron design
// matmul_single_core.py (canonical mm.cc port). Deterministic small int16
// inputs; exact int64 CPU reference; compares NPU output bit-for-bit.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int DIM_M = 256;
constexpr int DIM_K = 256;
constexpr int DIM_N = 256;

int main(int argc, const char *argv[]) {
  cxxopts::Options options("matmul-single-core-host");
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
  auto bo_c = xrt::bo(device, DIM_M * DIM_N * sizeof(int16_t),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  memcpy(bo_instr.map<void *>(), instr_v.data(),
         instr_v.size() * sizeof(uint32_t));

  int16_t *a = bo_a.map<int16_t *>();
  int16_t *b = bo_b.map<int16_t *>();
  int16_t *c = bo_c.map<int16_t *>();

  for (int i = 0; i < DIM_M * DIM_K; i++) a[i] = (int16_t)(i % 7);
  for (int i = 0; i < DIM_K * DIM_N; i++) b[i] = (int16_t)(i % 5);
  memset(c, 0, DIM_M * DIM_N * sizeof(int16_t));

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

  if (verbosity >= 1) std::cout << "Running Kernel.\n";
  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_c);
  run.wait();
  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int errors = 0;
  long max_abs_err = 0;
  for (int i = 0; i < DIM_M * DIM_N; i++) {
    long e = std::llabs(ref[i] - (long)c[i]);
    if (e > max_abs_err) max_abs_err = e;
    if (e != 0) errors++;
  }

  std::cout << "max_abs_err=" << max_abs_err << "\n";
  std::cout << "C[0:8] =";
  for (int j = 0; j < 8; j++) std::cout << " " << c[j];
  std::cout << "\n";

  std::ofstream fc("c_single.bin", std::ios::binary);
  fc.write((char *)c, DIM_M * DIM_N * sizeof(int16_t));
  std::ofstream fr("ref_single.bin", std::ios::binary);
  fr.write((char *)&ref[0], DIM_M * DIM_N * sizeof(int64_t));

  if (!errors) {
    std::cout << "PASS!" << std::endl;
    return 0;
  }
  std::cout << errors << " mismatches." << std::endl;
  return 1;
}
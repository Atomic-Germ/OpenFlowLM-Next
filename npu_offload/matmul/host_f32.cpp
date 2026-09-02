// host_f32.cpp -*- C++ -*-
// Variable-shape bf16 matmul host with f32 output for the tiled iron design.
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

int DIM_M = 64;
int DIM_K = 64;
int DIM_N = 64;

int main(int argc, const char *argv[]) {
  if (const char *dim = std::getenv("MM_DIM"))
    DIM_M = DIM_K = DIM_N = std::atoi(dim);
  if (const char *dim = std::getenv("MM_M")) DIM_M = std::atoi(dim);
  if (const char *dim = std::getenv("MM_K")) DIM_K = std::atoi(dim);
  if (const char *dim = std::getenv("MM_N")) DIM_N = std::atoi(dim);

  cxxopts::Options options("matmul-f32out-host");
  test_utils::add_default_options(options);
  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  xrt::device device;
  xrt::kernel kernel;
  test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                   vm["xclbin"].as<std::string>(),
                                   vm["kernel"].as<std::string>());

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(device, DIM_M * DIM_K * sizeof(uint16_t),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b = xrt::bo(device, DIM_K * DIM_N * sizeof(uint16_t),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_c = xrt::bo(device, DIM_M * DIM_N * sizeof(float),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  memcpy(bo_instr.map<void *>(), instr_v.data(),
         instr_v.size() * sizeof(uint32_t));

  uint16_t *a = bo_a.map<uint16_t *>();
  uint16_t *b = bo_b.map<uint16_t *>();
  float *c = bo_c.map<float *>();

  for (int i = 0; i < DIM_M * DIM_K; i++) a[i] = (uint16_t)(i % 7);  // placeholder, should be bf16
  for (int i = 0; i < DIM_K * DIM_N; i++) b[i] = (uint16_t)(i % 5);
  memset(c, 0, DIM_M * DIM_N * sizeof(float));

  // FP32 reference: A_bf16 @ B_bf16
  auto f32_to_bf16 = [](float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t lsb = (x >> 16) & 1u;
    x += 0x7fffu + lsb; return (uint16_t)(x >> 16);
  };
  auto bf16_to_f32 = [](uint16_t h) {
    float f; uint32_t x = (uint32_t)h << 16; std::memcpy(&f, &x, 4); return f;
  };

  std::vector<float> ref(DIM_M * DIM_N, 0.0f);
  for (int i = 0; i < DIM_M * DIM_K; i++) a[i] = f32_to_bf16(float(((i * 13) % 11) - 5) * 0.05f);
  for (int i = 0; i < DIM_K * DIM_N; i++) b[i] = f32_to_bf16(float(((i * 17) % 9) - 4) * 0.05f);

  for (int m = 0; m < DIM_M; m++)
    for (int n = 0; n < DIM_N; n++) {
      double acc = 0.0;
      for (int k = 0; k < DIM_K; k++)
        acc += (double)bf16_to_f32(a[m * DIM_K + k]) *
               (double)bf16_to_f32(b[k * DIM_N + n]);
      ref[m * DIM_N + n] = (float)acc;
    }

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1) std::cout << "Running Kernel.\n";
  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_c);
  run.wait();
  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int errors = 0;
  double max_abs_err = 0.0, max_rel_err = 0.0;
  for (int i = 0; i < DIM_M * DIM_N; i++) {
    float got = c[i];
    float ref_val = ref[i];
    double abs_err = std::fabs((double)got - (double)ref_val);
    double rel_err = std::fabs((double)got - (double)ref_val) /
                     std::max(std::fabs((double)ref_val), 1e-9);
    max_abs_err = std::max(max_abs_err, abs_err);
    max_rel_err = std::max(max_rel_err, rel_err);
    if (abs_err > 0.0001 + 0.001 * std::fabs((double)ref_val)) errors++;
  }

  std::cout << "max_abs_err=" << max_abs_err
            << " max_rel_err=" << max_rel_err << "\n";
  std::cout << "C[0:8] =";
  for (int j = 0; j < 8; j++) std::cout << " " << c[j];
  std::cout << "\n";

  if (!errors) {
    std::cout << "PASS!" << std::endl;
    return 0;
  }
  std::cout << errors << " mismatches." << std::endl;
  return 1;
}
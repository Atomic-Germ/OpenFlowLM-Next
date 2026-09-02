TEST_LIB=/home/atomic-germ/Projects/FastFlowLM_v1.0.1-add/ironvenv/lib/python3.13/site-packages/mlir_aie/runtime_lib/x86_64/test_lib
XRT=/opt/xilinx/xrt
g++ -O2 -std=c++17 host.cpp \
    -I${TEST_LIB}/include \
    -I${XRT}/include \
    ${TEST_LIB}/lib/libtest_utils.a \
    -L${XRT}/lib -lxrt_coreutil \
    -Wl,-rpath,${XRT}/lib \
    -o matmul.exe
echo "build exit=$?"

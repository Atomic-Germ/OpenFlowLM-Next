cmake_minimum_required(VERSION 3.22)
project(fastflowlm LANGUAGES CXX VERSION 0.1.0)

# Enable testing at the top level so `ctest` (run from the build root) discovers
# the tests registered under src/. The src/ subproject also calls it so a direct
# `cmake -S src` build remains testable.
enable_testing()

# The inference engine (its own project, "flm") lives under src/ and is built as
# a subdirectory so a single `cmake --preset` from the repo root configures,
# builds, tests and packages the whole distribution (engine + open NPU kernels).
#
#   cmake --preset linux-default        # configure (release + kernel export)
#   cmake --build --preset linux-default # build engine + xclbins
#   ctest --preset linux-default         # run the test suite
#   cmake --install build                # install into CMAKE_INSTALL_PREFIX
#   cpack --preset linux-package-rpm     # (or -deb / -tgz) package it
add_subdirectory(src)

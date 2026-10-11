### 💡 Trick: Building XRT / Firmware Without the 98GB Vitis Download

If you need to build or modify the host XRT runtime and require the Embedded Resource Trainer (ERT) firmware, you do not need to download the massive AMD Vitis installer. XRT only uses Vitis to access the MicroBlaze GCC compiler. 

You can bypass the AMD bloatware entirely by faking the Vitis environment:

1. **Install the standalone MicroBlaze compiler and required packages:**
   ```bash
   # On Ubuntu / Debian:
   sudo apt-get install gcc-microblaze-linux-gnu uuid-dev ocl-icd-opencl-dev

   # On RHEL / CentOS / Rocky Linux:
   sudo dnf install -y \
     gcc-microblaze-linux-gnu \
     binutils-microblaze-linux-gnu \
     gcc-c++-microblaze-linux-gnu \
     libuuid-devel \
     ocl-icd-devel \
     pybind11-devel \
     pybind11-json-devel \
     libstdc++-static \
     glibc-static \
     rpmbuild \
     curl-devel \
     fftw3-devel \
     libavcodec-free-devel \
     libavformat-free-devel \
     libavutil-free-devel \
     libswscale-free-devel \
     libswresample-free-devel \
     readline-devel \
     rust \
     cargo \
     git \
     cmake \
     ninja-build 
   ```
   *(Note: Ensure you install `ocl-icd-devel` rather than alternative OpenCL development headers, or the build may fail on OpenCL runtime hooks).*

2. **Create a fake Vitis directory structure:**
   ```bash
   mkdir -p ~/fake_vitis/gnu/microblaze/lin/bin
   ```

3. **Symlink the lightweight compiler into your dummy directory:**
   ```bash
   ln -s /usr/bin/microblaze-linux-gnu-gcc ~/fake_vitis/gnu/microblaze/lin/bin/microblaze-xilinx-elf-gcc
   ln -s /usr/bin/microblaze-linux-gnu-g++ ~/fake_vitis/gnu/microblaze/lin/bin/microblaze-xilinx-elf-g++
   ```

4. **Point your environment to the fake path before building:**
   ```bash
   export XILINX_VITIS=~/fake_vitis
   cd xdna-driver/xrt/build
   build.sh -npu
   ```

CMake will now find the required compiler binaries, allowing you to compile the ERT firmware smoothly without ever touching the official AMD installer.


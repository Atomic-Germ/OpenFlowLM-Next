# syntax=docker/dockerfile:1.7

FROM ubuntu:26.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    doxygen \
    curl \
    wget \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    cargo \
    rustc \
    clang \
    lld \
    patchelf \
    rpm \
    dpkg-dev \
    fakeroot \
    uuid-dev \
    libboost-all-dev \
    libcurl4-openssl-dev \
    libfftw3-dev \
    libreadline-dev \
    libncurses-dev \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libssl-dev \
    zlib1g-dev \
    libdrm-dev \
    libudev-dev \
    libxrt2 \
    libxrt-npu2 \
    libxrt-dev \
    libxrt-utils \
    libxrt-utils-npu \
    python3-xrt \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ironvenv-requirements.txt /code/ironvenv-requirements.txt

RUN python3 -m venv --system-site-packages /code/ironvenv \
 && /code/ironvenv/bin/python -m pip install --upgrade pip setuptools wheel \
 && /code/ironvenv/bin/python -m pip install -r /code/ironvenv-requirements.txt

# OpenFlowLM hard-codes /opt/xilinx/xrt in export-kernels.py, while Ubuntu
# packages XRT under /usr. Provide the layout expected by the project.
RUN mkdir -p /opt/xilinx/xrt/lib \
             /opt/xilinx/xrt/lib64 \
    && ln -sfn /usr/bin /opt/xilinx/xrt/bin \
    && ln -sfn /usr/include /opt/xilinx/xrt/include \
    && ln -sfn /usr/lib/x86_64-linux-gnu \
      /opt/xilinx/xrt/lib/x86_64-linux-gnu \
    && ln -sfn /usr/lib/x86_64-linux-gnu \
      /opt/xilinx/xrt/lib64/x86_64-linux-gnu \
    && ln -sfn /usr/lib/python3/dist-packages \
      /opt/xilinx/xrt/python

ENV VIRTUAL_ENV=/code/ironvenv
ENV XILINX_XRT=/opt/xilinx/xrt
ENV PATH="/code/ironvenv/bin:/opt/xilinx/xrt/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV LD_LIBRARY_PATH="/opt/xilinx/xrt/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"
ENV PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/lib/pkgconfig:/usr/share/pkgconfig"
ENV PYTHONPATH="/opt/xilinx/xrt/python"
ENV NPU_CACHE_HOME=/root/.npu/cache

# Sanity checks. These are deliberately individual so Docker shows the exact
# missing component if Ubuntu changes a package layout later.
RUN set -eux; \
    command -v cmake; \
    command -v ninja; \
    command -v cargo; \
    command -v clang; \
    command -v xclbinutil; \
    command -v aiebu-asm; \
    test -f /usr/include/xrt/xrt_bo.h; \
    pkg-config --modversion xrt; \
    /code/ironvenv/bin/python -c "import aie; import aie.iron; print('mlir-aie OK')"; \
    /code/ironvenv/bin/python -c "import pyxrt; print('pyxrt OK')"; \
    test -n "$(find /code/ironvenv/lib -type d -path '*/site-packages/llvm-aie/bin' -print -quit)"; \
    test -n "$(find /code/ironvenv/lib -type f -path '*/site-packages/llvm-aie/bin/clang' -print -quit)"

# Full kernel export includes open_npue/BERT sets which require a real NPU.
# Run the workflow only after starting the container with /dev/accel/accel0.
#
# Example:
#   docker run --rm -it \
#     --device=/dev/accel/accel0 \
#     -v "$PWD/oflm-build:/code/build" \
#     openflowlm-build:ubuntu26 \
#     cmake --workflow --preset linux-package
#
# DEB packaging:
#   cmake --preset linux-default
#   cmake --build --preset linux-default
#   ctest --preset linux-default
#   cpack --preset linux-package-deb

WORKDIR /code
CMD ["bash"]

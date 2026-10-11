#!/usr/bin/env bash
set -e

echo "🚀 Installing discovered Fedora build dependencies..."
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
  cmake

# Ensure uv is installed globally if not already present
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh | sh
    source $HOME/.local/bin/env
fi

echo "🧙 Creating Fake Vitis toolchain..."
FAKE_VITIS_DIR="$HOME/vitis"
mkdir -p "${FAKE_VITIS_DIR}/gnu/microblaze/lin/bin"

# Map the cross-compilers
ln -sf /usr/bin/microblaze-linux-gnu-gcc "${FAKE_VITIS_DIR}/gnu/microblaze/lin/bin/microblaze-xilinx-elf-gcc"
ln -sf /usr/bin/microblaze-linux-gnu-g++ "${FAKE_VITIS_DIR}/gnu/microblaze/lin/bin/microblaze-xilinx-elf-g++"

# Map toolchain dependencies XRT looks for in the same bucket
ln -sf $(command -v rustc) "${FAKE_VITIS_DIR}/gnu/microblaze/lin/bin/rust"
ln -sf $(command -v uv) "${FAKE_VITIS_DIR}/gnu/microblaze/lin/bin/uv"

# Export variable for the current active build script session
export XILINX_VITIS="${FAKE_VITIS_DIR}"
echo "Variables set: XILINX_VITIS=$XILINX_VITIS"

echo "🎯 Preparing OpenFlowLM-Next project workspace..."
# Assuming you are already in the recursively cloned OpenFlowLM-Next directory
if [ -f "CMakeLists.txt" ]; then
    # Bootstrapping the Python runtime properly
    uv python install 3.11
    
#    echo "💡 Creating and initializing ironvenv..."
#    uv venv ironvenv
#    
#    # Activate it so the upcoming build tools bind to it
#    source ironvenv/bin/activate
    
    echo "✅ Workspace is staged and ready for the MLIR-AIE build step!"
else
    echo "⚠️ Run this script right outside or inside your repository to complete the workflow setup."
fi

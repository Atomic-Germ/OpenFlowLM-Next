#!/usr/bin/env bash
# Build the llama.cpp reference dumper against the installed libllama.
#
# The library lives in /usr/local/lib64 on this box (Guanaco's install), and
# the headers in /usr/local/include. Override with LLAMA_INC / LLAMA_LIB if the
# install is somewhere else. Nothing here touches the tree's build.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
inc="${LLAMA_INC:-/usr/local/include}"
lib="${LLAMA_LIB:-/usr/local/lib64}"

g++ -O2 -std=c++17 -o "$here/dump_ref" "$here/dump_ref.cpp" \
    -I"$inc" -L"$lib" -Wl,-rpath,"$lib" \
    -lllama -lggml -lggml-base -lggml-cpu
echo "built $here/dump_ref"

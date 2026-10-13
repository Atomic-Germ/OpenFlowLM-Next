/// \file pack_command.hpp
/// \brief Process handoff from `oflm pack` to the bundled Q4NX builder.
#pragma once

/// \brief Run the bundled model packer with the given arguments.
/// \param argc Number of arguments after `pack`.
/// \param argv The arguments, unchanged.
/// \return The packer's exit status.
/// \note Deliberately a HANDOFF, not a port. The builder is a quantizer: it
///       packs GGUF and HF-safetensors into a Q4NX container whose bytes must
///       match what the NPU kernels read, and that job is done with torch,
///       numpy, gguf and safetensors. Reimplementing the bit layout in C++ is a
///       multi-week job whose failure mode is silently wrong weights rather than
///       a crash. The launcher beside oflm already sets up the environment and
///       remains the single source of truth for that, so this finds it and runs
///       it -- which also means the GGUF improvements land without touching oflm.
namespace pack_command {
int run(int argc, char* argv[]);
}

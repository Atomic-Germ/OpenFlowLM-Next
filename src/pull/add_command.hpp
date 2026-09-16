/// \file add_command.hpp
/// \brief Entry point for the bundled model installer.
#pragma once

namespace add_command {

/// Replace the current process with the bundled oflm-add implementation.
/// argc/argv contain only the arguments following `oflm add`.
int run(int argc, char* argv[]);

} // namespace add_command

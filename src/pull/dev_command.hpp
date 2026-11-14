/// \file dev_command.hpp
/// \brief Developer/toolchain workflows for `oflm dev`.
#pragma once

namespace dev_command {

/// Run developer tooling. argc/argv contain only arguments following `oflm dev`.
int run(int argc, char* argv[]);

} // namespace dev_command

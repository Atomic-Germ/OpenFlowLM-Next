#pragma once
/// \file config_paths.hpp
/// \brief The paths oflm reads and writes, and where they point right now.
/// \note A path that is not where the user thinks it is explains a lot of
///       "the model is not there" questions: the registry that is read, the
///       kernels that are loaded, the models that are found and the file the
///       settings are written to. Every one of them is an environment
///       variable, and none of them is visible from the outside.
#include <ostream>
#include <string>
#include <vector>

namespace oflm::config {

/// One path oflm reads or writes.
struct PathEntry {
    std::string var;      ///< the environment variable that sets it
    std::string what;     ///< what it is for
    std::string value;    ///< where it points right now
    std::string note;     ///< an extra remark (where the models really are)
    bool from_env = false;///< true when the environment sets it
    bool found = true;    ///< false when the path could not be resolved
};

/// The paths in use, resolved the way the commands resolve them.
/// \note The values are the ones in effect, not the ones in the environment:
///       a path that is not set is resolved to the default oflm would use.
std::vector<PathEntry> resolve_paths();

/// Print the paths, one per line, with the value each one has now.
void print_paths(std::ostream& os);

} // namespace oflm::config

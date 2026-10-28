#pragma once

#include <filesystem>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace pull {

// Replace an existing destination without removing it first. Both paths must
// be on the same filesystem (records and downloads stage beside their target).
inline std::error_code replace_file_atomically(const std::filesystem::path& staged,
                                                const std::filesystem::path& destination) {
#ifdef _WIN32
    if (MoveFileExW(staged.c_str(), destination.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) return {};
    return {static_cast<int>(GetLastError()), std::system_category()};
#else
    std::error_code error;
    std::filesystem::rename(staged, destination, error);
    return error;
#endif
}

}  // namespace pull

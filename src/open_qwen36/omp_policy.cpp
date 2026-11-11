/// \file omp_policy.cpp
/// \brief OpenMP startup policy without an XRT or model-engine dependency.
#include "open_qwen36/omp_policy.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __linux__
#include <unistd.h>
#endif

#ifdef OFLM_OMP_POLICY_FAIL_EXEC
static int failed_exec(const char*, char* const*) { errno = EACCES; return -1; }
#define execv failed_exec
#endif

namespace open_qwen36 {

bool qwen38_model(const char* family, const char* name) {
    if (!family || !name) return false;
    if (std::strcmp(family, "qwen3.5") != 0) return false;
    const char* prefix = "Qwen3.8-";
    return std::strncmp(name, prefix, std::strlen(prefix)) == 0;
}

#ifdef _WIN32
bool reexec_with_passive(char**) {
    return false;  // The engine's delay-loaded vcomp constructor handles Windows.
}
#elif defined(__linux__)
bool reexec_with_passive(char** argv) {
    if (std::getenv("OMP_WAIT_POLICY") || std::getenv("OFLM_OPEN_OMP_REEXEC")) return false;
    const char* off = std::getenv("OFLM_OPEN_OMP_PASSIVE");
    if (off && std::strcmp(off, "0") == 0) return false;
    if (!argv || !argv[0]) return false;
    if (setenv("OMP_WAIT_POLICY", "PASSIVE", 1) != 0) {
        std::perror("open_qwen36: cannot set OpenMP policy for re-exec");
        return false;
    }
    if (setenv("OFLM_OPEN_OMP_REEXEC", "1", 1) == 0) {
        execv("/proc/self/exe", argv);
        std::perror("open_qwen36: OpenMP re-exec failed");
    } else {
        std::perror("open_qwen36: cannot set OpenMP re-exec guard");
    }
    unsetenv("OMP_WAIT_POLICY");
    unsetenv("OFLM_OPEN_OMP_REEXEC");
    return false;
}
#else
bool reexec_with_passive(char**) {
    return false;
}
#endif

}  // namespace open_qwen36

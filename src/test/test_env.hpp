/// \file test_env.hpp
/// \brief setenv/unsetenv for the host-only tests on MSVC, which has neither
#pragma once
#include <cstdlib>

#ifdef _WIN32
// _putenv_s with an empty value removes the variable.
static inline int setenv(const char* name, const char* value, int /*overwrite*/) { return _putenv_s(name, value); }
static inline int unsetenv(const char* name) { return _putenv_s(name, ""); }
#endif

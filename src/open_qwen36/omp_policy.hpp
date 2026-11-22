/// \file omp_policy.hpp
/// \brief Early OpenMP wait-policy selection for the Qwen3.8 prefill workload.
#pragma once

namespace open_qwen36 {

/// Re-exec once with PASSIVE before libgomp initializes. An explicit policy and
/// OFLM_OPEN_OMP_PASSIVE=0 take precedence. A successful exec never returns.
bool reexec_with_passive(char** argv);

/// Whether the selected registry entry is the Qwen3.8 model measured for this policy.
bool qwen38_model(const char* family, const char* name);

}  // namespace open_qwen36

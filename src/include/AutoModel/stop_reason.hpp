/// \file stop_reason.hpp
/// \brief Why generation stopped, and its ENGINE-side spelling.
///
/// Split out of automodel.hpp so that code which only needs to name a stop
/// reason does not have to include every NPU model header to do it. The OpenAI
/// wire vocabulary is a DIFFERENT set and lives in server/openai_compat.hpp --
/// "cancel", "error" and "UNKNOWN" below are ours and must never reach a
/// `finish_reason` field.
#pragma once

#include <string>

typedef enum {
    EOT_DETECTED,
    MAX_LENGTH_REACHED,
    ERROR_DETECTED,
    CANCEL_DETECTED,
    TOOL_DETECTED
} stop_reason_t;

/// The engine-side name, used in logs and in oflm's own (non-OpenAI) responses.
inline std::string stop_reason_to_string(stop_reason_t reason){
    switch (reason){
        case EOT_DETECTED:
            return "stop";
        case MAX_LENGTH_REACHED:
            return "length";
        case CANCEL_DETECTED:
            return "cancel";
        case ERROR_DETECTED:
            return "error";
        case TOOL_DETECTED:
            return "tool_calls";
        default:
            return "UNKNOWN";
    }
}

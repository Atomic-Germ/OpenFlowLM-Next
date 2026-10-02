/// \file request_error.hpp
/// \brief The exception that means "what the client sent cannot be served".
///
/// Thrown where that is known -- a chat template refusing the conversation, say.
/// The server answers it with a 400 and any other exception with a 500
/// (openai_compat::exception_body, #135); what() is logged, never sent. Split out
/// like stop_reason.hpp, so the server can name it without every NPU model header.
#pragma once

#include <stdexcept>

struct request_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

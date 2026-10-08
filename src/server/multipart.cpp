/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file multipart.cpp
 * \brief MultiPart/form-data Parser
 * \author OpenFlowLM Team
 * \date 2025-10-16
 *  \version 0.9.24
 */

#include "multipart.hpp"

#include <cctype>

namespace {

std::string lower(std::string_view s) {
    std::string out(s);
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

std::string_view trim(std::string_view s) {
    size_t a = s.find_first_not_of(" \t");
    if (a == std::string_view::npos) return {};
    return s.substr(a, s.find_last_not_of(" \t") - a + 1);
}

// A quoted value is read whole, so `filename="x; name=y"` can't name the part.
std::string header_param(std::string_view v, std::string_view key) {
    size_t i = v.find(';');                     // past the value itself (form-data, ...)
    while (i != std::string_view::npos) {
        ++i;
        size_t eq = v.find_first_of("=;", i);
        if (eq == std::string_view::npos) return {};
        if (v[eq] == ';') { i = eq; continue; }   // a parameter with no value
        const std::string k = lower(trim(v.substr(i, eq - i)));
        size_t j = v.find_first_not_of(" \t", eq + 1);
        std::string val;
        if (j != std::string_view::npos && v[j] == '"') {
            for (++j; j < v.size() && v[j] != '"'; ++j) {
                // only \" and \\ escape: a lone backslash stays, as in a Windows path
                if (v[j] == '\\' && j + 1 < v.size() && (v[j + 1] == '"' || v[j + 1] == '\\')) ++j;
                val += v[j];
            }
            i = v.find(';', j);
        } else {
            i = j == std::string_view::npos ? j : v.find(';', j);
            if (j != std::string_view::npos) val = std::string(trim(v.substr(j, i == std::string_view::npos ? i : i - j)));
        }
        if (k == key) return val;
    }
    return {};
}

// Only the line naming the header, so another header's text can't answer for it.
std::string_view header_value(std::string_view headers, std::string_view name) {
    for (size_t pos = 0;;) {
        size_t eol = headers.find("\r\n", pos);
        std::string_view line = headers.substr(pos, eol == std::string_view::npos ? eol : eol - pos);
        size_t colon = line.find(':');
        if (colon != std::string_view::npos && lower(line.substr(0, colon)) == name) return line.substr(colon + 1);
        if (eol == std::string_view::npos) return {};
        pos = eol + 2;
    }
}

}  // namespace

///@brief multipart/form-data request parser
///@return parts of multipart/form-data, by name, in the order they came
std::multimap<std::string, MultipartPart> parse_multipart(const http::request<http::string_body>& req) {
    std::multimap<std::string, MultipartPart> parts;

    // 1. Extract the boundary from the Content-Type header. RFC 2046 allows it quoted
    //    (boundary="..."), and other parameters may follow it.
    std::string content_type_header = std::string(req[http::field::content_type]);
    std::string boundary = header_param(content_type_header, "boundary");
    if (boundary.empty()) {
        throw std::runtime_error("Invalid multipart/form-data: boundary not found.");
    }
    boundary = "--" + boundary;

    std::string_view body = req.body();
    size_t start_pos = 0;

    // 2. Use the boundary to split the request body
    while ((start_pos = body.find(boundary, start_pos)) != std::string_view::npos) {
        start_pos += boundary.length();
        if (body.substr(start_pos, 2) == "--") {
            break; // boundary ended
        }
        start_pos += 2; // skip \r\n

        size_t end_pos = body.find(boundary, start_pos);
        if (end_pos == std::string_view::npos) {
            break;
        }

        std::string_view part_data = body.substr(start_pos, end_pos - start_pos - 2); // minus the \r\n

        // 3. Parse each part
        size_t headers_end_pos = part_data.find("\r\n\r\n");
        if (headers_end_pos == std::string_view::npos) {
            continue;
        }

        std::string_view headers_sv = part_data.substr(0, headers_end_pos);
        MultipartPart part;
        part.content = std::string(part_data.substr(headers_end_pos + 4));

        // Header names are case-insensitive (RFC 7578 section 4.8)
        if (std::string_view cd = header_value(headers_sv, "content-disposition"); !cd.empty()) {
            part.name = header_param(cd, "name");
            part.filename = header_param(cd, "filename");
        }
        part.content_type = std::string(trim(header_value(headers_sv, "content-type")));

        if (!part.name.empty()) {
            std::string name = part.name;
            parts.emplace(std::move(name), std::move(part));
        }
    }

    return parts;
}

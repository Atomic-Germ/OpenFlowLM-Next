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

// A header parameter's value: name="value" or name=value (up to ';'). `lc` is the header
// text lowercased, for finding the key; the value is read from `orig`, as sent.
std::string header_param(std::string_view orig, const std::string& lc, const std::string& key, size_t from) {
    size_t pos = from;
    while ((pos = lc.find(key + "=", pos)) != std::string::npos) {
        // a whole parameter name, not the tail of another ("filename=" holds "name=")
        if (pos == 0 || lc[pos - 1] == ' ' || lc[pos - 1] == ';' || lc[pos - 1] == '\t') break;
        pos += key.size();
    }
    if (pos == std::string::npos) return {};
    pos += key.size() + 1;
    if (pos < orig.size() && orig[pos] == '"') {
        size_t end = orig.find('"', pos + 1);
        return std::string(orig.substr(pos + 1, end == std::string_view::npos ? std::string_view::npos : end - pos - 1));
    }
    size_t end = orig.find_first_of(";\r\n", pos);
    std::string v(orig.substr(pos, end == std::string_view::npos ? std::string_view::npos : end - pos));
    while (!v.empty() && (v.back() == ' ' || v.back() == '\t')) v.pop_back();
    return v;
}

}  // namespace

///@brief multipart/form-data request parser
///@return parts of multipart/form-data, by name, in the order they came
std::multimap<std::string, MultipartPart> parse_multipart(const http::request<http::string_body>& req) {
    std::multimap<std::string, MultipartPart> parts;

    // 1. Extract the boundary from the Content-Type header. RFC 2046 allows it quoted
    //    (boundary="..."), and other parameters may follow it.
    std::string content_type_header = std::string(req[http::field::content_type]);
    std::string boundary = header_param(content_type_header, lower(content_type_header), "boundary", 0);
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
        const std::string headers_lc = lower(headers_sv);
        MultipartPart part;
        part.content = std::string(part_data.substr(headers_end_pos + 4));

        // Header names are case-insensitive (RFC 7578 section 4.8)
        size_t cd_pos = headers_lc.find("content-disposition:");
        if (cd_pos != std::string::npos) {
            part.name = header_param(headers_sv, headers_lc, "name", cd_pos);
            part.filename = header_param(headers_sv, headers_lc, "filename", cd_pos);
        }
        size_t ct_pos = headers_lc.find("content-type:");
        if (ct_pos != std::string::npos) {
            size_t v = ct_pos + 13;
            size_t end = headers_sv.find("\r\n", v);
            std::string ct(headers_sv.substr(v, end == std::string_view::npos ? std::string_view::npos : end - v));
            size_t a = ct.find_first_not_of(" \t");
            part.content_type = a == std::string::npos ? std::string() : ct.substr(a);
        }

        if (!part.name.empty()) {
            std::string name = part.name;
            parts.emplace(std::move(name), std::move(part));
        }
    }

    return parts;
}

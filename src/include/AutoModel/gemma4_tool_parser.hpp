/// \file gemma4_tool_parser.hpp
/// \brief Gemma 4 tool-call text -> (name, JSON args), shared by the 12B and E2B/E4B models.
///
/// The model emits `<|tool_call>call:NAME{ARGS}<tool_call|>` where ARGS is relaxed JSON:
/// bare keys, strings wrapped in <|"|>...<|"|>, and enough variation on that to need a
/// real parser. One copy here; the two model files used to carry identical duplicates.

#pragma once
#include <nlohmann/json.hpp>
#include <cctype>
#include <cstdio>
#include <iostream>
#include <string>
#include <utility>

namespace gemma4_tools {
using json = nlohmann::ordered_json;

inline std::string trim_value(std::string value) {
    size_t start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }

    size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

// The model emits relaxed JSON for tool arguments:
//   - keys are bare identifiers (e.g. `name:`) — sometimes also "..." quoted
//   - string values are delimited by <|"|>...<|"|>, but the model frequently
//     omits the opener and/or replaces the closer with a plain `"`
//   - values can also be objects {..}, arrays [..], booleans, null, or numbers
//   - inside a string value, raw `"`, raw newlines, and even the literal
//     substring `<|"|>` can appear as content
//
// We rewrite the input into well-formed JSON via recursive-descent.
struct ArgsParser {
    const std::string& s;
    size_t i = 0;
    static constexpr size_t marker_len = 5;
    static constexpr const char* quote_marker_lit = "<|\"|>";

    explicit ArgsParser(const std::string& src) : s(src) {}

    void skip_ws() {
        while (i < s.size() &&
               std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }
    void skip_ws_at(size_t& pos) const {
        while (pos < s.size() &&
               std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
    }
    bool match_marker(size_t pos) const {
        return pos + marker_len <= s.size() &&
               s.compare(pos, marker_len, quote_marker_lit) == 0;
    }

    static void json_escape_char(std::string& out, char c) {
        switch (c) {
            case '"':  out.append("\\\""); break;
            case '\\': out.append("\\\\"); break;
            case '\n': out.append("\\n");  break;
            case '\r': out.append("\\r");  break;
            case '\t': out.append("\\t");  break;
            case '\b': out.append("\\b");  break;
            case '\f': out.append("\\f");  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x",
                                  static_cast<unsigned char>(c));
                    out.append(buf);
                } else {
                    out.push_back(c);
                }
                break;
        }
    }

    bool match_word(const char* w, size_t len) const {
        if (i + len > s.size()) return false;
        if (s.compare(i, len, w) != 0) return false;
        if (i + len == s.size()) return true;
        unsigned char nc = static_cast<unsigned char>(s[i + len]);
        return !(std::isalnum(nc) || nc == '_');
    }

    // Parse one JSON value. `terminators` is the set of chars that end a
    // bare (undelimited) string at depth 0 (e.g. ",}" inside an object,
    // ",]" inside an array, "" at the very top).
    std::string parse_value(const std::string& terminators) {
        skip_ws();
        if (i >= s.size()) return "null";
        char c = s[i];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (match_word("true",  4)) { i += 4; return "true";  }
        if (match_word("false", 5)) { i += 5; return "false"; }
        if (match_word("null",  4)) { i += 4; return "null";  }
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        return parse_string(terminators);
    }

    std::string parse_number() {
        size_t start = i;
        if (s[i] == '-') ++i;
        while (i < s.size()) {
            char c = s[i];
            if ((c >= '0' && c <= '9') || c == '.' ||
                c == 'e' || c == 'E' || c == '+' || c == '-') ++i;
            else break;
        }
        return s.substr(start, i - start);
    }

    std::string parse_key() {
        skip_ws();
        std::string key;
        if (i >= s.size()) return key;
        if (match_marker(i)) {
            i += marker_len;
            while (i < s.size() && !match_marker(i)) key.push_back(s[i++]);
            if (match_marker(i)) i += marker_len;
        } else if (s[i] == '"') {
            ++i;
            while (i < s.size() && s[i] != '"') {
                if (s[i] == '\\' && i + 1 < s.size()) {
                    key.push_back(s[i]);
                    key.push_back(s[i + 1]);
                    i += 2;
                } else {
                    key.push_back(s[i++]);
                }
            }
            if (i < s.size() && s[i] == '"') ++i;
        } else {
            while (i < s.size() &&
                   (std::isalnum(static_cast<unsigned char>(s[i])) ||
                    s[i] == '_')) {
                key.push_back(s[i++]);
            }
        }
        return key;
    }

    std::string parse_object() {
        // assumes s[i] == '{'
        ++i;
        std::string out = "{";
        bool first = true;
        while (i < s.size()) {
            skip_ws();
            if (i >= s.size()) break;
            if (s[i] == '}') { ++i; break; }

            std::string key = parse_key();
            if (key.empty()) {
                // can't make progress; bail
                if (i < s.size() && s[i] == '}') { ++i; }
                break;
            }
            skip_ws();
            if (i < s.size() && s[i] == ':') ++i;

            std::string val = parse_value(",}");

            if (!first) out.push_back(',');
            first = false;
            out.push_back('"');
            for (char kc : key) json_escape_char(out, kc);
            out.append("\":");
            out.append(val);

            skip_ws();
            if (i < s.size() && s[i] == ',') ++i;
        }
        out.push_back('}');
        return out;
    }

    std::string parse_array() {
        // assumes s[i] == '['
        ++i;
        std::string out = "[";
        bool first = true;
        while (i < s.size()) {
            skip_ws();
            if (i >= s.size()) break;
            if (s[i] == ']') { ++i; break; }

            std::string val = parse_value(",]");
            if (!first) out.push_back(',');
            first = false;
            out.append(val);

            skip_ws();
            if (i < s.size() && s[i] == ',') ++i;
        }
        out.push_back(']');
        return out;
    }

    // Parse a string value. Accepts three forms:
    //   - <|"|>...<|"|>  (or <|"|>...")  -- marker opener, marker or " closer
    //   - "..."                          -- regular JSON string
    //   - bare text                      -- ends at depth-0 terminator
    std::string parse_string(const std::string& terminators) {
        enum Mode { MARKER, QUOTE, BARE };
        Mode mode = BARE;
        if (match_marker(i)) { mode = MARKER; i += marker_len; }
        else if (s[i] == '"') { mode = QUOTE; ++i; }

        auto is_terminator = [&](size_t pos) {
            size_t k = pos;
            skip_ws_at(k);
            if (k >= s.size()) return true;
            return terminators.find(s[k]) != std::string::npos;
        };

        // QUOTE-mode only: honor JSON-style backslash escapes verbatim.
        auto consume_quote_escape = [&](std::string& out) {
            // s[i] == '\\'
            if (i + 1 >= s.size()) {
                json_escape_char(out, s[i]);
                ++i;
                return;
            }
            char nc = s[i + 1];
            switch (nc) {
                case '"': case '\\': case '/':
                case 'b': case 'f': case 'n': case 'r': case 't':
                    out.push_back('\\');
                    out.push_back(nc);
                    i += 2;
                    return;
                case 'u': {
                    if (i + 5 < s.size() &&
                        std::isxdigit(static_cast<unsigned char>(s[i + 2])) &&
                        std::isxdigit(static_cast<unsigned char>(s[i + 3])) &&
                        std::isxdigit(static_cast<unsigned char>(s[i + 4])) &&
                        std::isxdigit(static_cast<unsigned char>(s[i + 5]))) {
                        out.append(s, i, 6);
                        i += 6;
                        return;
                    }
                    break;
                }
                default: break;
            }
            json_escape_char(out, s[i]);
            ++i;
        };

        // For MARKER and BARE modes we first collect the raw content, then
        // JSON-encode it. Backslash policy is decided per-string:
        //   - if every `\` is followed by a valid JSON escape char, treat
        //     them as escapes (so e.g. `\n` becomes a real newline)
        //   - otherwise treat every `\` as a literal char (so Windows paths
        //     like `C:\Users\nock9\Desktop\abc.txt` are preserved verbatim)
        auto is_escape_char = [](char c) {
            return c == '"' || c == '\\' || c == '/' ||
                   c == 'b' || c == 'f' || c == 'n' ||
                   c == 'r' || c == 't' || c == 'u';
        };
        auto encode_raw = [&](const std::string& raw, std::string& out) {
            bool all_valid = true;
            for (size_t k = 0; k < raw.size(); ++k) {
                if (raw[k] == '\\') {
                    if (k + 1 >= raw.size() || !is_escape_char(raw[k + 1])) {
                        all_valid = false;
                        break;
                    }
                    ++k;
                }
            }
            for (size_t k = 0; k < raw.size(); ++k) {
                char c = raw[k];
                if (c == '\\' && all_valid && k + 1 < raw.size()) {
                    char nc = raw[k + 1];
                    if (nc == 'u' && k + 5 < raw.size() &&
                        std::isxdigit(static_cast<unsigned char>(raw[k + 2])) &&
                        std::isxdigit(static_cast<unsigned char>(raw[k + 3])) &&
                        std::isxdigit(static_cast<unsigned char>(raw[k + 4])) &&
                        std::isxdigit(static_cast<unsigned char>(raw[k + 5]))) {
                        out.append(raw, k, 6);
                        k += 5;
                    } else {
                        out.push_back('\\');
                        out.push_back(nc);
                        ++k;
                    }
                } else {
                    json_escape_char(out, c);
                }
            }
        };

        std::string value;
        std::string raw;       // used in MARKER / BARE modes
        int depth = 0;
        while (i < s.size()) {
            if (mode == MARKER) {
                if (match_marker(i)) {
                    size_t after = i + marker_len;
                    if (is_terminator(after)) { i = after; break; }
                    raw.append(quote_marker_lit, marker_len);
                    i = after;
                    continue;
                }
                if (s[i] == '"' || s[i] == '`') {
                    // The model occasionally substitutes a plain `"` or even
                    // a backtick `` ` `` for the closing <|"|> marker. Treat
                    // either as a close only when followed by a terminator,
                    // so they remain valid string content otherwise.
                    if (is_terminator(i + 1)) { ++i; break; }
                    raw.push_back(s[i]);
                    ++i;
                    continue;
                }
                raw.push_back(s[i]);
                ++i;
                continue;
            }
            if (mode == QUOTE) {
                char c = s[i];
                if (c == '\\') {
                    consume_quote_escape(value);
                    continue;
                }
                if (c == '"') { ++i; break; }
                json_escape_char(value, c);
                ++i;
                continue;
            }
            // BARE
            if (match_marker(i)) {
                size_t after = i + marker_len;
                if (depth == 0 && is_terminator(after)) { i = after; break; }
                raw.append(quote_marker_lit, marker_len);
                i = after;
                continue;
            }
            char c = s[i];
            if (depth == 0 && terminators.find(c) != std::string::npos) break;
            if (c == '{' || c == '[' || c == '(') ++depth;
            else if ((c == '}' || c == ']' || c == ')') && depth > 0) --depth;
            raw.push_back(c);
            ++i;
        }

        if (mode != QUOTE) {
            encode_raw(raw, value);
        }

        std::string out = "\"";
        out.append(value);
        out.push_back('"');
        return out;
    }
};

inline std::pair<std::string, json> parse_tool_content(std::string tool_content) {
    tool_content = trim_value(tool_content);

    const std::string prefix = "call:";
    if (tool_content.find(prefix) == 0) {
        tool_content = trim_value(tool_content.substr(prefix.length()));
    }

    // only has tool name but no args
    size_t brace_pos = tool_content.find('{');
    if (brace_pos == std::string::npos) {
        return {trim_value(tool_content), json::object()};
    }

    std::string tool_name = trim_value(tool_content.substr(0, brace_pos));
    std::string args_str = trim_value(tool_content.substr(brace_pos));

    // Rewrite the relaxed args into well-formed JSON via recursive-descent.
    ArgsParser parser(args_str);
    parser.skip_ws();
    std::string normalized;
    if (parser.i < args_str.size() && args_str[parser.i] == '{') {
        normalized = parser.parse_object();
    } else {
        // Unexpected shape; wrap whatever we get into an object so the
        // downstream JSON parse step still has a sensible structure.
        normalized = parser.parse_value("");
    }

    // Final: parse to a real JSON object; fall back to empty object on failure.
    json args_json = json::object();
    try {
        args_json = json::parse(normalized);
    } catch (const std::exception& e) {
        std::cerr << "[WARNING] Failed to parse tool args as JSON: "
                  << e.what() << std::endl;
        std::cerr << "Raw args: " << normalized << std::endl;
    }

    return {tool_name, args_json};
}

/// The model sometimes wraps the real call in a generic envelope, e.g.
/// `call:tool_call{args:{query:...},id:<|"|>memory_search<|"|>}` (ROCm/FastFlowLM#722).
/// Recover the named tool so the client sees a call it can execute.
inline void unwrap_envelope(std::string& name, json& args) {
    static const char* name_keys[] = {"name", "id", "function", "tool"};
    static const char* args_keys[] = {"args", "arguments", "parameters", "params", "input"};
    static const char* wrappers[] = {"tool_call", "call", "function", "tool", "function_call"};

    bool wrapped = false;
    for (const char* w : wrappers) if (name == w) wrapped = true;
    if (!wrapped || !args.is_object()) return;

    // A tool really named `function` or `tool` is legal, so the name alone is not enough -
    // every key has to be one an envelope would carry. Without this, a call like
    // `function{name:<|"|>x<|"|>,quantity:2}` would be read as an envelope and quantity dropped.
    for (auto& kv : args.items()) {
        bool known = false;
        for (const char* k : name_keys) if (kv.key() == k) known = true;
        for (const char* k : args_keys) if (kv.key() == k) known = true;
        if (!known) return;
    }

    std::string inner_name;
    for (const char* k : name_keys) {
        if (args.contains(k) && args[k].is_string()) { inner_name = args[k].get<std::string>(); break; }
    }
    if (inner_name.empty()) return;

    json inner_args = json::object();
    for (const char* k : args_keys) {
        if (args.contains(k) && args[k].is_object()) { inner_args = args[k]; break; }
    }
    name = inner_name;
    args = inner_args;
}

/// Parse one tool-call body (everything between <|tool_call> and <tool_call|>) into the
/// tool name and its arguments as a JSON object. Unparseable args come back as {}.
inline std::pair<std::string, json> parse_tool_call(std::string tool_content) {
    auto parsed = parse_tool_content(std::move(tool_content));
    unwrap_envelope(parsed.first, parsed.second);
    return parsed;
}

/// The chat template applies `| upper` to every parameter `type`, and minja throws when
/// that is a JSON-schema type array such as ["string", "null"] (Python's Jinja would
/// stringify it). Fold such arrays into the first non-null type plus `nullable: true`,
/// which the template already understands, so one optional field cannot fail the
/// whole request. Recurses through nested schemas.
template <typename J>
inline void normalize_schema_types(J& node) {
    if (node.is_array()) {
        for (auto& item : node) normalize_schema_types(item);
        return;
    }
    if (!node.is_object()) return;
    auto type_it = node.find("type");
    if (type_it != node.end() && type_it->is_array()) {
        std::string first;
        bool nullable = false;
        for (const auto& t : *type_it) {
            if (!t.is_string()) continue;
            const std::string s = t.template get<std::string>();
            if (s == "null") nullable = true;
            else if (first.empty()) first = s;
        }
        if (first.empty()) first = nullable ? "string" : "object";
        *type_it = first;
        if (nullable) node["nullable"] = true;
    }
    for (auto& kv : node.items()) {
        if (kv.key() == "type") continue;
        normalize_schema_types(kv.value());
    }
}

template <typename J>
inline J normalize_tools(J tools) {
    normalize_schema_types(tools);
    return tools;
}
}  // namespace gemma4_tools

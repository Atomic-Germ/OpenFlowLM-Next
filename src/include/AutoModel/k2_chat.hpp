// K2-Horizon chat plumbing (TOOLS-K2-HISTORY, TOOLS-K2-REASONING, TOOLS-K2-CALLS).
#pragma once
#include <nlohmann/json.hpp>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace k2_chat {
using json = nlohmann::ordered_json;

inline const std::vector<std::string>& think_ends() {
    static const std::vector<std::string> v = {"</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>"};
    return v;
}
inline const std::string CALLS_OPEN = "<ifm|tool_calls>";
inline const std::string CALLS_CLOSE = "</ifm|tool_calls>";
inline const std::string CALL_OPEN = "<ifm|tool_call>";
inline const std::string CALL_CLOSE = "</ifm|tool_call>";
inline const std::string KEY_OPEN = "<ifm|arg_key>";
inline const std::string KEY_CLOSE = "</ifm|arg_key>";
inline const std::string VALUE_OPEN = "<ifm|arg_value>";
inline const std::string VALUE_CLOSE = "</ifm|arg_value>";

inline std::string trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}

// The template raises on an assistant turn without a thinking field, and the server strips them.
inline json prepare_messages(json messages) {
    static const char* thinking[] = {"think", "think_fast", "think_faster", "reasoning_content", "reasoning"};
    for (auto& m : messages) {
        if (!m.is_object() || m.value("role", "") != "assistant") continue;
        bool has = false;
        for (const char* f : thinking) has = has || (m.contains(f) && m[f].is_string());
        if (!has) {
            for (const char* f : thinking) m.erase(f);
            m["reasoning_content"] = "";
        }
        if (!m.contains("tool_calls") || !m["tool_calls"].is_array()) continue;
        for (auto& tc : m["tool_calls"]) {
            json* fn = tc.is_object() && tc.contains("function") && tc["function"].is_object() ? &tc["function"] : &tc;
            if (!fn->is_object() || !fn->contains("arguments") || !(*fn)["arguments"].is_string()) continue;
            const std::string s = (*fn)["arguments"].get<std::string>();
            const json parsed = json::parse(s.empty() ? "{}" : s, nullptr, false);
            if (parsed.is_object()) (*fn)["arguments"] = parsed;
        }
    }
    return messages;
}

using ParamTypes = std::map<std::string, std::map<std::string, std::string>>;

inline ParamTypes param_types(const json& tools) {
    ParamTypes out;
    if (!tools.is_array()) return out;
    for (const auto& t : tools) {
        const json& fn = t.is_object() && t.contains("function") ? t["function"] : t;
        if (!fn.is_object() || !fn.contains("name") || !fn["name"].is_string()) continue;
        auto& params = out[fn["name"].get<std::string>()];
        if (!fn.contains("parameters") || !fn["parameters"].is_object()) continue;
        const json& props = fn["parameters"].value("properties", json::object());
        if (!props.is_object()) continue;
        for (const auto& [name, spec] : props.items()) {
            std::string type;
            if (spec.is_object() && spec.contains("type")) {
                const json& ty = spec["type"];
                if (ty.is_string()) type = ty.get<std::string>();
                else if (ty.is_array())
                    for (const auto& x : ty)
                        if (x.is_string() && x != "null") { type = x.get<std::string>(); break; }
            }
            params[name] = type;
        }
    }
    return out;
}

// The template writes string values raw and everything else as JSON, so "42" stays a string only by schema.
inline json arg_value(const std::string& raw, const std::string& type) {
    if (type == "string") return raw;
    const std::string t = trim(raw);
    const json v = json::parse(t, nullptr, false);
    return v.is_discarded() ? json(t) : v;
}

// The template's tool_call_format also has a json form, so accept both.
inline std::pair<std::string, json> parse_tool_call(const std::string& block, const ParamTypes& types) {
    json args = json::object();
    const std::string body = trim(block);
    if (!body.empty() && body.front() == '{') {
        const json j = json::parse(body, nullptr, false);
        if (j.is_object()) {
            std::string name = j.contains("name") && j["name"].is_string() ? j["name"].get<std::string>() : "";
            json a = j.value("arguments", json::object());
            if (a.is_string()) a = json::parse(a.get<std::string>(), nullptr, false);
            return {name, a.is_object() ? a : json::object()};
        }
    }
    size_t pos = block.find(KEY_OPEN);
    const std::string name = trim(block.substr(0, pos));
    const auto tool = types.find(name);
    while (pos != std::string::npos) {
        const size_t ks = pos + KEY_OPEN.size();
        const size_t ke = block.find(KEY_CLOSE, ks);
        if (ke == std::string::npos) break;
        const std::string key = trim(block.substr(ks, ke - ks));
        const size_t after_key = ke + KEY_CLOSE.size();
        const size_t next_key = block.find(KEY_OPEN, after_key);
        size_t vs = block.find(VALUE_OPEN, after_key);
        if (vs == std::string::npos || (next_key != std::string::npos && next_key < vs)) {
            args[key] = nullptr;
            pos = next_key;
            continue;
        }
        vs += VALUE_OPEN.size();
        size_t ve = block.find(VALUE_CLOSE, vs);
        const bool closed = ve != std::string::npos && (next_key == std::string::npos || ve < next_key);
        if (!closed) ve = next_key == std::string::npos ? block.size() : next_key;
        std::string type;
        if (tool != types.end()) {
            const auto p = tool->second.find(key);
            if (p != tool->second.end()) type = p->second;
        }
        args[key] = arg_value(block.substr(vs, ve - vs), type);
        pos = block.find(KEY_OPEN, closed ? ve + VALUE_CLOSE.size() : ve);
    }
    return {name, args};
}

struct Event {
    enum Kind { Wait, Reasoning, Content, Tool } kind = Wait;
    std::string text;
    std::string name;
    json args;
};

// One event per feed(), like the server's other parsers; the rest stays buffered for the next piece.
class StreamParser {
public:
    enum Mode { Reasoning, Content, Calls, Call };

    void reset(Mode start = Reasoning) {
        mode_ = start;
        buf_.clear();
        in_wrapper_ = false;
        strip_ = false;
    }
    bool empty() const { return buf_.empty(); }

    Event feed(const std::string& piece, bool final, const ParamTypes& types) {
        buf_ += piece;
        while (true) {
            if (mode_ == Reasoning) {
                size_t at = std::string::npos, len = 0;
                for (const auto& e : think_ends()) {
                    const size_t p = buf_.find(e);
                    if (p < at) { at = p; len = e.size(); }
                }
                if (at == std::string::npos) return emit(Event::Reasoning, final);
                if (at > 0) return take(Event::Reasoning, at);
                buf_.erase(0, len);
                mode_ = Content;
                strip_ = true;
                continue;
            }
            if (mode_ == Content) {
                if (strip_) {
                    const size_t a = buf_.find_first_not_of(" \t\r\n");
                    if (a == std::string::npos) {
                        if (final) buf_.clear();
                        return {};
                    }
                    buf_.erase(0, a);
                    strip_ = false;
                }
                const size_t pw = buf_.find(CALLS_OPEN), pc = buf_.find(CALL_OPEN);
                const size_t at = pw < pc ? pw : pc;
                if (at == std::string::npos) return emit(Event::Content, final);
                if (at > 0) return take(Event::Content, at);
                in_wrapper_ = at == pw;
                buf_.erase(0, in_wrapper_ ? CALLS_OPEN.size() : CALL_OPEN.size());
                mode_ = in_wrapper_ ? Calls : Call;
                continue;
            }
            if (mode_ == Calls) {
                const size_t a = buf_.find_first_not_of(" \t\r\n");
                if (a == std::string::npos) {
                    if (final) buf_.clear();
                    return {};
                }
                buf_.erase(0, a);
                if (buf_.compare(0, CALL_OPEN.size(), CALL_OPEN) == 0) {
                    buf_.erase(0, CALL_OPEN.size());
                    mode_ = Call;
                    continue;
                }
                if (buf_.compare(0, CALLS_CLOSE.size(), CALLS_CLOSE) == 0) {
                    buf_.erase(0, CALLS_CLOSE.size());
                    mode_ = Content;
                    strip_ = true;
                    continue;
                }
                if (!final && (is_prefix(buf_, CALL_OPEN) || is_prefix(buf_, CALLS_CLOSE))) return {};
                mode_ = Content;    // stray text inside the wrapper is content
                continue;
            }
            const size_t end = buf_.find(CALL_CLOSE);
            if (end == std::string::npos && !final) return {};
            if (end == std::string::npos && buf_.empty()) return {};
            const std::string block = buf_.substr(0, end);
            buf_.erase(0, end == std::string::npos ? buf_.size() : end + CALL_CLOSE.size());
            mode_ = in_wrapper_ ? Calls : Content;
            Event ev;
            ev.kind = Event::Tool;
            std::tie(ev.name, ev.args) = parse_tool_call(block, types);
            return ev;
        }
    }

private:
    Mode mode_ = Reasoning;
    std::string buf_;
    bool in_wrapper_ = false;
    bool strip_ = false;

    static bool is_prefix(const std::string& s, const std::string& of) {
        return s.size() < of.size() && of.compare(0, s.size(), s) == 0;
    }
    size_t held() const {
        static const std::vector<std::string> tags = [] {
            std::vector<std::string> t = think_ends();
            t.push_back(CALLS_OPEN);
            t.push_back(CALL_OPEN);
            return t;
        }();
        const size_t from = buf_.size() > 24 ? buf_.size() - 24 : 0;
        for (size_t p = buf_.find('<', from); p != std::string::npos; p = buf_.find('<', p + 1)) {
            const std::string tail = buf_.substr(p);
            for (const auto& t : tags)
                if (is_prefix(tail, t)) return buf_.size() - p;
        }
        return 0;
    }
    Event take(Event::Kind k, size_t n) {
        Event ev;
        ev.kind = k;
        ev.text = buf_.substr(0, n);
        buf_.erase(0, n);
        return ev;
    }
    Event emit(Event::Kind k, bool final) {
        const size_t n = buf_.size() - (final ? 0 : held());
        if (n == 0) return {};
        return take(k, n);
    }
};

struct Response {
    std::string reasoning;
    std::string content;
    std::vector<std::pair<std::string, json>> calls;
};

// The stream parser over the whole text, so streamed and non-streamed replies split identically.
inline Response parse_response(const std::string& text, const ParamTypes& types) {
    Response r;
    StreamParser p;
    p.reset();
    for (bool first = true;; first = false) {
        Event ev = p.feed(first ? text : "", true, types);
        if (ev.kind == Event::Wait) break;
        if (ev.kind == Event::Reasoning) r.reasoning += ev.text;
        else if (ev.kind == Event::Content) r.content += ev.text;
        else r.calls.emplace_back(ev.name, ev.args);
    }
    r.reasoning = trim(r.reasoning);
    r.content = trim(r.content);
    return r;
}

}  // namespace k2_chat

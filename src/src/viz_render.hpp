/// The flow-viz page's slot fill (specs/viz, VIZ-PAGE-FILL); no dependencies, so its test needs none.
#pragma once

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace viz_command {

/// Fill each /*VIZ_...*/ slot of the template once, in order; "</" is escaped so no payload closes its script tag.
inline std::string render(const std::string& tpl, const std::string& fonts, const std::string& data,
                          const std::string& explain, const std::string& model) {
    auto esc = [](const std::string& s) {
        std::string o;
        o.reserve(s.size() + 64);
        for (size_t i = 0; i < s.size(); ++i) {
            o += s[i];
            if (s[i] == '<' && i + 1 < s.size() && s[i + 1] == '/') o += '\\';
        }
        return o;
    };
    const std::pair<const char*, std::string> fill[] = {
        {"/*VIZ_FONTS*/", fonts}, {"/*VIZ_DATA*/", esc(data)}, {"/*VIZ_EXPLAIN*/", esc(explain)}, {"/*VIZ_MODEL*/", esc(model)}};
    std::string out;
    size_t at = 0;
    for (const auto& [slot, val] : fill) {
        size_t p = tpl.find(slot, at);
        if (p == std::string::npos) throw std::runtime_error(std::string("the viz page template has no ") + slot);
        out.append(tpl, at, p - at);
        out += val;
        at = p + std::strlen(slot);
    }
    out.append(tpl, at, std::string::npos);
    return out;
}

}  // namespace viz_command

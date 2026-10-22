// One JSON case per stdin line in, one {"text"} or {"error"} line out: check.py's protocol.
#include <fstream>
#include <iostream>
#include <sstream>
#include "minja/chat-template.hpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: render <chat_template.jinja> <bos> <eos> < cases.jsonl\n";
        return 2;
    }
    std::ifstream f(argv[1], std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    std::unique_ptr<minja::chat_template> tmpl;
    try {
        tmpl = std::make_unique<minja::chat_template>(ss.str(), argv[2], argv[3]);
    } catch (const std::exception& e) {
        std::cout << nlohmann::ordered_json{{"error", std::string("parse: ") + e.what()}}.dump() << std::endl;
        return 1;
    }
    const auto& caps = tmpl->original_caps();
    std::cerr << "caps: tools=" << caps.supports_tools << " tool_calls=" << caps.supports_tool_calls
              << " system=" << caps.supports_system_role << " typed_content=" << caps.requires_typed_content << "\n";
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        nlohmann::ordered_json out;
        try {
            auto c = nlohmann::ordered_json::parse(line);
            minja::chat_template_inputs in;
            in.messages = c.at("messages");
            if (c.contains("tools")) in.tools = c["tools"];
            in.add_generation_prompt = c.value("add_generation_prompt", true);
            if (c.contains("extra_context")) in.extra_context = c["extra_context"];
            minja::chat_template_options opts;
            opts.apply_polyfills = c.value("polyfills", false);
            out["text"] = tmpl->apply(in, opts);
        } catch (const std::exception& e) {
            out["error"] = e.what();
        }
        std::cout << out.dump() << std::endl;
    }
    return 0;
}

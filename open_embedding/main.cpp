/// \file main.cpp
/// \brief Standalone CLI for the open embedding engine (offline verification)
///
///   flm_open_embed --model-dir <dir> --text "..." [--prompt query|document]
///   flm_open_embed --model-dir <dir> --validate-anchor
///   flm_open_embed --model-dir <dir> --text "..." --compare oracle.json --name sample
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "open_embedding/engine.hpp"

static const char* kAnchorQuery = "Which planet is known as the Red Planet?";
static const char* kAnchorDocs[] = {
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
};
static const float kAnchorExpected[] = {0.3011f, 0.6359f, 0.4930f, 0.4889f};

static std::string read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    float dot = 0.f, na = 0.f, nb = 0.f;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na <= 0.f || nb <= 0.f) return 0.0f;
    return dot / std::sqrt(na * nb);
}

int main(int argc, char** argv) {
    std::string model_dir, text, prompt = "query", compare_json;
    bool validate_anchor = false, ids_only = false;
    std::string stages_file, layers_file, kp_file;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--model-dir") model_dir = next();
        else if (a == "--text") text = next();
        else if (a == "--prompt") prompt = next();
        else if (a == "--compare") compare_json = next();
        else if (a == "--validate-anchor") validate_anchor = true;
        else if (a == "--ids") ids_only = true;
        else if (a == "--stages") stages_file = next();
        else if (a == "--layers") layers_file = next();
        else if (a == "--kp") kp_file = next();
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); return 2; }
    }
    if (model_dir.empty()) { std::fprintf(stderr, "--model-dir required\n"); return 2; }

    open_embedding::Engine engine;
    if (!engine.load(model_dir)) return 1;

    open_embedding::task_type_t task = prompt == "document" ? open_embedding::task_type_t::task_document
                                                            : open_embedding::task_type_t::task_query;

    if (!kp_file.empty()) {
        auto kp = engine.debug_kp(text, task);
        nlohmann::json j;
        for (auto& [key, arr] : kp) {
            nlohmann::json v = nlohmann::json::array();
            for (float f : arr) v.push_back(f);
            j[key] = v;
        }
        j["gate_w_len"] = (long long)j["gate_raw"].size();
        std::ofstream(kp_file) << j.dump();
        return 0;
    }
    if (!layers_file.empty()) {
        auto layers = engine.debug_layers(text, task);
        nlohmann::json j = nlohmann::json::array();
        for (auto& st : layers) {
            nlohmann::json arr = nlohmann::json::array();
            for (float v : st) arr.push_back(v);
            j.push_back(arr);
        }
        std::ofstream(layers_file) << j.dump();
        return 0;
    }
    if (!stages_file.empty()) {
        auto stages = engine.debug_stages(text, task);
        nlohmann::json j;
        for (auto& st : stages) {
            nlohmann::json arr = nlohmann::json::array();
            for (float v : st) arr.push_back(v);
            j.push_back(arr);
        }
        std::ofstream(stages_file) << j.dump();
        return 0;
    }
    if (ids_only) {
        std::string pfx;
        std::vector<int32_t> ids = engine.debug_ids(text, task, &pfx);
        for (int id : ids) std::printf("%d ", id);
        std::printf("\n");
        return 0;
    }
    if (validate_anchor) {
        std::string qpfx;
        std::vector<float> qv = engine.embed(kAnchorQuery, open_embedding::task_type_t::task_query, &qpfx);
        std::printf("tokenized: %s%s\n", qpfx.c_str(), kAnchorQuery);
        bool all_ok = true;
        for (int i = 0; i < 4; i++) {
            std::string dpfx;
            std::vector<float> dv = engine.embed(kAnchorDocs[i], open_embedding::task_type_t::task_document, &dpfx);
            float sim = cosine(qv, dv);
            bool ok = std::fabs(sim - kAnchorExpected[i]) < 0.005f;
            all_ok = all_ok && ok;
            std::printf("  %.4f vs official %.4f  %s\n", sim, kAnchorExpected[i], ok ? "OK" : "MISMATCH");
        }
        std::printf("%s\n", all_ok ? "ANCHOR: PASS" : "ANCHOR: FAIL");
        return all_ok ? 0 : 1;
    }
    if (text.empty()) { std::fprintf(stderr, "--text required (or --validate-anchor)\n"); return 2; }

    std::string pfx;
    std::vector<float> v = engine.embed(text, task, &pfx);
    if (v.empty()) { std::fprintf(stderr, "embed failed\n"); return 1; }

    if (!compare_json.empty()) {
        nlohmann::json oracle = nlohmann::json::parse(read_all(compare_json));
        auto key = oracle.contains("name") ? oracle["name"].get<std::string>() : text;
        auto it = oracle.find(key);
        if (it == oracle.end()) { std::fprintf(stderr, "oracle has no key matching\n"); return 2; }
        std::vector<float> ref = it.value()["embedding"].get<std::vector<float>>();
        std::printf("cosine vs oracle: %.6f\n", cosine(v, ref));
        return 0;
    }

    std::printf("[%s] %s\n", prompt.c_str(), text.c_str());
    std::printf("[");
    for (size_t i = 0; i < v.size(); i++) std::printf("%s%.6f", i ? "," : "", v[i]);
    std::printf("]\n");
    return 0;
}
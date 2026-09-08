/// \file flm_add.cpp
/// \brief Native `flm add` installer for the open-kernel system.
///
/// Ports the installer logic that used to live in the standalone `flm-add`
/// Python package (utilities/flm-add) into the main application. The model
/// files are still copied/downloaded and registered in model_list.json, but
/// kernel linking now targets the *family* open_kernels set shipped with the
/// application (xclbins/<official>/open_kernels) instead of a closed per-model
/// .xclbin folder. A model links to its family's open_kernels regardless of
/// fine-tuning or derivate models (see AGENTS.md).
#include "flm_add.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "program_args.hpp"
#include "utils/utils.hpp"
#include "pull/download_model.hpp"

namespace flm_add {

namespace fs = std::filesystem;

// ------------------------------------------------------------------ constants

const std::vector<std::string> REQUIRED_FILES = {
    "config.json", "model.q4nx", "tokenizer.json", "tokenizer_config.json"};
const std::vector<std::string> OPTIONAL_FILES = {
    "chat_template.jinja", "vision_weight.q4nx", "audio_weight.q4nx"};
const std::vector<std::string> ALL_FILES = []() {
    std::vector<std::string> v = REQUIRED_FILES;
    v.insert(v.end(), OPTIONAL_FILES.begin(), OPTIONAL_FILES.end());
    return v;
}();

const std::vector<std::string> SYSTEM_LIST_CANDIDATES = {
    "/opt/fastflowlm/share/flm/model_list.json",
    "/usr/share/flm/model_list.json",
    "/usr/local/share/flm/model_list.json",
};

const std::vector<fs::path> SYSTEM_XCLBIN_PREFIXES = {
    "/opt/fastflowlm/share/flm",
    "/usr/share/flm",
    "/usr/local/share/flm",
};

// Dir-name prefix -> details.family, used only when no official entry can be
// matched by name. The official model_list.json is the primary source.
const std::vector<std::pair<std::string, std::string>> FAMILY_ALIASES = {
    {"qwen3.5-omni", "qwen3.5-omni"},
    {"qwen3.6", "qwen3.6-moe"},
    {"qwen3.5-moe", "qwen3.6-moe"},
    {"qwen3.5", "qwen3.5"},
    {"qwen3", "qwen3"},
    {"qwen2.5vl", "qwen2.5vl"},
    {"qwen2.5", "qwen2"},
    {"qwen2vl", "qwen2vl"},
    {"qwen2", "qwen2"},
    {"gemma4", "gemma4e"},
    {"gemma-4", "gemma4e"},
    {"gemma3", "gemma3"},
    {"llama3", "llama3"},
    {"llama", "llama3"},
    {"granite", "granite"},
    {"crow", "qwen3.5"},
    {"huihui", "qwen3.5"},
    {"qwythos", "qwen3.5"},
    {"qwopus", "qwen3.5"},
    {"darwin", "qwen3.6-moe"},
    {"deepseek-r1-0528", "deepseek-r1-0528"},
    {"deepseek-r1", "deepseek-r1"},
    {"deepseek", "deepseek-r1"},
    {"nanbeige4", "nanbeige"},
    {"nanbeige", "nanbeige"},
    {"gpt-oss", "gpt-oss"},
    {"lfm2.5", "lfm2.5-tk"},
    {"lfm2", "lfm2"},
    {"phi4", "phi4"},
    {"whisper-v3", "whisper-v3"},
    {"whisper", "whisper-v3"},
    {"embed-gemma", "embed-gemma"},
};

const std::vector<std::string> MODELSCOPE_HOSTS = {"modelscope.ai", "modelscope.cn", "modelscope.com"};
const std::vector<std::string> MS_DOMAINS = {"modelscope.ai", "modelscope.cn"};

// ------------------------------------------------------------------ logging

static void log_(const std::string& msg) { std::cerr << msg << std::endl; }
static void err_(const std::string& msg) { std::cerr << "[ERROR] " << msg << std::endl; }

// ------------------------------------------------------------------ json io

static nlohmann::json load_json(const fs::path& p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    nlohmann::json j;
    f >> j;
    return j;
}

static void save_json(const fs::path& p, const nlohmann::json& j) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p);
    f << j.dump(2) << std::endl;
}

// ------------------------------------------------------------------ path resolution

static fs::path find_system_model_list(const std::string& explicit_path) {
    if (!explicit_path.empty()) {
        fs::path p(explicit_path);
        if (fs::is_regular_file(p)) return p;
        throw std::runtime_error("system-list not found: " + explicit_path);
    }
    std::vector<fs::path> candidates;
    if (const char* exe = std::getenv("FLM_BIN_PATH")) {
        candidates.emplace_back(fs::path(exe).parent_path() / "model_list.json");
    }
    for (const auto& c : SYSTEM_LIST_CANDIDATES) candidates.emplace_back(c);
    for (const auto& c : candidates) {
        if (fs::is_regular_file(c)) return c;
    }
    try {
        fs::path app = fs::path(utils::get_executable_directory()) / "model_list.json";
        if (fs::is_regular_file(app)) return app;
    } catch (...) {}
    throw std::runtime_error(
        "Could not locate the system model_list.json. Pass --system-list.");
}

static fs::path user_registry_path(const std::string& arg) {
    if (!arg.empty()) return arg;
    if (const char* env = std::getenv("FLM_CONFIG_PATH")) return env;
    return fs::path(utils::get_models_directory()) / "model_list.json";
}

static fs::path models_root_dir(const std::string& arg) {
    if (!arg.empty()) return arg;
    if (const char* env = std::getenv("FLM_MODEL_PATH")) return fs::path(env) / "models";
    return fs::path(utils::get_models_directory()) / "models";
}

static fs::path system_xclbin_root() {
    fs::path root;
    try {
        root = utils::find_xclbin_path();
    } catch (...) {
        root.clear();
    }
    if (!root.empty() && fs::is_directory(root / "xclbins")) return root / "xclbins";
    for (const auto& prefix : SYSTEM_XCLBIN_PREFIXES) {
        if (fs::is_directory(prefix / "xclbins")) return prefix / "xclbins";
    }
    return {};
}

// ------------------------------------------------------------------ tag derivation

static std::string strip_npu2(const std::string& name) {
    std::regex re("-NPU2$", std::regex::icase);
    return std::regex_replace(name, re, "");
}

static std::pair<std::string, std::string> extract_size(const std::string& bare) {
    std::regex re(R"((\d+(?:\.\d+)?[Bb](?:-[A-Za-z]+\d+(?:\.\d+)?[A-Za-z]*)*))");
    std::smatch m;
    if (!std::regex_search(bare, m, re)) return {"", bare};
    std::string size = m.str(1);
    std::transform(size.begin(), size.end(), size.begin(), ::tolower);
    std::string rest = bare.substr(0, m.position()) + " " + bare.substr(m.position() + m.length());
    auto b = rest.find_first_not_of(" \t");
    auto e = rest.find_last_not_of(" \t");
    rest = (b == std::string::npos) ? "" : rest.substr(b, e - b + 1);
    return {size, rest};
}

static std::string derive_tag(const std::string& dir_name, const std::string& explicit_tag) {
    if (!explicit_tag.empty()) return explicit_tag;
    auto [size, rest] = extract_size(strip_npu2(dir_name));
    if (size.empty())
        throw std::runtime_error("Could not derive a size from '" + dir_name +
            "' (no 'NNb' marker). Pass --tag name:size.");
    std::regex tok_re(R"([-_ ]+)");
    std::vector<std::string> tokens;
    for (auto it = std::sregex_token_iterator(rest.begin(), rest.end(), tok_re, -1);
         it != std::sregex_token_iterator(); ++it) {
        if (!it->str().empty()) tokens.push_back(it->str());
    }
    if (tokens.empty())
        throw std::runtime_error("Could not derive a tag from the repo name. Pass --tag name:size.");
    std::string family = tokens[0];
    std::transform(family.begin(), family.end(), family.begin(), ::tolower);
    std::string variant;
    for (size_t i = 1; i < tokens.size(); ++i) {
        std::string t = tokens[i];
        if (std::regex_match(t, std::regex(R"(\d+(\.\d+)?[MmKk]?)"))) continue;
        std::transform(t.begin(), t.end(), t.begin(), ::tolower);
        variant = t;
        break;
    }
    return variant.empty() ? (family + ":" + size) : (family + "-" + variant + ":" + size);
}

// ------------------------------------------------------------------ official matching

struct Official {
    int common = 0;
    std::string bucket;
    std::string size;
    nlohmann::json info;
};

static std::optional<Official> match_official_entry(const nlohmann::json& reg, const std::string& dir_name) {
    std::optional<Official> best;
    std::vector<std::string> dir_tokens;
    {
        std::regex tok_re(R"([-_ ]+)");
        for (auto it = std::sregex_token_iterator(dir_name.begin(), dir_name.end(), tok_re, -1);
             it != std::sregex_token_iterator(); ++it)
            if (!it->str().empty()) dir_tokens.push_back(it->str());
    }
    const auto& models = reg.value("models", nlohmann::json::object());
    for (auto it = models.begin(); it != models.end(); ++it) {
        const std::string& bucket = it.key();
        const auto& sizes = it.value();
        for (auto sit = sizes.begin(); sit != sizes.end(); ++sit) {
            const std::string& sz = sit.key();
            const nlohmann::json& info = sit.value();
            std::string name = info.value("name", std::string(""));
            if (name.empty()) continue;
            int common = 0;
            std::vector<std::string> name_tokens;
            {
                std::regex tok_re(R"([-_ ]+)");
                for (auto ti = std::sregex_token_iterator(name.begin(), name.end(), tok_re, -1);
                     ti != std::sregex_token_iterator(); ++ti)
                    if (!ti->str().empty()) name_tokens.push_back(ti->str());
            }
            for (size_t k = 0; k < name_tokens.size() && k < dir_tokens.size(); ++k) {
                if (name_tokens[k] == dir_tokens[k]) ++common;
                else break;
            }
            if (common >= 2 && (!best || common > best->common)) {
                Official o; o.common = common; o.bucket = bucket; o.size = sz; o.info = info;
                best = o;
            }
        }
    }
    return best;
}

static std::vector<Official> official_entries_by_family(const nlohmann::json& reg, const std::string& family) {
    std::vector<Official> out;
    const auto& models = reg.value("models", nlohmann::json::object());
    for (auto it = models.begin(); it != models.end(); ++it) {
        const std::string& bucket = it.key();
        const auto& sizes = it.value();
        for (auto sit = sizes.begin(); sit != sizes.end(); ++sit) {
            const std::string& sz = sit.key();
            const nlohmann::json& info = sit.value();
            std::string fam = info.value("details", nlohmann::json::object()).value("family", std::string(""));
            if (fam == family) {
                Official o; o.common = 0; o.bucket = bucket; o.size = sz; o.info = info;
                out.push_back(o);
            }
        }
    }
    return out;
}

static std::optional<Official> match_official_by_family_size(const nlohmann::json& reg,
                                                              const std::string& family, uint64_t size) {
    if (family.empty() || size == 0) return std::nullopt;
    for (const auto& e : official_entries_by_family(reg, family))
        if (e.info.value("size", uint64_t(0)) == size) return e;
    return std::nullopt;
}

static std::pair<std::optional<Official>, std::string> resolve_official(
    const nlohmann::json& reg, const std::string& dir_name, const std::string& family, uint64_t size) {
    auto o = match_official_entry(reg, dir_name);
    if (o) return {o, ""};
    auto by_fs = match_official_by_family_size(reg, family, size);
    if (by_fs) return {by_fs, ""};
    auto entries = official_entries_by_family(reg, family);
    if (entries.size() == 1) {
        std::string note;
        if (size) {
            uint64_t off = entries[0].info.value("size", uint64_t(0));
            if (off && off != size) {
                std::ostringstream ss;
                ss << "tag size " << (size / 1e9) << "B differs from official " << (off / 1e9) << "B";
                note = ss.str();
            }
        }
        return {entries[0], note};
    }
    if (!entries.empty() && size) {
        const Official* best = &entries[0];
        uint64_t bestd = uint64_t(-1);
        for (const auto& e : entries) {
            uint64_t es = e.info.value("size", uint64_t(0));
            uint64_t d = (es > size) ? es - size : size - es;
            if (d < bestd) { bestd = d; best = &e; }
        }
        std::ostringstream ss;
        ss << "no exact size match for " << (size / 1e9) << "B; using "
           << (best->info.value("size", uint64_t(0)) / 1e9) << "B kernels";
        return {*best, ss.str()};
    }
    return {std::nullopt, ""};
}

static std::string derive_family(const nlohmann::json& reg, const std::string& dir_name,
                                 const std::string& explicit_family, const std::optional<Official>& base) {
    if (!explicit_family.empty()) return explicit_family;
    if (base && base->info.contains("details") && base->info["details"].contains("family"))
        return base->info["details"]["family"].get<std::string>();
    std::string lower = dir_name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    for (const auto& [prefix, family] : FAMILY_ALIASES) {
        std::string p = prefix;
        std::transform(p.begin(), p.end(), p.begin(), ::tolower);
        if (lower.rfind(p, 0) == 0) return family;
    }
    throw std::runtime_error("Could not determine details.family for '" + dir_name +
        "'. Pass --family (e.g. qwen3.5, qwen3.6-moe, nanbeige, llama3, ...).");
}

// ------------------------------------------------------------------ repo classification

static bool is_modelscope_host(const std::string& host) {
    std::string h = host;
    std::transform(h.begin(), h.end(), h.begin(), ::tolower);
    for (const auto& mh : MODELSCOPE_HOSTS) {
        if (h == mh) return true;
        if (h.size() > mh.size()) {
            std::string tail = h.substr(h.size() - mh.size());
            char c = h[h.size() - mh.size() - 1];
            if (tail == mh && (c == '.' || c == '-')) return true;
        }
    }
    return false;
}

static std::pair<std::string, std::string> split_remote_repo(const std::string& raw, bool& modelscope) {
    if (raw.rfind("https://", 0) != 0 && raw.rfind("http://", 0) != 0)
        return {"huggingface", raw};
    size_t q = raw.find("://");
    std::string rest = raw.substr(q + 3);
    size_t slash = rest.find('/');
    std::string host = (slash == std::string::npos) ? rest : rest.substr(0, slash);
    std::string path = (slash == std::string::npos) ? "" : rest.substr(slash + 1);
    std::vector<std::string> segs;
    {
        std::stringstream ss(path);
        std::string item;
        while (std::getline(ss, item, '/')) if (!item.empty()) segs.push_back(item);
    }
    static const std::vector<std::string> CUTS = {"resolve", "blob", "tree", "commit", "files", "discuss"};
    for (const auto& cut : CUTS) {
        auto pos = std::find(segs.begin(), segs.end(), cut);
        if (pos != segs.end()) segs.erase(pos, segs.end());
    }
    if (!segs.empty() && segs[0] == "models") segs.erase(segs.begin());
    modelscope = is_modelscope_host(host);
    std::string repo = segs.size() >= 2 ? (segs[0] + "/" + segs[1]) : (segs.empty() ? "" : segs[0]);
    return {modelscope ? "modelscope" : "huggingface", repo};
}

// ------------------------------------------------------------------ http helpers

static nlohmann::json http_json(const std::string& url) {
    std::string body = download_utils::download_string(url);
    if (body.empty()) throw std::runtime_error("empty response from " + url);
    return nlohmann::json::parse(body);
}

static bool verify_file(const fs::path& path, const std::string& expected_sha) {
    if (expected_sha.empty()) return true;
    std::string got = download_utils::calculate_file_sha256(path.string());
    if (got != expected_sha) {
        err_("sha256 mismatch for " + path.filename().string());
        return false;
    }
    return true;
}

static bool download_to(const std::string& url, const fs::path& dest,
                        uint64_t expected_size, const std::string& expected_sha,
                        bool verify, bool quiet) {
    fs::create_directories(dest.parent_path());
    if (!download_utils::download_file(url, dest.string(), false, "", nullptr)) return false;
    if (expected_size && fs::file_size(dest) != expected_size) {
        err_("size mismatch for " + dest.filename().string());
        return false;
    }
    if (verify && !verify_file(dest, expected_sha)) return false;
    return true;
}

// ------------------------------------------------------------------ asset acquisition

static std::map<std::string, nlohmann::json> hf_file_tree(const std::string& repo_id) {
    std::map<std::string, nlohmann::json> entries;
    nlohmann::json tree = http_json("https://huggingface.co/api/models/" + repo_id +
                                    "/tree/main?recursive=true");
    for (const auto& e : tree) {
        std::string p = e.value("path", std::string(""));
        if (!p.empty() && p.find('/') == std::string::npos) entries[p] = e;
    }
    return entries;
}

static std::pair<std::string, std::map<std::string, nlohmann::json>> ms_file_tree(const std::string& repo_id) {
    for (const auto& domain : MS_DOMAINS) {
        std::string url = "https://" + domain + "/api/v1/models/" + repo_id +
                          "/repo/files?Revision=master&Recursive=false";
        try {
            nlohmann::json tree = http_json(url);
            if (tree.value("Code", 0) == 200) {
                std::map<std::string, nlohmann::json> files;
                for (const auto& f : tree.value("Data", nlohmann::json::object()).value("Files", nlohmann::json::array()))
                    if (f.value("Path", std::string("")).size()) files[f["Path"]] = f;
                if (!files.empty()) return {domain, files};
            }
        } catch (...) {}
    }
    throw std::runtime_error("ModelScope repo not found (" + repo_id + ").");
}

static fs::path hf_cache_snapshot(const std::string& repo_id) {
    std::vector<fs::path> roots;
    for (const char* env : {"HF_HUB_CACHE", "HF_HOME"}) {
        if (const char* v = std::getenv(env)) {
            fs::path p(v);
            roots.push_back(p.filename() == "hub" ? p : p / "hub");
        }
    }
    roots.push_back(fs::path(utils::get_models_directory()).parent_path() / ".cache" / "huggingface" / "hub");
    std::string dir_name = "models--" + std::regex_replace(repo_id, std::regex("/"), "--");
    for (const auto& root : roots) {
        fs::path repo = root / dir_name;
        if (!fs::is_directory(repo)) continue;
        fs::path snaps = repo / "snapshots";
        if (!fs::is_directory(snaps)) continue;
        for (const auto& d : fs::directory_iterator(snaps))
            if (fs::is_directory(d.path()) && fs::is_regular_file(d.path() / "config.json")) return d.path();
    }
    return {};
}

static fs::path ms_cache_snapshot(const std::string& repo_id) {
    std::string org, name;
    auto pos = repo_id.find('/');
    org = (pos == std::string::npos) ? "" : repo_id.substr(0, pos);
    name = (pos == std::string::npos) ? repo_id : repo_id.substr(pos + 1);
    std::vector<fs::path> roots;
    if (const char* v = std::getenv("MODELSCOPE_CACHE")) roots.push_back(v);
    roots.push_back(fs::path(utils::get_models_directory()).parent_path() / ".cache" / "modelscope");
    for (const auto& root : roots) {
        for (const auto& base : {root, root / "models"}) {
            fs::path d = base / org / name;
            if (fs::is_regular_file(d / "config.json")) return d;
        }
    }
    return {};
}

static std::vector<std::string> fetch_assets(const std::string& repo_id, fs::path target,
                                             bool modelscope, bool verify, bool force, bool quiet) {
    std::vector<std::string> obtained;
    target = fs::absolute(target);
    fs::create_directories(target);
    if (modelscope) {
        auto [domain, entries] = ms_file_tree(repo_id);
        for (const auto& fname : ALL_FILES) {
            auto it = entries.find(fname);
            if (it == entries.end()) continue;
            fs::path dest = target / fname;
            if (fs::is_regular_file(dest) && !force) { obtained.push_back(fname); continue; }
            uint64_t size = it->second.value("Size", uint64_t(0));
            std::string sha = it->second.value("Sha256", std::string(""));
            std::transform(sha.begin(), sha.end(), sha.begin(), ::tolower);
            if (!quiet) log_("[INFO] Downloading " + fname + " from ModelScope (" + domain + ")...");
            if (!download_to("https://" + domain + "/models/" + repo_id + "/resolve/master/" + fname,
                             dest, size, sha, verify, quiet))
                throw std::runtime_error("failed to download " + fname);
            obtained.push_back(fname);
        }
        return obtained;
    }
    auto entries = hf_file_tree(repo_id);
    for (const auto& fname : ALL_FILES) {
        auto it = entries.find(fname);
        if (it == entries.end()) continue;
        fs::path dest = target / fname;
        if (fs::is_regular_file(dest) && !force) { obtained.push_back(fname); continue; }
        uint64_t size = 0;
        std::string sha;
        if (it->second.contains("lfs")) {
            size = it->second["lfs"].value("size", uint64_t(0));
            sha = it->second["lfs"].value("oid", std::string(""));
        } else {
            size = it->second.value("size", uint64_t(0));
        }
        if (!quiet) log_("[INFO] Downloading " + fname + "...");
        if (!download_to("https://huggingface.co/" + repo_id + "/resolve/main/" + fname,
                         dest, size, sha, verify, quiet))
            throw std::runtime_error("failed to download " + fname);
        obtained.push_back(fname);
    }
    return obtained;
}

static std::vector<std::string> copy_from_dir(const fs::path& src_dir, const fs::path& target, bool force) {
    std::vector<std::string> obtained;
    fs::create_directories(target);
    for (const auto& fname : ALL_FILES) {
        fs::path src = src_dir / fname;
        if (!fs::is_regular_file(src)) continue;
        fs::path dest = target / fname;
        if (fs::is_regular_file(dest) && !force) { obtained.push_back(fname); continue; }
        fs::copy_file(src, dest, fs::copy_options::overwrite_existing);
        obtained.push_back(fname);
    }
    return obtained;
}

// ------------------------------------------------------------------ size / entry

static uint64_t size_from_tag(const std::string& tag) {
    std::regex re(R"(.*:(\d+(?:\.\d+)?)b\b)", std::regex::icase);
    std::smatch m;
    if (!std::regex_match(tag, m, re)) return 0;
    return static_cast<uint64_t>(std::stod(m.str(1)) * 1'000'000'000.0);
}

static uint64_t estimate_size(const fs::path& config_path) {
    try {
        auto cfg = load_json(config_path);
        uint64_t hidden = cfg.value("hidden_size", 0u);
        uint64_t layers = cfg.value("num_hidden_layers", 0u);
        if (!hidden || !layers) return 0;
        uint64_t inter = cfg.value("intermediate_size", 0u);
        uint64_t per_layer = 12 * hidden * hidden;
        if (inter) per_layer += 3 * hidden * inter;
        uint64_t total = per_layer * layers + 2 * hidden * (cfg.value("vocab_size", hidden));
        total = static_cast<uint64_t>(std::max(std::round(total / 1e9 * 2) / 2 * 1e9, 1'000'000'000.0));
        return total;
    } catch (...) { return 0; }
}

static nlohmann::json build_entry(const std::optional<Official>& base, const std::string& dir_name,
                                  const std::vector<std::string>& files, uint64_t size) {
    nlohmann::json entry = base ? base->info : nlohmann::json::object();
    entry["name"] = dir_name;
    entry["files"] = files;
    entry["url"] = "";
    entry["file_url"] = "";
    entry["ms_url"] = "";
    if (!entry.contains("max_prefill_len")) entry["max_prefill_len"] = 4096;
    if (!entry.contains("default_context_length")) entry["default_context_length"] = 8192;
    if (!entry.contains("flm_min_version")) entry["flm_min_version"] = "0.9.45";
    if (!entry.contains("details")) entry["details"] = nlohmann::json::object();
    if (!entry["details"].contains("format")) entry["details"]["format"] = "NPU2";
    if (size) entry["size"] = size;
    bool vlm = std::any_of(files.begin(), files.end(),
        [](const std::string& f) { return f.rfind("vision", 0) == 0; });
    entry["vlm"] = vlm;
    return entry;
}

static void register_entry(const fs::path& user_list, const std::string& tag,
                           const nlohmann::json& entry, const nlohmann::json& system_reg) {
    nlohmann::json registry;
    if (fs::is_regular_file(user_list)) registry = load_json(user_list);
    else registry = nlohmann::json::parse(system_reg.dump());
    if (!registry.contains("model_path")) registry["model_path"] = "models";
    std::string bucket, size;
    {
        std::string t = tag;
        auto c = t.find(':');
        bucket = t.substr(0, c);
        size = (c == std::string::npos) ? "" : t.substr(c + 1);
    }
    registry["models"][bucket][size] = entry;
    save_json(user_list, registry);
}

// The app's file verification (ModelDownloader::verify_and_clean_files) reads a
// SEPARATE model_info.json keyed by tag (not model_list.json). Without an entry
// here, `flm serve <tag>` throws "key '<tag>' not found" during verification.
// Build a minimal but valid entry: one object per installed file with its path
// and size, and the LFS sha256 so the advisory hash check agrees.
static void register_model_info(const fs::path& user_list, const std::string& tag,
                                const std::vector<std::string>& files, const fs::path& model_dir) {
    fs::path info_path = user_list.parent_path() / "model_info.json";
    nlohmann::json info;
    if (fs::is_regular_file(info_path)) info = load_json(info_path);
    nlohmann::json files_arr = nlohmann::json::array();
    for (const auto& fname : files) {
        fs::path p = model_dir / fname;
        if (!fs::is_regular_file(p)) continue;
        uint64_t sz = fs::file_size(p);
        std::string sha = download_utils::calculate_file_sha256(p.string());
        files_arr.push_back(nlohmann::json::object({
            {"type", "file"},
            {"oid", sha},
            {"size", sz},
            {"lfs", nlohmann::json::object({{"oid", sha}, {"size", sz}})},
            {"path", fname},
        }));
    }
    info[tag] = files_arr;
    save_json(info_path, info);
}

// ------------------------------------------------------------------ open-kernel linking

// Find the family open_kernels source directory shipped with the app.
static fs::path resolve_family_open_kernels(const fs::path& xclbin_root,
                                            [[maybe_unused]] const std::string& family,
                                            const std::string& source_name) {
    if (!xclbin_root.empty() && !source_name.empty()) {
        fs::path cand = xclbin_root / source_name / "open_kernels";
        if (fs::is_regular_file(cand / "manifest.json")) return cand;
    }
    if (xclbin_root.empty()) return {};
    std::error_code ec;
    for (const auto& d : fs::directory_iterator(xclbin_root, ec)) {
        if (!d.is_directory()) continue;
        fs::path ok = d.path() / "open_kernels";
        if (fs::is_regular_file(ok / "manifest.json")) return ok;
    }
    return {};
}

static void link_open_kernels(const fs::path& models_root, const std::string& dir_name,
                              const fs::path& source, bool force, bool quiet) {
    if (source.empty()) {
        if (!quiet) log_("[WARN] No family open_kernels found; skipping link. Build/install the "
                        "open kernels for this family, or pass --xclbin-from NAME.");
        return;
    }
    fs::path link = models_root / dir_name / "open_kernels";
    std::error_code ec;
    if (fs::is_symlink(link)) {
        if (fs::read_symlink(link) == source) {
            if (!quiet) log_("[INFO] open_kernels link already in place: " + link.string());
            return;
        }
        fs::remove(link, ec);
    } else if (fs::exists(link, ec)) {
        if (force) fs::remove_all(link, ec);
        else throw std::runtime_error(link.string() + " already exists and is not a symlink. "
                                     "Remove it or pass --force.");
    }
    fs::create_directories(link.parent_path());
    fs::create_symlink(source, link, ec);
    if (ec) throw std::runtime_error("failed to symlink open_kernels: " + ec.message());
    if (!quiet) log_("[INFO] Linked open_kernels: " + link.string() + " -> " + source.string());
}

// ------------------------------------------------------------------ entry point

int run(const program_args_t& a) {
    std::string repo = a.model_tag;
    if (repo.empty()) {
        err_("`flm add` requires a repo argument (HF/ModelScope id, URL, or local directory).");
        return 1;
    }

    bool modelscope = a.modelscope;
    fs::path local_dir;
    if (fs::is_directory(repo)) local_dir = fs::absolute(repo);

    std::string dir_name;
    if (!local_dir.empty()) {
        dir_name = local_dir.filename().string();
    } else {
        std::string kind;
        std::tie(kind, repo) = split_remote_repo(repo, modelscope);
        if (kind == "modelscope") modelscope = true;
        dir_name = repo.substr(repo.find_last_of('/') + 1);
    }
    if (dir_name.empty()) {
        err_("Could not determine a model directory name from the repo.");
        return 1;
    }

    fs::path system_list = find_system_model_list(a.add_system_list);
    nlohmann::json system_reg = load_json(system_list);
    fs::path user_list = user_registry_path(a.add_config);
    fs::path models_root = models_root_dir(a.add_models_root);
    fs::path target = models_root / dir_name;

    std::string tag = derive_tag(dir_name, a.add_tag);
    auto official = match_official_entry(system_reg, dir_name);
    std::string family = derive_family(system_reg, dir_name, a.add_family, official);
    uint64_t size_value = official ? official->info.value("size", uint64_t(0)) : 0;
    if (!size_value) size_value = size_from_tag(tag);
    auto resolved = resolve_official(system_reg, dir_name, family, size_value);
    official = resolved.first;
    std::string note = resolved.second;
    std::string src_tag = official ? (official->bucket + ":" + official->size) : "";
    std::string xclbin_source = a.add_xclbin_from.empty()
        ? (official ? official->info.value("name", std::string("")) : "")
        : a.add_xclbin_from;

    if (a.add_dry_run) {
        std::cout << "repo directory : " << dir_name << std::endl;
        std::cout << "tag            : " << tag << std::endl;
        std::cout << "details.family : " << family << std::endl;
        std::cout << "official match : " << (src_tag.empty() ? "(none)" : src_tag) << std::endl;
        std::cout << "xclbin source  : " << (xclbin_source.empty() ? "(none)" : xclbin_source) << std::endl;
        std::cout << "models dir     : " << target.string() << std::endl;
        std::cout << "registry       : " << user_list.string() << std::endl;
        return 0;
    }

    if (!official) {
        if (!a.add_no_xclbin)
            log_("[WARN] No official model matched; will attempt family open_kernels link by family '" +
                 family + "'.");
    } else if (!note.empty()) {
        log_("[INFO] xclbins from official " + src_tag + " (" + note + ")");
    }

    // --- acquire model files ---
    std::vector<std::string> files;
    if (!local_dir.empty()) {
        if (!a.sub_process_mode) log_("[INFO] Using local model directory: " + local_dir.string());
        files = copy_from_dir(local_dir, target, a.force_redownload);
    } else {
        fs::path snapshot = modelscope ? ms_cache_snapshot(repo) : hf_cache_snapshot(repo);
        if (!snapshot.empty()) {
            log_("[INFO] Found local " + std::string(modelscope ? "ModelScope" : "HF") +
                 " cache: " + snapshot.string());
            files = copy_from_dir(snapshot, target, a.force_redownload);
        } else {
            log_("[INFO] Downloading model files from " + std::string(modelscope ? "ModelScope" : "Hugging Face") +
                 ": " + repo);
            files = fetch_assets(repo, target, modelscope, !a.add_no_verify, a.force_redownload, a.sub_process_mode);
        }
    }

    for (const auto& f : REQUIRED_FILES)
        if (!fs::is_regular_file(target / f))
            throw std::runtime_error("Model is missing required file: " + f);

    if (!size_value) size_value = estimate_size(target / "config.json");
    nlohmann::json entry = build_entry(official, dir_name, files, size_value);
    entry["details"]["family"] = family;

    register_entry(user_list, tag, entry, system_reg);
    log_("[INFO] Registered tag '" + tag + "' in " + user_list.string());
    register_model_info(user_list, tag, files, target);
    log_("[INFO] Registered file metadata for '" + tag + "' in " +
         user_list.parent_path().string() + "/model_info.json");

    if (!a.add_no_xclbin) {
        fs::path xroot = system_xclbin_root();
        fs::path source = resolve_family_open_kernels(xroot, family, xclbin_source);
        link_open_kernels(models_root, dir_name, source, a.force_redownload, a.sub_process_mode);
    }

    std::cout << "\nDone: " << dir_name << " installed to " << target.string() << std::endl;
    std::cout << "Run:  flm run " << tag << "   (or: flm serve " << tag << ")\n" << std::endl;
    return 0;
}

} // namespace flm_add

/// \file model_downloader.cpp
/// \brief Model downloader class
/// \author OpenFlowLM Team
/// \date 2025-06-24
/// \version 0.9.24
/// \note This class is used to download models from the huggingface
#include "model_downloader.hpp"
#include "utils/utils.hpp"
#include "download_model.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cctype>
#include <unordered_set>

namespace {

std::string percent_encode_filename(std::string_view filename) {
    static constexpr char kHex[] = "0123456789ABCDEF";
    std::string encoded;
    for (const unsigned char ch : filename) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' || ch == '.' || ch == '~') {
            encoded.push_back(static_cast<char>(ch));
        } else {
            encoded.push_back('%');
            encoded.push_back(kHex[ch >> 4]);
            encoded.push_back(kHex[ch & 0x0f]);
        }
    }
    return encoded;
}

bool is_hex_revision(const std::string& revision) {
    return revision.size() == 40 &&
           std::all_of(revision.begin(), revision.end(), [](unsigned char ch) {
               return std::isxdigit(ch) != 0;
           });
}

/// \brief load the pinned per-file records describing a model's download
/// \param model_info the resolved catalog entry
/// \param model_tag the resolved "family:size" tag
/// \note A model whose artifacts differ per NPU generation shares one tag
///       across platforms, so its entry names the record set explicitly via
///       "model_info_key"; everything else is keyed by its tag.
nlohmann::json load_model_file_records(const nlohmann::json& model_info,
                                       const std::string& model_tag) {
    const std::string key = model_info.value("model_info_key", model_tag);
    std::ifstream stream(utils::find_model_info());
    if (!stream.is_open()) {
        throw std::runtime_error("model_info.json could not be opened");
    }
    const nlohmann::json all = nlohmann::json::parse(stream);
    const auto it = all.find(key);
    // A model that is not in model_info.json has no remote records: it was
    // installed from a local directory, so there is nothing to check its
    // files against. An empty set says so, and the caller decides what to
    // do with it -- the pull path reports the gap, the check path has
    // nothing to verify.
    if (it == all.end()) return nlohmann::json::array();
    return *it;
}

const nlohmann::json& find_file_record(const nlohmann::json& records,
                                       const std::string& filename) {
    const auto record = std::find_if(records.begin(), records.end(), [&](const auto& value) {
        return value.at("path") == filename;
    });
    if (record == records.end()) {
        throw std::runtime_error("missing model_info record for " + filename);
    }
    return *record;
}

struct ResolvedModelFile {
    ModelFileSource source;
    std::string revision;
    std::uint64_t size;
    bool is_lfs;
    download_utils::HashAlgorithm hash_algorithm;
    std::string hash;
};

}  // namespace

bool uses_pinned_sources(const nlohmann::json& model_info) {
    const auto it = model_info.find("file_sources");
    return it != model_info.end() && it->is_object() && !it->empty();
}

ModelFileSource resolve_file_source(const nlohmann::json& model_info,
                                    std::string_view filename,
                                    bool use_modelscope) {
    // A present-but-misspelled table must fail loudly here rather than be
    // silently ignored; the strict gate in uses_pinned_sources only fires on
    // a non-empty object, so an empty table pins nothing and falls through.
    if (model_info.contains("file_sources")) {
        const auto& sources = model_info.at("file_sources");
        if (!sources.is_object()) {
            throw std::runtime_error("file_sources must be an object");
        }
        if (!sources.empty() && use_modelscope) {
            throw std::runtime_error("pinned Hugging Face per-file sources are required; --modelscope is not supported");
        }
        std::unordered_set<std::string> files;
        for (const auto& file : model_info.at("files")) {
            files.insert(file.get<std::string>());
        }
        for (const auto& [key, value] : sources.items()) {
            if (!files.contains(key)) {
                throw std::runtime_error("unknown file_sources key: " + key);
            }
            if (!value.is_object() || value.size() != 2 ||
                !value.contains("url") || !value.at("url").is_string() ||
                value.at("url").get<std::string>().empty()) {
                throw std::runtime_error("file source requires exactly a non-empty string url and revision");
            }
            if (!value.contains("revision") || !value.at("revision").is_string() ||
                !is_hex_revision(value.at("revision").get<std::string>())) {
                throw std::runtime_error("file source revision must be a 40-character hexadecimal string");
            }
        }
        const auto override = sources.find(std::string(filename));
        if (override != sources.end()) {
            const std::string base = override->at("url");
            const std::string revision = override->at("revision");
            return {base + "/resolve/" + revision + "/" +
                        percent_encode_filename(filename) + "?download=true",
                    revision};
        }
    }

    const std::string base_url = use_modelscope
        ? model_info.at("ms_url").get<std::string>()
        : model_info.at("url").get<std::string>();
    if (base_url.find("resolve") != std::string::npos) {
        return {base_url + "/" + std::string(filename) + "?download=true", {}};
    }
    return {base_url + "/resolve/main/" + std::string(filename) + "?download=true", {}};
}

namespace {

ResolvedModelFile resolve_model_file(const nlohmann::json& model_info,
                                     const nlohmann::json& records,
                                     const std::string& filename,
                                     bool use_modelscope) {
    const auto& record = find_file_record(records, filename);
    const bool is_lfs = record.contains("lfs");
    const bool has_explicit_sha256 = record.contains("sha256");
    // GGUF repositories commonly carry an explicit sha256 alongside (or
    // instead of) LFS metadata; prefer it whenever the record states one.
    const ModelFileSource source =
        resolve_file_source(model_info, filename, use_modelscope);
    return {
        source,
        source.revision,
        record.at("size").get<std::uint64_t>(),
        is_lfs,
        has_explicit_sha256 || is_lfs
            ? download_utils::HashAlgorithm::Sha256
            : download_utils::HashAlgorithm::GitBlobSha1,
        has_explicit_sha256
            ? record.at("sha256").get<std::string>()
            : (is_lfs ? record.at("lfs").at("oid").get<std::string>()
                      : record.at("oid").get<std::string>())};
}

}  // namespace

namespace {

std::vector<pull::RegistryFile> registry_for(const nlohmann::json& model_info, const std::string& tag,
                                             const std::vector<std::string>& files,
                                             std::vector<std::string>* unlisted) {
    nlohmann::json manifest;
    try {
        manifest = load_model_file_records(model_info, tag);
    } catch (const std::exception&) {}   // no manifest: every file is unlisted, checked for presence only
    return pull::registry_files(manifest, files, unlisted);
}

}  // namespace

/// \brief Constructor
/// \param models the model list
/// \return the model downloader
ModelDownloader::ModelDownloader(model_list& models) 
    : supported_models(models), curl_init() {
}

/// \brief Check if the model is downloaded
/// \param model_tag the model tag
/// \return true if the model is downloaded, false otherwise
ModelDownloader::ModelStatus ModelDownloader::is_model_downloaded(const std::string& model_tag, bool sub_process_mode, bool fast_check) {
    // An unresolvable tag stays Missing, as before: get_missing_files used to
    // swallow the lookup failure internally and report everything absent.
    std::string new_model_tag;
    nlohmann::json model_info;
    try {
        std::tie(new_model_tag, model_info) = supported_models.get_model_info(model_tag);
    } catch (const std::exception&) {
        return ModelStatus::Missing;
    }
    // A pinned entry downloads at an immutable revision, so presence alone
    // says nothing: a truncated or re-uploaded file with the right name
    // would otherwise read as Ready. The full hash pass below is what makes
    // `oflm list` (fast_check) cheap and everything else exact.
    const bool strict_integrity = uses_pinned_sources(model_info);
    auto missing_files = get_missing_files(new_model_tag);
    // `files` is authoritative: the GGUF embedding (embed-gemma:300m) ships
    // weights + tokenizer files and has no `config.json` to gate on. Testing the
    // missing set for `config.json` unconditionally made every such entry
    // look "not missing" when its sole weight was absent, then
    // `check_model_compatibility` called `LM_Config::from_pretrained` which
    // does `open(<dir>/config.json)` and on failure did `exit(1)` -- that
    // killed `oflm list` after 6 rows with "Failed to open file:
    // ~/.config/oflm/models/embeddinggemma-300M-GGUF". Even now that the
    // loader throws, a missing-config model must not reach the version check
    // at all.
    std::vector<std::string> model_files = model_info.value("files", std::vector<std::string>{});
    const bool requires_config =
        std::find(model_files.begin(), model_files.end(), "config.json") != model_files.end();
    if (!requires_config) {
        return missing_files.empty() &&
               (!strict_integrity || fast_check || verify_files(new_model_tag, false, sub_process_mode))
            ? ModelStatus::Ready : ModelStatus::Missing;
    }
    bool is_config_file_missing = std::find(missing_files.begin(), missing_files.end(), "config.json") != missing_files.end();
    ModelStatus modelstatus = ModelStatus::Missing;

    if (!is_config_file_missing) {
        try {
            modelstatus = check_model_compatibility(new_model_tag, sub_process_mode);
        } catch (const std::exception& e) {
            if (!sub_process_mode) {
                header_print("WARNING", std::string("Skipping version check for ") + new_model_tag + ": " + e.what());
            }
            return ModelStatus::Missing;
        }

        if (modelstatus == ModelStatus::Outdated) {
            if (!fast_check) {
                header_print("OFLM", "Checking outdated files...");
                verify_files(new_model_tag, false, sub_process_mode);
            }
        }
        else if (modelstatus == ModelStatus::Ready) {
            if (!missing_files.empty() ||
                (strict_integrity && !fast_check &&
                 !verify_files(new_model_tag, false, sub_process_mode))) {
                // config.json is present and the version check passed, but
                // other files (e.g. weights) are still missing -- or, for a
                // pinned entry, present but not the pinned bytes.
                modelstatus = ModelStatus::Missing;
            }
        }
    }
    return modelstatus;
}

/// \brief Check if the model is compatible with the current OFLM version
/// \param model_tag the model tag
/// \return true if the model is compatible, false otherwise
ModelDownloader::ModelStatus ModelDownloader::check_model_compatibility(const std::string& model_tag, bool sub_process_mode) {
    auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
    LM_Config config;
    config.from_pretrained(this->supported_models.get_model_path(new_model_tag));
    std::string oflm_version = config.oflm_version;
    // oflm_min_version, or the flm_min_version that a registry written before the
    // oflm rename carries (#41) -- the field was renamed in the DATA as well as the
    // code, and nothing read the old name.
    //
    // An entry with neither is not a reason to abort. The implicit conversion this
    // replaces threw `[json.exception.type_error.302] type must be string, but is
    // null` straight out of check_model_compatibility, which `oflm list` calls once
    // per entry -- so a single pre-rename or hand-written registry entry killed the
    // ENTIRE listing, naming neither the model nor the field. Measured on a user
    // registry carrying flm_min_version: 1 row printed, 39 lost, exit 1.
    std::string oflm_min_version;
    for (const char* key : {"oflm_min_version", "flm_min_version"}) {
        auto it = model_info.find(key);
        if (it != model_info.end() && it->is_string()) {
            oflm_min_version = it->get<std::string>();
            break;
        }
    }
    // Nothing to compare against -- the same reasoning as the "0.0.0" case below.
    if (oflm_min_version.empty()) {
        return ModelStatus::Ready;
    }

    // A CHECKPOINT THAT IS NOT AN OFLM ARTIFACT HAS NO VERSION TO COMPARE.
    //
    // LM_Config defaults oflm_version to "0.0.0" when config.json has no such
    // key -- and no upstream HuggingFace checkpoint has one, because it is a
    // field this project writes. So every model listed with the author's OWN
    // files reads as version 0, compares below the entry's oflm_min_version,
    // and is reported Outdated forever: `oflm list` shows a warning triangle
    // and ensure_*_model_loaded() re-pulls a complete, correct download on
    // every start.
    //
    // embed-gemma:300m is in exactly that state today, on a freshly pulled
    // tree: min 0.9.15 against a checkpoint that declares nothing.
    //
    // "absent" and "0.0.0" are already indistinguishable here, so treating the
    // default as "not versioned" changes nothing for any artifact that really
    // carries a version -- it only stops the check firing on models it was
    // never about.
    if (oflm_version == "0.0.0") {
        return ModelStatus::Ready;
    }
    int l_l, m_l, r_l; //left, middle, right on local version
    int l_r, m_r, r_r; //left, middle, right on requried version
    int l_f, m_f, r_f; //left, middle, right on oflm version
    sscanf(__OFLM_VERSION__, "%d.%d.%d", &l_f, &m_f, &r_f);
    sscanf(oflm_version.c_str(), "%d.%d.%d", &l_l, &m_l, &r_l);
    sscanf(oflm_min_version.c_str(), "%d.%d.%d", &l_r, &m_r, &r_r);
    uint32_t local_version_u32 = l_l * 1000000 + m_l * 1000 + r_l;
    uint32_t required_version_u32 = l_r * 1000000 + m_r * 1000 + r_r;
    uint32_t oflm_version_u32 = l_f * 1000000 + m_f * 1000 + r_f;

    if (local_version_u32 > oflm_version_u32) {
        if (!sub_process_mode) {
            header_print("WARNING", "Local model " + model_tag + " version: " + oflm_version + " > " + __OFLM_VERSION__);
            header_print("WARNING", "Please update OFLM to the latest version.");
        }
        return ModelStatus::Incompatible;
    }
    if (local_version_u32 < required_version_u32) {
        if (!sub_process_mode) {
            header_print("WARNING", "Local model " + model_tag + " version: " + oflm_version + " < " + oflm_min_version);
            // header_print("OFLM", "Re-pulling latest model...");
        }
        return ModelStatus::Outdated;
    }
    return ModelStatus::Ready;
}
/// \brief Pull the model
/// \param model_tag the model tag
/// \param force_redownload true if the model should be downloaded even if it is already downloaded
/// \return true if the model is downloaded, false otherwise
bool ModelDownloader::pull_model(const std::string& model_tag, bool use_modelscope, bool force_redownload) {
    try {
        // Get model info
        auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
        std::string model_name = model_info["name"];
        std::string model_server = use_modelscope ? "ModelScope" : "HuggingFace";
        if (use_modelscope && uses_pinned_sources(model_info)) {
            // Validate this before any ready-state early return.
            resolve_file_source(model_info, model_info.at("files").at(0).get<std::string>(), true);
        }
        
        header_print("OFLM", "Pulling model from " + model_server + "...");
        header_print("OFLM", "Model: " + new_model_tag);
        header_print("OFLM", "Name: " + model_name);
        if (uses_pinned_sources(model_info)) {
            for (const auto& [file, pin] : model_info.at("file_sources").items()) {
                header_print("OFLM", "Pinned: " + file + " @ " +
                             pin.at("revision").get<std::string>());
            }
        }

        // The plan below checks pinned bytes without deleting existing files.
        // Preflight only needs presence and version compatibility here.
        ModelDownloader::ModelStatus status = is_model_downloaded(new_model_tag, false, true);
        if (status == ModelStatus::Incompatible) {
            return true;
        }
        const bool pinned = uses_pinned_sources(model_info);

        const std::string model_path = supported_models.get_model_path(new_model_tag);
        std::vector<std::string> model_files = model_info["files"];
        std::vector<std::string> unlisted;
        const auto registry = registry_for(model_info, new_model_tag, model_files, &unlisted);
        if (!unlisted.empty()) {
            std::string names;
            for (const auto& n : unlisted) names += (names.empty() ? "" : ", ") + n;
            header_print("ERROR", "model_info.json describes none of these files required by model_list.json, "
                                  "so they cannot be downloaded: " + names);
            header_print("ERROR", "Adding a model needs an entry in BOTH files.");
            header_print("ERROR", "For models installed with `oflm add`, re-run `oflm add` with the original source.");
            return false;
        }

        // a pinned transport resumes its .part and verifies it; an unpinned one starts clean
        for (const auto& f : registry) {
            if (pinned) break;
            std::error_code ec;
            std::filesystem::remove(std::filesystem::path(model_path) / (f.path + ".part"), ec);   // a killed pull's leftovers
        }
        pull::Record record = pull::read_record(model_path);
        const bool all_recorded = std::all_of(registry.begin(), registry.end(), [&](const pull::RegistryFile& f) {
            return f.oid.empty() || record.count(f.path) != 0;
        });
        if (!pinned && status == ModelStatus::Ready && all_recorded && !force_redownload) {
            header_print("OFLM", "Model already downloaded. Use --force to re-download.");
            return true;
        }

        std::filesystem::create_directories(model_path);
        if (!all_recorded && !force_redownload) {
            header_print("OFLM", "No install record yet: checking the files already present against the registry (once)...");
        }
        const size_t recorded_before = record.size();
        const auto plan = pull::plan_pull(registry, model_path, &record, force_redownload, pinned);
        if (record.size() != recorded_before) {
            pull::write_record(model_path, record);
        }
        if (plan.empty()) {
            header_print("OFLM", "All files are present and current.");
            return !pinned || verify_files(new_model_tag, use_modelscope);
        }

        auto [downloads, sum_file_size] = build_download_list(new_model_tag, plan, use_modelscope);
        header_print("OFLM", "Files to download (" << plan.size() << ", " << std::fixed << std::setprecision(2)
                                                   << sum_file_size << " MB):");
        for (const auto& [f, why] : plan) {
            std::cout << "  - " << f.path << " (" << std::fixed << std::setprecision(2)
                      << static_cast<double>(f.size) / 1024 / 1024 << " MB, " << pull::describe(why) << ")" << std::endl;
        }

        // record each file as it lands, so an interrupted pull keeps what it finished
        auto progress = get_progress_callback();
        size_t recorded = 0;
        auto on_file = [&](size_t completed, size_t total) {
            for (; recorded < completed && recorded < plan.size(); ++recorded) {
                const pull::RegistryFile& f = plan[recorded].first;
                if (!f.oid.empty()) record[f.path] = f.oid;
            }
            pull::write_record(model_path, record);
            progress(completed, total);
        };
        bool success = download_utils::download_multiple_files(downloads, on_file);

        if (success) {
            header_print("OFLM", "Model downloaded successfully!");
            
            // Verify every final file using the same pinned metadata used to
            // download it. For a pinned entry the answer is authoritative, so
            // it is the return value; the legacy path keeps its presence
            // check and historical `true`.
            auto final_missing = get_missing_files(new_model_tag);
            const bool verified = final_missing.empty() &&
                (!uses_pinned_sources(model_info) ||
                 verify_files(new_model_tag, use_modelscope));
            if (verified) {
                header_print("OFLM", "All files verified successfully.");
            } else {
                header_print("WARNING", "Some files are missing or failed verification after download.");
            }
            if (!uses_pinned_sources(model_info)) return true;
            return verified;
        } else {
            header_print("ERROR", "Failed to download model files.");
            return false;
        }
        
    } catch (const std::exception& e) {
        header_print("ERROR", "Exception during download: " + std::string(e.what()));
        return false;
    }
}

/// \brief Model not found
/// \param model_tag the model tag
void ModelDownloader::model_not_found(const std::string& model_tag) {
    header_print("ERROR", "Model not found: " + model_tag);
    header_print("ERROR", "Supported models: ");
    nlohmann::json models = supported_models.get_all_models();
    for (const auto& model : models["models"]) {
        header_print("ERROR", "  - " + model["name"].get<std::string>());
    }
}

/// \brief Get missing files
/// \param model_tag the model tag
/// \return the missing files
std::vector<std::string> ModelDownloader::get_missing_files(const std::string& model_tag) {
    std::vector<std::string> missing_files;

    try {
        auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
        std::string model_path = supported_models.get_model_path(new_model_tag);
        std::vector<std::string> model_files = model_info["files"];
        std::vector<std::string> unlisted;
        const auto registry = registry_for(model_info, new_model_tag, model_files, &unlisted);
        const pull::Record record = pull::read_record(model_path);

        // PULL-STALE: the same test pull_model fetches by, so status and download agree
        for (const auto& f : registry) {
            const pull::Fetch why = pull::status_of(f, model_path, record);
            if (why == pull::Fetch::No) continue;
            if (why == pull::Fetch::Size) {
                std::error_code ec;
                const auto on_disk = std::filesystem::file_size(get_model_file_path(model_path, f.path), ec);
                // stderr, not stdout: `oflm list --json` reaches this and its stdout is one JSON document (#133)
                header_print_r("WARNING", f.path + " is " + std::to_string(on_disk) + " bytes, the manifest says " +
                                          std::to_string(f.size) + " -- treating it as missing");
            } else if (why == pull::Fetch::Oid) {
                header_print_r("WARNING", f.path + " was downloaded for another revision -- treating it as missing");
            }
            missing_files.push_back(f.path);
        }
        for (const auto& name : unlisted) {
            if (!file_exists(get_model_file_path(model_path, name))) missing_files.push_back(name);
        }
    } catch (const std::exception& e) {
        header_print("ERROR", "Error checking missing files: " + std::string(e.what()));
    }

    return missing_files;
}

/// \brief Get present files
/// \param model_tag the model tag
/// \return the present files
std::vector<std::string> ModelDownloader::get_present_files(const std::string& model_tag) {
    std::vector<std::string> present_files;
    
    try {
        auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
        std::string model_name = model_info["name"];
        std::string model_path = supported_models.get_model_path(new_model_tag);
        std::vector<std::string> model_files = model_info["files"];

        // Check if this is a VLM model (default to false if key doesn't exist)
        
        // Check each required model file
        for (int i = 0; i < model_files.size(); ++i) {
            std::string filename = model_files[i];
            std::string file_path = get_model_file_path(model_path, filename);
            if (file_exists(file_path)) {
                present_files.push_back(filename);
            }
        }     
    } catch (const std::exception& e) {
        header_print("ERROR", "Error checking present files: " + std::string(e.what()));
    }
    
    return present_files;
}

/// \brief Get progress callback
/// \return the progress callback
std::function<void(size_t, size_t)> ModelDownloader::get_progress_callback() {
    return [](size_t completed, size_t total) {
        if (total > 0) {
            double percentage = (static_cast<double>(completed) / total) * 100.0;
            std::cout << "\r[OFLM]  Overall progress:  " << completed << "/" << total << " files" << std::flush;
            
            std::cout << std::endl;
        }
    };
}

/// \brief Check if the file exists
/// \param file_path the file path
/// \return true if the file exists, false otherwise
bool ModelDownloader::file_exists(const std::string& file_path) {
    return std::filesystem::exists(file_path) && std::filesystem::is_regular_file(file_path);
}

/// \brief Get the model file path
/// \param model_path the model path
/// \param filename the filename
/// \return the model file path
std::string ModelDownloader::get_model_file_path(const std::string& model_path, const std::string& filename) {
    std::filesystem::path full_path = std::filesystem::path(model_path) / filename;
    return full_path.string();
}

/// \brief Build the download list
/// \param model_tag the model tag
/// \param modelscope true to download from ModelScope instead of HuggingFace
/// \param plan the files to fetch, in completion/record order
/// \return the download list
std::pair<nlohmann::json, float> ModelDownloader::build_download_list(
    const std::string& model_tag, const std::vector<std::pair<pull::RegistryFile, pull::Fetch>>& plan, bool modelscope) {
    nlohmann::json downloads = nlohmann::json::array();
    float sum_file_size = 0;
    auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
    const std::string model_path = supported_models.get_model_path(new_model_tag);
    // one entry per planned file, in plan order: pull_model's record keys off that order
    for (const auto& [f, why] : plan) {
        const ModelFileSource source = resolve_file_source(model_info, f.path, modelscope);
        const float file_size = static_cast<float>(f.size) / 1024 / 1024;
        sum_file_size += file_size;
        downloads.push_back(nlohmann::json{
            {"file", f.path},
            {"size", file_size},
            {"expected_size", f.size},
            {"url", source.url},
            {"localpath", get_model_file_path(model_path, f.path)},
            {"oid", f.oid},
            {"is_lfs", f.lfs},
            {"revision", source.revision},
        });
        // This marker opts into strict transport verification. Ordinary registry
        // hashes stay advisory even when size/hash metadata is available.
        if (uses_pinned_sources(model_info))
            downloads.back()["hash_algorithm"] = f.lfs ? "sha256" : "git_blob_sha1";
    }
    return std::make_pair(downloads, sum_file_size);
}

/// \brief Remove a model and all its files
/// \param model_tag the model tag
/// \return true if the model was successfully removed, false otherwise
/// \note Recursive: model directories routinely hold subdirectories (the
///       1_Pooling/2_Dense heads, a local npu_matmul_f32 set) and symlinks
///       (the open_kernels link oflm-add creates). remove_all deletes links
///       themselves, never their targets. The resolved directory must stay
///       inside the models root, so a hostile or hand-edited registry entry
///       whose name escapes it (a ".." segment) is refused, never followed.
///       Unknown tags are refused too: get_model_info falls back to a default
///       entry instead of throwing, and removing that would delete the wrong
///       model. The user-level xclbins symlink for the model (also an
///       oflm-add dropping, and only ever a symlink) is removed alongside.
bool ModelDownloader::remove_model(const std::string& model_tag, bool sub_process_mode) {
    try {
        // Membership first: get_model_info falls back to a default entry
        // (asserting on a registry without one) instead of throwing, so only
        // resolve tags the registry actually carries. Otherwise a typo could
        // delete the wrong model.
        std::string probe = model_tag;
        if (const auto slash = probe.find('/'); slash != std::string::npos) {
            probe = probe.substr(slash + 1);
        }
        if (!supported_models.is_model_supported(probe)) {
            header_print("ERROR", "Model not found: " + model_tag);
            model_not_found(model_tag);
            return false;
        }
        // Check if model exists in supported models by trying to get its info
        std::string new_model_tag;
        nlohmann::json model_info;
        try {
            std::tie(new_model_tag, model_info) = supported_models.get_model_info(model_tag);
        } catch (const std::exception& e) {
            header_print("ERROR", "Model not found: " + model_tag);
            model_not_found(model_tag);
            return false;
        }

        // Get model path
        std::string model_path = supported_models.get_model_path(new_model_tag);

        std::error_code ec;
        const auto root =
            std::filesystem::weakly_canonical(supported_models.get_model_root_path(), ec);
        if (ec) {
            header_print("ERROR", "Could not resolve models directory: " + ec.message());
            return false;
        }
        const auto target = std::filesystem::weakly_canonical(model_path, ec);
        if (ec) {
            header_print("ERROR", "Could not resolve model directory: " + ec.message());
            return false;
        }
        const auto rel = std::filesystem::relative(target, root, ec);
        if (ec || rel.empty() || rel.begin()->string() == "..") {
            header_print("ERROR", "Refusing to remove outside the models directory: " + model_path);
            return false;
        }

        // Check if model directory exists
        if (!std::filesystem::exists(target, ec)) {
            if (!sub_process_mode)
                header_print("OFLM", "Model directory does not exist: " + model_path);
            return true; // Consider it already removed
        }

        if (!sub_process_mode) {
            header_print("OFLM", "Removing model: " + new_model_tag);
            header_print("OFLM", "Path: " + model_path);
        }

        // Remove the whole tree: weights, nested head directories, and the
        // open_kernels symlink (the link itself, not the kernels it points at).
        const std::uintmax_t removed = std::filesystem::remove_all(target, ec);
        if (ec) {
            header_print("ERROR", "Failed to remove model directory: " + model_path +
                                  " (" + ec.message() + ")");
            return false;
        }
        if (!sub_process_mode)
            header_print("OFLM", "Successfully removed " + std::to_string(removed) +
                                 " entries and model directory.");

        // Drop the user-level xclbins symlink oflm-add created for this model
        // name, when one exists. Only symlinks are ever touched here: a real
        // directory (the system kernels) is left alone.
        const std::string link_name = model_info.value("name", std::string());
        if (!link_name.empty()) {
            for (const auto& xclbin_root : utils::xclbin_roots()) {
                std::error_code link_ec;
                const auto link =
                    std::filesystem::path(xclbin_root) / "xclbins" / link_name;
                if (std::filesystem::is_symlink(link, link_ec)) {
                    std::filesystem::remove(link, link_ec);
                    if (!link_ec && !sub_process_mode)
                        header_print("OFLM", "Removed xclbins link: " + link.string());
                }
            }
        }
        return true;
    } catch (const std::exception& e) {
        header_print("ERROR", "Exception during model removal: " + std::string(e.what()));
        return false;
    }
}

/// \brief Check hash of model files
/// \param model_tag the model tag
/// \return true if all files are present and compatible, false otherwise
bool ModelDownloader::check_model(const std::string& model_tag, bool use_modelscope, bool sub_process_mode) {
    auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
    if (use_modelscope && uses_pinned_sources(model_info)) {
        try {
            resolve_file_source(
                model_info, model_info.at("files").at(0).get<std::string>(), true);
        }
        catch (const std::exception& error) {
            header_print("ERROR", error.what());
            return false;
        }
    }
    header_print("OFLM", "Checking model: " + new_model_tag + "...\n");

    ModelStatus status = is_model_downloaded(new_model_tag, sub_process_mode);
    switch (status) {
        case ModelStatus::Missing:
            header_print("OFLM", "Model not found: " + new_model_tag);
            header_print("OFLM", "Use `oflm pull " + new_model_tag + "` to download it.");
            return true;
        case ModelStatus::Incompatible:
            header_print("OFLM", "Model is incompatible with this version of OpenFlowLM: " + new_model_tag);
            header_print("OFLM", "Use `oflm pull " + new_model_tag + "` to re-download it.");
            return true;
        case ModelStatus::Outdated:
        case ModelStatus::Ready: {
            bool ok = verify_files(new_model_tag, use_modelscope, sub_process_mode);
            if (!ok)
                header_print("OFLM", "Model check completed with errors. Use `oflm pull " + new_model_tag + "` to re-download corrupted files.");
            else
                header_print("OFLM", "Model check completed successfully. All files are present and compatible.");
            return true;
        }
    }
    return true;
}

/// \brief Verify each model file's hash against HuggingFace metadata and
///        report corrupt files without deleting them before a replacement exists.
/// \param model_tag the model tag
/// \param sub_process_mode if true, suppress informational logging
/// \return true if all files passed verification, false otherwise
/// \note Two regimes. Pinned (file_sources) entries are strict: the bytes on
///       disk must be the pinned revision's bytes, so a size or hash mismatch
///       fails verification -- the pull that follows stages and verifies a
///       replacement before overwriting the file. Everything else stays
///       advisory: a hash mismatch only warns, because the registry oid can
///       legitimately disagree with what a repo serves (LFS vs git-blob
///       hashing, re-uploads, mirrors), and deleting on such a disagreement
///       just forces endless re-downloads.
bool ModelDownloader::verify_files(const std::string& model_tag, bool use_modelscope, bool sub_process_mode) {
    bool any_error = false;
    try {
        auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
        std::vector<std::string> model_files = model_info["files"];
        std::string model_path = supported_models.get_model_path(new_model_tag);
        const bool strict = uses_pinned_sources(model_info);

        // GET HF api/models
        // The live HuggingFace API path would have made the local manifest
        // unnecessary; it stays commented out, so model_info.json is the
        // record source. model_info_key lets one tag share another entry's
        // per-generation records.
        const nlohmann::json records = load_model_file_records(model_info, new_model_tag);
        if (records.empty()) {
            // No manifest for this model: it was installed from a local
            // directory and has no remote record to check its files against.
            // That is not a defect of the installation, so it is not an
            // error -- the files that are there are all there is to check.
            if (!sub_process_mode) {
                header_print("OFLM", "No manifest for " + new_model_tag +
                                     "; nothing to verify against.");
            }
            return true;
        }

        for (const auto& filename : model_files) {
            if (!sub_process_mode) {
                header_print("OFLM", "Checking file: " + filename + "...");
            }

            // Unpinned entries keep their historical behavior: a file the
            // manifest does not describe is skipped, not failed. (The pull
            // path reports that gap loudly via missing_from_manifest.) A
            // pinned entry cannot be verified without its record, so that
            // throws below instead of silently passing.
            if (!strict) {
                const auto present = std::find_if(
                    records.begin(), records.end(),
                    [&](const nlohmann::json& f) { return f.at("path") == filename; });
                if (present == records.end()) {
                    continue;
                }
            }
            const auto file = resolve_model_file(model_info, records, filename, use_modelscope);
            std::string local_path = get_model_file_path(model_path, filename);

            // If the file isn't present locally, there's nothing to verify or
            // remove; treat as an error so the caller knows a re-pull is needed.
            if (!file_exists(local_path)) {
                any_error = true;
                header_print("OFLM", "File missing: " + filename);
                continue;
            }

            std::error_code size_error;
            const auto local_size = std::filesystem::file_size(local_path, size_error);
            const bool size_matches = !size_error && local_size == file.size;
            const std::string local_oid =
                file.hash_algorithm == download_utils::HashAlgorithm::Sha256
                    ? download_utils::calculate_file_sha256(local_path)
                    : download_utils::calculate_git_blob_oid(local_path);
            const bool hash_matches = local_oid == file.hash;

            if (size_matches && hash_matches) {
                if (!sub_process_mode) {
                    header_print("OFLM", "Success!");
                }
                continue;
            }

            if (!strict) {
                if (!sub_process_mode) {
                    header_print("WARN", "Hash differs for " + filename + "; continuing");
                }
                continue;
            }

            if (!sub_process_mode) {
                header_print("OFLM", "Fail!");
                header_print("OFLM", "Keeping file until a verified replacement is downloaded: " + filename);
            }
            any_error = true;
        }
    }
    catch (const std::exception& e) {
        header_print("ERROR", "Exception during file verification: " + std::string(e.what()));
        any_error = true;
    }
    return !any_error;
}

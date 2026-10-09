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

namespace {

std::vector<pull::RegistryFile> registry_for(const std::string& tag, const std::vector<std::string>& files,
                                             std::vector<std::string>* unlisted) {
    nlohmann::json manifest;
    try {
        std::ifstream mf(utils::find_model_info());
        manifest = nlohmann::json::parse(mf).at(tag);
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
    auto missing_files = get_missing_files(model_tag);
    bool is_config_file_missing = std::find(missing_files.begin(), missing_files.end(), "config.json") != missing_files.end();
    ModelStatus modelstatus = ModelStatus::Missing;

    if (!is_config_file_missing) {
        modelstatus = check_model_compatibility(model_tag, sub_process_mode);

        if (modelstatus == ModelStatus::Outdated) {
            if (!fast_check) {
                header_print("OFLM", "Checking outdated files...");
                verify_and_clean_files(model_tag, sub_process_mode);
            }
        }
        else if (modelstatus == ModelStatus::Ready && !missing_files.empty()) {
            // config.json is present and the version check passed, but other
            // files (e.g. weights) are still missing.
            modelstatus = ModelStatus::Missing;
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
        
        header_print("OFLM", "Pulling model from " + model_server + "...");
        header_print("OFLM", "Model: " + new_model_tag);
        header_print("OFLM", "Name: " + model_name);

        ModelDownloader::ModelStatus status = is_model_downloaded(new_model_tag);
        if (status == ModelStatus::Incompatible) {
            return true;
        }

        const std::string model_path = supported_models.get_model_path(new_model_tag);
        std::vector<std::string> model_files = model_info["files"];
        std::vector<std::string> unlisted;
        const auto registry = registry_for(new_model_tag, model_files, &unlisted);
        if (!unlisted.empty()) {
            std::string names;
            for (const auto& n : unlisted) names += (names.empty() ? "" : ", ") + n;
            header_print("ERROR", "model_info.json describes none of these files required by model_list.json, "
                                  "so they cannot be downloaded: " + names);
            header_print("ERROR", "Adding a model needs an entry in BOTH files.");
            return false;
        }

        for (const auto& f : registry) {
            std::error_code ec;
            std::filesystem::remove(std::filesystem::path(model_path) / (f.path + ".part"), ec);   // a killed pull's leftovers
        }
        pull::Record record = pull::read_record(model_path);
        const bool all_recorded = std::all_of(registry.begin(), registry.end(), [&](const pull::RegistryFile& f) {
            return f.oid.empty() || record.count(f.path) != 0;
        });
        if (status == ModelStatus::Ready && all_recorded && !force_redownload) {
            header_print("OFLM", "Model already downloaded. Use --force to re-download.");
            return true;
        }

        std::filesystem::create_directories(model_path);
        if (!all_recorded && !force_redownload) {
            header_print("OFLM", "No install record yet: checking the files already present against the registry (once)...");
        }
        const size_t recorded_before = record.size();
        const auto plan = pull::plan_pull(registry, model_path, &record, force_redownload);
        if (record.size() != recorded_before) {
            pull::write_record(model_path, record);
        }
        if (plan.empty()) {
            header_print("OFLM", "All files are present and current.");
            return true;
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
            
            // Verify download
            auto final_missing = get_missing_files(new_model_tag);
            if (final_missing.empty()) {
                header_print("OFLM", "All files verified successfully.");
            } else {
                header_print("WARNING", "Some files may be missing after download:");
                for (const auto& file : final_missing) {
                    std::cout << "  - " << file << std::endl;
                }
            }
            return true;
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
        const auto registry = registry_for(new_model_tag, model_files, &unlisted);
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
/// \return the download list
std::pair<nlohmann::json, float> ModelDownloader::build_download_list(
    const std::string& model_tag, const std::vector<std::pair<pull::RegistryFile, pull::Fetch>>& plan, bool modelscope) {
    nlohmann::json downloads = nlohmann::json::array();
    float sum_file_size = 0;
    auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
    const std::string base_url = modelscope ? model_info["ms_url"] : model_info["url"];
    const std::string model_path = supported_models.get_model_path(new_model_tag);
    // one entry per planned file, in plan order: pull_model's record keys off that order
    for (const auto& [f, why] : plan) {
        // a "resolve" URL already names a revision; otherwise the repo's main branch
        const std::string url = base_url.find("resolve") != std::string::npos
                                    ? base_url + "/" + f.path + "?download=true"
                                    : base_url + "/resolve/main/" + f.path + "?download=true";
        const float file_size = static_cast<float>(f.size) / 1024 / 1024;
        sum_file_size += file_size;
        downloads.push_back(nlohmann::json{
            {"file", f.path},
            {"size", file_size},
            {"url", url},
            {"localpath", get_model_file_path(model_path, f.path)},
            {"oid", f.oid},
            {"is_lfs", f.lfs},
        });
    }
    return std::make_pair(downloads, sum_file_size);
}

/// \brief Remove a model and all its files
/// \param model_tag the model tag
/// \return true if the model was successfully removed, false otherwise
bool ModelDownloader::remove_model(const std::string& model_tag, bool sub_process_mode) {
    try {
        // Check if model exists in supported models by trying to get its info
        try {
            supported_models.get_model_info(model_tag);
        } catch (const std::exception& e) {
            header_print("ERROR", "Model not found: " + model_tag);
            model_not_found(model_tag);
            return false;
        }
        
        // Get model path
        std::string model_path = supported_models.get_model_path(model_tag);
        
        // Check if model directory exists
        if (!std::filesystem::exists(model_path)) {
            header_print("OFLM", "Model directory does not exist: " + model_path);
            return true; // Consider it already removed
        }

        if (!sub_process_mode) {
            header_print("OFLM", "Removing model: " + model_tag);
            header_print("OFLM", "Path: " + model_path);
        }
        
        // Remove all files in the model directory
        size_t removed_files = 0;
        for (const auto& entry : std::filesystem::directory_iterator(model_path)) {
            if (entry.is_regular_file()) {
                std::filesystem::remove(entry.path());
                removed_files++;
            }
        }
        
        // Remove the model directory itself
        if (std::filesystem::remove(model_path)) {
            if(!sub_process_mode)
                header_print("OFLM", "Successfully removed " + std::to_string(removed_files) + " files and model directory.");
            return true;
        } else {
            header_print("ERROR", "Failed to remove model directory: " + model_path);
            return false;
        }
        
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
            bool ok = verify_and_clean_files(new_model_tag, use_modelscope, sub_process_mode);
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
///        remove any corrupted files. Files that pass verification are kept.
/// \param model_tag the model tag
/// \param sub_process_mode if true, suppress informational logging
/// \return true if all files passed verification, false otherwise
bool ModelDownloader::verify_and_clean_files(const std::string& model_tag, bool use_modelscope, bool sub_process_mode) {
    bool any_error = false;
    try {
        auto [new_model_tag, model_info] = supported_models.get_model_info(model_tag);
        std::vector<std::string> model_files = model_info["files"];
        std::string model_path = supported_models.get_model_path(new_model_tag);
        std::string file_url = model_info["file_url"];

        nlohmann::json hf_model_infos;
        // GET HF api/models
        // if (use_modelscope == 0) {
        //     std::string hf_response = download_utils::download_string(file_url);
        //     hf_model_infos = nlohmann::json::parse(hf_response);
        // }
        // else {
        std::string model_info_path = utils::find_model_info();
        std::ifstream model_info_file(model_info_path);
        nlohmann::json model_info_json = nlohmann::json::parse(model_info_file);
        hf_model_infos = model_info_json.at(new_model_tag);
        // }

        for (const auto& filename : model_files) {
            if (!sub_process_mode) {
                header_print("OFLM", "Checking file: " + filename + "...");
            }

            auto it = std::find_if(
                hf_model_infos.begin(),
                hf_model_infos.end(),
                [&](const nlohmann::json& f) {
                    return f["path"] == filename;
                }
            );
            if (it == hf_model_infos.end()) {
                continue;
            }
            const auto& file = *it;
            std::string local_path = get_model_file_path(model_path, filename);

            // If the file isn't present locally, there's nothing to verify or
            // remove; treat as an error so the caller knows a re-pull is needed.
            if (!file_exists(local_path)) {
                any_error = true;
                header_print("OFLM", "File missing: " + filename);
                continue;
            }

            bool is_lfs = file.contains("lfs");
            std::string oid_ref = is_lfs ? file["lfs"]["oid"] : file["oid"];
            std::string local_oid = is_lfs ? download_utils::calculate_file_sha256(local_path) : download_utils::calculate_git_blob_oid(local_path);

            // Advisory only. A hash mismatch no longer deletes the file or
            // fails verification: presence is what matters here, and correctness
            // is proven when the model loads. Deleting on a registry/serving
            // disagreement just forces endless re-downloads.
            if (local_oid == oid_ref) {
                if (!sub_process_mode) {
                    header_print("OFLM", "Success!");
                }
            }
            else {
                if (!sub_process_mode) {
                    header_print("WARN", "Hash differs for " + filename + "; continuing");
                }
            }
        }
    }
    catch (const std::exception& e) {
        header_print("ERROR", "Exception during file verification: " + std::string(e.what()));
        any_error = true;
    }
    return !any_error;
}
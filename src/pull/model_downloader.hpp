/// \file model_downloader.hpp
/// \brief Model downloader class
/// \author OpenFlowLM Team
/// \date 2025-06-24
/// \version 0.9.24
/// \note This class is used to download models from the huggingface
#pragma once

#include "model_list.hpp"
#include "lm_config.hpp"
#include "download_model.hpp"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <vector>
#include <string>
#include <string_view>
#include <iostream>

/// \brief one file's pinned download origin: a base repo URL plus the
///        immutable 40-hex commit it must be resolved at
/// \note Ported from the FastFlowLM fork (MIT). A catalog entry names these
///       per file under "file_sources"; entries without them behave exactly
///       as before.
struct ModelFileSource {
    std::string url;
    std::string revision;
};

/// \brief where one file downloads from: its file_sources pin, or the
///        entry's url/ms_url fallback
/// \throws std::runtime_error when the file_sources table is malformed, when
///         a pin is requested for --modelscope (pinned HF sources cannot be
///         served from a mirror), or when the table names an unknown file
ModelFileSource resolve_file_source(
    const nlohmann::json& model_info,
    std::string_view filename,
    bool use_modelscope);

/// \brief whether this entry carries pinned per-file sources
/// \note The strict-integrity gate keys off the pinning itself -- not a
///       backend id -- so any GGUF or otherwise pinned model gets
///       revision-pinned URLs and verified downloads.
bool uses_pinned_sources(const nlohmann::json& model_info);

class ModelDownloader {
public:

    enum class ModelStatus { Ready, Missing, Outdated,  Incompatible};

    ModelDownloader(model_list& models);
    
    // Check if model is already downloaded.
    // When fast_check is true, only the local presence + version compatibility
    // are checked; no HuggingFace metadata is fetched and no per-file hash
    // verification / cleanup is performed. Use this for cheap status queries
    // such as `oflm list`.
    ModelStatus is_model_downloaded(const std::string& model_tag, bool sub_process_mode=0, bool fast_check=false);
    
    // Download model files if not present
    bool pull_model(const std::string& model_tag, bool use_modelscope = false, bool force_redownload = false);
    
    // Get list of missing files for a model
    std::vector<std::string> get_missing_files(const std::string& model_tag);
    
    // Get list of present files for a model
    std::vector<std::string> get_present_files(const std::string& model_tag);
    
    // Remove a model and all its files
    bool remove_model(const std::string& model_tag, bool sub_process_mode=0);
    
    bool check_model(const std::string& model_tag, bool use_modelscope=0, bool sub_process_mode=0);

    // Get download progress callback
    std::function<void(size_t, size_t)> get_progress_callback();

    void model_not_found(const std::string& model_tag);

private:
    model_list& supported_models;
    download_utils::CurlInitializer curl_init;
    
    // Check if a specific file exists
    bool file_exists(const std::string& file_path);
    
    // Get the full path for a model file
    std::string get_model_file_path(const std::string& model_path, const std::string& filename);
    
    // Build download URLs for model files.
    // When force_redownload is true, files already on disk are re-queued
    // instead of skipped (previously --force with a complete tree downloaded
    // nothing and still reported success).
    std::pair<nlohmann::json, float> build_download_list(
        const std::string& model_tag, bool modelscope=0, bool force_redownload=false);

    // bool check_model_compatibility(const std::string& model_tag);
    ModelStatus check_model_compatibility(const std::string& model_tag, bool sub_process_mode=0);

    // Verify per-file integrity against HuggingFace metadata and remove any
    // corrupted files. Returns true if all files passed verification.
    bool verify_and_clean_files(const std::string& model_tag, bool use_modelscope=0, bool sub_process_mode=0);
}; 
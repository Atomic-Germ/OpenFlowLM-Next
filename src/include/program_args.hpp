/// \file options.hpp
/// \brief options file for the FastFlowLM project
/// \author FastFlowLM Team
/// \date 2026-02-24
/// \version 0.9.26
/// \note This file contains a struct for passing all user arguments from command line
/// \note This is to avoid keep add arguments to runner and serve
#pragma once

#include <string>

struct program_args_t {
    // common commands
    std::string command = "version";
    std::string model_tag = "model-faker";
    std::string power_mode = "performance";
    bool preemption = false;
    bool asr = false;
    bool embed = false;
    // Which embedding model --embed loads. Empty means the historical
    // default, embed-gemma:300m, so an existing command line is
    // unchanged. See all_embedding_model.hpp for the registry.
    std::string embedding_model = "";
    bool json_output = false;
    int ctx_length = -1; // let model decide
    int prefill_chunk_len = -1; // let model decide

    // handling input file
    std::string input_file_name = "";
    int iterations = 2;

    // specific commands
    int img_pre_resize = 3;

    // for list command
    std::string list_filter = "all";

    // for pull command
    bool force_redownload = false;
    
    // for download related command
    bool modelscope = false;

    // for add command (flm add: install a pre-converted Q4NX model)
    std::string add_tag = "";            // --tag (default: derived from repo name)
    std::string add_family = "";         // --family (details.family for engine dispatch)
    std::string add_config = "";         // --config (user model_list.json to update)
    std::string add_system_list = "";    // --system-list (official model_list.json for defaults)
    std::string add_models_root = "";    // --models-root (models directory)
    std::string add_xclbin_dir = "";     // --xclbin-dir (user xclbins directory)
    std::string add_xclbin_from = "";    // --xclbin-from (official model dir to link open_kernels from)
    bool add_no_xclbin = false;          // --no-xclbin
    bool add_no_verify = false;          // --no-verify
    bool add_dry_run = false;            // --dry-run

    // for serve command
    std::string host = "127.0.0.1";
    size_t max_socket_connections = 10;
    size_t max_npu_queue = 10;
    int port = -1; // default port
    bool cors = false;
    bool sub_process_mode = false;
    
    program_args_t() {}
};
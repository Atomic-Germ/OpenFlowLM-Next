/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file rest_handler.hpp
 * \brief RestHandler class and related declarations
 * \author OpenFlowLM Team
 * \date 2025-06-24
 *  \version 0.9.24
 */
#pragma once

#include "AutoModel/all_models.hpp"
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
#include "whisper/modeling_whisper.hpp"
#include "AutoEmbeddingModel/all_embedding_model.hpp"
#endif
#include "model_list.hpp"
#include "program_args.hpp"


#include "model_downloader.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
#include <functional>
#include "prompt_cache.hpp"

using json = nlohmann::ordered_json;

// Forward declaration
struct CancellationToken;
class Tokenizer;
namespace open_diffusion { class Engine; }

///@brief One file part of a /v1/images/edits form (`image`, `image[]` or `mask`)
struct ImageUpload {
    std::string field;
    std::string filename;
    std::string content_type;
    size_t bytes = 0;
    std::string data;            // the file's bytes (an edit's reference is decoded from them)
};

///@brief Stream callback type for sending streaming responses
using StreamResponseCallback = std::function<void(const json&, bool)>; // data, is_final

#include "server/openai_compat.hpp"
class RestHandler {
public:
    RestHandler(model_list& models, ModelDownloader& downloader, program_args_t& args);
    ~RestHandler();

    void handle_show(const json& request,
        std::function<void(const json&)> send_response,
        StreamResponseCallback send_streaming_response);

    void handle_generate(const json& request, 
                        std::function<void(const json&)> send_response,
                        StreamResponseCallback send_streaming_response,
                        std::shared_ptr<CancellationToken> cancellation_token = nullptr);

    void handle_chat(const json& request,
                    std::function<void(const json&)> send_response, 
                    StreamResponseCallback send_streaming_response,
                    std::shared_ptr<CancellationToken> cancellation_token = nullptr);
    

    void handle_embeddings(const json& request,
                          std::function<void(const json&)> send_response,
                          StreamResponseCallback send_streaming_response);
    

    void handle_models(const json& request,
                      std::function<void(const json&)> send_response,
                      StreamResponseCallback send_streaming_response);
    
    void handle_models_openai(const json& request,
                            std::function<void(const json&)> send_response,
                            StreamResponseCallback send_streaming_response);

    void handle_ps(const json& request,
                    std::function<void(const json&)> send_response,
                    StreamResponseCallback send_streaming_response);
    
    void handle_version(const json& request,
                       std::function<void(const json&)> send_response,
                       StreamResponseCallback send_streaming_response);
    
    // Placeholder handlers for unimplemented endpoints
    void handle_pull(const json& request,
                    std::function<void(const json&)> send_response,
                    StreamResponseCallback send_streaming_response);
    
    void handle_push(const json& request,
                    std::function<void(const json&)> send_response,
                    StreamResponseCallback send_streaming_response);
    
    void handle_delete(const json& request,
                      std::function<void(const json&)> send_response,
                      StreamResponseCallback send_streaming_response);
    
    void handle_copy(const json& request,
                    std::function<void(const json&)> send_response,
                    StreamResponseCallback send_streaming_response);
    
    void handle_create(const json& request,
                      std::function<void(const json&)> send_response,
                      StreamResponseCallback send_streaming_response);

    void handle_openai_chat_completion(const json& request,
                                      std::function<void(const json&)> send_response,
                                      StreamResponseCallback send_streaming_response,
                                      std::shared_ptr<CancellationToken> cancellation_token = nullptr);
    void handle_openai_audio_transcriptions(const json& request,
                                      std::function<void(const json&)> send_response,
                                      StreamResponseCallback send_streaming_response,
                                      std::shared_ptr<CancellationToken> cancellation_token = nullptr);
    void handle_openai_completion(const json& request,
        std::function<void(const json&)> send_response,
        StreamResponseCallback send_streaming_response,
        std::shared_ptr<CancellationToken> cancellation_token = nullptr);
    // specs/server-api: SERVER-IMAGES-*
    void handle_openai_images_generations(const json& request,
        std::function<void(const json&)> send_response,
        StreamResponseCallback send_streaming_response,
        std::shared_ptr<CancellationToken> cancellation_token = nullptr);
    /// \param fields the form's text fields (openai_compat::images_form_json)
    /// \param uploads its file parts
    void handle_openai_images_edits(const json& fields, const std::vector<ImageUpload>& uploads,
                                    std::shared_ptr<CancellationToken> cancellation_token,
        std::function<void(const json&)> send_response,
        StreamResponseCallback send_streaming_response);

private:
    using ModelLoad = openai_compat::ModelLoad;
    /// \param model_field_present the request carried a "model" key. Without it an
    ///        omitted field and an explicit "" are the same string -- see
    ///        openai_compat::preflight().
    ModelLoad ensure_model_loaded(const std::string& model_tag, bool model_field_present = false);
    void ensure_asr_model_loaded(const std::string& model_tag);
    void ensure_embed_model_loaded(const std::string& model_tag);
    /// The tag an image request names (or --imagemodel's, when it names none), resolved
    /// and checked BEFORE anything is unloaded. Empty json and *tag set, or the 400.
    json resolve_image_model(const json& request, std::string* tag);
    /// Load the image engine for `tag` unless it is loaded. Without --imagegen 1 this swaps
    /// the chat model off the NPU first (SERVER-IMAGES-RESIDENCY). Empty on success,
    /// else why it failed.
    std::string ensure_image_engine_loaded(const std::string& tag);
    /// Take the image engine off the NPU before a chat model loads, unless it is resident.
    void release_image_engine_for_chat();
    void configure_chat_engine_parameters(const json& options, const json& request);
    json build_nstream_response(std::string response_text,
                                stop_reason_t stop_reason = EOT_DETECTED);


    std::unique_ptr<AutoModel> auto_chat_engine;
    // The open diffusion engine and its prompt tokenizer.
    std::unique_ptr<open_diffusion::Engine> image_engine;
    std::unique_ptr<Tokenizer> image_tokenizer;
    std::string image_engine_tag;       // what image_engine was loaded for
    bool image_resident;                // --imagegen 1: loaded at startup, never swapped out
    std::string image_model_tag;        // --imagemodel: the model a request naming none gets
#ifndef FASTFLOWLM_LINUX_LIMITED_MODELS
    std::unique_ptr<Whisper> whisper_engine;
    std::unique_ptr<AutoEmbeddingModel> auto_embedding_engine;
#endif
    oflm_rt::device npu_device_inst;
    model_list& supported_models;
    ModelDownloader& downloader;
    std::string current_model_tag;
    std::string default_model_tag;
    bool modelscope;
    bool asr;
    std::string asr_model_tag;
    bool embed;
    // Which embedding model --embed loads, from --embeddingmodel.
    // Empty means embed-gemma:300m, so an existing command line keeps
    // its behaviour exactly.
    std::string embedding_model_tag;
    int prefill_chunk_len;
    int generate_context_id;
    int chat_context_id;
    int ctx_length;
    int img_pre_resize;
    std::string last_question;
    bool preemption;
    PromptCache prompt_cache;
};
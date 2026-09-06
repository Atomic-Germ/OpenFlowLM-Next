/// \file modeling_granite.cpp
/// \brief IBM Granite (dense) family. See modeling_granite.hpp.

#include "AutoModel/modeling_granite.hpp"

/************              Granite family            **************/
Granite::Granite(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Granite") {}

void Granite::load_model(std::string model_path, json model_info, int default_context_length,
                         bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);

    // The engine: the open one (open_qwen36/, the dense recipe's kernel set
    // under xclbins/<model>/open_kernels). Unlike llama3 / qwen3 / gemma3 there
    // is NO closed granite engine to fall back to -- llama_npu whitelists
    // hidden_size to {2048, 3072, 4096} and refuses Granite's 2560 outright --
    // so a missing kernel set is a named refusal, not a quiet substitution.
    // FLM_GRANITE_ENGINE=open is accepted (and is the only value that means
    // anything) so the variable behaves like the other families'.
#ifdef FLM_USE_OPEN_QWEN36
    const std::string kernels = open_qwen36::Engine::find_kernels(*this->lm_config);
    const char* sel = std::getenv("FLM_GRANITE_ENGINE");
    if (sel && std::string(sel) == "closed")
        throw std::runtime_error("FLM_GRANITE_ENGINE=closed: not implemented -- Granite has no closed engine "
                                 "(llama_npu refuses hidden_size 2560); build the open kernels instead");
    if (kernels.empty())
        throw std::runtime_error("no open kernels were found for " + this->lm_config->model_name +
                                 ". Granite runs on the open kernels only: build them with "
                                 "open_kernels/export_qwen36_kernels.py --model-dir <model dir>, or point "
                                 "FLM_OPEN_KERNELS_DIR at a built set.");
    header_print("FLM", "Granite on the open kernels (" + kernels + ")");
    auto eng = std::make_unique<open_qwen36::Engine>(*this->lm_config, this->npu_device_inst, this->MAX_L);
    eng->load_open_weights();
    this->lm_engine = std::move(eng);
#else
    throw std::runtime_error("Granite needs the open engine, which this build does not have "
                             "(FLM_USE_OPEN_QWEN36 is off -- it requires the XRT backend, not HRX)");
#endif

    this->lm_engine->clear_context();
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    // granite-4.2 is a reasoning model; its own generation_config ships
    // temperature 1.0 / top_p 0.95, and a repetition guard keeps long chains of
    // thought from looping.
    sampler_config config;
    config.top_k = 40;
    config.top_p = 0.95;
    config.min_p = 0.0;
    config.temperature = 1.0;
    config.rep_penalty = 1.05;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void Granite::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Granite::apply_chat_template(nlohmann::ordered_json& messages,
                                         nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    return this->chat_tmpl->apply(inputs);
}

bool Granite::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input,
                     std::function<bool()> is_cancelled) {
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    if (!input.messages.empty()) {
        templated_text = this->apply_chat_template(input.messages);
    }
    else if (!input.prompt.empty()) {
        nlohmann::ordered_json messages;
        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens = this->tokenizer->encode(templated_text);
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    return this->_shared_insert(meta_info, tokens, is_cancelled);
}

std::string Granite::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os,
                              std::function<bool()> is_cancelled) {
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string Granite::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input,
                                          int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->_shared_generate(meta_info, length_limit, os);
}

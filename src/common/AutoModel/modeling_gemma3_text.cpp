/// \file gemma3.cpp
/// \brief gemma3 class
/// \author OpenFlowLM Team
/// \date 2025-09-03
/// \version 0.9.24
/// \note This is a source file for the gemma3 class

#include "AutoModel/modeling_gemma3_text.hpp"
#include "config/settings_resolver.hpp"


/************              Gemma3_Text_Only family            **************/
Gemma3_Text_Only::Gemma3_Text_Only(oflm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "Gemma3_Text_Only") {}

void Gemma3_Text_Only::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {

    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption,
                             oflm::config::g_resolved_settings ? *oflm::config::g_resolved_settings : nlohmann::json::object());

    // The engine: the open kernels when a set is installed for this model;
    // the closed DLL only via OFLM_GEMMA_ENGINE=closed (AutoModel::_shared_select_open_engine).
    auto open_engine = this->_shared_select_open_engine("OFLM_GEMMA_ENGINE", "Gemma 3 (text)");
    if (open_engine) {
        this->lm_engine = std::move(open_engine);
    }
    else {
        // The closed Gemma 3 (text) engine is removed from this build entirely
        // (the open kernels are the only engine). This branch fires only when
        // OFLM_OFLM_GEMMA_ENGINE was set to 'closed', and names the one remaining remedy.
        throw std::runtime_error(
            "OFLM_GEMMA_ENGINE=closed requested, but this build ships no closed Gemma 3 (text) engine; the open kernels are the only engine -- install them with `oflm add`");
    }
    this->lm_engine->clear_context();
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    sampler_config config;
    config.top_k = 40;
    config.top_p = 0.95;
    config.min_p = 0.1;
    config.temperature = 0.8;
    config.rep_penalty = 1.05;
    config.freq_penalty = 1.05;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void Gemma3_Text_Only::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Gemma3_Text_Only::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    if (!tools.empty())
        inputs.tools = tools;
    return this->_shared_apply_template(inputs);
}

bool Gemma3_Text_Only::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // preprocess
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        templated_text = this->apply_chat_template(input.messages, input.tools);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;

        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens = this->tokenizer->encode(templated_text);
    
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());
    // hardware

    return this->_shared_insert(meta_info, tokens, is_cancelled);
}

std::string Gemma3_Text_Only::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string Gemma3_Text_Only::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    if (!this->insert(meta_info, input, is_cancelled)) {
        return "";
    }
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}
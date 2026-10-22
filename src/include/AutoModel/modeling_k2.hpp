// IFM K2-Horizon (dense), open kernels only; with uno.q4nx beside the weights, greedy requests decode by Uno (OPEN-UNO-DECODE).
#pragma once
#include "AutoModel/automodel.hpp"
#include "AutoModel/k2_chat.hpp"
#ifdef OFLM_USE_OPEN_QWEN36
#include "open_qwen36/engine.hpp"
#endif

/************              K2 family            **************/
class K2 : public AutoModel {
private:
    k2_chat::StreamParser parser_;
    k2_chat::ParamTypes tool_types_;
    int tool_seq_ = 0;

    void setup_tokenizer(std::string model_path);
    StreamResult stream_result(const k2_chat::Event& ev);

public:
    K2(oflm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;
    bool configure_parameter(std::string parameter_name, const std::any& value) override;
    NonStreamResult parse_nstream_content(const std::string response_text) override;
    StreamResult parse_stream_content(const std::string content) override;
    StreamResult parse_stream_content_final(const std::string content) override;
};

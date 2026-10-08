/// \file modeling_k2.hpp
/// \brief IFM K2-Horizon (dense) family, and K2-Horizon-7B-Uno's lossless speedup.
/// \note Open kernels only, like Granite. A model directory that carries uno.q4nx beside
///       model.q4nx (K2-Horizon-7B-Uno's diffusion LoRA) decodes greedy requests by Uno's
///       two-pass cycle (OPEN-UNO-DECODE): the same tokens as plain greedy decode, several a
///       cycle. Sampled requests, and greedy ones with a repetition penalty, decode as usual.

#pragma once
#include "AutoModel/automodel.hpp"
#ifdef OFLM_USE_OPEN_QWEN36
#include "open_qwen36/engine.hpp"
#endif

/************              K2 family            **************/
class K2 : public AutoModel {
private:
    void setup_tokenizer(std::string model_path);
    /// The greedy request Uno can take: top_k 1 and no penalty that reorders the logits.
    bool uno_applies() const;
    std::string generate_uno(chat_meta_info_t& meta_info, int length_limit, std::ostream& os,
                             std::function<bool()> is_cancelled);

public:
    K2(oflm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;
};

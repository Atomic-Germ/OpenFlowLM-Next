#include "AutoModel/modeling_k2.hpp"

/************              K2 family            **************/
K2::K2(oflm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "K2") {}

void K2::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);
#ifdef OFLM_USE_OPEN_QWEN36
    const std::string kernels = open_qwen36::Engine::find_kernels(*this->lm_config);
    if (kernels.empty())
        throw std::runtime_error("no open kernels were found for " + this->lm_config->model_name +
                                 ". K2-Horizon runs on the open kernels only: build them with "
                                 "open_kernels/export_qwen36_kernels.py --model-dir <model dir>, or point "
                                 "OFLM_OPEN_KERNELS_DIR at a built set.");
    auto eng = std::make_unique<open_qwen36::Engine>(*this->lm_config, this->npu_device_inst, this->MAX_L);
    eng->load_open_weights();
    header_print("OFLM", "K2-Horizon on the open kernels (" + kernels + ")" +
                             (eng->uno_ok() ? "; greedy requests decode by Uno's draft pass" : ""));
    this->lm_engine = std::move(eng);
#else
    throw std::runtime_error("K2-Horizon needs the open engine, which this build does not have "
                             "(OFLM_USE_OPEN_QWEN36 is off -- it requires the XRT backend, not HRX)");
#endif

    this->lm_engine->clear_context();
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    // greedy is where Uno is lossless, and K2's generation_config sets no sampling at all
    sampler_config config;
    config.top_k = 1;
    config.top_p = 1.0;
    config.min_p = 0.0;
    config.temperature = 1.0;
    config.rep_penalty = 1.0;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void K2::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string K2::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = k2_chat::prepare_messages(messages);
    inputs.extra_context = this->extra_context;
    if (!tools.empty())
        inputs.tools = tools;
    this->tool_types_ = k2_chat::param_types(tools);
    minja::chat_template_options opts;
    // minja's tool-call probe sends no thinking field, so it wrongly marks K2 for the polyfills
    opts.apply_polyfills = false;
    return this->_shared_apply_template(inputs, opts);
}

bool K2::configure_parameter(std::string parameter_name, const std::any& value) {
    if (parameter_name == "reasoning_effort") {
        const std::string* effort = std::any_cast<std::string>(&value);
        if (!effort) return false;
        if (*effort == "high" || *effort == "medium" || *effort == "low")
            this->extra_context["reasoning_effort"] = *effort;
        else
            header_print("WARNING", "K2-Horizon reasoning_effort must be 'low', 'medium' or 'high'");
        return true;
    }
    return AutoModel::configure_parameter(parameter_name, value);
}

StreamResult K2::stream_result(const k2_chat::Event& ev) {
    StreamResult r;
    switch (ev.kind) {
    case k2_chat::Event::Wait: r.type = StreamEventType::WAITING; break;
    case k2_chat::Event::Reasoning: r.type = StreamEventType::REASONING; r.content = ev.text; break;
    case k2_chat::Event::Content: r.type = StreamEventType::CONTENT; r.content = ev.text; break;
    case k2_chat::Event::Tool:
        r.type = StreamEventType::TOOL_DONE;
        r.tool_id = "call_" + std::to_string(std::time(nullptr)) + "_" + std::to_string(++this->tool_seq_);
        r.tool_name = ev.name;
        r.tool_args_str = ev.args.dump();
        break;
    }
    return r;
}

StreamResult K2::parse_stream_content(const std::string content) {
    return this->stream_result(this->parser_.feed(content, false, this->tool_types_));
}

StreamResult K2::parse_stream_content_final(const std::string content) {
    return this->stream_result(this->parser_.feed(content, true, this->tool_types_));
}

NonStreamResult K2::parse_nstream_content(const std::string response_text) {
    const k2_chat::Response r = k2_chat::parse_response(response_text, this->tool_types_);
    NonStreamResult result;
    result.content = r.content;
    result.reasoning_content = r.reasoning;
    for (const auto& [name, args] : r.calls) result.tool_calls_list.emplace_back(name, args.dump());
    if (!result.tool_calls_list.empty()) {
        result.tool_name = result.tool_calls_list[0].first;
        result.tool_args = result.tool_calls_list[0].second;
    }
    return result;
}

bool K2::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    if (!input.messages.empty()) {
        templated_text = this->apply_chat_template(input.messages, input.tools);
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

bool K2::uno_applies() const {
#ifdef OFLM_USE_OPEN_QWEN36
    auto* eng = dynamic_cast<open_qwen36::Engine*>(this->lm_engine.get());
    if (!eng || !eng->uno_ok() || !this->sampler) return false;
    const Sampler& s = *this->sampler;
    const bool penalties = s.repeat_last_n != 0 && (s.rep_penalty != 1.0f || s.freq_penalty != 0.0f || s.pre_penalty != 0.0f);
    return s.top_k == 1 && !penalties;
#else
    return false;
#endif
}

std::string K2::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os,
                         std::function<bool()> is_cancelled) {
    // the generation prompt always opens a think block, so every reply starts as reasoning
    this->parser_.reset();
    this->tool_seq_ = 0;
    return this->uno_applies() ? this->generate_uno(meta_info, length_limit, os, is_cancelled)
                               : this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string K2::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input,
                                     int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    if (!this->insert(meta_info, input, is_cancelled)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os, is_cancelled);
}

// _shared_generate's contract a cycle at a time; a cycle past the stop is cut back to where plain decode would stop.
std::string K2::generate_uno(chat_meta_info_t& meta_info, int length_limit, std::ostream& os,
                             std::function<bool()> is_cancelled) {
#ifdef OFLM_USE_OPEN_QWEN36
    auto* eng = dynamic_cast<open_qwen36::Engine*>(this->lm_engine.get());
    std::string result;
    assert(this->last_token != -1);
    stop_reason_t reason = EOT_DETECTED;
    int seed = this->last_token;
    this->token_history.push_back(seed);
    if (this->is_normal_token(seed)) {
        std::string token_str = this->tokenizer->run_time_decoder(seed);
        result += token_str;
        os << token_str << std::flush;
    }
    if (this->is_eos(seed)) return result;
    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();
    std::vector<int> committed;
    bool done = false;
    // a cycle writes up to L + 1 positions past the seed; stop cycling short of the capacity
    while (!done && this->total_tokens + 8 < this->MAX_L) {
        if (is_cancelled()) {
            reason = CANCEL_DETECTED;
            buffer_.clear();
            current_mode_ = StreamEventType::CONTENT;
            tool_name_.clear();
            is_in_tool_block_ = false;
            break;
        }
        committed.clear();
        this->profiler_list[DECODING_TIME].start();
        const int p = eng->uno_cycle(seed, committed);
        this->profiler_list[DECODING_TIME].stop(static_cast<int>(committed.size()));
        for (size_t i = 0; i < committed.size(); ++i) {
            const int t = committed[i];
            this->total_tokens++;
            this->profiler_list[TKOEN_DECODE_TIME].start();
            if (this->is_normal_token(t)) {
                std::string token_str = this->tokenizer->run_time_decoder(t);
                os << token_str << std::flush;
                result += token_str;
            }
            this->profiler_list[TKOEN_DECODE_TIME].stop(1);
            this->token_history.push_back(t);
            meta_info.generated_tokens++;
            const bool last = i + 1 == committed.size();
            if (this->is_eos(t)) {
                // token i sits at p + 1 + i; all but the cycle's last are already cached
                if (this->forward_on_eos) {
                    if (last) this->lm_engine->forward(t);
                    else eng->uno_seek(p + 2 + static_cast<int>(i));
                } else {
                    eng->uno_seek(p + 1 + static_cast<int>(i));
                }
                done = true;
                break;
            }
            if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)) {
                reason = MAX_LENGTH_REACHED;
                eng->uno_seek(p + 1 + static_cast<int>(i));     // as plain decode: the last emitted token uncached
                done = true;
                break;
            }
            if (last) seed = t;
        }
    }
    meta_info.decoding_duration = (uint64_t)(time_utils::cast_to_us(this->profiler_list[DECODING_TIME].get_total_time()).first) * 1e3;
    meta_info.stop_reason = reason;
    if (!done && reason != CANCEL_DETECTED) header_print("WARNING", "Max length reached, stopping generation...");
    std::cout << std::endl;
    header_print("OFLM", "Model RAW Output: \n" + result);
    return result;
#else
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
#endif
}

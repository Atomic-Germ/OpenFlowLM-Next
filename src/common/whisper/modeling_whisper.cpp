/// \file whisper_npu.hpp
/// \brief whisper_npu class
/// \author OpenFlowLM Team
/// \date 2025-10-17
/// \version 0.9.24
/// \note This is a source file for the modeling_whisper class
#include "whisper/modeling_whisper.hpp"
#include <cstdlib>
#include <chrono>
#include <cstdio>

// task 0180 Part B: end-to-end request profiling, timing only (no arithmetic
// changed). Host wall clock throughout -- these are stage buckets inside one
// /v1/audio/transcriptions request, printed to the server's own stdout log
// once per request, not an NPU performance claim (rule 1).
namespace {
double now_s_whisper() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
}  // namespace


Whisper::Whisper(oflm_rt::device* npu_device_inst){
    this->device = npu_device_inst;
    
    time_stamp = 0;
    audio_buffer.clear();
    fft_400 = std::make_unique<FFT400>();
    mel_feature = buffer<bf16>(128 * 3000);
    this->profiler_list.resize(PROFILER_TYPE_NUM);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i] = profiler();
        this->profiler_list[i].reset();
    }
    this->last_prefill_time = { 0, "us" };
}

void Whisper::load_model(std::string model_path, nlohmann::ordered_json model_info, bool enable_preemption) {
    header_print("OFLM", "Loading model: " << model_path);
    this->enable_preemption = enable_preemption;
    this->model_path = model_path;

    this->lm_config = std::make_unique<Whisper_Config>();
    this->lm_config->from_pretrained(this->model_path);

    this->engine.reset();
    this->engine = make_whisper_engine(this->model_path, *this->lm_config, this->device, enable_preemption);
    header_print("OFLM", "Whisper engine: " << this->engine->describe());
    this->setup_tokenizer(model_path);
    
    this->sampler.reset();

    sampler_config s_config;
    s_config.top_k = 1;
    s_config.top_p = 0.95;
    s_config.min_p = 0.1;
    s_config.temperature = 0.4;

    this->sampler = std::make_unique<Sampler>(this->lm_config->get("vocab_size"), s_config);
}

void Whisper::setup_tokenizer(std::string model_path) {
    // load tokenizer configurations
    #ifdef _WIN32
    std::string tokenizer_config_path = model_path + "\\tokenizer_config.json";
    #else
    std::string tokenizer_config_path = model_path + "/tokenizer_config.json";
    #endif
    std::ifstream fs_config(tokenizer_config_path, std::ios::in | std::ios::binary);
    if (fs_config.fail()) {
        std::cerr << "Cannot open " << tokenizer_config_path << std::endl;
        exit(1);
    }
    std::string data_config;
    fs_config.seekg(0, std::ios::end);
    size_t size_config = static_cast<size_t>(fs_config.tellg());
    fs_config.seekg(0, std::ios::beg);
    data_config.resize(size_config);
    fs_config.read(data_config.data(), size_config);
    fs_config.close();
    auto tokenizer_config = nlohmann::json::parse(data_config);
    // check if bos_token is null
    if (tokenizer_config["bos_token"].is_null()) {
        this->has_bos_token = false;
    }
    else {
        this->has_bos_token = true;
    }

    if (this->has_bos_token) {
        this->bos_token_id = tokenizer_config["bos_token_id"].get<int>();
    }
    else {
        this->bos_token_id = -1;
    }
    this->eos_token = tokenizer_config["eos_token"].get<std::string>();
    for (auto& token : tokenizer_config["eos_token_id"]) {
        this->eos_token_ids.push_back(token.get<int>());
    }
 
    this->tokenizer = std::make_unique<Tokenizer>(this->model_path);
    this->_build_time_map();
}

bool Whisper::load_audio(std::string& audio_path) {    
    this->audio_buffer = this->_load_audio(audio_path);
    header_print("info", "Length of audio: " + std::to_string(_S2T_(this->audio_buffer.size())) + " seconds");
    return true;
}

bool Whisper::load_audio(std::vector<uint8_t>& audio_data) {
    this->audio_buffer = this->_load_audio(audio_data);
    header_print("info", "Length of audio: " + std::to_string(_S2T_(this->audio_buffer.size())) + " seconds");
    return true;
}

std::string Whisper::_decode_protocol() {
    // Read once per process via a function-local static -- std::getenv is cheap but a
    // typo in the env var must be caught once, loudly, not read differently by two
    // callers racing a mutable env var mid-run.
    static const std::string protocol = [] {
        const char* env = std::getenv("OFLM_WHISPER_PROTOCOL");
        if (env == nullptr || std::string(env).empty()) {
            return std::string("legacy");
        }
        std::string v(env);
        if (v == "legacy" || v == "hf") {
            return v;
        }
        throw std::runtime_error("OFLM_WHISPER_PROTOCOL=" + v +
                                  ": unknown value, expected 'legacy' or 'hf' (unset defaults to 'legacy')");
    }();
    return protocol;
}

const whisper_hf::GenerationConfig& Whisper::_hf_gen_config() {
    if (!this->hf_gen_config_loaded_) {
        this->hf_gen_config_ = whisper_hf::GenerationConfig::load(this->model_path);
        this->hf_gen_config_loaded_ = true;
    }
    return this->hf_gen_config_;
}

int Whisper::_real_vocab_size() {
    if (this->real_vocab_size_ == 0) {
        std::ifstream f(this->model_path + "/config.json", std::ios::in | std::ios::binary);
        if (f.fail()) {
            throw std::runtime_error("cannot open " + this->model_path + "/config.json for vocab_size");
        }
        std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        auto j = nlohmann::json::parse(text);
        if (!j.contains("vocab_size") || j["vocab_size"].is_null()) {
            throw std::runtime_error(this->model_path + "/config.json has no 'vocab_size'");
        }
        this->real_vocab_size_ = j["vocab_size"].get<int>();
    }
    return this->real_vocab_size_;
}

std::pair<std::string, std::string> Whisper::generate(whisper_task_type_t task, bool enable_time_stamp, bool return_time_stamp, std::ostream& os) {
    const std::string protocol = this->_decode_protocol();
    if (protocol == "hf") {
        return this->_generate_hf(task, enable_time_stamp, return_time_stamp, os);
    }
    return this->_generate_legacy(task, enable_time_stamp, return_time_stamp, os);
}

std::pair<std::string, std::string> Whisper::_generate_legacy(whisper_task_type_t task, bool enable_time_stamp, bool return_time_stamp, std::ostream& os) {
    int length = this->audio_buffer.size();
    int current_idx = 0;
    int overlapping_samples = 5 * FS; // 
    int l_this_round = std::min(WINDOW_SAMPLES, length);
    int last_time_stamp = this->token_time_map_offset;
    int last_idx;
    bool last_chunk = false;
    std::string result;
    std::string language_detected;
    if ((!enable_time_stamp) && (return_time_stamp)){
        header_print("Error", "Return_time_stamp is true but timestamp is not enabled!");
        return std::make_pair("", "");
    }
    while (current_idx < length){
        bool allow_force_time_stamp = true;
        // std::cout << "Chunk " << _S2T_(current_idx) << "s to " << _S2T_(current_idx + l_this_round) << "s" << std::endl;
        float time_offset = _S2T_(current_idx);
        
        if (l_this_round == 0){
            break;
        }

        std::vector<float> audio_chunk(l_this_round);

        audio_chunk.insert(audio_chunk.begin(), this->audio_buffer.data() + current_idx, this->audio_buffer.data() + current_idx + l_this_round);
        
        _preprocess_audio(mel_feature, audio_chunk);
      
        // run whisper encoder
        this->engine->encode_audio(mel_feature); // encoded and pass kv-cache to decoder

        // decoder loop
        this->engine->clear_context();
        this->sampler->reset_penalties();

        last_idx = start_of_transcript; // the first token is fixed
        buffer<bf16> logits = this->engine->decode_audio(last_idx);
        last_idx = this->_sample_in_language(logits);
      
        // std::cout << "Language detected: " << this->tokenizer->run_time_decoder(last_idx) << "(" << langmap::to_language_name(this->tokenizer->run_time_decoder(last_idx)) << ")" << std::endl;
        language_detected = this->tokenizer->run_time_decoder(last_idx);

        if (task == e_translate) {
            //header_print("info", "translate is not supported! Do transcribe instead!");
            //task = e_transcribe;
            //buffer<bf16> logits = this->engine->decode_audio(50259); // en
            //logits = this->engine->decode_audio(translate_token);
            //last_idx = this->_sample_in_time_stamp(logits);
        }
        else if (task == e_transcribe) {
            buffer<bf16> logits = this->engine->decode_audio(transcribe_token);
            last_idx = this->_sample_in_time_stamp(logits);
        }
        else {
            header_print("Error", "Non-recongnized task!");
        }

        if (enable_time_stamp){
            if (return_time_stamp){
                std::string time_stamp = this->tokenizer->run_time_decoder(last_idx);
                std::string offset_time_stamp = this->_offset_time_stamp(time_stamp, time_offset);
                result += offset_time_stamp;
                os << offset_time_stamp << std::flush;
            }
        }
        else{
            buffer<bf16> logits = this->engine->decode_audio(no_time_stamp_token);
            last_idx = this->sampler->sample(logits);
            std::string token_str = this->tokenizer->run_time_decoder(last_idx);
            result += token_str;
            os << token_str << std::flush;
        }
        
        int watching_dog = 16;
        for (int i = 0; i < 448 - 3; i++){
            buffer<bf16> logits = this->engine->decode_audio(last_idx);
            if (watching_dog == 0 && allow_force_time_stamp){
                last_idx = this->_sample_in_time_stamp(logits);
                watching_dog = 16;
            }
            else{
                last_idx = this->sampler->sample(logits);
                if (watching_dog > 0){
                    watching_dog--;
                }
            }
            std::string token_str = this->tokenizer->run_time_decoder(last_idx);
            
            if (_is_normal_token(last_idx)){
                os << token_str << std::flush;
                result += token_str;
            }
            else if (return_time_stamp && _is_time_stemp(last_idx)){
                std::string offset_time_stamp = this->_offset_time_stamp(token_str, time_offset);
                os << offset_time_stamp << std::flush;
                result += offset_time_stamp;
            }

            if (enable_time_stamp && _is_time_stemp(last_idx)){
                last_time_stamp = last_idx;
                watching_dog = 16;
            }
         
            if (last_idx == 50257){
                break;
            }

            if (!_is_valid_utf8(token_str)){
                allow_force_time_stamp = false;
            }
            else {
                allow_force_time_stamp = true;
            }
        }

        
        if (l_this_round < WINDOW_SAMPLES){
            break;
        }
     
        if (enable_time_stamp){
            float end_time = _get_time(last_time_stamp);
            if (end_time == 0){
                end_time = 30;
            }
            l_this_round = end_time * FS;
        }

        
        current_idx += l_this_round;
 
        l_this_round = std::min(WINDOW_SAMPLES, length - current_idx);
        l_this_round = std::max(l_this_round, 0);
        
    }
    return std::make_pair(result, langmap::to_language_name(language_detected));
}

/// \brief the `hf` protocol -- see generation_hf.hpp for what each piece ports.
/// \note Structured as the same "one 30 s window at a time" outer loop as
///       _generate_legacy (this engine's decoder is a single sequential KV-cache, so
///       that shape is shared by construction, not a choice this port makes), with the
///       per-window body replaced: language is detected AND FED (the legacy protocol's
///       central defect -- modeling_whisper.cpp's old body called
///       `decode_audio(transcribe_token)` right after `decode_audio(SOT)`, silently
///       skipping the language token entirely, so every later step conditioned on a
///       [SOT, task] context instead of [SOT, lang, task]), suppress_tokens and
///       begin_suppress_tokens are applied every step (never applied at all before),
///       and timestamp pairing/monotonicity/the initial-timestamp cap come from
///       WhisperTimeStampLogitsProcessor instead of the old "force a timestamp every
///       16 tokens" watchdog, which is what produced both defects the task brief
///       described: mid-word cutoffs (the watchdog firing mid-word, unconditionally)
///       and "I'm sorry." loops (nothing ever suppressed the tokens that make that
///       phrase, and no monotonicity rule stopped it repeating).
std::pair<std::string, std::string> Whisper::_generate_hf(whisper_task_type_t task, bool enable_time_stamp,
                                                            bool return_time_stamp, std::ostream& os) {
    if ((!enable_time_stamp) && (return_time_stamp)) {
        header_print("Error", "Return_time_stamp is true but timestamp is not enabled!");
        return std::make_pair("", "");
    }

    const whisper_hf::GenerationConfig& gc = this->_hf_gen_config();
    const int vocab_size = this->_real_vocab_size();
    const std::vector<int> lang_ids = gc.lang_ids();
    const int timestamp_begin = gc.timestamp_begin();

    int task_token;
    if (task == e_transcribe) {
        auto it = gc.task_to_id.find("transcribe");
        if (it == gc.task_to_id.end()) {
            throw std::runtime_error("generation_config.json has no task_to_id['transcribe']");
        }
        task_token = it->second;
    } else if (task == e_translate) {
        auto it = gc.task_to_id.find("translate");
        if (it == gc.task_to_id.end()) {
            throw std::runtime_error("generation_config.json has no task_to_id['translate']");
        }
        task_token = it->second;
    } else {
        header_print("Error", "Non-recongnized task!");
        return std::make_pair("", "");
    }

    auto to_float_vec = [&](buffer<bf16>& logits) {
        std::vector<float> v(static_cast<size_t>(vocab_size));
        for (int i = 0; i < vocab_size; ++i) v[static_cast<size_t>(i)] = float(logits[i]);
        return v;
    };

    int length = static_cast<int>(this->audio_buffer.size());
    int current_idx = 0;
    int l_this_round = std::min(WINDOW_SAMPLES, length);
    std::string result;
    std::string language_detected;

    // task 0180 Part B: per-request stage timers (host wall clock; NOT an NPU
    // performance claim -- rule 1). Accumulated across every window this
    // request opens (almost always one, for the <=30s clips this was profiled
    // on) and printed once at the end of this function.
    double t_preprocess = 0, t_encode = 0, t_decode = 0, t_logits_copy = 0,
           t_logits_proc = 0, t_argmax = 0, t_tok_decode = 0;
    int64_t n_decode_steps = 0;

    while (current_idx < length) {
        if (l_this_round == 0) {
            break;
        }
        const float time_offset = _S2T_(current_idx);
        const float window_seconds = _S2T_(l_this_round);

        std::vector<float> audio_chunk(static_cast<size_t>(l_this_round));
        audio_chunk.insert(audio_chunk.begin(), this->audio_buffer.data() + current_idx,
                            this->audio_buffer.data() + current_idx + l_this_round);
        double t0 = now_s_whisper();
        _preprocess_audio(mel_feature, audio_chunk);
        t_preprocess += now_s_whisper() - t0;

        t0 = now_s_whisper();
        this->engine->encode_audio(mel_feature);
        t_encode += now_s_whisper() - t0;
        this->engine->clear_context();

        // [SOT] -> detect_language -> FEED the language token (the legacy protocol
        // never did the feed; see the function comment above).
        t0 = now_s_whisper();
        buffer<bf16> logits_buf = this->engine->decode_audio(start_of_transcript);
        t_decode += now_s_whisper() - t0;
        ++n_decode_steps;
        t0 = now_s_whisper();
        std::vector<float> sot_logits = to_float_vec(logits_buf);
        t_logits_copy += now_s_whisper() - t0;
        const int lang_id = whisper_hf::detect_language(sot_logits, lang_ids, vocab_size);
        language_detected = this->tokenizer->run_time_decoder(lang_id);

        t0 = now_s_whisper();
        logits_buf = this->engine->decode_audio(lang_id);  // context: [SOT, lang]
        t_decode += now_s_whisper() - t0; ++n_decode_steps;
        t0 = now_s_whisper();
        logits_buf = this->engine->decode_audio(task_token);  // context: [SOT, lang, task]
        t_decode += now_s_whisper() - t0; ++n_decode_steps;
        int begin_index = 3;
        if (!enable_time_stamp) {
            t0 = now_s_whisper();
            logits_buf = this->engine->decode_audio(no_time_stamp_token);  // + [notimestamps]
            t_decode += now_s_whisper() - t0; ++n_decode_steps;
            begin_index = 4;
        }

        whisper_hf::WhisperTimestampProcessor ts_proc(gc.no_timestamps_token_id, gc.eos_token_id,
                                                        gc.has_max_initial_timestamp_index,
                                                        gc.max_initial_timestamp_index);

        std::vector<int> generated;  // tokens produced since begin_index, this window
        const int max_new_tokens = std::max(0, gc.max_length - begin_index);

        for (int step = 0; step < max_new_tokens; ++step) {
            t0 = now_s_whisper();
            std::vector<float> logits = to_float_vec(logits_buf);
            t_logits_copy += now_s_whisper() - t0;
            const bool at_begin_index = generated.empty();

            t0 = now_s_whisper();
            whisper_hf::apply_suppress_tokens_at_begin(logits, gc.begin_suppress_tokens, at_begin_index, vocab_size);
            whisper_hf::apply_suppress_tokens(logits, gc.suppress_tokens, vocab_size);
            if (enable_time_stamp) {
                ts_proc.apply(logits, generated, vocab_size);
            }
            t_logits_proc += now_s_whisper() - t0;
            t0 = now_s_whisper();
            const int token = whisper_hf::argmax(logits, vocab_size);
            t_argmax += now_s_whisper() - t0;

            t0 = now_s_whisper();
            std::string token_str = this->tokenizer->run_time_decoder(token);
            t_tok_decode += now_s_whisper() - t0;
            if (token < gc.eos_token_id) {
                // an ordinary text token
                result += token_str;
                os << token_str << std::flush;
            } else if (return_time_stamp && enable_time_stamp && token >= timestamp_begin) {
                const std::string offset_str = this->_offset_time_stamp(token_str, time_offset);
                result += offset_str;
                os << offset_str << std::flush;
            }
            // else: a timestamp token that isn't being surfaced (enable_time_stamp &&
            // !return_time_stamp -- the server path) still gets fed below so the
            // decoder's own pairing state and the seek arithmetic see it; it just never
            // reaches `result`/`os`. Matches the legacy protocol's server-facing text.

            generated.push_back(token);
            if (token == gc.eos_token_id) {
                break;
            }
            t0 = now_s_whisper();
            logits_buf = this->engine->decode_audio(token);
            t_decode += now_s_whisper() - t0;
            ++n_decode_steps;
        }

        if (l_this_round < WINDOW_SAMPLES) {
            break;
        }

        float advance_seconds = window_seconds;
        if (enable_time_stamp) {
            advance_seconds = whisper_hf::compute_segment_offset_seconds(generated, timestamp_begin, window_seconds);
            if (advance_seconds <= 0.0f) {
                // Documented deviation from HF (generation_hf.hpp): a zero-second
                // offset is reachable (an unclosed trailing segment that opened at
                // exactly <|0.00|>) and would stall this engine's single sequential
                // window forever, where HF's batched seek loop just lets other batch
                // items carry the shared `seek` state forward regardless.
                header_print("Warning", "hf protocol: zero-length segment offset at t=" << time_offset
                                                                                          << "s, forcing full-window advance");
                advance_seconds = window_seconds;
            }
        }

        current_idx += _T2S_(advance_seconds);
        l_this_round = std::min(WINDOW_SAMPLES, length - current_idx);
        l_this_round = std::max(l_this_round, 0);
    }

    // task 0180 Part B: per-request stage breakdown, host wall clock, printed
    // once per request. `decode` includes the decoder's own host+NPU compute
    // AND engine_adapter.cpp's fp32->bf16 round of the returned logits (the
    // two are not separated -- see the task report); `logits copy` is the
    // matching bf16->fp32 copy of all vocab_padded elements back into
    // std::vector<float> on the caller's side (to_float_vec()), i.e. the
    // "51872 logits processed on the host per step" the task asked to measure.
    std::printf("[oflm] hf request stages (host wall clock; NOT an NPU perf claim): "
                "preprocess=%.1fms encode=%.1fms decode=%.1fms (%lld steps, %.3fms/step) "
                "logits_copy=%.1fms logits_proc=%.1fms argmax=%.1fms tok_decode=%.1fms\n",
                t_preprocess * 1e3, t_encode * 1e3, t_decode * 1e3,
                static_cast<long long>(n_decode_steps),
                n_decode_steps > 0 ? (t_decode * 1e3) / static_cast<double>(n_decode_steps) : 0.0,
                t_logits_copy * 1e3, t_logits_proc * 1e3, t_argmax * 1e3, t_tok_decode * 1e3);
    std::fflush(stdout);

    return std::make_pair(result, langmap::to_language_name(language_detected));
}


void Whisper::_build_time_map(){
    this->token_time_map.clear();
    auto ids = this->tokenizer->encode("<|0.00|>");
    this->token_time_map_offset = ids[0];
    for (float time = 0; time <= 30.00; time += 0.02){
        this->token_time_map.push_back(time);
    }

    this->total_time_stamps = this->token_time_map.size();
}

int Whisper::_sample_in_language(buffer<bf16>& logits){
    
    for (int i = 0; i < 50259; i++){
        logits[i] = -0x1.FEp127f;
    }
    for (int i = 50359; i < logits.size(); i++){
        logits[i] = -0x1.FEp127f;
    }
    return this->sampler->sample(logits);

}

int Whisper::_sample_in_time_stamp(buffer<bf16>& logits){
    for (int i = 0; i < this->token_time_map_offset; i++){
        logits[i] = -0x1.FEp127f;
    }
    for (int i = this->total_time_stamps + this->token_time_map_offset; i < logits.size(); i++){
        logits[i] = -0x1.FEp127f;
    }
    return this->sampler->sample(logits);
}
/// \file whisper_engine_closed.cpp
/// \brief The prebuilt whisper_npu behind the whisper_engine seam
/// \note The engine SELECTOR (make_whisper_engine(), the public factory
///       whisper_engine.hpp declares) moved to whisper_engine_select.cpp when
///       the open engine was wired in (phase 3b, issue #72) -- that file
///       decides open vs. closed and calls make_closed_whisper_engine() below
///       for the closed side. This file keeps only the closed engine itself,
///       so it stays buildable (and reviewable) independently of whether
///       OFLM_USE_OPEN_WHISPER is defined.
#include "whisper/whisper_engine.hpp"
#include "whisper/whisper_npu.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include <memory>
#include <string>

namespace {

/// Owns what Whisper::load_model used to build inline: the xclbin manager, the engine and
/// the one-shot Q4NX load. The calls forward unchanged.
class whisper_engine_closed final : public whisper_engine {
public:
    whisper_engine_closed(const std::string& model_path, Whisper_Config& config,
                          oflm_rt::device* device, bool enable_preemption) {
        npu = std::make_unique<npu_xclbin_manager>(npu_device::device_npu2, device, enable_preemption);
        engine = std::make_unique<whisper_npu>(config, npu.get(), 448);
        {
            Q4NX q4nx(model_path);
            engine->load_weights(q4nx);
        }
        engine->clear_context();
    }

    void encode_audio(buffer<bf16>& mel_feature) override { engine->encode_audio(mel_feature); }
    buffer<bf16> decode_audio(int last_id) override { return engine->decode_audio(last_id); }
    void clear_context() override { engine->clear_context(); }
    int get_current_context_length() override { return engine->get_current_context_length(); }
    std::string describe() const override { return "closed (whisper_npu)"; }

private:
    // Declaration order is destruction order reversed: the engine goes before the manager
    // whose hw_contexts it uses.
    std::unique_ptr<npu_xclbin_manager> npu;
    std::unique_ptr<whisper_npu> engine;
};

} // namespace

/// \brief Build the closed engine. Declared in whisper_engine_select.cpp, which is the
///        only caller -- this is not part of the public whisper_engine.hpp seam.
std::unique_ptr<whisper_engine> make_closed_whisper_engine(const std::string& model_path,
                                                           Whisper_Config& config,
                                                           oflm_rt::device* device,
                                                           bool enable_preemption) {
    return std::make_unique<whisper_engine_closed>(model_path, config, device, enable_preemption);
}

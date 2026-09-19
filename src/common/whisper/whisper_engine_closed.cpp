/// \file whisper_engine_closed.cpp
/// \brief The prebuilt whisper_npu behind the whisper_engine seam, and the engine selector
#include "whisper/whisper_engine.hpp"
#include "whisper/whisper_npu.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "utils/utils.hpp"
#include <stdexcept>

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

std::unique_ptr<whisper_engine> make_whisper_engine(const std::string& model_path,
                                                    Whisper_Config& config,
                                                    oflm_rt::device* device,
                                                    bool enable_preemption) {
    const std::string want = utils::getenv_oflm("OFLM_WHISPER_ENGINE");
    if (!want.empty() && want != "closed") {
        throw std::runtime_error("OFLM_WHISPER_ENGINE=" + want +
                                 ": this build has only the closed Whisper engine");
    }
    return std::make_unique<whisper_engine_closed>(model_path, config, device, enable_preemption);
}

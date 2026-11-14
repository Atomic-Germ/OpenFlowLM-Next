"""Execute the production generate body with host doubles, without model weights.

The method, preamble helper and metadata definition come directly from source;
only engine/tokenizer/sampler/profiling dependencies are replaced. This tests
control flow and metadata, not NPU arithmetic or tokenizer correctness.
"""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope='module')
def generation_binary(tmp_path_factory):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler required for executable generation regression')
    source = (ROOT / 'src/common/AutoModel/modeling_qwen3_5vl.cpp').read_text()
    method = source[source.index('std::string Qwen3_5VL::generate('):
                    source.index('std::string Qwen3_5VL::generate_with_prompt(')]
    header = (ROOT / 'src/include/AutoModel/modeling_qwen3_5vl.hpp').read_text()
    helper = header[header.index('int generation_preamble_rows() const'):].split('\n', 1)[0]
    header = (ROOT / 'src/include/AutoModel/automodel.hpp').read_text()
    metadata = header[header.index('struct chat_meta_info_t {'):].split('\n};', 1)[0] + '\n};'
    harness = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "AutoModel/stop_reason.hpp"
// The guard inside `generate()` now calls the shared headroom rule, so the extracted method
// needs its declaration to compile on its own.
#include "AutoModel/generation_headroom.hpp"
#define header_print(tag, text) do { std::ostringstream log; log << text; } while (false)
using bf16 = uint16_t;
template<class T> using buffer = std::vector<T>;
namespace time_utils {
using time_with_unit = std::pair<double, std::string>;
time_with_unit cast_to_us(time_with_unit t) { return t; }
}
struct Profiler {
    double elapsed = 17;
    int starts = 0, stops = 0;
    bool running = false;
    void reset() { elapsed = 0; starts = stops = 0; running = false; }
    void start() { ++starts; running = true; }
    void stop(int) { ++stops; elapsed += 10; running = false; }
    time_utils::time_with_unit get_total_time() { return {elapsed, "us"}; }
};
enum { DECODING_TIME, TKOEN_DECODE_TIME, SAMPLING_TIME };
enum class StreamEventType { CONTENT, TOOL };
struct Engine {
    int pos = 0;
    std::vector<int> calls;
    buffer<bf16> forward(int t) { calls.push_back(t); ++pos; return {}; }
    int get_current_context_length() { return pos; }
};
struct Tokenizer {
    int calls = 0;
    std::string run_time_decoder(int token) {
        ++calls;
        return token == 42 ? "[ordinary]" : token == 99 ? "[eos]" : "[preamble]";
    }
};
struct Sampler {
    int calls = 0;
    int eos_at = 0;
    int sample(const buffer<bf16>&) { return ++calls == eos_at ? 99 : 42; }
};
METADATA
struct Qwen3_5VL {
    bool enable_think = false;
    int MAX_L = 16, last_token = 1, think_start_id = 248068, think_end_id = 248069;
    size_t total_tokens = 0;
    std::vector<int> token_history;
    Profiler profiler_list[3];
    Engine engine;
    Tokenizer decoder;
    Sampler sampling;
    Engine* lm_engine = &engine;
    Tokenizer* tokenizer = &decoder;
    Sampler* sampler = &sampling;
    std::string buffer_ = "partial", tool_name_ = "tool";
    StreamEventType current_mode_ = StreamEventType::TOOL;
    bool is_in_tool_block_ = true;
    bool is_normal_token(int token) { return token == 42; }
    bool is_eos(int token) { return token == 99; }
    HELPER
    std::string generate(chat_meta_info_t&, int, std::ostream&, std::function<bool()>);
};
METHOD
int main(int argc, char** argv) {
    assert(argc == 3 || argc == 4);
    Qwen3_5VL model;
    model.enable_think = std::string(argv[1]) == "think";
    const std::string scenario = argv[2];
    const int preamble = model.generation_preamble_rows();
    const int largest = model.MAX_L - preamble - 1;
    const bool refused = scenario == "refuse" || scenario == "refuse-stale";
    const bool first_eos = scenario == "eos-first" || scenario == "eos-first-boundary";
    const bool boundary = scenario == "boundary" || scenario == "eos-first-boundary";
    model.total_tokens = refused ? largest + 1 : boundary ? largest : 1;
    model.engine.pos = static_cast<int>(model.total_tokens);
    const auto position = model.engine.pos;
    model.token_history = {123};
    const int available = model.MAX_L - position - preamble;
    const bool exhaust_context = scenario == "context" || scenario == "eos-at-context";
    model.sampling.eos_at = first_eos ? 1 : scenario == "eos" ? 2 : scenario == "eos-at-context" ? available : 0;
    chat_meta_info_t meta;
    meta.prompt_tokens = position;
    meta.total_duration = 123; meta.load_duration = 456; meta.prefill_duration = 789;
    if (scenario == "refuse-stale") {
        meta.stop_reason = CANCEL_DETECTED;
        meta.decoding_duration = 99999;
    }
    std::ostringstream output;
    const int limit = argc == 4 ? std::stoi(argv[3]) : exhaust_context ? 0 : 2;
    auto result = model.generate(meta, limit, output,
                                 [&] { return scenario == "cancel"; });
    assert(meta.prompt_tokens == position);
    assert(meta.total_duration == 123 && meta.load_duration == 456 && meta.prefill_duration == 789);
    assert(meta.decoding_duration == model.profiler_list[DECODING_TIME].elapsed * 1000);
    for (const auto& p : model.profiler_list) assert(!p.running);
    if (refused) {
        assert(meta.stop_reason == MAX_LENGTH_REACHED);
        assert(meta.generated_tokens == 0 && meta.decoding_duration == 0);
        assert(model.total_tokens == static_cast<size_t>(position) && model.engine.pos == position);
        assert(model.engine.calls.empty() && model.decoder.calls == 0 && model.sampling.calls == 0);
        assert(model.token_history == std::vector<int>{123});
        assert(model.profiler_list[DECODING_TIME].starts == 0);
        assert(model.profiler_list[TKOEN_DECODE_TIME].starts == 0);
        assert(result.empty() && output.str().empty());
    } else {
        assert(!result.empty() && result == output.str());
        const auto expected = model.enable_think ? std::vector<int>{model.think_start_id,198}
            : std::vector<int>{model.think_start_id,271,model.think_end_id,271};
        assert(model.engine.calls.size() >= static_cast<size_t>(preamble));
        assert(std::vector<int>(model.engine.calls.begin(),model.engine.calls.begin()+preamble) == expected);
        assert(meta.decoding_duration > 0);
        if (scenario == "output-cap" || first_eos) {
            const int count = first_eos ? 1 : limit > 0 && limit < available ? limit : available;
            assert(meta.generated_tokens == count);
            assert(meta.stop_reason == (first_eos ? EOT_DETECTED : MAX_LENGTH_REACHED));
            assert(model.sampling.calls == count);
            const int forwards = preamble + count - 1 + (first_eos ? 1 : 0);
            assert(model.engine.calls.size() == static_cast<size_t>(forwards));
            assert(model.engine.pos == position + forwards);
            assert(model.total_tokens == static_cast<size_t>(position + preamble + count));
            std::string expected_output = model.enable_think ? "[preamble][preamble]" : "";
            for (int i = 0; i < count; ++i) expected_output += first_eos ? "[eos]" : "[ordinary]";
            assert(output.str() == expected_output);
            if (first_eos) {
                assert(model.engine.calls.back() == 99);
                assert(model.token_history.back() == 99);
            }
        } else if (scenario == "boundary") {
            assert(meta.stop_reason == MAX_LENGTH_REACHED && meta.generated_tokens == 1);
            assert(model.total_tokens == static_cast<size_t>(model.MAX_L));
            assert(model.engine.calls.size() == static_cast<size_t>(preamble));
            assert(model.engine.pos == position+preamble);
        } else if (exhaust_context) {
            const bool eos = scenario == "eos-at-context";
            assert(meta.stop_reason == (eos ? EOT_DETECTED : MAX_LENGTH_REACHED));
            assert(available > 1 && meta.generated_tokens == available);
            assert(model.sampling.calls == available);
            assert(model.total_tokens == static_cast<size_t>(model.MAX_L));
            assert(model.engine.pos == model.MAX_L - (eos ? 0 : 1));
            assert(model.engine.calls.size() == static_cast<size_t>(preamble + available - (eos ? 0 : 1)));
            assert(model.token_history.back() == (eos ? 99 : 42));
        } else if (scenario == "cancel") {
            assert(meta.stop_reason == CANCEL_DETECTED && meta.generated_tokens == 1);
            assert(model.engine.calls.size() == static_cast<size_t>(preamble));
            assert(model.buffer_.empty() && model.tool_name_.empty() && !model.is_in_tool_block_);
            assert(model.current_mode_ == StreamEventType::CONTENT);
        } else {
            assert(meta.generated_tokens == 2);
            assert(meta.stop_reason == (scenario == "eos" ? EOT_DETECTED : MAX_LENGTH_REACHED));
        }
    }
}
'''.replace('METADATA', metadata).replace('HELPER', helper).replace('METHOD', method)
    work = tmp_path_factory.mktemp('qwen35-generation')
    cpp = work / 'generate_test.cpp'
    cpp.write_text(harness)
    binary = work / 'generate_test'
    subprocess.run([compiler, '-std=c++17', '-O0', '-I', str(ROOT / 'src/include'),
                    str(cpp), '-o', str(binary)], check=True, capture_output=True, text=True)
    return binary


@pytest.mark.parametrize('branch', ['think', 'no-think'])
@pytest.mark.parametrize('scenario', [
    'refuse', 'refuse-stale', 'boundary', 'limit', 'eos', 'cancel', 'context', 'eos-at-context',
])
def test_generation_finalizes_metadata_without_unwanted_forward_steps(generation_binary, branch, scenario):
    result = subprocess.run([str(generation_binary), branch, scenario], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('branch', ['think', 'no-think'])
@pytest.mark.parametrize('limit', [1, 2, 0, -1])
@pytest.mark.parametrize('scenario', ['output-cap', 'eos-first', 'eos-first-boundary'])
def test_generation_output_limit_and_first_eos(generation_binary, branch, limit, scenario):
    result = subprocess.run([str(generation_binary), branch, scenario, str(limit)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

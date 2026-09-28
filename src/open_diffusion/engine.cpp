// open_diffusion engine: replays export_bundle.py's schedule. See engine.hpp.
#include "engine.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <tuple>

#include "nlohmann/json.hpp"
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace open_diffusion {
namespace {

constexpr int kOpcode = 3;   // mlir-aie: run a DPU instruction sequence

std::vector<char> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    std::vector<char> d(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(d.data(), static_cast<std::streamsize>(d.size()));
    return d;
}

json read_json(const fs::path& p) {
    auto d = read_file(p);
    return json::parse(d.begin(), d.end());
}

void read_into(const fs::path& p, void* dst, size_t cap) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    size_t n = static_cast<size_t>(f.tellg());
    if (n > cap) throw std::runtime_error(p.string() + " is larger than its buffer");
    f.seekg(0);
    f.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
}

uint16_t to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7FFF + ((u >> 16) & 1);                       // round to nearest even
    return static_cast<uint16_t>(u >> 16);
}

// The manifest if dir holds a complete set of this format and layout; *why otherwise.
bool read_manifest(const fs::path& dir, const std::string& layout, json* out, std::string* why) {
    std::ifstream f(dir / "diffusion_kernels.json", std::ios::binary);
    if (!f) { *why = "no diffusion_kernels.json"; return false; }
    json j;
    try { f >> j; } catch (const json::exception&) { *why = "diffusion_kernels.json does not parse"; return false; }
    if (j.value("format", std::string()) != kKernelsFormat) {
        *why = "diffusion_kernels.json is not format " + std::string(kKernelsFormat);
        return false;
    }
    if (!j.value("complete", false)) { *why = "the kernel set is incomplete"; return false; }
    if (j.value("layout", std::string()) != layout) {
        *why = "the kernel set's layout " + j.value("layout", std::string("(none)")) +
               " is not the model's " + layout + " (built from other kernel code; rebuild one of them)";
        return false;
    }
    if (out) *out = std::move(j);
    return true;
}

}  // namespace

bool kernels_usable(const std::string& dir, const std::string& layout, std::string* why) {
    return read_manifest(dir, layout, nullptr, why);
}

std::string find_kernels(const std::string& model_dir, const std::string& env_dir,
                         const std::vector<std::string>& roots, std::string* how) {
    if (!env_dir.empty()) { *how = "OFLM_DIFFUSION_KERNELS_DIR"; return env_dir; }
    json bundle = read_json(fs::path(model_dir) / "bundle.json");
    std::string layout = bundle.at("layout").get<std::string>(), why;
    fs::path local = fs::path(model_dir) / "open_kernels";
    if (kernels_usable(local.string(), layout, &why)) { *how = "beside the model"; return local.string(); }
    // keyed on the family, not the model: a fine-tune of the same shape reuses the set
    std::string family = bundle.at("family").get<std::string>();
    for (const auto& r : roots) {
        fs::path cand = fs::path(r) / "xclbins" / family / "open_kernels";
        if (kernels_usable(cand.string(), layout, &why)) { *how = "an xclbins root"; return cand.string(); }
    }
    return {};
}

struct Engine::Impl {
    struct Stream {
        xrt::bo instr;
        std::vector<uint32_t> words;
    };
    struct Set {
        xrt::hw_context ctx;
        xrt::kernel kernel;
        fs::path dir;
        std::map<std::string, Stream> streams;
    };
    struct Buf {
        xrt::bo bo;
        size_t bytes = 0;
        std::map<std::pair<size_t, size_t>, xrt::bo> views;
    };
    struct Op {
        int set;
        std::string stream, phase;
        xrt::run run;
    };

    fs::path dir;
    json bundle, sched;
    int R = 0, T = 0, C = 0, max_tokens = 0, pad_id = 0, embed_dim = 0, token_row = 0;
    std::string templ;
    xrt::device dev{0u};
    std::vector<std::string> set_names;
    std::vector<std::unique_ptr<Set>> sets;
    std::map<std::string, Buf> bufs;
    std::vector<Op> ops;
    Stream* vl_stream = nullptr;
    std::vector<size_t> vl_words;
    std::ifstream embed;

    Stream& stream(int s, const std::string& name) {
        Set& set = *sets[s];
        auto it = set.streams.find(name);
        if (it != set.streams.end()) return it->second;
        auto d = read_file(set.dir / ("insts_" + name + ".bin"));
        if (d.size() % 4) throw std::runtime_error(name + ": instruction stream not word-sized");
        Stream st;
        st.words.resize(d.size() / 4);
        std::memcpy(st.words.data(), d.data(), d.size());
        st.instr = xrt::bo(dev, d.size(), xrt::bo::flags::cacheable, set.kernel.group_id(1));
        std::memcpy(st.instr.map<void*>(), d.data(), d.size());
        st.instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return set.streams.emplace(name, std::move(st)).first->second;
    }

    Buf& buf(const std::string& name) {
        auto it = bufs.find(name);
        if (it == bufs.end()) throw std::runtime_error("schedule names an unknown buffer " + name);
        return it->second;
    }

    // XRT sub-buffers are views of the ROOT allocation (views of views would put the
    // host pointer and the device address at different offsets).
    xrt::bo& view(const json& a) {
        Buf& b = buf(a[0].get<std::string>());
        size_t off = a[1].get<size_t>(), n = a[2].get<size_t>();
        if (off == 0 && (n == 0 || n == b.bytes)) return b.bo;
        if (n == 0) n = b.bytes - off;
        if (off + n > b.bytes) throw std::runtime_error("view past the end of " + a[0].get<std::string>());
        auto key = std::make_pair(off, n);
        auto it = b.views.find(key);
        if (it == b.views.end()) it = b.views.emplace(key, xrt::bo(b.bo, n, off)).first;
        return it->second;
    }

    void alloc(const std::string& name, size_t bytes, xrt::memory_group group) {
        Buf b;
        b.bo = xrt::bo(dev, bytes, xrt::bo::flags::host_only, group);
        b.bytes = bytes;
        std::memset(b.bo.map<void*>(), 0, bytes);
        bufs.emplace(name, std::move(b));
    }
};

Engine::Engine(const std::string& model_dir, const std::string& kernels_dir, int size)
    : impl_(std::make_unique<Impl>()) {
    Impl& m = *impl_;
    m.dir = model_dir;
    m.bundle = read_json(m.dir / "bundle.json");
    json manifest;
    std::string why;
    if (!read_manifest(kernels_dir, m.bundle.at("layout").get<std::string>(), &manifest, &why))
        throw std::runtime_error("kernel set " + kernels_dir + ": " + why);
    auto res = m.bundle.at("resolutions");
    if (!res.contains(std::to_string(size))) {
        std::vector<int> sizes;
        for (auto& [r, _] : res.items()) sizes.push_back(std::stoi(r));
        std::sort(sizes.begin(), sizes.end());
        std::string have;
        for (int r : sizes) have += (have.empty() ? "" : ", ") + std::to_string(r);
        throw std::runtime_error("unsupported size " + std::to_string(size) + " (supported: " + have + ")");
    }
    m.sched = read_json(m.dir / res.at(std::to_string(size)).get<std::string>());
    m.R = m.sched.at("R").get<int>();
    m.T = m.sched.at("image_tokens").get<int>();
    m.C = m.sched.at("latent_channels").get<int>();
    m.max_tokens = m.bundle.at("max_tokens").get<int>();
    m.pad_id = m.bundle.at("pad_id").get<int>();
    m.templ = m.bundle.at("prompt_template").get<std::string>();
    m.embed_dim = m.bundle.at("embed").at("dim").get<int>();
    m.token_row = m.sched.at("inputs").at("token_row_elems").get<int>();

    fs::path kdir = kernels_dir;
    for (auto& [name, sub] : manifest.at("sets").items()) {
        auto s = std::make_unique<Impl::Set>();
        s->dir = kdir / sub.get<std::string>();
        xrt::xclbin xcl((s->dir / "final.xclbin").string());
        auto uuid = m.dev.register_xclbin(xcl);
        s->ctx = xrt::hw_context(m.dev, uuid);
        s->kernel = xrt::kernel(s->ctx, "MLIR_AIE");
        m.set_names.push_back(name);
        m.sets.push_back(std::move(s));
    }
    // npu2 has one memory group for data arguments (npu_device.cpp); take arg 3's
    auto group = m.sets.front()->kernel.group_id(3);

    for (auto& [name, bytes] : m.sched.at("buffers").items())
        m.alloc(name, bytes.get<size_t>(), group);
    for (auto& [name, file] : m.sched.at("init").items()) {
        Impl::Buf& b = m.buf(name);
        read_into(m.dir / file.get<std::string>(), b.bo.map<void*>(), b.bytes);
    }
    for (auto& [name, b] : m.bufs) b.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    fs::path wpath = m.dir / m.bundle.at("weights_file").get<std::string>();
    std::ifstream wf(wpath, std::ios::binary | std::ios::ate);
    if (!wf) throw std::runtime_error("cannot open " + wpath.string());
    auto wsize = static_cast<size_t>(wf.tellg());
    for (auto& [name, w] : m.bundle.at("weights").items()) {
        size_t off = w.at("offset").get<size_t>(), bytes = w.at("bytes").get<size_t>();
        if (off + bytes > wsize) throw std::runtime_error(wpath.string() + " is shorter than " + name + " needs");
        m.alloc(name, bytes, group);
        Impl::Buf& b = m.buf(name);
        wf.seekg(static_cast<std::streamoff>(off));
        wf.read(b.bo.map<char*>(), static_cast<std::streamsize>(bytes));
        if (!wf) throw std::runtime_error("cannot read " + name + " from " + wpath.string());
        b.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    std::map<std::string, int> set_index;
    for (size_t i = 0; i < m.set_names.size(); ++i) set_index[m.set_names[i]] = static_cast<int>(i);
    for (const auto& o : m.sched.at("ops")) {
        Impl::Op op;
        op.set = set_index.at(o[0].get<std::string>());
        op.stream = o[1].get<std::string>();
        op.phase = o[3].get<std::string>();
        Impl::Stream& st = m.stream(op.set, op.stream);
        op.run = xrt::run(m.sets[op.set]->kernel);
        op.run.set_arg(0, kOpcode);
        op.run.set_arg(1, st.instr);
        op.run.set_arg(2, static_cast<int>(st.words.size()));
        int i = 3;
        for (const auto& a : o[2]) op.run.set_arg(i++, m.view(a));
        m.ops.push_back(std::move(op));
    }

    int fa = set_index.at("fa");
    m.vl_stream = &m.stream(fa, "te_attn");
    // the words are the kernel set's (a probe build found them), not the model's
    json fa_meta = read_json(kdir / manifest.at("sets").at("fa").get<std::string>() / "dit_fa.json");
    for (const auto& w : fa_meta.at("patch").at("te_attn").at("valid_len"))
        m.vl_words.push_back(w.get<size_t>());
    m.embed.open(m.dir / m.bundle.at("embed").at("file").get<std::string>(), std::ios::binary);
    if (!m.embed) throw std::runtime_error("cannot open the embedding table");
}

Engine::~Engine() = default;

int Engine::size() const { return impl_->R; }
int Engine::image_tokens() const { return impl_->T; }
int Engine::latent_channels() const { return impl_->C; }
int Engine::max_tokens() const { return impl_->max_tokens; }
int Engine::pad_id() const { return impl_->pad_id; }
const std::string& Engine::prompt_template() const { return impl_->templ; }

void Engine::set_tokens(const std::vector<int64_t>& ids) {
    Impl& m = *impl_;
    // the length as given, not up to the first pad id: a prompt may itself hold that token
    int n_real = static_cast<int>(ids.size());
    if (n_real == 0 || n_real > m.max_tokens)
        throw std::runtime_error("the prompt must have 1.." + std::to_string(m.max_tokens) + " tokens");
    Impl::Buf& xt = m.buf(m.sched.at("inputs").at("tokens").get<std::string>());
    auto* x = xt.bo.map<uint16_t*>();
    std::memset(x, 0, xt.bytes);
    int rows = m.bundle.at("embed").at("rows").get<int>();
    for (int t = 0; t < m.max_tokens; ++t) {
        int64_t id = t < n_real ? ids[t] : m.pad_id;
        if (id < 0 || id >= rows) throw std::runtime_error("token id out of range");
        m.embed.seekg(static_cast<std::streamoff>(id) * m.embed_dim * 2);
        m.embed.read(reinterpret_cast<char*>(x + static_cast<size_t>(t) * m.token_row), m.embed_dim * 2);
    }
    xt.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // te_attn masks keys at or past valid_len: the prompt's length
    Impl::Stream& st = *m.vl_stream;
    for (size_t w : m.vl_words) st.words[w] = static_cast<uint32_t>(n_real);
    std::memcpy(st.instr.map<void*>(), st.words.data(), st.words.size() * 4);
    st.instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

void Engine::set_noise(const std::vector<uint16_t>& bits) {
    Impl& m = *impl_;
    size_t n = static_cast<size_t>(m.T) * m.C;
    if (bits.size() != n) throw std::runtime_error("noise must be image_tokens x 128 values");
    Impl::Buf& lat = m.buf(m.sched.at("inputs").at("latents").get<std::string>());
    std::memcpy(lat.bo.map<void*>(), bits.data(), n * 2);
    lat.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, n * 2, 0);
}

std::vector<uint16_t> Engine::seeded_noise(uint64_t seed) const {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> out(static_cast<size_t>(impl_->T) * impl_->C);
    for (auto& v : out) v = to_bf16(nd(rng));
    return out;
}

Timing Engine::run(bool profile) {
    Impl& m = *impl_;
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    auto wait = [&](xrt::run& r, const Impl::Op& op) {
        auto st = r.wait();
        if (st != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error(m.set_names[op.set] + "/" + op.stream + ": state " +
                                     std::to_string(static_cast<int>(st)));
    };
    Timing t;
    // the runs queued on the current hardware context, oldest first; every one is waited
    // on (a run never waited on keeps a stale state and cannot be started again)
    std::vector<Impl::Op*> queued;
    auto drain = [&] {
        for (Impl::Op* q : queued) wait(q->run, *q);
        queued.clear();
    };
    int cur_set = -1;
    std::string cur_phase;
    auto t0 = clk::now(), tp = t0;
    for (auto& op : m.ops) {
        if (op.set != cur_set || op.phase != cur_phase) {
            // runs queued across two hardware contexts hang the array: drain first
            drain();
            if (op.phase != cur_phase) {
                auto now = clk::now();
                if (!cur_phase.empty()) t.phases.emplace_back(cur_phase, secs(tp, now));
                tp = now;
                cur_phase = op.phase;
            }
            cur_set = op.set;
        }
        auto ts = clk::now();
        op.run.start();
        queued.push_back(&op);
        if (profile) {
            drain();
            t.op_ms.push_back(1e3 * secs(ts, clk::now()));
        }
    }
    drain();
    auto end = clk::now();
    t.phases.emplace_back(cur_phase, secs(tp, end));
    t.total_s = secs(t0, end);
    return t;
}

std::vector<uint8_t> Engine::rgb() {
    Impl& m = *impl_;
    const auto& o = m.sched.at("outputs");
    Impl::Buf& b = m.buf(o.at("rgba").get<std::string>());
    size_t row = o.at("rgba_row_bytes").get<size_t>(), used = o.at("rgba_used_bytes").get<size_t>();
    size_t px = static_cast<size_t>(m.R) * m.R, rows = px * 4 / used;
    b.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, rows * row, 0);
    const auto* src = b.bo.map<const uint8_t*>();
    std::vector<uint8_t> out(px * 3);
    size_t k = 0;
    for (size_t r = 0; r < rows; ++r)
        for (size_t i = 0; i < used; i += 4, ++k) {
            const uint8_t* p = src + r * row + i;
            out[3 * k] = p[0];
            out[3 * k + 1] = p[1];
            out[3 * k + 2] = p[2];
        }
    return out;
}

std::vector<std::tuple<std::string, std::string, std::string>> Engine::ops() const {
    std::vector<std::tuple<std::string, std::string, std::string>> out;
    for (const auto& op : impl_->ops) out.emplace_back(impl_->set_names[op.set], op.stream, op.phase);
    return out;
}

// ------------------------------------------------------------------------ encoding

std::vector<uint8_t> Engine::encode(const std::string& format, int jpeg_quality) {
    auto px = rgb();
    int R = impl_->R;
    std::vector<uint8_t> out;
    auto sink = [](void* ctx, void* data, int n) {
        auto* v = static_cast<std::vector<uint8_t>*>(ctx);
        v->insert(v->end(), static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + n);
    };
    int ok = 0;
    if (format == "png")
        ok = stbi_write_png_to_func(sink, &out, R, R, 3, px.data(), 3 * R);
    else if (format == "jpeg")
        ok = stbi_write_jpg_to_func(sink, &out, R, R, 3, px.data(), std::clamp(jpeg_quality, 1, 100));
    else
        throw std::runtime_error("unsupported image format " + format + " (png or jpeg)");
    if (!ok) throw std::runtime_error("encoding the " + format + " failed");
    return out;
}

std::string format_for_path(const std::string& path) {
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ext == ".png") return "png";
    if (ext == ".jpg" || ext == ".jpeg") return "jpeg";
    return {};
}

}  // namespace open_diffusion

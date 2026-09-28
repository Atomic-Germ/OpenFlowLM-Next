// open_diffusion engine: replays export_bundle.py's schedule. See engine.hpp.
#include "engine.hpp"
#include "schedule.hpp"

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

bool available(std::string* why) {
    if (why) why->clear();
    return true;
}

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
    // An op's argument as the schedule names it; in step k of the step template it is at
    // off + k * stride.
    struct Arg {
        std::string buf;
        size_t off = 0, n = 0, stride = 0;
    };
    struct StepOp {
        int set;
        std::string stream;
        std::vector<Arg> args;
    };
    // One resolution: its schedule, activations and runs.
    struct Res {
        json sched;
        int R = 0, T = 0, C = 0, token_row = 0, bundle_steps = 0;
        std::map<std::string, Buf> bufs;
        std::vector<Op> head, tail;             // conditioning + text encoder; the VAE
        std::vector<StepOp> step_tmpl;          // step 0
        std::vector<std::vector<Op>> step_ops;  // step k's runs, built on first use
        std::vector<char> tf0, dt0;             // the bundle's TF / DT (its step count)
        int steps = 0;                          // the count TF / DT hold now
    };

    fs::path dir, kdir;
    json bundle, manifest;
    int max_tokens = 0, pad_id = 0, embed_dim = 0, bundle_steps = 0;
    std::string templ;
    xrt::device dev;
    std::vector<std::string> set_names;
    std::vector<std::unique_ptr<Set>> sets;
    std::map<std::string, int> set_index;
    xrt::memory_group group{};
    std::map<std::string, Buf> weights;         // shared by every resolution
    std::map<int, std::unique_ptr<Res>> res;
    Res* cur = nullptr;
    int cur_steps = 0;
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

    Buf& buf(Res& r, const std::string& name) {
        auto it = r.bufs.find(name);
        if (it != r.bufs.end()) return it->second;
        it = weights.find(name);
        if (it == weights.end()) throw std::runtime_error("schedule names an unknown buffer " + name);
        return it->second;
    }

    // XRT sub-buffers are views of the ROOT allocation (views of views would put the
    // host pointer and the device address at different offsets).
    xrt::bo& view(Res& r, const std::string& name, size_t off, size_t n) {
        Buf& b = buf(r, name);
        if (off == 0 && (n == 0 || n == b.bytes)) return b.bo;
        if (n == 0) n = b.bytes - off;
        if (off + n > b.bytes) throw std::runtime_error("view past the end of " + name);
        auto key = std::make_pair(off, n);
        auto it = b.views.find(key);
        if (it == b.views.end()) it = b.views.emplace(key, xrt::bo(b.bo, n, off)).first;
        return it->second;
    }

    Buf alloc(size_t bytes) {
        Buf b;
        b.bo = xrt::bo(dev, bytes, xrt::bo::flags::host_only, group);
        b.bytes = bytes;
        std::memset(b.bo.map<void*>(), 0, bytes);
        return b;
    }

    Op make_op(Res& r, int set, const std::string& name, const std::string& phase,
               const std::vector<Arg>& args, size_t k) {
        Op op;
        op.set = set;
        op.stream = name;
        op.phase = phase;
        Stream& st = stream(set, name);
        op.run = xrt::run(sets[set]->kernel);
        op.run.set_arg(0, kOpcode);
        op.run.set_arg(1, st.instr);
        op.run.set_arg(2, static_cast<int>(st.words.size()));
        int i = 3;
        for (const auto& a : args) op.run.set_arg(i++, view(r, a.buf, a.off + k * a.stride, a.n));
        return op;
    }

    Res& load_res(int size);
    void set_steps(Res& r, int steps);
    static std::vector<Arg> op_args(const json& o);
    static Res& selected(Res* r);
    static std::vector<Op*> op_order(Res& r, int steps);   // a selection's runs, in order
};

namespace {

// The schedule's buffers whose contents depend on the step count (klein_pipeline.py):
// the timestep features the conditioning GEMMs read, and each step's Euler dt.
constexpr const char* kTfBuf = "TF";
constexpr const char* kDtBuf = "DT";

bool is_step_phase(const std::string& p) {
    return p.size() > 4 && p.compare(0, 4, "step") == 0 &&
           std::all_of(p.begin() + 4, p.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

}  // namespace

std::vector<Engine::Impl::Arg> Engine::Impl::op_args(const json& o) {
    std::vector<Arg> out;
    for (const auto& a : o[2]) {
        Arg x;
        x.buf = a[0].get<std::string>();
        x.off = a[1].get<size_t>();
        x.n = a[2].get<size_t>();
        out.push_back(std::move(x));
    }
    return out;
}

Engine::Impl::Res& Engine::Impl::selected(Res* r) {
    if (!r) throw std::runtime_error("no resolution is selected (Engine::select)");
    return *r;
}

std::vector<Engine::Impl::Op*> Engine::Impl::op_order(Res& r, int steps) {
    std::vector<Op*> out;
    for (auto& op : r.head) out.push_back(&op);
    for (int k = 0; k < steps; ++k)
        for (auto& op : r.step_ops[k]) out.push_back(&op);
    for (auto& op : r.tail) out.push_back(&op);
    return out;
}

Engine::Impl::Res& Engine::Impl::load_res(int size) {
    auto found = res.find(size);
    if (found != res.end()) return *found->second;
    auto rp = std::make_unique<Res>();
    Res& r = *rp;
    r.sched = read_json(dir / bundle.at("resolutions").at(std::to_string(size)).get<std::string>());
    r.R = r.sched.at("R").get<int>();
    r.T = r.sched.at("image_tokens").get<int>();
    r.C = r.sched.at("latent_channels").get<int>();
    r.token_row = r.sched.at("inputs").at("token_row_elems").get<int>();
    r.bundle_steps = r.sched.at("steps").get<int>();

    // Split the op list: head, bundle_steps contiguous step groups, tail.
    std::vector<std::vector<const json*>> groups;
    std::vector<const json*> head, tail;
    for (const auto& o : r.sched.at("ops")) {
        std::string ph = o[3].get<std::string>();
        if (is_step_phase(ph)) {
            if (!tail.empty()) throw std::runtime_error("the schedule's steps are not contiguous");
            size_t k = std::stoul(ph.substr(4));
            if (k == groups.size()) groups.emplace_back();
            else if (k + 1 != groups.size())
                throw std::runtime_error("the schedule's steps are out of order at " + ph);
            groups[k].push_back(&o);
        } else {
            (groups.empty() ? head : tail).push_back(&o);
        }
    }
    if (groups.empty() || static_cast<int>(groups.size()) != r.bundle_steps)
        throw std::runtime_error("the schedule has " + std::to_string(groups.size()) +
                                 " step phases, not " + std::to_string(r.bundle_steps));

    // The step template, and each arg's stride from step 0 to step 1. Every later step
    // must lie on the same line, or step k of another count could not be derived.
    for (size_t i = 0; i < groups[0].size(); ++i) {
        const json& o = *groups[0][i];
        StepOp sp;
        sp.set = set_index.at(o[0].get<std::string>());
        sp.stream = o[1].get<std::string>();
        sp.args = op_args(o);
        for (size_t k = 1; k < groups.size(); ++k) {
            std::string where = "step " + std::to_string(k) + " op " + std::to_string(i);
            if (groups[k].size() != groups[0].size())
                throw std::runtime_error("the schedule's steps differ in length");
            const json& q = *groups[k][i];
            auto qa = op_args(q);
            if (q[0] != o[0] || q[1] != o[1] || qa.size() != sp.args.size())
                throw std::runtime_error(where + " is not step 0's");
            for (size_t j = 0; j < qa.size(); ++j) {
                Arg& a = sp.args[j];
                if (qa[j].buf != a.buf || qa[j].n != a.n || qa[j].off < a.off)
                    throw std::runtime_error(where + " reads another buffer than step 0's");
                if (k == 1) a.stride = qa[j].off - a.off;
                if (qa[j].off != a.off + k * a.stride)
                    throw std::runtime_error(where + " is not step 0's moved on by a fixed stride");
            }
        }
        r.step_tmpl.push_back(std::move(sp));
    }

    // Activations, with room for kMaxSteps where a view moves per step.
    std::map<std::string, size_t> need;
    for (const auto& sp : r.step_tmpl)
        for (const auto& a : sp.args)
            if (a.stride) {
                size_t end = a.off + static_cast<size_t>(kMaxSteps - 1) * a.stride + a.n;
                need[a.buf] = std::max(need[a.buf], end);
            }
    for (auto& [name, bytes] : r.sched.at("buffers").items()) {
        size_t n = bytes.get<size_t>();
        auto it = need.find(name);
        if (it != need.end()) n = std::max(n, it->second);
        r.bufs.emplace(name, alloc(n));
    }
    for (auto& [name, file] : r.sched.at("init").items()) {
        Buf& b = buf(r, name);
        fs::path p = dir / file.get<std::string>();
        read_into(p, b.bo.map<void*>(), b.bytes);
        if (name == kTfBuf) r.tf0 = read_file(p);
        if (name == kDtBuf) r.dt0 = read_file(p);
    }
    if (r.tf0.empty() || r.dt0.empty())
        throw std::runtime_error(std::string("the schedule does not initialise ") + kTfBuf + " and " + kDtBuf);
    for (auto& [name, b] : r.bufs) b.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    r.steps = r.bundle_steps;

    for (const json* o : head)
        r.head.push_back(make_op(r, set_index.at((*o)[0].get<std::string>()), (*o)[1].get<std::string>(),
                                 (*o)[3].get<std::string>(), op_args(*o), 0));
    for (const json* o : tail)
        r.tail.push_back(make_op(r, set_index.at((*o)[0].get<std::string>()), (*o)[1].get<std::string>(),
                                 (*o)[3].get<std::string>(), op_args(*o), 0));
    return *res.emplace(size, std::move(rp)).first->second;
}

void Engine::Impl::set_steps(Res& r, int steps) {
    while (static_cast<int>(r.step_ops.size()) < steps) {
        size_t k = r.step_ops.size();
        std::vector<Op> ops;
        for (const auto& sp : r.step_tmpl)
            ops.push_back(make_op(r, sp.set, sp.stream, "step" + std::to_string(k), sp.args, k));
        r.step_ops.push_back(std::move(ops));
    }
    if (r.steps == steps) return;
    Buf& tf = buf(r, kTfBuf);
    Buf& dt = buf(r, kDtBuf);
    std::memset(tf.bo.map<void*>(), 0, tf.bytes);
    std::memset(dt.bo.map<void*>(), 0, dt.bytes);
    if (steps == r.bundle_steps) {
        // the bundle's own bytes: its default image stays exactly what it was
        std::memcpy(tf.bo.map<void*>(), r.tf0.data(), std::min(tf.bytes, r.tf0.size()));
        std::memcpy(dt.bo.map<void*>(), r.dt0.data(), std::min(dt.bytes, r.dt0.size()));
    } else {
        auto sig = schedule::sigmas(r.T, steps);
        auto tfv = schedule::timestep_features(sig, steps);
        auto dtv = schedule::dt_params(sig, steps);
        if (tfv.size() * 2 > tf.bytes || dtv.size() * 2 > dt.bytes)
            throw std::runtime_error("the step count does not fit the schedule's TF / DT buffers");
        std::memcpy(tf.bo.map<void*>(), tfv.data(), tfv.size() * 2);
        std::memcpy(dt.bo.map<void*>(), dtv.data(), dtv.size() * 2);
    }
    tf.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    dt.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    r.steps = steps;
}

Engine::Engine(const std::string& model_dir, const std::string& kernels_dir, const oflm_rt::device* dev)
    : impl_(std::make_unique<Impl>()) {
    Impl& m = *impl_;
    m.dir = model_dir;
    m.kdir = kernels_dir;
    m.bundle = read_json(m.dir / "bundle.json");
    std::string why;
    if (!read_manifest(kernels_dir, m.bundle.at("layout").get<std::string>(), &m.manifest, &why))
        throw std::runtime_error("kernel set " + kernels_dir + ": " + why);
    m.max_tokens = m.bundle.at("max_tokens").get<int>();
    m.pad_id = m.bundle.at("pad_id").get<int>();
    m.templ = m.bundle.at("prompt_template").get<std::string>();
    m.embed_dim = m.bundle.at("embed").at("dim").get<int>();
    // every schedule of a bundle is made with one step count (export_bundle.py)
    const json& resolutions = m.bundle.at("resolutions");
    if (resolutions.empty()) throw std::runtime_error("the bundle has no resolutions");
    m.bundle_steps = read_json(m.dir / resolutions.begin().value().get<std::string>()).at("steps").get<int>();
    m.dev = dev ? *dev : xrt::device(0u);

    for (auto& [name, sub] : m.manifest.at("sets").items()) {
        auto s = std::make_unique<Impl::Set>();
        s->dir = m.kdir / sub.get<std::string>();
        xrt::xclbin xcl((s->dir / "final.xclbin").string());
        auto uuid = m.dev.register_xclbin(xcl);
        s->ctx = xrt::hw_context(m.dev, uuid);
        s->kernel = xrt::kernel(s->ctx, "MLIR_AIE");
        m.set_index[name] = static_cast<int>(m.sets.size());
        m.set_names.push_back(name);
        m.sets.push_back(std::move(s));
    }
    // npu2 has one memory group for data arguments (npu_device.cpp); take arg 3's
    m.group = m.sets.front()->kernel.group_id(3);

    fs::path wpath = m.dir / m.bundle.at("weights_file").get<std::string>();
    std::ifstream wf(wpath, std::ios::binary | std::ios::ate);
    if (!wf) throw std::runtime_error("cannot open " + wpath.string());
    auto wsize = static_cast<size_t>(wf.tellg());
    for (auto& [name, w] : m.bundle.at("weights").items()) {
        size_t off = w.at("offset").get<size_t>(), bytes = w.at("bytes").get<size_t>();
        if (off + bytes > wsize) throw std::runtime_error(wpath.string() + " is shorter than " + name + " needs");
        Impl::Buf b = m.alloc(bytes);
        wf.seekg(static_cast<std::streamoff>(off));
        wf.read(b.bo.map<char*>(), static_cast<std::streamsize>(bytes));
        if (!wf) throw std::runtime_error("cannot read " + name + " from " + wpath.string());
        b.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        m.weights.emplace(name, std::move(b));
    }

    m.vl_stream = &m.stream(m.set_index.at("fa"), "te_attn");
    // the words are the kernel set's (a probe build found them), not the model's
    json fa_meta = read_json(m.kdir / m.manifest.at("sets").at("fa").get<std::string>() / "dit_fa.json");
    for (const auto& w : fa_meta.at("patch").at("te_attn").at("valid_len"))
        m.vl_words.push_back(w.get<size_t>());
    m.embed.open(m.dir / m.bundle.at("embed").at("file").get<std::string>(), std::ios::binary);
    if (!m.embed) throw std::runtime_error("cannot open the embedding table");
}

Engine::Engine(const std::string& model_dir, const std::string& kernels_dir, int size)
    : Engine(model_dir, kernels_dir, static_cast<const oflm_rt::device*>(nullptr)) {
    select(size);
}

Engine::~Engine() = default;

std::vector<int> Engine::sizes() const {
    std::vector<int> out;
    for (auto& [r, _] : impl_->bundle.at("resolutions").items()) out.push_back(std::stoi(r));
    std::sort(out.begin(), out.end());
    return out;
}

int Engine::default_steps() const { return impl_->bundle_steps; }

void Engine::select(int size, int steps) {
    Impl& m = *impl_;
    if (!m.bundle.at("resolutions").contains(std::to_string(size))) {
        std::string have;
        for (int r : sizes()) have += (have.empty() ? "" : ", ") + std::to_string(r);
        throw std::runtime_error("unsupported size " + std::to_string(size) + " (supported: " + have + ")");
    }
    if (steps == 0) steps = m.bundle_steps;
    if (steps < 1 || steps > kMaxSteps)
        throw std::runtime_error("unsupported step count " + std::to_string(steps) + " (1.." +
                                 std::to_string(kMaxSteps) + ")");
    if (steps != m.bundle_steps && m.bundle_steps < 2)
        throw std::runtime_error("this bundle has a single step; no other count can be derived from it");
    Impl::Res& r = m.load_res(size);
    m.set_steps(r, steps);
    m.cur = &r;
    m.cur_steps = steps;
}

int Engine::size() const { return impl_->cur ? impl_->cur->R : 0; }
int Engine::steps() const { return impl_->cur_steps; }
int Engine::image_tokens() const { return Impl::selected(impl_->cur).T; }
int Engine::latent_channels() const { return Impl::selected(impl_->cur).C; }
int Engine::max_tokens() const { return impl_->max_tokens; }
int Engine::pad_id() const { return impl_->pad_id; }
const std::string& Engine::prompt_template() const { return impl_->templ; }

void Engine::set_tokens(const std::vector<int64_t>& ids) {
    Impl& m = *impl_;
    Impl::Res& r = Impl::selected(m.cur);
    // the length as given, not up to the first pad id: a prompt may itself hold that token
    int n_real = static_cast<int>(ids.size());
    if (n_real == 0 || n_real > m.max_tokens)
        throw std::runtime_error("the prompt must have 1.." + std::to_string(m.max_tokens) + " tokens");
    Impl::Buf& xt = m.buf(r, r.sched.at("inputs").at("tokens").get<std::string>());
    auto* x = xt.bo.map<uint16_t*>();
    std::memset(x, 0, xt.bytes);
    int rows = m.bundle.at("embed").at("rows").get<int>();
    for (int t = 0; t < m.max_tokens; ++t) {
        int64_t id = t < n_real ? ids[t] : m.pad_id;
        if (id < 0 || id >= rows) throw std::runtime_error("token id out of range");
        m.embed.seekg(static_cast<std::streamoff>(id) * m.embed_dim * 2);
        m.embed.read(reinterpret_cast<char*>(x + static_cast<size_t>(t) * r.token_row), m.embed_dim * 2);
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
    Impl::Res& r = Impl::selected(m.cur);
    size_t n = static_cast<size_t>(r.T) * r.C;
    if (bits.size() != n) throw std::runtime_error("noise must be image_tokens x 128 values");
    Impl::Buf& lat = m.buf(r, r.sched.at("inputs").at("latents").get<std::string>());
    std::memcpy(lat.bo.map<void*>(), bits.data(), n * 2);
    lat.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, n * 2, 0);
}

std::vector<uint16_t> Engine::seeded_noise(uint64_t seed) const {
    const Impl::Res& r = Impl::selected(impl_->cur);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> out(static_cast<size_t>(r.T) * r.C);
    for (auto& v : out) v = schedule::bf16_bits(nd(rng));
    return out;
}

Timing Engine::run(bool profile) {
    Impl& m = *impl_;
    Impl::Res& r = Impl::selected(m.cur);
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    auto wait = [&](xrt::run& run, const Impl::Op& op) {
        auto st = run.wait();
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
    for (Impl::Op* opp : Impl::op_order(r, m.cur_steps)) {
        Impl::Op& op = *opp;
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
    Impl::Res& r = Impl::selected(m.cur);
    const auto& o = r.sched.at("outputs");
    Impl::Buf& b = m.buf(r, o.at("rgba").get<std::string>());
    size_t row = o.at("rgba_row_bytes").get<size_t>(), used = o.at("rgba_used_bytes").get<size_t>();
    size_t px = static_cast<size_t>(r.R) * r.R, rows = px * 4 / used;
    b.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, rows * row, 0);
    const auto* src = b.bo.map<const uint8_t*>();
    std::vector<uint8_t> out(px * 3);
    size_t k = 0;
    for (size_t y = 0; y < rows; ++y)
        for (size_t i = 0; i < used; i += 4, ++k) {
            const uint8_t* p = src + y * row + i;
            out[3 * k] = p[0];
            out[3 * k + 1] = p[1];
            out[3 * k + 2] = p[2];
        }
    return out;
}

std::vector<std::tuple<std::string, std::string, std::string>> Engine::ops() const {
    std::vector<std::tuple<std::string, std::string, std::string>> out;
    for (const Impl::Op* op : Impl::op_order(Impl::selected(impl_->cur), impl_->cur_steps))
        out.emplace_back(impl_->set_names[op->set], op->stream, op->phase);
    return out;
}

// ------------------------------------------------------------------------ encoding

std::vector<uint8_t> Engine::encode(const std::string& format, int jpeg_quality) {
    auto px = rgb();
    int R = Impl::selected(impl_->cur).R;
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

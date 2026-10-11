// open_diffusion engine: replays export_bundle.py's schedule. See engine.hpp.
#include "engine.hpp"
#include "reference.hpp"
#include "schedule.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <future>
#include <fstream>
#include <map>
#include <optional>
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
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_kernel.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace open_diffusion {
namespace {

// Stretches (runlists) in flight on the context at most: the host blocks on the oldest
// beyond this.
constexpr size_t kWindow = 32;

// The context's QoS priority (amdxdna's "high"; normal is 0x200). A set configured by
// register writes is not one the firmware can restore: at normal priority, another
// process's context preempting ours mid-image hangs both (ERT_CMD_STATE_TIMEOUT, 8 of 8
// trials over four variants, utilities/reconfig-probe/contention_trial.ps1). At high
// priority the other context runs only between our stretches, and each stretch starts
// from a reset (6 of 6 trials, both sizes, the images byte-identical and the other
// process unharmed). OPEN-DIFFUSION-NPU-SHARING records this as the maintainer's decision.
constexpr uint32_t kPriority = 0x180;

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
    if (!j.contains("elf") || !j["elf"].is_object() || j["elf"].empty()) {
        *why = "diffusion_kernels.json names no ELF";
        return false;
    }
    for (auto& [res, elf] : j["elf"].items())
        if (!elf.is_string() || !fs::is_regular_file(dir / elf.get<std::string>())) {
            *why = "the kernel set's " + res + " ELF is missing";
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
    struct Buf {
        xrt::bo bo;
        size_t bytes = 0;
        std::map<std::pair<size_t, size_t>, xrt::bo> views;
    };
    // An op's argument as the schedule names it; in step k of the step template it is at
    // off + k * stride.
    struct Arg {
        std::string buf;
        size_t off = 0, n = 0, stride = 0;
    };
    struct Op {
        int set;
        std::string stream, phase;
        std::vector<Arg> args;
        size_t k = 0;
        xrt::run run;
        // te_attn: its valid_len head for the prompt's length, run first (compose_elf.py)
        bool has_pre = false;
        xrt::run pre;
    };
    struct StepOp {
        int set;
        std::string stream;
        std::vector<Arg> args;
    };
    // One configuration (a resolution, or an edit at one): its schedule, activations and runs.
    struct Res {
        json sched;
        std::string key;                        // "512", or "512e512" for an edit
        bool edit = false;
        int R = 0, T = 0, C = 0, token_row = 0, bundle_steps = 0;
        std::map<std::string, Buf> bufs;
        std::vector<Op> head, tail;             // conditioning + text encoder; the VAE
        std::vector<StepOp> step_tmpl;          // step 0
        // Two sets of step runs, bound to step k and reused for step k + 2: the driver
        // refuses past ~1800 live runs ("Cannot extend beyond 8 banks"), which one set per
        // step reached at 9 steps (8 for an edit). run() rebinds a set's moving arguments
        // at the phase boundary, after the step that last used it has drained.
        std::vector<std::vector<Op>> step_ops;
        std::vector<char> tf0, dt0;             // the bundle's TF / DT (its step count)
        int steps = 0;                          // the count TF / DT hold now
        int vl = 0;                             // the valid_len the te_attn heads are bound to
        // The resolution's ELF as its one hardware context, and its kernels.
        xrt::elf elf;
        xrt::hw_context ctx;
        std::map<std::string, xrt::ext::kernel> kernels;   // "<set>:<stream>", made on first use
        // Per set, its two configure-only kernels (main:cfg_<set>_a / _b), and per variant
        // one run per configure in an image. They differ only in the empty device they reset
        // the array with; alternating them keeps the firmware from skipping that reset.
        std::vector<std::array<xrt::ext::kernel, 2>> cfg_kernel;
        std::vector<std::array<std::deque<xrt::run>, 2>> cfg_runs;
        int flip = 0;                                      // the variant the next configure uses
    };

    fs::path dir, kdir;
    json bundle, manifest;
    int max_tokens = 0, pad_id = 0, embed_dim = 0, bundle_steps = 0;
    std::string templ;
    xrt::device dev;
    std::vector<std::string> set_names;
    std::map<std::string, int> set_index;
    std::map<std::string, Buf> weights;         // shared by every resolution
    std::map<std::string, std::unique_ptr<Res>> res;
    Res* cur = nullptr;
    int cur_steps = 0;
    std::string vl_stream, vl_head;             // te_attn, and its per-length head ("{n}")
    int vl_max = 0, vl = 0;                     // vl: the prompt's length (set_tokens)
    std::ifstream embed;

    // Creating a kernel walks all of the ELF's control code (~2 ms per MB): hence one ELF per
    // resolution, and te_attn's per-length heads split off small (compose_elf.py).
    xrt::ext::kernel& kernel(Res& r, const std::string& name) {
        auto it = r.kernels.find(name);
        if (it == r.kernels.end()) it = r.kernels.emplace(name, xrt::ext::kernel(r.ctx, name)).first;
        return it->second;
    }

    // te_attn's head for the prompt's length (before set_tokens, the unmasked one).
    std::string head_name(int set) const {
        std::string k = vl_head;
        k.replace(k.find("{n}"), 3, std::to_string(vl ? vl : vl_max));
        return set_names[set] + ":" + k;
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
        b.bo = xrt::ext::bo(dev, bytes);
        b.bytes = bytes;
        std::memset(b.bo.map<void*>(), 0, bytes);
        return b;
    }

    // A run of the op's kernel with its arguments bound (step k of the template).
    void bind(Res& r, Op& op) {
        op.run = xrt::run(kernel(r, set_names[op.set] + ":" + op.stream));
        int i = 0;
        for (const auto& a : op.args) op.run.set_arg(i++, view(r, a.buf, a.off + op.k * a.stride, a.n));
        op.has_pre = op.stream == vl_stream;
        if (op.has_pre) op.pre = xrt::run(kernel(r, head_name(op.set)));
    }

    Op make_op(Res& r, int set, const std::string& name, const std::string& phase,
               const std::vector<Arg>& args, size_t k) {
        Op op;
        op.set = set;
        op.stream = name;
        op.phase = phase;
        op.args = args;
        op.k = k;
        bind(r, op);
        return op;
    }

    // Point r's te_attn heads at the prompt's length if it changed since they were made.
    void bind_vl(Res& r) {
        if (r.vl == vl) return;
        for (auto& op : r.head)
            if (op.has_pre) op.pre = xrt::run(kernel(r, head_name(op.set)));
        r.vl = vl;
    }

    std::map<std::string, std::future<std::unique_ptr<Res>>> prepared;   // open_res started early
    std::unique_ptr<Res> open_res(int size, bool edit);
    Res& load_res(int size, bool edit);
    // A configuration's name (klein_pipeline.config_key) and its schedule file, or "".
    static std::string config_key(int size, bool edit) {
        return edit ? std::to_string(size) + "e" + std::to_string(size) : std::to_string(size);
    }
    std::string schedule_file(int size, bool edit) const {
        const char* map = edit ? "edits" : "resolutions";
        if (!bundle.contains(map) || !bundle.at(map).contains(std::to_string(size))) return {};
        return bundle.at(map).at(std::to_string(size)).get<std::string>();
    }
    void set_steps(Res& r, int steps);
    void init(const std::string& model_dir, const std::string& kernels_dir, const oflm_rt::device* dev,
              int prefetch_size);
    static std::vector<Arg> op_args(const json& o);
    static Res& selected(Res* r);
    // A selection's runs in order, each with the step it runs as (-1: head or tail).
    static std::vector<std::pair<Op*, int>> op_order(Res& r, int steps);
    // Point a step set's runs at step k (only the arguments that move per step).
    void rebind(Res& r, std::vector<Op>& set, int k);
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

std::vector<std::pair<Engine::Impl::Op*, int>> Engine::Impl::op_order(Res& r, int steps) {
    std::vector<std::pair<Op*, int>> out;
    for (auto& op : r.head) out.emplace_back(&op, -1);
    for (int k = 0; k < steps; ++k)
        for (auto& op : r.step_ops[k % 2]) out.emplace_back(&op, k);
    for (auto& op : r.tail) out.emplace_back(&op, -1);
    return out;
}

void Engine::Impl::rebind(Res& r, std::vector<Op>& set, int k) {
    for (auto& op : set) {
        if (op.k == static_cast<size_t>(k)) continue;
        op.k = static_cast<size_t>(k);
        int i = 0;
        for (const auto& a : op.args) {
            if (a.stride) op.run.set_arg(i, view(r, a.buf, a.off + op.k * a.stride, a.n));
            ++i;
        }
    }
}

// A resolution's schedule, its ELF as one hardware context, and a kernel for every stream it
// runs. It touches nothing but the new Res and read-only engine state, so the constructor
// runs it on a thread while the weights load: creating ~80 kernels costs ~1-2.5 s of CPU
// inside XRT, the weights ~5 s of reading.
std::unique_ptr<Engine::Impl::Res> Engine::Impl::open_res(int size, bool edit) {
    auto rp = std::make_unique<Res>();
    Res& r = *rp;
    r.key = config_key(size, edit);
    r.edit = edit;
    // the configuration's ELF: every set as one hardware context (compose_elf.py)
    const json& elfs = manifest.at("elf");
    if (!elfs.contains(r.key))
        throw std::runtime_error("kernel set " + kdir.string() + " has no ELF for " + r.key);
    fs::path elf = kdir / elfs.at(r.key).get<std::string>();
    r.elf = xrt::elf(elf.string());
    try {
        r.ctx = xrt::hw_context(dev, r.elf, {{"priority", kPriority}}, xrt::hw_context::access_mode::shared);
    } catch (const std::exception& e) {
        throw std::runtime_error("cannot open a hardware context from " + elf.string() +
                                 " (the NPU driver must support full ELFs with several PDIs): " + e.what());
    }
    for (const auto& name : set_names) {
        const json& c = manifest.at("cfg").at(name);
        if (!c.is_array() || c.size() != 2)
            throw std::runtime_error("kernel set " + kdir.string() + ": set " + name + " needs two cfg kernels");
        r.cfg_kernel.push_back({xrt::ext::kernel(r.ctx, c[0].get<std::string>()),
                                xrt::ext::kernel(r.ctx, c[1].get<std::string>())});
    }
    r.cfg_runs.resize(set_names.size());
    r.sched = read_json(dir / schedule_file(size, edit));
    r.R = r.sched.at("R").get<int>();
    r.T = r.sched.at("image_tokens").get<int>();
    r.C = r.sched.at("latent_channels").get<int>();
    r.token_row = r.sched.at("inputs").at("token_row_elems").get<int>();
    r.bundle_steps = r.sched.at("steps").get<int>();
    for (const auto& o : r.sched.at("ops")) kernel(r, o[0].get<std::string>() + ":" + o[1].get<std::string>());
    return rp;
}

Engine::Impl::Res& Engine::Impl::load_res(int size, bool edit) {
    const std::string key = config_key(size, edit);
    auto found = res.find(key);
    if (found != res.end()) return *found->second;
    std::unique_ptr<Res> rp;
    auto pending = prepared.find(key);
    if (pending != prepared.end()) {
        rp = pending->second.get();
        prepared.erase(pending);
    } else {
        rp = open_res(size, edit);
    }
    Res& r = *rp;

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
                // the last step's view, and the whole of its slot: set_steps writes DT as
                // kMaxSteps full slots (klein_pipeline.dt_params)
                size_t end = a.off + std::max(static_cast<size_t>(kMaxSteps - 1) * a.stride + a.n,
                                              static_cast<size_t>(kMaxSteps) * a.stride);
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
    r.vl = vl;
    return *res.emplace(key, std::move(rp)).first->second;
}

void Engine::Impl::set_steps(Res& r, int steps) {
    while (static_cast<int>(r.step_ops.size()) < std::min(steps, 2)) {
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

void Engine::Impl::init(const std::string& model_dir, const std::string& kernels_dir,
                        const oflm_rt::device* device, int prefetch_size) {
    Impl& m = *this;
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
    // Every configuration the MODEL advertises must have an ELF in the KERNEL SET.
    // A model repo's schedules and a kernel set's ELFs are written by two different
    // tools (q4nx-build --open-diffusion and export_dit_kernels.py), so a set built
    // for text-to-image only could otherwise load an edit model fine and fail at the
    // point the user asked for it -- open_res()'s "no ELF for 512e512". Naming the
    // half sets here, at load, is the difference.
    if (m.bundle.contains("edits")) {
        // read_manifest() has already proved `elf` is a non-empty object of files
        // that exist, so the only thing left to check is that it covers whatever
        // the model says it can do.
        const json& elfs = m.manifest.at("elf");
        for (auto it = m.bundle.at("edits").begin(); it != m.bundle.at("edits").end(); ++it) {
            if (!elfs.contains(it.key()))
                throw std::runtime_error(
                    "kernel set " + kernels_dir + " has no ELF for the edit configuration " +
                    it.key() + " this model's bundle advertises, so it was built from other "
                    "kernel code. Point OFLM_DIFFUSION_KERNELS_DIR at a set built for this "
                    "model, or place one in this directory.");
        }
    }
    m.bundle_steps = read_json(m.dir / resolutions.begin().value().get<std::string>()).at("steps").get<int>();
    m.dev = device ? *device : xrt::device(0u);

    for (const auto& s : m.manifest.at("sets")) {
        std::string name = s.get<std::string>();
        m.set_index[name] = static_cast<int>(m.set_names.size());
        m.set_names.push_back(name);
    }
    const json& vl = m.manifest.at("valid_len");
    m.vl_stream = vl.at("stream").get<std::string>();
    m.vl_head = vl.at("head").get<std::string>();
    m.vl_max = vl.at("max").get<int>();
    if (m.vl_head.find("{n}") == std::string::npos || m.vl_max < m.max_tokens)
        throw std::runtime_error("kernel set " + kernels_dir + ": its valid_len kernels do not cover " +
                                 std::to_string(m.max_tokens) + " tokens");
    if (prefetch_size && m.bundle.at("resolutions").contains(std::to_string(prefetch_size)) &&
        m.manifest.at("elf").contains(std::to_string(prefetch_size)))
        m.prepared.emplace(std::to_string(prefetch_size), std::async(std::launch::async, [&m, prefetch_size] {
            return m.open_res(prefetch_size, false);
        }));

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

    m.embed.open(m.dir / m.bundle.at("embed").at("file").get<std::string>(), std::ios::binary);
    if (!m.embed) throw std::runtime_error("cannot open the embedding table");
}

Engine::Engine(const std::string& model_dir, const std::string& kernels_dir, const oflm_rt::device* dev)
    : impl_(std::make_unique<Impl>()) {
    impl_->init(model_dir, kernels_dir, dev, 0);
}

Engine::Engine(const std::string& model_dir, const std::string& kernels_dir, int size)
    : impl_(std::make_unique<Impl>()) {
    // the size is known up front: its context and kernels are made while the weights load
    impl_->init(model_dir, kernels_dir, nullptr, size);
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

std::vector<int> Engine::edit_sizes() const {
    std::vector<int> out;
    if (impl_->bundle.contains("edits"))
        for (auto& [r, _] : impl_->bundle.at("edits").items()) out.push_back(std::stoi(r));
    std::sort(out.begin(), out.end());
    return out;
}

void Engine::select(int size, int steps, bool edit) {
    Impl& m = *impl_;
    if (m.schedule_file(size, edit).empty()) {
        std::string have;
        for (int r : edit ? edit_sizes() : sizes()) have += (have.empty() ? "" : ", ") + std::to_string(r);
        if (edit)
            throw std::runtime_error("edits are not supported at " + std::to_string(size) + " (supported: " +
                                     (have.empty() ? std::string("none in this model") : have) +
                                     "; the output is the reference's size)");
        throw std::runtime_error("unsupported size " + std::to_string(size) + " (supported: " + have + ")");
    }
    if (steps == 0) steps = m.bundle_steps;
    if (steps < 1 || steps > kMaxSteps)
        throw std::runtime_error("unsupported step count " + std::to_string(steps) + " (1.." +
                                 std::to_string(kMaxSteps) + ")");
    if (steps != m.bundle_steps && m.bundle_steps < 2)
        throw std::runtime_error("this bundle has a single step; no other count can be derived from it");
    Impl::Res& r = m.load_res(size, edit);
    m.set_steps(r, steps);
    m.cur = &r;
    m.cur_steps = steps;
}

int Engine::size() const { return impl_->cur ? impl_->cur->R : 0; }
bool Engine::editing() const { return impl_->cur && impl_->cur->edit; }
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
    // te_attn masks keys at or past valid_len, the prompt's length: its heads switch to
    // that length's kernel
    m.vl = n_real;
    m.bind_vl(r);
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

void Engine::set_reference(const std::vector<uint8_t>& rgb) {
    Impl& m = *impl_;
    Impl::Res& r = Impl::selected(m.cur);
    if (!r.edit) throw std::runtime_error("set_reference needs an edit selected (select(size, steps, true))");
    if (rgb.size() != static_cast<size_t>(r.R) * r.R * 3)
        throw std::runtime_error("the reference must be size x size x 3 RGB8");
    // the encoder's input: bf16 2 (x / 255) - 1 in channels 0-2 of a zero-bordered
    // [R + 2, pitch, channels] buffer (vae_encoder.py); the other channels and the border stay zero
    const json& in = r.sched.at("inputs");
    Impl::Buf& b = m.buf(r, in.at("reference").get<std::string>());
    size_t pitch = in.at("reference_pitch").get<size_t>(), ch = in.at("reference_channels").get<size_t>();
    if ((static_cast<size_t>(r.R) + 2) * pitch * ch * 2 > b.bytes)
        throw std::runtime_error("the schedule's reference buffer is too small");
    auto v = reference_bf16(rgb);
    auto* x = b.bo.map<uint16_t*>();
    for (int y = 0; y < r.R; ++y)
        for (int px = 0; px < r.R; ++px) {
            uint16_t* d = x + ((static_cast<size_t>(y) + 1) * pitch + px + 1) * ch;
            const uint16_t* s = v.data() + (static_cast<size_t>(y) * r.R + px) * 3;
            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
        }
    b.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
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
    m.bind_vl(r);
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    Timing t;
    // A stretch is one set's configure-only run and the ops after it on that set, submitted
    // as ONE xrt::runlist, which XRT executes atomically. The configuration is register
    // writes the firmware does not know about: if another process's context takes the NPU
    // between two of our commands, ours comes back without it and the next op hangs. So
    // every stretch configures first, and a switch can only fall between stretches.
    // Every stretch is waited on (a run never waited on cannot be started again).
    struct Stretch {
        xrt::runlist list;
        std::string what;
        int set = -1, v = 0;
        size_t cfg = 0;                           // its configure run, pool index
    };
    // A configure run goes back to its pool's free list once its stretch has completed, so
    // an image holds about as many as are in flight, not one per stretch: the driver gives
    // every run its own copy of its control code from a bounded heap ("Cannot extend beyond
    // 8 banks"), and a configure's is a whole set's register writes.
    std::vector<std::array<std::vector<size_t>, 2>> cfg_free(m.set_names.size());
    for (size_t set = 0; set < cfg_free.size(); ++set)
        for (int v = 0; v < 2; ++v)
            for (size_t i = r.cfg_runs[set][v].size(); i-- > 0;) cfg_free[set][v].push_back(i);
    std::deque<Stretch> inflight;
    std::optional<Stretch> open;
    auto wait_oldest = [&] {
        Stretch s = std::move(inflight.front());
        inflight.pop_front();
        try {
            s.list.wait();
        } catch (const xrt::runlist::command_error& e) {
            throw std::runtime_error(s.what + ": state " + std::to_string(static_cast<int>(e.get_command_state())));
        }
        cfg_free[s.set][s.v].push_back(s.cfg);
    };
    auto close = [&] {
        if (!open) return;
        if (inflight.size() >= kWindow) wait_oldest();
        open->list.execute();
        inflight.push_back(std::move(*open));
        open.reset();
    };
    auto drain = [&] {
        close();
        while (!inflight.empty()) wait_oldest();
    };
    int cur_set = -1;
    std::string cur_phase;
    auto t0 = clk::now(), tp = t0;
    try {
        for (auto [opp, k] : Impl::op_order(r, m.cur_steps)) {
            Impl::Op& op = *opp;
            const std::string phase = k < 0 ? op.phase : "step" + std::to_string(k);
            if (phase != cur_phase) {
                drain();
                auto now = clk::now();
                if (!cur_phase.empty()) t.phases.emplace_back(cur_phase, secs(tp, now));
                tp = now;
                cur_phase = phase;
                if (k >= 0) m.rebind(r, r.step_ops[k % 2], k);   // its last user has drained
            }
            auto ts = clk::now();
            if (!open || op.set != cur_set) {
                close();
                open = Stretch{xrt::runlist(r.ctx), m.set_names[op.set] + "/" + op.stream};
                int v = r.flip;
                r.flip ^= 1;
                auto& pool = r.cfg_runs[op.set][v];   // a deque: runs in a list keep their address
                auto& free = cfg_free[op.set][v];
                size_t i = pool.size();
                if (free.empty()) pool.emplace_back(r.cfg_kernel[op.set][v]);
                else i = free.back(), free.pop_back();
                open->list.add(pool[i]);
                open->set = op.set;
                open->v = v;
                open->cfg = i;
                cur_set = op.set;
            }
            if (op.has_pre) open->list.add(op.pre);
            open->list.add(op.run);
            if (profile) {
                drain();                              // the next op configures again
                t.op_ms.push_back(1e3 * secs(ts, clk::now()));
            }
        }
        drain();
    } catch (...) {
        // a failure must not leave runs executing: they could not be started again
        for (auto& s : inflight) {
            try { s.list.wait(); } catch (...) {}
        }
        throw;
    }
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
    for (auto [op, k] : Impl::op_order(Impl::selected(impl_->cur), impl_->cur_steps))
        out.emplace_back(impl_->set_names[op->set], op->stream, k < 0 ? op->phase : "step" + std::to_string(k));
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

/// \file npue_embedding.cpp
/// \brief The only translation unit that sees the NpuEmbeddings engine.
/// \date 2026-09-03
///
/// SPDX-License-Identifier: MIT
///
/// THIS FILE IS FORK-OWNED; src/open_npue/ is a synced copy from upstream and
/// must not be edited here. The separation is deliberate: the sync tool
/// overwrites everything under open_npue/ and knows nothing about this file.
///
/// It is listed in OPEN_NPUE_SOURCES in src/CMakeLists.txt, which means it
/// carries the engine's own compile flags -- `/arch:AVX2` (or `-mavx2 -mfma`)
/// and the open_npue include directory. Both matter, and the header explains
/// why at length: the engine is header-defined and half of an encode is AVX2
/// intrinsics behind `#if defined(__AVX2__)`, so a host translation unit that
/// includes it WITHOUT the flag instantiates the scalar path, and the linker
/// keeps a mixture of the two. Measured cost of that mistake: 1-cos 1.04e-04
/// on bge-base against the same engine in its own binary, on identical bytes.

#include "AutoEmbeddingModel/npue_embedding.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <thread>

#include "utils/utils.hpp"
#include "npue_encoder.hpp"
#include "npue_pack.hpp"

namespace {

/// Map the caller's task onto one of THIS model's prompt names.
///
/// THE MOST DANGEROUS SEAM IN THE WHOLE INTEGRATION, and the reason it refuses
/// rather than guessing: a wrongly-prefixed embedding is correctly shaped,
/// correctly normed and deterministic. Nothing downstream can tell that the
/// answer is for a different task.
///
/// Models differ in what they offer -- the BERT family has no prompt concept
/// at all, nomic has four names, EmbeddingGemma has fourteen -- so this asks
/// the CONTAINER what exists and takes the first candidate the model actually
/// declares. If a model has prompts and none of the candidates is among them,
/// that is a mapping this build does not know, and it throws naming what the
/// model does offer.
std::string prompt_for(const std::vector<std::string>& names,
                       embedding_task_type_t task_type) {
    if (names.empty()) return std::string();

    std::vector<std::string> candidates;
    switch (task_type) {
        case task_query:
            candidates = {"query", "search_query", "Retrieval-query",
                          "Retrieval"};                        break;
        case task_document:
            candidates = {"document", "search_document",
                          "Retrieval-document", "Retrieval"};  break;
        case task_clustering:
            candidates = {"clustering", "Clustering"};         break;
        case task_classification:
            candidates = {"classification", "Classification"}; break;
        case task_multilabel_classification:
            candidates = {"MultilabelClassification",
                          "classification", "Classification"}; break;
        case task_sentence_similarity:
            candidates = {"STS", "classification", "search_query"}; break;
        case task_summarization:
            candidates = {"Summarization", "clustering"};      break;
        case task_bitextmining:
            candidates = {"BitextMining", "search_query"};     break;
        case task_code_retrieval:
            candidates = {"Retrieval-query", "search_query", "query"}; break;
        case task_search_result:
            candidates = {"search_document", "document",
                          "Retrieval-document"};               break;
        default:
            candidates = {"search_query", "query"};            break;
    }
    for (const auto& c : candidates)
        if (std::find(names.begin(), names.end(), c) != names.end()) return c;

    std::string have;
    for (const auto& n : names) have += (have.empty() ? "" : ", ") + n;
    throw std::runtime_error(
        "NpueEmbedding: this model declares task prompts [" + have +
        "] and none of them matches the requested task. Refusing to pick one: "
        "a wrongly-prefixed embedding is correctly shaped and correctly "
        "normed, so nothing downstream could tell the answer is for a "
        "different task.");
}

int int_or(const json& j, const char* key, int fallback) {
    if (j.contains(key) && j[key].is_number_integer()) return j[key].get<int>();
    return fallback;
}

/// The `.npue` container: the checkpoint's weights, pre-tiled for the array.
///
/// PACKED ON FIRST RUN when it is absent, from the model author's own files in
/// the same directory. NOTHING IS RE-HOSTED: the weights a user gets are the
/// author's bytes with the author's hash, and the container -- those same
/// weights pre-tiled for the array -- is produced locally from them.
///
/// The packing DECISION (which architecture, which tile width, which pooling,
/// which source repository) is npue::prepare_model_auto() in the engine and not
/// here. That separation is the point: a second copy of that decision in this
/// repository would have to agree byte for byte with upstream's, and upstream's
/// byte-identity gate would stop being evidence about these containers the
/// moment the two diverged. So this calls one function and chooses nothing.
std::string find_container(const std::filesystem::path& dir, const json& info) {
    namespace fs = std::filesystem;
    if (info.contains("npue_container") && info["npue_container"].is_string()) {
        const fs::path p = dir / info["npue_container"].get<std::string>();
        if (fs::is_regular_file(p)) return p.string();
    }
    std::vector<fs::path> found;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end;
         it.increment(ec)) {
        if (it->path().extension() == ".npue") found.push_back(it->path());
    }
    if (found.size() == 1) return found.front().string();
    if (found.size() > 1) {
        // Two containers are two different datapaths (bf16 vs int8) or two
        // different sequence lengths, and they are not interchangeable.
        // Picking one by sort order is how upstream's tasks/0104 nearly
        // shipped the wrong datapath; name the choice instead.
        std::string names;
        for (const auto& f : found)
            names += (names.empty() ? "" : ", ") + f.filename().string();
        throw std::runtime_error(
            "NpueEmbedding: " + dir.string() + " holds several .npue "
            "containers (" + names + ") and they are not interchangeable -- "
            "they differ in datapath or sequence length. Name one with "
            "\"npue_container\" in the model entry.");
    }
    // Nothing packed yet: do it now, once, from the checkpoint beside us. A
    // first run therefore costs a pack -- tens of seconds for a 100M model --
    // and every run after it mmaps the result.
    if (!fs::is_regular_file(dir / "config.json"))
        throw std::runtime_error(
            "NpueEmbedding: " + dir.string() + " has neither a .npue container "
            "nor a config.json to pack one from. The container is the "
            "checkpoint's weights pre-tiled for the array; it is produced "
            "locally from the model author's own files, which means those "
            "files have to be here.");
    header_print("NPUE", "no container yet -- packing one from the checkpoint "
                         "in " + dir.string() + " (first run only)");
    npue::PrepareOptions po;
    po.checkpoint_dir = dir.string();
    // WHICH REPOSITORY THESE WEIGHTS CAME FROM, and this tree is the
    // authority on it: ModelDownloader fetched them from the entry's own
    // `url`. Upstream's packer falls back to a CHECKPOINT.json side-car, which
    // is a file NpuEmbeddings writes and a HuggingFace checkpoint does not
    // have -- so without this, packing a model fetched by `flm pull` refuses.
    // It refuses rather than guessing because a container that misattributes
    // its own weights is a licensing statement, and the fix is to tell it,
    // not to loosen it.
    if (info.contains("npue_source_repo") &&
        info["npue_source_repo"].is_string()) {
        po.source_repo = info["npue_source_repo"].get<std::string>();
    } else if (info.contains("url") && info["url"].is_string()) {
        const std::string url = info["url"].get<std::string>();
        const std::string host = "huggingface.co/";
        const size_t i = url.find(host);
        if (i != std::string::npos) po.source_repo = url.substr(i + host.size());
    }
    // tile_n is a property of the MODEL's widths, not a preference: the design
    // asserts N % (tile_n * n_cols) == 0, so bge-large's N in {1024,3072,4096}
    // needs 32 where hidden-768 models take 48. The model entry names it when
    // it is not the default.
    if (info.contains("npue_tile_n") && info["npue_tile_n"].is_number_integer())
        po.tile_n = info["npue_tile_n"].get<int>();
    po.log = [](const std::string& s) { header_print("NPUE", s); };
    return npue::prepare_model_auto(po);
}

/// The compiled design set. Same two-tier convention open_embedding uses: a
/// model-local copy wins, so a brand-new geometry can ship its kernels inside
/// its own model directory before it is promoted to a maintained family;
/// otherwise the app's installed xclbin tree.
///
/// The family name is a GEOMETRY, not a model: one BERT-768 set also serves
/// gte-multilingual and nomic, because their GEMM shapes match bit for bit.
std::string find_artifacts(const std::filesystem::path& dir, const json& info) {
    namespace fs = std::filesystem;
    auto has_design = [](const fs::path& d) {
        std::error_code ec;
        return fs::is_regular_file(d / "gemm_rtp" / "design.json", ec);
    };

    const fs::path local = dir / "npue_designs";
    if (has_design(local)) return local.string();

    std::string prefix;
    try {
        prefix = utils::find_xclbin_path();
    } catch (const std::exception&) {
        prefix.clear();
    }
    if (!prefix.empty() && info.contains("npue_design_family") &&
        info["npue_design_family"].is_string()) {
        const fs::path cand = fs::path(prefix) / "xclbins" /
                              info["npue_design_family"].get<std::string>();
        if (has_design(cand)) return cand.string();
    }
    throw std::runtime_error(
        "NpueEmbedding: no design set for this model.\n"
        "Looked for " + local.string() + "/gemm_rtp/design.json and, if the "
        "model entry names \"npue_design_family\", for that family under the "
        "installed xclbin tree.\n"
        "A design set is four instruction streams over one xclbin, compiled "
        "for this model's GEMM geometry -- one set serves every model whose "
        "shapes match, which is why it is keyed by geometry rather than by "
        "model name.\n"
        "The design sets are BUILT, not checked in: see "
        "npu_offload/gemm_rtp/README.md for the one command that builds this "
        "family. They ship pre-built in the distributed package only.");
}

}  // namespace

struct NpueEmbedding::Impl {
    std::string tag;
    std::vector<std::string> prompt_names;
    std::unique_ptr<npue::enc::Embedder> emb;
};

NpueEmbedding::NpueEmbedding(flm_rt::device* npu_device_inst, std::string tag)
    : AutoEmbeddingModel(npu_device_inst, tag), impl_(new Impl) {
    impl_->tag = std::move(tag);
}

NpueEmbedding::~NpueEmbedding() = default;

void NpueEmbedding::load_model(std::string model_path, json model_info,
                               bool enable_preemption) {
    (void)enable_preemption;   // the engine holds one hw_context, always
    namespace fs = std::filesystem;
    this->model_path = std::move(model_path);

    const fs::path dir(this->model_path);
    npue::enc::EmbedderOptions opt;
    opt.npue_path = find_container(dir, model_info);
    opt.artifacts_dir = find_artifacts(dir, model_info);
    // Threads and lanes are the host-side shape of the work, not model facts,
    // so they come from model_info if it says anything and from sensible
    // defaults otherwise. `lanes` overlaps one lane's host work with another's
    // array work; the array serialises either way.
    opt.threads = int_or(model_info, "npue_threads",
                         std::max(1u, std::thread::hardware_concurrency()));
    opt.lanes = int_or(model_info, "npue_lanes", 2);

    impl_->emb = std::make_unique<npue::enc::Embedder>(opt);
    impl_->prompt_names = impl_->emb->prompt_names();
    this->is_model_loaded = true;

    header_print("NPUE", "loaded " + impl_->emb->name() + " (" +
                         impl_->emb->source_repo() + "), hidden " +
                         std::to_string(impl_->emb->hidden()) + ", seq " +
                         std::to_string(impl_->emb->seq()) + ", " +
                         impl_->emb->datapath());
}

std::vector<float> NpueEmbedding::embed(std::string& text,
                                        embedding_task_type_t task_type) {
    if (!impl_->emb)
        throw std::runtime_error(
            "NpueEmbedding: embed() called before load_model()");
    std::vector<std::string> one{text};
    return impl_->emb->embed(one, prompt_for(impl_->prompt_names, task_type));
}

std::vector<float> NpueEmbedding::embed_batch(
        const std::vector<std::string>& texts,
        embedding_task_type_t task_type, int64_t* tokens) {
    if (!impl_->emb)
        throw std::runtime_error(
            "NpueEmbedding: embed_batch() called before load_model()");
    return impl_->emb->embed(texts, prompt_for(impl_->prompt_names, task_type),
                             tokens);
}

int64_t NpueEmbedding::hidden() const {
    return impl_->emb ? impl_->emb->hidden() : 0;
}

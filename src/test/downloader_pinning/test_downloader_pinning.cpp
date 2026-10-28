/// \file test_downloader_pinning.cpp
/// \brief Pinned per-file download sources (no network, no NPU)
/// \author OpenFlowLM Team
/// \note Covers source resolution and pull/verify behavior using synthetic
///       model trees and an HF cache. Unsupported URL schemes prevent network
///       access even if a cache regression falls through to the transport.
#include "model_downloader.hpp"
#include "download_model.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#define TEST_REQUIRE(condition)                                                   \
    do {                                                                          \
        if (!(condition)) {                                                        \
            throw std::runtime_error(std::string("requirement failed: ") +       \
                                     #condition);                                  \
        }                                                                         \
    } while (false)

inline void RequireContains(std::string_view text, std::string_view expected) {
    if (text.find(expected) == std::string_view::npos) {
        throw std::runtime_error("expected '" + std::string(text) +
                                 "' to contain '" + std::string(expected) + "'");
    }
}

template <typename Exception = std::exception, typename Callable>
std::string RequireThrows(Callable&& callable) {
    try {
        callable();
    } catch (const Exception& error) {
        return error.what();
    }
    throw std::runtime_error("expected exception was not thrown");
}

inline void RunTest(void (*test)(), const char* name) {
    try {
        test();
        std::cout << "PASS " << name << '\n';
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        std::exit(1);
    }
}

namespace {

constexpr const char* kRevA = "0123456789abcdef0123456789abcdef01234567";
constexpr const char* kRevB = "fedcba9876543210fedcba9876543210fedcba98";

nlohmann::json BaseInfo() {
    return {
        {"url", "https://huggingface.co/SomeOrg/SomeModel"},
        {"ms_url", "https://modelscope.cn/models/some/SomeModel"},
        {"files", {"config.json", "model.gguf"}},
    };
}

nlohmann::json PinnedInfo() {
    auto info = BaseInfo();
    info["file_sources"] = {
        {"model.gguf", {{"url", "https://huggingface.co/SomeOrg/SomeModel-GGUF"},
                        {"revision", kRevA}}},
    };
    return info;
}

void test_gate_fires_only_on_non_empty_table() {
    TEST_REQUIRE(!uses_pinned_sources(BaseInfo()));
    auto empty = BaseInfo();
    empty["file_sources"] = nlohmann::json::object();
    TEST_REQUIRE(!uses_pinned_sources(empty));
    auto wrong_type = BaseInfo();
    wrong_type["file_sources"] = nlohmann::json::array();
    TEST_REQUIRE(!uses_pinned_sources(wrong_type));
    TEST_REQUIRE(uses_pinned_sources(PinnedInfo()));
}

void test_legacy_fallback_urls() {
    const auto info = BaseInfo();
    const auto main = resolve_file_source(info, "config.json", false);
    TEST_REQUIRE(main.url ==
                 "https://huggingface.co/SomeOrg/SomeModel/resolve/main/config.json?download=true");
    TEST_REQUIRE(main.revision.empty());

    const auto ms = resolve_file_source(info, "config.json", true);
    TEST_REQUIRE(ms.url ==
                 "https://modelscope.cn/models/some/SomeModel/resolve/main/config.json?download=true");

    auto branched = BaseInfo();
    branched["url"] = "https://huggingface.co/SomeOrg/SomeModel/resolve/refs%2Fpr%2F3";
    const auto branch = resolve_file_source(branched, "config.json", false);
    TEST_REQUIRE(branch.url ==
                 "https://huggingface.co/SomeOrg/SomeModel/resolve/refs%2Fpr%2F3/config.json?download=true");
}

void test_pin_forms_revision_url() {
    const auto info = PinnedInfo();
    const auto pinned = resolve_file_source(info, "model.gguf", false);
    TEST_REQUIRE(pinned.url == std::string("https://huggingface.co/SomeOrg/SomeModel-GGUF/resolve/") +
                               kRevA + "/model.gguf?download=true");
    TEST_REQUIRE(pinned.revision == kRevA);

    // A file with no pin falls back to the entry URL even when the table exists.
    const auto plain = resolve_file_source(info, "config.json", false);
    TEST_REQUIRE(plain.revision.empty());
    RequireContains(plain.url, "/resolve/main/config.json");
}

void test_pin_percent_encodes_filenames() {
    auto info = BaseInfo();
    info["files"] = {"my model.gguf"};
    info["file_sources"] = {
        {"my model.gguf", {{"url", "https://huggingface.co/SomeOrg/M"}, {"revision", kRevB}}},
    };
    const auto pinned = resolve_file_source(info, "my model.gguf", false);
    RequireContains(pinned.url, "my%20model.gguf");
    TEST_REQUIRE(pinned.url.find(' ') == std::string::npos);
}

void test_modelscope_with_pins_is_rejected() {
    const auto info = PinnedInfo();
    const std::string message =
        RequireThrows([&] { resolve_file_source(info, "model.gguf", true); });
    RequireContains(message, "--modelscope");

    // ...but an empty table pins nothing, so mirrors stay usable.
    auto empty = BaseInfo();
    empty["file_sources"] = nlohmann::json::object();
    TEST_REQUIRE(!resolve_file_source(empty, "config.json", true).url.empty());
}

void test_malformed_tables_throw() {
    auto wrong_type = BaseInfo();
    wrong_type["file_sources"] = nlohmann::json::array();
    RequireThrows([&] { resolve_file_source(wrong_type, "config.json", false); });

    auto unknown_key = PinnedInfo();
    unknown_key["file_sources"]["nope.bin"] = {{"url", "https://huggingface.co/X"},
                                               {"revision", kRevA}};
    RequireThrows([&] { resolve_file_source(unknown_key, "model.gguf", false); });

    auto missing_url = PinnedInfo();
    missing_url["file_sources"]["model.gguf"] = {{"revision", kRevA}};
    RequireThrows([&] { resolve_file_source(missing_url, "model.gguf", false); });

    auto empty_url = PinnedInfo();
    empty_url["file_sources"]["model.gguf"] = {{"url", ""}, {"revision", kRevA}};
    RequireThrows([&] { resolve_file_source(empty_url, "model.gguf", false); });

    auto extra_key = PinnedInfo();
    extra_key["file_sources"]["model.gguf"] = {{"url", "https://huggingface.co/X"},
                                               {"revision", kRevA},
                                               {"mirror", "y"}};
    RequireThrows([&] { resolve_file_source(extra_key, "model.gguf", false); });

    auto short_rev = PinnedInfo();
    short_rev["file_sources"]["model.gguf"] = {{"url", "https://huggingface.co/X"},
                                               {"revision", "abc123"}};
    const std::string short_msg =
        RequireThrows([&] { resolve_file_source(short_rev, "model.gguf", false); });
    RequireContains(short_msg, "40-character");

    auto non_hex_rev = PinnedInfo();
    non_hex_rev["file_sources"]["model.gguf"] = {
        {"url", "https://huggingface.co/X"},
        {"revision", "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"}};
    RequireThrows([&] { resolve_file_source(non_hex_rev, "model.gguf", false); });

    // The whole table is validated even when the requested file has a pin:
    // a broken sibling must not ship silently.
    auto broken_sibling = BaseInfo();
    broken_sibling["file_sources"] = {
        {"config.json", {{"url", "https://huggingface.co/X"}, {"revision", kRevA}}},
        {"model.gguf", {{"url", "https://huggingface.co/X"}, {"revision", "nope"}}},
    };
    RequireThrows([&] { resolve_file_source(broken_sibling, "config.json", false); });
}

void WriteRegistry(const std::filesystem::path& list_path, const std::string& body) {
    std::ofstream out(list_path);
    out << body;
}

class ScopedEnv {
    std::string name_;
    std::string old_;
    bool present_;

    void set(const char* value) {
#ifdef _WIN32
        _putenv_s(name_.c_str(), value ? value : "");
#else
        if (value) setenv(name_.c_str(), value, 1);
        else unsetenv(name_.c_str());
#endif
    }
public:
    ScopedEnv(const char* name, const std::string& value) : name_(name) {
        const char* old = std::getenv(name);
        present_ = old != nullptr;
        if (old) old_ = old;
        set(value.c_str());
    }
    ~ScopedEnv() { set(present_ ? old_.c_str() : nullptr); }
};

std::string ReadBytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    TEST_REQUIRE(in.is_open());
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

struct PullFixture {
    const std::filesystem::path root = std::filesystem::temp_directory_path() / "oflm-pull-review-test";
    const std::filesystem::path model_dir = root / "models" / "PullReview";
    const std::filesystem::path snapshot = root / "hub" / "models--review--Fixture" / "snapshots" / kRevA;
    ScopedEnv manifest_env{"OFLM_MODELINFO_PATH", (root / "model_info.json").string()};
    ScopedEnv cache_env{"HF_HUB_CACHE", (root / "hub").string()};
    nlohmann::json catalog;
    pull::Record current_record;
    pull::Record old_record;

    PullFixture(bool pinned, bool with_config = true, bool outdated = false) {
        namespace fs = std::filesystem;
        fs::remove_all(root);
        fs::create_directories(model_dir);
        fs::create_directories(snapshot);
        // The cache resolver accepts this synthetic HF URL; curl cannot use
        // the scheme, so a cache miss fails locally without a server or port.
        const std::string url = "offline://huggingface.co/review/Fixture";
        nlohmann::json info = {
            {"name", "PullReview"}, {"url", url + "/resolve/" + kRevA},
            {"file_url", url + "/tree/main"}, {"ms_url", ""},
            {"oflm_min_version", "0.9.34"}, {"files", {"weights.bin"}},
        };
        if (pinned)
            info["file_sources"] = {{"weights.bin", {{"url", url}, {"revision", kRevA}}}};

        WriteRegistry(model_dir / "weights.bin", "old");
        WriteRegistry(snapshot / "weights.bin", "new");
        const std::string new_oid = pull::file_oid(snapshot / "weights.bin", true);
        current_record["weights.bin"] = new_oid;
        old_record["weights.bin"] = pull::file_oid(model_dir / "weights.bin", true);
        nlohmann::json records = nlohmann::json::array({
            {{"path", "weights.bin"}, {"size", 3}, {"sha256", new_oid}},
        });
        if (with_config) {
            info["files"].insert(info["files"].begin(), "config.json");
            WriteRegistry(model_dir / "config.json",
                          outdated ? R"({"oflm_version":"0.9.1"})" : R"({"oflm_version":"0.9.34"})");
            const auto config_oid = pull::file_oid(model_dir / "config.json", true);
            current_record["config.json"] = old_record["config.json"] = config_oid;
            records.push_back({{"path", "config.json"}, {"size", fs::file_size(model_dir / "config.json")},
                               {"sha256", config_oid}});
        }
        catalog = {{"model_path", "models"}, {"models", {{"review", {{"test", info}}}}}};
        WriteRegistry(root / "model_info.json", nlohmann::json{{"review:test", records}}.dump());
    }

    ~PullFixture() {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }
};

void test_failed_pinned_upgrade_preserves_existing_files() {
    PullFixture fixture(true);
    std::string root = fixture.root.string();
    model_list models(fixture.catalog, root);
    ModelDownloader downloader(models);
    pull::write_record(fixture.model_dir, fixture.old_record);
    std::filesystem::remove(fixture.snapshot / "weights.bin");
    WriteRegistry(fixture.model_dir / "weights.bin.part", "n");
    for (bool force : {false, true}) {
        TEST_REQUIRE(!downloader.pull_model("review:test", false, force));
        TEST_REQUIRE(ReadBytes(fixture.model_dir / "weights.bin") == "old");
        TEST_REQUIRE(pull::read_record(fixture.model_dir) == fixture.old_record);
    }
    // Once a verified replacement is available, the same upgrade succeeds.
    WriteRegistry(fixture.snapshot / "weights.bin", "new");
    TEST_REQUIRE(downloader.pull_model("review:test"));
    TEST_REQUIRE(ReadBytes(fixture.model_dir / "weights.bin") == "new");
    TEST_REQUIRE(pull::read_record(fixture.model_dir) == fixture.current_record);
    TEST_REQUIRE(!std::filesystem::exists(fixture.model_dir / "weights.bin.part"));
}

void test_pinned_status_preserves_and_pull_repairs_corruption() {
    for (bool with_config : {false, true}) {
        PullFixture fixture(true, with_config);
        std::string root = fixture.root.string();
        model_list models(fixture.catalog, root);
        ModelDownloader downloader(models);
        // Right size and current record, but wrong bytes: full verification
        // must reject it without deleting it; fast status remains metadata-only.
        pull::write_record(fixture.model_dir, fixture.current_record);
        TEST_REQUIRE(downloader.is_model_downloaded("review:test", true, true) ==
                     ModelDownloader::ModelStatus::Ready);
        TEST_REQUIRE(downloader.is_model_downloaded("review:test", true) ==
                     ModelDownloader::ModelStatus::Missing);
        TEST_REQUIRE(ReadBytes(fixture.model_dir / "weights.bin") == "old");
        TEST_REQUIRE(downloader.pull_model("review:test"));
        TEST_REQUIRE(ReadBytes(fixture.model_dir / "weights.bin") == "new");
        TEST_REQUIRE(pull::read_record(fixture.model_dir) == fixture.current_record);
    }
}

void test_outdated_pinned_status_preserves_existing_files() {
    PullFixture fixture(true, true, true);
    std::string root = fixture.root.string();
    model_list models(fixture.catalog, root);
    ModelDownloader downloader(models);
    TEST_REQUIRE(downloader.is_model_downloaded("review:test", true) ==
                 ModelDownloader::ModelStatus::Outdated);
    TEST_REQUIRE(ReadBytes(fixture.model_dir / "weights.bin") == "old");
}

void test_unpinned_migration_records_advisory_hash_mismatch() {
    PullFixture fixture(false);
    std::string root = fixture.root.string();
    model_list models(fixture.catalog, root);
    ModelDownloader downloader(models);
    // Neither the legacy install nor the served bytes match the registry.
    // The cache path exercises the actual download-list verification marker.
    WriteRegistry(fixture.snapshot / "weights.bin", "alt");
    TEST_REQUIRE(downloader.pull_model("review:test"));
    TEST_REQUIRE(ReadBytes(fixture.model_dir / "weights.bin") == "alt");
    TEST_REQUIRE(pull::read_record(fixture.model_dir) == fixture.current_record);
    TEST_REQUIRE(!std::filesystem::exists(fixture.model_dir / ".oflm-files.json.tmp"));
    // No source remains: a second pull must trust the recorded registry oid.
    std::filesystem::remove(fixture.snapshot / "weights.bin");
    TEST_REQUIRE(downloader.pull_model("review:test"));
    TEST_REQUIRE(ReadBytes(fixture.model_dir / "weights.bin") == "alt");
}

void test_pinned_staged_hash_failure_keeps_destination() {
    PullFixture fixture(true);
    download_utils::DownloadRequest request{
        "offline://unavailable", fixture.model_dir / "weights.bin", 3,
        download_utils::HashAlgorithm::Sha256, fixture.current_record.at("weights.bin")};
    WriteRegistry(fixture.model_dir / "weights.bin.part", "bad");
    TEST_REQUIRE(!download_utils::download_file_atomic(request));
    TEST_REQUIRE(ReadBytes(fixture.model_dir / "weights.bin") == "old");
    TEST_REQUIRE(!std::filesystem::exists(fixture.model_dir / "weights.bin.part"));
    WriteRegistry(fixture.model_dir / "weights.bin.part", "new");
    TEST_REQUIRE(download_utils::download_file_atomic(request));
    TEST_REQUIRE(ReadBytes(fixture.model_dir / "weights.bin") == "new");
}

void test_remove_deletes_tree_and_links() {
    // remove_model must delete the whole model tree -- nested head dirs and
    // the open_kernels symlink (the link itself, never its target) -- plus
    // the user-level xclbins symlink oflm-add created for the model name.
    namespace fs = std::filesystem;
    const fs::path sandbox = fs::temp_directory_path() / "oflm-remove-test";
    std::error_code ec;
    fs::remove_all(sandbox, ec);
    fs::create_directories(sandbox, ec);
    const fs::path store = sandbox / "store";
    const fs::path list_path = sandbox / "model_list.json";
    WriteRegistry(list_path, R"({
        "model_path": "models",
        "models": { "t": { "s": {
            "name": "OflmRemoveTest-S",
            "url": "https://example.invalid/T",
            "file_url": "https://example.invalid/T/tree",
            "ms_url": "",
            "files": ["a.gguf", "sub/b.gguf"]
        } } }
    })");

    std::string list_arg = list_path.string();
    std::string store_arg = store.string();
    model_list models(list_arg, store_arg);
    ModelDownloader downloader(models);

    const fs::path model_dir = store / "models" / "OflmRemoveTest-S";
    const fs::path kernel_target = sandbox / "kernels";
    fs::create_directories(model_dir / "sub", ec);
    fs::create_directories(kernel_target, ec);
    { std::ofstream(model_dir / "a.gguf") << "x"; }
    { std::ofstream(model_dir / "sub" / "b.gguf") << "y"; }
    { std::ofstream(kernel_target / "manifest.json") << "{}"; }
    fs::create_directory_symlink(kernel_target, model_dir / "open_kernels", ec);
    const bool links_ok = !ec;

    const fs::path xroot = sandbox / "xclbins-home";
    fs::create_directories(xroot / "xclbins", ec);
    if (links_ok) {
        fs::create_directory_symlink(kernel_target, xroot / "xclbins" / "OflmRemoveTest-S", ec);
    }
#ifdef _WIN32
    _putenv_s("OFLM_XCLBIN_PATH", xroot.string().c_str());
#else
    setenv("OFLM_XCLBIN_PATH", xroot.string().c_str(), 1);
#endif

    TEST_REQUIRE(downloader.remove_model("t:s", true));
    TEST_REQUIRE(!fs::exists(model_dir, ec));
    // The symlink targets are data that removal must never follow into.
    TEST_REQUIRE(fs::exists(kernel_target / "manifest.json", ec));
    if (links_ok) {
        TEST_REQUIRE(!fs::exists(xroot / "xclbins" / "OflmRemoveTest-S", ec));
    }

#ifdef _WIN32
    _putenv_s("OFLM_XCLBIN_PATH", "");
#else
    unsetenv("OFLM_XCLBIN_PATH");
#endif
    fs::remove_all(sandbox, ec);
}

void test_remove_refuses_unknown_and_escape() {
    namespace fs = std::filesystem;
    const fs::path sandbox = fs::temp_directory_path() / "oflm-remove-guard-test";
    std::error_code ec;
    fs::remove_all(sandbox, ec);
    fs::create_directories(sandbox, ec);
    const fs::path store = sandbox / "store";
    const fs::path list_path = sandbox / "model_list.json";
    WriteRegistry(list_path, R"({
        "model_path": "models",
        "models": {
            "t": { "s": {
                "name": "OflmRemoveGuard-S",
                "url": "https://example.invalid/T",
                "file_url": "https://example.invalid/T/tree",
                "ms_url": "",
                "files": ["a.gguf"]
            } },
            "e": { "v": {
                "name": "../../evil-bytes",
                "url": "https://example.invalid/E",
                "file_url": "https://example.invalid/E/tree",
                "ms_url": "",
                "files": ["a.gguf"]
            } }
        }
    })");

    std::string list_arg = list_path.string();
    std::string store_arg = store.string();
    model_list models(list_arg, store_arg);
    ModelDownloader downloader(models);

    // Unknown tags are refused: get_model_info falls back to a default entry
    // instead of throwing, and removing that would delete the wrong model.
    TEST_REQUIRE(!downloader.remove_model("nope:1b", true));

    // A registry name escaping the models root is refused, never followed.
    const fs::path evil = sandbox / "evil-bytes";
    fs::create_directories(evil, ec);
    { std::ofstream(evil / "a.gguf") << "victim"; }
    TEST_REQUIRE(!downloader.remove_model("e:v", true));
    TEST_REQUIRE(fs::exists(evil / "a.gguf", ec));

    fs::remove_all(sandbox, ec);
}

void test_hf_cache_reuse() {
    namespace fs = std::filesystem;
    // download_from_hf_cache: a synthetic HF cache under HF_HOME, and the
    // resolve URL for the same repo/file, must be copied (no network) rather
    // than re-downloaded -- and a size mismatch must not reuse. Only the
    // revision the URL names may be reused, and a pinned entry's hash must
    // verify before the cached copy is promoted.
    std::error_code ec;
    const std::string home = (fs::temp_directory_path() / "oflm-hf-cache-test").string();
    fs::remove_all(home, ec);
    const std::string rev = "0123456789abcdef0123456789abcdef01234567";
    const auto cached =
        fs::path(home) / "hub" / "models--org--Widget" / "snapshots" / rev / "widget.gguf";
    fs::create_directories(cached.parent_path(), ec);
    {
        std::ofstream of(cached.string());
        of << "fake gguf bytes 0123456789";
    }
    // A second snapshot of the same file name, belonging to a *different*
    // revision. Nothing may be reused from it.
    const std::string other_rev = "fedcba9876543210fedcba9876543210fedcba98";
    const auto other =
        fs::path(home) / "hub" / "models--org--Widget" / "snapshots" / other_rev / "widget.gguf";
    fs::create_directories(other.parent_path(), ec);
    {
        std::ofstream of(other.string());
        of << "other revision bytes";
    }
    // The URL names the branch "main", so the cache must dereference it.
    {
        fs::create_directories((fs::path(home) / "hub" / "models--org--Widget" / "refs").string(), ec);
        std::ofstream of((fs::path(home) / "hub" / "models--org--Widget" / "refs" / "main").string());
        of << rev << "\n";
    }
    const std::uint64_t expected = static_cast<std::uint64_t>(fs::file_size(cached, ec));
    const auto out = fs::temp_directory_path() / "oflm-hf-cache-out" / "model.gguf";
    fs::remove_all(out.parent_path(), ec);
    setenv("HF_HOME", home.c_str(), 1);
    unsetenv("HF_HUB_CACHE");
    using download_utils::HashAlgorithm;
    const std::string url =
        "https://huggingface.co/org/Widget/resolve/main/widget.gguf?download=true";
    // the HF-cache miss path first (a repo the cache does not hold)
    TEST_REQUIRE(!download_utils::download_from_hf_cache(
        "https://huggingface.co/other/Missing/resolve/main/x.gguf", out.string(), 0,
        HashAlgorithm::Sha256, ""));
    // the hit: without a size guard
    TEST_REQUIRE(download_utils::download_from_hf_cache(
        url, out.string(), 0, HashAlgorithm::Sha256, ""));
    TEST_REQUIRE(fs::is_regular_file(out, ec));
    TEST_REQUIRE(static_cast<std::uint64_t>(fs::file_size(out, ec)) == expected);
    // the hit, with the correct size
    TEST_REQUIRE(download_utils::download_from_hf_cache(
        url, out.string(), expected, HashAlgorithm::Sha256, ""));
    // a wrong size must NOT reuse (falls back to a real download)
    TEST_REQUIRE(!download_utils::download_from_hf_cache(
        url, out.string(), expected + 1, HashAlgorithm::Sha256, ""));
    // a pinned entry whose hash does not verify must NOT reuse, and must not
    // leave the wrong bytes promoted in its place
    TEST_REQUIRE(!download_utils::download_from_hf_cache(
        url, out.string(), expected, HashAlgorithm::Sha256,
        "0000000000000000000000000000000000000000000000000000000000000000"));
    // percent-encoded filename resolves in the cache too
    TEST_REQUIRE(download_utils::download_from_hf_cache(
        "https://huggingface.co/org/Widget/resolve/main/widget.gguf", out.string(), expected,
        HashAlgorithm::Sha256, ""));

    // A revision named by the URL is honoured even when other snapshots exist.
    const std::string rev_url =
        "https://huggingface.co/org/Widget/resolve/" + rev + "/widget.gguf";
    TEST_REQUIRE(download_utils::download_from_hf_cache(
        rev_url, out.string(), expected, HashAlgorithm::Sha256, ""));
    // A pinned URL naming a revision the cache does not hold must miss, even
    // though a same-named file exists under a different revision.
    const std::string absent_rev = "aaaabbbbccccddddeeeeffff0000111122223333";
    TEST_REQUIRE(!download_utils::download_from_hf_cache(
        "https://huggingface.co/org/Widget/resolve/" + absent_rev + "/widget.gguf",
        out.string(), 0, HashAlgorithm::Sha256, ""));
    // An uncached ref is a miss, not a licence to guess a snapshot.
    TEST_REQUIRE(!download_utils::download_from_hf_cache(
        "https://huggingface.co/org/Widget/resolve/some-branch/widget.gguf",
        out.string(), 0, HashAlgorithm::Sha256, ""));
    // A ref that resolves to other_rev must take other_rev's bytes (size 20).
    {
        std::ofstream of((fs::path(home) / "hub" / "models--org--Widget" / "refs" / "next").string());
        of << other_rev << "\n";
    }
    TEST_REQUIRE(download_utils::download_from_hf_cache(
        "https://huggingface.co/org/Widget/resolve/next/widget.gguf",
        out.string(), 20, HashAlgorithm::Sha256, ""));
    TEST_REQUIRE(static_cast<std::uint64_t>(fs::file_size(out, ec)) == 20);
    unsetenv("HF_HOME");
    fs::remove_all(home, ec);
    fs::remove_all(out.parent_path(), ec);
}

}  // namespace

int main() {
    RunTest(test_gate_fires_only_on_non_empty_table, "gate fires only on non-empty table");
    RunTest(test_legacy_fallback_urls, "legacy fallback urls");
    RunTest(test_pin_forms_revision_url, "pin forms revision url");
    RunTest(test_pin_percent_encodes_filenames, "pin percent-encodes filenames");
    RunTest(test_modelscope_with_pins_is_rejected, "modelscope with pins is rejected");
    RunTest(test_malformed_tables_throw, "malformed tables throw");
    RunTest(test_remove_deletes_tree_and_links, "remove deletes tree and links");
    RunTest(test_remove_refuses_unknown_and_escape, "remove refuses unknown and escape");
    RunTest(test_hf_cache_reuse, "hf cache reuse skips the download");
    RunTest(test_failed_pinned_upgrade_preserves_existing_files, "failed pinned upgrade preserves existing files");
    RunTest(test_pinned_status_preserves_and_pull_repairs_corruption, "pinned status preserves and pull repairs corruption");
    RunTest(test_outdated_pinned_status_preserves_existing_files, "outdated pinned status preserves existing files");
    RunTest(test_unpinned_migration_records_advisory_hash_mismatch, "unpinned migration records advisory hash mismatch");
    RunTest(test_pinned_staged_hash_failure_keeps_destination, "pinned staged hash failure keeps destination");
    std::cout << "All downloader pinning tests passed\n";
    return 0;
}

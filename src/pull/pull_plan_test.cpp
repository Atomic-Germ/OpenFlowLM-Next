// Traces: PULL-STALE, PULL-RECORD, PULL-FORCE (canonical spec: specs/pull/spec.md)
//
// pull_plan_test: which files `oflm pull` fetches, on files written to a temp directory. No network.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "install_record.hpp"

namespace fs = std::filesystem;
using pull::Fetch;

static int failures = 0;
static int checks = 0;

static void ok(bool cond, const std::string& what) {
    ++checks;
    if (!cond) ++failures;
    std::printf("%s  %s\n", cond ? "ok  " : "FAIL", what.c_str());
}

static void write(const fs::path& p, const std::string& bytes) {
    std::ofstream(p, std::ios::binary | std::ios::trunc) << bytes;
}

static std::string reasons(const std::vector<std::pair<pull::RegistryFile, Fetch>>& plan) {
    std::string s;
    for (const auto& [f, why] : plan) s += (s.empty() ? "" : ", ") + f.path + ": " + pull::describe(why);
    return s;
}

int main() {
    const fs::path dir = fs::temp_directory_path() / "oflm_pull_plan_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string abc_sha256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const std::string abc_blob = "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f";   // git hash-object of "abc"

    write(dir / "abc", "abc");
    ok(pull::file_oid(dir / "abc", true) == abc_sha256, "an LFS file's oid is its sha256");
    ok(pull::file_oid(dir / "abc", false) == abc_blob, "any other file's oid is its git blob oid");
    ok(pull::file_oid(dir / "absent", true).empty(), "an absent file has no oid");

    const nlohmann::json manifest = nlohmann::json::parse(R"([
        {"path": "weights.bin", "size": 3, "oid": "GITPOINTER", "lfs": {"oid": "LFSOID", "size": 3}},
        {"path": "config.json", "size": 3, "oid": "GITOID"}
    ])");
    std::vector<std::string> unlisted;
    const auto reg = pull::registry_files(manifest, {"config.json", "weights.bin", "extra.bin"}, &unlisted);
    ok(reg.size() == 2 && reg[0].path == "config.json" && reg[1].path == "weights.bin",
       "registry entries come in the model list's order");
    ok(reg[1].lfs && reg[1].oid == "LFSOID" && reg[1].size == 3, "an LFS entry's oid is lfs.oid");
    ok(!reg[0].lfs && reg[0].oid == "GITOID", "any other entry's oid is its git oid");
    ok(unlisted == std::vector<std::string>{"extra.bin"}, "a file the manifest does not describe is unlisted");

    // PULL-STALE: the status check
    const pull::RegistryFile abc{"abc", 3, abc_sha256, true};
    const pull::Record none;
    write(dir / "short", "ab");
    ok(pull::status_of({"absent", 3, abc_sha256, true}, dir, none) == Fetch::Absent, "status: an absent file");
    ok(pull::status_of({"short", 3, abc_sha256, true}, dir, none) == Fetch::Size, "status: a file of the wrong size");
    ok(pull::status_of(abc, dir, none) == Fetch::No, "status: an unrecorded file of the right size (status never hashes)");
    ok(pull::status_of(abc, dir, {{"abc", abc_sha256}}) == Fetch::No, "status: a file recorded for the registry's oid");
    ok(pull::status_of(abc, dir, {{"abc", "OLDOID"}}) == Fetch::Oid, "status: a file recorded for another oid");

    // PULL-RECORD: an install with no record is hashed once
    write(dir / "xyz", "xyz");   // the same size as the registry's file, other bytes: invisible to size
    const std::vector<pull::RegistryFile> files = {
        abc,
        {"xyz", 3, abc_sha256, true},
        {"short", 3, abc_sha256, true},
        {"absent", 3, abc_sha256, true},
    };
    pull::Record rec;
    auto plan = pull::plan_pull(files, dir, &rec, false);
    ok(reasons(plan) == "xyz: contents differ from the registry, short: wrong size, absent: missing",
       "pull: a same-size file with other bytes is fetched, as are the wrong-size and the absent  (" +
           reasons(plan) + ")");
    ok(rec == pull::Record{{"abc", abc_sha256}}, "pull: a file whose hash matches is recorded, not fetched");

    write(dir / "abc", "abd");
    ok(pull::plan_pull({abc}, dir, &rec, false).empty(),
       "pull: a recorded file is not hashed again (its record, not its bytes, says it is current)");
    pull::RegistryFile moved = abc;
    moved.oid = "NEWOID";
    ok(reasons(pull::plan_pull({moved}, dir, &rec, false)) == "abc: downloaded for another revision",
       "pull: a new registry oid fetches a recorded file");
    ok(pull::plan_pull({{"abc", 3, "", true}}, dir, &rec, false).empty(),
       "pull: a file the registry gives no oid is neither hashed nor fetched");

    // PULL-FORCE
    plan = pull::plan_pull(files, dir, &rec, true);
    bool all_forced = plan.size() == files.size();
    for (const auto& p : plan) all_forced = all_forced && p.second == Fetch::Forced;
    ok(all_forced, "--force fetches every file, present and current ones included");

    pull::write_record(dir, {{"a", "1"}, {"sub/b", "2"}});
    ok(pull::read_record(dir) == pull::Record{{"a", "1"}, {"sub/b", "2"}}, "the record reads back what was written");
    ok(!fs::exists(dir / (std::string(pull::kRecordFile) + ".tmp")), "...and leaves no temp file");
    write(dir / pull::kRecordFile, "{not json");
    ok(pull::read_record(dir).empty(), "an unreadable record reads as no record");
    ok(pull::read_record(dir / "nowhere").empty(), "an absent record reads as no record");

    fs::remove_all(dir);
    std::printf("\n%s (%d checks, %d failures)\n", failures ? "FAILED" : "PASS", checks, failures);
    return failures ? 1 : 0;
}

// Which of a model's files `oflm pull` fetches, and the record of what it installed (specs/pull).
#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"

namespace pull {

struct RegistryFile {
    std::string path;
    uint64_t size = 0;
    std::string oid;     // lfs.oid (sha256) for an LFS file, else the git blob oid
    bool lfs = false;
};

// path -> the registry oid the file was downloaded for (not a local hash: see PULL-RECORD)
using Record = std::map<std::string, std::string>;
constexpr const char* kRecordFile = ".oflm-files.json";

Record read_record(const std::filesystem::path& model_dir);
void write_record(const std::filesystem::path& model_dir, const Record& record);

enum class Fetch { No, Absent, Size, Oid, Hash, Forced };
const char* describe(Fetch why);

std::vector<RegistryFile> registry_files(const nlohmann::json& manifest, const std::vector<std::string>& files,
                                         std::vector<std::string>* unlisted);

// Status only: never reads a file's contents, so `oflm list` stays a stat per file.
Fetch status_of(const RegistryFile& f, const std::filesystem::path& model_dir, const Record& record);

// A present, right-size file the record does not name is hashed once; a match joins *record.
std::vector<std::pair<RegistryFile, Fetch>> plan_pull(const std::vector<RegistryFile>& files,
                                                      const std::filesystem::path& model_dir,
                                                      Record* record, bool force);

std::string file_oid(const std::filesystem::path& p, bool lfs);

}  // namespace pull

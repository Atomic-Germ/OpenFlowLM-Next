#include "install_record.hpp"

#include <fstream>
#include <stdexcept>
#include <system_error>

#include "picosha2.h"
#include "sha1.hpp"

namespace fs = std::filesystem;

namespace pull {

Record read_record(const fs::path& model_dir) {
    Record r;
    std::ifstream f(model_dir / kRecordFile, std::ios::binary);
    if (!f) return r;
    try {
        const nlohmann::json j = nlohmann::json::parse(f);
        for (const auto& [path, oid] : j.at("files").items())
            if (oid.is_string()) r[path] = oid.get<std::string>();
    } catch (const nlohmann::json::exception&) {
        r.clear();   // unreadable is no record: the next pull hashes the files again
    }
    return r;
}

void write_record(const fs::path& model_dir, const Record& record) {
    nlohmann::json files = nlohmann::json::object();
    for (const auto& [path, oid] : record) files[path] = oid;
    const fs::path tmp = model_dir / (std::string(kRecordFile) + ".tmp");
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f << nlohmann::json{{"version", 1}, {"files", files}}.dump(1) << '\n';
        if (!f) throw std::runtime_error("cannot write " + tmp.string());
    }
    fs::rename(tmp, model_dir / kRecordFile);
}

const char* describe(Fetch why) {
    switch (why) {
        case Fetch::No:     return "current";
        case Fetch::Absent: return "missing";
        case Fetch::Size:   return "wrong size";
        case Fetch::Oid:    return "downloaded for another revision";
        case Fetch::Hash:   return "contents differ from the registry";
        case Fetch::Forced: return "--force";
    }
    return "";
}

std::vector<RegistryFile> registry_files(const nlohmann::json& manifest, const std::vector<std::string>& files,
                                         std::vector<std::string>* unlisted) {
    std::vector<RegistryFile> out;
    for (const auto& name : files) {
        const nlohmann::json* e = nullptr;
        if (manifest.is_array())
            for (const auto& f : manifest)
                if (f.is_object() && f.value("path", std::string()) == name) { e = &f; break; }
        if (!e) {
            if (unlisted) unlisted->push_back(name);
            continue;
        }
        RegistryFile r;
        r.path = name;
        r.lfs = e->contains("lfs") && e->at("lfs").is_object();
        if (e->contains("size") && e->at("size").is_number())
            r.size = static_cast<uint64_t>(e->at("size").get<double>());
        const nlohmann::json& src = r.lfs ? e->at("lfs") : *e;
        if (src.contains("oid") && src.at("oid").is_string()) r.oid = src.at("oid").get<std::string>();
        out.push_back(std::move(r));
    }
    return out;
}

Fetch status_of(const RegistryFile& f, const fs::path& model_dir, const Record& record) {
    const fs::path p = model_dir / f.path;
    std::error_code ec;
    if (!fs::is_regular_file(p, ec)) return Fetch::Absent;
    const auto on_disk = fs::file_size(p, ec);
    if (!ec && f.size && on_disk != f.size) return Fetch::Size;
    auto it = record.find(f.path);
    if (it != record.end() && !f.oid.empty() && it->second != f.oid) return Fetch::Oid;
    return Fetch::No;
}

std::vector<std::pair<RegistryFile, Fetch>> plan_pull(const std::vector<RegistryFile>& files,
                                                      const fs::path& model_dir, Record* record, bool force) {
    std::vector<std::pair<RegistryFile, Fetch>> out;
    for (const auto& f : files) {
        Fetch why = force ? Fetch::Forced : status_of(f, model_dir, *record);
        if (why == Fetch::No && !f.oid.empty() && !record->count(f.path)) {
            if (file_oid(model_dir / f.path, f.lfs) == f.oid) (*record)[f.path] = f.oid;
            else why = Fetch::Hash;
        }
        if (why != Fetch::No) out.emplace_back(f, why);
    }
    return out;
}

std::string file_oid(const fs::path& p, bool lfs) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return {};
    if (!lfs) {
        std::error_code ec;
        SHA1 sha1;
        sha1.update(std::string("blob ") + std::to_string(fs::file_size(p, ec)) + '\0');
        sha1.update(in);
        return sha1.final();
    }
    picosha2::hash256_one_by_one hasher;
    std::vector<char> buf(4u << 20);
    while (in) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        hasher.process(buf.begin(), buf.begin() + in.gcount());
    }
    hasher.finish();
    return picosha2::get_hash_hex_string(hasher);
}

}  // namespace pull

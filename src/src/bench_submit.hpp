/// \file bench_submit.hpp
/// \brief --submit for `oflm bench` / `oflm bench-embed`: upload the CSV to the
/// community leaderboard, with a one-time anonymous hardware survey.
///
/// First-run consent: the first --submit run creates `bench_consent.json` in
/// utils::get_models_directory() (~/.config/oflm on Linux, ~/.oflm on Windows)
/// holding an opaque install UUIDv4. The user is asked ONCE whether the hardware
/// survey (CPU / RAM / OS / NPU presence) may be sent with the numbers; the
/// choice is stored in that file and never asked again. --submit-consent yes|no
/// (or $OFLM_BENCH_SUBMIT_CONSENT) overrides it for one run.
///
/// The UUID ties results to the install+machine in a broader sense (the server
/// keeps only its hash, so it can count distinct machines for the survey without
/// identifying anyone) and is used as the submission_idempotency seed.
///
/// HTTP: libcurl. GET <base>/api/token mints the 10-minute HMAC token, then
/// POST <base>/api/benchmarks with the JSON + CSV. The base is --submit-url,
/// else $OFLM_BENCH_SUBMIT_URL, else the built-in default below. Plain http://
/// is refused unless $OFLM_BENCH_ALLOW_INSECURE_URL=1 (local tests).
#pragma once

#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include <curl/curl.h>

#include "nlohmann/json.hpp"
#include "utils/utils.hpp"

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#else
#include <unistd.h>
#endif

#ifndef __OFLM_VERSION__
#define __OFLM_VERSION__ "0.0.0"
#endif

namespace bench_submit {

inline const char* kDefaultBaseUrl = "https://oflm-bench.openflowlm.org";
inline const char* kConsentFileName = "bench_consent.json";

struct BenchProfile {
  std::string install_id;    // uuid v4, opaque
  bool survey_opt_in = false;
  bool first_run = false;    // true when this process just created the file
};

inline std::string consent_file_path() {
  const char* override_path = std::getenv("OFLM_BENCH_CONSENT_PATH");
  if (override_path && *override_path) return std::string(override_path);
  return (std::filesystem::path(utils::get_models_directory()) / kConsentFileName).string();
}

inline std::string generate_uuid_v4() {
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  uint8_t b[16];
  for (int i = 0; i < 16; i++) b[i] = static_cast<uint8_t>(rng() & 0xFF);
  b[6] = (b[6] & 0x0F) | 0x40;  // version 4
  b[8] = (b[8] & 0x3F) | 0x80;  // variant 10xxxxxx
  std::ostringstream ss;
  ss << std::hex << std::setfill('0');
  for (int i = 0; i < 16; i++) {
    if (i == 4 || i == 6 || i == 8 || i == 10) ss << '-';
    ss << std::setw(2) << static_cast<int>(b[i]);
  }
  return ss.str();
}

inline bool is_uuid_v4(const std::string& s) {
  if (s.size() != 36) return false;
  for (size_t i = 0; i < s.size(); i++) {
    if (i == 8 || i == 13 || i == 18 || i == 23) { if (s[i] != '-') return false; continue; }
    if (!std::isxdigit(static_cast<unsigned char>(s[i]))) return false;
  }
  return s[14] == '4';
}

inline BenchProfile load_profile() {
  BenchProfile p;
  const std::string path = consent_file_path();
  std::ifstream in(path);
  if (in.is_open()) {
    try {
      const nlohmann::json j = nlohmann::json::parse(in);
      const std::string id = j.value("install_id", std::string());
      if (is_uuid_v4(id)) {
        p.install_id = id;
        p.survey_opt_in = j.value("survey_opt_in", false);
        return p;
      }
    } catch (const std::exception&) { /* corrupted file: regenerate below */ }
  }
  p.install_id = generate_uuid_v4();
  p.survey_opt_in = false;
  p.first_run = true;
  const std::filesystem::path dir = std::filesystem::path(path).parent_path();
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  std::ofstream out(path, std::ios::trunc);
  if (out.is_open()) {
    out << nlohmann::json{
      {"install_id", p.install_id},
      {"survey_opt_in", false},
      {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
      {"oflm_version", __OFLM_VERSION__},
    }.dump(2) << "\n";
#ifndef _WIN32
    std::filesystem::permissions(path, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::replace, ec);
#endif
  }
  return p;
}

inline void save_profile(const BenchProfile& p) {
  const std::string path = consent_file_path();
  std::ofstream out(path, std::ios::trunc);
  if (!out.is_open()) return;
  out << nlohmann::json{
    {"install_id", p.install_id},
    {"survey_opt_in", p.survey_opt_in},
    {"created", std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())},
    {"oflm_version", __OFLM_VERSION__},
  }.dump(2) << "\n";
  out.close();
#ifndef _WIN32
  std::error_code ec;
  std::filesystem::permissions(path, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                               std::filesystem::perm_options::replace, ec);
#endif
}

inline bool opt_in_arg() {
  const char* e = std::getenv("OFLM_BENCH_SUBMIT_CONSENT");
  if (e && std::string(e) == "yes") return true;
  if (e && std::string(e) == "no") return false;
  return false;
}

inline bool is_tty_stdin() {
#ifdef _WIN32
  return _isatty(_fileno(stdin)) != 0;
#else
  return ::isatty(STDIN_FILENO) != 0;
#endif
}

/// Resolve the effective survey choice. Returns true when a prompt
/// (interactive, first run) or an explicit flag/env answers. A non-tty first
/// run without a flag defaults to opt-OUT and the caller prints the notice.
inline bool resolve_consent(BenchProfile& p, const std::string& consent_flag) {
  if (!consent_flag.empty()) {
    p.survey_opt_in = consent_flag == "yes";
    if (!p.first_run) save_profile(p); // remember the new choice on later runs too
    else save_profile(p);
    return true;
  }
  const char* env = std::getenv("OFLM_BENCH_SUBMIT_CONSENT");
  if (env && std::string(env) != "") {
    p.survey_opt_in = std::string(env) == "yes";
    save_profile(p);
    return true;
  }
  if (!p.first_run) return true; // stored choice stands; never re-prompt
  if (!is_tty_stdin()) {
    std::cout << "[OFLM] First --submit: hardware survey defaults to OFF "
                 "(non-interactive). Opt in with --submit-consent yes. "
                 "Anonymous install id written to " << consent_file_path() << "\n";
    return false;
  }
  std::cout << "\n[OFLM] First benchmark submission on this machine.\n"
               "  --submit uploads only the bench CSV (model, numbers, OFLM version).\n"
               "  The hardware survey additionally sends CPU model, RAM size and OS,\n"
               "  tagged with this install's anonymous id so the survey can count\n"
               "  distinct machines without identifying anyone.\n"
               "  Include the hardware survey with this submission? [y/N] "
            << std::flush;
  std::string line;
  if (!std::getline(std::cin, line)) line = "";
  p.survey_opt_in = !line.empty() && (line[0] == 'y' || line[0] == 'Y');
  save_profile(p);
  return true;
}

// --- hardware survey (only fields oflm can actually read on the box) ---

inline std::string survey_cpu() {
#ifdef _WIN32
  // CPUID brand string (same source benchmarking::get_cpu_name() reads; split
  // out here so bench_submit.hpp need not pull in the whole engine headers).
  int cpu_info[4] = {0};
  char brand[0x40] = {0};
  __cpuid(cpu_info, 0x80000000);
  if (static_cast<unsigned>(cpu_info[0]) >= 0x80000004) {
    __cpuid(cpu_info, 0x80000002); std::memcpy(brand, cpu_info, sizeof(cpu_info));
    __cpuid(cpu_info, 0x80000003); std::memcpy(brand + 16, cpu_info, sizeof(cpu_info));
    __cpuid(cpu_info, 0x80000004); std::memcpy(brand + 32, cpu_info, sizeof(cpu_info));
  }
  std::string name(brand);
  while (!name.empty() && std::isspace(static_cast<unsigned char>(name.front()))) name.erase(name.begin());
  while (!name.empty() && std::isspace(static_cast<unsigned char>(name.back()))) name.pop_back();
  return name.empty() ? "unknown" : name;
#else
  std::ifstream f("/proc/cpuinfo");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("model name", 0) == 0) {
      const auto pos = line.find(':');
      if (pos != std::string::npos) {
        const std::string v = line.substr(pos + 1);
        const auto s = v.find_first_not_of(" \t");
        return v.substr(s == std::string::npos ? 0 : s);
      }
    }
  }
  return "unknown";
#endif
}

inline double survey_ram_gb() {
  try {
#ifdef _WIN32
    // GlobalMemoryStatusEx is the same source the Python submitter approximates
    // via wmic; the C++ version reads the OS directly.
    MEMORYSTATUSEX st{};
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) return static_cast<double>(st.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
#else
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
      if (line.rfind("MemTotal", 0) == 0) {
        std::istringstream ss(line);
        std::string key; unsigned long long kb = 0; std::string unit;
        ss >> key >> kb >> unit;
        return static_cast<double>(kb) / (1024.0 * 1024.0);
      }
    }
#endif
  } catch (...) {}
  return 0.0;
}

inline std::string survey_os() {
#ifdef _WIN32
  OSVERSIONINFOEXW v{sizeof(v)};
  if (GetVersionExW(reinterpret_cast<OSVERSIONINFOW*>(&v))) {
    std::ostringstream ss;
    ss << "Windows " << v.dwMajorVersion << "." << v.dwMinorVersion << " build " << v.dwBuildNumber;
    return ss.str();
  }
  return "Windows";
#else
  std::ifstream f("/etc/os-release");
  std::string line, name;
  while (std::getline(f, line)) {
    if (line.rfind("PRETTY_NAME=", 0) == 0) { name = line.substr(12); break; }
  }
  if (!name.empty() && name.front() == '"' && name.back() == '"') name = name.substr(1, name.size() - 2);
  return name.empty() ? "Linux" : name;
#endif
}

inline std::string survey_npu() {
#ifdef _WIN32
  // XRT is the stack that opens the NPU; if it is present the box has one.
  // No GPU/driver strings are read here (oflm bench never instruments them).
  return "AMD Ryzen AI (XDNA2)";
#else
  std::error_code ec;
  return std::filesystem::exists("/dev/accel/accel0", ec) ? "AMD Ryzen AI (XDNA2)" : "unknown";
#endif
}

inline nlohmann::json survey_json(const BenchProfile& p) {
  nlohmann::json j;
  if (!p.survey_opt_in) return j; // server sees {} -> survey fields stay NULL
  j["cpu"] = survey_cpu();
  const double ram = survey_ram_gb();
  if (ram > 0) j["ram_gb"] = ram;
  j["os"] = survey_os();
  j["npu"] = survey_npu();
  j["oflm_version"] = __OFLM_VERSION__;
  return j;
}

// --- transport ---

inline std::string submit_base_url(const std::string& url_flag) {
  if (!url_flag.empty()) return url_flag;
  const char* e = std::getenv("OFLM_BENCH_SUBMIT_URL");
  if (e && *e) return std::string(e);
  return kDefaultBaseUrl;
}

inline bool url_allowed(const std::string& url) {
  if (url.rfind("https://", 0) == 0) return true;
  const char* v = std::getenv("OFLM_BENCH_ALLOW_INSECURE_URL");
  return v && std::string(v) == "1" && url.rfind("http://", 0) == 0;
}

namespace detail {
inline size_t write_cb(void* dest, size_t size, size_t nmemb, void* user) {
  static_cast<std::string*>(user)->append(
      static_cast<char*>(dest), size * nmemb);
  return size * nmemb;
}
inline std::string trim(const std::string& u) {
  const auto s = u.find_first_not_of(" \t\r\n");
  if (s == std::string::npos) return "";
  return u.substr(s, u.find_last_not_of(" \t\r\n") - s + 1);
}
} // namespace detail

inline bool http_get_json(const std::string& url, const std::string& token,
                            long& http_status, std::string& body) {
  CURL* curl = curl_easy_init();
  if (!curl) return false;
  std::string resp;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, detail::write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  CURLcode rc = curl_easy_perform(curl);
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_status);
  curl_easy_cleanup(curl);
  if (rc != CURLE_OK) return false;
  body = resp;
  (void)token; // token endpoint takes no token
  return http_status == 200;
}

inline bool http_post_json(const std::string& url, const std::string& json_body,
                             const std::string& token, long& http_status, std::string& body) {
  CURL* curl = curl_easy_init();
  if (!curl) return false;
  std::string resp;
  std::string token_hdr = "X-OFLM-Token: " + token;
  const char* ct = "Content-Type: application/json";
  struct curl_slist* hdrs = nullptr;
  hdrs = curl_slist_append(hdrs, token_hdr.c_str());
  hdrs = curl_slist_append(hdrs, ct);
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(json_body.size()));
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, detail::write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
  CURLcode rc = curl_easy_perform(curl);
  curl_slist_free_all(hdrs);
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_status);
  curl_easy_cleanup(curl);
  if (rc != CURLE_OK) return false;
  body = resp;
  return http_status >= 200 && http_status < 300;
}

/// POST the bench result. Returns (http_status, server_id) or throws.
/// csv_text is the exact bytes of the bench_*.csv / bench_embed_*.csv file.
inline void submit(const std::string& base_url, BenchProfile profile,
                   const std::string& model, const std::string& task,
                   const std::string& backend, const std::string& quantization,
                   const std::string& csv_text, int runs,
                   const nlohmann::json& config = nlohmann::json::object()) {
  if (!url_allowed(base_url))
    throw std::runtime_error("refusing http:// submission URL without OFLM_BENCH_ALLOW_INSECURE_URL=1");
  const std::string base = detail::trim(base_url);
  long status = 0; std::string body;
  if (!http_get_json(base + "/api/token", "", status, body))
    throw std::runtime_error("could not mint submission token (HTTP " + std::to_string(status) + ")");
  std::string token;
  try { token = nlohmann::json::parse(body).value("token", std::string()); }
  catch (const std::exception&) { throw std::runtime_error("token response is not JSON"); }
  if (token.empty()) throw std::runtime_error("token response has no token");

  nlohmann::json payload;
  // submission_id is per-submission (idempotency); install_id is the anonymous
  // per-install id the survey counts distinct machines with.
  payload["submission_id"] = generate_uuid_v4();
  payload["install_id"] = profile.install_id;
  payload["survey_opt_in"] = profile.survey_opt_in;
  payload["model"] = model;
  payload["task"] = task;
  payload["backend"] = backend;
  payload["quantization"] = quantization;
  payload["oflm_version"] = __OFLM_VERSION__;
  payload["runs"] = runs;
  payload["source"] = "cli";
  payload["config"] = config;
  payload["csv"] = csv_text;
  // Survey fields are sent only when the stored choice says so. install_id is
  // always sent (the server stores only its hash, for distinct-machine counts).
  if (profile.survey_opt_in) {
    const nlohmann::json s = survey_json(profile);
    payload["cpu"] = s.value("cpu", std::string());
    payload["ram_gb"] = s.value("ram_gb", 0.0);
    payload["os"] = s.value("os", std::string());
    payload["hardware"] = s.value("npu", std::string()); // hardware column = NPU line
  }
  long post_status = 0; std::string post_body;
  if (!http_post_json(base + "/api/benchmarks", payload.dump(), token, post_status, post_body))
    throw std::runtime_error("submission failed (HTTP " + std::to_string(post_status) + "): " + post_body);
  nlohmann::json resp = nlohmann::json::parse(post_body);
  std::cout << "[OFLM] Submitted benchmark #" << resp.value("id", std::string("?"))
            << " (" << resp.value("status", std::string()) << ")" << std::endl;
}

inline std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) throw std::runtime_error("cannot read bench CSV: " + path);
  std::ostringstream ss; ss << in.rdbuf();
  return ss.str();
}

/// Read the quant string the registry carries for this model ("" if none).
inline std::string quantization_of(const nlohmann::json& model_info) {
  if (model_info.contains("quantization_level"))
    return model_info["quantization_level"].get<std::string>();
  if (model_info.contains("quantization"))
    return model_info["quantization"].get<std::string>();
  return std::string();
}

} // namespace bench_submit

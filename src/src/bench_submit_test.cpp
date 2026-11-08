/// \brief Unit tests for the pure half of `oflm bench --submit`.
/// \details No network, no device, no weights. Covers the parts a rigged
/// submission could silently break: the anonymous install id must be a stable
/// UUID, the consent file must persist the survey choice, an opt-out MUST
/// yield an empty survey (server sees no hardware), and http:// must be
/// refused by default. Run with OFLM_BENCH_CONSENT_PATH pointed at a tmp file.
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "bench_submit.hpp"

#define CHECK(cond, msg) if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; return 1; }

int main() {
  const std::string tmp = "/tmp/opencode/oflm-bench-submit-test/bench_consent.json";
  const std::filesystem::path dir = std::filesystem::path(tmp).parent_path();
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  setenv("OFLM_BENCH_CONSENT_PATH", tmp.c_str(), 1);
  // No consent flag, no env override anywhere in this process.
  unsetenv("OFLM_BENCH_SUBMIT_CONSENT");
  unsetenv("OFLM_BENCH_SUBMIT_URL");

  // First load creates the file with a stable uuid.
  bench_submit::BenchProfile p = bench_submit::load_profile();
  CHECK(p.first_run, "first load must report first_run");
  CHECK(p.survey_opt_in == false, "first run defaults to opt-out");
  CHECK(bench_submit::is_uuid_v4(p.install_id), "install id must be uuid v4");

  bench_submit::BenchProfile again = bench_submit::load_profile();
  CHECK(again.install_id == p.install_id, "install id must be stable across loads");
  CHECK(!again.first_run, "second load must not be first_run");

  // Flag consent persists; survey payload follows it exactly.
  p.survey_opt_in = true;
  bench_submit::save_profile(p);
  CHECK(bench_submit::load_profile().survey_opt_in, "consent must persist");
  nlohmann::json s_on = bench_submit::survey_json(bench_submit::load_profile());
  CHECK(s_on.is_object() && s_on.contains("cpu") && s_on.contains("os"),
        "opt-in survey must carry cpu+os");

  const std::string saved_id = bench_submit::load_profile().install_id;
  nlohmann::json s_off = bench_submit::survey_json(
      bench_submit::BenchProfile{saved_id, false, false});
  CHECK(s_off.empty(), "opt-out survey MUST be empty (server sees no hardware)");

  // URL policy: https always, http only with the test escape hatch.
  CHECK(bench_submit::url_allowed("https://x.test"), "https allowed");
  unsetenv("OFLM_BENCH_ALLOW_INSECURE_URL");
  CHECK(!bench_submit::url_allowed("http://127.0.0.1:8787"), "http refused by default");
  setenv("OFLM_BENCH_ALLOW_INSECURE_URL", "1", 1);
  CHECK(bench_submit::url_allowed("http://127.0.0.1:8787"), "http allowed for local tests");

  // Quantization readers: registry field wins, missing -> "".
  CHECK(bench_submit::quantization_of(nlohmann::json{{"quantization_level", "Q4_K"}}) == "Q4_K",
        "quantization_level read");
  CHECK(bench_submit::quantization_of(nlohmann::json{}) == "", "missing quant reads empty");

  std::cout << "bench_submit_test: all checks passed\n";
  return 0;
}
